#!/usr/bin/env python3
"""Convert the original FNO Darcy MAT files to the suite HDF5 contract.

This is benchmark-only glue.  It keeps the production loader unchanged, so a
1,250-case non-test pool is used to obtain the normal seeded 1,000/125/125
split.  The first 200 cases of the paper's second MAT file remain isolated for
post-training evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.io


FILES = {
    "piececonst_r421_N1024_smooth1.mat":
        "c8b1fd73a8bae85aa48afeb1fabd01204a03debe8e208524b72f615e13d2f664",
    "piececonst_r421_N1024_smooth2.mat":
        "c6522cf6cb3a8c4818268b2b07464a85e55ecd7206a23b4b1ac435bfae2e3eb5",
}
MIRROR = "https://huggingface.co/datasets/kmario23/standard-pde-benchmark"
DATASET_RECORD = "https://zenodo.org/records/12784353"
PAPER = "https://arxiv.org/abs/2010.08895"
OFFICIAL_RECIPE = (
    "https://github.com/li-Pingan/fourier-neural-operator/blob/main/"
    "FNO-torch.1.6/fourier_2d.py"
)
SOURCE_RESOLUTION = 421
STRIDE = 5
BENCHMARK_RESOLUTION = 85
PAPER_RESULT = 0.0108


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=here / "source" / "darcy")
    parser.add_argument("--train-output", type=Path)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument(
        "--paper-protocol", action="store_true",
        help=(
            "Write opt-in files whose unchanged seeded split contains exactly "
            "smooth1[0:1000], and encode the direct solution as the suite delta target"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fields(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = scipy.io.loadmat(path, variable_names=("coeff", "sol"))
    missing = {"coeff", "sol"} - raw.keys()
    if missing:
        raise ValueError(f"{path} is missing MAT fields: {sorted(missing)}")

    def orient(array: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(array)
        if array.shape == (1024, SOURCE_RESOLUTION, SOURCE_RESOLUTION):
            return array
        if array.shape == (SOURCE_RESOLUTION, SOURCE_RESOLUTION, 1024):
            return np.transpose(array, (2, 0, 1))
        raise ValueError(f"Unexpected {name} shape in {path}: {array.shape}")

    return orient(raw["coeff"], "coeff"), orient(raw["sol"], "sol")


def downsample(
    array: np.ndarray, indices: np.ndarray, dtype: np.dtype = np.float32
) -> np.ndarray:
    result = np.asarray(array[indices, ::STRIDE, ::STRIDE], dtype=dtype)
    expected = (len(indices), BENCHMARK_RESOLUTION, BENCHMARK_RESOLUTION)
    if result.shape != expected:
        raise ValueError(f"Downsampled shape {result.shape}, expected {expected}")
    return result


def initialize_file(path: Path, role: str, provenance: dict[str, object]) -> h5py.File:
    handle = h5py.File(path, "w")
    handle.attrs["format"] = "cae_ml_suite_mesh_hdf5_v1"
    handle.attrs["benchmark"] = "fno_darcy_85"
    handle.attrs["benchmark_role"] = role
    handle.attrs["benchmark_protocol"] = provenance["benchmark_protocol"]
    handle.attrs["target_encoding"] = provenance["target_encoding"]
    handle.attrs["num_timesteps"] = 2
    handle.create_group("data")
    topology = handle.create_group("topology")
    n_nodes = BENCHMARK_RESOLUTION**2
    chain = np.stack(
        [np.arange(n_nodes - 1, dtype=np.int64), np.arange(1, n_nodes, dtype=np.int64)]
    )
    topology.create_dataset("mesh_edge_chain", data=chain, compression="gzip")
    metadata = handle.create_group("metadata")
    metadata.create_dataset("provenance_json", data=json.dumps(provenance, sort_keys=True))
    return handle


def write_case(
    handle: h5py.File,
    sample_id: int,
    coeff: np.ndarray,
    solution: np.ndarray,
    coordinates: np.ndarray,
    source_file: str,
    source_index: int,
) -> None:
    n_nodes = BENCHMARK_RESOLUTION**2
    nodal_dtype = np.result_type(coeff.dtype, solution.dtype, np.float32)
    nodal = np.zeros((4, 2, n_nodes), dtype=nodal_dtype)
    nodal[0:2, :, :] = coordinates[:, None, :]
    nodal[3, 0, :] = coeff.reshape(-1)
    nodal[3, 1, :] = solution.reshape(-1)
    group = handle["data"].create_group(str(sample_id))
    group.create_dataset(
        "nodal_data", data=nodal, compression="gzip", compression_opts=4, shuffle=True
    )
    group["mesh_edge"] = handle["topology/mesh_edge_chain"]
    group.attrs["source_file"] = source_file
    group.attrs["source_index"] = int(source_index)


def audit_file(path: Path, expected_count: int) -> None:
    with h5py.File(path, "r") as handle:
        ids = sorted(int(key) for key in handle["data"])
        if ids != list(range(expected_count)):
            raise ValueError(f"{path}: non-contiguous sample IDs")
        for sample_id in ids:
            group = handle[f"data/{sample_id}"]
            nodal = group["nodal_data"]
            edges = group["mesh_edge"]
            if nodal.shape != (4, 2, BENCHMARK_RESOLUTION**2):
                raise ValueError(f"{path} sample {sample_id}: bad nodal shape {nodal.shape}")
            if edges.shape != (2, BENCHMARK_RESOLUTION**2 - 1):
                raise ValueError(f"{path} sample {sample_id}: bad edge shape {edges.shape}")
            if not np.all(np.isfinite(nodal[3, :, :])):
                raise ValueError(f"{path} sample {sample_id}: non-finite field")


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    paths = {name: args.source_dir / name for name in FILES}
    for name, expected in FILES.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, got {actual}")

    train_default = "darcy_paper_train.h5" if args.paper_protocol else "darcy_train.h5"
    test_default = "darcy_paper_test.h5" if args.paper_protocol else "darcy_test.h5"
    train_output = (args.train_output or here / train_default).resolve()
    test_output = (args.test_output or here / test_default).resolve()
    for output in (train_output, test_output):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it")
        output.parent.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, object] = {
        "benchmark": "FNO Darcy flow",
        "paper": PAPER,
        "paper_table_result_mean_relative_l2_s85": PAPER_RESULT,
        "official_recipe_mirror": OFFICIAL_RECIPE,
        "dataset_record": DATASET_RECORD,
        "download_mirror": MIRROR,
        "source_sha256": FILES,
        "source_resolution": SOURCE_RESOLUTION,
        "downsample_stride": STRIDE,
        "benchmark_resolution": BENCHMARK_RESOLUTION,
        "benchmark_protocol": (
            "paper_direct_solution_v1" if args.paper_protocol else "suite_residual_v1"
        ),
        "target_encoding": (
            "t0=coefficient,t1=coefficient+solution; unchanged residual target is direct solution"
            if args.paper_protocol else "t0=coefficient,t1=solution; residual target is solution-coefficient"
        ),
        "paper_training_cases": "smooth1 indices 0..999",
        "suite_training_pool": (
            "seed-42 optimization IDs contain exactly smooth1 indices 0..999; the remaining "
            "250 IDs contain smooth2 indices 200..449"
            if args.paper_protocol else
            "smooth1 indices 0..999 plus smooth2 indices 200..449; unchanged seeded "
            "80/10/10 split yields 1000/125/125"
        ),
        "isolated_test_cases": "smooth2 indices 0..199",
        "temporal_encoding_note": (
            "Paper protocol stores coefficient+solution at t1, making the unchanged suite's "
            "delta target equal the direct solution. Rollout t1 is coefficient+prediction; "
            "the benchmark evaluator subtracts t0 before applying the paper metric."
            if args.paper_protocol else
            "The unchanged suite represents the operator pair as one transition: "
            "coefficient at t0 and solution at t1. Saved inference t1 is the solution."
        ),
        "comparability_note": (
            "Opt-in paper protocol: exact optimization/test composition, direct-solution target, "
            "paper FNO core, decoded relative-L2 loss, Adam, and StepLR. The regular-grid data "
            "still passes through the suite's identity splat/sample adapter."
            if args.paper_protocol else
            "Same Darcy distribution, resolution, architecture scale, and 200-case metric. "
            "Not a strict reproduction: the suite shuffles a 1250-case non-test pool, "
            "optimizes normalized residual MSE, and retains its native scheduler/runtime."
        ),
    }

    train_tmp = train_output.with_suffix(train_output.suffix + ".tmp")
    test_tmp = test_output.with_suffix(test_output.suffix + ".tmp")
    for temporary in (train_tmp, test_tmp):
        if temporary.exists():
            temporary.unlink()

    axis = np.linspace(0.0, 1.0, BENCHMARK_RESOLUTION, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=0)

    try:
        storage_dtype = np.float64 if args.paper_protocol else np.float32
        with initialize_file(train_tmp, "suite_training_pool", provenance) as train_h5:
            coeff, solution = load_fields(paths["piececonst_r421_N1024_smooth1.mat"])
            indices = np.arange(1000, dtype=np.int64)
            coeff_ds = downsample(coeff, indices, dtype=storage_dtype)
            solution_ds = downsample(solution, indices, dtype=storage_dtype)
            if args.paper_protocol:
                shuffled_ids = np.arange(1250, dtype=np.int64)
                np.random.default_rng(42).shuffle(shuffled_ids)
                paper_train_ids = shuffled_ids[:1000]
                extra_pool_ids = shuffled_ids[1000:]
            else:
                paper_train_ids = np.arange(1000, dtype=np.int64)
                extra_pool_ids = np.arange(1000, 1250, dtype=np.int64)
            for i, sample_id in enumerate(paper_train_ids.tolist()):
                stored_t1 = coeff_ds[i] + solution_ds[i] if args.paper_protocol else solution_ds[i]
                write_case(
                    train_h5, sample_id, coeff_ds[i], stored_t1, coordinates,
                    "piececonst_r421_N1024_smooth1.mat", i,
                )
            del coeff, solution, coeff_ds, solution_ds

            coeff, solution = load_fields(paths["piececonst_r421_N1024_smooth2.mat"])
            extra_indices = np.arange(200, 450, dtype=np.int64)
            coeff_extra = downsample(coeff, extra_indices, dtype=storage_dtype)
            solution_extra = downsample(solution, extra_indices, dtype=storage_dtype)
            for offset, source_index in enumerate(extra_indices.tolist()):
                stored_t1 = (
                    coeff_extra[offset] + solution_extra[offset]
                    if args.paper_protocol else solution_extra[offset]
                )
                write_case(
                    train_h5, int(extra_pool_ids[offset]), coeff_extra[offset], stored_t1,
                    coordinates, "piececonst_r421_N1024_smooth2.mat", source_index,
                )

            with initialize_file(test_tmp, "published_test", provenance) as test_h5:
                test_indices = np.arange(200, dtype=np.int64)
                coeff_test = downsample(coeff, test_indices, dtype=storage_dtype)
                solution_test = downsample(solution, test_indices, dtype=storage_dtype)
                for sample_id in range(200):
                    stored_t1 = (
                        coeff_test[sample_id] + solution_test[sample_id]
                        if args.paper_protocol else solution_test[sample_id]
                    )
                    write_case(
                        test_h5, sample_id, coeff_test[sample_id], stored_t1,
                        coordinates, "piececonst_r421_N1024_smooth2.mat", sample_id,
                    )

        audit_file(train_tmp, 1250)
        audit_file(test_tmp, 200)
        os.replace(train_tmp, train_output)
        os.replace(test_tmp, test_output)
    finally:
        for temporary in (train_tmp, test_tmp):
            if temporary.exists():
                temporary.unlink()

    provenance_name = "darcy_paper.provenance.json" if args.paper_protocol else "darcy.provenance.json"
    provenance_path = train_output.parent / provenance_name
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {train_output} ({train_output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Wrote {test_output} ({test_output.stat().st_size / 1024**2:.1f} MiB)")
    if args.paper_protocol:
        print("Training pool: seed-42 optimization IDs map exactly to smooth1 indices 0..999")
        print("Target encoding: t1 - t0 = direct physical solution")
    else:
        print("Training pool: 1,250 -> unchanged seeded 1,000/125/125 split")
    print("Isolated paper test: smooth2 indices 0..199 (200 cases)")


if __name__ == "__main__":
    main()
