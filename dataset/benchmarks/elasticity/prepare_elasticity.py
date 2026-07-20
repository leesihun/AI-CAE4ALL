#!/usr/bin/env python3
"""Convert the public Geo-FNO/Transolver Elasticity arrays to suite HDF5.

The production loaders intentionally remain unchanged.  A 1,250-case training
pool therefore produces the suite's normal seeded 80/10/10 split: 1,000 cases
for optimization and 125 each for validation and internal testing.  The
published 200-case test partition is written to a physically separate file and
is only used for post-training inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np


FILES = {
    "Random_UnitCell_XY_10.npy": "29c615b7c8b0ef94252e4def4cd9999653b8759104619eb728b91a1cac5b665f",
    "Random_UnitCell_sigma_10.npy": "eb8102b580001bab80ee99bfc33289f727491e1d8edab2a4a94abc37a348fa1a",
}
MIRROR = "https://huggingface.co/datasets/asatheesh/PICore/tree/main/original_data"
OFFICIAL_CODE = "https://github.com/thuml/Transolver/blob/main/PDE-Solving-StandardBenchmark/exp_elas.py"
PAPER = "https://arxiv.org/abs/2402.02366"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=here / "source" / "original_data")
    parser.add_argument("--train-output", type=Path, default=here / "elasticity_train.h5")
    parser.add_argument("--test-output", type=Path, default=here / "elasticity_test.h5")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_hdf5(
    output: Path,
    xy: np.ndarray,
    sigma: np.ndarray,
    source_indices: np.ndarray,
    role: str,
    provenance: dict[str, object],
) -> None:
    """Write one self-contained suite dataset without changing loader semantics."""
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    chain_edges = np.stack(
        [np.arange(0, 971, dtype=np.int64), np.arange(1, 972, dtype=np.int64)], axis=0)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format"] = "cae_ml_suite_mesh_hdf5_v1"
            handle.attrs["benchmark"] = "elasticity"
            handle.attrs["benchmark_role"] = role
            data_group = handle.create_group("data")
            topology = handle.create_group("topology")
            topology.create_dataset("mesh_edge_chain", data=chain_edges, compression="gzip")
            metadata = handle.create_group("metadata")
            metadata.create_dataset("provenance_json", data=json.dumps(provenance, sort_keys=True))

            for sample_id, source_index in enumerate(source_indices.tolist()):
                group = data_group.create_group(str(sample_id))
                nodal = np.zeros((4, 1, 972), dtype=np.float32)
                nodal[0:2, 0, :] = np.asarray(xy[:, :, source_index].T, dtype=np.float32)
                nodal[3, 0, :] = np.asarray(sigma[:, source_index], dtype=np.float32)
                group.create_dataset(
                    "nodal_data", data=nodal, compression="gzip", compression_opts=4,
                    shuffle=True,
                )
                group["mesh_edge"] = topology["mesh_edge_chain"]
                group.attrs["source_index"] = source_index
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    source_paths = {name: args.source_dir / name for name in FILES}
    for name, expected in FILES.items():
        path = source_paths[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source file: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, got {actual}")

    xy = np.load(source_paths["Random_UnitCell_XY_10.npy"], mmap_mode="r")
    sigma = np.load(source_paths["Random_UnitCell_sigma_10.npy"], mmap_mode="r")
    if xy.shape != (972, 2, 2000) or sigma.shape != (972, 2000):
        raise ValueError(f"Unexpected source shapes: XY={xy.shape}, sigma={sigma.shape}")

    training_pool_source = np.arange(0, 1250, dtype=np.int64)
    test_source = np.arange(1800, 2000, dtype=np.int64)
    train_output = args.train_output.resolve()
    test_output = args.test_output.resolve()
    for output in (train_output, test_output):
        if output.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it.")
        output.parent.mkdir(parents=True, exist_ok=True)

    provenance = {
        "benchmark": "Geo-FNO / Transolver Elasticity",
        "paper": PAPER,
        "official_code": OFFICIAL_CODE,
        "download_mirror": MIRROR,
        "source_sha256": FILES,
        "source_shapes": {"xy": list(xy.shape), "sigma": list(sigma.shape)},
        "published_train_source_indices": [0, 999],
        "suite_training_pool_source_indices": [0, 1249],
        "suite_training_pool_size": 1250,
        "suite_split": "unchanged seeded 80/10/10 split (1000/125/125)",
        "additional_training_pool_source_indices": [1000, 1249],
        "published_test_source_indices": [1800, 1999],
        "mesh_points_per_sample": 972,
        "topology_note": "Chain edges satisfy the shared HDF5 contract; all benchmarked operator cores use positional_features=0 and do not consume mesh_edge.",
        "comparability_note": "The official 200-case test set is exact and isolated. This is not a strict paper reproduction: the unchanged loader shuffles a 1,250-case pool before selecting 1,000 optimization cases, and the suite retains normalized-MSE training plus its native scheduler/runtime instead of the authors' decoded relative-L2 training recipe.",
    }
    write_hdf5(train_output, xy, sigma, training_pool_source, "suite_training_pool", provenance)
    write_hdf5(test_output, xy, sigma, test_source, "published_test", provenance)

    provenance_path = train_output.parent / "elasticity.provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {train_output} ({train_output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"Wrote {test_output} ({test_output.stat().st_size / 1024**2:.1f} MiB)")
    print("As-is training pool: 1,250 cases -> 1,000/125/125 seeded random split")
    print("Isolated published test: 200 cases (source indices 1800..1999)")


if __name__ == "__main__":
    main()
