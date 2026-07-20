#!/usr/bin/env python3
"""Convert the official GINO CarCFD archive into benchmark HDF5.

The source archive is the official Zenodo release used by the maintained
NeuralOperator loader.  This converter deliberately preserves ``train.txt``
(500 cases) and ``test.txt`` (111 cases); it never routes those cases through
the suite's generic random splitter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import trimesh

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


ARCHIVE_MD5 = "24a46fe791085201d48ee5db7b6cfc86"
OFFICIAL_TRAIN_COUNT = 500
OFFICIAL_TEST_COUNT = 111
VERTEX_COUNT = 3586
FACE_COUNT = 7168
RAW_PRESSURE_COUNT = 3682
PRESSURE_MEAN = -37.11484334643704
# Reproduces the value returned by the released loader's incremental fitter.
PRESSURE_STD = 48.115568258070894
PRESSURE_EPS = 1.0e-7
PRESSURE_CROP = "concatenate(pressure[:16], pressure[112:])"
ZENODO_URL = "https://zenodo.org/records/13936501"
PAPER_ERA_ARCHITECTURE_COMMIT = "957f0b0fe540bf167f6138494297073d8aa97d98"
MAINTAINED_RECIPE_COMMIT = "86a8bc7812a31b42c4f7895693cf4ac11521c066"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_source = here / "source" / "extracted" / "processed-car-pressure-data"
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=default_source)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--sdf-chunk-size", type=int, default=32768)
    parser.add_argument(
        "--sdf-backend",
        choices=("open3d", "trimesh"),
        default="open3d",
        help=(
            "open3d matches the maintained NeuralOperator reference; trimesh is "
            "diagnostic-only and will not produce a paper-protocol artifact."
        ),
    )
    parser.add_argument(
        "--limit-train", type=int, default=0,
        help="Diagnostic-only prefix of train.txt; zero means all 500 cases.",
    )
    parser.add_argument(
        "--limit-test", type=int, default=0,
        help="Diagnostic-only prefix of test.txt; zero means all 111 cases.",
    )
    parser.add_argument("--skip-archive-md5", action="store_true")
    return parser.parse_args()


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[str]:
    values = [value.strip() for value in path.read_text(encoding="utf-8").split(",")]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return values


def pressure_for_case(data_root: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(np.load(data_root / f"press_{case_id}.npy"), dtype=np.float64).reshape(-1)
    if full.shape != (RAW_PRESSURE_COUNT,):
        raise ValueError(
            f"press_{case_id}.npy has shape {full.shape}, expected ({RAW_PRESSURE_COUNT},)."
        )
    cropped = np.concatenate((full[:16], full[112:]))
    if cropped.shape != (VERTEX_COUNT,):
        raise AssertionError(f"Pressure crop for {case_id} produced {cropped.shape}.")
    return full, cropped


def mesh_for_case(data_root: Path, case_id: str) -> trimesh.Trimesh:
    loaded = trimesh.load(
        data_root / f"mesh_{case_id}.ply",
        process=False,
        maintain_order=True,
    )
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"mesh_{case_id}.ply did not load as a single Trimesh.")
    if loaded.vertices.shape != (VERTEX_COUNT, 3):
        raise ValueError(
            f"mesh_{case_id}.ply has vertices {loaded.vertices.shape}, expected ({VERTEX_COUNT}, 3)."
        )
    if loaded.faces.shape != (FACE_COUNT, 3):
        raise ValueError(
            f"mesh_{case_id}.ply has faces {loaded.faces.shape}, expected ({FACE_COUNT}, 3)."
        )
    if not loaded.is_watertight:
        raise ValueError(f"mesh_{case_id}.ply is not watertight; signed distance is undefined.")
    return loaded


def signed_distance(
    mesh: trimesh.Trimesh,
    grid_points: np.ndarray,
    chunk_size: int,
    backend: str = "open3d",
) -> np.ndarray:
    values = np.empty(grid_points.shape[0], dtype=np.float32)
    if backend == "open3d":
        if o3d is None:
            raise ModuleNotFoundError(
                "Strict CarCFD SDF generation requires open3d. Install open3d or "
                "pass --sdf-backend trimesh for a diagnostic-only artifact."
            )
        legacy = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32)),
        )
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tensor_mesh)
    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        if backend == "open3d":
            query = o3d.core.Tensor(
                np.asarray(grid_points[start:end], dtype=np.float32),
                dtype=o3d.core.Dtype.Float32,
            )
            values[start:end] = scene.compute_signed_distance(query).numpy()
        elif backend == "trimesh":
            # trimesh is positive inside; Open3D/the released loader is negative inside.
            values[start:end] = -trimesh.proximity.signed_distance(
                mesh, grid_points[start:end]
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown SDF backend {backend!r}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Signed-distance evaluation produced NaN or Inf.")
    return values


def verify_pressure_normalizer(data_root: Path, train_ids: list[str]) -> dict[str, float]:
    total = 0.0
    square_total = 0.0
    count = 0
    for case_id in train_ids:
        full, _ = pressure_for_case(data_root, case_id)
        total += float(np.sum(full, dtype=np.float64))
        square_total += float(np.sum(full * full, dtype=np.float64))
        count += full.size
    mean = total / count
    population_std = math_sqrt_nonnegative(square_total / count - mean * mean)
    if not np.isclose(mean, PRESSURE_MEAN, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"Official pressure mean drifted: observed {mean:.15g}, expected {PRESSURE_MEAN:.15g}."
        )
    return {"computed_full_train_mean": mean, "computed_population_std": population_std}


def math_sqrt_nonnegative(value: float) -> float:
    return float(np.sqrt(max(0.0, value)))


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    data_root = source_root / "data"
    if not data_root.is_dir():
        raise FileNotFoundError(f"Canonical source data directory not found: {data_root}")
    # The archive contains a duplicate data/data tree.  Never discover files
    # recursively: the canonical outer directory above is authoritative.

    archive = source_root.parent.parent / "processed-car-pressure-data.tar.gz"
    if archive.exists() and not args.skip_archive_md5:
        observed_md5 = md5(archive)
        if observed_md5 != ARCHIVE_MD5:
            raise ValueError(f"Archive MD5 {observed_md5} != official {ARCHIVE_MD5}.")

    train_all = read_manifest(source_root / "train.txt")
    test_all = read_manifest(source_root / "test.txt")
    if len(train_all) != OFFICIAL_TRAIN_COUNT or len(test_all) != OFFICIAL_TEST_COUNT:
        raise ValueError(
            f"Official manifests must be 500/111, got {len(train_all)}/{len(test_all)}."
        )
    overlap = sorted(set(train_all) & set(test_all))
    if overlap:
        raise ValueError(f"Official train/test manifests overlap: {overlap[:10]}")

    train_ids = train_all[: args.limit_train or None]
    test_ids = test_all[: args.limit_test or None]
    diagnostic = (
        len(train_ids) != OFFICIAL_TRAIN_COUNT
        or len(test_ids) != OFFICIAL_TEST_COUNT
        or args.sdf_backend != "open3d"
    )
    if args.sdf_backend == "open3d" and o3d is None:
        raise ModuleNotFoundError(
            "--sdf-backend open3d was selected but open3d is not installed. "
            "The Trimesh backend is diagnostic-only."
        )
    resolution = int(args.resolution)
    if resolution < 2:
        raise ValueError("--resolution must be >= 2")
    output = (
        args.output
        or source_root.parent.parent.parent / f"carcfd_{'diagnostic_' if diagnostic else 'paper_'}r{resolution}.h5"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    bounds = np.asarray(np.loadtxt(source_root / "watertight_global_bounds.txt"), dtype=np.float64)
    if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
        raise ValueError(f"Invalid global bounds: shape={bounds.shape}, values={bounds}")
    axes = [np.linspace(bounds[0, axis], bounds[1, axis], resolution) for axis in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    grid_points = grid.reshape(-1, 3)

    pressure_audit = (
        verify_pressure_normalizer(data_root, train_all)
        if not diagnostic
        else {"computed_full_train_mean": None, "computed_population_std": None}
    )
    sdf_min = np.inf
    sdf_max = -np.inf

    with h5py.File(output, "w") as handle:
        handle.attrs["benchmark"] = "GINO ShapeNet Car pressure"
        handle.attrs["benchmark_protocol"] = (
            "gino_carcfd_hybrid_decoder_v1" if not diagnostic and resolution == 64
            else "gino_carcfd_diagnostic_v1"
        )
        handle.attrs["paper_target_mean_relative_l2"] = 0.0712
        handle.attrs["zenodo_url"] = ZENODO_URL
        handle.attrs["archive_md5"] = ARCHIVE_MD5
        handle.attrs["paper_era_architecture_commit"] = PAPER_ERA_ARCHITECTURE_COMMIT
        handle.attrs["maintained_recipe_commit"] = MAINTAINED_RECIPE_COMMIT
        handle.attrs["canonical_source_tree"] = "processed-car-pressure-data/data (outer tree only)"
        handle.attrs["pressure_crop"] = PRESSURE_CROP
        handle.attrs["pressure_mean"] = PRESSURE_MEAN
        handle.attrs["pressure_std"] = PRESSURE_STD
        handle.attrs["pressure_eps"] = PRESSURE_EPS
        handle.attrs["sdf_sign"] = "negative_inside"
        handle.attrs["sdf_backend"] = args.sdf_backend
        handle.attrs["strict_reference_sdf"] = args.sdf_backend == "open3d"
        handle.attrs["sdf_normalization"] = "train-global min/max to [1e-6, 1]"
        handle.attrs["coordinate_normalization"] = "raw global bounds to [-1, 1]^3"
        handle.attrs["grid_resolution"] = resolution
        handle.attrs["official_train_count"] = OFFICIAL_TRAIN_COUNT
        handle.attrs["official_test_count"] = OFFICIAL_TEST_COUNT
        handle.attrs["converted_train_count"] = len(train_ids)
        handle.attrs["converted_test_count"] = len(test_ids)
        handle.attrs["diagnostic_only"] = diagnostic or resolution != 64
        handle.attrs["global_bounds_min"] = bounds[0]
        handle.attrs["global_bounds_max"] = bounds[1]
        handle.attrs["pressure_audit_json"] = json.dumps(pressure_audit)

        string_dtype = h5py.string_dtype(encoding="utf-8")
        splits = handle.create_group("splits")
        splits.create_dataset("train_ids", data=np.asarray(train_ids, dtype=object), dtype=string_dtype)
        splits.create_dataset("test_ids", data=np.asarray(test_ids, dtype=object), dtype=string_dtype)
        data_group = handle.create_group("data")

        all_cases = [("train", case_id) for case_id in train_ids]
        all_cases.extend(("test", case_id) for case_id in test_ids)
        for index, (split, case_id) in enumerate(all_cases, start=1):
            print(f"[{index:03d}/{len(all_cases):03d}] {split} case {case_id}", flush=True)
            mesh = mesh_for_case(data_root, case_id)
            _, pressure = pressure_for_case(data_root, case_id)
            raw_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            normalized_vertices = (
                2.0 * (raw_vertices - bounds[0]) / (bounds[1] - bounds[0]) - 1.0
            ).astype(np.float32)
            sdf = signed_distance(
                mesh,
                grid_points,
                int(args.sdf_chunk_size),
                backend=args.sdf_backend,
            ).reshape(
                resolution, resolution, resolution
            )
            if split == "train":
                sdf_min = min(sdf_min, float(np.min(sdf)))
                sdf_max = max(sdf_max, float(np.max(sdf)))

            group = data_group.create_group(case_id)
            group.attrs["split"] = split
            group.attrs["mesh_source"] = f"data/mesh_{case_id}.ply"
            group.attrs["pressure_source"] = f"data/press_{case_id}.npy"
            group.create_dataset("pos", data=normalized_vertices, compression="gzip", shuffle=True)
            group.create_dataset("pos_raw", data=raw_vertices, compression="gzip", shuffle=True)
            group.create_dataset(
                "pressure", data=pressure.astype(np.float32), compression="gzip", shuffle=True
            )
            group.create_dataset(
                "faces", data=np.asarray(mesh.faces, dtype=np.int32), compression="gzip", shuffle=True
            )
            group.create_dataset("sdf", data=sdf, compression="gzip", shuffle=True)

        if not np.isfinite(sdf_min) or not np.isfinite(sdf_max) or sdf_max <= sdf_min:
            raise ValueError(f"Invalid train SDF range [{sdf_min}, {sdf_max}].")
        handle.attrs["sdf_train_min"] = sdf_min
        handle.attrs["sdf_train_max"] = sdf_max
        scale = sdf_max - sdf_min
        for case_id in train_ids + test_ids:
            dataset = data_group[case_id]["sdf"]
            raw = np.asarray(dataset, dtype=np.float32)
            normalized = 1.0e-6 + (raw - sdf_min) * ((1.0 - 1.0e-6) / scale)
            dataset[...] = normalized.astype(np.float32)

    provenance = {
        "output": str(output),
        "protocol": "diagnostic" if diagnostic or resolution != 64 else "paper_decoder",
        "resolution": resolution,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "official_manifest_counts": [OFFICIAL_TRAIN_COUNT, OFFICIAL_TEST_COUNT],
        "pressure_mean": PRESSURE_MEAN,
        "pressure_std": PRESSURE_STD,
        "pressure_eps": PRESSURE_EPS,
        "sdf_train_min": sdf_min,
        "sdf_train_max": sdf_max,
        "sdf_backend": args.sdf_backend,
        "strict_reference_sdf": args.sdf_backend == "open3d",
        "trimesh_equivalence_claimed": False,
        "sdf_backend_note": (
            "Open3D RaycastingScene is the maintained-reference backend; "
            "Trimesh uses the same negative-inside sign after negation but is "
            "diagnostic-only because numerical equivalence is not assumed."
        ),
        "source": ZENODO_URL,
        "archive_md5": ARCHIVE_MD5,
        "paper_era_architecture_commit": PAPER_ERA_ARCHITECTURE_COMMIT,
        "maintained_recipe_commit": MAINTAINED_RECIPE_COMMIT,
    }
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
