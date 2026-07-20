#!/usr/bin/env python3
"""Convert the Geo-FNO Plasticity MAT artifact to the suite mesh HDF5 format.

The converter is intentionally dataset-only.  It does not change any model or
loader code.  The HDF5 state has four channels so every current temporal
backend can use the unchanged ``input_var == output_var`` rollout contract:

    [u_x, u_y, u_z=0, die_profile]

The die profile is static, so its one-step target delta is exactly zero.  It is
also stored once as ``data/{sample_id}/die_profile`` for direct inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


SOURCE_NAME = "plas_N987_T20.mat"
SOURCE_SHA256 = "b58681e7777531b34889da46b506d2f18845c5d7b194859394dfd958cf28178f"
SOURCE_SHAPES = {"input": (987, 101), "output": (987, 101, 31, 20, 4)}
NUM_SAMPLES = 987
GRID_X = 101
GRID_Y = 31
NUM_NODES = GRID_X * GRID_Y
NUM_TIMESTEPS = 20
NUM_FEATURES = 8
NUM_CELLS = (GRID_X - 1) * (GRID_Y - 1)
OFFICIAL_TRAIN = np.arange(0, 900, dtype=np.int64)
OFFICIAL_UNUSED = np.arange(900, 907, dtype=np.int64)
OFFICIAL_TEST = np.arange(907, 987, dtype=np.int64)
FEATURE_NAMES = (
    "x_ref_mm",
    "y_ref_mm",
    "z_ref_mm",
    "u_x_mm",
    "u_y_mm",
    "u_z_mm",
    "die_profile_mm",
    "node_type",
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=here / SOURCE_NAME)
    parser.add_argument("--output", type=Path, default=here / "plasticity.h5")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete the checksum-verified MAT file only after exhaustive HDF5 validation.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structured_quad_topology() -> tuple[np.ndarray, np.ndarray]:
    """Return unique undirected perimeter edges and CPS4R-style quad cells."""
    node_ids = np.arange(NUM_NODES, dtype=np.int64).reshape(GRID_X, GRID_Y)
    vertical = np.stack(
        [node_ids[:, :-1].reshape(-1), node_ids[:, 1:].reshape(-1)], axis=0
    )
    horizontal = np.stack(
        [node_ids[:-1, :].reshape(-1), node_ids[1:, :].reshape(-1)], axis=0
    )
    edges = np.concatenate([vertical, horizontal], axis=1)
    cells = np.stack(
        [
            node_ids[:-1, :-1].reshape(-1),
            node_ids[1:, :-1].reshape(-1),
            node_ids[1:, 1:].reshape(-1),
            node_ids[:-1, 1:].reshape(-1),
        ],
        axis=0,
    )
    if edges.shape != (2, 6130) or cells.shape != (4, NUM_CELLS):
        raise AssertionError(f"Unexpected topology shapes: edges={edges.shape}, cells={cells.shape}")
    return edges, cells


def split_name(source_index: int) -> str:
    if source_index < 900:
        return "official_train"
    if source_index < 907:
        return "official_unused"
    return "official_test"


def make_nodal_case(
    die_profile: np.ndarray, raw_output: np.ndarray
) -> tuple[np.ndarray, float]:
    """Map one raw case to [features, time, nodes]."""
    current_xy = np.asarray(raw_output[..., 0:2], dtype=np.float64)
    displacement_xy = np.asarray(raw_output[..., 2:4], dtype=np.float64)
    reference_by_time = current_xy - displacement_xy
    reference_xy = np.mean(reference_by_time, axis=2)
    reference_variation = float(
        np.max(np.abs(reference_by_time - reference_xy[:, :, None, :]))
    )

    nodal = np.zeros((NUM_FEATURES, NUM_TIMESTEPS, NUM_NODES), dtype=np.float32)
    ref_flat = reference_xy.reshape(NUM_NODES, 2)
    nodal[0, :, :] = ref_flat[:, 0][None, :]
    nodal[1, :, :] = ref_flat[:, 1][None, :]

    disp_time_nodes = np.transpose(displacement_xy, (2, 0, 1, 3)).reshape(
        NUM_TIMESTEPS, NUM_NODES, 2
    )
    nodal[3, :, :] = disp_time_nodes[:, :, 0]
    nodal[4, :, :] = disp_time_nodes[:, :, 1]

    profile_nodes = np.repeat(
        np.asarray(die_profile, dtype=np.float32)[:, None], GRID_Y, axis=1
    ).reshape(NUM_NODES)
    nodal[6, :, :] = profile_nodes[None, :]
    # Features 2 (z), 5 (u_z), and 7 (node type) remain zero.
    return nodal, reference_variation


def provenance() -> dict[str, object]:
    return {
        "benchmark": "Geo-FNO / Transolver transient Plasticity",
        "source_file": SOURCE_NAME,
        "source_sha256": SOURCE_SHA256,
        "source_shapes": {key: list(value) for key, value in SOURCE_SHAPES.items()},
        "official_code": (
            "https://github.com/neuraloperator/Geo-FNO/blob/main/"
            "plasticity/plasticity_3d.py"
        ),
        "paper": "https://jmlr.org/papers/volume24/23-0064/23-0064.pdf",
        "download_mirror": (
            "https://huggingface.co/datasets/kmario23/standard-pde-benchmark/"
            "blob/main/plasticity/plas_N987_T20.mat"
        ),
        "logical_grid": [GRID_X, GRID_Y],
        "nodes_per_sample": NUM_NODES,
        "quad_cells_per_sample": NUM_CELLS,
        "undirected_mesh_edges_per_sample": 6130,
        "time_states": NUM_TIMESTEPS,
        "normalized_time_coordinates": np.linspace(0.0, 1.0, NUM_TIMESTEPS).tolist(),
        "nodal_feature_order": list(FEATURE_NAMES),
        "model_state_order": ["u_x_mm", "u_y_mm", "u_z_mm", "die_profile_mm"],
        "recommended_input_var": 4,
        "recommended_output_var": 4,
        "metric_channels": ["u_x_mm", "u_y_mm", "u_z_mm"],
        "conditioning_channel": "die_profile_mm (static; target delta is zero)",
        "coordinate_reconstruction": (
            "reference_xy = mean_t(raw_current_xy - raw_displacement_xy); "
            "current_xy = reference_xy + displacement_xy"
        ),
        "official_split": {
            "train_source_indices": [0, 899],
            "unused_source_indices": [900, 906],
            "test_source_indices": [907, 986],
        },
        "loader_note": (
            "The current suite loaders use their own seeded 80/10/10 split and do not "
            "consume stored split metadata. The official source split is retained for "
            "benchmark tooling and provenance."
        ),
    }


def create_hdf5(
    temporary: Path, die_inputs: np.ndarray, raw_outputs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    edges, cells = structured_quad_topology()
    prov = provenance()
    feature_min = np.full(NUM_FEATURES, np.inf, dtype=np.float64)
    feature_max = np.full(NUM_FEATURES, -np.inf, dtype=np.float64)
    feature_sum = np.zeros(NUM_FEATURES, dtype=np.float64)
    feature_sumsq = np.zeros(NUM_FEATURES, dtype=np.float64)
    feature_count = 0
    max_reference_variation = 0.0

    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = "cae_ml_suite_mesh_hdf5_v1"
        handle.attrs["benchmark"] = "plasticity"
        handle.attrs["num_samples"] = NUM_SAMPLES
        handle.attrs["num_features"] = NUM_FEATURES
        handle.attrs["num_timesteps"] = NUM_TIMESTEPS
        handle.attrs["num_nodes_per_sample"] = NUM_NODES
        handle.attrs["state_channels"] = 4
        handle.attrs["source_sha256"] = SOURCE_SHA256
        handle.attrs["source_mat_retained"] = False

        data_group = handle.create_group("data")
        topology = handle.create_group("topology")
        topology.create_dataset(
            "mesh_edge_structured_quad", data=edges, compression="gzip", shuffle=True
        )
        topology.create_dataset("quad_cells", data=cells, compression="gzip", shuffle=True)

        metadata = handle.create_group("metadata")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        metadata.create_dataset(
            "feature_names", data=np.asarray(FEATURE_NAMES, dtype=object), dtype=string_dtype
        )
        metadata.create_dataset(
            "time_normalized", data=np.linspace(0.0, 1.0, NUM_TIMESTEPS, dtype=np.float32)
        )
        metadata.create_dataset("provenance_json", data=json.dumps(prov, sort_keys=True))

        splits = handle.create_group("splits")
        splits.create_dataset("train", data=OFFICIAL_TRAIN)
        splits.create_dataset("val", data=np.empty(0, dtype=np.int64))
        splits.create_dataset("test", data=OFFICIAL_TEST)
        splits.create_dataset("unused", data=OFFICIAL_UNUSED)
        metadata["splits"] = splits

        for source_index in range(NUM_SAMPLES):
            nodal, reference_variation = make_nodal_case(
                die_inputs[source_index], raw_outputs[source_index]
            )
            max_reference_variation = max(max_reference_variation, reference_variation)

            group = data_group.create_group(str(source_index))
            group.create_dataset(
                "nodal_data",
                data=nodal,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                chunks=(NUM_FEATURES, 1, NUM_NODES),
            )
            group["mesh_edge"] = topology["mesh_edge_structured_quad"]
            group.create_dataset(
                "die_profile",
                data=np.asarray(die_inputs[source_index], dtype=np.float32),
                compression="gzip",
                shuffle=True,
            )
            sample_metadata = group.create_group("metadata")
            sample_metadata.attrs["source_filename"] = SOURCE_NAME
            sample_metadata.attrs["source_index"] = source_index
            sample_metadata.attrs["split"] = split_name(source_index)
            sample_metadata.attrs["num_nodes"] = NUM_NODES
            sample_metadata.attrs["num_edges"] = edges.shape[1]
            sample_metadata.attrs["num_cells"] = NUM_CELLS
            sample_metadata.attrs["num_timesteps"] = NUM_TIMESTEPS
            sample_metadata.attrs["max_reference_time_variation"] = reference_variation

            case_min = np.min(nodal, axis=(1, 2)).astype(np.float32)
            case_max = np.max(nodal, axis=(1, 2)).astype(np.float32)
            case_mean = np.mean(nodal, axis=(1, 2), dtype=np.float64).astype(np.float32)
            case_std = np.std(nodal, axis=(1, 2), dtype=np.float64).astype(np.float32)
            sample_metadata.create_dataset("feature_min", data=case_min)
            sample_metadata.create_dataset("feature_max", data=case_max)
            sample_metadata.create_dataset("feature_mean", data=case_mean)
            sample_metadata.create_dataset("feature_std", data=case_std)

            flat = nodal.reshape(NUM_FEATURES, -1).astype(np.float64, copy=False)
            feature_min = np.minimum(feature_min, np.min(flat, axis=1))
            feature_max = np.maximum(feature_max, np.max(flat, axis=1))
            feature_sum += np.sum(flat, axis=1)
            feature_sumsq += np.sum(flat * flat, axis=1)
            feature_count += flat.shape[1]

        global_mean = feature_sum / feature_count
        global_var = np.maximum(feature_sumsq / feature_count - global_mean**2, 0.0)
        normalization = metadata.create_group("normalization_params")
        normalization.create_dataset("min", data=feature_min.astype(np.float32))
        normalization.create_dataset("max", data=feature_max.astype(np.float32))
        normalization.create_dataset("mean", data=global_mean.astype(np.float32))
        normalization.create_dataset("std", data=np.sqrt(global_var).astype(np.float32))
        handle.attrs["max_reference_time_variation"] = max_reference_variation

    return edges, cells, feature_min, max_reference_variation


def audit_hdf5(
    path: Path, die_inputs: np.ndarray, raw_outputs: np.ndarray, expected_edges: np.ndarray
) -> dict[str, float]:
    max_displacement_error = 0.0
    max_die_error = 0.0
    max_coordinate_reconstruction_error = 0.0
    with h5py.File(path, "r") as handle:
        if set(handle.keys()) != {"data", "metadata", "splits", "topology"}:
            raise ValueError(f"Unexpected root groups: {sorted(handle.keys())}")
        if len(handle["data"]) != NUM_SAMPLES:
            raise ValueError(f"Expected {NUM_SAMPLES} cases, found {len(handle['data'])}")
        if not np.array_equal(handle["topology/mesh_edge_structured_quad"][:], expected_edges):
            raise ValueError("Stored structured topology does not match the generated topology")
        if not np.array_equal(handle["splits/train"][:], OFFICIAL_TRAIN):
            raise ValueError("Official train split mismatch")
        if not np.array_equal(handle["splits/test"][:], OFFICIAL_TEST):
            raise ValueError("Official test split mismatch")
        if not np.array_equal(handle["splits/unused"][:], OFFICIAL_UNUSED):
            raise ValueError("Official unused split mismatch")

        for source_index in range(NUM_SAMPLES):
            group = handle[f"data/{source_index}"]
            nodal = group["nodal_data"][:]
            if nodal.shape != (NUM_FEATURES, NUM_TIMESTEPS, NUM_NODES):
                raise ValueError(f"Case {source_index}: invalid nodal shape {nodal.shape}")
            if group["mesh_edge"].shape != (2, 6130):
                raise ValueError(f"Case {source_index}: invalid edge shape")
            if not np.all(np.isfinite(nodal)):
                raise ValueError(f"Case {source_index}: non-finite nodal data")
            if np.any(nodal[[2, 5, 7]] != 0.0):
                raise ValueError(f"Case {source_index}: padded z/node-type channels are not zero")

            expected_displacement = np.transpose(
                raw_outputs[source_index, ..., 2:4], (2, 0, 1, 3)
            ).reshape(NUM_TIMESTEPS, NUM_NODES, 2)
            stored_displacement = np.stack([nodal[3], nodal[4]], axis=-1)
            max_displacement_error = max(
                max_displacement_error,
                float(np.max(np.abs(stored_displacement - expected_displacement))),
            )

            expected_die = np.asarray(die_inputs[source_index], dtype=np.float32)
            max_die_error = max(
                max_die_error,
                float(np.max(np.abs(group["die_profile"][:] - expected_die))),
            )
            profile_nodes = np.repeat(expected_die[:, None], GRID_Y, axis=1).reshape(NUM_NODES)
            max_die_error = max(
                max_die_error,
                float(np.max(np.abs(nodal[6] - profile_nodes[None, :]))),
            )

            stored_ref = np.stack([nodal[0, 0], nodal[1, 0]], axis=-1).reshape(
                GRID_X, GRID_Y, 2
            )
            reconstructed = stored_ref[:, :, None, :] + raw_outputs[source_index, ..., 2:4]
            max_coordinate_reconstruction_error = max(
                max_coordinate_reconstruction_error,
                float(np.max(np.abs(reconstructed - raw_outputs[source_index, ..., 0:2]))),
            )

    if max_displacement_error > 1e-6:
        raise ValueError(f"Displacement conversion error too large: {max_displacement_error}")
    if max_die_error > 1e-6:
        raise ValueError(f"Die-profile conversion error too large: {max_die_error}")
    if max_coordinate_reconstruction_error > 3e-5:
        raise ValueError(
            "Coordinate reconstruction error too large: "
            f"{max_coordinate_reconstruction_error}"
        )
    return {
        "max_displacement_error": max_displacement_error,
        "max_die_profile_error": max_die_error,
        "max_coordinate_reconstruction_error": max_coordinate_reconstruction_error,
    }


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source MAT file: {source}")
    actual_sha = sha256(source)
    if actual_sha != SOURCE_SHA256:
        raise ValueError(
            f"Source SHA256 mismatch: expected {SOURCE_SHA256}, got {actual_sha}"
        )
    if output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    raw = loadmat(source, variable_names=("input", "output"))
    if set(SOURCE_SHAPES) - raw.keys():
        raise ValueError(f"MAT file is missing required fields: {sorted(set(SOURCE_SHAPES) - raw.keys())}")
    die_inputs = np.asarray(raw["input"])
    raw_outputs = np.asarray(raw["output"])
    if die_inputs.shape != SOURCE_SHAPES["input"] or raw_outputs.shape != SOURCE_SHAPES["output"]:
        raise ValueError(
            f"Unexpected source shapes: input={die_inputs.shape}, output={raw_outputs.shape}"
        )
    if not np.all(np.isfinite(die_inputs)) or not np.all(np.isfinite(raw_outputs)):
        raise ValueError("Source arrays contain non-finite values")

    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        edges, _, _, max_reference_variation = create_hdf5(
            temporary, die_inputs, raw_outputs
        )
        audit = audit_hdf5(temporary, die_inputs, raw_outputs, edges)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    prov = provenance()
    prov["conversion_audit"] = audit
    prov["max_reference_time_variation"] = max_reference_variation
    prov["converted_hdf5_sha256"] = sha256(output)
    prov["source_mat_deleted"] = bool(args.delete_source)
    provenance_path = output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.delete_source:
        # The exact source was resolved, checksum-verified, converted, and audited above.
        source.unlink()

    print(f"Wrote {output} ({output.stat().st_size / 1024**2:.1f} MiB)")
    print(f"HDF5 SHA256: {prov['converted_hdf5_sha256']}")
    print(f"Samples/times/nodes: {NUM_SAMPLES}/{NUM_TIMESTEPS}/{NUM_NODES}")
    print(f"Nodal layout: ({NUM_FEATURES}, {NUM_TIMESTEPS}, {NUM_NODES}) per sample")
    print(f"Max displacement conversion error: {audit['max_displacement_error']:.3e}")
    print(
        "Max coordinate reconstruction error: "
        f"{audit['max_coordinate_reconstruction_error']:.3e}"
    )
    print(f"Source MAT deleted: {args.delete_source}")


if __name__ == "__main__":
    main()
