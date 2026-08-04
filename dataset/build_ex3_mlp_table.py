#!/usr/bin/env python3
"""Build the ex3 scalar-QoI MLP table from a reordered full-resolution file.

X contains the six global flight/control conditions.  Y contains the
surface-area-weighted mean of the four aerodynamic fields (Cp and Cf_xyz).
This keeps the MLP in its intended scalar-parametric-surrogate role while using
targets derived exactly from the same cases as the mesh-field methods.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


INPUT_NAMES = (
    "mach",
    "aoa_deg",
    "aileron_inboard_deg",
    "aileron_outboard_deg",
    "elevator_deg",
    "htp_deg",
)
OUTPUT_NAMES = (
    "area_weighted_cp",
    "area_weighted_cf_x",
    "area_weighted_cf_y",
    "area_weighted_cf_z",
)
EXPECTED_FEATURES = (
    "x_coord", "y_coord", "z_coord",
    "cp", "cf_x", "cf_y", "cf_z",
    *INPUT_NAMES,
    "normal_x", "normal_y", "normal_z", "surface_area",
)


def _decode(values) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


def build_table(source_path: Path, output_path: Path) -> None:
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        partial_path.unlink()

    try:
        with h5py.File(source_path, "r") as source:
            names = _decode(source["metadata/feature_names"][...])
            if names != EXPECTED_FEATURES:
                raise ValueError(f"unexpected reordered ex3 features: {names}")
            sample_ids = sorted(source["data"].keys(), key=int)
            x = np.empty((len(sample_ids), len(INPUT_NAMES)), dtype=np.float32)
            y = np.empty((len(sample_ids), len(OUTPUT_NAMES)), dtype=np.float32)

            for row, sample_id in enumerate(sample_ids):
                nodal = source[f"data/{sample_id}/nodal_data"]
                conditions = np.asarray(nodal[7:13, 0, :], dtype=np.float32)
                spread = np.ptp(conditions, axis=1)
                scale = np.maximum(np.abs(conditions[:, 0]), 1.0)
                if np.any(spread > 1e-4 * scale):
                    raise ValueError(f"sample {sample_id}: global condition rows vary across nodes")
                x[row] = conditions[:, 0]

                fields = np.asarray(nodal[3:7, 0, :], dtype=np.float64)
                area = np.asarray(nodal[16, 0, :], dtype=np.float64)
                if not np.all(np.isfinite(fields)) or not np.all(np.isfinite(area)):
                    raise ValueError(f"sample {sample_id}: non-finite field or area values")
                area_sum = float(area.sum())
                if area_sum <= 0:
                    raise ValueError(f"sample {sample_id}: non-positive total surface area")
                y[row] = np.asarray((fields * area[None, :]).sum(axis=1) / area_sum, dtype=np.float32)

        with h5py.File(partial_path, "w") as output:
            output.create_dataset("X", data=x)
            output.create_dataset("Y", data=y)
            output.create_dataset("input_names", data=np.asarray(INPUT_NAMES, dtype="S"))
            output.create_dataset("output_names", data=np.asarray(OUTPUT_NAMES, dtype="S"))
            output.create_dataset("sample_ids", data=np.asarray(sample_ids, dtype="S"))
            output.attrs["source_file"] = source_path.name
            output.attrs["target_definition"] = "surface-area-weighted mean over full-resolution surface nodes"

        with h5py.File(partial_path, "r") as check:
            if check["X"].shape != x.shape or check["Y"].shape != y.shape:
                raise ValueError("written MLP table shape validation failed")
            if not np.array_equal(check["X"][...], x) or not np.array_equal(check["Y"][...], y):
                raise ValueError("written MLP table value validation failed")
        os.replace(partial_path, output_path)
        print(f"wrote {output_path}: X{x.shape}, Y{y.shape}")
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build_table(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
