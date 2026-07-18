#!/usr/bin/env python3
"""Small process-isolated HDF5 schema probe used by the top-level launcher."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def _mesh_report(handle):
    errors = []
    warnings = []
    metadata = {}
    if "data" not in handle:
        return {"errors": ["Missing root group 'data'."], "warnings": warnings, "metadata": metadata}
    keys = list(handle["data"].keys())
    numeric_keys = [key for key in keys if str(key).isdigit()]
    if not numeric_keys:
        return {"errors": ["Group 'data' has no numeric sample IDs."], "warnings": warnings, "metadata": metadata}
    sample_id = sorted(numeric_keys, key=int)[0]
    sample = handle["data"][sample_id]
    if "nodal_data" not in sample:
        errors.append(f"Sample {sample_id} is missing nodal_data.")
    if "mesh_edge" not in sample:
        errors.append(f"Sample {sample_id} is missing mesh_edge.")
    if not errors:
        nodal_shape = tuple(int(v) for v in sample["nodal_data"].shape)
        edge_shape = tuple(int(v) for v in sample["mesh_edge"].shape)
        metadata.update({"sample_count": len(numeric_keys), "sample_id": sample_id, "nodal_shape": nodal_shape, "edge_shape": edge_shape})
        if len(nodal_shape) != 3:
            errors.append(f"nodal_data must have rank 3 [F,T,N]; got {nodal_shape}.")
        if len(edge_shape) != 2 or edge_shape[0] != 2:
            errors.append(f"mesh_edge must have shape [2,E]; got {edge_shape}.")
    return {"errors": errors, "warnings": warnings, "metadata": metadata}


def _sdf_report(handle):
    errors = []
    warnings = []
    metadata = {}
    if "shapes" not in handle:
        return {"errors": ["Missing root group 'shapes'."], "warnings": warnings, "metadata": metadata}
    keys = list(handle["shapes"].keys())
    if not keys:
        return {"errors": ["Group 'shapes' contains no shapes."], "warnings": warnings, "metadata": metadata}
    sample_id = sorted(keys)[0]
    shape = handle["shapes"][sample_id]
    required = ("surface_points", "surface_normals", "sdf_points", "sdf_values", "cond")
    for name in required:
        if name not in shape:
            errors.append(f"Shape {sample_id} is missing {name}.")
    if not errors:
        metadata = {"shape_count": len(keys), "shape_id": sample_id, "arrays": {name: tuple(int(v) for v in shape[name].shape) for name in required}}
        points = metadata["arrays"]["surface_points"]
        normals = metadata["arrays"]["surface_normals"]
        if points != normals:
            errors.append(f"surface_points and surface_normals shapes differ: {points} vs {normals}.")
    return {"errors": errors, "warnings": warnings, "metadata": metadata}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print(json.dumps({"errors": ["usage: dataset_probe.py <mesh_hdf5|sdf_hdf5> <path>"], "warnings": [], "metadata": {}}))
        return 2
    kind, raw_path = argv
    path = Path(raw_path)
    try:
        import h5py
        with h5py.File(path, "r") as handle:
            result = _mesh_report(handle) if kind == "mesh_hdf5" else _sdf_report(handle)
    except Exception as exc:
        result = {"errors": [f"Could not inspect HDF5: {type(exc).__name__}: {exc}"], "warnings": [], "metadata": {}}
    print(json.dumps(result))
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
