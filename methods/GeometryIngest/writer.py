"""Write ingested geometry into the shared mesh HDF5 contract.

Layout (per dataset/DATASET_FORMAT.md):
    data/{id}/nodal_data   [F, T=1, N]  rows 0:3 = physical coords, 3: fields
    data/{id}/mesh_edge    [2, E]       (graph emit only)
    data/{id}/metadata     attrs: source_filename, num_nodes, num_edges, num_cells
    metadata/feature_names
    root attrs: num_samples, num_features, num_timesteps

Solution-field rows are zero-filled: a geometry carries no solution, so the file
is an *inference initial condition*. A solver or trained model fills the fields.
"""

from __future__ import annotations

import os

import numpy as np


def build_nodal_data(coords: np.ndarray, num_fields: int) -> np.ndarray:
    """[3 + num_fields, 1, N]: coords in rows 0:3, zero-filled fields after."""
    n = coords.shape[0]
    nodal = np.zeros((3 + num_fields, 1, n), dtype=np.float32)
    nodal[0:3, 0, :] = coords.T.astype(np.float32)
    return nodal


def _feature_names(num_fields: int) -> list[str]:
    return ["x", "y", "z"] + [f"field_{k}" for k in range(num_fields)]


def write_contract(out_path: str, samples: list[dict], num_fields: int,
                   with_edges: bool = True) -> None:
    """Write ``samples`` (dicts with 'coords', optional 'mesh_edge', metadata)."""
    import h5py

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with h5py.File(out_path, "w") as h5:
        data_grp = h5.require_group("data")
        for i, s in enumerate(samples, start=1):
            coords = s["coords"]
            g = data_grp.create_group(str(i))
            g.create_dataset("nodal_data", data=build_nodal_data(coords, num_fields),
                             compression="gzip", compression_opts=4)
            num_edges = 0
            if with_edges and s.get("mesh_edge") is not None:
                g.create_dataset("mesh_edge", data=s["mesh_edge"],
                                 compression="gzip", compression_opts=4)
                num_edges = int(s["mesh_edge"].shape[1])
            md = g.create_group("metadata")
            md.attrs["source_filename"] = s.get("source", "")
            md.attrs["num_nodes"] = int(coords.shape[0])
            md.attrs["num_edges"] = num_edges
            md.attrs["num_cells"] = int(s.get("num_cells", 0))

        meta = h5.require_group("metadata")
        meta.create_dataset("feature_names",
                            data=np.array(_feature_names(num_fields),
                                          dtype=h5py.string_dtype()))
        h5.attrs["num_samples"] = len(samples)
        h5.attrs["num_features"] = 3 + num_fields
        h5.attrs["num_timesteps"] = 1
