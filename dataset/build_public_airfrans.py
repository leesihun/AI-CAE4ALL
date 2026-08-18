#!/usr/bin/env python
"""Convert the AirfRANS dataset (VTU/VTP inside Dataset.zip) into the shared
mesh HDF5 contract (see dataset/DATASET_FORMAT.md, dataset/PUBLIC_DATASETS.md).

    python dataset/build_public_airfrans.py --task scarce_train
    python dataset/build_public_airfrans.py --task reynolds_train
    python dataset/build_public_airfrans.py --task aoa_test

`--task` names come straight from the dataset's own Dataset/manifest.json:
full_train, scarce_train, reynolds_train, aoa_train, full_test, reynolds_test,
aoa_test. Each task is written as its own file (split_strategy hdf5 is not
implemented in this runtime).

Row layout (13 features, matches dataset/PUBLIC_DATASETS.md):
    rows 0:3    x, y, z(=0)
    rows 3:7    u_x, u_y, p, nu_t              state       input_var=output_var=4
    rows 7:12   u_inf_x, u_inf_y, sdf, n_x, n_y  conditions  cond_var=5
    row  12     node_type                        0 volume, 1 airfoil surface

node_type is derived from `implicit_distance == 0` (the VTU's own SDF field is
exactly zero on the airfoil boundary -- verified empirically, not assumed).
n_x/n_y for volume points are nearest-neighbor-propagated from the surface
polydata's `Normals` array (the surface's own points get their own normal
back at distance 0), matching the AirfRANS paper's own feature construction.

Keep `augment_geometry False` downstream: u_inf and the normals are vector
components (see DATASET_FORMAT.md's rotation caveat for conditioning rows).
"""
import argparse
import io
import json
import os
import re
import tempfile
import zipfile

import h5py
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

FEATURE_NAMES = [
    "x_coord", "y_coord", "z_coord", "velocity_x", "velocity_y", "pressure", "nu_t",
    "u_inf_x", "u_inf_y", "sdf", "normal_x", "normal_y", "node_type",
]

NAME_RE = re.compile(r"^airFoil2D_SST_([-\d.]+)_([-\d.]+)_")


def read_vtk_member(z, member_name, suffix):
    data = z.read(member_name)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return pv.read(path)
    finally:
        os.unlink(path)


def quad_ring_edges(cells_dict):
    """Unique undirected boundary edges from VTK QUAD connectivity (ring order,
    not all C(4,2) pairs -- a quad's diagonals are not mesh edges)."""
    pairs = set()
    for cell_type, conn in cells_dict.items():
        if conn.shape[1] != 4:
            raise ValueError(f"expected quads (4 verts), got cell_type={cell_type} shape={conn.shape}")
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            u = conn[:, a].astype(np.int64)
            v = conn[:, b].astype(np.int64)
            lo = np.minimum(u, v)
            hi = np.maximum(u, v)
            pairs.update(zip(lo.tolist(), hi.tolist()))
    arr = np.array(sorted(pairs), dtype=np.int32).T
    return arr


def build_sample(z, case):
    m = NAME_RE.match(case)
    if not m:
        raise ValueError(f"unrecognized case name format: {case}")
    u_inf = float(m.group(1))
    aoa_deg = float(m.group(2))
    aoa_rad = np.deg2rad(aoa_deg)
    u_inf_x = u_inf * np.cos(aoa_rad)
    u_inf_y = u_inf * np.sin(aoa_rad)

    internal = read_vtk_member(z, f"Dataset/{case}/{case}_internal.vtu", ".vtu")
    surface = read_vtk_member(z, f"Dataset/{case}/{case}_aerofoil.vtp", ".vtp")

    pts = internal.points  # [N, 3]
    N = pts.shape[0]
    p = internal.point_data["p"].astype(np.float32)
    U = internal.point_data["U"].astype(np.float32)  # [N, 3]
    nut = internal.point_data["nut"].astype(np.float32)
    sdf = internal.point_data["implicit_distance"].astype(np.float32)
    node_type = (np.isclose(sdf, 0.0, atol=1e-6)).astype(np.float32)

    # Normals are zero everywhere except exactly at surface nodes (node_type==1)
    # -- verified against the existing ex5_airfrans_full_test.h5: normal magnitude
    # is 1.0 for exactly the surface-point fraction of nodes and 0.0 elsewhere, i.e.
    # NOT nearest-neighbor-propagated into the volume.
    surf_pts = surface.points[:, :2]
    surf_normals = surface.point_data["Normals"][:, :2].astype(np.float32)
    tree = cKDTree(surf_pts)
    normals = np.zeros((N, 2), dtype=np.float32)
    surf_node_idx = np.where(node_type > 0.5)[0]
    if surf_node_idx.size:
        dist, nn_idx = tree.query(pts[surf_node_idx, :2], k=1)
        assert dist.max() < 1e-6, f"surface node not coincident with aerofoil.vtp point, max dist={dist.max()}"
        normals[surf_node_idx] = surf_normals[nn_idx]

    nodal = np.zeros((13, 1, N), dtype=np.float32)
    nodal[0, 0, :] = pts[:, 0]
    nodal[1, 0, :] = pts[:, 1]
    # row 2 (z) stays 0 -- planar 2D case, matches existing ex5 files
    nodal[3, 0, :] = U[:, 0]
    nodal[4, 0, :] = U[:, 1]
    nodal[5, 0, :] = p
    nodal[6, 0, :] = nut
    nodal[7, 0, :] = u_inf_x
    nodal[8, 0, :] = u_inf_y
    nodal[9, 0, :] = sdf
    nodal[10, 0, :] = normals[:, 0]
    nodal[11, 0, :] = normals[:, 1]
    nodal[12, 0, :] = node_type

    edges = quad_ring_edges(internal.cells_dict)
    return nodal, edges, u_inf, aoa_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--raw-zip", default=os.environ.get("RAW", "D:/CAE_datasets_raw") + "/airfrans/Dataset.zip")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N cases (for smoke testing).")
    args = ap.parse_args()

    z = zipfile.ZipFile(args.raw_zip)
    with z.open("Dataset/manifest.json") as f:
        manifest = json.load(f)
    if args.task not in manifest:
        raise SystemExit(f"unknown task {args.task!r}; available: {sorted(manifest.keys())}")
    cases = manifest[args.task]
    if args.limit:
        cases = cases[: args.limit]

    out_name = args.out_name or f"ex5_airfrans_{args.task}.h5"
    out_path = os.path.join(args.out_dir, out_name)
    tmp_path = out_path + ".tmp"

    n_features = len(FEATURE_NAMES)
    running = dict(
        count=0,
        sum=np.zeros(n_features, dtype=np.float64),
        sumsq=np.zeros(n_features, dtype=np.float64),
        min=np.full(n_features, np.inf, dtype=np.float64),
        max=np.full(n_features, -np.inf, dtype=np.float64),
    )

    with h5py.File(tmp_path, "w") as f:
        data_grp = f.create_group("data")
        for i, case in enumerate(cases, start=1):
            nodal, edges, u_inf, aoa_deg = build_sample(z, case)
            F, T, N = nodal.shape

            flat = nodal.reshape(F, -1).astype(np.float64)
            running["count"] += flat.shape[1]
            running["sum"] += flat.sum(axis=1)
            running["sumsq"] += (flat ** 2).sum(axis=1)
            running["min"] = np.minimum(running["min"], flat.min(axis=1))
            running["max"] = np.maximum(running["max"], flat.max(axis=1))

            sg = data_grp.create_group(str(i))
            sg.create_dataset("nodal_data", data=nodal, compression="gzip", compression_opts=4)
            sg.create_dataset("mesh_edge", data=edges, compression="gzip", compression_opts=4)
            md = sg.create_group("metadata")
            md.attrs["source_filename"] = "AirfRANS Dataset.zip"
            md.attrs["filename_id"] = case
            md.attrs["num_nodes"] = N
            md.attrs["num_edges"] = edges.shape[1]
            md.attrs["num_timesteps"] = T
            md.attrs["inlet_velocity"] = u_inf
            md.attrs["angle_of_attack_deg"] = aoa_deg
            md.create_dataset("feature_min", data=nodal.min(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_max", data=nodal.max(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_mean", data=nodal.mean(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_std", data=nodal.std(axis=(1, 2)).astype(np.float32))
            print(f"[airfrans/{args.task}] sample {i}/{len(cases)} ({case}): N={N} E={edges.shape[1]}")

        f.attrs["num_samples"] = len(cases)
        f.attrs["num_features"] = n_features
        f.attrs["num_timesteps"] = 1

        mean = running["sum"] / running["count"]
        var = running["sumsq"] / running["count"] - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))

        top_md = f.create_group("metadata")
        top_md.create_dataset("feature_names", data=np.array(FEATURE_NAMES, dtype="S10"))
        norm = top_md.create_group("normalization_params")
        norm.create_dataset("min", data=running["min"].astype(np.float32))
        norm.create_dataset("max", data=running["max"].astype(np.float32))
        norm.create_dataset("mean", data=mean.astype(np.float32))
        norm.create_dataset("std", data=std.astype(np.float32))
        splits = top_md.create_group("splits")
        splits.create_dataset("train", data=np.array([], dtype=np.int64))
        splits.create_dataset("val", data=np.array([], dtype=np.int64))
        splits.create_dataset("test", data=np.array([], dtype=np.int64))

        f.attrs["builder_input_var"] = 4
        f.attrs["builder_output_var"] = 4
        f.attrs["builder_cond_var"] = 5

    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({len(cases)} samples)")


if __name__ == "__main__":
    main()
