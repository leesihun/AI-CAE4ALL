#!/usr/bin/env python
"""Convert Zongyi Li's Geo-FNO plasticity benchmark (plas_N987_T20.mat) into
the shared mesh HDF5 contract (dataset/DATASET_FORMAT.md).

Source: https://github.com/zongyi-li/Geo-FNO (Google Drive folder
"Geo-PDE datasets", plasticity/plas_N987_T20.mat, file id
14CPGK_ljae5c6dm2nRraY2kIDt39JX3d). Raw file expected at
$RAW/geo_fno/plasticity/plas_N987_T20.mat (default RAW=D:/CAE_datasets_raw).

    python dataset/build_geo_fno_plasticity.py

`.mat` contents (verified empirically, not assumed):
    input  [987, 101]              per-sample die-profile boundary parameterization
    output [987, 101, 31, 20, 4]   101x31 structured node grid, 20 timesteps, 4 channels

Channels 0/1 of `output` are themselves grid coordinates (they shift slightly
over time -- an updated-Lagrangian mesh) and are used here as the fixed T=0
reference position; channels 2/3 are literal displacement (exactly zero at
t=0, matching this repo's rows[3:3+input_var] "state" convention). `input`'s
101 die-profile values are broadcast across the 31-axis as a 4th, time-
constant state channel -- this matches
configs/benchmarks/plasticity/config_train_meshgraphnets.txt's own comment
("State: ux, uy, uz=0, static die profile") exactly, verified before writing
this converter, not guessed after.

Row layout (7 rows, T=20): rows 0:3 = x,y,z(=0); rows 3:7 = ux, uy, uz(=0),
die_profile; input_var=output_var=4, cond_var=0.

mesh_edge is a structured 101x31 GRID adjacency (4-connected), not KNN --
this dataset ships a real regular mesh, unlike elasticity's bare point cloud.

Split: first 900 samples -> ex9.h5 (train), last 87 -> ex9_infer.h5 (test) --
a reasonable round held-out split (not a bit-exact paper replication, this
benchmark's own literature doesn't fix one), renamed to this suite's
exN.h5/exN_infer.h5 convention. Sample order in the raw .mat is presumed i.i.d.
(no evidence of parameter-sorted ordering), so unlike ex4-ex6 this is a plain
held-out split, not an engineered extrapolation test.
"""
import argparse
import os

import h5py
import numpy as np
import scipy.io as sio


def grid_edges(n_i, n_j):
    """4-connected structured-grid adjacency, node index = i*n_j + j."""
    pairs = []
    for i in range(n_i):
        for j in range(n_j):
            idx = i * n_j + j
            if i + 1 < n_i:
                pairs.append((idx, idx + n_j))
            if j + 1 < n_j:
                pairs.append((idx, idx + 1))
    return np.array(pairs, dtype=np.int32).T  # [2, E]


def write_split(out_path, inp, out, sample_ids, edges):
    n_features = 7
    n_i, n_j, T = out.shape[1], out.shape[2], out.shape[3]
    N = n_i * n_j
    running = dict(
        count=0,
        sum=np.zeros(n_features, dtype=np.float64),
        sumsq=np.zeros(n_features, dtype=np.float64),
        min=np.full(n_features, np.inf, dtype=np.float64),
        max=np.full(n_features, -np.inf, dtype=np.float64),
    )
    tmp_path = out_path + ".tmp"
    with h5py.File(tmp_path, "w") as f:
        data_grp = f.create_group("data")
        for out_i, src_i in enumerate(sample_ids, start=1):
            sample = out[src_i]  # [101, 31, 20, 4]
            x0 = sample[:, :, 0, 0].reshape(N)
            y0 = sample[:, :, 0, 1].reshape(N)
            ux = sample[:, :, :, 2].reshape(N, T)
            uy = sample[:, :, :, 3].reshape(N, T)
            die = np.broadcast_to(inp[src_i][:, None], (n_i, n_j)).reshape(N)

            nodal = np.zeros((n_features, T, N), dtype=np.float32)
            nodal[0, :, :] = x0[None, :]
            nodal[1, :, :] = y0[None, :]
            nodal[3, :, :] = ux.T
            nodal[4, :, :] = uy.T
            nodal[6, :, :] = die[None, :]

            flat = nodal.reshape(n_features, -1).astype(np.float64)
            running["count"] += flat.shape[1]
            running["sum"] += flat.sum(axis=1)
            running["sumsq"] += (flat ** 2).sum(axis=1)
            running["min"] = np.minimum(running["min"], flat.min(axis=1))
            running["max"] = np.maximum(running["max"], flat.max(axis=1))

            sg = data_grp.create_group(str(out_i))
            sg.create_dataset("nodal_data", data=nodal, compression="gzip", compression_opts=4)
            sg.create_dataset("mesh_edge", data=edges, compression="gzip", compression_opts=4)
            md = sg.create_group("metadata")
            md.attrs["source_filename"] = "plas_N987_T20.mat"
            md.attrs["filename_id"] = f"plasticity_{src_i:04d}"
            md.attrs["num_nodes"] = N
            md.attrs["num_edges"] = edges.shape[1]
            md.attrs["num_timesteps"] = T
            md.create_dataset("feature_min", data=nodal.min(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_max", data=nodal.max(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_mean", data=nodal.mean(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_std", data=nodal.std(axis=(1, 2)).astype(np.float32))
            if out_i % 100 == 0 or out_i == len(sample_ids):
                print(f"[plasticity] {os.path.basename(out_path)}: {out_i}/{len(sample_ids)}")

        f.attrs["num_samples"] = len(sample_ids)
        f.attrs["num_features"] = n_features
        f.attrs["num_timesteps"] = T

        mean = running["sum"] / running["count"]
        var = running["sumsq"] / running["count"] - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        top_md = f.create_group("metadata")
        top_md.create_dataset(
            "feature_names",
            data=np.array(["x_coord", "y_coord", "z_coord", "ux", "uy", "uz", "die_profile"], dtype="S10"),
        )
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
        f.attrs["builder_cond_var"] = 0

    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({len(sample_ids)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-file", default=os.environ.get("RAW", "D:/CAE_datasets_raw") + "/geo_fno/plasticity/plas_N987_T20.mat")
    ap.add_argument("--out-dir", default="dataset")
    args = ap.parse_args()

    d = sio.loadmat(args.raw_file)
    inp = d["input"]  # [987, 101]
    out = d["output"]  # [987, 101, 31, 20, 4]
    n_total = out.shape[0]
    assert n_total == 987, f"unexpected sample count {n_total}"

    edges = grid_edges(out.shape[1], out.shape[2])

    os.makedirs(args.out_dir, exist_ok=True)
    write_split(os.path.join(args.out_dir, "ex9.h5"), inp, out, list(range(0, 900)), edges)
    write_split(os.path.join(args.out_dir, "ex9_infer.h5"), inp, out, list(range(900, 987)), edges)


if __name__ == "__main__":
    main()
