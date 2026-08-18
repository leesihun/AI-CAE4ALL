#!/usr/bin/env python
"""Convert Zongyi Li's Geo-FNO "Random_UnitCell" elasticity benchmark into the
shared mesh HDF5 contract (dataset/DATASET_FORMAT.md).

Source: https://github.com/zongyi-li/Geo-FNO (Google Drive folder
"Geo-PDE datasets", elasticity/Meshes/Random_UnitCell_{XY,sigma,rr,theta}_10.npy).
Raw files expected at $RAW/geo_fno/elasticity/*.npy (default RAW=D:/CAE_datasets_raw).

    python dataset/build_geo_fno_elasticity.py

972-node unit cell with an arbitrary-shaped void at the center, 2000 total
samples. No explicit connectivity ships with this dataset (it's a Lagrangian
point set, not a stored mesh) -- mesh_edge here is a deterministic k=6 nearest-
neighbor proximity graph, same convention as the AirfRANS smoke tier.

Row layout (static, T=1): rows 0:3 = x,y,z(=0); row 3 = sigma (von Mises
stress); input_var=output_var=1, cond_var=0, matching
configs/benchmarks/elasticity/config_train_transolver.txt exactly (positional_features 0,
use_node_types False).

Split: first 1000 samples -> ex8.h5 (train), last 200 -> ex8_infer.h5 (test) --
the ntrain=1000/ntest=200 convention used throughout Zongyi Li's own Geo-FNO
loading scripts for this file, unchanged, just renamed to this suite's
exN.h5/exN_infer.h5 convention. NOT an extrapolation split like ex4-ex6: this
is a paper-replication benchmark, so the point is to match the published
evaluation protocol exactly (for comparability against reported FNO/Transolver
numbers), not to engineer a harder generalization test.
"""
import argparse
import os

import h5py
import numpy as np
from scipy.spatial import cKDTree


def knn_edges(xy, k=6):
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k + 1)  # includes self at col 0
    pairs = set()
    for i in range(xy.shape[0]):
        for j in idx[i, 1:]:
            j = int(j)
            key = (i, j) if i < j else (j, i)
            pairs.add(key)
    arr = np.array(sorted(pairs), dtype=np.int32).T
    return arr


def write_split(out_path, xy, sigma, sample_ids):
    n_features = 4  # x, y, z, sigma
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
            pts = xy[:, :, src_i]  # [972, 2]
            s = sigma[:, src_i]  # [972]
            N = pts.shape[0]
            nodal = np.zeros((4, 1, N), dtype=np.float32)
            nodal[0, 0, :] = pts[:, 0]
            nodal[1, 0, :] = pts[:, 1]
            nodal[3, 0, :] = s

            edges = knn_edges(pts, k=6)

            flat = nodal.reshape(4, -1).astype(np.float64)
            running["count"] += flat.shape[1]
            running["sum"] += flat.sum(axis=1)
            running["sumsq"] += (flat ** 2).sum(axis=1)
            running["min"] = np.minimum(running["min"], flat.min(axis=1))
            running["max"] = np.maximum(running["max"], flat.max(axis=1))

            sg = data_grp.create_group(str(out_i))
            sg.create_dataset("nodal_data", data=nodal, compression="gzip", compression_opts=4)
            sg.create_dataset("mesh_edge", data=edges, compression="gzip", compression_opts=4)
            md = sg.create_group("metadata")
            md.attrs["source_filename"] = "Random_UnitCell_*_10.npy"
            md.attrs["filename_id"] = f"unitcell_{src_i:04d}"
            md.attrs["num_nodes"] = N
            md.attrs["num_edges"] = edges.shape[1]
            md.attrs["num_timesteps"] = 1
            md.create_dataset("feature_min", data=nodal.min(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_max", data=nodal.max(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_mean", data=nodal.mean(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_std", data=nodal.std(axis=(1, 2)).astype(np.float32))
            if out_i % 100 == 0 or out_i == len(sample_ids):
                print(f"[elasticity] {os.path.basename(out_path)}: {out_i}/{len(sample_ids)}")

        f.attrs["num_samples"] = len(sample_ids)
        f.attrs["num_features"] = n_features
        f.attrs["num_timesteps"] = 1

        mean = running["sum"] / running["count"]
        var = running["sumsq"] / running["count"] - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        top_md = f.create_group("metadata")
        top_md.create_dataset("feature_names", data=np.array(["x_coord", "y_coord", "z_coord", "sigma"], dtype="S10"))
        norm = top_md.create_group("normalization_params")
        norm.create_dataset("min", data=running["min"].astype(np.float32))
        norm.create_dataset("max", data=running["max"].astype(np.float32))
        norm.create_dataset("mean", data=mean.astype(np.float32))
        norm.create_dataset("std", data=std.astype(np.float32))
        splits = top_md.create_group("splits")
        splits.create_dataset("train", data=np.array([], dtype=np.int64))
        splits.create_dataset("val", data=np.array([], dtype=np.int64))
        splits.create_dataset("test", data=np.array([], dtype=np.int64))
        f.attrs["builder_input_var"] = 1
        f.attrs["builder_output_var"] = 1
        f.attrs["builder_cond_var"] = 0

    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({len(sample_ids)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.environ.get("RAW", "D:/CAE_datasets_raw") + "/geo_fno/elasticity")
    ap.add_argument("--out-dir", default="dataset")
    args = ap.parse_args()

    xy = np.load(os.path.join(args.raw_dir, "Random_UnitCell_XY_10.npy"))  # [972, 2, 2000]
    sigma = np.load(os.path.join(args.raw_dir, "Random_UnitCell_sigma_10.npy"))  # [972, 2000]
    n_total = xy.shape[-1]
    assert n_total == 2000 and sigma.shape[-1] == 2000, f"unexpected sample count {n_total}"

    os.makedirs(args.out_dir, exist_ok=True)
    write_split(os.path.join(args.out_dir, "ex8.h5"), xy, sigma, list(range(0, 1000)))
    write_split(os.path.join(args.out_dir, "ex8_infer.h5"), xy, sigma, list(range(1800, 2000)))


if __name__ == "__main__":
    main()
