#!/usr/bin/env python
"""Convert DeepMind meshgraphnets TFRecord datasets into the shared mesh HDF5
contract (see dataset/DATASET_FORMAT.md and dataset/PUBLIC_DATASETS.md).

Reads TFRecords with tf.train.Example decoding only (no tf.data training
ops, no GPU) so it is cheap to run once to produce the HDF5 files that the
rest of the suite consumes.

    python dataset/build_public_mgn.py --dataset cylinder_flow   --tier main
    python dataset/build_public_mgn.py --dataset deforming_plate --tier smoke
    python dataset/build_public_mgn.py --dataset flag_simple     --tier smoke

Raw TFRecords are expected at $RAW/meshgraphnets/<dataset>/{meta.json,
train.tfrecord}, staged by fetch_public_datasets.sh (default
RAW=D:/CAE_datasets_raw).
"""
import argparse
import itertools
import json
import os

import h5py
import numpy as np
import tensorflow as tf


TIERS = {
    # (n_traj, stride, n_frames). smoke/arrt are STRIDED subsamples of the full
    # trajectory, not truncations -- verified empirically against the existing
    # ex4_*_smoke.h5 (stride 6, ceil(full_t/6) frames) and ex4_*_arrt.h5
    # (fixed stride 8, 25 frames, i.e. raw indices 0,8,...,192) files.
    "smoke": dict(n_traj=20, stride=6, n_frames=None),
    "arrt": dict(n_traj=50, stride=8, n_frames=25),
    "main": dict(n_traj=100, stride=1, n_frames=None),
}

NP_DTYPE = {"int32": np.int32, "float32": np.float32}


def iter_trajectories(raw_dir, meta, max_traj):
    """Yield {field_name: ndarray[shape_per_meta_with_true_node_count]} dicts."""
    rec_path = os.path.join(raw_dir, "train.tfrecord")
    ds = tf.data.TFRecordDataset(rec_path)
    for raw in itertools.islice(ds, max_traj):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        feats = ex.features.feature
        traj = {}
        for name in meta["field_names"]:
            spec = meta["features"][name]
            dt = NP_DTYPE[spec["dtype"]]
            raw_bytes = feats[name].bytes_list.value[0]
            arr = np.frombuffer(raw_bytes, dtype=dt)
            shape = spec["shape"]
            time_dim, chan_dim = shape[0], shape[-1]
            n_nodes = arr.size // (time_dim * chan_dim)
            traj[name] = arr.reshape(time_dim, n_nodes, chan_dim)
        yield traj


def edges_from_cells(cells, k):
    """Unique undirected edges from a [C, k] simplex array (k=3 tri, k=4 tet)."""
    pairs = set()
    idx_pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    for cell in cells:
        for a, b in idx_pairs:
            u, v = int(cell[a]), int(cell[b])
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            pairs.add(key)
    arr = np.array(sorted(pairs), dtype=np.int32).T  # [2, E]
    if arr.size == 0:
        return np.zeros((2, 0), dtype=np.int32)
    return arr


def time_indices(tier, full_t):
    stride = tier["stride"]
    n_frames = tier["n_frames"]
    if n_frames is None:
        n_frames = -(-full_t // stride)  # ceil(full_t / stride)
    idx = np.arange(n_frames) * stride
    return idx[idx < full_t]


def build_cylinder_flow(traj, t_idx):
    mesh_pos = traj["mesh_pos"][0]  # [N, 2]
    node_type = traj["node_type"][0, :, 0]  # [N] raw codes
    velocity = traj["velocity"][t_idx]  # [T, N, 2]
    pressure = traj["pressure"][t_idx]  # [T, N, 1]
    cells = traj["cells"][0]  # [C, 3]
    N = mesh_pos.shape[0]
    T = velocity.shape[0]
    nodal = np.zeros((7, T, N), dtype=np.float32)
    nodal[0] = mesh_pos[:, 0][None, :]
    nodal[1] = mesh_pos[:, 1][None, :]
    # nodal[2] stays 0 (planar)
    nodal[3] = velocity[:, :, 0]
    nodal[4] = velocity[:, :, 1]
    nodal[5] = pressure[:, :, 0]
    nodal[6] = node_type[None, :].astype(np.float32)
    edges = edges_from_cells(cells, 3)
    return nodal, edges, dict(input_var=3, output_var=3, cond_var=0)


def build_deforming_plate(traj, t_idx):
    mesh_pos = traj["mesh_pos"][0]  # [N, 3]
    node_type = traj["node_type"][0, :, 0]
    world_pos = traj["world_pos"][t_idx]  # [T, N, 3]
    stress = traj["stress"][t_idx]  # [T, N, 1]
    cells = traj["cells"][0]  # [C, 4] tets
    N = mesh_pos.shape[0]
    T = world_pos.shape[0]
    nodal = np.zeros((8, T, N), dtype=np.float32)
    nodal[0] = mesh_pos[:, 0][None, :]
    nodal[1] = mesh_pos[:, 1][None, :]
    nodal[2] = mesh_pos[:, 2][None, :]
    disp = world_pos - mesh_pos[None, :, :]
    nodal[3] = disp[:, :, 0]
    nodal[4] = disp[:, :, 1]
    nodal[5] = disp[:, :, 2]
    nodal[6] = stress[:, :, 0]
    nodal[7] = node_type[None, :].astype(np.float32)
    edges = edges_from_cells(cells, 4)
    return nodal, edges, dict(input_var=4, output_var=4, cond_var=0)


def build_flag_simple(traj, t_idx):
    mesh_pos = traj["mesh_pos"][0]  # [N, 2] flat rest template
    node_type = traj["node_type"][0, :, 0]  # verified constant over time
    world_pos = traj["world_pos"][t_idx]  # [T, N, 3]
    cells = traj["cells"][0]  # [C, 3] triangles
    N = mesh_pos.shape[0]
    T = world_pos.shape[0]
    mesh_pos3 = np.zeros((N, 3), dtype=np.float32)
    mesh_pos3[:, 0] = mesh_pos[:, 0]
    mesh_pos3[:, 1] = mesh_pos[:, 1]
    nodal = np.zeros((7, T, N), dtype=np.float32)
    nodal[0] = mesh_pos3[:, 0][None, :]
    nodal[1] = mesh_pos3[:, 1][None, :]
    nodal[2] = mesh_pos3[:, 2][None, :]
    disp = world_pos - mesh_pos3[None, :, :]
    nodal[3] = disp[:, :, 0]
    nodal[4] = disp[:, :, 1]
    nodal[5] = disp[:, :, 2]
    nodal[6] = node_type[None, :].astype(np.float32)
    edges = edges_from_cells(cells, 3)
    return nodal, edges, dict(input_var=3, output_var=3, cond_var=0)


BUILDERS = {
    "cylinder_flow": (build_cylinder_flow, ["x_coord", "y_coord", "z_coord", "velocity_x", "velocity_y", "pressure", "node_type"]),
    "deforming_plate": (build_deforming_plate, ["x_coord", "y_coord", "z_coord", "disp_x", "disp_y", "disp_z", "stress", "node_type"]),
    "flag_simple": (build_flag_simple, ["x_coord", "y_coord", "z_coord", "disp_x", "disp_y", "disp_z", "node_type"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(BUILDERS.keys()))
    ap.add_argument("--tier", required=True, choices=sorted(TIERS.keys()))
    ap.add_argument("--raw-root", default=os.environ.get("RAW", "D:/CAE_datasets_raw") + "/meshgraphnets")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--out-name", default=None, help="Override output filename (default: ex4_<dataset>[_<tier>].h5)")
    args = ap.parse_args()

    builder_fn, feature_names = BUILDERS[args.dataset]
    tier = TIERS[args.tier]
    raw_dir = os.path.join(args.raw_root, args.dataset)
    with open(os.path.join(raw_dir, "meta.json")) as f:
        meta = json.load(f)

    suffix = "" if args.tier == "main" else f"_{args.tier}"
    out_name = args.out_name or f"ex4_{args.dataset}{suffix}.h5"
    out_path = os.path.join(args.out_dir, out_name)

    n_features = len(feature_names)
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
        sample_id = 0
        for traj in iter_trajectories(raw_dir, meta, tier["n_traj"]):
            sample_id += 1
            full_t = traj[meta["field_names"][-1]].shape[0]
            t_idx = time_indices(tier, full_t)
            nodal, edges, var_info = builder_fn(traj, t_idx)
            F, T, N = nodal.shape

            flat = nodal.reshape(F, -1).astype(np.float64)
            running["count"] += flat.shape[1]
            running["sum"] += flat.sum(axis=1)
            running["sumsq"] += (flat ** 2).sum(axis=1)
            running["min"] = np.minimum(running["min"], flat.min(axis=1))
            running["max"] = np.maximum(running["max"], flat.max(axis=1))

            sg = data_grp.create_group(str(sample_id))
            sg.create_dataset("nodal_data", data=nodal, compression="gzip", compression_opts=4)
            sg.create_dataset("mesh_edge", data=edges, compression="gzip", compression_opts=4)
            md = sg.create_group("metadata")
            md.attrs["source_filename"] = "train.tfrecord"
            md.attrs["filename_id"] = f"{args.dataset}_train_{sample_id:04d}"
            md.attrs["num_nodes"] = N
            md.attrs["num_edges"] = edges.shape[1]
            md.attrs["num_timesteps"] = T
            md.create_dataset("feature_min", data=nodal.min(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_max", data=nodal.max(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_mean", data=nodal.mean(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_std", data=nodal.std(axis=(1, 2)).astype(np.float32))
            print(f"[{args.dataset}/{args.tier}] sample {sample_id}: N={N} T={T} E={edges.shape[1]}")

        f.attrs["num_samples"] = sample_id
        f.attrs["num_features"] = n_features
        f.attrs["num_timesteps"] = T  # last sample's T; tiers are T-homogeneous per dataset

        mean = running["sum"] / running["count"]
        var = running["sumsq"] / running["count"] - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))

        top_md = f.create_group("metadata")
        top_md.create_dataset("feature_names", data=np.array(feature_names, dtype="S10"))
        norm = top_md.create_group("normalization_params")
        norm.create_dataset("min", data=running["min"].astype(np.float32))
        norm.create_dataset("max", data=running["max"].astype(np.float32))
        norm.create_dataset("mean", data=mean.astype(np.float32))
        norm.create_dataset("std", data=std.astype(np.float32))
        splits = top_md.create_group("splits")
        splits.create_dataset("train", data=np.array([], dtype=np.int64))
        splits.create_dataset("val", data=np.array([], dtype=np.int64))
        splits.create_dataset("test", data=np.array([], dtype=np.int64))

        for k, v in var_info.items():
            f.attrs[f"builder_{k}"] = v

    os.replace(tmp_path, out_path)
    print(f"wrote {out_path} ({sample_id} samples)")


if __name__ == "__main__":
    main()
