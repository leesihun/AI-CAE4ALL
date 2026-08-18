#!/usr/bin/env python
"""Convert DeepMind meshgraphnets TFRecord datasets into the shared mesh HDF5
contract (see dataset/DATASET_FORMAT.md).

Produces exactly two files per dataset -- a full-resolution train set and an
extrapolation test set -- no smoke/arrt tiers:

    python dataset/build_public_mgn.py --dataset cylinder_flow
    python dataset/build_public_mgn.py --dataset deforming_plate
    python dataset/build_public_mgn.py --dataset flag_simple

Reads `--n-train + --n-infer` trajectories (default 100 + 20) from
train.tfrecord at full temporal resolution (no striding/truncation), scores
each trajectory with a dataset-specific scalar "axis" tied to its driving
physical parameter, sorts by that axis, and assigns the *highest* values to
the infer/test split so evaluation genuinely extrapolates beyond the training
range rather than just being a held-out sample of the same distribution:

  - cylinder_flow: inflow speed magnitude at the INFLOW boundary (node_type
    4) -- this is literally the per-trajectory randomized control parameter
    in the DeepMind sim.
  - deforming_plate: indentation depth of the rigid OBSTACLE (node_type 1)
    over the trajectory -- the prescribed kinematic BC driving the contact.
  - flag_simple: overall cloth displacement magnitude (node_type 0). The raw
    data has no stored global "wind" parameter and the HANDLE nodes barely
    move across trajectories, so there is no clean input-parameter to split
    on; this is an outcome-based proxy for motion severity, not a true
    control-parameter extrapolation like the other two.

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


NP_DTYPE = {"int32": np.int32, "float32": np.float32}

EX_SLOT = {"cylinder_flow": "ex4", "deforming_plate": "ex5", "flag_simple": "ex6"}


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


# --- per-dataset extrapolation axis ----------------------------------------

def axis_cylinder_flow(traj):
    nt = traj["node_type"][0, :, 0]
    vel0 = traj["velocity"][0]
    mask = nt == 4  # INFLOW
    return float(np.linalg.norm(vel0[mask], axis=-1).mean())


def axis_deforming_plate(traj):
    nt = traj["node_type"][0, :, 0]
    mask = nt == 1  # OBSTACLE (prescribed rigid indenter)
    centroid_z = traj["world_pos"][:, mask, 2].mean(axis=1)
    return float(centroid_z.max() - centroid_z.min())


def axis_flag_simple(traj):
    nt = traj["node_type"][0, :, 0]
    mask = nt == 0  # free cloth body -- see module docstring caveat
    mesh_pos = traj["mesh_pos"][0]
    disp = traj["world_pos"][:, mask, :2] - mesh_pos[mask][None]
    return float(np.linalg.norm(disp, axis=-1).mean())


AXIS_FN = {
    "cylinder_flow": axis_cylinder_flow,
    "deforming_plate": axis_deforming_plate,
    "flag_simple": axis_flag_simple,
}


# --- per-dataset HDF5 row layout --------------------------------------------

def build_cylinder_flow(traj):
    mesh_pos = traj["mesh_pos"][0]  # [N, 2]
    node_type = traj["node_type"][0, :, 0]  # [N] raw codes
    velocity = traj["velocity"]  # [T, N, 2]
    pressure = traj["pressure"]  # [T, N, 1]
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


def build_deforming_plate(traj):
    mesh_pos = traj["mesh_pos"][0]  # [N, 3]
    node_type = traj["node_type"][0, :, 0]
    world_pos = traj["world_pos"]  # [T, N, 3]
    stress = traj["stress"]  # [T, N, 1]
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


def build_flag_simple(traj):
    mesh_pos = traj["mesh_pos"][0]  # [N, 2] flat rest template
    node_type = traj["node_type"][0, :, 0]  # verified constant over time
    world_pos = traj["world_pos"]  # [T, N, 3]
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


def write_h5(out_path, records, feature_names, var_info, dataset, split_name):
    """records: list of (axis_value, nodal[F,T,N], edges[2,E])."""
    n_features = len(feature_names)
    running = dict(
        count=0,
        sum=np.zeros(n_features, dtype=np.float64),
        sumsq=np.zeros(n_features, dtype=np.float64),
        min=np.full(n_features, np.inf, dtype=np.float64),
        max=np.full(n_features, -np.inf, dtype=np.float64),
    )

    tmp_path = out_path + ".tmp"
    T_last = None
    with h5py.File(tmp_path, "w") as f:
        data_grp = f.create_group("data")
        for sample_id, (axis_val, nodal, edges) in enumerate(records, start=1):
            F, T, N = nodal.shape
            T_last = T

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
            md.attrs["filename_id"] = f"{dataset}_{split_name}_{sample_id:04d}"
            md.attrs["extrapolation_axis"] = axis_val
            md.attrs["num_nodes"] = N
            md.attrs["num_edges"] = edges.shape[1]
            md.attrs["num_timesteps"] = T
            md.create_dataset("feature_min", data=nodal.min(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_max", data=nodal.max(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_mean", data=nodal.mean(axis=(1, 2)).astype(np.float32))
            md.create_dataset("feature_std", data=nodal.std(axis=(1, 2)).astype(np.float32))
            print(f"[{dataset}/{split_name}] sample {sample_id}: N={N} T={T} E={edges.shape[1]} axis={axis_val:.4f}")

        f.attrs["num_samples"] = len(records)
        f.attrs["num_features"] = n_features
        f.attrs["num_timesteps"] = T_last

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
    print(f"wrote {out_path} ({len(records)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(BUILDERS.keys()))
    ap.add_argument("--raw-root", default=os.environ.get("RAW", "D:/CAE_datasets_raw") + "/meshgraphnets")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--n-infer", type=int, default=20)
    args = ap.parse_args()

    builder_fn, feature_names = BUILDERS[args.dataset]
    axis_fn = AXIS_FN[args.dataset]
    slot = EX_SLOT[args.dataset]
    raw_dir = os.path.join(args.raw_root, args.dataset)
    with open(os.path.join(raw_dir, "meta.json")) as f:
        meta = json.load(f)

    n_total = args.n_train + args.n_infer
    records = []
    var_info = None
    for i, traj in enumerate(iter_trajectories(raw_dir, meta, n_total)):
        nodal, edges, var_info = builder_fn(traj)
        axis_val = axis_fn(traj)
        records.append((axis_val, nodal, edges))
        print(f"[{args.dataset}] decoded {i + 1}/{n_total} axis={axis_val:.4f} N={nodal.shape[2]}")

    records.sort(key=lambda r: r[0])
    train_records = records[: args.n_train]
    infer_records = records[args.n_train :]  # highest axis values -> extrapolation

    write_h5(os.path.join(args.out_dir, f"{slot}.h5"), train_records, feature_names, var_info, args.dataset, "train")
    write_h5(os.path.join(args.out_dir, f"{slot}_infer.h5"), infer_records, feature_names, var_info, args.dataset, "infer")


if __name__ == "__main__":
    main()
