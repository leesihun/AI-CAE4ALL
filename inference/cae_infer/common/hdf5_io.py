"""Shared MGN-contract HDF5 I/O: `data/{sample_id}/{nodal_data,mesh_edge}` in,
an atomic-write rollout `.h5` out. Used by the neural_operator, transolver, and
meshgraphnets(_v) family drivers so the read/write contract cannot drift
between them (mirrors Neural_Operator/inference_profiles/rollout.py).
"""

import os

import h5py
import numpy as np


def list_sample_ids(input_path: str):
    with h5py.File(input_path, "r") as f:
        return sorted(int(k) for k in f["data"].keys())


def read_sample(input_path: str, sample_id):
    with h5py.File(input_path, "r") as f:
        nodal_data = f[f"data/{sample_id}/nodal_data"][:]
        mesh_edge = f[f"data/{sample_id}/mesh_edge"][:]
    return nodal_data, mesh_edge


def write_rollout_output(output_dir: str, sample_id, ref_pos: np.ndarray, mesh_edge: np.ndarray,
                          all_states: np.ndarray, part_ids, output_var: int, model_path: str,
                          source_label: str, total_time_s: float) -> str:
    """Atomic MGN-style writer: write to a temp file, `os.replace` into place
    so a crash never leaves a successful-looking partial output. Returns the
    final path."""
    os.makedirs(output_dir, exist_ok=True)
    num_steps = all_states.shape[0] - 1
    num_nodes = ref_pos.shape[0]
    output_filename = f"rollout_sample{sample_id}_steps{num_steps}.h5"
    final_path = os.path.join(output_dir, output_filename)
    tmp_path = final_path + ".tmp"

    num_save_features = 3 + output_var + 1
    nodal_data = np.zeros((num_save_features, num_steps + 1, num_nodes), dtype=np.float32)
    nodal_data[0, :, :] = ref_pos[:, 0]
    nodal_data[1, :, :] = ref_pos[:, 1]
    nodal_data[2, :, :] = ref_pos[:, 2]
    for ch in range(output_var):
        nodal_data[3 + ch, :, :] = all_states[:, :, ch]
    if part_ids is not None:
        nodal_data[3 + output_var, :, :] = part_ids[np.newaxis, :]

    with h5py.File(tmp_path, "w") as f:
        f.attrs["num_samples"] = 1
        f.attrs["num_features"] = num_save_features
        f.attrs["num_timesteps"] = num_steps + 1

        data_grp = f.create_group("data")
        sample_grp = data_grp.create_group(str(sample_id))
        sample_grp.create_dataset("nodal_data", data=nodal_data,
                                   compression="gzip", compression_opts=4)
        sample_grp.create_dataset("mesh_edge", data=mesh_edge)

        meta_grp = sample_grp.create_group("metadata")
        meta_grp.attrs["sample_id"] = sample_id
        meta_grp.attrs["num_nodes"] = num_nodes
        meta_grp.attrs["num_edges"] = mesh_edge.shape[1]
        meta_grp.attrs["num_timesteps"] = num_steps + 1
        meta_grp.attrs["model_path"] = model_path
        meta_grp.attrs["source"] = source_label
        meta_grp.attrs["total_rollout_time_s"] = total_time_s

        global_meta = f.create_group("metadata")
        feature_names = [b"x_coord", b"y_coord", b"z_coord"]
        feature_names += [f"output_{i}".encode() for i in range(output_var)]
        feature_names += [b"Part No."]
        global_meta.create_dataset("feature_names", data=np.array(feature_names))
        f.flush()

    os.replace(tmp_path, final_path)
    return final_path


def write_mgn_rollout_output(output_dir: str, sample_id, ref_pos: np.ndarray, mesh_edge: np.ndarray,
                              all_states: np.ndarray, part_ids, output_var: int, normalization: dict,
                              model_path: str, source_label: str, total_time_s: float,
                              output_filename: str = None, vae_sample_idx=None) -> str:
    """MeshGraphNets-specific atomic writer: mirrors the exact HDF5 schema the
    native `MeshGraphNets/inference_profiles/rollout.py` and
    `MeshGraphNets - variational/inference_profiles/rollout.py` write --
    `write_rollout_output` above deliberately omits fields those two rollouts
    always include (per-feature min/max/mean/std, the `normalization_params`
    group with node/edge/delta stats, and MGN's semantic feature names instead
    of generic `output_i`), so downstream tooling built against the native
    rollout output keeps working unchanged against the bundle's output.

    `normalization` is the checkpoint's `normalization` dict (or the
    per-sample `_SampleContext` equivalent for meshgraphnets_v); only
    node_mean/std, edge_mean/std, delta_mean/std are read from it -- extra
    keys (world_edge_radius, coarse_edge_means/stds, ...) are ignored.

    `vae_sample_idx`, when not None, is recorded as a metadata attr (matches
    the variational rollout's per-trajectory HDF5 outputs).
    """
    os.makedirs(output_dir, exist_ok=True)
    num_steps = all_states.shape[0] - 1
    num_nodes = ref_pos.shape[0]
    if output_filename is None:
        output_filename = f"rollout_sample{sample_id}_steps{num_steps}.h5"
    final_path = os.path.join(output_dir, output_filename)
    tmp_path = final_path + ".tmp"

    node_mean = normalization["node_mean"]
    node_std = normalization["node_std"]
    edge_mean = normalization["edge_mean"]
    edge_std = normalization["edge_std"]
    delta_mean = normalization["delta_mean"]
    delta_std = normalization["delta_std"]

    num_save_features = 3 + output_var + 1
    nodal_data = np.zeros((num_save_features, num_steps + 1, num_nodes), dtype=np.float32)
    nodal_data[0, :, :] = ref_pos[:, 0]
    nodal_data[1, :, :] = ref_pos[:, 1]
    nodal_data[2, :, :] = ref_pos[:, 2]
    for ch in range(output_var):
        nodal_data[3 + ch, :, :] = all_states[:, :, ch]
    if part_ids is not None:
        nodal_data[3 + output_var, :, :] = part_ids[np.newaxis, :]

    with h5py.File(tmp_path, "w") as f:
        f.attrs["num_samples"] = 1
        f.attrs["num_features"] = num_save_features
        f.attrs["num_timesteps"] = num_steps + 1

        sample_grp = f.create_group("data").create_group(str(sample_id))
        sample_grp.create_dataset("nodal_data", data=nodal_data,
                                   compression="gzip", compression_opts=4)
        sample_grp.create_dataset("mesh_edge", data=mesh_edge)

        meta_grp = sample_grp.create_group("metadata")
        meta_grp.attrs["sample_id"] = sample_id
        meta_grp.attrs["num_nodes"] = num_nodes
        meta_grp.attrs["num_edges"] = mesh_edge.shape[1]
        meta_grp.attrs["num_timesteps"] = num_steps + 1
        meta_grp.attrs["model_path"] = model_path
        meta_grp.attrs["config_file"] = source_label
        meta_grp.attrs["total_rollout_time_s"] = total_time_s
        if vae_sample_idx is not None:
            meta_grp.attrs["vae_sample_idx"] = vae_sample_idx

        all_feature_names = [
            b"x_coord", b"y_coord", b"z_coord",
            b"x_disp(mm)", b"y_disp(mm)", b"z_disp(mm)",
            b"stress(MPa)", b"Part No.",
        ]
        feature_names = np.array(all_feature_names[:3 + output_var] + [b"Part No."])
        meta_grp.create_dataset("feature_min", data=np.array(
            [nodal_data[i].min() for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset("feature_max", data=np.array(
            [nodal_data[i].max() for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset("feature_mean", data=np.array(
            [nodal_data[i].mean() for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset("feature_std", data=np.array(
            [nodal_data[i].std() for i in range(num_save_features)], dtype=np.float32))

        global_meta = f.create_group("metadata")
        global_meta.create_dataset("feature_names", data=feature_names)
        norm_grp = global_meta.create_group("normalization_params")
        norm_grp.create_dataset("node_mean", data=node_mean)
        norm_grp.create_dataset("node_std", data=node_std)
        norm_grp.create_dataset("edge_mean", data=edge_mean)
        norm_grp.create_dataset("edge_std", data=edge_std)
        norm_grp.create_dataset("delta_mean", data=delta_mean)
        norm_grp.create_dataset("delta_std", data=delta_std)
        f.flush()

    os.replace(tmp_path, final_path)
    return final_path
