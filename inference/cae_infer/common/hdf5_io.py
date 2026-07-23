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
