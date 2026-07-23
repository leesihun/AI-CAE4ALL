"""Explicit-args refactor of Neural_Operator/inference_profiles/rollout.py's
`run_rollout`. Covers all four registered cores (point_deeponet, deeponet,
fno, gino) through one checkpoint-led path -- the checkpoint alone decides
which architecture gets rebuilt (model/factory.py::build_model_from_checkpoint).

CPU-only: no gpu_ids branch, no torch_cluster (radius_neighbors.py falls back
to its scipy cKDTree path automatically when torch_cluster is absent).
"""

import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from common.hdf5_io import write_rollout_output
from general_modules.normalize import normalize_positions, normalize_node_features
from general_modules.positional_features import compute_positional_features
from model.factory import build_model_from_checkpoint
from query_decode import decode_in_chunks

SCHEMA_VERSION = "deeponet_repo_v1"


def run(checkpoint: str, input: str, output: str, device: torch.device,
        timesteps: int = None, query_chunk_size: int = 0, **_ignored) -> str:

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = ckpt.get("schema_version")
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint '{checkpoint}' has schema_version={schema!r}, expected "
            f"'{SCHEMA_VERSION}'."
        )

    selected_model = ckpt["selected_model"]
    config = {"model": selected_model}
    print(f"  Family: neural_operator / model: {selected_model}")

    model, data_spec, coordinate_domain = build_model_from_checkpoint(config, ckpt)
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    norm = ckpt["normalization"]
    node_mean, node_std = norm["node_mean"], norm["node_std"]
    delta_mean, delta_std = norm["delta_mean"], norm["delta_std"]
    position_scale = norm["position_scale"]
    node_type_to_idx = norm.get("node_type_to_idx")
    num_node_types = norm.get("num_node_types")

    input_var = data_spec.input_var
    output_var = data_spec.output_var
    num_pos_features = data_spec.positional_dim
    use_node_types = data_spec.node_type_dim > 0

    with h5py.File(input, "r") as f:
        sample_ids = sorted(int(k) for k in f["data"].keys())
    print(f"  Found {len(sample_ids)} sample(s): {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    os.makedirs(output, exist_ok=True)
    last_path = None

    for sample_id in sample_ids:
        with h5py.File(input, "r") as f:
            nodal_data = f[f"data/{sample_id}/nodal_data"][:]
            mesh_edge = f[f"data/{sample_id}/mesh_edge"][:]

        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"\n  Sample {sample_id}: nodal_data {nodal_data.shape}, "
              f"mesh_edge {mesh_edge.shape[1]} (unidirectional)")

        steps = timesteps
        if steps is None:
            if num_timesteps > 1:
                steps = num_timesteps - 1
                print(f"  Auto-set rollout steps to {steps} (full trajectory)")
            else:
                raise ValueError(
                    f"timesteps not given and dataset has only {num_timesteps} "
                    "timestep(s); pass --timesteps."
                )

        ref_pos = nodal_data[:3, 0, :].T.astype(np.float32)
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

        part_ids = None
        if use_node_types:
            if num_features <= 7:
                raise ValueError(
                    f"Checkpoint requires node types but '{input}' sample "
                    f"{sample_id} has only {num_features} feature rows."
                )
            part_ids = nodal_data[-1, 0, :].astype(np.int32)

        pos_feat = None
        if num_pos_features > 0:
            pos_feat = compute_positional_features(ref_pos, edge_index, num_pos_features)

        pos_normalized = normalize_positions(ref_pos, position_scale)
        pos_t = torch.from_numpy(ref_pos).to(device)
        pos_norm_t = torch.from_numpy(pos_normalized.astype(np.float32)).to(device)
        edge_index_t = torch.from_numpy(edge_index).long().to(device)
        batch_t = torch.zeros(num_nodes, dtype=torch.long, device=device)
        ptr_t = torch.tensor([0, num_nodes], dtype=torch.long, device=device)

        if num_timesteps == 1:
            current_state = np.zeros((num_nodes, input_var), dtype=np.float32)
        else:
            current_state = nodal_data[3:3 + input_var, 0, :].T.astype(np.float32)

        all_states = np.zeros((steps + 1, num_nodes, output_var), dtype=np.float32)
        all_states[0] = current_state[:, :output_var]

        chunk_size = int(query_chunk_size or 0)

        rollout_start = time.time()
        with torch.no_grad():
            for step in range(steps):
                x_raw = current_state if pos_feat is None else np.concatenate([current_state, pos_feat], axis=1)
                x_norm = normalize_node_features(
                    x_raw, node_mean, node_std,
                    node_types=part_ids if use_node_types else None,
                    node_type_to_idx=node_type_to_idx, num_node_types=num_node_types,
                )
                x_t = torch.from_numpy(x_norm.astype(np.float32)).to(device)

                graph = Data(x=x_t, pos=pos_t, pos_normalized=pos_norm_t, edge_index=edge_index_t)
                graph.batch = batch_t
                graph.ptr = ptr_t
                graph.sample_id = int(sample_id)

                if chunk_size > 0:
                    encoded = model.encode_operator(graph)
                    predicted_norm = decode_in_chunks(model, encoded, graph, chunk_size)
                else:
                    predicted_norm, _ = model(graph, add_noise=False)
                predicted = predicted_norm.cpu().numpy() * delta_std + delta_mean

                if num_timesteps == 1:
                    current_state[:, :output_var] = predicted
                else:
                    current_state[:, :output_var] = current_state[:, :output_var] + predicted
                all_states[step + 1] = current_state[:, :output_var]

                if steps > 0 and (step % max(1, steps // 20) == 0 or step == steps - 1):
                    print(f"    step {step + 1}/{steps}")

        total_time = time.time() - rollout_start
        print(f"  Rollout done in {total_time:.2f}s" +
              (f" ({total_time / steps:.3f}s/step)" if steps > 0 else ""))

        last_path = write_rollout_output(
            output, sample_id, ref_pos, mesh_edge, all_states, part_ids,
            output_var, checkpoint, "cae_infer/neural_operator", total_time,
        )
        print(f"  Saved: {last_path}")

    print(f"\nInference complete. Processed {len(sample_ids)} scene(s).")
    return last_path
