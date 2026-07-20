"""Checkpoint-led static inference and autoregressive temporal rollout
(IMPLEMENTATION_PLAN.md section 14), adapted from MeshGraphNets'
inference_profiles/rollout.py. Recomputes the operator encoding every
timestep (the physical state changes every step); geometry (positions,
positional features, edges) is read once per sample.
"""

import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from general_modules.mesh_dataset import normalize_positions, normalize_node_features
from general_modules.positional_features import compute_positional_features
from inference_profiles.query_decode import decode_in_chunks
from model.factory import build_model_from_checkpoint
from training_profiles.setup import SCHEMA_VERSION


def _validate_checkpoint_schema(checkpoint, path):
    schema = checkpoint.get('schema_version')
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint '{path}' has schema_version={schema!r}, expected "
            f"'{SCHEMA_VERSION}'. MeshGraphNets/Transolver/older checkpoints are "
            "not compatible with this repository's checkpoint contract (section 13)."
        )


def run_rollout(config, config_filename='config.txt'):
    """Perform checkpoint-led static inference or autoregressive rollout."""
    print("\n" + "=" * 60)
    print("CHECKPOINT-LED INFERENCE / ROLLOUT")
    print("=" * 60)

    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]

    if torch.cuda.is_available() and gpu_ids[0] >= 0:
        gpu_id = gpu_ids[0]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU {gpu_id}, device: {device}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")

    model_path = config.get('modelpath')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    _validate_checkpoint_schema(checkpoint, model_path)

    selected_model = checkpoint['selected_model']
    requested_model = str(config.get('model', selected_model)).lower()
    if requested_model != selected_model:
        raise ValueError(
            f"Config requests model='{requested_model}' but checkpoint was trained as "
            f"'{selected_model}'. Loading a checkpoint under a different model is not supported."
        )
    config['model'] = selected_model

    print("\nInitializing model from checkpoint metadata...")
    model, data_spec, coordinate_domain = build_model_from_checkpoint(config, checkpoint)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint valid loss: {checkpoint.get('valid_loss', 'unknown')}")

    norm = checkpoint['normalization']
    node_mean, node_std = norm['node_mean'], norm['node_std']
    delta_mean, delta_std = norm['delta_mean'], norm['delta_std']
    position_scale = norm['position_scale']
    node_type_to_idx = norm.get('node_type_to_idx')
    num_node_types = norm.get('num_node_types')

    input_var = data_spec.input_var
    output_var = data_spec.output_var
    num_pos_features = data_spec.positional_dim
    use_node_types = data_spec.node_type_dim > 0

    dataset_dir = config.get('infer_dataset')
    num_rollout_steps_cfg = config.get('infer_timesteps')

    print("\nLoading inference dataset...")
    print(f"  Dataset: {dataset_dir}")
    with h5py.File(dataset_dir, 'r') as f:
        sample_ids = sorted(int(k) for k in f['data'].keys())
    print(f"  Found {len(sample_ids)} samples: {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    output_dir = config.get('inference_output_dir', 'outputs/rollout')
    os.makedirs(output_dir, exist_ok=True)

    for sample_id in sample_ids:
        with h5py.File(dataset_dir, 'r') as f:
            nodal_data = f[f'data/{sample_id}/nodal_data'][:]
            mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]

        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"\n  Sample {sample_id}: nodal_data {nodal_data.shape}, "
              f"mesh_edge {mesh_edge.shape[1]} (unidirectional)")

        steps = num_rollout_steps_cfg
        if steps is None:
            if num_timesteps > 1:
                steps = num_timesteps - 1
                print(f"  Auto-set rollout steps to {steps} (full trajectory)")
            else:
                raise ValueError(
                    f"infer_timesteps not specified and dataset has only {num_timesteps} "
                    "timestep(s); set infer_timesteps in the config."
                )

        ref_pos = nodal_data[:3, 0, :].T.astype(np.float32)
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

        part_ids = None
        if use_node_types:
            if num_features <= 7:
                raise ValueError(
                    f"Checkpoint requires node types but '{dataset_dir}' sample "
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

        infer_chunk_size = int(config.get('infer_query_chunk_size', 0))

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
                # Static geometry per sample -> lets GINO reuse its cached
                # neighbor edge index across all rollout timesteps.
                graph.sample_id = int(sample_id)

                if infer_chunk_size > 0:
                    encoded = model.encode_operator(graph)
                    predicted_norm = decode_in_chunks(model, encoded, graph, infer_chunk_size)
                else:
                    predicted_norm, _ = model(graph, add_noise=False)
                predicted = predicted_norm.cpu().numpy() * delta_std + delta_mean

                if num_timesteps == 1:
                    # Static case: the "delta" target IS the direct field (section 4.2).
                    current_state[:, :output_var] = predicted
                else:
                    current_state[:, :output_var] = current_state[:, :output_var] + predicted
                all_states[step + 1] = current_state[:, :output_var]

                if steps > 0 and (step % max(1, steps // 20) == 0 or step == steps - 1):
                    print(f"    step {step + 1}/{steps}")

        total_time = time.time() - rollout_start
        if steps > 0:
            print(f"  Rollout done in {total_time:.2f}s ({total_time / steps:.3f}s/step)")
        else:
            print(f"  Done in {total_time:.2f}s (no steps executed)")

        _write_rollout_output(
            output_dir, sample_id, ref_pos, mesh_edge, all_states, part_ids,
            output_var, model_path, config_filename, total_time,
        )

    print(f"\nRollout inference complete. Processed {len(sample_ids)} scene(s) = "
          f"{len(sample_ids)} output file(s).")


def _write_rollout_output(output_dir, sample_id, ref_pos, mesh_edge, all_states,
                          part_ids, output_var, model_path, config_filename, total_time_s):
    """Atomic MGN-style HDF5 writer (section 14.4): write to a temp file and
    `os.replace` into place so a crash never leaves a successful-looking
    partial output."""
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

    with h5py.File(tmp_path, 'w') as f:
        f.attrs['num_samples'] = 1
        f.attrs['num_features'] = num_save_features
        f.attrs['num_timesteps'] = num_steps + 1

        data_grp = f.create_group('data')
        sample_grp = data_grp.create_group(str(sample_id))
        sample_grp.create_dataset('nodal_data', data=nodal_data,
                                  compression='gzip', compression_opts=4)
        sample_grp.create_dataset('mesh_edge', data=mesh_edge)

        meta_grp = sample_grp.create_group('metadata')
        meta_grp.attrs['sample_id'] = sample_id
        meta_grp.attrs['num_nodes'] = num_nodes
        meta_grp.attrs['num_edges'] = mesh_edge.shape[1]
        meta_grp.attrs['num_timesteps'] = num_steps + 1
        meta_grp.attrs['model_path'] = model_path
        meta_grp.attrs['config_file'] = config_filename
        meta_grp.attrs['total_rollout_time_s'] = total_time_s

        global_meta = f.create_group('metadata')
        feature_names = [b'x_coord', b'y_coord', b'z_coord']
        feature_names += [f'output_{i}'.encode() for i in range(output_var)]
        feature_names += [b'Part No.']
        global_meta.create_dataset('feature_names', data=np.array(feature_names))
        f.flush()

    os.replace(tmp_path, final_path)
    file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"  Saved: {final_path} ({file_size_mb:.1f} MB)")
