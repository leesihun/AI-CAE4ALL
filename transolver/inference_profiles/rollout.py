"""Checkpoint-led inference (IMPLEMENTATION_PLAN.md section 11): static direct
prediction and temporal autoregressive rollout, dispatching to decoupled
two-stage inference when `infer_mode decoupled`. Reuses MeshGraphDataset and
its shared normalize_positions/normalize_node_features/denormalize_delta
helpers for point-sample construction, so training and inference cannot drift
apart (section 11's explicit requirement).
"""

import os
import time

import h5py
import numpy as np
import torch

from general_modules.config_validation import validate_checkpoint
from general_modules.mesh_dataset import (
    MeshGraphDataset, denormalize_delta, normalize_node_features, normalize_positions,
)
from general_modules.positional_features import compute_positional_features
from model.Transolver import Transolver


def _resolve_device(config):
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
    return device


def _load_checkpoint_and_overlay(config, device):
    model_path = config.get('modelpath')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, model_path)

    model_config = checkpoint['model_config']
    print("\n  Model config loaded from checkpoint:")
    for k, v in model_config.items():
        old_val = config.get(k)
        config[k] = v
        if old_val is not None and old_val != v:
            print(f"    {k}: {old_val} -> {v} (overridden by checkpoint)")
        else:
            print(f"    {k}: {v}")

    return checkpoint


def _build_model_and_load_weights(config, checkpoint, device):
    model = Transolver(config, str(device)).to(device)
    if 'ema_state_dict' in checkpoint:
        ema_sd = checkpoint['ema_state_dict']
        model_sd = {k[len('module.'):]: v for k, v in ema_sd.items() if k.startswith('module.')}
        model.load_state_dict(model_sd)
        print("  Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Loaded training weights from checkpoint (no EMA available)")
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    return model


def _write_sample_hdf5(output_dir, sample_id, steps, ref_pos, output_states, output_dim,
                        part_ids, checkpoint_path, config_filename, elapsed_s):
    """MGN-compatible layout (section 11): nodal_data = [ref xyz, predicted
    fields, Part No.], one file per sample."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rollout_sample{sample_id}_steps{steps}.h5")

    num_nodes = ref_pos.shape[0]
    num_timesteps = output_states.shape[0]
    num_save_features = 3 + output_dim + 1
    nodal_data = np.zeros((num_save_features, num_timesteps, num_nodes), dtype=np.float32)
    nodal_data[0, :, :] = ref_pos[:, 0]
    nodal_data[1, :, :] = ref_pos[:, 1]
    nodal_data[2, :, :] = ref_pos[:, 2]
    for ch in range(output_dim):
        nodal_data[3 + ch, :, :] = output_states[:, :, ch]
    if part_ids is not None:
        nodal_data[3 + output_dim, :, :] = part_ids[np.newaxis, :]

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_samples'] = 1
        f.attrs['num_features'] = num_save_features
        f.attrs['num_timesteps'] = num_timesteps

        data_grp = f.create_group('data')
        sample_grp = data_grp.create_group(str(sample_id))
        sample_grp.create_dataset('nodal_data', data=nodal_data, compression='gzip', compression_opts=4)

        meta_grp = sample_grp.create_group('metadata')
        meta_grp.attrs['sample_id'] = sample_id
        meta_grp.attrs['num_nodes'] = num_nodes
        meta_grp.attrs['num_timesteps'] = num_timesteps
        meta_grp.attrs['model_path'] = checkpoint_path
        meta_grp.attrs['config_file'] = config_filename
        meta_grp.attrs['total_rollout_time_s'] = elapsed_s

    print(f"  Saved: {output_path}")
    return output_path


def run_inference(config, config_filename='config.txt'):
    """Dispatch on infer_mode (section 11)."""
    infer_mode = config.get('infer_mode', 'direct')
    if infer_mode == 'decoupled':
        from inference_profiles.decoupled import run_decoupled_inference
        return run_decoupled_inference(config, config_filename)
    return run_direct_inference(config, config_filename)


def run_direct_inference(config, config_filename='config.txt'):
    print("\n" + "=" * 60)
    print("DIRECT INFERENCE (static prediction / temporal rollout)")
    print("=" * 60)

    device = _resolve_device(config)
    checkpoint = _load_checkpoint_and_overlay(config, device)
    model = _build_model_and_load_weights(config, checkpoint, device)

    infer_dataset_path = config.get('infer_dataset')
    dataset = MeshGraphDataset(infer_dataset_path, config)
    dataset.inherit_preprocessing_from_dict(checkpoint['normalization'])

    output_dir = config.get('inference_output_dir', 'outputs/rollout')
    modelpath = config.get('modelpath')

    if dataset.num_timesteps == 1:
        _run_static_direct(model, dataset, device, output_dir, config, modelpath, config_filename)
    else:
        _run_temporal_rollout(model, dataset, device, output_dir, config, modelpath, config_filename)

    print(f"\nInference complete. Processed {len(dataset.sample_ids)} sample(s).")


def _run_static_direct(model, dataset, device, output_dir, config, modelpath, config_filename):
    """Section 11: reproduce training input exactly -- zero physical input,
    normalized reference geometry, denormalized final field. Reusing
    MeshGraphDataset.__getitem__ guarantees this (it already zeros x_phys for
    T=1 samples), instead of re-deriving the contract by hand."""
    output_dim = config['output_var']
    with torch.no_grad():
        for idx, sample_id in enumerate(dataset.sample_ids):
            start = time.time()
            graph = dataset[idx].to(device)
            pred_norm, _ = model(graph, add_noise=False)
            pred_denorm = denormalize_delta(pred_norm.cpu().numpy(), dataset.delta_mean, dataset.delta_std)
            elapsed = time.time() - start

            ref_pos = graph.pos.cpu().numpy()
            part_ids = graph.part_ids.cpu().numpy() if graph.part_ids is not None else None
            output_states = pred_denorm[np.newaxis, :, :]  # [1, N, output_dim]

            _write_sample_hdf5(
                output_dir, sample_id, steps=0, ref_pos=ref_pos, output_states=output_states,
                output_dim=output_dim, part_ids=part_ids, checkpoint_path=modelpath,
                config_filename=config_filename, elapsed_s=elapsed,
            )


def _run_temporal_rollout(model, dataset, device, output_dir, config, modelpath, config_filename):
    """Section 11: state_t -> normalize -> predict normalized delta ->
    denormalize -> state_{t+1} = state_t + delta. Uses the shared
    normalize_positions/normalize_node_features helpers directly (state
    evolves in memory, so __getitem__ -- which always reads from disk --
    cannot be reused step-to-step)."""
    input_dim = config['input_var']
    output_dim = config['output_var']
    num_pos_features = int(config.get('positional_features', 0))
    use_node_types = config.get('use_node_types', False)
    num_rollout_steps = config.get('infer_timesteps')

    import h5py as h5

    with torch.no_grad():
        for sample_id in dataset.sample_ids:
            with h5.File(dataset.h5_file, 'r') as f:
                dset = f[f'data/{sample_id}/nodal_data']
                num_features, num_timesteps, num_nodes = dset.shape
                # Only timestep 0 seeds the rollout; state evolves in memory
                # from here on, so read just that slice -- not the whole
                # trajectory (section 4/5's I/O lesson from MeshGraphNets'
                # own dataloader history applies here too).
                step0 = dset[:, 0, :]  # [F, N]
                mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]

            steps = num_rollout_steps if num_rollout_steps is not None else num_timesteps - 1

            ref_pos = step0[:3, :].T
            current_state = step0[3:3 + input_dim, :].T.astype(np.float32).copy()
            part_ids = step0[-1, :].astype(np.int32) if (use_node_types and num_features > 7) else None

            edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)
            pos_feat = (compute_positional_features(ref_pos, edge_index, num_pos_features)
                       if num_pos_features > 0 else None)
            pos_normalized = normalize_positions(ref_pos, dataset.position_scale)
            pos_normalized_t = torch.from_numpy(pos_normalized.astype(np.float32)).to(device)

            all_states = np.zeros((steps + 1, num_nodes, output_dim), dtype=np.float32)
            all_states[0] = current_state[:, :output_dim]

            start = time.time()
            for step in range(steps):
                x_raw = np.concatenate([current_state, pos_feat], axis=1) if pos_feat is not None else current_state
                x_norm = normalize_node_features(
                    x_raw, dataset.node_mean, dataset.node_std,
                    node_types=part_ids if use_node_types else None,
                    node_type_to_idx=dataset.node_type_to_idx, num_node_types=dataset.num_node_types,
                )

                from torch_geometric.data import Data
                graph = Data(
                    x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
                    pos_normalized=pos_normalized_t,
                    edge_index=torch.from_numpy(edge_index).long().to(device),
                )
                pred_norm, _ = model(graph, add_noise=False)
                pred_denorm = denormalize_delta(pred_norm.cpu().numpy(), dataset.delta_mean, dataset.delta_std)

                current_state[:, :output_dim] = current_state[:, :output_dim] + pred_denorm
                all_states[step + 1] = current_state[:, :output_dim]

            elapsed = time.time() - start
            print(f"  Sample {sample_id}: {steps} steps in {elapsed:.2f}s")

            _write_sample_hdf5(
                output_dir, sample_id, steps=steps, ref_pos=ref_pos, output_states=all_states,
                output_dim=output_dim, part_ids=part_ids, checkpoint_path=modelpath,
                config_filename=config_filename, elapsed_s=elapsed,
            )
