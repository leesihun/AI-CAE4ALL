"""Explicit-args refactor of Transolver/inference_profiles/rollout.py's
`run_direct_inference` (static direct prediction + temporal autoregressive
rollout) with an optional decoupled two-stage decode path folded in from
inference_profiles/decoupled.py (see `query_chunk_size` below).

CPU-only: no gpu_ids branch (the `device` arg is already resolved by the
caller to torch.device('cpu')).

This driver intentionally does NOT depend on general_modules.mesh_dataset's
`MeshGraphDataset` -- that class pulls in general_modules/dataset_stats.py
(train-split z-score fitting) and general_modules/time_integration.py
(AR-OT/AR-RT window bookkeeping), both training-only machinery the bundle
never needs (mirrors the trim already applied to the neural_operator family:
general_modules/normalize.py extracts only the pure normalize/denormalize
functions). Both call sites below (static and temporal) instead read HDF5
and build torch_geometric.data.Data graphs directly, exactly reproducing
what MeshGraphDataset.__getitem__ / _run_temporal_rollout do for an
is_training=False, non-augmented sample.
"""

import time

import numpy as np
import torch
from torch_geometric.data import Data

from common.hdf5_io import list_sample_ids, read_sample, write_rollout_output
from general_modules.normalize import denormalize_delta, normalize_node_features, normalize_positions
from general_modules.positional_features import compute_positional_features
from model.Transolver import Transolver


def _load_checkpoint_and_overlay(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Minimal schema check (config_validation.validate_checkpoint is training/
    # CLI-side validation machinery and is not vendored into the bundle).
    if "checkpoint_version" not in ckpt:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' has no checkpoint_version; refusing to guess "
            "metadata for an unversioned or foreign (e.g. MeshGraphNets/Neural_Operator) "
            "checkpoint."
        )
    for required in ("model_state_dict", "normalization", "model_config"):
        if required not in ckpt:
            raise ValueError(f"Checkpoint '{checkpoint_path}' is missing required key '{required}'.")

    model_config = ckpt["model_config"]
    config = {}
    print("\n  Model config loaded from checkpoint:")
    for k, v in model_config.items():
        config[k] = v
        print(f"    {k}: {v}")

    return ckpt, config


def _build_model_and_load_weights(config, ckpt, device):
    model = Transolver(config, str(device)).to(device)
    if "ema_state_dict" in ckpt:
        ema_sd = ckpt["ema_state_dict"]
        model_sd = {k[len("module."):]: v for k, v in ema_sd.items() if k.startswith("module.")}
        # NOTE: the live Transolver/inference_profiles/rollout.py calls
        # load_state_dict without a `strict=` argument, i.e. strict=True (the
        # nn.Module default). Preserved verbatim here -- do not add
        # strict=False.
        model.load_state_dict(model_sd)
        print("  Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded training weights from checkpoint (no EMA available)")
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    return model


def run(checkpoint: str, input: str, output: str, device,
        timesteps: int = None, query_chunk_size: int = 0, **_ignored) -> str:

    ckpt, config = _load_checkpoint_and_overlay(checkpoint)
    model = _build_model_and_load_weights(config, ckpt, device)

    norm = ckpt["normalization"]
    node_mean, node_std = norm["node_mean"], norm["node_std"]
    delta_mean, delta_std = norm["delta_mean"], norm["delta_std"]
    position_scale = norm["position_scale"]
    node_type_to_idx = norm.get("node_type_to_idx")
    num_node_types = norm.get("num_node_types")

    input_var = config.get("input_var")
    output_var = config.get("output_var")
    num_pos_features = int(config.get("positional_features", 0))
    use_node_types = bool(config.get("use_node_types", False))

    sample_ids = list_sample_ids(input)
    print(f"  Found {len(sample_ids)} sample(s): {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    chunk_size = int(query_chunk_size or 0)
    last_path = None

    for sample_id in sample_ids:
        nodal_data, mesh_edge = read_sample(input, sample_id)
        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"\n  Sample {sample_id}: nodal_data {nodal_data.shape}, "
              f"mesh_edge {mesh_edge.shape[1]} (unidirectional)")

        if num_timesteps == 1:
            last_path = _run_static_direct(
                model, device, nodal_data, mesh_edge, sample_id, checkpoint, output,
                input_var, output_var, num_pos_features, use_node_types,
                node_mean, node_std, delta_mean, delta_std, position_scale,
                node_type_to_idx, num_node_types, chunk_size,
            )
        else:
            last_path = _run_temporal_rollout(
                model, device, nodal_data, mesh_edge, sample_id, checkpoint, output,
                input_var, output_var, num_pos_features, use_node_types,
                node_mean, node_std, delta_mean, delta_std, position_scale,
                node_type_to_idx, num_node_types, timesteps, chunk_size,
            )

    print(f"\nInference complete. Processed {len(sample_ids)} scene(s).")
    return last_path


def _run_static_direct(model, device, nodal_data, mesh_edge, sample_id, checkpoint, output,
                        input_var, output_var, num_pos_features, use_node_types,
                        node_mean, node_std, delta_mean, delta_std, position_scale,
                        node_type_to_idx, num_node_types, chunk_size):
    """T==1: reproduce MeshGraphDataset.__getitem__'s static branch -- zero
    physical input, normalized reference geometry, denormalized field output.
    Static case's "delta" target IS the direct field (fit that way by
    training-side dataset_stats), so denormalize_delta gives the field
    directly -- no addition to a zero state (matches Neural_Operator's T==1
    convention, and rollout.py's own comment to that effect)."""
    num_features, _, num_nodes = nodal_data.shape

    ref_pos = nodal_data[:3, 0, :].T.astype(np.float32)
    edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

    part_ids = None
    if use_node_types:
        if num_features <= 7:
            raise ValueError(
                f"Checkpoint requires node types but sample {sample_id} has only "
                f"{num_features} feature rows."
            )
        part_ids = nodal_data[-1, 0, :].astype(np.int32)

    x_phys = np.zeros((num_nodes, input_var), dtype=np.float32)
    pos_feat = None
    if num_pos_features > 0:
        pos_feat = compute_positional_features(ref_pos, edge_index, num_pos_features)
    x_raw = np.concatenate([x_phys, pos_feat], axis=1) if pos_feat is not None else x_phys

    pos_normalized = normalize_positions(ref_pos, position_scale)
    x_norm = normalize_node_features(
        x_raw, node_mean, node_std,
        node_types=part_ids if use_node_types else None,
        node_type_to_idx=node_type_to_idx, num_node_types=num_node_types,
    )

    graph = Data(
        x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
        pos=torch.from_numpy(ref_pos).to(device),
        pos_normalized=torch.from_numpy(pos_normalized.astype(np.float32)).to(device),
        edge_index=torch.from_numpy(edge_index).long().to(device),
    )
    graph.ptr = torch.tensor([0, num_nodes], dtype=torch.long, device=device)
    graph.sample_id = int(sample_id)

    start = time.time()
    with torch.no_grad():
        if chunk_size > 0:
            # Decoupled two-stage decode (inference_profiles/decoupled.py):
            # mathematically identical to the ordinary forward when
            # query_graph == cache_graph (its default), just tiled for
            # memory. Only supported for static (T==1) samples -- the native
            # decoupled.py raises NotImplementedError for T>1 datasets.
            pred_norm, _ = model.forward_decoupled(graph, infer_chunk_size=chunk_size)
        else:
            pred_norm, _ = model(graph, add_noise=False)
    pred_denorm = denormalize_delta(pred_norm.cpu().numpy(), delta_mean, delta_std)
    elapsed = time.time() - start

    all_states = pred_denorm[np.newaxis, :, :].astype(np.float32)  # [1, N, output_var]

    last_path = write_rollout_output(
        output, sample_id, ref_pos, mesh_edge, all_states, part_ids,
        output_var, checkpoint, "cae_infer/transolver", elapsed,
    )
    print(f"  Saved: {last_path}")
    return last_path


def _run_temporal_rollout(model, device, nodal_data, mesh_edge, sample_id, checkpoint, output,
                           input_var, output_var, num_pos_features, use_node_types,
                           node_mean, node_std, delta_mean, delta_std, position_scale,
                           node_type_to_idx, num_node_types, timesteps, chunk_size):
    """T>1: state_t -> normalize -> predict normalized delta -> denormalize ->
    state_{t+1} = state_t + delta. `chunk_size`/decoupled decode is not
    supported here (native decoupled.py is static-only); the arg is ignored
    with a one-time notice."""
    num_features, num_timesteps, num_nodes = nodal_data.shape

    if chunk_size > 0:
        print(f"  [transolver] query_chunk_size={chunk_size} ignored: decoupled/chunked "
              f"decode only supports static (T==1) samples; sample {sample_id} has "
              f"{num_timesteps} timesteps.")

    steps = timesteps if timesteps is not None else num_timesteps - 1

    ref_pos = nodal_data[:3, 0, :].T.astype(np.float32)
    current_state = nodal_data[3:3 + input_var, 0, :].T.astype(np.float32).copy()
    part_ids = nodal_data[-1, 0, :].astype(np.int32) if (use_node_types and num_features > 7) else None

    edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)
    pos_feat = (compute_positional_features(ref_pos, edge_index, num_pos_features)
                if num_pos_features > 0 else None)
    pos_normalized = normalize_positions(ref_pos, position_scale)
    pos_normalized_t = torch.from_numpy(pos_normalized.astype(np.float32)).to(device)
    edge_index_t = torch.from_numpy(edge_index).long().to(device)

    all_states = np.zeros((steps + 1, num_nodes, output_var), dtype=np.float32)
    all_states[0] = current_state[:, :output_var]

    start = time.time()
    with torch.no_grad():
        for step in range(steps):
            x_raw = (np.concatenate([current_state, pos_feat], axis=1)
                      if pos_feat is not None else current_state)
            x_norm = normalize_node_features(
                x_raw, node_mean, node_std,
                node_types=part_ids if use_node_types else None,
                node_type_to_idx=node_type_to_idx, num_node_types=num_node_types,
            )

            graph = Data(
                x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
                pos_normalized=pos_normalized_t,
                edge_index=edge_index_t,
            )
            pred_norm, _ = model(graph, add_noise=False)
            pred_denorm = denormalize_delta(pred_norm.cpu().numpy(), delta_mean, delta_std)

            current_state[:, :output_var] = current_state[:, :output_var] + pred_denorm
            all_states[step + 1] = current_state[:, :output_var]

            if steps > 0 and (step % max(1, steps // 20) == 0 or step == steps - 1):
                print(f"    step {step + 1}/{steps}")

    elapsed = time.time() - start
    print(f"  Sample {sample_id}: {steps} steps in {elapsed:.2f}s"
          + (f" ({elapsed / steps:.3f}s/step)" if steps > 0 else ""))

    last_path = write_rollout_output(
        output, sample_id, ref_pos, mesh_edge, all_states, part_ids,
        output_var, checkpoint, "cae_infer/transolver", elapsed,
    )
    print(f"  Saved: {last_path}")
    return last_path
