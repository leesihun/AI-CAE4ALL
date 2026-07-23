"""Explicit-args refactor of MeshGraphNets/inference_profiles/rollout.py's
`run_rollout`. CPU-only: no gpu_ids branch (the native GPU/CPU selection
block is dropped -- `device` is always torch.device('cpu'), resolved by the
caller before `run()` is invoked).

Unlike the Neural_Operator family (which drops edge attributes entirely),
MGN consumes edge features every step and optionally world edges + a
multiscale coarsening hierarchy. This driver reproduces the live rollout's
graph-construction logic verbatim (INFERENCE_BUNDLE_PLAN.md section 5.3 /
landmine 6):
  - edge_attr: `compute_edge_attr(ref_pos, deformed_pos, edge_index)`,
    z-score normalized with the checkpoint's `edge_mean`/`edge_std`.
  - world edges: built from `normalization['world_edge_radius']` via
    `general_modules/world_edges.py::compute_world_edges`. The bundle never
    vendors torch_cluster, so `HAS_TORCH_CLUSTER` is always False and the
    scipy `cKDTree` backend always runs -- matches the plan's CPU-only
    constraint (section 1 item 6) and its landmine-1 parity caveat (a
    checkpoint trained with the torch_cluster radius backend must be
    re-validated against the scipy path; the two backends are not
    guaranteed bit-identical).
  - multiscale coarsening: `model/coarsening.py`'s `MultiscaleData` +
    `general_modules/multiscale_helpers.py`'s `build_multiscale_hierarchy` /
    `attach_coarse_levels_to_graph`, using `normalization['coarse_edge_means'
    /'coarse_edge_stds']` when present (falls back to the flat edge stats,
    exactly like the native rollout).

`query_chunk_size` is accepted for signature parity with the other family
drivers but is a no-op: MGN has no query-decode path (confirmed against the
live rollout.py -- it has no `encode_operator`/`decode_in_chunks` analogue).
"""

import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Data

from common.hdf5_io import write_mgn_rollout_output
from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.positional_features import compute_positional_features
from general_modules.removed_feature_guard import validate_checkpoint
from general_modules.world_edges import HAS_TORCH_CLUSTER, compute_world_edges
from model.MeshGraphNets import MeshGraphNets

try:
    from model.coarsening import MultiscaleData
    from general_modules.multiscale_helpers import (
        attach_coarse_levels_to_graph,
        build_multiscale_hierarchy,
    )
    HAS_COARSENING = True
except ImportError:
    HAS_COARSENING = False


def run(checkpoint: str, input: str, output: str, device: torch.device,
        timesteps: int = None, query_chunk_size: int = 0, **_ignored) -> str:

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(ckpt, checkpoint)

    if "normalization" not in ckpt:
        raise KeyError(
            f"Checkpoint '{checkpoint}' does not contain normalization statistics."
        )
    norm = ckpt["normalization"]
    node_mean, node_std = norm["node_mean"], norm["node_std"]
    edge_mean, edge_std = norm["edge_mean"], norm["edge_std"]
    delta_mean, delta_std = norm["delta_mean"], norm["delta_std"]
    if "coarse_edge_means" in norm:
        coarse_edge_means = norm["coarse_edge_means"]
        coarse_edge_stds = norm["coarse_edge_stds"]
    else:
        coarse_edge_means = [edge_mean]
        coarse_edge_stds = [edge_std]
    print("  Normalization stats loaded from checkpoint")

    # Overlay every checkpointed architecture key into a fresh config dict --
    # same overlay pattern as Neural_Operator/model/factory.py::build_model_from_checkpoint.
    config = {}
    model_config = ckpt.get("model_config", {})
    if not model_config:
        raise KeyError(f"Checkpoint '{checkpoint}' has no 'model_config'.")
    for k, v in model_config.items():
        config[k] = v
    print(f"  Model config loaded from checkpoint: {sorted(model_config)}")

    use_node_types = config.get("use_node_types")
    node_type_to_idx = norm.get("node_type_to_idx")
    num_node_types = norm.get("num_node_types")
    if use_node_types and num_node_types:
        config["num_node_types"] = num_node_types
        print(f"  Node types: {num_node_types} types, mapping: {node_type_to_idx}")

    use_world_edges = bool(config.get("use_world_edges"))
    world_edge_radius = norm.get("world_edge_radius")
    # Runtime-only world-edge knobs -- not part of model_config (they don't
    # affect weight shapes), so they aren't recoverable from the checkpoint.
    # Defaults match MeshGraphNets/training_profiles native config.txt
    # defaults; override via infer(checkpoint, ..., world_max_num_neighbors=...)
    # if a non-default value matters for a given checkpoint.
    world_max_num_neighbors = int(_ignored.get("world_max_num_neighbors", 64))
    requested_backend = str(_ignored.get("world_edge_backend", "scipy_kdtree")).lower()
    world_edge_backend = (
        "torch_cluster" if requested_backend == "torch_cluster" and HAS_TORCH_CLUSTER
        else "scipy_kdtree"
    )
    if use_world_edges:
        print(f"  World edges: radius={world_edge_radius}, backend={world_edge_backend}")

    # [FLAG FOR REVIEW] `coarse_world_edges` is read by both the model
    # constructor (EncoderProcessorDecoder._build_multiscale_processor, via
    # `config.get('coarse_world_edges', False)`) and the rollout's coarse
    # hierarchy attachment -- but it is NOT written into model_config by
    # MeshGraphNets/training_profiles/setup.py::build_model_config (verified
    # against the live function). For a checkpoint trained with
    # use_multiscale=True, use_world_edges=True, AND coarse_world_edges=True,
    # levels i>0 use HybridNodeBlock (3*latent_dim input, extra
    # world_eb_module) instead of NodeBlock (2*latent_dim) -- a real weight
    # SHAPE difference load_state_dict cannot paper over. Since this knob
    # isn't recoverable from the checkpoint, this driver defaults it to False
    # (matching both the model's and the native rollout's own default);
    # pass infer(checkpoint, ..., coarse_world_edges=True) to override for a
    # checkpoint that needs it.
    coarse_world_edges = bool(_ignored.get("coarse_world_edges", False))
    config["coarse_world_edges"] = coarse_world_edges

    print("\nInitializing model...")
    model = MeshGraphNets(config, str(device)).to(device)
    if "ema_state_dict" in ckpt:
        ema_sd = ckpt["ema_state_dict"]
        model_sd = {k[len("module."):]: v for k, v in ema_sd.items() if k.startswith("module.")}
        model.load_state_dict(model_sd)
        print("  Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded training weights from checkpoint (no EMA available)")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")

    input_dim = config.get("input_var")
    output_dim = config.get("output_var")
    num_pos_features = int(config.get("positional_features", 0))

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

        ref_pos = nodal_data[:3, 0, :].T
        initial_state = nodal_data[3:3 + input_dim, 0, :].T
        part_ids = (nodal_data[-1, 0, :].astype(np.int32)
                    if use_node_types and num_features > 7 else None)
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

        pos_features = None
        if num_pos_features > 0:
            pos_features = compute_positional_features(ref_pos, edge_index, num_pos_features)
            print(f"  Positional features: {pos_features.shape}")

        use_multiscale = bool(config.get("use_multiscale", False))
        multiscale_levels = int(config.get("multiscale_levels", 1))
        raw_ct = config.get("coarsening_type", "bfs")
        if isinstance(raw_ct, list):
            coarsening_types = [str(t).strip().lower() for t in raw_ct]
        else:
            coarsening_types = [str(raw_ct).strip().lower()] * multiscale_levels
        if len(coarsening_types) == 1 and multiscale_levels > 1:
            coarsening_types = coarsening_types * multiscale_levels

        raw_vc = config.get("voronoi_clusters", None)
        if raw_vc is None:
            voronoi_clusters = [0] * multiscale_levels
        elif isinstance(raw_vc, list):
            voronoi_clusters = [int(v) for v in raw_vc]
        else:
            voronoi_clusters = [int(raw_vc)] * multiscale_levels
        if len(voronoi_clusters) == 1 and multiscale_levels > 1:
            voronoi_clusters = voronoi_clusters * multiscale_levels

        coarse_hierarchy = None
        if use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True but model/coarsening.py could not be imported")
            coarse_hierarchy = build_multiscale_hierarchy(
                edge_index, num_nodes, ref_pos,
                multiscale_levels, coarsening_types, voronoi_clusters,
            )
            current_n_report = num_nodes
            for level, entry in enumerate(coarse_hierarchy):
                method = coarsening_types[level] if level < len(coarsening_types) else "bfs"
                n_c = entry["n_c"]
                print(f"  Coarsening level {level} ({method}): {current_n_report} -> {n_c} nodes "
                      f"({n_c / current_n_report * 100:.1f}%)")
                current_n_report = n_c

        all_states = np.zeros((steps + 1, num_nodes, output_dim), dtype=np.float32)
        all_states[0] = initial_state[:, :output_dim]
        current_state = initial_state.copy()

        rollout_start = time.time()
        with torch.no_grad():
            for step in range(steps):
                if pos_features is not None:
                    x_raw = np.concatenate([current_state, pos_features], axis=1)
                else:
                    x_raw = current_state
                x_norm = (x_raw - node_mean) / node_std

                if use_node_types and part_ids is not None and node_type_to_idx is not None:
                    node_type_indices = np.array(
                        [node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32
                    )
                    node_type_onehot = np.zeros((num_nodes, num_node_types), dtype=np.float32)
                    node_type_onehot[np.arange(num_nodes), node_type_indices] = 1.0
                    x_norm = np.concatenate([x_norm, node_type_onehot], axis=1)

                displacement = current_state[:, :3]
                deformed_pos = ref_pos + displacement
                edge_attr_raw = compute_edge_attr(ref_pos, deformed_pos, edge_index)
                edge_attr_norm = (edge_attr_raw - edge_mean) / edge_std

                DataClass = MultiscaleData if use_multiscale else Data
                graph = DataClass(
                    x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
                    edge_index=torch.from_numpy(edge_index).long().to(device),
                    edge_attr=torch.from_numpy(edge_attr_norm.astype(np.float32)).to(device),
                    pos=torch.from_numpy(ref_pos.astype(np.float32)).to(device),
                )

                if use_world_edges and world_edge_radius is not None:
                    world_ei, world_ea = compute_world_edges(
                        ref_pos, deformed_pos, edge_index,
                        radius=world_edge_radius,
                        max_num_neighbors=world_max_num_neighbors,
                        backend=world_edge_backend,
                        device=device,
                        edge_mean=edge_mean, edge_std=edge_std,
                    )
                    graph.world_edge_index = torch.from_numpy(world_ei).long().to(device)
                    graph.world_edge_attr = torch.from_numpy(world_ea.astype(np.float32)).to(device)
                else:
                    graph.world_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    graph.world_edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)

                if use_multiscale and coarse_hierarchy is not None:
                    world_ei_for_coarse = (
                        graph.world_edge_index.cpu().numpy()
                        if use_world_edges and coarse_world_edges else None
                    )
                    attach_coarse_levels_to_graph(
                        graph, coarse_hierarchy, ref_pos, deformed_pos,
                        coarse_edge_means, coarse_edge_stds,
                        device=device, world_edge_index=world_ei_for_coarse,
                    )

                predicted_delta_norm, _ = model(graph)
                predicted_delta = predicted_delta_norm.cpu().numpy() * delta_std + delta_mean
                current_state[:, :output_dim] = current_state[:, :output_dim] + predicted_delta
                all_states[step + 1] = current_state[:, :output_dim]

                if steps > 0 and (step % max(1, steps // 20) == 0 or step == steps - 1):
                    print(f"    step {step + 1}/{steps}")

        total_time = time.time() - rollout_start
        print(f"  Rollout done in {total_time:.2f}s" +
              (f" ({total_time / steps:.3f}s/step)" if steps > 0 else ""))

        last_path = write_mgn_rollout_output(
            output, sample_id, ref_pos, mesh_edge, all_states, part_ids, output_dim,
            norm, checkpoint, "cae_infer/meshgraphnets", total_time,
        )
        print(f"  Saved: {last_path}")

    print(f"\nInference complete. Processed {len(sample_ids)} scene(s).")
    return last_path
