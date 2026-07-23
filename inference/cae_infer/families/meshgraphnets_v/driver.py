"""Explicit-args refactor of
`MeshGraphNets - variational/inference_profiles/rollout.py`'s `run_rollout`.
CPU-only: no gpu_ids branch, and the CUDA-only auto-VRAM VAE-batch-sizing /
OOM-retry logic is dropped (the native code itself falls back to
`vae_batch_size=1` whenever `device.type != 'cuda'`, so this bundle simply
always takes that path). The eval-dataset z_disp spread-histogram feature
(matplotlib, `eval_dataset` ground-truth compare, `os.startfile` viewer) is
diagnostic/plotting tooling, not part of the rollout contract -- dropped
entirely (documented trim; see the family report).

Shares the vanilla `meshgraphnets` family's edge-attribute / world-edge /
multiscale-coarsening graph construction verbatim (both rollouts build graphs
identically) -- see that driver's docstring for the landmine-6 detail. This
file adds the VAE / conditional-prior latent sampling on top.

[VERIFIED against the live rollout.py -- contradicts one INFERENCE_BUNDLE_PLAN.md
claim] The plan's section 5.4 / section 1 says the variational rollout
"resamples the latent per rollout step". The live code does NOT do that: a
single z is drawn once, at step 0 of each trajectory (from the conditional
prior's `.sample()` when available, else `torch.randn`), and is then held
FIXED for every remaining step of that trajectory via `fixed_z=z_batch` on
every subsequent `model(...)` call (see `_run_batch` below, and
`MeshGraphNets - variational/inference_profiles/rollout.py` lines ~600-629).
This driver preserves the live behavior exactly (sample-once-per-trajectory,
fixed for the rollout), not the plan's description. No explicit seeding is
performed anywhere in the native sampling path (no `torch.manual_seed` call
in rollout.py, `MeshGraphNets.py`, or `conditional_prior.py`) -- reproducibility
across runs depends entirely on the caller's global torch RNG state, exactly
as in the original repo. This driver does not add seeding beyond that (no
hidden manual_seed), preserving bit-for-bit parity with an unseeded native run.
"""

import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from common.hdf5_io import write_mgn_rollout_output
from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.positional_features import compute_positional_features
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


class _SampleContext:
    """Static per-scene data shared by every rollout step and z-sample.

    Ported verbatim from the native rollout.py's `_SampleContext` (same
    method bodies) -- kept as a class here too since `build_step_graph` is
    called once per (z-sample, step) pair and porting it to a flat loop would
    risk transcription drift from the live graph-construction logic.
    """

    def __init__(self, config, checkpoint_norm, ref_pos, edge_index, part_ids, device,
                 world_max_num_neighbors=64, world_edge_backend="scipy_kdtree",
                 coarse_world_edges=False):
        self.device = device
        self.ref_pos = ref_pos
        self.edge_index = edge_index
        self.num_nodes = ref_pos.shape[0]

        norm = checkpoint_norm
        self.node_mean, self.node_std = norm["node_mean"], norm["node_std"]
        self.edge_mean, self.edge_std = norm["edge_mean"], norm["edge_std"]
        self.delta_mean, self.delta_std = norm["delta_mean"], norm["delta_std"]
        if "coarse_edge_means" in norm:
            self.coarse_edge_means = norm["coarse_edge_means"]
            self.coarse_edge_stds = norm["coarse_edge_stds"]
        else:
            self.coarse_edge_means = [self.edge_mean]
            self.coarse_edge_stds = [self.edge_std]

        num_pos_features = int(config.get("positional_features", 0))
        self.pos_features = None
        if num_pos_features > 0:
            self.pos_features = compute_positional_features(ref_pos, edge_index, num_pos_features)
            print(f"  Positional features: {self.pos_features.shape}")

        self.node_type_onehot = None
        if config.get("use_node_types") and part_ids is not None:
            node_type_to_idx = norm.get("node_type_to_idx")
            num_node_types = norm.get("num_node_types")
            if node_type_to_idx is not None and num_node_types:
                indices = np.array([node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32)
                onehot = np.zeros((self.num_nodes, num_node_types), dtype=np.float32)
                onehot[np.arange(self.num_nodes), indices] = 1.0
                self.node_type_onehot = onehot

        self.use_world_edges = bool(config.get("use_world_edges"))
        self.world_edge_radius = norm.get("world_edge_radius")
        self.world_max_num_neighbors = world_max_num_neighbors
        self.world_edge_backend = (
            "torch_cluster" if world_edge_backend == "torch_cluster" and HAS_TORCH_CLUSTER
            else "scipy_kdtree"
        )
        if self.use_world_edges:
            print(f"  World edges: radius={self.world_edge_radius}, backend={self.world_edge_backend}")

        self.use_multiscale = bool(config.get("use_multiscale", False))
        self.use_coarse_world_edges = coarse_world_edges
        self.hierarchy = None
        if self.use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True but model/coarsening.py could not be imported")
            multiscale_levels = int(config.get("multiscale_levels", 1))
            raw_ct = config.get("coarsening_type", "bfs")
            if isinstance(raw_ct, list):
                coarsening_types = [str(t).strip().lower() for t in raw_ct]
            else:
                coarsening_types = [str(raw_ct).strip().lower()]
            if len(coarsening_types) == 1 and multiscale_levels > 1:
                coarsening_types = coarsening_types * multiscale_levels
            raw_vc = config.get("voronoi_clusters", None)
            if raw_vc is None:
                voronoi_clusters = [0] * multiscale_levels
            elif isinstance(raw_vc, list):
                voronoi_clusters = [int(v) for v in raw_vc]
            else:
                voronoi_clusters = [int(raw_vc)]
            if len(voronoi_clusters) == 1 and multiscale_levels > 1:
                voronoi_clusters = voronoi_clusters * multiscale_levels

            self.hierarchy = build_multiscale_hierarchy(
                edge_index, self.num_nodes, ref_pos,
                multiscale_levels, coarsening_types, voronoi_clusters,
            )
            current_n = self.num_nodes
            for level, entry in enumerate(self.hierarchy):
                method = coarsening_types[level] if level < len(coarsening_types) else "bfs"
                n_c = entry["n_c"]
                print(f"  Coarsening level {level} ({method}): {current_n} -> {n_c} nodes "
                      f"({n_c / current_n * 100:.1f}%)")
                current_n = n_c

    def build_step_graph(self, current_state):
        device = self.device
        if self.pos_features is not None:
            x_raw = np.concatenate([current_state, self.pos_features], axis=1)
        else:
            x_raw = current_state
        x_norm = (x_raw - self.node_mean) / self.node_std
        if self.node_type_onehot is not None:
            x_norm = np.concatenate([x_norm, self.node_type_onehot], axis=1)

        deformed_pos = self.ref_pos + current_state[:, :3]
        edge_attr = (compute_edge_attr(self.ref_pos, deformed_pos, self.edge_index)
                     - self.edge_mean) / self.edge_std

        DataClass = MultiscaleData if self.use_multiscale else Data
        graph = DataClass(
            x=torch.from_numpy(x_norm.astype(np.float32)).to(device),
            edge_index=torch.from_numpy(self.edge_index).long().to(device),
            edge_attr=torch.from_numpy(edge_attr.astype(np.float32)).to(device),
            pos=torch.from_numpy(self.ref_pos.astype(np.float32)).to(device),
        )

        if self.use_world_edges and self.world_edge_radius is not None:
            world_ei, world_ea = compute_world_edges(
                self.ref_pos, deformed_pos, self.edge_index,
                radius=self.world_edge_radius,
                max_num_neighbors=self.world_max_num_neighbors,
                backend=self.world_edge_backend,
                device=device,
                edge_mean=self.edge_mean, edge_std=self.edge_std,
            )
            graph.world_edge_index = torch.from_numpy(world_ei).long().to(device)
            graph.world_edge_attr = torch.from_numpy(world_ea.astype(np.float32)).to(device)
        else:
            graph.world_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            graph.world_edge_attr = torch.zeros((0, EDGE_FEATURE_DIM), dtype=torch.float32, device=device)

        if self.use_multiscale and self.hierarchy is not None:
            world_ei_for_coarse = (
                graph.world_edge_index.cpu().numpy()
                if self.use_world_edges and self.use_coarse_world_edges else None
            )
            attach_coarse_levels_to_graph(
                graph, self.hierarchy, self.ref_pos, deformed_pos,
                self.coarse_edge_means, self.coarse_edge_stds,
                device=device, world_edge_index=world_ei_for_coarse,
            )

        return graph


def _load_model_from_checkpoint(config, checkpoint, device):
    model = MeshGraphNets(config, str(device)).to(device)
    if "ema_state_dict" in checkpoint:
        ema_sd = checkpoint["ema_state_dict"]
        model_sd = {k[len("module."):]: v for k, v in ema_sd.items() if k.startswith("module.")}
        model.load_state_dict(model_sd)
        print("  Loaded EMA weights from checkpoint")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded training weights from checkpoint (no EMA available)")
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    return model


def _load_conditional_prior(config, checkpoint, model, device):
    """Return the conditional prior module to sample z from, or None (-> N(0,I))."""
    if getattr(model, "prior", None) is not None:
        model.prior.eval()
        return model.prior
    if "conditional_prior_state_dict" in checkpoint:
        from model.conditional_prior import ConditionalMixturePrior
        prior_config = dict(config)
        prior_config.update(checkpoint.get("conditional_prior_config", {}))
        prior = ConditionalMixturePrior(prior_config).to(device)
        prior.load_state_dict(checkpoint["conditional_prior_state_dict"])
        prior.eval()
        return prior
    return None


def run(checkpoint: str, input: str, output: str, device: torch.device,
        timesteps: int = None, query_chunk_size: int = 0, **_ignored) -> str:
    # query_chunk_size: no query-decode path in MGN(-variational); no-op,
    # accepted only for signature parity with the other family drivers.

    print(f"Loading checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)

    if "normalization" not in ckpt:
        raise KeyError(f"Checkpoint '{checkpoint}' does not contain normalization statistics.")
    norm = ckpt["normalization"]
    print("  Normalization stats loaded from checkpoint")

    config = {}
    model_config = ckpt.get("model_config", {})
    if not model_config:
        raise KeyError(f"Checkpoint '{checkpoint}' has no 'model_config'.")
    # Back-compat shim from the live rollout.py: pre-FM checkpoints predate
    # 'prior_family' and are all Gaussian-mixture priors.
    if ("prior_family" not in model_config
            and str(model_config.get("prior_type", "")).lower().strip() == "gnn_e2e"):
        config["prior_family"] = "gmm"
        print("  prior_family: gmm (implied by pre-FM checkpoint)")
    for k, v in model_config.items():
        config[k] = v
    print(f"  Model config loaded from checkpoint: {sorted(model_config)}")

    if config.get("use_node_types") and norm.get("num_node_types"):
        config["num_node_types"] = norm["num_node_types"]
        print(f"  Node types: {norm['num_node_types']} types, mapping: {norm.get('node_type_to_idx')}")

    # [FLAG FOR REVIEW] Same gap as the vanilla meshgraphnets driver:
    # `coarse_world_edges` sizes HybridNodeBlock vs NodeBlock at multiscale
    # levels i>0 (a real weight-shape difference) but is not written into
    # model_config by MeshGraphNets - variational/training_profiles/setup.py
    # ::build_model_config (verified). Defaults False; override via
    # infer(checkpoint, ..., coarse_world_edges=True) if a checkpoint needs it.
    coarse_world_edges = bool(_ignored.get("coarse_world_edges", False))
    config["coarse_world_edges"] = coarse_world_edges
    world_max_num_neighbors = int(_ignored.get("world_max_num_neighbors", 64))
    world_edge_backend = str(_ignored.get("world_edge_backend", "scipy_kdtree")).lower()

    print("\nInitializing model...")
    model = _load_model_from_checkpoint(config, ckpt, device)

    use_vae = bool(config.get("use_vae", False))
    vae_latent_dim = int(config.get("vae_latent_dim", 8))
    use_conditional_prior = bool(config.get("use_conditional_prior", True))
    prior_temperature = float(_ignored.get("prior_temperature", 1.0))
    # Number of independent z-sampled trajectories per scene, and how many of
    # those are advanced together in one batched forward pass. The native
    # rollout only auto-sizes/grows this batch on CUDA; on CPU it always
    # falls back to 1, so this bundle (CPU-only) does the same by default.
    num_vae_samples = int(_ignored.get("num_vae_samples", 1)) if use_vae else 1
    vae_batch_size = max(1, int(_ignored.get("vae_batch_size", 1))) if use_vae else 1

    conditional_prior = None
    if use_vae and use_conditional_prior:
        conditional_prior = _load_conditional_prior(config, ckpt, model, device)

    if use_vae:
        if conditional_prior is not None:
            if getattr(conditional_prior, "family", "gmm") == "fm":
                sampler_desc = (f"conditional flow-matching prior "
                                 f"({conditional_prior.num_steps} Euler steps, temp={prior_temperature:g})")
            else:
                sampler_desc = (f"conditional mixture prior "
                                 f"({conditional_prior.num_components} components, temp={prior_temperature:g})")
        else:
            sampler_desc = "N(0, I)"
        print(f"  VAE sampling: {num_vae_samples} sample(s) per scene "
              f"(z_dim={vae_latent_dim}, batch_size={vae_batch_size}, prior={sampler_desc})")

    input_dim = config.get("input_var")
    output_dim = config.get("output_var")

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
        print(f"\n  Sample {sample_id}: {num_nodes} nodes, {mesh_edge.shape[1]} edges, "
              f"{num_timesteps} dataset timestep(s)")

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
                    if config.get("use_node_types") and num_features > 7 else None)
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)

        ctx = _SampleContext(
            config, norm, ref_pos, edge_index, part_ids, device,
            world_max_num_neighbors=world_max_num_neighbors,
            world_edge_backend=world_edge_backend,
            coarse_world_edges=coarse_world_edges,
        )

        def _run_batch(batch_start, requested_batch_size):
            B = min(requested_batch_size, num_vae_samples - batch_start)
            states = [initial_state.copy() for _ in range(B)]
            all_states = np.zeros((B, steps + 1, num_nodes, output_dim), dtype=np.float32)
            for b in range(B):
                all_states[b, 0] = initial_state[:, :output_dim]

            # z sampled ONCE per trajectory batch, at step 0 -- then held
            # FIXED for the rest of the rollout (see module docstring: this
            # is the live behavior, not a per-step resample).
            z_batch = None
            rollout_start = time.time()

            with torch.no_grad():
                for step in range(steps):
                    if step == 0:
                        graphs = [ctx.build_step_graph(states[0])] * B
                        if use_vae:
                            if conditional_prior is not None:
                                prior_batch = Batch.from_data_list(graphs)
                                z_batch = conditional_prior.sample(
                                    prior_batch, temperature=prior_temperature,
                                ).to(device)
                            else:
                                z_batch = torch.randn(B, vae_latent_dim, device=device)
                    else:
                        graphs = [ctx.build_step_graph(states[b]) for b in range(B)]

                    batch_graph = Batch.from_data_list(graphs)
                    predicted, _, _, _, _ = model(batch_graph, fixed_z=z_batch)
                    predicted = predicted.view(B, num_nodes, output_dim).cpu().numpy()

                    for b in range(B):
                        delta = predicted[b] * ctx.delta_std + ctx.delta_mean
                        states[b][:, :output_dim] += delta
                        all_states[b, step + 1] = states[b][:, :output_dim]

                    if steps > 0 and (step % max(1, steps // 20) == 0 or step == steps - 1):
                        print(f"    step {step + 1}/{steps}")

            return B, all_states, time.time() - rollout_start

        batch_start = 0
        batch_counter = 0
        while batch_start < num_vae_samples:
            requested_batch_size = min(vae_batch_size, num_vae_samples - batch_start)
            if use_vae:
                batch_counter += 1
                print(f"  batch {batch_counter}: z-samples "
                      f"{batch_start}-{batch_start + requested_batch_size - 1} "
                      f"(B={requested_batch_size})")
            B, all_states, rollout_time = _run_batch(batch_start, requested_batch_size)

            for b in range(B):
                vae_idx = batch_start + b
                if num_vae_samples > 1:
                    filename = f"rollout_sample{sample_id}_vaesample{vae_idx}_steps{steps}.h5"
                else:
                    filename = f"rollout_sample{sample_id}_steps{steps}.h5"

                last_path = write_mgn_rollout_output(
                    output, sample_id, ref_pos, mesh_edge, all_states[b], part_ids, output_dim,
                    norm, checkpoint, "cae_infer/meshgraphnets_v", rollout_time,
                    output_filename=filename,
                    vae_sample_idx=vae_idx if use_vae else None,
                )
                print(f"  Saved: {last_path}")

            batch_start += B

    total_outputs = len(sample_ids) * num_vae_samples
    print(f"\nInference complete. Processed {len(sample_ids)} scene(s) x "
          f"{num_vae_samples} VAE sample(s) = {total_outputs} output file(s).")
    return last_path
