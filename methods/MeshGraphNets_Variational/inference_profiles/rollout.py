import json
import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch, Data

from general_modules.state_geometry import deformed_positions
from general_modules.renamed_keys import migrate_checkpoint_config
from general_modules.edge_features import EDGE_FEATURE_DIM, compute_edge_attr
from general_modules.positional_features import compute_positional_features
from general_modules.world_edges import HAS_TORCH_CLUSTER, compute_world_edges
from model.MeshGraphNets import MeshGraphNets

# Multiscale coarsening (only needed when use_multiscale=True)
try:
    from model.coarsening import MultiscaleData
    from general_modules.multiscale_helpers import (
        attach_coarse_levels_to_graph,
        build_multiscale_hierarchy,
    )
    HAS_COARSENING = True
except ImportError:
    HAS_COARSENING = False


from inference_profiles.spread import (  # noqa: F401  (re-exported)
    SPREAD_CHANNEL_DEFAULT,
    SPREAD_STATS,
    VECTOR_SPREAD_STATS,
    SPREAD_STAT_DEFAULT,
    _format_bytes,
    _is_cuda_oom,
    _spread_stat,
    _spread_stat_expr,
    _spread_index,
    _eval_dataset_spreads,
    _crps_ensemble,
    _w2_exact,
    _spread_scores,
    _open_in_viewer,
    _plot_spread_histogram,
    _eval_dataset_groups,
    _group_order,
    _spread_by_group,
)


class _SampleContext:
    """Static per-scene data shared by every rollout step and z-sample.

    Bundles the mesh (topology, reference geometry, positional features,
    optional node-type one-hot, optional coarsening hierarchy) plus the
    normalization stats needed to build a normalized step graph.
    """

    def __init__(self, config, checkpoint_norm, ref_pos, edge_index, part_ids, device,
                 cond_feat=None):
        self.geometry_config = config
        self.device = device
        self.ref_pos = ref_pos                  # [N, 3]
        self.edge_index = edge_index            # [2, 2M] bidirectional
        self.num_nodes = ref_pos.shape[0]
        self.input_dim = int(config.get('input_var'))
        # Input-only conditioning rows [N, cond_var]: known values, static
        # across every rollout step and z-sample.
        self.cond_feat = cond_feat

        norm = checkpoint_norm
        self.node_mean, self.node_std = norm['node_mean'], norm['node_std']
        self.edge_mean, self.edge_std = norm['edge_mean'], norm['edge_std']
        self.delta_mean, self.delta_std = norm['delta_mean'], norm['delta_std']
        if 'coarse_edge_means' in norm:
            self.coarse_edge_means = norm['coarse_edge_means']
            self.coarse_edge_stds = norm['coarse_edge_stds']
        else:
            self.coarse_edge_means = [self.edge_mean]
            self.coarse_edge_stds = [self.edge_std]

        # Positional features (geometry + topology, static across steps)
        num_pos_features = int(config.get('positional_features', 0))
        self.pos_features = None
        if num_pos_features > 0:
            self.pos_features = compute_positional_features(
                ref_pos, edge_index, num_pos_features
            )
            print(f"  Positional features: {self.pos_features.shape}")

        # Node-type one-hot (static across steps and z-samples)
        self.node_type_onehot = None
        if config.get('use_node_types') and part_ids is not None:
            node_type_to_idx = norm.get('node_type_to_idx')
            num_node_types = norm.get('num_node_types')
            if node_type_to_idx is not None and num_node_types:
                indices = np.array([node_type_to_idx[int(t)] for t in part_ids], dtype=np.int32)
                onehot = np.zeros((self.num_nodes, num_node_types), dtype=np.float32)
                onehot[np.arange(self.num_nodes), indices] = 1.0
                self.node_type_onehot = onehot

        # World edges
        self.use_world_edges = bool(config.get('use_world_edges'))
        self.world_edge_radius = norm.get('world_edge_radius')
        self.world_max_num_neighbors = config.get('world_max_num_neighbors', 64)
        requested_backend = config.get('world_edge_backend', 'scipy_kdtree').lower()
        self.world_edge_backend = (
            'torch_cluster' if requested_backend == 'torch_cluster' and HAS_TORCH_CLUSTER
            else 'scipy_kdtree'
        )
        if self.use_world_edges:
            print(f"  World edges: radius={self.world_edge_radius}, backend={self.world_edge_backend}")

        # Coarsening hierarchy (static — same for all rollout steps)
        self.use_multiscale = bool(config.get('use_multiscale', False))
        self.use_coarse_world_edges = bool(config.get('coarse_world_edges', False))
        self.hierarchy = None
        if self.use_multiscale:
            if not HAS_COARSENING:
                raise ImportError("use_multiscale=True but model/coarsening.py could not be imported")
            multiscale_levels = int(config.get('multiscale_levels', 1))
            raw_ct = config.get('coarsening_type', 'bfs')
            if isinstance(raw_ct, list):
                coarsening_types = [str(t).strip().lower() for t in raw_ct]
            else:
                coarsening_types = [str(raw_ct).strip().lower()]
            if len(coarsening_types) == 1 and multiscale_levels > 1:
                coarsening_types = coarsening_types * multiscale_levels
            raw_vc = config.get('voronoi_clusters', None)
            if raw_vc is None:
                voronoi_clusters = [0] * multiscale_levels
            elif isinstance(raw_vc, list):
                voronoi_clusters = [int(v) for v in raw_vc]
            else:
                voronoi_clusters = [int(raw_vc)]
            if len(voronoi_clusters) == 1 and multiscale_levels > 1:
                voronoi_clusters = voronoi_clusters * multiscale_levels

            # FPS picks its first seed with an unseeded np.random.randint
            # (model/coarsening.py::_fps_euclidean), so an unpinned rollout
            # builds a DIFFERENT partition on every run -- measured ~50% of node
            # pairs regrouped between two builds of the same mesh. Training, by
            # contrast, seeds per sample id (multiscale_cache.py::variant_seed),
            # so it always saw one fixed partition.
            #
            # `hierarchy_seed` pins the draw. Give it a LIST and the rollout
            # builds one hierarchy per seed and rotates them across z-sample
            # batches: the partition stops being a per-scene lottery whose bad
            # outcome poisons all inference_samples at once, and becomes a nuisance
            # variable the sample set marginalises over. Each trajectory records
            # the seed it used, so an outlier can be traced back to a partition.
            raw_seed = config.get('hierarchy_seed')
            if raw_seed is None:
                self.hierarchy_seeds = [None]
            elif isinstance(raw_seed, (list, tuple)):
                self.hierarchy_seeds = [int(s) for s in raw_seed]
            else:
                self.hierarchy_seeds = [int(raw_seed)]

            self.hierarchies = []
            for hseed in self.hierarchy_seeds:
                rng_state = np.random.get_state() if hseed is not None else None
                if hseed is not None:
                    np.random.seed(int(hseed) & 0x7FFFFFFF)
                try:
                    self.hierarchies.append(build_multiscale_hierarchy(
                        edge_index, self.num_nodes, ref_pos,
                        multiscale_levels, coarsening_types, voronoi_clusters,
                    ))
                finally:
                    if rng_state is not None:
                        np.random.set_state(rng_state)

            if self.hierarchy_seeds == [None]:
                print("  Coarsening hierarchy: UNPINNED -- this rollout is not "
                      "reproducible. Set `hierarchy_seed` (a list rotates "
                      "partitions across z-sample batches).")
            else:
                print(f"  Coarsening hierarchy: {len(self.hierarchies)} partition(s), "
                      f"seeds={self.hierarchy_seeds}"
                      + (" (rotated across z-sample batches)"
                         if len(self.hierarchies) > 1 else " (fixed)"))

            self.hierarchy = self.hierarchies[0]
            self.active_hierarchy_idx = 0
            current_n = self.num_nodes
            for level, entry in enumerate(self.hierarchy):
                method = coarsening_types[level] if level < len(coarsening_types) else 'bfs'
                n_c = entry['n_c']
                print(f"  Coarsening level {level} ({method}): {current_n} → {n_c} nodes "
                      f"({n_c / current_n * 100:.1f}%)")
                current_n = n_c

    def use_hierarchy(self, batch_index):
        """Select which cached partition the next batch of graphs is built on."""
        if not getattr(self, 'hierarchies', None):
            return
        self.active_hierarchy_idx = int(batch_index) % len(self.hierarchies)
        self.hierarchy = self.hierarchies[self.active_hierarchy_idx]

    def active_hierarchy_seed(self):
        seeds = getattr(self, 'hierarchy_seeds', [None])
        return seeds[getattr(self, 'active_hierarchy_idx', 0) % len(seeds)]

    def build_step_graph(self, current_state):
        """Build one normalized PyG graph for the given physical state [N, input_dim]."""
        device = self.device
        # Same [state | conditions | positional] layout the dataloader builds
        # (general_modules/mesh_dataset.py::__getitem__).
        blocks = [current_state]
        if self.cond_feat is not None:
            blocks.append(self.cond_feat)
        if self.pos_features is not None:
            blocks.append(self.pos_features)
        x_raw = np.concatenate(blocks, axis=1) if len(blocks) > 1 else current_state
        x_norm = (x_raw - self.node_mean) / self.node_std
        if self.node_type_onehot is not None:
            x_norm = np.concatenate([x_norm, self.node_type_onehot], axis=1)

        deformed_pos = deformed_positions(self.ref_pos, current_state, self.geometry_config, self.input_dim)
        edge_attr = (compute_edge_attr(self.ref_pos, deformed_pos, self.edge_index, self.geometry_config.get('periodic_box'))
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
                periodic_box=self.geometry_config.get('periodic_box'),
            )

        return graph



def _rollout_feature_names(dataset_dir, output_dim):
    """Channel names written into each rollout file.

    Prefer the inference dataset's own declared names (metadata/feature_names)
    for the coordinate and state rows, so an evaluator can pair prediction and
    truth channels by name. The legacy NASA-CRM list below used to be written
    for *every* dataset, so ex9's ux/uy came out labelled x_disp(mm)/y_disp(mm),
    matched nothing but x/y/z_coord in the truth file, and the Studio scored the
    coordinates as a perfect model. The trailing row is the node-type/part id.
    """
    legacy = [b'x_coord', b'y_coord', b'z_coord',
              b'x_disp(mm)', b'y_disp(mm)', b'z_disp(mm)',
              b'stress(MPa)', b'Part No.']
    keep = 3 + int(output_dim)
    names = None
    try:
        with h5py.File(dataset_dir, 'r') as f:
            if 'metadata/feature_names' in f:
                raw = f['metadata/feature_names'][()]
                names = [n if isinstance(n, bytes) else str(n).encode('utf-8') for n in raw]
    except Exception:
        names = None
    if names and len(names) >= keep:
        return np.array(names[:keep] + [b'Part No.'])
    return np.array(legacy[:keep] + [b'Part No.'])

def _save_rollout_h5(output_path, sample_id, all_states, ctx, part_ids, output_dim,
                     num_steps, model_path, config_filename, rollout_time_s,
                     vae_sample_idx=None, hierarchy_seed=None, hierarchy_idx=None,
                     feature_names=None):
    """Write one trajectory to HDF5 following DATASET_FORMAT.md.

    all_states: [num_steps + 1, N, output_dim] predicted physical states.
    """
    num_nodes = ctx.num_nodes
    num_save_features = 3 + output_dim + 1  # xyz + outputs + part number

    # nodal_data layout: [x, y, z, <output channels>, part_number]
    nodal_data = np.zeros((num_save_features, num_steps + 1, num_nodes), dtype=np.float32)
    nodal_data[0, :, :] = ctx.ref_pos[:, 0]
    nodal_data[1, :, :] = ctx.ref_pos[:, 1]
    nodal_data[2, :, :] = ctx.ref_pos[:, 2]
    for ch in range(output_dim):
        nodal_data[3 + ch, :, :] = all_states[:, :, ch]
    if part_ids is not None:
        nodal_data[3 + output_dim, :, :] = part_ids[np.newaxis, :]

    # mesh_edge stored unidirectional (first half of the bidirectional index)
    mesh_edge = ctx.edge_index[:, :ctx.edge_index.shape[1] // 2]

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_samples'] = 1
        f.attrs['num_features'] = num_save_features
        f.attrs['num_timesteps'] = num_steps + 1
        # The evaluator has to know how many of these rows are PREDICTIONS.
        # With only num_features it had to infer them, which is how the
        # coordinate rows and the trailing node-type row reached a score.
        f.attrs['output_var'] = int(output_dim)

        sample_grp = f.create_group('data').create_group(str(sample_id))
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
        meta_grp.attrs['total_rollout_time_s'] = rollout_time_s
        if vae_sample_idx is not None:
            meta_grp.attrs['vae_sample_idx'] = vae_sample_idx
        # Which coarsening partition produced this trajectory. Lets an outlier be
        # attributed to a partition rather than to the latent draw.
        if hierarchy_idx is not None:
            meta_grp.attrs['hierarchy_idx'] = int(hierarchy_idx)
        meta_grp.attrs['hierarchy_seed'] = (
            -1 if hierarchy_seed is None else int(hierarchy_seed))

        if feature_names is None:
            feature_names = _rollout_feature_names(None, output_dim)
        meta_grp.create_dataset('feature_min',  data=np.array([nodal_data[i].min()  for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset('feature_max',  data=np.array([nodal_data[i].max()  for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset('feature_mean', data=np.array([nodal_data[i].mean() for i in range(num_save_features)], dtype=np.float32))
        meta_grp.create_dataset('feature_std',  data=np.array([nodal_data[i].std()  for i in range(num_save_features)], dtype=np.float32))

        global_meta = f.create_group('metadata')
        global_meta.create_dataset('feature_names', data=feature_names)
        norm_grp = global_meta.create_group('normalization_params')
        norm_grp.create_dataset('node_mean',  data=ctx.node_mean)
        norm_grp.create_dataset('node_std',   data=ctx.node_std)
        norm_grp.create_dataset('edge_mean',  data=ctx.edge_mean)
        norm_grp.create_dataset('edge_std',   data=ctx.edge_std)
        norm_grp.create_dataset('delta_mean', data=ctx.delta_mean)
        norm_grp.create_dataset('delta_std',  data=ctx.delta_std)
        f.flush()


def _load_weights(model, state_dict, source):
    """Load weights, dropping tensors the current model no longer defines.

    Checkpoints trained before the auxiliary term became a peak-to-valley loss
    on the decoded field carry `aux_decoder.*` parameters. That head is gone, so
    a strict load rejects them and every pre-change checkpoint -- including the
    trained sweep arms a new run is compared against -- becomes unloadable. The
    head was training-only and never ran at inference, so dropping its weights
    changes no prediction.
    """
    stale = [k for k in state_dict if k.startswith('aux_decoder.')]
    if stale:
        state_dict = {k: v for k, v in state_dict.items() if k not in stale}
        print(f'  Dropped {len(stale)} legacy aux_decoder tensor(s) '
              f'(training-only head, no longer in the model)')
    # Checkpoints written before latent standardization existed carry no
    # prior.z_shift / prior.z_scale. Their defaults ARE the old behaviour
    # (identity), so absence is not an error -- but only for these two.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed = {'prior.z_shift', 'prior.z_scale'}
    unresolved = [k for k in missing if k not in allowed]
    if unresolved or unexpected:
        raise RuntimeError(
            f'checkpoint does not match the model: missing {unresolved}, '
            f'unexpected {list(unexpected)}')
    if missing:
        print(f'  Legacy checkpoint: {len(missing)} latent-standardization '
              f'buffer(s) defaulted to identity')
    print(f'  Loaded {source} from checkpoint')


def _load_model_from_checkpoint(config, checkpoint, device):
    """Rebuild MeshGraphNets from a checkpoint and load (EMA-preferred) weights."""
    model = MeshGraphNets(config, str(device)).to(device)

    if 'ema_state_dict' in checkpoint:
        # AveragedModel saves keys as "module.<key>" + "n_averaged" buffer.
        # Strip the "module." prefix to get the raw model state dict directly,
        # avoiding a fragile AveragedModel reconstruction at inference.
        ema_sd = checkpoint['ema_state_dict']
        model_sd = {k[len('module.'):]: v for k, v in ema_sd.items() if k.startswith('module.')}
        _load_weights(model, model_sd, 'EMA weights')
    else:
        _load_weights(model, checkpoint['model_state_dict'],
                      'training weights (no EMA available)')
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Checkpoint valid loss: {checkpoint.get('valid_loss', 'unknown')}")
    return model


def _load_conditional_prior(config, checkpoint, model, device):
    """Return the conditional prior module to sample z from, or None.

    The joint-trained prior is a submodule of MeshGraphNets and was already
    loaded by load_state_dict. Legacy checkpoints stored a separately-trained
    ConditionalMixturePrior state dict instead.
    """
    if getattr(model, 'prior', None) is not None:
        model.prior.eval()
        return model.prior
    if 'conditional_prior_state_dict' in checkpoint:
        from model.conditional_prior import ConditionalMixturePrior
        prior_config = dict(config)
        prior_config.update(checkpoint.get('conditional_prior_config', {}))
        prior = ConditionalMixturePrior(prior_config).to(device)
        prior.load_state_dict(checkpoint['conditional_prior_state_dict'])
        prior.eval()
        return prior
    return None


def run_rollout(config, config_filename='config.txt'):
    """
    Perform autoregressive rollout inference.

    Given an initial condition from an HDF5 dataset and a pretrained model
    checkpoint, iteratively predicts the next state from the current state:

        1. Load initial state at t=0
        2. For each step t -> t+1:
            a. Normalize current state -> build graph
            b. Forward pass -> predicted normalized delta
            c. Denormalize delta;  state_{t+1} = state_t + delta
        3. Save all predicted timesteps to HDF5

    With use_vae=True, `inference_samples` independent trajectories are generated
    per scene, each with its own z drawn from the conditional prior (fallback:
    N(0, I)). `inference_batch_size` trajectories are advanced together in one
    batched forward pass.
    """
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE ROLLOUT INFERENCE")
    print("=" * 60)

    # ---- Device --------------------------------------------------------
    gpu_ids = config.get('gpu_ids')
    if not isinstance(gpu_ids, list):
        gpu_ids = [gpu_ids]
    if torch.cuda.is_available() and gpu_ids[0] >= 0:
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # ---- Checkpoint (weights + normalization + model_config) ------------
    model_path = config.get('modelpath')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'normalization' not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain normalization statistics. "
            "Re-train or re-save the model with the updated training code that "
            "includes normalization stats in the checkpoint."
        )
    norm = checkpoint['normalization']
    print(f"  Normalization stats loaded from checkpoint")

    # Override config with model_config from checkpoint if available
    if 'model_config' in checkpoint:
        model_config = migrate_checkpoint_config(checkpoint['model_config'])
        # Back-compat: checkpoints saved before prior_family existed are all
        # Gaussian-mixture priors. Without this shim the new 'fm' default
        # would rebuild them as FM priors and state_dict loading would fail.
        if ('prior_family' not in model_config
                and str(model_config.get('prior_type', '')).lower().strip() == 'gnn_e2e'):
            config['prior_family'] = 'gmm'
            print("\n  prior_family: gmm (implied by pre-FM checkpoint)")
        print(f"\n  Model config loaded from checkpoint:")
        for k, v in model_config.items():
            old_val = config.get(k)
            config[k] = v
            if old_val is not None and old_val != v:
                print(f"    {k}: {old_val} -> {v} (overridden by checkpoint)")
            else:
                print(f"    {k}: {v}")
    else:
        print(f"\n  WARNING: No model_config in checkpoint, using config file values")

    if config.get('use_node_types') and norm.get('num_node_types'):
        config['num_node_types'] = norm['num_node_types']
        print(f"  Node types: {norm['num_node_types']} types, mapping: {norm.get('node_type_to_idx')}")

    # ---- Model + prior ---------------------------------------------------
    print("\nInitializing model...")
    model = _load_model_from_checkpoint(config, checkpoint, device)

    use_vae = config.get('use_vae', False)
    vae_latent_dim = int(config.get('vae_latent_dim', 8))
    use_conditional_prior = bool(config.get('use_conditional_prior', True))
    prior_temperature = float(config.get('prior_temperature', 1.0))
    inference_samples = int(config.get('inference_samples', 1)) if use_vae else 1
    raw_inference_batch_size = config.get('inference_batch_size', 1)
    auto_inference_batch_size = (
        use_vae
        and isinstance(raw_inference_batch_size, str)
        and raw_inference_batch_size.strip().lower() == 'auto'
    )
    if use_vae and not auto_inference_batch_size:
        inference_batch_size = max(1, int(raw_inference_batch_size))
    else:
        inference_batch_size = 1
    inference_batch_vram_fraction = float(config.get('inference_batch_vram_fraction', 0.70))
    inference_batch_vram_fraction = min(max(inference_batch_vram_fraction, 0.01), 1.0)
    inference_batch_size_min = max(1, int(config.get('inference_batch_size_min', 1)))
    inference_batch_size_max = max(0, int(config.get('inference_batch_size_max', 0)))

    conditional_prior = None
    if use_vae and use_conditional_prior:
        conditional_prior = _load_conditional_prior(config, checkpoint, model, device)

    if use_vae:
        if conditional_prior is not None:
            if getattr(conditional_prior, 'family', 'gmm') == 'fm':
                sampler_desc = (f"conditional flow-matching prior "
                                f"({conditional_prior.num_steps} {conditional_prior.solver.capitalize()} steps, "
                                f"temp={prior_temperature:g})")
            else:
                sampler_desc = (f"conditional mixture prior "
                                f"({conditional_prior.num_components} components, temp={prior_temperature:g})")
        else:
            sampler_desc = "N(0, I)"
        batch_desc = (
            f"auto, target={inference_batch_vram_fraction:.0%} free VRAM"
            if auto_inference_batch_size else str(inference_batch_size)
        )
        if auto_inference_batch_size and device.type != 'cuda':
            batch_desc = "auto requested, using 1 on CPU"
        print(f"  VAE sampling: {inference_samples} sample(s) per scene "
              f"(z_dim={vae_latent_dim}, batch_size={batch_desc}, prior={sampler_desc})")

    # ---- Rollout inputs --------------------------------------------------
    dataset_dir = config.get('infer_dataset')
    num_rollout_steps = config.get('infer_timesteps')
    input_dim = config.get('input_var')
    output_dim = config.get('output_var')
    cond_dim = int(config.get('cond_var', 0) or 0)
    # Resolved once, not per sample: names come from the inference dataset.
    rollout_feature_names = _rollout_feature_names(dataset_dir, output_dim)
    trained_timesteps = config.get('num_timesteps')
    static_training = trained_timesteps is not None and int(trained_timesteps) == 1

    # Inline z_disp spread-histogram (generated vs ground-truth eval dataset).
    # Enabled when an `eval_dataset` is given in the config (no GT -> no compare).
    output_dir = config.get('inference_output_dir', '../../output/meshgraphnets-v/rollout')
    # One HDF5 is written per (scene, VAE sample). A distribution study wants
    # hundreds of draws per scene, which is hundreds of files per scene and tens
    # of thousands across a sweep -- so `save_rollouts False` keeps the sampling
    # and the histogram but writes no trajectory files. The spread values are
    # still collected, printed, and dumped to spread_values.npz.
    save_rollouts = bool(config.get('save_rollouts', True))
    eval_dataset = config.get('eval_dataset')
    make_histogram = bool(config.get('make_histogram', use_vae and eval_dataset is not None))
    histogram_bins = int(config.get('histogram_bins', 60))
    # Latent inflation factors for the FM prior (see ConditionalFMPrior.sample_n).
    # A single value applies to every draw; a list is cycled batch by batch so
    # ONE inference pass yields the whole lam-vs-spread curve, each generated
    # spread tagged with the lam that produced it (spread_values.npz: gen_lam).
    _infl = config.get('latent_inflation', 1.0)
    latent_inflation = [float(v) for v in (_infl if isinstance(_infl, (list, tuple)) else [_infl])]
    if any(v <= 0 for v in latent_inflation):
        raise ValueError(f'latent_inflation must be positive, got {latent_inflation}')
    histogram_clip_quantile = float(config.get('histogram_clip_quantile', 0.0))
    show_histogram = bool(config.get('show_histogram', True))
    # Output-channel index the spread statistic is taken over; the ground-truth
    # side reads nodal_data row 3 + this. See SPREAD_CHANNEL_DEFAULT; a list of
    # channels goes with a vector stat (SPREAD_STATS).
    z_gen_idx, spread_stat = _spread_index(config)
    spread_label = str(config.get('spread_label', f'channel {z_gen_idx}'))
    generated_spreads = []  # one spread scalar per generated rollout trajectory
    # Which scene each of those came from, same order. This is what lets
    # the scorer split within-scene ensemble width from between-scene
    # variation of the conditional mean; pooled, the two are inseparable.
    generated_scene_ids = []
    generated_lams = []       # the inflation factor each draw was produced with

    print(f"\nLoading initial conditions...")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Rollout steps: {num_rollout_steps}")

    with h5py.File(dataset_dir, 'r') as f:
        sample_ids = sorted([int(k) for k in f['data'].keys()])
    print(f"  Found {len(sample_ids)} samples: {sample_ids[:10]}{'...' if len(sample_ids) > 10 else ''}")

    os.makedirs(output_dir, exist_ok=True)

    for sample_id in sample_ids:
        with h5py.File(dataset_dir, 'r') as f:
            nodal_data = f[f'data/{sample_id}/nodal_data'][:]  # [features, time, nodes]
            mesh_edge = f[f'data/{sample_id}/mesh_edge'][:]    # [2, M]

        num_features, num_timesteps, num_nodes = nodal_data.shape
        print(f"\n{'=' * 60}")
        print(f"Sample {sample_id}: {num_nodes} nodes, {mesh_edge.shape[1]} edges, "
              f"{num_timesteps} dataset timestep(s)")

        # Resolve rollout length for this sample
        steps = num_rollout_steps
        if steps is None:
            if num_timesteps > 1:
                steps = num_timesteps - 1
                print(f"  Auto-set rollout steps to {steps} (full trajectory)")
            else:
                raise ValueError(
                    f"infer_timesteps not specified and dataset has only {num_timesteps} "
                    f"timestep(s). Please set infer_timesteps in the config."
                )

        # nodal_data layout: [x, y, z, x_disp, y_disp, z_disp, stress, (part_number)]
        ref_pos = nodal_data[:3, 0, :].T                     # [N, 3]

        # Static training (T=1) feeds a ZERO state and regresses the field
        # directly; temporal training feeds state_t and regresses a delta.
        # Inference must reproduce whichever contract the checkpoint was fit
        # under -- seeding a statically-trained model with the ground-truth
        # field both leaks the answer and divides by its degenerate zero-state
        # std. Fall back to the inference file's own T for older checkpoints.
        is_static = static_training if trained_timesteps is not None else (num_timesteps == 1)
        if is_static:
            initial_state = np.zeros((num_nodes, input_dim), dtype=np.float32)
            if steps != 1:
                print(f"  INFO: checkpoint was trained on static (T=1) data; "
                      f"clamping {steps} requested step(s) to 1.")
                steps = 1
        else:
            initial_state = nodal_data[3:3 + input_dim, 0, :].T.astype(np.float32)  # [N, input_dim]

        # Input-only conditioning rows: always the real stored values, never
        # zeroed, never advanced by the rollout.
        cond_feat = None
        if cond_dim > 0:
            cond_start = 3 + input_dim
            cond_feat = nodal_data[cond_start:cond_start + cond_dim, 0, :].T.astype(np.float32)

        part_ids = (nodal_data[-1, 0, :].astype(np.int32)
                    if config.get('use_node_types') and num_features > 7 else None)
        edge_index = np.concatenate([mesh_edge, mesh_edge[[1, 0], :]], axis=1)  # [2, 2M]

        ctx = _SampleContext(config, norm, ref_pos, edge_index, part_ids, device,
                             cond_feat=cond_feat)

        batch_index = {'n': 0}

        def _run_batch(batch_start, requested_batch_size):
            # Rotate the coarsening partition per batch so a single unlucky
            # partition cannot poison every z-sample of this scene at once.
            ctx.use_hierarchy(batch_index['n'])
            batch_index['n'] += 1
            # Cycle the inflation list across batches; every draw in this
            # batch is tagged with it in _save_batch.
            batch_index['lam'] = latent_inflation[(batch_index['n'] - 1) % len(latent_inflation)]
            B = min(requested_batch_size, inference_samples - batch_start)
            states = [initial_state.copy() for _ in range(B)]
            all_states = np.zeros((B, steps + 1, num_nodes, output_dim), dtype=np.float32)
            for b in range(B):
                all_states[b, 0] = initial_state[:, :output_dim]

            z_batch = None  # sampled once per trajectory batch at step 0
            rollout_start = time.time()

            with torch.no_grad():
                for step in range(steps):
                    # Step 0: all B states are identical; build one graph and
                    # replicate. Step 1+: states have diverged, so build each graph.
                    if step == 0:
                        graphs = [ctx.build_step_graph(states[0])] * B
                        if use_vae:
                            prior_batch = Batch.from_data_list(graphs)
                            if conditional_prior is not None:
                                z_batch = conditional_prior.sample(
                                    prior_batch, temperature=prior_temperature,
                                    inflation=batch_index['lam'],
                                ).to(device)
                            else:
                                # Independent noise PER z-slot (and per coarse
                                # node for latent_type local/hybrid), matching
                                # how training draws z; a single [B, D] z
                                # replicated across slots is a distribution the
                                # decoder never trained on. The coarse-node rows
                                # stay valid for every step: the hierarchy is
                                # fixed within a batch.
                                z_batch = getattr(model, 'model', model).standard_normal_z(
                                    prior_batch.to(device))
                    else:
                        graphs = [ctx.build_step_graph(states[b]) for b in range(B)]

                    batch_graph = Batch.from_data_list(graphs)
                    predicted, _, _, _, _ = model(batch_graph, fixed_z=z_batch)
                    predicted = predicted.view(B, num_nodes, output_dim).cpu().numpy()

                    for b in range(B):
                        delta = predicted[b] * ctx.delta_std + ctx.delta_mean
                        states[b][:, :output_dim] += delta
                        all_states[b, step + 1] = states[b][:, :output_dim]

            return B, all_states, time.time() - rollout_start

        def _save_batch(batch_start, B, all_states, rollout_time):
            # Save one HDF5 per trajectory
            for b in range(B):
                vae_idx = batch_start + b

                if make_histogram and output_dim > np.max(z_gen_idx):
                    generated_spreads.append(
                        # [b, -1] first: one index expression mixing scalars,
                        # a slice and a channel LIST moves the list axis to
                        # the front ([C, N]), which mag_max would misread.
                        _spread_stat(all_states[b, -1][:, z_gen_idx], spread_stat)
                    )
                    generated_scene_ids.append(str(sample_id))
                    generated_lams.append(float(batch_index['lam']))

                if not save_rollouts:
                    continue

                if inference_samples > 1:
                    filename = f"rollout_sample{sample_id}_vaesample{vae_idx}_steps{steps}.h5"
                else:
                    filename = f"rollout_sample{sample_id}_steps{steps}.h5"
                output_path = os.path.join(output_dir, filename)

                _save_rollout_h5(
                    output_path, sample_id, all_states[b], ctx, part_ids, output_dim,
                    steps, model_path, config_filename, rollout_time,
                    vae_sample_idx=vae_idx if use_vae else None,
                    hierarchy_seed=ctx.active_hierarchy_seed(),
                    hierarchy_idx=getattr(ctx, 'active_hierarchy_idx', 0),
                    feature_names=rollout_feature_names,
                )
                if b == B - 1 or B <= 4:
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"  Saved {filename}  ({size_mb:.1f} MB)")

        active_inference_batch_size = inference_batch_size
        batch_start = 0

        if use_vae and auto_inference_batch_size and device.type == 'cuda' and inference_samples > 0:
            print(f"\nRollout: {inference_samples} z-sample(s) x {steps} step(s), "
                  f"batch_size=auto (target {inference_batch_vram_fraction:.0%} of free VRAM)")
            print("  Auto batch probe: running z-sample 0 with batch_size=1")
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            free_before, total_bytes = torch.cuda.mem_get_info(device)
            allocated_before = torch.cuda.memory_allocated(device)
            reserved_before = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)

            B, all_states, rollout_time = _run_batch(0, 1)

            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            probe_extra_allocated = max(0, peak_allocated - allocated_before)
            probe_extra_reserved = max(0, peak_reserved - reserved_before)
            probe_extra = max(1, probe_extra_allocated, probe_extra_reserved)
            torch.cuda.empty_cache()
            free_after, _ = torch.cuda.mem_get_info(device)

            _save_batch(0, B, all_states, rollout_time)
            batch_start = B

            remaining = inference_samples - batch_start
            target_extra = int(free_after * inference_batch_vram_fraction)
            raw_choice = max(1, int(target_extra // probe_extra))
            if inference_batch_size_max:
                raw_choice = min(raw_choice, inference_batch_size_max)
            if remaining > 0:
                active_inference_batch_size = min(
                    remaining,
                    max(inference_batch_size_min, raw_choice),
                )
            else:
                active_inference_batch_size = 1

            print("  Auto batch memory:")
            print(f"    free before probe: {_format_bytes(free_before)} / {_format_bytes(total_bytes)}")
            print(f"    free after probe:  {_format_bytes(free_after)}")
            print(f"    probe peak extra:  {_format_bytes(probe_extra)} "
                  f"(allocated {_format_bytes(probe_extra_allocated)}, "
                  f"reserved {_format_bytes(probe_extra_reserved)})")
            print(f"    target extra:      {_format_bytes(target_extra)} "
                  f"({inference_batch_vram_fraction:.0%} of free)")
            print(f"    selected batch:    {active_inference_batch_size}")
        else:
            if use_vae:
                if auto_inference_batch_size:
                    print("\nRollout: auto VAE batch sizing requested, but CUDA is unavailable; "
                          "using batch_size=1")
                num_batches = (inference_samples + active_inference_batch_size - 1) // active_inference_batch_size
                print(f"\nRollout: {inference_samples} z-sample(s) x {steps} step(s), "
                      f"batch_size={active_inference_batch_size} -> {num_batches} forward batch(es)")

        batch_counter = 0
        while batch_start < inference_samples:
            requested_batch_size = min(active_inference_batch_size, inference_samples - batch_start)
            if use_vae:
                batch_counter += 1
                print(f"  batch {batch_counter}: z-samples "
                      f"{batch_start}-{batch_start + requested_batch_size - 1} "
                      f"(B={requested_batch_size})")
            try:
                B, all_states, rollout_time = _run_batch(batch_start, requested_batch_size)
            except RuntimeError as exc:
                if (auto_inference_batch_size and device.type == 'cuda'
                        and requested_batch_size > 1 and _is_cuda_oom(exc)):
                    new_batch_size = max(1, requested_batch_size // 2)
                    print(f"  CUDA OOM at batch_size={requested_batch_size}; "
                          f"retrying z-sample {batch_start} with batch_size={new_batch_size}")
                    active_inference_batch_size = new_batch_size
                    if use_vae:
                        batch_counter -= 1
                    torch.cuda.empty_cache()
                    continue
                raise
            _save_batch(batch_start, B, all_states, rollout_time)
            batch_start += B

    total_outputs = len(sample_ids) * inference_samples
    print(f"\nRollout inference complete. Processed {len(sample_ids)} scene(s) x "
          f"{inference_samples} VAE sample(s) = {total_outputs} output file(s).")

    # ---- z_disp spread histogram: generated vs ground-truth eval set ------
    if make_histogram:
        print("\n" + "=" * 60)
        print("HISTOGRAM COMPARE "
              f"({_spread_stat_expr(spread_label, spread_stat)}, final timestep)")
        print("=" * 60)
        if not eval_dataset:
            print("  Skipped: no `eval_dataset` set in config "
                  "(needed for the ground-truth comparison).")
        elif not os.path.exists(str(eval_dataset)):
            print(f"  Skipped: eval_dataset not found: {eval_dataset}")
        elif not generated_spreads:
            print("  Skipped: no generated spread values were collected "
                  f"(output_var={output_dim} has no channel {z_gen_idx}).")
        else:
            try:
                gt, gt_scene = _eval_dataset_spreads(
                    str(eval_dataset), z_gen_idx, spread_stat)
                gen = np.asarray(generated_spreads, dtype=np.float64)
                print(f"  GT spread values  (1 per eval sample): {gt.size:,}")
                print(f"  Gen spread values (1 per rollout):     {gen.size:,}")
                # Machine-readable lines so a sweep report can tabulate these
                # without re-reading the plot; prefixed so a log grep is exact.
                for tag, col in (("GT", gt), ("GEN", gen)):
                    print(f"  [SPREAD] {tag} mean={col.mean():.6e} std={col.std():.6e} "
                          f"min={col.min():.6e} max={col.max():.6e} n={col.size}")
                lams = np.asarray(generated_lams, dtype=np.float64)
                if lams.size == gen.size and np.unique(lams).size > 1:
                    gsd = float(gt.std()) or 1.0
                    for lam in np.unique(lams):
                        col = gen[lams == lam]
                        print(f"  [SPREAD] GEN lam={lam:g} mean={col.mean():.6e} "
                              f"std={col.std():.6e} sd_ratio={col.std() / gsd:.3f} n={col.size}")
                # Raw values too: with save_rollouts False this is the ONLY
                # surviving record, and it lets a report overlay every arm on one
                # axis instead of leaving 16 separate PNGs to eyeball.
                npz_path = os.path.join(output_dir, 'spread_values.npz')
                os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
                # Scene labels alongside the values: see
                # _eval_dataset_spreads for why they are load-bearing.
                np.savez_compressed(
                    npz_path, gt=gt, gen=gen,
                    gt_scene=np.asarray([str(s) for s in gt_scene]),
                    gen_scene=np.asarray([str(s) for s in generated_scene_ids]),
                    gen_lam=np.asarray(generated_lams, dtype=np.float64))
                print(f"  [SPREAD] values -> {npz_path}")

                # Ensemble scores, paired scene by scene. Written beside the
                # values so a campaign can tabulate every arm without rerunning
                # inference; the headline four are stamped onto the figure.
                scores = _spread_scores(gt, gt_scene, gen, generated_scene_ids)
                scores['channel'] = z_gen_idx
                scores['label'] = spread_label
                scores['stat'] = spread_stat
                scores['eval_dataset'] = str(eval_dataset)
                for key, target in (('crps_norm', 0.0), ('spread_skill', 1.0),
                                    ('pit_ks', 0.0), ('pit_tails', 0.04),
                                    ('w1_norm', 0.0), ('dmean_norm', 0.0),
                                    ('sd_ratio', 1.0)):
                    if key in scores:
                        print(f"  [SCORE] {key}={scores[key]:.4f} (target {target:g})")
                scores_path = os.path.join(output_dir, 'spread_metrics.json')
                with open(scores_path, 'w', encoding='utf-8') as fh:
                    json.dump(scores, fh, indent=2)
                print(f"  [SCORE] metrics -> {scores_path}")

                hist_path = os.path.join(output_dir, 'histogram_compare.png')
                _plot_spread_histogram(
                    gt, gen, hist_path,
                    eval_path=str(eval_dataset), rollout_dir=output_dir,
                    bins=histogram_bins, clip_quantile=histogram_clip_quantile,
                    show=show_histogram, label=spread_label, scores=scores,
                    stat=spread_stat,
                )
            except Exception as exc:  # never let plotting break a finished rollout
                print(f"  Histogram generation failed: {exc}")
            # One figure per held-out geometry (split_group_attr), beside the
            # pooled one: the pooled curve hides per-geometry width errors.
            group_attr = config.get('split_group_attr')
            if group_attr and generated_spreads:
                try:
                    groups = _eval_dataset_groups(str(eval_dataset), str(group_attr))
                    if groups:
                        _spread_by_group(
                            gt, gt_scene, gen, generated_scene_ids, groups,
                            str(group_attr), output_dir,
                            eval_path=str(eval_dataset), bins=histogram_bins,
                            clip_quantile=histogram_clip_quantile,
                            label=spread_label, stat=spread_stat)
                    else:
                        print(f"  Per-group histograms skipped: no `{group_attr}` "
                              f"metadata attribute in {eval_dataset}")
                except Exception as exc:
                    print(f"  Per-group histogram generation failed: {exc}")
