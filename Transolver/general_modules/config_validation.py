"""Centralized config validation for the Transolver runtime.

Two entry points:
    validate_config(config, source)      -- static checks; called by load_config
                                             on every parse, before any dataset load.
    validate_temporal_contract(config)   -- called once num_timesteps is known
                                             (training_profiles.setup.build_dataset_splits);
                                             enforces input_var == output_var for T > 1.

Section 8 of IMPLEMENTATION_PLAN.md is the authoritative spec for every rule below.
"""

import os

VALID_MODES = {'train', 'inference'}
VALID_ATTENTION_KERNELS = {'naive', 'slice_space'}
VALID_INFER_MODES = {'direct', 'decoupled'}
# 'ddp'        -- data parallelism (one full mesh per rank; ~N throughput).
# 'node_shard' -- Phase 7: split ONE mesh's nodes across ranks and all-reduce
#                 the slice aggregates in the forward, pooling VRAM across GPUs.
VALID_PARALLEL_MODES = {'ddp', 'node_shard'}
# Accept the MeshGraphNets name the user knows, but map it to the mechanism that
# actually pools VRAM for Transolver. MGN's layer-wise split has no analogue here
# (params are tiny; memory is per-node activations) -- see IMPLEMENTATION_PLAN.md 6.6.
PARALLEL_MODE_ALIASES = {'model_split': 'node_shard'}

# MGN-only keys that change architecture or preprocessing. Their presence means
# an MGN config was likely passed by mistake; failing loudly beats silently
# ignoring a switch the user thought was active (section 4/8).
MGN_ARCHITECTURE_KEYS = {
    'edge_var', 'message_passing_num', 'mp_per_level', 'coarsening_type',
    'voronoi_clusters', 'multiscale_levels', 'world_radius_multiplier',
    'world_max_num_neighbors', 'world_edge_backend', 'coarse_world_edges',
}

REMOVED_MODES = {'train_prior', 'train_with_prior'}


def _format_list(values):
    return ', '.join(sorted(str(v) for v in values))


def _as_list(value):
    return value if isinstance(value, list) else [value]


def validate_config(config, source='configuration'):
    """Static validation: no dataset access, safe to run immediately after parsing."""
    mode = str(config.get('mode', '')).lower()
    if mode in REMOVED_MODES:
        raise ValueError(
            f"Unsupported mode '{mode}' in {source}; this checkout supports only "
            f"'train' and 'inference'."
        )
    if 'mode' in config and mode not in VALID_MODES:
        raise ValueError(f"{source}: mode must be one of {sorted(VALID_MODES)}, got '{mode}'")

    model_name = str(config.get('model', '')).lower()
    if 'model' in config and model_name != 'transolver':
        raise ValueError(f"{source}: model must be 'transolver', got '{model_name}'")

    mgn_keys = MGN_ARCHITECTURE_KEYS.intersection(config.keys())
    if mgn_keys:
        raise ValueError(
            f"{source} contains MeshGraphNets-only architecture/preprocessing keys "
            f"that Transolver does not support: {_format_list(mgn_keys)}. "
            f"This checkout has no code path for them; remove them from the config "
            f"rather than relying on them being silently ignored."
        )

    parallel_mode = str(config.get('parallel_mode', 'ddp')).lower().strip()
    if parallel_mode in PARALLEL_MODE_ALIASES:
        canonical = PARALLEL_MODE_ALIASES[parallel_mode]
        print(
            f"{source}: parallel_mode '{parallel_mode}' is accepted as an alias for "
            f"'{canonical}'. This is node-sharded Physics-Attention: one mesh's nodes "
            f"are split across ranks and the slice aggregates (num/den) are all-reduced "
            f"inside the forward, pooling VRAM across GPUs. It is NOT MeshGraphNets-style "
            f"layer splitting (which has no Transolver analogue -- see IMPLEMENTATION_PLAN.md 6.6)."
        )
        if isinstance(config, dict):
            config['parallel_mode'] = canonical
        parallel_mode = canonical
    if parallel_mode not in VALID_PARALLEL_MODES:
        raise ValueError(
            f"{source}: parallel_mode must be one of {sorted(VALID_PARALLEL_MODES)} "
            f"(alias: {sorted(PARALLEL_MODE_ALIASES)}); got '{parallel_mode}'."
        )
    if parallel_mode == 'node_shard':
        kernel = config.get('attention_kernel', 'naive')
        if kernel != 'slice_space':
            raise ValueError(
                f"{source}: parallel_mode 'node_shard' requires attention_kernel "
                f"'slice_space' (the naive kernel materializes node-space projections "
                f"and cannot shard the reduction dimension); got attention_kernel='{kernel}'."
            )

    if config.get('use_world_edges', False):
        raise ValueError(f"{source}: use_world_edges must be False (baseline Transolver does not consume edges)")
    if config.get('use_multiscale', False):
        raise ValueError(f"{source}: use_multiscale must be False (not implemented in this checkout)")

    coord_norm = config.get('coordinate_normalization', 'centered_isotropic')
    if coord_norm != 'centered_isotropic':
        raise ValueError(
            f"{source}: coordinate_normalization must be 'centered_isotropic' in phase 1, "
            f"got '{coord_norm}'"
        )

    # --- architecture shape checks (only when the relevant keys are present,
    #     so this function is also usable on partial/model_config dicts) ---
    if 'latent_dim' in config and config['latent_dim'] <= 0:
        raise ValueError(f"{source}: latent_dim must be > 0, got {config['latent_dim']}")
    if 'num_layers' in config and config['num_layers'] <= 0:
        raise ValueError(f"{source}: num_layers must be > 0, got {config['num_layers']}")
    if 'num_heads' in config and config['num_heads'] <= 0:
        raise ValueError(f"{source}: num_heads must be > 0, got {config['num_heads']}")
    if 'slice_num' in config and config['slice_num'] <= 0:
        raise ValueError(f"{source}: slice_num must be > 0, got {config['slice_num']}")
    if 'latent_dim' in config and 'num_heads' in config:
        if config['latent_dim'] % config['num_heads'] != 0:
            raise ValueError(
                f"{source}: latent_dim ({config['latent_dim']}) must be divisible by "
                f"num_heads ({config['num_heads']})"
            )

    attention_kernel = config.get('attention_kernel', 'naive')
    if attention_kernel not in VALID_ATTENTION_KERNELS:
        raise ValueError(
            f"{source}: attention_kernel must be one of {sorted(VALID_ATTENTION_KERNELS)}, "
            f"got '{attention_kernel}'"
        )

    chunk_size = config.get('chunk_size', 0)
    if not isinstance(chunk_size, int) or chunk_size < 0:
        raise ValueError(f"{source}: chunk_size must be a non-negative integer, got {chunk_size!r}")
    infer_chunk_size = config.get('infer_chunk_size', 0)
    if not isinstance(infer_chunk_size, int) or infer_chunk_size < 0:
        raise ValueError(
            f"{source}: infer_chunk_size must be a non-negative integer, got {infer_chunk_size!r}"
        )
    if chunk_size > 0 and attention_kernel != 'slice_space':
        raise ValueError(
            f"{source}: chunk_size > 0 requires attention_kernel == 'slice_space' "
            f"(the naive kernel materializes node-space projections and cannot tile); "
            f"got attention_kernel='{attention_kernel}', chunk_size={chunk_size}"
        )

    infer_mode = config.get('infer_mode', 'direct')
    if infer_mode not in VALID_INFER_MODES:
        raise ValueError(
            f"{source}: infer_mode must be one of {sorted(VALID_INFER_MODES)}, got '{infer_mode}'"
        )

    t_init = config.get('temperature_init', 0.5)
    t_min = config.get('temperature_min', 0.1)
    t_max = config.get('temperature_max', 5.0)
    if not (0 < t_min <= t_init <= t_max):
        raise ValueError(
            f"{source}: temperature bounds must satisfy 0 < temperature_min <= "
            f"temperature_init <= temperature_max, got min={t_min}, init={t_init}, max={t_max}"
        )

    output_var = config.get('output_var')
    loss_weights = config.get('feature_loss_weights')
    if output_var is not None and loss_weights is not None:
        loss_weights = _as_list(loss_weights)
        if len(loss_weights) != output_var:
            raise ValueError(
                f"{source}: feature_loss_weights has {len(loss_weights)} entries, "
                f"expected output_var={output_var}"
            )

    # --- path existence, mode-dependent ---
    if mode == 'train':
        dataset_dir = config.get('dataset_dir')
        if dataset_dir and not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"{source}: dataset_dir does not exist: {dataset_dir}")
    elif mode == 'inference':
        modelpath = config.get('modelpath')
        if modelpath and not os.path.exists(modelpath):
            raise FileNotFoundError(f"{source}: modelpath does not exist: {modelpath}")
        infer_dataset = config.get('infer_dataset')
        if infer_dataset and not os.path.exists(infer_dataset):
            raise FileNotFoundError(f"{source}: infer_dataset does not exist: {infer_dataset}")


def validate_temporal_contract(config):
    """Called once num_timesteps is known (after dataset load).

    Temporal (T > 1) autoregressive datasets require input_var == output_var:
    the delta target is computed as state[t+1, :output_var] - state[t, :input_var],
    which only broadcasts correctly when the widths match (plan section 5.3).
    """
    num_timesteps = config.get('num_timesteps')
    if num_timesteps is not None and num_timesteps > 1:
        input_var = config.get('input_var')
        output_var = config.get('output_var')
        if input_var != output_var:
            raise ValueError(
                f"Temporal dataset (num_timesteps={num_timesteps}) requires "
                f"input_var == output_var, got input_var={input_var}, output_var={output_var}"
            )


def validate_checkpoint(checkpoint, source='checkpoint'):
    """Reject checkpoints missing required schema, or from an incompatible model."""
    required = ('model_state_dict', 'normalization', 'model_config')
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise ValueError(f"{source} is missing required keys: {_format_list(missing)}")

    model_config = checkpoint.get('model_config', {})
    if isinstance(model_config, dict):
        validate_config(model_config, f"{source} model_config")

    version = checkpoint.get('checkpoint_version')
    if version is None:
        raise ValueError(
            f"{source} has no checkpoint_version; refusing to guess metadata for an "
            f"unversioned or foreign (e.g. MeshGraphNets) checkpoint."
        )
