"""Config key registry and fail-fast validation (IMPLEMENTATION_PLAN.md section 11).

Mirrors MeshGraphNets' general_modules/removed_feature_guard.py fail-fast
pattern: unknown or legacy keys raise immediately, before any HDF5 file is
opened. Unlike MGN, this repository additionally validates per-model
architecture keys (section 11.4) and only after the active `model` is known.
"""

MODEL_NAMES = {"point_deeponet", "deeponet", "fno", "gino"}

# Keys recognized regardless of which model is selected (section 11.2).
COMMON_KEYS = {
    "model", "mode", "gpu_ids", "parallel_mode",
    "log_file_dir", "modelpath", "dataset_dir", "infer_dataset",
    "inference_output_dir", "infer_timesteps", "split_seed",
    "input_var", "output_var", "cond_var", "feature_loss_weights",
    "positional_features", "use_node_types",
    "coordinate_normalization", "operator_dim", "dimension_tolerance",
    "grid_padding", "out_of_bounds_policy",
    "sdf_source", "sdf_sidecar",
    "global_condition_features", "integration_weight_source",
    "training_epochs", "batch_size", "learningr", "weight_decay",
    "warmup_epochs", "num_workers", "prefetch_factor", "grad_accum_steps",
    "max_grad_norm", "use_parallel_stats", "train_eval_subset_size",
    "std_noise", "noise_gamma", "noise_std_ratio",
    # Time integration (training_profiles/ar_rollout.py): ar_ot (default) or ar_rt.
    "time_integration",
    "augment_geometry",
    "use_amp", "use_checkpointing", "use_ema", "ema_decay", "use_compile",
    "val_interval", "test_interval", "test_max_batches", "test_batch_idx",
    "plot_feature_idx", "display_trainset", "display_testset",
    "checkpoint_interval",
    "train_query_chunk_size", "infer_query_chunk_size",
    "write_preprocessing",
    "use_world_edges", "use_multiscale",
    "profile_batches",
    "pipeline_microbatches",
    # injected at runtime by setup.py / dataset construction, not user-set,
    # but must be tolerated when a saved config is echoed back:
    "num_timesteps", "num_node_types", "_pin_memory", "_ddp_port", "log_dir",
    "_paper_target_mean", "_paper_target_std", "_norm_stats",
}

POINT_DEEPONET_KEYS = {
    "point_variant", "point_sensor_count", "point_sampling",
    "point_resample_each_epoch", "point_hidden_channels", "point_feature_dim",
    "pointnet_depth", "pointnet_activation", "pointnet_norm",
    "point_branch_merge", "point_condition_depth", "point_trunk_depth",
    "point_refiner_depth", "point_siren_omega0", "point_output_activation",
}

DEEPONET_KEYS = {
    "deeponet_branch_source", "deeponet_sensor_resolution",
    "deeponet_hidden_channels", "deeponet_branch_depth",
    "deeponet_trunk_depth", "deeponet_basis_dim", "deeponet_activation",
    "deeponet_multi_output", "deeponet_max_branch_params",
}

FNO_KEYS = {
    "fno_grid_resolution", "fno_modes", "fno_hidden_channels", "fno_layers",
    "fno_use_channel_mlp", "fno_norm", "fno_variant",
}

GINO_KEYS = {
    "gino_variant", "gino_grid_resolution", "gino_fno_modes",
    "gino_fno_hidden_channels", "gino_fno_layers", "gino_in_radius",
    "gino_out_radius", "gino_kernel_hidden", "gino_max_empty_input_fraction",
    "gino_query_chunk_size", "gino_use_torch_cluster",
    "gino_group_shared_geometry", "gino_cache_neighbors",
    # Opt-in ShapeNet Car paper decoder. These keys are inert for mesh_state.
    "gino_tucker_rank", "gino_channel_mlp_expansion", "gino_lifting_hidden",
    "gino_kernel_widths", "gino_projection_widths", "gino_max_num_neighbors",
    "gino_pos_embedding_type", "gino_coord_embed_dim",
    "gino_include_grid_coordinates", "gino_domain_padding",
}

ALL_MODEL_KEYS = {
    "point_deeponet": POINT_DEEPONET_KEYS,
    "deeponet": DEEPONET_KEYS,
    "fno": FNO_KEYS,
    "gino": GINO_KEYS,
}

ALL_KNOWN_KEYS = set(COMMON_KEYS)
for _keys in ALL_MODEL_KEYS.values():
    ALL_KNOWN_KEYS |= _keys

# Legacy/removed keys from the MGN checkout (message-passing GNN, VAE branch,
# world edges, multiscale) that must never silently do nothing here.
# parallel_mode=model_split IS supported (parallelism/, MGN-style pipeline
# split, fno/gino only) -- it is a parallel_mode value, not a key.
REMOVED_KEYS = {
    "message_passing_num", "latent_dim", "edge_var",
    "world_radius_multiplier", "world_max_num_neighbors", "world_edge_backend",
    "coarse_world_edges", "multiscale_levels", "mp_per_level",
    "coarsening_type", "voronoi_clusters", "coarse_cache_per_worker",
    "use_vae", "vae_latent_dim", "vae_mp_layers", "vae_graph_aware",
    "free_bits", "posterior_min_std", "lambda_mmd", "lambda_kl", "lambda_det",
}

# Models with a sequential latent stack that pipeline model-split can cut.
# DeepONet/Point-DeepONet are parallel branch/trunk pairs -- nothing to cut.
SPLIT_CAPABLE_MODELS = {"fno", "gino"}


def _format_list(values):
    return ", ".join(sorted(str(v) for v in values))


def _as_list(value):
    return value if isinstance(value, list) else [value]


def _require_choice(config, name, default, accepted, source):
    value = str(config.get(name, default)).lower()
    if value not in accepted:
        raise ValueError(
            f"{source}: {name} must be one of {_format_list(accepted)}, got '{value}'."
        )
    return value


def _require_int_entries(config, name, minimum, source, lengths=None):
    if name not in config:
        return []
    raw_values = _as_list(config[name])
    if lengths is not None and len(raw_values) not in lengths:
        raise ValueError(
            f"{source}: {name} must contain "
            f"{' or '.join(str(v) for v in sorted(lengths))} values, got {raw_values!r}."
        )
    parsed = []
    for raw in raw_values:
        if not isinstance(raw, int) or isinstance(raw, bool) or raw < minimum:
            raise ValueError(
                f"{source}: {name} entries must be integers >= {minimum}, got {raw_values!r}."
            )
        parsed.append(raw)
    return parsed


def _require_positive_number(config, name, source, allow_zero=False):
    if name not in config:
        return
    raw = config[name]
    if isinstance(raw, bool):
        raise ValueError(f"{source}: {name} must be a number, got {raw!r}.")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {name} must be a number, got {raw!r}.") from exc
    if value < 0 if allow_zero else value <= 0:
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{source}: {name} must be {relation}, got {raw!r}.")


def _validate_static_model_options(config, model_name, source):
    """Validate config-only architecture constraints before opening a dataset.

    The model factory repeats dataset-dependent checks later. Keeping the
    closed choices and simple ranges here prevents a config from passing the
    native preflight only to fail during model construction.
    """
    if model_name == "point_deeponet":
        variant = _require_choice(config, "point_variant", "mesh_state", {"mesh_state"}, source)
        _require_choice(config, "point_sampling", "random", {"random"}, source)
        _require_choice(config, "pointnet_activation", "relu", {"relu"}, source)
        _require_choice(config, "pointnet_norm", "batch", {"batch"}, source)
        _require_choice(config, "point_branch_merge", "sum", {"sum"}, source)
        _require_choice(config, "point_output_activation", "identity", {"identity", "tanh"}, source)
        if variant != "mesh_state":  # defensive; _require_choice currently makes this unreachable
            raise ValueError(f"{source}: point_variant={variant} is not executable by the current dataset adapter.")
        _require_int_entries(config, "point_sensor_count", 0, source)
        for name in (
            "point_hidden_channels", "point_feature_dim", "pointnet_depth",
            "point_condition_depth", "point_trunk_depth", "point_refiner_depth",
        ):
            _require_int_entries(config, name, 1, source)
        _require_positive_number(config, "point_siren_omega0", source)

    elif model_name == "deeponet":
        _require_choice(config, "deeponet_branch_source", "fixed_sensors", {"fixed_sensors"}, source)
        _require_choice(config, "deeponet_multi_output", "split_both", {"split_both"}, source)
        _require_choice(config, "deeponet_activation", "silu", {"relu", "silu", "gelu", "tanh"}, source)
        resolution = _require_int_entries(
            config, "deeponet_sensor_resolution", 2, source, lengths={2, 3}
        )
        configured_dim = config.get("operator_dim", "auto")
        if configured_dim in (2, 3) and resolution and len(resolution) != configured_dim:
            raise ValueError(
                f"{source}: deeponet_sensor_resolution has {len(resolution)} entries, "
                f"expected operator_dim={configured_dim}."
            )
        for name in (
            "deeponet_hidden_channels", "deeponet_branch_depth",
            "deeponet_trunk_depth", "deeponet_basis_dim",
        ):
            _require_int_entries(config, name, 1, source)
        _require_positive_number(config, "deeponet_max_branch_params", source)

    elif model_name == "fno":
        _require_choice(config, "fno_variant", "mesh", {"mesh", "paper_darcy"}, source)
        _require_choice(config, "fno_norm", "none", {"none"}, source)
        resolution = _require_int_entries(config, "fno_grid_resolution", 2, source, lengths={2, 3})
        modes = _require_int_entries(config, "fno_modes", 1, source, lengths={2, 3})
        configured_dim = config.get("operator_dim", "auto")
        if resolution and modes and len(resolution) != len(modes):
            raise ValueError(f"{source}: fno_grid_resolution and fno_modes must have the same length.")
        if configured_dim in (2, 3) and resolution and len(resolution) != configured_dim:
            raise ValueError(
                f"{source}: fno_grid_resolution has {len(resolution)} entries, "
                f"expected operator_dim={configured_dim}."
            )
        if resolution and modes and len(resolution) == len(modes):
            for index, (size, mode) in enumerate(zip(resolution, modes)):
                limit = size // 2 + 1 if index == len(resolution) - 1 else size // 2
                if mode > limit:
                    raise ValueError(
                        f"{source}: fno_modes[{index}]={mode} exceeds limit {limit} "
                        f"for fno_grid_resolution[{index}]={size}."
                    )
        _require_int_entries(config, "fno_hidden_channels", 1, source)
        _require_int_entries(config, "fno_layers", 1, source)

    elif model_name == "gino":
        variant = _require_choice(
            config, "gino_variant", "mesh_state", {"mesh_state", "paper_decoder"}, source
        )
        lengths = {1, 3} if variant == "paper_decoder" else {2, 3}
        resolution = _require_int_entries(config, "gino_grid_resolution", 2, source, lengths=lengths)
        modes = _require_int_entries(config, "gino_fno_modes", 1, source, lengths=lengths)
        if variant == "paper_decoder":
            if len(resolution) == 1:
                resolution *= 3
            if len(modes) == 1:
                modes *= 3
            if resolution and modes and len(resolution) == len(modes):
                for index, (size, mode) in enumerate(zip(resolution, modes)):
                    if mode > size:
                        raise ValueError(
                            f"{source}: gino_fno_modes[{index}]={mode} exceeds "
                            f"gino_grid_resolution[{index}]={size}."
                        )
            _require_choice(
                config, "gino_pos_embedding_type", "nerf", {"nerf", "paper_2023"}, source
            )
            if config.get("gino_include_grid_coordinates", True) is not True:
                raise ValueError(
                    f"{source}: gino_include_grid_coordinates must be True for paper_decoder."
                )
            for name in ("gino_coord_embed_dim", "gino_lifting_hidden"):
                _require_int_entries(config, name, 1, source)
            for name in ("gino_channel_mlp_expansion",):
                _require_positive_number(config, name, source)
            if "gino_tucker_rank" in config:
                rank = float(config["gino_tucker_rank"])
                if not 0 < rank <= 1:
                    raise ValueError(f"{source}: gino_tucker_rank must be in (0, 1].")
            for name in ("gino_kernel_widths", "gino_projection_widths"):
                _require_int_entries(config, name, 1, source)
        else:
            if resolution and modes and len(resolution) != len(modes):
                raise ValueError(
                    f"{source}: gino_grid_resolution and gino_fno_modes must have the same length."
                )
            if resolution and modes and len(resolution) == len(modes):
                for index, (size, mode) in enumerate(zip(resolution, modes)):
                    limit = size // 2 + 1 if index == len(resolution) - 1 else size // 2
                    if mode > limit:
                        raise ValueError(
                            f"{source}: gino_fno_modes[{index}]={mode} exceeds limit {limit} "
                            f"for gino_grid_resolution[{index}]={size}."
                        )
        for name in (
            "gino_fno_hidden_channels", "gino_fno_layers", "gino_kernel_hidden",
        ):
            _require_int_entries(config, name, 1, source)
        _require_int_entries(config, "gino_max_num_neighbors", 0, source)
        for name in ("gino_in_radius", "gino_out_radius"):
            _require_positive_number(config, name, source)


def validate_common_config(config, source="configuration"):
    """Fail fast on removed keys, unrecognized keys, and bad common values.

    Called from load_config() before dataset/model construction, mirroring
    MGN's removed_feature_guard.validate_config fail-fast placement.
    """
    removed = REMOVED_KEYS.intersection(config.keys())
    if removed:
        raise ValueError(
            f"{source} contains removed/unsupported keys: {_format_list(removed)}. "
            "This repository has no message-passing GNN, VAE branch, world "
            "edges, or multiscale coarsening (IMPLEMENTATION_PLAN.md section 5.1)."
        )

    unknown = set(config.keys()) - ALL_KNOWN_KEYS
    if unknown:
        raise ValueError(
            f"{source} contains unknown keys: {_format_list(unknown)}. "
            "See IMPLEMENTATION_PLAN.md section 11.2 for the full key table."
        )

    if "use_parallel_stats" in config and not isinstance(config["use_parallel_stats"], bool):
        raise ValueError(
            f"{source}: use_parallel_stats must be true or false, "
            f"got {config['use_parallel_stats']!r}."
        )
    if "train_eval_subset_size" in config:
        subset_size = config["train_eval_subset_size"]
        if not isinstance(subset_size, int) or isinstance(subset_size, bool) or subset_size <= 0:
            raise ValueError(
                f"{source}: train_eval_subset_size must be a positive integer, "
                f"got {subset_size!r}."
            )

    model_name = str(config.get("model", "")).lower()
    if "model" in config and model_name not in MODEL_NAMES:
        raise ValueError(
            f"{source}: model='{model_name}' is not one of {_format_list(MODEL_NAMES)}."
        )
    if model_name in MODEL_NAMES:
        _validate_static_model_options(config, model_name, source)

    mode = str(config.get("mode", "")).lower()
    if "mode" in config and mode not in ("train", "inference"):
        raise ValueError(f"{source}: mode='{mode}' must be 'train' or 'inference'.")

    parallel_mode = str(config.get("parallel_mode", "ddp")).lower()
    if parallel_mode not in ("ddp", "model_split"):
        raise ValueError(
            f"{source}: parallel_mode must be 'ddp' or 'model_split', got '{parallel_mode}'."
        )
    if parallel_mode == "model_split":
        if "model" in config and model_name not in SPLIT_CAPABLE_MODELS:
            raise ValueError(
                f"{source}: parallel_mode=model_split supports only "
                f"{_format_list(SPLIT_CAPABLE_MODELS)} (models with a sequential "
                f"latent stack to cut into pipeline stages); model='{model_name}' "
                "has a parallel branch/trunk structure. Use parallel_mode ddp."
            )
        if config.get("augment_geometry", False):
            raise ValueError(
                f"{source}: augment_geometry must be False with parallel_mode="
                "model_split: the first and last pipeline stage each load their "
                "own copy of every sample, and the unseeded per-item rotation "
                "would rotate the input and the target differently."
            )
        if (model_name == "gino" and
                str(config.get("gino_variant", "mesh_state")).lower() == "paper_decoder"):
            raise ValueError(
                f"{source}: gino_variant=paper_decoder does not implement the "
                "pipeline model-split protocol. Use parallel_mode ddp."
            )

    if config.get("use_world_edges", False):
        raise ValueError(f"{source}: use_world_edges must be False (section 3).")
    if config.get("use_multiscale", False):
        raise ValueError(f"{source}: use_multiscale must be False (section 3).")

    coordinate_normalization = str(
        config.get("coordinate_normalization", "centered_isotropic")
    ).lower()
    if coordinate_normalization != "centered_isotropic":
        raise ValueError(
            f"{source}: coordinate_normalization must be centered_isotropic, "
            f"got '{coordinate_normalization}'."
        )
    _require_positive_number(config, "dimension_tolerance", source)
    _require_positive_number(config, "grid_padding", source, allow_zero=True)

    if config.get("write_preprocessing", False):
        raise ValueError(
            f"{source}: write_preprocessing must be False. This repository never "
            "writes normalization statistics back into a source HDF5 (deliberate "
            "divergence from MeshGraphNets, IMPLEMENTATION_PLAN.md section 4.1); "
            "the key is accepted only as False for config compatibility."
        )

    gc_features = config.get("global_condition_features", "none")
    if not (isinstance(gc_features, str) and gc_features == "none"):
        raise ValueError(
            f"{source}: global_condition_features={gc_features!r} is declared, but "
            "the dataset loader does not yet attach global_conditions to graphs "
            "(no shipped dataset provides them, IMPLEMENTATION_PLAN.md section 4.1). "
            "Set global_condition_features none."
        )

    input_var = config.get("input_var")
    output_var = config.get("output_var")
    if input_var is not None and (not isinstance(input_var, int) or input_var <= 0):
        raise ValueError(f"{source}: input_var must be a positive int, got {input_var!r}.")
    if output_var is not None and (not isinstance(output_var, int) or output_var <= 0):
        raise ValueError(f"{source}: output_var must be a positive int, got {output_var!r}.")

    cond_var = config.get("cond_var")
    if cond_var is not None and (not isinstance(cond_var, int) or cond_var < 0):
        raise ValueError(f"{source}: cond_var must be a non-negative int, got {cond_var!r}.")

    loss_weights = config.get("feature_loss_weights")
    if loss_weights is not None and output_var is not None:
        if not isinstance(loss_weights, list):
            loss_weights = [loss_weights]
        if len(loss_weights) != output_var:
            raise ValueError(
                f"{source}: feature_loss_weights has {len(loss_weights)} entries, "
                f"expected output_var={output_var}."
            )

    op_dim = config.get("operator_dim", "auto")
    if isinstance(op_dim, str) and op_dim not in ("auto",):
        raise ValueError(f"{source}: operator_dim must be 'auto', 2, or 3, got {op_dim!r}.")
    if isinstance(op_dim, int) and op_dim not in (2, 3):
        raise ValueError(f"{source}: operator_dim must be 2 or 3, got {op_dim!r}.")

    oob = str(config.get("out_of_bounds_policy", "error")).lower()
    if oob not in ("error", "clamp"):
        raise ValueError(f"{source}: out_of_bounds_policy must be 'error' or 'clamp', got '{oob}'.")

    sdf_source = str(config.get("sdf_source", "none")).lower()
    if sdf_source not in ("none", "dataset", "sidecar"):
        raise ValueError(f"{source}: sdf_source must be none/dataset/sidecar, got '{sdf_source}'.")

    integration_weights = str(config.get("integration_weight_source", "none")).lower()
    if integration_weights != "none":
        raise ValueError(
            f"{source}: integration_weight_source must be none; the current "
            "dataset adapter never supplies integration weights."
        )


def validate_temporal_contract(config):
    """input_var must equal output_var whenever the dataset has T > 1 (rollout needs it).

    `cond_var` is deliberately not part of this constraint: conditioning rows
    are input-only and static across the unroll, so they never have to be
    predicted for the next step to be constructible.
    """
    num_timesteps = config.get("num_timesteps")
    if num_timesteps is None or num_timesteps <= 1:
        return
    input_var = config.get("input_var")
    output_var = config.get("output_var")
    if input_var != output_var:
        raise ValueError(
            f"Temporal dataset (num_timesteps={num_timesteps}) requires "
            f"input_var == output_var for autoregressive rollout, got "
            f"input_var={input_var}, output_var={output_var}."
        )


def validate_model_config(config, data_spec):
    """Dispatch to the active architecture's validator. Only called by the factory."""
    model_name = str(config.get("model", "")).lower()
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown model '{model_name}'; expected one of {_format_list(MODEL_NAMES)}.")

    active_keys = ALL_MODEL_KEYS[model_name]
    inactive = set()
    for other_name, other_keys in ALL_MODEL_KEYS.items():
        if other_name == model_name:
            continue
        present = other_keys.intersection(config.keys())
        inactive |= present
    if inactive:
        print(f"[config] Recognized-but-inactive keys for model='{model_name}': "
              f"{_format_list(inactive)}")

    if data_spec.operator_dim not in (2, 3):
        raise ValueError(
            f"operator_dim resolved to {data_spec.operator_dim}; expected 2 or 3. "
            "Check dimension_tolerance and training geometry."
        )

    from model import factory as _factory  # local import: avoids a cycle at module load
    _factory.VALIDATORS[model_name](config, data_spec)
