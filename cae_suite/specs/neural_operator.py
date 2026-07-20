from __future__ import annotations

from ..diagnostics import Severity
from .base import (
    MethodSpec,
    PathKind,
    PathRule,
    SpecValidationContext,
    as_list,
    integer,
    numeric,
    validate_common_values,
    validate_nonnegative_int_fields,
    validate_positive_fields,
)


COMMON_KEYS = frozenset(
    {
        "model", "mode", "gpu_ids", "parallel_mode", "log_file_dir", "modelpath",
        "dataset_dir", "infer_dataset", "inference_output_dir", "infer_timesteps",
        "split_seed", "input_var", "output_var", "feature_loss_weights",
        "positional_features", "use_node_types", "coordinate_normalization", "operator_dim",
        "dimension_tolerance", "grid_padding", "out_of_bounds_policy", "sdf_source",
        "sdf_sidecar", "global_condition_features", "integration_weight_source",
        "training_epochs", "batch_size", "learningr", "weight_decay", "warmup_epochs",
        "num_workers", "prefetch_factor", "grad_accum_steps", "max_grad_norm",
        "std_noise", "noise_gamma", "noise_std_ratio", "augment_geometry", "use_amp",
        "use_checkpointing", "use_ema", "ema_decay", "use_compile", "val_interval",
        "test_interval", "test_max_batches", "test_batch_idx", "plot_feature_idx",
        "display_trainset", "display_testset", "checkpoint_interval",
        "train_query_chunk_size", "infer_query_chunk_size", "write_preprocessing",
        "use_world_edges", "use_multiscale", "profile_batches", "pipeline_microbatches",
        # Injected at runtime by setup.py/dataset construction; the native
        # registry tolerates them when a saved config is echoed back.
        "num_timesteps", "num_node_types", "_pin_memory", "_ddp_port", "log_dir",
    }
)

POINT_KEYS = frozenset(
    {
        "point_variant", "point_sensor_count", "point_sampling", "point_resample_each_epoch",
        "point_hidden_channels", "point_feature_dim", "pointnet_depth", "pointnet_activation",
        "pointnet_norm", "point_branch_merge", "point_condition_depth", "point_trunk_depth",
        "point_refiner_depth", "point_siren_omega0", "point_output_activation",
    }
)
DEEPO_KEYS = frozenset(
    {
        "deeponet_branch_source", "deeponet_sensor_resolution", "deeponet_hidden_channels",
        "deeponet_branch_depth", "deeponet_trunk_depth", "deeponet_basis_dim",
        "deeponet_activation", "deeponet_multi_output", "deeponet_max_branch_params",
    }
)
FNO_KEYS = frozenset(
    {"fno_grid_resolution", "fno_modes", "fno_hidden_channels", "fno_layers", "fno_use_channel_mlp", "fno_norm", "fno_variant"}
)
GINO_KEYS = frozenset(
    {
        "gino_variant", "gino_grid_resolution", "gino_fno_modes", "gino_fno_hidden_channels",
        "gino_fno_layers", "gino_in_radius", "gino_out_radius", "gino_kernel_hidden",
        "gino_max_empty_input_fraction", "gino_query_chunk_size", "gino_use_torch_cluster",
        "gino_group_shared_geometry",
    }
)

VARIANT_KEYS = {
    "point_deeponet": POINT_KEYS,
    "deeponet": DEEPO_KEYS,
    "fno": FNO_KEYS,
    "gino": GINO_KEYS,
}

# Exact mirror of Neural_Operator/general_modules/config_validation.py REMOVED_KEYS.
NO_REMOVED_KEYS = frozenset(
    {
        "message_passing_num", "latent_dim", "edge_var", "world_radius_multiplier",
        "world_max_num_neighbors", "world_edge_backend", "coarse_world_edges",
        "multiscale_levels", "mp_per_level", "coarsening_type", "voronoi_clusters",
        "coarse_cache_per_worker", "use_vae", "vae_latent_dim", "vae_mp_layers",
        "vae_graph_aware", "free_bits", "posterior_min_std", "lambda_mmd",
        "lambda_kl", "lambda_det",
    }
)

REQUIRED_VARIANT_TRAIN = {
    "point_deeponet": frozenset({"point_sensor_count", "point_hidden_channels", "point_feature_dim", "pointnet_depth", "point_trunk_depth"}),
    "deeponet": frozenset({"deeponet_sensor_resolution", "deeponet_hidden_channels", "deeponet_branch_depth", "deeponet_trunk_depth", "deeponet_basis_dim"}),
    "fno": frozenset({"fno_grid_resolution", "fno_modes", "fno_hidden_channels", "fno_layers"}),
    "gino": frozenset({"gino_grid_resolution", "gino_fno_modes", "gino_fno_hidden_channels", "gino_fno_layers", "gino_in_radius", "gino_out_radius", "gino_kernel_hidden"}),
}


def _validate_grid_modes(ctx: SpecValidationContext, resolution_name: str, modes_name: str, code: str) -> None:
    if resolution_name not in ctx.values or modes_name not in ctx.values:
        return
    resolution = as_list(ctx.values[resolution_name])
    modes = as_list(ctx.values[modes_name])
    if len(resolution) != len(modes):
        ctx.add(code, Severity.ERROR, f"{modes_name} and {resolution_name} must have the same dimensionality.", field_name=modes_name)
        return
    for index, (size_raw, mode_raw) in enumerate(zip(resolution, modes)):
        size = integer(size_raw)
        mode = integer(mode_raw)
        if size is None or size <= 0 or mode is None or mode <= 0:
            ctx.add(code, Severity.ERROR, f"Grid sizes and Fourier modes must be positive integers (axis {index}).", field_name=modes_name)
            continue
        # rFFT uses size//2 + 1 on the last axis; this conservative bound catches
        # definitely invalid configs while the native validator remains authoritative.
        limit = size // 2 + 1 if index == len(resolution) - 1 else size
        if mode > limit:
            ctx.add(code, Severity.ERROR, f"{modes_name}[{index}]={mode} exceeds the supported grid limit {limit} for size {size}.", field_name=modes_name)


def validate_neural_operator(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values
    model = ctx.model_id

    if ctx.mode == "train":
        for name in REQUIRED_VARIANT_TRAIN[model]:
            if name not in values:
                ctx.add("NOVAR-REQ", Severity.ERROR, f"{name} is required for {model} training.", field_name=name)

    inactive = set()
    for other_model, keys in VARIANT_KEYS.items():
        if other_model != model:
            inactive.update(keys.intersection(values))
    for name in sorted(inactive):
        ctx.add(
            "NOVAR-INACTIVE",
            Severity.WARNING,
            f"{name} configures a different Neural Operator variant and is inactive for model={model}.",
            field_name=name,
            promote_in_strict=True,
        )

    removed = NO_REMOVED_KEYS.intersection(values)
    for name in sorted(removed):
        ctx.add("NOVAR-REMOVED", Severity.ERROR, f"{name} is a MeshGraphNets key and is unsupported by Neural Operator.", field_name=name)

    if values.get("write_preprocessing", False) is not False:
        ctx.add("NOVAR-WRITE-001", Severity.ERROR, "write_preprocessing must be False; Neural Operator keeps source HDF5 files read-only.", field_name="write_preprocessing")
    if values.get("use_world_edges", False) is not False:
        ctx.add("NOVAR-WORLD-001", Severity.ERROR, "use_world_edges must be False for Neural Operator.", field_name="use_world_edges")
    if values.get("use_multiscale", False) is not False:
        ctx.add("NOVAR-MULTI-001", Severity.ERROR, "use_multiscale must be False for Neural Operator.", field_name="use_multiscale")

    parallel = str(values.get("parallel_mode", "ddp")).lower()
    if parallel not in {"ddp", "model_split"}:
        ctx.add("NOVAR-PARALLEL-001", Severity.ERROR, "parallel_mode must be 'ddp' or 'model_split'.", field_name="parallel_mode")
    if parallel == "model_split":
        if model not in {"fno", "gino"}:
            ctx.add("NOVAR-PARALLEL-002", Severity.ERROR, "model_split supports only fno and gino.", field_name="parallel_mode")
        if values.get("augment_geometry", False) is True:
            ctx.add("NOVAR-PARALLEL-003", Severity.ERROR, "augment_geometry must be False with model_split.", field_name="augment_geometry")
        if len(as_list(values.get("gpu_ids", []))) < 2:
            ctx.add("NOVAR-PARALLEL-004", Severity.ERROR, "model_split requires at least two gpu_ids.", field_name="gpu_ids")

    validate_nonnegative_int_fields(ctx, ("train_query_chunk_size", "infer_query_chunk_size", "gino_query_chunk_size"), "NOVAR-CHUNK-001")
    if model == "fno":
        _validate_grid_modes(ctx, "fno_grid_resolution", "fno_modes", "NOVAR-FNO-001")
        validate_positive_fields(ctx, ("fno_hidden_channels", "fno_layers"), "NOVAR-FNO-POSITIVE")
    elif model == "gino":
        _validate_grid_modes(ctx, "gino_grid_resolution", "gino_fno_modes", "NOVAR-GINO-001")
        validate_positive_fields(ctx, ("gino_fno_hidden_channels", "gino_fno_layers", "gino_in_radius", "gino_out_radius", "gino_kernel_hidden"), "NOVAR-GINO-POSITIVE")
        if "gino_max_empty_input_fraction" in values:
            value = numeric(values["gino_max_empty_input_fraction"])
            if value is None or not 0 <= value <= 1:
                ctx.add("NOVAR-GINO-COVERAGE", Severity.ERROR, "gino_max_empty_input_fraction must be in [0, 1].", field_name="gino_max_empty_input_fraction")
    elif model == "point_deeponet":
        validate_positive_fields(ctx, ("point_sensor_count", "point_hidden_channels", "point_feature_dim", "pointnet_depth", "point_trunk_depth"), "NOVAR-POINT-POSITIVE")
    elif model == "deeponet":
        validate_positive_fields(ctx, ("deeponet_hidden_channels", "deeponet_branch_depth", "deeponet_trunk_depth", "deeponet_basis_dim"), "NOVAR-DEEP-POSITIVE")

    if str(values.get("sdf_source", "none")).lower() == "sidecar" and str(values.get("sdf_sidecar", "none")).lower() in {"", "none"}:
        ctx.add("NOVAR-SDF-001", Severity.ERROR, "sdf_source=sidecar requires sdf_sidecar.", field_name="sdf_sidecar")

    if ctx.mode == "inference":
        ctx.add("NOVAR-CKPT-001", Severity.NOTICE, "The checkpoint supplies the selected model architecture and adapter configuration during inference.", field_name="modelpath")


def build_neural_operator_spec() -> MethodSpec:
    all_keys = COMMON_KEYS | POINT_KEYS | DEEPO_KEYS | FNO_KEYS | GINO_KEYS | NO_REMOVED_KEYS
    return MethodSpec(
        spec_id="neural_operator",
        display_name="Neural Operator",
        model_ids=("point_deeponet", "deeponet", "fno", "gino"),
        repository="Neural_Operator",
        entrypoint="main.py",
        valid_modes=("train", "inference"),
        known_keys=all_keys,
        required_by_mode={
            "train": frozenset({"dataset_dir", "modelpath", "input_var", "output_var", "training_epochs", "batch_size", "learningr", "coordinate_normalization"}),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var"}),
        },
        recommended_by_mode={"train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode", "write_preprocessing"})},
        defaults={"parallel_mode": "ddp", "write_preprocessing": False, "use_world_edges": False, "use_multiscale": False, "train_query_chunk_size": 0, "infer_query_chunk_size": 0},
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/rollout"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
            PathRule("sdf_sidecar", PathKind.INPUT_FILE),
        ),
        validators=(validate_neural_operator,),
        import_modules=("torch", "h5py", "torch_geometric", "scipy"),
        dataset_kind="mesh_hdf5",
    )
