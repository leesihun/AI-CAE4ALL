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
        "split_seed", "input_var", "output_var", "cond_var", "feature_loss_weights",
        "positional_features", "use_node_types", "coordinate_normalization", "operator_dim",
        "dimension_tolerance", "grid_padding", "out_of_bounds_policy", "sdf_source",
        "sdf_sidecar", "global_condition_features", "integration_weight_source",
        "training_epochs", "batch_size", "learningr", "weight_decay", "warmup_epochs",
        "num_workers", "prefetch_factor", "grad_accum_steps", "max_grad_norm",
        "use_parallel_stats", "train_eval_subset_size",
        "std_noise", "noise_gamma", "noise_std_ratio", "augment_geometry", "use_amp",
        # Time integration (CONFIGURATION_REFERENCE.md section 3.5): ar_ot or ar_rt.
        "time_integration",
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
# Exact mirror of Neural_Operator/general_modules/config_validation.py GINO_KEYS.
GINO_KEYS = frozenset(
    {
        "gino_variant", "gino_grid_resolution", "gino_fno_modes", "gino_fno_hidden_channels",
        "gino_fno_layers", "gino_in_radius", "gino_out_radius", "gino_kernel_hidden",
        "gino_max_empty_input_fraction", "gino_query_chunk_size", "gino_use_torch_cluster",
        "gino_group_shared_geometry", "gino_cache_neighbors",
        # Opt-in ShapeNet Car paper decoder. These keys are inert for mesh_state.
        "gino_tucker_rank", "gino_channel_mlp_expansion", "gino_lifting_hidden",
        "gino_kernel_widths", "gino_projection_widths", "gino_max_num_neighbors",
        "gino_pos_embedding_type", "gino_coord_embed_dim",
        "gino_include_grid_coordinates", "gino_domain_padding",
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
    expected_dim = integer(ctx.values.get("operator_dim"))
    if len(resolution) not in {2, 3} or (expected_dim in {2, 3} and len(resolution) != expected_dim):
        expected = str(expected_dim) if expected_dim in {2, 3} else "2 or 3"
        ctx.add(code, Severity.ERROR, f"{resolution_name} and {modes_name} must each contain {expected} entries.", field_name=resolution_name)
        return
    for index, (size_raw, mode_raw) in enumerate(zip(resolution, modes)):
        size = integer(size_raw)
        mode = integer(mode_raw)
        if size is None or size < 2 or mode is None or mode <= 0:
            ctx.add(code, Severity.ERROR, f"Grid sizes must be integers >= 2 and Fourier modes must be positive integers (axis {index}).", field_name=modes_name)
            continue
        # SpectralConvNd stores separate +/- corners on every non-final axis;
        # allowing more than half the grid makes those corners overlap. The
        # final axis is an rFFT and therefore has size//2 + 1 coefficients.
        limit = size // 2 + 1 if index == len(resolution) - 1 else size // 2
        if mode > limit:
            ctx.add(code, Severity.ERROR, f"{modes_name}[{index}]={mode} exceeds the supported grid limit {limit} for size {size}.", field_name=modes_name)


def _validate_minimum_int_entries(
    ctx: SpecValidationContext, name: str, minimum: int, code: str
) -> None:
    if name not in ctx.values:
        return
    for index, raw in enumerate(as_list(ctx.values[name])):
        value = integer(raw)
        if value is None or value < minimum:
            ctx.add(
                code,
                Severity.ERROR,
                f"{name}[{index}] must be an integer >= {minimum}; got {raw!r}.",
                field_name=name,
            )


def _validate_paper_decoder_grid(ctx: SpecValidationContext) -> None:
    """Validate the CarCFD decoder's centered-total-frequency convention.

    Unlike SpectralConvNd, the paper decoder interprets each configured mode as
    the total centered span and converts only the final axis to rFFT storage.
    Scalars are expanded to all three axes by the native implementation.
    """
    for name, minimum in (("gino_grid_resolution", 2), ("gino_fno_modes", 1)):
        if name not in ctx.values:
            continue
        entries = as_list(ctx.values[name])
        if len(entries) not in {1, 3}:
            ctx.add(
                "NOVAR-GINO-PAPER-GRID",
                Severity.ERROR,
                f"{name} must be a scalar or exactly three integers for gino_variant=paper_decoder.",
                field_name=name,
            )
            continue
        _validate_minimum_int_entries(ctx, name, minimum, "NOVAR-GINO-PAPER-GRID")
    if "gino_grid_resolution" in ctx.values and "gino_fno_modes" in ctx.values:
        resolution = list(as_list(ctx.values["gino_grid_resolution"]))
        modes = list(as_list(ctx.values["gino_fno_modes"]))
        if len(resolution) == 1:
            resolution *= 3
        if len(modes) == 1:
            modes *= 3
        if len(resolution) == len(modes) == 3:
            for index, (size_raw, mode_raw) in enumerate(zip(resolution, modes)):
                size, mode = integer(size_raw), integer(mode_raw)
                if size is not None and mode is not None and mode > size:
                    ctx.add(
                        "NOVAR-GINO-PAPER-GRID",
                        Severity.ERROR,
                        f"gino_fno_modes[{index}]={mode} exceeds gino_grid_resolution[{index}]={size}.",
                        field_name="gino_fno_modes",
                    )


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

    if str(values.get("coordinate_normalization", "centered_isotropic")).lower() != "centered_isotropic":
        ctx.add("NOVAR-COORD-001", Severity.ERROR, "coordinate_normalization must be centered_isotropic.", field_name="coordinate_normalization")
    operator_dim = values.get("operator_dim", "auto")
    if not (str(operator_dim).lower() == "auto" or integer(operator_dim) in {2, 3}):
        ctx.add("NOVAR-DIM-001", Severity.ERROR, "operator_dim must be auto, 2, or 3.", field_name="operator_dim")
    if "dimension_tolerance" in values:
        tolerance = numeric(values["dimension_tolerance"])
        if tolerance is None or tolerance <= 0:
            ctx.add("NOVAR-DIM-TOL", Severity.ERROR, "dimension_tolerance must be > 0.", field_name="dimension_tolerance")
    if "grid_padding" in values:
        padding = numeric(values["grid_padding"])
        if padding is None or padding < 0:
            ctx.add("NOVAR-GRID-PADDING", Severity.ERROR, "grid_padding must be >= 0.", field_name="grid_padding")
    if str(values.get("out_of_bounds_policy", "error")).lower() not in {"error", "clamp"}:
        ctx.add("NOVAR-OOB-001", Severity.ERROR, "out_of_bounds_policy must be error or clamp.", field_name="out_of_bounds_policy")
    sdf_source = str(values.get("sdf_source", "none")).lower()
    if sdf_source not in {"none", "dataset", "sidecar"}:
        ctx.add("NOVAR-SDF-SOURCE", Severity.ERROR, "sdf_source must be none, dataset, or sidecar.", field_name="sdf_source")
    if str(values.get("integration_weight_source", "none")).lower() != "none":
        ctx.add(
            "NOVAR-INTEGRATION-WEIGHTS",
            Severity.ERROR,
            "integration_weight_source must be none; the current dataset adapter never supplies integration weights.",
            field_name="integration_weight_source",
        )

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
    validate_positive_fields(ctx, ("train_eval_subset_size",), "NOVAR-EVAL-SUBSET")
    if model == "fno":
        _validate_grid_modes(ctx, "fno_grid_resolution", "fno_modes", "NOVAR-FNO-001")
        validate_positive_fields(ctx, ("fno_hidden_channels", "fno_layers"), "NOVAR-FNO-POSITIVE")
        if str(values.get("fno_variant", "mesh")).lower() not in {"mesh", "paper_darcy"}:
            ctx.add("NOVAR-FNO-VARIANT", Severity.ERROR, "fno_variant must be mesh or paper_darcy.", field_name="fno_variant")
        if str(values.get("fno_norm", "none")).lower() != "none":
            ctx.add("NOVAR-FNO-NORM", Severity.ERROR, "fno_norm must be none.", field_name="fno_norm")
    elif model == "gino":
        variant = str(values.get("gino_variant", "mesh_state")).lower()
        if variant not in {"mesh_state", "paper_decoder"}:
            ctx.add(
                "NOVAR-GINO-VARIANT",
                Severity.ERROR,
                "gino_variant must be 'mesh_state' or the opt-in 'paper_decoder'.",
                field_name="gino_variant",
            )
        if parallel == "model_split" and variant == "paper_decoder":
            ctx.add(
                "NOVAR-GINO-PARALLEL",
                Severity.ERROR,
                "gino_variant=paper_decoder does not implement model-split execution; use parallel_mode=ddp.",
                field_name="parallel_mode",
            )
        if variant == "paper_decoder":
            _validate_paper_decoder_grid(ctx)
        else:
            _validate_grid_modes(ctx, "gino_grid_resolution", "gino_fno_modes", "NOVAR-GINO-001")
        validate_positive_fields(ctx, ("gino_fno_hidden_channels", "gino_fno_layers", "gino_in_radius", "gino_out_radius", "gino_kernel_hidden"), "NOVAR-GINO-POSITIVE")
        validate_nonnegative_int_fields(ctx, ("gino_max_num_neighbors",), "NOVAR-GINO-NEIGHBORS")
        if "gino_max_empty_input_fraction" in values:
            value = numeric(values["gino_max_empty_input_fraction"])
            if value is None or not 0 <= value <= 1:
                ctx.add("NOVAR-GINO-COVERAGE", Severity.ERROR, "gino_max_empty_input_fraction must be in [0, 1].", field_name="gino_max_empty_input_fraction")
        if "gino_domain_padding" in values:
            value = numeric(values["gino_domain_padding"])
            if value is None or not 0 <= value < 1:
                ctx.add("NOVAR-GINO-PADDING", Severity.ERROR, "gino_domain_padding must be in [0, 1).", field_name="gino_domain_padding")
            elif variant != "paper_decoder":
                ctx.add(
                    "NOVAR-GINO-PADDING-INACTIVE",
                    Severity.WARNING,
                    "gino_domain_padding is active only for gino_variant=paper_decoder.",
                    field_name="gino_domain_padding",
                    promote_in_strict=True,
                )
        if variant == "paper_decoder":
            if str(values.get("gino_pos_embedding_type", "nerf")).lower() not in {"nerf", "paper_2023"}:
                ctx.add("NOVAR-GINO-EMBED", Severity.ERROR, "gino_pos_embedding_type must be nerf or paper_2023.", field_name="gino_pos_embedding_type")
            validate_positive_fields(
                ctx,
                ("gino_coord_embed_dim", "gino_lifting_hidden", "gino_channel_mlp_expansion"),
                "NOVAR-GINO-PAPER-POSITIVE",
            )
            for name in ("gino_kernel_widths", "gino_projection_widths"):
                if name in values:
                    for raw in as_list(values[name]):
                        if integer(raw) is None or integer(raw) <= 0:
                            ctx.add("NOVAR-GINO-WIDTHS", Severity.ERROR, f"{name} entries must be positive integers.", field_name=name)
                            break
            if "gino_tucker_rank" in values:
                rank = numeric(values["gino_tucker_rank"])
                if rank is None or not 0 < rank <= 1:
                    ctx.add("NOVAR-GINO-TUCKER", Severity.ERROR, "gino_tucker_rank must be in (0, 1].", field_name="gino_tucker_rank")
            if values.get("gino_include_grid_coordinates", True) is not True:
                ctx.add("NOVAR-GINO-GRID-COORDS", Severity.ERROR, "gino_include_grid_coordinates must be True for paper_decoder.", field_name="gino_include_grid_coordinates")
    elif model == "point_deeponet":
        validate_nonnegative_int_fields(ctx, ("point_sensor_count",), "NOVAR-POINT-SENSORS")
        validate_positive_fields(ctx, ("point_hidden_channels", "point_feature_dim", "pointnet_depth", "point_condition_depth", "point_trunk_depth", "point_refiner_depth", "point_siren_omega0"), "NOVAR-POINT-POSITIVE")
        closed = {
            "point_variant": ({"mesh_state"}, "mesh_state (the paper profile is not executable while global conditions are unavailable)"),
            "point_sampling": ({"random"}, "random"),
            "pointnet_activation": ({"relu"}, "relu"),
            "pointnet_norm": ({"batch"}, "batch"),
            "point_branch_merge": ({"sum"}, "sum"),
            "point_output_activation": ({"identity", "tanh"}, "identity or tanh"),
        }
        for name, (accepted, label) in closed.items():
            if name in values and str(values[name]).lower() not in accepted:
                ctx.add("NOVAR-POINT-CHOICE", Severity.ERROR, f"{name} must be {label}.", field_name=name)
    elif model == "deeponet":
        validate_positive_fields(ctx, ("deeponet_hidden_channels", "deeponet_branch_depth", "deeponet_trunk_depth", "deeponet_basis_dim"), "NOVAR-DEEP-POSITIVE")
        _validate_minimum_int_entries(ctx, "deeponet_sensor_resolution", 2, "NOVAR-DEEP-RESOLUTION")
        if "deeponet_sensor_resolution" in values:
            sensor_resolution = as_list(values["deeponet_sensor_resolution"])
            expected_dim = integer(values.get("operator_dim"))
            if len(sensor_resolution) not in {2, 3} or (expected_dim in {2, 3} and len(sensor_resolution) != expected_dim):
                expected = str(expected_dim) if expected_dim in {2, 3} else "2 or 3"
                ctx.add("NOVAR-DEEP-RESOLUTION", Severity.ERROR, f"deeponet_sensor_resolution must contain {expected} entries.", field_name="deeponet_sensor_resolution")
        if "deeponet_max_branch_params" in values and (numeric(values["deeponet_max_branch_params"]) is None or numeric(values["deeponet_max_branch_params"]) <= 0):
            ctx.add("NOVAR-DEEP-MAX-PARAMS", Severity.ERROR, "deeponet_max_branch_params must be > 0.", field_name="deeponet_max_branch_params")
        if str(values.get("deeponet_branch_source", "fixed_sensors")).lower() != "fixed_sensors":
            ctx.add("NOVAR-DEEP-BRANCH", Severity.ERROR, "deeponet_branch_source must be fixed_sensors; global conditions are not attached by the current dataset loader.", field_name="deeponet_branch_source")
        if str(values.get("deeponet_multi_output", "split_both")).lower() != "split_both":
            ctx.add("NOVAR-DEEP-OUTPUT", Severity.ERROR, "deeponet_multi_output must be split_both.", field_name="deeponet_multi_output")
        if str(values.get("deeponet_activation", "silu")).lower() not in {"relu", "silu", "gelu", "tanh"}:
            ctx.add("NOVAR-DEEP-ACTIVATION", Severity.ERROR, "deeponet_activation must be relu, silu, gelu, or tanh.", field_name="deeponet_activation")

    if sdf_source == "sidecar" and str(values.get("sdf_sidecar", "none")).lower() in {"", "none"}:
        ctx.add("NOVAR-SDF-001", Severity.ERROR, "sdf_source=sidecar requires sdf_sidecar.", field_name="sdf_sidecar")

    if ctx.mode == "inference":
        ctx.add("NOVAR-CKPT-001", Severity.NOTICE, "The checkpoint supplies the selected model architecture and adapter configuration during inference.", field_name="modelpath")


def build_neural_operator_spec() -> MethodSpec:
    all_keys = COMMON_KEYS | POINT_KEYS | DEEPO_KEYS | FNO_KEYS | GINO_KEYS | NO_REMOVED_KEYS
    return MethodSpec(
        spec_id="neural_operator",
        display_name="Neural Operator",
        model_ids=("point_deeponet", "deeponet", "fno", "gino"),
        repository="methods/Neural_Operator",
        entrypoint="main.py",
        valid_modes=("train", "inference"),
        known_keys=all_keys,
        required_by_mode={
            "train": frozenset({"dataset_dir", "modelpath", "input_var", "output_var", "training_epochs", "batch_size", "learningr", "coordinate_normalization"}),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var"}),
        },
        recommended_by_mode={"train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode", "write_preprocessing"})},
        defaults={"parallel_mode": "ddp", "write_preprocessing": False, "use_world_edges": False, "use_multiscale": False, "use_parallel_stats": True, "train_eval_subset_size": 128, "train_query_chunk_size": 0, "infer_query_chunk_size": 0},
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
