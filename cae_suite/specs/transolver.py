from __future__ import annotations

from pathlib import Path

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


TRANSOLVER_KEYS = frozenset(
    {
        "model", "mode", "gpu_ids", "parallel_mode", "log_file_dir", "modelpath",
        "dataset_dir", "infer_dataset", "inference_output_dir", "infer_timesteps",
        "split_seed", "input_var", "output_var", "feature_loss_weights",
        "positional_features", "use_node_types", "coordinate_normalization", "latent_dim",
        "num_layers", "num_heads", "slice_num", "attention_kernel", "chunk_size",
        "infer_mode", "infer_chunk_size", "mlp_ratio", "dropout", "temperature_init",
        "temperature_min", "temperature_max", "small_output_init", "training_epochs",
        "batch_size", "learningr", "weight_decay", "warmup_epochs", "num_workers",
        "prefetch_factor", "grad_accum_steps", "max_grad_norm", "std_noise", "noise_gamma",
        "augment_geometry", "use_amp", "use_checkpointing", "use_ema", "ema_decay",
        "use_compile", "val_interval", "test_interval", "test_max_batches",
        "use_world_edges", "use_multiscale", "write_preprocessing",
        "max_train_batches", "max_val_batches", "display_trainset",
        "write_test_predictions", "use_parallel_stats",
    }
)

TRANSOLVER_MGN_KEYS = frozenset(
    {
        "edge_var", "message_passing_num", "mp_per_level", "coarsening_type",
        "voronoi_clusters", "multiscale_levels", "world_radius_multiplier",
        "world_max_num_neighbors", "world_edge_backend", "coarse_world_edges",
    }
)


def validate_transolver(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values
    if ctx.mode == "train":
        for name in ("latent_dim", "num_layers", "num_heads", "slice_num", "attention_kernel"):
            if name not in values:
                ctx.add("TRANS-REQ", Severity.ERROR, f"{name} is required for Transolver training.", field_name=name)

    validate_positive_fields(ctx, ("latent_dim", "num_layers", "num_heads", "slice_num", "mlp_ratio"), "TRANS-POSITIVE")
    latent = integer(values.get("latent_dim"))
    heads = integer(values.get("num_heads"))
    if latent is not None and heads is not None and heads > 0 and latent % heads != 0:
        ctx.add("TRANS-HEADS-001", Severity.ERROR, f"latent_dim ({latent}) must be divisible by num_heads ({heads}).", field_name="num_heads")

    kernel = str(values.get("attention_kernel", "naive")).lower()
    if kernel not in {"naive", "slice_space"}:
        ctx.add("TRANS-KERNEL-001", Severity.ERROR, "attention_kernel must be 'naive' or 'slice_space'.", field_name="attention_kernel")
    validate_nonnegative_int_fields(ctx, ("chunk_size", "infer_chunk_size", "max_train_batches", "max_val_batches"), "TRANS-CHUNK-VALUE")
    chunk = integer(values.get("chunk_size", 0))
    if chunk is not None and chunk > 0 and kernel != "slice_space":
        ctx.add("TRANS-CHUNK-001", Severity.ERROR, "chunk_size > 0 requires attention_kernel=slice_space.", field_name="chunk_size")

    infer_mode = str(values.get("infer_mode", "direct")).lower()
    if infer_mode not in {"direct", "decoupled"}:
        ctx.add("TRANS-INFER-001", Severity.ERROR, "infer_mode must be 'direct' or 'decoupled'.", field_name="infer_mode")

    parallel = str(values.get("parallel_mode", "ddp")).lower()
    if parallel == "model_split":
        ctx.add("TRANS-PARALLEL-ALIAS", Severity.NOTICE, "parallel_mode=model_split is a native alias for node_shard in Transolver.", field_name="parallel_mode")
        parallel = "node_shard"
    if parallel not in {"ddp", "node_shard"}:
        ctx.add("TRANS-PARALLEL-001", Severity.ERROR, "parallel_mode must be 'ddp' or 'node_shard'.", field_name="parallel_mode")
    if parallel == "node_shard":
        if kernel != "slice_space":
            ctx.add("TRANS-PARALLEL-002", Severity.ERROR, "node_shard requires attention_kernel=slice_space.", field_name="attention_kernel")
        if len(as_list(values.get("gpu_ids", []))) < 2:
            ctx.add("TRANS-PARALLEL-003", Severity.ERROR, "node_shard requires at least two gpu_ids.", field_name="gpu_ids")

    if values.get("use_world_edges", False) is not False:
        ctx.add("TRANS-WORLD-001", Severity.ERROR, "use_world_edges must be False; Transolver does not consume world edges.", field_name="use_world_edges")
    if values.get("use_multiscale", False) is not False:
        ctx.add("TRANS-MULTI-001", Severity.ERROR, "use_multiscale must be False for Transolver.", field_name="use_multiscale")
    if values.get("write_preprocessing", False) is not False:
        dataset_name = Path(str(values.get("dataset_dir", ""))).name
        if ctx.mode != "train" or not dataset_name.endswith("_transolver_runtime.h5"):
            ctx.add(
                "TRANS-WRITE-001",
                Severity.ERROR,
                "write_preprocessing=True is allowed only for a dedicated "
                "*_transolver_runtime.h5 training copy.",
                field_name="write_preprocessing",
            )
    if str(values.get("coordinate_normalization", "centered_isotropic")).lower() != "centered_isotropic":
        ctx.add("TRANS-COORD-001", Severity.ERROR, "coordinate_normalization must be centered_isotropic.", field_name="coordinate_normalization")

    t_min = numeric(values.get("temperature_min", 0.1))
    t_init = numeric(values.get("temperature_init", 0.5))
    t_max = numeric(values.get("temperature_max", 5.0))
    if None in {t_min, t_init, t_max} or not (0 < float(t_min) <= float(t_init) <= float(t_max)):
        ctx.add("TRANS-TEMP-001", Severity.ERROR, "Temperature bounds must satisfy 0 < temperature_min <= temperature_init <= temperature_max.", field_name="temperature_init")

    forbidden = TRANSOLVER_MGN_KEYS.intersection(values)
    for name in sorted(forbidden):
        ctx.add("TRANS-MGN-KEY", Severity.ERROR, f"{name} is a MeshGraphNets-only key and is unsupported by Transolver.", field_name=name)

    if ctx.mode == "inference":
        ctx.add("TRANS-CKPT-001", Severity.NOTICE, "Transolver inference restores architecture and normalization from the checkpoint.", field_name="modelpath")


def build_transolver_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="transolver",
        display_name="Transolver",
        model_ids=("transolver",),
        repository="Transolver",
        entrypoint="Transolver_main.py",
        valid_modes=("train", "inference"),
        known_keys=TRANSOLVER_KEYS | TRANSOLVER_MGN_KEYS,
        required_by_mode={
            "train": frozenset({"dataset_dir", "modelpath", "input_var", "output_var", "training_epochs", "batch_size", "learningr", "coordinate_normalization"}),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var"}),
        },
        recommended_by_mode={"train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode", "write_preprocessing"})},
        defaults={"parallel_mode": "ddp", "attention_kernel": "naive", "infer_mode": "direct", "chunk_size": 0, "infer_chunk_size": 0, "coordinate_normalization": "centered_isotropic"},
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/rollout"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_transolver,),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
