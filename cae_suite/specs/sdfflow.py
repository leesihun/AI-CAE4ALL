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
    validate_positive_fields,
)


SDFFLOW_KEYS = frozenset(
    {
        "model", "mode", "gpu_ids", "log_file_dir", "output_dir", "vae_modelpath",
        "fm_modelpath", "dataset_dir", "split_seed", "num_encoder_points",
        "num_query_points", "latent_tokens", "latent_dim", "decoder_type",
        "decoder_hidden", "decoder_layers", "decoder_heads", "encoder_dim", "encoder_heads",
        "encoder_blocks", "fourier_bands", "kl_weight", "clamp_dist", "training_epochs",
        "deterministic_warmup_epochs", "posterior_noise_warmup_epochs",
        "posterior_noise_max_scale", "kl_warmup_epochs", "init_vae_modelpath",
        "overfit_all_shapes", "overfit_num_shapes",
        "batch_size", "learningr", "weight_decay", "warmup_epochs", "num_workers",
        "use_amp", "use_ema", "ema_decay", "val_interval", "test_interval",
        "num_test_shapes", "mc_resolution_test", "encode_batch_size", "use_conditions",
        "condition_names", "condition_clip", "min_condition_std", "cond_dropout",
        "fm_hidden", "fm_blocks", "fm_cond_hidden", "ode_steps",
        "num_samples", "seed", "mc_resolution", "cond_values", "cfg_scale",
        "max_condition_z", "condition_ood_policy", "latent_clip", "candidate_multiplier",
        "source_num_samples", "sample_index_a", "sample_index_b", "alpha",
        "plot_dpi", "plot_max_faces", "input_mesh",
        "pipeline_log_file", "skip_completed_stages",
        "vae_log_file_dir", "vae_training_epochs", "vae_batch_size",
        "vae_learningr", "vae_weight_decay", "vae_warmup_epochs",
        "vae_num_workers", "vae_use_amp", "vae_use_ema", "vae_ema_decay",
        "vae_val_interval", "vae_test_interval", "vae_num_test_shapes",
        "vae_mc_resolution_test",
        "fm_log_file_dir", "fm_training_epochs", "fm_batch_size",
        "fm_learningr", "fm_weight_decay", "fm_warmup_epochs",
        "fm_num_workers", "fm_use_amp", "fm_use_ema", "fm_ema_decay",
        "fm_val_interval", "fm_test_interval", "fm_num_test_shapes",
        "fm_mc_resolution_test",
    }
)


def validate_sdfflow(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values
    gpu_ids = as_list(values.get("gpu_ids", []))
    if len(gpu_ids) > 1:
        ctx.add(
            "SDF-GPU-001",
            Severity.WARNING,
            "SDFFlow is single-process and will use only the first GPU ID.",
            field_name="gpu_ids",
            promote_in_strict=True,
        )

    validate_positive_fields(
        ctx,
        (
            "num_encoder_points", "num_query_points", "latent_tokens", "latent_dim",
            "decoder_hidden", "decoder_layers", "decoder_heads", "encoder_dim", "encoder_heads",
            "encoder_blocks", "fourier_bands", "fm_hidden", "fm_blocks",
            "fm_cond_hidden", "ode_steps", "num_samples", "mc_resolution",
            "mc_resolution_test", "encode_batch_size", "candidate_multiplier",
            "source_num_samples", "plot_dpi",
            "vae_training_epochs", "vae_batch_size", "vae_learningr",
            "vae_warmup_epochs", "vae_val_interval", "vae_test_interval",
            "vae_num_test_shapes", "vae_mc_resolution_test",
            "fm_training_epochs", "fm_batch_size", "fm_learningr",
            "fm_warmup_epochs", "fm_val_interval", "fm_test_interval",
            "fm_num_test_shapes", "fm_mc_resolution_test",
        ),
        "SDF-POSITIVE-001",
    )

    if "decoder_type" in values and str(values["decoder_type"]).lower() not in {"mlp", "attention"}:
        ctx.add("SDF-DECODER-001", Severity.ERROR, "decoder_type must be 'mlp' or 'attention'.", field_name="decoder_type")

    for field_name in (
        "deterministic_warmup_epochs",
        "posterior_noise_warmup_epochs",
        "kl_warmup_epochs",
    ):
        if field_name in values:
            value = integer(values[field_name])
            if value is None or value < 0:
                ctx.add(
                    "SDF-WARMUP-001",
                    Severity.ERROR,
                    f"{field_name} must be a nonnegative integer.",
                    field_name=field_name,
                )

    if "posterior_noise_max_scale" in values:
        noise_scale = numeric(values["posterior_noise_max_scale"])
        if noise_scale is None or noise_scale < 0:
            ctx.add(
                "SDF-NOISE-001",
                Severity.ERROR,
                "posterior_noise_max_scale must be a nonnegative number.",
                field_name="posterior_noise_max_scale",
            )

    if ctx.mode in {"train", "train_vae"}:
        if integer(values.get("latent_tokens", 1)) and integer(values.get("latent_tokens", 1)) > 1 and str(values.get("decoder_type", "mlp")).lower() == "mlp":
            ctx.add(
                "SDF-TOKENS-001",
                Severity.WARNING,
                "latent_tokens > 1 is normally paired with decoder_type=attention; verify the intended Tier-2 architecture.",
                field_name="decoder_type",
                promote_in_strict=True,
            )

    if ctx.mode in {"train", "train_fm"}:
        if values.get("use_conditions", False) is True:
            dropout = numeric(values.get("cond_dropout"))
            if dropout is None:
                ctx.add("SDF-COND-001", Severity.ERROR, "cond_dropout is required when use_conditions=True.", field_name="cond_dropout")
            elif not 0 <= dropout < 1:
                ctx.add("SDF-COND-002", Severity.ERROR, "cond_dropout must be in [0, 1).", field_name="cond_dropout")

    if ctx.mode == "sample":
        if "cfg_scale" in values:
            scale = numeric(values["cfg_scale"])
            if scale is None or scale < 0:
                ctx.add("SDF-CFG-001", Severity.ERROR, "cfg_scale must be a nonnegative number.", field_name="cfg_scale")
        if "cond_values" in values:
            ctx.add("SDF-COND-META", Severity.NOTICE, "cond_values length and order will be checked against FM checkpoint metadata by the native runtime.", field_name="cond_values")
        if "condition_ood_policy" in values and str(values["condition_ood_policy"]).lower() not in {"error", "warn", "clamp"}:
            ctx.add("SDF-COND-OOD-001", Severity.ERROR, "condition_ood_policy must be error, warn, or clamp.", field_name="condition_ood_policy")

    if ctx.mode == "interpolate":
        index_a = integer(values.get("sample_index_a"))
        index_b = integer(values.get("sample_index_b"))
        if index_a is not None and index_a < 0:
            ctx.add("SDF-INTERP-001", Severity.ERROR, "sample_index_a must be nonnegative.", field_name="sample_index_a")
        if index_b is not None and index_b < 0:
            ctx.add("SDF-INTERP-002", Severity.ERROR, "sample_index_b must be nonnegative.", field_name="sample_index_b")
        if index_a is not None and index_b is not None and index_a == index_b:
            ctx.add("SDF-INTERP-003", Severity.ERROR, "Interpolation endpoint indices must be distinct.", field_name="sample_index_b")
        alpha = numeric(values.get("alpha"))
        if alpha is not None and not 0 <= alpha <= 1:
            ctx.add("SDF-INTERP-004", Severity.ERROR, "alpha must be within [0, 1].", field_name="alpha")
        max_faces = integer(values.get("plot_max_faces"))
        if max_faces is not None and max_faces < 0:
            ctx.add("SDF-INTERP-005", Severity.ERROR, "plot_max_faces must be nonnegative.", field_name="plot_max_faces")

    if ctx.mode == "reconstruct" and "input_mesh" in values:
        suffix = str(values["input_mesh"]).lower().rsplit(".", 1)[-1]
        if suffix not in {"stl", "obj", "ply", "off", "glb", "gltf"}:
            ctx.add("SDF-MESH-001", Severity.WARNING, "input_mesh has an uncommon extension for trimesh reconstruction.", field_name="input_mesh", promote_in_strict=True)


def build_sdfflow_spec() -> MethodSpec:
    train_common = frozenset({"dataset_dir", "output_dir", "training_epochs", "batch_size", "learningr"})
    return MethodSpec(
        spec_id="sdfflow",
        display_name="SDFFlow",
        model_ids=("sdfflow",),
        repository="Geometry_generation",
        entrypoint="SDFFlow_main.py",
        valid_modes=("train", "train_vae", "train_fm", "sample", "reconstruct", "interpolate"),
        known_keys=SDFFLOW_KEYS,
        required_by_mode={
            "train": frozenset({
                "dataset_dir", "output_dir", "vae_modelpath", "fm_modelpath",
                "latent_tokens", "latent_dim", "decoder_type", "decoder_hidden",
                "decoder_layers", "encoder_dim", "encoder_heads", "encoder_blocks",
                "num_encoder_points", "num_query_points", "fm_hidden", "fm_blocks",
                "fm_cond_hidden", "vae_training_epochs", "vae_batch_size",
                "vae_learningr", "fm_training_epochs", "fm_batch_size", "fm_learningr",
            }),
            "train_vae": train_common | frozenset({"vae_modelpath", "latent_tokens", "latent_dim", "decoder_type", "decoder_hidden", "decoder_layers", "encoder_dim", "encoder_heads", "encoder_blocks", "num_encoder_points", "num_query_points"}),
            "train_fm": train_common | frozenset({"vae_modelpath", "fm_modelpath", "fm_hidden", "fm_blocks", "fm_cond_hidden"}),
            "sample": frozenset({"vae_modelpath", "fm_modelpath", "output_dir", "num_samples", "seed", "ode_steps", "mc_resolution"}),
            "reconstruct": frozenset({"vae_modelpath", "input_mesh", "output_dir", "mc_resolution"}),
            "interpolate": frozenset({"vae_modelpath", "fm_modelpath", "output_dir", "seed", "source_num_samples", "sample_index_a", "sample_index_b", "alpha", "ode_steps", "mc_resolution"}),
        },
        recommended_by_mode={
            "train": frozenset({"split_seed", "use_conditions", "skip_completed_stages"}),
            "train_vae": frozenset({"split_seed", "use_ema", "use_amp"}),
            "train_fm": frozenset({"split_seed", "use_conditions", "use_ema", "use_amp"}),
            "sample": frozenset({"cfg_scale"}),
        },
        defaults={},
        defaults_by_mode={
            "train": {"skip_completed_stages": True},
            "train_vae": {"use_ema": False, "use_amp": False},
            "train_fm": {"use_conditions": False, "use_ema": False, "use_amp": False},
            "sample": {"cfg_scale": 2.0, "ode_steps": 50, "mc_resolution": 128},
            "reconstruct": {"mc_resolution": 128},
            "interpolate": {"alpha": 0.5, "ode_steps": 50, "mc_resolution": 128, "plot_dpi": 180, "plot_max_faces": 0},
        },
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train", "train_vae", "train_fm"})),
            PathRule("init_vae_modelpath", PathKind.INPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.INPUT_FILE, frozenset({"train_fm", "sample", "reconstruct", "interpolate"})),
            PathRule("fm_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_fm"})),
            PathRule("fm_modelpath", PathKind.INPUT_FILE, frozenset({"sample", "interpolate"})),
            PathRule("input_mesh", PathKind.INPUT_FILE, frozenset({"reconstruct"})),
            PathRule("output_dir", PathKind.OUTPUT_DIR),
        ),
        validators=(validate_sdfflow,),
        import_modules=("torch", "h5py", "numpy", "trimesh", "skimage"),
        dataset_kind="sdf_hdf5",
    )
