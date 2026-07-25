from __future__ import annotations

from ..diagnostics import Severity
from .base import (
    MethodSpec,
    PathKind,
    PathRule,
    SpecValidationContext,
    as_list,
    integer,
    validate_common_values,
    validate_positive_fields,
)


# The launcher validates against these; an unlisted key becomes CFG-UNKNOWN-001.
# SimulGenVAE mirrors SDFFlow's stage-prefixed scheme: per-stage training knobs are
# carried as vae_*/lc_* in the combined `train` mode and stripped to the unprefixed
# names each worker reads (build_stage_config).
SIMULGENVAE_KEYS = frozenset(
    {
        # common / shared
        "model", "mode", "gpu_ids", "parallel_mode", "dataset_dir", "split_seed",
        "output_dir", "num_workers", "log_file_dir",
        "pipeline_log_file", "skip_completed_stages",
        # data mapping (SimulGenVAE-specific)
        "num_var", "field_start_row", "node_start", "node_end", "timesteps_reduced",
        "network_size", "load_all", "plot_mode",
        # checkpoints
        "vae_modelpath", "lc_modelpath", "init_vae_modelpath",
        # VAE arch (unprefixed, shared into the vae stage)
        "latent_dim", "latent_dim_end", "num_filter_enc", "loss_type", "alpha",
        "init_beta_divisor", "beta_target", "kl_warmup_epochs", "kl_warmup_start_frac",
        "recon_iter", "val_interval",
        # LC arch
        "lc_filter", "lc_data_type", "param_dir", "param_data_type", "lc_dropout",
        "use_spatial_attention",
        # per-stage training knobs (vae_*/lc_* prefixes) and their unprefixed forms
        "training_epochs", "batch_size", "learningr", "weight_decay", "warmup_epochs",
        "use_amp", "use_ema", "ema_decay",
        "vae_training_epochs", "vae_batch_size", "vae_learningr", "vae_weight_decay",
        "vae_warmup_epochs", "vae_num_workers", "vae_use_amp", "vae_use_ema",
        "vae_ema_decay", "vae_log_file_dir",
        "lc_training_epochs", "lc_batch_size", "lc_learningr", "lc_weight_decay",
        "lc_warmup_epochs", "lc_num_workers", "lc_use_amp", "lc_use_ema",
        "lc_ema_decay", "lc_log_file_dir",
    }
)

_NETWORK_SIZES = {"small", "large"}
_LC_DATA_TYPES = {"csv", "image"}
_LOSS_TYPES = {1, 2, 3, 4}


def validate_simulgenvae(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values

    par_mode = str(values.get("parallel_mode", "single")).lower()
    if par_mode not in {"single", "ddp", "fsdp"}:
        ctx.add("SGV-PAR-001", Severity.ERROR,
                "parallel_mode must be single, ddp, or fsdp.", field_name="parallel_mode")
    gpu_ids = as_list(values.get("gpu_ids", []))
    if len(gpu_ids) > 1 and par_mode == "single":
        ctx.add("SGV-GPU-001", Severity.WARNING,
                "parallel_mode=single uses only the first GPU ID; set parallel_mode ddp "
                "to use all listed GPUs.", field_name="gpu_ids", promote_in_strict=True)
    if par_mode in {"ddp", "fsdp"} and len(gpu_ids) < 2:
        ctx.add("SGV-PAR-002", Severity.WARNING,
                f"parallel_mode={par_mode} lists <2 GPUs; it will fall back to single-GPU training.",
                field_name="gpu_ids", promote_in_strict=True)

    if "network_size" in values and str(values["network_size"]).lower() not in _NETWORK_SIZES:
        ctx.add("SGV-SIZE-001", Severity.ERROR,
                "network_size must be 'small' or 'large'.", field_name="network_size")

    if "lc_data_type" in values and str(values["lc_data_type"]).lower() not in _LC_DATA_TYPES:
        ctx.add("SGV-LC-001", Severity.ERROR,
                "lc_data_type must be 'csv' or 'image'.", field_name="lc_data_type")

    if "loss_type" in values and integer(values["loss_type"]) not in _LOSS_TYPES:
        ctx.add("SGV-LOSS-001", Severity.ERROR,
                "loss_type must be 1 (MSE), 2 (MAE), 3 (smoothL1), or 4 (Huber).",
                field_name="loss_type")

    num_var = integer(values.get("num_var"))
    if num_var is not None and num_var < 1:
        ctx.add("SGV-VAR-001", Severity.ERROR, "num_var must be >= 1.", field_name="num_var")
    for name in ("field_start_row", "node_start", "node_end", "timesteps_reduced"):
        v = integer(values.get(name))
        if v is not None and v < 0:
            ctx.add("SGV-INT-001", Severity.ERROR,
                    f"{name} must be a nonnegative integer.", field_name=name)

    validate_positive_fields(
        ctx,
        ("latent_dim", "latent_dim_end", "vae_training_epochs", "vae_batch_size",
         "vae_learningr", "lc_training_epochs", "lc_batch_size", "lc_learningr"),
        "SGV-POSITIVE-001",
    )


def build_simulgenvae_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="simulgenvae",
        display_name="SimulGenVAE",
        model_ids=("simulgenvae",),
        repository="SimulGenVAE",
        entrypoint="SimulGenVAE_main.py",
        valid_modes=("train", "train_vae", "train_lc", "reconstruct"),
        known_keys=SIMULGENVAE_KEYS,
        required_by_mode={
            "train": frozenset({
                "dataset_dir", "vae_modelpath", "lc_modelpath", "num_filter_enc",
                "latent_dim", "latent_dim_end", "lc_filter", "lc_data_type", "param_dir",
                "vae_training_epochs", "vae_batch_size", "vae_learningr",
                "lc_training_epochs", "lc_batch_size", "lc_learningr",
            }),
            "train_vae": frozenset({
                "dataset_dir", "vae_modelpath", "num_filter_enc", "latent_dim",
                "latent_dim_end", "training_epochs", "batch_size", "learningr",
            }),
            "train_lc": frozenset({
                "dataset_dir", "vae_modelpath", "lc_modelpath", "num_filter_enc",
                "latent_dim", "latent_dim_end", "lc_filter", "lc_data_type", "param_dir",
                "training_epochs", "batch_size", "learningr",
            }),
            "reconstruct": frozenset({
                "dataset_dir", "vae_modelpath", "lc_modelpath", "num_filter_enc",
                "latent_dim", "latent_dim_end", "lc_filter", "lc_data_type", "param_dir",
                "output_dir",
            }),
        },
        recommended_by_mode={
            "train": frozenset({"split_seed", "network_size", "skip_completed_stages"}),
            "train_vae": frozenset({"split_seed", "network_size", "alpha"}),
            "train_lc": frozenset({"split_seed", "param_data_type"}),
        },
        defaults={
            "parallel_mode": "single", "network_size": "small", "num_var": 1,
            "field_start_row": 3, "loss_type": 1, "split_seed": 42, "plot_mode": 2,
        },
        defaults_by_mode={
            "train": {"skip_completed_stages": True},
            "train_vae": {"alpha": 1.0, "init_beta_divisor": 4},
        },
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE,
                     frozenset({"train", "train_vae", "train_lc", "reconstruct"})),
            # param_dir is a .csv file for lc_data_type=csv but a directory for image,
            # so its existence is validated by the native loader, not a static PathRule.
            PathRule("init_vae_modelpath", PathKind.INPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.INPUT_FILE, frozenset({"train_lc", "reconstruct"})),
            PathRule("lc_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_lc"})),
            PathRule("lc_modelpath", PathKind.INPUT_FILE, frozenset({"reconstruct"})),
            PathRule("output_dir", PathKind.OUTPUT_DIR),
        ),
        validators=(validate_simulgenvae,),
        import_modules=("torch", "numpy", "h5py", "sklearn"),
        dataset_kind="mesh_hdf5",
        native_probe=True,
    )
