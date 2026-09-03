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
        "output_dir", "num_workers", "log_file_dir", "fsdp_min_params",
        "pipeline_log_file", "skip_completed_stages",
        # data mapping (SimulGenVAE-specific)
        "num_var", "cond_var", "field_start_row", "node_start", "node_end", "timesteps_reduced",
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
# 'hdf5' sources the conditioner's inputs from the mesh HDF5's own cond_var
# rows instead of a separate param_dir file (see SimulGenVAE/general_modules/
# fom_dataset.py::read_conditions_from_hdf5).
_LC_DATA_TYPES = {"csv", "image", "hdf5"}
_LC_MODES = ("train", "train_lc", "reconstruct")
_LOSS_TYPES = {1, 2, 3, 4}
_REMOVED_NOOP_KEYS = {"load_all", "plot_mode", "recon_iter"}


def validate_simulgenvae(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values

    for name in sorted(_REMOVED_NOOP_KEYS.intersection(values)):
        ctx.add(
            "SGV-REMOVED-NOOP",
            Severity.ERROR,
            f"{name} belongs to the retired standalone/pickle pipeline and has no HDF5-native runtime effect.",
            field_name=name,
        )

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

    lc_data_type = str(values.get("lc_data_type", "")).lower()
    if "lc_data_type" in values and lc_data_type not in _LC_DATA_TYPES:
        ctx.add("SGV-LC-001", Severity.ERROR,
                "lc_data_type must be 'csv', 'image' or 'hdf5'.", field_name="lc_data_type")

    # param_dir is required for the file-backed conditioner sources only; the
    # 'hdf5' source reads the conditions out of dataset_dir instead. It is not
    # in required_by_mode for that reason, so enforce it here.
    if ctx.mode in _LC_MODES and lc_data_type in ("csv", "image") and "param_dir" not in values:
        ctx.add("SGV-LC-002", Severity.ERROR,
                f"param_dir is required when lc_data_type is '{lc_data_type}'.",
                field_name="lc_data_type")

    cond_var = integer(values.get("cond_var"))
    if cond_var is not None and cond_var < 0:
        ctx.add("SGV-COND-001", Severity.ERROR,
                "cond_var must be a nonnegative integer.", field_name="cond_var")
    if lc_data_type == "hdf5" and not cond_var:
        ctx.add("SGV-COND-002", Severity.ERROR,
                "lc_data_type 'hdf5' requires cond_var >= 1: it names how many "
                "conditioning rows follow the field rows in nodal_data.",
                field_name="cond_var")

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
         "vae_learningr", "lc_training_epochs", "lc_batch_size", "lc_learningr",
         "fsdp_min_params"),
        "SGV-POSITIVE-001",
    )


def build_simulgenvae_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="simulgenvae",
        display_name="SimulGenVAE",
        model_ids=("simulgenvae",),
        repository="methods/SimulGenVAE",
        entrypoint="SimulGenVAE_main.py",
        valid_modes=("train", "train_vae", "train_lc", "reconstruct"),
        known_keys=SIMULGENVAE_KEYS,
        required_by_mode={
            "train": frozenset({
                "dataset_dir", "vae_modelpath", "lc_modelpath", "num_filter_enc",
                "latent_dim", "latent_dim_end", "lc_filter", "lc_data_type",
                "vae_training_epochs", "vae_batch_size", "vae_learningr",
                "lc_training_epochs", "lc_batch_size", "lc_learningr",
            }),
            "train_vae": frozenset({
                "dataset_dir", "vae_modelpath", "num_filter_enc", "latent_dim",
                "latent_dim_end", "training_epochs", "batch_size", "learningr",
            }),
            "train_lc": frozenset({
                "dataset_dir", "vae_modelpath", "lc_modelpath", "num_filter_enc",
                "latent_dim", "latent_dim_end", "lc_filter", "lc_data_type",
                "training_epochs", "batch_size", "learningr",
            }),
            "reconstruct": frozenset({
                "dataset_dir", "vae_modelpath", "lc_modelpath", "num_filter_enc",
                "latent_dim", "latent_dim_end", "lc_filter", "lc_data_type",
                "output_dir",
            }),
        },
        recommended_by_mode={
            "train": frozenset({"split_seed", "network_size", "skip_completed_stages"}),
            "train_vae": frozenset({"split_seed", "network_size", "alpha"}),
            "train_lc": frozenset({"split_seed", "param_data_type"}),
        },
        defaults={
            "parallel_mode": "single", "fsdp_min_params": 1_000_000,
            "network_size": "small", "num_var": 1,
            "field_start_row": 3, "loss_type": 1, "split_seed": 42,
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
