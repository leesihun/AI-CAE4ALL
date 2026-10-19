from __future__ import annotations

from ..diagnostics import Severity
from .base import (
    MethodSpec,
    PathKind,
    PathRule,
    RESUME_KEYS,
    SpecValidationContext,
    as_list,
    integer,
    numeric,
    validate_common_values,
    validate_nonnegative_int_fields,
    validate_positive_fields,
    validate_resume,
)
from .meshgraphnets_variational import validate_spread_keys


# Keys read by methods/SAR (SAR_main.py, training_profiles/, inference_profiles/,
# general_modules/mesh_dataset.py). SAR follows SDFFlow/SimulGenVAE's stage
# prefixes: vae_* configures the graph VAE (stage 1), sar_* the
# scale-autoregressive model over it (stage 2); both are read in `train`.
SAR_KEYS = frozenset(
    {
        # common
        "model", "mode", "gpu_ids", "num_workers", "training_seed",
        # data mapping (shared mesh HDF5: rows 3:3+output_var generated,
        # then cond_var input-only rows, node type in the last row)
        "dataset_dir", "infer_dataset", "eval_dataset", "split_seed", "split_group_attr",
        "output_var", "cond_var", "use_node_types", "edge_attr_r",
        "draws_per_epoch", "val_draws_per_geometry",
        # checkpoints / artifacts
        "vae_modelpath", "modelpath", "log_file_dir", "vae_log_file_dir", "sar_log_file_dir",
        "inference_output_dir",
        # periodic validation loss + pictures (beside the stage log)
        "val_interval", "test_interval", "display_testset", "display_trainset", "viz_num_samples",
        # runtime switches of the vendored Transolver (no effect on the maths)
        "dense_attention", "use_checkpointing", "grad_clip",
        # stage 1: graph VAE (upstream sar.nn.VAE)
        "vae_num_layers", "vae_d_model", "vae_latent_features", "vae_dropout",
        "vae_kl_weight", "vae_noise_std",
        "vae_training_epochs", "vae_batch_size", "vae_grad_accum", "vae_learningr",
        "vae_lr_factor", "vae_lr_patience", "vae_lr_threshold",
        # stage 2: SAR (upstream sar.nn.SAR)
        "sar_num_iters", "sar_num_cond_layers", "sar_num_ar_layers", "sar_num_sampler_layers",
        "sar_d_model", "sar_num_slices", "sar_num_heads", "sar_dropout", "sar_noise_std",
        "sar_sampler_batch_size",
        "sar_training_epochs", "sar_batch_size", "sar_grad_accum", "sar_learningr",
        "sar_lr_factor", "sar_lr_patience", "sar_lr_threshold",
        # sampling (inference + the periodic SAR picture)
        "num_samples", "sample_seed", "sample_scope", "sample_steps", "sample_batch_size",
        "save_rollouts", "make_histogram", "show_histogram", "histogram_bins",
        "histogram_clip_quantile", "spread_channel", "spread_label", "spread_stat",
    }
)

# Mirror of methods/SAR/inference_profiles/sampler.py (sample_scope).
SAMPLE_SCOPES = ("geometry", "scene")
_TRAIN_MODES = frozenset({"train", "train_vae", "train_sar"})
_VAE_STAGE = ("vae_training_epochs", "vae_batch_size", "vae_learningr", "vae_lr_patience")
_SAR_STAGE = ("sar_training_epochs", "sar_batch_size", "sar_learningr", "sar_lr_patience")


def validate_sar(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values

    gpu_ids = as_list(values.get("gpu_ids", []))
    if len(gpu_ids) > 1:
        ctx.add("SAR-GPU-001", Severity.WARNING,
                "SAR trains and samples on one device; only the first gpu_ids entry is used.",
                field_name="gpu_ids", promote_in_strict=True)

    # Upstream's multi-scale VAE path expects scalar parent offsets that its own
    # MeshCoarsening does not produce, and every upstream example trains a
    # single-scale VAE; SAR_main raises on a list, so catch it here.
    if "vae_num_layers" in values:
        layers = as_list(values["vae_num_layers"])
        if len(layers) != 1 or integer(layers[0]) is None or integer(layers[0]) < 1:
            ctx.add("SAR-VAE-SCALES", Severity.ERROR,
                    "vae_num_layers must be ONE positive integer (message-passing layers of the "
                    f"single-scale VAE, upstream 2); got {values['vae_num_layers']!r}.",
                    field_name="vae_num_layers")

    num_iters = integer(values.get("sar_num_iters", 3))
    if num_iters is None or num_iters < 1:
        ctx.add("SAR-ITERS-001", Severity.ERROR,
                f"sar_num_iters must be an integer >= 1; got {values.get('sar_num_iters')!r}.",
                field_name="sar_num_iters")
    elif "sample_steps" in values:
        steps = as_list(values["sample_steps"])
        if any(integer(s) is None or integer(s) < 1 for s in steps):
            ctx.add("SAR-STEPS-001", Severity.ERROR,
                    "sample_steps must be positive integers (flow-matching steps per scale).",
                    field_name="sample_steps")
        elif len(steps) not in (1, num_iters):
            ctx.add("SAR-STEPS-002", Severity.ERROR,
                    f"sample_steps lists {len(steps)} values but sar_num_iters is {num_iters}: "
                    "give one value per scale (coarse to fine) or a single value for all.",
                    field_name="sample_steps")

    scope = values.get("sample_scope")
    if scope is not None and str(scope).lower() not in SAMPLE_SCOPES:
        ctx.add("SAR-SCOPE-001", Severity.ERROR,
                f"sample_scope must be one of {', '.join(SAMPLE_SCOPES)}; got {scope!r}.",
                field_name="sample_scope")

    cond_var = integer(values.get("cond_var", 0))
    if cond_var is None or cond_var < 0:
        ctx.add("SAR-COND-001", Severity.ERROR, "cond_var must be a nonnegative integer.",
                field_name="cond_var")

    if values.get("edge_attr_r") is not None and str(values["edge_attr_r"]).lower() != "auto":
        r = numeric(values["edge_attr_r"])
        if r is None or r <= 0:
            ctx.add("SAR-EDGE-001", Severity.ERROR,
                    "edge_attr_r must be 'auto' or a positive length (upstream ScaleEdgeAttr r).",
                    field_name="edge_attr_r")

    for name in ("vae_lr_factor", "sar_lr_factor"):
        if name in values:
            f = numeric(values[name])
            if f is None or not 0 < f < 1:
                ctx.add("SAR-LR-001", Severity.ERROR, f"{name} must be in (0, 1); got {values[name]!r}.",
                        field_name=name)

    if ctx.mode == "train_sar" and values.get("vae_modelpath") == values.get("modelpath"):
        ctx.add("SAR-PATH-001", Severity.ERROR,
                "vae_modelpath and modelpath name the same file; the SAR stage would overwrite its VAE.",
                field_name="modelpath")

    validate_positive_fields(
        ctx,
        ("vae_d_model", "vae_latent_features", "vae_training_epochs", "vae_batch_size",
         "vae_grad_accum", "vae_learningr", "vae_lr_patience",
         "sar_num_cond_layers", "sar_num_ar_layers", "sar_num_sampler_layers", "sar_d_model",
         "sar_num_slices", "sar_num_heads", "sar_sampler_batch_size", "sar_training_epochs",
         "sar_batch_size", "sar_grad_accum", "sar_learningr", "sar_lr_patience",
         "num_samples", "sample_batch_size", "draws_per_epoch", "val_draws_per_geometry",
         "viz_num_samples", "histogram_bins"),
        "SAR-POSITIVE-001",
    )
    validate_nonnegative_int_fields(
        ctx, ("val_interval", "test_interval", "sample_seed", "training_seed", "split_seed"),
        "SAR-INT-001")
    validate_spread_keys(ctx)


_TRAIN_DEFAULTS = {
    "split_seed": 42, "edge_attr_r": "auto", "draws_per_epoch": 1, "val_draws_per_geometry": 1,
    "num_workers": 0, "use_checkpointing": False, "val_interval": 10, "test_interval": 0,
    "display_testset": True,
    "display_trainset": False,
    "vae_num_layers": 2, "vae_d_model": 128, "vae_dropout": 0.1, "vae_kl_weight": "1e-6",
    "vae_noise_std": 0.01, "vae_grad_accum": 1, "vae_lr_factor": 0.1, "vae_lr_threshold": 0.0,
    "sar_num_iters": 3, "sar_num_cond_layers": 8, "sar_num_ar_layers": 8,
    "sar_num_sampler_layers": 8, "sar_d_model": 128, "sar_num_slices": 64, "sar_num_heads": 8,
    "sar_dropout": 0.1, "sar_noise_std": 0.01, "sar_sampler_batch_size": 4, "sar_grad_accum": 1,
    "sar_lr_factor": 0.1, "sar_lr_threshold": 0.0,
}


def build_sar_spec() -> MethodSpec:
    stage_common = ("dataset_dir", "vae_modelpath", "output_var")
    return MethodSpec(
        spec_id="sar",
        display_name="SAR (scale-autoregressive flow matching)",
        model_ids=("sar",),
        repository="methods/SAR",
        entrypoint="SAR_main.py",
        valid_modes=("train", "train_vae", "train_sar", "inference"),
        known_keys=SAR_KEYS | RESUME_KEYS,
        required_by_mode={
            "train": frozenset({*stage_common, "modelpath", *_VAE_STAGE, *_SAR_STAGE}),
            "train_vae": frozenset({*stage_common, *_VAE_STAGE}),
            "train_sar": frozenset({*stage_common, "modelpath", *_SAR_STAGE}),
            "inference": frozenset({"vae_modelpath", "modelpath", "infer_dataset", "output_var",
                                    "inference_output_dir"}),
        },
        recommended_by_mode={
            "train": frozenset({"split_seed", "split_group_attr", "cond_var", "log_file_dir"}),
            "train_vae": frozenset({"split_seed", "split_group_attr", "cond_var", "log_file_dir"}),
            "train_sar": frozenset({"split_seed", "split_group_attr", "cond_var", "log_file_dir"}),
            "inference": frozenset({"num_samples", "eval_dataset", "cond_var"}),
        },
        # Upstream examples/Wing values (methods/SAR/README.md). Architecture and
        # optimiser keys are training-only: inference rebuilds both models from
        # their checkpoints, so they are not listed as inference defaults.
        defaults={
            "split_group_attr": "shape_idx", "cond_var": 0, "use_node_types": True,
            "dense_attention": True, "sample_steps": "20, 11, 2", "sample_batch_size": 8,
        },
        defaults_by_mode={
            **{mode: dict(_TRAIN_DEFAULTS) for mode in ("train", "train_vae", "train_sar")},
            "inference": {"num_samples": 64, "sample_seed": 0, "sample_scope": "geometry",
                          "save_rollouts": False, "make_histogram": True, "spread_stat": "range",
                          "spread_channel": 2},
        },
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, _TRAIN_MODES),
            PathRule("vae_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.INPUT_FILE, frozenset({"train_sar", "inference"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_sar"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            # inference reads only the two checkpoints and the infer/eval files;
            # dataset_dir is not an inference input.
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("eval_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
            PathRule("log_file_dir", PathKind.OUTPUT_FILE, _TRAIN_MODES),
        ),
        validators=(validate_sar, validate_resume),
        # Upstream also lists lightning, wandb, pot, vtk, torch_cluster: the port
        # replaced Lightning with a plain loop and vendors none of the code that
        # imports the others (methods/SAR/README.md).
        import_modules=("torch", "numpy", "h5py", "torch_geometric", "tqdm"),
        dataset_kind="mesh_hdf5",
        native_probe=True,
    )
