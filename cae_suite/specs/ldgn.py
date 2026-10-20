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


# Keys read by methods/LDGN (LDGN_main.py, training_profiles/, inference_profiles/,
# general_modules/mesh_dataset.py). Stage prefixes as SAR: vae_* configures the
# variational graph autoencoder (stage 1), ldgn_* the latent diffusion model over
# it (stage 2); both are read in `train`.
LDGN_KEYS = frozenset(
    {
        # common
        "model", "mode", "gpu_ids", "num_workers", "training_seed", "grad_clip",
        # data mapping (shared mesh HDF5: rows 3:3+output_var generated,
        # then cond_var input-only rows, node type in the last row)
        "dataset_dir", "infer_dataset", "eval_dataset", "split_seed", "split_group_attr",
        "output_var", "cond_var", "use_node_types", "edge_attr_r", "rel_pos_ratios",
        "draws_per_epoch", "val_draws_per_geometry",
        # checkpoints / artifacts
        "vae_modelpath", "modelpath", "log_file_dir", "vae_log_file_dir", "ldgn_log_file_dir",
        "inference_output_dir",
        # periodic validation loss + pictures (beside the stage log)
        "val_interval", "test_interval", "display_testset", "display_trainset", "viz_num_samples",
        # stage 1: VGAE (upstream dgn4cfd.nn.VGAE, examples/Wing/train_ae.py)
        "vae_depths", "vae_fnns_depth", "vae_fnns_width", "vae_latent_features", "vae_aggr",
        "vae_dropout", "vae_norm_latents", "vae_kl_reg",
        "vae_training_epochs", "vae_batch_size", "vae_grad_accum", "vae_learningr",
        "vae_lr_factor", "vae_lr_patience", "vae_lr_threshold",
        # stage 2: LDGN (upstream dgn4cfd.nn.LatentDiffusionGraphNet, train_ldgn.py)
        "ldgn_depths", "ldgn_fnns_depth", "ldgn_fnns_width", "ldgn_aggr", "ldgn_dropout",
        "ldgn_diffusion_steps", "ldgn_schedule", "ldgn_learnable_variance",
        "ldgn_lambda_vlb", "ldgn_step_sampler",
        "ldgn_training_epochs", "ldgn_batch_size", "ldgn_grad_accum", "ldgn_learningr",
        "ldgn_lr_factor", "ldgn_lr_patience", "ldgn_lr_threshold",
        # sampling (inference + the periodic LDGN picture)
        "num_samples", "sample_seed", "sample_scope", "sample_steps", "sample_step_schedule",
        "sample_batch_size",
        "save_rollouts", "make_histogram", "show_histogram", "histogram_bins",
        "histogram_clip_quantile", "spread_channel", "spread_label", "spread_stat",
    }
)

# Mirrors of the vendored dgn4cfd asserts and methods/LDGN/inference_profiles/sampler.py.
AGGREGATIONS = ("sum", "mean")                  # MultiScaleGnn
DIFFUSION_SCHEDULES = ("linear", "cosine")      # DiffusionProcess
STEP_SAMPLERS = ("importance", "uniform")       # TrainingSettings
SAMPLE_STEP_SCHEDULES = ("linear", "exp")       # DiffusionStepsGenerator
SAMPLE_SCOPES = ("geometry", "scene")
_TRAIN_MODES = frozenset({"train", "train_vae", "train_ldgn"})
_VAE_STAGE = ("vae_training_epochs", "vae_batch_size", "vae_learningr", "vae_lr_patience")
_LDGN_STAGE = ("ldgn_training_epochs", "ldgn_batch_size", "ldgn_learningr", "ldgn_lr_patience")


def _depths(ctx: SpecValidationContext, name: str, minimum: int, code: str, why: str) -> None:
    values = ctx.values
    if name not in values:
        return
    depths = as_list(values[name])
    if len(depths) < minimum or any(integer(d) is None or integer(d) < 1 for d in depths):
        ctx.add(code, Severity.ERROR,
                f"{name} must list at least {minimum} positive integers ({why}); "
                f"got {values[name]!r}.", field_name=name)


def _choice(ctx: SpecValidationContext, name: str, choices: tuple[str, ...], code: str) -> None:
    value = ctx.values.get(name)
    if value is not None and str(value).lower() not in choices:
        ctx.add(code, Severity.ERROR, f"{name} must be one of {', '.join(choices)}; got {value!r}.",
                field_name=name)


def validate_ldgn(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values

    gpu_ids = as_list(values.get("gpu_ids", []))
    if len(gpu_ids) > 1:
        ctx.add("LDGN-GPU-001", Severity.WARNING,
                "LDGN trains and samples on one device; only the first gpu_ids entry is used.",
                field_name="gpu_ids", promote_in_strict=True)

    # The VGAE encodes onto its coarsest level, so it needs two levels at least;
    # the denoiser starts there (upstream Wing: [2,2,1] and [1,2,2,2]).
    _depths(ctx, "vae_depths", 2, "LDGN-DEPTHS-001",
            "message-passing layers per VGAE level, fine to coarse")
    _depths(ctx, "ldgn_depths", 1, "LDGN-DEPTHS-002",
            "message-passing layers per denoiser level, from the VGAE's coarsest")
    for name in ("vae_fnns_depth", "ldgn_fnns_depth"):
        if name in values and (integer(values[name]) is None or integer(values[name]) < 2):
            ctx.add("LDGN-FNN-001", Severity.ERROR, f"{name} must be an integer >= 2; got {values[name]!r}.",
                    field_name=name)
    for name in ("vae_dropout", "ldgn_dropout"):
        if name in values:
            p = numeric(values[name])
            if p is None or not 0 <= p < 1:
                ctx.add("LDGN-DROPOUT-001", Severity.ERROR, f"{name} must be in [0, 1); got {values[name]!r}.",
                        field_name=name)

    _choice(ctx, "vae_aggr", AGGREGATIONS, "LDGN-AGGR-001")
    _choice(ctx, "ldgn_aggr", AGGREGATIONS, "LDGN-AGGR-001")
    _choice(ctx, "ldgn_schedule", DIFFUSION_SCHEDULES, "LDGN-SCHEDULE-001")
    _choice(ctx, "ldgn_step_sampler", STEP_SAMPLERS, "LDGN-SAMPLER-001")
    _choice(ctx, "sample_step_schedule", SAMPLE_STEP_SCHEDULES, "LDGN-SAMPLER-002")
    _choice(ctx, "sample_scope", SAMPLE_SCOPES, "LDGN-SCOPE-001")

    # The linear schedule scales its betas by 1000/N and clips them at 0.999;
    # short schedules reach alpha_bar = 0 and sample NaN (methods/LDGN/README.md).
    steps = integer(values.get("ldgn_diffusion_steps", 1000))
    if steps is None or steps < 2:
        ctx.add("LDGN-STEPS-001", Severity.ERROR,
                f"ldgn_diffusion_steps must be an integer >= 2; got {values.get('ldgn_diffusion_steps')!r}.",
                field_name="ldgn_diffusion_steps")
    else:
        if ctx.mode in _TRAIN_MODES and steps < 100:
            ctx.add("LDGN-STEPS-002", Severity.WARNING,
                    f"ldgn_diffusion_steps {steps} is far below upstream's 1000; short linear "
                    "schedules clip their betas and can sample NaN.",
                    field_name="ldgn_diffusion_steps", promote_in_strict=True)
        # Inference reads the trained step count from the checkpoint, so the
        # upper bound is only checked against the config's own value in training.
        if "sample_steps" in values:
            n = integer(values["sample_steps"])
            upper = steps if ctx.mode in _TRAIN_MODES else None
            if n is None or n < 2 or (upper is not None and n > upper):
                ctx.add("LDGN-STEPS-003", Severity.ERROR,
                        "sample_steps must be ONE integer >= 2 (denoising steps, a subset of the "
                        f"{'%d ' % upper if upper else ''}trained diffusion steps); got {values['sample_steps']!r}.",
                        field_name="sample_steps")

    cond_var = integer(values.get("cond_var", 0))
    if cond_var is None or cond_var < 0:
        ctx.add("LDGN-COND-001", Severity.ERROR, "cond_var must be a nonnegative integer.",
                field_name="cond_var")

    if values.get("edge_attr_r") is not None and str(values["edge_attr_r"]).lower() != "auto":
        r = numeric(values["edge_attr_r"])
        if r is None or r <= 0:
            ctx.add("LDGN-EDGE-001", Severity.ERROR,
                    "edge_attr_r must be 'auto' or a positive length (upstream ScaleEdgeAttr r).",
                    field_name="edge_attr_r")

    if values.get("rel_pos_ratios") is not None:
        raw = values["rel_pos_ratios"]
        ratios = [numeric(v) for v in (raw if isinstance(raw, list) else [raw])]
        if any(r is None or r <= 0 for r in ratios):
            ctx.add("LDGN-EDGE-002", Severity.ERROR,
                    "rel_pos_ratios must be positive numbers: the coarsening radius of each level "
                    "as a multiple of edge_attr_r (upstream Wing 1, 2, 4, 8, 13.33, 26.67; "
                    f"Ellipse 1, 3, 7.5, 15); got {raw!r}.",
                    field_name="rel_pos_ratios")

    for name in ("vae_lr_factor", "ldgn_lr_factor"):
        if name in values:
            f = numeric(values[name])
            if f is None or not 0 < f < 1:
                ctx.add("LDGN-LR-001", Severity.ERROR, f"{name} must be in (0, 1); got {values[name]!r}.",
                        field_name=name)

    if ctx.mode in ("train", "train_ldgn") and values.get("vae_modelpath") == values.get("modelpath"):
        ctx.add("LDGN-PATH-001", Severity.ERROR,
                "vae_modelpath and modelpath name the same file; the LDGN stage would overwrite its VGAE.",
                field_name="modelpath")

    validate_positive_fields(
        ctx,
        ("vae_fnns_width", "vae_latent_features", "vae_training_epochs", "vae_batch_size",
         "vae_grad_accum", "vae_learningr", "vae_lr_patience",
         "ldgn_fnns_width", "ldgn_training_epochs", "ldgn_batch_size", "ldgn_grad_accum",
         "ldgn_learningr", "ldgn_lr_patience",
         "num_samples", "sample_batch_size", "draws_per_epoch", "val_draws_per_geometry",
         "viz_num_samples", "histogram_bins"),
        "LDGN-POSITIVE-001",
    )
    validate_nonnegative_int_fields(
        ctx, ("val_interval", "test_interval", "sample_seed", "training_seed", "split_seed"),
        "LDGN-INT-001")
    validate_spread_keys(ctx)


_TRAIN_DEFAULTS = {
    "split_seed": 42, "edge_attr_r": "auto", "draws_per_epoch": 1, "val_draws_per_geometry": 1,
    "num_workers": 0, "val_interval": 10, "test_interval": 0, "grad_clip": 1.0,
    "display_testset": True,
    "display_trainset": False,
    "vae_depths": "2, 2, 1", "vae_fnns_depth": 2, "vae_fnns_width": 126, "vae_latent_features": 1,
    "vae_aggr": "sum", "vae_dropout": 0.1, "vae_norm_latents": True, "vae_kl_reg": "1e-6",
    "vae_grad_accum": 1, "vae_lr_factor": 0.1, "vae_lr_threshold": 0.0,
    "ldgn_depths": "1, 2, 2, 2", "ldgn_fnns_depth": 2, "ldgn_fnns_width": 128, "ldgn_aggr": "sum",
    "ldgn_dropout": 0.1, "ldgn_diffusion_steps": 1000, "ldgn_schedule": "linear",
    "ldgn_learnable_variance": True, "ldgn_lambda_vlb": "1e-3", "ldgn_step_sampler": "importance",
    "ldgn_grad_accum": 1, "ldgn_lr_factor": 0.1, "ldgn_lr_threshold": 0.0,
}


def build_ldgn_spec() -> MethodSpec:
    stage_common = ("dataset_dir", "vae_modelpath", "output_var")
    return MethodSpec(
        spec_id="ldgn",
        display_name="LDGN (latent diffusion graph net)",
        model_ids=("ldgn",),
        repository="methods/LDGN",
        entrypoint="LDGN_main.py",
        valid_modes=("train", "train_vae", "train_ldgn", "inference"),
        known_keys=LDGN_KEYS | RESUME_KEYS,
        required_by_mode={
            "train": frozenset({*stage_common, "modelpath", *_VAE_STAGE, *_LDGN_STAGE}),
            "train_vae": frozenset({*stage_common, *_VAE_STAGE}),
            "train_ldgn": frozenset({*stage_common, "modelpath", *_LDGN_STAGE}),
            # The LDGN checkpoint carries the frozen VGAE (arch + weights).
            "inference": frozenset({"modelpath", "infer_dataset", "output_var",
                                    "inference_output_dir"}),
        },
        recommended_by_mode={
            "train": frozenset({"split_seed", "split_group_attr", "cond_var", "log_file_dir"}),
            "train_vae": frozenset({"split_seed", "split_group_attr", "cond_var", "log_file_dir"}),
            "train_ldgn": frozenset({"split_seed", "split_group_attr", "cond_var", "log_file_dir"}),
            "inference": frozenset({"num_samples", "eval_dataset", "cond_var"}),
        },
        # Upstream examples/Wing values (methods/LDGN/README.md). Architecture and
        # optimiser keys are training-only: inference rebuilds the model from its
        # checkpoint, so they are not listed as inference defaults.
        defaults={
            "split_group_attr": "shape_idx", "cond_var": 0, "use_node_types": True,
            "sample_steps": 50, "sample_step_schedule": "linear", "sample_batch_size": 8,
        },
        defaults_by_mode={
            **{mode: dict(_TRAIN_DEFAULTS) for mode in ("train", "train_vae", "train_ldgn")},
            "inference": {"num_samples": 64, "sample_seed": 0, "sample_scope": "geometry",
                          "save_rollouts": False, "make_histogram": True, "spread_stat": "range",
                          "spread_channel": 2},
        },
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, _TRAIN_MODES),
            PathRule("vae_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.INPUT_FILE, frozenset({"train_ldgn"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_ldgn"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("eval_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
            PathRule("log_file_dir", PathKind.OUTPUT_FILE, _TRAIN_MODES),
        ),
        validators=(validate_ldgn, validate_resume),
        # Upstream also lists scipy, tensorboard, vtk, bayesian-torch, pot and
        # torch_cluster; the port vendors none of the code that imports them.
        import_modules=("torch", "numpy", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
        native_probe=True,
    )
