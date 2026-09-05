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
        "fm_arch", "fm_heads", "fm_time_sampling", "fm_time_logit_mean",
        "fm_time_logit_std",
        "surface_weight", "normal_weight", "eikonal_weight", "hybrid_grad_points",
        "encoder_self_attention",
        "parallel_mode", "fsdp_min_params", "num_workers",
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
        # mode `optimize`: closed-loop generate -> mesh -> FEA -> search
        "opt_subspace_dim", "opt_subspace_seed", "opt_condition_dims",
        "opt_latent_range", "opt_shell_scale",
        "opt_budget", "opt_popsize", "opt_sigma0", "opt_seed", "opt_baseline_size",
        "opt_load_cases", "opt_length_scale", "opt_stress_percentile",
        "opt_mesh_size_max", "opt_target_faces",
        "opt_material_e", "opt_material_nu", "opt_material_rho", "opt_yield_stress",
        "opt_stress_margin", "opt_disp_margin", "opt_stress_weight", "opt_disp_weight",
        "opt_verify_resolution", "opt_verify_target_faces", "opt_verify_mesh_size_max",
        # optimize: AI-surrogate analysis backend (HI-MGN in place of gmsh + FEA)
        "opt_analysis", "opt_surrogate_checkpoint", "opt_surrogate_config",
        "opt_surrogate_target_nodes",
        # v3 VAE recipe: geometry-anchored (FPS) encoder queries and a relative
        # posterior-std floor; parent-grouped dataset split; best-val checkpoint
        "encoder_query_type", "posterior_min_std_rel", "split_by_parent",
        "vae_best_modelpath",
        # decoder-frozen latent refinement (reconstruct + evaluate)
        "latent_refine_steps", "latent_refine_lr", "latent_refine_prior_weight",
        # mode `evaluate`: held-out reconstruction metrics
        "eval_split", "eval_num_shapes", "eval_seed",
        # mode `interpolate`: slerp in FM noise space (default), lerp in latent space,
        # or a fixed-noise condition sweep (`cond_sweep`)
        "interpolation_space", "cond_values_a", "cond_values_b", "sweep_steps",
        # Conditional generation (CONDITIONAL_GENERATION_DESIGN_2026-09.md):
        # per-dimension condition dropout so a request may leave entries 'nan',
        # plus the explicit drop-all term that keeps the CFG branch trained
        "cond_dropout_mode", "cond_dropout_all_prob",
        # sample-time descriptor accuracy tools -- C2 calibrated endpoint guidance,
        # E2 proxy-Jacobian Newton correction, and the calibration they share
        "guidance_enabled", "guidance_t_start", "guidance_eta", "guidance_step_mode",
        "guidance_targets", "soft_descriptor_resolution", "soft_descriptor_tau",
        "descriptor_calibration_path",
        "newton_rounds", "newton_step_cap_rms", "newton_line_search_tries",
        "newton_measure_resolution",
        # how decoded meshes are re-measured against FEA-named conditions
        "condition_audit",
        # mode `evaluate`: eval_task descriptor_calibration | conditional
        "eval_task", "eval_methods", "calibration_num_shapes", "calibration_samples_per_shape",
        "calibration_min_r2", "eval_exclude_shapes",
    }
)

GEOMETRIC_CONDITION_NAMES = frozenset({"bbox_x", "bbox_y", "bbox_z", "volume", "area"})
# The only descriptors the soft SDF proxy (general_modules/descriptor_proxy.py)
# can compute; guidance / Newton silently skip every other target name.
PROXY_GUIDABLE_NAMES = frozenset({"volume", "area"})
COND_DROPOUT_MODES = frozenset({"all", "per_dim"})
GUIDANCE_STEP_MODES = frozenset({"velocity_dt", "per_step_jump"})
CONDITION_AUDITS = frozenset({"geometric", "fea", "surrogate"})
EVAL_TASKS = frozenset({"reconstruction", "descriptor_calibration", "conditional"})
EVAL_METHODS = frozenset({"plain", "rejection", "c2", "e2", "c2e2"})
INTERPOLATION_SPACES = frozenset({"slerp_noise", "lerp_latent", "cond_sweep"})
# Methods that read the descriptor calibration artifact.
CALIBRATED_EVAL_METHODS = frozenset({"c2", "e2", "c2e2"})
# evaluate.py::DEFAULT_EVAL_METHODS -- what an evaluate config that omits
# `eval_methods` actually runs. It contains e2, which needs the calibration.
NATIVE_DEFAULT_EVAL_METHODS = ("plain", "rejection", "e2")


def _flag(value) -> bool:
    """Truthiness the way the native `bool(config.get(key, False))` reads it.

    `guidance_enabled 1` parses to the int 1, not the bool True: `bool(1)` turns
    guidance ON natively, while an `is True` test in the spec left it OFF and
    skipped the ERROR that requires `descriptor_calibration_path`. `1` is a
    spelling this repo actually uses for flags.
    """
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "no", "off"}
    return bool(value)


def _names(value) -> list[str]:
    """A comma-list config value as lowercase stripped names ('' entries dropped)."""
    return [str(item).strip().lower() for item in as_list(value) if str(item).strip()]


def _is_condition_entry(value) -> bool:
    """A cond_values entry: a number, or the literal 'nan' (= unspecified)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, str) and (value.strip().lower() == "nan" or numeric(value) is not None)


def _has_nan_entry(value) -> bool:
    return any(isinstance(item, str) and item.strip().lower() == "nan" for item in as_list(value))


def _calibration_file(ctx: SpecValidationContext):
    """(configured, path or None): the descriptor calibration artifact resolved
    against the method repository, or None when the key is absent / not a string."""
    value = ctx.values.get("descriptor_calibration_path")
    if not isinstance(value, str) or not value.strip():
        return False, None
    from ..path_checks import resolve_native_path
    return True, resolve_native_path(value, ctx.repository_root)


def _validate_condition_entries(ctx: SpecValidationContext, field_name: str) -> None:
    values = ctx.values
    if field_name not in values:
        return
    bad = [item for item in as_list(values[field_name]) if not _is_condition_entry(item)]
    if bad:
        ctx.add(
            "SDF-COND-003",
            Severity.ERROR,
            f"{field_name} entries must be numbers (stored space; natural log for log_* names) "
            f"or the literal 'nan' for an unspecified condition; got {bad!r}.",
            field_name=field_name,
        )


def _validate_structural_backend(ctx: SpecValidationContext) -> None:
    """Value checks for the `opt_*` structural settings, wherever they appear.

    `condition_audit fea|surrogate` made these keys second- and third-mode
    consumed: `sample.py` and (through its import) `evaluate.py` read
    `opt_length_scale`, `opt_material_*`, `opt_yield_stress`,
    `opt_stress_percentile`, `opt_mesh_size_max`, `opt_target_faces`,
    `opt_surrogate_*`. They used to be validated only under `mode optimize`, so
    `opt_material_nu 0.9` (a singular isotropic stiffness) passed `--check
    --strict` in `sample` and produced silently garbage audited stresses.
    """
    values = ctx.values
    validate_positive_fields(
        ctx,
        ("opt_target_faces", "opt_mesh_size_max", "opt_length_scale",
         "opt_material_e", "opt_material_rho", "opt_yield_stress",
         "opt_surrogate_target_nodes"),
        "SDF-OPT-POSITIVE-001",
    )
    nu = numeric(values.get("opt_material_nu"))
    if nu is not None and not -1.0 < nu < 0.5:
        ctx.add("SDF-OPT-NU-001", Severity.ERROR,
                "opt_material_nu must lie in (-1, 0.5) for an isotropic solid.",
                field_name="opt_material_nu")
    percentile = numeric(values.get("opt_stress_percentile"))
    if percentile is not None and not 0 < percentile <= 100:
        ctx.add("SDF-OPT-PERCENTILE-001", Severity.ERROR,
                "opt_stress_percentile must lie in (0, 100]: it selects a percentile of the "
                "nodal von Mises field.",
                field_name="opt_stress_percentile")
    audit = str(values.get("condition_audit", "")).strip().lower()
    if audit == "surrogate":
        for field_name, label in (
            ("opt_surrogate_checkpoint", "HI-MGN checkpoint"),
            ("opt_surrogate_config", "HI-MGN inference config"),
        ):
            if not str(values.get(field_name, "")).strip():
                ctx.add(
                    "SDF-OPT-SURROGATE-001",
                    Severity.ERROR,
                    f"{field_name} is required when condition_audit=surrogate ({label}); "
                    "without it the run does the whole ODE and Marching Cubes pass, then "
                    "silently falls back to the geometric audit.",
                    field_name=field_name,
                    hint="Choose an existing file or set condition_audit geometric.",
                )


def _validate_descriptor_tools(ctx: SpecValidationContext) -> None:
    """Value checks for the guidance / Newton / soft-proxy keys, wherever they appear
    (sample, and evaluate with eval_task conditional)."""
    values = ctx.values
    if "guidance_step_mode" in values and str(values["guidance_step_mode"]).strip().lower() not in GUIDANCE_STEP_MODES:
        ctx.add(
            "SDF-GUIDE-001",
            Severity.ERROR,
            "guidance_step_mode must be 'velocity_dt' (correction scaled by dt so the total "
            "strength is NFE-invariant; equals the pilot at 50 steps) or 'per_step_jump' "
            "(the pilot's per-step state jump; total strength grows with ode_steps).",
            field_name="guidance_step_mode",
        )
    if "guidance_eta" in values:
        eta = numeric(values["guidance_eta"])
        if eta is None or eta <= 0:
            ctx.add("SDF-GUIDE-002", Severity.ERROR, "guidance_eta must be > 0.", field_name="guidance_eta")
    if "soft_descriptor_tau" in values:
        tau = numeric(values["soft_descriptor_tau"])
        if tau is None or tau <= 0:
            ctx.add(
                "SDF-GUIDE-003",
                Severity.ERROR,
                "soft_descriptor_tau must be > 0 (the sigmoid temperature of the soft occupancy, "
                "in normalized SDF units; the pilot used 0.032).",
                field_name="soft_descriptor_tau",
            )
    validate_positive_fields(ctx, ("soft_descriptor_resolution",), "SDF-GUIDE-003")
    if "guidance_t_start" in values:
        t_start = numeric(values["guidance_t_start"])
        if t_start is None or not 0 <= t_start < 1:
            ctx.add(
                "SDF-GUIDE-004",
                Severity.ERROR,
                "guidance_t_start must lie in [0, 1): guidance acts on Euler states with "
                "t_start <= t < 1.",
                field_name="guidance_t_start",
            )
    if "guidance_targets" in values:
        unguidable = sorted(set(_names(values["guidance_targets"])) - PROXY_GUIDABLE_NAMES)
        if unguidable:
            ctx.add(
                "SDF-GUIDE-005",
                Severity.WARNING,
                f"guidance_targets {unguidable} have no soft SDF proxy; only "
                f"{sorted(PROXY_GUIDABLE_NAMES)} can be guided or Newton-corrected and the "
                "native run ignores the rest with a note (FEA-named conditions are measured "
                "by condition_audit fea|surrogate instead).",
                field_name="guidance_targets",
                promote_in_strict=True,
            )
    validate_nonnegative_int_fields(ctx, ("newton_rounds",), "SDF-NEWTON-001")
    validate_positive_fields(
        ctx,
        ("newton_step_cap_rms", "newton_line_search_tries", "newton_measure_resolution"),
        "SDF-NEWTON-002",
    )
    if "newton_line_search_tries" in values and integer(values["newton_line_search_tries"]) is None:
        ctx.add("SDF-NEWTON-002", Severity.ERROR, "newton_line_search_tries must be a positive integer.",
                field_name="newton_line_search_tries")
    if "condition_audit" in values and str(values["condition_audit"]).strip().lower() not in CONDITION_AUDITS:
        ctx.add(
            "SDF-AUDIT-001",
            Severity.ERROR,
            "condition_audit must be 'geometric' (measure volume/area/bbox on the decoded mesh), "
            "'fea' (also mesh and solve the GE load cases with design_loop; needs gmsh + pyamg) "
            "or 'surrogate' (design_loop's HI-MGN surrogate; needs opt_surrogate_config / "
            "opt_surrogate_checkpoint).",
            field_name="condition_audit",
        )
    validate_positive_fields(
        ctx, ("calibration_num_shapes", "calibration_samples_per_shape"), "SDF-CALIB-004")
    if "calibration_min_r2" in values:
        min_r2 = numeric(values["calibration_min_r2"])
        if min_r2 is None or not 0 <= min_r2 <= 1:
            ctx.add(
                "SDF-CALIB-005",
                Severity.ERROR,
                "calibration_min_r2 must lie in [0, 1]: it is the per-descriptor fit quality "
                "below which the calibration task refuses to save (0 disables the floor).",
                field_name="calibration_min_r2",
            )
    _validate_structural_backend(ctx)


def validate_sdfflow(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values
    par_mode = str(values.get("parallel_mode", "single")).lower()
    if par_mode not in {"single", "ddp", "fsdp"}:
        ctx.add("SDF-PAR-001", Severity.ERROR,
                "parallel_mode must be single, ddp, or fsdp.", field_name="parallel_mode")
    gpu_ids = as_list(values.get("gpu_ids", []))
    if len(gpu_ids) > 1 and par_mode == "single":
        ctx.add(
            "SDF-GPU-001",
            Severity.WARNING,
            "parallel_mode=single uses only the first GPU ID; set parallel_mode "
            "ddp (or fsdp) to use all listed GPUs.",
            field_name="gpu_ids",
            promote_in_strict=True,
        )
    if par_mode in {"ddp", "fsdp"} and len(gpu_ids) < 2:
        ctx.add(
            "SDF-PAR-002",
            Severity.WARNING,
            f"parallel_mode={par_mode} lists <2 GPUs; it will fall back to "
            "single-GPU training.",
            field_name="gpu_ids",
            promote_in_strict=True,
        )

    validate_positive_fields(
        ctx,
        (
            "num_encoder_points", "num_query_points", "latent_tokens", "latent_dim",
            "decoder_hidden", "decoder_layers", "decoder_heads", "encoder_dim", "encoder_heads",
            "encoder_blocks", "fourier_bands", "fm_hidden", "fm_blocks",
            "fm_cond_hidden", "fm_heads", "ode_steps", "num_samples", "mc_resolution",
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

    if "fm_arch" in values and str(values["fm_arch"]).lower() not in {"mlp", "dit"}:
        ctx.add("SDF-FMARCH-001", Severity.ERROR, "fm_arch must be 'mlp' or 'dit'.", field_name="fm_arch")

    if "fm_time_sampling" in values and str(values["fm_time_sampling"]).lower() not in {"uniform", "logit_normal"}:
        ctx.add("SDF-FMTIME-001", Severity.ERROR, "fm_time_sampling must be 'uniform' or 'logit_normal'.", field_name="fm_time_sampling")

    if ctx.mode in {"train", "train_fm"} and str(values.get("fm_arch", "mlp")).lower() == "dit" \
            and integer(values.get("latent_tokens", 1)) == 1:
        ctx.add(
            "SDF-FMARCH-002",
            Severity.WARNING,
            "fm_arch=dit over a single latent token has no tokens to attend across; "
            "pair it with latent_tokens > 1 (VecSet) for any benefit.",
            field_name="fm_arch",
            promote_in_strict=True,
        )

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

    if "posterior_min_std_rel" in values:
        min_std_rel = numeric(values["posterior_min_std_rel"])
        if min_std_rel is None or min_std_rel < 0:
            ctx.add(
                "SDF-NOISE-002",
                Severity.ERROR,
                "posterior_min_std_rel must be a nonnegative number (0 disables the "
                "posterior std floor).",
                field_name="posterior_min_std_rel",
            )

    if "encoder_query_type" in values:
        query_type = str(values["encoder_query_type"]).strip().lower()
        if query_type not in {"learned", "fps"}:
            ctx.add(
                "SDF-QUERY-001",
                Severity.ERROR,
                "encoder_query_type must be 'learned' (nn.Parameter queries) or 'fps' "
                "(farthest-point-sampled input points as queries).",
                field_name="encoder_query_type",
            )
        if query_type == "fps":
            tokens = integer(values.get("latent_tokens", 1))
            points = integer(values.get("num_encoder_points"))
            if tokens is not None and points is not None and tokens > points:
                ctx.add(
                    "SDF-QUERY-003",
                    Severity.ERROR,
                    f"encoder_query_type=fps needs latent_tokens ({tokens}) <= "
                    f"num_encoder_points ({points}): there are not enough input points to "
                    "anchor one query each, and the fallback repeats tokens.",
                    field_name="latent_tokens",
                )
        if query_type == "fps" and str(values.get("decoder_type", "mlp")).strip().lower() == "mlp":
            ctx.add(
                "SDF-QUERY-002",
                Severity.WARNING,
                "encoder_query_type=fps makes the latent token order depend on the input "
                "point set; the flattened-token mlp decoder then sees an input-dependent "
                "channel layout. Pair fps with decoder_type=attention.",
                field_name="encoder_query_type",
                promote_in_strict=True,
            )

    # Decoder-frozen latent refinement (reconstruct + evaluate). Shared by both
    # modes, so validated wherever the keys appear.
    validate_nonnegative_int_fields(ctx, ("latent_refine_steps",), "SDF-REFINE-001")
    if "latent_refine_lr" in values:
        refine_lr = numeric(values["latent_refine_lr"])
        if refine_lr is None or refine_lr <= 0:
            ctx.add(
                "SDF-REFINE-002",
                Severity.ERROR,
                "latent_refine_lr must be a positive number.",
                field_name="latent_refine_lr",
            )
    if "latent_refine_prior_weight" in values:
        prior_weight = numeric(values["latent_refine_prior_weight"])
        if prior_weight is None or prior_weight < 0:
            ctx.add(
                "SDF-REFINE-003",
                Severity.ERROR,
                "latent_refine_prior_weight must be a nonnegative number.",
                field_name="latent_refine_prior_weight",
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

    # Condition dropout mode (train / train_fm; stored in the FM checkpoint and
    # read back by every inference mode). Validated wherever it appears.
    dropout_mode = str(values.get("cond_dropout_mode", "all")).strip().lower()
    if "cond_dropout_mode" in values and dropout_mode not in COND_DROPOUT_MODES:
        ctx.add(
            "SDF-CDROP-001",
            Severity.ERROR,
            "cond_dropout_mode must be 'all' (legacy: one Bernoulli mask per sample and a "
            "learned null embedding) or 'per_dim' (independent mask per condition entry, "
            "learned null_values + mask input; the mode that allows 'nan' = unspecified "
            "entries in cond_values at inference).",
            field_name="cond_dropout_mode",
        )

    if "cond_dropout_all_prob" in values:
        drop_all = numeric(values["cond_dropout_all_prob"])
        if drop_all is None or not 0 <= drop_all < 1:
            ctx.add(
                "SDF-CDROP-002",
                Severity.ERROR,
                "cond_dropout_all_prob must lie in [0, 1): it is the probability that a per_dim "
                "training row has EVERY condition masked, i.e. how often the CFG unconditional "
                "branch is trained at all.",
                field_name="cond_dropout_all_prob",
            )
        elif drop_all == 0 and dropout_mode == "per_dim":
            ctx.add(
                "SDF-CDROP-003",
                Severity.WARNING,
                "cond_dropout_all_prob 0 with cond_dropout_mode per_dim leaves the "
                "unconditional (all-masked) row to chance at probability cond_dropout ** "
                "cond_dim -- 6e-5 for six conditions at 0.2. The CFG branch is then near "
                "initialization and cfg_scale must stay 1.0.",
                field_name="cond_dropout_all_prob",
                promote_in_strict=True,
            )

    if ctx.mode in {"train", "train_fm"}:
        # `use_conditions 1` is the int 1 and the native reader is bool(); an
        # `is True` test here left a conditional run unvalidated (same class as
        # the guidance_enabled bug _flag documents).
        if _flag(values.get("use_conditions", False)):
            dropout = numeric(values.get("cond_dropout"))
            if dropout is None:
                ctx.add("SDF-COND-001", Severity.ERROR, "cond_dropout is required when use_conditions=True.", field_name="cond_dropout")
            elif not 0 <= dropout < 1:
                ctx.add("SDF-COND-002", Severity.ERROR, "cond_dropout must be in [0, 1).", field_name="cond_dropout")
        fea_names = sorted(set(_names(values.get("condition_names", []))) - GEOMETRIC_CONDITION_NAMES)
        if fea_names:
            ctx.add(
                "SDF-COND-FEA-001",
                Severity.NOTICE,
                f"condition_names {fea_names} are not geometric descriptors: the dataset must "
                "carry the FEA sidecar (root 'cond_extra', written by "
                "methods/SDFFlow/add_fea_conditions.py) and the names must be in "
                "general_modules/condition_names.py; the native run validates both.",
                field_name="condition_names",
            )
            if dropout_mode != "per_dim":
                ctx.add(
                    "SDF-COND-FEA-002",
                    Severity.NOTICE,
                    "FEA-named conditions with cond_dropout_mode all: inference must then "
                    "specify EVERY condition; a partial request ('nan' entries) needs "
                    "cond_dropout_mode per_dim.",
                    field_name="cond_dropout_mode",
                )

    _validate_descriptor_tools(ctx)
    for field_name in ("cond_values", "cond_values_a", "cond_values_b"):
        _validate_condition_entries(ctx, field_name)

    if ctx.mode == "sample":
        if "cfg_scale" in values:
            scale = numeric(values["cfg_scale"])
            if scale is None or scale < 0:
                ctx.add("SDF-CFG-001", Severity.ERROR, "cfg_scale must be a nonnegative number.", field_name="cfg_scale")
        if "cond_values" in values:
            ctx.add("SDF-COND-META", Severity.NOTICE, "cond_values length and order will be checked against FM checkpoint metadata by the native runtime.", field_name="cond_values")
            if _has_nan_entry(values["cond_values"]):
                ctx.add(
                    "SDF-COND-PARTIAL-001",
                    Severity.NOTICE,
                    "cond_values leaves entries unspecified ('nan'): the FM checkpoint must have "
                    "been trained with cond_dropout_mode per_dim, or the native run raises.",
                    field_name="cond_values",
                )
        if "condition_ood_policy" in values and str(values["condition_ood_policy"]).lower() not in {"error", "warn", "clamp"}:
            ctx.add("SDF-COND-OOD-001", Severity.ERROR, "condition_ood_policy must be error, warn, or clamp.", field_name="condition_ood_policy")
        guidance_on = _flag(values.get("guidance_enabled", False))
        newton_on = (integer(values.get("newton_rounds", 0)) or 0) > 0
        if guidance_on or newton_on:
            configured, calibration = _calibration_file(ctx)
            if "cond_values" not in values:
                ctx.add(
                    "SDF-GUIDE-006",
                    Severity.WARNING,
                    "guidance_enabled / newton_rounds act only on a conditional request; without "
                    "cond_values both are inactive for this run.",
                    field_name="guidance_enabled" if guidance_on else "newton_rounds",
                    promote_in_strict=True,
                )
            if not configured:
                ctx.add(
                    "SDF-CALIB-001",
                    Severity.ERROR,
                    "descriptor_calibration_path is required when guidance_enabled is True or "
                    "newton_rounds > 0: C2/E2 work in calibrated proxy units.",
                    field_name="descriptor_calibration_path",
                    hint="Create it with mode evaluate + eval_task descriptor_calibration "
                    "(configs/SDFFlow/config_calibrate_descriptors.txt) on the same VAE/FM pair.",
                )
            elif calibration is not None and not calibration.is_file():
                ctx.add(
                    "SDF-CALIB-002",
                    Severity.NOTICE,
                    f"descriptor_calibration_path does not exist yet: {calibration}. The native run "
                    "will refuse to start; run the descriptor_calibration evaluate task first.",
                    field_name="descriptor_calibration_path",
                )

    if ctx.mode == "interpolate":
        space = str(values.get("interpolation_space", "slerp_noise")).strip().lower()
        if space in {"slerp_noise", "lerp_latent"} and "sample_index_b" not in values:
            ctx.add(
                "SDF-INTERP-007",
                Severity.ERROR,
                f"sample_index_b is required for interpolation_space {space} (the second "
                "endpoint of the interpolation).",
                field_name="sample_index_b",
                hint="Add a 'sample_index_b <index>' line, or switch to interpolation_space "
                "cond_sweep, which sweeps one noise row across conditions.",
            )
        if space == "cond_sweep":
            for field_name in ("cond_values_a", "cond_values_b"):
                if field_name not in values:
                    ctx.add(
                        "SDF-SWEEP-003",
                        Severity.ERROR,
                        f"{field_name} is required for interpolation_space cond_sweep (the "
                        "condition vector at the start / end of the sweep, in FM checkpoint "
                        "cond_names order; 'nan' = unspecified, per_dim checkpoints only).",
                        field_name=field_name,
                    )
            if "cond_values_a" in values and "cond_values_b" in values:
                len_a = len(as_list(values["cond_values_a"]))
                len_b = len(as_list(values["cond_values_b"]))
                if len_a != len_b:
                    ctx.add(
                        "SDF-SWEEP-004",
                        Severity.ERROR,
                        f"cond_values_a has {len_a} entries but cond_values_b has {len_b}; both "
                        "follow the FM checkpoint's cond_names order.",
                        field_name="cond_values_b",
                    )
            if "cond_values" in values:
                ctx.add(
                    "SDF-SWEEP-005",
                    Severity.WARNING,
                    "cond_values is ignored by interpolation_space cond_sweep; the sweep runs "
                    "between cond_values_a and cond_values_b.",
                    field_name="cond_values",
                    promote_in_strict=True,
                )
        validate_nonnegative_int_fields(ctx, ("sweep_steps",), "SDF-SWEEP-001")
        if space == "cond_sweep":
            # Native precondition, not a preference: interpolate.py::sweep_alphas
            # raises ValueError('sweep_steps must be an integer >= 2 ...') after
            # both checkpoints have been loaded.
            steps = integer(values.get("sweep_steps", 5))
            if steps is None or steps < 2:
                ctx.add(
                    "SDF-SWEEP-001",
                    Severity.ERROR,
                    f"sweep_steps must be an integer >= 2 for interpolation_space cond_sweep "
                    f"(got {values.get('sweep_steps')!r}); the sweep interpolates between "
                    "cond_values_a and cond_values_b and needs both endpoints.",
                    field_name="sweep_steps",
                )
        elif "sweep_steps" in values:
            ctx.add(
                "SDF-SWEEP-002",
                Severity.NOTICE,
                f"sweep_steps is read only by interpolation_space cond_sweep; ignored for "
                f"{space}.",
                field_name="sweep_steps",
            )
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
        if "interpolation_space" in values and str(values["interpolation_space"]).strip().lower() not in INTERPOLATION_SPACES:
            ctx.add(
                "SDF-INTERP-006",
                Severity.ERROR,
                "interpolation_space must be 'slerp_noise' (spherical interpolation of the FM "
                "source noise; endpoints reproduce the original samples), 'lerp_latent' "
                "(legacy torch.lerp in normalized latent space), or 'cond_sweep' (one fixed "
                "noise row integrated under sweep_steps conditions lerped from cond_values_a "
                "to cond_values_b).",
                field_name="interpolation_space",
            )

    if ctx.mode == "evaluate":
        if "eval_split" in values and str(values["eval_split"]).strip().lower() not in {"train", "val", "test"}:
            ctx.add("SDF-EVAL-001", Severity.ERROR, "eval_split must be 'train', 'val', or 'test'.", field_name="eval_split")
        validate_nonnegative_int_fields(ctx, ("eval_num_shapes",), "SDF-EVAL-005")
        if "eval_seed" in values and integer(values["eval_seed"]) is None:
            ctx.add("SDF-EVAL-006", Severity.ERROR, "eval_seed must be an integer.", field_name="eval_seed")
        if "eval_exclude_shapes" in values:
            bad = [item for item in as_list(values["eval_exclude_shapes"])
                   if integer(item) is None or integer(item) < 0]
            if bad:
                ctx.add(
                    "SDF-EVAL-009",
                    Severity.ERROR,
                    f"eval_exclude_shapes must be nonnegative HDF5 shape indices; got {bad!r}.",
                    field_name="eval_exclude_shapes",
                )
        task = str(values.get("eval_task", "reconstruction")).strip().lower()
        if "eval_task" in values and task not in EVAL_TASKS:
            ctx.add(
                "SDF-EVAL-002",
                Severity.ERROR,
                "eval_task must be 'reconstruction' (held-out VAE reconstruction metrics), "
                "'descriptor_calibration' (fit the soft-proxy affine calibration and write "
                "descriptor_calibration_path) or 'conditional' (paired-noise condition-accuracy "
                "benchmark of plain / rejection / c2 / e2 / c2e2 sampling).",
                field_name="eval_task",
            )
        # SDF-EVAL-003 checks what was WRITTEN; the calibration-need test below
        # has to use what the native run will EFFECTIVELY use, because
        # evaluate.py's own default (plain, rejection, e2) already contains e2,
        # which reads descriptor_calibration_path.
        methods_written = "eval_methods" in values
        methods = set(_names(values["eval_methods"])) if methods_written else set()
        effective_methods = methods if methods_written else set(NATIVE_DEFAULT_EVAL_METHODS)
        unknown_methods = sorted(methods - EVAL_METHODS)
        if unknown_methods:
            ctx.add(
                "SDF-EVAL-003",
                Severity.ERROR,
                f"eval_methods contains unknown method(s) {unknown_methods}; allowed: "
                f"{sorted(EVAL_METHODS)}.",
                field_name="eval_methods",
            )
        if task in EVAL_TASKS and task != "reconstruction":
            if not str(values.get("fm_modelpath", "")).strip():
                ctx.add(
                    "SDF-EVAL-004",
                    Severity.ERROR,
                    f"fm_modelpath is required for eval_task {task}: the task samples the "
                    "flow-matching model (reconstruction scores the VAE alone).",
                    field_name="fm_modelpath",
                    hint="Point fm_modelpath at the FM checkpoint trained with vae_modelpath.",
                )
            if methods_written and task != "conditional":
                ctx.add(
                    "SDF-EVAL-008",
                    Severity.NOTICE,
                    f"eval_methods is read only by eval_task conditional; ignored for {task}.",
                    field_name="eval_methods",
                )
        elif methods_written:
            ctx.add(
                "SDF-EVAL-008",
                Severity.NOTICE,
                "eval_methods is read only by eval_task conditional; ignored for reconstruction.",
                field_name="eval_methods",
            )
        configured, calibration = _calibration_file(ctx)
        needs_calibration_input = (task == "conditional"
                                   and bool(effective_methods & CALIBRATED_EVAL_METHODS))
        if task == "descriptor_calibration" and not configured:
            ctx.add(
                "SDF-EVAL-007",
                Severity.ERROR,
                "descriptor_calibration_path is required for eval_task descriptor_calibration: "
                "it is where the fitted DescriptorCalibration is written.",
                field_name="descriptor_calibration_path",
            )
        elif task == "descriptor_calibration" and calibration is not None and calibration.is_file():
            ctx.add(
                "SDF-CALIB-003",
                Severity.NOTICE,
                f"descriptor_calibration_path already exists and will be overwritten: {calibration}",
                field_name="descriptor_calibration_path",
            )
        if needs_calibration_input:
            if not configured:
                ctx.add(
                    "SDF-EVAL-007",
                    Severity.ERROR,
                    "descriptor_calibration_path is required when eval_methods include c2, e2 or "
                    f"c2e2: those methods work in calibrated proxy units. Effective eval_methods "
                    f"here: {sorted(effective_methods)}"
                    + ("" if methods_written else " (evaluate.py's own default, since the key is "
                       "absent)."),
                    field_name="descriptor_calibration_path",
                    hint="Create it with eval_task descriptor_calibration "
                    "(configs/SDFFlow/config_calibrate_descriptors.txt) on the same VAE/FM pair.",
                )
            elif calibration is not None and not calibration.is_file():
                ctx.add(
                    "SDF-CALIB-002",
                    Severity.NOTICE,
                    f"descriptor_calibration_path does not exist yet: {calibration}. Run the "
                    "descriptor_calibration evaluate task first.",
                    field_name="descriptor_calibration_path",
                )

    if ctx.mode == "optimize":
        # The value checks shared with the condition audit (opt_length_scale,
        # opt_material_*, opt_yield_stress, opt_stress_percentile,
        # opt_target_faces, opt_mesh_size_max, opt_surrogate_target_nodes,
        # opt_material_nu) live in `_validate_structural_backend`, which runs in
        # every mode. Only the search-specific ones are left here.
        validate_positive_fields(
            ctx,
            ("opt_subspace_dim", "opt_budget", "opt_popsize", "opt_baseline_size",
             "opt_sigma0", "opt_verify_resolution", "opt_verify_target_faces",
             "opt_latent_range", "opt_shell_scale"),
            "SDF-OPT-POSITIVE-001",
        )
        analysis_backend = str(values.get("opt_analysis", "fea")).strip().lower()
        if analysis_backend not in {"fea", "surrogate"}:
            ctx.add(
                "SDF-OPT-ANALYSIS-001",
                Severity.ERROR,
                "opt_analysis must be 'fea' or 'surrogate'.",
                field_name="opt_analysis",
            )
        elif analysis_backend == "fea":
            # These controls define the actual gmsh discretization, but are
            # meaningless on the surface-graph surrogate path. Keeping them in
            # the mode-wide recommended set made an otherwise complete
            # surrogate config fail strict preflight for missing FEA settings.
            for field_name in ("opt_target_faces", "opt_mesh_size_max"):
                if not str(values.get(field_name, "")).strip():
                    ctx.add(
                        "SDF-OPT-FEA-REC-001",
                        Severity.WARNING,
                        f"{field_name} is recommended when opt_analysis=fea; "
                        "verify the published meshing default is intended.",
                        field_name=field_name,
                        promote_in_strict=True,
                    )
        elif analysis_backend == "surrogate":
            for field_name, label in (
                ("opt_surrogate_checkpoint", "HI-MGN checkpoint"),
                ("opt_surrogate_config", "HI-MGN inference config"),
            ):
                if not str(values.get(field_name, "")).strip():
                    ctx.add(
                        "SDF-OPT-SURROGATE-001",
                        Severity.ERROR,
                        f"{field_name} is required when opt_analysis=surrogate ({label}).",
                        field_name=field_name,
                        hint="Choose an existing file or switch opt_analysis back to fea.",
                    )
        known_cases = {"vertical", "horizontal", "diagonal", "torsion"}
        cases = {str(c).strip().lower() for c in as_list(values.get("opt_load_cases", []))}
        unknown = cases - known_cases
        if unknown:
            ctx.add("SDF-OPT-LOAD-001", Severity.ERROR,
                    f"opt_load_cases contains unknown case(s) {sorted(unknown)}; "
                    f"available: {sorted(known_cases)}.", field_name="opt_load_cases")
        known_dims = {"bbox_x", "bbox_y", "bbox_z", "volume", "area"}
        dims = {str(d).strip().lower() for d in as_list(values.get("opt_condition_dims", []))}
        if dims - known_dims:
            ctx.add("SDF-OPT-COND-001", Severity.ERROR,
                    f"opt_condition_dims contains unknown descriptor(s) "
                    f"{sorted(dims - known_dims)}; available: {sorted(known_dims)}.",
                    field_name="opt_condition_dims")
        if "bbox_y" in dims:
            ctx.add("SDF-OPT-COND-002", Severity.WARNING,
                    "bbox_y has zero train-split standard deviation in DeepJEB; "
                    "searching over it moves nothing.",
                    field_name="opt_condition_dims", promote_in_strict=True)
        popsize = integer(values.get("opt_popsize"))
        budget = integer(values.get("opt_budget"))
        if popsize and budget and budget < 2 * popsize:
            ctx.add("SDF-OPT-BUDGET-001", Severity.WARNING,
                    f"opt_budget={budget} allows fewer than two CMA-ES generations at "
                    f"opt_popsize={popsize}; the search cannot adapt.",
                    field_name="opt_budget", promote_in_strict=True)

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
        repository="methods/SDFFlow",
        entrypoint="SDFFlow_main.py",
        valid_modes=("train", "train_vae", "train_fm", "sample", "reconstruct", "interpolate", "optimize", "evaluate"),
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
            # sample_index_b is required by SDF-INTERP-007 for slerp_noise / lerp_latent
            # only: interpolation_space cond_sweep integrates ONE noise row (sample_index_a)
            # under a sweep of conditions and has no second endpoint. alpha keeps its
            # mode default (0.5) below.
            "interpolate": frozenset({"vae_modelpath", "fm_modelpath", "output_dir", "seed", "source_num_samples", "sample_index_a", "ode_steps", "mc_resolution"}),
            "optimize": frozenset({"vae_modelpath", "fm_modelpath", "output_dir", "seed",
                                   "ode_steps", "mc_resolution", "opt_subspace_dim",
                                   "opt_budget", "opt_popsize", "opt_baseline_size",
                                   "opt_load_cases"}),
            "evaluate": frozenset({"vae_modelpath", "dataset_dir", "output_dir"}),
        },
        recommended_by_mode={
            # `seed` seeds torch/numpy/python and the train DataLoader shuffle in
            # every training worker. Without it a sweep has no noise floor: two
            # arms can differ purely by run-to-run variance and nothing says so.
            "train": frozenset({"seed", "split_seed", "use_conditions", "skip_completed_stages"}),
            "train_vae": frozenset({"seed", "split_seed", "use_ema", "use_amp"}),
            "train_fm": frozenset({"seed", "split_seed", "use_conditions", "use_ema", "use_amp"}),
            "sample": frozenset({"cfg_scale"}),
            "optimize": frozenset({"opt_length_scale", "opt_material_e", "opt_yield_stress"}),
            # split_by_parent decides which shapes are held out at all; an
            # evaluate config that omits it silently inherits the checkpoint's.
            "evaluate": frozenset({"split_seed", "split_by_parent", "eval_split"}),
        },
        defaults={},
        defaults_by_mode={
            # cond_dropout_mode 'all' is model/velocity_net.py's default and reproduces
            # the pre-conditional parameter set exactly, so every existing FM checkpoint
            # keeps loading; per_dim is opt-in per training run.
            "train": {"skip_completed_stages": True, "cond_dropout_mode": "all",
                      "cond_dropout_all_prob": 0.1},
            "train_vae": {"use_ema": False, "use_amp": False},
            "train_fm": {"use_conditions": False, "use_ema": False, "use_amp": False,
                         "cond_dropout_mode": "all", "cond_dropout_all_prob": 0.1},
            # cfg_scale 1.0 is the native default (inference_profiles/sample.py) and the
            # documented conservative setting: the in-repo measurement put volume error at
            # 8.5% -> 21.2% going from 1.0 to 3.0. Everything below cfg_scale mirrors
            # sample.py's own `config.get(key, default)` calls: the two switches
            # (guidance_enabled, newton_rounds) are off, and the remaining values are the
            # pilot settings that only take effect once a switch is on.
            "sample": {"cfg_scale": 1.0, "ode_steps": 50, "mc_resolution": 128,
                       "guidance_enabled": False, "guidance_t_start": 0.3,
                       "guidance_eta": 0.1, "guidance_step_mode": "velocity_dt",
                       "guidance_targets": "volume,area",
                       "soft_descriptor_resolution": 48, "soft_descriptor_tau": 0.032,
                       "newton_rounds": 0, "newton_step_cap_rms": 0.12,
                       "newton_line_search_tries": 3, "newton_measure_resolution": 128,
                       "condition_audit": "geometric"},
            # latent_refine_lr / _prior_weight are defaulted (not just known)
            # so Studio's required rows for this mode have a value the
            # SDF-REFINE-002 validator accepts; lr 0 would be rejected.
            "reconstruct": {"mc_resolution": 128, "latent_refine_steps": 0,
                            "latent_refine_lr": 0.01, "latent_refine_prior_weight": 0.0},
            "interpolate": {"alpha": 0.5, "ode_steps": 50, "mc_resolution": 128, "plot_dpi": 180, "plot_max_faces": 0,
                            "interpolation_space": "slerp_noise", "sweep_steps": 5},
            # eval_methods / calibration_* are the task-specific defaults of
            # inference_profiles/evaluate.py. newton_rounds is deliberately absent here:
            # evaluate's `e2` method setdefaults it to 3 (a benchmark with 0 rounds is not
            # a benchmark), which is a different default from sample's 0, so a single
            # mode-wide value would misreport one of the two.
            "evaluate": {
                "eval_split": "val", "eval_num_shapes": 0, "eval_seed": 0, "mc_resolution": 128,
                "latent_refine_steps": 0, "latent_refine_lr": 0.01, "latent_refine_prior_weight": 0.0,
                "eval_task": "reconstruction", "condition_audit": "geometric",
                "eval_methods": "plain,rejection,e2",
                "calibration_num_shapes": 64, "calibration_samples_per_shape": 4,
                "guidance_t_start": 0.3, "guidance_eta": 0.1,
                "guidance_step_mode": "velocity_dt", "guidance_targets": "volume,area",
                "soft_descriptor_resolution": 48, "soft_descriptor_tau": 0.032,
                "newton_step_cap_rms": 0.12, "newton_line_search_tries": 3,
                # newton_measure_resolution follows mc_resolution natively (both
                # 128), so E2 accepts steps on the grid the audit and the
                # calibration use. candidate_multiplier's evaluate default is 4,
                # different from sample's native 1.
                "newton_measure_resolution": 128, "candidate_multiplier": 4,
                "calibration_min_r2": 0.5,
            },
            "optimize": {
                "ode_steps": 50, "mc_resolution": 128, "opt_subspace_dim": 12,
                "opt_condition_dims": "volume,area", "opt_budget": 120, "opt_popsize": 8,
                "opt_sigma0": 1.0, "opt_baseline_size": 12, "opt_load_cases": "vertical,diagonal",
                "opt_length_scale": 0.19 / 1.8, "opt_stress_percentile": 99.5,
                "opt_mesh_size_max": 0.05, "opt_target_faces": 12000,
                "opt_material_e": 113.8e9, "opt_material_nu": 0.342,
                "opt_material_rho": 4430.0, "opt_yield_stress": 903e6,
                "opt_stress_margin": 1.0, "opt_disp_margin": 1.0,
                "opt_stress_weight": 6.0, "opt_disp_weight": 3.0,
                "opt_analysis": "fea", "opt_surrogate_target_nodes": 5000,
                "opt_verify_resolution": 160, "opt_verify_target_faces": 30000,
                "opt_verify_mesh_size_max": 0.035,
            },
        },
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train", "train_vae", "train_fm", "evaluate"})),
            PathRule("init_vae_modelpath", PathKind.INPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_best_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_vae"})),
            PathRule("vae_modelpath", PathKind.INPUT_FILE, frozenset({"train_fm", "sample", "reconstruct", "interpolate", "optimize", "evaluate"})),
            PathRule("fm_modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_fm"})),
            # evaluate reads the FM only for eval_task descriptor_calibration / conditional;
            # SDF-EVAL-004 makes the key mandatory there, and this rule checks the file
            # whenever the key is present (a reconstruction config simply omits it).
            PathRule("fm_modelpath", PathKind.INPUT_FILE, frozenset({"sample", "interpolate", "optimize", "evaluate"})),
            # descriptor_calibration_path is an OUTPUT of eval_task descriptor_calibration
            # and an INPUT of sample guidance/Newton and of eval_task conditional
            # (c2/e2/c2e2), so only the evaluate side gets a PathRule -- OUTPUT_FILE, which
            # is the writability check the calibration task needs. There is deliberately NO
            # INPUT_FILE rule for `sample`: the file legitimately does not exist until the
            # calibration task has run, so a hard rule would make the documented ordering
            # (calibrate, then sample) fail preflight. validate_sdfflow reports the input
            # side per task instead (SDF-CALIB-001 ERROR when the key is missing while
            # guidance/Newton is on, SDF-CALIB-002 NOTICE when the file is not there yet,
            # SDF-CALIB-003 / SDF-EVAL-007 for the calibration task itself).
            # Caveat: under eval_task conditional the file is an input, so an existing one
            # also draws PATH-OUTPUT-EXISTS ("may overwrite"); that warning is spurious for
            # that task -- SDF-CALIB-002/003 are the accurate pair.
            PathRule("descriptor_calibration_path", PathKind.OUTPUT_FILE, frozenset({"evaluate"})),
            PathRule("input_mesh", PathKind.INPUT_FILE, frozenset({"reconstruct"})),
            # Also read by `condition_audit surrogate` in sample / evaluate, where a
            # typo used to cost the whole ODE + Marching Cubes pass before the
            # backend silently downgraded to a geometric audit.
            PathRule("opt_surrogate_checkpoint", PathKind.INPUT_FILE,
                     frozenset({"optimize", "sample", "evaluate"})),
            PathRule("opt_surrogate_config", PathKind.INPUT_FILE,
                     frozenset({"optimize", "sample", "evaluate"})),
            PathRule("output_dir", PathKind.OUTPUT_DIR),
        ),
        validators=(validate_sdfflow,),
        import_modules=("torch", "h5py", "numpy", "trimesh", "skimage"),
        dataset_kind="sdf_hdf5",
    )
