from __future__ import annotations

from ..diagnostics import Severity
from .base import MethodSpec, PathKind, PathRule, SpecValidationContext, integer, numeric
from .meshgraphnets import validate_meshgraphnets
from .meshgraphnets_variational import VAR_KEYS


# The backbone is the HI-MGN V-cycle, so the whole mesh/multiscale key surface
# is inherited verbatim from the variational spec. What differs is only the
# generative machinery: every latent/prior key is gone, replaced by four flow
# keys. Reusing VAR_KEYS rather than re-listing them means a future change to
# the shared backbone stays in one place.
FLOW_ONLY_KEYS = frozenset(
    {
        # Number of ODE steps used at inference. A SAMPLING-TIME choice, not an
        # architecture one: the same checkpoint integrates at any K without
        # retraining, because what was learned is a continuous velocity field
        # and K is only its quadrature resolution.
        "flow_steps",
        # 'heun' (2nd-order trapezoid, 2 evaluations/step) or 'euler'.
        "flow_solver",
        # Fourier octaves in the time embedding. ARCHITECTURE-DEFINING: it sets
        # the AdaLN input width, so a checkpoint only loads under the value it
        # trained with. Persisted in model_config.
        "flow_time_freqs",
        # Where along the path the training budget is spent. 'uniform' weights
        # every t equally; 'logitnormal' concentrates on the middle, which is
        # where the velocity is hardest to predict. Training-time only -- it
        # changes convergence speed, not the optimum, so two runs that differ
        # only here are directly comparable.
        "flow_t_sampling", "flow_t_logit_scale",
        # Parameterization, expressed as a loss weight so the velocity head (and
        # therefore the exact ODE) is untouched. 'uniform' = velocity
        # prediction; 'x0' = data prediction, (1-s*t)^2, which concentrates the
        # budget on the DETERMINISTIC end of the path.
        "flow_loss_weighting",
        # Share of training graphs pinned to t=0, where the objective collapses
        # to a pure deterministic regression on E[y|g]. This is what trains the
        # deterministic mode instead of merely reading it out.
        "flow_det_prob",
        # Inference readout: 'sample' (integrate), 'mean' (1 forward at t=0) or
        # 'ensemble_mean' (average of num_vae_samples draws).
        "flow_predict",
        # Validation sampling: how many ODE steps and how many ensemble members.
        # Cheaper than inference on purpose -- validation runs every val_interval.
        "val_flow_steps", "val_num_samples",
        # Checkpoint selection: 'recon' (one-step velocity regression on the
        # held-out split) or 'crps' (the sampling-based ensemble score, which is
        # the metric that mirrors inference).
        "best_by",
    }
)

# Latent/prior keys from the variational tree. This method has no posterior and
# no learned prior, so any of these in a config is a stale copy-paste and would
# silently do nothing.
REMOVED_LATENT_KEYS = frozenset(
    {
        "use_vae", "vae_latent_dim", "vae_mp_layers", "vae_graph_aware",
        "posterior_min_std", "num_z", "z_conditioning", "mmd_bandwidth",
        "mmd_gather_ranks", "lambda_mmd", "beta_aux", "alpha_recon", "recon_loss",
        "prior_type", "use_conditional_prior", "prior_family", "prior_nll_weight",
        "prior_fm_steps", "prior_fm_solver", "prior_mp_layers", "prior_hidden_dim",
        "prior_temperature", "prior_kl_reg_weight", "prior_cov_rank",
        "prior_min_std", "prior_mixture_components", "prior_grad_to_encoder",
        "vae_valid_prior_samples",
        "gamma_es", "es_samples", "es_steps", "es_noise_source", "es_start_epoch",
    }
)

FLOW_KEYS = (VAR_KEYS - REMOVED_LATENT_KEYS) | FLOW_ONLY_KEYS


def validate_chi_mgnflow(ctx: SpecValidationContext) -> None:
    validate_meshgraphnets(ctx)
    values = ctx.values

    for name in sorted(REMOVED_LATENT_KEYS.intersection(values)):
        ctx.add(
            "FLOW-REMOVED",
            Severity.WARNING,
            f"{name} belongs to the variational method; cHI-MGNflow has no latent "
            f"or learned prior and ignores it.",
            field_name=name,
            hint="Delete the line. See cHI-MGNflow/README.md for the replacement controls.",
            promote_in_strict=True,
        )

    for name in ("flow_steps", "val_flow_steps", "val_num_samples", "flow_time_freqs"):
        if name in values:
            v = integer(values[name])
            if v is None or v < 1:
                ctx.add(
                    "FLOW-POSITIVE",
                    Severity.ERROR,
                    f"{name} must be an integer >= 1.",
                    field_name=name,
                )

    if "flow_solver" in values and str(values["flow_solver"]).lower().strip() not in {"heun", "euler"}:
        ctx.add("FLOW-SOLVER", Severity.ERROR,
                "flow_solver must be 'heun' or 'euler'.", field_name="flow_solver")

    if ("flow_t_sampling" in values
            and str(values["flow_t_sampling"]).lower().strip() not in {"uniform", "logitnormal"}):
        ctx.add("FLOW-TSAMPLING", Severity.ERROR,
                "flow_t_sampling must be 'uniform' or 'logitnormal'.",
                field_name="flow_t_sampling")

    if "flow_t_logit_scale" in values:
        v = numeric(values["flow_t_logit_scale"])
        if v is None or v <= 0:
            ctx.add("FLOW-TSCALE", Severity.ERROR,
                    "flow_t_logit_scale must be > 0.", field_name="flow_t_logit_scale")

    # Sampling cost is the one real expense of this method, and it is easy to
    # walk into by copying num_vae_samples from a one-shot config where each
    # sample was a single forward.
    steps = integer(values.get("flow_steps", 30)) or 30
    solver = str(values.get("flow_solver", "heun")).lower().strip()
    per_sample = steps * (2 if solver == "heun" else 1)
    n_samples = integer(values.get("num_vae_samples", 1)) or 1
    if ctx.mode == "inference" and per_sample * n_samples > 20000:
        ctx.add(
            "FLOW-COST",
            Severity.WARNING,
            f"num_vae_samples={n_samples} x {per_sample} forwards/sample = "
            f"{per_sample * n_samples:,} network forwards per scene.",
            field_name="num_vae_samples",
            hint="Lower num_vae_samples, or flow_steps (K is a sampling-time "
                 "choice -- no retraining needed).",
        )

    if str(values.get("best_by", "recon")).lower().strip() not in {"recon", "crps", "det"}:
        ctx.add("FLOW-BESTBY", Severity.ERROR,
                "best_by must be 'recon', 'crps' or 'det'.", field_name="best_by")

    if ("flow_loss_weighting" in values
            and str(values["flow_loss_weighting"]).lower().strip() not in {"uniform", "x0"}):
        ctx.add("FLOW-WEIGHTING", Severity.ERROR,
                "flow_loss_weighting must be 'uniform' or 'x0'.",
                field_name="flow_loss_weighting")

    if ("flow_predict" in values
            and str(values["flow_predict"]).lower().strip()
            not in {"sample", "mean", "ensemble_mean"}):
        ctx.add("FLOW-PREDICT", Severity.ERROR,
                "flow_predict must be 'sample', 'mean' or 'ensemble_mean'.",
                field_name="flow_predict")

    if "flow_det_prob" in values:
        v = numeric(values["flow_det_prob"])
        if v is None or not 0.0 <= v < 1.0:
            ctx.add("FLOW-DETPROB", Severity.ERROR,
                    "flow_det_prob must be in [0, 1).", field_name="flow_det_prob")

    # x0 weighting and logit-normal t sampling pull in opposite directions:
    # the first concentrates budget at t=0, the second de-emphasises both ends.
    if (str(values.get("flow_loss_weighting", "uniform")).lower().strip() == "x0"
            and str(values.get("flow_t_sampling", "uniform")).lower().strip() == "logitnormal"):
        ctx.add("FLOW-WEIGHT-CONFLICT", Severity.WARNING,
                "flow_loss_weighting x0 concentrates the budget at t=0 while "
                "flow_t_sampling logitnormal de-emphasises both endpoints; they "
                "largely cancel.",
                field_name="flow_loss_weighting",
                hint="Treat them as alternatives, not as a stack.")

    if values.get("use_multiscale", False) is not True:
        ctx.add(
            "FLOW-FLAT",
            Severity.WARNING,
            "use_multiscale is off. At t~0 the network sees white noise plus the "
            "geometry and must recover global structure from it, which a flat "
            "message-passing stack cannot do within its receptive field.",
            field_name="use_multiscale",
        )


def build_chi_mgnflow_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="chi_mgnflow",
        display_name="cHI-MGNflow",
        model_ids=("chi-mgnflow",),
        repository="cHI-MGNflow",
        entrypoint="CHiMGNFlow_main.py",
        valid_modes=("train", "inference"),
        known_keys=FLOW_KEYS | REMOVED_LATENT_KEYS,
        required_by_mode={
            "train": frozenset({
                "dataset_dir", "modelpath", "input_var", "output_var", "edge_var",
                "latent_dim", "training_epochs", "batch_size", "learningr",
            }),
            "inference": frozenset({
                "modelpath", "infer_dataset", "input_var", "output_var", "edge_var",
            }),
        },
        recommended_by_mode={
            "train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode",
                                "flow_steps", "best_by"}),
        },
        defaults={
            "parallel_mode": "ddp",
            "use_multiscale": True,
            "flow_steps": 30,
            "flow_solver": "heun",
            "flow_time_freqs": 16,
            "val_num_samples": 8,
            "best_by": "crps",
        },
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/rollout"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_chi_mgnflow,),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
