from __future__ import annotations

from ..diagnostics import Severity
from .base import MethodSpec, PathKind, PathRule, SpecValidationContext, integer, numeric
from .meshgraphnets import validate_meshgraphnets
from .meshgraphnets_variational import (VAR_KEYS, VAR_REMOVED_KEYS,
                                        validate_spread_keys)


# The backbone is the HI-MGN V-cycle, so the whole mesh/multiscale key surface
# is inherited verbatim from the variational spec. What differs is the
# generative machinery: this is now an LDGN-style two-stage model (Lino,
# Pfaff & Thuerey, ICLR 2025, arXiv:2504.02843) -- a near-lossless compressor
# (stage 1, 'train_ae') onto a coarse spatial latent, and a flow-matching
# prior over that latent alone (stage 2, 'train_prior'), with 'train' running
# both in sequence. Every key from the variational tree's pooled-vector VAE +
# generic prior family is gone, replaced by AE_PRIOR_KEYS (the two-stage
# architecture/mode surface) and FLOW_ONLY_KEYS (the flow-matching ODE
# surface, unchanged: the prior is still a flow, just over a coarse latent
# instead of the full field). Reusing VAR_KEYS rather than re-listing the
# mesh/multiscale keys means a future change to the shared backbone stays in
# one place.
AE_PRIOR_KEYS = frozenset(
    {
        # Channels per COARSE-MESH NODE in the compressed latent (not one
        # pooled vector -- see model/autoencoder.py). ARCHITECTURE-DEFINING:
        # sizes both the compressor's bottleneck and the prior's input/output,
        # so a checkpoint only loads under the value it trained with.
        "latent_ch",
        # KL(q(z|y) || N(0,I)) weight for the stage-1 compressor. Kept tiny
        # (paper: ~1e-6) on purpose -- this is a near-lossless compressor, not
        # a generative bottleneck; the generative job is stage 2's flow.
        "ae_kl_weight",
        # AdaLN-Zero GnBlocks in the stage-2 LatentFlowPrior's trunk.
        "prior_blocks",
        # mode 'train_prior' only: path to a checkpoint saved by a completed
        # 'train_ae' run. Loaded once at construction and frozen; the
        # compressor never receives gradient during stage 2.
        "ae_checkpoint",
        # mode 'train' (combined) only: stage-1 epoch budget. Stage 2 then
        # runs for training_epochs, mirroring SimulGenVAE's combined pipeline.
        "ae_epochs",
    }
)

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
        # Field row the peak-to-valley auxiliary term scores; that term lives
        # on the variational decoder, which this method does not have.
        "pv_channel",
        "prior_type", "use_conditional_prior", "prior_family", "prior_nll_weight",
        "prior_fm_steps", "prior_fm_solver", "prior_mp_layers", "prior_hidden_dim",
        "prior_temperature", "prior_kl_reg_weight", "prior_cov_rank",
        # Inference-only latent inflation for the FM prior; flow has no latent.
        "latent_inflation",
        "prior_min_std", "prior_mixture_components", "prior_grad_to_encoder",
        "vae_valid_prior_samples",
        "gamma_es", "es_samples", "es_steps", "es_noise_source", "es_start_epoch",
    }
) | VAR_REMOVED_KEYS

# Controls that only existed in the copied variational model-split path.  They
# remain known so preflight can diagnose an old config precisely, but they are
# not part of the executable cHI-MGNflow surface.
FLOW_RUNTIME_REMOVED_KEYS = frozenset(
    {"pipeline_microbatches", "std_noise", "noise_gamma", "noise_std_ratio"}
)

# Output-parameterization keys from the retired single-stage, field-space
# flow design (network reparameterized its output as the clean field 'x'
# instead of velocity; flow_head_eps floored the resulting division near
# t=1). The LDGN rewrite's prior always emits velocity over the
# coarse latent, so there is no field-space division left to floor, and
# flow_loss_weighting='x0' gets the same data-prediction effect as a LOSS
# REWEIGHTING on that one untouched head. Kept known (not silently
# CFG-UNKNOWN) so a leftover config gets a precise diagnostic instead of a
# generic one.
FLOW_HEAD_REMOVED_KEYS = frozenset({"flow_head", "flow_head_eps"})

FLOW_KEYS = (VAR_KEYS - REMOVED_LATENT_KEYS - FLOW_RUNTIME_REMOVED_KEYS) | FLOW_ONLY_KEYS | AE_PRIOR_KEYS

# Shared base for every training mode: what it takes to construct the full
# LatentDiffusionGraphNet (compressor + prior), even in 'train_prior' where
# the compressor's weights are immediately overwritten by a frozen checkpoint.
_BASE_TRAIN_REQUIRED = frozenset({
    "dataset_dir", "modelpath", "input_var", "output_var", "edge_var",
    "latent_dim", "training_epochs", "batch_size", "learningr",
})


def validate_chi_mgnflow(ctx: SpecValidationContext) -> None:
    validate_meshgraphnets(ctx)
    values = ctx.values
    validate_spread_keys(ctx)

    for name in sorted(REMOVED_LATENT_KEYS.intersection(values)):
        ctx.add(
            "FLOW-REMOVED",
            Severity.WARNING,
            f"{name} belongs to the variational method; cHI-MGNflow has no latent "
            f"or learned prior and ignores it.",
            field_name=name,
            hint="Delete the line. See methods/HI_MGNFlow/README.md for the replacement controls.",
            promote_in_strict=True,
        )

    for name in sorted(FLOW_HEAD_REMOVED_KEYS.intersection(values)):
        ctx.add(
            "FLOW-HEAD-REMOVED",
            Severity.WARNING,
            f"{name} belongs to cHI-MGNflow's retired single-stage field-space "
            "flow design (superseded by the LDGN-style two-stage compressor + "
            "coarse-latent flow prior) and is not read by the current model.",
            field_name=name,
            hint="Delete the line; flow_loss_weighting=x0 gets the equivalent "
                 "data-prediction effect on the one velocity head that remains.",
            promote_in_strict=True,
        )

    for name in sorted(FLOW_RUNTIME_REMOVED_KEYS.intersection(values)):
        value = numeric(values[name])
        if name == "std_noise" and value == 0.0:
            ctx.add(
                "FLOW-LEGACY-NOISE",
                Severity.NOTICE,
                "std_noise=0 is accepted only for compatibility with older configs; "
                "the cHI-MGNflow trainer does not implement MGN input noise.",
                field_name=name,
                hint="Delete the line.",
            )
            continue
        ctx.add(
            "FLOW-RUNTIME-REMOVED",
            Severity.ERROR,
            f"{name} belongs to the unavailable copied model-split/noise path and "
            "is not consumed by cHI-MGNflow.",
            field_name=name,
            hint="Delete the line; cHI-MGNflow supports parallel_mode ddp only.",
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
            Severity.ERROR,
            "cHI-MGNflow's LDGN-style architecture compresses the field onto a "
            "coarse mesh level, which requires use_multiscale=True (with "
            "multiscale_levels >= 1 and a matching mp_per_level). A flat "
            "(non-hierarchical) HI-MGN has no coarse level to place the latent "
            "on -- this now raises inside LatentDiffusionGraphNet.__init__, "
            "not just a discouraged setting.",
            field_name="use_multiscale",
        )

    # message_passing_num is a MeshGraphNets-family key inherited via VAR_KEYS;
    # cHI-MGNflow's V-cycle depth comes entirely from mp_per_level (one entry
    # per pre/coarsest/post stage) and never reads message_passing_num.
    if "message_passing_num" in values:
        ctx.add(
            "FLOW-MPNUM-INERT",
            Severity.NOTICE,
            "message_passing_num is not read by cHI-MGNflow; the V-cycle's "
            "per-level message-passing depth is set entirely by mp_per_level.",
            field_name="message_passing_num",
            hint="Delete the line, or set mp_per_level if you meant to change depth.",
        )

    if "latent_ch" in values:
        v = integer(values["latent_ch"])
        if v is None or v < 1:
            ctx.add("FLOW-LATENTCH", Severity.ERROR,
                    "latent_ch must be an integer >= 1.", field_name="latent_ch")

    if "prior_blocks" in values:
        v = integer(values["prior_blocks"])
        if v is None or v < 1:
            ctx.add("FLOW-PRIORBLOCKS", Severity.ERROR,
                    "prior_blocks must be an integer >= 1.", field_name="prior_blocks")

    if "ae_kl_weight" in values:
        v = numeric(values["ae_kl_weight"])
        if v is None or v < 0:
            ctx.add("FLOW-AEKL", Severity.ERROR,
                    "ae_kl_weight must be a number >= 0.", field_name="ae_kl_weight")
        elif v > 1e-3:
            ctx.add(
                "FLOW-AEKL-LARGE",
                Severity.WARNING,
                f"ae_kl_weight={v:g} is large for a near-lossless compressor "
                "(paper: ~1e-6). A strongly regularised stage-1 latent starves "
                "stage 2's reconstruction quality independent of how well the "
                "flow prior fits it.",
                field_name="ae_kl_weight",
            )

    if "ae_epochs" in values:
        v = integer(values["ae_epochs"])
        if v is None or v < 1:
            ctx.add("FLOW-AEEPOCHS", Severity.ERROR,
                    "ae_epochs must be an integer >= 1.", field_name="ae_epochs")


def build_chi_mgnflow_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="chi_mgnflow",
        display_name="cHI-MGNflow",
        model_ids=("chi-mgnflow",),
        repository="methods/HI_MGNFlow",
        entrypoint="CHiMGNFlow_main.py",
        valid_modes=("train", "train_ae", "train_prior", "inference"),
        known_keys=FLOW_KEYS | REMOVED_LATENT_KEYS | FLOW_RUNTIME_REMOVED_KEYS | FLOW_HEAD_REMOVED_KEYS,
        required_by_mode={
            "train": _BASE_TRAIN_REQUIRED | {"ae_epochs"},
            "train_ae": _BASE_TRAIN_REQUIRED,
            "train_prior": _BASE_TRAIN_REQUIRED | {"ae_checkpoint"},
            "inference": frozenset({
                "modelpath", "infer_dataset", "input_var", "output_var", "edge_var",
            }),
        },
        recommended_by_mode={
            "train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode",
                                "flow_steps", "best_by"}),
            "train_ae": frozenset({"feature_loss_weights", "split_seed", "parallel_mode"}),
            "train_prior": frozenset({"split_seed", "parallel_mode", "flow_steps", "best_by"}),
        },
        defaults={
            "parallel_mode": "ddp",
            "use_multiscale": True,
            "latent_ch": 4,
            "ae_kl_weight": 1e-6,
            "prior_blocks": 4,
            "flow_steps": 30,
            "flow_solver": "heun",
            "flow_time_freqs": 16,
            "val_num_samples": 8,
            "best_by": "crps",
        },
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/rollout"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train", "train_ae", "train_prior"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train", "train_ae", "train_prior"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            # Stage 2 alone loads a frozen stage-1 checkpoint at construction
            # time, before training even starts -- an input, not an output.
            PathRule("ae_checkpoint", PathKind.INPUT_FILE, frozenset({"train_prior"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            # Scoring-only, but a stale path here is SILENT: rollout.py prints a
            # skip at the very end of a finished run, so the histogram is simply
            # missing after the GPU time is already spent. Validating it as an
            # input file also brings PATH-CASE-001, which catches SAOI vs saoi.
            PathRule("eval_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_chi_mgnflow,),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
