from __future__ import annotations

from ..diagnostics import Severity
from .base import MethodSpec, PathKind, PathRule, RESUME_KEYS, SpecValidationContext, integer, numeric, validate_positive_fields, validate_resume
from .meshgraphnets import check_renamed_keys, validate_meshgraphnets


# Keys consumed by the live variational runtime (traced from config[...] /
# config.get(...) reachable from methods/MeshGraphNets_Variational/MeshGraphNets_main.py).
VAR_KEYS = frozenset(
    {
        "use_deformed_edge", "displacement_indices", "periodic_box", "write_preprocessing",
        # Group samples by data/{sid}/metadata.attrs[<attr>] so every draw of
        # one geometry lands in the same split (general_modules/grouped_split.py).
        "split_group_attr",
        "model", "mode", "gpu_ids", "parallel_mode", "log_file_dir", "modelpath",
        "dataset_dir", "infer_dataset", "eval_dataset", "inference_output_dir",
        "infer_timesteps", "split_seed", "input_var", "output_var", "cond_var",
        # split_seed decides data-split MEMBERSHIP; training_seed is a separate
        # stream (init, shuffle, CUDA, stochastic latent -- single_training.py).
        # Two runs differing only in training_seed share a split, which is what
        # makes them a seed replicate rather than a different experiment.
        "training_seed",
        # Inference-only: inflate the FM prior's z around its per-graph center
        # (ConditionalFMPrior.sample_n). Scalar or list; a list is cycled per
        # draw batch so one pass yields the lam-vs-spread curve.
        "latent_inflation",
        "feature_loss_weights", "edge_var", "positional_features",
        "message_passing_num", "training_epochs", "batch_size", "learningr",
        "latent_dim", "num_workers", "prefetch_factor", "std_noise", "noise_gamma",
        # Time integration (CONFIGURATION_REFERENCE.md section 3.5): ar_ot or ar_rt.
        "time_integration",
        "noise_std_ratio", "weight_decay", "warmup_epochs", "augment_geometry",
        "grad_accum_steps", "use_checkpointing", "use_amp", "use_ema", "ema_decay",
        "use_compile", "test_interval", "val_interval", "test_max_batches",
        # Batch for the rank-0 (unsharded) validation loader. Defaults to
        # batch_size; set it when a DDP run lowers batch_size per rank.
        "val_batch_size",
        # Epoch at which a joint run freezes the simulator and trains ONLY the
        # conditional prior for the rest (latent standardization fitted, fresh
        # cosine). 0 = never. Single-GPU only.
        "prior_freeze_epoch",
        # False takes page-locking out of the data path. Worth it when several
        # DDP ranks contend for the driver-serialized pin, which is the
        # documented bottleneck for large variable-size graph batches.
        "pin_memory",
        "display_trainset", "display_testset", "use_node_types", "use_world_edges",
        "world_radius_multiplier", "world_max_num_neighbors", "world_edge_backend",
        "coarse_world_edges", "use_parallel_stats", "static_cache_per_worker",
        "hierarchy_cache_dir", "hierarchy_cache_build_workers",
        "hierarchy_cache_wait_timeout", "hierarchy_cache_keep",
        # Coarsening-partition variance control: hierarchy_variants caches K
        # independently-seeded partitions per sample so training can rotate them
        # per epoch; hierarchy_seed pins the (otherwise unseeded) FPS draw at
        # inference so a rollout is reproducible.
        "hierarchy_variants", "hierarchy_seed",
        "test_batch_idx", "plot_feature_idx",
        "use_multiscale", "coarsening_type", "voronoi_clusters", "multiscale_levels",
        "mp_per_level", "pipeline_microbatches", "make_histogram", "show_histogram",
        # inference: write one HDF5 per (scene, VAE sample), or only the
        # spread histogram + spread_values.npz. False makes a many-draw
        # distribution study affordable on disk.
        "save_rollouts",
        "histogram_bins", "histogram_clip_quantile",
        # Which output channel the spread statistic is taken over, and its name
        # on the plot. The ground-truth row is 3 + spread_channel, so one key
        # points both sides at the same field; it used to be hardcoded to the
        # displacement layout's z_disp row.
        # ...and `spread_stat` is how that channel is reduced to one number per
        # realization: 'range' (max - min, the original), 'mean' or 'std'. It is
        # a key because max - min is degenerate on a saturating or
        # displacement-driven field, where it reports the boundary condition
        # rather than the solution and every calibration score built on it is
        # noise. 'mag_max' is the vector stat: max over nodes of |u| across a
        # LIST of channels (`spread_channel 0, 1, 2` = total displacement).
        "spread_channel", "spread_label", "spread_stat",
        # VAE / conditional-prior branch
        "use_vae", "vae_latent_dim", "vae_encoder_mp_layers", "vae_graph_aware",
        # How z reaches the processor: 'concat' (legacy Linear([x, z]) fuser) or
        # 'adaln' (AdaLN-Zero modulation, identity at init). It changes the
        # parameter names, so a checkpoint is tied to the value it trained with.
        "z_conditioning",
        # Where z lives on the V-cycle (model/latent.py): 'global' (num_z
        # per-graph slots, HI-MGN-V), 'local' (latent_ch channels per coarsest
        # node, HI-MGNflow), or 'hybrid' (both). Architecture-defining, persisted
        # in the checkpoint. vae_local_mp_layers is the depth of the coarse-node
        # posterior head. latent_reg picks the regularizer pulling the posterior
        # toward N(0, I): 'mmd' (lambda_mmd) or per-sample 'kl' (kl_weight).
        "latent_type", "latent_ch", "vae_local_mp_layers", "latent_reg", "kl_weight",
        # All-gather z across DDP ranks before MMD so the two-sample estimator
        # sees the GLOBAL batch instead of the per-rank one (default True).
        "mmd_gather_ranks",
        "inference_batch_size", "inference_batch_size_max", "inference_batch_size_min",
        "inference_batch_vram_fraction", "vae_valid_prior_samples", "recon_loss",
        "alpha_recon", "beta_aux", "pv_channel", "lambda_mmd", "mmd_bandwidth",
        "posterior_min_std", "num_z", "inference_samples", "prior_type",
        "use_conditional_prior", "prior_family", "prior_nll_weight",
        "prior_fm_steps", "prior_fm_solver", "prior_mp_layers", "prior_hidden_dim",
        "prior_temperature", "prior_kl_reg_weight", "prior_cov_rank",
        # FM velocity-MLP width (defaults to prior_hidden_dim) and the
        # condition-dependent mean/scale head of the FM base; both persist in
        # the checkpoint model_config.
        "prior_velocity_hidden_dim", "prior_fm_moments",
        # The rest of the conditional-prior architecture persisted by
        # training_profiles/setup.py::build_model_config: condition-encoder width,
        # the FM velocity network ('mlp' or 'residual_film') and its depth, the
        # moment head's width / loss weight / scale clamp, and the number of
        # moment-only calibration epochs that follow prior_freeze_epoch.
        "prior_condition_hidden_dim", "prior_fm_velocity_arch", "prior_fm_blocks",
        "prior_fm_moment_hidden_dim", "prior_fm_moment_weight",
        "prior_fm_moment_min_scale", "prior_fm_moment_max_scale",
        "prior_moment_calibration_epochs",
        # Weight on the flow-matching gradient that reaches the ENCODER
        # (0 = legacy detached one-way coupling, 1 = full CVAE rate term).
        "prior_grad_to_encoder",
        "prior_min_std", "prior_mixture_components",
        # Checkpoint-selection metric: 'recon' (posterior validation loss,
        # default) or 'crps' (inference-mirroring learned-prior CRPS).
        "best_by",
    }
)

# Keys from removed variational branches (post-hoc GMM prior, deterministic-z
# auxiliary losses, broadcast unpool, legacy positional encoding). The variational
# runtime has no removed-feature guard, so it silently ignores them.
VAR_REMOVED_KEYS = frozenset(
    {
        "free_bits", "fit_latent_gmm", "gmm_components", "gmm_covariance_type",
        "gmm_reg_covar", "lambda_kl", "lambda_det", "alpha_prior_max",
        "residual_scale", "bipartite_unpool", "positional_encoding",
        # Energy-score generative term, removed from the objective.
        "gamma_es", "es_samples", "es_steps", "es_noise_source", "es_start_epoch",
    }
)


# Renamed on 2026-10-22 (the geometry pair is checked by validate_meshgraphnets).
# Mirror of methods/MeshGraphNets_Variational/general_modules/renamed_keys.py.
VAR_RENAMED_KEYS = {
    "vae_mp_layers": "vae_encoder_mp_layers",
    "num_vae_samples": "inference_samples",
    "vae_batch_size": "inference_batch_size",
    "vae_batch_size_min": "inference_batch_size_min",
    "vae_batch_size_max": "inference_batch_size_max",
    "vae_batch_vram_fraction": "inference_batch_vram_fraction",
}


def validate_variational(ctx: SpecValidationContext) -> None:
    validate_meshgraphnets(ctx)
    check_renamed_keys(ctx, VAR_RENAMED_KEYS)
    values = ctx.values

    for name in sorted(VAR_REMOVED_KEYS.intersection(values)):
        ctx.add(
            "MGNV-REMOVED",
            Severity.WARNING,
            f"{name} was removed from the variational runtime; the native code silently ignores it.",
            field_name=name,
            hint="Delete the line, or check docs/CONFIG_REFERENCE.md for the replacement control.",
            promote_in_strict=True,
        )

    if values.get("use_vae", False) is True:
        for name in ("vae_latent_dim", "vae_encoder_mp_layers", "recon_loss", "vae_graph_aware"):
            if name not in values:
                ctx.add(
                    "MGNV-VAE-DEFAULT",
                    Severity.NOTICE,
                    f"{name} is absent while use_vae=True; the native VAE default will be used.",
                    field_name=name,
                )
        validate_positive_fields(ctx, ("vae_latent_dim", "vae_encoder_mp_layers"), "MGNV-VAE-POSITIVE")

        # Same default as the runtime's resolve_prior_type (training_profiles/setup.py).
        prior_type = str(values.get("prior_type") or "gnn_e2e").lower()
        if prior_type == "gnn_e2e":
            for name in ("prior_family", "prior_mp_layers", "prior_hidden_dim"):
                if name not in values:
                    ctx.add("MGNV-PRIOR-DEFAULT", Severity.NOTICE, f"{name} is absent; the native conditional-prior default will be used.", field_name=name)
            family = str(values.get("prior_family", "fm")).lower()
            if family not in {"fm", "gmm"}:
                ctx.add("MGNV-PRIOR-FAMILY", Severity.ERROR, "prior_family must be 'fm' or 'gmm'.", field_name="prior_family")
            if family == "fm" and "prior_fm_steps" not in values:
                ctx.add("MGNV-FM-DEFAULT", Severity.NOTICE, "prior_fm_steps is absent; the native default of 20 will be used.", field_name="prior_fm_steps")
            if family == "gmm" and "prior_mixture_components" not in values:
                ctx.add(
                    "MGNV-GMM-REC",
                    Severity.WARNING,
                    "prior_mixture_components is not set for a GMM prior; verify the native default is intended.",
                    field_name="prior_mixture_components",
                    promote_in_strict=True,
                )

            solver = str(values.get("prior_fm_solver", "heun")).lower()
            if solver not in {"heun", "euler"}:
                ctx.add(
                    "MGNV-FM-SOLVER",
                    Severity.ERROR,
                    "prior_fm_solver must be 'heun' (2nd-order, default) or 'euler'.",
                    field_name="prior_fm_solver",
                )

            arch = str(values.get("prior_fm_velocity_arch", "mlp")).lower()
            if arch not in {"mlp", "residual_film"}:
                ctx.add(
                    "MGNV-FM-ARCH",
                    Severity.ERROR,
                    "prior_fm_velocity_arch must be 'mlp' (default) or 'residual_film'.",
                    field_name="prior_fm_velocity_arch",
                )

        try:
            grad_enc = float(values.get("prior_grad_to_encoder", 0) or 0)
        except (TypeError, ValueError):
            grad_enc = 0.0
            ctx.add(
                "MGNV-PRIOR-GRAD",
                Severity.ERROR,
                "prior_grad_to_encoder must be a number in [0, 1].",
                field_name="prior_grad_to_encoder",
            )
        # The peak-to-valley term is computed on the decoder output. The
        # model_split pipeline runs the VAE stage before the decoder and returns
        # a zero there, so beta_aux would silently drop out of the objective.
        if str(values.get("parallel_mode", "ddp")).lower().strip() == "model_split":
            try:
                b_aux = float(values.get("beta_aux", 1.0) or 0)
            except (TypeError, ValueError):
                b_aux = 0.0
            if b_aux > 0.0:
                ctx.add(
                    "MGNV-AUX-PIPELINE",
                    Severity.ERROR,
                    "beta_aux weights a peak-to-valley loss on the DECODED field, "
                    "but parallel_mode model_split computes the VAE stage before "
                    "the decoder and cannot evaluate it -- the term would silently "
                    "be zero.",
                    field_name="beta_aux",
                    hint="Use parallel_mode ddp, or set beta_aux 0.",
                )
        # prior_freeze_epoch: an epoch index inside the run, single card only.
        raw_freeze = values.get("prior_freeze_epoch", 0)
        try:
            freeze = int(float(raw_freeze or 0))
        except (TypeError, ValueError):
            freeze = -1
        if freeze < 0:
            ctx.add("MGNV-PRIOR-FREEZE-001", Severity.ERROR,
                    "prior_freeze_epoch must be a non-negative integer (0 = never).",
                    field_name="prior_freeze_epoch")
        elif freeze > 0:
            try:
                epochs = int(float(values.get("training_epochs", 0) or 0))
            except (TypeError, ValueError):
                epochs = 0
            if epochs and freeze >= epochs:
                ctx.add("MGNV-PRIOR-FREEZE-002", Severity.ERROR,
                        f"prior_freeze_epoch {freeze} is not inside the run "
                        f"(training_epochs {epochs}); the prior-only tail would be empty.",
                        field_name="prior_freeze_epoch")
            gpus = values.get("gpu_ids", 0)
            n_gpu = len(gpus) if isinstance(gpus, (list, tuple)) else 1
            if n_gpu > 1:
                ctx.add("MGNV-PRIOR-FREEZE-003", Severity.ERROR,
                        "prior_freeze_epoch is implemented for a single GPU only: the "
                        "prior-only tail rebuilds the optimizer over a parameter subset "
                        "and fits buffers from a full training-set pass, neither of "
                        "which is wired through DDP ranks.",
                        field_name="gpu_ids",
                        hint="one arm per card, or prior_freeze_epoch 0")
        if grad_enc > 0.0:
            try:
                beta_aux = float(values.get("beta_aux", 1.0) or 0)
            except (TypeError, ValueError):
                beta_aux = 0.0
            if beta_aux <= 0.0:
                ctx.add(
                    "MGNV-PRIOR-GRAD-AUX",
                    Severity.WARNING,
                    "prior_grad_to_encoder > 0 pressures the posterior to be "
                    "predictable from the graph alone. MMD does not guard against "
                    "that (a deterministic z = h(g) matches N(0,I) perfectly); "
                    "beta_aux is the I(z;y) floor that does -- it makes the decoded "
                    "field reproduce each realization's own peak-to-valley, which "
                    "a z independent of y cannot do -- and it is 0 here.",
                    field_name="beta_aux",
                    hint="Keep beta_aux > 0 (1.0 is the native default) while the "
                         "encoder-side coupling is open.",
                    promote_in_strict=True,
                )

        z_cond = values.get("z_conditioning")
        if z_cond is not None and str(z_cond).lower() not in {"concat", "adaln"}:
            ctx.add(
                "MGNV-ZCOND",
                Severity.ERROR,
                "z_conditioning must be 'concat' (legacy fuser) or 'adaln' (AdaLN-Zero).",
                field_name="z_conditioning",
            )

        validate_latent_placement(ctx)

    # The native trainers branch only on 'crps' and rank everything else by the
    # reconstruction loss, so 'det' (a HI-MGNflow criterion) was accepted here
    # and then silently trained as 'recon' while the log said "best by det".
    if "best_by" in values and str(values["best_by"]).lower().strip() not in {"recon", "crps"}:
        ctx.add(
            "MGNV-BESTBY",
            Severity.ERROR,
            "best_by must be 'recon' or 'crps' for MeshGraphNets-V.",
            field_name="best_by",
        )

    if ctx.mode == "inference" and values.get("use_vae", False) is True:
        if "inference_samples" not in values:
            ctx.add("MGNV-SAMPLES-DEFAULT", Severity.NOTICE, "inference_samples is absent; the native default of 1 will be used.", field_name="inference_samples")
        elif integer(values["inference_samples"]) is None or integer(values["inference_samples"]) <= 0:
            ctx.add("MGNV-SAMPLES-VALUE", Severity.ERROR, "inference_samples must be a positive integer.", field_name="inference_samples")
        elif (integer(values["inference_samples"]) and integer(values["inference_samples"]) > 1000
                and values.get("save_rollouts", True) is not False):
            # The warning is about FILES, one per (scene, draw). With
            # save_rollouts False none are written, so a large draw count is
            # only compute and this would be a false alarm on every config of a
            # distribution study.
            ctx.add(
                "MGNV-SAMPLES-WORKLOAD",
                Severity.WARNING,
                f"inference_samples={values['inference_samples']} writes that many rollout "
                f"HDF5s per scene. Set save_rollouts False if only the spread "
                f"histogram is wanted.",
                field_name="inference_samples",
            )
        if "prior_temperature" in values and (numeric(values["prior_temperature"]) is None or numeric(values["prior_temperature"]) <= 0):
            ctx.add("MGNV-TEMP-001", Severity.ERROR, "prior_temperature must be > 0.", field_name="prior_temperature")
        if values.get("use_conditional_prior", False) is True:
            ctx.add(
                "MGNV-CKPT-OVERRIDE",
                Severity.NOTICE,
                "The checkpoint model_config may override use_conditional_prior and related inference fields.",
                field_name="use_conditional_prior",
            )

    validate_spread_keys(ctx)


# Mirror of LATENT_TYPES / LATENT_REGS in the runtime's model/latent.py.
LATENT_TYPES = ("global", "local", "hybrid")
LATENT_REGS = ("mmd", "kl")


def validate_latent_placement(ctx: SpecValidationContext) -> None:
    """latent_type / latent_reg and the features only the global latent has.

    local and hybrid put z_l on the coarsest V-cycle level and swap the prior
    for CoarseFMPrior, a GNN flow over that level. It has no GMM family, no
    conditional-moment base, and no residual_film velocity; the runtime raises
    on the first and silently ignores the other two, so all three are errors
    here rather than a run that is not the one configured.
    """
    values = ctx.values
    latent_type = str(values.get("latent_type", "global")).lower().strip()
    if latent_type not in LATENT_TYPES:
        ctx.add("MGNV-LATENT-TYPE", Severity.ERROR,
                "latent_type must be 'global' (per-graph slots), 'local' (per coarsest "
                "node) or 'hybrid' (both).",
                field_name="latent_type")
        return
    reg = str(values.get("latent_reg", "mmd")).lower().strip()
    if reg not in LATENT_REGS:
        ctx.add("MGNV-LATENT-REG", Severity.ERROR,
                "latent_reg must be 'mmd' (weight lambda_mmd) or 'kl' (per-sample KL, "
                "weight kl_weight).",
                field_name="latent_reg")
    if reg == "kl" and "kl_weight" not in values:
        ctx.add("MGNV-LATENT-KL-DEFAULT", Severity.NOTICE,
                "kl_weight is absent; the native default of 1e-3 will be used.",
                field_name="kl_weight")
    if "kl_weight" in values:
        w = numeric(values["kl_weight"])
        if w is None or w < 0:
            ctx.add("MGNV-LATENT-KL-WEIGHT", Severity.ERROR,
                    "kl_weight must be a non-negative number.", field_name="kl_weight")
        elif reg != "kl":
            ctx.add("MGNV-LATENT-KL-WEIGHT", Severity.WARNING,
                    "kl_weight is set but latent_reg is not 'kl'; it is ignored.",
                    field_name="kl_weight", promote_in_strict=True)
    if reg == "kl" and "lambda_mmd" in values:
        ctx.add("MGNV-LATENT-MMD-WEIGHT", Severity.WARNING,
                "lambda_mmd is set but latent_reg is 'kl'; it is ignored (kl_weight "
                "weights the KL term).",
                field_name="lambda_mmd", promote_in_strict=True)

    if latent_type == "global":
        for name in ("latent_ch", "vae_local_mp_layers"):
            if name in values:
                ctx.add("MGNV-LATENT-INACTIVE", Severity.WARNING,
                        f"{name} only applies to latent_type local/hybrid; it is ignored "
                        f"with latent_type global.",
                        field_name=name, promote_in_strict=True)
        return

    validate_positive_fields(ctx, ("latent_ch", "vae_local_mp_layers"), "MGNV-LATENT-POSITIVE")
    if ctx.mode != "train":
        # Inference rebuilds the architecture from the checkpoint model_config.
        return
    why = f"latent_type {latent_type} "
    if values.get("use_multiscale", False) is not True:
        ctx.add("MGNV-LATENT-MULTISCALE", Severity.ERROR,
                why + "places z_l on the coarsest V-cycle level, which needs "
                      "use_multiscale True.",
                field_name="use_multiscale")
    if str(values.get("prior_family", "fm")).lower() != "fm":
        ctx.add("MGNV-LATENT-PRIOR", Severity.ERROR,
                why + "has only the flow-matching coarse-level prior; prior_family must be fm.",
                field_name="prior_family")
    if values.get("prior_fm_moments", False) is True:
        ctx.add("MGNV-LATENT-PRIOR", Severity.ERROR,
                why + "has no conditional-moment base; prior_fm_moments would be ignored.",
                field_name="prior_fm_moments")
    if str(values.get("prior_fm_velocity_arch", "mlp")).lower() != "mlp":
        ctx.add("MGNV-LATENT-PRIOR", Severity.ERROR,
                why + "uses a coarse-level GNN velocity; prior_fm_velocity_arch would be "
                      "ignored. Its depth is prior_fm_blocks.",
                field_name="prior_fm_velocity_arch")
    if (integer(values.get("prior_moment_calibration_epochs", 0)) or 0) > 0:
        ctx.add("MGNV-LATENT-PRIOR", Severity.ERROR,
                why + "has no moment head to calibrate; prior_moment_calibration_epochs "
                      "must be 0.",
                field_name="prior_moment_calibration_epochs")
    if str(values.get("parallel_mode", "ddp")).lower().strip() == "model_split":
        ctx.add("MGNV-LATENT-PIPELINE", Severity.ERROR,
                why + "is not wired through the model_split pipeline (global only).",
                field_name="parallel_mode")


# Mirror of SPREAD_STATS / VECTOR_SPREAD_STATS in both trees'
# inference_profiles/rollout.py (the launcher imports no method code).
SPREAD_STATS = ("range", "mean", "std", "mag_max")
VECTOR_SPREAD_STATS = ("mag_max",)


def validate_spread_keys(ctx: SpecValidationContext) -> None:
    """Shared checks for the spread-histogram keys.

    Not gated on use_vae: the spread comparison in
    `inference_profiles/rollout.py` reads these keys whenever eval_dataset is
    set.
    """
    values = ctx.values
    if ctx.mode == "inference":
        stat = values.get("spread_stat")
        stat = None if stat is None else str(stat).lower()
        if stat is not None and stat not in SPREAD_STATS:
            ctx.add(
                "MGNV-SPREAD-STAT",
                Severity.ERROR,
                "spread_stat must be 'range' (max - min), 'mean', 'std', or 'mag_max' "
                "(max |u| over the spread_channel list).",
                field_name="spread_stat",
            )
        raw = values.get("spread_channel")
        listed = isinstance(raw, (list, tuple))
        channels = ([integer(c) for c in raw] if listed
                    else [integer(raw)] if "spread_channel" in values else [])
        out_var = integer(values.get("output_var"))
        if listed and stat not in VECTOR_SPREAD_STATS:
            # rollout.py raises on this too, but only after the model has loaded.
            ctx.add("MGNV-SPREAD-CHANNEL", Severity.ERROR,
                    f"spread_channel lists {len(raw)} channels, but spread_stat "
                    f"{stat or 'range'} reduces one; only 'mag_max' takes a list.",
                    field_name="spread_channel")
        elif any(c is None or c < 0 for c in channels):
            ctx.add("MGNV-SPREAD-CHANNEL", Severity.ERROR,
                    "spread_channel is an index into the output block and must be a "
                    "non-negative integer (or a list of them for mag_max).",
                    field_name="spread_channel")
        elif channels and out_var is not None and max(channels) >= out_var:
            # Silently out of range means the histogram block is skipped after
            # the whole rollout has run; catching it here costs nothing.
            ctx.add(
                "MGNV-SPREAD-CHANNEL",
                Severity.ERROR,
                f"spread_channel {raw} is outside the output block "
                f"(output_var={out_var}); no spread histogram would be produced.",
                field_name="spread_channel",
            )
        if values.get("eval_dataset") is None and (
                "spread_channel" in values or "spread_stat" in values
                or values.get("make_histogram") is True):
            ctx.add(
                "MGNV-SPREAD-EVAL",
                Severity.WARNING,
                "The spread keys are set but eval_dataset is not, so there is no "
                "ground truth to compare against and no histogram or score is written.",
                field_name="eval_dataset",
            )


def build_variational_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="meshgraphnets_variational",
        display_name="MeshGraphNets Variational",
        model_ids=("meshgraphnets-v",),
        repository="methods/MeshGraphNets_Variational",
        entrypoint="MeshGraphNets_main.py",
        valid_modes=("train", "inference"),
        known_keys=VAR_KEYS | VAR_REMOVED_KEYS | RESUME_KEYS,
        required_by_mode={
            "train": frozenset({"dataset_dir", "modelpath", "input_var", "output_var", "edge_var", "latent_dim", "training_epochs", "batch_size", "learningr"}),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var", "edge_var"}),
        },
        recommended_by_mode={"train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode"})},
        defaults={"parallel_mode": "ddp", "use_vae": False, "use_conditional_prior": False, "use_multiscale": False},
        defaults_by_mode={"inference": {"inference_output_dir": "../../output/meshgraphnets-v/rollout"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            # Scoring-only, but a stale path here is SILENT: rollout.py prints a
            # skip at the very end of a finished run, so the histogram is simply
            # missing after the GPU time is already spent. Validating it as an
            # input file also brings PATH-CASE-001, which catches SAOI vs saoi.
            PathRule("eval_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_variational, validate_resume),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
