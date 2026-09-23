from __future__ import annotations

from ..diagnostics import Severity
from .base import MethodSpec, PathKind, PathRule, SpecValidationContext, integer, numeric, validate_positive_fields
from .meshgraphnets import validate_meshgraphnets


# Keys consumed by the live variational runtime (traced from config[...] /
# config.get(...) reachable from methods/MeshGraphNets_Variational/MeshGraphNets_main.py).
VAR_KEYS = frozenset(
    {
        "geometry_state_mode", "displacement_state_indices", "periodic_box", "write_preprocessing",
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
        # noise.
        "spread_channel", "spread_label", "spread_stat",
        # VAE / conditional-prior branch
        "use_vae", "vae_latent_dim", "vae_mp_layers", "vae_graph_aware",
        # How z reaches the processor: 'concat' (legacy Linear([x, z]) fuser) or
        # 'adaln' (AdaLN-Zero modulation, identity at init). It changes the
        # parameter names, so a checkpoint is tied to the value it trained with.
        "z_conditioning",
        # All-gather z across DDP ranks before MMD so the two-sample estimator
        # sees the GLOBAL batch instead of the per-rank one (default True).
        "mmd_gather_ranks",
        "vae_batch_size", "vae_batch_size_max", "vae_batch_size_min",
        "vae_batch_vram_fraction", "vae_valid_prior_samples", "recon_loss",
        "alpha_recon", "beta_aux", "pv_channel", "lambda_mmd", "mmd_bandwidth",
        "posterior_min_std", "num_z", "num_vae_samples", "prior_type",
        "use_conditional_prior", "prior_family", "prior_nll_weight",
        "prior_fm_steps", "prior_fm_solver", "prior_mp_layers", "prior_hidden_dim",
        "prior_temperature", "prior_kl_reg_weight", "prior_cov_rank",
        # FM velocity-MLP width (defaults to prior_hidden_dim) and the
        # condition-dependent mean/scale head of the FM base; both persist in
        # the checkpoint model_config.
        "prior_velocity_hidden_dim", "prior_fm_moments",
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


def validate_variational(ctx: SpecValidationContext) -> None:
    validate_meshgraphnets(ctx)
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
        for name in ("vae_latent_dim", "vae_mp_layers", "recon_loss", "vae_graph_aware"):
            if name not in values:
                ctx.add(
                    "MGNV-VAE-DEFAULT",
                    Severity.NOTICE,
                    f"{name} is absent while use_vae=True; the native VAE default will be used.",
                    field_name=name,
                )
        validate_positive_fields(ctx, ("vae_latent_dim", "vae_mp_layers"), "MGNV-VAE-POSITIVE")

        prior_type = str(values.get("prior_type", "none")).lower()
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

    if ctx.mode == "inference" and values.get("use_vae", False) is True:
        if "num_vae_samples" not in values:
            ctx.add("MGNV-SAMPLES-DEFAULT", Severity.NOTICE, "num_vae_samples is absent; the native default of 1 will be used.", field_name="num_vae_samples")
        elif integer(values["num_vae_samples"]) is None or integer(values["num_vae_samples"]) <= 0:
            ctx.add("MGNV-SAMPLES-VALUE", Severity.ERROR, "num_vae_samples must be a positive integer.", field_name="num_vae_samples")
        elif (integer(values["num_vae_samples"]) and integer(values["num_vae_samples"]) > 1000
                and values.get("save_rollouts", True) is not False):
            # The warning is about FILES, one per (scene, draw). With
            # save_rollouts False none are written, so a large draw count is
            # only compute and this would be a false alarm on every config of a
            # distribution study.
            ctx.add(
                "MGNV-SAMPLES-WORKLOAD",
                Severity.WARNING,
                f"num_vae_samples={values['num_vae_samples']} writes that many rollout "
                f"HDF5s per scene. Set save_rollouts False if only the spread "
                f"histogram is wanted.",
                field_name="num_vae_samples",
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


def validate_spread_keys(ctx: SpecValidationContext) -> None:
    """Shared checks for the spread-histogram keys.

    Called by both this spec and chi_mgnflow: the two routes run the same
    `inference_profiles/rollout.py` comparison and read the same four keys, so
    a check that lived in only one of them would pass a broken config on the
    other. Not gated on use_vae -- chi-mgnflow sets that in the runtime rather
    than the config.
    """
    values = ctx.values
    if ctx.mode == "inference":
        stat = values.get("spread_stat")
        if stat is not None and str(stat).lower() not in {"range", "mean", "std"}:
            ctx.add(
                "MGNV-SPREAD-STAT",
                Severity.ERROR,
                "spread_stat must be 'range' (max - min), 'mean' or 'std'.",
                field_name="spread_stat",
            )
        channel = integer(values.get("spread_channel")) if "spread_channel" in values else None
        out_var = integer(values.get("output_var"))
        if channel is not None and channel < 0:
            ctx.add("MGNV-SPREAD-CHANNEL", Severity.ERROR,
                    "spread_channel is an index into the output block and cannot be negative.",
                    field_name="spread_channel")
        elif channel is not None and out_var is not None and channel >= out_var:
            # Silently out of range means the histogram block is skipped after
            # the whole rollout has run; catching it here costs nothing.
            ctx.add(
                "MGNV-SPREAD-CHANNEL",
                Severity.ERROR,
                f"spread_channel {channel} is outside the output block "
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
        known_keys=VAR_KEYS | VAR_REMOVED_KEYS,
        required_by_mode={
            "train": frozenset({"dataset_dir", "modelpath", "input_var", "output_var", "edge_var", "latent_dim", "training_epochs", "batch_size", "learningr"}),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var", "edge_var"}),
        },
        recommended_by_mode={"train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode"})},
        defaults={"parallel_mode": "ddp", "use_vae": False, "use_conditional_prior": False, "use_multiscale": False},
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/rollout"}},
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
        validators=(validate_variational,),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
