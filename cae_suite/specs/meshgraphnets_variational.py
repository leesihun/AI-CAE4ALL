from __future__ import annotations

from ..diagnostics import Severity
from .base import MethodSpec, PathKind, PathRule, SpecValidationContext, integer, numeric, validate_positive_fields
from .meshgraphnets import validate_meshgraphnets


# Keys consumed by the live variational runtime (traced from config[...] /
# config.get(...) reachable from "MeshGraphNets - variational"/MeshGraphNets_main.py).
VAR_KEYS = frozenset(
    {
        "model", "mode", "gpu_ids", "parallel_mode", "log_file_dir", "modelpath",
        "dataset_dir", "infer_dataset", "eval_dataset", "inference_output_dir",
        "infer_timesteps", "split_seed", "input_var", "output_var",
        "feature_loss_weights", "edge_var", "positional_features",
        "message_passing_num", "training_epochs", "batch_size", "learningr",
        "latent_dim", "num_workers", "prefetch_factor", "std_noise", "noise_gamma",
        # Time integration (CONFIGURATION_REFERENCE.md section 3.5): ar_ot or ar_rt.
        "time_integration",
        "noise_std_ratio", "weight_decay", "warmup_epochs", "augment_geometry",
        "grad_accum_steps", "use_checkpointing", "use_amp", "use_ema", "ema_decay",
        "use_compile", "test_interval", "val_interval", "test_max_batches",
        "display_trainset", "display_testset", "use_node_types", "use_world_edges",
        "world_radius_multiplier", "world_max_num_neighbors", "world_edge_backend",
        "coarse_world_edges", "use_parallel_stats", "static_cache_per_worker",
        "hierarchy_cache_dir", "hierarchy_cache_build_workers",
        "hierarchy_cache_wait_timeout", "test_batch_idx", "plot_feature_idx",
        "use_multiscale", "coarsening_type", "voronoi_clusters", "multiscale_levels",
        "mp_per_level", "pipeline_microbatches", "make_histogram", "show_histogram",
        "histogram_bins", "histogram_clip_quantile",
        # VAE / conditional-prior branch
        "use_vae", "vae_latent_dim", "vae_mp_layers", "vae_graph_aware",
        "vae_batch_size", "vae_batch_size_max", "vae_batch_size_min",
        "vae_batch_vram_fraction", "vae_valid_prior_samples", "recon_loss",
        "alpha_recon", "beta_aux", "lambda_mmd", "mmd_bandwidth",
        "posterior_min_std", "num_z", "num_vae_samples", "prior_type",
        "use_conditional_prior", "prior_family", "prior_nll_weight",
        "prior_fm_steps", "prior_mp_layers", "prior_hidden_dim",
        "prior_temperature", "prior_kl_reg_weight", "prior_cov_rank",
        "prior_min_std", "prior_mixture_components",
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

    if ctx.mode == "inference" and values.get("use_vae", False) is True:
        if "num_vae_samples" not in values:
            ctx.add("MGNV-SAMPLES-DEFAULT", Severity.NOTICE, "num_vae_samples is absent; the native default of 1 will be used.", field_name="num_vae_samples")
        elif integer(values["num_vae_samples"]) is None or integer(values["num_vae_samples"]) <= 0:
            ctx.add("MGNV-SAMPLES-VALUE", Severity.ERROR, "num_vae_samples must be a positive integer.", field_name="num_vae_samples")
        elif integer(values["num_vae_samples"]) and integer(values["num_vae_samples"]) > 1000:
            ctx.add(
                "MGNV-SAMPLES-WORKLOAD",
                Severity.WARNING,
                f"num_vae_samples={values['num_vae_samples']} can produce a large number of rollout artifacts.",
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


def build_variational_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="meshgraphnets_variational",
        display_name="MeshGraphNets Variational",
        model_ids=("meshgraphnets-v",),
        repository="MeshGraphNets - variational",
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
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_variational,),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
