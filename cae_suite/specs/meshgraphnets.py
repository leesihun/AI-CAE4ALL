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


# Keys consumed by the live deterministic runtime (traced from config[...] /
# config.get(...) reachable from MeshGraphNets_main.py).
MGN_KEYS = frozenset(
    {
        "model", "mode", "gpu_ids", "parallel_mode", "log_file_dir", "modelpath",
        "dataset_dir", "infer_dataset", "inference_output_dir", "infer_timesteps",
        "split_seed", "input_var", "output_var", "feature_loss_weights", "edge_var",
        "positional_features", "message_passing_num", "training_epochs", "batch_size",
        "learningr", "latent_dim", "num_workers", "prefetch_factor", "std_noise",
        "noise_gamma", "noise_std_ratio", "weight_decay", "warmup_epochs",
        "augment_geometry", "grad_accum_steps", "use_checkpointing",
        "use_amp", "use_ema", "ema_decay", "use_compile", "test_interval",
        "val_interval", "test_max_batches", "train_eval_subset_size",
        "display_trainset", "display_testset", "use_node_types", "use_world_edges",
        "world_radius_multiplier", "world_max_num_neighbors", "world_edge_backend",
        "coarse_world_edges", "coarse_cache_per_worker", "use_parallel_stats",
        "test_batch_idx", "plot_feature_idx", "use_multiscale", "coarsening_type",
        "voronoi_clusters", "multiscale_levels", "mp_per_level", "profile_batches",
        "pipeline_microbatches",
        # Time integration (CONFIGURATION_REFERENCE.md section 3.5): ar_ot or ar_rt.
        "time_integration",
    }
)

# Exact mirror of MeshGraphNets/general_modules/removed_feature_guard.py
# REMOVED_CONFIG_KEYS: the native loader raises on these before any HDF5 access.
MGN_NATIVE_REMOVED_KEYS = frozenset(
    {
        "use_vae", "vae_latent_dim", "vae_mp_layers", "vae_graph_aware", "free_bits",
        "posterior_min_std", "lambda_mmd", "lambda_kl", "lambda_det", "alpha_recon",
        "beta_aux", "num_vae_samples", "vae_valid_prior_samples", "fit_latent_gmm",
        "gmm_components", "gmm_covariance_type", "gmm_reg_covar",
        "train_conditional_prior", "use_conditional_prior", "prior_temperature",
        "prior_mixture_components", "prior_hidden_dim", "prior_mp_layers",
        "prior_min_std", "prior_loss_type", "prior_epochs", "prior_learningr",
        "prior_batch_size", "prior_num_workers", "prior_val_interval",
        "prior_diagnose_interval", "prior_mc_samples", "resume_prior", "num_z",
    }
)

# Variational-runtime keys the deterministic guard does not reject; the native
# deterministic runtime silently ignores them, so their presence is a strong
# wrong-repository signal but not a native hard failure.
MGN_VARIATIONAL_IGNORED_KEYS = frozenset(
    {
        "recon_loss", "mmd_bandwidth", "prior_type", "prior_family",
        "prior_nll_weight", "prior_fm_steps", "prior_kl_reg_weight",
        "prior_cov_rank", "vae_batch_size", "vae_batch_size_max",
        "vae_batch_size_min", "vae_batch_vram_fraction", "eval_dataset",
        "hierarchy_cache_dir", "hierarchy_cache_build_workers",
        "hierarchy_cache_wait_timeout", "static_cache_per_worker",
        "make_histogram", "show_histogram", "histogram_bins",
        "histogram_clip_quantile",
    }
)


def validate_meshgraphnets(ctx: SpecValidationContext) -> None:
    validate_common_values(ctx)
    values = ctx.values
    if "edge_var" in values and integer(values["edge_var"]) != 8:
        ctx.add(
            "MGN-EDGE-001",
            Severity.ERROR,
            f"edge_var must be 8 for MeshGraphNets; got {values['edge_var']!r}.",
            field_name="edge_var",
        )

    use_multiscale = values.get("use_multiscale", False) is True
    if use_multiscale:
        for name in ("coarsening_type", "multiscale_levels", "voronoi_clusters", "mp_per_level"):
            if name not in values:
                ctx.add(
                    "MGN-MULTI-REQ",
                    Severity.ERROR,
                    f"{name} is required when use_multiscale=True.",
                    field_name=name,
                )
        coarsening_modes = {"bfs", "voronoi_centroid", "voronoi_inherit", "voronoi_seedmean"}
        for entry in as_list(values.get("coarsening_type", [])):
            entry_name = str(entry).lower()
            if entry_name == "voronoi":
                ctx.add(
                    "MGN-COARSEN-001",
                    Severity.ERROR,
                    "The bare coarsening_type 'voronoi' alias was removed; the native hierarchy build raises on it.",
                    field_name="coarsening_type",
                    hint="Use voronoi_centroid (the direct successor), voronoi_inherit, or voronoi_seedmean.",
                )
            elif entry_name not in coarsening_modes:
                ctx.add(
                    "MGN-COARSEN-002",
                    Severity.ERROR,
                    f"Unknown coarsening_type entry {entry_name!r}.",
                    field_name="coarsening_type",
                    hint="Valid per-level modes: bfs, voronoi_centroid, voronoi_inherit, voronoi_seedmean.",
                )
        levels = integer(values.get("multiscale_levels"))
        mp = values.get("mp_per_level")
        clusters = values.get("voronoi_clusters")
        if levels is not None and levels <= 0:
            ctx.add("MGN-MULTI-LEVEL", Severity.ERROR, "multiscale_levels must be > 0.", field_name="multiscale_levels")
        if levels is not None and mp is not None:
            length = len(as_list(mp))
            expected = 2 * levels + 1
            if length != expected:
                ctx.add(
                    "MGN-MULTI-001",
                    Severity.ERROR,
                    f"mp_per_level must contain {expected} entries for multiscale_levels={levels}; found {length}.",
                    field_name="mp_per_level",
                )
        if levels is not None and clusters is not None and len(as_list(clusters)) not in {1, levels}:
            ctx.add(
                "MGN-MULTI-002",
                Severity.ERROR,
                f"voronoi_clusters must contain one reusable value or {levels} per-level entries; found {len(as_list(clusters))}.",
                field_name="voronoi_clusters",
            )
    elif "message_passing_num" not in values:
        ctx.add(
            "MGN-MP-REQ",
            Severity.ERROR,
            "message_passing_num is required when use_multiscale is false or absent.",
            field_name="message_passing_num",
        )

    validate_positive_fields(ctx, ("message_passing_num",), "MGN-POSITIVE-001")

    parallel_mode = str(values.get("parallel_mode", "ddp")).lower()
    if parallel_mode not in {"ddp", "model_split"}:
        ctx.add(
            "MGN-PARALLEL-001",
            Severity.ERROR,
            "parallel_mode must be 'ddp' or 'model_split'.",
            field_name="parallel_mode",
        )
    if parallel_mode == "model_split":
        gpu_ids = as_list(values.get("gpu_ids", []))
        if len(gpu_ids) < 2:
            ctx.add(
                "MGN-PARALLEL-002",
                Severity.ERROR,
                "parallel_mode=model_split requires at least two gpu_ids.",
                field_name="gpu_ids",
            )

    if ctx.model_id == "meshgraphnets":
        for name in sorted(MGN_NATIVE_REMOVED_KEYS.intersection(values)):
            ctx.add(
                "MGN-REMOVED-VAR",
                Severity.ERROR,
                f"{name} is rejected by the deterministic runtime's removed-feature guard. Use model meshgraphnets-v for the variational runtime.",
                field_name=name,
            )
        for name in sorted(MGN_VARIATIONAL_IGNORED_KEYS.intersection(values)):
            ctx.add(
                "MGN-VARIATIONAL-IGNORED",
                Severity.WARNING,
                f"{name} configures the variational runtime; the deterministic runtime silently ignores it. Use model meshgraphnets-v if the variational behavior is intended.",
                field_name=name,
                promote_in_strict=True,
            )


def build_meshgraphnets_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="meshgraphnets",
        display_name="MeshGraphNets",
        model_ids=("meshgraphnets",),
        repository="MeshGraphNets",
        entrypoint="MeshGraphNets_main.py",
        valid_modes=("train", "inference"),
        known_keys=MGN_KEYS | MGN_NATIVE_REMOVED_KEYS | MGN_VARIATIONAL_IGNORED_KEYS,
        required_by_mode={
            "train": frozenset({"dataset_dir", "modelpath", "input_var", "output_var", "edge_var", "latent_dim", "training_epochs", "batch_size", "learningr"}),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var", "edge_var"}),
        },
        recommended_by_mode={"train": frozenset({"feature_loss_weights", "split_seed", "parallel_mode"})},
        defaults={"parallel_mode": "ddp", "use_multiscale": False, "use_world_edges": False, "use_checkpointing": False, "use_ema": False},
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/rollout"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_meshgraphnets,),
        import_modules=("torch", "h5py", "torch_geometric"),
        dataset_kind="mesh_hdf5",
    )
