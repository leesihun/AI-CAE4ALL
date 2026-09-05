import { keys } from "./text.js";

export const ICONS = {
  cad: "⌑",
  data: "▤",
  parameters: "☷",
  model: "◇",
  prepare: "◈",
  run: "▶",
  optimize: "▥",
  evaluate: "≋",
  output: "⇩",
  docs: "▧",
  system: "◉",
  artifact: "▣"
};

export const TYPE_META = {
  geometry: { color: "#4c7f71", label: "geometry" },
  dataset: { color: "#3c7193", label: "HDF5 dataset" },
  parameters: { color: "#b0713f", label: "design parameters" },
  checkpoint: { color: "#795991", label: "saved model" },
  field: { color: "#b6534d", label: "field / response" },
  candidates: { color: "#708a38", label: "CAD candidates" },
  metrics: { color: "#1f7c66", label: "metrics" },
  report: { color: "#7a6946", label: "report" },
  artifact: { color: "#68736e", label: "artifact" },
  endpoint: { color: "#4d6874", label: "endpoint" }
};

export const KEY_CATALOGS = {
  mlp: keys(`activation batch_size checkpoint_interval dataset_dir dropout ema_decay gpu_ids hidden_layers infer_dataset inference_output_dir input_normalization input_var learningr log_file_dir loss max_grad_norm mode model modelpath norm num_workers output_activation output_normalization output_var prefetch_factor split_seed training_epochs use_amp use_compile use_ema val_interval warmup_epochs weight_decay`),
  meshgraphnets: keys(`alpha_recon augment_geometry batch_size beta_aux coarse_cache_per_worker coarse_world_edges coarsening_type cond_var dataset_dir display_testset display_trainset edge_var ema_decay eval_dataset feature_loss_weights fit_latent_gmm free_bits gmm_components gmm_covariance_type gmm_reg_covar gpu_ids grad_accum_steps hierarchy_cache_build_workers hierarchy_cache_dir hierarchy_cache_keep hierarchy_cache_wait_timeout histogram_bins histogram_clip_quantile infer_dataset infer_timesteps inference_output_dir init_modelpath input_var lambda_det lambda_kl lambda_mmd latent_dim learned_interpolation learningr log_file_dir make_histogram message_passing_num mmd_bandwidth mode model modelpath mp_per_level multiscale_levels noise_gamma noise_std_ratio num_vae_samples num_workers num_z output_var parallel_mode pin_memory pipeline_microbatches plot_feature_idx pool_heads pool_type positional_features posterior_min_std prefetch_factor prior_batch_size prior_cov_rank prior_diagnose_interval prior_epochs prior_family prior_fm_steps prior_hidden_dim prior_kl_reg_weight prior_learningr prior_loss_type prior_mc_samples prior_min_std prior_mixture_components prior_mp_layers prior_nll_weight prior_num_workers prior_temperature prior_type prior_val_interval profile_batches recon_loss resume_prior show_histogram split_seed static_cache_per_worker std_noise test_batch_idx test_interval test_max_batches time_integration train_conditional_prior train_eval_subset_size training_epochs unpool_type use_amp use_checkpointing use_compile use_conditional_prior use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_batch_size vae_batch_size_max vae_batch_size_min vae_batch_vram_fraction vae_graph_aware vae_latent_dim vae_mp_layers vae_valid_prior_samples val_interval voronoi_branches voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier`),
  meshgraphnetsV: keys(`alpha_prior_max alpha_recon augment_geometry batch_size best_by beta_aux bipartite_unpool coarse_world_edges coarsening_type cond_var dataset_dir display_testset display_trainset edge_var ema_decay eval_dataset feature_loss_weights fit_latent_gmm free_bits gmm_components gmm_covariance_type gmm_reg_covar gpu_ids grad_accum_steps hierarchy_cache_build_workers hierarchy_cache_dir hierarchy_cache_keep hierarchy_cache_wait_timeout hierarchy_seed hierarchy_variants histogram_bins histogram_clip_quantile infer_dataset infer_timesteps inference_output_dir input_var lambda_det lambda_kl lambda_mmd latent_dim learningr log_file_dir make_histogram message_passing_num mmd_bandwidth mmd_gather_ranks mode model modelpath mp_per_level multiscale_levels noise_gamma noise_std_ratio num_vae_samples num_workers num_z output_var parallel_mode pipeline_microbatches plot_feature_idx positional_encoding positional_features posterior_min_std prefetch_factor prior_cov_rank prior_family prior_fm_solver prior_fm_steps prior_grad_to_encoder prior_hidden_dim prior_kl_reg_weight prior_min_std prior_mixture_components prior_mp_layers prior_nll_weight prior_temperature prior_type recon_loss residual_scale save_rollouts show_histogram split_seed static_cache_per_worker std_noise test_batch_idx test_interval test_max_batches time_integration training_epochs use_amp use_checkpointing use_compile use_conditional_prior use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_batch_size vae_batch_size_max vae_batch_size_min vae_batch_vram_fraction vae_graph_aware vae_latent_dim vae_mp_layers vae_valid_prior_samples val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier z_conditioning`),
  operator: keys(`_ddp_port _pin_memory augment_geometry batch_size checkpoint_interval coarse_cache_per_worker coarse_world_edges coarsening_type cond_var coordinate_normalization dataset_dir deeponet_activation deeponet_basis_dim deeponet_branch_depth deeponet_branch_source deeponet_hidden_channels deeponet_max_branch_params deeponet_multi_output deeponet_sensor_resolution deeponet_trunk_depth dimension_tolerance display_testset display_trainset edge_var ema_decay feature_loss_weights fno_grid_resolution fno_hidden_channels fno_layers fno_modes fno_norm fno_use_channel_mlp fno_variant free_bits gino_cache_neighbors gino_channel_mlp_expansion gino_coord_embed_dim gino_fno_hidden_channels gino_fno_layers gino_fno_modes gino_grid_resolution gino_group_shared_geometry gino_in_radius gino_include_grid_coordinates gino_kernel_hidden gino_kernel_widths gino_lifting_hidden gino_max_empty_input_fraction gino_max_num_neighbors gino_out_radius gino_pos_embedding_type gino_projection_widths gino_query_chunk_size gino_tucker_rank gino_use_torch_cluster gino_variant global_condition_features gpu_ids grad_accum_steps grid_padding infer_dataset infer_query_chunk_size infer_timesteps inference_output_dir input_var integration_weight_source lambda_det lambda_kl lambda_mmd latent_dim learningr log_dir log_file_dir max_grad_norm message_passing_num mode model modelpath mp_per_level multiscale_levels noise_gamma noise_std_ratio num_node_types num_timesteps num_workers operator_dim out_of_bounds_policy output_var parallel_mode pipeline_microbatches plot_feature_idx point_branch_merge point_condition_depth point_feature_dim point_hidden_channels point_output_activation point_refiner_depth point_resample_each_epoch point_sampling point_sensor_count point_siren_omega0 point_trunk_depth point_variant pointnet_activation pointnet_depth pointnet_norm positional_features posterior_min_std prefetch_factor profile_batches sdf_sidecar sdf_source split_seed std_noise test_batch_idx test_interval test_max_batches time_integration train_eval_subset_size train_query_chunk_size training_epochs use_amp use_checkpointing use_compile use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_graph_aware vae_latent_dim vae_mp_layers val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier write_preprocessing`),
  transolver: keys(`amortized_cache_nodes amortized_query_nodes amortized_training attention_kernel augment_geometry batch_size chunk_size coarse_world_edges coarsening_type cond_var coordinate_normalization dataset_dir display_trainset dropout edge_var ema_decay feature_loss_weights gpu_ids grad_accum_steps infer_chunk_size infer_dataset infer_mode infer_timesteps inference_output_dir input_var latent_dim learningr log_file_dir max_grad_norm max_train_batches max_val_batches message_passing_num mlp_ratio mode model modelpath mp_per_level multiscale_levels noise_gamma num_heads num_layers num_workers output_var parallel_mode positional_features prefetch_factor slice_num small_output_init split_seed std_noise temperature_init temperature_max temperature_min test_batch_idx test_interval test_max_batches time_integration training_epochs use_amp use_checkpointing use_compile use_ema use_multiscale use_node_types use_parallel_stats use_world_edges val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier write_preprocessing write_test_predictions`),
  sdfflow: keys(`alpha batch_size candidate_multiplier cfg_scale clamp_dist cond_dropout cond_values condition_clip condition_names condition_ood_policy dataset_dir decoder_heads decoder_hidden decoder_layers decoder_type deterministic_warmup_epochs eikonal_weight ema_decay encode_batch_size encoder_blocks encoder_dim encoder_heads encoder_self_attention fm_arch fm_batch_size fm_blocks fm_cond_hidden fm_ema_decay fm_heads fm_hidden fm_learningr fm_log_file_dir fm_mc_resolution_test fm_modelpath fm_num_test_shapes fm_num_workers fm_test_interval fm_time_logit_mean fm_time_logit_std fm_time_sampling fm_training_epochs fm_use_amp fm_use_ema fm_val_interval fm_warmup_epochs fm_weight_decay fourier_bands fsdp_min_params gpu_ids hybrid_grad_points init_vae_modelpath input_mesh kl_warmup_epochs kl_weight latent_clip latent_dim latent_tokens learningr log_file_dir max_condition_z mc_resolution mc_resolution_test min_condition_std mode model normal_weight num_encoder_points num_query_points num_samples num_test_shapes num_workers ode_steps output_dir overfit_all_shapes overfit_num_shapes parallel_mode pipeline_log_file plot_dpi plot_max_faces posterior_noise_max_scale posterior_noise_warmup_epochs sample_index_a sample_index_b seed skip_completed_stages source_num_samples split_seed surface_weight test_interval training_epochs use_amp use_conditions use_ema vae_batch_size vae_ema_decay vae_learningr vae_log_file_dir vae_mc_resolution_test vae_modelpath vae_num_test_shapes vae_num_workers vae_test_interval vae_training_epochs vae_use_amp vae_use_ema vae_val_interval vae_warmup_epochs vae_weight_decay val_interval warmup_epochs weight_decay opt_subspace_dim opt_subspace_seed opt_condition_dims opt_latent_range opt_shell_scale opt_budget opt_popsize opt_sigma0 opt_baseline_size opt_load_cases opt_length_scale opt_stress_percentile opt_mesh_size_max opt_target_faces opt_material_e opt_material_nu opt_material_rho opt_yield_stress opt_stress_margin opt_disp_margin opt_stress_weight opt_disp_weight opt_verify_resolution opt_verify_target_faces opt_verify_mesh_size_max opt_analysis opt_surrogate_checkpoint opt_surrogate_config opt_surrogate_target_nodes`),
  simulgenvae: keys(`alpha batch_size beta_target cond_var dataset_dir ema_decay field_start_row gpu_ids init_beta_divisor init_vae_modelpath kl_warmup_epochs kl_warmup_start_frac latent_dim latent_dim_end lc_batch_size lc_data_type lc_dropout lc_ema_decay lc_filter lc_learningr lc_log_file_dir lc_modelpath lc_num_workers lc_training_epochs lc_use_amp lc_use_ema lc_warmup_epochs lc_weight_decay learningr load_all log_file_dir loss_type mode model network_size node_end node_start num_filter_enc num_var num_workers output_dir parallel_mode param_data_type param_dir pipeline_log_file plot_mode recon_iter skip_completed_stages split_seed timesteps_reduced training_epochs use_amp use_ema use_spatial_attention vae_batch_size vae_ema_decay vae_learningr vae_log_file_dir vae_modelpath vae_num_workers vae_training_epochs vae_use_amp vae_use_ema vae_warmup_epochs vae_weight_decay val_interval warmup_epochs weight_decay`)
};

// Native-only controls discovered after the original compact catalogs were
// published. Keep them explicit here so Studio authoring and MethodSpecs stay
// aligned without hiding them in an unrelated model family's long key line.
KEY_CATALOGS.operator = [...new Set([...KEY_CATALOGS.operator, "gino_domain_padding"])].sort();
export const SIMULGEN_REMOVED_NOOPS = new Set(["load_all", "plot_mode", "recon_iter"]);
KEY_CATALOGS.simulgenvae = [...new Set([...KEY_CATALOGS.simulgenvae, "fsdp_min_params"])]
  .sort();
KEY_CATALOGS.meshgraphnetsV = [...new Set([
  ...KEY_CATALOGS.meshgraphnetsV,
  ...keys(`gamma_es es_samples es_steps es_noise_source es_start_epoch`)
])].sort();
// SDFFlow v3 recipe + `evaluate` mode (cae_suite/specs/sdfflow.py SDFFLOW_KEYS):
// FPS encoder queries, relative posterior-std floor, parent-grouped split,
// best-val VAE checkpoint, decoder-frozen latent refinement, held-out
// evaluation controls, and the interpolation-space switch.
KEY_CATALOGS.sdfflow = [...new Set([
  ...KEY_CATALOGS.sdfflow,
  "opt_seed",
  ...keys(`encoder_query_type posterior_min_std_rel split_by_parent vae_best_modelpath latent_refine_steps latent_refine_lr latent_refine_prior_weight eval_split eval_num_shapes eval_seed interpolation_space`),
  // Conditional generation (docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md):
  // per-dimension condition dropout, sample-time descriptor guidance (C2) and
  // Newton correction (E2) with their shared calibration artifact, the FEA
  // condition audit, the fixed-noise condition sweep, and the evaluate tasks.
  ...keys(`cond_dropout_mode guidance_enabled guidance_t_start guidance_eta guidance_step_mode guidance_targets soft_descriptor_resolution soft_descriptor_tau descriptor_calibration_path newton_rounds newton_step_cap_rms newton_line_search_tries newton_measure_resolution condition_audit cond_values_a cond_values_b sweep_steps eval_task eval_methods calibration_num_shapes calibration_samples_per_shape cond_dropout_all_prob calibration_min_r2 eval_exclude_shapes`)
])].sort();

// cHI-MGNflow deliberately shares the complete MeshGraphNets-V diagnostic
// surface. The backend keeps stale latent/prior keys in `known_keys` so strict
// preflight can explain that they are rejected instead of calling them unknown;
// the flow controls are the only model-specific additions.
KEY_CATALOGS.chiMgnflow = [...new Set([
  ...KEY_CATALOGS.meshgraphnetsV,
  ...keys(`flow_steps flow_solver flow_time_freqs flow_t_sampling flow_t_logit_scale flow_loss_weighting flow_det_prob flow_predict val_flow_steps val_num_samples gamma_es es_samples es_steps es_noise_source es_start_epoch`)
])].sort();

export const MODEL_CATALOG = {
  simulgenvae: {
    label: "SimulGen-VAE", short: "SGV", accent: "#a45b6e", modes: ["train", "train_vae", "train_lc", "reconstruct"],
    keys: KEY_CATALOGS.simulgenvae,
    description: "Hierarchical VAE plus a latent conditioner (CSV, condition image, or the dataset's own cond_var rows) for parametric transient or static simulation fields.",
    dataset: "fixed-geometry mesh HDF5",
    defaults: {
      model: "simulgenvae", mode: "train", gpu_ids: "0", parallel_mode: "single",
      // ex9, not ex1: SimulGen-VAE needs ONE fixed (T, N) for every sample and
      // ex1.h5 holds 100 different shapes, so this default could never train
      // (STUDIO-SGV-FIXED-001). num_var 2 = ux,uy per
      // configs/SimulGenVAE/ex9/config_train_vae.txt; cond_var 2 + lc_data_type
      // hdf5 reads the conditioner from ex9's own trailing input-only rows,
      // which is why no param_dir is set -- the ex1_conditions.csv this used to
      // point at does not exist on disk (STUDIO-SGV-PARAM-001).
      dataset_dir: "../../dataset/ex9.h5", split_seed: "42", output_dir: "../../output/simulgenvae/studio",
      vae_modelpath: "../../output/simulgenvae/studio/simulgenvae_vae.pth",
      lc_modelpath: "../../output/simulgenvae/studio/simulgenvae_lc.pth",
      num_var: "2", cond_var: "2", field_start_row: "3", node_start: "0", node_end: "0", timesteps_reduced: "0",
      latent_dim: "8", latent_dim_end: "32", num_filter_enc: "1024 512 256 128",
      network_size: "small", loss_type: "1", alpha: "1000000", init_beta_divisor: "4",
      vae_training_epochs: "2000", vae_batch_size: "16", vae_learningr: "0.001",
      lc_filter: "32 64 128 256 512 1024", lc_data_type: "hdf5",
      lc_dropout: "0.2", use_spatial_attention: "1",
      lc_training_epochs: "5000", lc_batch_size: "64", lc_learningr: "0.001",
      // train_vae / train_lc use the generic trio rather than the vae_*/lc_*
      // variants, so without these a block switched to either mode reported three
      // missing required keys. The stage presets still override them per stage.
      training_epochs: "2000", batch_size: "16", learningr: "0.001",
      skip_completed_stages: "True"
    }
  },
  mlp: {
    label: "Simple MLP", short: "MLP", accent: "#a66a45", modes: ["train", "inference"], keys: KEY_CATALOGS.mlp,
    description: "Tabular X[S,N] to Y[S,M] regression for global engineering responses.", dataset: "table HDF5",
    defaults: { model: "mlp", mode: "train", gpu_ids: "0", modelpath: "../../output/mlp/studio/mlp.pth", hidden_layers: "256,256,128", activation: "gelu", training_epochs: "200", batch_size: "32", learningr: "0.001" }
  },
  meshgraphnets: {
    label: "MeshGraphNets", short: "MGN", accent: "#19715e", modes: ["train", "inference"], keys: KEY_CATALOGS.meshgraphnets,
    description: "Deterministic graph-network simulator with Flat, HI-MGN, and BSMS-GNN presets.", dataset: "mesh HDF5",
    // edge_var is required and MGN-EDGE-001 rejects anything but 8, so leaving it
    // unset only ever produced a required-field error the user had to guess at.
    defaults: { model: "meshgraphnets", mode: "train", gpu_ids: "0", modelpath: "../../output/meshgraphnets/studio/meshgraphnets.pth", message_passing_num: "15", latent_dim: "128", edge_var: "8", use_multiscale: "False", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  "meshgraphnets-v": {
    label: "MeshGraphNets-V", short: "MGN-V", accent: "#407d69", modes: ["train", "inference"], keys: KEY_CATALOGS.meshgraphnetsV,
    description: "Variational mesh simulator with conditional priors and stochastic trajectories.", dataset: "mesh HDF5",
    defaults: { model: "meshgraphnets-v", mode: "train", gpu_ids: "0", modelpath: "../../output/meshgraphnets_v/studio/meshgraphnets_v.pth", latent_dim: "128", edge_var: "8", vae_latent_dim: "256", num_vae_samples: "32", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  "chi-mgnflow": {
    label: "cHI-MGNflow", short: "cHI-FM", accent: "#286d7a", modes: ["train", "inference"], keys: KEY_CATALOGS.chiMgnflow,
    description: "Conditional hierarchical MeshGraphNet with flow-matching field generation and deterministic or ensemble readout.", dataset: "mesh HDF5",
    defaults: {
      model: "chi-mgnflow", mode: "train", gpu_ids: "0", parallel_mode: "ddp",
      modelpath: "../../output/chi-mgnflow/studio/chi_mgnflow.pth",
      latent_dim: "128", edge_var: "8", use_multiscale: "True",
      message_passing_num: "15", coarsening_type: "voronoi_seedmean",
      multiscale_levels: "2", voronoi_clusters: "500,100", mp_per_level: "3,4,6,4,3",
      hierarchy_variants: "1", hierarchy_seed: "1234",
      flow_steps: "30", flow_solver: "heun", flow_time_freqs: "16",
      flow_t_sampling: "uniform", flow_loss_weighting: "uniform", flow_det_prob: "0",
      flow_predict: "sample", val_flow_steps: "10", val_num_samples: "8", best_by: "crps",
      training_epochs: "500", batch_size: "4", learningr: "0.0001"
    }
  },
  point_deeponet: {
    label: "Point-DeepONet", short: "P-DON", accent: "#3d718d", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Point-conditioned operator for arbitrary-query field prediction.", dataset: "mesh HDF5",
    // point_feature_dim / pointnet_depth / point_trunk_depth were missing here --
    // NOVAR-REQ rejects a Point-DeepONet train block on all three, same class of
    // bug as sdfflow's missing dataset_dir. Values mirror
    // configs/Neural_Operator/ex9/config_train_point_deeponet_plasticity.txt,
    // the one benchmarked, working Point-DeepONet configuration in the repo.
    defaults: { model: "point_deeponet", mode: "train", gpu_ids: "0", modelpath: "../../output/point_deeponet/studio/point_deeponet.pth", coordinate_normalization: "centered_isotropic", point_sensor_count: "2048", point_hidden_channels: "128", point_feature_dim: "128", pointnet_depth: "3", point_trunk_depth: "3", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  deeponet: {
    label: "DeepONet", short: "DON", accent: "#526f9d", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Branch/trunk neural operator on the shared mesh dataset contract.", dataset: "mesh HDF5",
    // deeponet_sensor_resolution / deeponet_basis_dim were missing -- both
    // NOVAR-REQ required. Values mirror
    // configs/Neural_Operator/ex9/config_train_deeponet_plasticity.txt.
    defaults: { model: "deeponet", mode: "train", gpu_ids: "0", modelpath: "../../output/deeponet/studio/deeponet.pth", coordinate_normalization: "centered_isotropic", deeponet_sensor_resolution: "32,16", deeponet_hidden_channels: "256", deeponet_branch_depth: "3", deeponet_trunk_depth: "3", deeponet_basis_dim: "128", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  fno: {
    label: "FNO", short: "FNO", accent: "#6561a3", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Fourier Neural Operator with explicit grid and spectral-mode controls.", dataset: "mesh HDF5",
    defaults: { model: "fno", mode: "train", gpu_ids: "0", modelpath: "../../output/fno/studio/fno.pth", coordinate_normalization: "centered_isotropic", fno_grid_resolution: "64,64,64", fno_modes: "16,16,16", fno_hidden_channels: "64", fno_layers: "4", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  gino: {
    label: "GINO", short: "GINO", accent: "#7d5c99", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Geometry-informed neural operator for irregular domains.", dataset: "mesh HDF5",
    // gino_fno_hidden_channels / gino_fno_layers / gino_kernel_hidden were
    // missing -- all three NOVAR-REQ required, so a freshly dropped GINO block
    // never reached preflight. Values mirror
    // configs/Neural_Operator/ex9/config_train_gino_plasticity.txt.
    defaults: { model: "gino", mode: "train", gpu_ids: "0", modelpath: "../../output/gino/studio/gino.pth", coordinate_normalization: "centered_isotropic", gino_grid_resolution: "64,64,64", gino_fno_modes: "16,16,16", gino_fno_hidden_channels: "64", gino_fno_layers: "4", gino_kernel_hidden: "64", gino_in_radius: "0.05", gino_out_radius: "0.05", training_epochs: "500", batch_size: "2", learningr: "0.0001" }
  },
  transolver: {
    label: "Transolver3", short: "TR3", accent: "#9a5e55", modes: ["train", "inference"], keys: KEY_CATALOGS.transolver,
    description: "Physics-Attention with slice-space and node-sharded execution controls.", dataset: "mesh HDF5",
    defaults: { model: "transolver", mode: "train", gpu_ids: "0", modelpath: "../../output/transolver/studio/transolver.pth", coordinate_normalization: "centered_isotropic", latent_dim: "256", num_layers: "10", num_heads: "8", slice_num: "128", attention_kernel: "slice_space", use_checkpointing: "True", training_epochs: "500", batch_size: "1", grad_accum_steps: "4", learningr: "0.0001" }
  },
  sdfflow: {
    label: "SDFFlow", short: "SDF", accent: "#8b7837", modes: ["train", "train_vae", "train_fm", "sample", "reconstruct", "interpolate", "optimize", "evaluate"], keys: KEY_CATALOGS.sdfflow,
    description: "SDF-VAE and conditional flow matching for CAD candidate generation.", dataset: "SDF HDF5",
    // SDFFlow's train route requires 25 keys. Without architecture/checkpoint
    // defaults a freshly added block needed 18 hand-typed values before it could
    // even reach preflight, so these mirror configs/SDFFlow/config_train.txt
    // (decoder_type=attention is the VecSet pairing for latent_tokens > 1).
    defaults: {
      model: "sdfflow", mode: "train", gpu_ids: "0", latent_dim: "256", latent_tokens: "32",
      // dataset_dir was missing here -- sdfflow train is the only model whose
      // freshly dropped block failed preflight on a required field nothing in
      // this table ever set (CFG-REQ-001). deepjeb.h5 is the only staged SDF
      // dataset; its cond_names attr is exactly this five-column order.
      dataset_dir: "../../dataset/deepjeb.h5", split_seed: "42",
      // use_conditions defaults to FALSE in the spec, so shipping
      // condition_names without it left the FM stage unconditioned and the
      // whole list inert -- and the downstream "generative" pipeline, which
      // steers generation with cond_values, had nothing to steer. Turning it on
      // makes cond_dropout mandatory (SDF-COND-001); 0.1 is the checked-in
      // value and is what enables classifier-free guidance at sample time.
      use_conditions: "True", cond_dropout: "0.1",
      condition_names: "bbox_x,bbox_y,bbox_z,volume,area",
      // Reuse a finished, config-compatible VAE stage instead of retraining it
      // when only the FM stage changed.
      skip_completed_stages: "True",
      output_dir: "../../output/geometry_generation/studio",
      vae_modelpath: "../../output/geometry_generation/studio/sdfflow_vae.pth",
      fm_modelpath: "../../output/geometry_generation/studio/sdfflow_fm.pth",
      num_encoder_points: "4096", num_query_points: "4096",
      decoder_type: "attention", decoder_hidden: "512", decoder_layers: "8",
      encoder_dim: "256", encoder_heads: "4", encoder_blocks: "2",
      fm_hidden: "256", fm_blocks: "4", fm_cond_hidden: "128",
      vae_training_epochs: "500", vae_batch_size: "4", vae_learningr: "0.0005",
      fm_training_epochs: "500", fm_batch_size: "64", fm_learningr: "0.0002",
      training_epochs: "500", batch_size: "4", learningr: "0.0005"
    }
  }
};

export const REQUIRED = {
  simulgenvae: {
    train: keys(`model mode gpu_ids dataset_dir vae_modelpath lc_modelpath num_filter_enc latent_dim latent_dim_end lc_filter lc_data_type param_dir vae_training_epochs vae_batch_size vae_learningr lc_training_epochs lc_batch_size lc_learningr`),
    train_vae: keys(`model mode gpu_ids dataset_dir vae_modelpath num_filter_enc latent_dim latent_dim_end training_epochs batch_size learningr`),
    train_lc: keys(`model mode gpu_ids dataset_dir vae_modelpath lc_modelpath num_filter_enc latent_dim latent_dim_end lc_filter lc_data_type param_dir training_epochs batch_size learningr`),
    reconstruct: keys(`model mode gpu_ids dataset_dir vae_modelpath lc_modelpath num_filter_enc latent_dim latent_dim_end lc_filter lc_data_type param_dir output_dir`)
  },
  mlp: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var training_epochs batch_size learningr`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
  },
  meshgraphnets: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var edge_var latent_dim training_epochs batch_size learningr`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var edge_var`)
  },
  "meshgraphnets-v": {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var edge_var latent_dim training_epochs batch_size learningr`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var edge_var`)
  },
  "chi-mgnflow": {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var edge_var latent_dim training_epochs batch_size learningr`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var edge_var`)
  },
  point_deeponet: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var coordinate_normalization training_epochs batch_size learningr point_sensor_count point_hidden_channels point_feature_dim pointnet_depth point_trunk_depth`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
  },
  deeponet: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var coordinate_normalization training_epochs batch_size learningr deeponet_sensor_resolution deeponet_hidden_channels deeponet_branch_depth deeponet_trunk_depth deeponet_basis_dim`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
  },
  fno: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var coordinate_normalization training_epochs batch_size learningr fno_grid_resolution fno_modes fno_hidden_channels fno_layers`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
  },
  gino: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var coordinate_normalization training_epochs batch_size learningr gino_grid_resolution gino_fno_modes gino_fno_hidden_channels gino_fno_layers gino_in_radius gino_out_radius gino_kernel_hidden`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
  },
  transolver: {
    train: keys(`model mode gpu_ids dataset_dir modelpath input_var output_var coordinate_normalization training_epochs batch_size learningr latent_dim num_layers num_heads slice_num attention_kernel`),
    inference: keys(`model mode gpu_ids modelpath infer_dataset input_var output_var`)
  },
  sdfflow: {
    train: keys(`model mode gpu_ids dataset_dir output_dir vae_modelpath vae_training_epochs vae_batch_size vae_learningr fm_modelpath fm_training_epochs fm_batch_size fm_learningr latent_dim latent_tokens encoder_dim encoder_heads encoder_blocks decoder_type decoder_hidden decoder_layers num_encoder_points num_query_points fm_hidden fm_blocks fm_cond_hidden`),
    train_vae: keys(`model mode gpu_ids dataset_dir output_dir vae_modelpath training_epochs batch_size learningr latent_dim latent_tokens encoder_dim encoder_heads encoder_blocks decoder_type decoder_hidden decoder_layers num_encoder_points num_query_points`),
    train_fm: keys(`model mode gpu_ids dataset_dir output_dir vae_modelpath fm_modelpath training_epochs batch_size learningr fm_hidden fm_blocks fm_cond_hidden`),
    sample: keys(`model mode gpu_ids vae_modelpath fm_modelpath output_dir num_samples seed ode_steps mc_resolution`),
    reconstruct: keys(`model mode gpu_ids vae_modelpath input_mesh output_dir mc_resolution latent_refine_steps latent_refine_lr latent_refine_prior_weight`),
    // sample_index_b is conditional (SDF-INTERP-007): slerp_noise / lerp_latent
    // need the second endpoint, interpolation_space cond_sweep integrates the one
    // noise row sample_index_a under a condition sweep. alpha has a spec default.
    interpolate: keys(`model mode gpu_ids vae_modelpath fm_modelpath output_dir seed source_num_samples sample_index_a ode_steps mc_resolution interpolation_space`),
    optimize: keys(`model mode gpu_ids vae_modelpath fm_modelpath output_dir seed ode_steps mc_resolution opt_subspace_dim opt_budget opt_popsize opt_baseline_size opt_load_cases`),
    // Held-out VAE reconstruction metrics; the split keys must match the
    // training run so the evaluated shapes are the ones the VAE never saw.
    evaluate: keys(`model mode gpu_ids vae_modelpath dataset_dir output_dir split_seed split_by_parent eval_split eval_num_shapes eval_seed mc_resolution latent_refine_steps latent_refine_lr latent_refine_prior_weight`)
  }
};

export const CHOICES = {
  activation: ["relu", "gelu", "silu", "tanh"],
  norm: ["none", "batch", "layer"],
  input_normalization: ["standard", "minmax", "none"],
  output_normalization: ["standard", "minmax", "none"],
  output_activation: ["none", "relu", "sigmoid", "tanh", "softplus"],
  loss: ["mse", "mae", "huber"],
  network_size: ["small", "large"],
  // 'hdf5' reads the conditioner vector from the dataset's own cond_var rows
  // instead of a separate param_dir file; omitting it hid that path entirely.
  lc_data_type: ["csv", "image", "hdf5"],
  loss_type: ["1", "2", "3", "4"],
  parallel_mode: ["single", "ddp", "fsdp"],
  attention_kernel: ["naive", "slice_space"],
  infer_mode: ["direct", "decoupled"],
  coordinate_normalization: ["centered_isotropic"],
  time_integration: ["ar_ot", "ar_rt"],
  // The bare 'voronoi' alias was removed and the native hierarchy build raises on
  // it (MGN-COARSEN-001); voronoi_seedmean is what every checked-in config uses.
  coarsening_type: ["bfs", "voronoi_centroid", "voronoi_inherit", "voronoi_seedmean"],
  condition_ood_policy: ["warn", "clamp", "error"],
  // fea is the exact mesh-and-solve path; surrogate swaps in a trained HI-MGN
  // forward pass -- see the Optimization block's own notes on when that
  // checkpoint's numbers can be trusted before switching a live run to it.
  opt_analysis: ["fea", "surrogate"],
  decoder_type: ["mlp", "attention"],
  // SDFFlow v3 / evaluate-mode enums. Each is enforced by a spec validator
  // (SDF-EVAL-001, SDF-QUERY-001, and the interpolation_space check), so
  // without them the sheet took free text the launcher then rejected.
  eval_split: ["train", "val", "test"],
  encoder_query_type: ["learned", "fps"],
  interpolation_space: ["slerp_noise", "lerp_latent", "cond_sweep"],
  // SDFFlow conditional generation enums, each mirrored by a spec validator:
  // SDF-CDROP-001, SDF-GUIDE-001, SDF-AUDIT-001, SDF-EVAL-002. eval_methods is
  // a comma list (subset of plain,rejection,c2,e2,c2e2; SDF-EVAL-003), not an enum.
  cond_dropout_mode: ["all", "per_dim"],
  guidance_step_mode: ["velocity_dt", "per_step_jump"],
  condition_audit: ["geometric", "fea", "surrogate"],
  eval_task: ["reconstruction", "descriptor_calibration", "conditional"],
  fm_arch: ["mlp", "dit"],
  fm_time_sampling: ["uniform", "logit_normal"],
  point_variant: ["mesh_state"],
  point_sampling: ["random"],
  pointnet_activation: ["relu"],
  pointnet_norm: ["batch"],
  point_branch_merge: ["sum"],
  point_output_activation: ["identity", "tanh"],
  deeponet_branch_source: ["fixed_sensors"],
  deeponet_multi_output: ["split_both"],
  deeponet_activation: ["relu", "silu", "gelu", "tanh"],
  fno_variant: ["mesh", "paper_darcy"],
  fno_norm: ["none"],
  out_of_bounds_policy: ["error", "clamp"],
  sdf_source: ["none", "dataset", "sidecar"],
  gino_variant: ["mesh_state", "paper_decoder"],
  flow_solver: ["heun", "euler"],
  flow_t_sampling: ["uniform", "logitnormal"],
  flow_loss_weighting: ["uniform", "x0"],
  flow_predict: ["sample", "mean", "ensemble_mean"],
  best_by: ["recon", "crps", "det"]
};

export const PARALLEL_MODE_CHOICES = Object.freeze({
  simulgenvae: ["single", "ddp", "fsdp"],
  sdfflow: ["single", "ddp", "fsdp"],
  transolver: ["ddp", "node_shard"],
  fno: ["ddp", "model_split"],
  gino: ["ddp", "model_split"],
  meshgraphnets: ["ddp", "model_split"],
  "meshgraphnets-v": ["ddp", "model_split"],
  "chi-mgnflow": ["ddp"]
});

export const BOOLEAN_KEYS = new Set(keys(`augment_geometry bipartite_unpool coarse_world_edges display_testset display_trainset encoder_self_attention fit_latent_gmm fm_use_amp fm_use_ema gino_cache_neighbors guidance_enabled gino_group_shared_geometry gino_include_grid_coordinates gino_use_torch_cluster load_all make_histogram overfit_all_shapes point_resample_each_epoch resume_prior show_histogram skip_completed_stages small_output_init split_by_parent train_conditional_prior use_amp use_checkpointing use_compile use_conditional_prior use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_graph_aware vae_use_amp vae_use_ema lc_use_amp lc_use_ema write_preprocessing write_test_predictions`));
BOOLEAN_KEYS.delete("load_all");
export const OPERATOR_REMOVED = new Set(keys(`message_passing_num latent_dim edge_var world_radius_multiplier world_max_num_neighbors world_edge_backend coarse_world_edges multiscale_levels mp_per_level coarsening_type voronoi_clusters coarse_cache_per_worker use_vae vae_latent_dim vae_mp_layers vae_graph_aware free_bits posterior_min_std lambda_mmd lambda_kl lambda_det`));
export const TRANSOLVER_REJECTED = new Set(keys(`edge_var message_passing_num mp_per_level coarsening_type voronoi_clusters multiscale_levels world_radius_multiplier world_max_num_neighbors world_edge_backend coarse_world_edges`));
// Exact mirrors of cae_suite/specs/meshgraphnets.py. The deterministic runtime
// RAISES on the first set (removed_feature_guard) and silently ignores the
// second, but the spec still lists both in known_keys so preflight can explain
// them -- which meant the config sheet showed all 56 as ordinary editable
// fields with generic help, and only a preflight round-trip revealed that
// setting one is an error.
export const MGN_NATIVE_REMOVED = new Set(keys(`alpha_recon beta_aux fit_latent_gmm free_bits gmm_components gmm_covariance_type gmm_reg_covar lambda_det lambda_kl lambda_mmd num_vae_samples num_z posterior_min_std prior_batch_size prior_diagnose_interval prior_epochs prior_hidden_dim prior_learningr prior_loss_type prior_mc_samples prior_min_std prior_mixture_components prior_mp_layers prior_num_workers prior_temperature prior_val_interval resume_prior train_conditional_prior use_conditional_prior use_vae vae_graph_aware vae_latent_dim vae_mp_layers vae_valid_prior_samples`));
export const MGN_VARIATIONAL_IGNORED = new Set(keys(`eval_dataset hierarchy_cache_build_workers hierarchy_cache_dir hierarchy_cache_keep hierarchy_cache_wait_timeout histogram_bins histogram_clip_quantile make_histogram mmd_bandwidth prior_cov_rank prior_family prior_fm_steps prior_kl_reg_weight prior_nll_weight prior_type recon_loss show_histogram static_cache_per_worker vae_batch_size vae_batch_size_max vae_batch_size_min vae_batch_vram_fraction`));

export const VARIATIONAL_REMOVED = new Set(keys(`alpha_prior_max bipartite_unpool fit_latent_gmm free_bits gmm_components gmm_covariance_type gmm_reg_covar lambda_det lambda_kl positional_encoding residual_scale gamma_es es_samples es_steps es_noise_source es_start_epoch`));
export const CHI_FLOW_REMOVED = new Set(keys(`use_vae vae_latent_dim vae_mp_layers vae_graph_aware posterior_min_std num_z z_conditioning mmd_bandwidth mmd_gather_ranks lambda_mmd beta_aux alpha_recon recon_loss prior_type use_conditional_prior prior_family prior_nll_weight prior_fm_steps prior_fm_solver prior_mp_layers prior_hidden_dim prior_temperature prior_kl_reg_weight prior_cov_rank prior_min_std prior_mixture_components prior_grad_to_encoder vae_valid_prior_samples gamma_es es_samples es_steps es_noise_source es_start_epoch pipeline_microbatches std_noise noise_gamma noise_std_ratio`));
VARIATIONAL_REMOVED.forEach(key => CHI_FLOW_REMOVED.add(key));

export const BLOCK_SPECS = {
  "source.cad": {
    label: "CAD", category: "Sources", icon: "cad", accent: "#4c7f71", visual: "geometry", maturity: "native",
    description: "Select STEP, IGES, STL, PLY, or OBJ geometry and inspect every body.",
    inputs: [], outputs: [{ id: "geometry", type: "geometry", label: "geometry" }],
    // `units` is a statement, not a control: nothing in Studio or GeometryIngest
    // reads it and the ingest performs no unit conversion. It used to be an
    // editable "mm", which implied both. Rendered read-only by inspector.js.
    defaults: { path: "", units: "as stored in the file (no unit conversion)" }, sampleLabel: "CAD / mesh geometry"
  },
  "source.hdf5": {
    label: "HDF5 Dataset", category: "Sources", icon: "data", accent: "#3c7193", visual: "dataset", maturity: "native",
    description: "Browse samples, fields, geometry, splits, statistics, and verified parameter bindings.",
    inputs: [{ id: "parameters", type: "parameters", label: "parameter overlay", required: false }],
    outputs: [{ id: "data", type: "dataset", label: "configured dataset" }],
    // ex9 is what every shipped template trains on and the only staged dataset
    // every route (SimulGen-VAE included) accepts; a bare dataset block used to
    // default to ex1.h5, which SimulGen-VAE rejects outright.
    defaults: { path: "dataset/ex9.h5", split: "seeded 80/10/10", edit_mode: "immutable overlay" }, sampleLabel: "900 samples"
  },
  "source.parameters": {
    label: "Design Parameters", category: "Sources", icon: "parameters", accent: "#b0713f", visual: "parameters", maturity: "adapter",
    description: "Bind named scalar, vector, profile, CSV, or image conditions to compatible dataset inputs.",
    inputs: [], outputs: [{ id: "parameters", type: "parameters", label: "parameter overlay" }],
    defaults: { binding: "CSV condition columns", value: "0.50", range_policy: "warn outside training range" }, sampleLabel: "HDF5 input / output map"
  },
  "source.checkpoint": {
    label: "Saved ML Model", category: "Sources", icon: "model", accent: "#795991", visual: "checkpoint", maturity: "native",
    description: "Load a .pth checkpoint or compatible multi-stage model bundle with lineage.",
    inputs: [], outputs: [{ id: "model", type: "checkpoint", label: "saved model" }],
    defaults: { path: "", version: "best", compatibility: "auto-detect" }, sampleLabel: "Select checkpoint", workspace: "deploy"
  },
  "prep.geometry": {
    label: "Geometry → HDF5 Dataset", category: "Preparation", icon: "prepare", accent: "#29745f", visual: "dataset", maturity: "native",
    description: "Convert CAD or surface geometry to the shared graph or point-cloud HDF5 contract.",
    inputs: [{ id: "geometry", type: "geometry", label: "geometry", required: true }],
    outputs: [{ id: "data", type: "dataset", label: "dataset" }],
    defaults: {
      mode: "inspect", reader: "auto", mesh_type: "surface", emit: "graph",
      num_fields: "3", num_points: "4096", resample_method: "fps", seed: "42",
      // gmsh target element size; 0 keeps gmsh's own default. These are the only
      // controls over mesh density when meshing CAD, so they must be reachable.
      mesh_size_min: "0", mesh_size_max: "0",
      output_dataset: "../../studio/runtime/geometry-ingest/geometry.h5", limit: "0"
    },
    sampleLabel: "Geometry ingest results", executable: true
  },
  "run.inference": {
    label: "Inference Run", category: "Execution", icon: "run", accent: "#b76b2a", visual: "field", maturity: "native",
    description: "Run a sample, batch, rollout, variational ensemble, or SimulGen-VAE reconstruction.",
    inputs: [
      { id: "data", type: "dataset", label: "dataset", required: true },
      { id: "model", type: "checkpoint", label: "saved model", required: true },
      { id: "parameters", type: "parameters", label: "parameters", required: false }
    ],
    outputs: [{ id: "prediction", type: "field", label: "predictions" }, { id: "metrics", type: "metrics", label: "run statistics" }],
    defaults: {
      // `mode` and `viewer` are facts, not controls (nothing reads either; the
      // step's mode comes from the connected model block and the viewer is
      // fixed). Both used to be editable prose. inspector.js renders them
      // read-only per block type, because `mode` IS a real control elsewhere.
      mode: "resolved from the connected model block",
      // Which method to launch when no trainer is on the canvas. Read back from
      // the checkpoint itself when it can be identified; editable because a
      // checkpoint that records nothing about its family would otherwise leave
      // the block permanently unrunnable.
      model_id: "",
      gpu_ids: "0", infer_timesteps: "", inference_output_dir: "",
      batch_size: "16", num_workers: "", infer_chunk_size: "", infer_query_chunk_size: "",
      num_vae_samples: "", vae_batch_size: "",
      flow_steps: "", flow_solver: "", flow_predict: "",
      viewer: "prediction · truth · error · timestep player · distributions"
    }, sampleLabel: "20 reconstructions", executable: true
  },
  "run.cad_generator": {
    label: "CAD Generator", category: "Execution", icon: "run", accent: "#817336", visual: "candidates", maturity: "native",
    description: "Conditionally generate, reconstruct, or blend CAD candidates with SDFFlow.",
    inputs: [{ id: "parameters", type: "parameters", label: "design parameters" }, { id: "model", type: "checkpoint", label: "SDFFlow model", required: true }],
    outputs: [{ id: "candidates", type: "candidates", label: "CAD candidates" }],
    defaults: {
      mode: "sample", num_samples: "24", cfg_scale: "2.5", ode_steps: "50",
      mc_resolution: "128", seed: "42", cond_values: "",
      geometry_checks: "automatic: connected + watertight + bounds",
      // Only load-bearing in `mode optimize` (generate -> analyze -> search);
      // inert otherwise. Kept visible here rather than buried in the
      // generator model's own full-config panel since it is the one switch
      // that decides whether "analyze" is the exact FEA solve or the faster,
      // currently unproven HI-MGN surrogate -- see that model's own notes.
      opt_analysis: "fea"
    }, sampleLabel: "24 CAD candidates", executable: true
  },
  "optimize.design": {
    label: "Optimization", category: "Optimization", icon: "optimize", accent: "#8a613b", visual: "pareto", maturity: "adapter",
    // One input, not three. The block ranks the rows of one candidate/evaluation
    // CSV: connectedOptimizationCsv() reads a csv path off the connected source
    // and nothing else is consulted, so the former "physics evaluators"
    // (checkpoint) and "search variables" (parameters) ports accepted links that
    // changed no result -- a wire that promises an effect it cannot have.
    description: "Rank the rows of one candidate/evaluation CSV: hard constraints, then the feasible Pareto front with crowding-distance top-k.",
    inputs: [
      { id: "candidates", type: "candidates", label: "candidate / evaluation CSV", required: true }
    ],
    outputs: [{ id: "selected", type: "candidates", label: "Pareto designs" }, { id: "metrics", type: "metrics", label: "objective table" }, { id: "report", type: "report", label: "optimization report" }],
    defaults: {
      mode: "evaluate CSV candidate batch", csv_path: "",
      objectives: "volume,bbox_z", directions: "min,max",
      // No default constraint. "watertight >= 1" is a column of the DeepJEB
      // candidate table and of nothing else, so with the constraint box now
      // validated against the selected CSV the block arrived DISABLED for every
      // other input. Constraints are optional; the datalist suggests the real
      // columns of whichever CSV is chosen.
      constraints: "", selection: "feasible Pareto + diverse top-k", top_k: "6"
    }, sampleLabel: "No report yet", workspace: "optimization"
  },
  "evaluate.predictions": {
    label: "Evaluate Predictions", category: "Evaluation", icon: "evaluate", accent: "#1f7c66", visual: "parity", maturity: "adapter",
    description: "Compute compatible scalar and field metrics, distributions, and synchronized error views.",
    inputs: [{ id: "prediction", type: "field", label: "predictions", required: true }, { id: "truth", type: "dataset", label: "ground truth", required: true }],
    outputs: [{ id: "metrics", type: "metrics", label: "metrics" }, { id: "report", type: "report", label: "evaluation report" }],
    defaults: {
      // The metric set and its aggregates are fixed in studio_backend/analysis.py
      // (relative_l2, mae, rmse, max_absolute_error, r2; mean/median/p95/min/max).
      // These two rows state that; they were editable text that read as if the
      // user could choose, and understated what is actually computed.
      metrics: "relative L2 · MAE · RMSE · max abs error · R²", aggregate: "mean · median · p95 · min · max", error_view: "absolute + relative",
      mapping_mode: "schema", field_pairs: "", mapping_confirmed: "False"
    }, sampleLabel: "100 sample metrics", workspace: "evaluation"
  },
  "evaluate.training_metrics": {
    label: "Train Metrics", category: "Evaluation", icon: "evaluate", accent: "#167864", visual: "training", maturity: "native",
    description: "Plot every metric discovered in a connected model run, with per-metric include and exclude controls.",
    inputs: [{ id: "metrics", type: "metrics", label: "training metrics", required: true }],
    outputs: [{ id: "metrics", type: "metrics", label: "selected metrics" }, { id: "report", type: "report", label: "metric plots" }],
    defaults: { job_id: "", excluded_metrics: "", smoothing: "0" }, sampleLabel: "all discovered metrics",
    isMetricsViewer: true
  },
  "evaluate.compare": {
    label: "Compare Models", category: "Evaluation", icon: "evaluate", accent: "#5e6f90", visual: "ranking", maturity: "native",
    description: "Compare graph-connected training runs or user-selected evaluation CSVs while keeping their source lineage visible.",
    inputs: [{ id: "metrics", type: "metrics", label: "model runs", required: true, multiple: true }],
    outputs: [{ id: "report", type: "report", label: "comparison" }],
    defaults: { metric: "", direction: "min", qualification: "same held-out set" }, sampleLabel: "10 compared models", workspace: "comparison"
  },
  "output.export": {
    label: "Export Results", category: "Outputs", icon: "output", accent: "#596560", visual: "export", maturity: "adapter",
    description: "Export selected datasets, fields, CAD, metrics, reports, or model files with provenance.",
    inputs: [{ id: "input", type: "artifact", label: "artifact", required: true }],
    outputs: [{ id: "files", type: "artifact", label: "files" }],
    // Both rows used to be untrue AND editable: export_artifact() in
    // studio_backend/analysis.py copies a file or ZIPs a directory verbatim --
    // there is no VTK/STL/HTML conversion -- and always writes under
    // studio/runtime/exports/<label>-<token>, never to output/studio/run_001.
    // export_label IS editable and is the only field that changes the output --
    // it names the directory the copy/ZIP lands in. Leaving it out of defaults
    // meant the block's inspector showed two read-only statements and no control.
    defaults: { export_label: "ai-cae4all-artifact", format: "as-is copy (file) or ZIP (directory) · no conversion", path: "studio/runtime/exports/<label>-<token>" }, sampleLabel: "13 files", workspace: "export"
  },
  "deploy.api": {
    label: "API Deployment", category: "Deployment", icon: "output", accent: "#4d6874", visual: "export", maturity: "adapter",
    description: "Run the validated portable inference API or build the Windows inference executable.",
    inputs: [{ id: "model", type: "checkpoint", label: "saved model", required: true }, { id: "data", type: "dataset", label: "sample data", required: false }],
    outputs: [{ id: "endpoint", type: "endpoint", label: "endpoint" }],
    defaults: {
      checkpoint_path: "", input_path: "", output_name: "studio-inference",
      timesteps: "", num_samples: "", ode_steps: "", cond_values: ""
    }, sampleLabel: "No deployment job yet", workspace: "deploy"
  }
};

function installModelBlock(modelId, model) {
  BLOCK_SPECS[`model.${modelId}`] = {
    label: model.label,
    category: "Models",
    icon: "model",
    accent: model.accent,
    visual: modelId === "simulgenvae" ? "latent" : "training",
    maturity: "native",
    description: model.description,
    inputs: [
      { id: "data", type: "dataset", label: "training data", required: true },
      { id: "parameters", type: "parameters", label: "conditions", required: false },
      { id: "resume", type: "checkpoint", label: "resume / stage model", required: false }
    ],
    outputs: [{ id: "model", type: "checkpoint", label: modelId === "simulgenvae" ? "VAE + LC bundle" : "saved model" }, { id: "metrics", type: "metrics", label: "training metrics" }],
    defaults: { ...model.defaults },
    sampleLabel: modelId === "simulgenvae" ? "VAE stage 2 / 2" : "epoch 342 / 500",
    modelId, executable: true,
    isModel: true
  };
}

Object.entries(MODEL_CATALOG).forEach(([modelId, model]) => installModelBlock(modelId, model));

/**
 * Keep a newly registered trainable route usable even before constants.js is
 * updated. Curated entries retain their richer copy/defaults; the live
 * MethodSpec always owns keys, modes, required fields, and published defaults.
 */
export function registerLiveModel(model) {
  if (!model?.model || model.spec_id === "geometry_ingest") return false;
  const modelId = String(model.model);
  const stringifyDefault = value => typeof value === "boolean"
    ? (value ? "True" : "False")
    : String(value ?? "");
  const liveDefaults = Object.fromEntries(
    Object.entries(model.defaults || {}).map(([key, value]) => [key, stringifyDefault(value)])
  );
  const liveDefaultsByMode = Object.fromEntries(
    Object.entries(model.defaults_by_mode || {}).map(([mode, defaults]) => [
      mode,
      Object.fromEntries(Object.entries(defaults || {}).map(([key, value]) => [key, stringifyDefault(value)]))
    ])
  );
  const existing = MODEL_CATALOG[modelId];
  const local = existing || {
    label: model.method || modelId,
    short: modelId.slice(0, 7).toUpperCase(),
    accent: "#486f78",
    description: `Live ${model.method || modelId} route published by the backend registry.`,
    defaults: { model: modelId, mode: model.modes?.[0] || "train", gpu_ids: "0" }
  };
  Object.assign(local, {
    keys: [...(model.known_keys || [])],
    modes: [...(model.modes || [])],
    required: model.required || {},
    dataset: model.dataset_kind || local.dataset || "backend-defined dataset",
    backend: model,
    backendDefaults: liveDefaults,
    backendDefaultsByMode: liveDefaultsByMode,
    defaults: { ...(local.defaults || {}), ...liveDefaults }
  });
  MODEL_CATALOG[modelId] = local;
  if (!BLOCK_SPECS[`model.${modelId}`]) installModelBlock(modelId, local);
  else BLOCK_SPECS[`model.${modelId}`].defaults = { ...local.defaults };
  return !existing;
}

/**
 * Every mesh surrogate is wired the same way, so the shipped per-model pipelines
 * are generated instead of copy-pasted: one dataset feeds a trainer, a metrics
 * viewer hangs off the trainer, and inference → evaluation → export closes the loop.
 *
 * They all point at ex9 (plasticity: 900 samples x 3131 nodes x 20 steps), the
 * smallest staged mesh dataset, and mirror the checked-in ex9 configs under
 * `configs/<method>/ex9/` exactly — including `cond_var 2`, without which 50% of
 * the loss would sit on constant targets.
 */
const EX9_MESH = {
  dataset_dir: "../../dataset/ex9.h5",
  infer_dataset: "../../dataset/ex9_infer.h5",
  input_var: "2", output_var: "2", cond_var: "2",
  feature_loss_weights: "1.0, 1.0", positional_features: "4",
  use_node_types: "False", infer_timesteps: "19", split_seed: "42"
};

function meshPipeline(name, modelType, trainerConfig = {}, inferenceConfig = {}) {
  return {
    name,
    nodes: [
      ["dataset", "source.hdf5", 35, 70, { path: "dataset/ex9.h5" }],
      // The inference dataset is its own source block on purpose. Wiring the
      // training dataset into the Inference block made the graph *say* "predict
      // the data you trained on", and because the graph wins over the trainer's
      // own infer_dataset, that is exactly what it did -- silently replacing
      // ex9_infer.h5 with ex9.h5 on every run. Two blocks keep the held-out
      // split visible on the canvas, which is where the user can see it.
      ["infer_dataset_src", "source.hdf5", 35, 300, { path: "dataset/ex9_infer.h5" }],
      ["trainer", modelType, 330, 70, { ...EX9_MESH, ...trainerConfig }],
      ["train_metrics", "evaluate.training_metrics", 625, 390],
      // inferenceConfig: run-time readout choices that belong to the Inference
      // block, not the trainer (see chimgnflow).
      ["inference", "run.inference", 625, 70, { ...inferenceConfig }],
      ["evaluation", "evaluate.predictions", 920, 70],
      ["export", "output.export", 1215, 70]
    ],
    edges: [
      ["dataset", "data", "trainer", "data"],
      ["trainer", "metrics", "train_metrics", "metrics"],
      ["infer_dataset_src", "data", "inference", "data"],
      ["trainer", "model", "inference", "model"],
      ["inference", "prediction", "evaluation", "prediction"],
      ["infer_dataset_src", "data", "evaluation", "truth"],
      ["evaluation", "report", "export", "input"]
    ]
  };
}

export const TEMPLATES = {
  // Default pipeline. Mirrors configs/MeshGraphNets/ex9/config_train_hi_plasticity.txt:
  // a 2-level Voronoi hierarchy (voronoi_seedmean — the bare "voronoi" alias was
  // removed and the native hierarchy build raises on it).
  himgn: meshPipeline("HI-MGN multiscale (ex9 plasticity)", "model.meshgraphnets", {
    message_passing_num: "15", latent_dim: "128", edge_var: "8",
    use_multiscale: "True", coarsening_type: "voronoi_seedmean",
    voronoi_clusters: "500, 100", multiscale_levels: "2", mp_per_level: "4, 6, 8, 6, 4",
    use_world_edges: "False", training_epochs: "500", batch_size: "4", learningr: "0.0001",
    weight_decay: "0.0001", warmup_epochs: "3", use_amp: "True", use_ema: "True",
    ema_decay: "0.99", val_interval: "5", num_workers: "2"
  }),
  /**
   * SimulGen-VAE, on ex9 rather than ex1, reading its conditions out of the
   * dataset instead of a CSV. Both changes fix a template that could not run:
   *
   *  - ex1.h5 is NOT fixed geometry. SimulGen-VAE flattens the field rows into a
   *    dense [samples, channels, time] tensor, so every sample must share one
   *    (T, N); ex1 holds 100 *different* shapes and preflight rejected it with
   *    STUDIO-SGV-FIXED-001. The node even carried a "fixed N and T required"
   *    label while pointing at the one staged dataset that violates it. ex9 is
   *    (T=20, N=3131) for all 900 samples -- and is what every other template
   *    here already targets.
   *  - dataset/ex1_conditions.csv does not exist on disk (STUDIO-SGV-PARAM-001).
   *    `lc_data_type hdf5` + `cond_var 2` reads the conditioner's inputs from
   *    ex9's own trailing input-only rows (uz, die_profile), which removes the
   *    external file entirely -- so the separate parameters block and its two
   *    edges are gone, matching how every mesh template in this file is wired.
   *
   * num_var 2 = ux,uy (rows 3:5), mirroring configs/SimulGenVAE/ex9/
   * config_train_vae.txt; the trailing rows are conditions, not field channels.
   */
  simulgen: {
    name: "SimulGen-VAE reconstruction (ex9)",
    nodes: [
      ["dataset", "source.hdf5", 35, 70, { path: "dataset/ex9.h5", compatibility: "fixed N and T required" }],
      // Held out on purpose: reconstructing the training split is not evidence.
      ["infer_dataset_src", "source.hdf5", 35, 300, { path: "dataset/ex9_infer.h5" }],
      ["simulgen", "model.simulgenvae", 330, 70, {
        dataset_dir: "../../dataset/ex9.h5", num_var: "2", field_start_row: "3",
        cond_var: "2", lc_data_type: "hdf5", vae_training_epochs: "20"
      }],
      ["train_metrics", "evaluate.training_metrics", 625, 390],
      ["inference", "run.inference", 625, 70, { mode: "SimulGen-VAE reconstruct" }],
      ["evaluation", "evaluate.predictions", 920, 70],
      ["export", "output.export", 1215, 70]
    ],
    edges: [
      ["dataset", "data", "simulgen", "data"],
      ["simulgen", "metrics", "train_metrics", "metrics"],
      ["infer_dataset_src", "data", "inference", "data"],
      ["simulgen", "model", "inference", "model"],
      ["inference", "prediction", "evaluation", "prediction"],
      ["infer_dataset_src", "data", "evaluation", "truth"],
      ["evaluation", "report", "export", "input"]
    ]
  },
  geometry: {
    // `ingest`, not `inspect`. The template wires the ingest block's output into
    // an Export block, but `inspect` is a dry run -- pipeline.py returns after
    // printing "no file written", and geometryConfigText only emits
    // output_dataset for `ingest` -- so the shipped pipeline's second half
    // pointed at a file its first half could never produce. `inspect` is still
    // one dropdown away for a dry check.
    name: "Geometry to HDF5 (ingest)",
    nodes: [
      ["cad", "source.cad", 35, 125, { path: "" }],
      // reader "auto", not "trimesh": the CAD block accepts .step/.stp/.iges/.igs,
      // which trimesh cannot read at all. auto routes CAD to gmsh and meshes to
      // trimesh, so it is correct for every extension the picker offers.
      ["ingest", "prep.geometry", 365, 125, { mode: "ingest", reader: "auto", mesh_type: "surface", emit: "graph, pointcloud", num_points: "4096", limit: "3" }],
      // No `path` override: exports always land under studio/runtime/exports/.
      ["export", "output.export", 710, 125]
    ],
    edges: [
      ["cad", "geometry", "ingest", "geometry"],
      ["ingest", "data", "export", "input"]
    ]
  },
  // No coarsening_type here on purpose. This is the flat (single-level) arm, so
  // the key is inert -- and the value it used to carry, "none", is not one of
  // the four the spec accepts, so the config sheet's own dropdown could not
  // represent it. An inert key with an illegal value is worse than no key.
  physics: meshPipeline("MeshGraphNets flat (ex9 plasticity)", "model.meshgraphnets", {
    message_passing_num: "15", latent_dim: "128", edge_var: "8",
    use_multiscale: "False",
    training_epochs: "500", batch_size: "4", learningr: "0.0001"
  }),
  meshgraphnets_v: meshPipeline("MeshGraphNets-V variational (ex9)", "model.meshgraphnets-v", {
    message_passing_num: "15", latent_dim: "128", edge_var: "8",
    use_vae: "True", vae_latent_dim: "256", num_vae_samples: "32",
    training_epochs: "500", batch_size: "4", learningr: "0.0001"
  }),
  // Mirrors configs/HI_MGNFlow/ex9/config_ex9_fm_v.txt (the FM_V arm: velocity
  // prediction with uniform time weighting -- the sampling-optimal
  // parameterization and the arm every other cHI-MGNflow config in the SAOI
  // sweep is compared against). training_epochs 25 is that config's real
  // budget, not a placeholder: flow-matching plus the multiscale hierarchy plus
  // an 8-sample validation ensemble makes each epoch far heavier than a plain
  // MeshGraphNets epoch on the same ex9 dataset, so 500 would never realistically
  // finish as a default demo run. Was previously the one live route (of 12) with
  // no pipeline template at all -- dragging its block in required hand-wiring
  // every port from scratch.
  chimgnflow: meshPipeline("cHI-MGNflow (ex9 plasticity)", "model.chi-mgnflow", {
    message_passing_num: "15", latent_dim: "96", edge_var: "8", parallel_mode: "ddp",
    use_multiscale: "True", coarsening_type: "voronoi_seedmean",
    voronoi_clusters: "500, 100", multiscale_levels: "2", mp_per_level: "3, 4, 6, 4, 3",
    hierarchy_variants: "1", hierarchy_seed: "1234", hierarchy_cache_keep: "True",
    time_integration: "ar_ot",
    flow_steps: "20", flow_solver: "heun", flow_time_freqs: "16", flow_t_sampling: "uniform",
    flow_loss_weighting: "uniform", flow_det_prob: "0", flow_predict: "sample",
    val_flow_steps: "8", val_num_samples: "4", best_by: "crps",
    training_epochs: "25", batch_size: "8", learningr: "0.0003",
    use_checkpointing: "True", use_amp: "True", use_ema: "True", ema_decay: "0.99",
    val_interval: "5", num_workers: "2", augment_geometry: "False",
    grad_accum_steps: "1"
  }, {
    // Readout for the Inference block, not the trainer -- and `sample` here is
    // measured, not assumed. The reasoning for `mean` is seductive: Evaluate
    // computes relative L2 and R^2, which are pointwise metrics, so scoring one
    // stochastic draw looks like counting the model's variance as error. Both
    // were run end to end on this exact 25-epoch checkpoint over all 87 held-out
    // ex9 samples:
    //
    //     flow_predict sample   relative L2 1.25   R^2 mean -0.96 (median -0.44)
    //     flow_predict mean     relative L2 1.53   R^2 mean -1.76 (median -1.41)
    //
    // `mean` is the deterministic t=0 readout, and this arm mirrors FM_V
    // (flow_loss_weighting uniform, flow_det_prob 0) -- it spends no training
    // budget at t=0, so reading the mode out of it is reading out something it
    // never trained. configs/HI_MGNFlow/ex9/config_ex9_fm_x0.txt is the arm
    // built for that readout (x0 weighting + flow_det_prob 0.25 + best_by det).
    // For a better pointwise score without changing the training arm, raise
    // num_vae_samples and use flow_predict ensemble_mean, at N x the cost.
    flow_predict: "sample", num_vae_samples: "1"
  }),
  transolver: meshPipeline("Transolver (ex9 plasticity)", "model.transolver", {
    latent_dim: "128", num_layers: "10", num_heads: "8", slice_num: "128",
    attention_kernel: "slice_space", training_epochs: "500", batch_size: "8", learningr: "0.001"
  }),
  // Grid/mode axes below are 2 entries, not 3: ex9 is a 2-D forging domain
  // (Geo-FNO plasticity, 987 cases x 20 states x 3131 nodes). A 3-entry
  // "64,64,64"/"16,16,16" grid was shipped here for a while -- it's internally
  // consistent (same length both sides) so it clears NOVAR-FNO-001/NOVAR-GINO-001
  // at the spec layer, but it fits a 3rd axis that does not exist in this
  // dataset. Values below instead mirror
  // configs/Neural_Operator/ex9/config_train_fno_plasticity.txt exactly, the
  // benchmarked, actually-run FNO configuration on this data.
  fno: meshPipeline("FNO (ex9 plasticity)", "model.fno", {
    fno_variant: "mesh", fno_grid_resolution: "64, 32", fno_modes: "16, 12", fno_hidden_channels: "64",
    fno_layers: "4", training_epochs: "500", batch_size: "20", learningr: "0.001"
  }),
  // Mirrors configs/Neural_Operator/ex9/config_train_gino_plasticity.txt (same
  // 2-D dimensionality note as fno above).
  gino: meshPipeline("GINO (ex9 plasticity)", "model.gino", {
    gino_variant: "mesh_state", gino_grid_resolution: "64, 32", gino_fno_modes: "16, 12",
    gino_fno_hidden_channels: "64", gino_fno_layers: "4", gino_kernel_hidden: "64",
    gino_in_radius: "0.08", gino_out_radius: "0.08",
    training_epochs: "500", batch_size: "1", learningr: "0.001"
  }),
  // Mirrors configs/Neural_Operator/ex9/config_train_deeponet_plasticity.txt.
  // deeponet_sensor_resolution and deeponet_basis_dim were previously absent
  // here -- both NOVAR-REQ required, so this template failed preflight before
  // a user could even open it.
  deeponet: meshPipeline("DeepONet (ex9 plasticity)", "model.deeponet", {
    deeponet_sensor_resolution: "32, 16", deeponet_hidden_channels: "256",
    deeponet_branch_depth: "3", deeponet_trunk_depth: "3", deeponet_basis_dim: "128",
    deeponet_activation: "silu",
    training_epochs: "500", batch_size: "20", learningr: "0.001"
  }),
  // Mirrors configs/Neural_Operator/ex9/config_train_point_deeponet_plasticity.txt.
  // point_feature_dim, pointnet_depth, and point_trunk_depth were previously
  // absent -- all three NOVAR-REQ required, same missing-required-field bug as
  // the other three operator templates above.
  point_deeponet: meshPipeline("Point-DeepONet (ex9 plasticity)", "model.point_deeponet", {
    point_sensor_count: "2048", point_hidden_channels: "128", point_feature_dim: "128",
    pointnet_depth: "3", point_trunk_depth: "3",
    training_epochs: "500", batch_size: "2", learningr: "0.001"
  }),
  parametric: {
    name: "Parametric response estimation",
    nodes: [
      ["parameters", "source.parameters", 35, 295],
      ["dataset", "source.hdf5", 35, 55, { path: "dataset/mlp/train.h5" }],
      // Same reason as meshPipeline: this template used to feed the *training*
      // table into the Inference block, so the shipped pipeline scored the MLP
      // on rows it had already fitted. The held-out split was sitting unused
      // next to it the whole time.
      ["infer_dataset_src", "source.hdf5", 35, 470, { path: "dataset/mlp/infer.h5" }],
      // input_var/output_var are stated rather than left to the user here, and
      // that is safe *only* because MLP is the suite's one tabular route
      // (dataset_kind=table_hdf5): the contract is X[S,N] -> Y[S,M] with no
      // cond_var rows, so 3 and 2 are literally X.shape[1] and Y.shape[1] of
      // dataset/mlp/train.h5. Do not copy this to a mesh template -- there
      // input_var is NOT the feature-row count and hardcoding it reintroduces
      // the ex9 constant-target class of bug. Without these three keys the
      // shipped template failed preflight with CFG-REQ-001 x3 before the user
      // touched anything.
      ["mlp", "model.mlp", 330, 115, {
        dataset_dir: "../../dataset/mlp/train.h5", infer_dataset: "../../dataset/mlp/infer.h5",
        input_var: "3", output_var: "2", split_seed: "42", val_interval: "5"
      }],
      ["train_metrics", "evaluate.training_metrics", 625, 430],
      ["inference", "run.inference", 625, 115],
      ["evaluation", "evaluate.predictions", 920, 115],
      ["export", "output.export", 1215, 115]
    ],
    edges: [
      ["parameters", "parameters", "dataset", "parameters"],
      ["dataset", "data", "mlp", "data"],
      ["parameters", "parameters", "mlp", "parameters"],
      ["mlp", "metrics", "train_metrics", "metrics"],
      ["infer_dataset_src", "data", "inference", "data"],
      ["mlp", "model", "inference", "model"],
      ["parameters", "parameters", "inference", "parameters"],
      ["inference", "prediction", "evaluation", "prediction"],
      ["infer_dataset_src", "data", "evaluation", "truth"],
      ["evaluation", "report", "export", "input"]
    ]
  },
  // The only other template touching sdfflow ("generative", below) assumes a
  // *pretrained* checkpoint already exists -- its model.sdfflow block is
  // configured in mode "sample", not "train". Before this template, SDFFlow
  // was the only mode among the suite's 12 live routes with no from-scratch
  // pipeline: training it required hand-typing the merged VAE->FM route onto
  // a bare block. This mirrors configs/SDFFlow/config_train.txt's merged
  // `mode train` pipeline on the real deepjeb.h5 SDF dataset (2138 shapes),
  // then feeds the resulting checkpoint straight into a sample-mode CAD
  // generator to prove the trained model actually produces candidates --
  // the natural on-ramp into the "generative" design-optimization template.
  sdfflow_train: {
    name: "SDFFlow train (DeepJEB)",
    nodes: [
      ["dataset", "source.hdf5", 35, 70, { path: "dataset/deepjeb.h5" }],
      ["trainer", "model.sdfflow", 330, 70, { mode: "train" }],
      ["train_metrics", "evaluate.training_metrics", 330, 390],
      ["generator", "run.cad_generator", 625, 70, { mode: "sample" }],
      ["export", "output.export", 920, 70]
    ],
    edges: [
      ["dataset", "data", "trainer", "data"],
      ["trainer", "metrics", "train_metrics", "metrics"],
      ["trainer", "model", "generator", "model"],
      ["generator", "candidates", "export", "input"]
    ]
  },
  generative: {
    // The name states the prerequisite because this template legitimately
    // preflights RED on a fresh checkout: its two SDFFlow checkpoints do not
    // exist until something trains them. "sdfflow_train" writes to exactly
    // these two paths, so running that first turns both errors green.
    name: "Design optimization (needs trained SDFFlow)",
    nodes: [
      ["parameters", "source.parameters", 35, 300],
      // num_samples/seed are required for sdfflow `sample` and were absent, so
      // this block reported two CFG-REQ-001 errors on a freshly loaded template.
      // They match the run.cad_generator block it feeds, which is where the
      // user actually reads them.
      ["generator_model", "model.sdfflow", 35, 55, { mode: "sample", num_samples: "24", seed: "42" }],
      ["generator", "run.cad_generator", 330, 125],
      ["optimization", "optimize.design", 625, 125, {
        objectives: "volume,bbox_z", directions: "min,max", constraints: "watertight >= 1"
      }],
      ["export", "output.export", 920, 125]
    ],
    edges: [
      ["parameters", "parameters", "generator", "parameters"],
      ["generator_model", "model", "generator", "model"],
      ["generator", "candidates", "optimization", "candidates"],
      ["optimization", "selected", "export", "input"]
    ]
  },
  blank: { name: "Untitled pipeline", nodes: [], edges: [] }
};

export const STUDIO_SECTIONS = {
  data: {
    label: "Data", icon: "data", color: "#3c7193", note: "Geometry, datasets, conditions",
    title: "Data workspace", description: "Curate every object entering a pipeline, inspect individual samples, preserve split provenance, and expose exact HDF5 contracts.",
    cards: [
      ["HDF5 dataset catalog", "data", "native", "Samples, geometry, topology, fields, timesteps, splits, statistics, and parameter bindings.", ["mesh HDF5", "table HDF5", "SDF HDF5"], "source.hdf5"],
      ["SimulGen fixed-geometry gate", "data", "native", "Verify uniform node count N and timestep count T before SimulGen-VAE training or reconstruction.", ["fixed N", "fixed T", "field rows"], "model.simulgenvae"],
      ["Condition sources", "parameters", "adapter", "CSV rows or condition images ordered to match sorted SimulGen sample IDs.", ["csv", "image", "ordering"], "source.parameters"],
      ["Geometry ingest", "prepare", "native", "Convert STEP, IGES, STL, PLY, or OBJ into the shared graph and point-cloud HDF5 contract.", ["CAD", "Gmsh", "trimesh"], "prep.geometry"],
      ["Dataset compatibility", "prepare", "native", "Inspect shapes, feature counts, node types, fields, and model-specific compatibility before execution.", ["schema", "diagnostics", "preflight"], "source.hdf5"],
      ["Immutable parameter overlays", "parameters", "adapter", "Apply verified parameter changes without rewriting the source HDF5 artifact.", ["binding", "range guard", "lineage"], "source.parameters"]
    ]
  },
  experiments: {
    label: "Experiments", icon: "evaluate", color: "#1f7c66", note: "Training, sweeps, sessions",
    title: "Experiment workspace", description: "Training is durable work. Track sessions, GPU resources, checkpoints, stage reuse, comparisons, and restartable lineage.",
    cards: [
      ["Train Metrics", "evaluate", "native", "Plot every metric discovered in actual model-run logs, then include or exclude individual series.", ["all by default", "per-metric toggle", "actual logs"], "evaluate.training_metrics"],
      ["SimulGen staged training", "model", "native", "Train the merged VAE → latent-conditioner pipeline or execute either stage independently.", ["train", "train_vae", "train_lc"], "model.simulgenvae"],
      ["Stage compatibility", "model", "native", "Reuse completed VAE/LC checkpoints only when stage metadata and configuration remain compatible.", ["skip completed", "VAE", "LC"], "model.simulgenvae"],
      ["Hyperparameter sweeps", "parameters", "adapter", "Schedule child configurations and compare validation, resource, and reconstruction metrics.", ["grid", "random", "parent/child"]],
      ["Resource and VRAM", "system", "adapter", "Measure real peak allocated memory, throughput, batch feasibility, and multi-GPU behavior.", ["peak allocated", "DDP", "FSDP"]],
      ["Cross-model comparison", "evaluate", "native", "Compare graph-connected run histories, then rank selected held-out outputs only when their metric and test set are genuinely comparable.", ["multiple runs", "same test set", "lineage"], "evaluate.compare"]
    ]
  },
  optimization: {
    label: "Optimization", icon: "optimize", color: "#8a613b", note: "Objectives, constraints, Pareto",
    title: "Generated-design evaluation and optimization", description: "Turn candidate galleries into engineering decisions while keeping geometry evidence, physics predictions, constraints, Pareto trade-offs, and verification separate.",
    cards: [
      ["Geometry feasibility", "prepare", "adapter", "Consume explicit geometry-check columns from the selected candidate CSV as hard constraints.", ["CSV gate", "condition column", "traceable"], "optimize.design"],
      ["Surrogate evaluation", "model", "adapter", "Consume actual outputs from linked inference or benchmark runs; the optimizer does not invent physics scores.", ["actual CSV", "multi-model", "lineage"], "optimize.design"],
      ["Objectives and constraints", "parameters", "native", "Select numeric columns, min/max direction, hard inequalities, and a top-k budget.", ["min/max", "inequalities", "columns"], "optimize.design"],
      ["Pareto and diversity", "optimize", "native", "Compute the feasible non-dominated set and crowding-distance top-k from actual numeric rows.", ["Pareto front", "top-k", "crowding"], "optimize.design"],
      ["Iterative search", "run", "roadmap", "Use DOE, evolutionary, or constrained Bayesian search with caching and an evaluation budget.", ["DOE", "NSGA-II", "Bayesian"], "optimize.design"],
      ["Solver verification", "prepare", "roadmap", "Re-simulate finalists, measure surrogate discrepancy, and approve, reject, or retrain.", ["truth gate", "OOD", "active learning"], "optimize.design"]
    ]
  },
  evaluation: {
    label: "Evaluation", icon: "evaluate", color: "#1f7c66", note: "Field metrics and errors",
    title: "Prediction evaluation", description: "Compare real rollout or reconstructed HDF5 fields against a real ground-truth HDF5 file and persist per-sample evidence.",
    cards: [
      ["Relative L2", "evaluate", "native", "Global per-sample field norm error with a finite-value mask and explicit field-row selection.", ["relative L2", "per sample", "actual arrays"], "evaluate.predictions"],
      ["Absolute errors", "evaluate", "native", "MAE, RMSE, and maximum absolute error over matched feature, timestep, and node dimensions.", ["MAE", "RMSE", "max error"], "evaluate.predictions"],
      ["Explained variance", "evaluate", "native", "R² with mean, median, p95, minimum, and maximum aggregation.", ["R²", "aggregate", "JSON"], "evaluate.predictions"]
    ]
  },
  comparison: {
    label: "Compare", icon: "evaluate", color: "#5e6f90", note: "Rank actual model results",
    title: "Cross-model comparison", description: "Resolve graph-connected run histories by persisted node lineage, and separately rank user-selected benchmark or evaluation CSV group means.",
    cards: [
      ["CSV mean ranking", "evaluate", "native", "Choose the grouping column, numeric metric, and min/max direction; matching finite observations are averaged before ranking. The user remains responsible for selecting comparable evidence.", ["CSV", "mean", "evidence"], "evaluate.compare"],
      ["Benchmark handoff", "evaluate", "native", "Use checked comparison.csv artifacts from benchmark campaigns without copying values into the UI.", ["benchmark", "evidence", "JSON"], "evaluate.compare"]
    ]
  },
  export: {
    label: "Export", icon: "output", color: "#596560", note: "Copy or archive artifacts",
    title: "Artifact export", description: "Copy a selected real file or ZIP a selected artifact directory into the isolated frontend runtime and download it.",
    cards: [
      ["Export file", "output", "native", "Copy a real HDF5, CSV, JSON, checkpoint, or report while preserving the source artifact.", ["copy", "provenance", "download"], "output.export"],
      ["Export directory", "output", "native", "Create a ZIP archive under studio/runtime/exports for multi-file inference and geometry outputs.", ["ZIP", "folder", "download"], "output.export"]
    ]
  },
  models: {
    label: "Models", icon: "model", color: "#76568e", note: "Routes, keys, checkpoints",
    title: "Model and checkpoint workspace", description: "Browse every installed route, its real mode and configuration contract, checkpoint compatibility, dataset requirements, and intended engineering use.",
    modelCards: true
  },
  benchmarks: {
    label: "Campaigns", icon: "evaluate", color: "#8a613b", note: "Checked-in configs and preflight",
    title: "Benchmark campaign workspace", description: "Inspect, preflight, and load the checked-in cross-method campaign roster. A present or passing config is not evidence that its dataset was staged, training ran, or a published result was reproduced.",
    cards: [
      ["Cross-method roster", "evaluate", "adapter", "Browse the ex4-ex9 model arms named by configs/campaigns/benchmarks_all/roster.tsv.", ["roster", "ex4-ex9", "config"]],
      ["Native preflight", "evaluate", "native", "Run the launcher contract checks for one checked-in config before loading it into the graph.", ["validation", "warnings", "errors"]],
      ["Evidence boundary", "output", "adapter", "Campaign membership and preflight status do not create a score or validate a scientific claim.", ["not executed", "no score", "lineage"]]
    ]
  },
  artifacts: {
    label: "Artifacts", icon: "artifact", color: "#68736e", note: "Fields, geometry, models",
    title: "Artifact workspace", description: "Search typed outputs across pipelines, inspect individual samples, compare versions, and follow provenance to the producing block and run.",
    cards: [
      ["SimulGen model bundles", "model", "native", "VAE and latent-conditioner checkpoints with stage, architecture, scaling, and config metadata.", ["vae.pth", "lc.pth", "compatibility"], "source.checkpoint"],
      ["Reconstructed fields", "artifact", "native", "Per-sample field reconstruction, ground truth, MSE, timestep player, and HDF5 export.", ["field", "truth", "MSE"], "run.inference"],
      ["Generated CAD", "artifact", "native", "Candidate gallery, geometry checks, conditions, latent provenance, and selection state.", ["STL", "gallery", "validity"], "run.cad_generator"],
      ["Evaluation reports", "evaluate", "adapter", "Metrics, error maps, distributions, worst samples, and exported evidence.", ["CSV", "JSON", "HTML"], "evaluate.predictions"],
      ["Optimization reports", "optimize", "native", "Fixed-batch candidate status, constraints, Pareto set, crowding selection, and source-row provenance.", ["Pareto", "top-k", "provenance"], "optimize.design"],
      ["Use in pipeline", "run", "adapter", "Insert a compatible concrete source block from any pinned artifact version.", ["typed source", "pinned", "lineage"]]
    ]
  },
  deploy: {
    label: "Deploy", icon: "output", color: "#4d6874", note: "Executable and API handoff",
    title: "Deployment workspace", description: "Package validated saved models for portable inference or publish them behind governed, versioned APIs.",
    cards: [
      ["Inference bundle", "output", "native", "CPU-oriented, family-detecting standalone inference with parity and limitations reports.", ["portable", "family detect", "parity"]],
      ["Build .exe", "output", "native", "PyInstaller one-folder executable for a no-Python Windows target.", ["Windows", "CPU", "smoke test"]],
      ["SimulGen bundle", "model", "adapter", "Package the VAE + LC checkpoints, condition schema, scaling, and reconstruction entrypoint together.", ["dual checkpoint", "condition schema", "field output"], "model.simulgenvae"],
      ["Local inference API", "output", "adapter", "Run family-detected CPU inference through POST /api/inference/run with job status, logs, and cancellation.", ["REST", "jobs", "health"], "deploy.api"],
      ["Model contract", "docs", "roadmap", "Generate request/response schema from checkpoint feature and condition metadata.", ["schema", "validation", "examples"]],
      ["Deployment audit", "system", "roadmap", "Retain versions, health, requests, resource limits, and rollback evidence.", ["governance", "audit", "health"]]
    ]
  },
  system: {
    label: "System", icon: "system", color: "#5a746b", note: "Environments, GPUs, diagnostics",
    title: "System workspace", description: "Understand interpreters, environments, dependencies, CUDA devices, resources, and diagnostic health before running expensive work.",
    cards: [
      ["Installation health", "system", "native", "Registry, repository, entrypoint, imports, config examples, and native probe status.", ["live registry", "entrypoints", "imports"]],
      ["Environment routing", "system", "native", "Suite interpreter plus repository-root-relative backend interpreters.", ["Windows", "Ubuntu", "venv"]],
      ["GPU inventory", "system", "adapter", "Device model, VRAM, CUDA capability, allocated/reserved memory, and owning processes.", ["CUDA", "VRAM", "process"]],
      ["SimulGen parallel modes", "model", "native", "Single GPU, in-process DDP, and FSDP selection with GPU-count warnings.", ["single", "ddp", "fsdp"], "model.simulgenvae"],
      ["Config audit", "prepare", "native", "Parse and validate every checked-in config with exact diagnostic codes, run below.", ["parse", "spec", "paths"]],
      ["Diagnostic layers", "system", "native", "Graph, config, route, filesystem, environment, dataset, checkpoint, native dry-run, and command.", ["deep links", "fix hints", "JSON"]]
    ]
  },
  docs: {
    // docCards: the badge follows the live document count when the server is up,
    // because that is what this section actually shows. The cards below render
    // only in the offline fallback.
    docCards: true,
    label: "Docs", icon: "docs", color: "#377264", note: "Methods, configs, datasets",
    title: "Documentation workspace", description: "Open contextual method, configuration, dataset, inference, benchmark, and implementation documentation without leaving Studio.",
    cards: [
      ["SimulGen-VAE README", "docs", "native", "Overview, architecture, training concepts, data conditioning, performance, and troubleshooting.", ["methods/SimulGenVAE/README.md", "method"], "model.simulgenvae"],
      ["SimulGen technical documentation", "docs", "native", "Hierarchical latent design, VAE/LC components, training pipeline, losses, and implementation notes.", ["docs/research/simulgenvae/TECHNICAL_DOCUMENTATION.md", "architecture"], "model.simulgenvae"],
      ["SimulGen checked configs", "docs", "native", "Merged train, VAE-only, LC-only, and reconstruction examples matching the live route.", ["4 modes", "flat config", "examples"], "model.simulgenvae"],
      ["Configuration reference", "docs", "native", "Suite-wide keys, modes, validation semantics, path resolution, and examples.", ["docs/CONFIGURATION.md", "spec"]],
      ["Dataset format", "docs", "native", "Shared mesh HDF5 groups, shapes, field conventions, splits, and metadata.", ["docs/reference/DATASET_FORMAT.md", "HDF5"]],
      ["Studio boundary", "docs", "native", "The Studio lives in studio/; narrow checkpoint-family and portable-routing compatibility fixes preserve truthful native integration.", ["studio/", "local bridge", "native routes"]]
    ]
  }
};

export const CONFIG_SECTIONS = ["Required", "Data & output", "Architecture", "Training", "Resources & runtime", "Inference & evaluation", "Optimization", "Advanced", "Inactive / rejected"];

export const HELP = {
  model: "Exact live registry ID. The display name may be friendlier, but saved config uses this route.",
  mode: "Only modes accepted by the selected live route are listed.",
  dataset_dir: "Input HDF5 path, resolved relative to the method repository by existing launcher rules.",
  gpu_ids: "One CUDA ID or a comma-separated list. SimulGen DDP/FSDP requires multiple IDs to distribute.",
  vae_modelpath: "VAE checkpoint path. SimulGen uses it across VAE/LC stages; SDFFlow uses it to decode latent shapes.",
  fm_modelpath: "SDFFlow flow-matching checkpoint. It defines the learned latent distribution used by sample, interpolate, and optimize modes.",
  lc_modelpath: "SimulGen latent-conditioner checkpoint output for train/train_lc and input for reconstruct.",
  num_filter_enc: "Space-separated progressive VAE encoder widths; hierarchical levels equal length minus one.",
  // HELP is keyed by config key, and several keys are shared across routes with
  // different meanings -- so shared entries name each meaning. The old
  // latent_dim text ("per-level hierarchical latent width") was true only for
  // SimulGen-VAE and shown, unchanged, on the other ten routes.
  latent_dim: "Hidden width. MeshGraphNets family: node/edge latent size; Transolver: token width (must be divisible by num_heads); SDFFlow: channels per latent token; SimulGen-VAE: per-level hierarchical latent width.",
  latent_dim_end: "Main SimulGen latent width.",
  lc_filter: "Space-separated latent-conditioner widths.",
  lc_data_type: "csv: MLP conditioner over an ordered CSV (param_dir); image: CNN conditioner over condition images (param_dir); hdf5: MLP conditioner over the dataset's own cond_var rows, no separate file.",
  param_dir: "CSV file or image directory ordered to match sorted HDF5 sample IDs.",
  field_start_row: "First physical-field row; the checked example leaves rows 0:3 for reference coordinates.",
  num_var: "Number of physical field components supplied to the VAE.",
  node_end: "Optional node slice end; zero means all nodes.",
  timesteps_reduced: "Optional timestep truncation; zero means the full trajectory.",
  loss_type: "1=MSE, 2=MAE, 3=SmoothL1, 4=Huber.",
  skip_completed_stages: "Reuse a complete stage only when its checkpoint metadata is configuration-compatible.",
  parallel_mode: "Backend validator filters valid modes and reports insufficient GPU counts.",
  learningr: "Native flat-config spelling for learning rate.",
  use_checkpointing: "Activation checkpointing trades additional compute for a lower training memory peak.",
  coordinate_normalization: "Coordinate transform recorded with the checkpoint and restored for inference.",
  _ddp_port: "Runtime-injected internal key, visible for completeness but normally not user-authored.",
  _pin_memory: "Runtime-injected internal key, visible for completeness but normally not user-authored.",
  flow_steps: "Inference ODE integration steps. This changes sampling cost and accuracy, not checkpoint architecture.",
  flow_solver: "Heun uses two velocity evaluations per step; Euler uses one.",
  flow_time_freqs: "Fourier time-embedding width. This is checkpoint architecture and must match training.",
  flow_t_sampling: "Training-time path sampling: uniform or middle-focused logitnormal.",
  flow_t_logit_scale: "Positive scale of the logit-normal training-time sampler.",
  flow_loss_weighting: "uniform predicts velocity uniformly; x0 emphasizes the deterministic path endpoint.",
  flow_det_prob: "Fraction of training graphs pinned to t=0. Must be in [0, 1).",
  flow_predict: "sample integrates one trajectory; mean is the deterministic t=0 readout; ensemble_mean averages generated draws.",
  num_vae_samples: "Number of stochastic field trajectories. Used by MeshGraphNets-V and cHI-MGNflow inference.",
  vae_batch_size: "How many stochastic trajectories are evaluated together; lower it to reduce inference memory.",
  infer_chunk_size: "Transolver inference chunk size. Leave blank to use the checkpoint/model configuration.",
  infer_query_chunk_size: "Neural-operator query decode chunk size. Leave blank for the checkpoint/model default.",
  val_flow_steps: "Cheaper ODE step count used during validation.",
  val_num_samples: "Number of validation ensemble members used for sampling metrics.",
  best_by: "Checkpoint selection metric: reconstruction, CRPS, or deterministic validation error.",
  opt_subspace_dim: "Number of orthonormal flow-noise directions exposed to the optimizer; a smaller subspace makes the search cheaper and easier to interpret.",
  opt_subspace_seed: "Seed that fixes the orthonormal latent search basis. Keep it fixed when comparing optimization runs.",
  opt_condition_dims: "Comma-separated trained FM condition names that become design variables, such as volume and area.",
  opt_shell_scale: "Maximum latent-noise shell radius multiplier used to keep candidates near the learned distribution.",
  opt_latent_range: "Per-coordinate bound for latent search variables inside the selected subspace.",
  opt_target_faces: "Target surface-face count used for each search-stage FEA mesh.",
  opt_mesh_size_max: "Maximum gmsh element size, in normalized geometry units, during the search stage.",
  opt_load_cases: "Comma-separated structural load cases evaluated for every candidate: vertical, horizontal, diagonal, or torsion.",
  opt_length_scale: "Metres per normalized geometry unit; required to convert generated shapes into physical FEA dimensions.",
  opt_material_e: "Young's modulus in pascals for the structural solve.",
  opt_material_nu: "Poisson ratio for the structural material model.",
  opt_material_rho: "Material density in kg/m^3, used to compute mass.",
  opt_yield_stress: "Yield stress in pascals used for the stress feasibility constraint.",
  opt_stress_percentile: "Nodal von Mises percentile treated as peak stress; a percentile is more robust than a single-node maximum.",
  opt_baseline_size: "Number of random generated designs used to calibrate active stress and displacement allowables.",
  opt_stress_margin: "Multiplier applied to the baseline peak-stress allowance; values below one are stricter.",
  opt_disp_margin: "Multiplier applied to the baseline displacement allowance; values below one are stricter.",
  opt_stress_weight: "Quadratic objective penalty for exceeding the stress allowance.",
  opt_disp_weight: "Quadratic objective penalty for exceeding the displacement allowance.",
  opt_budget: "Maximum number of generate-mesh-solve evaluations in the CMA-ES search.",
  opt_popsize: "CMA-ES population evaluated per generation.",
  opt_sigma0: "Initial CMA-ES step size in normalized design coordinates.",
  opt_seed: "Seed for the baseline population and CMA-ES trajectory.",
  opt_analysis: "Analysis backend for optimize mode. fea runs the exact mesh-and-solve path; surrogate runs the faster HI-MGN approximation and must not be treated as structurally trustworthy until its checkpoint is validated on representative held-out data.",
  opt_surrogate_checkpoint: "HI-MGN checkpoint used only when opt_analysis is surrogate. It must match the model family and feature contract named by opt_surrogate_config.",
  opt_surrogate_config: "HI-MGN inference config used only when opt_analysis is surrogate. It defines the dataset schema, variables, normalization, graph construction, and rollout settings expected by the checkpoint.",
  opt_surrogate_target_nodes: "Approximate surface-node count sampled for each surrogate evaluation. Larger values preserve more geometric detail but increase graph construction and inference cost.",
  opt_verify_resolution: "Marching Cubes resolution used to regenerate the final winner for verification.",
  opt_verify_target_faces: "Surface-face target for the final refined verification mesh.",
  opt_verify_mesh_size_max: "Maximum gmsh element size for the final refined verification solve.",

  // --- Shared data contract and training keys -------------------------------
  // These are the rows a user sees on every model block and in every "Required"
  // tab, and none of them had help: the sheet fell back to "Manual input is
  // retained because the live spec does not publish a closed value set".
  cond_var: "Trailing input-only rows after the state rows: conditions the model reads (boundary, flight or process parameters) but never predicts. They get their own normalization and are carried unchanged through a rollout. 0 reproduces the pre-conditioning behaviour.",
  feature_loss_weights: "One weight per output_var channel, comma-separated; scales each channel's share of the loss.",
  positional_features: "Number of positional feature channels appended to every node input (encoded coordinates).",
  infer_timesteps: "Rollout length in steps. For a dataset with T timesteps the paired training config uses T-1: the first frame is the initial condition.",
  time_integration: "ar_ot: one-step, teacher-forced training. ar_rt: full-trajectory autoregressive rollout training, slower per epoch but trained against its own compounding error.",
  split_seed: "Seed of the deterministic train / validation / test split.",
  warmup_epochs: "Epochs of linear learning-rate warm-up before the schedule takes over.",
  weight_decay: "Optimizer L2 weight decay.",
  use_amp: "Mixed-precision training: bfloat16 where the GPU supports it, else float16.",
  use_ema: "Keep an exponential moving average of the weights and validate and checkpoint with it.",
  ema_decay: "EMA decay per step. 0.99 to 0.999 is typical; higher is smoother and slower to follow.",
  val_interval: "Run validation every N epochs.",
  num_workers: "DataLoader worker processes. The mesh loaders here are input-bound, so more than 2 to 4 rarely helps.",
  grad_accum_steps: "Micro-batches accumulated before each optimizer step; the effective batch is batch_size times this.",

  // --- MeshGraphNets family -------------------------------------------------
  edge_var: "Edge feature count. Fixed at 8 for the MeshGraphNets family (mesh- and world-space relative position plus their norms); MGN-EDGE-001 rejects any other value.",
  use_multiscale: "True builds the HI-MGN V-cycle hierarchy and makes coarsening_type, multiscale_levels, voronoi_clusters and mp_per_level mandatory. False runs the flat stack sized by message_passing_num.",
  message_passing_num: "Message-passing blocks in the flat model. Ignored when use_multiscale is True; mp_per_level sizes each level instead.",
  coarsening_type: "How each coarser level is built: bfs, voronoi_centroid, voronoi_inherit or voronoi_seedmean (what every checked-in HI-MGN config uses). The bare 'voronoi' alias was removed and the native build raises on it.",
  multiscale_levels: "Number of coarser levels below the fine mesh. mp_per_level then needs 2 x levels + 1 entries; voronoi_clusters one value or one per level.",
  voronoi_clusters: "Target node count per coarse level, e.g. 500, 100 for two levels on a 3131-node mesh. A single value is reused for every level.",
  mp_per_level: "Message-passing blocks per V-cycle stage, listed pre-levels, coarsest, post-levels: 3, 4, 6, 4, 3 for two levels.",
  hierarchy_variants: "How many independently seeded coarsening partitions to cache per sample; training rotates through them per epoch as augmentation. 1 disables that.",
  hierarchy_seed: "Seed of the coarsening partition. Pinning it makes inference rebuild the hierarchy the model trained on; unseeded, about half the partition differs between runs.",
  hierarchy_cache_keep: "Keep the on-disk hierarchy cache (dataset/<name>.mscache.*) after the run instead of deleting it.",
  vae_latent_dim: "MeshGraphNets-V: width of the per-graph stochastic latent sampled at every rollout step.",

  // --- Transolver -----------------------------------------------------------
  num_layers: "Transolver blocks.",
  num_heads: "Attention heads. latent_dim must be divisible by it (TRANS-HEADS-001).",
  slice_num: "Physics-Attention slices per head. Attention memory scales with heads x nodes x slice_num per layer, so this, not latent_dim, drives VRAM.",
  attention_kernel: "naive materializes the full attention; slice_space streams it and is required by chunk_size, amortized_training and node_shard.",

  // --- MLP -------------------------------------------------------------------
  hidden_layers: "Comma-separated hidden layer widths, e.g. 256,256,128.",
  activation: "Hidden-layer activation.",

  // --- Neural operators -----------------------------------------------------
  fno_grid_resolution: "Latent grid size per spatial axis: 2 or 3 entries matching the data's dimensionality (ex9 is 2-D: 64, 32). Mesh values are interpolated onto this grid.",
  fno_modes: "Retained Fourier modes per axis, same entry count as fno_grid_resolution. Each must be at most grid/2 (last axis grid/2 + 1).",
  fno_hidden_channels: "Channel width of the spectral layers.",
  fno_layers: "Number of spectral convolution layers.",
  gino_grid_resolution: "Latent grid per axis for the FNO core; the graph-kernel encoder maps mesh points onto it and the decoder maps back. 2 or 3 entries.",
  gino_fno_modes: "Fourier modes per axis for the latent FNO core; same rules as fno_modes.",
  gino_fno_hidden_channels: "Channel width of the latent FNO core.",
  gino_fno_layers: "Spectral layers in the latent FNO core.",
  gino_kernel_hidden: "Hidden width of the graph-kernel MLPs that move features between mesh points and the grid.",
  gino_in_radius: "Neighbourhood radius, in normalized coordinates, for the mesh-to-grid kernel integration. Too small leaves grid cells empty (see gino_max_empty_input_fraction).",
  gino_out_radius: "Neighbourhood radius, in normalized coordinates, for the grid-to-query kernel integration.",
  deeponet_sensor_resolution: "Fixed sensor grid per axis at which the branch net samples the input field (2 or 3 entries).",
  deeponet_hidden_channels: "Width of the branch and trunk MLPs.",
  deeponet_branch_depth: "Branch-net layers.",
  deeponet_trunk_depth: "Trunk-net layers.",
  deeponet_basis_dim: "Number of basis functions the branch and trunk outputs are contracted over (the DeepONet p).",
  point_sensor_count: "Points sampled per sample as the branch input; point_resample_each_epoch redraws them every epoch.",
  point_hidden_channels: "Width of the point encoder and trunk MLPs.",
  point_feature_dim: "Pooled per-sample feature width produced by the PointNet branch.",
  pointnet_depth: "PointNet layers before pooling.",
  point_trunk_depth: "Trunk layers evaluated at each query point.",

  // --- SimulGen-VAE -----------------------------------------------------------
  output_dir: "Directory for periodic reconstruction / generation test artifacts and for reconstruct- and sample-mode outputs.",
  node_start: "Optional node-axis slice start (0 = first node); pairs with node_end.",
  network_size: "small or large: preset width of the VAE encoder / decoder blocks.",
  alpha: "SimulGen-VAE: reconstruction-loss weight relative to the KL term (1e6 in the checked configs). SDFFlow interpolate: position in [0, 1] between the two endpoints -- of the FM source noise under interpolation_space slerp_noise (the default, so the result is an on-manifold sample and its latent distances to the endpoints are not symmetric), of the normalized latents under lerp_latent.",
  init_beta_divisor: "Initial KL weight is 10^-divisor; it warms up to beta_target over kl_warmup_epochs.",
  vae_training_epochs: "Epochs for the VAE stage of the merged pipeline. The standalone train_vae mode uses training_epochs instead.",
  vae_learningr: "Learning rate for the VAE stage.",
  lc_dropout: "Dropout in the latent conditioner.",
  use_spatial_attention: "1 enables spatial attention in the image (CNN) conditioner; inert for csv and hdf5 conditions.",
  lc_training_epochs: "Epochs for the latent-conditioner stage of the merged pipeline. The standalone train_lc mode uses training_epochs.",
  lc_batch_size: "Batch size for the latent-conditioner stage.",
  lc_learningr: "Learning rate for the latent-conditioner stage.",

  // --- SDFFlow ---------------------------------------------------------------
  latent_tokens: "Latent tokens per shape. 1 is a single global code (DeepSDF-style, pairs with decoder_type mlp); more than 1 pairs with decoder_type attention (VecSet-style).",
  use_conditions: "Condition the flow-matching stage on the shape descriptors in condition_names. Off, the generator is unconditional and cond_values are ignored.",
  cond_dropout: "Fraction of FM batches trained with the null condition. Required when use_conditions is True, and what makes classifier-free guidance (cfg_scale) possible.",
  condition_names: "Subset and order of the dataset's cond_names attribute used as FM conditions. cond_values at sample time follow this order.",
  num_encoder_points: "Surface points (with normals) drawn per shape, per epoch, for the VAE encoder. Keep it BELOW the number the dataset stores: set equal to it, the without-replacement draw is just a permutation of one fixed point set, and because the encoder is permutation-invariant the shape then presents a bit-identical input every epoch -- which removes the only surface augmentation. The v3 recipe draws 6144 of the 8192 stored.",
  num_query_points: "SDF query points per shape used to supervise the decoder.",
  decoder_type: "mlp: DeepSDF-style coordinate MLP. attention: VecSet-style cross-attention over the latent tokens.",
  decoder_hidden: "Decoder hidden width (mlp) or attention d_model.",
  decoder_layers: "MLP layers (mlp) or cross-attention blocks (attention).",
  encoder_dim: "Point-cloud cross-attention encoder d_model.",
  encoder_heads: "Encoder cross-attention heads.",
  encoder_blocks: "Encoder cross-attention blocks.",
  fm_hidden: "Velocity-network hidden width.",
  fm_blocks: "AdaLN-Zero residual blocks in the velocity network.",
  fm_cond_hidden: "Timestep and condition embedding width.",
  fm_training_epochs: "Epochs for the flow-matching stage of the merged pipeline. The standalone train_fm mode uses training_epochs.",
  fm_batch_size: "Latents per step in the flow-matching stage.",
  fm_learningr: "Learning rate for the flow-matching stage.",
  num_samples: "Shapes to generate in sample mode.",
  seed: "Seed for the latent noise, and for the CMA-ES trajectory in optimize mode. In the training modes it is the run seed: each rank seeds torch/numpy/python and its train DataLoader shuffle with seed + rank, while model construction is put back on the rank-independent base seed. Leave it out for the legacy unseeded run -- but then two runs of the same config differ, and a sweep has no noise floor to judge its arms against.",
  ode_steps: "Integration steps for flow-matching sampling. A sampling-time cost, not architecture: the same checkpoint runs at any step count.",
  mc_resolution: "Marching-cubes grid resolution for the output surface; cost grows with the cube of this.",
  input_mesh: "Mesh file to encode and reconstruct in reconstruct mode.",
  source_num_samples: "Number of latents drawn before sample_index_a and sample_index_b pick the two to interpolate.",
  sample_index_a: "Index of the first drawn sample to interpolate from.",
  sample_index_b: "Index of the second drawn sample to interpolate to.",

  // --- Geometry ingest -------------------------------------------------------
  input_geometry: "A single CAD/mesh file, or a directory that is walked recursively for meshes. Resolved relative to the method repository like every other path key.",
  output_dataset: "Destination HDF5 written by `ingest`. `inspect` is a dry run and writes nothing, so this key is only emitted for `ingest`.",
  reader: "auto picks gmsh for CAD (STEP/IGES/BREP) and trimesh for meshes (STL/OBJ/PLY). trimesh cannot read CAD at all; gmsh can do both but is slower.",
  mesh_type: "surface keeps the boundary triangles; volume asks gmsh for a tetrahedral mesh, which only works for watertight solids.",
  emit: "Which contract to write: graph (nodes + mesh_edge, for MeshGraphNets) or pointcloud (sampled points, for the operators and Transolver). Both can be written at once.",
  num_fields: "How many zero-filled field rows to allocate after the coordinates, so the file matches the shared dataset contract a model expects.",
  num_points: "Points sampled per shape for the point-cloud output. Ignored when emit is graph only.",
  resample_method: "fps spreads points evenly by farthest-point sampling; random is cheaper and clumpier.",
  mesh_size_min: "gmsh minimum element size, in the geometry's own units. 0 leaves gmsh's default.",
  mesh_size_max: "gmsh maximum element size, in the geometry's own units. This is the main control over mesh density; 0 leaves gmsh's default.",
  limit: "Process at most this many input files. 0 means all of them.",

  // --- SDFFlow v3 encoder / split / evaluate ---------------------------------
  encoder_query_type: "learned uses trained query vectors; fps uses farthest-point-sampled input points as the queries. fps makes the latent token ORDER depend on the input point set, so pair it with decoder_type attention -- the flattened-token mlp decoder would see an input-dependent channel layout (SDF-QUERY-002).",
  posterior_min_std_rel: "Floor on the posterior standard deviation, relative to the latent scale. 0 disables the floor.",
  vae_best_modelpath: "Where the best-validation VAE checkpoint is written, alongside the last-epoch vae_modelpath.",
  split_by_parent: "Keep every shape derived from one parent geometry on the same side of the train/val/test split. Off, near-duplicate variants of the same part land on both sides and the held-out score is optimistic.",
  eval_split: "Which split to score: train, val, or test. Use val for model selection and touch test once, for the final winner.",
  eval_num_shapes: "How many shapes of the split to score. 0 scores all of them; N scores a random subset of that size, drawn with eval_seed -- not the first N, because under split_by_parent the split is ordered parent by parent and a head slice would be one or two part families.",
  eval_seed: "Seed for the deterministic encoder subsample drawn from each shape.",
  latent_refine_steps: "Adam steps that optimize the latent against the shape's stored SDF labels with the decoder frozen. 0 reports the encoder mean only; above 0 the stored query points are halved -- refinement fits one half and BOTH the encoder (enc_*) and refined (ref_*) metrics are scored on the other -- so the gap separates encoder error from decoder capacity without measuring fit. Expect near-zero movement on an undertrained checkpoint, whose decoder is barely z-sensitive.",
  latent_refine_lr: "Adam learning rate for that latent refinement.",
  latent_refine_prior_weight: "Weight on ||z - z_encoder||^2 during refinement -- summed over the latent scalars and averaged over the batch, so it keeps its meaning as the latent grows. Keeps the refined latent near the encoder's estimate; calibrate against latent_shift_l2 in the per-shape rows. 0 lets the latent drift freely.",
  interpolation_space: "slerp_noise interpolates the FM source noise spherically, so the endpoints reproduce the original samples. lerp_latent is the legacy straight line in normalized latent space. cond_sweep keeps ONE noise row (sample_index_a) fixed and integrates it under sweep_steps conditions lerped from cond_values_a to cond_values_b -- the controllable morph a designer asks for (lighter, stiffer, ...), scored per panel by the geometric audit.",

  // --- SDFFlow conditional generation ---------------------------------------
  cond_dropout_mode: "all (legacy): one Bernoulli mask per sample and a learned null embedding -- every condition must be given at inference. per_dim: an independent mask per condition entry with a learned null value and the mask fed to the network, so a sample request may leave entries 'nan' (unspecified) and the model fills them in from the ones given. Stored in the FM checkpoint; the two modes do not share parameters.",
  cond_values_a: "Condition vector at the START of a cond_sweep, in the FM checkpoint's cond_names order and stored space (natural log for log_* FEA names). 'nan' = unspecified (per_dim checkpoints only).",
  cond_values_b: "Condition vector at the END of a cond_sweep; same order and length as cond_values_a. The sweep is a straight line between the two in normalized condition space.",
  sweep_steps: "Panels in a cond_sweep: alphas linspace(0, 1, sweep_steps); each is integrated from the same noise row and written as sample_<seed>_sweep_<k>.stl plus one strip PNG. Fewer than 2 shows no morph.",
  guidance_enabled: "C2: calibrated endpoint-prediction guidance during the FM ODE. After each Euler step the one-step endpoint x1_hat is decoded through the soft SDF proxy and the latent is nudged toward the requested volume/area (in calibrated proxy units). Needs descriptor_calibration_path and a conditional request (cond_values). Off by default; measured on the ex1 pilot only.",
  guidance_t_start: "Guidance acts on Euler states with t_start <= t < 1; the endpoint estimate is unreliable at small t, so the early trajectory is left alone. Pilot value 0.3.",
  guidance_eta: "Guidance strength (> 0). The per-sample RMS-normalized gradient is scaled by eta * (1 - t). Pilot value 0.1.",
  guidance_step_mode: "velocity_dt: the correction is a velocity integrated over dt, so the total strength does not change with ode_steps (equals the pilot at 50 steps). per_step_jump: the pilot's per-step state jump, whose total grows with ode_steps -- only for reproducing the 50-step pilot.",
  guidance_targets: "Which requested conditions guidance / Newton act on. Only volume and area have a soft SDF proxy; other names are ignored with a note, and FEA-named conditions are measured by condition_audit instead.",
  soft_descriptor_resolution: "Cell-centre grid resolution of the differentiable soft volume/area proxy (48 in the pilot). Must match the value the calibration was fitted with -- the calibration file refuses a mismatch.",
  soft_descriptor_tau: "Sigmoid temperature of the soft occupancy, in normalized SDF units (0.032 in the pilot). Must match the calibration.",
  descriptor_calibration_path: "The affine calibration proxy = a * true + b of the soft proxy against the real Marching Cubes measurement, fitted per descriptor on the val split by evaluate + eval_task descriptor_calibration. Bound to the exact VAE/FM pair and proxy settings (SHA-256 checked). Read by sample guidance/Newton and by eval_task conditional; written by the calibration task.",
  newton_rounds: "E2: rounds of proxy-Jacobian Newton correction applied to each retained latent after sampling. Each round measures the decoded mesh, takes a damped least-squares step toward the requested volume/area, and keeps it only if the TRUE relative residual drops. 0 = off. Pilot: 3 rounds, volume median error 7.6% -> 0.28%.",
  newton_step_cap_rms: "Cap on one Newton step, as coordinate RMS in the normalized flat latent (cap = value * sqrt(latent_flat_dim)). 0.12 in the pilot; the uncapped step collapsed shapes.",
  newton_line_search_tries: "Backtracking tries per round: dz, dz/2, dz/4 for 3. A step is accepted only if the decoded mesh is valid and the true residual norm decreases.",
  newton_measure_resolution: "Marching Cubes resolution of the true measurement inside the Newton loop. Defaults to mc_resolution, and that is what it should stay: the calibration slope is fitted against the export path at mc_resolution and the audit reports there, so a different grid makes E2 converge a residual it is not scored on (MC volume moves ~0.05-0.1% between res 96 and 128). DescriptorCalibration.check_compatible refuses a mismatched artifact.",
  condition_audit: "How decoded meshes are re-measured against the request. geometric: volume/area/bbox from the mesh (always runs). fea: also gmsh-mesh and solve the GE load cases with design_loop (needs gmsh + pyamg). surrogate: design_loop's HI-MGN surrogate (needs opt_surrogate_config / opt_surrogate_checkpoint). fea/surrogate fall back to geometric with one printed message when unavailable; the metadata records which backend ran. Both are relative-only: tet4 is stiff and the solver reports a stress percentile.",
  eval_task: "reconstruction: held-out VAE reconstruction metrics (default). descriptor_calibration: fit and write descriptor_calibration_path from calibration_num_shapes x calibration_samples_per_shape samples of the split's conditions. conditional: paired-noise condition-accuracy benchmark of eval_methods against the split's true conditions (needs fm_modelpath).",
  eval_methods: "Comma list for eval_task conditional, subset of plain, rejection, c2, e2, c2e2. All methods start from the same seeded noise per shape; rejection uses candidate_multiplier, c2 the guidance_* keys, e2 the newton_* keys (both need descriptor_calibration_path).",
  calibration_num_shapes: "Shapes of eval_split whose true conditions seed the calibration samples (eval_task descriptor_calibration).",
  calibration_samples_per_shape: "Samples generated per calibration shape; the affine fit sees calibration_num_shapes x this many (proxy, true) pairs per descriptor.",
  cond_dropout_all_prob: "per_dim only: probability that a training row has EVERY condition masked, i.e. how often the classifier-free-guidance unconditional branch is trained. Independent per-entry dropout alone would draw that row with probability cond_dropout ** cond_dim (6e-5 for six conditions at 0.2), leaving the CFG branch near initialization. 0 restores the chance-only behaviour, and cfg_scale must then stay 1.0.",
  calibration_min_r2: "eval_task descriptor_calibration: refuse to save a calibration whose per-descriptor fit is weaker than this (0 disables). C2/E2 apply proxy_target = a * true + b, so a poorly determined slope makes the correction worse rather than weaker -- the pilot's uncalibrated round 1 took volume error from 7.6% to 23%.",
  eval_exclude_shapes: "Comma list of HDF5 shape indices to drop from the evaluate pool before the seeded subset is drawn. DeepJEB index 2099 (131_561) is a partial STL carrying full-bracket labels: no generator can hit its conditions, and at n=32 one such shape dominates the p95 column of every method."

};

export const INPUT_SOURCE_META = {
  "source.cad": {
    key: "path",
    kind: "geometry",
    label: "CAD / mesh input",
    // Mirrors _MESH_GLOBS in studio_backend/paths.py exactly: .msh was offered
    // and cannot be read, .brep is read and was not offered.
    accept: ".stl,.ply,.obj,.off,.step,.stp,.igs,.iges,.brep"
  },
  "source.hdf5": {
    key: "path",
    kind: "dataset",
    label: "HDF5 input",
    accept: ".h5,.hdf5"
  },
  "source.parameters": {
    key: "binding",
    kind: "dataset",
    label: "Parameter input",
    accept: ".csv,.json"
  },
  "source.checkpoint": {
    key: "path",
    kind: "checkpoint",
    label: "Saved model input",
    accept: ".pth,.pt,.ckpt"
  }
};

/* Manual zoom stops at 45%: below that a node's text is unreadable, so stepping
   there with +/- is never what someone wants. "Fit graph" is the exception --
   it is a request to see the whole pipeline, and refusing to zoom out is how it
   used to leave blocks stranded under the toolbar. Saved views are clamped to
   the fit range so a fitted workspace reloads exactly as it was left. */
export const MIN_ZOOM = .45;
export const FIT_MIN_ZOOM = .22;
export const MAX_ZOOM = 1.6;

export const NODE_WIDTH = 272;
export const NODE_HEADER_HEIGHT = 58;
export const NODE_PREVIEW_HEIGHT = 105;
export const PORT_START_Y = 183;
export const PORT_GAP = 30;
export const NODE_FOOTER_HEIGHT = 78;
