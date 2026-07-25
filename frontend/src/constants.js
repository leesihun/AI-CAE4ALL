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
  meshgraphnets: keys(`alpha_recon augment_geometry batch_size beta_aux coarse_cache_per_worker coarse_world_edges coarsening_type dataset_dir display_testset display_trainset edge_var ema_decay eval_dataset feature_loss_weights fit_latent_gmm free_bits gmm_components gmm_covariance_type gmm_reg_covar gpu_ids grad_accum_steps hierarchy_cache_build_workers hierarchy_cache_dir hierarchy_cache_wait_timeout histogram_bins histogram_clip_quantile infer_dataset infer_timesteps inference_output_dir input_var lambda_det lambda_kl lambda_mmd latent_dim learningr log_file_dir make_histogram message_passing_num mmd_bandwidth mode model modelpath mp_per_level multiscale_levels noise_gamma noise_std_ratio num_vae_samples num_workers num_z output_var parallel_mode pipeline_microbatches plot_feature_idx positional_features posterior_min_std prefetch_factor prior_batch_size prior_cov_rank prior_diagnose_interval prior_epochs prior_family prior_fm_steps prior_hidden_dim prior_kl_reg_weight prior_learningr prior_loss_type prior_mc_samples prior_min_std prior_mixture_components prior_mp_layers prior_nll_weight prior_num_workers prior_temperature prior_type prior_val_interval profile_batches recon_loss resume_prior show_histogram split_seed static_cache_per_worker std_noise test_batch_idx test_interval test_max_batches time_integration train_conditional_prior train_eval_subset_size training_epochs use_amp use_checkpointing use_compile use_conditional_prior use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_batch_size vae_batch_size_max vae_batch_size_min vae_batch_vram_fraction vae_graph_aware vae_latent_dim vae_mp_layers vae_valid_prior_samples val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier`),
  meshgraphnetsV: keys(`alpha_prior_max alpha_recon augment_geometry batch_size beta_aux bipartite_unpool coarse_world_edges coarsening_type dataset_dir display_testset display_trainset edge_var ema_decay eval_dataset feature_loss_weights fit_latent_gmm free_bits gmm_components gmm_covariance_type gmm_reg_covar gpu_ids grad_accum_steps hierarchy_cache_build_workers hierarchy_cache_dir hierarchy_cache_wait_timeout histogram_bins histogram_clip_quantile infer_dataset infer_timesteps inference_output_dir input_var lambda_det lambda_kl lambda_mmd latent_dim learningr log_file_dir make_histogram message_passing_num mmd_bandwidth mode model modelpath mp_per_level multiscale_levels noise_gamma noise_std_ratio num_vae_samples num_workers num_z output_var parallel_mode pipeline_microbatches plot_feature_idx positional_encoding positional_features posterior_min_std prefetch_factor prior_cov_rank prior_family prior_fm_steps prior_hidden_dim prior_kl_reg_weight prior_min_std prior_mixture_components prior_mp_layers prior_nll_weight prior_temperature prior_type recon_loss residual_scale show_histogram split_seed static_cache_per_worker std_noise test_batch_idx test_interval test_max_batches time_integration training_epochs use_amp use_checkpointing use_compile use_conditional_prior use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_batch_size vae_batch_size_max vae_batch_size_min vae_batch_vram_fraction vae_graph_aware vae_latent_dim vae_mp_layers vae_valid_prior_samples val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier`),
  operator: keys(`_ddp_port _pin_memory augment_geometry batch_size checkpoint_interval coarse_cache_per_worker coarse_world_edges coarsening_type coordinate_normalization dataset_dir deeponet_activation deeponet_basis_dim deeponet_branch_depth deeponet_branch_source deeponet_hidden_channels deeponet_max_branch_params deeponet_multi_output deeponet_sensor_resolution deeponet_trunk_depth dimension_tolerance display_testset display_trainset edge_var ema_decay feature_loss_weights fno_grid_resolution fno_hidden_channels fno_layers fno_modes fno_norm fno_use_channel_mlp fno_variant free_bits gino_cache_neighbors gino_channel_mlp_expansion gino_coord_embed_dim gino_fno_hidden_channels gino_fno_layers gino_fno_modes gino_grid_resolution gino_group_shared_geometry gino_in_radius gino_include_grid_coordinates gino_kernel_hidden gino_kernel_widths gino_lifting_hidden gino_max_empty_input_fraction gino_max_num_neighbors gino_out_radius gino_pos_embedding_type gino_projection_widths gino_query_chunk_size gino_tucker_rank gino_use_torch_cluster gino_variant global_condition_features gpu_ids grad_accum_steps grid_padding infer_dataset infer_query_chunk_size infer_timesteps inference_output_dir input_var integration_weight_source lambda_det lambda_kl lambda_mmd latent_dim learningr log_dir log_file_dir max_grad_norm message_passing_num mode model modelpath mp_per_level multiscale_levels noise_gamma noise_std_ratio num_node_types num_timesteps num_workers operator_dim out_of_bounds_policy output_var parallel_mode pipeline_microbatches plot_feature_idx point_branch_merge point_condition_depth point_feature_dim point_hidden_channels point_output_activation point_refiner_depth point_resample_each_epoch point_sampling point_sensor_count point_siren_omega0 point_trunk_depth point_variant pointnet_activation pointnet_depth pointnet_norm positional_features posterior_min_std prefetch_factor profile_batches sdf_sidecar sdf_source split_seed std_noise test_batch_idx test_interval test_max_batches time_integration train_query_chunk_size training_epochs use_amp use_checkpointing use_compile use_ema use_multiscale use_node_types use_vae use_world_edges vae_graph_aware vae_latent_dim vae_mp_layers val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier write_preprocessing`),
  transolver: keys(`attention_kernel augment_geometry batch_size chunk_size coarse_world_edges coarsening_type coordinate_normalization dataset_dir display_trainset dropout edge_var ema_decay feature_loss_weights gpu_ids grad_accum_steps infer_chunk_size infer_dataset infer_mode infer_timesteps inference_output_dir input_var latent_dim learningr log_file_dir max_grad_norm max_train_batches max_val_batches message_passing_num mlp_ratio mode model modelpath mp_per_level multiscale_levels noise_gamma num_heads num_layers num_workers output_var parallel_mode positional_features prefetch_factor slice_num small_output_init split_seed std_noise temperature_init temperature_max temperature_min test_interval test_max_batches time_integration training_epochs use_amp use_checkpointing use_compile use_ema use_multiscale use_node_types use_parallel_stats use_world_edges val_interval voronoi_clusters warmup_epochs weight_decay world_edge_backend world_max_num_neighbors world_radius_multiplier write_preprocessing write_test_predictions`),
  sdfflow: keys(`alpha batch_size candidate_multiplier cfg_scale clamp_dist cond_dropout cond_values condition_clip condition_names condition_ood_policy dataset_dir decoder_heads decoder_hidden decoder_layers decoder_type deterministic_warmup_epochs eikonal_weight ema_decay encode_batch_size encoder_blocks encoder_dim encoder_heads encoder_self_attention fm_arch fm_batch_size fm_blocks fm_cond_hidden fm_ema_decay fm_heads fm_hidden fm_learningr fm_log_file_dir fm_mc_resolution_test fm_modelpath fm_num_test_shapes fm_num_workers fm_test_interval fm_time_logit_mean fm_time_logit_std fm_time_sampling fm_training_epochs fm_use_amp fm_use_ema fm_val_interval fm_warmup_epochs fm_weight_decay fourier_bands fsdp_min_params gpu_ids hybrid_grad_points init_vae_modelpath input_mesh kl_warmup_epochs kl_weight latent_clip latent_dim latent_tokens learningr log_file_dir max_condition_z mc_resolution mc_resolution_test min_condition_std mode model normal_weight num_encoder_points num_query_points num_samples num_test_shapes num_workers ode_steps output_dir overfit_all_shapes overfit_num_shapes parallel_mode pipeline_log_file plot_dpi plot_max_faces posterior_noise_max_scale posterior_noise_warmup_epochs sample_index_a sample_index_b seed skip_completed_stages source_num_samples split_seed surface_weight test_interval training_epochs use_amp use_conditions use_ema vae_batch_size vae_ema_decay vae_learningr vae_log_file_dir vae_mc_resolution_test vae_modelpath vae_num_test_shapes vae_num_workers vae_test_interval vae_training_epochs vae_use_amp vae_use_ema vae_val_interval vae_warmup_epochs vae_weight_decay val_interval warmup_epochs weight_decay`),
  simulgenvae: keys(`alpha batch_size beta_target dataset_dir ema_decay field_start_row gpu_ids init_beta_divisor init_vae_modelpath kl_warmup_epochs kl_warmup_start_frac latent_dim latent_dim_end lc_batch_size lc_data_type lc_dropout lc_ema_decay lc_filter lc_learningr lc_log_file_dir lc_modelpath lc_num_workers lc_training_epochs lc_use_amp lc_use_ema lc_warmup_epochs lc_weight_decay learningr load_all log_file_dir loss_type mode model network_size node_end node_start num_filter_enc num_var num_workers output_dir parallel_mode param_data_type param_dir pipeline_log_file plot_mode recon_iter skip_completed_stages split_seed timesteps_reduced training_epochs use_amp use_ema use_spatial_attention vae_batch_size vae_ema_decay vae_learningr vae_log_file_dir vae_modelpath vae_num_workers vae_training_epochs vae_use_amp vae_use_ema vae_warmup_epochs vae_weight_decay val_interval warmup_epochs weight_decay`)
};

export const MODEL_CATALOG = {
  simulgenvae: {
    label: "SimulGen-VAE", short: "SGV", accent: "#a45b6e", modes: ["train", "train_vae", "train_lc", "reconstruct"],
    keys: KEY_CATALOGS.simulgenvae,
    description: "Hierarchical VAE plus CSV/image latent conditioner for parametric transient or static simulation fields.",
    dataset: "fixed-geometry mesh HDF5",
    defaults: {
      model: "simulgenvae", mode: "train", gpu_ids: "0", parallel_mode: "single",
      dataset_dir: "../dataset/ex1.h5", split_seed: "42", output_dir: "../output/simulgenvae/ex1",
      vae_modelpath: "../output/simulgenvae/ex1/simulgenvae_vae.pth",
      lc_modelpath: "../output/simulgenvae/ex1/simulgenvae_lc.pth",
      num_var: "1", field_start_row: "3", node_start: "0", node_end: "0", timesteps_reduced: "0",
      latent_dim: "8", latent_dim_end: "32", num_filter_enc: "1024 512 256 128",
      network_size: "small", loss_type: "1", alpha: "1000000", init_beta_divisor: "4",
      vae_training_epochs: "2000", vae_batch_size: "16", vae_learningr: "0.001",
      lc_filter: "32 64 128 256 512 1024", lc_data_type: "csv",
      param_dir: "../dataset/ex1_conditions.csv", lc_dropout: "0.2", use_spatial_attention: "1",
      lc_training_epochs: "5000", lc_batch_size: "64", lc_learningr: "0.001",
      skip_completed_stages: "True"
    }
  },
  mlp: {
    label: "Simple MLP", short: "MLP", accent: "#a66a45", modes: ["train", "inference"], keys: KEY_CATALOGS.mlp,
    description: "Tabular X[S,N] to Y[S,M] regression for global engineering responses.", dataset: "table HDF5",
    defaults: { model: "mlp", mode: "train", gpu_ids: "0", hidden_layers: "256,256,128", activation: "gelu", training_epochs: "200", batch_size: "32", learningr: "0.001" }
  },
  meshgraphnets: {
    label: "MeshGraphNets", short: "MGN", accent: "#19715e", modes: ["train", "inference"], keys: KEY_CATALOGS.meshgraphnets,
    description: "Deterministic graph-network simulator with Flat, HI-MGN, and BSMS-GNN presets.", dataset: "mesh HDF5",
    defaults: { model: "meshgraphnets", mode: "train", gpu_ids: "0", message_passing_num: "15", latent_dim: "128", use_multiscale: "False", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  "meshgraphnets-v": {
    label: "MeshGraphNets-V", short: "MGN-V", accent: "#407d69", modes: ["train", "inference"], keys: KEY_CATALOGS.meshgraphnetsV,
    description: "Variational mesh simulator with conditional priors and stochastic trajectories.", dataset: "mesh HDF5",
    defaults: { model: "meshgraphnets-v", mode: "train", gpu_ids: "0", latent_dim: "128", vae_latent_dim: "256", num_vae_samples: "32", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  point_deeponet: {
    label: "Point-DeepONet", short: "P-DON", accent: "#3d718d", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Point-conditioned operator for arbitrary-query field prediction.", dataset: "mesh HDF5",
    defaults: { model: "point_deeponet", mode: "train", gpu_ids: "0", point_sensor_count: "1024", point_hidden_channels: "256", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  deeponet: {
    label: "DeepONet", short: "DON", accent: "#526f9d", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Branch/trunk neural operator on the shared mesh dataset contract.", dataset: "mesh HDF5",
    defaults: { model: "deeponet", mode: "train", gpu_ids: "0", deeponet_hidden_channels: "256", deeponet_branch_depth: "4", deeponet_trunk_depth: "4", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  fno: {
    label: "FNO", short: "FNO", accent: "#6561a3", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Fourier Neural Operator with explicit grid and spectral-mode controls.", dataset: "mesh HDF5",
    defaults: { model: "fno", mode: "train", gpu_ids: "0", fno_grid_resolution: "64,64,64", fno_modes: "16,16,16", fno_hidden_channels: "64", fno_layers: "4", training_epochs: "500", batch_size: "4", learningr: "0.0001" }
  },
  gino: {
    label: "GINO", short: "GINO", accent: "#7d5c99", modes: ["train", "inference"], keys: KEY_CATALOGS.operator,
    description: "Geometry-informed neural operator for irregular domains.", dataset: "mesh HDF5",
    defaults: { model: "gino", mode: "train", gpu_ids: "0", gino_grid_resolution: "64,64,64", gino_fno_modes: "16,16,16", gino_in_radius: "0.05", gino_out_radius: "0.05", training_epochs: "500", batch_size: "2", learningr: "0.0001" }
  },
  transolver: {
    label: "Transolver3", short: "TR3", accent: "#9a5e55", modes: ["train", "inference"], keys: KEY_CATALOGS.transolver,
    description: "Physics-Attention with slice-space and node-sharded execution controls.", dataset: "mesh HDF5",
    defaults: { model: "transolver", mode: "train", gpu_ids: "0", latent_dim: "256", num_layers: "10", num_heads: "8", slice_num: "128", attention_kernel: "slice_space", use_checkpointing: "True", batch_size: "1", grad_accum_steps: "4", learningr: "0.0001" }
  },
  sdfflow: {
    label: "SDFFlow", short: "SDF", accent: "#8b7837", modes: ["train", "train_vae", "train_fm", "sample", "reconstruct", "interpolate"], keys: KEY_CATALOGS.sdfflow,
    description: "SDF-VAE and conditional flow matching for CAD candidate generation.", dataset: "SDF HDF5",
    defaults: { model: "sdfflow", mode: "train", gpu_ids: "0", latent_dim: "256", latent_tokens: "32", condition_names: "bbox_x,bbox_y,bbox_z,volume,area", vae_training_epochs: "500", fm_training_epochs: "500" }
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
    reconstruct: keys(`model mode gpu_ids vae_modelpath input_mesh output_dir mc_resolution`),
    interpolate: keys(`model mode gpu_ids vae_modelpath fm_modelpath output_dir seed source_num_samples sample_index_a sample_index_b alpha ode_steps mc_resolution`)
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
  lc_data_type: ["csv", "image"],
  loss_type: ["1", "2", "3", "4"],
  plot_mode: ["0", "1", "2"],
  parallel_mode: ["single", "ddp", "fsdp"],
  attention_kernel: ["naive", "slice_space"],
  infer_mode: ["direct", "decoupled"],
  coordinate_normalization: ["centered_isotropic"],
  time_integration: ["ar_ot", "ar_rt"],
  coarsening_type: ["none", "voronoi", "bfs"],
  condition_ood_policy: ["warn", "clip", "error"],
  decoder_type: ["cross_attention", "implicit"],
  fm_arch: ["transformer", "mlp"],
  fm_time_sampling: ["uniform", "logit_normal"],
  point_sampling: ["random", "fps"],
  out_of_bounds_policy: ["error", "clamp", "zero"],
  sdf_source: ["none", "sidecar"]
};

export const BOOLEAN_KEYS = new Set(keys(`augment_geometry bipartite_unpool coarse_world_edges display_testset display_trainset encoder_self_attention fit_latent_gmm fm_use_amp fm_use_ema gino_cache_neighbors gino_group_shared_geometry gino_include_grid_coordinates gino_use_torch_cluster load_all make_histogram overfit_all_shapes point_resample_each_epoch resume_prior show_histogram skip_completed_stages small_output_init train_conditional_prior use_amp use_checkpointing use_compile use_conditional_prior use_ema use_multiscale use_node_types use_parallel_stats use_vae use_world_edges vae_graph_aware vae_use_amp vae_use_ema lc_use_amp lc_use_ema write_preprocessing write_test_predictions`));
export const OPERATOR_REMOVED = new Set(keys(`message_passing_num latent_dim edge_var world_radius_multiplier world_max_num_neighbors world_edge_backend coarse_world_edges multiscale_levels mp_per_level coarsening_type voronoi_clusters coarse_cache_per_worker use_vae vae_latent_dim vae_mp_layers vae_graph_aware free_bits posterior_min_std lambda_mmd lambda_kl lambda_det`));
export const TRANSOLVER_REJECTED = new Set(keys(`edge_var message_passing_num mp_per_level coarsening_type voronoi_clusters multiscale_levels world_radius_multiplier world_max_num_neighbors world_edge_backend coarse_world_edges`));

export const BLOCK_SPECS = {
  "source.cad": {
    label: "CAD", category: "Sources", icon: "cad", accent: "#4c7f71", visual: "geometry", maturity: "native",
    description: "Select STEP, IGES, STL, PLY, or OBJ geometry and inspect every body.",
    inputs: [], outputs: [{ id: "geometry", type: "geometry", label: "geometry" }],
    defaults: { path: "dataset/cad/bracket.step", units: "mm" }, sampleLabel: "12 CAD bodies"
  },
  "source.hdf5": {
    label: "HDF5 Dataset", category: "Sources", icon: "data", accent: "#3c7193", visual: "dataset", maturity: "native",
    description: "Browse samples, fields, geometry, splits, statistics, and verified parameter bindings.",
    inputs: [{ id: "parameters", type: "parameters", label: "parameter overlay", required: false }],
    outputs: [{ id: "data", type: "dataset", label: "configured dataset" }],
    defaults: { path: "dataset/ex1.h5", split: "seeded 80/10/10", edit_mode: "immutable overlay" }, sampleLabel: "100 samples"
  },
  "source.parameters": {
    label: "Design Parameters", category: "Sources", icon: "parameters", accent: "#b0713f", visual: "parameters", maturity: "adapter",
    description: "Bind named scalar, vector, profile, CSV, or image conditions to compatible dataset inputs.",
    inputs: [], outputs: [{ id: "parameters", type: "parameters", label: "parameter overlay" }],
    defaults: { binding: "CSV condition columns", value: "0.50", range_policy: "warn outside training range" }, sampleLabel: "5 parameters"
  },
  "source.checkpoint": {
    label: "Saved ML Model", category: "Sources", icon: "model", accent: "#795991", visual: "checkpoint", maturity: "native",
    description: "Load a .pth checkpoint or compatible multi-stage model bundle with lineage.",
    inputs: [], outputs: [{ id: "model", type: "checkpoint", label: "saved model" }],
    defaults: { path: "output/model.pth", version: "best", compatibility: "auto-detect" }, sampleLabel: "1 model bundle"
  },
  "prep.geometry": {
    label: "Geometry → HDF5 Dataset", category: "Preparation", icon: "prepare", accent: "#29745f", visual: "dataset", maturity: "native",
    description: "Convert CAD or surface geometry to the shared graph or point-cloud HDF5 contract.",
    inputs: [{ id: "geometry", type: "geometry", label: "geometry", required: true }],
    outputs: [{ id: "data", type: "dataset", label: "dataset" }],
    defaults: {
      mode: "inspect", reader: "auto", mesh_type: "surface", emit: "graph",
      num_fields: "3", num_points: "4096", resample_method: "fps", seed: "42",
      output_dataset: "../../frontend/runtime/geometry-ingest/geometry.h5", limit: "0"
    },
    sampleLabel: "Geometry ingest results"
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
    defaults: { mode: "auto from saved model", batch_size: "16", viewer: "field + samples + distributions" }, sampleLabel: "20 reconstructions"
  },
  "run.cad_generator": {
    label: "CAD Generator", category: "Execution", icon: "run", accent: "#817336", visual: "candidates", maturity: "native",
    description: "Conditionally generate, reconstruct, or blend CAD candidates with SDFFlow.",
    inputs: [{ id: "parameters", type: "parameters", label: "design parameters" }, { id: "model", type: "checkpoint", label: "SDFFlow model", required: true }],
    outputs: [{ id: "candidates", type: "candidates", label: "CAD candidates" }],
    defaults: { mode: "conditional generation", candidates: "24", guidance: "2.5", geometry_checks: "connected + watertight + bounds" }, sampleLabel: "24 CAD candidates"
  },
  "optimize.design": {
    label: "Optimization", category: "Optimization", icon: "optimize", accent: "#8a613b", visual: "candidates", maturity: "adapter",
    description: "Evaluate designs with geometry gates, physics objectives, constraints, uncertainty, and Pareto trade-offs.",
    inputs: [
      { id: "candidates", type: "candidates", label: "CAD candidates", required: true },
      { id: "models", type: "checkpoint", label: "physics evaluators", required: false },
      { id: "parameters", type: "parameters", label: "search variables", required: false }
    ],
    outputs: [{ id: "selected", type: "candidates", label: "Pareto designs" }, { id: "metrics", type: "metrics", label: "objective table" }, { id: "report", type: "report", label: "optimization report" }],
    defaults: { mode: "evaluate fixed candidate batch", objectives: "min peak stress; min mass", constraints: "watertight; displacement ≤ limit", selection: "feasible Pareto + diverse top-k", top_k: "6" }, sampleLabel: "6 Pareto candidates"
  },
  "evaluate.predictions": {
    label: "Evaluate Predictions", category: "Evaluation", icon: "evaluate", accent: "#1f7c66", visual: "training", maturity: "adapter",
    description: "Compute compatible scalar and field metrics, distributions, and synchronized error views.",
    inputs: [{ id: "prediction", type: "field", label: "predictions", required: true }, { id: "truth", type: "dataset", label: "ground truth", required: true }],
    outputs: [{ id: "metrics", type: "metrics", label: "metrics" }, { id: "report", type: "report", label: "evaluation report" }],
    defaults: { metrics: "relative L2, MAE, RMSE", aggregate: "mean + median + p95", error_view: "absolute + relative" }, sampleLabel: "100 sample metrics"
  },
  "evaluate.compare": {
    label: "Compare Models", category: "Evaluation", icon: "evaluate", accent: "#5e6f90", visual: "training", maturity: "adapter",
    description: "Compare accuracy, latency, throughput, VRAM, size, uncertainty, and synchronized samples.",
    inputs: [{ id: "metrics", type: "metrics", label: "model runs", required: true }],
    outputs: [{ id: "report", type: "report", label: "comparison" }],
    defaults: { x: "relative L2", y: "throughput", qualification: "same held-out set" }, sampleLabel: "10 compared models"
  },
  "output.export": {
    label: "Export Results", category: "Outputs", icon: "output", accent: "#596560", visual: "export", maturity: "adapter",
    description: "Export selected datasets, fields, CAD, metrics, reports, or model files with provenance.",
    inputs: [{ id: "input", type: "artifact", label: "artifact", required: true }],
    outputs: [{ id: "files", type: "artifact", label: "files" }],
    defaults: { format: "auto: HDF5 / VTK / STL / CSV / JSON / HTML", path: "output/studio/run_001" }, sampleLabel: "13 files"
  },
  "deploy.api": {
    label: "API Deployment", category: "Deployment", icon: "output", accent: "#4d6874", visual: "export", maturity: "adapter",
    description: "Run the validated portable inference API or build the Windows inference executable.",
    inputs: [{ id: "model", type: "checkpoint", label: "saved model", required: true }],
    outputs: [{ id: "endpoint", type: "endpoint", label: "endpoint" }],
    defaults: { target: "local / server", device: "CPU / GPU", auth: "token", openapi: "enabled" }, sampleLabel: "3 revisions"
  }
};

Object.entries(MODEL_CATALOG).forEach(([modelId, model]) => {
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
    modelId,
    isModel: true
  };
});

export const TEMPLATES = {
  simulgen: {
    name: "SimulGen-VAE field reconstruction",
    nodes: [
      ["dataset", "source.hdf5", 35, 70, { path: "dataset/ex1.h5", compatibility: "fixed N and T required" }],
      ["conditions", "source.parameters", 35, 355, { binding: "dataset/ex1_conditions.csv" }],
      ["simulgen", "model.simulgenvae", 360, 145],
      ["inference", "run.inference", 705, 145, { mode: "SimulGen-VAE reconstruct" }],
      ["evaluation", "evaluate.predictions", 1050, 145],
      ["export", "output.export", 1395, 145]
    ],
    edges: [
      ["dataset", "data", "simulgen", "data"],
      ["conditions", "parameters", "simulgen", "parameters"],
      ["dataset", "data", "inference", "data"],
      ["simulgen", "model", "inference", "model"],
      ["conditions", "parameters", "inference", "parameters"],
      ["inference", "prediction", "evaluation", "prediction"],
      ["dataset", "data", "evaluation", "truth"],
      ["evaluation", "report", "export", "input"]
    ]
  },
  geometry: {
    name: "Geometry to HDF5 inspection",
    nodes: [
      ["cad", "source.cad", 35, 125, { path: "dataset/benchmarks/gino_carcfd/source/extracted/processed-car-pressure-data/data", units: "dataset native" }],
      ["ingest", "prep.geometry", 365, 125, { mode: "inspect", reader: "trimesh", mesh_type: "surface", emit: "graph, pointcloud", num_points: "4096", limit: "3" }],
      ["export", "output.export", 710, 125, { path: "frontend/runtime/geometry-ingest" }]
    ],
    edges: [
      ["cad", "geometry", "ingest", "geometry"],
      ["ingest", "data", "export", "input"]
    ]
  },
  physics: {
    name: "Physics from CAD",
    nodes: [
      ["dataset", "source.hdf5", 35, 125],
      ["trainer", "model.meshgraphnets", 330, 70],
      ["inference", "run.inference", 625, 70],
      ["evaluation", "evaluate.predictions", 920, 70],
      ["export", "output.export", 1215, 70]
    ],
    edges: [
      ["dataset", "data", "trainer", "data"],
      ["dataset", "data", "inference", "data"],
      ["trainer", "model", "inference", "model"],
      ["inference", "prediction", "evaluation", "prediction"],
      ["dataset", "data", "evaluation", "truth"],
      ["evaluation", "report", "export", "input"]
    ]
  },
  parametric: {
    name: "Parametric response estimation",
    nodes: [
      ["parameters", "source.parameters", 35, 295],
      ["dataset", "source.hdf5", 35, 55, { path: "dataset/mlp/train.h5" }],
      ["mlp", "model.mlp", 330, 115],
      ["inference", "run.inference", 625, 115],
      ["evaluation", "evaluate.predictions", 920, 115],
      ["export", "output.export", 1215, 115]
    ],
    edges: [
      ["parameters", "parameters", "dataset", "parameters"],
      ["dataset", "data", "mlp", "data"],
      ["parameters", "parameters", "mlp", "parameters"],
      ["dataset", "data", "inference", "data"],
      ["mlp", "model", "inference", "model"],
      ["parameters", "parameters", "inference", "parameters"],
      ["inference", "prediction", "evaluation", "prediction"],
      ["dataset", "data", "evaluation", "truth"],
      ["evaluation", "report", "export", "input"]
    ]
  },
  generative: {
    name: "Generative design optimization",
    nodes: [
      ["parameters", "source.parameters", 35, 300],
      ["generator_model", "model.sdfflow", 35, 55],
      ["generator", "run.cad_generator", 330, 125],
      ["physics_model", "source.checkpoint", 330, 390, { path: "checkpoints/physics_surrogate.pth" }],
      ["optimization", "optimize.design", 625, 125],
      ["export", "output.export", 920, 125]
    ],
    edges: [
      ["parameters", "parameters", "generator", "parameters"],
      ["generator_model", "model", "generator", "model"],
      ["generator", "candidates", "optimization", "candidates"],
      ["physics_model", "model", "optimization", "models"],
      ["parameters", "parameters", "optimization", "parameters"],
      ["optimization", "selected", "export", "input"]
    ]
  },
  blank: { name: "Untitled pipeline", nodes: [], edges: [] }
};

export const STUDIO_SECTIONS = {
  data: {
    label: "Data", icon: "data", color: "#3c7193", note: "Geometry, datasets, conditions",
    title: "Data workspace", description: "Curate every object entering a pipeline, inspect individual samples, preserve split provenance, and expose exact HDF5 contracts.",
    stats: [["2,138", "DeepJEB shapes"], ["3", "dataset contracts"]],
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
    stats: [["2", "active sessions"], ["44", "checkpoints"]],
    cards: [
      ["Training sessions", "evaluate", "native", "Live loss, throughput, GPU memory, checkpoints, logs, stop, resume, and block ownership.", ["durable jobs", "resume", "logs"]],
      ["SimulGen staged training", "model", "native", "Train the merged VAE → latent-conditioner pipeline or execute either stage independently.", ["train", "train_vae", "train_lc"], "model.simulgenvae"],
      ["Stage compatibility", "model", "native", "Reuse completed VAE/LC checkpoints only when stage metadata and configuration remain compatible.", ["skip completed", "VAE", "LC"], "model.simulgenvae"],
      ["Hyperparameter sweeps", "parameters", "adapter", "Schedule child configurations and compare validation, resource, and reconstruction metrics.", ["grid", "random", "parent/child"]],
      ["Resource and VRAM", "system", "adapter", "Measure real peak allocated memory, throughput, batch feasibility, and multi-GPU behavior.", ["peak allocated", "DDP", "FSDP"]],
      ["Cross-model comparison", "evaluate", "adapter", "Compare accuracy, latency, VRAM, parameter count, uncertainty, and sample views.", ["same test set", "qualification", "evidence"], "evaluate.compare"]
    ]
  },
  optimization: {
    label: "Optimization", icon: "optimize", color: "#8a613b", note: "Objectives, constraints, Pareto",
    title: "Generated-design evaluation and optimization", description: "Turn candidate galleries into engineering decisions while keeping geometry evidence, physics predictions, constraints, uncertainty, Pareto trade-offs, and verification separate.",
    stats: [["1", "real Pareto engine"], ["2", "future search layers"]],
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
    stats: [["5", "field metrics"], ["1", "per-sample CSV"]],
    cards: [
      ["Relative L2", "evaluate", "native", "Global per-sample field norm error with a finite-value mask and explicit field-row selection.", ["relative L2", "per sample", "actual arrays"], "evaluate.predictions"],
      ["Absolute errors", "evaluate", "native", "MAE, RMSE, and maximum absolute error over matched feature, timestep, and node dimensions.", ["MAE", "RMSE", "max error"], "evaluate.predictions"],
      ["Explained variance", "evaluate", "native", "R² with mean, median, p95, minimum, and maximum aggregation.", ["R²", "aggregate", "JSON"], "evaluate.predictions"]
    ]
  },
  comparison: {
    label: "Compare", icon: "evaluate", color: "#5e6f90", note: "Rank actual model results",
    title: "Cross-model comparison", description: "Rank actual benchmark or evaluation CSV rows by a chosen numeric metric while preserving the source row and evidence path.",
    stats: [["2", "directions"], ["200", "ranked rows"]],
    cards: [
      ["Qualified metric ranking", "evaluate", "native", "Choose the model column, metric column, and min/max direction from an actual output CSV.", ["CSV", "rank", "source row"], "evaluate.compare"],
      ["Benchmark handoff", "evaluate", "native", "Use checked comparison.csv artifacts from benchmark campaigns without copying values into the UI.", ["benchmark", "evidence", "JSON"], "evaluate.compare"]
    ]
  },
  export: {
    label: "Export", icon: "output", color: "#596560", note: "Copy or archive artifacts",
    title: "Artifact export", description: "Copy a selected real file or ZIP a selected artifact directory into the isolated frontend runtime and download it.",
    stats: [["1", "click download"], ["0", "backend rewrites"]],
    cards: [
      ["Export file", "output", "native", "Copy a real HDF5, CSV, JSON, checkpoint, or report while preserving the source artifact.", ["copy", "provenance", "download"], "output.export"],
      ["Export directory", "output", "native", "Create a ZIP archive under frontend/runtime/exports for multi-file inference and geometry outputs.", ["ZIP", "folder", "download"], "output.export"]
    ]
  },
  models: {
    label: "Models", icon: "model", color: "#76568e", note: "10 live model IDs",
    title: "Model and checkpoint workspace", description: "Browse every installed route, its real mode and configuration contract, checkpoint compatibility, dataset requirements, and intended engineering use.",
    stats: [["10", "trainable IDs"], ["28", "route modes"]],
    modelCards: true
  },
  benchmarks: {
    label: "Benchmarks", icon: "evaluate", color: "#8a613b", note: "Locked protocols and reports",
    title: "Benchmark workspace", description: "Run named, qualified protocols. Smoke, full, and paper-equivalent evidence remain visibly distinct.",
    stats: [["6", "benchmark families"], ["3", "qualification levels"]],
    cards: [
      ["Elasticity accuracy", "evaluate", "adapter", "Prepare, preflight, train, infer, evaluate relative L2, and compare supported models.", ["smoke", "full", "paper"]],
      ["Transient plasticity", "evaluate", "adapter", "Seeded split, exact test HDF5, model scheduling, rollout evaluation, and resource probes.", ["campaign", "rollout", "7 models"]],
      ["SimulGen reconstruction", "model", "adapter", "Evaluate fixed-geometry field compression/reconstruction MSE and condition-driven generation quality.", ["VAE", "LC", "field MSE"], "model.simulgenvae"],
      ["FNO Darcy", "evaluate", "adapter", "Own-paper and opt-in paper-validation protocols with relative L2 qualification.", ["FNO", "Darcy", "qualification"]],
      ["GINO CarCFD", "evaluate", "adapter", "Source verification, voxel/SDF preparation, training, and CarCFD evaluation.", ["GINO", "CarCFD", "provenance"]],
      ["Protocol reports", "output", "adapter", "Generate HTML, CSV, and JSON reports with run budgets and evidence attached.", ["traceable", "qualified", "portable"]]
    ]
  },
  artifacts: {
    label: "Artifacts", icon: "artifact", color: "#68736e", note: "Fields, geometry, models",
    title: "Artifact workspace", description: "Search typed outputs across pipelines, inspect individual samples, compare versions, and follow provenance to the producing block and run.",
    stats: [["837", "HDF5 artifacts"], ["44", "saved models"]],
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
    stats: [["8", "bundle families"], ["1", "local API contract"]],
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
    stats: [["11", "healthy routes"], ["6", "CUDA devices"]],
    cards: [
      ["Installation health", "system", "native", "Registry, repository, entrypoint, imports, config examples, and native probe status.", ["11 OK", "entrypoints", "imports"]],
      ["Environment routing", "system", "native", "Suite interpreter plus repository-root-relative backend interpreters.", ["Windows", "Ubuntu", "venv"]],
      ["GPU inventory", "system", "adapter", "Device model, VRAM, CUDA capability, allocated/reserved memory, and owning processes.", ["CUDA", "VRAM", "process"]],
      ["SimulGen parallel modes", "model", "native", "Single GPU, in-process DDP, and FSDP selection with GPU-count warnings.", ["single", "ddp", "fsdp"], "model.simulgenvae"],
      ["Config audit", "prepare", "native", "Parse and validate every checked-in config with exact diagnostic codes, run below.", ["parse", "spec", "paths"]],
      ["Diagnostic layers", "system", "native", "Graph, config, route, filesystem, environment, dataset, checkpoint, native dry-run, and command.", ["deep links", "fix hints", "JSON"]]
    ]
  },
  docs: {
    label: "Docs", icon: "docs", color: "#377264", note: "Methods, configs, datasets",
    title: "Documentation workspace", description: "Open contextual method, configuration, dataset, inference, benchmark, and implementation documentation without leaving Studio.",
    stats: [["12", "method guides"], ["67", "SimulGen keys"]],
    cards: [
      ["SimulGen-VAE README", "docs", "native", "Overview, architecture, training concepts, data conditioning, performance, and troubleshooting.", ["SimulGenVAE/README.md", "method"], "model.simulgenvae"],
      ["SimulGen technical documentation", "docs", "native", "Hierarchical latent design, VAE/LC components, training pipeline, losses, and implementation notes.", ["TECHNICAL_DOCUMENTATION.md", "architecture"], "model.simulgenvae"],
      ["SimulGen checked configs", "docs", "native", "Merged train, VAE-only, LC-only, and reconstruction examples matching the live route.", ["4 modes", "flat config", "examples"], "model.simulgenvae"],
      ["Configuration reference", "docs", "native", "Suite-wide keys, modes, validation semantics, path resolution, and examples.", ["CONFIGURATION_REFERENCE.md", "spec"]],
      ["Dataset format", "docs", "native", "Shared mesh HDF5 groups, shapes, field conventions, splits, and metadata.", ["DATASET_FORMAT.md", "HDF5"]],
      ["Frontend boundary", "docs", "native", "This application is self-contained in frontend/ and does not modify backend code.", ["frontend/", "static", "isolated"]]
    ]
  }
};

export const CONFIG_SECTIONS = ["Required", "Data & output", "Architecture", "Training", "Resources & runtime", "Inference & evaluation", "Advanced", "Inactive / rejected"];

export const HELP = {
  model: "Exact live registry ID. The display name may be friendlier, but saved config uses this route.",
  mode: "Only modes accepted by the selected live route are listed.",
  dataset_dir: "Input HDF5 path, resolved relative to the method repository by existing launcher rules.",
  gpu_ids: "One CUDA ID or a comma-separated list. SimulGen DDP/FSDP requires multiple IDs to distribute.",
  vae_modelpath: "SimulGen VAE checkpoint output for train/train_vae and input for train_lc/reconstruct.",
  lc_modelpath: "SimulGen latent-conditioner checkpoint output for train/train_lc and input for reconstruct.",
  num_filter_enc: "Space-separated progressive VAE encoder widths; hierarchical levels equal length minus one.",
  latent_dim: "Per-level hierarchical latent width.",
  latent_dim_end: "Main SimulGen latent width.",
  lc_filter: "Space-separated latent-conditioner widths.",
  lc_data_type: "csv selects the MLP conditioner; image selects the CNN conditioner.",
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
  _pin_memory: "Runtime-injected internal key, visible for completeness but normally not user-authored."
};

export const INPUT_SOURCE_META = {
  "source.cad": {
    key: "path",
    kind: "geometry",
    label: "CAD / mesh input",
    accept: ".stl,.step,.stp,.iges,.igs,.obj,.ply,.off,.msh"
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

export const NODE_WIDTH = 272;
export const NODE_HEADER_HEIGHT = 58;
export const NODE_PREVIEW_HEIGHT = 105;
export const PORT_START_Y = 183;
export const PORT_GAP = 30;
export const NODE_FOOTER_HEIGHT = 78;
