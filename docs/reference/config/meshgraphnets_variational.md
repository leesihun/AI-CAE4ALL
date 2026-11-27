# MeshGraphNets-V configuration reference

This page covers the keys that `model meshgraphnets-v` reads. The authority is
[ai_cae4all/specs/meshgraphnets_variational.py](../../../ai_cae4all/specs/meshgraphnets_variational.py):

- A key not listed there is a `CFG-UNKNOWN-001` warning.
- A key in `VAR_REMOVED_KEYS` is an error (`MGNV-REMOVED`).

The method overview is in
[04_MeshGraphNets_Variational.md](../../methods/04_MeshGraphNets_Variational.md).
The config grammar is in [CONFIGURATION.md](../../CONFIGURATION.md): lines of
`key value`, `%` comment lines and `#` inline comments, lowercased keys, and
comma lists.

Run every config through the launcher:

```bash
python AI_CAE4ALL_main.py --config <cfg> --check
python AI_CAE4ALL_main.py --config <cfg>
python AI_CAE4ALL_main.py --describe meshgraphnets-v   # required fields per mode
```

The native process runs with cwd `methods/MeshGraphNets_Variational/`, so path
keys are written `../../dataset/...` and `../../output/...`.

## Execution

| Key | Meaning |
| --- | --- |
| `model` | `meshgraphnets-v` |
| `mode` | `train` or `inference` |
| `gpu_ids` | `-1` = CPU, one id = single process, a list = multi-GPU |
| `parallel_mode` | `ddp` (default) or `model_split` (pipeline split, `latent_type global` only) |
| `pipeline_microbatches` | `model_split`: batches pipelined per optimizer step (1F1B; default `2 × stages`). Effective batch = `batch_size × pipeline_microbatches` |
| `modelpath` | Checkpoint to save to (train) or load from (inference) |
| `log_file_dir` | Epoch-log path; periodic PNG/HDF5 snapshots are written beside it |
| `resume_training`, `resume_interval_minutes` | Mid-training resume (default off, 15 min); single-process only. See [00 §7](../../methods/00_shared_foundations.md#7-mid-training-resume) |

Required fields:

- **`train`:** `modelpath`, `dataset_dir`, `input_var`, `output_var`,
  `edge_var`, `latent_dim`, `batch_size`, `learningr`, `training_epochs`.
- **`inference`:** `modelpath`, `infer_dataset`, `input_var`, `output_var`,
  `edge_var`.

## Dataset

| Key | Meaning |
| --- | --- |
| `dataset_dir` | Training mesh HDF5 ([DATASET_FORMAT.md](../DATASET_FORMAT.md)) |
| `infer_dataset`, `infer_timesteps` | Rollout initial conditions and step count |
| `split_seed` | Seed of the 80/10/10 split that training builds from sorted sample IDs (default 42). Stored `metadata/splits` is ignored |
| `split_group_attr` | Keeps every sample sharing `data/{sid}/metadata.attrs[<attr>]` in one split; at inference it also adds per-group spread figures |
| `eval_group_attr` | Metadata attribute that names one geometry in the scored file, for the per-group spread figures and the `misc/` diagnosis tools (default: `split_group_attr`). Set it when a split group holds several geometries or `eval_dataset` lacks the split attribute (ex3: split by `split_group`, score by `shape_idx`) |
| `write_preprocessing` | Write the train-derived normalizers back into the HDF5 (default true) |
| `use_parallel_stats` | Parallel normalization-stat computation (default true) |
| `static_cache_per_worker` | Per-worker LRU cap for positional features, non-multiscale runs (default 64) |
| `num_workers`, `prefetch_factor`, `pin_memory` | DataLoader settings. `pin_memory` defaults to CUDA availability |

## Input and model shape

| Key | Meaning |
| --- | --- |
| `input_var` / `output_var` | State channels read / predicted. `T > 1` needs them equal |
| `cond_var` | Input-only condition rows after the state block (default 0). See [00 §4](../../methods/00_shared_foundations.md#4-input-only-conditions-cond_var) |
| `edge_var` | Must be `8` |
| `positional_features` | Extra rotation-invariant node features (centroid distance, mean edge length, RWPE) |
| `use_node_types` | One-hot node type from the **last** `nodal_data` row; `num_node_types` is derived from the data |
| `use_deformed_edge` | Default true: the deformed half of the edge features follows reference + displacement. Replaces `geometry_state_mode` |
| `displacement_indices` | Three state columns holding the displacement; `-1` pads an axis. Replaces `displacement_state_indices` |
| `periodic_box` | Three box lengths for the minimum-image convention on relative vectors; `0` = not periodic on that axis |
| `latent_dim` | Processor width |
| `message_passing_num` | Flat processor depth (`use_multiscale false`) |

The node input width is
`input_var + cond_var + positional_features + num_node_types`.

## Multiscale

The V-cycle and coarsening are described in
[02_HI-MGN.md](../../methods/02_HI-MGN.md). Unpooling is always the learned
`UnpoolBlock`.

| Key | Meaning |
| --- | --- |
| `use_multiscale` | Enable the V-cycle |
| `multiscale_levels` | Levels `L` |
| `mp_per_level` | `2L + 1` block counts: down arm, coarsest, up arm |
| `coarsening_type` | `bfs`, `voronoi_centroid`, `voronoi_inherit`, `voronoi_seedmean`, or one per level |
| `voronoi_clusters` | Cluster count, scalar or per level |
| `hierarchy_cache_dir`, `hierarchy_cache_build_workers`, `hierarchy_cache_wait_timeout` (36000 s), `hierarchy_cache_keep` (false) | Shared per-run `*.mscache.<digest>.h5` cache, deleted after training unless kept |
| `hierarchy_variants` | K independently seeded partitions per sample, rotated across epochs. Validation stays on variant 0. Costs K× cache build time and disk. `1` = one fixed partition |
| `hierarchy_seed` | Seeds the FPS draw at inference; a list rotates seeds across draw batches |

## World edges

| Key | Meaning |
| --- | --- |
| `use_world_edges` | Radius edges on deformed positions, disjoint from mesh edges, using the mesh-edge normalization |
| `world_radius_multiplier` | Radius = this × minimum mesh-edge length (default 1.5) |
| `world_max_num_neighbors` | `torch_cluster` neighbor cap (default 64; truncates silently) |
| `world_edge_backend` | `scipy_kdtree` (default), `torch_cluster`, or `auto` |
| `coarse_world_edges` | World edges on every coarse level, normalized with that level's coarse-edge statistics. Needs `use_world_edges` and `use_multiscale` |

## Latent and posterior

| Key | Meaning |
| --- | --- |
| `use_vae` | Enable the latent path (default false) |
| `latent_type` | `global` (default, HI-MGN-V), `local` (HI-MGNflow), or `hybrid` (HI-MGN-hybrid). See [04](../../methods/04_MeshGraphNets_Variational.md). Stored in `model_config`. `local`/`hybrid` need `use_multiscale`, `prior_family fm`, no `prior_fm_moments`, `prior_fm_velocity_arch mlp`, `prior_moment_calibration_epochs 0`, and no `model_split` |
| `vae_latent_dim` | Width of each `z_g` slot |
| `num_z` | `z_g` slots (default `multiscale_levels + 1`, otherwise 1) |
| `latent_ch` | `z_l` channels per coarsest node (default 4; `local`/`hybrid`) |
| `vae_encoder_mp_layers` | Posterior encoder depth (default 5) |
| `vae_local_mp_layers` | Coarse-node posterior head depth (default 2; `local`/`hybrid`) |
| `vae_graph_aware` | Fuse the input `x` into the posterior |
| `posterior_min_std` | Posterior std floor (about 0.05). It is also the noise level the decoder learns to tolerate |
| `posterior_max_std_g` | Cap on the `z_g` std (default 0 = none; must be ≥ `posterior_min_std`) |
| `z_conditioning` | `concat` (default, `Linear([x, z])`) or `adaln` (AdaLN-Zero, identity at init). Renames parameters, so a checkpoint loads only under its trained value |
| `local_inject` | `all` (default) or `coarsest`: which levels read `z_l`. Changes behaviour only |
| `local_dropout` | `hybrid`, training only: the share of graphs whose `z_l` is withheld from the processor, in `[0, 1)` |
| `local_warmup_epochs` | `hybrid`: withhold every `z_l` for the first N epochs; checkpoint selection restarts at epoch N. Single GPU |
| `zg_anchor_weight` | `hybrid`, needs `local_warmup_epochs`: KL that keeps `z_g` near its end-of-warmup posterior |
| `stochastic_mode` | `vae` (default): posterior encoder + learned prior. `noise`: `z ~ N(0, I)` in training and inference, no encoder and no prior; trained on the fair CRPS of `noise_draws` forwards. `latent_type` still sets the z layout. Every encoder, regularizer and prior key is ignored; `prior_freeze_epoch`, `local_warmup_epochs`, `zg_anchor_weight` and `model_split` are errors. Checkpoint architecture |
| `noise_draws` | `noise` only: forwards per batch, each with its own `z` (default 4, at least 2). Training cost grows about linearly with it |
| `noise_loss` | `noise` only: `crps` (default) is the fair CRPS per node and feature; `energy` is the fair energy score per graph, `mean_k d(x_k, y) − 1/(2K(K−1)) Σ_{i≠j} d(x_i, x_j)` with `d` the feature-weighted RMS difference over the graph's nodes, so the whole field is scored as one vector. Training only; validation still reports the per-node CRPS |

## Objective

The full objective is written out in
[04](../../methods/04_MeshGraphNets_Variational.md#training-objective). Every
weight acts relative to `alpha_recon`.

| Key | Meaning |
| --- | --- |
| `recon_loss` | `huber` (default, δ = 1) or `mse` (keeps extreme amplitudes better) |
| `alpha_recon` | Reconstruction weight |
| `latent_reg` | `mmd` (default) or `kl` |
| `lambda_mmd` | Aggregate-posterior MMD weight (spread modelling: about 0.1). Ignored with `kl` |
| `lambda_mmd_g` | `hybrid` + `mmd`: `z_g` MMD weight, relative to `lambda_mmd` (default `lambda_mmd`; 0 drops it) |
| `kl_weight` | Per-sample KL weight with `latent_reg kl` (default 1e-3) |
| `mmd_bandwidth` | `fixed` (saturates at high `vae_latent_dim`) or `median` (scales with the batch) |
| `mmd_cond_width` | `0` = plain MMD. `> 0` compares (input summary, z) with (input summary, N(0, I)) through a product kernel, so it is zero only when q(z \| geometry) = N(0, I) for every geometry. Needs `mmd` |
| `mmd_gather_ranks` | All-gather `z` across DDP ranks before the MMD (default true) |
| `beta_aux` | Peak-to-valley MSE on the decoded field (spread modelling: about 1.0). Rejected with `model_split` |
| `pv_channel` | Output channel that term reads (default 2) |
| `feature_loss_weights` | Per-channel reconstruction weights, normalized to sum to 1 |

## Conditional prior

| Key | Meaning |
| --- | --- |
| `prior_type` | `gnn_e2e` (the runtime default): prior trained jointly, saved as `prior.*` |
| `use_conditional_prior` | Sample `z` from the prior at inference (otherwise N(0, I)). The training default is false and it is stored in `model_config`, which overrides the inference config, so set it at **training** time |
| `prior_family` | `fm` (default) or `gmm` (legacy). Pre-FM checkpoints load as `gmm` |
| `prior_nll_weight` | Prior loss weight (default 1.0) |
| `prior_grad_to_encoder` | Share of the prior gradient that reaches the encoder: 0 (default) = detached, 1 = full CVAE rate term. Needs `beta_aux > 0` (`MGNV-PRIOR-GRAD-AUX`) |
| `prior_mp_layers` | Prior trunk depth (default 10) |
| `prior_hidden_dim`, `prior_condition_hidden_dim`, `prior_velocity_hidden_dim` | Widths (default `latent_dim`, then `prior_hidden_dim`) |
| `prior_fm_steps` | ODE steps at sampling time (default 20); sampling only |
| `prior_fm_solver` | `heun` (default) or `euler` |
| `prior_fm_time_power` | `1` (default) = uniform time. `p > 1` draws `t = u^p` in training and uses `t_k = (k/steps)^p` in the sampler, which is dense near the noise end |
| `prior_fm_velocity_arch`, `prior_fm_blocks` | `mlp` (default) or `residual_film` with N blocks (default 4). With `local`/`hybrid`, `prior_fm_blocks` is the depth of the coarse prior |
| `prior_fm_moments` + `prior_fm_moment_hidden_dim`, `prior_fm_moment_weight` (1.0), `prior_fm_moment_min_scale` / `prior_fm_moment_max_scale` (0.03 / 20) | Condition-dependent Gaussian base for the global FM prior |
| `prior_arch` | `local`/`hybrid` coarse prior (`CoarseFMPrior`): `gnn` (default) or `transformer` |
| `prior_param` | Head target: `velocity` (default) or `x` |
| `prior_trunk` | Where the coarse prior's per-coarse-node condition comes from. `fine` (default): `prior_mp_layers` blocks of its own on the fine mesh, then the cluster mean. `multiscale`: the same blocks, then a pool and `prior_mp_layers` more blocks on every coarser level. `decoder`: no trunk of its own; the simulator's descending arm with the latent set to zero, read without gradient (`prior_mp_layers` is unused). Checkpoint architecture |
| `prior_tf_heads`, `prior_tf_dropout` | Transformer heads (default 4; must divide the width) and training dropout (default 0) |
| `prior_fm_draws` | `(z0, t)` draws per posterior sample in the coarse prior loss (default 1) |
| `prior_cond_noise` | Relative noise on the coarse prior's context in training (default 0) |
| `prior_freeze_epoch` | From this epoch, freeze the simulator and train only the prior with a fresh cosine. The joint best is kept as `<stem>.joint.pth`. 0 = never; single GPU |
| `prior_moment_calibration_epochs` | With `prior_fm_moments`: the first N prior-only epochs fit only the moment head |
| `prior_mixture_components` (50), `prior_min_std` (0.1), `prior_cov_rank` (0), `prior_kl_reg_weight` (0.02) | `gmm` only |

## Training and performance

| Key | Meaning |
| --- | --- |
| `training_epochs`, `batch_size`, `learningr` | Epochs, batch size, Adam learning rate |
| `val_batch_size` | Validation batch size (default `batch_size`) |
| `weight_decay` | Coupled L2 in Adam (default 0) |
| `warmup_epochs` | `LinearLR` warmup before cosine restarts (default 3) |
| `training_seed` | Seeds single-process training (default 42); DDP ignores it |
| `std_noise`, `noise_gamma` (0.1) | Input noise and target correction; `noise_std_ratio` is computed by setup |
| `time_integration` | `ar_ot` (default) or `ar_rt` (needs `T > 1`) |
| `grad_accum_steps` | `1` = per batch, `N` = accumulate N, `0` = the whole epoch |
| `augment_geometry` | Training only: random z rotation and x/y reflection |
| `use_checkpointing`, `use_amp`, `use_ema`, `ema_decay` | Activation checkpointing, bf16 autocast, EMA model |
| `use_compile` | Compile each plain `GnBlock` (`model/compiled_block.py`); falls back to eager if the self-test fails |
| `amp_layernorm` | Retired; accepted and ignored. Under AMP, LayerNorm always runs in bf16 |
| `best_by` | Checkpoint selection: `recon` (default) or `crps` (learned-prior CRPS, which mirrors inference) |
| `val_interval`, `test_interval`, `test_batch_idx`, `test_max_batches` (200) | Validation and visualization cadence |
| `vae_valid_prior_samples` | Prior draws per graph for validation CRPS (default 8) |
| `plot_feature_idx`, `display_testset`, `display_trainset` | Periodic visualization |

To fit a run into VRAM, try these in order:

1. lower `batch_size`;
2. `use_checkpointing`;
3. `use_amp`;
4. drop `use_ema`;
5. lighten the loader (`num_workers`, `prefetch_factor`, `pin_memory false`);
6. `grad_accum_steps` (effective batch = `batch_size × steps`);
7. `model_split`.

## Inference

| Key | Meaning |
| --- | --- |
| `inference_output_dir` | Rollout directory (default `../../output/meshgraphnets-v/rollout`) |
| `inference_samples` | Draws per scene (default 1); `z` is drawn once per trajectory |
| `inference_batch_size`, `inference_batch_size_min` (1), `inference_batch_size_max` (0 = none), `inference_batch_vram_fraction` (0.70) | Trajectories per batched forward; if unset, it is sized automatically |
| `prior_temperature` | `fm`: scales the initial noise std by √T. `gmm`: divides the logits and scales the std by √T |
| `latent_inflation` | `fm`: widens `z` around the ODE image of the origin after sampling; a list is cycled per batch (each draw is tagged `gen_lam`) |
| `save_rollouts` | Write one HDF5 per (scene, draw) (default true). With `false`, only the spread outputs remain |
| `eval_dataset` | Ground-truth HDF5 for spread scoring |
| `spread_channel` | Output channel to score (default 2). Only `mag_max` accepts a list |
| `spread_stat` | `range` (max − min, default), `mean`, `std`, or `mag_max` (max \|u\| over the channel list) at the final step |
| `spread_label` | Axis label |
| `make_histogram`, `show_histogram`, `histogram_bins` (60), `histogram_clip_quantile` (0) | Histogram controls. `make_histogram` defaults to on when `use_vae` and `eval_dataset` are set |

The spread keys are checked by `MGNV-SPREAD-STAT`, `MGNV-SPREAD-CHANNEL` and
`MGNV-SPREAD-EVAL`.

These files are written to `inference_output_dir`:

```text
rollout_sample{sid}_steps{N}.h5
rollout_sample{sid}_vaesample{k}_steps{N}.h5
spread_values.npz  spread_metrics.json  histogram_compare.png    # with eval_dataset
```

The spread scoring is in `inference_profiles/spread.py`, and the run also
prints `[SPREAD]` and `[SCORE]` lines.

## Checkpoint contents

- `model_state_dict` and, optionally, `ema_state_dict`. The prior is the
  `prior.*` submodule. Legacy checkpoints with a separate
  `conditional_prior_state_dict` still load.
- The optimizer and scheduler state, `train_loss` and `valid_loss`.
- `normalization`: node, edge and delta statistics, plus the world-edge
  radius, the node-type mapping and the coarse-edge statistics when they
  apply.
- `model_config`: the architecture keys. At inference these override the
  config file, including `use_conditional_prior`.

## Removed and renamed keys

**Removed** (`MGNV-REMOVED`, error):

- `alpha_prior_max`, `bipartite_unpool`, `fit_latent_gmm`, `free_bits`
- `gmm_components`, `gmm_covariance_type`, `gmm_reg_covar`
- `lambda_det`, `lambda_kl`, `positional_encoding`, `residual_scale`
- `gamma_es`, `es_samples`, `es_steps`, `es_noise_source`, `es_start_epoch`

**Renamed** (`CFG-RENAMED-001`, error; the native loader refuses them too):

| Old | New |
| --- | --- |
| `vae_mp_layers` | `vae_encoder_mp_layers` |
| `num_vae_samples` | `inference_samples` |
| `vae_batch_size` | `inference_batch_size` |
| `vae_batch_size_min` | `inference_batch_size_min` |
| `vae_batch_size_max` | `inference_batch_size_max` |
| `vae_batch_vram_fraction` | `inference_batch_vram_fraction` |
| `geometry_state_mode` | `use_deformed_edge` |
| `displacement_state_indices` | `displacement_indices` |

Any other retired key is simply unknown (`CFG-UNKNOWN-001`).

These MGN-only keys are unknown to the MGN-V spec and are not read by its
runtime: `pool_type`, `pool_heads`, `unpool_type`, `learned_interpolation`,
`voronoi_branches`, `init_modelpath`, `coarse_cache_per_worker` and
`train_eval_subset_size`.
