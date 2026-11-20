# 00 — Shared foundations of the mesh models

These are the concepts that MeshGraphNets ([01](01_MeshGraphNets_MGN.md)),
HI-MGN ([02](02_HI-MGN.md)), BSMS-GNN ([03](03_BSMS-GNN.md)) and MGN-V
([04](04_MeshGraphNets_Variational.md)) have in common. Transolver and the
Neural Operators read the same dataset, but they use their own blocks.

The two trees, [methods/MeshGraphNets/](../../methods/MeshGraphNets/) and
[methods/MeshGraphNets_Variational/](../../methods/MeshGraphNets_Variational/),
share no imports. The modules named below exist in both trees unless the text
says otherwise.

## 1. Dataset contract

The full specification is in
[DATASET_FORMAT.md](../reference/DATASET_FORMAT.md). In short:

- The data for one sample is `data/{sample_id}/nodal_data` with shape
  `[F, T, N]`, plus `data/{sample_id}/mesh_edge` with shape `[2, E]`.
- The rows of `nodal_data` are:

  | Rows | Content | Role |
  | --- | --- | --- |
  | `0:3` | reference coordinates | `graph.pos`; never an input channel or a prediction |
  | `3 : 3+input_var` | state | input and output |
  | next `cond_var` rows | conditions | input only (see [§4](#4-input-only-conditions-cond_var)) |
  | last row (if present) | node type / part label | copied through to rollouts |

- **Static (`T = 1`):** the state input is zeroed, and the target is the
  absolute field `nodal_data[3:3+output_var]`.
- **Temporal (`T > 1`):** the input is the state at step `t`, and the target
  is the delta `state[t+1] − state[t]`. The AR loop needs
  `input_var == output_var`.
- **Splits:** training makes its own 80/10/10 split from the sorted sample
  IDs, seeded by `split_seed` (default 42). It ignores any splits stored in
  the file. With `split_group_attr`, every sample that shares
  `data/{sid}/metadata.attrs[<attr>]` lands in the same split
  ([grouped_split.py](../../methods/MeshGraphNets/general_modules/grouped_split.py)).
- **Normalizers:** z-score statistics for nodes, edges and deltas come from
  the train split. They are stored in the checkpoint. Training also writes
  them back into the HDF5 unless `write_preprocessing false` is set.

## 2. Graph encoding

### Node features

```text
graph.x = [ state (input_var) | conditions (cond_var) | positional (positional_features) | node-type one-hot (num_node_types) ]
```

- `positional_features` are rotation-invariant features, in this order:
  1. distance from the centroid,
  2. mean incident edge length,
  3. RWPE (random-walk return probabilities at k = 2, 4, 8, 16, 32).

  See [positional_features.py](../../methods/MeshGraphNets/general_modules/positional_features.py).
- `use_node_types` reads the **last** row of `nodal_data` at `t = 0`. It maps
  the observed labels to contiguous indices and appends a one-hot vector.
  `num_node_types` is derived from the data and cannot be set in a config.
  The mapping is saved in the checkpoint.

### Edge features

`edge_var` must be `8` (spec code `MGN-EDGE-001`). Edge features are
recomputed when a sample is loaded, never stored in the file:

```text
[ deformed dx, dy, dz, |d|,  reference dx, dy, dz, |d| ]
```

The code is in [edge_features.py](../../methods/MeshGraphNets/general_modules/edge_features.py).

- With `use_deformed_edge true` (the default), the deformed geometry is the
  reference plus the state columns listed in `displacement_indices`. Use `-1`
  to pad a missing axis with zero.
- With `use_deformed_edge false`, both halves use the reference geometry.
- These two keys replace `geometry_state_mode` and
  `displacement_state_indices`. The launcher (`CFG-RENAMED-001`) and the
  native loader both refuse the old names.
- A `1e-12` epsilon inside the square root keeps the gradient of `|d|` finite
  at coincident nodes.
- MGN-V also accepts `periodic_box` (three lengths, `0` = not periodic) to
  apply the minimum-image convention to relative vectors.

### World edges (optional)

With `use_world_edges true`, radius edges are added from the **deformed**
positions ([world_edges.py](../../methods/MeshGraphNets/general_modules/world_edges.py)):

- **Radius:** `world_radius_multiplier` (default 1.5) × the minimum mesh-edge
  length in the training data. The radius is stored in
  `checkpoint['normalization']`, so inference uses the same value.
- **Disjoint from the mesh:** pairs that are already mesh edges are removed.
- **Features:** the same 8-D layout as mesh edges, normalized with the
  mesh-edge statistics.
- **Backend:** `world_edge_backend` is `scipy_kdtree` (the dataset default),
  `torch_cluster`, or `auto`. `world_max_num_neighbors` (default 64) caps the
  neighbors per node, and `torch_cluster` silently truncates past that cap.
- **During a rollout:** the edges are rebuilt at every step from the
  predicted geometry.
- **On coarse levels:** with `use_multiscale`, `coarse_world_edges true` adds
  world edges to every coarse level too.

## 3. Encode–process–decode blocks

The blocks live in `model/mlp.py`, `model/encoder_decoder.py` and
`model/blocks.py`.

- **`build_mlp`:** `Linear → SiLU → Linear → SiLU → Linear`, followed by a
  LayerNorm. Weights use Kaiming-uniform init and biases start at zero. Under
  `use_amp`, the LayerNorm runs on the bf16 activations.
- **Encoder:** one node MLP, one mesh-edge MLP (`8 → latent_dim`), and a
  world-edge MLP when world edges are on.
- **`GnBlock`:**
  1. The edge MLP takes `[x_s, x_r, e]` and updates `e`.
  2. Messages are **sum**-aggregated at the receiver.
  3. The node MLP takes `[x, Σe]` and updates `x`.

  Both updates are residual. With world edges, `HybridNodeBlock` reads
  `[x, Σe_mesh, Σe_world]` instead. The first Linear is split per input
  (`split_linear`), so the `[E, 3·latent_dim]` concatenation is never built.
- **Decoder:** an MLP **without** LayerNorm. When `num_timesteps` is unset or
  greater than 1, its last layer is scaled by 0.01, so a temporal model starts
  near "no change".

## 4. Input-only conditions (`cond_var`)

`cond_var` rows are known per-node conditions. Examples are loads,
temperatures and process settings.

- The model reads them at every step but never predicts them.
- They have their own normalization statistics.
- They are present even when the static state input is zeroed.
- An AR rollout carries them through unchanged.

`cond_var 0` (the default) gives the unconditioned model.

## 5. Training and rollout

| Aspect | MeshGraphNets | MGN-V |
| --- | --- | --- |
| Loss | MSE on the normalized target | `recon_loss`: `huber` (default, δ = 1) or `mse`, weighted by `alpha_recon`, plus the VAE terms ([04](04_MeshGraphNets_Variational.md#training-objective)) |
| Per-channel weights | `feature_loss_weights`, normalized to sum to 1 | same |
| Optimizer | fused AdamW with decoupled `weight_decay` (default 0) | Adam with coupled `weight_decay` (default 0) |
| Schedule | `LinearLR` warmup over `warmup_epochs` (default 3), then `CosineAnnealingWarmRestarts` over the remaining epochs | same |
| Gradient clipping | max-norm 3.0 | same |
| Precision | `use_amp`: bf16 autocast where native, otherwise fp16 with a GradScaler | same |
| Memory | `use_checkpointing` (activation recomputation), `grad_accum_steps` (`0` = the whole epoch) | same |
| EMA | `use_ema`, `ema_decay`; the EMA model is used for validation and saved with the checkpoint | same |
| Checkpoint written | once, at the end of training (an interrupted run saves nothing) | on every new best by `best_by`; the last epoch if no epoch validated |

**Input noise (`std_noise`):** during training, Gaussian noise is added to
the leading `output_var` state channels. It is in normalized units. The
deformed half of every mesh and world edge feature is then rebuilt from the
perturbed positions. Coarse edges and the world-edge connectivity are not
re-searched. The target is corrected by
`noise_gamma × noise × noise_std_ratio`, where `noise_std_ratio` is computed
as `node_std / delta_std` and not set by hand. See
[input_noise.py](../../methods/MeshGraphNets/general_modules/input_noise.py).

**Time integration (`time_integration`)** follows arXiv:2510.15201:

- `ar_ot` (the default) trains on one-step pairs with teacher forcing.
- `ar_rt` unrolls the whole training trajectory and backpropagates through
  it ([ar_rollout.py](../../methods/MeshGraphNets/training_profiles/ar_rollout.py)).
  It needs `num_timesteps > 1`.

**Inference (`mode inference`)** rolls the model forward for
`infer_timesteps` steps from each sample of `infer_dataset`. After every step
it rebuilds the deformed edge features, and the world edges if they are on.
The checkpoint's `model_config` overrides the config file for every
architecture key. Changing such a key only in the inference config therefore
has no effect.

**Periodic visualization:** every `test_interval` epochs, a 2×2 PNG is
written next to the epoch log at `log_file_dir`. It shows prediction against
truth, normalized and denormalized, coloured by `plot_feature_idx`. Set
`display_testset` / `display_trainset` to choose which split is drawn.

## 6. Multiscale basics

`use_multiscale true` replaces the flat processor with a V-cycle over a
coarsening hierarchy. [02](02_HI-MGN.md) describes it in full. In brief:

- `mp_per_level` has `2·multiscale_levels + 1` entries: the down arm, the
  coarsest level, then the up arm.
- `coarsening_type` is one mode per level: `bfs`, `voronoi_centroid`,
  `voronoi_inherit` or `voronoi_seedmean`.
- `voronoi_clusters` sets the Voronoi cluster counts. The spec requires the
  key whenever multiscale is on, even for `bfs`.
- Hierarchies are computed once per run into a shared
  `*.mscache.<digest>.h5` file
  ([multiscale_cache.py](../../methods/MeshGraphNets/general_modules/multiscale_cache.py)).
  The file is deleted after training unless `hierarchy_cache_keep true` is
  set. The relevant keys are `hierarchy_cache_dir`,
  `hierarchy_cache_build_workers` and `hierarchy_cache_wait_timeout`
  (default 36000 s).

## 7. Mid-training resume

`resume_training true` saves `<modelpath>.resume` at the first epoch boundary
after every `resume_interval_minutes` (default 15). The file holds the model,
EMA, optimizer, scheduler and RNG state. A restart with the same config
fingerprint continues from that file, and the file is deleted once the final
checkpoint is written. Multi-GPU and `model_split` training reject the key
([resume_state.py](../../methods/MeshGraphNets/general_modules/resume_state.py),
identical in every training repo).

## 8. Config parsing

The launcher ([ai_cae4all/config_parser.py](../../ai_cae4all/config_parser.py))
and each tree's `general_modules/load_config.py` parse configs the same way:

- Every line is `key value`.
- A line that starts with `%` is skipped, and `#` starts an inline comment.
- Keys are lowercased. String values are lowercased too, except path keys.
- A single value is a bare scalar, and comma-separated values become a list.
- `1e-4` stays a string, so consumers convert numbers explicitly.
- `true`/`false` become booleans.
- A BOM or a duplicate key is an error.

See [CONFIGURATION.md](../CONFIGURATION.md) for the full grammar.
