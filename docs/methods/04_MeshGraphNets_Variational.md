# 04 — MeshGraphNets-V (conditional graph VAE)

| | |
| --- | --- |
| Route | `model meshgraphnets-v` |
| Directory | [methods/MeshGraphNets_Variational/](../../methods/MeshGraphNets_Variational/) |
| Entrypoint | [MeshGraphNets_Variational_main.py](../../methods/MeshGraphNets_Variational/MeshGraphNets_Variational_main.py) |
| Spec | [ai_cae4all/specs/meshgraphnets_variational.py](../../ai_cae4all/specs/meshgraphnets_variational.py) |
| Key reference | [reference/config/meshgraphnets_variational.md](../reference/config/meshgraphnets_variational.md) |

MGN-V is the suite's **one-to-many** model. Use it when one input, such as a
geometry with its boundary conditions, has a *distribution* of outputs, as with
manufacturing spread or several FE draws per geometry.

It wraps the [MGN](01_MeshGraphNets_MGN.md) / [HI-MGN](02_HI-MGN.md) simulator
in a conditional VAE. A latent `z` carries the sample-to-sample variation, and
a learned prior `p(z | graph)` supplies `z` at inference. On data with a single
realization per input it has nothing to model, so use MGN there.

The display name follows where the latent sits (`latent_type`, with
`use_multiscale`):

| `latent_type` | Name | Latent |
| --- | --- | --- |
| `global` (default) | HI-MGN-V | `z_g`: `num_z` per-graph slots of `vae_latent_dim` (default `num_z = multiscale_levels + 1`) |
| `local` | HI-MGNflow | `z_l`: `latent_ch` channels (default 4) per coarsest-level node |
| `hybrid` | HI-MGN-hybrid | both, carried as `Latent(g, l)` |

HI-MGNflow was formerly a separate `chi-mgnflow` route. It was merged into
this one on 2026-10-19, and the old repo is in git history at commit `9db47d6`.

Read [00](00_shared_foundations.md) and [02](02_HI-MGN.md) first. The encoder,
processor, decoder, edge features, world edges, multiscale V-cycle and rollout
are inherited from them.

MGN-V does **not** carry over these MGN-only options: `pool_type`,
`unpool_type`, `voronoi_branches`, `learned_interpolation` and broadcast
unpooling. Unpooling is always the learned `UnpoolBlock`.

## Architecture

**Simulator.** The simulator is
[MeshGraphNets.py](../../methods/MeshGraphNets_Variational/model/MeshGraphNets.py),
with the flat or V-cycle processor. `z` is injected before every processor
block. `z_conditioning` selects how:

- **`concat`** (default): a bare `Linear([x, z])`. Because `z` is broadcast
  over the nodes, this adds one constant vector per graph. It is also a
  non-residual Linear with a gain of about 1.33 per block under Kaiming init,
  so the residual stream grows with depth.
- **`adaln`**: `AdaLNZero` in
  [blocks.py](../../methods/MeshGraphNets_Variational/model/blocks.py). `z`
  emits `(shift, scale, gate)`, and the block computes:

  ```text
  x_mod = (1 + scale)·x + shift
  x_out = x + gate·(Block(x_mod) − x_mod)
  ```

  It is zero-initialized with a gate bias of 1, so at init it is exactly the
  unconditioned block.

The two options name their parameters differently, so a checkpoint loads only
under the value it was trained with.

**Posterior.** `GNNVariationalEncoder`
([vae.py](../../methods/MeshGraphNets_Variational/model/vae.py)) encodes the
target `graph.y`:

1. It encodes `y`. With `vae_graph_aware`, it fuses in the input `x` as well.
2. It runs `vae_encoder_mp_layers` `GnBlock`s.
3. It pools with attention and applies `mu`/`logvar` heads to produce `z_g`.
   For `local`/`hybrid`, a coarse-node head of depth `vae_local_mp_layers`
   produces `z_l`.

`posterior_min_std` floors the posterior std, and `posterior_max_std_g` caps
the std of `z_g`. Latent placement and the regularizer choices live in
[latent.py](../../methods/MeshGraphNets_Variational/model/latent.py).

**Conditional prior.** The prior is in
[conditional_prior.py](../../methods/MeshGraphNets_Variational/model/conditional_prior.py),
and `build_conditional_prior` picks the class. It is trained jointly
(`prior_type gnn_e2e`) and saved as the `prior.*` submodule:

| Prior | When | What it is |
| --- | --- | --- |
| `ConditionalFMPrior` | `global`, `prior_family fm` (default) | Conditional flow matching from `N(0, I)`, or from a per-graph Gaussian with `prior_fm_moments`. The velocity net is `mlp` or `residual_film`. Sampled with `prior_fm_steps` steps of `heun` (default) or `euler`. |
| `ConditionalMixturePrior` | `global`, `prior_family gmm` | Graph-conditioned Gaussian mixture, with optional low-rank covariance. Legacy. |
| `CoarseFMPrior` | `local` / `hybrid` | Flow matching over the coarsest-level nodes. `prior_arch` is `gnn` (default) or `transformer` (AdaLN-Zero attention). `prior_param` sets whether the head predicts `velocity` (default) or `x`. |

`prior_fm_time_power p > 1` concentrates both the training times and the
sampler grid near the noise end.

**Constraints on `local`/`hybrid`** (`validate_latent_placement`):

- `use_multiscale true`
- `prior_family fm`
- no `prior_fm_moments`
- `prior_fm_velocity_arch mlp`
- `prior_moment_calibration_epochs 0`
- no `model_split`

`model_split` supports only `global`.

## Training objective

```text
loss = alpha_recon      · recon(f(z_q, g), y)        huber (default) or mse
     + lambda_mmd       · MMD(q(z) ‖ N(0, I))         latent_reg mmd (default); or kl_weight · KL with latent_reg kl
     + beta_aux         · PV(f(z_q, g), y)            peak-to-valley of decoded channel pv_channel
     + prior_nll_weight · prior loss                  FM MSE, or mixture NLL (+ prior_kl_reg_weight)
```

**MMD.** The MMD matches only the *aggregate* posterior to `N(0, I)`, so the
latent stays the ball that the flow-matching prior integrates from. Two keys
refine it:

- `mmd_gather_ranks` (default true) gathers `z` across DDP ranks, so the
  estimator sees the global batch.
- `mmd_cond_width > 0` conditions the MMD on a per-graph input summary.

**Peak-to-valley term.** `_pv_loss` fits each realization's own
`max − min` on the decoded field. That is what makes the decoded spread
depend on `z`. Note:

- For `T > 1` it fits the per-step increment (with a warning).
- `model_split` rejects it (`MGNV-AUX-PIPELINE`).

**`prior_grad_to_encoder`** sets how much of the prior gradient reaches the
encoder:

- `0` (default) detaches the posterior, so the prior chases a fixed `q`.
- `1` closes the CVAE rate term.

Opening it rewards a `z` that is predictable from the graph alone. Only
`beta_aux > 0` guards against that collapse, which is why `MGNV-PRIOR-GRAD-AUX`
warns when `beta_aux` is 0.

**Scaling.** All the regularizer weights act *relative to* `alpha_recon`.
Watch `mmd` and `fm_p` against `total` on the progress bar. A starting point
for spread modelling is `lambda_mmd ≈ 0.1` and `beta_aux ≈ 1.0`.

**Training mechanics.** MGN-V differs from MGN in these ways:

- **Optimizer:** plain Adam with coupled `weight_decay`.
- **Allocator and precision:** the main script sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and enables TF32.
- **Checkpointing:** a checkpoint is saved on every new best validation score
  under `best_by`, which is `recon` (default) or `crps`, the learned-prior
  CRPS. Use `crps` when the run exists to generate.
- **Prior-only tail:** `prior_freeze_epoch` (single GPU) freezes the
  simulator and trains only the prior for the remaining epochs.
- **Hierarchy variants:** `hierarchy_variants` rotates K seeded coarsening
  partitions across epochs. `hierarchy_seed` pins the FPS draw at inference.
- **Seed:** `training_seed` seeds single-process training. DDP ignores it.
- **Speed:** `use_compile` compiles each plain `GnBlock`
  ([compiled_block.py](../../methods/MeshGraphNets_Variational/model/compiled_block.py)),
  with an eager fallback.

## Data contract

The dataset is the shared mesh HDF5
([DATASET_FORMAT.md](../reference/DATASET_FORMAT.md)). The data must share a
topology and carry real spread: either several realizations per input, or one
realization per input across many inputs.

`split_group_attr` keeps every draw of a geometry in the same split, which is
required for multi-draw files such as ex3. `cond_var`, `periodic_box` and
world edges work as described in [00](00_shared_foundations.md).

## Modes and keys

The modes are `train` and `inference`. Run
`python AI_CAE4ALL_main.py --describe meshgraphnets-v` for the required fields
of each mode. Every key, with its default and validator, is in the
[key reference](../reference/config/meshgraphnets_variational.md).

**Inference** draws `z` once per trajectory and holds it for the whole
rollout:

- `inference_samples` sets the number of draws per scene.
- `use_conditional_prior` comes from the checkpoint's `model_config`. Its
  training default is `false`, so set it at **training** time.
- `prior_temperature` and `latent_inflation` widen the draws.

## Outputs

These paths are resolved from `methods/MeshGraphNets_Variational/`, and every
shipped config points them under `../../output/meshgraphnets-v/`.

| File | Where |
| --- | --- |
| Best checkpoint (plus `<modelpath>.resume` while `resume_training` is on) | `modelpath` |
| Epoch log, periodic PNG/HDF5 | `log_file_dir` |
| `rollout_sample{sid}_steps{N}.h5` (deterministic) or `rollout_sample{sid}_vaesample{k}_steps{N}.h5` (VAE draws) | `inference_output_dir` (default `../../output/meshgraphnets-v/rollout`); not written when `save_rollouts false` |
| `spread_values.npz`, `spread_metrics.json`, `histogram_compare.png`, plus per-group figures with `split_group_attr` | `inference_output_dir`, when `eval_dataset` is set |

The spread statistic is `spread_stat` applied to the output channel
`spread_channel` at the final step. Ensemble scoring (CRPS, W2,
spread-skill) is in
[inference_profiles/spread.py](../../methods/MeshGraphNets_Variational/inference_profiles/spread.py).
Offline diagnostics, such as posterior vs. prior and the prior diagnosis, are
indexed in
[misc/README.md](../../methods/MeshGraphNets_Variational/misc/README.md).

## Configs

| Tree | Content |
| --- | --- |
| [probabilistic/ex{1,2,3}/baseline/](../../configs/MeshGraphNets_Variational/probabilistic/ex1/baseline/config_train_himgn_v.txt) | `config_{train,infer}_himgn_v.txt` baselines |
| [probabilistic/ex3/latent_type/](../../configs/MeshGraphNets_Variational/probabilistic/ex3/latent_type/) | `{global,local,hybrid} × {mmd,kl}` |
| [SAOI_run/](../../configs/MeshGraphNets_Variational/SAOI_run/README.md), [SAOI_hybrid_run/](../../configs/MeshGraphNets_Variational/SAOI_hybrid_run/README.md) | SAOI production runs |
| [hyperparameter_sweep/](../../configs/MeshGraphNets_Variational/hyperparameter_sweep/README.md) | B300 sweep rounds r8–r13 |

## Tests and validation

```bash
python AI_CAE4ALL_main.py --config configs/MeshGraphNets_Variational/probabilistic/ex3/baseline/config_train_himgn_v.txt --check
cd methods/MeshGraphNets_Variational && python -m pytest -q tests/
```

The tests cover these areas:

- latent types and coarse-prior variants
- flow-matching priors (`test_fm_v2`) and prior freezing
- the peak-to-valley loss
- periodic edges and edge geometry (`test_state_geometry`)
- multiscale normalization and the hierarchy cache
- grouped splits, `cond_var`, AR rollout and the optimizer contract
- SAOI sweep preflight

## Known limitations

- **Many interacting weights.** `alpha_recon`, `lambda_mmd`/`kl_weight`,
  `beta_aux`, `prior_nll_weight` and the posterior std floors all interact.
  Mis-weighting them collapses the spread or blurs the mean.
- **Checkpoint-coupled config.** `model_config` overrides the inference
  config, including `use_conditional_prior`.
- **Narrow ensembles.** Generated ensembles tend to be narrower than the
  truth. The [sweep README](../../configs/MeshGraphNets_Variational/hyperparameter_sweep/README.md)
  tracks the current state.
- **Compute cost.** The posterior encoder and the prior trunk add substantial
  compute on top of the simulator.
- **Inherited caveats.** MGN-V inherits HI-MGN's coarsening cost and
  node-order dependence, and MGN's rollout error accumulation.
- **Removed keys.** Some old keys are rejected as `MGNV-REMOVED`; see the
  [key reference](../reference/config/meshgraphnets_variational.md#removed-and-renamed-keys).
