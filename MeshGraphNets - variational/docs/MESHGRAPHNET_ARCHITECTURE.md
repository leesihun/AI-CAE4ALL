# MeshGraphNets Architecture

This document describes the architecture implemented in the current codebase.
The authoritative modules are:

- `model/MeshGraphNets.py`
- `model/encoder_decoder.py`
- `model/blocks.py`
- `model/mlp.py`
- `model/vae.py`
- `model/conditional_prior.py`
- `training_profiles/training_loop.py`
- `training_profiles/posthoc_prior.py`

## Runtime Entry Points

`MeshGraphNets_main.py` loads a text config and dispatches by `mode`:

| Mode | Runtime path |
| --- | --- |
| `train` | simulator training through `single_training.py` or DDP in `distributed_training.py` |

| `inference` | rollout through `inference_profiles/rollout.py` |

With multiple `gpu_ids`, the default path is DDP data parallelism. Setting
`parallel_mode model_split` uses the experimental pipeline split launcher under
`parallelism/`.

## Graph Contract

Each training sample becomes a PyG `Data` or `MultiscaleData` object with:

| Field | Shape | Meaning |
| --- | --- | --- |
| `x` | `[N, input_var + positional_features + optional node_types]` | normalized node input |
| `y` | `[N, output_var]` | normalized target delta |
| `pos` | `[N, 3]` | unnormalized reference position |
| `edge_index` | `[2, E]` | bidirectional mesh edges |
| `edge_attr` | `[E, 8]` | normalized mesh edge features |
| `world_edge_index` | `[2, E_world]` | optional radius edges, empty when disabled |
| `world_edge_attr` | `[E_world, 8]` | optional normalized world-edge features |
| `part_ids` | `[N]` | optional raw part numbers for visualization |

`edge_var` must be `8`. The code validates it against `EDGE_FEATURE_DIM` in
`general_modules/edge_features.py`.

Edge feature order:

```text
deformed_dx, deformed_dy, deformed_dz, deformed_dist,
ref_dx, ref_dy, ref_dz, ref_dist
```

For single-timestep datasets, the node input physical channels are zeros and the
target is the final displacement/state. For multi-timestep datasets, the node
input is state `t` and the target is `state[t + 1] - state[t]`.

## MLP Building Block

`model/mlp.py::build_mlp` builds:

```text
Linear -> SiLU -> Linear -> SiLU -> Linear -> optional LayerNorm
```

LayerNorm is enabled for encoders, processor blocks, coarse edge encoders,
unpool blocks, VAE encoders, and the conditional prior. It is disabled for the
final simulator decoder and the VAE auxiliary decoder.

Weights are initialized with Kaiming uniform for `nn.Linear` and zero bias.

## Flat Encoder-Processor-Decoder

The top-level wrapper `MeshGraphNets` owns an `EncoderProcessorDecoder` as
`model`.

### Encoder

`Encoder` maps raw node and edge features into `latent_dim`:

- node encoder: node input size to `latent_dim`
- mesh edge encoder: 8-D edge features to `latent_dim`
- optional world edge encoder: 8-D world-edge features to `latent_dim`

### Processor

When `use_multiscale False`, the processor is a `ModuleList` of
`message_passing_num` `GnBlock` layers.

Each `GnBlock` does:

1. Mesh edge update from sender node, receiver node, and current mesh edge state.
2. Optional world edge update with the same edge-block structure.
3. Node update from current node state plus summed incoming mesh edge messages.
4. If world edges are enabled, node update also receives summed incoming
   world-edge messages through `HybridNodeBlock`.
5. Residual node and edge updates (unscaled: `x + delta`).

Aggregation is `sum`, matching the NVIDIA PhysicsNeMo deforming-plate style in
the local comments.

### Decoder

`Decoder` maps latent node states to `output_var` normalized delta channels. The
decoder MLP has no final LayerNorm. For delta prediction (`num_timesteps` absent
or greater than 1), the last decoder layer is scaled by `0.01` at construction
time so the initial prior is close to no change.

## Multiscale V-Cycle

When `use_multiscale True`, the flat `message_passing_num` processor is replaced
by a V-cycle built in `EncoderProcessorDecoder._build_multiscale_processor`.

For `L = multiscale_levels`, `mp_per_level` must contain `2 * L + 1` integers:

```text
[pre_0, pre_1, ..., pre_(L-1), coarsest, post_(L-1), ..., post_1, post_0]
```

The default fallback for one level is:

```text
[fine_mp_pre, coarse_mp_num, fine_mp_post]
```

The forward pass is:

1. Encode the fine graph.
2. Run pre-blocks on each level.
3. Save a fine-level skip state.
4. Pool node states by `fine_to_coarse_i`.
5. Encode the corresponding coarse edge attributes.
6. Run coarsest blocks.
7. Unpool back up each level.
8. Merge skip state and unpooled state with `skip_projs[i]`, a linear projection
   from `2 * latent_dim` to `latent_dim`.
9. Run post-blocks and decode.

Unpooling is always a learned `UnpoolBlock` over
`unpool_edge_index_i` using coarse state, fine skip state, and relative position.
Otherwise, coarse node states are broadcast to fine nodes by cluster assignment.

There is no global gated skip module in the current code. Skip merging is the
per-level linear `skip_projs` described above.

World-edge message passing only applies on the original fine level. Coarse-level
blocks are constructed with `use_world_edges False`.

## VAE Branch

The VAE branch is the core mechanism for spread modeling. The latent `z` captures
the part of the output that varies across manufactured samples (the "spread") while
the graph processor captures the part that is determined by geometry and boundary
conditions. During inference, sampling different `z` values from the prior
`p(z|graph)` produces distinct but physically plausible output variants for the
same input mesh.

When `use_vae True`, the model adds a graph variational encoder and injects a
global latent `z` into the simulator processor.

### Posterior Encoder

`GNNVariationalEncoder` encodes target delta `y` into a graph-level latent:

1. Encode `y` with an MLP.
2. If `vae_graph_aware True`, encode graph input `x` with a second MLP and fuse
   it with the encoded `y`.
3. Encode 8-D mesh edge attributes.
4. Run `vae_mp_layers` `GnBlock` layers.
5. Pool with `GlobalAttention`.
6. Predict `mu` and `logvar`.
7. Reparameterize to sample `z`.

The posterior regularizer used in training is MMD between sampled posterior `z`
and `N(0,I)`. It is a two-SAMPLE statistic, so its effective sample count is the
number of `z` rows it sees, and it is evaluated inside the model forward — i.e.
on the PER-RANK batch. `mmd_gather_ranks` (default `True`) all-gathers `z` across
DDP ranks first, so the estimator sees the global batch; `z` is `[B, num_z, D]`,
so the collective costs nothing. Every rank draws its own `N(0,I)` reference, and
`torch.distributed.nn.all_gather`'s sum-reduced backward combined with DDP's
mean over ranks gives exactly the gradient of the rank-averaged MMD.

### Latent Injection

During training, the simulator uses posterior `z`. During inference, it uses a
fixed `z`, conditional-prior sample, legacy GMM sample, or standard normal sample
depending on checkpoint contents and config.

`z_conditioning` selects the mechanism. Both live under the same attribute
names (`z_fusers`, `ms_z_fusers_pre/post/coarsest`), one module per processor
block, with separate lists for each pre arm, the coarsest arm, and each post arm;
pooled batch assignments map graph-level `z` onto coarse nodes. Their parameter
paths differ (`...0.weight` vs `...0.net.1.weight`), so loading a checkpoint
under the wrong mode fails with a clean missing/unexpected-key error rather than
a silent shape mismatch. The value is persisted in the checkpoint's
`model_config`; checkpoints from before it existed load as `concat`, which is
what they trained with.

**`concat` (legacy default)** — `x <- Linear([x, z])` before each block. Two
defects, both structural:

1. `z` is broadcast identically to every node of a graph, so
   `Linear([x, z]) = W_x x + W_z z + b` and the `W_z z` half is ONE constant
   vector added to all nodes. `z` can translate the node-feature cloud and
   nothing else — no scaling, no gating, no spatial selection. All spatial
   structure of the stochastic field has to be synthesized by message passing
   from a uniform offset.
2. The fuser is a bare `nn.Linear` — no residual, no normalization, no
   nonlinearity — inserted between every pair of blocks. Under the repo's
   `kaiming_uniform_(nonlinearity='relu')` its x-half has gain ~1.33, which
   compounds over the processor. Simulated on the SAOI V-cycle
   (`latent_dim 128`, `mp_per_level 4,6,8,6,4`) the residual stream reaches
   ~2.8e2 at the coarsest level against ~4.4 for the same stack without fusers.
   Because `UnpoolBlock.node_mlp` ends in a LayerNorm, `h_up` is always ~1 while
   the skip branch carries that inflated scale, so the coarse-to-fine merge
   weights the V-cycle's global information at 1:28 (level 1) and 1:4.8
   (level 0), against 1:3.3 and 1:2.2 without fusers. The multiscale arm is
   attenuated by the very mechanism meant to inject `z`.

**`adaln`** — AdaLN-Zero (Peebles & Xie, DiT 2023), in `model/blocks.py`:

```text
shift, scale, gate = MLP(z)                       # three [N, latent_dim] vectors
x_mod = (1 + scale) * x + shift                   # modulate the block INPUT
x_out = x + gate * (Block(x_mod) - x_mod)         # gate the update, keep the highway
```

The final Linear is zero-initialized with bias `(0, 0, 1)`, so at init
`scale=0, shift=0, gate=1` and the module is exactly the unconditioned block:
`max|f(z1) - f(z2)| = 0` measured, and the decoder-input rms drops from 32.2 to
3.7 on a 12-block stack. `scale` and `gate` are multiplicative, which is the
control a pure additive offset never had. Note the gate bias is `1`, not DiT's
`0`: a zero gate would kill the `scale`/`shift` gradients at init. It is also
cheaper than the fuser it replaces (`3 * z_dim * D` MACs per node against
`(D + z_dim) * D`). The modulation runs INSIDE the gradient-checkpointed region
so its three `[N, latent_dim]` tensors are recomputed rather than stored.

The auxiliary decoder predicts per-graph target mean and standard deviation from
`z`, and its loss is weighted by `beta_aux`.

## Conditional Prior

`model/conditional_prior.py` provides the mesh-conditioned prior `p(z|graph)`,
trained **jointly** with the simulator when `use_vae True` and
`prior_type gnn_e2e`. It lives as the `prior` submodule of `MeshGraphNets`, so
DDP and EMA wrap it together with the simulator and it is saved inside the
normal `model_state_dict`.

A shared graph trunk uses the same node input accounting as the simulator:

```text
input_var + positional_features + optional num_node_types
```

Its edge input is also `edge_var`, which must be 8. The trunk encodes node and
edge features, runs `prior_mp_layers` graph blocks, and pools with attentional
aggregation into one conditioning vector per graph. `prior_family` selects the
density head:

- `fm` (default): a conditional flow-matching velocity field. Training is MSE
  regression on straight-line interpolation paths toward fresh posterior
  samples, resampled every step so the target is the smoothed posterior cloud
  rather than fixed points. Whether those samples are detached is
  `prior_grad_to_encoder` (see the training objective below). Sampling
  integrates the ODE for `prior_fm_steps` steps with `prior_fm_solver` —
  `heun` (default, 2nd-order trapezoid) or `euler`. The velocity is most curved
  near `t = 1`, which is where a first-order step overshoots into the `z` tails;
  on `dz/dt = z`, Heun at 30 steps is ~27x more accurate than Euler at 100 and
  ~40% cheaper. Temperature scales the initial noise std by `sqrt(temperature)`.
- `gmm`: mixture logits, means, and log-stds for `prior_mixture_components`
  Gaussian components (optionally low-rank covariance via `prior_cov_rank`).
  Training is mixture NLL plus a small `prior_kl_reg_weight` analytical-KL
  anchor. Temperature divides logits and scales stds by `sqrt(temperature)`.

At inference, `rollout.py` samples from the conditional prior when
`use_vae True` and `use_conditional_prior True` hold after checkpoint
model_config overrides, and the checkpoint carries a prior (the joint-trained
submodule, or a legacy separately-saved `conditional_prior_state_dict`).
Otherwise it samples `N(0,I)`.

## Training Objective

The model's goal is **manufacturing spread modeling**: given multiple manufactured
objects that share mesh topology but differ in physical outputs due to production
variability, the VAE must learn the spread of that output distribution and enable
generation of realistic new samples at inference time (potentially more samples
than the training set).

The simulator predicts normalized target deltas. Reconstruction loss is Huber
loss with `delta=1.0`, optionally weighted per output channel by
`feature_loss_weights` after normalizing those weights to sum to 1.

Without VAE:

```text
loss = reconstruction_loss
```

With VAE:

```text
loss = alpha_recon      * reconstruction_loss
     + lambda_mmd       * mmd_loss           # AGGREGATE rate: q(z) ↔ N(0,I)
     + beta_aux         * auxiliary_loss     # I(z;y) floor: z → per-graph output stats
     + prior_nll_weight * prior_loss         # CONDITIONAL rate: FM (fm) or NLL (gmm)
   [ + prior_kl_reg_weight * kl_anchor ]     # gmm family only
```

The CVAE bound has exactly two terms, reconstruction and the rate
`KL(q(z|y,g) || p(z|g))`. `p(z|g)` is a flow-matching model with no closed-form
density, so the rate is approximated by TWO different surrogates that constrain
different things and therefore do not fight:

- **MMD** constrains only the MARGINAL `q(z)`, saying nothing about which graph
  maps where. It keeps the latent the well-scaled unit ball that the FM prior
  integrates FROM. A minibatch of different graphs is not a problem for it —
  `q(z)` is by definition a mixture over the dataset, so a batch of B graphs is
  B samples from exactly that mixture.
- **The FM term** constrains the CONDITIONAL structure inside that ball.

`prior_grad_to_encoder` decides whether the second one is the rate term at all.
At `0.0` (the historical default) the posterior is detached, so the gradient
reaches only the prior: the prior chases a `q` that never moves toward it, and
the rate term is simply absent from the objective. At `1.0` the loop is closed.

Opening it adds a pressure toward a `z` that is PREDICTABLE FROM THE GRAPH, and
MMD is not a sufficient guard against that: a deterministic `z = h(g)` whose
distribution over graphs happens to be `N(0,I)` satisfies MMD perfectly. What
keeps `y` in the latent is the reconstruction term and `beta_aux`'s explicit
I(z;y) floor — so **`beta_aux` must stay > 0 whenever `prior_grad_to_encoder`
is > 0**, and a rising `aux` loss during training is the signal that this
collapse is under way. MMD does still rule out the simpler failure of `q`
collapsing to a constant.

All three regularizer weights are only meaningful RELATIVE to `alpha_recon`. At
`alpha_recon 1000` with the others at 1 each is ~0.1% of the objective, i.e.
effectively off — and that matters most for `prior_grad_to_encoder`, whose rate
term competes with reconstruction INSIDE the encoder. Watch `mmd` and `fm_p`
against `total` on the progress bar in the first epochs.

An energy-score term over decodes of prior-sampled `z` (`gamma_es`, `es_*`) was
removed from the objective; the launcher now flags those keys as
`MGNV-REMOVED`.

**Spread modeling guidance for loss weights:**

- `lambda_mmd` should remain low (≈ 0.1). A non-zero residual MMD is acceptable
  and expected: the true aggregate posterior encodes real spread structure that does
  not match an isotropic Gaussian. Forcing MMD → 0 erases that structure.
- `beta_aux` should remain high (≈ 1.0). The auxiliary decoder forces `z` to
  predict per-graph output mean and standard deviation. Without this anchor the
  encoder collapses all spread into a small z subspace (mode collapse).
- The deterministic z=0 auxiliary pass (`lambda_det`) was removed from the code:
  it conflicts directly with the spread objective by punishing z for carrying
  information the graph cannot predict alone. Do not reintroduce it.

Other training behavior:

- Adam optimizer, fused when CUDA is available.
- Linear warmup followed by cosine warm restarts.
- bfloat16 autocast when `use_amp True`.
- gradient clipping with max norm `3.0`.
- optional EMA shadow model.
- optional activation checkpointing during training.
- optional `torch.compile(dynamic=True)`.

## Checkpoint And Inference Behavior

Training checkpoints store:

- model, optimizer, scheduler states
- optional EMA state
- train and validation losses
- train-split normalization
- architecture-critical `model_config`
- optional VAE prior diagnostics

Inference first loads normalization, then applies checkpoint `model_config` over
the runtime config. This is intentional for shape safety, but it means changing
architecture or prior keys only in an inference config may not take effect if
the checkpoint stored different values.

Rollout writes HDF5 files under `inference_output_dir` or `outputs/rollout` by
default. The saved nodal layout is:

```text
x, y, z, predicted output channels..., Part No.
```
