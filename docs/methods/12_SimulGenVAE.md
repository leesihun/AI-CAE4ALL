# 12 — SimulGenVAE (hierarchical VAE + latent conditioner)

## Summary

SimulGenVAE compresses whole simulation responses into a hierarchical latent
space. A separate **latent conditioner (LC)** then maps design conditions to that
latent space, and decoding the predicted latent gives the full response.

It is a **fixed-geometry, dense** model. It reads the shared mesh HDF5, but every
sample must have the same node count and timestep count. The mesh is never used
as a graph.

| | |
| --- | --- |
| Route id | `simulgenvae` |
| Directory | [`methods/SimulGenVAE/`](../../methods/SimulGenVAE/) |
| Entrypoint | [`SimulGenVAE_main.py`](../../methods/SimulGenVAE/SimulGenVAE_main.py) |
| Spec | [`ai_cae4all/specs/simulgenvae.py`](../../ai_cae4all/specs/simulgenvae.py) |
| Maintainer notes | [`methods/SimulGenVAE/CLAUDE.md`](../../methods/SimulGenVAE/CLAUDE.md) |
| Paper | LSH-VAE, Engineering with Computers (2023), [doi:10.1007/s00366-023-01916-6](https://doi.org/10.1007/s00366-023-01916-6) |

The campaign key is `lsh_vae`, after the paper. It is part of the
`output/dataset_matrix/` run paths, so it stays.

## Architecture

The field tensor is `[B, C, T]` with `C = num_var × N`. Each node-variable pair
is a channel, and time is the sequence axis. All convolutions are 1-D over time.

**VAE** ([`model/vae.py`](../../methods/SimulGenVAE/model/vae.py))

- **Encoder** ([`model/encoder.py`](../../methods/SimulGenVAE/model/encoder.py))
  - Blocks: one `ConvBlock` + residual block per entry of `num_filter_enc`.
    `network_size small` uses 1×1 convs; `large` adds a k=3 conv.
  - Levels: each level projects its `[C_i, T]` features to a `latent_dim` vector
    `xs_i` through a linear layer.
  - Main latent: the deepest features give the main latent's `mu`, `log_var`
    (size `latent_dim_end`).
  - The encoder returns the `len(num_filter_enc) − 1` shallower `xs`, deepest
    first.
- **Decoder** ([`model/decoder.py`](../../methods/SimulGenVAE/model/decoder.py))
  - Filters mirror the encoder.
  - Input: `z` (sampled from the main posterior) is lifted to `[C, T]`, then
    each level applies a `ConvTranspose1d` block and a residual block.
  - Per-level latent (NVAE-style, a residual posterior):
    - `condition_z` gives a prior `(mu, log_var)` from the decoder features.
    - `condition_xz` gives `(Δmu, Δlog_var)` from `[xs_i | decoder features]`.
    - The posterior is `N(mu + Δmu, exp(log_var + Δlog_var))`.
    - The sampled `[C_i, T]` latent is added to the next level's input. In
      `mode='fix'` (reconstruction) the mean is used instead of a sample.
  - Output: a 1×1 conv to `C` channels, then `Tanh`, with no normalization in
    between.
- **Norms**: GroupNorm with one group throughout, i.e. LayerNorm over
  `(C, T)` (`group_norm_groups`).
- **Spectral norm** is applied to every conv, transposed conv and linear layer
  (`model/common.py::add_sn`), both in training and at reconstruction.

**Losses** ([`model/losses.py`](../../methods/SimulGenVAE/model/losses.py))

```text
recon    loss_type 1 MSE | 2 MAE | 3 SmoothL1 | 4 Huber
KL_main  0.5 Σ (mu² + σ² − log σ² − 1)
KL_i     0.5 Σ (Δmu²/σ² + Δσ² − log Δσ² − 1)        per hierarchical level
loss     alpha · recon + beta · (KL_main + Σ KL_i)
```

The KL weight `beta` follows a schedule (`train_vae.py::_kl_beta`):

1. It holds at `10^(−init_beta_divisor)` until `kl_warmup_start_frac ×` total
   epochs.
2. It then ramps linearly to `beta_target` over `kl_warmup_epochs`. With the
   default of 0, the ramp ends at 0.8 × total.

**Latent conditioner** ([`model/latent_conditioner.py`](../../methods/SimulGenVAE/model/latent_conditioner.py),
[`model/latent_conditioner_cnn.py`](../../methods/SimulGenVAE/model/latent_conditioner_cnn.py))

- **Training data**: the frozen VAE encodes every sample. The LC regresses two
  targets:
  - main: `mu`, shape `[S, latent_dim_end]`
  - hier: the stacked `xs`, shape `[S, L, latent_dim]`
- **Loss**: `10 · MSE(main) + MSE(hier)`, with gradient clipping at 10. Targets
  and csv/hdf5 inputs are min-max scaled on the train split.
- **Model**: `LatentConditioner` (an MLP trunk with two heads, widths
  `lc_filter`) serves `csv` and `hdf5` inputs. `LatentConditionerImg` (ResNet
  blocks with squeeze-excitation and optional spatial attention) serves `image`
  inputs.

**Reconstruct** ([`inference_profiles/reconstruct.py`](../../methods/SimulGenVAE/inference_profiles/reconstruct.py))

1. Run conditions through the LC and invert the min-max scaling.
2. Decode with `vae.decoder(z, xs, mode='fix')`.
3. Unscale with the **VAE checkpoint's** normalization. It is never refit on
   the new file.

**Training setup** ([`training_profiles/setup.py`](../../methods/SimulGenVAE/training_profiles/setup.py)):
fused AdamW, with linear warmup into `CosineAnnealingWarmRestarts`.

## Data contract

The input is the mesh HDF5 of [DATASET_FORMAT.md](../reference/DATASET_FORMAT.md),
read densely by
[`general_modules/fom_dataset.py`](../../methods/SimulGenVAE/general_modules/fom_dataset.py):

- **Field rows** are `[field_start_row : field_start_row + num_var]` of
  `nodal_data`. `field_start_row` defaults to 3, so coordinates are skipped.
- **Shapes are fixed.** Every sample must share `(T, N)`; a mismatch is a hard
  error. `node_start` / `node_end` crop nodes and `timesteps_reduced` subsamples
  time.
- **Scaling** is min-max to `(−0.7, 0.7)`, fit on the train split and stored in
  the checkpoint.
- **Splits** are a seeded 80/10/10 over sorted sample ids, and need at least 3
  samples. The LC stage must see the same split manifest and seed as the VAE.

LC inputs (`lc_data_type`):

| Value | Source |
| --- | --- |
| `hdf5` | the `cond_var` rows after the field rows. They must be constant per sample; needs `cond_var ≥ 1`, no `param_dir` |
| `csv` | headerless CSV at `param_dir`, one row per sample, in sorted-id order |
| `image` | images under `param_dir` (`param_data_type`, default `.png`), resized to 256², binarized, flattened |

Checked-in slots are under `configs/SimulGenVAE/deterministic/<slot>/baseline/`:

| Slot | Training file | Fields | LC input |
| --- | --- | --- | --- |
| `ex3_full` | `ex3_NASA_CRM_full.h5` | `num_var 4` | `hdf5`, `cond_var 6` |
| `ex3_mid` | `derived/config_matrix/ex3_NASA_CRM_mid_canonical.h5` | `num_var 4` | `hdf5`, `cond_var 6` |
| `ex6` | `ex6_flag_simple.h5` | `num_var 3` | `csv` (`derived/config_matrix/ex6_flag_simple_conditions.csv`) |
| `ex9` | `ex9_plasticity.h5` | `num_var 2` | `csv` (`derived/config_matrix/ex9_plasticity_conditions.csv`) |

Each slot has a `config_train_lsh_vae.txt` and a `config_infer_lsh_vae.txt`. The
infer config runs `reconstruct` on the slot's `_infer.h5` file.

## Modes and keys

| Mode | Runs | Required beyond the architecture keys |
| --- | --- | --- |
| `train` | VAE, then LC | `dataset_dir`, `vae_modelpath`, `lc_modelpath`, `vae_*` / `lc_*` `training_epochs`, `batch_size`, `learningr` |
| `train_vae` | VAE only | `dataset_dir`, `vae_modelpath`, unprefixed `training_epochs`, `batch_size`, `learningr` |
| `train_lc` | LC only, on a frozen VAE | as above, plus `lc_modelpath` |
| `reconstruct` | LC → decoder on a new file | `dataset_dir`, both model paths, `output_dir` |

The architecture keys are `num_filter_enc`, `latent_dim` and `latent_dim_end`.
Every mode except `train_vae` also needs `lc_filter` and `lc_data_type`.
`param_dir` is required for `csv` and `image` LC inputs.

**Prefixes.** In `train` mode, per-stage keys carry `vae_` / `lc_` prefixes.
[`train_pipeline.py`](../../methods/SimulGenVAE/training_profiles/train_pipeline.py)
strips the prefix for each stage. **Skipping.** With `skip_completed_stages`
(default true), the pipeline reuses a stage whose checkpoint is complete and
matches the config.

| Key | Default | Notes |
| --- | --- | --- |
| `num_var` / `field_start_row` | 1 / 3 | field rows |
| `network_size` | `small` | `small`, `large` |
| `loss_type` | 1 | 1 MSE, 2 MAE, 3 SmoothL1, 4 Huber |
| `alpha` | 1.0 | recon weight; the checked-in configs use `1e6` |
| `init_beta_divisor`, `beta_target`, `kl_warmup_start_frac`, `kl_warmup_epochs` | 4, 1.0, 0.3, 0 | KL schedule above |
| `lc_dropout`, `use_spatial_attention` | — | LC regularization; image LC |
| `parallel_mode` | `single` | `single`, `ddp`, `fsdp` |
| `split_seed` | 42 | must match between stages |
| `val_interval`, `test_interval`, `display_testset`, `num_test_samples`, `plot_dpi` | — | validation and pictures |
| `*_use_amp`, `*_use_ema`, `*_ema_decay` | off | per stage |
| `resume_training`, `resume_interval_minutes` | off, 15 | single-GPU only |

`load_all`, `plot_mode` and `recon_iter` are rejected by the spec with
`SGV-REMOVED-NOOP`.

**Multi-GPU.** With `parallel_mode ddp` (or `fsdp`) and more than one GPU in
`gpu_ids`, the training modes spawn one worker per GPU in-process (`mp.spawn`,
[`general_modules/distributed.py`](../../methods/SimulGenVAE/general_modules/distributed.py)).

## Outputs

Paths come from the config. The checked-in slots write to
`../../output/dataset_matrix/deterministic/<slot>/lsh_vae/`:

| Artifact | Contents |
| --- | --- |
| `vae.pth` | `stage 'vae'`, model and EMA state, config, normalization, split manifest, `num_channels`, `num_time`, `num_filter_enc` |
| `lc.pth` | `stage 'lc'`, LC state, latent and input normalization, `num_levels` |
| `pipeline.log`, `vae.log`, `lc.log` | stage logs |
| `vae_recon_epoch#####.png`, `lc_parity_epoch#####.png` | periodic truth / reconstruction / error, and latent parity, beside the logs |
| `infer/reconstructions.h5` | `data/{sid}/nodal_field [num_var, T, N]`; the console prints the mean field MSE |

## Tests and validation

```bash
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/deterministic/ex6/baseline/config_train_lsh_vae.txt --check
cd methods/SimulGenVAE && python -m pytest -q tests/
```

The tests cover:

- the config contract
- dense HDF5 loading
- HDF5 conditions
- reconstruct normalization
- split provenance

## Known limitations

- **One mesh per model.** Different node or timestep counts need a new model or
  resampling.
- **Memory scales with N.** The first layers are `num_var × N` channels wide, so
  large meshes are expensive.
- **No sampling modes.** `sample` and `interpolate` are not ported; generation is
  condition → LC → decoder only. Data augmentation is not ported either.
- **The shallowest `xs` is never decoded.** The decoder conditions on only
  `len(num_filter_enc) − 2` of the `len(num_filter_enc) − 1` encoder levels; the
  last level breaks before conditioning. The shallowest `xs` is still encoded and
  regressed by the LC.
- **DDP/NCCL not validated.** Multi-GPU has not been validated on a real NCCL
  cluster.
