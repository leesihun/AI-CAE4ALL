# 07 — FNO (mesh-adapted Fourier Neural Operator)

- **`model`**: `fno`
- **Repo / entrypoint**: `Neural_Operator/` → `main.py`
- **Key source**: `model/fno.py`, `model/spectral.py`, `model/adapters/grid.py`
- **Prereqs**: [00_shared_foundations.md](00_shared_foundations.md) (§1 data)

---

## What it does

FNO learns a solution operator by working in the **frequency domain**: it lifts the
input field to a set of channels, applies several **spectral convolution layers**
(pointwise multiplication of the low Fourier modes by learned weights), and projects
back. Because a spectral convolution is **global** over the grid, one FNO layer
couples the whole domain at once — the opposite of MGN's local message passing.

FNO needs a **regular grid**. Since this suite's data lives on irregular meshes, the
model is **"mesh-adapted"**: a deterministic **splat** projects the ragged mesh onto a
fixed grid (`fno_grid_resolution`), the spectral core runs, and a final
**`grid_sample`** interpolates predictions back to the original node coordinates. The
splat/sample projection error is an explicit part of this baseline.

The spectral core is **native** (`model/spectral.py`) — no external `neuraloperator`
dependency. A `paper_darcy` variant reproduces the original Li et al. 2021 Darcy-flow
FNO exactly for benchmarking.

---

## Capabilities

- **Global receptive field per layer** via Fourier modes — long-range coupling is free.
- **N-dimensional** spectral conv (`d ∈ {2,3}`, auto-detected via `operator_dim`).
- **Cheap once on the grid**: cost scales with grid size and retained modes, not mesh
  node count.
- **`paper_darcy` variant** for exact benchmark reproduction (85×85 grid, ReLU blocks,
  paper init).
- **Pipeline model-split** across GPUs (`parallel_mode model_split`): entry (splat +
  lifting) → latent FNO blocks → exit (projection + grid_sample).
- **Activation checkpointing** on the spectral blocks (exact recomputation).
- Static or autoregressive temporal; DDP.

## Strengths

- **Excellent for smooth, globally-coupled fields** on domains that are close to
  grid-like — spectral bias favors smooth solutions.
- **Fast and parameter-efficient** relative to dense global models: only the retained
  low modes carry weights.
- **Depth-independent global mixing** — a few layers already see the whole domain.
- **fp32 FFT islands** keep the spectral math numerically safe even under bf16 autocast.

## Weaknesses

- **Grid projection is lossy for irregular geometry**: splatting a thin/curved mesh
  onto a coarse grid blurs boundaries — the main accuracy limiter here.
- **Grid resolution × modes cost**: memory/params grow with `∏resolution` and
  `∏modes`; high resolution is expensive.
- **Mode truncation caps detail**: features above `fno_modes` are simply not
  represented.
- **Not mesh-native**: no notion of connectivity; occupancy/density channels partly
  compensate but do not restore topology.
- Resolution/modes are **architecture changes, not memory knobs**.

---

## Network structure

```mermaid
flowchart TD
    M["mesh nodes: graph.x (+sdf)"] --> SPL["splat → regular grid\n[field | occupancy | density | coords | conditions]"]
    SPL --> LIFT["Lifting (1×1 conv)\nin_channels → hidden"]
    LIFT --> BLK

    subgraph BLK["fno_layers × Fourier block"]
        direction TB
        SC["SpectralConvNd\n(rFFT → truncate to modes → learned weights → irFFT)"] --> ADD["+"]
        PW["pointwise 1×1 conv"] --> ADD
        ADD --> CMLP["+ optional channel MLP"]
        CMLP --> ACT["GELU"]
    end

    BLK --> PROJ["Projection (1×1 conv → GELU → 1×1 conv)\nhidden → output_var"]
    PROJ --> SMP["grid_sample back to node coords"]
    SMP --> OUT["prediction [N, output_var]"]
```

### Grid assembly (`_assemble_grid`)

`splat(values, coords, …, resolution)` yields the field grid plus **occupancy** and
**density** maps; concatenated with a normalized **coordinate grid** and (optional)
broadcast global conditions. The `paper_darcy` variant instead uses exactly
`[coefficient, x, y]` on an identity 85×85 grid.

### Spectral convolution (`model/spectral.py::SpectralConvNd`)

Standard FNO corner-truncation scheme (Li et al. 2021):

- `torch.fft.rfftn` over spatial dims (**fp32**, autocast disabled).
- `2^(d-1)` learned weight blocks cover every sign combination of the non-final axes
  (the real-FFT halves the final axis).
- Each retained corner is multiplied by its complex weight
  (`einsum('bi...,io...->bo...')`); the rest are zero.
- `torch.fft.irfftn` back to real space.
- Weights are stored as **real tensors with a trailing size-2 dim** and viewed as
  complex only inside `forward`, so fused AdamW (which rejects complex params) works.

### Block, lifting, projection

Each block = `SpectralConvNd + pointwise 1×1 conv (+ channel MLP) → GELU`. Lifting and
projection are 1×1 convolutions (`Conv2d`/`Conv3d` with kernel 1 = per-position
Linear). For temporal runs the projection's last layer starts scaled by `0.01`.

---

## Shared Neural-Operator config keys

**These keys are common to all three backends** (`deeponet`, `point_deeponet`,
`fno`) and are referenced from the other operator docs.

### Execution, dataset, common shape

| Key | Meaning |
| --- | --- |
| `model` / `mode` / `gpu_ids` | Backend selector, `train`/`inference`, device id(s) |
| `parallel_mode` | `ddp` or `model_split` (fno only, ≥2 GPUs, `augment_geometry False`) |
| `log_file_dir` / `modelpath` | Log path / checkpoint path |
| `dataset_dir` / `infer_dataset` / `inference_output_dir` / `infer_timesteps` | Data + rollout I/O |
| `split_seed` / `split_group_attr` | Deterministic 80/10/10 split seed (default 42; `training_profiles/setup.py:41`); optional HDF5 attribute to group samples by so a group never straddles the split |
| `input_var` / `output_var` / `cond_var` / `feature_loss_weights` | Predicted-channel counts, trailing **input-only** conditioner rows, per-channel weights |
| `positional_features` / `use_node_types` | Extra node features / one-hot node types |
| `operator_dim` | `auto`/`2`/`3` spatial dimensionality |
| `coordinate_normalization` | Must be `centered_isotropic` |
| `dimension_tolerance` / `grid_padding` / `out_of_bounds_policy` | Domain fitting + OOB policy |
| `sdf_source` / `sdf_sidecar` / `global_condition_features` / `integration_weight_source` | `sdf_source` is `none`/`dataset`/`sidecar`/`mesh`; the other two **must** be `none` (no shipped dataset supplies conditions or quadrature weights) |

### Optimization & runtime

| Key | Meaning |
| --- | --- |
| `training_epochs` / `batch_size` / `learningr` / `weight_decay` / `warmup_epochs` | AdamW + schedule |
| `num_workers` / `prefetch_factor` / `grad_accum_steps` / `max_grad_norm` | Loader + step controls |
| `use_parallel_stats` / `train_eval_subset_size` / `profile_batches` / `pipeline_microbatches` | Distributed normalizer reduction, train-set eval subsample, profiling, model-split microbatching |
| `std_noise` / `noise_gamma` / `noise_std_ratio` / `augment_geometry` | Noise injection + geometry augmentation |
| `use_amp` / `use_checkpointing` / `use_ema` / `ema_decay` / `use_compile` | Precision, memory, EMA, compile |
| `train_query_chunk_size` / `infer_query_chunk_size` | Query-decode chunking (exact) |
| `val_interval` / `test_interval` / `test_max_batches` / `test_batch_idx` / `plot_feature_idx` / `display_*` / `write_test_predictions` / `checkpoint_interval` | Evaluation & visualization |
| `write_preprocessing` / `use_world_edges` / `use_multiscale` | Must stay `False` (operators ignore MGN edges/hierarchy) |
| `time_integration` | `ar_ot` (default) or `ar_rt` |

### FNO-specific keys

| Key | Meaning |
| --- | --- |
| `fno_variant` | `mesh` (default) or `paper_darcy` (exact Darcy benchmark) |
| `fno_grid_resolution` | Regular grid size per active axis (comma list, each ≥ 2) |
| `fno_modes` | Retained Fourier modes per axis (validated against resolution) |
| `fno_hidden_channels` | Spectral channel width (default 64) |
| `fno_layers` | Spectral block count (default 4) |
| `fno_use_channel_mlp` | Add a channel MLP in each block (default True) |
| `fno_norm` | `none` (only accepted in this baseline) |

### FNO config sketch

```text
model               fno
mode                train
dataset_dir         ../../dataset/deterministic/ex1_static_thermoelastic.h5
input_var           4
output_var          4
cond_var            0
positional_features 4
use_node_types      True
operator_dim        2
fno_variant         mesh
fno_grid_resolution 128, 32
fno_modes           16, 12
fno_hidden_channels 64
fno_layers          4
fno_use_channel_mlp True
fno_norm            none
```

> Paths are **cwd-relative to `methods/Neural_Operator/`**, hence `../../dataset/...`
> and `../../output/...`. The excerpt is the canonical ex1 config verbatim: the grid
> is **anisotropic** (`128, 32`), fitted to the domain's aspect ratio, not the square
> grid a sketch invites. `fno_modes[i]` is capped at `resolution[i]//2` (`//2 + 1` on
> the last axis) by `general_modules/config_validation.py`.
