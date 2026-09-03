# CLAUDE.md

This file provides guidance to Claude Code when working inside `SimulGenVAE/`.
For the monorepo/launcher conventions (routing, preflight, per-method venvs), see
the root [../CLAUDE.md](../../CLAUDE.md); this file is authoritative for
SimulGenVAE's own architecture, data contract, and validation steps.

## What this repo is

SimulGenVAE is a **hierarchical VAE** for parametric simulation fields plus a
**latent conditioner (LC)** that maps physical parameters (csv) or images to the
VAE's latent space, so a new set of conditions can be decoded into a full field
without ever re-running the FOM solver. It is routed through the AI-CAE4ALL
launcher as `model simulgenvae`:

```bash
python ../AI_CAE4ALL_main.py --config ../configs/SimulGenVAE/ex1/config_train.txt
```

It is **structurally the same shape as `methods/SDFFlow/` (SDFFlow)** — a VAE
stage plus a second generative stage — and mirrors that repo's layout, config
conventions, checkpoint format, and in-process distributed training almost
exactly. If you know SDFFlow, you already know most of this repo:

| SDFFlow | SimulGenVAE | Role |
| --- | --- | --- |
| `train_vae` | `train_vae` | Stage 1: train the (SDF / hierarchical) VAE |
| `train_fm` | `train_lc` | Stage 2: train the second network against frozen VAE latents |
| `train` | `train` | Sequential pipeline: stage 1 → stage 2, with checkpoint reuse |
| `sample`/`interpolate` | — | Not yet implemented for SimulGenVAE (see Follow-ups) |
| `reconstruct` | `reconstruct` | Conditions → LC → VAE decode → field HDF5 |

## Layout

```text
SimulGenVAE_main.py          # --config entrypoint; mode dispatch + mp.spawn for parallel_mode ddp
general_modules/
  load_config.py             # flat key/value parser (copy of the suite convention)
  distributed.py             # in-process DDP helpers (copy of SDFFlow's; verbatim)
  fom_dataset.py             # HDF5-native FOM loader: mesh HDF5 -> dense [N,C,T] tensor
model/
  vae.py                     # VAE(latent_dim_end, latent_dim, num_filter_enc, ...) -- hierarchical VAE
  encoder.py  decoder.py     # hierarchical encoder/decoder (1D conv blocks + per-level latents)
  common.py                  # weight init, spectral norm, residual blocks
  losses.py                  # KL divergence (main + per-level delta-KL)
  latent_conditioner.py      # MLP latent conditioner (csv conditions)
  latent_conditioner_cnn.py  # CNN latent conditioner (image conditions)
training_profiles/
  setup.py                   # device/optimizer/EMA/log/checkpoint helpers + build_vae/build_lc factories
  train_vae.py                # vae_worker(config, cfg_file) -- stage 1
  train_lc.py                 # lc_worker(config, cfg_file)  -- stage 2
  train_pipeline.py           # train_pipeline(config, cfg_file): VAE -> LC, with checkpoint-reuse logic
inference_profiles/
  reconstruct.py              # run_reconstruct(config, cfg_file) -- conditions -> field HDF5
tests/
  conftest.py                 # synthetic mesh-HDF5 fixtures
  test_fom_dataset.py          # loader shape/error tests + checkpoint round-trip
```

## Config conventions (mirrors SDFFlow's stage-prefix scheme)

Same flat `key value` format as every other method (`#` inline comment, `%`/`'`
line comments, comma/space → list, `true`/`false` → bool). **Values are
lowercased** by the shared parser, **except path-valued keys** (`dataset_dir`,
`param_dir`, `output_dir`, `*_modelpath`, the log dirs — see
`general_modules/load_config.py::PATH_KEYS`), which keep the case you wrote, so
mixed-case paths are fine.

Per-stage training knobs are carried with `vae_`/`lc_` prefixes in the combined
`train` mode and stripped to the unprefixed names each worker reads
(`training_profiles/train_pipeline.py::build_stage_config`,
`_STAGE_SETTING_SUFFIXES = log_file_dir, training_epochs, batch_size, learningr,
weight_decay, warmup_epochs, num_workers, use_amp, use_ema, ema_decay`). In
standalone `train_vae`/`train_lc` mode, use the **unprefixed** names directly
(see `configs/SimulGenVAE/ex1/config_train_vae.txt` / `config_train_lc.txt`).

Full key catalog: CONFIGURATION_REFERENCE.md §9.11 (root repo). Quick map from
the original interactive `condition.txt`/`preset.txt` naming, for anyone porting
an old config:

| Old (`condition.txt` / `preset.txt`) | New |
| --- | --- |
| `Training_epochs` | `vae_training_epochs` (or `training_epochs` in `train_vae`) |
| `Batch_size` | `vae_batch_size` / `batch_size` |
| `LearningR` | `vae_learningr` / `learningr` |
| `Latent_dim` | `latent_dim` (hierarchical, per level) |
| `Latent_dim_end` | `latent_dim_end` (main) |
| `Loss_type` | `loss_type` |
| `alpha` | `alpha` |
| preset.txt line 3 (`init_beta_divisor`) | `init_beta_divisor` |
| preset.txt line 4 (VAE filters) | `num_filter_enc` |
| preset.txt line 5 (LC filters) | `lc_filter` |
| `n_epoch` | `lc_training_epochs` / `training_epochs` (in `train_lc`) |
| `latent_conditioner_lr` | `lc_learningr` / `learningr` |
| `latent_conditioner_batch` | `lc_batch_size` / `batch_size` |
| `input_type` | `lc_data_type` (`image`→`image`, `csvs`→`csv`) |
| `param_dir` / `param_data_type` | unchanged |
| `latent_conditioner_dropout_rate` | `lc_dropout` |

Retired entirely: `input_data/condition.txt`, `preset.txt`, `launch_ddp.py`
(torchrun), the interactive `SimulGen-VAE.py` CLI, and `dataset*.pickle` loading
— all superseded by `--config` + the HDF5-native loader.

## Dataset: HDF5-native, fixed-geometry dense FOM

SimulGenVAE reads the **same mesh HDF5 contract** as the rest of the suite
(`data/{sample_id}/nodal_data[num_features, num_timesteps, num_nodes]`, rows
`0:3` reference coordinates, `3:` physical fields — see
`../dataset/DATASET_FORMAT.md`). Because the VAE is a dense 1D-conv network, it
needs a dense `[num_samples, num_channels, num_time]` tensor, built by
`general_modules/fom_dataset.py::load_fom_from_hdf5`:

```text
nodal_data[F, T, N]
  -> physical rows [field_start_row : field_start_row + num_var]   -> [num_var, T, N]
  -> optional node_start:node_end (N axis), timesteps_reduced (T axis)
  -> reorder to [T, num_var*N]
stack over samples -> [num_samples, T, num_var*N] -> transpose -> [num_samples, num_var*N, T]
```

**Every sample must share the same (T, N)** — this is inherent to the dense FOM
VAE, not a launcher limitation. A mismatch raises `ValueError` listing the
offending sample ids; `mesh_edge` is part of the shared contract but is not read.
`num_channels`/`num_time`/`num_samples` are derived from the file and written
back onto the in-memory config dict (not required as separate keys). Field values
are MinMax-scaled to `[-0.7, 0.7]` (the original SimulGenVAE convention); the
scaler is returned as a plain dict (`fit_minmax`/`apply_minmax`/`invert_minmax`)
so it folds into the checkpoint `normalization` payload rather than a loose
`.pkl` file.

`build_dataset_splits(config, split_seed)` does the seeded 80/10/10 split (suite
convention, `general_modules/fom_dataset.py`, same shape as SDFFlow's
`sdf_dataset.build_dataset_splits`).

LC conditioning inputs come from `read_conditions`, which has three sources:

| `lc_data_type` | Source | `param_dir` |
| --- | --- | --- |
| `csv` | headerless CSV `[num_samples, features]` | required |
| `image` | PNG/JPG under the directory (edge-filtered, /255) | required |
| `hdf5` | the mesh HDF5's own conditioning rows | **not used** |

For `csv`/`image`, rows/images must be ordered to match the mesh HDF5's **sorted
integer sample IDs**.

`lc_data_type hdf5` (`read_conditions_from_hdf5`) takes rows
`[field_start_row + num_var : ... + cond_var]` of `nodal_data` straight out of
`dataset_dir` — the same input-only conditioning rows the mesh methods consume
via `cond_var` (see `../dataset/DATASET_FORMAT.md`). It needs `cond_var >= 1`,
returns `[num_samples, cond_var]` in sorted-sample-ID order, and uses the MLP
conditioner like `csv` does. Because those rows are per-sample constants
broadcast to every node, node 0 recovers them exactly; the loader **verifies
that constancy** and raises if a row varies across nodes, so pointing `cond_var`
at a genuinely spatial field fails loudly instead of silently training on one
arbitrary node's value.

## Hierarchical latent space

Encoding a FOM sample yields a *dual* latent output:

- a **main** latent `z_main` of dim `latent_dim_end` (e.g. 32),
- a stack of **hierarchical** latents `z_hier`, one per encoder level below the
  top (`num_levels = len(num_filter_enc) - 1`), each of dim `latent_dim` (e.g. 8).

`model/encoder.py::Encoder.forward` returns `(mu, log_var, xs)` where `xs` is
that per-level list (already ordered coarse→fine for the decoder); `model/
decoder.py::Decoder.forward(z, xs, mode=...)` consumes the main latent plus the
per-level list, adding each level's residual KL (`losses.py::kl_2`) against the
previous level's prior. `mode='fix'` (near-deterministic, tiny std) is what
`inference_profiles/reconstruct.py` uses to decode LC predictions.

The **latent conditioner regresses both targets**: `training_profiles/
train_lc.py::_encode_latents` runs the frozen VAE encoder over every FOM sample to
produce `(main[N, Dm], hier[N, L, Dh])`, fits an independent MinMax scaler for
each, and trains the LC (`model/latent_conditioner.py` or `latent_conditioner_cnn.py`)
to predict both with loss `10 * MSE(main) + MSE(hier)` (weighting main latent
recovery higher, the original SimulGenVAE convention).

## Checkpoint format

Both stages save a dict payload via `training_profiles/setup.py::save_checkpoint`
(mirrors SDFFlow, and is what `cae_suite/checkpoint_probe.py` expects):

```text
vae: {stage: 'vae', epoch, model_state, ema_state, config, normalization,
      num_channels, num_time, num_filter_enc}
lc:  {stage: 'lc',  epoch, model_state, config,
      normalization: {main, hier, input}, num_levels, input_shape}
```

`config` always carries `model: simulgenvae`, so the launcher's checkpoint probe
validates model + stage consistency. There are no more loose `model_save/*.pkl`
scaler files or whole-model `torch.save(model)` pickles — everything a stage
needs to be reloaded standalone lives in its own checkpoint dict.

**Important:** `model/vae.py::VAE` is initialized with `add_sn` (spectral norm)
applied to every conv/linear layer, which renames each `weight` to
`weight_orig`/`weight_u`/`weight_v` in the state dict. Any code that rebuilds a
`VAE` and loads a saved `model_state` **must call `model.apply(add_sn)` before
`load_state_dict`** (see `train_lc.py` and `reconstruct.py`) — building it
"plain" will fail to load with a missing/unexpected-key mismatch.

## Multi-GPU (in-process, mirrors SDFFlow)

`general_modules/distributed.py` is copied verbatim from `methods/SDFFlow/`.
`SimulGenVAE_main.py` dispatches `train`/`train_vae`/`train_lc` through
`_train_dispatch(rank, world_size, gpu_ids, config, cfg_file)`; when
`parallel_mode` is `ddp`/`fsdp` and more than one GPU is listed, it self-spawns
one rank per GPU via `torch.multiprocessing.spawn` (no `torchrun` needed). Each
worker uses `worker_device`, `wrap_model`, `DistributedSampler`,
`reduce_epoch_mean`, `main_process_only`/`is_main_process`, and
`full_state_dict`; rank 0 owns logging, validation, and checkpoint writes. NCCL
needs Linux + CUDA — on Windows, or without CUDA, use `parallel_mode single`
(the default).

## Validation / test commands

```bash
# From the AI-CAE4ALL repo root:
python AI_CAE4ALL_main.py --describe simulgenvae
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/ex1/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/SimulGenVAE/ex1/config_train.txt --dry-run

# From this repo (method venv):
python -m pytest -q tests/test_fom_dataset.py
```

`--check`/`--dry-run` runs the full layered preflight: spec validation → path
checks → environment probe (imports `torch, numpy, h5py, sklearn` in the method's
Python) → dataset schema probe (`dataset_kind mesh_hdf5`, reuses
`cae_suite/dataset_probe.py::_mesh_report`) → native probe (imports
`general_modules.load_config`). A clean `--check` does not confirm fixed-geometry
uniformity across *all* samples (the schema probe only inspects the first); rely
on the loader's own `ValueError` (surfaced when the native process actually
trains) for that.

## Follow-ups / known gaps

- `sample`/`interpolate` modes (unconditional or interpolated generation without
  physical conditions) are not implemented — only conditioned reconstruction via
  `reconstruct`.
- Multi-GPU (`parallel_mode ddp`) is implemented per the SDFFlow pattern but has
  only been exercised via the CPU gloo path in this integration; validate NCCL/CUDA
  behavior on a real multi-GPU box before relying on it in production.
- Data augmentation (the original `audiomentations`-based pipeline) was not
  ported into the HDF5-native path; the current loader trains on the field data
  as-is.
