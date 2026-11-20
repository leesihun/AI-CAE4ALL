# 11 — MLP (tabular parametric surrogate)

## Summary

The MLP maps **N scalar inputs to M scalar outputs**, `X[S,N] → Y[S,M]`. It has
no mesh, nodes, edges or timesteps. Use it when you have a table from a DOE or a
parameter sweep and want a fast surrogate for scalar quantities of interest, for
example `[E, thickness, load] → [max_stress, max_disp, freq_1]`. It runs on CPU.

It is the only ML method here that does not read the mesh contract. For fields on
a mesh, use MGN, an operator or Transolver.

| | |
| --- | --- |
| Route id | `mlp` |
| Directory | [`methods/MLP/`](../../methods/MLP/) |
| Entrypoint | [`MLP_main.py`](../../methods/MLP/MLP_main.py) |
| Spec | [`ai_cae4all/specs/mlp.py`](../../ai_cae4all/specs/mlp.py) |
| Maintainer notes | [`methods/MLP/CLAUDE.md`](../../methods/MLP/CLAUDE.md) |

There is no paper. It is a plain fully connected regressor.

## Architecture

- [`mlp/model.py`](../../methods/MLP/mlp/model.py) (`MLP`, `build_model`)
  - One `Linear → [norm] → activation → [dropout]` block per width in
    `hidden_layers`.
  - A linear head with an optional `output_activation`.
- [`mlp/train.py`](../../methods/MLP/mlp/train.py)
  - Optimizer: AdamW, with linear warmup then a cosine schedule.
  - Loss: `mse`, `mae` or `huber`.
  - Precision: optional bf16 autocast, on CUDA only.
  - EMA: optional (`use_ema`), with decay warmed up as `min(decay, (1+s)/(10+s))`
    so short runs do not keep the random initialization.
- [`mlp/data.py`](../../methods/MLP/mlp/data.py)
  - Split: seeded 80/10/10 train/val/test. Fewer than 10 samples fall back to
    all-train. The test split is not used.
  - Normalization: `Normalizer` is fit on the train split only and stored in the
    checkpoint, as plain lists that `checkpoint_probe` reads with
    `weights_only=True`.
- [`mlp/infer.py`](../../methods/MLP/mlp/infer.py)
  - Checks the width of `X` against the checkpoint.
  - Prefers the EMA weights.
- [`mlp/config.py`](../../methods/MLP/mlp/config.py) is the native validator.
  The spec sets `native_probe False`, so the launcher's spec validator is the
  preflight gate.

## Data contract

The data is tabular HDF5, `dataset_kind table_hdf5`. See the "Tabular Parametric
Dataset" section of [DATASET_FORMAT.md](../reference/DATASET_FORMAT.md).

```text
X             float [S, N]   N must equal input_var
Y             float [S, M]   M must equal output_var; optional for inference
input_names   str   [N]      optional
output_names  str   [M]      optional
```

Preflight cross-checks the widths of `X` and `Y` against `input_var` and
`output_var` (`DATASET-FEATURES-001/002`). Two datasets ship:

- `dataset/deterministic/ex3_NASA_CRM_mlp_train.h5`: `X [105, 6]`, `Y [105, 4]`
- `dataset/deterministic/ex3_NASA_CRM_mlp_test.h5`: 44 samples

## Modes and keys

| Mode | Required |
| --- | --- |
| `train` | `dataset_dir`, `modelpath`, `input_var`, `output_var`, `training_epochs`, `batch_size`, `learningr` |
| `inference` | `modelpath`, `infer_dataset`, `input_var`, `output_var` |

| Key | Default | Choices |
| --- | --- | --- |
| `hidden_layers` | `256,256,128` | widths |
| `activation` | `gelu` | `relu`, `gelu`, `silu`, `tanh` |
| `norm` | `none` | `none`, `batch`, `layer` |
| `dropout` | 0 | |
| `input_normalization`, `output_normalization` | `standard` | `standard`, `minmax`, `none` |
| `output_activation` | `none` | `none`, `relu`, `sigmoid`, `tanh`, `softplus` |
| `loss` | `mse` | `mse`, `mae`, `huber` |
| `inference_output_dir` | `../../output/mlp/predictions` | |

Other keys:

- `weight_decay`, `warmup_epochs`, `max_grad_norm`
- `use_amp`, `use_ema`, `ema_decay`, `use_compile`
- `val_interval`, `checkpoint_interval`, `test_interval`, `display_testset`
- `plot_dpi`, `plot_max_points`, `split_seed`

Set `gpu_ids -1` for CPU. MLP is the one training method without
`resume_training`.

There is no checked-in MLP config. `configs/MLP/deterministic/` is empty, and MLP
is not in the dataset_matrix campaign. The Studio MLP template is the runnable
starting point. It reads the two ex3 tables and writes to
`../../output/mlp/studio/mlp.pth`.

## Outputs

| Artifact | Written by |
| --- | --- |
| `modelpath`, one file: written on each val improvement, every `checkpoint_interval` epochs, and at the end if val never improved | `train` |
| `parity_val_epoch#####.png`, one panel per output column, beside `log_file_dir` (or the checkpoint) | `train`, every `test_interval` |
| `<inference_output_dir>/predictions.h5` (`X`, `Y_pred`, `Y_true`), plus per-output MAE/RMSE printed to the console | `inference` |

## Tests and validation

```bash
cd methods/MLP && python -m pytest -q tests/
```

The tests are [`test_config_validation.py`](../../methods/MLP/tests/test_config_validation.py)
and [`test_mlp_pipeline.py`](../../methods/MLP/tests/test_mlp_pipeline.py).

## Known limitations

- **Fixed-length scalar vectors only.** There is no notion of mesh or spatial
  resolution.
- **Column order is the contract.** The columns of `X` and `Y` must match the
  trained model's.
- **Periodic saves overwrite the best one.** `checkpoint_interval > 0` writes to
  the same `modelpath`, so the file holds the last write, not necessarily the
  best validation epoch.
- **Extrapolation is unguarded.** Predictions outside the training parameter box
  are unreliable.
