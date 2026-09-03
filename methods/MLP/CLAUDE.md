# CLAUDE.md — MLP Surrogate

Guidance for Claude Code when working inside the `MLP/` method repo. This file is
**authoritative** for the MLP method's data contract and architecture; the
root [CLAUDE.md](../../CLAUDE.md) covers the launcher and cross-cutting conventions.

## What this repo is

A deliberately simple **parametric surrogate**: a fully-connected MLP that maps a
vector of **N scalar design/simulation parameters → M scalar quantities of
interest** (e.g. `[E, thickness, load] → [max_stress, max_disp]`). It is the
tabular regression method in the suite — it has **no meshes, edges, or
timesteps**, so it does *not* use the mesh HDF5 contract the other methods share.

Registered as launcher model `mlp` (repository `MLP/`, entrypoint `MLP_main.py`).
Runs through the suite launcher or standalone:

```bash
python AI_CAE4ALL_main.py --config configs/MLP/ex1/config_train_mlp.txt --check
python AI_CAE4ALL_main.py --config configs/MLP/ex1/config_train_mlp.txt
python MLP/MLP_main.py --config <abs-or-repo-relative config>   # standalone
```

`mode` lives inside the config: **`train`** or **`inference`**. Config paths
resolve from this repo root (`MLP/`).

## Dataset contract (tabular X/Y HDF5)

The **only** dataset format. A single HDF5 file per split:

| Object | Shape | Meaning |
| --- | --- | --- |
| `X` | `[num_samples, N]` float | input parameters (must equal `input_var`) |
| `Y` | `[num_samples, M]` float | output quantities (must equal `output_var`) |
| `input_names` | `[N]` str | optional column labels |
| `output_names` | `[M]` str | optional column labels |

`Y` is optional for inference (present → per-output MAE/RMSE are reported). The
launcher preflight (`dataset_kind="table_hdf5"`) cross-checks `X`/`Y` widths
against `input_var`/`output_var` before launch. See
[dataset/DATASET_FORMAT.md](../../docs/reference/DATASET_FORMAT.md).

## Architecture facts

- `mlp/model.py` — `Linear → [Norm] → Activation → [Dropout]` blocks from
  `hidden_layers`, then a linear head + optional `output_activation`.
- `mlp/data.py` — `load_xy` (HDF5 → float32) and `Normalizer`
  (`standard|minmax|none`, fit on the **train split only**, guards constant columns).
- `mlp/train.py` — AdamW + linear-warmup→cosine LR, `mse|mae|huber` loss,
  deterministic 80/10/10 split by `split_seed`, optional bf16 AMP and EMA.
- `mlp/infer.py` — loads the checkpoint (prefers EMA weights), inverse-normalizes,
  writes `predictions.h5` (`X`, `Y_pred`, and `Y_true` if provided).

**Normalization lives in the checkpoint, not the dataset** — source HDF5 stays
read-only. The checkpoint dict carries `model_state`, `selected_model="mlp"`,
`config` (architecture), `normalization` (input/output stats as plain lists so
the launcher's `checkpoint_probe` can read it with `weights_only=True`), and
`ema_state` when EMA is on.

## Config keys

Routing/optimizer/runtime keys are shared with the suite (`input_var`=N,
`output_var`=M, `training_epochs`, `batch_size`, `learningr`, …). MLP-specific:
`hidden_layers`, `activation`, `dropout`, `norm`, `input_normalization`,
`output_normalization`, `output_activation`, `loss`. The full catalog lives in
[CONFIGURATION_REFERENCE.md](../../docs/CONFIGURATION.md) section 9.10 and the
launcher spec `cae_suite/specs/mlp.py` (the validation source of truth).

## Tests

```bash
cd MLP && python -m pytest -q tests/test_mlp_pipeline.py
```

Builds a tiny synthetic X/Y HDF5, trains a few CPU epochs, and asserts a
checkpoint is written and inference produces a `[S, M]` prediction array.

## When you change something

- **New/renamed config key** → update `mlp/config.py` **and** the launcher spec
  `known_keys` in `cae_suite/specs/mlp.py`, or the launcher will reject a valid
  config or accept an invalid one.
- **New architecture option** → thread it through `mlp/config.py` (`Params` +
  `params_from_config`), `mlp/model.py`, and the checkpoint `config` block so
  inference can rebuild the model.
