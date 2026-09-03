# MLP Surrogate (`mlp`)

> **The one non-mesh ML method.** Every other method in this suite predicts a
> *field on a mesh*; the MLP predicts a handful of **scalars from scalars**. It has
> no nodes, edges, timesteps, or positional features, so it uses a separate
> **tabular** data contract, not the shared mesh HDF5.

- **`model` value:** `mlp`
- **Repo / entrypoint:** `MLP/` · `MLP_main.py`
- **Modes:** `train` · `inference`
- **Family:** parametric regression / surrogate modeling
- **Own docs:** [MLP/CLAUDE.md](../../methods/MLP/CLAUDE.md)

## What it does

Learns a deterministic map **N scalar design/simulation parameters → M scalar
quantities of interest** — a surrogate for a DOE (design of experiments) or
parameter sweep. Typical use: replace a slow solver so an optimizer/UQ loop can
evaluate `[E, thickness, load, …] → [max_stress, max_disp, freq_1, …]` in
microseconds.

It is a plain fully-connected network — `Linear → [Norm] → Activation →
[Dropout]` blocks from `hidden_layers`, then a linear head with an optional
output activation ([MLP/mlp/model.py](../../methods/MLP/mlp/model.py)). The point of the
method is not architectural novelty; it is being a **first-class, launcher-routed,
preflight-validated** member of the suite for the common "I just have a table of
inputs and outputs" case.

## Data contract (tabular `X`/`Y` HDF5)

A single HDF5 per split — see
[dataset/DATASET_FORMAT.md](../reference/DATASET_FORMAT.md) → *Tabular Parametric
Dataset*:

```
X             float [num_samples, N]   input parameters   (must equal input_var)
Y             float [num_samples, M]   output quantities  (must equal output_var)
input_names   str   [N]                optional column labels
output_names  str   [M]                optional column labels
```

`Y` is optional for inference — when present, `infer.py` reports per-output
MAE/RMSE. The launcher validates this as `dataset_kind=table_hdf5`
([cae_suite/dataset_probe.py](../../cae_suite/dataset_probe.py)) and **cross-checks
`X`/`Y` widths against `input_var`/`output_var` before launch** (`DATASET-FEATURES-001/002`).
A tiny sample generator ships at
[dataset/mlp/make_sample.py](../../dataset/mlp/make_sample.py).

## Training details

[MLP/mlp/train.py](../../methods/MLP/mlp/train.py): AdamW with a linear-warmup→cosine LR
schedule, `mse` / `mae` / `huber` loss, a deterministic 80/10/10 train/val/test
split by `split_seed`, optional bf16 AMP and EMA. **Normalization
(`standard`/`minmax`/`none`) is fit on the train split only and stored in the
checkpoint**, so the source HDF5 stays read-only. The checkpoint carries
`selected_model="mlp"`, the architecture, the normalization stats (as plain lists,
so the launcher's `checkpoint_probe` reads it with `weights_only=True`), and the
EMA weights when enabled; inference prefers EMA.

> **EMA note:** decay is *warmed up* (timm/TF style — a low effective decay early)
> so EMA stays useful on short runs instead of retaining a chunk of the random
> initialization.

## When to use it

| If you need to… | This method does… |
| --- | --- |
| Turn a DOE / parameter-sweep table into a fast surrogate | Regress N inputs → M outputs directly |
| Predict scalar QoIs (max stress, drag, a few eigenfrequencies) from design variables | Exactly its job — no mesh required |
| Drive an optimizer or uncertainty-quantification loop cheaply | Microsecond forward passes on CPU |

**Not** for it: predicting a *field over a mesh* (use MGN / an operator /
Transolver), or generating geometry (SDFFlow). If your inputs/outputs are spatial
fields rather than a fixed-length scalar vector, this is the wrong method.

## Config keys

See [CONFIGURATION_REFERENCE.md §9.10](../CONFIGURATION.md) for the
full catalog; the launcher spec
[cae_suite/specs/mlp.py](../../cae_suite/specs/mlp.py) is the validation source of
truth. Minimal train example:

```
model mlp
mode train
gpu_ids -1
dataset_dir ../dataset/mlp/train.h5
modelpath ../output/mlp/ex1/mlp.pth
input_var 3
output_var 2
hidden_layers 256, 256, 128
activation gelu
loss mse
training_epochs 200
batch_size 32
learningr 0.001
```

Checked-in templates: `configs/MLP/ex1/config_train_mlp.txt` and
`config_infer_mlp.txt`.

## Dependencies

`torch`, `h5py`, `numpy` (no mesh/graph libraries; no GPU required — `gpu_ids -1`
runs on CPU). `native_probe=False`: there is no separate native config validator,
so the launcher's spec validator plus `MLP/mlp/config.py::validate` are the gates.

## Caveats

1. **Fixed-length scalar vectors only** — inputs and outputs are `N` and `M`
   numbers per sample; there is no notion of a variable mesh or spatial resolution.
2. **Column order is the contract** — `X`/`Y` columns must line up with the
   `input_var`/`output_var` the model was trained on. Inference re-checks `X`
   width against the checkpoint.
3. **Extrapolation is unguarded** — like any regressor, predictions far outside
   the training parameter box are unreliable; keep the DOE representative.
