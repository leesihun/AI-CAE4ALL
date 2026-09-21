# cHI-MGNflow SAOI

One recipe, two board halves. Run the whole thing from the repository root:

```bash
bash configs/HI_MGNFlow/SAOI_run/run_all.sh
```

That checks the training and evaluation data, trains `bot` and `top` from
scratch in parallel (one GPU each), then runs inference on each of the three
evaluation sets.

**Inference is what draws the figure.** With `make_histogram True` — which the
infer configs set — `inference_profiles/rollout.py` writes the GT-versus-
generated peak-to-valley histogram, `spread_values.npz` and the
machine-readable `[SPREAD]` log lines itself. There is no separate figure,
diagnostic or ranking stage.

Everything lands under:

```text
output/chi-mgnflow/saoi_run/
    bot.pth  top.pth         checkpoints (compressor + prior in one file)
    bot.log  top.log         training logs
    infer/<half>/<eval set>/ rollout metrics, spread_values.npz, histogram PNG
    run_logs/                per-stage launcher logs
```

## Files

| File | Purpose |
|---|---|
| `config_train_bot.txt` / `config_train_top.txt` | the two training jobs, GPU 0 and 1 |
| `config_infer_<half>_<tag>.txt` | 6 = 2 halves × 3 eval sets |
| `run_all.sh` | preflight → train → infer |
| `check_eval_inputs.py` | verifies each `_infer_`/`_compare_` pair before any GPU work |

Eval tags are `s26fe_main`, `s26fe_sec`, `sm_l345u_main`, matching
`configs/MeshGraphNets_Variational/SAOI_run/` so the two methods stay directly
comparable: same parts, same 2000 draws, same output layout under a different
`output/` root.

These configs are written directly — there is no generator. The eight-arm
hyperparameter sweep that used to live here, and the `gen_configs.py` that
rendered it, are gone; editing a config means editing the file.

## The two arms

| Arm | Data | GPU | Checkpoint |
|---|---|---|---|
| `bot` | bottom board half | 0 | `bot.pth` |
| `top` | top board half | 1 | `top.pth` |

They are not variants of one recipe — they are different datasets. The recipe
is identical and fixed; the two files differ only in `dataset_dir`, the output
paths and `gpu_ids`.

One card per half, so `parallel_mode ddp` runs at world_size 1. The two halves
are separate processes, not ranks — nothing is shared between them except the
hierarchy cache code path, which coordinates through an exclusive lock file and
is safe to run concurrently.

## What the model now is

cHI-MGNflow is **two-stage** (LDGN-style, after Lino, Pfaff & Thuerey, ICLR
2025). `mode train` runs both stages in one process:

| Stage | What it is | Epochs | Key knobs |
|---|---|---|---|
| 1 | near-lossless multiscale compressor (`HierarchyEncoder` + `MultiscaleDecoder`) putting the whole field on the **coarsest** mesh level as `latent_ch` channels per coarse node | `ae_epochs 600` | `latent_ch`, `ae_kl_weight`, `latent_dim` |
| 2 | flow-matching velocity prior over **that coarse latent only**, against a frozen stage 1 | `training_epochs 2000` | `prior_blocks`, `flow_time_freqs`, `flow_t_sampling` |

Two consequences run through every number in these configs:

- **Stage 1 sets the ceiling.** Stage 2 never sees the fine mesh except through
  the compressed latent, and stage 1 is frozen while it trains, so anything the
  compressor cannot represent is lost for good. `ae_epochs 600` is deliberately
  generous for that reason.
- **The ODE is cheap.** It integrates over ~100 coarse nodes, not the full
  mesh; one draw costs a single full-mesh encode/decode plus
  `flow_steps × 2` small forwards. That is why `val_flow_steps 20` is
  affordable during training and why `flow_steps` is a free dial at inference.

### Keys that no longer exist

These belonged to the retired single-stage, field-space flow. They are absent
from these configs on purpose, and the launcher flags them if they come back:

| Key | Diagnostic | Replacement |
|---|---|---|
| `flow_head`, `flow_head_eps` | `FLOW-HEAD-REMOVED` | `flow_loss_weighting x0` for the data-prediction effect |
| `pipeline_microbatches`, `noise_gamma`, `noise_std_ratio` | `FLOW-RUNTIME-REMOVED` (error) | — |
| `std_noise` | `FLOW-LEGACY-NOISE` | — MGN input noise is not implemented here |
| `message_passing_num` | `FLOW-MPNUM-INERT` | depth comes from `mp_per_level` |

`use_multiscale True` is now mandatory (`FLOW-FLAT`): a flat stack has no
coarse level to compress onto.

## If the first run underfits

The recipe writes the model's own defaults out explicitly rather than tuning
them, so there is one obvious place to look for each symptom:

| Symptom | Reach for | Why |
|---|---|---|
| reconstruction error high even before sampling | `latent_ch 4 → 8` | stage 1 is the ceiling; more channels per coarse node |
| reconstruction fine, draws too narrow or biased | `prior_blocks 4 → 6/8` | prior capacity, and it is cheap — the blocks run on coarse nodes |
| latent drifting or unstable | `ae_kl_weight` (>1e-3 raises `FLOW-AEKL-LARGE`) | the only regularizer on the latent scale |
| sample quality plateaus with steps | `flow_steps` at inference | sampling-time only; no retraining |

`flow_time_freqs` is the exception: it sets the AdaLN input width, so it is
architecture-defining and a checkpoint only loads under the value it was
trained with.

## What the histogram is scored against

For one evaluation case the condition is fixed:

```text
c = graph geometry + mesh topology + thickness/recorded condition rows
```

The `_infer_` file is ONE part's geometry; the matching `_compare_` file holds
the 125 physical realizations of that same part. That is why the target width
ratio is exactly **1** — the model draws 2000 samples under the condition and
their peak-to-valley spread should match the real ones. This is a calibration
measurement, not an accuracy measurement.

`check_eval_inputs.py` verifies the pairing before any GPU work, because an
`_infer_`/`_compare_` mismatch compares two different parts and reads as a
spread defect. It checks the node coordinates, mesh topology and conditioning
rows of every realization against the inference file, not just that both paths
exist.

`--check` reports `FLOW-COST` on each infer config: 2000 draws × 30 Heun steps
× 2 = 120,000 forwards per scene. That count is real but its cost model is
pre-rewrite — those forwards are now coarse-mesh only. The 2000 full-mesh
encode/decode passes are the actual expense, and 2000 is what MGN-V draws, so
the two histograms are built from the same sample count.

### No `split_seed` in the inference configs

Unlike the MGN-V campaign, these do not carry one. `rollout.py` takes the
normalization statistics **from the checkpoint** (`checkpoint['normalization']`,
and it refuses to run without them), so nothing at inference re-derives a split.

### What the checkpoint owns

`rollout.py` overwrites `config[k]` from `checkpoint['model_config']` for every
recorded key, **except** `flow_steps` and `flow_solver`, where the config wins.
So the architecture block repeated in each infer config is documentation: it
describes the run on its own and makes a mismatch print loudly, but it cannot
change the model. A key like `use_checkpointing` set in an infer config would
be silently overridden — which is why none of them set one.

## Dataset paths are case-sensitive

`dataset_dir`, `infer_dataset` and `eval_dataset` are all in `PATH_KEYS`
(`general_modules/load_config.py`, mirrored in `cae_suite/config_parser.py`),
so the case written in the config is the case opened. Keep `SAOI/`,
`S26FE-MAIN`, `S26FE-SEC` and `SM-L345U-MAIN` uppercase as written.

## Operational controls

The bare command runs everything. Stages can be repeated without editing
configs:

```bash
# re-run inference (and its histograms) on existing checkpoints
TRAIN=0 bash configs/HI_MGNFlow/SAOI_run/run_all.sh

# one half, one evaluation set
ARMS=bot INFER_TAGS=s26fe_main TRAIN=0 \
bash configs/HI_MGNFlow/SAOI_run/run_all.sh
```

Variables: `ARMS`, `INFER_TAGS`, `STAGGER`, `PYTHON`, and the stage switches
`PREFLIGHT`, `STRICT_PREFLIGHT`, `EVAL_PREFLIGHT`, `TRAIN`, `INFER`.

Both preflights default to on. A preflight failure aborts before any GPU work.
After training, the script accepts only checkpoints newer than each arm's
launch marker, so an old checkpoint cannot be inferred after a failed fresh
run — which matters here because there is **no resume**: `cosine_T0` is
`epochs - warmup`, so a killed job restarts from zero and one launch is one
complete model.

`hierarchy_cache_keep True` leaves each half's multiscale cache beside its
training HDF5. Nothing deletes it for you:

```bash
rm -f dataset/SAOI/saoi_train_*.mscache.*.h5
```
