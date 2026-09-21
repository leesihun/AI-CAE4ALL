# MeshGraphNets-V SAOI

One recipe, two board halves. Run the whole thing from the repository root:

```bash
bash configs/MeshGraphNets_Variational/SAOI_run/run_all.sh
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
output/meshgraphnets-v/saoi_run/
    bot.pth  top.pth         checkpoints
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
`configs/HI_MGNFlow/SAOI_run/` so the two methods stay directly comparable.

These configs are written directly — there is no generator. The
`SAOI_all_input` production profile they used to be rendered from is gone, so
editing a config means editing the file.

## The two arms

| Arm | Data | GPU | Checkpoint |
|---|---|---|---|
| `bot` | bottom board half | 0 | `bot.pth` |
| `top` | top board half | 1 | `top.pth` |

They are not variants of one recipe — they are different datasets. The recipe
is identical and fixed: `training_seed 20260916`, `split_seed 42`, batch size
16, 1000 epochs with the encoder/decoder frozen at epoch 700 so the last 300
fit only the prior on a fresh cosine, `best_by crps`, and no FM gradient into
the posterior encoder (`prior_grad_to_encoder 0.0`).

One card per half, so `parallel_mode ddp` runs at world_size 1. The two halves
are separate processes, not ranks — nothing is shared between them.

## Why there are no FM-v2 keys in these configs

This work used to be an eight-arm ablation over the flow-matching prior:
`bot`/`top` × P0–P3, where P0 was the untouched baseline, P1 widened the
velocity MLP 256 → 512, P2 replaced it with four residual FiLM blocks at width
512, and P3 added a graph-conditioned moment head `mu(c), L(c)` on top of P2.

That campaign is finished. **P0 won**, and P0 is exactly the native default, so
these configs set none of those twelve keys rather than restating them.

| Question | Verdict |
|---|---|
| Does velocity capacity help? (P1, 256 → 512) | **No — actively harmful.** The largest replicated effect in the sweep: `sd_ratio` fell 9.5% on `bot` and 29.9% on `top`. |
| Does time/condition modulation help? (P2, residual FiLM) | **No — a clean null** in both halves. |
| Do explicit conditional moments help? (P3) | Helped the conditional mean, not the width. |

P0 won `sd_ratio`, `PITtails` and `PIT_KS` in **both** halves. Its single loss
was `W1/sd`, by under 2% — inside single-seed noise, and a conditional-mean
metric rather than the width and calibration this model exists to get right.

The defaults that carry that verdict, for the record:

```text
prior_fm_velocity_arch          mlp     two-hidden-layer velocity MLP
prior_velocity_hidden_dim       = prior_hidden_dim, 256
prior_fm_moments                False   global standardization only
prior_moment_calibration_epochs 0       no moment stage
```

The 1000/700 schedule matters for the same reason: that is what the ablation
was measured under. Moving to a longer schedule, or opening
`prior_grad_to_encoder`, makes a run incomparable to the evidence that picked
the recipe — it would need its own controls.

## What the histogram is scored against

For one evaluation case the condition is fixed:

```text
c = graph geometry + mesh topology + thickness/recorded condition rows
    + configured node/B.C. types
```

The `_infer_` file is ONE part's geometry; the matching `_compare_` file holds
the physical realizations of that same part. That is why the target width ratio
is exactly **1** — the model draws 2000 samples under the condition and their
peak-to-valley spread should match the real ones. This is a calibration
measurement, not an accuracy measurement.

`check_eval_inputs.py` verifies the pairing before any GPU work, because an
`_infer_`/`_compare_` mismatch compares two different parts and reads as a
spread defect.

`split_seed 42` appears in the inference configs for the same reason it appears
in training: the split decides which 80% the normalizers are fit on, and those
normalizers denormalize every spread.

## Dataset paths are case-sensitive

`dataset_dir`, `infer_dataset` and `eval_dataset` are all in `PATH_KEYS`
(`general_modules/load_config.py`, mirrored in `cae_suite/config_parser.py`),
so the case written in the config is the case opened. A stale lowercase twin of
`dataset/SAOI/` exists on disk holding a different vintage; it opens without
error and yields wrong numbers. Keep `SAOI/`, `S26FE-MAIN`, `S26FE-SEC` and
`SM-L345U-MAIN` uppercase as written.

## Operational controls

The bare command runs everything. Stages can be repeated without editing
configs:

```bash
# re-run inference (and its histograms) on existing checkpoints
TRAIN=0 bash configs/MeshGraphNets_Variational/SAOI_run/run_all.sh

# one half, one evaluation set
ARMS=bot INFER_TAGS=s26fe_main TRAIN=0 \
bash configs/MeshGraphNets_Variational/SAOI_run/run_all.sh
```

Variables: `ARMS`, `INFER_TAGS`, `STAGGER`, `PYTHON`, and the stage switches
`PREFLIGHT`, `STRICT_PREFLIGHT`, `EVAL_PREFLIGHT`, `TRAIN`, `INFER`.

Both preflights default to on. A preflight failure aborts before any GPU work.
After training, the script accepts only checkpoints newer than each arm's
launch marker, so an old checkpoint cannot be inferred after a failed fresh run.
