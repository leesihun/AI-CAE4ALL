# MeshGraphNets-V SAOI FM-v2

One recipe, two board halves. Run the whole thing from the repository root:

```bash
bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh
```

That generates the configs, checks the training and evaluation data, trains
`bot` and `top` from scratch in parallel (one GPU each), then runs inference on
each of the three evaluation sets.

**Inference is what draws the figure.** With `make_histogram True` — which the
generated infer configs set — `inference_profiles/rollout.py` writes the
GT-versus-generated peak-to-valley histogram, `spread_values.npz` and the
machine-readable `[SPREAD]` log lines itself. There is no separate figure,
diagnostic or ranking stage in this pipeline.

Everything lands under:

```text
output/meshgraphnets-v/saoi_fm_v2_sweep/
    bot.pth  top.pth         checkpoints
    bot.log  top.log         training logs
    infer/                   rollout metrics, spread_values.npz, histogram PNGs
    run_logs/                per-stage launcher logs
```

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

## Why the FM-v2 keys are absent from these configs

This directory used to hold an eight-arm ablation over the flow-matching prior:
`bot`/`top` × P0–P3, where P0 was the untouched baseline, P1 widened the
velocity MLP 256 → 512, P2 replaced it with four residual FiLM blocks at width
512, and P3 added a graph-conditioned moment head `mu(c), L(c)` on top of P2.

That campaign is finished. **P0 won**, and P0 is exactly the native default, so
`gen_configs.py` now sets none of those twelve keys rather than restating them.

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

The retired arms' artifacts (`1.pth` … `8.pth`) are untouched in the same
output directory. `bot.pth`/`top.pth` are new names, which is why the arms were
renamed rather than renumbered: a renumbered arm `2` would load `2.pth`, the
retired bot-P1 checkpoint. The trained P0 arms can be kept without retraining:

```bash
cp output/meshgraphnets-v/saoi_fm_v2_sweep/1.pth output/meshgraphnets-v/saoi_fm_v2_sweep/bot.pth
cp output/meshgraphnets-v/saoi_fm_v2_sweep/5.pth output/meshgraphnets-v/saoi_fm_v2_sweep/top.pth
```

## What the histogram is scored against

For one evaluation case the condition is fixed:

```text
c = graph geometry + mesh topology + thickness/recorded condition rows
    + configured node/B.C. types
```

The `_infer_` file is ONE part's geometry; the matching `_compare_` file holds
125 physical realizations of that same part. That is why the target width ratio
is exactly **1** — the model draws 2000 samples under the condition and their
peak-to-valley spread should match the 125 real ones.

`check_eval_inputs.py` verifies the pairing before any GPU work, because an
`_infer_`/`_compare_` mismatch compares two different parts and reads as a
spread defect.

`split_seed 42` appears in the inference configs for the same reason it appears
in training: the split decides which 80% the normalizers are fit on, and those
normalizers denormalize every spread.

## Operational controls

The bare command runs everything. Stages can be repeated without editing
configs:

```bash
# re-run inference (and its histograms) on existing checkpoints
TRAIN=0 bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh

# one half, one evaluation set
ARMS=bot INFER_TAGS=s26fe_main TRAIN=0 \
bash configs/MeshGraphNets_Variational/SAOI_sweep/run_sweep.sh
```

Variables: `ARMS`, `INFER_TAGS`, `STAGGER`, `PYTHON`, and the stage switches
`GENERATE`, `PREFLIGHT`, `STRICT_PREFLIGHT`, `EVAL_PREFLIGHT`, `TRAIN`,
`INFER`.

Config regeneration and both preflights default to on. A preflight failure
aborts before any GPU work. After training, the script accepts only checkpoints
newer than each arm's launch marker, so an old checkpoint cannot be inferred
after a failed fresh run.
