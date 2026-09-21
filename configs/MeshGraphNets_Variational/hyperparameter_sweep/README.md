# MeshGraphNets-V SAOI — hyperparameter sweep

Eight arms, trained independently on both halves of the SAOI part (`bot` and
`top` — 16 training jobs total, nothing shared between the halves). Run one
half at a time from the repository root:

```bash
bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run_bot.out 2>&1

HALF=top bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep_top/run_top.out 2>&1
```

Train → infer → posterior-vs-prior → one report, per half. See "Running both
halves" below for why these run one after the other, not backgrounded together.

## What this sweep is aimed at

Posterior reconstruction is excellent while the generated ensemble is **~2×
too narrow** — `sd_ratio` 0.45–0.53 on every arm of the retired FM-v2 ablation,
with `|dmean|/sd < 0.3`. The location is right; the width is not. Both MGN-V
and cHI-MGNflow show it, which is the signature of a structural problem rather
than a badly chosen hyperparameter.

`misc/posterior_vs_prior.py` names the mechanism: **the prior puts its variance
in latent directions the decoder does not read.** An inflation sweep already
measured the consequence — 2.5× on `z` bought only 1.35× on the field.

So the axes below are not capacity axes. The retired eight-arm ablation
(`bot`/`top` × P0–P3) already exhausted that question and the answer was that
capacity is not the bottleneck: widening the velocity MLP 256 → 512 was the
largest replicated effect in the whole campaign and it made `sd_ratio` **worse**
(−9.5% bot, −29.9% top). Every arm here changes *where the variance goes*, not
how much model there is.

## The eight arms

Every arm is exactly one config key away from `base`, which is the SAOI_run
recipe verbatim for its half (`bot` or `top`). The swept line in each file is
tagged `# SWEPT` and carries its own rationale, so a config read on its own
explains itself.

| Arm | Change | Hypothesis |
|---|---|---|
| `base` | — | reference; identical to `configs/.../SAOI_run/config_train_<half>.txt` but for output paths |
| `seed` | `training_seed` 20260916 → 20270317 | **the noise floor.** Nothing else differs, so any gap is scatter by construction |
| `zdim8` | `vae_latent_dim` 16 → 8 | z over-capacity: unused latent dimensions are exactly where the prior can waste variance |
| `zdim4` | `vae_latent_dim` 16 → 4 | same axis, far enough out to tell a trend from a lucky point |
| `pmin15` | `posterior_min_std` 0.05 → 0.15 | the decoder is only trained on z near a posterior mean, so a distant prior draw is off-distribution and gets squashed |
| `pmin30` | `posterior_min_std` 0.05 → 0.30 | same axis, 6× the baseline floor |
| `g2e` | `prior_grad_to_encoder` 0.0 → 1.0 | closed, p is fitted to a fixed q and nothing pulls q toward what p can represent |
| `mmd10` | `lambda_mmd` 1 → 10 | push the aggregate posterior toward N(0,I) so the flow has a better-conditioned target |

`zdim8`, `zdim4`, `pmin15` and `pmin30` are the two axes the PVP decomposition
points at directly (capacity, posterior floor) — mechanically simple,
decoder-side hypotheses, plus the `base`/`seed` pair every arm is judged
against. `g2e` and `mmd10` reweight the objective instead: each moves more
than one mechanism at once, and `g2e` carries a known collapse risk.

A ninth arm, `arecon` (`alpha_recon` 1000 → 100), is **benched, not deleted**:
recon outweighs every distributional term by 10³, so it is a direction finder
confounded on purpose rather than a clean hypothesis, and unlike the eight
above it has no `top` counterpart. Its `bot` config still runs standalone —
`ARMS=arecon bash run_sweep.sh` — it just is not part of either half's default
roster.

Two more arms are deliberately **absent** (no config at all, on either half).
`prior_temperature` is not a width knob — it scales the *start* of the ODE,
leaving the trajectories the velocity field was trained on, which is why
`latent_inflation` exists. `prior_min_std` clamps the *mixture* prior and does
nothing under `prior_family fm`.

### `seed` is not one of eight results

It is the measurement that makes the other seven readable. The retired
ablation had no seed replicate, which is why its sub-2% `W1/sd` result could
never be interpreted. `analyze_sweep.py` refuses to call anything an effect
unless it clears the base-vs-seed gap **and** moves the same direction on all
three eval sets. If you drop `seed` to save a card, the sweep produces an
ordering and no conclusion.

## Running both halves

`bot` and `top` are two physically distinct halves of the SAOI part, each
with its own training dataset and its own three eval sets. They share
nothing — not a checkpoint, not a report — so `HALF` just points the whole
script (train, infer, PVP, report) at one half's configs and one half's
output root:

```bash
# bot half (default) -- writes output/meshgraphnets-v/saoi_sweep/
bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run_bot.out 2>&1

# top half -- writes output/meshgraphnets-v/saoi_sweep_top/
HALF=top bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep_top/run_top.out 2>&1
```

Run them one after the other, not concurrently: both default to the same
eight cards (`$GPUS`), and `Batch_size 16` being per-rank means every arm
needs a card to itself, so overlapping the two invocations would silently put
two arms on some cards. To actually run them at once, split `$GPUS` between
the two instead, e.g. `GPUS="0 1 2 3"` for bot and `GPUS="4 5 6 7"` for top —
each then takes two waves instead of one (see "Cost and scheduling" below).

`ARMS` and `REPORT_ARMS` still take an explicit override under either half —
setting one only replaces that half's default for that variable, e.g. to
re-run a single arm on `top` without touching any of `bot`'s checkpoints.

## What it produces

Per half, under that half's own output root:

```text
output/meshgraphnets-v/saoi_sweep/            HALF=bot (default)
output/meshgraphnets-v/saoi_sweep_top/        HALF=top
    <arm>.pth  <arm>.log                       8 checkpoints + training logs
    infer/<arm>/<eval set>/spread_values.npz   the whole inflation curve
    diag/posterior_vs_prior_<arm>_<tag>.json   the four-ensemble decomposition
    run_logs/report.txt                        this half's report (8 arms)
```

That is 24 inference runs and 24 PVP dumps per half (8 arms × 3 eval sets),
48 of each across both halves, and 16 checkpoints total.

**The inflation curve is free.** `latent_inflation` is a list, and `rollout.py`
cycles it batch by batch while tagging every draw with the λ that produced it
(`gen_lam`). One inference pass therefore yields λ = 1.0 … 3.0 in one go. Do not
split it into one run per λ.

**The histogram PNG is not the deliverable here.** For a multi-λ run it pools
every λ into one generated histogram. `analyze_sweep.py` reads the `.npz` and
never looks at the PNG.

## Reading the report

`analyze_sweep.py` prints five tables. They answer three questions and the order
matters:

1. **Is anything bigger than noise?** — table 2 sets the floor from `base` vs
   `seed`; table 4 tests every arm against it.
2. **Is the defect in the prior or the decoder?** — table 3, the PVP
   decomposition. If decoding a posterior *sample* already loses the width, no
   prior-side arm can put it back and most of this sweep was aimed at the wrong
   stage. **Read table 3 before table 4 means anything.** The `READING` block at
   the end states which case the baseline landed in.
3. **How far can post-hoc rescaling go?** — table 5. The `eff` column is
   (sd_ratio gain)/(λ gain); prior art on this model is ~0.54. An arm that
   *raises* `eff` has improved the alignment between the prior and the
   directions the decoder reads — which is the mechanism worth keeping, even if
   its λ=1.0 ranking is unremarkable.

Run it standalone at any time, including on a partial sweep — it reports its own
coverage and marks an arm missing a part as `incomplete`, not as a weak effect:

```bash
python configs/MeshGraphNets_Variational/hyperparameter_sweep/analyze_sweep.py \
    --json output/meshgraphnets-v/saoi_sweep/report.json --figure
```

## Cost and scheduling

`Batch_size 16` is **per rank**, so each arm gets exactly one card — two-card
DDP would make the global batch 32 and break comparability with `base` and with
SAOI_run. Arms are dealt round-robin onto `$GPUS` and each lane runs its arms in
series; `CUDA_VISIBLE_DEVICES` does the assignment, so every config asks for
local device 0 and nothing in the configs knows about the machine.

`GPUS` defaults to `0 1 2 3 4 5 6 7` — eight lanes, and eight arms divide
evenly across them: round-robin dealing puts exactly one arm per lane, so
each half's training is **one wave of 1000-epoch training**, no card idle.
Running both halves sequentially (see "Running both halves" above) is two
such waves back to back, not four — there is no second wave to avoid within
a half. The schedule cannot be shortened further within one wave:
`prior_freeze_epoch 700` means the last 300 epochs are the only ones that fit
the prior, and the whole point of `base` is that it matches the recipe the
evidence was collected under. If the box actually has fewer than eight cards,
pass `GPUS` explicitly (see below) — nothing in the configs assumes a GPU
count, only the default.

Inference and PVP are cheap by comparison and reuse the same lanes.

## Operational controls

```bash
# checkpoints already exist: redo inference, diagnosis and the report
TRAIN=0 bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh

# just re-read what is on disk
TRAIN=0 INFER=0 PVP=0 bash .../run_sweep.sh

# a subset, on two cards
ARMS="base seed pmin30" GPUS="0 1" bash .../run_sweep.sh

# bot now, top later (see "Running both halves" above)
bash .../run_sweep.sh
HALF=top bash .../run_sweep.sh

# top's report only, re-read from disk, without g2e/mmd10 muddying it
HALF=top TRAIN=0 INFER=0 PVP=0 REPORT_ARMS="base seed zdim8" bash .../run_sweep.sh

# arecon, benched but not deleted -- bot half only, it has no top counterpart
ARMS=arecon REPORT_ARMS="base seed arecon" bash .../run_sweep.sh
```

Variables: `ARMS`, `REPORT_ARMS`, `HALF`, `INFER_TAGS`, `GPUS`, `PYTHON`,
`METHOD_PYTHON`, and the stage switches `EVAL_PREFLIGHT`, `PREFLIGHT`,
`STRICT_PREFLIGHT`, `TRAIN`, `INFER`, `PVP`, `REPORT`. `HALF` (`bot`, `top`,
or unset, defaulting to `bot`) picks the dataset and the output root; `ARMS`
(what this invocation trains) and `REPORT_ARMS` (what the report covers) both
default to the same eight-arm roster on either half, and each takes its own
explicit override independent of `HALF`.

`METHOD_PYTHON` is resolved from `ai_cae4all.local.toml` automatically, because
`posterior_vs_prior.py` imports the method package directly instead of going
through the launcher. Override it if that lookup fails.

After training, the script accepts only checkpoints newer than each arm's launch
marker, so a stale checkpoint cannot flow into inference after a failed run.

## Dataset paths are case-sensitive

`dataset_dir`, `infer_dataset` and `eval_dataset` are all in `PATH_KEYS`
(`general_modules/load_config.py`, mirrored in `cae_suite/config_parser.py`), so
the case written in the config is the case opened. A stale lowercase twin of
`dataset/SAOI/` exists on disk holding a different vintage; it opens without
error and yields wrong numbers. Keep `SAOI/`, `S26FE-MAIN`, `S26FE-SEC` and
`SM-L345U-MAIN` uppercase as written.

## One thing the sweep cannot fix

The objective has **no term that penalizes a too-narrow ensemble**. It is
`alpha_recon*recon + lambda_mmd*mmd + beta_aux*aux + prior_nll_weight*fm`;
`crps` appears only as the checkpoint-selection metric (`best_by crps`,
`_crps_from_samples` in `training_profiles/training_loop.py`) and carries no
gradient. There are no energy-score (`es_*`) keys anywhere in the MGN-V or
HI_MGNFlow trees.

Every arm here is therefore an *indirect* lever on width. If the report says no
arm cleared the floor, that is the result — the next step is a new term in the
objective, not more points on these axes.
