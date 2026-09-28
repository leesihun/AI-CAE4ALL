# MeshGraphNets-V SAOI — hyperparameter sweep

Eight arms, trained independently on both halves of the SAOI part (`bot` and
`top` — 16 jobs total, nothing shared between the halves). One invocation
runs all 16 from the repository root, queued over every GPU on the machine:

```bash
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/run_sweep.out 2>&1 &
```

Each job is train → infer → posterior-vs-prior for one arm on one half; a
report per half follows. Nothing starts until the environment, datasets and
configs have all been checked — see "Before any GPU time is spent".

## What this sweep is aimed at

The target defect is a generated ensemble that is **too narrow**: each eval
set is one held-out part × 125 realizations, so a correct model has
`sd_ratio` = 1, and the location (`|dmean|/sd`) is not the problem. No result
from outside this folder is used as evidence here; the sweep has to establish
its own baseline, its own noise floor and its own effects.

Width can be lost in exactly two places, and every arm is chosen to act on one
of them:

- **the decoder** — it is only ever trained on `z` drawn from a posterior, so
  a prior draw that lands where no posterior sample went is off-distribution
  and gets squashed, or lands in a latent direction the decoder never learned
  to read;
- **the prior** — the flow fits the aggregate posterior but nothing asks its
  per-condition spread to be right.

`misc/posterior_vs_prior.py` separates the two: it decodes the truth,
the posterior mean, a posterior *sample* and a prior draw for the same
condition. If the posterior-sample ensemble is already narrow, the loss is in
the decoder and no prior-side arm can recover it; if the posterior sample is
wide and the prior draw is narrow, the loss is in the prior.

## The eight arms

Every arm is exactly one config key away from `base`, which is the SAOI_run
recipe verbatim for its half (`bot` or `top`). The swept line in each file is
tagged `# SWEPT` and carries its own rationale, so a config read on its own
explains itself.

| Arm | Change | Stage | Hypothesis |
|---|---|---|---|
| `base` | — | — | reference; identical to `configs/.../SAOI_run/config_train_<half>.txt` but for output paths |
| `seed` | `training_seed` 20260916 → 20270317 | — | **the noise floor.** Nothing else differs, so any gap is scatter by construction |
| `zdim4` | `vae_latent_dim` 16 → 4 | decoder | fewer latent directions leave the prior fewer unread ones to put variance in |
| `pmin30` | `posterior_min_std` 0.05 → 0.30 | decoder | the decoder is trained on a 6× wider z cloud, so a prior draw is no longer off-distribution |
| `aux100` | `beta_aux` 10 → 100 | decoder | the aux term is the only one tied to the scored statistic: each posterior z must reproduce its own realization's peak-to-valley, which forces the decoder to read z in the direction the spread lives in |
| `arecon` | `alpha_recon` 1000 → 100 | both | recon outweighs every distributional term 10³:1. Under Adam with per-group clipping at `prior_grad_to_encoder 0` this is the same run as `beta_aux` ×10 **and** `lambda_mmd` ×10, so `arecon` − `aux100` is the MMD share |
| `fmmom` | `prior_fm_moments` False → True | prior | the only prior-side arm: the flow starts from N(μ(c), s(c)²) with a learned per-condition width and fits only the residual, so width is an explicit output. Zero-initialized, i.e. identical to `base` at step 0 |
| `g2e` | `prior_grad_to_encoder` 0.0 → 1.0 | coupling | closed, the prior is fitted to a fixed posterior and nothing pulls the posterior toward what the prior can represent. Carries a known collapse risk (the encoder can shrink q toward p) |

How to read them together:

- **posterior vs. prior says decoder-bound** → `zdim4`, `pmin30`, `aux100` are the arms that
  can move it; `fmmom` should do nothing, and that null is informative.
- **posterior vs. prior says prior-bound** → `fmmom` is the direct test; `g2e` is the
  indirect one.
- **`arecon` vs `aux100`** splits the one arm the user asked for into its
  two mechanisms without spending a card on `mmd10`.

Benched, not deleted — the configs still run standalone on either half, they
are just not in the default roster:

| Arm | Change | Why benched |
|---|---|---|
| `zdim8` | `vae_latent_dim` 16 → 8 | midpoint of the `zdim4` axis; run it only if `zdim4` clears the floor |
| `pmin15` | `posterior_min_std` 0.05 → 0.15 | midpoint of the `pmin30` axis; same rule |
| `mmd10` | `lambda_mmd` 1 → 10 | `arecon` − `aux100` already estimates the MMD share |

Deliberately **absent** (no config at all): inference-only knobs, because they
are free on any trained checkpoint and need no training arm —
`latent_inflation` is already a curve inside every inference pass,
`prior_temperature` only scales the start of the ODE, and `prior_min_std`
does nothing under `prior_family fm`. The legacy `z_conditioning concat` fuser
is not an arm either: it has a documented per-block gain defect, so a result
from it would measure the defect.

### `seed` is not one of eight results

It is the measurement that makes the other seven readable: without a seed
replicate a small difference between two arms cannot be told from training
scatter. `analyze_sweep.py` refuses to call anything an effect
unless it clears the base-vs-seed gap **and** moves the same direction on all
three eval sets. If you drop `seed` to save a card, the sweep produces an
ordering and no conclusion.

## Both halves, one invocation

`bot` and `top` are two physically distinct halves of the SAOI part, each
with its own training dataset and its own three eval sets. They share
nothing — not a checkpoint, not a report — so each gets its own output root,
and a single `bash run_sweep.sh` always runs both: 8 arms × 2 halves = **16
jobs** in one queue.

```text
output/meshgraphnets-v/saoi_sweep/            bot
output/meshgraphnets-v/saoi_sweep_top/        top
```

A job is the whole chain for one arm on one half — train → 3 inference runs
→ 3 posterior vs. prior dumps — and it runs on one card from start to finish. A failed job
stops only itself; the other jobs carry on, and the final summary lists every
job as `ok` or with the stage that failed. `ARMS` and `REPORT_ARMS` apply
identically to both halves.

## What it produces

Per half, under that half's own output root:

```text
output/meshgraphnets-v/saoi_sweep/            bot
output/meshgraphnets-v/saoi_sweep_top/        top
    <arm>.pth  <arm>.log                       8 checkpoints + training logs
    infer/<arm>/<eval set>/spread_values.npz   the whole inflation curve
    diag/posterior_vs_prior_<arm>_<tag>.json   the four-ensemble decomposition
    run_logs/report.txt                        this half's report (8 arms)
```

That is 24 inference runs and 24 posterior vs. prior dumps per half (8 arms × 3 eval sets),
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
2. **Is the defect in the prior or the decoder?** — table 3, the posterior vs. prior
   decomposition. If decoding a posterior *sample* already loses the width, no
   prior-side arm can put it back and most of this sweep was aimed at the wrong
   stage. **Read table 3 before table 4 means anything.** The `READING` block at
   the end states which case the baseline landed in.
3. **How far can post-hoc rescaling go?** — table 5. The `eff` column is
   (sd_ratio gain)/(λ gain), and `base` sets its reference value. An arm that
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

`Batch_size 16` is **per rank**, so each job gets exactly one card — two-card
DDP would make the global batch 32 and break comparability with `base` and with
SAOI_run. `CUDA_VISIBLE_DEVICES` does the assignment, so every config asks for
local device 0 and nothing in the configs knows about the machine.

`GPUS` defaults to **every GPU the MGN-V interpreter can see on the machine it
runs on**, and an index that does not exist there aborts the run before
anything starts. The 16 jobs sit in one queue; each card takes the next
unclaimed job when it finishes the last (claims are atomic `mkdir`s under
`saoi_sweep/run_logs/queue_<ts>/`), so 8 cards finish in 2 training waves and 4
cards in 4, with no re-dealing. The schedule cannot be shortened within a wave:
`prior_freeze_epoch 700` means the last 300 epochs are the only ones that fit
the prior, and `base` must match the recipe the evidence was collected under.

Do not start a second sweep on the same cards: every job assumes a card to
itself.

## Before any GPU time is spent

The script refuses to start training unless all four checks pass, and prints
the reason when one fails:

1. **environment**: the launcher python imports `cae_suite`; the MGN-V
   interpreter (`METHOD_PYTHON`, else `ai_cae4all.local.toml`) imports torch,
   torch_geometric and h5py and sees CUDA; every `GPUS` index exists.
2. **datasets**: every `dataset_dir` / `infer_dataset` / `eval_dataset` named
   by a selected config exists **with the case written in the config**.
3. **infer/compare pairs**: `SAOI_run/check_eval_inputs.py`.
4. **`--check`**: the launcher's full preflight on all 16 train configs.

Everything the script prints is also written to
`output/meshgraphnets-v/saoi_sweep/run_logs/sweep_<timestamp>.out`, so there is
a log even when it was started without a redirect.

## Operational controls

```bash
# both halves, default roster (the normal way to run this)
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/run_sweep.out 2>&1 &

# checkpoints already exist: redo inference, diagnosis and the report
TRAIN=0 bash .../run_sweep.sh

# just re-read what is on disk
TRAIN=0 INFER=0 POSTERIOR_VS_PRIOR=0 bash .../run_sweep.sh

# a subset, on two cards
ARMS="base seed pmin30" GPUS="0 1" bash .../run_sweep.sh

# report only, trimmed roster
TRAIN=0 INFER=0 POSTERIOR_VS_PRIOR=0 REPORT_ARMS="base seed fmmom" bash .../run_sweep.sh

# a benched arm (zdim8, pmin15, mmd10) -- runs on both halves like any other
ARMS=zdim8 REPORT_ARMS="base seed zdim4 zdim8" bash .../run_sweep.sh
```

Variables: `ARMS`, `REPORT_ARMS`, `INFER_TAGS`, `GPUS`, `PYTHON`,
`METHOD_PYTHON`, and the stage switches `EVAL_PREFLIGHT`, `PREFLIGHT`,
`TRAIN`, `INFER`, `POSTERIOR_VS_PRIOR`, `REPORT`.

`METHOD_PYTHON` is used for `posterior_vs_prior.py`, which imports the method
package directly instead of going through the launcher.

A job accepts only a checkpoint newer than its launch marker, and an inference
run only counts if it wrote `spread_values.npz`, so a stale checkpoint or a
silent skip cannot pass as a result.

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
