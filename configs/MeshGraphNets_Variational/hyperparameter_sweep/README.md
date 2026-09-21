# MeshGraphNets-V SAOI — hyperparameter sweep

Nine arms on the `bot` half. Run the whole thing from the repository root:

```bash
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run.out 2>&1 &
```

Train → infer → posterior-vs-prior → one report.

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

## The nine arms

Every arm is exactly one config key away from `base`, which is the SAOI_run
`bot` recipe verbatim. The swept line in each file is tagged `# SWEPT` and
carries its own rationale, so a config read on its own explains itself.

| Phase | Arm | Change | Hypothesis |
|---|---|---|---|
| A | `base` | — | reference; identical to `configs/.../SAOI_run/config_train_bot.txt` but for output paths |
| A | `seed` | `training_seed` 20260916 → 20270317 | **the noise floor.** Nothing else differs, so any gap is scatter by construction |
| A | `zdim8` | `vae_latent_dim` 16 → 8 | z over-capacity: unused latent dimensions are exactly where the prior can waste variance |
| A | `zdim4` | `vae_latent_dim` 16 → 4 | same axis, far enough out to tell a trend from a lucky point |
| A | `pmin15` | `posterior_min_std` 0.05 → 0.15 | the decoder is only trained on z near a posterior mean, so a distant prior draw is off-distribution and gets squashed |
| A | `pmin30` | `posterior_min_std` 0.05 → 0.30 | same axis, 6× the baseline floor |
| B | `g2e` | `prior_grad_to_encoder` 0.0 → 1.0 | closed, p is fitted to a fixed q and nothing pulls q toward what p can represent |
| B | `mmd10` | `lambda_mmd` 1 → 10 | push the aggregate posterior toward N(0,I) so the flow has a better-conditioned target |
| B | `arecon` | `alpha_recon` 1000 → 100 | recon outweighs every distributional term by 10³; a direction finder, confounded on purpose |

Phase A is the two axes the PVP decomposition points at directly (capacity,
posterior floor) — mechanically simple, decoder-side hypotheses, plus the
`base`/`seed` pair both phases are judged against. Phase B is the two axes
that reweight the objective and the one that opens a new gradient path —
each moves more than one mechanism at once, and `g2e` carries a known
collapse risk. See "Running it in two phases" below.

Two arms are deliberately **absent**. `prior_temperature` is not a width knob —
it scales the *start* of the ODE, leaving the trajectories the velocity field
was trained on, which is why `latent_inflation` exists. `prior_min_std` clamps
the *mixture* prior and does nothing under `prior_family fm`.

### `seed` is not one of nine results

It is the measurement that makes the other eight readable. The retired ablation
had no seed replicate, which is why its sub-2% `W1/sd` result could never be
interpreted. `analyze_sweep.py` refuses to call anything an effect unless it
clears the base-vs-seed gap **and** moves the same direction on all three eval
sets. If you drop `seed` to save a card, the sweep produces an ordering and no
conclusion.

## Running it in two phases

Nine arms do not divide evenly across eight cards (see "Cost and scheduling"
below), and the two halves of the table above carry different risk. `PHASE`
splits the campaign into two invocations, each ending in its own report,
instead of forcing one nine-arm pass to run two waves before printing
anything:

```bash
# phase A: base, seed, zdim8, zdim4, pmin15, pmin30 -- six arms, one wave
PHASE=A bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run_a.out 2>&1

# read output/meshgraphnets-v/saoi_sweep/run_logs/report_a.txt, then:

# phase B: g2e, mmd10, arecon -- three arms, one wave, reusing phase A's checkpoints
PHASE=B bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh \
    > output/meshgraphnets-v/saoi_sweep/run_b.out 2>&1
```

Both phases fit in a single wave on eight cards (six arms, then three), so the
two-phase run is not slower than the unphased one — it is the same nine
arm-slots of training, just split where the report can land in the middle.

Phase B trains only its own three arms, but its report needs `base` and
`seed` too, for the same noise floor described above — so phase B reports on
`base seed g2e mmd10 arecon`, and the two floor checkpoints are read from
phase A's `output/.../base.pth` and `seed.pth` rather than retrained. If those
files are missing, phase B's TRAIN stage prints `WARNING: .../base.pth not
found -- run PHASE=A first` and continues anyway; the report that follows will
simply have no floor to compare against (see "`seed` is not one of nine
results" above), so that warning is the only signal that something is wrong.

Each phase writes its own tagged report, `report_a.txt` and `report_b.txt`, so
neither overwrites the other. Leaving `PHASE` unset trains and reports on all
nine arms in one pass and writes the untagged `report.txt`, exactly as before
`PHASE` existed. `ARMS` and `REPORT_ARMS` still take an explicit override under
either phase — setting one only replaces that phase's default for that
variable, e.g. to re-run a single arm inside phase A without the other five.

## What it produces

```text
output/meshgraphnets-v/saoi_sweep/
    <arm>.pth  <arm>.log                       9 checkpoints + training logs
    infer/<arm>/<eval set>/spread_values.npz   the whole inflation curve
    diag/posterior_vs_prior_<arm>_<tag>.json   the four-ensemble decomposition
    run_logs/report.txt                        the report (all nine arms, PHASE unset)
    run_logs/report_a.txt                      PHASE=A's report (six arms)
    run_logs/report_b.txt                      PHASE=B's report (three arms + base/seed)
```

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

`GPUS` defaults to `0 1 2 3 4 5 6 7` — eight lanes. Nine arms do not divide
evenly across eight, so round-robin dealing puts two arms on lane 0 and one
arm on each of the other seven lanes: the campaign's wall-clock is **two waves
of 1000-epoch training**, with seven of the eight cards idle during the
second one. Splitting the run with `PHASE` ("Running it in two phases" above)
avoids that entirely — six arms and three arms both fit in a single wave, so
only the unphased nine-arm pass ever pays for a second wave. The schedule
cannot be shortened further within one phase: `prior_freeze_epoch 700`
means the last 300 epochs are the only ones that fit the prior, and the whole
point of `base` is that it matches the recipe the evidence was collected
under. If the box actually has fewer than eight cards, pass `GPUS` explicitly
(see below) — nothing in the configs assumes a GPU count, only the default.

Inference and PVP are cheap by comparison and reuse the same lanes.

## Operational controls

```bash
# checkpoints already exist: redo inference, diagnosis and the report
TRAIN=0 bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh

# just re-read what is on disk
TRAIN=0 INFER=0 PVP=0 bash .../run_sweep.sh

# a subset, on two cards
ARMS="base seed pmin30" GPUS="0 1" bash .../run_sweep.sh

# phase A now, phase B later (see "Running it in two phases" above)
PHASE=A bash .../run_sweep.sh
PHASE=B bash .../run_sweep.sh

# phase B's report only, re-read from disk, without g2e/mmd10 muddying it
PHASE=B TRAIN=0 INFER=0 PVP=0 REPORT_ARMS="base seed arecon" bash .../run_sweep.sh
```

Variables: `ARMS`, `REPORT_ARMS`, `PHASE`, `INFER_TAGS`, `GPUS`, `PYTHON`,
`METHOD_PYTHON`, and the stage switches `EVAL_PREFLIGHT`, `PREFLIGHT`,
`STRICT_PREFLIGHT`, `TRAIN`, `INFER`, `PVP`, `REPORT`. `PHASE` (`A`, `B`, or
unset) sets defaults for both `ARMS` (what this invocation trains) and
`REPORT_ARMS` (what the report covers); setting either explicitly overrides
just that one.

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
