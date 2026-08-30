# cHI-MGNflow SAOI Wave B — sweep

A 2^(5-1) resolution-V half fraction: **five factors in sixteen arms**, two arms
per GPU across GPUs 0–7. Trains on `saoi_train_bot.h5`, infers every arm against
the three held-out `*_bot` eval sets, and scores the grid into one report.

Deliberately the same shape as `configs/MeshGraphNets-V/SAOI_sweep3/` — same
data, same eval sets, same report format — so the two methods can be read side
by side. Everything except the three scripts and this file is **generated**:

```bash
python configs/cHI-MGNflow/SAOI_sweepB/gen_sweep_configs.py
```

## Design

Defining relation **`I = ABCDE`**: `learningr` is `A xor B xor C xor D`, not
free. All 5 main effects and all 10 two-factor interactions are clean.

| | factor | level 0 | level 1 |
| --- | --- | --- | --- |
| A | `batch_size` | `b16` 16 | `b32` 32 |
| B | `flow_t_sampling` | `tu` uniform | `tl` logit-normal |
| C | `voronoi_clusters` | `c1k` 1000,100 | `c2k` 2000,250 |
| D | capacity | `k0` 128 / `4,6,8,6,4` | `k1` 192 / `6,8,12,8,6` |
| E | `learningr` | `lr1` 1e-4 | `lr3` 3e-4 |

`flow_steps` and `flow_solver` are **not** here on purpose: they are
sampling-time choices, so the same checkpoint integrates at any K and sweeping
them over training runs would burn the budget on what inference answers for
free. They belong to Wave A (`cHI-MGNflow/docs/SWEEP_PLAN.md`).

**`voronoi_clusters` is the only swept key that enters the coarsening cache
signature**, so this grid builds **two** caches, not one. `run_sweep.sh` warms
one arm per level and `cache_ready()` counts files rather than testing
existence — a single-file check would release all 16 arms as soon as the first
cache appeared, leaving 15 of them to race on the second build.

## Before you launch: Wave 0

`training_epochs` is a **placeholder (6000)**. Flow matching needs more steps
than deterministic regression because the target `y − z0` carries irreducible
noise of size `Var(y|g)`, and that multiple has never been measured on this
data. Launch at a guessed budget and the ranking becomes a function of
convergence speed rather than of the factors.

`docs/SWEEP_PLAN.md` Wave 0 is two arms (~1 day): a deterministic HI-MGN and a
flow arm on the same backbone/data/budget, compared on where their loss curves
flatten. Set `FIXED_TRAIN['training_epochs']` from that and regenerate.

## Cost — read this before starting the inference stage

Unlike the variational method, **one draw is not one forward**: a draw
integrates the ODE, so it costs `flow_steps x 2` forwards under Heun.

| | per draw | per scene at 2000 draws |
| --- | --- | --- |
| MeshGraphNets-V | 1 forward | 2,000 |
| cHI-MGNflow, `flow_steps 12` | 24 forwards | 48,000 |
| cHI-MGNflow, `flow_steps 30` | 60 forwards | 120,000 |

The stage total is `16 arms x 3 eval sets x scenes x that`. `INFER_STEPS` is set
to **12**, not the production 30, precisely because of this: K is a sampling-time
choice, and the spread statistic (max − min over nodes) is far less sensitive to
integration error than a per-node field. If Wave A shows the spread histogram
still moving between K=12 and K=30, raise `INFER_STEPS` and re-run **only the
inference stage** — no retraining. The launcher's `FLOW-COST` warning prints the
exact figure for every config.

## Running it

```bash
rm -f dataset/saoi/saoi_train_bot.mscache.*.h5      # BOTH caches
nohup bash configs/cHI-MGNflow/SAOI_sweepB/run_sweep.sh > sweep.out 2>&1 &
```

preflight (64 configs) → warm both caches → train 16 → infer 48 → score.
Env knobs: `ARMS`, `PREFLIGHT`, `INFER`, `INFER_TAGS`, `SCORE`,
`WARM_TIMEOUT`, `CACHE_COUNT_REQUIRED`.

## What comes out

```
outputs/saoi_sweepB/sweep_results.md        the report
outputs/saoi_sweepB/warpage_<tag>.png       GT + all 16 arms on ONE axis, ranked by W1
output/chi-mgnflow/saoi_sweepB/<arm>.pth
output/chi-mgnflow/saoi_sweepB/infer/<arm>/<tag>/histogram_compare.png
output/chi-mgnflow/saoi_sweepB/infer/<arm>/<tag>/spread_values.npz
```

The report carries the per-arm table, **main effects** (8 vs 8), **two-factor
interactions**, and the **warpage spread table** — `max(z_disp) − min(z_disp)`
per realization against ground truth, normalized by the GT spread's own std so
the three eval sets are comparable. `W1/sd` is the ranking column; `sd ratio < 1`
is under-dispersion.

**There is no rank-histogram stage.** The variational tree's
`misc/eval_distribution.py` samples through a latent prior this method does not
have; porting it means reimplementing its draw loop against the ODE sampler.
Until then the report is training-log CRPS plus the spread comparison, and the
`eval` columns read `-`.

## Watch these in the first hour

1. **`VRAM peak=` on a `b32_*_k1` arm** — the worst corner is batch 32 ×
   capacity k1 × `c2k`. If the pair on a card does not fit, drop the batch
   levels to `('8', '16')` in the generator and regenerate.
2. **Both cache files appear** before the other 14 arms launch — the runner
   prints the count it is waiting for.
3. **CRPS actually moving** by the first few `val_interval`s. If it is flat, the
   epoch budget is the problem, not the factors — go do Wave 0.
