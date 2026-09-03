# cHI-MGNflow SAOI Wave B — sweep

A 2^(4-1) resolution-IV half fraction: **four factors in eight arms**, one arm
per GPU across GPUs 0–7. Trains on `saoi_train_bot.h5`, infers every arm against
the three held-out `*_bot` eval sets, and scores the grid into one report.

Everything here except the three scripts and this file is **generated**:

```bash
python configs/HI_MGNFlow/SAOI_sweepB/gen_sweep_configs.py
```

## Layout

| Path | Authored? | What |
| --- | --- | --- |
| `gen_sweep_configs.py` | yes | Emits all 32 configs from the production ones in `../SAOI_all_input/` |
| `run_sweep.sh` | yes | preflight → cache warm → train → infer → score, one command |
| `score_sweep.py` | yes | Builds `sweep_results.md` + the warpage overlay figures |
| `config_train_<arm>.txt` | generated (8) | One training arm |
| `config_infer_<arm>_<tag>.txt` | generated (24) | One arm × one eval set |

The base is `../SAOI_all_input/config_train_bot.txt`; the inference configs
are derived from that folder's three `*_bot` infer configs. A non-swept key
that needs to change should change there, not here.

## The design

Defining relation **`I = ABCD`**: the fourth factor is `A xor B xor C`, not
free. All 4 main effects are estimable clean; the six 2-factor interactions
collapse into **three confounded pairs** — `AB=CD`, `AC=BD`, `AD=BC` — so a
large effect on one pair cannot be attributed to either half without a
follow-up run. With one run per cell there is no replication either way, so
3-factor terms were never trustworthy — this is the shape that keeps main
effects clean at the lowest run count.

| | factor | level 0 | level 1 |
| --- | --- | --- | --- |
| A | `batch_size` | `b16` 16 | `b32` 32 |
| B | `flow_t_sampling` | `tu` uniform | `tl` logit-normal |
| C | capacity | `k0` `latent_dim` 128 / `mp_per_level` 4,6,8,6,4 (28 blocks) | `k1` 192 / 6,8,12,8,6 (40 blocks) |
| D | `learningr` (generated) | `lr1` 1e-4 | `lr3` 3e-4 |

Arm names encode the cell: `<b16\|b32>_<tu\|tl>_<k0\|k1>_<lr1\|lr3>`.

**This replaced an earlier 5-factor / 16-arm design** (`voronoi_clusters` as a
fifth axis, two arms per GPU, 2000-epoch budget). At a measured 500 s/epoch
that was multiple GPU-weeks and risked OOM from card-sharing. `voronoi_clusters`
was dropped rather than any of the other four: it is the only key that would
enter the coarsening cache signature, so sweeping it means building and
warming **two** caches instead of one — real infrastructure cost for a
hypothesis unlikely to separate at a short, budget-limited run. It stays fixed
at the production value (`1000, 100`).

`flow_steps` and `flow_solver` are **not** swept, on purpose: they are
sampling-time choices, so the same checkpoint integrates at any `K` and
sweeping them over training runs would burn arms on what inference answers for
free. They belong to Wave A (`docs/research/hi_mgnflow/SWEEP_PLAN.md`).

## Before you launch: Wave 0

`training_epochs` is fixed at **1000**, set from a Wave 0 measurement (a
deterministic HI-MGN vs. a flow arm on the same backbone/data/budget, compared
on where their loss curves flatten) combined with a measured **500 s/epoch**
on this dataset and backbone. **1000 epochs is ~5.8 days per arm — a
BUDGET-LIMITED comparison, not a converged one.** Read the report that way:
"best at this budget," not an asymptotic ranking. If the budget changes,
update `FIXED_TRAIN['training_epochs']` in the generator and regenerate.

## GPU packing

One arm per GPU (`gpu_ids = arm index`) — no card sharing, so there is no
VRAM-exposure from co-residency and no complement-pairing logic is needed.

```
gpu 0  arm 1  (b16 tu k0 lr1)      gpu 4  arm 5  (b32 tu k0 lr3)
gpu 1  arm 2  (b16 tu k1 lr3)      gpu 5  arm 6  (b32 tu k1 lr1)
gpu 2  arm 3  (b16 tl k0 lr3)      gpu 6  arm 7  (b32 tl k0 lr1)
gpu 3  arm 4  (b16 tl k1 lr1)      gpu 7  arm 8  (b32 tl k1 lr3)
```

None of the swept keys enter the coarsening cache signature (`voronoi_clusters`
is fixed, not an axis here), so all 8 arms share **ONE** `*.mscache.*.h5`.

## Running it

```bash
rm -f dataset/saoi/saoi_train_bot.mscache.*.h5
nohup bash configs/HI_MGNFlow/SAOI_sweepB/run_sweep.sh > sweep.out 2>&1 &
tail -f sweep.out
```

`run_sweep.sh` preflights all 32 configs, launches ONE arm to build the shared
cache (aborting the batch if it dies, so a config error costs minutes not
days), then the other 7, then inference, then scoring.

| env | default | effect |
| --- | --- | --- |
| `ARMS` | all 8 | subset to run |
| `PREFLIGHT` | 1 | `--check` every arm before launching any |
| `TRAIN` | 1 | `0` skips training and goes straight to infer + score on checkpoints already on disk |
| `WARM_TIMEOUT` | 21600 | seconds to wait for the shared cache |
| `INFER` | 1 | run the per-arm inference stage |
| `INFER_TAGS` | all 3 | which eval sets to infer |
| `SCORE` | 1 | build the report when training ends |

## What comes out

```
output/chi-mgnflow/saoi_sweepB/sweep_results.md      the report — read/paste this
output/chi-mgnflow/saoi_sweepB/sweep_results.json    everything, incl. per-arm log detail
output/chi-mgnflow/saoi_sweepB/warpage_<tag>.png     GT + all 8 arms on ONE axis, ranked by W1
output/chi-mgnflow/saoi_sweepB/run_logs/<arm>.log    per-arm transcripts
output/chi-mgnflow/saoi_sweepB/<arm>.pth
output/chi-mgnflow/saoi_sweepB/infer/<arm>/<tag>/histogram_compare.png
output/chi-mgnflow/saoi_sweepB/infer/<arm>/<tag>/spread_values.npz
```

The report carries: the per-arm table (best CRPS, valid `fm` loss, ensemble
spread ratio, one-step deterministic MSE), **main effects** (4 vs 4 per
factor), **confounded two-factor effects** (3 pairs, each the sum of its
alias), and the **warpage spread table** — `max(z_disp) − min(z_disp)` per
realization, generated against ground truth, normalized by the GT spread's
own std so the three eval sets are comparable. `W1/sd` is the ranking column;
`sd ratio < 1` is the classic under-dispersion failure.

**There is no rank-histogram / verification-rank stage.** The variational
tree's `misc/eval_distribution.py` samples through a learned latent prior that
this method does not have; the warpage comparison is this sweep's calibration
signal instead.

The inference configs set `save_rollouts False`, so **no trajectory HDF5s are
written** — scene × draws would be thousands of files across the grid.
`num_vae_samples` is 2000 per scene against production's 5000; raise
`INFER_SAMPLES` in the generator if the histograms look ragged. Unlike the
variational method, **one draw is not one forward** here — a draw integrates
the ODE, so it costs `flow_steps x 2` forwards under Heun. `INFER_STEPS` is set
to 12, not the production 30, because `K` is purely a sampling-time choice and
the spread statistic is far less sensitive to integration error than a
per-node field; the launcher's `FLOW-COST` warning prints the exact forward
count for every config.

## Watch these in the first hour

1. **`VRAM peak=` on a `b32_*_k1` arm** — batch 32 at the wider/deeper capacity
   level is the worst corner, even with one arm per card.
2. **The first few epochs' log line**: `Train fm=... | Valid fm=... | CRPS ...`.
   If CRPS is flat from the first `val_interval` on, the epoch budget is the
   problem, not the factors — that is what Wave 0 exists to catch beforehand.
3. **`[FlowDiag] spread/gt=`** in the transcript — this is the same
   under-/over-dispersion signal as the warpage table's `sd ratio`, but
   measured on the train split during training rather than on a held-out set.

## Caveats

- 1000 epochs is a **budget-limited**, not converged, comparison — see "Before
  you launch" above.
- Resolution IV means a large confounded-pair effect needs a follow-up run to
  attribute to one member or the other.
- Only the three `*_bot` eval sets are inferred: the sweep trains on
  `saoi_train_bot.h5`, so no `*_top` checkpoint exists for these arms.
