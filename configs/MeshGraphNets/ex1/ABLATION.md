# HI-MGN ex1 ablation — runbook

Everything here is driven by [`ablation.py`](ablation.py). All 22 configs in
this directory named `config_{train,infer}_abl_*.txt` are **generated** — edit
`config_train1.txt` (the historical baseline, no longer checked in) or the
`ARMS` table in `ablation.py`
and re-run `gen`, never the generated files. That is the whole point: an arm
that drifted from the baseline by a hand edit would silently invalidate the
comparison it exists to make.

Design background: [`../../../MeshGraphNets/EX1_HIERARCHY_ABLATION.md`](../../../docs/research/meshgraphnets/EX1_HIERARCHY_ABLATION.md)
and [`../../../MeshGraphNets/COARSENING_ABLATION_DESIGN.md`](../../../docs/research/meshgraphnets/COARSENING_ABLATION_DESIGN.md).

---

## The five axes

Baseline is `config_train1.txt` as it stood when the 22 arms were generated:
`voronoi_clusters 5000, 100`, `multiscale_levels 2`, `mp_per_level 4, 6, 8, 6, 4`,
`coarsening_type voronoi_seedmean`, learned prolongation, `std_noise 0.1`, 2000
epochs, `Batch_size 1`. Nothing else is touched by any arm.

**Note:** `config_train1.txt` was later changed to `std_noise 0.01` to match the
rest of the ex1 suite. The generated `config_{train,infer}_abl_*.txt` files were
deliberately *not* regenerated, so this ablation study stays frozen at
`std_noise 0.1` as its own internal baseline — the two files have intentionally
diverged. Re-run `ablation.py gen` only if you want the study to pick up 0.01
(and re-read the "Known limitations" note below, which was written about the
0.1 value).

| axis | arms |
| --- | --- |
| **voronoi** | 1 stage `100` / **2 stage `5000, 100`** / 3 stage `10000, 1000, 100` |
| **mp** | `5,6,6,6,5` / **`4,6,8,6,4`** / `7,5,4,5,7` (total fixed at 28) |
| **coarsen** | **`voronoi_seedmean`** / `voronoi_inherit` |
| **totalmp** | `2,3,4,3,2` (14) / **`4,6,8,6,4`** (28) / `6,8,10,8,6` (38) |
| **interp** | **`learned_interpolation True`** / `False` (broadcast) |

The baseline sits in all five axes, so the 13 table cells are only **9 distinct
configurations**. The run list is **11**: those 9 plus two more runs of the
baseline.

Those two are not waste. Nothing in the training path calls
`torch.manual_seed` — `split_seed` fixes only the data split — so weight init
and geometry augmentation are a fresh random draw every run. The three baseline
runs are this study's **only** measurement of run-to-run noise. Without them,
a ΔR² of a few 1e-3 between two arms cannot be distinguished from scatter, and
every conclusion is unfalsifiable.

---

## Cost, measured before training

`params` is an exact count from constructing each model. `GFLOP/fwd` is
analytic, over per-level `(N, E, E_up)` built by the repo's own
`build_multiscale_hierarchy` on 5 samples spanning ex1's 6× size range
(N = 14,605 … 88,582).

| arm | axis | params | GFLOP/fwd | vs base | GPU |
| --- | --- | ---: | ---: | ---: | ---: |
| `base` | baseline | 4,667,652 | 526.4 | 1.00× | 4 |
| `base_rep1` | baseline | 4,667,652 | 526.4 | 1.00× | 5 |
| `base_rep2` | baseline | 4,667,652 | 526.4 | 1.00× | 6 |
| `vc_1stage` | voronoi | 4,467,588 | 849.7 | **1.61×** | 0 |
| `vc_3stage` | voronoi | 4,867,716 | 468.4 | 0.89× | 5 |
| `mp_flat` | mp | 4,667,652 | 626.3 | 1.19× | 3 |
| `mp_fine` | mp | 4,667,652 | 815.2 | 1.55× | 1 |
| `ct_inherit` | coarsen | 4,667,652 | 526.4 | 1.00× | 7 |
| `tmp_14` | totalmp | 2,585,348 | 293.0 | 0.56× | 6 |
| `tmp_38` | totalmp | 6,155,012 | 748.7 | 1.42× | 2 |
| `interp_bcast` | interp | 4,402,180 | 472.2 | 0.90× | 4 |

Two things this table settles up front:

- **Fixing the block count at 28 does not fix compute.** The grid still spans
  0.56×–1.61×, because a level-0 GnBlock costs ~450× a coarsest-level one — so
  `vc_1stage` (16 of its 28 blocks on the full mesh) is the *most* expensive arm
  and `vc_3stage` is *cheaper* than baseline, the opposite of the intuition that
  deeper hierarchies cost more. Report GFLOP beside every R².
- **`ct_inherit` is the only completely free comparison** — identical params,
  identical FLOPs, identical hierarchy topology. `voronoi_seedmean` and
  `voronoi_inherit` call the same coarsener with the same arguments; the only
  difference is that `inherit` writes `coarse_seed_idx_{i}`, which switches the
  model from mean-pooling to gathering at the FPS seed.

GPU packing makespan is **1.90 baseline-forward units**, which is optimal: 11
runs on 8 GPUs forces exactly three doubled-up GPUs, and the cheapest possible
pairing of the six smallest arms is 1.00+0.90 / 1.00+0.89 / 1.00+0.56. Use
`--gpus 16` to spread across both nodes; makespan then drops to 1.61.

---

## Running it

[`run_ablation.sh`](run_ablation.sh) is the entry point. It runs the configs
that are already in this directory and **never regenerates them**, so a re-run
cannot silently change what is being compared mid-study.

```bash
./configs/MeshGraphNets/ex1/run_ablation.sh              # cost -> train -> infer -> eval -> report
./configs/MeshGraphNets/ex1/run_ablation.sh train        # a single stage
./configs/MeshGraphNets/ex1/run_ablation.sh infer eval report
DRY=1 ./configs/MeshGraphNets/ex1/run_ablation.sh        # print the plan, launch nothing
PY=/path/to/venv/bin/python ./configs/MeshGraphNets/ex1/run_ablation.sh
```

| stage | what it does | needs GPU |
| --- | --- | --- |
| `cost` | builds each model for an exact param count, builds hierarchies for FLOPs → `cost.json` | no |
| `train` | launches all 11 training runs at once, one process per arm | yes |
| `infer` | rolls out each checkpoint on `ex1_infer.h5` | yes |
| `eval` | R² / RMSE / peak error → `scores.json` | no |
| `report` | `report.md` + `report.csv` | no |

The runner discovers arms by globbing `config_train_abl_*.txt` and reads each
arm's GPU back out of its config's `gpu_ids` line, so the config is the single
source of truth and the two cannot disagree. All 11 launch simultaneously;
GPUs 4, 5 and 6 each host two concurrent processes.

A failed arm is logged and the run continues — one dead arm should not discard
the other ten. Later stages skip arms whose artifacts are missing and `report`
prints `n/a` for them.

Outputs land in `output/meshgraphnets/ex1/ablation/`: `runner.log`, `cost.json`,
`scores.json`, `report.md`, `report.csv`, per-arm `train_*.stdout` /
`infer_*.stdout`, and `infer_<arm>/` rollout HDF5s.

### Regenerating the configs

Only when the baseline or the `ARMS` table actually changes:

```bash
python configs/MeshGraphNets/ex1/ablation.py gen --gpus 8
```

This rewrites all 22 configs and re-bakes `gpu_ids`. Run `cost` first if the
arm list changed, so the GPU packing uses real FLOPs rather than a round-robin
placeholder. `ablation.py all` also exists but includes `gen`; prefer
`run_ablation.sh` for anything you intend to compare.

---

## Scoring

Every arm is scored on **`dataset/ex1_infer.h5`**. `config_train1.txt` and the
generated `config_train_abl_*.txt` arms used to ship `infer_dataset
hex_dataset.h5` (whose state rows are all zero and cannot serve as ground
truth) — inert in `mode train` but misleading, so all ex1 train configs now
point `infer_dataset` at `ex1_infer.h5` directly, same as the infer configs.

**R² is the mean of per-channel `1 − SS_res/SS_tot`** on the denormalized
field, computed by `ablation.py`. Two deliberate choices:

- It is **not** the `_pearson_r2` in `general_modules/mesh_utils_fast.py`. That
  one exists for figure titles and is scale- and bias-blind — a prediction at 2×
  the truth scores 1.0, which would rank arms by correlation rather than by
  accuracy.
- It is **not** R² pooled over the stacked `[4, N]` array. Pooling takes
  `SS_tot` about a single mean spanning displacements (mm) and stress (MPa), so
  the largest-variance channel silently becomes the entire metric. Per-channel
  R² and the pooled value are both kept in `scores.json` for inspection.

Peak |stress| relative error is reported alongside, because restriction
operators trade it against RMSE — `ct_inherit` is the designed case, and an arm
that loses R² while gaining peak retention is not simply worse.

---

## Known limitations

**`ex1_infer.h5` holds one sample** (N = 42,265). All 11 arms are ranked on a
single mesh, so ΔR² between arms carries the variance of one geometry on top of
the run-to-run noise the baseline replicates measure. The replicate spread is
the reference to judge against; treat gaps smaller than ~2σ of it as nothing.
If a tighter comparison is wanted later, scoring the 10-sample held-out split of
`ex1.h5` is the natural addition.

**`gen` and `cost` are verified by execution; `train` / `infer` / `eval` /
`report` are not.** They need GPUs and checkpoints that do not exist yet, so
they have only been syntax-checked. `eval` and `report` run after training is
already banked, so a fix there is cheap — that is why the pipeline is staged
this way.

**Preflight passes.** `--check` on the generated configs reports zero spec
warnings; on a machine with fewer than 8 GPUs it reports `ENV-CUDA-002` for the
arms assigned to high GPU indices, which disappears on the 8-GPU node.

**`std_noise 0.1`** is inherited from `config_train1.txt` and applies to every
arm equally, so the comparison is fair — but ex1 is T=1 with no rollout, and
input-noise injection exists for autoregressive stability, so it may be
suppressing the absolute R² of the whole study.
