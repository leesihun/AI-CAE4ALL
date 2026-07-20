# Plasticity seven-model comparison report

**Snapshot:** 2026-07-20 08:35 KST (UTC+09:00)  
**Status:** five-epoch preliminary Plasticity campaign running; FNO and
DeepONet have passed train/infer/evaluate, MeshGraphNets is training  
**Policy:** use the listed checked-in configs, preserve the authoritative
dataset, add concurrency only from strict resource evidence, and do not commit

## Benchmark contract

The suite-native Plasticity artifact is available and matches its recorded
provenance hash. It contains 987 cases, 20 time states, and a common
`101 x 31` mesh with 3,131 nodes. The checked-in seed-42 suite split gives 789
training, 98 validation, and 100 held-out test cases. Inference uses only the
paired `plasticity_seed42_test.h5` artifact and performs 19 autoregressive
updates from time state 0.

All seven training configs pass the current suite, native, filesystem, and
environment preflight:

| Model | Training config | Inference config | Status |
|---|---|---|---|
| MeshGraphNets | `configs/benchmarks/plasticity/config_train_meshgraphnets.txt` | `configs/benchmarks/plasticity/config_infer_meshgraphnets.txt` | training on GPU; isolated dataset copy hash-verified |
| HI-MeshGraphNets | `configs/benchmarks/plasticity/config_train_hi_meshgraphnets.txt` | `configs/benchmarks/plasticity/config_infer_hi_meshgraphnets.txt` | next in GPU queue; independent isolated dataset copy hash-verified |
| Point-DeepONet | `configs/benchmarks/plasticity/config_train_point_deeponet.txt` | `configs/benchmarks/plasticity/config_infer_point_deeponet.txt` | queued |
| DeepONet | `configs/benchmarks/plasticity/config_train_deeponet.txt` | `configs/benchmarks/plasticity/config_infer_deeponet.txt` | complete: checkpoint, 100 rollouts, strict evaluation passed |
| FNO | `configs/benchmarks/plasticity/config_train_fno.txt` | `configs/benchmarks/plasticity/config_infer_fno.txt` | complete: checkpoint, 100 rollouts, strict evaluation passed |
| GINO | `configs/benchmarks/plasticity/config_train_gino.txt` | `configs/benchmarks/plasticity/config_infer_gino.txt` | queued |
| Transolver-3 (direct associative/tiled path) | `configs/benchmarks/plasticity/config_train_transolver.txt` | `configs/benchmarks/plasticity/config_infer_transolver.txt` | queued; internal key remains `transolver` |

Inference preflight cannot pass until each corresponding `model.pth` exists;
that is expected and is not a config defect.

## Fair comparison metric

The comparison will use the same 100 held-out IDs and all predicted time states
1 through 19 for every model. It will score de-normalized physical displacement
`u_x`, `u_y`, and `u_z`, while excluding the static die-profile state.

The primary ranking metric is the arithmetic mean over cases of full-trajectory
relative L2:

```text
mean_case ||u_pred[:, 1:20] - u_true[:, 1:20]||_2
          / ||u_true[:, 1:20]||_2
```

The isolated evaluator additionally reports global relative L2, the mean of
per-case time-averaged per-timestep relative L2, final-timestep mean relative
L2, per-component RMSE/MAE, and recorded inference time. Normalized validation
losses are not used to rank models because each architecture's normalized
objective and output parameterization are not a shared physical error scale.

The evaluator pins the exact 100 test IDs and the held-out HDF5 SHA-256, checks
all 1,900 case/time rows, requires case metadata and seed-state identity,
enforces backend-specific mesh-edge/provenance contracts, and hashes every
rollout and result CSV. The comparison is explicitly incomplete and exits with
failure if any of the seven model results is absent or invalid. Focused
Plasticity evaluator/comparator/campaign regressions pass.

## Execution and safety

Every checked-in config targets the same 8 GiB GPU. The safe default remains
one GPU job at a time, but the campaign may run an exact certified GPU pair and
one strictly eligible DeepONet CPU lane. It rejects evidence unless the index,
identity, result path, SHA-256, source/selected-ID identity, resource limits,
and completion fields all reconstruct exactly.

The common budget is 14,991 transition pairs per epoch for 500 epochs, an
effective batch of four, 3,748 optimizer updates per epoch, 1,874,000 total
updates, and 7,495,500 pair exposures. All seven use AdamW (`lr=1e-4`, weight
decay `1e-4`), three warmup epochs, EMA `0.99`, gradient clipping at `3`, and
an epoch-level schedule. The last update contains three examples for every
profile. Physical layouts are MGN `4x1`, HI-MGN `4x1` with safe `2x2`/`1x4`
profiles, Point-DeepONet fixed `2x2`, DeepONet/FNO `4x1` with safe fallbacks,
and GINO/Transolver-3 `1x4`. Point-DeepONet's final `2+1` accumulation window
averages the two batch means equally, so the singleton receives 50% rather than
33.3% of that final update (1 of 3,748 updates per epoch). No sample is dropped,
and the hot training loop is left unchanged.

### Preliminary execution-budget status - 2026-07-20 07:41 KST

The checked-in training configs remain unchanged at the default 500 epochs.
That value is the suite's common fairness budget, not a claim that every
counterpart paper used 500 epochs. The source protocols use different datasets
and budget units: official MeshGraphNets uses 10,000,000 optimizer steps;
HI-MGN has no separate published training protocol; Point-DeepONet uses 40,000
iterations; the selected 2D fractional-Laplacian DeepONet validation uses 5,000
dataset epochs; Geo-FNO Plasticity uses 501 epochs; GINO CarCFD uses 100 epochs;
and Transolver v1 Plasticity uses 500 epochs. These values were checked against
the [MeshGraphNets runner](https://github.com/google-deepmind/deepmind-research/blob/master/meshgraphnets/run_model.py),
the local [Point-DeepONet source](../point_deeponet/source/point_deeponet_official/5.Point_DeepONet/main.py),
[2D DeepONet config](../../../configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt),
[Geo-FNO Plasticity source](https://github.com/neuraloperator/Geo-FNO/blob/main/plasticity/plasticity_3d.py),
[GINO CarCFD config](../gino_carcfd/source/neuraloperator_official/config/otno_carcfd_config.py),
and local [Transolver v1 source](../elasticity/source/transolver_official/PDE-Solving-StandardBenchmark/exp_plas.py).

The latest measured single-GPU rates project the full 500-epoch suite budget as
follows. An explicit five-epoch preliminary run was launched at 07:13 KST,
materialized only in runtime configs through `--epoch-budget 5`; it is exactly
1/100 of the configured epoch count. FNO completed training, all 100 held-out
rollouts, and strict evaluation. Its mean per-case full-trajectory relative L2
is `1.0462108584` (global full-trajectory relative L2 `0.9629133499`, final-time
mean `0.9536549842`). The first campaign attempt then exposed a comparator bug:
the evaluator correctly wrote the HDF5 float32 time coordinate, while the
comparator accepted only a float64 `t/19` representation. The comparator now
strictly accepts either representation; the FNO checkpoint and artifacts passed
full recovery validation. The cancellation-affected DeepONet and Transolver
partials were preserved under `failed_attempts/attempt_20260720T073442`, and a
recovery campaign started at 07:40 KST. FNO is skipped as validated complete;
DeepONet runs on the certified CPU lane while MeshGraphNets runs on the GPU.
HI-MGN, Point-DeepONet, GINO, and Transolver remain queued. The durable live
state is in `output/benchmarks/plasticity/campaign_status.json`, with two-hour
snapshots in `output/benchmarks/plasticity/campaign_monitor/`.

Live check at 2026-07-20 08:35 KST: the recovery campaign process and its
two-hour watcher are both alive with no campaign error. DeepONet completed the
certified 8-thread CPU lane in 2,711.34 seconds, produced the expected epoch-4
checkpoint, generated all 100 held-out 19-step rollouts, and passed strict
evaluation. Its validation objective improved on every epoch: `0.82733`,
`0.14493`, `0.12946`, `0.12005`, and `0.11758`. MeshGraphNets remains on the
GPU and has started epoch index 2 of 4; GPU utilization was 97--98% using about
3.9 GiB of 8 GiB. Its epoch 0 and 1 train objectives were `0.31612` and
`0.13008`, while validation improved from `0.19351` to `0.10459`. The
remaining GPU queue is HI-MeshGraphNets, Point-DeepONet,
GINO, then Transolver. The next automatic two-hour snapshot is due at
approximately 09:39 KST.

| Completed five-epoch result | Mean per-case full trajectory relative L2 | Global full trajectory relative L2 | Final-time mean relative L2 | Time-averaged per-timestep mean relative L2 |
|---|---:|---:|---:|---:|
| DeepONet | `0.6725580` | `0.7339620` | `0.5986507` | `0.8220935` |
| FNO | `1.0462109` | `0.9629133` | `0.9536550` | `1.0416079` |

| Model | Projected 500-epoch time (h) | Linear 5-epoch planning estimate (h) |
|---|---:|---:|
| MeshGraphNets | 399.02 | 3.9902 |
| HI-MeshGraphNets | 396.60 | 3.9660 |
| Point-DeepONet | 190.55 | 1.9055 |
| DeepONet | 185.63 | 1.8563 |
| FNO | 204.26 | 2.0426 |
| GINO | 444.20 | 4.4420 |
| Transolver-3 | 928.89 | 9.2889 |

The five-epoch values are linear planning projections, not completed runtimes.
This preliminary pass tests execution and short-horizon behavior; it is neither
paper-equivalent nor eligible to populate the final 500-epoch comparison. FNO
and DeepONet exact held-out results are available above; the other five remain
unavailable until their training, inference, and strict rollout-evaluation
phases finish.

Transolver uses the supported T3-equivalent `slice_space` attention path with
tiling (`chunk_size=1024`) while retaining direct temporal inference. The
local attention, block, full-model, gradient, and cache paths were executed
against official Transolver-3 commit
`ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37`; every parity check passed, with
relative discrepancies below `1e-12` (normally around `1e-14` or less). The
upstream DrivAer T3 epoch budgets are task-specific and are not copied into
Plasticity.

The campaign does not automatically retry an OOM with another profile into the
same output. MeshGraphNets and Transolver-3 do not produce their final
`model.pth` until epoch 499; any partial or ambiguous output is refused.

Both MeshGraphNets variants write train-fit normalization metadata into their
configured datasets. To preserve the authoritative provenance hash and avoid
cross-variant HDF5 writes, the baseline points to
`plasticity_meshgraphnets_runtime.h5` and HI-MGN points to
`plasticity_hi_meshgraphnets_runtime.h5`. Both copies were created and their
SHA-256 values were verified identical to the source before any metadata write.
This requires no shared model-runtime change.

The resource-gated campaign is executing through the explicit preliminary-run
waiver and five-epoch runtime budget; it does not claim paper equivalence. It
materializes runtime-only batch/device configs, owns concurrent child trees
independently, writes atomic `campaign_status.json` state and per-attempt logs,
and follows `train -> checkpoint gate -> infer -> evaluate` for every model.
Baseline MGN, HI-MGN, and Transolver-3 have separate writable HDF5 files; the
Transolver copy is created atomically and accepted only after
normalization-only drift audit. Final seven-model comparison remains forbidden
until all seven strict results exist.

## Results

Ranks below are provisional among completed five-epoch models only. A value is
entered only after the model finishes training and all 100 held-out rollouts
and its output passes the external metric audit.

| Rank | Model | Mean case trajectory relL2 | Global relL2 | Final-time mean relL2 | Status |
|---:|---|---:|---:|---:|---|
| - | MeshGraphNets | pending | pending | pending | training |
| - | HI-MeshGraphNets | pending | pending | pending | queued |
| - | Point-DeepONet | pending | pending | pending | queued |
| 1 | DeepONet | `0.6725580` | `0.7339620` | `0.5986507` | five-epoch strict result complete |
| 2 | FNO | `1.0462109` | `0.9629133` | `0.9536550` | five-epoch strict result complete |
| - | GINO | pending | pending | pending | queued |
| - | Transolver-3 (direct associative/tiled path) | pending | pending | pending | queued |
