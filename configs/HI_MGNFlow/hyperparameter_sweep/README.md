# cHI-MGNflow SAOI parametric study (one 8-GPU B300 node)

Eight arms on the SAOI warpage data, one GPU per arm. Each arm trains both
board halves (`bot`, `top`) and scores each half on its three evaluation sets.
Every arm is the SAOI_run recipe (`configs/HI_MGNFlow/SAOI_run/`) plus the
shared sweep layout plus the change listed below. Each config's `%` header
names its change and its question; in a train config the changed lines are
marked `SWEPT`.

| arm | change on top of SAOI_run | question |
| --- | --- | --- |
| `ctrl` | none | Reference: the local-only prior, whose ensembles were ~2x too narrow and whose bias flipped sign across parts |
| `tok` | `prior_global token` | Does a global token give the prior the cross-node coherence it lacked on ex3, and with it the missing width? |
| `tok_seed` | tok, repeated unchanged | Run-to-run noise. HI_MGNFlow sets no seed, so each launch draws a fresh init and data order. An arm gap smaller than tok vs tok_seed is noise |
| `tok_recon` | tok + `best_by recon` | Does selecting on sampled CRPS keep mean-collapsed (too narrow) checkpoints? |
| `tok_kl` | tok + `ae_kl_weight 1e-3` | A smoother, better-scaled latent (1000x the KL). It is the largest value without the launcher's `FLOW-AEKL-LARGE` warning |
| `tok_c8` | tok + `latent_ch 8` | Is `latent_ch 4` a ceiling? Sweep 1 put the knee at 8 for both halves |
| `tok_g50` | tok + `voronoi_clusters 1000, 50` + `latent_ch 8` | Same latent size (50 x 8 = 100 x 4) on half as many coarse nodes: is the narrowness a coordination problem? |
| `tok_w256` | tok + `latent_dim 256` | Twice the trunk width in both stages: is either stage capacity-bound? |

Shared layout, which changes nothing about what is learned:

- `gpu_ids 0`: the runner assigns the card through `CUDA_VISIBLE_DEVICES`.
- Outputs go under `output/chi-mgnflow/saoi_b300_sweep/<arm>/<half>/`, one
  directory per half, so each half's periodic pictures stay with its own log.
- `write_preprocessing False`: 16 jobs read the same two training files, and
  none of them opens one for writing. Normalization travels in the
  checkpoint.
- A sweep-private hierarchy cache under `saoi_b300_sweep/mscache/`, one
  directory per coarsening (`v1000_100`, and `v1000_50` for `tok_g50`). It
  sits out of reach of SAOI_run's cache beside the dataset. Its lock lets one
  job build it while the others wait.
- `hierarchy_cache_keep False`, and the runner removes any leftover at the end.

Everything else (600 compressor epochs, 2000 prior epochs, b16, val every 25,
`best_by crps`, 30 Heun steps, 2000 draws per part) is SAOI_run's, unchanged.

## Files

| File | Purpose |
| --- | --- |
| `config_train_<arm>_<half>.txt` | 16 = 8 arms x 2 halves |
| `config_infer_<arm>_<half>_<eval set>.txt` | 48 = 16 x 3 eval sets (`s26fe_main`, `s26fe_sec`, `sm_l345u_main`) |
| `run_sweep.sh` | preflight, train, infer, report |

The eval-pair check is SAOI_run's `check_eval_inputs.py`, run against this
directory. There is no copy here.

This directory used to hold the earlier SAOI sweeps (sweeps 1-3). They are in
git history: `git show 5b61f5b:configs/HI_MGNFlow/hyperparameter_sweep/README.md`.

## Run

Prerequisites on the node:

- **The SAOI data under `dataset/SAOI/`, named exactly as the configs spell
  it.** Linux opens only that case:
  - `saoi_train_{bot,top}.h5`
  - `test_{S26FE-MAIN,S26FE-SEC,SM-L345U-MAIN}_{infer,compare}_{bot,top}.h5`
- **A cHI-MGNflow Python environment.** Point at it with
  `ai_cae4all.local.toml` or `METHOD_PYTHON=`. The interpreter needs torch
  built for Blackwell (sm_100, CUDA 12.8 or newer).
- **A checkout that has `prior_global`** (commit 5b61f5b or later).

```bash
nohup bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh \
    > output/chi-mgnflow/saoi_b300_sweep.out 2>&1 &
```

Before any GPU time is spent, the runner checks these things and stops on the
first one that fails:

1. The environment can import torch, torch_geometric and h5py, and can run a
   bf16 matmul and a scatter on the card.
2. Every GPU index in `GPUS` exists on the node.
3. The checkout has `prior_global`. Otherwise every `tok*` arm would quietly
   train the `ctrl` prior.
4. All 14 dataset files (2 training, 12 eval) exist with the case written in
   the configs.
5. Each `_infer_` file and its `_compare_` file describe the same part.
6. The launcher `--check` passes on all 64 configs.

Each card then takes the next unclaimed arm and runs that arm's two halves
side by side. Each half trains, then runs its three eval sets in series. With
fewer than 8 GPUs the remaining arms wait for a free card.

Overrides:

- `GPUS="0 1 2 3"`
- `ARMS="tok ctrl"`
- `HALVES=bot`
- `INFER_TAGS=s26fe_main`
- `PARALLEL_HALVES=0`: the two halves of an arm run one after the other.
- `KEEP_CACHE=1`: keep the hierarchy caches after the run.
- `PYTHON=`: the launcher interpreter.
- `METHOD_PYTHON=`: the cHI-MGNflow interpreter, instead of the one
  `ai_cae4all.local.toml` names.
- `PROBE_TIMEOUT=900`: how many seconds the environment probe may take. Its
  steps print as they run, so a stall shows the step it is stuck in.
- `THREADS_PER_JOB=4`: the CPU threads of each trainer's own torch pool. BLAS
  (numpy and torch matmuls) runs on one thread in every process. Without
  these caps, every process starts about one thread per core, and two sweeps
  on one node keep the CPU at 100% while the GPUs wait.
- `PREFLIGHT=0`, `EVAL_PREFLIGHT=0`, `TRAIN=0`, `INFER=0` or `REPORT=0` skips
  that stage.

To rebuild the report without running anything, use
`TRAIN=0 INFER=0 bash .../run_sweep.sh`. To re-score one arm on its existing
checkpoints, use `ARMS=tok TRAIN=0 bash .../run_sweep.sh`.

There is no resume, so a killed half restarts from epoch 0. The runner only
accepts a checkpoint newer than that half's launch marker, so an old
`model.pth` never flows into inference.

## Time

**Unknown.** There are no SAOI timing logs, neither from SAOI_run nor from the
old sweep. Each half is 600 compressor epochs plus 2000 prior epochs at b16,
with a sampled validation every 25 prior epochs. Inference is 2000 draws per
part, three parts per half.

Two halves share each card, and 16 jobs share the node's CPUs and disk, which
can stretch every arm. `tok_w256` is the slowest arm. Read the first epoch
lines in `<arm>/<half>/train.log` and extrapolate from them.

## Output

Everything lands under `output/chi-mgnflow/saoi_b300_sweep/`:

```text
<arm>/<half>/model.pth                       kept checkpoint (compressor + prior)
<arm>/<half>/train.log                       per-epoch AE/prior lines (+ periodic pictures)
<arm>/<half>/infer/<eval set>/
    spread_values.npz                        125 FEA truths + 2000 generated values
    histogram_compare.png                    FEA truth vs generated
    spread_metrics.json                      pooled scores (see below)
run_logs/<arm>_<half>.{train,infer_<eval set>}.out   launcher + trainer stdout
run_logs/checks/*.log                        preflight reports
run_logs/report.txt                          the tables below
run_logs/histograms.png                      arms x 6 cells, truth vs generated
run_logs/sweep_<timestamp>.out               the runner's own log
```

Each eval set is one part, so its `_infer_` file and its `_compare_` file
carry different sample IDs. `rollout.py` therefore pairs no scenes and
`spread_metrics.json` holds only the pooled scores (`sd_ratio`, `w1_norm`,
`dmean_norm`). The report computes the rest from `spread_values.npz`. Every
draw and every truth describe the same part, so the pooled ranks are the
calibration.

`report.txt` has two tables:

- **One row per arm x half x eval set**, with these columns:

  | column | meaning | target |
  | --- | --- | --- |
  | `sd_ratio` | sd(generated) / sd(truth) | 1 |
  | `W1` | 1-Wasserstein / sd(truth) | 0 |
  | `dmean` | mean bias / sd(truth) | 0 |
  | `CRPS` | of each truth against the full ensemble, / sd(truth) | low |
  | `tails` | share of truths in the outer 2% of the draws | 0.04 |
  | `pit_ks` | KS distance of the truth ranks from uniform | 0 |

- **One row per arm, ranked by mean W1 over the 6 cells.** It also shows the
  mean |log sd_ratio| (width error) and the mean |dmean| (bias). For each half
  it gives the kept prior epoch, with that epoch's validation CRPS and spread,
  and the training hours.

  The last line gives the tok vs tok_seed gap. That gap is the noise floor
  for every other comparison.
