# cHI-MGNflow parametric study (B300 round 2)

Round 2 runs on two 8-GPU B300 nodes, one study per node. Each node runs
this method's runner and the MeshGraphNets-V runner side by side, so every
card holds one cHI job and one MGN-V job:

| node | study | data | cHI-MGNflow jobs | MGN-V jobs |
| --- | --- | --- | --- | --- |
| 1 | [`SAOI/`](SAOI/) | `dataset/SAOI/saoi_train_{bot,top}.h5` | 4 arms x 2 board halves | 4 arms x 2 board halves |
| 2 | [`probabilistic/`](probabilistic/) | `dataset/probabilistic/ex3_shell_buckling{,_infer}.h5` | 8 arms | 8 arms |

The MGN-V half of each study lives in
[`configs/MeshGraphNets_Variational/hyperparameter_sweep/`](../../MeshGraphNets_Variational/hyperparameter_sweep/),
with a byte-identical copy of every script here.

The results come back as **typed text only**. Nothing is copied off the
nodes. Each runner therefore ends with a block of short lines (see "Reading
the block") that holds its whole result and is short enough to retype.

| File | Purpose |
| --- | --- |
| `run_sweep.sh SAOI\|probabilistic` | one method on one node, end to end: checks, data, preflight, the 8 jobs, the summary |
| `brief.py SAOI\|probabilistic\|r1` | prints the type-back block at any time, during or after the run |
| `SAOI/saoi_heldout.py` | SAOI only: geometry labels, the held-out parts and their inference configs |
| `SAOI/*.txt`, `probabilistic/*.txt` | round 2 configs; every header names the arm's change and its question |
| `SAOI/r1/`, `probabilistic/r1/` | round 1, with its own runner and README |

## SAOI: does the model carry a part's scatter to a new geometry?

Round 1 ([`SAOI/r1/`](SAOI/r1/)) found that the `prior_global token` arm fits
parts it trained on. It misses unseen parts in both width and shape, even
though most of those parts lie inside the training range. Every round-1 arm
split **samples** 80/10/10. Validation therefore held more realizations of
the training parts, and `best_by crps` kept the epoch that reproduced those
parts best.

Round 2 asks why the unseen parts miss, and asks it of both methods:

| arm (cHI / MGN-V) | change | question |
| --- | --- | --- |
| `c_smp` / `v_smp` | none: round-1 `tok` / MGN-V `SAOI_run`, sample split | the reference |
| `c_grp` / `v_grp` | `split_group_attr geometry_id` on the labelled `_bygeom.h5` copy | does selecting the checkpoint on **unseen geometries** help on unseen parts? |
| `c_grp_rep` / `v_grp_rep` | `c_grp` repeated / `v_grp` + `training_seed 20261012` | the noise floor; `c_grp` leaves `training_seed` unset, so the cHI repeat draws a fresh seed per launch (printed at start) |
| `c_nopart` / `v_nopart` | `*_grp` + `cond_var 2` (the Part No. row is no longer read) | is the arbitrary part ID, fed in as a scalar, what lets the model memorize parts? |

A geometry is the node coordinates plus the mesh. Variants that differ only in
thickness or part number stay on one side of the split.

Each job is scored on two families of parts:

- **O**: the three official eval parts (`s26fe_main`, `s26fe_sec`,
  `sm_l345u_main`). They are unseen by every arm.
- **H**: up to 6 test-role parts per half, cut by `saoi_heldout.py`. They are
  SEEN by `*_smp`, which trained on about 80% of their realizations
  (`split_<half>.json` records the exact share). They are UNSEEN by `*_grp` and
  `*_nopart`, which trained on none. The parts are the same for every arm and
  both methods, so `H(smp) - H(grp)` is the generalization gap at fixed parts.

Reading the SAOI results:

| result | meaning |
| --- | --- |
| `c_grp` beats `c_smp` on O by more than the `c_grp` vs `c_grp_rep` gap | checkpoint selection on seen parts was part of the miss |
| H of `*_grp` close to H of `*_smp` | a new in-range geometry costs nothing; the official parts are hard for another reason |
| H of `*_grp` far worse than H of `*_smp` | the model does not transfer scatter to a new geometry, so more geometries are needed, or conditions that say what separates parts |
| `*_nopart` beats `*_grp` | the part-ID row is a memorization channel; drop it from the recipe |
| MGN-V vs cHI on the same arm | which method transfers better, under identical splits |

Every cHI arm differs from `configs/HI_MGNFlow/SAOI_run/` only by its change
and the shared layout: `gpu_ids 0`, outputs under
`output/b300_r2/saoi/<arm>/<half>/`, `write_preprocessing False`, and a
private hierarchy cache. It runs 600 compressor epochs plus 2000 prior
epochs. Inference draws 2000 samples per official part and 500 per held-out
part.

## probabilistic (ex3 buckling): what does each capacity change buy?

Round 1 ([`probabilistic/r1/`](probabilistic/r1/)) added one change to `tok`
per arm. Round 2 adds all of them at once (`c_full`), then removes one change
per arm:

| arm | relative to `c_tok` | question |
| --- | --- | --- |
| `c_tok` | none | does round 1's `tok` reproduce on this node? Ties the rounds together |
| `c_full` | `prior_blocks 8`, `latent_dim 256`, `voronoi_clusters 4096,64` + `latent_ch 16`, `ae_epochs 120` + `training_epochs 240` | is the round-1 gain additive? |
| `c_full_rep` | `c_full` with `training_seed 43` (c_full: 42) | the noise floor for every leave-one-out difference |
| `c_full_b4` | `c_full` with `prior_blocks 4` | what the deeper prior contributes |
| `c_full_w128` | `c_full` with `latent_dim 128` | what the wider network contributes |
| `c_full_g256` | `c_full` with `voronoi_clusters 4096,256` + `latent_ch 4` | what the coarser latent grid contributes (same latent size) |
| `c_full_g512` | `c_full` with `voronoi_clusters 4096,512` + `latent_ch 4` | the other direction: a finer grid, twice the latent size |
| `c_full_short` | `c_full` with 60 + 120 epochs | what the doubled epochs contribute |

Inference draws 64 samples per scene of `ex3_shell_buckling_infer.h5`. The
cHI jobs go heaviest first and the MGN-V jobs lightest first, so card k pairs
the k-th heaviest cHI job with the k-th lightest MGN-V job and the two
biggest jobs never share a card.

## Run

Prerequisites on each node:

- A checkout of `main` with `prior_global` and `split_group_attr`. The runner
  checks both.
- The data, named exactly as the configs spell it, because Linux opens only
  that case:
  - SAOI node: `dataset/SAOI/saoi_train_{bot,top}.h5` and
    `test_{S26FE-MAIN,S26FE-SEC,SM-L345U-MAIN}_{infer,compare}_{bot,top}.h5`.
  - ex3 node: `dataset/probabilistic/ex3_shell_buckling.h5` and
    `ex3_shell_buckling_infer.h5`.
- A cHI-MGNflow and a MeshGraphNets-V interpreter, set in
  `ai_cae4all.local.toml` or through `METHOD_PYTHON=` on each runner. Each
  needs torch built for Blackwell (sm_100, CUDA 12.8 or newer).

From the repo root:

```bash
# node 1 (SAOI)
nohup bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh SAOI > output/b300_r2_saoi_chi.out 2>&1 &
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh SAOI > output/b300_r2_saoi_mgnv.out 2>&1 &

# node 2 (shell buckling)
nohup bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh probabilistic > output/b300_r2_ex3_chi.out 2>&1 &
nohup bash configs/MeshGraphNets_Variational/hyperparameter_sweep/run_sweep.sh probabilistic > output/b300_r2_ex3_mgnv.out 2>&1 &
```

Start both runners of a node together. Each runner stops before any GPU time
is spent if any of the following fails, and prints the reason as a type-back
block (`R2 <saoi|ex3> <chi|mgnv> ABORT <reason>`):

1. **Environment.** The launcher Python imports `cae_suite`. This method's
   interpreter imports torch, torch_geometric and h5py, and runs a bf16 matmul
   and a scatter on a GPU.
2. **Data.**
   - SAOI node: `SAOI/saoi_heldout.py` writes
     `dataset/SAOI/saoi_train_<half>_bygeom.h5` (a second full copy of each
     training file), the held-out parts and this method's configs for them.
     The two runners take turns under a lock; the second reuses what the
     first wrote.
   - Both nodes: every dataset a config names exists, with the case as written.
3. **Preflight.** The launcher `--check` passes on every config, 8 at a time
   per runner. On the SAOI node that is up to 80 checks per method: 8 train
   configs, 24 official inference configs and up to 48 held-out ones. On the
   ex3 node it is 16.

The jobs then start. Each job trains and then runs its inference configs one
after another. A failed inference does not stop the next one. A slot that
finishes takes the next unclaimed job of its own method, so a node with fewer
GPUs (`GPUS="0 1 2 3"`) also works, only slower.

Overrides:

| override | effect |
| --- | --- |
| `GPUS="0 1 2 3"` | which cards to use (default: every visible card) |
| `JOBS_PER_GPU=2` | jobs per card; use 2 when this method runs alone on the node |
| `JOBS="c_grp.bot c_grp.top"` | run only these jobs, in this order (SAOI job = `<arm>.<half>`, ex3 job = `<arm>`) |
| `THREADS_PER_JOB=4` | the torch thread pool per job; BLAS is pinned to one thread. 16 jobs x (4 + 4 loader workers) = 128 cores |
| `TRAIN=0` | inference only, on existing checkpoints; the run hours stay those of the training run |
| `INFER=0`, `PREFLIGHT=0` | skip that stage |
| `PYTHON=`, `METHOD_PYTHON=` | the launcher and this method's interpreter |
| `PROBE_TIMEOUT=900` | seconds the environment probe may take |

There is no resume: a killed job restarts from epoch 0. To run one job again,
use `JOBS=c_grp.bot bash configs/HI_MGNFlow/hyperparameter_sweep/run_sweep.sh SAOI`.

## Reading the block

Each runner prints its block at the end, so a node gives two: one for cHI
(`R2 <node> chi ...`) and one for MGN-V (`R2 <node> mgnv ...`). `brief.py`
prints the same block at any time:

```bash
python configs/HI_MGNFlow/hyperparameter_sweep/brief.py SAOI            # or probabilistic
python configs/HI_MGNFlow/hyperparameter_sweep/brief.py r1              # round 1, both studies
```

Every number is multiplied by 100 and rounded, so `23` means 0.23.

```text
==== TYPE THIS BACK ====
R2 saoi chi 31h D8/8 F0 | bot G120 v12 t12 H6 | top G118 v11 t12 H6
c_smp 41 62 18 12 88 6 14
c_grp 37 70 15 33 74 11 15
...
==== END ====
```

The lines in the example are invented to show the format.

**Header.** `31h` is the runner's wall time and `D8/8` the finished jobs.
`F0` counts the jobs that failed. On SAOI, each half then lists its
geometries (`G`), validation geometries (`v`), test geometries (`t`) and the
held-out parts that were cut (`H`).

**SAOI line.** It reads `arm Ow Os Ob Hw Hs Hb h`, with O and H as defined above:

| column | meaning | target |
| --- | --- | --- |
| `w` | mean `w1_norm`, the distribution distance divided by the truth sd | 0 |
| `s` | geometric-mean `sd_ratio`, generated width over true width | 100 |
| `b` | mean `abs(dmean_norm)`, the mean offset divided by the truth sd | 0 |
| `h` | training hours of the slower half | |

**ex3 line.** It reads `arm w1 sd bias crps skill h`. Here `bias` is signed:
negative means the generated mean sits below the truth. `skill` is
`spread_skill`.

**Suffixes.**

- `F:<stage>.<half> <error>`: the first failure. `+N` means more failures.
- `run NN%`: still training, with that share of epochs done.
- `wait`: queued, or still building its hierarchy cache.
- `inf`: running inference.

`brief.py SAOI --long` prints every number per eval set. That output is for
reading on the node, not for typing.

## Output

```text
output/b300_r2/<saoi|ex3>/<arm>/[<half>/]model.pth, train.log, infer/<eval set>/
output/b300_r2/<saoi|ex3>/run_logs/chi/<job>.{train,infer_<tag>}.out   stdout per stage
output/b300_r2/<saoi|ex3>/run_logs/chi/status/                         what brief.py reads
output/b300_r2/<saoi|ex3>/run_logs/chi/sweep_<timestamp>.out           the runner's own log
output/b300_r2/saoi/heldout/{data,configs/chi,split_<half>.json}       the held-out parts
```

The MGN-V runner writes the same tree with `mgnv` in place of `chi`; the two
share only `heldout/data` and the `_bygeom.h5` copies.

## Time

**Unknown.** No SAOI timing log exists from any round. Round 1 ran two sweeps
per node without thread caps, so its hours are not comparable. Read the first
epoch lines of each `train.log`, or the `run NN%` suffix, and extrapolate from
those.
