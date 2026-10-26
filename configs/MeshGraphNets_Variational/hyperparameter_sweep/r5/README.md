# B300 round 5: ex3 shell buckling, 13 jobs as one queue on node 135's 7 cards

Round 5 replaces round 4's two `long` jobs, which are still running. Stop them
first (see [Starting](#starting)). Only node 135 is available, without its
GPU1, so the run has 7 cards (0, 2–7). The budget is about 72 h.

Every job trains on `dataset/probabilistic/ex3_shell_buckling.h5` with round
3's split (`split_seed 42`, grouped by `shape_idx`). It is then scored on the 8
held-out geometries × 64 FE results (`spread_metrics.json`, plus one
`histogram_compare` per geometry). After inference, each job runs
[../plots.py](../plots.py), which writes the W2 ranking (`infer/spread_w2.json`)
and draws the loss curves beside the epoch logs.

No seeds this round: every arm is seed 42. Without a second seed, W2
differences of about 5–10 points are noise.

## Starting

On node 135, from the repo root, after pulling this checkout:

```
touch output/b300_r4/stop_135
ps -eo pid,etime,cmd | grep "_long.txt" | grep -v grep
kill <each PID listed>
nvidia-smi
cd configs/MeshGraphNets_Variational/hyperparameter_sweep/r5
bash run_135.sh
```

The stop file keeps round 4's driver from starting anything else. Kill by PID
only, never with `pkill -f`. `nvidia-smi` should show the two cards free before
round 5 starts.

The checks run in the terminal. When they pass, the jobs detach, so the
terminal can close. To see results, run
`python configs/MeshGraphNets_Variational/hyperparameter_sweep/r5/brief.py`.
[run_node.sh](run_node.sh) lists the overrides (`GPUS`, `JOBS`, `TRAIN=0`, …).

**After a stop or crash**, name the unfinished jobs:

```
JOBS="hybrid_long local_long" bash run_135.sh
```

Every training config has `resume_training True`, so each continues from its
last saved state, written about every 15 minutes. A finished job named again
trains from scratch.

**To end the queue early**, run `touch output/b300_r5/stop_135` from the repo
root. Running jobs finish, and no new job starts.

## The 13 jobs

[jobs.txt](jobs.txt) lists them. Jobs 1–7 start at once, one per card. Each
card that finishes takes the next line.

| # | job | change from round 4 | question |
| --- | --- | --- | --- |
| 1 | `hybrid_long` | none (round 4's `hybrid_long`, 240 epochs) | is 120 epochs too short? |
| 2 | `local_long` | none (round 4's `local_long`, 240 epochs) | is 120 epochs too short? |
| 3 | `hybrid_mp28` | `mp_per_level 2, 3, 5, 3, 2 → 4, 6, 8, 6, 4` | does a deeper V-cycle decoder fit better? |
| 4 | `sar_wing_std_short` | epoch caps `3000 / 2000 → 600 / 1000` | the SAR row within 72 h |
| 5 | `hybrid_v128c8` | `voronoi_clusters 4096, 256 → 4096, 128`, `latent_ch 4 → 8` | fewer coarse nodes for the prior to coordinate |
| 6 | `hybrid_v8192` | `voronoi_clusters 4096, 256 → 8192, 256` | does the middle level matter? |
| 7 | `hybrid_prior_deep` | `prior_mp_layers 3 → 6`, `prior_fm_blocks 4 → 8` | is the joint FM prior too shallow? |
| 8 | `infl_hybrid` | inference only, round 4's `hybrid_mmd` | does inflating z widen the spread? |
| 9 | `infl_local` | inference only, round 4's `local_mmd` | same |
| 10 | `infl_global` | inference only, round 4's `global_mmd` | same |
| 11 | `ldgn_wing_std` | none (round 4's, not run there) | the LDGN row with z-scored targets |
| 12 | `hybrid_v512c2` | `voronoi_clusters 4096, 256 → 4096, 512`, `latent_ch 4 → 2` | more coarse nodes, fewer channels each |
| 13 | `hybrid_prior_wide` | `prior_hidden_dim 128 → 256` | is the joint FM prior too narrow? |

The MGN-V arms change round 4's `hybrid_mmd` (120 epochs, seed 42, batch 4 × 1,
`use_checkpointing`). Each change is in both the train and the infer config,
because inference rebuilds the model from the config.

**Rough wall clock (estimates, not measured).** Round 4, with 7 jobs on these
cards, took about 27–30 h for the 120-epoch MGN-V jobs. On that basis:

| card | jobs | estimate |
| --- | --- | --- |
| GPU 0 | `hybrid_long` | about 73 h (41% after 30 h in round 4); it can run past 72 h |
| GPU 2 | `local_long` | about 68 h (44% after 30 h) |
| GPU 3 | `hybrid_mp28` | about 40–50 h (the fine level is 96% of the compute; twice its blocks) |
| GPU 4 | `sar_wing_std_short` | VAE about 30 h (3 min per epoch in round 4) + SAR stage at most about 33 h |
| GPU 5–7 | a 120-epoch arm, then the rest of the queue | about 30 h + about 30 h |

The three inference-only jobs take a card for a short time each (not timed).
`ldgn_wing_std` took 26 h in round 4 as `ldgn_wing`.

## Changes, one by one

**`mp_per_level 4, 6, 8, 6, 4`.** This is 28 blocks, as in the deterministic
HI-MGN configs. `message_passing_num` is set to 28 to match. With
`use_multiscale` it is not read.

**`voronoi_clusters`.** The local part of the hybrid latent sits on the
coarsest clusters: 256 nodes × `latent_ch 4` = 1024 numbers. Changing the
coarsest level alone would change the latent size too. So `v128c8` and
`v512c2` keep about 1024 numbers and change only how they are spread.
`v8192` changes the middle level only. Each setting has its own hierarchy cache
(`output/b300_r5/mscache/mgnv_v<fine>_<coarse>`).

**The FM prior.** It is the flow-matching generator of z given the geometry.
At inference it starts from z0 ~ N(0, I) and integrates its velocity network
with Heun for 30 steps. `prior_deep` doubles its fine-mesh trunk and its
coarse velocity blocks. `prior_wide` doubles its width; the velocity network
follows `prior_hidden_dim` because `prior_velocity_hidden_dim` is unset. A
deeper local prior barely helped earlier (4 → 16 blocks), so neither arm is
expected to change much.

**`latent_inflation 1, 1.25, 1.5, 2`.** It moves each z away from the prior's
image of z0 = 0 by the factor λ. The infer configs set
`inference_batch_size 16` with `inference_samples 64`. The factor list cycles
per batch and restarts per geometry, so every held-out geometry gets 16 draws
at each λ. Picking λ on this held-out file is a diagnosis, not a tuned setting.
- `spread_metrics.json`, its crps and the per-geometry histograms pool every λ.
- `spread_w2.json` scores each λ on its own (`by_inflation`), and the headline
  is λ = 1.
- To draw one λ's histograms, take its draws from `infer/spread_values.npz`
  (`gen_lam`).

These jobs read round 4's checkpoints under `output/b300_r4/`. The runner
refuses to start if one is missing.

**SAR.** 600 VAE epochs is about 30 h. The VAE's patience is 250, so the
plateau schedule can lower the learning rate at most twice. The SAR stage's cap
of 1000 epochs is about 33 h or less (round 3's SAR stage ran at most 2 min per
epoch). SAR and LDGN have no wall-clock cap key.

## Layout

- MGN-V as round 4: `batch_size 4 × grad_accum_steps 1`,
  `use_checkpointing True`, `gpu_ids 0`; the runner sets
  `CUDA_VISIBLE_DEVICES`.
- Each job rebuilds a missing hierarchy cache and deletes it after training,
  unless another job still has it open.
- Outputs go to `output/b300_r5/<job>/`. The plotting stdout goes to
  `output/b300_r5/run_logs/135/<job>.plots.out`. A plotting failure does not
  fail the job.
- **Not measured yet:** memory and time for `hybrid_mp28`, `hybrid_v8192`,
  `hybrid_prior_deep` and `hybrid_prior_wide`. Watch the first epochs
  (`tail -f output/b300_r5/run_135.out`).

## The block to type back

```
R5 <node> <h>h D<done>/<jobs> F<failed>
floor p50 p5 p95
<job> w2 b s shape sdg crps h w2h [flags]
infl_hybrid@1.5 w2 b s shape sdg - -
```

Every number is ×100. The ranking column is `w2`: per held-out geometry, the
W2 distance between the draws and the 64 FE results over that geometry's FE
sd, then the root mean square over the 8 geometries. It splits exactly as
w2² = b² + s² + shape² (mean offset, width error, the rest). `sdg` is 100 when
the width is right and below 100 when too narrow. An inference-only job prints
one extra line per λ. [brief.py](brief.py) explains the flags.

**The noise level (`floor`, `w2h`).** The 64 FE results of each geometry are
themselves a sample. [../plots.py](../plots.py) cuts them at random into two
halves of 32, 200 times with a fixed seed, and scores one half against the
other: `floor` is the median of that W2 and its 5–95% range. It is the score a
perfect sampler would get at this sample size. `w2h` scores 32 of the job's
draws against the same halves, so compare `w2h` with `floor`, not `w2`. A
`w2h` inside the floor range means the job cannot be told from FE scatter.
The infl jobs have 16 draws per λ, too few for this, so their `w2h` is `-`.
The figure is `output/b300_r5/noise_floor.png`: one median split per geometry
(half A against half B) and the floor histogram with each job's `w2h`.

To add both to jobs that already finished (their `spread_w2.json` predates
them and is recomputed), run from the repo root:

```
PY=$(python -c "from pathlib import Path; from cae_suite.settings import LocalSettings; print(LocalSettings.load(Path('.')).resolve_python('meshgraphnets-v', 'meshgraphnets_variational'))")
"$PY" -B configs/MeshGraphNets_Variational/hyperparameter_sweep/plots.py configs/MeshGraphNets_Variational/hyperparameter_sweep/r5 --no-loss
python -B configs/MeshGraphNets_Variational/hyperparameter_sweep/r5/brief.py 135
```
