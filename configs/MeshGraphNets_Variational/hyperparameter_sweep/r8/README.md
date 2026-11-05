# B300 round 8: HI-MGN-hybrid at batch 16. Epochs, MMD and KL weight, the flow-matching prior

Goal: round 7 finished only 2 of 14 jobs in 72 hours, and those two said the
same thing: the batch, not the step count, sets the cost. `e480_bs16` (alone on
a card) scored 68 and `e240_bs8` 74, against the base's 69 (r6 `hybrid_e240`,
batch 4), in less time than any batch-4 run. So round 8 trains at batch 16. Its
base (`hybrid_base`, job 2) is r6's `hybrid_e240` with only the batch changed,
and every other job changes one thing from it. One run per setting, seed 42, no replicates.

Everything here is generated. `make_sweep.py` writes `config_train_*`,
`config_infer_*`, `jobs.txt`, `run_135.sh` and the job table at the end of this
file. `--plan` prints the queue and writes nothing. Edit `make_sweep.py`, never
a generated file.

## Before running: round 7 and the packing

1. **Stop round 7 first** (or let it finish). Its jobs have `resume_training
   True`, so `kill_all.sh` loses at most 15 minutes of each, and
   `bash ../r7/run_135.sh` continues them later.
2. **Measure, then pick `--pack`** (jobs per card). On the node:

       python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/speed.py

   It prints seconds per epoch and VRAM peak for round 6 and 7's runs, grouped
   by batch size, checkpointing and jobs per card. Round 6 has batch 16 with
   checkpointing at 2 per card; round 7 has batch 16 alone without it. Type the
   block back. The generated default is `--pack 1`.

### Why not 3 or 4 jobs per card at batch 16

Memory decides it, not compute. The worst batch of 16 ex3 samples has about
1.04 M nodes, which is ~180 GB without gradient checkpointing (17.1 GB per
100 k nodes, measured). A 288 GB card holds one such job. Two or more fit only
with checkpointing, which costs about 1.52x per step (measured on a 3090).
Packing pays only when MPS sharing gains more than that:

| jobs per card | checkpointing | per-card throughput vs. one alone (if the guesses hold) |
| --- | --- | --- |
| 1 | off (~180 GB) | 1 |
| 2 | on (~55 GB each, a guess) | 2 / (1.5 x 1.52) = 0.88 |
| 3 | on | 3 / (2.1 x 1.52) = 0.94 |

The sharing factors (1.5 for two per card, 2.1 for three) are guesses. Batch 4
gave up to 1.54x from three per card on a 3090, because a batch-4 step is
launch-bound and leaves the GPU idle between kernels. A batch-16 step on 1 M
nodes runs large kernels and should leave much less to share. If `speed.py`
shows batch 16 alone near 100 % utilization, keep `--pack 1`. The direct test: `--pack 2`
wins when `speed.py`'s `b16 c1 n2` seconds per epoch are under twice its
`b16 c0 n1`. Then regenerate:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/make_sweep.py --pack 2 [--force]

`--pack` rewrites `run_135.sh` (GPUS, THREADS_PER_JOB) and each config's
`use_checkpointing` (on only where the estimated peak of PACK jobs would not
fit). Neither enters the resume fingerprint, so a running round can be stopped,
regenerated with another `--pack` and restarted; every job resumes. `--force`
is needed once `output/b300_r8` exists.

## Running it (node 135, cards 0 and 2-7)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/run_135.sh

This is the same runner as rounds 6 and 7. The checks (environment, data,
launcher `--check`) run in the terminal. When they pass, the jobs detach and
the slots take the lines of `jobs.txt` in order. At any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **CPU:** about 120 of the 128 CPUs, split evenly over the slots
  (`THREADS_PER_JOB` = 120 / slots - 4 DataLoader workers): 13 at `--pack 1`.
- Restart, overrides, `stop_135`, `kill_all.sh`, MPS, `use_compile` and the
  hierarchy cache work as in `../r6/README.md`.

## The design

| axis | jobs | what it asks |
| --- | --- | --- |
| base | `base` (240 epochs, batch 16, lr 1e-4) | does batch 16 at 240 epochs hold r6/r7's ~70? |
| epochs | `e360`, `e720` | with the base (240) and r7's `e480_bs16` (68, this base at 480 epochs): does w2 keep falling with training? |
| learning rate | `lr2e-4` | batch 16 has 1/4 of the base's steps: does a larger step make up for it (sqrt scaling from batch 4)? One value only |
| batch | `bs32_lr2e-4` | one size up (checkpointing always: ~330 GB without) |
| MMD weight | `mmd` 0.1, 10, 100, 1000, 10000 (base 1) | where does a stronger latent match start to help or hurt? |
| KL | `kl` 0.1, 1, 10, 100 | is the per-sample KL a better regularizer than the aggregate MMD? |
| flow-matching prior, size | `fmbig` (`prior_fm_blocks` 8, `prior_hidden_dim` 256) | is the prior's velocity network too small at batch 16 and 240 epochs? |
| flow-matching prior, sampler | `base_steps10`, `base_steps100` (sampling only) | `prior_fm_steps` 10 and 100 instead of 30, on the base's checkpoint: does the ODE solver add error? |

- **Why redo round 7's MMD and KL arms:** they ran at batch 4, and only the
  batch-8 one finished. MMD is estimated within a batch, so at batch 16 it sees
  four times as many latents per estimate. Its value at batch 16 is the one that
  matters from here on.
- **Queue order:** with 7 slots, the first 7 jobs make the first wave: `e720`
  first (about 4 days, so it does not set the makespan), then the base, the
  learning rate, the middle of the MMD and KL ranges. The two sampling-only
  jobs come next: they wait for the base (about 1.4 days) and take 2 hours each.
  Then `e360`, `fmbig`, `bs32` and the ends of the ranges.
- **The flow-matching prior, already tried:** at batch 4 and 120 epochs (r6),
  `prior_fm_blocks` 8 scored 82 and `prior_hidden_dim` 256 90 against the
  reference's 88/88/107, so both were within seed noise; `prior_temperature`
  0.5 and 2 and `latent_inflation` above 1 made every placement worse. So
  `fmbig` changes both at once at the new batch. `prior_nll_weight` is left
  out: only the prior reads the flow-matching loss, and under Adam scaling one
  network's only loss changes nothing (gradient clipping aside).
- **Not here, and why:** a longer prior-only tail at a fixed simulator cannot
  be set from the config. The joint phase's cosine spans `training_epochs`, so
  `training_epochs 400, prior_freeze_epoch 160` also trains a different
  simulator (its learning rate is still high at epoch 160), and the comparison
  with the base would mix the two.
- `prior_freeze_epoch` follows the epochs at 2/3 (160 at 240). KL configs drop
  `lambda_mmd` and `mmd_bandwidth` (the launcher warns that they are ignored).
- `make_sweep.py` checks the base key by key against
  `../r6/config_train_hybrid_e240.txt` and `config_infer_hybrid_e240.txt`. The
  only allowed difference is the training batch size (plus paths and build-only
  keys).
- **Time:** the plan assumes 0.13 h per batch-16 epoch alone (round 7's
  `e480_bs16` finished 480 epochs inside 72 h). That is a bound, not a
  measurement; `speed.py` replaces it.

## Reading the results

`brief.py` prints one line per finished job, `# w2`, best first. Read every job
against job 2 (`base`), and the base against r6 `hybrid_e240` 69, r7
`e240_bs8` 74 and `e480_bs16` 68. One run per setting: three seeds of one r6
config spread over 88-107 (sd about 11), so a difference of 10 or less is not an
effect on its own. Trust a trend along an axis over a single job. W2 scale,
floor (27) and caveats are as in `../r6/README.md`.

The sampling-only jobs share the base's checkpoint, so for them the seed noise
of training drops out: `base_steps10`/`base_steps100` against the base is a
direct reading of the sampler, and a few points of difference already count.

Every per-geometry histogram (`infer/histogram_compare_shape_idx_<k>.png`)
carries its own `w2_norm` (W2 over that geometry's FE sd, the number the
ranking squares and averages) and, in the title, the RMS over the 8 geometries
(the ranking's w2 itself; `brief.py` prints it x100).

## Jobs (generated by make_sweep.py; everything below is rewritten)

15 trainings + 2 sampling-only jobs, seed 42, one run each; --pack 1: 1 per card on GPUs 0, 2, 3, 4, 5, 6, 7 = 7 slots, 13 torch threads per job. Estimated makespan 4.2 days (GUESSED speeds: 0.13 h per batch-16 epoch alone, x1.52 with checkpointing, x1.0 per job at 1 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_e720 | 720 epochs (with 240 and r7's 480: does it keep improving?) | 0 | off | 180 | 96 | 0.0-4.0 |
| 2 | hybrid_base | the base: 240 epochs, batch 16, lr 1e-4, lambda_mmd 1 | 2 | off | 180 | 33 | 0.0-1.4 |
| 3 | hybrid_lr2e-4 | learning rate 2e-4 (sqrt scaling from batch 4) | 3 | off | 180 | 33 | 0.0-1.4 |
| 4 | hybrid_mmd10 | lambda_mmd 10 | 4 | off | 180 | 33 | 0.0-1.4 |
| 5 | hybrid_mmd100 | lambda_mmd 100 | 5 | off | 180 | 33 | 0.0-1.4 |
| 6 | hybrid_kl1 | latent_reg kl, kl_weight 1 | 6 | off | 180 | 33 | 0.0-1.4 |
| 7 | hybrid_kl10 | latent_reg kl, kl_weight 10 | 7 | off | 180 | 33 | 0.0-1.4 |
| 8 | hybrid_base_steps10 | sampling only: prior_fm_steps 10 on the base's checkpoint | 2 | - | - | 2 | 1.4-1.5 |
| 9 | hybrid_base_steps100 | sampling only: prior_fm_steps 100 on the base's checkpoint | 3 | - | - | 2 | 1.4-1.5 |
| 10 | hybrid_e360 | 360 epochs | 4 | off | 180 | 49 | 1.4-3.4 |
| 11 | hybrid_fmbig | larger flow-matching prior: prior_fm_blocks 8, prior_hidden_dim 256 | 5 | off | 220 | 33 | 1.4-2.8 |
| 12 | hybrid_bs32_lr2e-4 | batch 32 at learning rate 2e-4 (checkpointing always: ~330 GB without) | 6 | on | 99 | 42 | 1.4-3.1 |
| 13 | hybrid_mmd0.1 | lambda_mmd 0.1 | 7 | off | 180 | 33 | 1.4-2.8 |
| 14 | hybrid_mmd1000 | lambda_mmd 1000 | 2 | off | 180 | 33 | 1.5-2.9 |
| 15 | hybrid_mmd10000 | lambda_mmd 10000 | 3 | off | 180 | 33 | 1.5-2.9 |
| 16 | hybrid_kl0.1 | latent_reg kl, kl_weight 0.1 | 5 | off | 180 | 33 | 2.8-4.2 |
| 17 | hybrid_kl100 | latent_reg kl, kl_weight 100 | 7 | off | 180 | 33 | 2.8-4.2 |
