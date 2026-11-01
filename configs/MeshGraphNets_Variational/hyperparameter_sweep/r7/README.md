# B300 round 7: HI-MGN-hybrid only. Epochs x batch size, MMD and KL weight

Goal: round 6 left HI-MGN-hybrid (`latent_type hybrid`) as the placement to
take forward, and 240 epochs as its best setting (`hybrid_e240`, w2 69,
against 88, 88 and 107 for three seeds at 120 epochs). Round 7 takes that run
as the **base** and changes one thing per job: longer training (with larger
batches so it does not take twice as long), the MMD weight up to 10000, and
the per-sample KL in place of MMD. One run per setting, seed 42, no replicates.

Everything here is generated. `make_sweep.py` writes `config_train_*`,
`config_infer_*`, `jobs.txt` and the job table at the end of this file.
`--plan` prints the queue and its estimated hours and writes nothing. Edit
`make_sweep.py`, never a generated config.

## Running it (node 135, cards 0 and 2-7)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r7/run_135.sh

This is the same runner as round 6. The checks (environment, data, launcher
`--check`) run in the terminal. When they pass, the jobs detach and the 11
slots take the lines of `jobs.txt` in order. At any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r7/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **GPU allocation.** `run_135.sh` lists GPUs 0, 2 and 3 once and GPUs 4-7
  twice (`0 2 3 4 5 6 7 4 5 6 7`): 11 slots. The three 480-epoch jobs are the
  first lines of `jobs.txt`, and the runner deals job *k* to slot *k* in its
  first round (on a restart too, over the unfinished jobs), so they always get
  a card to themselves: no MPS sharing (about 1.4x faster per job) and enough
  memory to run without gradient checkpointing even at batch 16 (~180 GB of
  the 288 GB). The other eleven share cards 4-7 two at a time; whatever comes
  next takes the first free slot, alone or shared. `make_sweep.py` refuses to
  generate when `run_135.sh`'s GPUS disagrees with its `ALONE_GPUS` /
  `SHARED_GPUS`. Do not reorder with `JOBS=`, or an e480 job can land on a
  shared card.
- **CPU:** 11 jobs x (6 torch threads + 4 DataLoader workers) = 110 of the
  128 CPUs, hence `THREADS_PER_JOB 6`.
- **Round 6 must be finished or stopped first** (`kill_all.sh` stops every
  round's runner).
- **Code since round 6** (bc0221f, 2026-10-30, the NVIDIA MGN port) changes the
  numerics only by bf16 rounding; the state dict is the same, so round 6's
  `hybrid_e240` stays a fair base. `amp_layernorm` is retired (AMP always keeps
  LayerNorm in bf16, which is what r6 did with `amp_layernorm True`), and
  round 7's configs drop the key. `make_sweep.py` checks the base key by key
  against `../r6/config_train_hybrid_e240.txt` and `config_infer_hybrid_e240.txt`.
- Restart, overrides, `stop_135`, `kill_all.sh`, MPS, `use_compile`,
  checkpointing and the hierarchy cache work as in `../r6/README.md`. No job
  needs gradient checkpointing; the tightest shared one is `e240_bs8`
  (~97 GB of the 120 GB per-job budget).

## The design

| axis | jobs | what it asks |
| --- | --- | --- |
| epochs x batch | `e480`, `e480_bs8`, `e480_bs16`, `e360`, `e240_bs8` | does training longer keep helping, and is it the epochs or the optimizer steps? |
| MMD weight | `e240_mmd` 0.1, 10, 100, 1000, 10000 (base 1) | where does a stronger latent match start to help or hurt? |
| KL | `e240_kl` 0.1, 1, 10, 100 | is the per-sample KL a better regularizer than the aggregate MMD? |

Optimizer steps against the base (240 epochs, batch 4); the learning rate is
not rescaled with the batch size:

| | batch 4 | batch 8 | batch 16 |
| --- | --- | --- | --- |
| 240 epochs | 1x (**base**, r6: 69) | 0.5x `e240_bs8` | - |
| 360 epochs | 1.5x `e360` | - | - |
| 480 epochs | 2x `e480` | 1x `e480_bs8` | 0.5x `e480_bs16` |

`e480_bs8` has the base's step count with twice the samples per step, so
against the base it separates "more data passes" from "more steps"; `e480_bs16`
and `e240_bs8` are the cheap ends. Batch 8 at 120 epochs scored 79 in r6
against 88/88/107 at batch 4, so larger batches are not expected to hurt.

- **MMD**: with Adam and per-group gradient clipping only weight *ratios*
  matter, so `lambda_mmd 10000` against `alpha_recon 1000` makes the MMD
  dominate the encoder gradient. That is the point: find where it starts to
  hurt. r6 at 120 epochs: 0.1 76, 1 88, 100 71.
- **KL** (`latent_reg kl`): a per-sample KL averaged per latent coordinate.
  Round 3's `kl_weight 1e-3` did nothing next to `alpha_recon 1000`, so the
  range starts at 0.1. `lambda_mmd` and `mmd_bandwidth` are dropped from these
  configs, because the launcher warns that they are ignored.
- `prior_freeze_epoch` follows the epochs at 2/3 (160 at 240, 320 at 480).
- **Time**: the plan below uses round 6's estimate (34 h per 120 epochs alone
  with checkpointing, x1.4 two per card, 1/1.52 without checkpointing). Round
  6 ran at about 0.6-0.7x of it, so expect about 3.5 days rather than 5.4.
  Eleven jobs start at once; the last three (`e240_kl1`, `kl10`, `kl100`)
  take the first slots to free up, about 1.5-2 days in.

## Reading the results

`brief.py` prints one line per finished job, `# w2`, best first. Read every
job against the base, round 6's `hybrid_e240` = 69. One run per setting means
one number per setting: in r6 three seeds of the same 120-epoch config spread
over 88-107 (sd about 11), so a difference of 10 or less from the base is not
an effect on its own; trust a trend along an axis (the MMD or KL curve, the
epoch/step table) over any single job. W2 scale, floor and caveats are as in
`../r6/README.md` (floor 27, 100 = the exact FE mean with no spread).
Selection on the held-out file still biases the best numbers downward.

## Jobs (generated by make_sweep.py; everything below is rewritten)

14 trainings, seed 42, one run each, on 11 slots (GPUs 0, 2, 3 one job each, 4, 5, 6, 7 two each); estimated makespan 5.4 days (34 h per 120 epochs alone with checkpointing, x1.4 two per card, 1/1.52 without checkpointing, per-epoch batch factors {2: 1.3, 4: 1.0, 8: 0.85, 16: 0.75}; round 6 ran at about 0.6-0.7x of such estimates).

| # | job | change from the base (r6 hybrid_e240) | GPU | checkpointing | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_e480 | epochs 480 (2x the base's steps); alone on a card | 0 (alone) | off | 91 | 0.0-3.8 |
| 2 | hybrid_e480_bs8 | epochs 480, batch size 8 (the base's steps); alone on a card | 2 (alone) | off | 78 | 0.0-3.3 |
| 3 | hybrid_e480_bs16 | epochs 480, batch size 16 (half the base's steps); alone on a card | 3 (alone) | off | 69 | 0.0-2.9 |
| 4 | hybrid_e360 | epochs 360 (1.5x the base's steps) | 4 | off | 96 | 0.0-4.0 |
| 5 | hybrid_e240_bs8 | batch size 8 (half the base's steps) | 5 | off | 55 | 0.0-2.3 |
| 6 | hybrid_e240_mmd0.1 | lambda_mmd 0.1 | 6 | off | 65 | 0.0-2.7 |
| 7 | hybrid_e240_mmd10 | lambda_mmd 10 | 7 | off | 65 | 0.0-2.7 |
| 8 | hybrid_e240_mmd100 | lambda_mmd 100 | 4 | off | 65 | 0.0-2.7 |
| 9 | hybrid_e240_mmd1000 | lambda_mmd 1000 | 5 | off | 65 | 0.0-2.7 |
| 10 | hybrid_e240_mmd10000 | lambda_mmd 10000 | 6 | off | 65 | 0.0-2.7 |
| 11 | hybrid_e240_kl0.1 | latent_reg kl, kl_weight 0.1 | 7 | off | 65 | 0.0-2.7 |
| 12 | hybrid_e240_kl1 | latent_reg kl, kl_weight 1 | 5 | off | 65 | 2.3-5.0 |
| 13 | hybrid_e240_kl10 | latent_reg kl, kl_weight 10 | 6 | off | 65 | 2.7-5.4 |
| 14 | hybrid_e240_kl100 | latent_reg kl, kl_weight 100 | 7 | off | 65 | 2.7-5.4 |
