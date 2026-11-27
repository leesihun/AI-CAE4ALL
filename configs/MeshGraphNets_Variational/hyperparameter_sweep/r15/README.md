# B300 round 15: round 14's best setting again, on the augmented ex3 dataset

Model: HI-MGN-hybrid (`meshgraphnets-v`, `latent_type hybrid`) on ex3 shell
buckling. Node 135, fourteen jobs on all eight cards, all at once, MPS off,
about 60 hours (the user, 2026-11-27).

## What it asks

Round 14 had one setting clearly below the base: `joint_tfk8_dec` (a8), 55/57
against the base's 66/95/75. It was two seeds, on 150 training geometries. This
round runs it again and builds on it:

1. **Does 56 come back?** `joint_tfk8_dec` with three seeds beside the base
   with three seeds, both on the new data.
2. **Is it the trunk?** `joint_tfk8` (round 14's a6, 68/81) is `joint_tfk8_dec`
   without `prior_trunk decoder`.
3. **Does the per-sample KL add to it?** `joint_tfk8_dec_kl10`. Round 14 never
   ran the three together: KL 10 helped the GNN prior (74) and hurt the
   transformer prior without the trunk (84).
4. **Is it still improving?** `joint_tfk8_dec_e450`: the same for 450 epochs
   (1.9x the updates).
5. **Does the new data help the base?** The base here against the base of
   rounds 8-10 (66/95/75), on the 8 held-out geometries those rounds scored.
6. **A small check, two jobs:** `stochastic_mode noise` trained on the energy
   score per graph instead of the CRPS per node (round 14: 125 and 245).

## The data (changed 2026-11-26)

| file | holds | size |
| --- | --- | --- |
| `dataset/probabilistic/ex3_shell_buckling.h5` | 210 geometries x 64 draws = 13,440 samples: the 150 of the earlier rounds and 60 around the buckle / no-buckle transition (`shape_idx` 300-359) | 10.08 GB |
| `dataset/probabilistic/ex3_shell_buckling_infer.h5` | 16 held-out geometries x 256 draws = 4,096 scenes: the 8 of the earlier rounds (`shape_idx` 150-157) and 8 more (200-207) | 2.94 GB |

**Both files are git-ignored: copy them to node 135 before starting.** The
names are the old ones; `run_node.sh` reads the attributes and stops if either
file is the old one.

- **Split.** Every train config has `split_group_attr split_group` (was
  `shape_idx`). Each of the 60 new geometries shares a `split_group` with one
  training geometry, so with `split_seed 42` the 120/15/15 partition of the
  earlier rounds is kept and all 60 train: 180 training geometries, 11,520
  samples, 720 steps per epoch at batch 16 (was 480). `split_group_attr
  shape_idx` on this file would deal the 210 geometries again and move old
  training geometries into the test split.
- **Scoring.** A `split_group` can hold two geometries, so every infer config
  has `eval_group_attr shape_idx` (new native key; default = the split
  attribute): the scores and the diagnosis are per geometry. The held-out file
  has no `split_group` and needs none (inference never splits).
- **Comparing with earlier rounds.** Only the 8 geometries 150-157. `brief.py`
  prints each setting's score on them (the per-seed numbers and `m`) and on
  200-207 (`x`) separately. The two groups also differ in kind (the file's
  `contents` attribute): 150-157 each lie outside the training ranges in one
  factor, 200-207 inside them, where a shell goes from not buckling to
  buckling.

## The settings

| | setting | what it is | seeds | epochs |
| --- | --- | --- | --- | --- |
| a0 | `base` | two parts: the prior alone from epoch 160 (2/3), its loss does not reach the encoder | 42, 43, 44 | 240 |
| a1 | `joint_tfk8` | round 14's a6: `prior_freeze_epoch 0`, `prior_grad_to_encoder 1`; `prior_arch transformer`, `prior_fm_draws 8`, `local_inject coarsest`, `lambda_mmd_g 0` | 42, 43 | 240 |
| a2 | `joint_tfk8_dec` | round 14's a8: a1 + `prior_trunk decoder` | 42, 43, 44 | 240 |
| a3 | `joint_tfk8_dec_kl10` | a2 with `latent_reg kl`, `kl_weight 10` (no MMD) | 42, 43 | 240 |
| a4 | `joint_tfk8_dec_e450` | a2 for 450 epochs | 42, 43 | 450 |
| a5 | `noise_energy` | `stochastic_mode noise`, `noise_draws 2`, `noise_loss energy`; no encoder, no prior | 42 | 200 |
| a6 | `noise_global_energy` | a5 with `latent_type global` (one vector per sample, no z_l) | 42 | 200 |

**Epochs.** 240 epochs of 720 steps are the 172,800 updates of round 14's 360
epochs of 480. The two noise jobs train 200 epochs (144,000 updates): a noise
step costs about 1.14 of a base step, and at 240 epochs their card would end
5-10 hours after the others. Round 14's noise jobs were stopped at about
150,000 updates.

**`prior_trunk decoder`** (see round 14's README): the prior has no trunk of
its own. The simulator's descending arm is run once more at z = 0 without
gradient, and the prior reads its coarsest-level features.

**`noise_loss energy`** (new native key; `crps` is the default). The fair
energy score of the `noise_draws` forwards, one value per graph: a graph's
whole output field is one vector, so the loss also sees how the draws co-vary
across nodes. The CRPS is a sum of per-node scores and does not. Validation
and `best_by crps` are unchanged.

## What is already known against it

- **`joint_tfk8_dec`**: two seeds. Round 13's `kl10` was 69/72 on two seeds
  and its round-14 counterpart came out at 74, level with the base's best
  seeds.
- **450 epochs**: in round 8 (the GNN prior in two parts, one seed each) the
  longer runs were 66 at 1.5x the epochs and 77 at 3x. A longer run of a8 has
  no result.
- **KL 10**: on the transformer prior without the trunk it lost (84 against
  74).
- **Energy-score noise**: round 14's CRPS noise lost to the base by a wide
  margin (125, and 245 twice, which looks like a collapsed spread). The energy
  score is another loss on the same generator; nothing says it closes a gap
  that size. Two jobs, one seed each: a result near the base would need a
  second seed before it counts.
- **The new data**: the 60 new geometries are placed where the held-out error
  was largest (the transition). A lower score on 150-157 can come from the
  data alone, which is why the base is run again.

## Before starting

1. Pull. `run_node.sh` stops with "Pull first." if the checkout lacks
   `noise_loss` or `eval_group_attr`.
2. Copy the two dataset files (above) to `dataset/probabilistic/` on 135.
3. Rounds 14 and 15 cannot run on 135 together (the same cards).

## Running it (node 135, cards 0-7)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r15/run_135.sh

`run_135.sh` sets `MPS=0` itself; the two jobs of a card time-share it. 4
torch threads per job (`THREADS_PER_JOB` overrides).

| card | its jobs | checkpointing | est. end (model) |
| --- | --- | --- | --- |
| 4 | `base` 42, `joint_tfk8_dec` 43 | on | 67 h |
| 5 | `base` 43, `joint_tfk8_dec` 44 | on | 67 h |
| 6 | `base` 44, `joint_tfk8_dec_kl10` 42 | on | 67 h |
| 7 | `joint_tfk8_dec` 42, `joint_tfk8` 42 | on | 67 h |
| 2 | `joint_tfk8` 43, `joint_tfk8_dec_kl10` 43 | on | 67 h |
| 1 | `noise_energy` 42, `noise_global_energy` 42 | on | 67 h |
| 0 | `joint_tfk8_dec_e450` 42 | off | 50 h |
| 3 | `joint_tfk8_dec_e450` 43 | off | 50 h |

No card holds two seeds of one setting (`make_sweep.py` refuses otherwise).
Each two-job card pairs a job counted x1.0 with a `decoder`-trunk job (x0.88);
the two shorter noise jobs share one card. To change the layout, edit `ORDER`
in `make_sweep.py` and regenerate with `--force`; running jobs resume.

At any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r15/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **a0-a6**: the settings in the table above.
- **Mark**: `+` every seed beats every seed of the other setting, `-` every
  seed loses, `=` neither. a1 has one mark, against the base. a2 has two:
  against the base, then against a1. a3 and a4 have their second mark against
  a2. a5 and a6 have one seed and so no mark: read their number against the
  base's three.
- **The per-seed numbers and `m`**: W2 on the held-out geometries 150-157, the
  8 of every earlier round. **`x`**: the same on 200-207, mean over the seeds.
- **ho / tr / ts**: floor / posterior sample / prior on the 16 held-out
  geometries, on 16 of the 180 training geometries, and on the 15 geometries
  of the test split. a5 and a6 have none (no encoder, no prior).
- **g, l, n, s**: as in round 14.

## Diagnosis and histograms (automatic)

As in round 14: `run_135.sh` starts `diagnose_135.sh`, which diagnoses each
finished job and draws one histogram per geometry at the end:

    output/b300_r15/_diag/unseen_<setting>_<tr|ts|ho>.png

`brief.py` prints PRIOR DIAGNOSIS above the block (one decision per setting:
CONFIRMED 1, RULED OUT 2, OPEN 3, WORSE 4, MIXED 5) and the block's `pd` / `pb`
lines carry it:

| setting | reference | cause it tests |
| --- | --- | --- |
| `joint_tfk8` | the base | 3 the prior does not fit its training geometries |
| `joint_tfk8_dec` | `joint_tfk8` | 4 missed on new geometries inside the training range |
| `joint_tfk8_dec_kl10` | `joint_tfk8_dec` | 24 the posteriors are points with gaps |
| `joint_tfk8_dec_e450` | `joint_tfk8_dec` | 3 the prior does not fit its training geometries |
| the two noise settings | | none: `diagnose_135.sh` skips them |

`kill_all.sh` stops the watcher too.

## Memory and time

Peak memory follows the largest batch: 16 samples drawn from the new file
reach 1.14M nodes (the old file: 1.05M), about 197 GB for the base without
gradient checkpointing. Two such jobs do not fit one card (260 GB counted of
288), so the twelve jobs on two-job cards have checkpointing on (x1.52 per
step, the gradients unchanged; peaks about 53-69 GB, a guess at 0.3 of the
full peak). The two 450-epoch jobs are alone on a card with checkpointing off
(about 174 GB).

Time: round 8 measured 277 s per epoch of 480 steps alone on a card; 720 steps
and 1.2% more nodes give 0.117 h per epoch. With checkpointing x1.52, each job
x1.5 slower while two share a card (a guess), 4 h of inference per job (4,096
scenes x 4 draws). The model gives **67 h for the two-job cards and 50 h for
the long jobs**. Round 14 ran at about 0.8 of what the same model gave, which
would be **about 54 h**. The diagnosis runs after that (minutes per job).

The CPU: 14 jobs x (4 torch threads + 4 loader workers) = 112 processes on
128 cores.

If the round has to end early: `touch output/b300_r15/stop_135` lets running
jobs finish; `kill_all.sh` stops them now. Every job writes a resume state
every 15 minutes, so the same command continues it.

## Jobs (generated by make_sweep.py; everything below is rewritten)

14 trainings (7 settings: base seeds 42/43/44; joint_tfk8 seeds 42/43; joint_tfk8_dec seeds 42/43/44; joint_tfk8_dec_kl10 seeds 42/43; joint_tfk8_dec_e450 seeds 42/43; noise_energy seeds 42; noise_global_energy seeds 42), all at once on GPUs 0, 1, 2, 3, 4, 5, 6, 7 (GPUS 4 5 6 7 2 1 0 3 4 5 6 7 2 1), 4 torch threads per job, MPS off. Estimated end of the last inference: 67 h as modelled, about 54 h if the round runs at round 14's 0.8 of the model (GUESSED speeds: 0.117 h per batch-16 epoch alone, x1.52 with checkpointing, each job x1.5 while 2 share a card, 4 h of inference per job; cost per job from cost_scale: noise x1.14, prior_trunk decoder x0.88; peaks with checkpointing guessed at 0.3 of the full peak). The diagnosis (diagnose_135.sh) runs after that.

| # | job | what it is | GPU | epochs | checkpointing | est. peak GB | est. training h | est. end h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_base_s42 | the base: the prior alone from epoch 160, its loss does not reach the encoder; seed 42 | 4 | 240 | on | 60 | 63 | 67 |
| 2 | hybrid_base_s43 | the base: the prior alone from epoch 160, its loss does not reach the encoder; seed 43 | 5 | 240 | on | 60 | 63 | 67 |
| 3 | hybrid_base_s44 | the base: the prior alone from epoch 160, its loss does not reach the encoder; seed 44 | 6 | 240 | on | 60 | 63 | 67 |
| 4 | hybrid_joint_tfk8_dec_s42 | round 14's a8: no prior-only part and the prior's loss reaches the encoder (prior_freeze_epoch 0, prior_grad_to_encoder 1); transformer prior, 8 flow draws per sample, z_l read at the coarsest level, no MMD on z_g; the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder); seed 42 | 7 | 240 | on | 53 | 56 | 62 |
| 5 | hybrid_joint_tfk8_s43 | round 14's a6: no prior-only part and the prior's loss reaches the encoder; transformer prior, 8 flow draws per sample, z_l read at the coarsest level, no MMD on z_g; seed 43 | 2 | 240 | on | 60 | 63 | 67 |
| 6 | hybrid_noise_energy_s42 | stochastic_mode noise: no encoder, no prior; z_g and z_l ~ N(0,I), 2 draws per batch, loss = their fair energy score per graph (noise_loss energy); 200 epochs; seed 42 | 1 | 200 | on | 69 | 61 | 67 |
| 7 | hybrid_joint_tfk8_dec_e450_s42 | round 14's a8 for 450 epochs (the others: 240); seed 42 | 0 | 450 | off | 174 | 46 | 50 |
| 8 | hybrid_joint_tfk8_dec_e450_s43 | round 14's a8 for 450 epochs (the others: 240); seed 43 | 3 | 450 | off | 174 | 46 | 50 |
| 9 | hybrid_joint_tfk8_dec_s43 | round 14's a8: no prior-only part and the prior's loss reaches the encoder (prior_freeze_epoch 0, prior_grad_to_encoder 1); transformer prior, 8 flow draws per sample, z_l read at the coarsest level, no MMD on z_g; the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder); seed 43 | 4 | 240 | on | 53 | 56 | 62 |
| 10 | hybrid_joint_tfk8_dec_s44 | round 14's a8: no prior-only part and the prior's loss reaches the encoder (prior_freeze_epoch 0, prior_grad_to_encoder 1); transformer prior, 8 flow draws per sample, z_l read at the coarsest level, no MMD on z_g; the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder); seed 44 | 5 | 240 | on | 53 | 56 | 62 |
| 11 | hybrid_joint_tfk8_dec_kl10_s42 | round 14's a8 with the per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 10); seed 42 | 6 | 240 | on | 53 | 56 | 62 |
| 12 | hybrid_joint_tfk8_s42 | round 14's a6: no prior-only part and the prior's loss reaches the encoder; transformer prior, 8 flow draws per sample, z_l read at the coarsest level, no MMD on z_g; seed 42 | 7 | 240 | on | 60 | 63 | 67 |
| 13 | hybrid_joint_tfk8_dec_kl10_s43 | round 14's a8 with the per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 10); seed 43 | 2 | 240 | on | 53 | 56 | 62 |
| 14 | noise_global_energy_s42 | stochastic_mode noise with latent_type global (one N(0,I) vector per sample, no z_l), 2 draws per batch, loss = their fair energy score per graph; 200 epochs; seed 42 | 1 | 200 | on | 69 | 61 | 67 |
