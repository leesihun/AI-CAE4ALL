# B300 round 13: a latent forced into order, and a plainer model

Model: HI-MGN-hybrid (`meshgraphnets-v`, `latent_type hybrid`) on ex3 shell
buckling. Node 135, twenty jobs, two per card on cards 0–7 (sixteen at once,
four as slots free).

## What it asks

The posterior holds on unseen geometries (held-out 32 against a floor of 27);
the prior does not (66/95/75). So the question is what the prior is asked to
learn. Three things are known about the latent of the base:

- z_g is unused: the decoder gives the same result with z_g swapped or zeroed.
- z_l carries the draw. Each of its rows looks N(0,1), but a z ~ N(0,I) sent
  straight to the decoder scores 578 on the held-out geometries against the
  prior's 83: the 256 coarse nodes together are far from N(0,I).
- The posteriors are narrow (width 0.51 of the spread of their means; 0.03 in
  the 360-epoch run): points with gaps between them.

This round asks two questions on round 12's `joint` setting
(`prior_freeze_epoch 0`, `prior_grad_to_encoder 1`):

1. **Does a latent forced into order give a prior that holds on unseen
   geometries?** Two existing keys force it: a wider posterior
   (`posterior_min_std`) and the per-sample KL to N(0,I) (`latent_reg kl`).
   The KL is the direct one: with every sample's posterior near N(0,I), every
   geometry's latent is near N(0,I), and the prior has little left to learn.
2. **Does the model need all its parts?** `latent_type local` drops z_g;
   `local_joint_kl1` is the plainest model here: one latent field, the usual
   VAE term, a prior trained with it.

Two more settings change how the prior is trained (`prior_fm_time_power`) and
one weights the MMD by how alike two geometries are (`mmd_cond_width`).

`brief.py` prints two numbers per setting that say whether the latent is in
order (`n`, `s`, below), so the round can tell "the latent is in order and the
prior is still off" from "the latent never got into order".

## The settings

| setting | change from `joint` | seeds |
| --- | --- | --- |
| base (not run here) | two parts: `prior_freeze_epoch 240`, `prior_grad_to_encoder 0` | 42, 43, 44 |
| `joint` | none (round 12's setting; the control here) | 42, 43 |
| `joint_tp2` | `prior_fm_time_power 2`: P(t < 0.1) 0.10 → 0.32 | 42, 43 |
| `joint_pms03` | `posterior_min_std 0.3` | 42, 43 |
| `joint_pms05` | `posterior_min_std 0.5` | 42, 43 |
| `joint_kl1` | `latent_reg kl`, `kl_weight 1` (no MMD) | 42, 43 |
| `joint_kl10` | `latent_reg kl`, `kl_weight 10` | 42, 43 |
| `local_joint` | `latent_type local` (no z_g) | 42, 43 |
| `local_joint_kl1` | `latent_type local`, `latent_reg kl`, `kl_weight 1` | 42, 43 |
| `joint_tp3` (second wave) | `prior_fm_time_power 3`: P(t < 0.1) → 0.46 | 42, 43 |
| `joint_cmmd1` (second wave) | `mmd_cond_width 1` | 42, 43 |

The base's three runs are r8 `hybrid_e360`, r9 `hybrid_base_s43` and r10
`hybrid_base_s44`; `brief.py` reads them where those rounds left them.
Everything else equals the base: 360 epochs, batch 16, `beta_aux 10`,
`best_by crps`.

## What is already known against it

- Round 8 ran the per-sample KL at 0.1, 1, 10, 100 on the two-part schedule
  (240 epochs, one seed each): 95, 112, 85, 99 against 85. No gain, but nobody
  looked at whether the latent was in order in those runs.
  **`organized_135.sh` reads that off the trained runs**, in minutes, without
  training (below).
- The 360-epoch run has the narrowest posteriors and the best score of the
  base's seeds (within seed noise).
- `local` against `hybrid` on the two-part schedule: 94–108 against 66–107.

## The flow time (`prior_fm_time_power`)

The flow-matching loss is nearly blind to how much of each outcome it makes,
except near the noise end, where the outcomes still overlap. With power p the
training time is drawn as u^p and the sampler steps are dense at the same end.
p = 1 is the old behaviour, bit for bit.

## The MMD weighted by geometry (`mmd_cond_width`)

Each graph gets a fixed summary of its input: the mean and std over its nodes
of every node input feature, and over its edges of every edge feature. Each
column is standardized over the batch. The kernel between two graphs is an RBF
whose squared width is `mmd_cond_width` times the median squared distance
between two graphs in the batch. Every z kernel of the MMD is multiplied by
it, and the averages are weighted by it, so a constant kernel gives the plain
MMD exactly.

Measured on ex3 batches of 16 it is weak where the draw lives: a batch holds
about one pair of draws of the same geometry in 120, and for z_l the statistic
sees an offset or a scale of a geometry's pooled rows (8–24 times the value of
independent rows) but hardly a per-node dependence on the geometry (1.6–3.2
times). The per-node form is the per-sample KL, which is why the KL settings
run first and this one last. The narrower width (0.25) left almost only each
graph against itself and shrinks z_g, so it is not run.

## Where each can fail, and where it shows

- **pms03/pms05, kl1/kl10**: the posterior is wider by design, so its second
  number sits above the floor. Too strong and the decoder stops telling draws
  apart: the second number rises far above the floor and `l` falls. A KL that
  is too strong ends in a decoder that ignores z (prior near 100: the mean
  with no spread).
- **tp2/tp3**: fewer training steps near the data end. If the decoded fields
  get blurrier, the third number (prior) on `tr` rises above `joint`'s.
- **local**: if z_g did carry something under `joint`, the prior gets worse
  than `joint`'s.
- **cmmd1**: if the pull wins over reconstruction, z loses the draw's
  information; the second number leaves the floor.

On a 12 x 10 toy grid (CPU) every setting trains, validates and samples. That
says the code paths work; it says nothing about ex3.

## Before starting

Pull first. `run_node.sh` stops with "Pull first." if the checkout lacks
`prior_fm_time_power` or `mmd_cond_width`. Both default to the old behaviour.

Round 12 and round 13 cannot run on 135 together. `joint` is in both; the other
three settings of round 12 (`joint_g01`, `joint_kl`, `joint_tfk8`) are not here
(round 12's `joint_kl` used `kl_weight 0.01`).

## Running it (node 135, cards 0–7, two jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r13/run_135.sh

The same runner as rounds 6–12. Sixteen jobs start at once: jobs 1–8 on cards
0–7 (seed 42 of the first eight settings), jobs 9–16 on cards 0–7 again (their
seed 43; card k holds both seeds of one setting). Jobs 17–20 (`joint_tp3`,
`joint_cmmd1`) each take the first slot that frees. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r13/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **a0–a10**: base, `joint`, `joint_tp2`, `joint_pms03`, `joint_pms05`,
  `joint_kl1`, `joint_kl10`, `joint_tp3`, `local_joint`, `joint_cmmd1`,
  `local_joint_kl1`.
- **Mark**: `+` every seed beats every seed of the other setting, `-` every
  seed loses, `=` neither. a1 has one mark, against the base. a2–a9 have two:
  against the base, then against a1. a10 is marked against a8 (`local_joint`).
- **ho / tr / ts**: floor / posterior sample / prior on the 8 held-out
  geometries, on 16 training geometries, and on the 15 geometries of the test
  split.
- **g, l**: how much the decoded draw depends on z_g, on z_l (0 = ignored;
  `g-` = no z_g).
- **n**: a z ~ N(0,I), which the prior never saw, decoded on the held-out
  geometries and scored like the prior. Equal to the prior's number = the
  decoder reads any N(0,I) draw as a valid result, so the geometry does not
  move the latent. The base: 578.
- **s**: the posterior's width over the spread of its means, for z_l (x100).
  Near 0 = points with gaps; 100 or more = the samples of a geometry overlap.
  The base: 51.

Reading n with the prior: n down and the prior down = the order of the latent
was the cause. n down and the prior unchanged = it was not. n unchanged = the
setting did not put the latent in order.

## The same two numbers from runs that already exist

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r13/organized_135.sh [GPU]

Round 8's KL runs (0.1, 1, 10, 100), its base and its `mmd10000` are trained
and scored; this diagnoses each on the held-out file (forward passes only,
~10–20 GB, minutes per run, in the foreground) and prints one short block
(`ORG 135`). It can run beside round 13 on any card with room. A run whose
checkpoint is gone prints `none`. When none of round 8's checkpoints is left,
it takes the base's seeds 43/44 (rounds 9, 10) and round 11's `local` runs
instead (`ORG 135 other`): those have no KL, so they give n, s, c for the base
and the plainer model but cannot say what the KL does; round 13's KL settings
answer that.

Each line ends in a third number, `c`: the largest over the smallest z_l cloud
size among the 8 held-out geometries (x100). It says whether a normalizer per
geometry (a head that predicts each geometry's latent mean and scale, the flow
learning the rest) would have anything to rescale: 100 = every geometry's
latent already has the same size. The normalizer over all geometries together
already exists (`zl_shift` / `zl_scale`, fitted at the prior freeze) and is on
in the base.

## Diagnosis and histograms (automatic)

As in round 12: `run_135.sh` starts `diagnose_135.sh`, which diagnoses each
finished job and draws one histogram per geometry at the end:

    output/b300_r13/_diag/unseen_<setting>_<tr|ts|ho>.png

To draw them earlier: `python .../r13/unseen_brief.py --figures` (MGN-V
interpreter). `kill_all.sh` stops the watcher too.

## Memory and time

Two jobs per card: without checkpointing one job peaks at about 180 GB, over
half of the card, so checkpointing is on (peak guessed at about 55 GB). About
65 h per job; sixteen at once, so the first eight settings are done after
about 2.7 days and the last two after about 5.4 — a guess from r8's epoch
time, x1.52 for checkpointing, x1.5 for sharing a card.

## Jobs (generated by make_sweep.py; everything below is rewritten)

20 trainings (10 settings: joint seeds 42/43; joint_tp2 seeds 42/43; joint_pms03 seeds 42/43; joint_pms05 seeds 42/43; joint_kl1 seeds 42/43; joint_kl10 seeds 42/43; local_joint seeds 42/43; local_joint_kl1 seeds 42/43; joint_tp3 seeds 42/43; joint_cmmd1 seeds 42/43; the base is not run: its 42/43/44 = r8 hybrid_e360, r9 hybrid_base_s43, r10 hybrid_base_s44); --pack 2: 2 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 16 slots, 3 torch threads per job. Estimated makespan 5.4 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, x1.5 per job at 2 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_joint_s42 | r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 42 | 0 | on | 55 | 65 | 0.0-2.7 |
| 2 | hybrid_joint_tp2_s42 | joint, flow time dense at the noise end (prior_fm_time_power 2); seed 42 | 1 | on | 55 | 65 | 0.0-2.7 |
| 3 | hybrid_joint_pms03_s42 | joint, wider posterior (posterior_min_std 0.3); seed 42 | 2 | on | 55 | 65 | 0.0-2.7 |
| 4 | hybrid_joint_pms05_s42 | joint, wider posterior (posterior_min_std 0.5); seed 42 | 3 | on | 55 | 65 | 0.0-2.7 |
| 5 | hybrid_joint_kl1_s42 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 1); seed 42 | 4 | on | 55 | 65 | 0.0-2.7 |
| 6 | hybrid_joint_kl10_s42 | joint, per-sample KL, stronger (kl_weight 10); seed 42 | 5 | on | 55 | 65 | 0.0-2.7 |
| 7 | local_joint_s42 | joint without z_g (latent_type local); seed 42 | 6 | on | 55 | 65 | 0.0-2.7 |
| 8 | local_joint_kl1_s42 | joint without z_g, per-sample KL (latent_type local, kl_weight 1); seed 42 | 7 | on | 55 | 65 | 0.0-2.7 |
| 9 | hybrid_joint_s43 | r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 43 | 0 | on | 55 | 65 | 0.0-2.7 |
| 10 | hybrid_joint_tp2_s43 | joint, flow time dense at the noise end (prior_fm_time_power 2); seed 43 | 1 | on | 55 | 65 | 0.0-2.7 |
| 11 | hybrid_joint_pms03_s43 | joint, wider posterior (posterior_min_std 0.3); seed 43 | 2 | on | 55 | 65 | 0.0-2.7 |
| 12 | hybrid_joint_pms05_s43 | joint, wider posterior (posterior_min_std 0.5); seed 43 | 3 | on | 55 | 65 | 0.0-2.7 |
| 13 | hybrid_joint_kl1_s43 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 1); seed 43 | 4 | on | 55 | 65 | 0.0-2.7 |
| 14 | hybrid_joint_kl10_s43 | joint, per-sample KL, stronger (kl_weight 10); seed 43 | 5 | on | 55 | 65 | 0.0-2.7 |
| 15 | local_joint_s43 | joint without z_g (latent_type local); seed 43 | 6 | on | 55 | 65 | 0.0-2.7 |
| 16 | local_joint_kl1_s43 | joint without z_g, per-sample KL (latent_type local, kl_weight 1); seed 43 | 7 | on | 55 | 65 | 0.0-2.7 |
| 17 | hybrid_joint_tp3_s42 | joint, flow time denser at the noise end (prior_fm_time_power 3); seed 42 | 0 | on | 55 | 65 | 2.7-5.4 |
| 18 | hybrid_joint_tp3_s43 | joint, flow time denser at the noise end (prior_fm_time_power 3); seed 43 | 1 | on | 55 | 65 | 2.7-5.4 |
| 19 | hybrid_joint_cmmd1_s42 | joint, MMD weighted by geometry likeness (mmd_cond_width 1); seed 42 | 2 | on | 55 | 65 | 2.7-5.4 |
| 20 | hybrid_joint_cmmd1_s43 | joint, MMD weighted by geometry likeness (mmd_cond_width 1); seed 43 | 3 | on | 55 | 65 | 2.7-5.4 |
