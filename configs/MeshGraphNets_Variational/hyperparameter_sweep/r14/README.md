# B300 round 14: the prior (GNN or transformer) x the latent regularizer (MMD or KL 10, 100) x what the prior reads of the geometry

Model: HI-MGN-hybrid (`meshgraphnets-v`, `latent_type hybrid`) on ex3 shell
buckling. Node 135, twenty-four jobs, **three per card on all eight cards, all
at once** (the user, 2026-11-24), gradient checkpointing on, MPS off.

## What it asks

Round 13 ran twelve settings on round 12's `joint` setting. Two things moved:

- `joint_kl10` (the per-sample KL at weight 10 in place of the MMD) put the
  latent in order (a decoded N(0,I) draw: 551 → 211) and had both seeds low on
  the held-out geometries (69/72; the base's seeds: 66/95/75).
- Before that, round 9's `tf_k8` (a transformer prior trained on 8 flow draws
  per sample) was the only prior that fitted the training geometries better
  without a warm-up (tr 72/70). It was never run on `joint`.

A third thing was never varied at all: **what the prior reads of the
geometry**. Every training config of rounds 8–13 has the same trunk (3
message-passing blocks on the fine mesh, averaged into the coarsest clusters);
only the network after it changed. Round 11's diagnosis put 21 of the 53 points
of held-out gap on "missed on new geometries inside the training range" and 28
on the fit of the training geometries themselves.

So this round has three columns of change on `joint`:

| | MMD | KL 10 | KL 100 | trunk `decoder` | trunk `multiscale` |
| --- | --- | --- | --- | --- | --- |
| GNN prior | `joint` | `joint_kl10` | `joint_kl100` | `joint_dec` | `joint_ms` |
| transformer prior | `joint_tfk8` | `joint_tfk8_kl10` | not run | `joint_tfk8_dec` | not run |

The trunk settings keep the MMD. Eight settings x seeds 42/43. Every run is
`joint`: `prior_freeze_epoch 0` (one objective for all 360 epochs) and
`prior_grad_to_encoder 1` (the prior's loss reaches the encoder).

Four questions:

1. Does the KL help along a line (MMD → 10 → 100), or was 69/72 at weight 10
   two lucky seeds? `joint_kl10` repeats round 13's run.
2. Does a prior that sees the whole coarse graph hold better on unseen
   geometries, on `joint`?
3. Do the two add up (`joint_tfk8_kl10`)?
4. Does the prior do better when its condition already holds the whole shape
   (`joint_dec`, `joint_ms`, `joint_tfk8_dec`)?

**Changed 2026-11-24.** The round was first a 2 x 4 table with KL 1, 10, 100.
Three settings gave their place to the trunk settings: `joint_kl1` (round 13
ran it: 106/84, worse than the base), `joint_tfk8_kl1` and `joint_tfk8_kl100`.
To bring one back: its weight into `GNN_KL` / `TF_KL` and its two seeds into
`ORDER` in `make_sweep.py`, and its row into `SETTINGS` and `PRIOR_TESTS` in
`brief.py`.

**Added 2026-11-24: no prior at all.** Every question above is about the
prior. Two settings remove it: `stochastic_mode noise` (new native key) builds
no encoder and no prior; z ~ N(0,I) goes through the same AdaLN paths in
training and at inference, and the loss is the CRPS of `noise_draws` (2)
forwards of the same batch (two draws per input; the spread term needs two).
What the decoder is trained on is what it is sampled with, so there is no gap
between posterior and prior to close. The fifth question:

5. Does a sampler with no prior (`noise`, `noise_global`) reach the base on the
   held-out geometries, and does a z_l help it (`noise` vs `noise_global`)?

**Added 2026-11-24: three jobs per card.** All 24 slots start at once, so
two settings were added to fill the last four:

- `joint_kl10_dec`: KL 10 and `prior_trunk decoder` together, the change that
  put the latent in order in round 13 and the one aimed at unseen geometries.
  Read against `joint_kl10` (does the decoder's view add to KL 10?).
- `noise_coarse`: `noise` with `local_inject coarsest`. z_l enters only at the
  coarsest level, so the ascending arm spreads it into a smooth field. It sits
  between `noise_global` (one vector) and `noise` (z_l at every level).

## The settings

| setting | change from `joint` | seeds |
| --- | --- | --- |
| base (not run here) | two parts: `prior_freeze_epoch 240`, `prior_grad_to_encoder 0` | 42, 43, 44 |
| `joint` | none (the control; it died in round 13) | 42, 43 |
| `joint_kl10` | `latent_reg kl`, `kl_weight 10` (no MMD) | 42, 43 |
| `joint_kl100` | `latent_reg kl`, `kl_weight 100` | 42, 43 |
| `joint_dec` | `prior_trunk decoder` | 42, 43 |
| `joint_ms` | `prior_trunk multiscale` | 42, 43 |
| `joint_tfk8` | `prior_arch transformer`, `prior_fm_draws 8`, `local_inject coarsest`, `lambda_mmd_g 0` | 42, 43 |
| `joint_tfk8_kl10` | `joint_tfk8` + `latent_reg kl`, `kl_weight 10` | 42, 43 |
| `joint_tfk8_dec` | `joint_tfk8` + `prior_trunk decoder` | 42, 43 |
| `noise` | the base with `stochastic_mode noise`, `noise_draws 2`; the encoder, regularizer, aux and prior keys are not in the config | 42, 43 |
| `noise_global` | `noise` + `latent_type global` (one N(0,I) vector per sample, no z_l) | 42, 43 |
| `joint_kl10_dec` | `joint_kl10` + `prior_trunk decoder` | 42, 43 |
| `noise_coarse` | `noise` + `local_inject coarsest` | 42, 43 |

The noise settings start from the base, not from `joint` (there is no prior to
train jointly). `prior_freeze_epoch`, `local_warmup_epochs` and
`zg_anchor_weight` above 0 are rejected in noise mode.

**The transformer row is not the transformer alone.** `joint_tfk8` is round 9's
recipe: four keys change together (the network, 8 flow draws per sample, z_l
read by the decoder at the coarsest level only, no MMD on z_g). If the row
wins, a later round separates them.

With `latent_reg kl` the MMD keys (`lambda_mmd`, `lambda_mmd_g`,
`mmd_bandwidth`) are not in the config.

## The prior

All settings use `CoarseFMPrior` (`model/conditional_prior.py`): a trunk that
turns the input graph into one feature row per coarsest node (~256 per shape),
a velocity network on those nodes, the flow-matching loss, the sampler
(`prior_fm_steps 30`).

### The velocity network (`prior_arch`)

- **GNN** (`gnn`, the default): `prior_fm_blocks` (4) message-passing blocks on
  the coarse graph's edges. A coarse node sees its 4-hop neighbourhood; z_g
  talks to the nodes through one per-graph token that reads the node mean
  before every block.
- **Transformer** (`transformer`): one token per coarse node (its noisy z_l,
  its trunk feature, a code of its position) plus one token per z_g slot; 4
  blocks of self-attention over all tokens of a graph (4 heads, width 128).
  Every coarse node sees every other one from the first block; the coarse
  edges are not used. The flow time and a pooled summary of the geometry enter
  every block as one condition (a shift, scale and gate of its normalization).
  At initialization the velocity is exactly 0.

`prior_fm_draws 8`: each posterior sample is regressed from 8 independent
(noise, time) pairs per step. The trunk runs once, so the flow gets 8 times the
supervision at little cost.

### The trunk (`prior_trunk`, new in this round)

- **`fine`** (the default; every earlier round): the prior's own node and edge
  encoders over the raw normalized inputs, `prior_mp_layers` (3) blocks on the
  fine mesh, then the mean over each coarsest cluster. A cluster holds 126–284
  fine nodes; each of them saw 3 hops. The condition of a coarse node is a
  local average, and nothing in it knows the rest of the shape.
- **`multiscale`**: the same, then the prior's own descending arm: at every
  coarser level a mean pool, that level's edges and `prior_mp_layers` more
  blocks (here 4096 nodes, then ~256). Trained by the prior's loss alone.
- **`decoder`**: no trunk of its own. The simulator's descending arm (its
  encoder, the 4 fine and 6 mid-level blocks, the pools) is run once more at
  z = 0 without gradient, and a small MLP reads its coarsest-level features
  (one 128-vector per coarsest node: what the decoder itself holds there
  before it reads z). The prior's loss does not reach the simulator through
  it. `prior_mp_layers` is unused (it stays in the config, inert).

With `multiscale` and `decoder` the pooled summary of the geometry (the z_g
token's input, the transformer's condition) is taken over the coarsest-level
features too.

`joint_dec` changes two things at once: the condition has seen the whole
shape, and it comes from features trained by the reconstruction loss.
`joint_ms` has the first only. Both down = the wider view; `joint_dec` alone =
the decoder's features.

## What is already known against it

- **KL 100**: round 8's two-part `kl100` scored 99 (the mean with no spread).
  The expected failure is a decoder that stops reading z.
- **KL 1** (not run here): round 13's `joint_kl1` was 106/84, worse than the
  base.
- **KL 10**: its prior on the training geometries was 111 and its posterior
  sample on the held-out geometries 41 (floor 27): the latent is in order, the
  decoder pays for it. 69/72 against the base's 66/95/75 is inside the seed
  range.
- **tf_k8**: round 9's numbers (tr 72/70, ho 80/71) are from the two-part
  setting. On the held-out geometries it did not beat the base there.
- **The trunks**: there is no result for either, on any dataset. Against them:
  round 9's transformer priors, which see the whole coarse graph, fitted the
  training geometries (65–72) and missed the held-out ones (105–108); a wider
  view can also be used to memorize 150 geometries. With `decoder` the
  features move for as long as the simulator trains, so the prior's condition
  is not fixed. With `local_inject all` (`joint_dec`) the descending arm is
  trained with posterior samples of z_l, and z = 0 is a value it rarely sees.
- **noise**: no result on ex3. Against it: with 2 draws the CRPS spread term
  is a noisy estimate, so the spread may come out too narrow; and a z ~ N(0,I)
  carries nothing about the FE draw it is paired with, so the decoder has to
  learn the whole spread from the loss alone (no encoder points it at the
  right draw).
- Round 13's `joint` and `joint_pms05` died (both seeds each, cards 0 and 3,
  two jobs per card under MPS). `joint` is rerun here; `joint_pms05` is not.

## Where each can fail, and where it shows

- **KL** (a2, a3, a7): the posterior is wider by design, so the second number
  sits above the floor. Too strong: the second number rises towards 100 and
  `l` falls towards 0 (the decoder ignores z); the prior then scores near 100
  on every set.
- **Transformer** (a6–a8): it can fit the training geometries by position and
  not hold on unseen ones: the prior on `tr` falls and on `ho` it does not
  (MIXED).
- **Trunk** (a4, a5, a8): the same shape of failure: the prior on `tr` falls
  and on `ts` / `ho` it does not. No change on any set = the prior's input is
  not the cause.
- **Noise** (a9, a10, a12): only the w2 and the per-geometry histograms of the
  inference run (`make_histogram True`). Too narrow a spread shows as w2
  near 100 with the right mean; the histograms tell the two apart.

On a 12 x 10 toy grid every setting trains, validates, samples and reloads from
its checkpoint (on the CPU, and on a GPU with fp16 AMP). That says the code
paths work; it says nothing about ex3. The compiled blocks (`use_compile`)
were not exercised there (no Triton on the Windows machine); a compiled call
that fails falls back to eager.

## Before starting

Pull first. `run_node.sh` stops with "Pull first." if the checkout lacks round
13's keys or this round's `prior_trunk` or `stochastic_mode noise`. Round 13 and round 14 cannot run on
135 together (the same cards).

## Running it (node 135, cards 0–7, three jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/run_135.sh

`run_135.sh` sets `MPS=0` itself (not an override: round 13 lost both seeds of
two settings with two jobs per card under MPS); the three jobs of a card
time-share it. 2 torch threads per job (`THREADS_PER_JOB` overrides).

| card | its three jobs (guessed cost: noise x1.14, trunk `decoder` x0.88, others x1.0) | load | est. end |
| --- | --- | --- | --- |
| 4 | `joint` 42, `noise` 42, `joint_dec` 42 | 3.02 | 4.0 days |
| 5 | `joint` 43, `noise` 43, `joint_dec` 43 | 3.02 | 4.0 days |
| 6 | `joint_kl10` 42, `joint_tfk8` 42, `joint_tfk8_kl10` 42 | 3.00 | 3.9 days |
| 7 | `joint_kl10` 43, `joint_tfk8` 43, `joint_tfk8_kl10` 43 | 3.00 | 3.9 days |
| 1 | `joint_kl100` 42, `noise_global` 42, `joint_tfk8_dec` 42 | 3.02 | 4.0 days |
| 2 | `joint_kl100` 43, `noise_global` 43, `joint_tfk8_dec` 43 | 3.02 | 4.0 days |
| 0 | `joint_ms` 42, `noise_coarse` 42, `joint_kl10_dec` 42 | 3.02 | 4.0 days |
| 3 | `joint_ms` 43, `noise_coarse` 43, `joint_kl10_dec` 43 | 3.02 | 4.0 days |

Balanced by guessed cost: each of the six noise jobs (the heaviest) shares a
card with a `decoder`-trunk job (the lightest) and one other; the two cards
left hold three ordinary jobs. If the `decoder` guess is wrong, the noise
cards are at most 0.14 job heavier, which is still the best that six noise
jobs on eight cards allow. A setting's two seeds are on neighbouring cards,
never one (`make_sweep.py` refuses otherwise). The settings added last are on
cards 0 and 3, where round 13's jobs died. Every job finishes at about the
same time; there is no early answer.

To go back to one job per card (checkpointing off, the rest queued):
`python -B .../r14/make_sweep.py --pack 1 --force`. Resume ignores
`use_checkpointing` and the slot layout, so running jobs continue.

At any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **a0–a12**: base, `joint`, `joint_kl10`, `joint_kl100`, `joint_dec`,
  `joint_ms`, `joint_tfk8`, `joint_tfk8_kl10`, `joint_tfk8_dec`, `noise`,
  `noise_global`, `joint_kl10_dec`, `noise_coarse`. Read a1–a3 as one line
  (the GNN prior: MMD, KL 10, KL 100), a4 and a5 against a1 (the trunk), a6–a8
  as the transformer prior (MMD, KL 10, trunk `decoder`), a11 against a2
  (KL 10 with and without the trunk), a9, a10, a12 against the base (no prior).
- **Mark**: `+` every seed beats every seed of the other setting, `-` every
  seed loses, `=` neither. a1 and a9 have one mark, against the base. a2–a6
  have two: against the base, then against a1. a7 and a8 have their second
  mark against a6, a10 and a12 against a9, a11 against a2. a9, a10 and a12
  have the w2 only (no ho / tr / ts, g, l, n, s: there is no encoder or prior
  to diagnose).
- **ho / tr / ts**: floor / posterior sample / prior on the 8 held-out
  geometries, on 16 training geometries, and on the 15 geometries of the test
  split.
- **g, l**: how much the decoded draw depends on z_g, on z_l (0 = ignored).
- **n**: a z ~ N(0,I), which the prior never saw, decoded on the held-out
  geometries and scored like the prior. Equal to the prior's number = the
  decoder reads any N(0,I) draw as a valid result. The base: 551.
- **s**: the posterior's width over the spread of its means, for z_l (x100).
  Near 0 = points with gaps; 100 or more = the samples of a geometry overlap.
  The base: 2.

Reading n with the prior: n down and the prior down = the order of the latent
was the cause. n down and the prior unchanged = it was not. n unchanged = the
setting did not put the latent in order.

## Diagnosis, histograms and the decision per setting (automatic)

As in round 13: `run_135.sh` starts `diagnose_135.sh`, which diagnoses each
finished job (the posterior diagnosis, then round 11's prior diagnosis) and
draws one histogram per geometry at the end:

    output/b300_r14/_diag/unseen_<setting>_<tr|ts|ho>.png

`brief.py` prints PRIOR DIAGNOSIS above the block, one decision per setting
(CONFIRMED 1, RULED OUT 2, OPEN 3, WORSE 4, MIXED 5; see round 13's README
for how it is made), and the block's `pd` / `pb` lines carry it:

| setting | cause it tests |
| --- | --- |
| `joint` | 36 the prior trained on a frozen encoder |
| the three KL settings | 24 the posteriors are points with gaps |
| `joint_tfk8` | 3 the prior does not fit its training geometries |
| `joint_dec`, `joint_ms`, `joint_tfk8_dec`, `joint_kl10_dec` | 4 missed on new geometries inside the training range |
| `noise`, `noise_global`, `noise_coarse` | none: `diagnose_135.sh` skips them (no posterior, no prior); their `pd` / `pb` are `-` |

`python .../r14/brief.py --prior a4` prints the full prior diagnosis of one
setting. `kill_all.sh` stops the watcher too.

## Memory and time

Three jobs per card: a job alone peaks at about 180 GB of the card's 260, so
gradient checkpointing is on for all 24 (x1.52 per step, the gradients
unchanged; `tests/test_noise_mode.py` checks this for noise mode too). Peaks
with checkpointing: about 55 GB (63 GB for noise, 49 GB for the `decoder`
trunk), about 167 GB on each card.

Time: about 28 h of training alone on a card (r8: 277 s per epoch, 360
epochs), x1.52 for checkpointing, each job x2.1 slower while three share the
card (a guess; never measured at three jobs per card), plus inference. The
plan lets a card's jobs speed up as its others finish: **every card ends at
3.9–4.0 days.** Without MPS the three jobs time-slice; if that gives no
overlap at all the factor is 3, not 2.1, and the round takes nearer 5.5 days.
One job per card in waves would take about 3.7 days for 20 jobs; this layout
trades some speed for all 24 finishing together.

The CPU: 24 jobs x (2 torch threads + 4 loader workers) = 144 processes busy
on 128 cores. If epochs are slow and the GPUs idle, lower `num_workers` or
`THREADS_PER_JOB`.

The noise settings are guessed from block counts: the base runs 14 fine-mesh
blocks per step (simulator 8, posterior encoder 3, prior 3); noise runs the
simulator twice (2 x 8 = 16), so x1.14. Not measured. The transformer settings carry 8 flow
draws per sample through the prior only (the trunk runs once), so they should
cost little more; that is not measured.

The trunks are not measured on ex3 either. `decoder` adds one forward-only
pass of the descending arm (4 fine and 6 mid-level blocks, nothing kept for
backward) and drops the prior's 3 fine blocks with their backward: counted as
(14 − 3 + 4/3) / 14 = x0.88. `multiscale` adds 3 blocks at 4096 nodes and 3 at
~256 (about 1% of the fine level's work): counted as x1.0.

## Jobs (generated by make_sweep.py; everything below is rewritten)

24 trainings (12 settings: joint seeds 42/43; joint_kl10 seeds 42/43; joint_kl100 seeds 42/43; joint_dec seeds 42/43; joint_ms seeds 42/43; joint_tfk8 seeds 42/43; joint_tfk8_kl10 seeds 42/43; joint_tfk8_dec seeds 42/43; noise seeds 42/43; noise_global seeds 42/43; joint_kl10_dec seeds 42/43; noise_coarse seeds 42/43; the base is not run: its 42/43/44 = r8 hybrid_e360, r9 hybrid_base_s43, r10 hybrid_base_s44); --pack 3: 3 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 24 slots (slot order 4 5 6 7 1 2 0 3 4 5 6 7 1 2 0 3 4 5 6 7 1 2 0 3), 2 torch threads per job, MPS off. Estimated makespan 4.0 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, each job x2.1 while 3 share a card; cost per job from cost_scale: noise x1.14, prior_trunk decoder x0.88; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_joint_s42 | r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 42 | 4 | on | 55 | 90 | 0.0-3.7 |
| 2 | hybrid_joint_s43 | r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 43 | 5 | on | 55 | 90 | 0.0-3.7 |
| 3 | hybrid_joint_kl10_s42 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 10); seed 42 | 6 | on | 55 | 93 | 0.0-3.9 |
| 4 | hybrid_joint_kl10_s43 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 10); seed 43 | 7 | on | 55 | 93 | 0.0-3.9 |
| 5 | hybrid_joint_kl100_s42 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 100); seed 42 | 1 | on | 55 | 90 | 0.0-3.7 |
| 6 | hybrid_joint_kl100_s43 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 100); seed 43 | 2 | on | 55 | 90 | 0.0-3.7 |
| 7 | hybrid_joint_ms_s42 | joint, the prior's own trunk goes down to the coarsest level (prior_trunk multiscale); seed 42 | 0 | on | 55 | 90 | 0.0-3.7 |
| 8 | hybrid_joint_ms_s43 | joint, the prior's own trunk goes down to the coarsest level (prior_trunk multiscale); seed 43 | 3 | on | 55 | 90 | 0.0-3.7 |
| 9 | hybrid_noise_s42 | stochastic_mode noise: no encoder, no prior; z_g and z_l ~ N(0,I), 2 draws per batch, loss = their CRPS; seed 42 | 4 | on | 63 | 96 | 0.0-4.0 |
| 10 | hybrid_noise_s43 | stochastic_mode noise: no encoder, no prior; z_g and z_l ~ N(0,I), 2 draws per batch, loss = their CRPS; seed 43 | 5 | on | 63 | 96 | 0.0-4.0 |
| 11 | hybrid_joint_tfk8_s42 | joint with r9's tf_k8 prior: transformer, 8 flow draws per sample, z_l injected at the coarsest level only; seed 42 | 6 | on | 55 | 93 | 0.0-3.9 |
| 12 | hybrid_joint_tfk8_s43 | joint with r9's tf_k8 prior: transformer, 8 flow draws per sample, z_l injected at the coarsest level only; seed 43 | 7 | on | 55 | 93 | 0.0-3.9 |
| 13 | noise_global_s42 | stochastic_mode noise with latent_type global: one N(0,I) vector per sample (no z_l), 2 draws per batch, loss = their CRPS; seed 42 | 1 | on | 63 | 96 | 0.0-4.0 |
| 14 | noise_global_s43 | stochastic_mode noise with latent_type global: one N(0,I) vector per sample (no z_l), 2 draws per batch, loss = their CRPS; seed 43 | 2 | on | 63 | 96 | 0.0-4.0 |
| 15 | hybrid_noise_coarse_s42 | stochastic_mode noise, z_l read at the coarsest level only (local_inject coarsest), 2 draws per batch, loss = their CRPS; seed 42 | 0 | on | 63 | 96 | 0.0-4.0 |
| 16 | hybrid_noise_coarse_s43 | stochastic_mode noise, z_l read at the coarsest level only (local_inject coarsest), 2 draws per batch, loss = their CRPS; seed 43 | 3 | on | 63 | 96 | 0.0-4.0 |
| 17 | hybrid_joint_dec_s42 | joint, the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder; no trunk of its own); seed 42 | 4 | on | 49 | 82 | 0.0-3.4 |
| 18 | hybrid_joint_dec_s43 | joint, the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder; no trunk of its own); seed 43 | 5 | on | 49 | 82 | 0.0-3.4 |
| 19 | hybrid_joint_tfk8_kl10_s42 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 10); seed 42 | 6 | on | 55 | 93 | 0.0-3.9 |
| 20 | hybrid_joint_tfk8_kl10_s43 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 10); seed 43 | 7 | on | 55 | 93 | 0.0-3.9 |
| 21 | hybrid_joint_tfk8_dec_s42 | joint, r9's tf_k8 prior, the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder; no trunk of its own); seed 42 | 1 | on | 49 | 82 | 0.0-3.4 |
| 22 | hybrid_joint_tfk8_dec_s43 | joint, r9's tf_k8 prior, the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder; no trunk of its own); seed 43 | 2 | on | 49 | 82 | 0.0-3.4 |
| 23 | hybrid_joint_kl10_dec_s42 | joint, per-sample KL (kl_weight 10) and the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder; no trunk of its own); seed 42 | 0 | on | 49 | 82 | 0.0-3.4 |
| 24 | hybrid_joint_kl10_dec_s43 | joint, per-sample KL (kl_weight 10) and the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder; no trunk of its own); seed 43 | 3 | on | 49 | 82 | 0.0-3.4 |
