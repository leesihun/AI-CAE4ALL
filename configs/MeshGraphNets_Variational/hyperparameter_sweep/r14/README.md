# B300 round 14: the prior (GNN or transformer) x the latent regularizer (MMD or KL 1, 10, 100)

Model: HI-MGN-hybrid (`meshgraphnets-v`, `latent_type hybrid`) on ex3 shell
buckling. Node 135, sixteen jobs, **one per card on all eight cards** (eight at
once, the other eight as cards free), MPS off.

## What it asks

Round 13 ran twelve settings on round 12's `joint` setting. Two things moved:

- `joint_kl10` (the per-sample KL at weight 10 in place of the MMD) put the
  latent in order (a decoded N(0,I) draw: 551 → 211) and had both seeds low on
  the held-out geometries (69/72; the base's seeds: 66/95/75).
- Before that, round 9's `tf_k8` (a transformer prior trained on 8 flow draws
  per sample) was the only prior that fitted the training geometries better
  without a warm-up (tr 72/70). It was never run on `joint`.

So this round crosses the two, with the KL weight a factor 10 apart:

| | MMD | KL 1 | KL 10 | KL 100 |
| --- | --- | --- | --- | --- |
| GNN prior | `joint` | `joint_kl1` | `joint_kl10` | `joint_kl100` |
| transformer prior | `joint_tfk8` | `joint_tfk8_kl1` | `joint_tfk8_kl10` | `joint_tfk8_kl100` |

Eight settings x seeds 42/43. Every run is `joint`: `prior_freeze_epoch 0` (one
objective for all 360 epochs) and `prior_grad_to_encoder 1` (the prior's loss
reaches the encoder).

Three questions:

1. Does the KL help along a line (1 → 10 → 100), or was 69/72 at weight 10 two
   lucky seeds? `joint_kl1` and `joint_kl10` repeat round 13's runs.
2. Does a prior that sees the whole coarse graph hold better on unseen
   geometries, on `joint`?
3. Do the two add up (`joint_tfk8_kl10`)?

## The settings

| setting | change from `joint` | seeds |
| --- | --- | --- |
| base (not run here) | two parts: `prior_freeze_epoch 240`, `prior_grad_to_encoder 0` | 42, 43, 44 |
| `joint` | none (the control; it died in round 13) | 42, 43 |
| `joint_kl1` | `latent_reg kl`, `kl_weight 1` (no MMD) | 42, 43 |
| `joint_kl10` | `latent_reg kl`, `kl_weight 10` | 42, 43 |
| `joint_kl100` | `latent_reg kl`, `kl_weight 100` | 42, 43 |
| `joint_tfk8` | `prior_arch transformer`, `prior_fm_draws 8`, `local_inject coarsest`, `lambda_mmd_g 0` | 42, 43 |
| `joint_tfk8_kl1` | `joint_tfk8` + `latent_reg kl`, `kl_weight 1` | 42, 43 |
| `joint_tfk8_kl10` | `joint_tfk8` + `kl_weight 10` | 42, 43 |
| `joint_tfk8_kl100` | `joint_tfk8` + `kl_weight 100` | 42, 43 |

**The transformer row is not the transformer alone.** `joint_tfk8` is round 9's
recipe: four keys change together (the network, 8 flow draws per sample, z_l
read by the decoder at the coarsest level only, no MMD on z_g). If the row
wins, a later round separates them.

With `latent_reg kl` the MMD keys (`lambda_mmd`, `lambda_mmd_g`,
`mmd_bandwidth`) are not in the config.

## The two priors

Both are `CoarseFMPrior` (`model/conditional_prior.py`) and share everything
but the velocity network: the trunk (message passing on the fine mesh, its node
features mean-pooled into the ~256 coarsest clusters), the flow-matching loss,
the sampler (`prior_fm_steps 30`).

- **GNN** (`prior_arch gnn`, the default): `prior_fm_blocks` (4) message-passing
  blocks on the coarse graph's edges. A coarse node sees its 4-hop
  neighbourhood; z_g talks to the nodes through one per-graph token that reads
  the node mean before every block.
- **Transformer** (`prior_arch transformer`): one token per coarse node (its
  noisy z_l, its pooled trunk feature, a code of its position) plus one token
  per z_g slot; 4 blocks of self-attention over all tokens of a graph (4
  heads, width 128). Every coarse node sees every other one from the first
  block; the coarse edges are not used. The flow time and a pooled summary of
  the geometry enter every block as one condition (a shift, scale and gate of
  its normalization). At initialization the velocity is exactly 0.

`prior_fm_draws 8`: each posterior sample is regressed from 8 independent
(noise, time) pairs per step. The trunk runs once, so the flow gets 8 times the
supervision at little cost.

## What is already known against it

- **KL 100**: round 8's two-part `kl100` scored 99 (the mean with no spread).
  The expected failure is a decoder that stops reading z.
- **KL 1**: round 13's `joint_kl1` was 106/84, worse than the base.
- **KL 10**: its prior on the training geometries was 111 and its posterior
  sample on the held-out geometries 41 (floor 27): the latent is in order, the
  decoder pays for it. 69/72 against the base's 66/95/75 is inside the seed
  range.
- **tf_k8**: round 9's numbers (tr 72/70, ho 80/71) are from the two-part
  setting. On the held-out geometries it did not beat the base there.
- Round 13's `joint` and `joint_pms05` died (both seeds each, cards 0 and 3,
  two jobs per card under MPS). `joint` is rerun here; `joint_pms05` is not.

## Where each can fail, and where it shows

- **KL** (a2–a4, a6–a8): the posterior is wider by design, so the second number
  sits above the floor. Too strong: the second number rises towards 100 and
  `l` falls towards 0 (the decoder ignores z); the prior then scores near 100
  on every set.
- **Transformer** (a5–a8): it can fit the training geometries by position and
  not hold on unseen ones: the prior on `tr` falls and on `ho` it does not
  (MIXED).

On a 12 x 10 toy grid (CPU) every setting trains, validates and samples. That
says the code paths work; it says nothing about ex3.

## Before starting

Pull first. `run_node.sh` stops with "Pull first." if the checkout lacks round
13's keys. Round 13 and round 14 cannot run on 135 together (the same cards).

## Running it (node 135, cards 0–7, one job per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/run_135.sh

`run_135.sh` sets `MPS=0` itself (not an override: one job per card has nothing
to share, and round 13 lost two settings under MPS). Slot order: cards
4 5 6 7 1 2 0 3. Cards 0 and 3, where round 13's jobs died, are last.

| wave | cards 4, 5, 6, 7 | cards 1, 2, 0, 3 |
| --- | --- | --- |
| 1 (0–30 h) | `joint` 42, `joint_tfk8_kl10` 42, `joint` 43, `joint_tfk8_kl10` 43 | `joint_tfk8` 42, `joint_kl100` 42, `joint_tfk8` 43, `joint_kl100` 43 |
| 2 (30–60 h) | `joint_tfk8_kl100` 42, `joint_tfk8_kl1` 42, `joint_kl10` 42, `joint_kl1` 42 | `joint_tfk8_kl100` 43, `joint_tfk8_kl1` 43, `joint_kl10` 43, `joint_kl1` 43 |

A setting's two seeds are never on one card. Wave 1 holds what is not known
yet; with round 13's `joint_kl1` and `joint_kl10` the GNN line is complete
after it. Wave 2's cards are taken as they free, so the table's wave 2 is the
order, not a fixed card.

At any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **a0–a8**: base, `joint`, `joint_kl1`, `joint_kl10`, `joint_kl100`,
  `joint_tfk8`, `joint_tfk8_kl1`, `joint_tfk8_kl10`, `joint_tfk8_kl100`.
  Read a1–a4 as one line (the GNN prior: MMD, KL 1, 10, 100) and a5–a8 as the
  same line with the transformer prior.
- **Mark**: `+` every seed beats every seed of the other setting, `-` every
  seed loses, `=` neither. a1 has one mark, against the base. a2–a5 have two:
  against the base, then against a1. a6–a8 have their second mark against a5.
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
| the six KL settings | 24 the posteriors are points with gaps |
| `joint_tfk8` | 3 the prior does not fit its training geometries |

`python .../r14/brief.py --prior a3` prints the full prior diagnosis of one
setting. `kill_all.sh` stops the watcher too.

## Memory and time

One job per card: checkpointing is off (a job peaks at about 180 GB of the
card's 260). About 28 h of training (r8 measured 277 s per epoch alone on a
card, 360 epochs) plus about 2 h of inference: about 30 h per job, two waves,
**about 60 h (2.5 days)** for the round. The transformer settings carry 8 flow
draws per sample through the prior only (the trunk runs once), so they should
cost little more; that is not measured.

## Jobs (generated by make_sweep.py; everything below is rewritten)

16 trainings (8 settings: joint seeds 42/43; joint_kl1 seeds 42/43; joint_kl10 seeds 42/43; joint_kl100 seeds 42/43; joint_tfk8 seeds 42/43; joint_tfk8_kl1 seeds 42/43; joint_tfk8_kl10 seeds 42/43; joint_tfk8_kl100 seeds 42/43; the base is not run: its 42/43/44 = r8 hybrid_e360, r9 hybrid_base_s43, r10 hybrid_base_s44); --pack 1: 1 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 8 slots (slot order 4 5 6 7 1 2 0 3), 11 torch threads per job, MPS off. Estimated makespan 2.5 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, x1.0 per job at 1 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_joint_s42 | r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 42 | 4 | off | 180 | 30 | 0.0-1.2 |
| 2 | hybrid_joint_tfk8_kl10_s42 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 10); seed 42 | 5 | off | 180 | 30 | 0.0-1.2 |
| 3 | hybrid_joint_s43 | r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 43 | 6 | off | 180 | 30 | 0.0-1.2 |
| 4 | hybrid_joint_tfk8_kl10_s43 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 10); seed 43 | 7 | off | 180 | 30 | 0.0-1.2 |
| 5 | hybrid_joint_tfk8_s42 | joint with r9's tf_k8 prior: transformer, 8 flow draws per sample, z_l injected at the coarsest level only; seed 42 | 1 | off | 180 | 30 | 0.0-1.2 |
| 6 | hybrid_joint_kl100_s42 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 100); seed 42 | 2 | off | 180 | 30 | 0.0-1.2 |
| 7 | hybrid_joint_tfk8_s43 | joint with r9's tf_k8 prior: transformer, 8 flow draws per sample, z_l injected at the coarsest level only; seed 43 | 0 | off | 180 | 30 | 0.0-1.2 |
| 8 | hybrid_joint_kl100_s43 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 100); seed 43 | 3 | off | 180 | 30 | 0.0-1.2 |
| 9 | hybrid_joint_tfk8_kl100_s42 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 100); seed 42 | 4 | off | 180 | 30 | 1.2-2.5 |
| 10 | hybrid_joint_tfk8_kl1_s42 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 1); seed 42 | 5 | off | 180 | 30 | 1.2-2.5 |
| 11 | hybrid_joint_kl10_s42 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 10); seed 42 | 6 | off | 180 | 30 | 1.2-2.5 |
| 12 | hybrid_joint_kl1_s42 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 1); seed 42 | 7 | off | 180 | 30 | 1.2-2.5 |
| 13 | hybrid_joint_tfk8_kl100_s43 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 100); seed 43 | 1 | off | 180 | 30 | 1.2-2.5 |
| 14 | hybrid_joint_tfk8_kl1_s43 | joint, r9's tf_k8 prior, per-sample KL (kl_weight 1); seed 43 | 2 | off | 180 | 30 | 1.2-2.5 |
| 15 | hybrid_joint_kl10_s43 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 10); seed 43 | 0 | off | 180 | 30 | 1.2-2.5 |
| 16 | hybrid_joint_kl1_s43 | joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, kl_weight 1); seed 43 | 3 | off | 180 | 30 | 1.2-2.5 |
