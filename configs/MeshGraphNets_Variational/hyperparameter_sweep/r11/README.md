# B300 round 11: where the latent lives, and a z_g the decoder reads

Sixteen runs on node 135 (two per card), three questions:

1. **Placement.** Global, local and hybrid at today's recipe (batch 16, 360
   epochs). Round 6 compared them at batch 4 and 120 epochs.
2. **A living z_g.** Can the decoder be made to read z_g, and does the prior
   get better when it does?
3. **The best result.** Two additions to the best setting so far (`tfk8`).

## What rounds 8–10 established

- **The whole error is the prior's.** The posterior sample sits at the floor
  on held-out geometries (29 against 27); the prior sample is at 76–80.
- **The prior is wrong on training geometries too** (71–82 against a floor of
  56), so it is not only generalization. The GNN prior under-fits; the
  transformer prior fits training geometries and over-fits.
- **z_g is dead in every arm** (`g0 l100`). Its posterior σ is about 9× the
  spread of its mean, so the decoder learned to ignore it, and z_l (256 nodes ×
  4 channels) carries the whole draw. The hybrid has in effect been a local
  model.
- **Not known: why hybrid led local in round 6** (88/88/107 against
  108/108/108). An earlier draft of this file put it down to the hybrid
  prior's per-graph token. Round 8's anatomy contradicts that: cutting the
  token's write-back left the prior's score unchanged. The two ranges touch,
  so it may be seed noise. The `local` arms measure it again.
- **Against a living z_g:** `latent_type global` was the worst placement in
  round 6 (144–190) and got worse with more training. Nobody measured whether
  its posterior or its prior failed. The `global` arm does.

Round 10, as w2 per seed and the diagnosis (`ho`/`tr` = floor / posterior
sample / prior):

| r10 arm | w2 42/43/44 | m | ho | tr | g / l |
| --- | --- | --- | --- | --- | --- |
| base | 66/95/75 | 78 | 27/29/78 | 56/57/82 | 0 / 100 |
| gnn_k8 | 62/94/81 | 79 | 27/29/80 | 56/56/72 | 0 / 100 |
| **tfk8_r9** (r9's gfix_tf_k8) | **78/66** | **72** | 27/33/**76** | 56/56/**71** | 0 / 101 |
| tfd | 90/102/128 | 107 | 27/33/108 | 56/56/84 | 0 / 100 |
| tfd_cn | 107/80/82 | 90 | 27/35/94 | 56/56/73 | 0 / 100 |
| tfd_d2 | 91/105/126 | 107 | 27/34/111 | 56/56/78 | 0 / 100 |
| tfd_p3 | 88/95/106 | 96 | 27/35/96 | 56/56/96 | 0 / 100 |

## The arms

| arm | change | seeds | what it asks |
| --- | --- | --- | --- |
| `base` | none, not run (r8 `hybrid_e360`, r9 `hybrid_base_s43`, r10 `hybrid_base_s44`) | 42, 43, 44 | the reference |
| `global` | `latent_type global` | 42 | where global fails: posterior or prior |
| `local` | `latent_type local` | 42, 43 | is the hybrid with a dead z_g anything but local? |
| `local_tfk8` | `local` with `tfk8`'s prior | 42, 43 | the same, under the best prior (against `tfk8`) |
| `wu` | `local_warmup_epochs 60` | 42, 43 | z_l withheld for 60 epochs: does z_g stay in use after? |
| `wu_a1` | `wu` + `zg_anchor_weight 1` | 42 | does z_g need an anchor to stay (against `wu`) |
| `ld50` | `local_dropout 0.5` | 42, 43 | z_l withheld for half the graphs, all training |
| `cap05` | `posterior_max_std_g 0.05` | 42, 43 | z_g with no noise from the first step |
| (`tfk8`) | r9's `gfix_tf_k8`, not run | 42, 43 | the best setting so far |
| `tfk8_cn` | `tfk8` + `prior_cond_noise 0.1` | 42, 43 | condition noise against the transformer's over-fit |
| `tfk8_wu` | `tfk8` + `local_warmup_epochs 60` | 42, 43 | a living z_g under the best prior |

The three z_g keys, all default 0 (= the behaviour up to round 10):

- **`local_warmup_epochs N`** zeroes z_l for every sample in the first N
  epochs, so only z_g can carry the draw. At epoch N the best/last checkpoint
  pick is reset and a snapshot of z_g's posterior is taken. 60 is well before
  the prior freeze (epoch 240).
- **`zg_anchor_weight w`** adds, after the warmup, w × KL(q(z_g | x) ‖ that
  snapshot) per sample. The recipe it comes from (TS-DSAE) needs it against a
  KL term that pushes z_g back to noise. Under MMD nothing does, so `wu` may
  hold on its own. One seed is enough here: the question is `g`, not w2.
- **`posterior_max_std_g`** caps the σ of z_g's posterior.

Things to know before reading the result:

- **`tfk8_wu` has no regularizer on z_g** (`lambda_mmd_g 0`, part of `tfk8`)
  while z_g carries the draw. Only the σ floor acts on it. The prior
  standardizes its z_g target at the freeze, not before.
- **`global` has one seed.** Its answer is the diagnosis (posterior against
  prior), which a single seed shows if global is as far off as in round 6.
- **`wu` and `ld50` can revive z_g and still not help.** z_g then holds
  whatever 32 numbers can; it may be the geometry rather than the draw.

## Reading the result

Seed noise is large (the base: 66/95/75). An arm counts as better only if
every seed beats every seed of the arm it is compared with; a difference under
about 15 says nothing. Each `aN` line carries two marks, the first against the
base, the second against the arm it extends (`wu_a1` against `wu`; `tfk8_cn`,
`tfk8_wu` and `local_tfk8` against `tfk8`).

| question | read | meaning |
| --- | --- | --- |
| Is hybrid more than local? | `local` against the base, `local_tfk8` against `tfk8` | equal: the hybrid as trained is a local model. Local better: z_g costs something |
| Where does global fail? | `global`'s `ho` middle number | far above 27: 32 numbers cannot carry the draw. Near 27 with a bad prior: the prior fails even in 32 numbers, so moving the draw into z_g will not rescue it |
| Did z_g come alive? | `g` on `wu`, `wu_a1`, `ld50`, `cap05`, `tfk8_wu` | 0: the method does not work. Above 0: read the next row |
| Does a living z_g help? | w2 and the prior (`ho`/`tr` third number) of those arms | better than the base: keep the hybrid, with that key. Not better: the placement to keep is `local` |
| Best setting | lowest `m` among arms whose seeds all beat `tfk8`'s 78/66 | the setting for the final runs |

Everything here is generated. `make_sweep.py` writes `config_train_*`,
`config_infer_*`, `jobs.txt`, `run_135.sh` and the job table at the end of this
file. `--plan` prints the plan and writes nothing. Edit `make_sweep.py`, never a
generated file. Its self-check confirms that the reference runs (R1–R5) have
exactly the configs of the settings they stand for, apart from seeds and paths.

## Before starting

Pull first. The three z_g keys default to 0, so a resuming job of an older
round runs exactly as before. `run_node.sh` stops with "Pull first." if the
checkout lacks round 11's code. `local_warmup_epochs` is single-GPU only (the
DDP trainer rejects it); every job here is one GPU.

## Running it (node 135, all 8 cards, 2 jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/run_135.sh

The same runner as rounds 6–10. The checks run in the terminal; when they pass,
all 16 jobs start at once, job k on card k mod 8. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

## Where the error comes from (automatic)

`run_135.sh` starts `diagnose_135.sh` as in round 10, for this round's jobs
only. R1–R5 were diagnosed by the round 9 and 10 watchers, and `brief.py`
reads them from `output/b300_r9/` and `output/b300_r10/`. Each arm line reads:

    a4 ++ 60/62 m61 ho 27/33/70 tr 56/56/72 g30 l70

- **Marks**, as above. `global` and `wu_a1` have one seed, so no marks.
- **w2** per seed, then **m**, their mean.
- **ho / tr**: floor / posterior sample / prior on held-out and training
  geometries.
- **g / l**: how much the decoded draw's correlation with its FE result drops
  when z_g, or z_l, is swapped with another draw's. `-` on the global and
  local arms, which have one latent and nothing to swap.

Progress is in `tail -f output/b300_r11/_diag/diag.out`. `kill_all.sh` stops
the watcher too.

- **Memory and time** as in round 10: two jobs per card with gradient
  checkpointing, about 55 GB peak and 65 h per job.
- **If two per card turns out slower,** run `kill_all.sh`, then
  `make_sweep.py --pack 1 --force`, and start the round again. Every job
  resumes.
- **Shared tools.** Restarts, overrides, `stop_135`, `kill_all.sh`, MPS,
  `use_compile` and the hierarchy cache work as in [the sweep index](../README.md#shared-runner-tools).

## Unseen geometries inside the range, and the seeds (after the round)

The round scores the prior on 8 held-out geometries outside the training
ranges (`ho`) and on 16 training geometries (`tr`). It does not say how the
prior does on a new geometry *inside* the ranges, or why two seeds of one
setting differ by 30. One more pass answers both, with no training:

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/unseen_135.sh

It runs the same diagnosis on the test split of the training file (`ts`: 15
geometries x 64 FE draws, never trained on) for this round's 16 jobs and
R1–R5, one run per free card, minutes each. Progress is in
`tail -f output/b300_r11/_diag/unseen.out`; `kill_all.sh` stops it. Then:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/unseen_brief.py

and type back the block. Each arm line reads:

    a0 tr 82/85/80 s30 ts 48 99/104/97 p90 s40 ho 78/95/75 p70 s55

- **tr / ts / ho**: the prior per seed on each set; on `ts` the posterior
  sample comes first. The floors are on the `fl` line.
- **p**: the seeds pooled, each draw from a model picked at random, the same
  number of draws as one seed. It cannot be above the seeds' RMS; how far it
  falls below is the error the seeds do not share.
- **s**: one seed's prior draws against another's, over the FE sd. Near the
  floor, the seeds draw the same distribution.
- **bd**: how much one checkpoint's score moves with the draws alone.
- **T / G**: the `ts` and `ho` geometries one by one, for the base and tfk8.

| the block shows | it means |
| --- | --- |
| `ts` prior as bad as `ho` | the prior fails on any new geometry: 120 training geometries are too few |
| `ts` prior near its floor | the remaining error is extrapolation outside the ranges only |
| `p` far below the seeds, `s` large | the seeds err in different directions: averaging over models removes that part |
| `p` about the seeds, `s` small | the seeds share one error: more seeds or an ensemble do not help |
| `s` grows from `tr` to `ts` to `ho` | the farther from the training geometries, the less the data pins the prior down |

One histogram per geometry (FE results and each seed's prior draws) is written
to `output/b300_r11/_diag/unseen_<arm>_<set>.png`.

### Why the seeds differ

The totals are RMS values over few geometries, each over its own FE sd, so one
geometry, or a few draws in it, can carry a total. A second block says which
(no GPU, seconds, for the base's three seeds on `ho`):

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/unseen_brief.py --why

Above the block it prints one plain sentence per line, so the reading does not
have to be done by hand. The block is 5 lines of whole numbers with spaces
between them; the last number of every line is a check digit (the last digit of
the sum of the line's other numbers). `--why --long` adds `ts` and tfk8 (15
lines, not for typing).

    F 16 21 7                       floors: ho trimmed, ho one scale
    T a0 ho 62 66 64 70 95 74 1     per seed: trimmed score, then one-scale score
    D a0 ho 5 160 49 6 9 104 3      geometry, worst seed, best seed, h, m, f
    B a0 ho 7 104 88 97 18 4        geometry, best seed, h, m, f

- **trimmed**: the usual score without the 10% of the draws that are farthest
  off.
- **one scale**: W2 in the field's units over the RMS of the geometries' FE sd,
  so a geometry whose FE barely scatters cannot dominate.
- **D**: a geometry on which the seeds disagree most. **B**: the geometry that
  is bad in every seed.
- **h**: how many % of the draws carry 90% of the error. **m**: % of the error
  that is the mean. **f**: the geometry's FE sd as % of the median geometry's.

| the block shows | it means |
| --- | --- |
| seeds close on the trimmed score, `h` about 10 or less | the seeds differ in a few draws that land somewhere else (the other regime); the rest of the distribution agrees |
| seeds close on the one-scale score, `f` far below 100 | the seeds differ on geometries where FE barely scatters: the division by a small FE sd makes the gap |
| neither closes it, `h` about 90, `m` large | a seed puts the whole cloud in the wrong place: the prior is not pinned down there |
| a `B` line with `f` far below 100 | that geometry's score is inflated in every seed; it is not a failure of one seed |

## What is wrong with the prior (after the round)

The encoder and decoder are near the floor; the prior is not. One more pass
says where the prior's gap comes from and how much each part is, with no
training and no reading by hand:

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/prior_135.sh

It runs `misc/prior_diagnosis.py` on R1–R5 and this round's local / hybrid
jobs (not the global run: another prior class), up to four parts per run: `ho`
(8 held-out geometries x 256 draws), `tr` (16 training geometries), `ts`
(the 15 test geometries) and `jt` (`ho` again with `<checkpoint>.joint.pth`,
the best checkpoint of the epochs before the prior freeze; left out when a run
has no such file). One run per free card, the base and tfk8 first;
`PRIOR_ONLY="base tfk8"` runs those alone. Progress is in
`tail -f output/b300_r11/_diag/prior.out`; `kill_all.sh` stops it. Then:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/prior_brief.py

prints, in English, the verdict of the base in sentences, a `CAUSES` table
that marks every candidate cause `YES` (found), `no` (ruled out by a number) or
`?` (this run cannot tell, with the reason), the premise and conclusion of
every other setting, and the block to type back (`ho` `tr` `ts` `ve` `in` `ex`
`cs`: whole numbers, the last of each line a check digit; `cs` is the numbers
of the causes found, `u`, then the numbers that stay open). `--long` prints
every setting's full verdict, `--all` adds one typed line per other setting.

The reference is exact. A geometry's FE draws give n posteriors; their mixture
is what the prior should draw, and its flow-matching velocity has a closed
form. Everything is compared with that:

| line of the verdict | the question | how it is measured |
| --- | --- | --- |
| `premise` | is it the prior at all | floor, posterior sample (`P`), prior (`Q`) |
| `check` | can the method be trusted | the exact flow through the model's sampler (`O`) must score as `P` |
| `where` | training fit, new geometries or extrapolation | the gap `Q - P` on `tr`, `ts`, `ho` |
| `what` | mean, width, shape or joint structure | the prior's draws corrected one step at a time (`Qc`, `Qcs`, `Qm`) |
| `when` | the early (which regime) or the late part of the flow | exact flow before t = 0.5, the model after (`R5`); the model until t, exact after (`c10`..`c90`) |
| `onto` | off the posterior's support, or on it in the wrong proportion | the draws read off at the nearest posterior component, no decoding (`S`) |
| `latent` | z_g, z_l, or how the two are paired | only one of them from the prior (`Qg`, `Ql`) against two unrelated posterior samples (`Px`); two unrelated prior draws (`Qx`) |
| `sampler` | too few steps, temperature | the prior at 4x the steps (`Qf`) |
| `field` | underfit or memorised, and is the geometry used | velocity error over signal per time, `tr` against `ts`/`ho`, beside a field that ignores the geometry and the best per-geometry Gaussian |
| `input` | is the geometry's input outside the training set | distance of the input embedding from the training geometries, and its correlation with the gap |
| `output` | does the prior's cloud follow the geometry | R2 of its centre against the posterior's, `z_l` and `z_g` apart |
| `target` | is the target itself learnable | one posterior's width over the spacing of the draws; size after standardization |
| `calib` | tails and mode shares | PIT below 5% / above 95%, share of the upper mode |
| `weight` | does one geometry carry the total | share of the squared sum, the score without it |
| `train` | was the kept epoch too early or past the best | the epoch log of the training config beside the inference config: where the kept epoch lies, validation CRPS still falling or turned up |
| `EMA` | do the EMA weights cost the prior | the checkpoint's trained weights run as well (`Pr`, `Qr`): `(Q - P) - (Qr - Pr)`; posterior means of the two encoders |
| `tail` | did the prior-only epochs help | the gap at the joint-phase checkpoint (`jt`) against the gap at the kept one |

The shares in % split one gap in a fixed order; lines cut the same gap in
different directions and must not be added. One thing stays `?` by design:
what training the prior and the encoder together would give instead of a prior
fitted to a frozen posterior. No single checkpoint holds that; r12 is that run.

## Jobs (generated by make_sweep.py; everything below is rewritten)

16 trainings (9 arms; seeds 42, 43, global and wu_a1 42 only; the base is not run: its 42/43/44 = r8 hybrid_e360, r9 hybrid_base_s43, r10 hybrid_base_s44); --pack 2: 2 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 16 slots, 3 torch threads per job. Estimated makespan 2.7 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, x1.5 per job at 2 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | global_s42 | latent_type global: one z of vae_latent_dim 32 per graph, no z_l; seed 42 | 0 | on | 55 | 65 | 0.0-2.7 |
| 2 | local_s42 | latent_type local: z_l only, the base's GNN prior; seed 42 | 1 | on | 55 | 65 | 0.0-2.7 |
| 3 | local_tfk8_s42 | latent_type local with tfk8's prior (local_inject coarsest, transformer prior, prior_fm_draws 8); seed 42 | 2 | on | 55 | 65 | 0.0-2.7 |
| 4 | hybrid_wu_s42 | z_l withheld for every graph in the first 60 epochs (local_warmup_epochs 60), no anchor; seed 42 | 3 | on | 55 | 65 | 0.0-2.7 |
| 5 | hybrid_ld50_s42 | z_l withheld for half the graphs of every batch (local_dropout 0.5); seed 42 | 4 | on | 55 | 65 | 0.0-2.7 |
| 6 | hybrid_cap05_s42 | z_g's posterior sigma capped at the floor, 0.05 (posterior_max_std_g 0.05); seed 42 | 5 | on | 55 | 65 | 0.0-2.7 |
| 7 | hybrid_tfk8_cn_s42 | r9's gfix_tf_k8 (gfix + transformer prior + prior_fm_draws 8) + prior_cond_noise 0.1, no dropout; seed 42 | 6 | on | 55 | 65 | 0.0-2.7 |
| 8 | hybrid_tfk8_wu_s42 | r9's gfix_tf_k8 + z_l withheld in the first 60 epochs (local_warmup_epochs 60); seed 42 | 7 | on | 55 | 65 | 0.0-2.7 |
| 9 | local_s43 | latent_type local: z_l only, the base's GNN prior; seed 43 | 0 | on | 55 | 65 | 0.0-2.7 |
| 10 | local_tfk8_s43 | latent_type local with tfk8's prior (local_inject coarsest, transformer prior, prior_fm_draws 8); seed 43 | 1 | on | 55 | 65 | 0.0-2.7 |
| 11 | hybrid_wu_s43 | z_l withheld for every graph in the first 60 epochs (local_warmup_epochs 60), no anchor; seed 43 | 2 | on | 55 | 65 | 0.0-2.7 |
| 12 | hybrid_ld50_s43 | z_l withheld for half the graphs of every batch (local_dropout 0.5); seed 43 | 3 | on | 55 | 65 | 0.0-2.7 |
| 13 | hybrid_cap05_s43 | z_g's posterior sigma capped at the floor, 0.05 (posterior_max_std_g 0.05); seed 43 | 4 | on | 55 | 65 | 0.0-2.7 |
| 14 | hybrid_tfk8_cn_s43 | r9's gfix_tf_k8 (gfix + transformer prior + prior_fm_draws 8) + prior_cond_noise 0.1, no dropout; seed 43 | 5 | on | 55 | 65 | 0.0-2.7 |
| 15 | hybrid_tfk8_wu_s43 | r9's gfix_tf_k8 + z_l withheld in the first 60 epochs (local_warmup_epochs 60); seed 43 | 6 | on | 55 | 65 | 0.0-2.7 |
| 16 | hybrid_wu_a1_s42 | wu + each sample's z_g anchored to its epoch-60 posterior (zg_anchor_weight 1); seed 42 | 7 | on | 55 | 65 | 0.0-2.7 |
