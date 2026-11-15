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
  `use_compile` and the hierarchy cache work as in `../r6/README.md`.

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
