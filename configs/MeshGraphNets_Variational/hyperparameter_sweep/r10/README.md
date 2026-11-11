# B300 round 10: the HI-MGN-hybrid prior against over-fitting, three seeds per arm

Goal: keep what round 9 found to work, the transformer prior with 8 FM draws
(`gfix_tf_k8`), and stop it over-fitting the 120 training geometries. Round 9's
results, as w2 per seed and the diagnosis
(`ho`/`tr` = floor / posterior sample / prior):

| r9 arm | w2 42/43 | ho | tr |
| --- | --- | --- | --- |
| base | 66/95 | 27/29/79 | 56/56/81 |
| gfix | 94/77 | 27/34/92 | 56/56/101 |
| tf | 95/119 | 27/30/105 | 56/55/65 |
| gfix_tf_x | 88/101 | 27/33/97 | 56/56/84 |
| **gfix_tf_k8** | **78/66** | 27/33/**76** | 56/56/**71** |
| gfix_tf_ch2 | 103/115 | 27/32/108 | 56/56/72 |
| gfix_tf_x_k8 | 96/114 | 27/35/108 | 56/56/69 |

- **The posterior sample sits at the floor** in every arm (ho 27 → 29–35). The
  encoder and decoder are fine; the whole gap is the prior's.
- **The GNN prior under-fits but generalizes.** On training geometries its
  sample scores 81–101 against the posterior sample's 56, and on held-out
  geometries it does about as well.
- **The transformer prior fits but over-fits.** On training geometries it
  reaches 65–72, but on held-out ones 105–108. Only `gfix_tf_k8` kept the gap
  small (tr 71, ho 76), and it was the best arm.
- **z_g stays dead** (`g` ≈ 0) even with `gfix`. `prior_param x` and
  `latent_ch 2` made things worse.
- **The seed spread is large.** The base alone gave 66 and 95, so every arm here
  runs three seeds.

## The arms

Every transformer arm trains with dropout, set by the new key
**`prior_tf_dropout`**. It applies to the attention weights and to both
residual branches (attention output and MLP) of each block, and only in
training: eval, inference and the diagnosis are unchanged. Its default is 0,
which is exactly the behaviour up to round 9. It adds no parameter, so any
checkpoint loads with any value.

| arm | change from the base | what it asks |
| --- | --- | --- |
| `base` | none; seed 44 only (42 = r8 `hybrid_e360`, 43 = r9 `hybrid_base_s43`) | the base's seed noise, now from three seeds |
| `gnn_k8` | `prior_fm_draws 8` | does k8 also help the under-fitting GNN prior? |
| (`tfk8_r9`) | r9's `gfix_tf_k8`, seeds 42 and 43, not run here | the same setting at dropout 0 |
| `tfd` | `lambda_mmd_g 0`, `local_inject coarsest`, `prior_arch transformer`, `prior_fm_draws 8`, `prior_tf_dropout 0.1` | r9's best setting with dropout 0.1. Against `tfk8_r9`, this is the dropout's effect |
| `tfd_p3` | `tfd` with `prior_tf_dropout 0.3` | is 0.1 too weak for 120 geometries? |
| `tfd_cn` | `tfd` + `prior_cond_noise 0.1` | does noise on what the prior reads stop it memorizing geometries? |
| `tfd_d2` | `tfd` + `prior_fm_blocks 2` | is a 4-block transformer more capacity than 120 geometries support? |

- **Dropout dose.** `tfk8_r9` → `tfd` → `tfd_p3` is dropout 0 → 0.1 → 0.3 on
  the same setting. 0.1 is the usual transformer value (Transformer, BERT,
  ViT, DDPM). Small-data diffusion nets go to 0.3 (iDDPM on CIFAR-10). If
  `tfd_p3` lowers the held-out (`ho`) prior without raising the training
  (`tr`) prior much, the transformer wants more.
- **`prior_cond_noise σ`** (a new key) adds Gaussian noise to the context the
  prior reads, in training only. That context is the trunk's pooled node
  features and the pooled vector. The noise is σ × each row's RMS, so the
  network cannot make it smaller by scaling up its features. Each of the k FM
  draws gets its own noise. The default 0 changes nothing.
- **Considered and not run.**
  - *A longer EMA (`ema_decay 0.999`).* It averages the whole model, so the
    simulator would change too and the arm would not isolate the prior.
  - *Dropout in the prior's GNN trunk* (`prior_type gnn_e2e`, 3 layers, which
    feeds the transformer its condition). The over-fitting may sit there as
    much as in the transformer. It needs one more key; a next round if
    `tfd_p3` does not close the gap.
  - *Weight decay on the prior only.* Adam's `weight_decay` here is coupled L2,
    and AdamW-sized values (0.05) would crush the weights. The base keeps 1e-4.
  - *Local or neighbourhood attention.* Too much new code for one round.
  - *Already in place:* a model is picked by CRPS on unseen validation
    geometries, with the pick reset at the start of the prior-only tail;
    `augment_geometry`; EMA 0.99; weight decay 1e-4.

**Seeds.** `training_seed` is 42, 43 and 44 for every arm. Arm `k`'s three
seeds run on three different cards.

**Reading the result.** An arm counts as better only if every seed beats every
seed of the arm it is compared with. Each `aN` line carries two marks:

- the first against the base;
- the second against the arm it extends (`tfd` against `tfk8_r9`, the other
  `tfd_*` arms against `tfd`).

Everything here is generated. `make_sweep.py` writes `config_train_*`,
`config_infer_*`, `jobs.txt`, `run_135.sh` and the job table at the end of this
file. `--plan` prints the plan and writes nothing. Edit `make_sweep.py`, never a
generated file. Its self-check confirms that the reference runs (R1–R4) have
exactly the configs of the settings they stand for, apart from seeds and paths.

## Before starting

Pull first. `prior_tf_dropout` defaults to 0, so round 9's jobs, if one still
resumes, keep running exactly as before. `run_node.sh` stops with "Pull
first." if the checkout lacks round 10's code.

## Running it (node 135, all 8 cards, 2 jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r10/run_135.sh

This is the runner from rounds 6–9. The checks (environment, data, launcher
`--check`) run in the terminal. When they pass, all 16 jobs start at once, and
job k runs on card k mod 8. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r10/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

## Where the error comes from (automatic)

`run_135.sh` starts `diagnose_135.sh` as in round 9. It diagnoses only this
round's jobs, on the held-out file (`ho`) and on 16 training geometries (`tr`).
The reference runs R1–R4 were already diagnosed by round 9's watcher, and
`brief.py` reads them from `output/b300_r9/`. Each arm line reads:

    a3 ++ 60/62/64 m62 ho 27/33/70 tr 56/56/72 g0 l100

- **Marks**, as above.
- **w2** per seed, then **m**, their mean.
- **ho / tr**: floor / posterior sample / prior on held-out and training
  geometries.
- **g / l**: how much the decoded draw's correlation with its FE result drops
  when z_g, or z_l, is swapped with another draw's.

The anti-over-fitting arms work if `ho` prior drops toward `tr` prior. A `tr`
prior that only rises means the arm under-fits instead.

Progress is in `tail -f output/b300_r10/_diag/diag.out`. `kill_all.sh` stops
the watcher too.

- **Memory and time** are as in round 9:
  - two jobs per card with gradient checkpointing;
  - about 55 GB peak per job (a guess);
  - about 65 h per job, all 16 finishing together.
- **Compare speed** using the brief's `run` line, which shows seconds per
  epoch. Dropout adds almost nothing. `tfd_d2`'s prior is half as deep and may
  be faster.
- **If two per card turns out slower,** run `kill_all.sh`, then
  `make_sweep.py --pack 1 --force`, and start the round again. Every job
  resumes.
- **Shared tools.** Restarts, overrides, `stop_135`, `kill_all.sh`, MPS,
  `use_compile` and the hierarchy cache work as in `../r6/README.md`.
## Jobs (generated by make_sweep.py; everything below is rewritten)

16 trainings (6 arms; seeds 42, 43, 44; the base only 44, its 42 = r8 hybrid_e360, its 43 = r9 hybrid_base_s43); --pack 2: 2 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 16 slots, 3 torch threads per job. Estimated makespan 2.7 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, x1.5 per job at 2 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_gnn_k8_s42 | the base's GNN prior with 8 FM draws per posterior sample (prior_fm_draws 8); seed 42 | 0 | on | 55 | 65 | 0.0-2.7 |
| 2 | hybrid_tfd_s42 | r9's gfix_tf_k8 (gfix + transformer prior + prior_fm_draws 8) with transformer dropout 0.1 (prior_tf_dropout 0.1); seed 42 | 1 | on | 55 | 65 | 0.0-2.7 |
| 3 | hybrid_tfd_p3_s42 | tfd with dropout 0.3 (prior_tf_dropout 0.3); seed 42 | 2 | on | 55 | 65 | 0.0-2.7 |
| 4 | hybrid_tfd_cn_s42 | tfd + noise on the prior's condition (prior_cond_noise 0.1); seed 42 | 3 | on | 55 | 65 | 0.0-2.7 |
| 5 | hybrid_tfd_d2_s42 | tfd with 2 transformer blocks instead of 4 (prior_fm_blocks 2); seed 42 | 4 | on | 55 | 65 | 0.0-2.7 |
| 6 | hybrid_gnn_k8_s43 | the base's GNN prior with 8 FM draws per posterior sample (prior_fm_draws 8); seed 43 | 5 | on | 55 | 65 | 0.0-2.7 |
| 7 | hybrid_tfd_s43 | r9's gfix_tf_k8 (gfix + transformer prior + prior_fm_draws 8) with transformer dropout 0.1 (prior_tf_dropout 0.1); seed 43 | 6 | on | 55 | 65 | 0.0-2.7 |
| 8 | hybrid_tfd_p3_s43 | tfd with dropout 0.3 (prior_tf_dropout 0.3); seed 43 | 7 | on | 55 | 65 | 0.0-2.7 |
| 9 | hybrid_tfd_cn_s43 | tfd + noise on the prior's condition (prior_cond_noise 0.1); seed 43 | 0 | on | 55 | 65 | 0.0-2.7 |
| 10 | hybrid_tfd_d2_s43 | tfd with 2 transformer blocks instead of 4 (prior_fm_blocks 2); seed 43 | 1 | on | 55 | 65 | 0.0-2.7 |
| 11 | hybrid_gnn_k8_s44 | the base's GNN prior with 8 FM draws per posterior sample (prior_fm_draws 8); seed 44 | 2 | on | 55 | 65 | 0.0-2.7 |
| 12 | hybrid_tfd_s44 | r9's gfix_tf_k8 (gfix + transformer prior + prior_fm_draws 8) with transformer dropout 0.1 (prior_tf_dropout 0.1); seed 44 | 3 | on | 55 | 65 | 0.0-2.7 |
| 13 | hybrid_tfd_p3_s44 | tfd with dropout 0.3 (prior_tf_dropout 0.3); seed 44 | 4 | on | 55 | 65 | 0.0-2.7 |
| 14 | hybrid_tfd_cn_s44 | tfd + noise on the prior's condition (prior_cond_noise 0.1); seed 44 | 5 | on | 55 | 65 | 0.0-2.7 |
| 15 | hybrid_tfd_d2_s44 | tfd with 2 transformer blocks instead of 4 (prior_fm_blocks 2); seed 44 | 6 | on | 55 | 65 | 0.0-2.7 |
| 16 | hybrid_base_s44 | the base: r8 hybrid_e360 (seed 42 = r8's run, seed 43 = r9's hybrid_base_s43); seed 44 | 7 | on | 55 | 65 | 0.0-2.7 |
