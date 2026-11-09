# B300 round 9: the HI-MGN-hybrid coarse prior, two seeds, all 16 at once

Goal: close the gap between the prior and the posterior. Round 8's best
setting was 360 epochs at batch 16 (`hybrid_e360`, w2 66 against a floor of
27). Its diagnosis showed where the error comes from:

- **The decoder ignores z_g.** Shuffling z_g across scenes leaves w2 unchanged
  (48 = 48), so the global latent carries nothing the decoder reads.
- **The whole prior gap is in z_l**, and it is not an extrapolation problem.
  Even on training geometries the prior sample scores 94, against 58 for the
  posterior sample. A larger GNN prior did not help: `fmbig` and `pm6` were
  both within noise.

So the base here is `hybrid_e360`, unchanged. Every arm changes either the
coarse prior or the path z_g takes into the decoder. The new keys are
documented in `docs/reference/config/meshgraphnets_variational.md`.

| arm | change from the base | what it asks |
| --- | --- | --- |
| `base` | none (seed 43; seed 42 is r8's `hybrid_e360` itself) | the base's seed noise |
| `gfix` | `lambda_mmd_g 0`, `local_inject coarsest` | does z_g get used once it is the only latent the finer levels see, and is free of the MMD pull? |
| `tf` | `prior_arch transformer` | can full attention over the 256 coarsest nodes fit z_l where the GNN could not? |
| `gfix_tf` | both (seeds 42, 43, 44) | the main candidate |
| `gfix_tf_x` | + `prior_param x` | is predicting the clean latent easier than predicting the velocity? |
| `gfix_tf_k8` | + `prior_fm_draws 8` | does it help to train the prior with more (z0, t) draws per posterior sample? |
| `gfix_tf_ch2` | + `latent_ch 2` | is the local latent too large for the prior? |
| `gfix_tf_x_k8` | + both | do x and k8 add up? |

- **Seeds.** `training_seed` is 42 and 43 for every arm; `gfix_tf` also runs
  44. With r8's run as the base's seed 42, every arm has two seeds and the main
  candidate has three.
- **Reading the result.** r6 put the seed spread of a single setting at about
  ±10 w2 (88/88/107 at batch 4 and 120 epochs). An arm counts as better only
  if both of its seeds beat both of the base's.

Everything here is generated. `make_sweep.py` writes `config_train_*`,
`config_infer_*`, `jobs.txt`, `run_135.sh` and the job table at the end of this
file. `--plan` prints the plan and writes nothing. Edit `make_sweep.py`, never
a generated file.

## Running it (node 135, all 8 cards, 2 jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r9/run_135.sh

This is the runner from rounds 6–8. The checks (environment, data, launcher
`--check`) run in the terminal. When they pass, all 16 jobs start at once:
job k runs on card k mod 8. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r9/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

## Where the error comes from (automatic)

When the jobs have started, `run_135.sh` also starts `diagnose_135.sh`. It waits
in the background. As each job finishes, it runs
`misc/posterior_vs_prior_groups.py` on that job twice, on the held-out file
(`ho`) and on 16 training geometries (`tr`). It does the same once for r8's
`hybrid_e360`, the base's seed 42. Each part is forward passes only, about
10–20 GB, on a card with 40 GB free. The block then ends with one line per arm:

    a3 + 52/55/58 ho 27/38/52 tr 25/30/34 g9 l30

That line reads as follows, and each arm against its neighbour names one cause:

- **Posterior vs. prior.** `ho` is floor / posterior sample / prior. The step
  from floor to posterior sample is the encoder and decoder. The step from
  posterior sample to prior is the prior.
- **Under-fitted or not generalizing.** `tr` is the same split on training
  geometries. If `tr` prior is still far above `tr` post, the prior is
  under-fitted. If it is close there but far on `ho`, the prior does not carry
  over to new geometries.
- **Which latent carries the draw.** `g` and `l` show how much the decoded
  draw's correlation with its FE result drops when z_g, or z_l, is swapped
  with another draw's. Near 0, the decoder ignores that part. r8 had g ≈ 0.
- **Which change caused it.**
  - `gfix` against `base`: the z_g path.
  - `tf` against `base`: the prior architecture.
  - `gfix_tf` against both: whether the two add up.
  - `_x`, `_k8` and `_ch2` against `gfix_tf`: one prior factor each.

Progress is in `tail -f output/b300_r9/_diag/diag.out`. `kill_all.sh` stops
the watcher too. Run `bash .../r9/diagnose_135.sh` to start it again; finished
parts are kept.

- **Memory: two per card needs checkpointing.** A batch-16 job peaks at about
  180 GB without gradient checkpointing, so two do not fit on a 288 GB card.
  With it, the peak is about 55 GB (a guess), at about 1.52x per step.
- **Time.** Both factors below are guesses, so the estimate is too: x1.52 for
  checkpointing and x1.5 for sharing a card make about 65 h per job, with all
  16 finishing together.
  - One per card without checkpointing would run two waves of about 30 h each,
    about 60 h in all.
  - Round 8 was host-launch bound (the GPU was active 65 % of the time). If
    two per card fills that idle time, the sharing costs less than x1.5.
  - The brief's `run` line shows each job's seconds per epoch. Compare them
    with r8's 277 s alone.
  - If two per card turns out slower, stop the round with `kill_all.sh`, run
    `make_sweep.py --pack 1 --force` and start it again. Every job resumes.
- **CPU.** About 120 of the 128 CPUs are split over 16 jobs: 3 torch threads
  and 4 DataLoader workers each.
- **Shared tools.** Restarts, overrides, `stop_135`, `kill_all.sh`, MPS,
  `use_compile` and the hierarchy cache work as in `../r6/README.md`.

## Jobs (generated by make_sweep.py; everything below is rewritten)

16 trainings (8 arms; seeds 42, 43, gfix_tf also 44; the base's seed 42 = r8 hybrid_e360); --pack 2: 2 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 16 slots, 3 torch threads per job. Estimated makespan 2.7 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, x1.5 per job at 2 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_gfix_s42 | z_g without MMD (lambda_mmd_g 0), z_l only at the coarsest level (local_inject coarsest); seed 42 | 0 | on | 55 | 65 | 0.0-2.7 |
| 2 | hybrid_tf_s42 | coarse prior as a transformer (prior_arch transformer), z_g path as the base; seed 42 | 1 | on | 55 | 65 | 0.0-2.7 |
| 3 | hybrid_gfix_tf_s42 | gfix + transformer prior: the main candidate; seed 42 | 2 | on | 55 | 65 | 0.0-2.7 |
| 4 | hybrid_gfix_tf_x_s42 | gfix_tf, the prior predicts the clean latent (prior_param x); seed 42 | 3 | on | 55 | 65 | 0.0-2.7 |
| 5 | hybrid_gfix_tf_k8_s42 | gfix_tf, 8 FM draws per posterior sample (prior_fm_draws 8); seed 42 | 4 | on | 55 | 65 | 0.0-2.7 |
| 6 | hybrid_gfix_tf_ch2_s42 | gfix_tf, latent_ch 2 (half the local latent); seed 42 | 5 | on | 55 | 65 | 0.0-2.7 |
| 7 | hybrid_gfix_tf_x_k8_s42 | gfix_tf with prior_param x and prior_fm_draws 8; seed 42 | 6 | on | 55 | 65 | 0.0-2.7 |
| 8 | hybrid_gfix_s43 | z_g without MMD (lambda_mmd_g 0), z_l only at the coarsest level (local_inject coarsest); seed 43 | 7 | on | 55 | 65 | 0.0-2.7 |
| 9 | hybrid_tf_s43 | coarse prior as a transformer (prior_arch transformer), z_g path as the base; seed 43 | 0 | on | 55 | 65 | 0.0-2.7 |
| 10 | hybrid_gfix_tf_s43 | gfix + transformer prior: the main candidate; seed 43 | 1 | on | 55 | 65 | 0.0-2.7 |
| 11 | hybrid_gfix_tf_x_s43 | gfix_tf, the prior predicts the clean latent (prior_param x); seed 43 | 2 | on | 55 | 65 | 0.0-2.7 |
| 12 | hybrid_gfix_tf_k8_s43 | gfix_tf, 8 FM draws per posterior sample (prior_fm_draws 8); seed 43 | 3 | on | 55 | 65 | 0.0-2.7 |
| 13 | hybrid_gfix_tf_ch2_s43 | gfix_tf, latent_ch 2 (half the local latent); seed 43 | 4 | on | 55 | 65 | 0.0-2.7 |
| 14 | hybrid_gfix_tf_x_k8_s43 | gfix_tf with prior_param x and prior_fm_draws 8; seed 43 | 5 | on | 55 | 65 | 0.0-2.7 |
| 15 | hybrid_base_s43 | the base: r8 hybrid_e360 (seed 42 = r8's run, w2 66); seed 43 | 6 | on | 55 | 65 | 0.0-2.7 |
| 16 | hybrid_gfix_tf_s44 | gfix + transformer prior: the main candidate; seed 44 | 7 | on | 55 | 65 | 0.0-2.7 |
