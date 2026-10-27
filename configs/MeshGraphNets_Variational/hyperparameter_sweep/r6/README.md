# B300 round 6: placement sweep, one knob at a time

Goal: how each knob moves the sample quality of each latent placement of MGN-V
on ex3 shell buckling, before the placements are compared for the paper:

| placement | display name | `latent_type` |
| --- | --- | --- |
| global | HI-MGN-V | `global`: one latent vector per shell |
| local | HI-MGNflow | `local`: `latent_ch` channels per coarsest V-cycle node |
| hybrid | HI-MGN-hybrid | `hybrid`: global vector + per-coarse-node channels |

Everything here is generated: `make_sweep.py` writes
`config_train_*`, `config_infer_*`, `jobs.txt` and the job table at the end of
this file (`--plan` prints the
queue and its estimated hours and writes nothing). Edit `make_sweep.py`, never
a generated config. The round-based generator it replaced is in
`junk/legacy_placement_rounds_2026-10-26/`.

## Running it (node 135, cards 0 and 2-7, two jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r6/run_135.sh

Checks (environment, data, launcher `--check`) run in the terminal; when they
pass, the jobs detach. 14 slots (the 7 cards, each listed twice) take the
lines of `jobs.txt` in order. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r6/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **Before the first run**: `dataset/probabilistic/ex3_shell_buckling_infer.h5`
  on the node must be the **256-draw file** (1.57 GB; root attributes
  `draws_per_shape 256`, `num_samples 2048`). Stage 2 refuses the 64-draw file.
- **After a stop or crash** run the same command again: finished jobs are
  skipped, unfinished trainings continue from their resume state.
- **Two jobs per card is unmeasured**: memory (B300) is not the concern; the
  slowdown of each job and the CPU contention (`THREADS_PER_JOB 4`,
  `num_workers 4`, 14 jobs) are. Read the epoch times in the first brief.
  Fewer slots: `GPUS="0 2 3 4 5 6 7" bash run_135.sh`.
- **Sampling-only jobs** (`*_ref_infl`, `*_ref_t*`) sample `<placement>_ref`'s
  checkpoint; a slot that takes one before that training ends waits for it
  (state `wait`, brief's `wait` line).
- `run_node.sh`'s header lists every override (`JOBS=`, `REDO=1`, `TRAIN=0`, ...).
- **Kill everything** (every round's runner and its jobs, any python left
  running from the repository, then the cache build `.lock`/`.tmp.*` files a
  killed builder leaves): `bash .../r6/kill_all.sh -n` lists, without `-n` it
  kills (by PID; SIGTERM, then SIGKILL after 20 s). Killed trainings resume.
- **The hierarchy cache** (9,600 samples x 2 variants, shared by every job) is
  built once, by the first job, while the others wait: 48 processes
  (`hierarchy_cache_build_workers 48`, on the 128-core node), with gzip done in
  the workers. Rounds 4-5 built it with 8.

## The design

Every job is the **reference** of its placement with **exactly one knob
moved**. The reference is round 4's `<placement>_mmd` with the deterministic
HI-MGN's depth (`mp_per_level 4, 6, 8, 6, 4`, so `message_passing_num 28`);
`make_sweep.py` checks this key by key against `../r4/config_train_<p>_mmd.txt`
and, for hybrid, `../r5/config_train_hybrid_mp28.txt`.

**Fixed**: `mp_per_level 4, 6, 8, 6, 4`; `voronoi_clusters 4096, 256`;
`beta_aux 10`; `split_seed`, `hierarchy_seed` 42 (every job sees the same split
and the same coarsening); everything else in the reference.

| knob | config key | reference | moved to | placements |
| --- | --- | --- | --- | --- |
| learning rate | `learningr` | 1e-4 | 3e-5, 3e-4, 1e-3 | all |
| epochs | `training_epochs` | 120 | 60, 240 | all |
| batch size | `batch_size` | 4 | 2, 8, 16 | all |
| reconstruction weight | `alpha_recon` | 1000 | 100, 10000 | all |
| backbone width | `latent_dim` | 128 | 64, 256 | all |
| MMD weight | `lambda_mmd` | 1 | 0.1, 10, 100 | all |
| global latent size | `vae_latent_dim` | 32 | 8, 16, 64, 128 (hybrid: 16, 64) | global, hybrid |
| local latent channels | `latent_ch` | 4 | 1, 2, 8, 16 (hybrid: 2, 8) | local, hybrid |
| prior width | `prior_hidden_dim` | 128 | 64, 256 | all |
| prior depth | `prior_mp_layers` | 3 | 2, 6 | all |
| coarse FM prior blocks | `prior_fm_blocks` | 4 | 2, 8 | local, hybrid |
| posterior std floor | `posterior_min_std` | 0.05 | 0.01, 0.1 | all |
| training seed | `training_seed` | 42 | 43, 44 | all |
| latent inflation (sampling only) | `latent_inflation` | 1 | 1.25, 1.5, 2 | all |
| prior temperature (sampling only) | `prior_temperature` | 1 | 0.5, 2 | all |
| deterministic line | `use_vae False` | - | `ref_det`, once | - |

98 jobs: 89 trainings, 9 sampling-only. Notes on the values:

- `prior_freeze_epoch` follows `training_epochs` at 2/3 (80 at 120).
- Batch size moves alone: the learning rate is not rescaled. The MMD term is
  estimated within a batch, so batch size also moves its strength.
- `alpha_recon` and `lambda_mmd` overlap: with Adam and gradient clipping,
  `alpha_recon` x10 behaves like `lambda_mmd` and `beta_aux` /10. Both are run
  because `alpha_recon` also moves `beta_aux` relative to the reconstruction.
- `prior_fm_blocks` is inert for global (its FM prior uses the MLP velocity).
- The seed jobs give the noise every difference is read against: a knob whose
  w2 differs from its reference by less than the spread of the three seeds
  (42, 43, 44) has not shown an effect.
- Order: the slowest jobs first (240 epochs, width 256), then references and
  seeds, then the knobs; sampling-only jobs after the 60-epoch ones, when
  their reference has usually finished.
- The table at the end lists every job with its estimated hours. The estimate is a
  guess anchored on round 5's `hybrid_mp28` (34 h for 120 epochs, one job per
  card), x1.5 for two per card, width^1.5 and guessed batch factors: about
  16 days on 14 slots.

## Reading the results

- Rank by the headline `w2` on the 256-draw held-out file: per geometry, W2
  between the model's 1024 draws (4 per scene) and the 256 FE results, over
  that geometry's FE sd; RMS over the 8 geometries.
- Per knob, read the change against the reference and against the seed spread.
  The per-geometry histograms (`plots.py`, in each run dir) show what moved.
- **Selection uses the held-out (test) file**, by choice. This biases the
  best numbers downward (winner's curse); a final run of the chosen configs
  with fresh seeds removes the training-seed part, not the part from choosing
  on these 8 geometries. The paper says so.

## How big is a W2? (x100, on the 256-draw file)

| reference | w2 |
| --- | --- |
| FE against FE, 128 vs 128 (the **floor**, median and 5-95 %) | 27 (19-36) |
| FE against FE, 32 vs 224 | 38 |
| perfect sampler, 1024 draws vs the 256 FE (estimate) | ~15 |
| unimodal Gaussian with the right mean and sd | 49 |
| right distribution shifted by 0.5 sd | 50 |
| buckling probability off by 0.1 / 0.2 / 0.3 | 53 / 82 / 113 |
| **exact FE mean, no spread** (a perfect deterministic model) | **100** |
| right distribution shifted by 1 sd | 100 |
| ignores the geometry (one distribution for all 8) | 1250 |

So 100 is large: it is what a perfect point prediction scores. Round 5's local
(89-92 on the 64-draw file) was barely better than that and global (180-214)
was worse than a point prediction 1 sd off. Round-5 numbers used the 64-draw
file (floor 52) and are not comparable; they are not rescored in r6.

## Sources (deep research, 2026-10)

- Google Research, *Deep Learning Tuning Playbook*: separate the scientific
  knob (here the placement) from nuisance knobs and tune the nuisance knobs
  separately for every value of the scientific one; quasi-random search over
  one-at-a-time. This sweep is one-at-a-time by choice: it maps what each knob
  does around one reference; a combined search, if any, comes after it.
- Lucic et al., *Are GANs Created Equal?* (NeurIPS 2018); Dodge et al., *Show
  Your Work* (EMNLP 2019): compare models under equal, stated budgets and
  report best-so-far against trials.
- Bergstra & Bengio (JMLR 2012): random/quasi-random search beats grids when
  few knobs matter; learning rate on a log scale.
- Bouthillier et al. (MLSys 2021): seed variance is comparable to many
  reported gains; rerun the winners with fresh seeds.
- Cawley & Talbot (JMLR 2010): selecting on the evaluation data biases it;
  pre-state the rule and disclose it.
- arXiv 2501.01423: latent size and the prior/generator size must be chosen
  together, on the generative metric, not on reconstruction.
- SAR (arXiv 2604.11403) and LDGN (arXiv 2504.02843) papers for the baselines'
  own settings. Giving them the same treatment is advisable but outside this sweep.
- Caveat: no source gave verified numeric ranges for this model family; the
  ranges above are this sweep's choice. "Never tune epochs" was checked and is
  not supported by the sources.
## Jobs (generated by make_sweep.py; everything below is rewritten)

98 jobs (89 trainings, 9 sampling only) on 14 slots; estimated makespan 16 days (guess: 34 h per 120 epochs on one card, x1.5 for two per card, width^1.5, batch factors {2: 1.3, 4: 1.0, 8: 0.85, 16: 0.75}).

| # | job | change from the reference | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- |
| 1 | global_e240 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 240 (reference 120) | 104 | 0.0-4.3 |
| 2 | local_e240 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 240 (reference 120) | 104 | 0.0-4.3 |
| 3 | hybrid_e240 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 240 (reference 120) | 104 | 0.0-4.3 |
| 4 | global_w256 | backbone width (latent_dim): latent_dim 256 (reference 128) | 146 | 0.0-6.1 |
| 5 | local_w256 | backbone width (latent_dim): latent_dim 256 (reference 128) | 146 | 0.0-6.1 |
| 6 | hybrid_w256 | backbone width (latent_dim): latent_dim 256 (reference 128) | 146 | 0.0-6.1 |
| 7 | global_ref | none (the reference) | 53 | 0.0-2.2 |
| 8 | local_ref | none (the reference) | 53 | 0.0-2.2 |
| 9 | hybrid_ref | none (the reference) | 53 | 0.0-2.2 |
| 10 | global_ref_s43 | training_seed 43 (reference 42) | 53 | 0.0-2.2 |
| 11 | global_ref_s44 | training_seed 44 (reference 42) | 53 | 0.0-2.2 |
| 12 | local_ref_s43 | training_seed 43 (reference 42) | 53 | 0.0-2.2 |
| 13 | local_ref_s44 | training_seed 44 (reference 42) | 53 | 0.0-2.2 |
| 14 | hybrid_ref_s43 | training_seed 43 (reference 42) | 53 | 0.0-2.2 |
| 15 | hybrid_ref_s44 | training_seed 44 (reference 42) | 53 | 2.2-4.4 |
| 16 | global_lr3e-5 | learning rate: learningr 3e-5 (reference 1e-4) | 53 | 2.2-4.4 |
| 17 | global_lr3e-4 | learning rate: learningr 3e-4 (reference 1e-4) | 53 | 2.2-4.4 |
| 18 | global_lr1e-3 | learning rate: learningr 1e-3 (reference 1e-4) | 53 | 2.2-4.4 |
| 19 | local_lr3e-5 | learning rate: learningr 3e-5 (reference 1e-4) | 53 | 2.2-4.4 |
| 20 | local_lr3e-4 | learning rate: learningr 3e-4 (reference 1e-4) | 53 | 2.2-4.4 |
| 21 | local_lr1e-3 | learning rate: learningr 1e-3 (reference 1e-4) | 53 | 2.2-4.4 |
| 22 | hybrid_lr3e-5 | learning rate: learningr 3e-5 (reference 1e-4) | 53 | 2.2-4.4 |
| 23 | hybrid_lr3e-4 | learning rate: learningr 3e-4 (reference 1e-4) | 53 | 4.3-6.5 |
| 24 | hybrid_lr1e-3 | learning rate: learningr 1e-3 (reference 1e-4) | 53 | 4.3-6.5 |
| 25 | global_e60 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 60 (reference 120) | 28 | 4.3-5.5 |
| 26 | local_e60 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 60 (reference 120) | 28 | 4.4-5.6 |
| 27 | hybrid_e60 | epochs (prior_freeze_epoch follows at 2/3): training_epochs 60 (reference 120) | 28 | 4.4-5.6 |
| 28 | global_ref_infl | sampling only: latent_inflation 1, 1.25, 1.5, 2 on global_ref | 4 | 4.4-4.6 |
| 29 | global_ref_t0.5 | sampling only: prior_temperature 0.5 on global_ref | 2 | 4.4-4.5 |
| 30 | global_ref_t2 | sampling only: prior_temperature 2 on global_ref | 2 | 4.4-4.5 |
| 31 | local_ref_infl | sampling only: latent_inflation 1, 1.25, 1.5, 2 on local_ref | 4 | 4.4-4.6 |
| 32 | local_ref_t0.5 | sampling only: prior_temperature 0.5 on local_ref | 2 | 4.4-4.5 |
| 33 | local_ref_t2 | sampling only: prior_temperature 2 on local_ref | 2 | 4.4-4.5 |
| 34 | hybrid_ref_infl | sampling only: latent_inflation 1, 1.25, 1.5, 2 on hybrid_ref | 4 | 4.5-4.7 |
| 35 | hybrid_ref_t0.5 | sampling only: prior_temperature 0.5 on hybrid_ref | 2 | 4.5-4.6 |
| 36 | hybrid_ref_t2 | sampling only: prior_temperature 2 on hybrid_ref | 2 | 4.5-4.6 |
| 37 | global_bs2 | batch size (learning rate not rescaled): batch_size 2 (reference 4) | 68 | 4.5-7.3 |
| 38 | global_bs8 | batch size (learning rate not rescaled): batch_size 8 (reference 4) | 45 | 4.6-6.5 |
| 39 | global_bs16 | batch size (learning rate not rescaled): batch_size 16 (reference 4) | 40 | 4.6-6.3 |
| 40 | local_bs2 | batch size (learning rate not rescaled): batch_size 2 (reference 4) | 68 | 4.6-7.4 |
| 41 | local_bs8 | batch size (learning rate not rescaled): batch_size 8 (reference 4) | 45 | 4.6-6.5 |
| 42 | local_bs16 | batch size (learning rate not rescaled): batch_size 16 (reference 4) | 40 | 4.7-6.3 |
| 43 | hybrid_bs2 | batch size (learning rate not rescaled): batch_size 2 (reference 4) | 68 | 5.5-8.3 |
| 44 | hybrid_bs8 | batch size (learning rate not rescaled): batch_size 8 (reference 4) | 45 | 5.6-7.5 |
| 45 | hybrid_bs16 | batch size (learning rate not rescaled): batch_size 16 (reference 4) | 40 | 5.6-7.2 |
| 46 | global_ar100 | reconstruction weight: alpha_recon 100 (reference 1000) | 53 | 6.1-8.3 |
| 47 | global_ar10000 | reconstruction weight: alpha_recon 10000 (reference 1000) | 53 | 6.1-8.3 |
| 48 | local_ar100 | reconstruction weight: alpha_recon 100 (reference 1000) | 53 | 6.1-8.3 |
| 49 | local_ar10000 | reconstruction weight: alpha_recon 10000 (reference 1000) | 53 | 6.3-8.5 |
| 50 | hybrid_ar100 | reconstruction weight: alpha_recon 100 (reference 1000) | 53 | 6.3-8.6 |
| 51 | hybrid_ar10000 | reconstruction weight: alpha_recon 10000 (reference 1000) | 53 | 6.5-8.7 |
| 52 | global_w64 | backbone width (latent_dim): latent_dim 64 (reference 128) | 20 | 6.5-7.3 |
| 53 | local_w64 | backbone width (latent_dim): latent_dim 64 (reference 128) | 20 | 6.5-7.4 |
| 54 | hybrid_w64 | backbone width (latent_dim): latent_dim 64 (reference 128) | 20 | 6.5-7.4 |
| 55 | global_mmd0.1 | MMD weight: lambda_mmd 0.1 (reference 1) | 53 | 7.2-9.4 |
| 56 | global_mmd10 | MMD weight: lambda_mmd 10 (reference 1) | 53 | 7.3-9.5 |
| 57 | global_mmd100 | MMD weight: lambda_mmd 100 (reference 1) | 53 | 7.3-9.6 |
| 58 | local_mmd0.1 | MMD weight: lambda_mmd 0.1 (reference 1) | 53 | 7.4-9.6 |
| 59 | local_mmd10 | MMD weight: lambda_mmd 10 (reference 1) | 53 | 7.4-9.6 |
| 60 | local_mmd100 | MMD weight: lambda_mmd 100 (reference 1) | 53 | 7.4-9.6 |
| 61 | hybrid_mmd0.1 | MMD weight: lambda_mmd 0.1 (reference 1) | 53 | 7.5-9.7 |
| 62 | hybrid_mmd10 | MMD weight: lambda_mmd 10 (reference 1) | 53 | 8.3-10.5 |
| 63 | hybrid_mmd100 | MMD weight: lambda_mmd 100 (reference 1) | 53 | 8.3-10.5 |
| 64 | global_ld8 | global latent vector size: vae_latent_dim 8 (reference 32) | 53 | 8.3-10.5 |
| 65 | global_ld16 | global latent vector size: vae_latent_dim 16 (reference 32) | 53 | 8.3-10.5 |
| 66 | global_ld64 | global latent vector size: vae_latent_dim 64 (reference 32) | 53 | 8.5-10.7 |
| 67 | global_ld128 | global latent vector size: vae_latent_dim 128 (reference 32) | 53 | 8.6-10.8 |
| 68 | hybrid_ld16 | global latent vector size: vae_latent_dim 16 (reference 32) | 53 | 8.7-10.9 |
| 69 | hybrid_ld64 | global latent vector size: vae_latent_dim 64 (reference 32) | 53 | 9.4-11.7 |
| 70 | local_ch1 | latent channels per coarse node: latent_ch 1 (reference 4) | 53 | 9.5-11.7 |
| 71 | local_ch2 | latent channels per coarse node: latent_ch 2 (reference 4) | 53 | 9.6-11.8 |
| 72 | local_ch8 | latent channels per coarse node: latent_ch 8 (reference 4) | 53 | 9.6-11.8 |
| 73 | local_ch16 | latent channels per coarse node: latent_ch 16 (reference 4) | 53 | 9.6-11.8 |
| 74 | hybrid_ch2 | latent channels per coarse node: latent_ch 2 (reference 4) | 53 | 9.6-11.8 |
| 75 | hybrid_ch8 | latent channels per coarse node: latent_ch 8 (reference 4) | 53 | 9.7-11.9 |
| 76 | global_ph64 | prior width: prior_hidden_dim 64 (reference 128) | 53 | 10.5-12.7 |
| 77 | global_ph256 | prior width: prior_hidden_dim 256 (reference 128) | 53 | 10.5-12.7 |
| 78 | local_ph64 | prior width: prior_hidden_dim 64 (reference 128) | 53 | 10.5-12.7 |
| 79 | local_ph256 | prior width: prior_hidden_dim 256 (reference 128) | 53 | 10.5-12.7 |
| 80 | hybrid_ph64 | prior width: prior_hidden_dim 64 (reference 128) | 53 | 10.7-12.9 |
| 81 | hybrid_ph256 | prior width: prior_hidden_dim 256 (reference 128) | 53 | 10.8-13.0 |
| 82 | global_pm2 | prior message-passing layers: prior_mp_layers 2 (reference 3) | 53 | 10.9-13.1 |
| 83 | global_pm6 | prior message-passing layers: prior_mp_layers 6 (reference 3) | 53 | 11.7-13.9 |
| 84 | local_pm2 | prior message-passing layers: prior_mp_layers 2 (reference 3) | 53 | 11.7-13.9 |
| 85 | local_pm6 | prior message-passing layers: prior_mp_layers 6 (reference 3) | 53 | 11.8-14.0 |
| 86 | hybrid_pm2 | prior message-passing layers: prior_mp_layers 2 (reference 3) | 53 | 11.8-14.0 |
| 87 | hybrid_pm6 | prior message-passing layers: prior_mp_layers 6 (reference 3) | 53 | 11.8-14.0 |
| 88 | local_fb2 | coarse FM prior blocks: prior_fm_blocks 2 (reference 4) | 53 | 11.8-14.1 |
| 89 | local_fb8 | coarse FM prior blocks: prior_fm_blocks 8 (reference 4) | 53 | 11.9-14.1 |
| 90 | hybrid_fb2 | coarse FM prior blocks: prior_fm_blocks 2 (reference 4) | 53 | 12.7-14.9 |
| 91 | hybrid_fb8 | coarse FM prior blocks: prior_fm_blocks 8 (reference 4) | 53 | 12.7-14.9 |
| 92 | global_pms0.01 | posterior std floor: posterior_min_std 0.01 (reference 0.05) | 53 | 12.7-14.9 |
| 93 | global_pms0.1 | posterior std floor: posterior_min_std 0.1 (reference 0.05) | 53 | 12.7-15.0 |
| 94 | local_pms0.01 | posterior std floor: posterior_min_std 0.01 (reference 0.05) | 53 | 12.9-15.1 |
| 95 | local_pms0.1 | posterior std floor: posterior_min_std 0.1 (reference 0.05) | 53 | 13.0-15.2 |
| 96 | hybrid_pms0.01 | posterior std floor: posterior_min_std 0.01 (reference 0.05) | 53 | 13.1-15.3 |
| 97 | hybrid_pms0.1 | posterior std floor: posterior_min_std 0.1 (reference 0.05) | 53 | 13.9-16.1 |
| 98 | ref_det | deterministic HI-MGN (use_vae False) on the same backbone: the line a sampler must beat | 33 | 13.9-15.3 |
