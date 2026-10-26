# B300 placement sweep r6: one knob at a time

Goal: how each knob moves the sample quality of each latent placement of MGN-V
on ex3 shell buckling, before the placements are compared for the paper:

| placement | display name | `latent_type` |
| --- | --- | --- |
| global | HI-MGN-V | `global`: one latent vector per shell |
| local | HI-MGNflow | `local`: `latent_ch` channels per coarsest V-cycle node |
| hybrid | HI-MGN-hybrid | `hybrid`: global vector + per-coarse-node channels |

Everything here is generated: `make_sweep.py` writes
`r6/{config_train_*,config_infer_*,jobs.txt,run_135.sh,README.md}` (`--plan` prints the
queue and its estimated hours and writes nothing). Edit `make_sweep.py`, never
a generated config. The round-based generator it replaced is in
`junk/legacy_placement_rounds_2026-10-26/`.

## Running it (node 135, cards 0 and 2-7, two jobs per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/placement/r6/run_135.sh

Checks (environment, data, launcher `--check`) run in the terminal; when they
pass, the jobs detach. 14 slots (the 7 cards, each listed twice) take the
lines of `r6/jobs.txt` in order. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/placement/brief.py r6

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.

- **Before the first run**: `dataset/probabilistic/ex3_shell_buckling_infer.h5`
  on the node must be the **256-draw file** (1.57 GB; root attributes
  `draws_per_shape 256`, `num_samples 2048`). Stage 2 refuses the 64-draw file.
- **After a stop or crash** run the same command again: finished jobs are
  skipped, unfinished trainings continue from their resume state.
- **Two jobs per card is unmeasured**: memory (B300) is not the concern; the
  slowdown of each job and the CPU contention (`THREADS_PER_JOB 4`,
  `num_workers 4`, 14 jobs) are. Read the epoch times in the first brief.
  Fewer slots: `GPUS="0 2 3 4 5 6 7" bash r6/run_135.sh`.
- **Sampling-only jobs** (`*_ref_infl`, `*_ref_t*`) sample `<placement>_ref`'s
  checkpoint; a slot that takes one before that training ends waits for it
  (state `wait`, brief's `wait` line).
- `run_node.sh`'s header lists every override (`JOBS=`, `REDO=1`, `TRAIN=0`, ...).

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
- `r6/README.md` lists every job with its estimated hours. The estimate is a
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
