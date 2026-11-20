# B300 round 12: the prior trained to the end with the encoder, its loss sent into the encoder

Model: HI-MGN-hybrid (`meshgraphnets-v`, `latent_type hybrid`) on ex3 shell
buckling. Node 135, eight jobs, one per card.

## What it asks

Up to round 11 every run trained in two parts:

1. Epochs 0–240: encoder, decoder and prior together, with the prior's loss
   stopped before the encoder (`prior_grad_to_encoder 0`).
2. Epochs 240–360: the prior alone, on the frozen posterior
   (`prior_freeze_epoch 240`).

In every earlier round the posterior sample sits at the floor and the whole
error is the prior's. In that schedule the encoder is never asked to produce
a latent the prior can hit. Every run of round 12 trains one objective from the
first epoch to the last and lets the prior's loss reach the encoder (`joint`).
Three of the four settings add one change to that.

## The settings

| setting | change from `joint` | why | seeds |
| --- | --- | --- | --- |
| base (not run here) | two parts: `prior_freeze_epoch 240`, `prior_grad_to_encoder 0` | the reference | 42, 43, 44 |
| `joint` | `prior_freeze_epoch 0`, `prior_grad_to_encoder 1` | the setting of this round | 42, 43 |
| `joint_g01` | `prior_grad_to_encoder 0.1` | the same with a tenth of the pull on the encoder, in case the full pull costs the posterior | 42, 43 |
| `joint_kl` | `latent_reg kl`, `kl_weight 0.01` (no MMD) | the prior's loss has no term that rewards a spread-out posterior; the KL has one | 42, 43 |
| `joint_tfk8` | `prior_arch transformer`, `prior_fm_draws 8`, `local_inject coarsest`, `lambda_mmd_g 0` | the best prior of rounds 9–11; 8 draws also give the encoder a less noisy gradient | 42, 43 |

The base's three runs are r8 `hybrid_e360`, r9 `hybrid_base_s43` and r10
`hybrid_base_s44`; `brief.py` reads them where those rounds left them.
Everything else equals the base: 360 epochs, batch 16, `prior_nll_weight 1`,
`beta_aux 10`, `best_by crps`.

`kl_weight 0.01`: the KL is a mean per latent coordinate, about 5 at the
posterior this model reaches (sd at the 0.05 floor, means of a few units), so
0.01 puts it at a few percent to a few tens of percent of the reconstruction
term. This is an estimate, not a measurement.

Earlier rounds, all on the two-part schedule and one seed each: round 3 ran
`kl_weight` 0.001 and saw no effect; round 8 ran 0.1, 1, 10 and 100 and scored
95, 112, 85 and 99 on the held-out shapes against the base's 85. No weight
helped there, and 0.01 is weaker than every weight round 8 ran. The only new
part here is the joint training, where the KL is the counterweight to the
prior's pull on the encoder.

`joint_tfk8`: r11's `local_tfk8_s43` (the same prior on `latent_type local`)
went to NaN once, cause unknown.

## What `prior_freeze_epoch 0` changes besides the schedule

- **The checkpoint kept.** With a freeze, the best validation CRPS of the
  prior-only part (epochs 240–360). Without one, the best of the whole run.
- **The prior's target.** Its standardization statistics are fitted at the
  freeze. Without a freeze they stay at identity.

## What can go wrong, and where it shows

The prior's loss has no term that rewards a spread-out posterior. Sent into
the encoder, it pulls the posterior towards what the geometry alone predicts.
Reconstruction and `beta_aux` hold against it. If they lose, the encoder drops
the draw's information: the prior becomes easy to hit and the decoded draws
stop matching FE.

On a 12 x 10 toy grid (CPU, one seed, 120 epochs, this round's loss weights)
the full pull shrank the spread of the z_g means by about a third and of the
z_l means by about a tenth, and the reconstruction did not get worse. That says
the code path trains; it does not say what happens on ex3.

Each setting's line of the block carries floor / posterior sample / prior on
three sets of geometries. The **second** number is the check: it has to stay at
the floor. A setting whose second number rose has a prior score that means
nothing.

## Before starting

Pull first. `run_node.sh` stops with "Pull first." if the checkout lacks the
two keys. Both default to the base's behaviour, so a resuming job of an older
round runs as before.

## Running it (node 135, cards 0–7, one job per card)

    bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r12/run_135.sh

The same runner as rounds 6–11. The checks run in the terminal; when they pass,
the eight jobs start at once, job k on card k-1. Then, at any time:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r12/brief.py

and type back the block between `==== TYPE THIS BACK ====` and `==== END ====`.
It is one line per setting:

    a0 66/95/75 m78 ho 27/29/78 tr 56/57/82 ts 45/48/100 g0 l100
    a2 += 52/61 m57 ho 27/29/57 tr 56/57/59 ts 45/46/60 g0 l100

- **a0–a4**: base, `joint`, `joint_g01`, `joint_kl`, `joint_tfk8`.
- **Mark**: `+` every seed beats every seed of the other setting, `-` every
  seed loses, `=` neither. a1 has one mark, against the base. a2–a4 have two:
  against the base, then against a1.
- **w2** per seed, then **m**, their mean.
- **ho / tr / ts**: floor / posterior sample / prior on the 8 held-out
  geometries, on 16 training geometries, and on the 15 geometries of the test
  split (never trained on, inside the training ranges).
- **g / l**: how much the decoded draw's correlation with its FE result drops
  when z_g, or z_l, is swapped with another draw's.

## Diagnosis and histograms (automatic)

`run_135.sh` starts `diagnose_135.sh`. Each time a job is trained and scored,
it runs `posterior_vs_prior_groups.py` on that job's checkpoint for the three
sets above. When the round is done it draws one histogram per geometry:

    output/b300_r12/_diag/unseen_<setting>_<tr|ts|ho>.png

One panel per geometry: the FE results in grey, each seed's prior draws as a
line, the W2 of each seed in the legend. `<setting>` is `base`, `joint`,
`joint_g01`, `joint_kl` or `joint_tfk8`. To draw them earlier, on what is
finished:

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r12/unseen_brief.py --figures

(with the MGN-V interpreter, which has matplotlib). Progress is in
`tail -f output/b300_r12/_diag/diag.out`. `kill_all.sh` stops the watcher too.

## Memory and time

One job per card without gradient checkpointing: about 180 GB peak and 30 h
per job, estimated from the base. A joint epoch trains the encoder and the
decoder for all 360 epochs, where the base trains only the prior for the last
120, so every job can take longer than that. `joint_tfk8` adds 8 prior draws
per sample on the 256-node level, under 1% of the nodes.

To run two per card instead: `kill_all.sh`, then `make_sweep.py --pack 2
--force`, and start the round again. Every job resumes. Restarts, overrides,
`stop_135`, `kill_all.sh`, MPS, `use_compile` and the hierarchy cache work as
in [the sweep index](../README.md#shared-runner-tools).

## Jobs (generated by make_sweep.py; everything below is rewritten)

8 trainings (4 settings: joint seeds 42/43; joint_g01 seeds 42/43; joint_kl seeds 42/43; joint_tfk8 seeds 42/43; the base is not run: its 42/43/44 = r8 hybrid_e360, r9 hybrid_base_s43, r10 hybrid_base_s44); --pack 1: 1 per card on GPUs 0, 1, 2, 3, 4, 5, 6, 7 = 8 slots, 11 torch threads per job. Estimated makespan 1.2 days (GUESSED speeds: 0.077 h per batch-16 epoch alone, x1.52 with checkpointing, x1.0 per job at 1 per card; peaks with checkpointing guessed at 0.3 of the full peak).

| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | hybrid_joint_s42 | no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 42 | 0 | off | 180 | 30 | 0.0-1.2 |
| 2 | hybrid_joint_g01_s42 | joint, the prior's loss reaching the encoder at a tenth of the strength (prior_grad_to_encoder 0.1); seed 42 | 1 | off | 180 | 30 | 0.0-1.2 |
| 3 | hybrid_joint_kl_s42 | joint, latent_reg kl (kl_weight 0.01) in place of mmd; seed 42 | 2 | off | 180 | 30 | 0.0-1.2 |
| 4 | hybrid_joint_tfk8_s42 | joint, the prior of r9's gfix_tf_k8: transformer, prior_fm_draws 8, local_inject coarsest, lambda_mmd_g 0; seed 42 | 3 | off | 180 | 30 | 0.0-1.2 |
| 5 | hybrid_joint_s43 | no prior-only part (prior_freeze_epoch 0) and the prior's loss reaches the encoder (prior_grad_to_encoder 1); seed 43 | 4 | off | 180 | 30 | 0.0-1.2 |
| 6 | hybrid_joint_g01_s43 | joint, the prior's loss reaching the encoder at a tenth of the strength (prior_grad_to_encoder 0.1); seed 43 | 5 | off | 180 | 30 | 0.0-1.2 |
| 7 | hybrid_joint_kl_s43 | joint, latent_reg kl (kl_weight 0.01) in place of mmd; seed 43 | 6 | off | 180 | 30 | 0.0-1.2 |
| 8 | hybrid_joint_tfk8_s43 | joint, the prior of r9's gfix_tf_k8: transformer, prior_fm_draws 8, local_inject coarsest, lambda_mmd_g 0; seed 43 | 7 | off | 180 | 30 | 0.0-1.2 |
