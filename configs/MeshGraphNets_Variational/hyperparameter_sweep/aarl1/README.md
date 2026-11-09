# aarl round 1: HI-MGN-hybrid on ex3, ablations, then tuning

HI-MGN-hybrid is MGN-V with `latent_type hybrid`. This round runs on aarl (4× RTX 3090, a shared box) and may
hold **two cards at most** (0 and 1). Cards 2 and 3 belong to other users. [LOG.md](LOG.md) is the journal:
every decision, every result, and what each result changed.

## Run it

```bash
python configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/make_sweep.py --plan   # queue + hours
python configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/make_sweep.py          # writes configs, jobs.txt, run_aarl.sh
bash   configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/run_aarl.sh            # on aarl, from the copy in /data/Lee/mgnv_hx3/AI-CAE4ALL
python configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/brief.py               # the ranking, at any time
bash   configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/diagnose.sh            # posterior vs. prior on hybrid_ref, once it is done
LOOP=1800 bash configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/pull.sh        # LOCAL: copy output/aarl1 down every 30 min
```

All artifacts go to `output/aarl1/<job>/`. To stop the queue, `touch output/aarl1/stop_aarl`: running jobs
finish and no new job starts. `kill_all.sh` kills only this round's processes, by PID file and cwd.

## Score

`w2` (×100) is the RMS, over the 8 held-out geometries, of each geometry's 1-D Wasserstein-2 distance between
the predicted and FE distributions of max|u|, divided by that geometry's FE standard deviation. The held-out
file has 256 FE draws per geometry. Inference uses 4 draws per scene, so 1024 predicted draws per geometry.

- **27** is the floor: two halves of the FE draws scored against each other (19–36 per geometry).
- **100** is the exact FE mean with no spread.
- **Seed noise is about 11**: r6 `hybrid_ref` scored 88, 88 and 107 on seeds 42, 43 and 44. An arm in this
  round runs once, so a difference of 10 or less from the reference is not a result.

## What earlier rounds already settled

All rounds ran on B300 node 135. Round 6 used the recipe this round copies: batch 4, 120 epochs, coarsening
4096/256, `mp_per_level 4, 6, 8, 6, 4`.

| Round | Finding | What this round does with it |
| --- | --- | --- |
| r3–r5 | hybrid beats local beats global. Most of the error is a **mean offset** on unseen geometries, not a spread problem. Sample inflation above 1 hurts. | Hybrid only. Read each arm's per-geometry histograms for the offset, not just w2. |
| r6 | One knob at a time around `hybrid_ref` (88/88/107). Only 120 → 240 epochs (69) clearly helps. mmd100 71, ar10000 76, pm2 76, mmd0.1 76 and pm6 77 sit inside seed noise; where both directions of a knob "beat" the reference, that is noise. Temperature 0.5/2 hurts. | Hyperparameters are not re-swept one at a time at batch 4. |
| r7 | e480 at batch 16 scored 68; e240 at batch 8 scored 74. Large batches beat DDP. | Phase 2 trains longer. A 3090 cannot hold a batch of 16 (about 1M nodes). |
| r8 (running) | Batch 16: e360/e720, lr 2e-4, bs32, MMD 0.1–10000, KL 0.1–100, a bigger FM prior, 10/100 FM steps. | None of it is duplicated here. Phase 2 reads its results when they arrive. |

**What no round has done is remove a part of the method.** Every earlier arm changed a value. Phase 1 is
therefore the ablation table.

## Phase 1: ablations (batch 4, 120 epochs, seed 42)

Each arm removes one component of `hybrid_ref`. The reference is retrained here first, because the 3090 needs
gradient checkpointing. Checkpointing is exact in the forward pass, so the reference should land inside r6's
seed spread. If it does not, that is the first entry in LOG.md.

| Arm | Removes | Question |
| --- | --- | --- |
| `ref` | nothing | Does aarl reproduce r6 (88/88/107)? |
| `mmd0` | the latent regularizer (`lambda_mmd 0`) | Does MMD matter, or is the posterior std floor enough? r6 moved its weight ×0.1 and ×100 and both read as noise. |
| `aux0` | the peak-to-valley loss (`beta_aux 0`) | Does the max\|u\| statistic the score uses need its own loss? |
| `nofreeze` | the prior-only tail (`prior_freeze_epoch 0`) | Does training the prior alone on a fixed encoder for the last third help? |
| `noaug` | Z rotation and reflection (`augment_geometry False`) | Does augmentation help on unseen geometries (the mean offset)? |
| `concat` | AdaLN-Zero z injection (`z_conditioning concat`) | Is the conditioning mechanism the reason z is used? |
| `flatenc` | geometry in the posterior encoder (`vae_graph_aware False`) | Does the encoder need the mesh to read the imperfection? |
| `hv1` | the second coarsening per sample (`hierarchy_variants 1`) | Is hierarchy variation acting as augmentation? |
| `noema` | weight EMA (`use_ema False`) | Does EMA steady the score? |
| `nocond` | thickness and imperfection amplitude as inputs (`cond_var 0`) | How much of the spread do the two scalar conditions carry? |

Two more ablations need no training. On the finished `ref`, `diagnose.sh` decodes the 256 FE draws of each
held-out geometry four ways:

- posterior mean
- posterior sample
- learned-prior sample
- **z ~ N(0, I)**, which removes the learned conditional prior

The gap between posterior and prior shows how much of the w2 above the floor (27) comes from the prior and how
much from the encoder and decoder. The N(0, I) row shows what the learned prior is worth.

Not ablated:

- `use_deformed_edge`: a no-op on a static case (T = 1).
- `use_multiscale` and the FM prior: hybrid requires both.
- Removing the VAE: `ref_det` in r6 already did.
- Temperature: excluded at the user's request.

## Phase 2: tuning (240 epochs and up)

Phase 2 combines, at 240 epochs and above, the directions that held up:

- r6's e240 (69) and r7's e480 (68)
- whatever phase 1 shows matters
- the diagnosis: if the prior dominates, prior capacity and FM steps; if the encoder/decoder dominates,
  width, alpha_recon and the mean offset
- r8's batch-16 results, once they arrive

Each phase-2 arm is added to `TUNE` in make_sweep.py with a LOG.md entry naming the result that chose it.
Phase-2 winners are re-run on a second seed before anything is called better, because one seed carries about
11 points of noise.

## Budget

At most 2 cards. EP_H and PEAK_GB in make_sweep.py are measured on aarl with probe runs (LOG.md, 2026-11-07),
and `--plan` gives the makespan from them.

## Jobs (generated by make_sweep.py; everything below is rewritten)

10 trainings + 0 sampling-only jobs, seed 42, one run each; --pack 2: 2 per card on GPUs 0, 1 = 4 slots, 6 torch threads per job, gradient checkpointing on. Estimated makespan 8.4 days (batch 4 alone on a 3090, measured: 0.41 h per joint epoch, 0.17 h per prior-only epoch; x1.58 per job at 2 per card).

| # | phase | job | change from the reference | GPU | est. h | est. start-end (days) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | hybrid_ref | the reference: r6 hybrid_ref's recipe, retrained on aarl (3090, checkpointing) | 0 | 67 | 0.0-2.8 |
| 2 | 1 | hybrid_mmd0 | no latent regularizer: lambda_mmd 0 (only the posterior std floor is left) | 1 | 67 | 0.0-2.8 |
| 3 | 1 | hybrid_aux0 | no peak-to-valley loss on the decoded field: beta_aux 0 | 0 | 67 | 0.0-2.8 |
| 4 | 1 | hybrid_nofreeze | no prior-only tail: prior_freeze_epoch 0 (encoder, decoder and prior train together to the end) | 1 | 82 | 0.0-3.4 |
| 5 | 1 | hybrid_noaug | no geometry augmentation: augment_geometry False (no Z rotation / reflection) | 0 | 67 | 2.8-5.6 |
| 6 | 1 | hybrid_concat | z enters by concatenation instead of AdaLN-Zero: z_conditioning concat | 1 | 67 | 2.8-5.6 |
| 7 | 1 | hybrid_flatenc | the posterior encoder does not see the geometry: vae_graph_aware False | 0 | 67 | 2.8-5.6 |
| 8 | 1 | hybrid_hv1 | one coarsening per sample instead of two: hierarchy_variants 1 | 1 | 67 | 3.4-6.2 |
| 9 | 1 | hybrid_noema | no weight EMA: use_ema False | 0 | 67 | 5.6-8.4 |
| 10 | 1 | hybrid_nocond | the conditions (thickness, imperfection delta) are not inputs: cond_var 0 | 1 | 67 | 5.6-8.4 |
