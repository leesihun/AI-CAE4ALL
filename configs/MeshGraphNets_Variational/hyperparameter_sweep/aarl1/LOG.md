# aarl1 journal

Newest entry last. Each entry records what was done, what was measured, and what it changed. All times are KST.
Scores are w2×100: 27 is the floor, 100 is the FE mean with no spread, seed noise is about 11 (README.md).

## 2026-11-07: setup

**Request.** An ablation study and hyperparameter tuning of HI-MGN-hybrid (MGN-V, `latent_type hybrid`) on
ex3, run on aarl, built on the results of rounds r3–r8, using at most 2 GPUs at a time, with everything
recorded.

**aarl state (19:00).**

- 4× RTX 3090 24 GB, 80 cores, 251 GB RAM, load average about 38.
- Cards 2 and 3 run another user's LDGN jobs (root, PIDs 897743 and 905553, configs in
  `/data/Lee/lee_scratch/sarldgn/cfg/`). This round uses cards **0 and 1 only** and must never touch 2 or 3.
- Code is copied (tar, no git) to `/data/Lee/mgnv_hx3/AI-CAE4ALL`, apart from the other users' checkouts.
- Python is `/data/Lee/envs/mgn310/bin/python`.
- Datasets are in `dataset/probabilistic/`: `ex3_shell_buckling.h5`, plus `ex3_shell_buckling_infer.h5`
  with 256 draws per geometry.

**Design decisions.**

1. **Phase 1 is ablations, not hyperparameters.** r6, r7 and r8 already moved every scalar knob of the hybrid
   one at a time. Only more epochs was credible (r6 e240 69, r7 e480 bs16 68). No round has removed a part of
   the method. The arms are listed in README.md.
2. **The recipe is r6 `hybrid_ref`** (batch 4, 120 epochs, prior freeze at 80). That makes every arm
   comparable with r6's three reference seeds (88, 88, 107) and r6's single-knob table.
   `make_sweep.py` checks that the reference equals r6's config key for key, ignoring build-only keys and paths.
3. **Gradient checkpointing is on for every training.** Without it, batch 4 peaks at about 57 GB on the
   worst batch (329k nodes); a 3090 has 24 GB.
4. **The reference is retrained on aarl** instead of reusing r6's checkpoints. The hardware and checkpointing
   differ, and the posterior vs. prior diagnosis needs a checkpoint from this round.
5. **The learned prior is ablated without training.** `use_conditional_prior False` in an inference config
   does nothing, because the checkpoint's `model_config` overrides it (`inference_profiles/rollout.py`,
   `training_profiles/setup.py:254`). So `misc/posterior_vs_prior_groups.py` gained a fifth ensemble,
   `std_normal`, which decodes z ~ N(0, I) with the same structure as the prior draw. Its JSON keys are
   additive, so r8's `extend_135.sh` still reads the file.
6. **Dropped arms.**
   - `use_deformed_edge False`: a no-op on a static (T = 1) case.
   - No-multiscale and no-FM-prior: hybrid requires both.
   - `use_vae False`: r6 `ref_det` already ran it.
   - Temperature: excluded at the user's request.
   - `prior_freeze_epoch` must be 0 (never) or inside the run, and freezing works on one GPU only. Every job
     here uses one GPU.
7. **The scripts are r8's machinery, adapted** (`run_node.sh`, `brief.py`, `kill_all.sh`, `diagnose.sh`):
   node `aarl`, root `output/aarl1`, its own MPS pipe `/tmp/aarl1_mps_<uid>`, and the stop file
   `output/aarl1/stop_aarl`.
   - `kill_all.sh` kills by PID file and by cwd inside this checkout only. The other users' jobs run from
     `/data/Lee/lee_scratch`, outside it.

**Probe (19:07).** Two-epoch probe runs, `configs/aarl_probe/probe_bs{4,8}.txt` (aarl copy only), are
measuring the epoch time and peak VRAM with checkpointing at batch 4 (GPU 0) and batch 8 (GPU 1). The hierarchy
cache build (19,200 entries, 32 workers) takes about 26 minutes before the first epoch.

**Probe results (20:35).** Two epochs each, alone on a card, while the other user's jobs ran on cards 2–3:

| | batch 4 (GPU 0) | batch 8 (GPU 1) |
| --- | --- | --- |
| joint epoch, train pass | 1920 it × 0.76 s ≈ 24 min | 960 it × 1.38 s ≈ 22 min |
| prior-only epoch (`PriorFit`) | 1920 it at 3.3 it/s ≈ 10 min | 960 it at 1.8 it/s ≈ 9 min |
| CRPS validation (every 10 epochs) | 4.5 min | 4 min |
| VRAM peak / reserved | **9.37 / 10.0 GB** | 16.7 / 17.8 GB |
| start (cache build 26 min + normalization stats) | ~45 min | shared cache |

- Reserved is 1.07× the peak, because `expandable_segments` is on. The fragmentation growth up to 2.3×
  seen without it does not happen here.
- A batch-4, 120-epoch run alone on a card is about 41 h: 80 joint epochs × 0.41 h, 40 prior-only epochs
  × 0.17 h, plus validation and start.

**Diagnosis script smoke test.** `posterior_vs_prior_groups.py` ran on the bs4 probe checkpoint (2
geometries, 16 FE results each): it printed and wrote all five ensembles, the N(0, I) row included. The
numbers are meaningless after 2 epochs. The probe outputs and configs were then deleted from aarl.

**Decision: two jobs per card (`--pack 2`, MPS).** 2 × 10 GB fits a 24 GB card. With 4 slots, the plan is
7.5 days for phase 1, against 9.4 days at one job per card, assuming each job runs 1.4× slower when paired
(a guess). That 1.4× is measured in the first hours. If pairing gives less than ~1.15× the throughput of
one job per card, the round is regenerated at `--pack 1` and the jobs resume from their resume states.
Each job gets 6 torch threads and 4 DataLoader workers.

**Launch (20:41).** `run_aarl.sh` launched with `PYTHON`/`MGNV_PYTHON=/data/Lee/envs/mgn310/bin/python`
and `DETACH=1`. Environment, data and preflight passed: 20 of 20 configs OK, and the held-out file is the
256-draw one. The MPS daemon is on `/tmp/aarl1_mps_0`, driver PID 2337117. First wave: `hybrid_ref` and
`hybrid_aux0` on GPU 0, `hybrid_mmd0` and `hybrid_nofreeze` on GPU 1.

**Paired speed measured (21:46).** About 300–430 iterations into the first joint epoch, two jobs per card
under MPS:

| job | GPU | s/it |
| --- | --- | --- |
| ref | 0 | 1.24 |
| aux0 | 0 | 1.20 |
| mmd0 | 1 | 1.24 |
| nofreeze | 1 | 1.13 |

- Alone on a card a job runs at 0.76 s/it, so each job is about **1.58× slower** when paired.
- Per card that is **1.27× the throughput** of one job, above the 1.15× threshold, so `--pack 2` stays.
- Each card uses 19.4 GB.
- `SHARE[2]` in make_sweep.py is now 1.58 (measured). Replanned: phase 1 ends in about 8.4 days, around
  2026-11-15 to 16. The first wave (`ref`, `mmd0`, `aux0`) ends in about 2.8 days; `nofreeze` takes longer
  (3.4 days), because it never switches to the cheaper prior-only epochs.
- The ablation arms are active: `mmd0` logs λ = 0.0.

## 2026-11-09: status check (06:28)

- All four first-wave jobs are at epoch 48 of 120, about 0.65 h per joint epoch (32.4 h elapsed), which
  matches the plan. The driver is alive and there are no errors.
- They show the same VRAM peak in every epoch because all use `training_seed 42`, so they draw the same
  batch order.
- Reserved memory has crept up from 10.0 to 11.4 GB per job, leaving 22.4 of 24 GB used per card. That is a
  1.6 GB margin, so watch it.
- The ablations show in the training losses:
  - `mmd0`: its MMD term is 1.68, against 0.65 in the reference. With the weight at 0, the latent drifts
    away from N(0, I).
  - `aux0`: its peak-to-valley error is 2.0e-3, against 5e-4 in the reference. With the weight at 0, that
    error is no longer trained down.
- Expected ends: `ref`, `mmd0` and `aux0` about 2026-11-10 16:00 (inference included); `nofreeze` about
  2026-11-11 morning.
- Card 3 became free when the other user's job ended. The round stays on cards 0 and 1.

## 2026-11-09: round stopped at the user's request (15:30)

- The user said: "다 죽여. 지금 변경했어. 아직 안해도돼" (kill everything; they changed something; no need to
  run it yet).
- `kill_all.sh` signalled 79 processes. Before running it I checked that only this round's processes had
  their cwd in the checkout. The other user's jobs (905553 on card 2, 2888998 on card 3) are untouched and
  still running.
- Cleanup: MPS daemon gone, `/tmp/aarl1_mps_0` removed, the 6.5 GB hierarchy cache removed. Cards 0 and 1
  are empty.
- When stopped, `ref`, `aux0` and `mmd0` were at epoch 62 of 120 and `nofreeze` at 61. That is inside the
  joint phase, so no inference and no score.
- Their `model.pth` (best so far) and train logs are pulled to local `output/aarl1/`.
- The resume states (`*.resume`) are still on aarl. They resume only under the same config fingerprint, so
  they are useless if the change touches these configs.
- The local pull loop is stopped.
