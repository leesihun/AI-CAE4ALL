# B300 round 4: ex3 shell buckling, 7 jobs on node 135's 7 cards

Round 4 replaces round 3; stop round 3 before starting it. Only node 135 is
available, without its GPU1, so the run has 7 cards (0, 2–7): one job per card,
for SAR, LDGN and the three latent placements (global, local, hybrid). SAR and
LDGN run with our target scaling (`target_scaling standard`, see
[SAR and LDGN](#sar-and-ldgn)); the upstream default is `minmax`.

Every job trains on `dataset/probabilistic/ex3_shell_buckling.h5` with round 3's
split (`split_seed 42`, grouped by `shape_idx`). It is then scored on the 8
held-out geometries × 64 FE results (`spread_metrics.json`, plus one
`histogram_compare` per geometry). After inference, each job runs
[../plots.py](../plots.py), which draws the loss curves (every component plus
the total, train and validation) beside the epoch logs.

```
bash run_135.sh        # from this directory
```

The checks run in the terminal. When they pass, the jobs detach, so the
terminal can close. To see results, run
`python configs/MeshGraphNets_Variational/hyperparameter_sweep/r4/brief.py`.
[run_node.sh](run_node.sh) lists the overrides (`GPUS`, `JOBS`, `TRAIN=0`, …).

**Beside a run that is still going**, start a second one with `LANE=b`. It has
its own `run_135_b.out` and pid file, and it leaves the first run's hierarchy
cache and run hours alone:

```
LANE=b GPUS="0 2" JOBS="sar_wing_std ldgn_wing_std" bash run_135.sh
```

## The 7 jobs

[jobs.txt](jobs.txt) lists them; job k starts on the k-th card.

| # | job | card | question |
| --- | --- | --- | --- |
| 1 | `sar_wing_std` | GPU 0 | the SAR row: the Wing recipe with z-scored targets |
| 2 | `ldgn_wing_std` | GPU 2 | the LDGN row: the Wing recipe with z-scored targets |
| 3 | `global_mmd` | GPU 3 | HI-MGN-V reference |
| 4 | `local_mmd` | GPU 4 | HI-MGNflow reference |
| 5 | `hybrid_mmd` | GPU 5 | HI-MGN-hybrid reference |
| 6 | `local_long` | GPU 6 | is 120 epochs too short? (round 2: longer training calibrated this placement) |
| 7 | `hybrid_long` | GPU 7 | is 120 epochs too short? |

**Rough wall clock**, from round 3's hours with 8 jobs per node (120 epochs:
global about 15 h, local about 12 h; `long` about twice that). Hybrid, SAR and
LDGN have not been timed. On that basis the 120-epoch jobs end in about
half a day and the two `long` jobs in about a day and a half.

## Later jobs (commented out in jobs.txt)

Their configs exist and pass `--check`. Each line sits at the end of
[jobs.txt](jobs.txt) behind a `# `. To run some of them, delete the `# ` of
their lines and name them, or the 7 jobs above train again:

```
JOBS="himgn_det global_long" bash run_135.sh
```

With more active jobs than cards, the runner deals the first 7 and then gives
each card that finishes the next job, in file order.

| job | question |
| --- | --- |
| `himgn_det` | how far is one deterministic field from the 64 FE results? |
| `global_long` | is 120 epochs too short? (global's limit is the encoder) |
| `hybrid_mmd_s43`, `global_mmd_s43`, `local_mmd_s43` | second seed: how large a difference is noise? |
| `global_zl`, `local_ch1` | capacity control (see below) |
| `hybrid_ch1`, `hybrid_prior2x` | does hybrid use its global part; is the joint prior's capacity the limit? |
| `sar_3x4_std` | is SAR's 5.25M parameters too many for about 120 geometries? |
| `ldgn_eflow_std` | does LDGN need a 6-level denoiser (EllipseFlow: 5)? |
| `sar_wing`, `ldgn_wing` | the Wing recipes with upstream min/max scaling (the first round-4 launch ran these) |

**To end a queue early**, run `touch output/b300_r4/stop_135` from the repo
root. Running jobs finish, and no new job starts. The next `bash run_135.sh`
deletes the file. To kill everything now, use the PID procedure that stopped
round 3.

## Why these arms

Each arm tests a cause we have already measured, or measures the noise.

- **The regularizer does nothing.** In round 3, `kl`, `mmd10` and `mmd0` all
  fell inside the seed range (5–8 crps points) for both global and local. They
  are dropped. Every arm uses `latent_reg mmd`, `lambda_mmd 1`.
- **Global loses amplitude before the prior.** Measured on ex3 earlier
  (2026-10-02 to 10-05):
  - The posterior mean is already 0.56–0.81× the FE amplitude, and the prior
    matches the posterior.
  - A 4× larger z changed nothing.
  - A z optimized per draw through the frozen decoder recovers 0.87–0.94×, so
    the encoder's single pooled vector is the limit.
  - So the global arms have no prior or regularizer changes. Their extra arm is
    the capacity control below.
- **Local decodes well but its prior is too narrow per geometry.**
  - Its posterior reconstructs amplitude (mean 1.00, sd 0.99).
  - The prior's sd ratio is about 0.5–0.6. Each coarse node's marginal is
    right; what is missing is the coordination across nodes.
  - A 4× deeper local prior (4 → 16 blocks) barely helped.
  - Hybrid's joint prior, with a per-graph token that reads and writes every
    node, is the structural fix. The hybrid arms test it.
- **Longer training calibrated the local placement in round 2**
  (spread_skill .56 → .99). Round 3's `long` arms did not finish, so each
  placement runs `long` again.
- **Two seeds per placement** (the later `*_s43` jobs) give the paper's
  placement table its own noise estimate.

## MGN-V arms

The base is round 3's config of the same name (`r3/config_*_<arm>.txt`), with
the same batch 4 × 1, `use_checkpointing`, 120 epochs and seed 42 unless the
change says otherwise. Every change is in both the train and the infer config,
because inference rebuilds the model from the config.

| job | change from round 3 |
| --- | --- |
| `global_mmd`, `local_mmd`, `hybrid_mmd` | none |
| `*_mmd_s43` | `training_seed 43` |
| `*_long` | `training_epochs 240`, `prior_freeze_epoch 160` |
| `global_zl` | `vae_latent_dim 32 → 128` (3 slots × 128 = 384 latent numbers) |
| `local_ch1` | `latent_ch 4 → 1` (about 256 numbers instead of 1024) |
| `hybrid_ch1` | `latent_ch 4 → 1`; the global part stays 3 × 32 |
| `hybrid_prior2x` | `prior_hidden_dim 128 → 256`, `prior_mp_layers 3 → 6`, `prior_fm_blocks 4 → 8` |
| `himgn_det` | none (`global_mmd` with `use_vae False`) |

**How to read the capacity control.** Local has about 10× more latent numbers
than global (1024 vs 96). If `local_ch1` (about 256) still beats `global_zl`
(384), then the placement decides the score, not the size of z.

## SAR and LDGN

**Target scaling.** `target_scaling` picks how the generated field (ux, uy,
uz) is scaled over the train split. `minmax`, upstream's `ScaleAttr` to
[-1, 1], is **the default**. `standard`, per-channel mean and standard
deviation, is our deviation, and every SAR/LDGN job we run uses it (the `_std`
jobs). The reason: in the first round-4 launch, `ldgn_wing`'s VGAE drew ux and
uy as flat lines (train and validation reconstruction about 8.9e-3, KL about
6), while uz fit. Under min/max the lateral draw-to-draw variation is a small
share of the loss. Both decoders end in a linear layer, so z-scores outside
[-1, 1] are reachable. The choice is stored in the VAE checkpoint, and the
infer configs repeat it, so a checkpoint trained the other way is refused.
`output/b300_r4/sar_wing/` and `ldgn_wing/` keep that first launch's min/max
runs.

**SAR.** The base is round 3's `sar_s42` (`configs/SAR/hyperparameter_sweep/r3/`),
the upstream `examples/Wing/training_*.py` script (SAR-3x8: 3 scales, 8/8/8
layers, 64 slices).

| job | change | source |
| --- | --- | --- |
| `sar_wing` | `vae_num_layers 2 → 4`; VAE batch `2 × 32 → 1 × 64` | released Wing checkpoints, paper App. A.2 (the script says 2 layers; the released VAE has 4) |
| `sar_wing_std` | `sar_wing` + `target_scaling standard` | ours (see above) |
| `sar_3x4_std` | `sar_wing_std` + cond/AR/sampler layers `8 → 4` | Wing SAR-3x4, paper Table 3 |

The VAE is not checkpointed (`use_checkpointing` wraps only the Transolver
blocks), so a 4-layer VAE roughly doubles its activation memory. A batch of
1 × 64 keeps the effective batch at 64.

**LDGN.** The base is round 3's `ldgn_s42` (`configs/LDGN/hyperparameter_sweep/r3/`),
the upstream Wing recipe as checked in. The paper publishes one recipe per
dataset, not a sweep.

| job | change | source |
| --- | --- | --- |
| `ldgn_wing` | none (seed 42) | Wing |
| `ldgn_wing_std` | `target_scaling standard` | ours (see above) |
| `ldgn_eflow_std` | `ldgn_wing_std` + `ldgn_depths 1, 2, 2`, `vae_norm_latents False`, `rel_pos_ratios 1, 2, 4, 8, 16`, patience 50 / 50 | EllipseFlow |

`ldgn_diffusion_steps` stays at 1000. The checkpoint stores `rel_pos_ratios`,
so the infer configs do not repeat them.

## Free extras after training (not jobs)

Sampling settings are inference-only. To sweep them on a finished checkpoint,
copy its infer config and change the key (and `inference_output_dir`):

- SAR: `sample_steps` `20, 11, 2` / `15, 8, 2` / `10, 6, 1` / `5, 3, 1` / `3, 2, 1` (paper Figs 11/12).
- LDGN: `sample_steps` `20` / `50` / `100` / `1000` (paper Fig. 19b axis).
- MGN-V: `prior_temperature`. The per-geometry width was too narrow in round 3,
  so a temperature above 1 is the cheapest test.

## Not in round 4 at all

These were in the 23-job plan and have no configs now. Each would take a card
for a day or more:

- **A third seed for hybrid** (`hybrid_mmd_s44`).
- **`sar_wing_d8`** (8 draws per epoch, the upstream epoch length). It can take
  up to 8× the wall clock of `sar_wing`.
- **`sar_i5`** (5 scales).
- **`ldgn_wing_long`** (upstream caps 5000 / 50000).
- **`ldgn_ellipse`** (4 levels).
- **The regularizer arms.** They had no measurable effect in round 3.

To add one, generate its configs and append its line to `jobs.txt`.

## Layout

- The MGN-V layout is unchanged from round 3: `batch_size 4 × grad_accum_steps 1`
  with `use_checkpointing True`. The node has one run-private hierarchy cache
  at `output/b300_r4/mscache/`. Each job rebuilds it if it is missing and
  deletes it after training, unless another job still has it open.
- `gpu_ids 0`; the runner sets `CUDA_VISIBLE_DEVICES`.
- Outputs go to `output/b300_r4/<job>/`. Loss plots are written as
  `loss_<log>.png` beside each epoch log. The plotting stdout goes to
  `output/b300_r4/run_logs/<node>/<job>.plots.out`. A plotting failure does
  not fail the job.
- **Not measured yet:** memory and time on a B300 for `global_zl`,
  `hybrid_prior2x`, the 4-layer SAR VAE, and both LDGN recipes. Target scaling
  changes neither.
  Watch the first epochs (`tail -f output/b300_r4/run_135.out`).

## The block to type back

```
R4 <node> <h>h D<done>/<jobs> F<failed>
<job> w2 b s shape sdg crps h [flags]
```

Every number is ×100. The ranking column is `w2`: per held-out geometry, the
W2 distance between the 64 draws and the 64 FE results over that geometry's
FE sd, then the root mean square over the 8 geometries. It splits exactly as
w2² = b² + s² + shape² (mean offset, width error, the rest); `sdg` is 100 when
the width is right and below 100 when too narrow. [../plots.py](../plots.py)
writes these (`infer/spread_w2.json`) after each job; for a job that finished
before it could, run `python ../plots.py r4` with a method's Python first.
[brief.py](brief.py) explains the flags.
