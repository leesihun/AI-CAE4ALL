# B300 round 4: ex3 shell buckling, 23 jobs on three nodes

Round 4 replaces round 3; stop round 3 before starting it. One job per card,
no queue. Node 135 leaves out its GPU1 (the card where round 3's
`global_mmd_s43` died), so it runs 7 jobs on cards 0, 2–7.

Every job trains on `dataset/probabilistic/ex3_shell_buckling.h5` with round 3's
split (`split_seed 42`, grouped by `shape_idx`). It is then scored on the 8
held-out geometries × 64 FE results (`spread_metrics.json`, plus one
`histogram_compare` per geometry). After inference, each job runs
[../plots.py](../plots.py), which draws the loss curves (every component plus
the total, train and validation) beside the epoch logs.

| node | command (from this directory) | what it runs |
| --- | --- | --- |
| 135 | `bash run_135.sh` | SAR × 4 (published configurations), `hybrid_ch1`, `hybrid_prior2x`, `himgn_det` |
| 136 | `bash run_136.sh` | LDGN × 4 (published recipes), `global_mmd`, `global_mmd_s43`, `global_long`, `global_zl` |
| 137 | `bash run_137.sh` | `local_mmd`, `local_mmd_s43`, `local_long`, `local_ch1`, `hybrid_mmd`, `hybrid_mmd_s43`, `hybrid_mmd_s44`, `hybrid_long` |

SAR and LDGN stay on 135 and 136, the two nodes whose SAR and LDGN venvs round 3
already used. The checks run in the terminal. When they pass, the jobs detach,
so the terminal can close. To see results, run
`python configs/MeshGraphNets_Variational/hyperparameter_sweep/r4/brief.py`.
[run_node.sh](run_node.sh) lists the overrides (`GPUS`, `JOBS`, `TRAIN=0`, …).
[jobs.txt](jobs.txt) is the job table. Job k of a node starts on the node's k-th card.

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
- **Two seeds per placement** give the paper's placement table its own noise
  estimate. Hybrid gets a third seed: it is the newest model, and whether its
  decoder uses the global part may depend on the seed.

## MGN-V arms

The base is round 3's config of the same name (`r3/config_*_<arm>.txt`), with
the same batch 4 × 1, `use_checkpointing`, 120 epochs and seed 42 unless the
change says otherwise. Every change is in both the train and the infer config,
because inference rebuilds the model from the config.

| job | change from round 3 | question |
| --- | --- | --- |
| `global_mmd`, `local_mmd`, `hybrid_mmd` | none | reference of each placement |
| `*_mmd_s43`, `hybrid_mmd_s44` | `training_seed 43` / `44` | how large a difference is noise? |
| `global_long`, `local_long`, `hybrid_long` | `training_epochs 240`, `prior_freeze_epoch 160` | is 120 epochs too short? |
| `global_zl` | `vae_latent_dim 32 → 128` (3 slots × 128 = 384 latent numbers) | capacity control, with `local_ch1` |
| `local_ch1` | `latent_ch 4 → 1` (about 256 numbers instead of 1024) | capacity control; does a smaller coarse latent let the prior keep the nodes coherent? |
| `hybrid_ch1` | `latent_ch 4 → 1`; the global part stays 3 × 32 | with less room in the local part, does the decoder use the global part? |
| `hybrid_prior2x` | `prior_hidden_dim 128 → 256`, `prior_mp_layers 3 → 6`, `prior_fm_blocks 4 → 8` | is the joint prior's capacity the limit on per-geometry width? |
| `himgn_det` | none (`global_mmd` with `use_vae False`) | how far is one deterministic field from the 64 FE results? |

**How to read the capacity control.** Local has about 10× more latent numbers
than global (1024 vs 96). If `local_ch1` (about 256) still beats `global_zl`
(384), then the placement decides the score, not the size of z.

## SAR: the published Wing configurations

Base: round 3's `sar_s42` (`configs/SAR/hyperparameter_sweep/r3/`), which is
the upstream `examples/Wing/training_*.py` script (SAR-3x8: 3 scales, 8/8/8
layers, 64 slices) with seed 42 and `draws_per_epoch 1`. The upstream script
says the VAE has 2 layers. But the released Wing VAE checkpoint has 4 layers,
every released Wing SAR loads that VAE, and paper App. A.2 says four. So
`sar_wing` is the reference and the others build on it.

| job | change | source | question |
| --- | --- | --- | --- |
| `sar_wing` | `vae_num_layers 2 → 4`; VAE batch `2 × 32 → 1 × 64` | released Wing checkpoints, paper App. A.2 | the published SAR row |
| `sar_wing_d8` | `sar_wing` + `draws_per_epoch 1 → 8` | Wing epoch length (about 1,000 items, read from the checkpoints' `global_step / epoch`) | does SAR stop too early at ex3's short epochs? |
| `sar_i5` | `sar_wing` + `sar_num_iters 5`, `sample_steps 20, 16, 11, 6, 2` | released `sar-Wing-i5-8_8_8` | does a coarser top scale capture the buckling-mode choice? |
| `sar_3x4` | `sar_wing` + cond/AR/sampler layers `8 → 4` | Wing SAR-3x4, paper Table 3 | is 5.25M parameters too many for about 120 geometries? |

- **VAE batch 1 × 64.** The VAE is not checkpointed (`use_checkpointing` wraps
  only the Transolver blocks), so a 4-layer VAE roughly doubles its activation
  memory. 1 × 64 keeps the effective batch at 64.
- **`sar_wing_d8`** sees 8 draws per geometry per epoch. Its patience
  therefore means as many optimizer steps as upstream. It can take up to about
  8× the wall clock of `sar_wing`, possibly days.
- **Not run:**
  - `sar-Wing-i3-4_4_4-n0` and `-4_8_8`, which are small variations of
    `sar_3x4`.
  - The Ellipse configuration. At ex3's epoch length its patience of 50/20
    epochs is about 80 optimizer steps.
  - The ablations that need new code.

## LDGN: the published dataset recipes

Base: round 3's `ldgn_s42` (`configs/LDGN/hyperparameter_sweep/r3/`), which is
the upstream Wing recipe as checked in. The paper publishes no LDGN
hyperparameter sweep, only one recipe per dataset (Ellipse, EllipseFlow,
Wing). So the arms are those three recipes plus upstream's longer Wing caps.

| job | change | source | question |
| --- | --- | --- | --- |
| `ldgn_wing` | none (seed 42) | Wing | the published LDGN row |
| `ldgn_wing_long` | epoch caps `3000 / 5000 → 5000 / 50000` | upstream Wing caps | do the port's caps cut training short? (the released Wing AE ran 4415 epochs; the released LDGN was still at lr 1e-6 at epoch 4006) |
| `ldgn_eflow` | `ldgn_depths 1, 2, 2`, `vae_norm_latents False`, `rel_pos_ratios 1, 2, 4, 8, 16`, patience 50 / 50 | EllipseFlow | how deep does the denoiser need to be (Wing 6 levels, EllipseFlow 5)? |
| `ldgn_ellipse` | `ldgn_depths 1, 2`, `vae_fnns_width 128`, `rel_pos_ratios 1, 3, 7.5, 15`, patience 50 / 50, batch `2 × 32` (upstream 64) | Ellipse | does a shallow denoiser (4 levels) still find the buckling mode? |

- The number of levels is `len(vae_depths) + len(ldgn_depths) − 1`, so
  `rel_pos_ratios` has 4, 5 or 6 entries.
- The checkpoint stores the ratios, so the infer configs do not repeat them.
- `ldgn_diffusion_steps` stays at 1000 in every arm.
- `ldgn_eflow` and `ldgn_ellipse` stop on the learning-rate threshold. Their
  patience of 50 epochs is short at ex3's epoch length, so expect them to end
  early. That is part of what they test.

## Free extras after training (not jobs)

Sampling settings are inference-only. To sweep them on a finished checkpoint,
copy its infer config and change the key (and `inference_output_dir`):

- SAR: `sample_steps` `20, 11, 2` / `15, 8, 2` / `10, 6, 1` / `5, 3, 1` / `3, 2, 1` (paper Figs 11/12).
- LDGN: `sample_steps` `20` / `50` / `100` / `1000` (paper Fig. 19b axis).
- MGN-V: `prior_temperature`. The per-geometry width was too narrow in round 3,
  so a temperature above 1 is the cheapest test.

## What round 4 does not cover

- **A third seed for global and local.** Round 3's finished seeds (global 2,
  local 3) give the spread: 5–8 crps points.
- **The regularizer arms.** They had no measurable effect in round 3.

## Layout

- The MGN-V layout is unchanged from round 3: `batch_size 4 × grad_accum_steps 1`
  with `use_checkpointing True`. Each node has one run-private hierarchy cache
  at `output/b300_r4/mscache/`, deleted after training.
- `gpu_ids 0`; the runner sets `CUDA_VISIBLE_DEVICES`.
- Outputs go to `output/b300_r4/<job>/`. Loss plots are written as
  `loss_<log>.png` beside each epoch log. The plotting stdout goes to
  `output/b300_r4/run_logs/<node>/<job>.plots.out`. A plotting failure does
  not fail the job.
- **Not measured yet:** memory and time on a B300 for `global_zl`,
  `hybrid_prior2x`, the 4-layer SAR VAE, `sar_wing_d8`, and every LDGN recipe.
  Watch the first epochs (`tail -f output/b300_r4/run_<node>.out`).

## The block to type back

```
R4 <node> <h>h D<done>/<jobs> F<failed>
<job> w1 sd bias crps skill h [flags]
```

Every number is ×100. `bias` is signed. `sd` is 100 when the generated width
is right. [brief.py](brief.py) explains the flags.
