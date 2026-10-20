# B300 round 4: ex3 shell buckling, 24 jobs on three 8-GPU nodes

One job per GPU, eight per node, no queue. Every job trains on
`dataset/probabilistic/ex3_shell_buckling.h5` with the same split as round 3
(`split_seed 42`, grouped by `shape_idx`), and is then scored on the 8 held-out
geometries × 64 FE results (`spread_metrics.json`, plus one
`histogram_compare` per geometry). After its inference, each job runs
[../plots.py](../plots.py), which draws the loss curves (every component plus
the total, train and validation) beside the epoch logs.

| node | command (from this directory) | what it runs |
| --- | --- | --- |
| 135 | `bash run_135.sh` | SAR × 6 (published configurations), HI-MGN-V `global_zs`, `global_zl` |
| 136 | `bash run_136.sh` | LDGN × 6 (published configurations), HI-MGN-V `global_prior2x`, `global_pmin` |
| 137 | `bash run_137.sh` | HI-MGNflow (`local`) × 4, HI-MGN-hybrid × 4: `zs`, `zl`, `prior2x`, `pmin` |

SAR and LDGN stay on 135 and 136, the two nodes whose SAR and LDGN venvs round 3
already used. The checks run in the terminal. When they pass, the jobs detach,
so the terminal can close. Results:
`python configs/MeshGraphNets_Variational/hyperparameter_sweep/r4/brief.py`.
[run_node.sh](run_node.sh) lists the overrides (`GPUS`, `JOBS`, `TRAIN=0`, …).
[jobs.txt](jobs.txt) is the job table. Job k of a node starts on card k.

## MGN-V: four changes per placement

Each MGN-V job is round 3's `<placement>_mmd` (`r3/config_*_<placement>_mmd.txt`,
with the same batch 4 × 1, `use_checkpointing`, 120 epochs, and seed 42) with
one change. The change is in both the train and the infer config, because
inference rebuilds the model from the config. `global` runs every change on
135/136 and `local` and `hybrid` run them on 137, so each change can be read
across the three placements.

| job suffix | change from `<placement>_mmd` | question |
| --- | --- | --- |
| `zs` | latent 4× smaller: `vae_latent_dim 32 → 8` (global), `latent_ch 4 → 1` (local), both (hybrid) | is z over-sized? |
| `zl` | latent 4× larger: `vae_latent_dim 32 → 128`, `latent_ch 4 → 16` | does a richer z widen the per-geometry spread? |
| `prior2x` | `prior_hidden_dim 128 → 256`, `prior_mp_layers 3 → 6`, `prior_fm_blocks 4 → 8` | is the conditional prior the limit? (round 3: every placement too narrow per geometry, skill ≈ 0.4) |
| `pmin` | `posterior_min_std 0.05 → 0.2` | a rate limit on z by noise instead of size: does calibration change? |

The round-3 `<placement>_mmd` runs are the reference for all four, so round 4 has
no reference job of its own.

## SAR: the published Wing configurations

Base: round 3's `sar_s42` (`configs/SAR/hyperparameter_sweep/r3/`), which is
the upstream `examples/Wing/training_*.py` script (SAR-3x8: 3 scales, 8/8/8
layers, 64 slices) with seed 42 and `draws_per_epoch 1`. The upstream script
says the VAE has 2 layers, but the released Wing VAE checkpoint has 4 layers,
every released Wing SAR loads that VAE, and paper App. A.2 says four. So
`sar_wing` is the first arm and the others build on it.

| job | change | source | question |
| --- | --- | --- | --- |
| `sar_wing` | `vae_num_layers 2 → 4`; VAE batch `2 × 32 → 1 × 64` | released Wing checkpoints, paper App. A.2 | the published SAR row |
| `sar_wing_d8` | `sar_wing` + `draws_per_epoch 1 → 8` | Wing epoch length (≈1,000 items, read from the checkpoints' `global_step / epoch`) | does SAR stop too early at ex3's short epochs? |
| `sar_i5` | `sar_wing` + `sar_num_iters 5`, `sample_steps 20, 16, 11, 6, 2` | released `sar-Wing-i5-8_8_8` | does a coarser top scale capture the buckling-mode choice? |
| `sar_3x4` | `sar_wing` + cond/AR/sampler layers `8 → 4` | Wing SAR-3x4, paper Table 3 | is 5.25M parameters too many for about 120 geometries? |
| `sar_3x4_n0` | `sar_3x4` + `sar_noise_std 0.01 → 0` | released `sar-Wing-i3-4_4_4-n0` | does the coarse-to-fine training noise matter? |
| `sar_4_8_8` | `sar_wing` + `sar_num_cond_layers 8 → 4` | released `sar-Wing-i3-4_8_8` | is condition-encoder depth wasted? |

- **VAE batch 1 × 64.** The VAE is not checkpointed (`use_checkpointing` wraps
  only the Transolver blocks), so a 4-layer VAE roughly doubles its activation
  memory. 1 × 64 keeps the effective batch of 64.
- **`sar_wing_d8`** sees 8 draws per geometry per epoch. Its patience
  therefore means as many optimizer steps as upstream. It costs up to about
  8× the wall clock of `sar_wing`, possibly days.
- **Not run:**
  - The Ellipse configuration. At ex3's epoch length its patience of 50/20
    epochs is about 80 optimizer steps; it needs `draws_per_epoch` ≈ 48.
  - The ablations that need new code: no condition encoder (the spec rejects
    `sar_num_cond_layers 0`), the non-latent SAR, the ResNet sampler, and
    RandomizeScales.

## LDGN: the published dataset recipes

Base: round 3's `ldgn_s42` (`configs/LDGN/hyperparameter_sweep/r3/`), which is
the upstream Wing recipe as checked in. The paper publishes no LDGN
hyperparameter sweep, only one recipe per dataset (Ellipse, EllipseFlow, Wing),
so the arms are those recipes, the Wing recipe's own discrepancies, and seeds.

| job | change | source | question |
| --- | --- | --- | --- |
| `ldgn_wing_s44` | `training_seed 44` | Wing | a third seed beside round 3's 42 and 43 |
| `ldgn_wing_long` | epoch caps `3000 / 5000 → 5000 / 50000` | upstream Wing caps | do the port's caps cut training short? (the released Wing AE ran 4415 epochs; the released LDGN was still at lr 1e-6 at epoch 4006) |
| `ldgn_wing_ae200` | `vae_lr_patience 250 → 200` | released Wing AE checkpoint (the script says 250) | minor: the released autoencoder's schedule |
| `ldgn_eflow` | `ldgn_depths 1, 2, 2`, `vae_norm_latents False`, `rel_pos_ratios 1, 2, 4, 8, 16`, patience 50 / 50 | EllipseFlow | how deep does the denoiser need to be (Wing 6 levels, EllipseFlow 5)? |
| `ldgn_eflow_s43` | `ldgn_eflow` + `training_seed 43` | EllipseFlow | seed spread of that recipe |
| `ldgn_ellipse` | `ldgn_depths 1, 2`, `vae_fnns_width 128`, `rel_pos_ratios 1, 3, 7.5, 15`, patience 50 / 50, batch `2 × 32` (upstream 64) | Ellipse | does a shallow denoiser (4 levels) still find the buckling mode? |

- The number of levels is `len(vae_depths) + len(ldgn_depths) − 1`, so
  `rel_pos_ratios` has 4, 5 or 6 entries.
- The checkpoint stores the ratios, so the infer configs do not repeat them.
- `ldgn_diffusion_steps` stays at 1000 in every arm.
- Like round 3, `ldgn_eflow` and `ldgn_ellipse` stop on the learning-rate
  threshold. Their patience of 50 epochs is short at ex3's epoch length, about
  120 items per epoch with 2 × 16 accumulation, so expect them to end early.
  That is part of what they test.

## Free extras after training (not jobs)

Sampling steps are inference-only. They can be swept on any finished
checkpoint by copying its infer config and changing `sample_steps` (and
`inference_output_dir`):

- SAR: `20, 11, 2` / `15, 8, 2` / `10, 6, 1` / `5, 3, 1` / `3, 2, 1` (paper Figs 11/12).
- LDGN: `20` / `50` / `100` / `1000` (paper Fig. 19b axis).
- MGN-V: `prior_temperature`.

## Layout

- The MGN-V layout is unchanged from round 3: `batch_size 4 × grad_accum_steps 1`
  with `use_checkpointing True`. There is one run-private hierarchy cache per
  node, at `output/b300_r4/mscache/`, and it is deleted after training.
- `gpu_ids 0`; the runner sets `CUDA_VISIBLE_DEVICES`.
- Outputs go to `output/b300_r4/<job>/`. Loss plots are written as
  `loss_<log>.png` beside each epoch log; the plotting stdout goes to
  `output/b300_r4/run_logs/<node>/<job>.plots.out`. A plotting failure does
  not fail the job.
- **Not measured yet:** memory and time on a B300 for `zl` and `prior2x`, the
  4-layer SAR VAE, `sar_wing_d8`, and every LDGN recipe other than Wing. Watch
  the first epochs (`tail -f output/b300_r4/run_<node>.out`).

## The block to type back

```
R4 <node> <h>h D<done>/<jobs> F<failed>
<job> w1 sd bias crps skill h [flags]
```

Every number is ×100. `bias` is signed. `sd` is 100 when the generated width
is right. [brief.py](brief.py) explains the flags.
