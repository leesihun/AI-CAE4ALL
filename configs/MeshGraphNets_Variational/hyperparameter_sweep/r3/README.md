# B300 round 3: ex3 shell buckling, 24 jobs on three 8-GPU nodes

One job per GPU, eight per node. Every job trains on
`dataset/probabilistic/ex3_shell_buckling.h5` (same split as before:
`split_seed 42`, grouped by `shape_idx`). It is then scored on the 8 held-out
geometries × 64 FE results (`spread_metrics.json`).

| node | command (from this directory) | what it runs |
| --- | --- | --- |
| 135 | `bash run_135.sh` | HI-MGN-V (`latent_type global`) × 7, SAR seed 42 |
| 136 | `bash run_136.sh` | HI-MGNflow (`latent_type local`) × 7, SAR seed 43 |
| 137 | `bash run_137.sh` | HI-MGN-hybrid (`latent_type hybrid`) × 7, deterministic HI-MGN |

The checks run in the terminal. When they pass, the jobs detach, so the
terminal can close. Results:
`python configs/MeshGraphNets_Variational/hyperparameter_sweep/r3/brief.py`.
[run_node.sh](run_node.sh) lists the overrides (`GPUS`, `JOBS`, `TRAIN=0`, …).
[jobs.txt](jobs.txt) is the job table. Job k of a node runs on card k.

## The seven variants of each placement

Each MGN-V job is the checked-in recipe in
`configs/MeshGraphNets_Variational/probabilistic/ex3/latent_type/` with one
change.

| job suffix | change from `<placement>_mmd` | question |
| --- | --- | --- |
| `mmd` | none: `latent_reg mmd`, `lambda_mmd 1`, seed 42 | reference |
| `mmd_s43` | `training_seed 43` | training-seed spread: how large a difference is noise? |
| `mmd_s44` | `training_seed 44` | (same, third seed) |
| `kl` | `latent_reg kl` (`kl_weight 1e-3`) | does the latent regularizer matter? |
| `mmd10` | `lambda_mmd 10` | less mean offset? (round 2: the only MGN-V change that helped) |
| `mmd0` | `lambda_mmd 0` | is a regularizer needed at all? |
| `long` | `training_epochs 240`, `prior_freeze_epoch 160` | is 120 epochs too short? (round 2: longer training calibrated HI-MGNflow) |

`long` takes about twice as long as the others.

## The eighth job of each node

| job | node | what it is | question |
| --- | --- | --- | --- |
| `sar_s42` | 135 | SAR (ported upstream code), `configs/SAR/probabilistic/ex3/` as checked in, seed 42 | the published one-to-many baseline on the same split and metrics |
| `sar_s43` | 136 | the same, seed 43 | SAR's own seed spread |
| `himgn_det` | 137 | `global_mmd` with the latent removed (`use_vae False`): one field per input | how far is a deterministic prediction from the 64 FE results? |

## Layout (changes nothing learned)

- `batch_size 8 × grad_accum_steps 1`, the recipe's effective batch of 8 in
  one step.
- `gpu_ids 0`; the runner sets `CUDA_VISIBLE_DEVICES`.
- One run-private hierarchy cache per node, deleted after training
  (`hierarchy_cache_keep False`).
- Outputs go to `output/b300_r3/<job>/`.

SAR keeps its checked-in batches (VAE 2 × 32, SAR 2 × 16).

Not measured yet: memory and time of SAR, `local` and `hybrid` on a B300.
Check the first epochs (`tail -f output/b300_r3/run_<node>.out`).

## The block to type back

```
R3 <node> <h>h D<done>/<jobs> F<failed>
<job> w1 sd bias crps skill h [flags]
```

Every number is ×100. `bias` is signed. `sd` is 100 when the generated width
is right. [brief.py](brief.py) explains the flags.
