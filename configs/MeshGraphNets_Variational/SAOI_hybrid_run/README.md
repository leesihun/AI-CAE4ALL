# HI-MGN-hybrid SAOI

The same job as [../SAOI_run](../SAOI_run/README.md) — two board halves,
train then infer on three evaluation sets — with the model swapped for the
**HI-MGN-hybrid base** (`latent_type hybrid`) of the ex3 B300 rounds. Run it
from the repository root:

```bash
bash configs/MeshGraphNets_Variational/SAOI_hybrid_run/run_all.sh
```

Everything lands under:

```text
output/meshgraphnets-v/saoi_hybrid_run/
    bot.pth  top.pth         checkpoints
    bot.log  top.log         training logs
    infer/<half>/<eval set>/ rollout metrics, spread_values.npz, histogram PNG
    run_logs/                per-stage launcher logs
```

`run_all.sh` is `../SAOI_run/run_all.sh` with the paths changed, and it calls
`../SAOI_run/check_eval_inputs.py` for the infer/compare pairing check (the
eval files are the same). Same overrides: `ARMS`, `INFER_TAGS`, `STAGGER`,
`PYTHON`, `PREFLIGHT`, `STRICT_PREFLIGHT`, `EVAL_PREFLIGHT`, `TRAIN`, `INFER`.

**GPUs:** bot on 0, top on 1 — the same cards as `SAOI_run`. Change `gpu_ids`
in the configs before running both at once.

## What comes from where

| From | Keys |
|---|---|
| r10 `hybrid_base` (= r8 `hybrid_e360`, r9 `hybrid_base_s43`) | every model and training key except the schedule: batch 16, `voronoi_clusters 4096, 256`, `mp_per_level 4,6,8,6,4`, `latent_type hybrid`, `latent_ch 4`, `vae_latent_dim 32`, encoder 3 / local 2 layers, prior 3 × 128, MMD, `beta_aux 10`, `weight_decay 1e-4`, `std_noise 0`, `hierarchy_variants 2`, `hierarchy_seed 42`, `best_by crps` |
| every MGN-V SAOI run (`SAOI_run`, `SAOI_sweep`, `SAOI_sweep3`, B300 `hyperparameter_sweep/SAOI`) | the schedule: `training_epochs 1000`, `prior_freeze_epoch 700` |
| `SAOI_run` | the data contract and the run identity: dataset/eval paths, `cond_var 3`, `feature_loss_weights 0.1, 0.1, 1`, `use_node_types False`, `split_seed 42`, `training_seed 20260916`, `inference_samples 2000`, the spread statistic (native default, peak-to-valley) |

## Against SAOI_run

| | SAOI_run | SAOI_hybrid_run |
|---|---|---|
| latent | global, `vae_latent_dim 16` | hybrid: z_g 32 + z_l 4 ch on 256 coarse nodes |
| coarsening | 1000, 100; 4 variants | 4096, 256; 2 variants |
| VAE encoder / prior | 5 layers / 5 × 256 | 3 layers / 3 × 128 |
| schedule | 1000 epochs, freeze 700 | same |
| input noise | `std_noise 0.01` | 0.0 |
| inference hierarchy | `hierarchy_seed 0,1,2,3` (rotated) | 42 |

The schedule is the SAOI one, not the ex3 hybrid one (360 / 240), so the two
SAOI runs differ only in the model.

The 4096/256 coarsening has not been run on SAOI. SAOI meshes are larger than
ex3's, so the node count is not the limit, but batch 16 with this hierarchy on
one card has not been measured; if it OOMs, lower `batch_size` before the
coarsening, since the coarse grid is what the local latent lives on.
