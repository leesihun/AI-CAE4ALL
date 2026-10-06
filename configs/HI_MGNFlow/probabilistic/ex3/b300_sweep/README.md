# cHI-MGNflow ex3 sweep (one 8-GPU B300 node)

Eight arms on ex3 shell buckling, one GPU each: train (AE, then prior) and then
score on the held-out geometries. Every arm is the ex3 baseline
(`configs/HI_MGNFlow/probabilistic/ex3/baseline/`) plus the B300 layout and
the changes listed below. Each config's `%` header names its change and its
question.

| arm | change on top of the baseline | question |
| --- | --- | --- |
| `ctrl` | `prior_global none` | Reference: the local-only prior, whose samples were ~0.5x too narrow |
| `tok` | `prior_global token` | Does a per-graph global token give the prior the cross-node coherence it lacked? |
| `tok_seed` | tok, repeated unchanged | Run-to-run noise floor. HI_MGNFlow sets no seed (`training_seed` is not read), so each launch draws a fresh init and data order. Any gap smaller than tok vs tok_seed is noise |
| `tok_b8` | tok + `prior_blocks 8` | Deeper prior: more global/local exchange rounds per flow step |
| `tok_w256` | tok + `latent_dim 256` | Bigger network: AE and prior both twice as wide |
| `tok_long` | tok + `ae_epochs 120`, `training_epochs 240` | Longer training: twice the epochs in both stages |
| `tok_kl` | tok + `ae_kl_weight 1e-4` | A smoother AE latent (100x the KL) that is easier for the prior to fit |
| `tok_g64` | tok + `voronoi_clusters 4096, 64`, `latent_ch 16` | Same latent size (64 x 16 = 256 x 4) on 4x fewer coarse nodes to coordinate |

Shared B300 layout, which changes nothing about what is learned:

- `batch_size 8` x `grad_accum_steps 1`: the same effective batch as the
  baseline's 2 x 4.
- `num_workers 8`.
- `gpu_ids 0`: the runner assigns the card through `CUDA_VISIBLE_DEVICES`.
- `hierarchy_cache_keep False`.

One card per arm keeps the global batch equal across arms. Seven arms share one
hierarchy cache next to the dataset; `tok_g64` builds its own. The cache lock
lets one job build while the others wait, and a job never deletes a cache that
another job still has open.

## Run

Prerequisites on the node:

- **The ex3 datasets:** `dataset/probabilistic/ex3_shell_buckling.h5` and
  `dataset/probabilistic/ex3_shell_buckling_infer.h5`.
- **A cHI-MGNflow Python environment.** Point at it with
  `ai_cae4all.local.toml` or `METHOD_PYTHON=`. The interpreter needs torch
  built for Blackwell (sm_100, CUDA 12.8 or newer).
- **A checkout that has `prior_global`.**

```bash
nohup bash configs/HI_MGNFlow/probabilistic/ex3/b300_sweep/run_sweep.sh \
    > output/chi-mgnflow/run_sweep.out 2>&1 &
```

Before any GPU time is spent, the runner checks four things and stops on the
first one that fails:

1. the environment can import torch, torch_geometric and h5py, and can run a
   bf16 matmul and a scatter on the card;
2. every GPU index in `GPUS` exists on the node;
3. the checkout has `prior_global` (otherwise every `tok*` arm would quietly
   train the `ctrl` prior);
4. the dataset files exist, and the launcher `--check` passes on all 16 configs.

Each card then takes the next unclaimed arm, longest first, so fewer than 8
GPUs also works.

Overrides:

- `GPUS="0 1 2 3"`
- `ARMS="tok ctrl"`
- `PYTHON=` (the launcher interpreter)
- `METHOD_PYTHON=`
- `PREFLIGHT=0`, `TRAIN=0`, `INFER=0` or `REPORT=0` to skip a stage

To rebuild the report without running anything, use
`TRAIN=0 INFER=0 bash .../run_sweep.sh`.

## Time

On a 3090, the baseline runs at b2 x 4:

| stage | per epoch | epochs | peak VRAM |
| --- | --- | --- | --- |
| AE | ~1170 s | 60 | 11.4 GB |
| prior | ~450 s, plus a sampled validation every 10 epochs | 120 | 3.4 GB |

That is ~43 h end to end. The B300 numbers below assume a 5-8x speedup at b8.
Treat them as a guess until the first epoch lines arrive in `train.log`.

| arms | estimate |
| --- | --- |
| ctrl, tok, tok_seed, tok_kl, tok_g64 | ~6-9 h |
| tok_b8 | ~1.2x that |
| tok_long | ~12-18 h |
| tok_w256 | ~15-25 h |

Eight jobs share the node's CPUs and disk, which can stretch every arm.

## Output

Everything lands under `output/chi-mgnflow/ex3_b300_sweep/`:

```text
<arm>/model.pth                          kept checkpoint (AE + prior)
<arm>/train.log                          per-epoch AE/prior lines (+ periodic test pictures)
<arm>/infer/spread_metrics.json          held-out max|u| ensemble scores
<arm>/infer/histogram_compare.png        FEA truth vs generated, pooled
<arm>/infer/spread_values.npz            per-scene values behind the histogram
run_logs/<arm>.{train,infer}.out         launcher + trainer stdout
run_logs/<arm>.{train,infer}.check.log   preflight reports
run_logs/report.txt                      one row per arm (below)
run_logs/sweep_<timestamp>.out           the runner's own log
```

`report.txt` has one row per arm. Its columns are `sd_ratio`, `w1_norm`,
`crps_norm`, `spread_skill` and `pit_ks` from `spread_metrics.json`; the kept
prior epoch with its validation CRPS and spread; and the wall-clock hours.

On the same held-out set the current ex3 model scores `sd_ratio` 0.81,
`crps_norm` 0.36 and `w1_norm` 0.22. The targets are 1, 0 and 0.
