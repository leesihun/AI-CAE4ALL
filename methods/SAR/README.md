# SAR — scale-autoregressive flow matching (port)

A port of the official SAR code, Lino & Thuerey (2026), "One Scale at a Time:
Scale-Autoregressive Modeling for Fluid Flow Distributions"
([arXiv:2604.11403](https://arxiv.org/abs/2604.11403),
[tum-pbs/SAR](https://github.com/tum-pbs/SAR) at commit `1d1609fd`, Apache-2.0;
see [LICENSE](LICENSE) and [NOTICE](NOTICE)). It is a baseline for the
one-to-many comparison on ex3 shell buckling. It sits beside MGN-V
(`meshgraphnets-v`) and is scored the same way.

The model is SAR as published. It has two stages:

1. **Graph VAE.** A graph VAE encodes a field on the mesh into a per-node latent.
2. **SAR stage.** Over the frozen VAE, the SAR stage generates that latent one
   mesh scale at a time, coarse to fine, by flow matching. Each scale is
   conditioned on the geometry, the conditions and the coarser scales already
   generated.

Only the data I/O, the conditioning inputs and the entrypoint are new.

## Mapping onto the shared mesh HDF5

A SAR *system* is one geometry, meaning every sample that shares the
`split_group_attr` value (`shape_idx` on ex3). Its *states* are that geometry's
FEA draws. The model learns `p(field | geometry, conditions)`.

| `nodal_data` rows (ex3) | SAR input | handling |
| --- | --- | --- |
| `0:3` x, y, z at t=0 | `pos`, and `edge_attr = pos[dst] - pos[src]` on the bidirectional mesh edges | train-set centre removed; one isotropic factor (train max \|coord\|). `edge_attr` is divided by `2r` (upstream `ScaleEdgeAttr`), with `r = edge_attr_r`, `auto` = half the mean train edge length |
| `3:3+output_var` = ux, uy, uz at the **final** timestep | `target` (generated) | per channel min/max to [-1, 1] over the train split (upstream `ScaleAttr`) |
| next `cond_var` rows = thickness, delta at **t=0** | `glob` | per channel min/max to [-1, 1] over the train split; 0 if constant |
| last row = node type (0 interior, 1 clamped rim, 2 loaded rim) | `loc` one-hot | `use_node_types True` |

Nothing else in the file is read. The `latent` group and every metadata
attribute except `split_group_attr` are never read; ex3 stores the hidden
imperfection parameters there. So the spread has to come from SAR's sampler,
as it would for an unknown input. `tests/test_adapter.py::test_hidden_parameters_never_read`
checks this.

The split is the same as MGN-V's: `split_seed 42` with ratios 0.8/0.1/0.1,
grouped by `shape_idx`, so no geometry has draws on both sides of the split.

## Modes

| `mode` | does | writes |
| --- | --- | --- |
| `train` | VAE stage, then SAR stage | `vae_modelpath`, `modelpath` |
| `train_vae` | VAE stage only | `vae_modelpath` |
| `train_sar` | SAR stage over the trained VAE at `vae_modelpath` | `modelpath` |
| `inference` | `num_samples` draws per held-out system | the same files as MGN-V (below) |

Each stage writes its epoch log beside `log_file_dir`, as `train_vae.log` and
`train_sar.log`; `vae_log_file_dir` and `sar_log_file_dir` override the paths.
The periodic pictures (`test_interval`, `display_testset`, `display_trainset`)
land in the same directory:

- VAE stage: a parity scatter of reconstruction against truth, one panel per
  output channel;
- SAR stage: a histogram of the spread statistic, comparing the system's FEA
  draws with `viz_num_samples` SAR samples, for a val system (and a train
  system under `train/`).

A validation loss is printed every `val_interval` epochs. It is the stage's
own training loss on fixed val draws under a fixed RNG.

### Inference output

The `inference` mode writes MeshGraphNets_Variational's spread tail with the
same file names and the same (copied) code:

- `spread_values.npz`;
- `spread_metrics.json`;
- `histogram_compare.png`.

With `save_rollouts True` it also writes one
`rollout_sample{sid}_vaesample{k}_steps1.h5` per draw, in MGN-V's layout:

- `nodal_data [3+out+1, 2, N]`, where t=0 is the zero state and t=1 the
  sample;
- an `output_var` attribute;
- the node-type row.

The ex3 scoring therefore reads a SAR run like an MGN-V run.

`sample_scope` controls how ensembles map to scenes:

- **`geometry`** (the default) draws one `num_samples` ensemble per system and
  credits it to every scene of that system. A system here is the
  `split_group_attr` value plus identical condition rows (grouped by a hash of
  those rows). SAR sees nothing else, so those scenes have the same conditional
  distribution.
- **`scene`** draws a fresh ensemble per scene, like MGN-V's loop, at
  n_scenes × the cost.

## Run

Use the GPU server. The launcher sets the working directory to
`methods/SAR/`, so all paths in the configs are `../../...`.

```bash
# validate (from the repo root)
python AI_CAE4ALL_main.py --config configs/SAR/probabilistic/ex3/config_train_sar.txt --check
python AI_CAE4ALL_main.py --config configs/SAR/probabilistic/ex3/config_train_sar.txt --dry-run

# train both stages, then sample 64 fields per held-out geometry
python AI_CAE4ALL_main.py --config configs/SAR/probabilistic/ex3/config_train_sar.txt
python AI_CAE4ALL_main.py --config configs/SAR/probabilistic/ex3/config_infer_sar.txt

# or natively
cd methods/SAR
python SAR_main.py --config ../../configs/SAR/probabilistic/ex3/config_train_sar.txt
python SAR_main.py --config ../../configs/SAR/probabilistic/ex3/config_infer_sar.txt
```

`resume_training true` works as in the other trainers (see the root
CLAUDE.md). With it, a `mode train` run whose VAE stage already finished under
the same config keeps that VAE and continues with the SAR stage.

Check the first epochs' lines before leaving a run unattended. Memory and
throughput on ex3's 46–87k-node shells have not been measured. The config turns
on `dense_attention` and `use_checkpointing` for that reason.

Tests run on a tiny synthetic mesh HDF5 on the CPU, in about 30 s:

```bash
cd methods/SAR && python -m pytest -q tests/
```

## Deviations from upstream

The algorithm is unchanged. The following are unchanged from upstream:

- the VAE, the condition encoder, the masker, SAR's training loss, the scale
  sampling and the flow-matching sampler;
- the Guillard `MeshCoarsening`, `RandomizeScales` and `ScaleEdgeAttr`;
- the Wing hyperparameters.

Everything below is a change in plumbing, scale or I/O.

1. **PyTorch Lightning and wandb removed.** `sar/nn/model.py` is a plain
   `nn.Module`. Its `fit` reproduces the Lightning run upstream configured:
   - Adam;
   - gradient accumulation (`*_grad_accum`), with the last partial group
     stepped;
   - optional gradient clipping, off by default as upstream;
   - `ReduceLROnPlateau(factor, patience, eps=0)` stepped once per epoch on
     the epoch-mean `train_loss`;
   - upstream's `LearningRateEarlyStopping`: stop at the start of an epoch
     whose LR fell below `*_lr_threshold`;
   - the **last** weights are kept, as upstream's `ModelCheckpoint` did.

   `save_hyperparameters` and `load_from_checkpoint` are reimplemented
   without Lightning, and checkpoints load with `weights_only=True`. Single
   GPU only, since upstream's DDP path came from Lightning.
2. **`SAR.sample_n` sample count fixed.** Upstream computed each sampler
   batch as `min(batch_size, num_samples - len(samples))`. But `samples` holds
   batches, so a `num_samples` that is not a multiple of the batch returned
   too many fields (10 at batch 4 gave 12). The port counts samples instead.
   When `num_samples` is a multiple of the batch, the output is identical.
3. **Memory-lean Transolver attention (`dense_attention`, default on).**
   Upstream's `PhysicsAttention` forms a `[nodes, heads, slices, d_head]`
   tensor twice per layer. That is 8192 floats per node, about 2.9 GB per layer
   for one 87k-node shell. The dense path computes the same two contractions as
   padded einsums. It matches the reference to float summation order
   (`tests/test_sar_port.py`). The reference path is kept, and is used when the
   batch vector is unsorted or when `dense_attention False`.
4. **Activation checkpointing (`use_checkpointing`)** around each Transolver
   block while training. It gives the same gradients (tested). Neither switch
   is a hyperparameter or is saved in the checkpoint.
5. **Data.** Upstream's dataset classes (`datasets.py`, with downloaders for
   Ellipse, EllipseFlow and Wing) are replaced by the mesh-HDF5 adapter in
   `general_modules/mesh_dataset.py`:
   - the mapping and normalization described above;
   - train-split min/max instead of Wing's hard-coded `vmin/vmax`;
   - `edge_attr_r auto` instead of Wing's fixed 0.015.

   A system's static graph and scale hierarchy are built once per run and
   cloned into every draw. Upstream re-ran the transforms per item, which
   gives the same result because coarsening is deterministic.
6. **What the model sees.** `glob` (the conditions) is read at t=0 and the
   target at the final timestep. ex3 is quasi-static, and the generated
   quantity is the final post-buckling displacement.
7. **Draws per epoch.** One random draw per training system per epoch
   (`draws_per_epoch 1`), as upstream's one random state per system.
8. **Batch size cut with accumulation.** Wing's VAE batch of 32 × 2 becomes
   2 × 32, and the SAR batch of 8 × 4 becomes 2 × 16, so the effective batches
   stay 64 and 32 on large shells.
9. **Epoch caps.** Upstream trained until the LR threshold, with max_epochs
   1e6. `vae_training_epochs 3000` and `sar_training_epochs 2000` are only
   wall-clock caps. The LR-threshold stop still applies.
10. **Single-scale VAE only.** `vae_num_layers` must be one value, which is
    what every upstream example uses (`num_layers=[2,]`). Upstream's
    multi-scale VAE path expects scalar parent offsets that its own
    `MeshCoarsening` default does not produce.
11. **Inference grouping.** `sample_scope geometry` groups scenes by geometry
    plus a hash of the condition rows (see above). This is a cost saving, not a
    model change.
12. **Additions with no upstream counterpart:**
    - the validation loss, which is logged only and selects nothing;
    - the periodic pictures;
    - mid-training resume (`general_modules/resume_state.py`, the suite's shared
      copy);
    - checkpoint extras: `stage`, the train-split `normalization`, the run
      `config`, the scale counts and `vae_modelpath`, which the SAR stage and
      the sampler check against the config.

    DataLoader worker RNG is not restored on resume. With `num_workers > 0` a
    resumed run is statistically equivalent, not bit-identical.
13. **Log format.** Epochs print as `Epoch k/max ...` on stdout, which the
    Studio parses, and as `Elapsed: ...s Epoch k ...` in the stage log.
    Upstream logged only to wandb.
14. **Not vendored:**
    - `datasets.py`, `metrics.py` (POT Wasserstein metrics), `plot.py` (vtk)
      and `nn/callbacks.py`;
    - `nn/flow_matching/fmt.py` (the single-scale FMT baseline);
    - `transforms/connect.py`;
    - the notebooks and the example checkpoints.

    With them gone, the port needs no `lightning`, `wandb`, `pot`, `vtk`,
    `torchvision`, `scipy` or `torch_cluster` (see
    [requirements.txt](requirements.txt)).
