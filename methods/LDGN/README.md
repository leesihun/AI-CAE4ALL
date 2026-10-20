# LDGN — latent diffusion graph net (port)

A port of the official dgn4cfd code, Lino, Pfaff & Thuerey (ICLR 2025), "Learning
Distributions of Complex Fluid Simulations with Diffusion Graph Networks"
([OpenReview](https://openreview.net/forum?id=uKZdlihDDn),
[tum-pbs/dgn4cfd](https://github.com/tum-pbs/dgn4cfd) at commit `68311caa`,
Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Only the latent
variant (LDGN) is vendored. It is a baseline for the one-to-many comparison on
ex3 shell buckling, beside SAR and MGN-V (`meshgraphnets-v`), and is scored the
same way.

The model is the LDGN as published. It has two stages:

1. **VGAE.** A multi-scale variational graph autoencoder encodes a field on the
   mesh into a latent on a coarse Guillard level of that mesh. Its condition
   encoder turns geometry and conditions into latent node and edge features.
2. **LDGN stage.** Over the frozen VGAE, a multi-scale GNN denoiser learns a
   DDPM (1000 linear steps, learned variance, hybrid loss with importance step
   sampling) in that latent space. A sample is a denoised latent passed
   through the VGAE decoder.

Only the data I/O, the conditioning inputs and the entrypoint are new.

## Mapping onto the shared mesh HDF5

An LDGN *system* is one geometry: every sample that shares the
`split_group_attr` value (`shape_idx` on ex3). Its *states* are that geometry's
FEA draws. The model learns `p(field | geometry, conditions)`.

| `nodal_data` rows (ex3) | LDGN input | handling |
| --- | --- | --- |
| `0:3` x, y, z at t=0 | `pos`, and `edge_attr = pos[dst] - pos[src]` on the bidirectional mesh edges | train-set centre removed; one isotropic factor (train max \|coord\|). `edge_attr` is divided by `2r` (upstream `ScaleEdgeAttr`), with `r = edge_attr_r`, `auto` = half the mean train edge length |
| `3:3+output_var` = ux, uy, uz at the **final** timestep | `target` (generated) | per channel min/max to [-1, 1] over the train split (upstream `ScaleAttr`) |
| next `cond_var` rows = thickness, delta at **t=0** | `glob` | per channel min/max to [-1, 1] over the train split; 0 if constant |
| last row = node type (0 interior, 1 clamped rim, 2 loaded rim) | `loc` one-hot | `use_node_types True` |

Nothing else in the file is read. The `latent` group and every metadata
attribute except `split_group_attr` are never read; ex3 stores the hidden
imperfection parameters there. So the spread has to come from the diffusion
sampler, as it would for an unknown input.

The split is the same as MGN-V's and SAR's: `split_seed 42` with ratios
0.8/0.1/0.1, grouped by `shape_idx`, so no geometry has draws on both sides.

### Scale hierarchy

Upstream's Wing scripts run `MeshCoarsening` with
`rel_pos_scaling = [0.015, 0.03, 0.06, 0.12, 0.2, 0.4]` after
`ScaleEdgeAttr(0.015)`. The first entry is the edge scale `r`, so here the list
is `r × (1, 2, 4, 8, 40/3, 80/3)` (doubling past six levels). The VGAE uses the
first `len(vae_depths)` levels. The LDGN denoiser starts at the VGAE's coarsest
level and uses `len(ldgn_depths)` levels from there, so both stages build
`len(vae_depths) + len(ldgn_depths) - 1` levels (6 with the Wing depths
`[2,2,1]` and `[1,2,2,2]`). A geometry's levels are built once per run and
cloned into every draw.

## Modes

| `mode` | does | writes |
| --- | --- | --- |
| `train` | VGAE stage, then LDGN stage | `vae_modelpath`, `modelpath` |
| `train_vae` | VGAE stage only | `vae_modelpath` |
| `train_ldgn` | LDGN stage over the trained VGAE at `vae_modelpath` | `modelpath` |
| `inference` | `num_samples` draws per held-out system | the same files as MGN-V (below) |

`modelpath` is self-contained: it stores the VGAE's arch, and the frozen VGAE's
weights are part of the LDGN state dict. Inference reads only `modelpath`.

Each stage writes its epoch log beside `log_file_dir`, as `train_vae.log` and
`train_ldgn.log`; `vae_log_file_dir` and `ldgn_log_file_dir` override them.
The periodic pictures (`test_interval`, `display_testset`, `display_trainset`)
land in the same directory:

- VGAE stage: `vae_parity_ep{k}.png`, a parity scatter of reconstruction
  against truth, one panel per output channel;
- LDGN stage: `ldgn_samples_ep{k}.png`, a histogram of the spread statistic
  comparing the system's FEA draws with `viz_num_samples` LDGN samples, for a
  val system (and a train system under `train/`).

A validation loss is printed every `val_interval` epochs. It is the stage's
own training loss on fixed val draws under a fixed RNG, and it selects nothing.

### Inference output

The `inference` mode writes MeshGraphNets_Variational's spread tail with the
same file names and the same (copied) code: `spread_values.npz`,
`spread_metrics.json` and `histogram_compare.png`. With `save_rollouts True` it
also writes one `rollout_sample{sid}_vaesample{k}_steps1.h5` per draw in MGN-V's
layout (`nodal_data [3+out+1, 2, N]`, t=0 the zero state and t=1 the sample, an
`output_var` attribute and the node-type row). The ex3 scoring therefore reads
an LDGN run like an MGN-V run.

Sampling is upstream's ancestral DDPM sampler over `sample_steps` of the
trained steps. The Wing notebook uses 50 linearly spaced steps of 1000
(`sample_steps 50`, `sample_step_schedule linear`).

`sample_scope` works as in SAR: `geometry` (the default) draws one ensemble per
(geometry, condition rows) and credits it to every scene of that system;
`scene` draws a fresh ensemble per scene, at n_scenes × the cost.

## Run

Use the GPU server. The launcher sets the working directory to
`methods/LDGN/`, so all paths in the configs are `../../...`.

```bash
# validate (from the repo root)
python AI_CAE4ALL_main.py --config configs/LDGN/probabilistic/ex3/config_train_ldgn.txt --check

# train both stages, then sample 64 fields per held-out geometry
python AI_CAE4ALL_main.py --config configs/LDGN/probabilistic/ex3/config_train_ldgn.txt
python AI_CAE4ALL_main.py --config configs/LDGN/probabilistic/ex3/config_infer_ldgn.txt
```

`resume_training true` works as in the other trainers (see the root
CLAUDE.md). With it, a `mode train` run whose VGAE stage already finished under
the same config keeps that VGAE and continues with the LDGN stage.

Memory and throughput on ex3's 46–87k-node shells have not been measured.
Check the first epochs' lines before leaving a run unattended.

Tests run on a tiny synthetic mesh HDF5 on the CPU, in about 30 s:

```bash
cd methods/LDGN && python -m pytest -q tests/
```

## Deviations from upstream

The algorithm is unchanged. The following are unchanged from upstream:

- the VGAE, its condition encoder, `VaeLoss`;
- the LDGN denoiser, `DiffusionProcess`, `HybridLoss`, the importance step
  sampler and the DDPM sampler;
- the Guillard `MeshCoarsening`;
- the Wing hyperparameters, except where listed below.

Everything below is a change in plumbing, scale or I/O.

1. **Training loop.** Upstream's `Model.fit` (TensorBoard, tqdm, periodic
   `.chk` files, mixed precision) is replaced by a plain loop with the same
   recipe:
   - Adam over the trainable parameters;
   - gradient-norm clipping at 1 from the first epoch;
   - `ReduceLROnPlateau(factor 0.1, patience 250, eps=0)` stepped once per
     epoch on the epoch-mean training loss;
   - stop at the start of an epoch whose LR fell below `*_lr_threshold`
     (upstream `stopping 1e-8`);
   - the last weights are kept.

   Gradient accumulation (`*_grad_accum`) is new; the last partial group is
   stepped. Checkpoints load with `weights_only=True` (the activation class is
   stored by name, the `DiffusionProcess` by its constructor arguments).
   Single GPU only.
2. **`sample_n` fixes** (`DiffusionModel` and `VGAE`). The batched path
   counted generated *batches* against `num_samples`, so more samples than
   asked could come back. The one-by-one path reused one graph that `sample`
   mutates (its `edge_index`/`batch` become the coarsest level's), so the second
   call ran on a corrupted graph. Both are fixed.
3. **`learnable_variance False`** raised `AttributeError` upstream (it read
   `out_node_features` before setting it). Fixed. The Wing recipe uses `True`,
   which was not affected.
4. **`LazyBatchNorm1d` → `BatchNorm1d(1)`** in the VGAE's latent
   normalization. The lazy module only ever inferred one feature. The buffers
   and state-dict keys are the same.
5. **Self-contained LDGN checkpoint.** The LDGN can be rebuilt from the VGAE's
   arch alone (its weights are in the LDGN state dict), so sampling needs no
   separate VGAE file.
6. **Data.** Upstream's dataset classes and the Wing transforms
   (`ScaleEdgeAttr`, `EdgeCondFreeStream`, `ScaleAttr`, `Copy`) are replaced by
   the mesh-HDF5 adapter in `general_modules/mesh_dataset.py`:
   - the mapping and normalization described above;
   - train-split min/max instead of Wing's hard-coded `vmin/vmax`;
   - `edge_attr_r auto` instead of Wing's fixed 0.015, with the coarsening
     radii scaled by the same ratios (`rel_pos_ratios`, default Wing's
     1, 2, 4, 8, 13.33, 26.67; Ellipse's are 1, 3, 7.5, 15);
   - the conditions are the node-type one-hot plus `cond_var` rows instead of
     Wing's free-stream edge conditions. When a file has neither, `loc` is a
     constant 1, since upstream's encoders need at least one condition feature.
7. **What the model sees.** `glob` is read at t=0 and the target at the final
   timestep. ex3 is quasi-static, and the generated quantity is the final
   post-buckling displacement.
8. **Draws per epoch.** One random draw per training system per epoch
   (`draws_per_epoch 1`), as upstream's one random state per system.
9. **Batch size cut with accumulation.** Wing's batch of 32 becomes 2 × 16 in
   both stages, so the effective batch stays 32 on 46–87k-node shells.
10. **Epoch caps.** Wing trains the VGAE for up to 5000 epochs and the LDGN for
    up to 50000, both also stopped by the LR threshold. ex3 caps them at
    `vae_training_epochs 3000` (as SAR's VAE) and `ldgn_training_epochs 5000`.
    These are wall-clock caps; the LR-threshold stop still applies.
11. **Short diffusion schedules.** The linear schedule scales its betas by
    `1000 / ldgn_diffusion_steps` and clips them at 0.999. Below about 50 steps
    the coarse sub-steps reach ᾱ = 0 in float32 and sampling returns NaN.
    This is upstream behaviour; keep `ldgn_diffusion_steps` at 1000 (the tests
    use 50).
12. **Additions with no upstream counterpart:**
    - the validation loss, which is logged only;
    - the periodic pictures;
    - mid-training resume (`general_modules/resume_state.py`, the suite's shared
      copy); the importance step sampler's loss history is part of the resume
      state;
    - checkpoint extras: `stage`, the train-split `normalization`, the run
      `config`, `num_scales`, `autoencoder_arch` and `vae_modelpath`, which the
      LDGN stage and the sampler check against the config.

    DataLoader worker RNG is not restored on resume. With `num_workers > 0` a
    resumed run is statistically equivalent, not bit-identical.
13. **Log format.** Epochs print as `Epoch k/max ...` on stdout, which the
    Studio parses, and as `Elapsed: ...s Epoch k ...` in the stage log.
14. **Not vendored:** the non-latent DGN, flow matching (FMGN/LFMGN), the
    Bayesian/Gaussian-mixture/vanilla GNN baselines, the dataset downloaders,
    `metrics.py`, `plot.py`, the other transforms and the examples. With them
    gone the port needs no `scipy`, `tensorboard`, `vtk`, `bayesian-torch`,
    `pot`, `torch_cluster` or `torchvision` (see
    [requirements.txt](requirements.txt)).
