# SDFFlow maintainer notes

This file records the live repository contract for agents and maintainers.
`SDFFlow_main.py`, the checked-in configs, and their consumers remain the
authoritative implementation.

## Commands and working directories

From the `AI-CAE4ALL` root:

```bash
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_sample.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_sample_extrapolation.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_interpolate.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_optimize.txt
```

From `Geometry_generation`:

```bash
python build_dataset.py --output dataset/synthetic256.h5 --synthetic 256
python SDFFlow_main.py --config ../configs/Geometry_generation/config_train.txt
python SDFFlow_main.py --config ../configs/Geometry_generation/config_sample.txt
```

The config parser accepts flat `key value` text, lowercases keys and string
values, and treats `%` lines as comments. Path-valued keys (`dataset_dir`,
`output_dir`, `input_mesh`, `*_modelpath`, the log dirs — see
`general_modules/load_config.py::PATH_KEYS`) are exempt from the value
lowercasing and keep the case you wrote. Relative native paths resolve from
the `Geometry_generation` repository even when the suite launcher is used.

Valid modes are `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`,
`interpolate`, and `optimize`. Production training uses `train`; the two split
training modes are retained for targeted debugging and have no checked-in split
configs.

## Multi-GPU (`parallel_mode`)

The training modes (`train`, `train_vae`, `train_fm`) support `parallel_mode`:

- `single` (default) — one process/one GPU; unchanged legacy behavior.
- `ddp` — Distributed Data Parallel. `SDFFlow_main.py` self-spawns one worker
  per GPU in `gpu_ids` (a self-picked free TCP port; no `torchrun`). The batch
  is sharded, gradients all-reduced. Rank 0 owns validation, the periodic test,
  logging, and checkpoint writes. Use when the model fits on one GPU (on a
  288 GB B300, almost everything). The checked-in `config_train_b300.txt` is an
  8-GPU DDP example.
- `fsdp` — Fully Sharded Data Parallel; SDFFlow's "model split". Shards
  params/grads/optimizer for a velocity DiT too large for one GPU. Requires
  CUDA/NCCL; disables EMA and performs its own bf16. `fsdp_min_params` sets the
  auto-wrap granularity.

Distributed plumbing lives in `general_modules/distributed.py`. Sampling,
reconstruction, and interpolation remain single-process. The suite launch
command is unchanged (`python SDFFlow_main.py --config ...`); the spawn happens
inside the process when `parallel_mode` is ddp/fsdp and >1 GPU is listed.
DDP/FSDP was validated with a 2-rank gloo/CPU run of the merged pipeline
(including the hybrid VAE double-backward); FSDP's CUDA path is standard PyTorch
FSDP1 but is only exercised on the GPU server.

## Merged training invariants

`training_profiles/train_pipeline.py` converts the merged config into native
VAE and FM stage configs. Every supported stage setting is written as
`vae_<setting>` or `fm_<setting>` and loses that prefix before its worker is
called. Shared dataset, architecture, checkpoint, and conditioning fields are
copied to both workers.

The pipeline contract is:

1. Inspect `vae_modelpath` for the expected `stage`, final epoch, and relevant
   saved config fields.
2. Train or reuse the VAE according to `skip_completed_stages`.
3. Refuse to start FM unless the VAE checkpoint verifies as complete.
4. Release unused stage memory before FM.
5. Reuse FM only if the VAE was reused and the FM checkpoint is complete and
   compatible. A newly trained VAE always invalidates old FM reuse.

Do not restore separate production configs or launch stages independently from
automation. That reintroduces idle-GPU gaps and permits stale VAE/FM pairings.

Canonical artifacts are:

```text
ex1/train.log
ex1/train_vae.log
ex1/train_fm.log
outputs/ex1/sdfflow_vae.pth
outputs/ex1/sdfflow_fm.pth
outputs/ex1/samples/
outputs/ex1/samples_extrapolation/
outputs/ex1/interpolation/
```

## Data and condition invariants

The HDF5 layout is
`shapes/{index:05d}/{surface_points,surface_normals,sdf_points,sdf_values,cond}`.
Root `cond_names` and every `cond` row contain the raw five descriptors:

```text
bbox_x, bbox_y, bbox_z, volume, area
```

FM training may select a subset through `condition_names`. The shipped DeepJEB
config uses `bbox_x,bbox_z,volume,area`; do not add `bbox_y` without first
showing that its train-split standard deviation exceeds `min_condition_std`.
The selected names, statistics, extrema, and clipping limit are stored in the
FM checkpoint and define `cond_values` order at inference.

Conditioned sampling enforces `max_condition_z` with an `error`, `warn`, or
`clamp` policy. The checked-in extrapolation config uses `error`, applies
`latent_clip`, and uses `candidate_multiplier` so decoded candidates can be
ranked by actual geometric-condition error. Preserve the requested and actual
condition audit in `sample_<seed>_meta.json` when changing inference.

`cfg_scale 1.0` is the live conservative setting. Higher CFG is a strength and
diversity tradeoff, not a substitute for condition accuracy or OOD validation.

## Interpolation invariants

`inference_profiles/interpolate.py` reproduces an unconditional FM batch from
`seed` and `source_num_samples`, then applies `torch.lerp` in normalized FM
latent space. Endpoint indices must be distinct and in range, and `alpha` must
be within `[0, 1]`. Keep `source_num_samples` equal to the original sampling
batch size when reproducing existing endpoint indices. Conditional
interpolation is intentionally rejected for now.

The mode writes three STLs, a triptych PNG, and JSON metadata. A missing zero
crossing is a hard failure because all three comparison meshes are required.

## Closed-loop optimization (`optimize`)

`inference_profiles/optimize.py` drives `design_loop/` around the frozen
VAE + FM pair: **generate -> mesh -> analyze -> score -> search**. It trains
nothing; it searches the geometry the generator already knows.

**The design vector is FM noise, not a latent and not a condition.** It is a
`opt_subspace_dim`-dimensional orthonormal slice of the 256-d flow-matching
noise space (`opt_subspace_seed` fixes the basis, `seed` the out-of-subspace
remainder), optionally extended by `opt_condition_dims` descriptor conditions.
Every point therefore integrates through the ODE to an on-manifold shape, which
is what makes a derivative-free search viable. The composed noise is projected
back onto the Gaussian shell, so the search cannot walk off the prior. Do not
substitute a raw latent parameterization: latents off the FM prior decode to
shapes the VAE never trained on.

`bbox_y` has exactly zero train-split standard deviation in DeepJEB and
`bbox_x` a 0.45% coefficient of variation, so `opt_condition_dims` defaults to
`volume,area` -- the roughly 1.5 descriptor degrees of freedom that actually
move (`SDFFlow_ENCODER` findings in
`GEOMETRY_UPGRADE_MESHING_SEMANTIC_2026-08.md`). The launcher warns on
`bbox_y`.

Keep `opt_latent_range` equal to `opt_shell_scale` (both default 1.25). That
inscribes the search box's corner exactly on the shell for any
`opt_subspace_dim`. A wider box is mostly *degenerate*, not merely generous: the
composed noise is rescaled back onto the shell, which keeps direction and
discards magnitude, so every point on a ray beyond the radius decodes to the
identical shape. `output/geometry_generation/ex1/optimization_widebox/` is a
kept 200-evaluation run at `latent_range 3.0` against the same 4.33 shell --
its typical draw had norm about 6.0 and was always clipped, and it spent its
whole budget repairing constraints instead of shedding mass. Compare it with
`optimization/` before widening the box again.

`bounds()` takes no arguments on purpose. It is called from both the baseline
sampler and the CMA-ES setup, and an earlier version that accepted a
`latent_range` argument and cached it on the instance let the second, no-argument
call silently reset the configured range. `tests/test_design_loop.py` pins that
down.

**Boundary conditions come from a geometric rule, because DeepJEB carries no
per-shape semantic labels.** The dataset is rigidly aligned -- y is the long
axis at extent 1.8 for every sample -- and a 600-sample occupancy study shows
the mounting plate is the low-z slab with pads at both y ends and the loaded
interface is the lug rising in +z near y = 0. `design_loop/problem.py` fixes
the bottom face of the two end pads and applies the GE bracket-challenge loads
to the lug crown. The thresholds are module constants (`MOUNT_ABS_Y`,
`LUG_ABS_Y`, ...); a shape missing either interface raises and is scored as a
failed design rather than silently analyzed as a cantilever.

**Allowables are calibrated, not assumed.** A random population of
`opt_baseline_size` designs is analyzed first and the stress/deflection limits
are set to its medians. Absolute limits taken from the material would leave the
constraints inactive -- the baseline sits near 7% of Ti-6Al-4V yield -- and the
objective would collapse to unconstrained mass minimization. The comparison
baseline reported at the end is the *best-scoring* population member, not the
median.

**The solver is 4-node tetrahedra, and tet4 is stiff.** `tests/test_design_loop.py`
pins it down: the constant-strain patch test is exact to 1e-9, rigid-body modes
store no energy, and load resultants match to machine precision -- but on a
slender cantilever the predicted tip deflection reaches only 0.40 / 0.65 / 0.83 /
0.90 of the Timoshenko value at 288 / 1.3k / 6k / 16.5k tets. Deflection and
stress from this loop are therefore **optimistic in absolute terms**; the
comparison between two designs at equal discretization is what the loop is for.
`second_order=True` in `design_loop/mesher.py` produces the tet10 mesh DeepJEB's
own FEA uses, but `fea.py` assembles tet4 only -- wiring a tet10 element is the
change to make before quoting absolute stresses.

**The search mesh is a ranking device.** `opt_target_faces` / `opt_mesh_size_max`
size the per-evaluation mesh; the winner and the baseline are then re-analyzed
together on the finer `opt_verify_*` mesh. `summary.json` records the
same-design shift between the two meshes under `mesh_sensitivity`; treat the
search-phase feasibility flag as relative only.

Artifacts in `output_dir`: `optimized.stl`, `baseline.stl`, `summary.json`,
`history.json` (every evaluation, including failures and timings),
`convergence.png`, `report.md`.

## Architecture and checkpoint facts

- SDF sign is negative inside and positive outside. Shapes occupy roughly
  `[-0.9, 0.9]^3`; queries cover `[-1, 1]^3`.
- VAE training supports deterministic, posterior-noise, and KL warmups. FM
  consumes normalized encoder means rather than posterior samples.
- The Tier-1 latent is `latent_tokens x latent_dim`; the checked-in model uses
  one global token and the MLP decoder.
- FM uses rectified flow with AdaLN-Zero blocks. Latent and selected-condition
  statistics come from the train split and are stored in its checkpoint.
- Checkpoints store config and stage metadata. Inference rebuilds architecture
  from checkpoint config and prefers `ema_state` when present.
- The FM checkpoint records the exact `vae_modelpath`; architecture fields in
  sample configs do not override checkpoint-owned models.

## Key files

| File | Role |
| --- | --- |
| `SDFFlow_main.py` | Config load, mode dispatch, distributed spawn dispatch |
| `general_modules/distributed.py` | DDP/FSDP setup, spawn, rank gating, sharded state-dict gather |
| `build_dataset.py` | Real meshes or synthetic primitives to HDF5 |
| `general_modules/sdf_sampling.py` | Normalization, SDF samples, descriptors, repair path |
| `general_modules/sdf_dataset.py` | Lazy-open HDF5 dataset, seeded split, condition statistics |
| `general_modules/mesh_extraction.py` | Latent to SDF grid to Marching Cubes mesh/report |
| `model/sdf_vae.py` | Encoder, latent bottleneck, SDF decoders, reconstruction loss |
| `model/velocity_net.py` | Conditional rectified-flow loss and ODE sampler |
| `training_profiles/train_pipeline.py` | Sequential orchestration and compatibility-based reuse |
| `training_profiles/train_vae.py` | VAE stage worker |
| `training_profiles/train_fm.py` | Latent cache, condition selection, FM stage worker |
| `training_profiles/setup.py` | Device, optimizer/scheduler, EMA, logging, checkpoints |
| `inference_profiles/sample.py` | Sampling, OOD guard, candidate ranking, reconstruction |
| `inference_profiles/interpolate.py` | Reproducible latent interpolation and triptych output |
| `inference_profiles/optimize.py` | `optimize` mode: config to loop, calibration, verification, reporting |
| `design_loop/generator.py` | Design vector to FM noise to latent to SDF grid to surface mesh |
| `design_loop/mesher.py` | gmsh tetrahedralization (reparametrization-free; see its docstring) |
| `design_loop/fea.py` | Tet4 linear elasticity, AMG solve, von Mises recovery |
| `design_loop/problem.py` | Bracket interfaces, GE load cases, mass objective with penalties |
| `design_loop/loop.py` | Evaluator, baseline calibration, CMA-ES driver, history |

## Validation after changes

At minimum, run from the suite root:

```bash
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_sample.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_interpolate.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_optimize.txt --check
```

> The previously documented `pytest tests/test_sdfflow_pipeline.py
> tests/test_checked_in_configs.py tests/test_required_field_matrix.py` does not
> exist in this checkout (nor anywhere else in the tree) -- as with the stale
> root `testpaths`, per-config `--check` is the live validation. Verified
> 2026-08-30.

The `optimize` mode adds `gmsh`, `pyamg`, and `cma` to the requirement set.

Also run `python -m py_compile` on modified Python files. Documentation changes
must keep the five canonical config names, the four selected DeepJEB condition
names, and the canonical `outputs/ex1` paths synchronized.

`GEOMETRY_GENERATION_RESEARCH.md` is design context, not implementation truth.
