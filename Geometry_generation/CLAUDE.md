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
```

From `Geometry_generation`:

```bash
python build_dataset.py --output dataset/synthetic256.h5 --synthetic 256
python SDFFlow_main.py --config ../configs/Geometry_generation/config_train.txt
python SDFFlow_main.py --config ../configs/Geometry_generation/config_sample.txt
```

The config parser accepts flat `key value` text, lowercases keys and string
values, and treats `%` lines as comments. Relative native paths resolve from
the `Geometry_generation` repository even when the suite launcher is used.

Valid modes are `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`, and
`interpolate`. Production training uses `train`; the two split training modes
are retained for targeted debugging and have no checked-in split configs.

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
| `SDFFlow_main.py` | Config load and mode dispatch |
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

## Validation after changes

At minimum, run from the suite root:

```bash
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt --check
python -m pytest -q tests/test_sdfflow_pipeline.py tests/test_checked_in_configs.py tests/test_required_field_matrix.py
```

Also run `python -m py_compile` on modified Python files. Documentation changes
must keep the four canonical config names, the four selected DeepJEB condition
names, and the canonical `outputs/ex1` paths synchronized.

`GEOMETRY_GENERATION_RESEARCH.md` is design context, not implementation truth.
