# SDFFlow geometry generation

SDFFlow trains an SDF variational autoencoder (VAE) and then a latent
flow-matching (FM) model. The VAE converts geometry to and from a compact
latent representation; the FM learns to generate those latents. Generated
latents are decoded to an SDF grid and exported through Marching Cubes as STL.

## Recommended workflow

From the `CAE_ML_Suite` root, validate and run the merged training config:

```powershell
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_train.txt --check
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_train.txt
```

The single `mode train` job runs the VAE first and starts FM training only
after it verifies that the VAE checkpoint completed successfully. This keeps
the GPU occupied without requiring a second manual launch.

After training, generate or compare shapes with:

```powershell
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_sample.txt
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_sample_extrapolation.txt
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_interpolate.txt
```

Direct backend commands are also supported. Run these from
`Geometry_generation` so relative paths keep their native meaning:

```powershell
python SDFFlow_main.py --config ex1/config_train.txt
python SDFFlow_main.py --config ex1/config_sample.txt
```

## Canonical configs and artifacts

| Config | Purpose | Main output |
| --- | --- | --- |
| `ex1/config_train.txt` | Sequential VAE -> FM training on DeepJEB | `outputs/ex1/sdfflow_vae.pth`, then `outputs/ex1/sdfflow_fm.pth` |
| `ex1/config_sample.txt` | Reproducible unconditional generation | `outputs/ex1/samples/` |
| `ex1/config_sample_extrapolation.txt` | Guarded, one-axis conditional extrapolation | `outputs/ex1/samples_extrapolation/` |
| `ex1/config_interpolate.txt` | Reproduce samples 0 and 1 and decode their latent interpolation | `outputs/ex1/interpolation/` |

Training writes the pipeline log to `ex1/train.log`, with stage logs at
`ex1/train_vae.log` and `ex1/train_fm.log`. These paths describe the runtime
contract; checkpoints and outputs are created only after the corresponding
jobs run.

The old split config names are not the production interface. Native
`train_vae` and `train_fm` modes remain available for focused debugging, but
the checked-in training config is intentionally merged.

## Pipeline restart behavior

`skip_completed_stages True` is safe to use when relaunching
`config_train.txt`:

- A checkpoint is reused only when its stage, completed epoch, and relevant
  saved config fields match the requested stage.
- An incomplete or incompatible VAE is retrained, and FM does not start until
  the replacement VAE passes verification.
- If the VAE was retrained, an existing FM checkpoint is treated as stale and
  FM is retrained against the new VAE.
- If both compatible checkpoints are complete, both stages are reused.

Stage-specific training controls in the merged config use `vae_` and `fm_`
prefixes. For example, `vae_training_epochs` becomes `training_epochs` for the
VAE worker, while `fm_training_epochs` becomes `training_epochs` for the FM
worker. Shared architecture, dataset, checkpoint, and condition fields remain
unprefixed.

## Dataset and conditioning contract

Build a synthetic smoke dataset or a real-mesh dataset from this repository:

```powershell
python build_dataset.py --output dataset/synthetic256.h5 --synthetic 256
python build_dataset.py --output dataset/parts.h5 --mesh_dir ./meshes --repair
```

The HDF5 dataset stores five descriptors in this fixed order:

```text
bbox_x, bbox_y, bbox_z, volume, area
```

The current DeepJEB FM deliberately selects only:

```text
bbox_x, bbox_z, volume, area
```

That selected order comes from `condition_names` in `config_train.txt` and is
saved in the FM checkpoint. `bbox_y` is excluded because normalization makes
that dimension effectively constant in this dataset. Any `cond_values` list
must match the checkpoint's selected `cond_names`, not the raw five-column
HDF5 order.

`config_sample.txt` omits `cond_values`, so it draws reproducible random
samples from the model's unconditional branch even though the FM was trained
conditionally. `cfg_scale 1.0` is plain conditional guidance when conditions
are supplied; increasing it can reduce diversity and does not guarantee
physical accuracy.

The extrapolation config moves only `volume` slightly beyond the observed
training maximum. It also:

- rejects requests beyond `max_condition_z` by default;
- clips extreme normalized latents with `latent_clip`;
- decodes extra candidates and ranks them by measured geometric descriptors;
- records requested, normalized, extrapolated, and actual conditions in the
  sample metadata.

Extrapolation remains an out-of-distribution experiment, not evidence that the
model is reliable far outside the training range.

## Interpolation

`config_interpolate.txt` recreates the same seed-0, 32-latent unconditional
batch used by `config_sample.txt`, selects indices 0 and 1, and linearly
interpolates them in normalized FM latent space with `alpha 0.5`. It exports:

- the two endpoint STLs and interpolated STL;
- a three-panel PNG comparison;
- JSON metadata with paths, mesh reports, interpolation space, and latent
  distances.

`source_num_samples` must match the original sampled batch because seeded RNG
reproduction depends on the tensor shape. The current interpolation mode is
unconditional and requires `0 <= alpha <= 1`.

## Output contracts

Sampling writes `sample_<seed>_<index>.stl` for valid zero crossings and one
`sample_<seed>_meta.json` file. The JSON also lists rejected candidates, so a
requested index can appear without an STL. Conditional runs include a
condition audit based on the descriptors measured from each decoded mesh.

Reconstruction is still available as an advanced native mode. A minimal config
needs `mode reconstruct`, `vae_modelpath`, `input_mesh`, `output_dir`, and
`mc_resolution`; it writes `<input_basename>_recon.stl`.

## SDF conventions

- Shapes are normalized to fit inside approximately `[-0.9, 0.9]^3`; queries
  cover `[-1, 1]^3`.
- SDF is negative inside and positive outside. The dataset builder flips the
  sign returned by `trimesh.signed_distance`.
- Reconstruction loss truncates SDF targets to `clamp_dist` (default `0.1`),
  while predictions remain unclamped so out-of-band errors retain gradients.
- Real input meshes must be watertight after any requested repair.

See the suite-level
[`DATASET_CONFIG_OUTPUT_REFERENCE.md`](../DATASET_CONFIG_OUTPUT_REFERENCE.md)
for the complete config, checkpoint, and output schema. The research document
[`GEOMETRY_GENERATION_RESEARCH.md`](GEOMETRY_GENERATION_RESEARCH.md) explains
the design motivation but is not the runtime source of truth.
