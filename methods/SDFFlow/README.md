# SDFFlow geometry generation

SDFFlow trains an SDF variational autoencoder (VAE) and then a latent
flow-matching (FM) model. The VAE converts geometry to and from a compact
latent representation; the FM learns to generate those latents. Generated
latents are decoded to an SDF grid and exported through Marching Cubes as STL.

## Recommended workflow

From the `AI-CAE4ALL` root, validate and run the recommended merged training
config (the v3 recipe; its header explains every setting):

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3.txt
```

The single `mode train` job runs the VAE first and starts FM training only
after it verifies that the VAE checkpoint completed successfully. This keeps
the GPU occupied without requiring a second manual launch.

Score the trained VAE on its held-out split, then generate or compare shapes:

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample_extrapolation.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_interpolate.txt
```

The three sampling/interpolation configs point at the `ex1` checkpoints; to
use the v3 model, point their `vae_modelpath` / `fm_modelpath` at
`output/geometry_generation/ex4/`.

The FEA-conditioned track (`ex5`) lets a designer ask for a bracket by the
numbers that matter -- volume, area, the peak stress of a load case, the first
mode -- and leave the rest unspecified. It needs DeepJEB's FEA labels appended
to the dataset once, then trains, calibrates, samples, sweeps and benchmarks:

```bash
cd methods/SDFFlow && python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 \
    --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv --dry_run   # then without --dry_run
cd ../..
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3_fea.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_calibrate_descriptors.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample_conditional.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_cond_sweep.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate_conditional.txt
```

No ex5 checkpoint exists yet; the design, the label analysis behind the
condition set and the list of what is unverified until it trains are in
[`docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md`](../../docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md).

Direct backend commands are also supported. Run these from
`methods/SDFFlow` so relative paths keep their native meaning:

```bash
python SDFFlow_main.py --config ../../configs/SDFFlow/config_train_v3.txt
python SDFFlow_main.py --config ../../configs/SDFFlow/config_sample.txt
```

## Canonical configs and artifacts

| Config | Purpose | Main output |
| --- | --- | --- |
| `configs/SDFFlow/config_train_v3.txt` | **Recommended** single-GPU VAE -> FM training on DeepJEB (32 x 32 FPS-anchored VecSet latent, parent-grouped split, real posterior noise, `volume,area` conditions) | `output/geometry_generation/ex4/sdfflow_vae.pth`, `sdfflow_vae_best.pth`, then `sdfflow_fm.pth` |
| `configs/SDFFlow/config_train_b300.txt` | 8-GPU DDP twin of v3 (batch/LR/epoch arithmetic in its header) | `output/geometry_generation/ex3/` |
| `configs/SDFFlow/config_train_v2.txt` | The "v2 control": 32 x 64 VecSet + DiT + hybrid loss, architecture frozen | `output/geometry_generation/ex2/` |
| `configs/SDFFlow/config_train.txt` | Tier-1 control: one 256-d token, MLP decoder | `output/geometry_generation/ex1/sdfflow_vae.pth`, then `output/geometry_generation/ex1/sdfflow_fm.pth` |
| `configs/SDFFlow/config_train_v3_fea.txt` | v3 conditioned on `volume,area` + four DeepJEB FEA labels (`cond_dropout_mode per_dim`); needs the `cond_extra` sidecar from `add_fea_conditions.py` | `output/geometry_generation/ex5/sdfflow_vae.pth`, `sdfflow_fm.pth` |
| `configs/SDFFlow/config_evaluate.txt` | Held-out reconstruction metrics for the ex4 VAE (encoder mean and refined latent) | `output/geometry_generation/ex4/eval/eval_val.{json,csv}` |
| `configs/SDFFlow/config_calibrate_descriptors.txt` | `eval_task descriptor_calibration`: affine calibration of the soft volume/area proxy on the ex5 val split | `output/geometry_generation/ex5/eval/descriptor_calibration.pth` |
| `configs/SDFFlow/config_evaluate_conditional.txt` | `eval_task conditional`: paired-noise condition-accuracy benchmark (plain / rejection / e2) on the ex5 test split | `output/geometry_generation/ex5/eval_conditional/eval_conditional.{json,csv}` |
| `configs/SDFFlow/config_sample.txt` | Reproducible unconditional generation | `output/geometry_generation/ex1/samples/` |
| `configs/SDFFlow/config_sample_extrapolation.txt` | Guarded, one-axis conditional extrapolation | `output/geometry_generation/ex1/samples_extrapolation/` |
| `configs/SDFFlow/config_sample_conditional.txt` | Partial request (two of six conditions unspecified) with candidate ranking and E2 Newton correction on the ex5 pair | `output/geometry_generation/ex5/samples_conditional/` |
| `configs/SDFFlow/config_interpolate.txt` | Reproduce samples 0 and 1 and decode a noise-space slerp between them | `output/geometry_generation/ex1/interpolation/` |
| `configs/SDFFlow/config_cond_sweep.txt` | `interpolation_space cond_sweep`: one noise row decoded under a five-step condition morph | `output/geometry_generation/ex5/cond_sweep/` |
| `configs/SDFFlow/config_optimize.txt`, `config_optimize_surrogate.txt` | Closed-loop design search over the trained generator | `output/geometry_generation/ex1/optimization*/` |
| `configs/SDFFlow/arms/A0.txt` .. `A9.txt` | VAE ablations of the v3 recipe (`mode train_vae`). Most move one axis; `A1` is an omnibus ex1-architecture control and `A9` is A0 at a second `seed`, the sweep's noise floor. See [`arms/README.md`](../../configs/SDFFlow/arms/README.md) | `output/geometry_generation/arms/<label>/` |

Training writes the pipeline log to `output/geometry_generation/<exN>/train.log`,
with stage logs at `<exN>/train_vae.log` and `<exN>/train_fm.log`. The
historical `geometry_generation` output slug is retained for checkpoint
compatibility; the runtime itself now lives under `methods/SDFFlow/`.

The checked-in production training configs are merged (`mode train`). The
native `train_vae` mode is what the `arms/` sweep uses to compare VAE recipes
before the FM is trained once for the winner; `train_fm` remains available for
focused debugging.

Optional `vae_best_modelpath` saves, after every validation, the EMA (or raw)
model with the best validation SDF loss so far, in the same payload format as
the final checkpoint. The final save to `vae_modelpath` is unchanged and stays
the file the pipeline's completeness check and the FM stage read.

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
python methods/SDFFlow/build_dataset.py --output dataset/synthetic256.h5 --synthetic 256
python methods/SDFFlow/build_dataset.py --output dataset/parts.h5 --mesh_dir ./meshes --repair --near_sigmas 0.01,0.05
```

`--near_sigmas` (default `0.01,0.05`) lists the near-surface Gaussian offsets;
one is drawn uniformly per near point. The builder records its provenance as
root attrs (`num_surface`, `max_faces`, `sharp_edge_fraction`,
`sharp_edge_angle`, `near_sigmas`, `seed`, `sdf_backend`). Signed distances
use `igl` when importable, else Open3D's `RaycastingScene` (sign verified so
inside is negative), else trimesh; the backend is printed once per process.

### Held-out split

DeepJEB's 2138 brackets are variants of 263 parent geometries (the `source`
attr is `<parent>_<variant>.stl`). The per-shape random split placed a sibling
of every val/test shape into train, so validation measured memorization.
`split_by_parent True` (used by every v3-era config) permutes the parents with
`split_seed` and assigns whole parents to train/val/test at ~80/10/10 of the
shape count. `train_vae`, `train_fm`, and `evaluate` share the same split
function, and the val/test datasets (and the FM latent-cache encode pass)
subsample points deterministically, so every run and rank sees identical
inputs. The key defaults to `False` (legacy per-shape split).

The HDF5 dataset stores five descriptors in this fixed order:

```text
bbox_x, bbox_y, bbox_z, volume, area
```

The ex1/ex2 DeepJEB FMs (`config_train.txt`, `config_train_v2.txt`) select:

```text
bbox_x, bbox_z, volume, area
```

and the v3 recipe (`config_train_v3.txt`, `config_train_b300.txt`) selects only:

```text
volume, area
```

The selected order comes from `condition_names` in the training config and is
saved in the FM checkpoint. `bbox_y` is excluded because normalization makes
that dimension exactly constant in this dataset, and v3 also drops `bbox_x`
(0.45% coefficient of variation) because conditioning on a near-constant only
adds noise to the CFG branch. Any `cond_values` list must match the
checkpoint's selected `cond_names`, not the raw five-column HDF5 order.

### FEA-label conditions (the `cond_extra` sidecar)

DeepJEB ships per-design FEA labels (mass, per-load-case max von Mises stress
and max displacement, the first two eigenfrequencies, ...). `add_fea_conditions.py`
appends them to an existing HDF5 as the root dataset `cond_extra`
(`[num_shapes, k]`, row i = shape i) with the attrs `cond_extra_names`,
`cond_extra_source`, `cond_extra_transforms` and `cond_extra_created`; nothing
else in the file is touched, and a file without the sidecar reads exactly as
before. The join is the per-shape `source` basename without extension against
the CSV's `item_name` (2138/2138); the builder refuses unmatched shapes and an
existing sidecar unless told otherwise (`--allow_missing`, `--overwrite`), and
`--dry_run` prints the per-name statistics without writing:

```bash
cd methods/SDFFlow
python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv --dry_run
python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv
python add_fea_conditions.py --list_names
```

The dataset then reports `cond_names = bbox_x, bbox_y, bbox_z, volume, area` +
the sidecar names, and `condition_names` selects from the merged list. Names and
transforms come from `general_modules/condition_names.py`: stress, displacement
and frequency labels are stored as **natural logs** (`log_max_ver_stress_mpa`,
`log_max_tor_magdisp_mm`, `log_first_mode_freq_hz`, ...), mass and the absolute
volume/area as identity. Every condition value the FM sees, every `cond_values`
entry and every sidecar row is in that stored space; `from_stored(name, value)`
converts back to MPa / mm / Hz / kg. The ex5 recipe conditions on the
decorrelated six

```text
volume, area, log_max_ver_stress_mpa, log_max_dia_stress_mpa, log_max_tor_stress_mpa, log_first_mode_freq_hz
```

and deliberately not on `mass_kg` (identical to `volume` at constant density),
`surface_area_mm2`, the displacements (r 0.91 with their stress), the horizontal
case or the 2nd mode -- see the research note for the analysis.

### Partial requests (`cond_dropout_mode per_dim`)

Under the default `cond_dropout_mode all` the FM learns one null embedding and
a request must specify every condition. `per_dim` drops each condition entry
independently during training, feeds the model the observed-mask alongside the
(null-filled) values, and thereby lets a sample request leave entries
unspecified: write the literal `nan` in `cond_values` (`0.30,4.8,6.8024,nan,nan,8.1017`).
`sample.py` builds the mask from the `nan`s and raises a clear error when the
checkpoint was trained with `all`. An unspecified entry is filled in by the
model from the entries given, through their correlations -- it is not "free".

### Sample-time accuracy: candidate ranking, E2 Newton correction, C2 guidance

`candidate_multiplier` decodes N times as many candidates and keeps the best by
measured geometric condition error. `newton_rounds > 0` (E2) then corrects each
retained latent: a differentiable soft volume/area proxy provides the Jacobian,
the real Marching Cubes measurement decides whether a damped step is accepted
(pilot on ex1: volume median error 7.6% -> 0.28% in three rounds).
`guidance_enabled` (C2) steers the ODE itself toward the requested volume/area
through a one-step endpoint prediction (pilot: 7.6% -> 1.7%). Both work in
calibrated proxy units and need `descriptor_calibration_path`, the artifact
`config_calibrate_descriptors.txt` writes for the exact VAE/FM pair; a mismatch
is refused. Both act on `volume` / `area` only. `condition_audit fea|surrogate`
re-measures FEA-named conditions on the decoded meshes with `design_loop`
(relative-only numbers; falls back to the geometric audit with one message when
gmsh/pyamg or the surrogate are unavailable).

`config_sample.txt` omits `cond_values`, so it draws reproducible random
samples from the model's unconditional branch even though the FM was trained
conditionally. `cfg_scale 1.0` is plain conditional guidance when conditions
are supplied; increasing it can reduce diversity and does not guarantee
physical accuracy -- on ex1 it made the volume error 2.5x worse.

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
batch used by `config_sample.txt`, selects indices 0 and 1, and interpolates
between them at `alpha 0.5`. `interpolation_space` chooses how:

- `slerp_noise` (default): the two endpoints' FM *source noise* vectors are
  spherically interpolated and all three noises are integrated through the FM
  ODE, so the endpoints reproduce the original samples exactly and the midpoint
  is itself an on-manifold sample.
- `lerp_latent` (legacy): `torch.lerp` in normalized FM latent space, a
  straight line the FM never trained on.
- `cond_sweep` (`config_cond_sweep.txt`): one fixed noise row
  (`sample_index_a`) integrated `sweep_steps` times while the condition vector
  moves from `cond_values_a` to `cond_values_b` in normalized condition space --
  the controllable morph of a conditional checkpoint. Writes
  `sample_<seed>_sweep_<k>.stl`, a strip PNG and metadata with each panel's
  requested and measured conditions and `body_count_raw`. `nan` entries are
  allowed for `per_dim` checkpoints; run several seeds before reading a trend.

The two noise-space modes export:

- the two endpoint STLs and interpolated STL;
- a three-panel PNG comparison;
- JSON metadata with paths, mesh reports (including `body_count_raw`, the
  component count before the largest body is kept), the interpolation space,
  and the eps-space and latent-space endpoint distances.

`source_num_samples` must match the original sampled batch because seeded RNG
reproduction depends on the tensor shape. The current interpolation mode is
unconditional and requires `0 <= alpha <= 1`.

## Evaluation

`config_evaluate.txt` (`mode evaluate`) scores a trained VAE on a split it
never saw. The architecture and the encoder point budget come from the
checkpoint's saved config, and so does the split: the split keys
(`split_seed`, `split_by_parent`, `overfit_all_shapes`, `overfit_num_shapes`)
default to the checkpoint's values and are overridden only where the run config
actually sets them. Omitting them therefore reproduces the training split, which
is what you want; setting them to something else silently rescores a different
split and the "held-out" shapes are no longer held out. The evaluation keys
(`eval_split`, `eval_num_shapes`, `eval_seed`, `mc_resolution`,
`latent_refine_*`) and `dataset_dir` always come from the run config.

Every shape is encoded deterministically, optionally refined (below), decoded at
`mc_resolution`, and meshed. The report lists per-shape and aggregate:

- `surface_mean` / `p95` / `max` -- distance from every stored GT surface point
  to the reconstructed mesh, computed with `open3d`'s `RaycastingScene` when it
  imports and `trimesh.proximity.closest_point` otherwise (the choice is
  recorded as `surface_distance_backend`),
- `pred_to_gt_mean` / `p95` and `chamfer_mean` -- the reverse direction (8192
  points sampled on the reconstruction to the GT surface cloud) and the average
  of the two means. The one-sided number alone rewards a noisy space-filling
  reconstruction, because every GT point still finds some nearby surface,
- `sdf_l1`, `sign_accuracy`, `sign_balanced_accuracy` (the mean of the inside
  and outside rates, so its trivial baseline is 0.5 rather than the
  majority-class `positive_fraction` the rows also record),
- `body_count_raw`, watertightness, and validity.

With `latent_refine_steps > 0` both the encoder-mean (`enc_*`) and refined
(`ref_*`) rows are reported, and the stored query points are halved by a seeded
mask: refinement fits one half and both prefixes are scored on the other, so
`ref_ - enc_` is a held-out comparison. The fit-half numbers are labelled
`ref_*_insample`. Output: `<output_dir>/eval_<split>.json` and
`eval_<split>.csv`.

`latent_refine_steps` / `latent_refine_lr` / `latent_refine_prior_weight` run
Adam on the latent alone with the decoder frozen, minimizing the truncated-L1
SDF loss plus a surface term and an optional pull toward the encoder's latent.
That pull is `||z - z0||^2` **summed** over the latent scalars and averaged over
the batch, so its weight does not shrink as the latent grows; `config_evaluate.txt`
ships `0.0`. The same keys are accepted by `mode reconstruct`, where the labels
are sampled from the normalized input mesh. Default `0` = encoder mean only.

Refinement says little on an **undertrained** checkpoint: a decoder that has not
converged is nearly z-insensitive, so the loss barely moves and whichever of
`ref_` / `enc_` wins in mesh space is marching-cubes noise. Check
`refine_loss_first` against `refine_loss_last` before reading the gap.

### Other evaluate tasks (`eval_task`)

`eval_task reconstruction` is the default and everything above.
`descriptor_calibration` (`config_calibrate_descriptors.txt`) samples
`calibration_num_shapes x calibration_samples_per_shape` latents under the
split's true conditions, measures each through the soft proxy and through the
export path, fits `proxy = a * true + b` per descriptor and writes
`descriptor_calibration_path`; calibrate on `val`. `conditional`
(`config_evaluate_conditional.txt`) benchmarks the FM's condition accuracy: for
`eval_num_shapes` test shapes the target is the shape's own stored condition
vector, every method in `eval_methods` (`plain`, `rejection`, `c2`, `e2`,
`c2e2`) starts from the same seeded noise, and the report lists per method and
condition the relative error in raw units (median / p95), validity, latent drift,
NFE and wall time. Both need `fm_modelpath`. The parent-grouped test split is
in-distribution in condition space, so this measures realising seen condition
values with unseen bracket families, not extrapolation.

## Output contracts

Sampling writes `sample_<seed>_<index>.stl` for valid zero crossings and one
`sample_<seed>_meta.json` file. The JSON also lists rejected candidates, so a
requested index can appear without an STL. Conditional runs include a
condition audit based on the descriptors measured from each decoded mesh,
which entries of the request were specified, the `cond_dropout_mode` of the
checkpoint, the Newton history and latent drift when `newton_rounds > 0`, the
guidance settings when enabled, the NFE per candidate, and which
`condition_audit` backend actually ran.

Reconstruction is still available as an advanced native mode. A minimal config
needs `mode reconstruct`, `vae_modelpath`, `input_mesh`, `output_dir`, and
`mc_resolution`; it writes `<input_basename>_recon.stl`. Add
`latent_refine_steps` (with `latent_refine_lr`, `latent_refine_prior_weight`)
to refine the encoder's latent against SDF labels sampled from the input mesh
before decoding.

## SDF conventions

- Shapes are normalized to fit inside approximately `[-0.9, 0.9]^3`; queries
  cover `[-1, 1]^3`.
- SDF is negative inside and positive outside. The dataset builder flips the
  sign returned by `trimesh.signed_distance`.
- Reconstruction loss truncates SDF targets to `clamp_dist` (default `0.1`),
  while predictions remain unclamped so out-of-band errors retain gradients.
- Real input meshes must be watertight after any requested repair.

See the suite-level [configuration reference](../../docs/CONFIGURATION.md)
for the complete config, checkpoint, and output schema. The research document
[`GEOMETRY_GENERATION_RESEARCH.md`](../../docs/research/sdfflow/GEOMETRY_GENERATION_RESEARCH.md) explains
the design motivation but is not the runtime source of truth.
