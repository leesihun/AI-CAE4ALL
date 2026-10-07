# SDFFlow geometry generation

SDFFlow trains an SDF variational autoencoder (VAE) and then a latent
flow-matching (FM) model. The VAE converts geometry to and from a compact
latent representation; the FM learns to generate those latents. Generated
latents are decoded to an SDF grid and exported through Marching Cubes as STL.

## Recommended workflow

Every checked-in config lives under `configs/SDFFlow/geometry_generation/<exN>/baseline/`, one
directory per dataset (each config's header states what it adapts from the
paper and what it does not):

| slot | dataset | conditions |
| --- | --- | --- |
| `ex1` | DeepJEB brackets (`ex1_deepjeb.h5`, parent-grouped split) | `volume, area` |
| `ex2` | DrivAerML cars | the 16 geometry parameters + 5 force coefficients |
| `ex3` | MCB nut subset | `bbox_x, bbox_y, bbox_z, volume, area`, five `class_*` one-hots, `hole_count` |

From the `AI-CAE4ALL` root, validate and run the merged training config:

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/geometry_generation/ex1/baseline/config_train_sdfflow.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/geometry_generation/ex1/baseline/config_train_sdfflow.txt
```

The single `mode train` job runs the VAE first and starts FM training only
after it verifies that the VAE checkpoint completed successfully. This keeps
the GPU occupied without requiring a second manual launch.

Score the trained VAE on its held-out split, then generate or compare shapes:

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/geometry_generation/ex1/baseline/config_evaluate_sdfflow.txt     # mode evaluate
python AI_CAE4ALL_main.py --config configs/SDFFlow/geometry_generation/ex1/baseline/config_infer_sdfflow.txt        # mode sample
python AI_CAE4ALL_main.py --config configs/SDFFlow/geometry_generation/ex1/baseline/config_interpolate_sdfflow.txt  # mode interpolate
```

The `ex1` configs are the DeepJEB arm of the dataset matrix and set `gpu_ids`
for an 8-GPU box; on a smaller machine preflight reports `ENV-CUDA-002` until
you change it.

The FEA-conditioned track lets a designer ask for a bracket by the numbers that
matter -- volume, area, the peak stress of a load case, the first mode -- and
leave the rest unspecified. It needs DeepJEB's FEA labels appended to the
dataset once:

```bash
cd methods/SDFFlow && python add_fea_conditions.py --h5 ../../dataset/geometry_generation/ex1_deepjeb.h5     --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv --dry_run   # then without --dry_run
```

No checked-in config trains it any more: set `condition_names` to the FEA
names below and `cond_dropout_mode per_dim` in a copy of the ex1 training
config. The design, the label analysis behind the condition set and the list
of what is unverified until it trains are in
[`docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md`](../../docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md).

Direct backend commands are also supported. Run these from
`methods/SDFFlow` so relative paths keep their native meaning:

```bash
python SDFFlow_main.py --config ../../configs/SDFFlow/geometry_generation/ex1/baseline/config_train_sdfflow.txt
python SDFFlow_main.py --config ../../configs/SDFFlow/geometry_generation/ex1/baseline/config_infer_sdfflow.txt
```

## Canonical configs and artifacts

Each slot has the same four configs; `ex1` adds the closed loop and `ex3` the
two conditional-accuracy tasks. `<out>` is the slot's output directory:
`output/dataset_matrix/geometry_generation/ex1/sdfflow/` for ex1 and
`output/geometry_generation/<exN>_<dataset>/sdfflow/` for ex2 and ex3.

| Config (`<exN>/baseline/`) | Purpose | Main output |
| --- | --- | --- |
| `config_train_sdfflow.txt` | Merged VAE -> FM training (`mode train`) | `<out>/vae.pth`, then `<out>/fm.pth` and `fm_best.pth` |
| `config_evaluate_sdfflow.txt` | `eval_task reconstruction`: held-out reconstruction metrics for the VAE | `<out>/eval/eval_test.{json,csv}` |
| `config_infer_sdfflow.txt` | Reproducible unconditional generation (`mode sample`) | `<out>/samples/` |
| `config_interpolate_sdfflow.txt` | Reproduce samples 0 and 1 and decode a noise-space slerp between them | `<out>/interpolation/` |
| `ex1/.../config_optimize_sdfflow.txt` | `mode optimize`: closed-loop bracket design search over the trained generator (`opt_analysis fea`; the Studio's surrogate preset switches it to the ex10 HI-MGN) | `<out>/optimize/` |
| `ex3/.../config_calibrate_sdfflow.txt` | `eval_task descriptor_calibration`: affine calibration of the soft volume/area proxy on the val split | `<out>/eval/descriptor_calibration.pth` |
| `ex3/.../config_evaluate_conditional_sdfflow.txt` | `eval_task conditional`: paired-noise condition-accuracy benchmark on the test split | `<out>/eval/eval_conditional.{json,csv}` |

The merged config writes the pipeline log to `pipeline_log_file` and the stage
logs to `vae_log_file_dir` / `fm_log_file_dir` (`<out>/pipeline.log`,
`vae.log`, `fm.log`). The historical `geometry_generation` output slug is
retained for checkpoint compatibility; the runtime itself lives under
`methods/SDFFlow/`.

The checked-in training configs are merged (`mode train`). The native
`train_vae` mode compares VAE recipes before the FM is trained once for the
winner; `train_fm` remains available for focused debugging.

Optional `vae_best_modelpath` saves, after every validation, the EMA (or raw)
model with the best validation SDF loss so far, in the same payload format as
the final checkpoint. The final save to `vae_modelpath` is unchanged and stays
the file the pipeline's completeness check and the FM stage read.

Optional `fm_best_modelpath` does the same for the FM stage, writing the full
checkpoint payload (the frozen VAE embedded, so the file is a stand-alone
inference artifact) whenever the validation loss improves; `fm_modelpath` still
gets the final epoch. Selecting an FM epoch is safe because the FM is the last
stage and its objective has no warmup ramp, so every validation value is
comparable. Do not try the same trick on the VAE: the FM embeds and normalizes
against the exact VAE it trained on, so a different VAE epoch means retraining
the FM. Set the VAE's epoch budget instead.

## Pipeline restart behavior

`skip_completed_stages True` (set in every checked-in training config) is safe
to use when relaunching `config_train_sdfflow.txt`:

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

The DeepJEB (ex1) training config selects only:

```text
volume, area
```

and the MCB (ex3) config all five.

The selected order comes from `condition_names` in the training config and is
saved in the FM checkpoint. `bbox_y` is excluded because normalization makes
that dimension exactly constant in DeepJEB, and ex1 also drops `bbox_x`
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
python add_fea_conditions.py --h5 ../../dataset/geometry_generation/ex1_deepjeb.h5 --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv --dry_run
python add_fea_conditions.py --h5 ../../dataset/geometry_generation/ex1_deepjeb.h5 --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv
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
`config_calibrate_sdfflow.txt` (ex3) writes for the exact VAE/FM pair; a mismatch
is refused. Both act on `volume` / `area` only. `condition_audit fea|surrogate`
re-measures FEA-named conditions on the decoded meshes with `design_loop`
(relative-only numbers; falls back to the geometric audit with one message when
gmsh/pyamg or the surrogate are unavailable).

`config_infer_sdfflow.txt` omits `cond_values`, so it draws reproducible random
samples from the model's unconditional branch even though the FM was trained
conditionally. `cfg_scale 1.0` is plain conditional guidance when conditions
are supplied; increasing it can reduce diversity and does not guarantee
physical accuracy -- on ex1 it made the volume error 2.5x worse.

A conditional request (`cond_values`) that moves one condition beyond the
observed training range is an extrapolation experiment. `mode sample`:

- rejects a request beyond `max_condition_z` sigmas (3 in the infer configs)
  under the default `condition_ood_policy error` (`warn` / `clamp` relax it);
- clips extreme normalized latents when `latent_clip` is set (default off);
- decodes extra candidates and ranks them by measured geometric descriptors
  when `candidate_multiplier` > 1;
- records requested, normalized, extrapolated, and actual conditions in the
  sample metadata.

Extrapolation remains an out-of-distribution experiment, not evidence that the
model is reliable far outside the training range.

## Interpolation

`config_interpolate_sdfflow.txt` recreates the seed-42, `source_num_samples`
(16) unconditional batch, selects indices 0 and 1, and interpolates between them
at `alpha 0.5`. On ex2 and ex3 that is the batch `config_infer_sdfflow.txt` draws
(`num_samples 16`); ex1's infer config draws 209, and the noise comes from a
device generator whose draw depends on the tensor shape, so its endpoints are
not that run's samples 0 and 1. `interpolation_space` chooses how:

- `slerp_noise` (default): the two endpoints' FM *source noise* vectors are
  spherically interpolated and all three noises are integrated through the FM
  ODE, so the endpoints reproduce the original samples exactly and the midpoint
  is itself an on-manifold sample.
- `lerp_latent` (legacy): `torch.lerp` in normalized FM latent space, a
  straight line the FM never trained on.
- `cond_sweep` (no checked-in config): one fixed noise row
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

`config_evaluate_sdfflow.txt` (`mode evaluate`) scores a trained VAE on a split it
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
the batch, so its weight does not shrink as the latent grows; the evaluate
configs leave it (and `latent_refine_steps`) at the default `0`. The same keys are accepted by `mode reconstruct`, where the labels
are sampled from the normalized input mesh. Default `0` = encoder mean only.

Refinement says little on an **undertrained** checkpoint: a decoder that has not
converged is nearly z-insensitive, so the loss barely moves and whichever of
`ref_` / `enc_` wins in mesh space is marching-cubes noise. Check
`refine_loss_first` against `refine_loss_last` before reading the gap.

### Other evaluate tasks (`eval_task`)

`eval_task reconstruction` is the default and everything above.
`descriptor_calibration` (`ex3/.../config_calibrate_sdfflow.txt`) samples
`calibration_num_shapes x calibration_samples_per_shape` latents under the
split's true conditions, measures each through the soft proxy and through the
export path, fits `proxy = a * true + b` per descriptor and writes
`descriptor_calibration_path`; calibrate on `val`. `conditional`
(`ex3/.../config_evaluate_conditional_sdfflow.txt`) benchmarks the FM's condition accuracy: for
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
