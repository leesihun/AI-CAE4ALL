# SDFFlow maintainer notes

This file records the live repository contract for agents and maintainers.
`SDFFlow_main.py`, the checked-in configs, and their consumers remain the
authoritative implementation.

## Commands and working directories

From the `AI-CAE4ALL` root:

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample_extrapolation.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_interpolate.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_optimize.txt
# FEA-conditioned track (ex5): sidecar first (see "Data and condition invariants"), then
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3_fea.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_calibrate_descriptors.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample_conditional.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_cond_sweep.txt
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate_conditional.txt
```

The checked-in config roster under `configs/SDFFlow/`:

| Config | Mode | Output | Role |
| --- | --- | --- | --- |
| `config_train.txt` | train | `ex1/` | Tier-1 control: one 256-d token, MLP decoder, four conditions |
| `config_train_v2.txt` | train | `ex2/` | "v2 control": 32 x 64 VecSet + DiT + hybrid loss; architecture frozen, KL corrected to ex1 parity |
| `config_train_v3.txt` | train | `ex4/` | **Recommended single-GPU recipe** (see "v3 recipe" below) |
| `config_train_b300.txt` | train | `ex3/` | 8-GPU DDP twin of v3 (batch/LR/epoch arithmetic in its header) |
| `config_train_v3_fea.txt` | train | `ex5/` | v3 conditioned on `volume,area` + four DeepJEB FEA labels with `cond_dropout_mode per_dim`; needs the `cond_extra` sidecar (see "Data and condition invariants") |
| `config_evaluate.txt` | evaluate | `ex4/eval/` | Held-out VAE reconstruction metrics for the ex4 VAE (`eval_task reconstruction`) |
| `config_calibrate_descriptors.txt` | evaluate | `ex5/eval/descriptor_calibration.pth` | `eval_task descriptor_calibration`: fits the soft-proxy affine calibration on the ex5 val split |
| `config_evaluate_conditional.txt` | evaluate | `ex5/eval_conditional/` | `eval_task conditional`: paired-noise condition-accuracy benchmark (plain / rejection / e2) on the ex5 test split |
| `config_sample.txt`, `config_sample_extrapolation.txt` | sample | `ex1/samples*/` | Unconditional / guarded conditional generation |
| `config_sample_conditional.txt` | sample | `ex5/samples_conditional/` | Partial request (two `nan` entries) + candidate ranking + E2 Newton correction on the ex5 pair |
| `config_interpolate.txt` | interpolate | `ex1/interpolation/` | Noise-space slerp between two reproduced samples |
| `config_cond_sweep.txt` | interpolate | `ex5/cond_sweep/` | `interpolation_space cond_sweep`: one noise row under a five-step condition morph |
| `config_optimize.txt`, `config_optimize_surrogate.txt` | optimize | `ex1/optimization*/` | Closed-loop design search (FEA / HI-MGN backend) |
| `arms/A0.txt` .. `arms/A9.txt` | train_vae | `arms/<label>/` | VAE ablations of v3 (A1 is an omnibus control, A9 the seed-repeat noise floor); see `arms/README.md` |

The ex5 track is a design that has not been trained yet: every ex5 config
preflights with `PATH-INPUT-001` on the missing checkpoints, and everything
its mechanisms claim beyond the ex1 pilot measurements is listed as unverified in
`docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md`.

From `methods/SDFFlow`:

```bash
python build_dataset.py --output ../../dataset/synthetic256.h5 --synthetic 256
python SDFFlow_main.py --config ../../configs/SDFFlow/config_train_v3.txt
python SDFFlow_main.py --config ../../configs/SDFFlow/config_sample.txt
```

The config parser accepts flat `key value` text, lowercases keys and string
values, and treats `%` lines as comments. Path-valued keys (`dataset_dir`,
`output_dir`, `input_mesh`, `*_modelpath`, `descriptor_calibration_path`, the
log dirs — see `general_modules/load_config.py::PATH_KEYS`) are exempt from the
value lowercasing and keep the case you wrote. Relative native paths resolve from
the `methods/SDFFlow` repository even when the suite launcher is used.

Valid modes are `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`,
`interpolate`, `optimize`, and `evaluate`. Production training uses `train`.
`train_vae` is what the `configs/SDFFlow/arms/` VAE sweep runs (the FM is then
trained once, for the winning arm, through `train` with `skip_completed_stages`
reusing that VAE); `train_fm` is retained for targeted debugging. `evaluate`
scores a trained VAE on a held-out split (below).

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
output/geometry_generation/ex1/train.log
output/geometry_generation/ex1/train_vae.log
output/geometry_generation/ex1/train_fm.log
output/geometry_generation/ex1/sdfflow_vae.pth
output/geometry_generation/ex1/sdfflow_fm.pth
output/geometry_generation/ex1/samples/
output/geometry_generation/ex1/samples_extrapolation/
output/geometry_generation/ex1/interpolation/
```

(Paths are shown from the suite root. The configs spell them `../../output/...`
because the native process runs with its cwd set to `methods/SDFFlow`. Nothing
is ever written inside the method directory -- `output/` at the repo root is the
only artifact destination, and `geometry_generation` is SDFFlow's historical
slug there.)

and, for the v3 recipe (`config_train_v3.txt`):

```text
output/geometry_generation/ex4/sdfflow_vae.pth        final-epoch VAE (the pipeline's completeness check reads this)
output/geometry_generation/ex4/sdfflow_vae_best.pth   best-validation VAE (vae_best_modelpath; same payload format)
output/geometry_generation/ex4/sdfflow_fm.pth
output/geometry_generation/ex4/eval/eval_val.json     config_evaluate.txt
output/geometry_generation/ex4/eval/eval_val.csv
```

`vae_best_modelpath` is optional and unprefixed (it is not a `vae_<setting>`
stage key, so `build_stage_config` passes it through to the VAE worker
untouched). When present, `train_vae.py` saves the EMA (or raw) model with the
best validation SDF loss so far after every validation, with `epoch` set to
that epoch. The final save to `vae_modelpath` is unchanged, and it is the
final checkpoint -- not the best one -- that the pipeline's completeness check
and the FM stage read. Point `config_evaluate.txt` at whichever you want scored.

## Data and condition invariants

The HDF5 layout is
`shapes/{index:05d}/{surface_points,surface_normals,sdf_points,sdf_values,cond}`.
Root `cond_names` and every `cond` row contain the raw five descriptors:

```text
bbox_x, bbox_y, bbox_z, volume, area
```

FM training may select a subset through `condition_names`. `config_train.txt`
(ex1) and `config_train_v2.txt` (ex2) use `bbox_x,bbox_z,volume,area`; the v3
recipe (`config_train_v3.txt`, `config_train_b300.txt`) uses only `volume,area`,
because `bbox_y` has exactly zero train-split standard deviation and `bbox_x` a
0.45% coefficient of variation in DeepJEB -- conditioning on near-constants only
adds noise to the CFG branch. Do not add `bbox_y` without first showing that its
train-split standard deviation exceeds `min_condition_std`.
The selected names, statistics, extrema, and clipping limit are stored in the
FM checkpoint and define `cond_values` order at inference.

**FEA-label conditions live in a sidecar, not in the `cond` rows.** DeepJEB's
per-design FEA labels (`bracket_labels.csv`: mass, per-load-case max von Mises
stress and max displacement, the first two eigenfrequencies, ...) are appended to
an existing HDF5 by `add_fea_conditions.py` as the root dataset `cond_extra`
(float32 `[num_shapes, k]`, row i = shape i) with root attrs `cond_extra_names`,
`cond_extra_source`, `cond_extra_transforms` (JSON name -> `identity|log`) and
`cond_extra_created`. `general_modules/sdf_dataset.py` merges them:
`cond_names = base names + extra names`, `cond_dim = len`, and `get_cond` /
`__getitem__['cond']` return the concatenation; `compute_cond_stats` is
unchanged. A file without `cond_extra` behaves exactly as before, so the sidecar
is append-only and backward compatible. The join key is the per-shape `source`
basename without extension (`DeepJEB/SurfaceMesh/101_428.stl` -> `101_428`)
against the CSV's `item_name` (2138/2138 match); the builder refuses unmatched
shapes (`--allow_missing` writes NaN rows with a WARNING), refuses to replace an
existing sidecar without `--overwrite`, and prints per-name mean/std/min/max of
the stored values. `--dry_run` does everything but write.

`general_modules/condition_names.py` is the name registry: `GEOMETRIC_NAMES`
(the five descriptors above) and `FEA_CONDITIONS`, a dict `name -> {csv_column,
transform, unit, kind, load_case}`. Stress, displacement and frequency labels are
stored as **natural logs** (`log_max_ver_stress_mpa`, `log_max_ver_magdisp_mm`,
`log_first_mode_freq_hz`, ... ; raw skew +0.7..+1.3 becomes <= 0.6), mass and
the absolute volume/area as identity (`mass_kg`, `volume_mm3`,
`surface_area_mm2`). Conditions in the FM checkpoint, `cond_values` in a sample
config and the dataset rows are all in **stored** space -- `to_stored(name,
raw)` / `from_stored(name, stored)` convert (identity for geometric names),
`describe(name)` returns the entry or `None`, `is_fea(name)` tells the two
families apart, and `describe(...)['load_case']` is the key into
`design_loop/problem.py::LOAD_CASES` when a condition is re-measured on a decoded
mesh. All names are lowercase.

**The FEA condition set is decorrelated, not exhaustive.** `config_train_v3_fea.txt`
(ex5) conditions on `volume,area,log_max_ver_stress_mpa,log_max_dia_stress_mpa,
log_max_tor_stress_mpa,log_first_mode_freq_hz` -- participation ratio 2.53
against 1.43 for v3's `volume,area`. The rest of the registry is deliberately
left out as duplicates: `mass_kg` and `volume_mm3` are `4.793 x volume`
(constant Ti-6Al-4V density, extent CV 0.8%), `surface_area_mm2` is
`10468 x area`, the displacements are 0.58-0.85 predictable from geometry and
r 0.91 with their own stress, the horizontal case is recovered from the vertical
one, and the 2nd mode is r 0.95 with the 1st. Torsion stress is the one label
orthogonal to everything else (R^2 -0.04 from geometry) and also the riskiest
(ICC 0.06); expect its conditional accuracy to be the lowest and audit it only
relatively. Keep the sidecar complete anyway -- the audit and the reports read
names the FM is not conditioned on. Shape h5 index 2099 (`131_561`, a v3
**test**-split shape) is a partial STL carrying full-bracket labels; drop it
from any conditional evaluation. The analysis behind every number here is
`docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md`.

**Partial conditions -- `cond_dropout_mode` (`all` default | `per_dim`).**
`model/velocity_net.py` keeps the legacy behaviour under `all`: one Bernoulli
mask per sample, a learned `null_cond` embedding, parameter set and forward
outputs bit-identical to the pre-key file, so every existing FM checkpoint loads
unchanged. `per_dim` draws an independent Bernoulli mask per condition entry with
probability `cond_dropout`, feeds `concat([where(mask, cond, null_values),
mask.float()])` (width `2 * cond_dim`, `null_values` a learned `nn.Parameter`
`[cond_dim]`) to the condition MLP, and uses the all-masked row as the
unconditional branch. `forward(z, t, cond=None, cond_mask=None)` accepts no
mask, a `[B]` bool mask (legacy semantics) or a `[B, cond_dim]` bool mask; in
mode `all` a `[B, cond_dim]` mask is reduced with `.all(dim=1)` and a warning.
At inference a `cond_values` entry written as the literal `nan` means
"unspecified": `sample.py` builds `cond_mask = ~isnan` and raises a clear
`ValueError` if the checkpoint was not trained `per_dim`. The mode is stored in
the FM checkpoint config; nothing else is needed. `cond_dropout_all_prob`
(default 0.1, per_dim only) is the EXPLICIT drop-all term: independent
per-entry dropout would draw the fully unconditional row -- the branch
`sample_latents` evaluates under CFG -- with probability `cond_dropout **
cond_dim`, 6.4e-5 at the ex5 settings (about 111 rows in a 1000-epoch run), so
without it a CFG ablation on such a checkpoint measures a starved branch rather
than CFG. Setting it to 0 restores the chance-only behaviour and then
`cfg_scale` must stay 1.0 (`SDF-CDROP-003` warns). The one thing the mode still
does NOT do: an unspecified entry is not "free" -- the model fills it in from
the entries given through their correlations (stress and displacement are about
-0.7 with volume).

Conditioned sampling enforces `max_condition_z` with an `error`, `warn`, or
`clamp` policy. The checked-in extrapolation config uses `error`, applies
`latent_clip`, and uses `candidate_multiplier` so decoded candidates can be
ranked by actual geometric-condition error. Preserve the requested and actual
condition audit in `sample_<seed>_meta.json` when changing inference. The ex5
test-split targets reach `|z|` 3.82 (area) and 3.21 (torsion stress) under
train statistics, so a conditional evaluation on it needs `max_condition_z >= 4`.

`cfg_scale 1.0` is the live conservative setting. Higher CFG is a strength and
diversity tradeoff, not a substitute for condition accuracy or OOD validation:
measured on ex1 (36 samples, 6 held-out shapes), `cfg_scale 3.0` made the
volume error 2.5x worse than 1.0 (8.5% -> 21.2%). Condition accuracy comes from
the sample-time tools below and from `candidate_multiplier`, not from CFG.

## Sample-time descriptor accuracy (C2 guidance, E2 Newton, calibration, audit)

Three opt-in mechanisms, all bit-for-bit inert when off, implement the pilot
recipes of `docs/research/sdfflow/GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md`
section 2 (measured there on the ex1 checkpoint: plain conditional volume median
error 7.6%, C2 1.7%, E2 0.28%, C2+E2 0.077%). They act only on a conditional
request and only on **geometric** targets that the soft SDF proxy can compute
(`volume`, `area`); FEA-named targets are skipped with a printed note and are
measured after decoding by the audit instead.

- `general_modules/descriptor_proxy.py::soft_descriptors(vae, z_flat, names,
  resolution=48, tau=0.032, bound=1.0, chunk=32768)` -- differentiable volume and
  area on a **cell-centre** grid (`h = 2 * bound / resolution`, centres at
  `-bound + h * (i + 0.5)`): `volume = sum(sigmoid(-sdf / tau)) * h^3`, `area =
  sum(||grad occ||) * h^3` by finite differences on the occupancy grid. `sphere_sdf`,
  `box_sdf` and `MockDecoder` (latent `[:, 0]` = sphere radius) exist so the tools
  are testable without a trained VAE; `tests/test_conditional_tools.py` checks
  them against closed-form values (soft volume of an exact sphere to < 0.01%).
- `general_modules/descriptor_calibration.py` -- the proxy is biased but almost
  linear in the Marching Cubes measurement (ex1: volume `soft ~= 0.86 true +
  0.40`, R^2 0.98; area R^2 0.60), so C2/E2 work in **proxy units**:
  `proxy_target = a * true + b` (`DescriptorCalibration.proxy_target`, the
  pilot's forward map). `calibrate(vae, fm_model, latent_mean, latent_std,
  cond_batches | z_batches, ...)` generates, decodes each latent once through the
  proxy and once through the export path at `measure_resolution`, and fits
  `fit_affine(proxy, true) -> {a, b, r2, n}` per name. The artifact
  (`save`/`load`, a plain `torch.save` dict) records the VAE and FM SHA-256s,
  `resolution`, `tau`, `measure_resolution`, `cond_names`, split and counts;
  `check_compatible(vae_path, fm_path, resolution, tau, measure_resolution=None)`
  raises `ValueError` on any mismatch, because silently reusing stale
  coefficients is what turned the pilot's uncalibrated guidance into a 23%
  volume error; passing `measure_resolution` also pins the Marching Cubes grid
  the slope was fitted against, which `sample`/`evaluate` do (the effective
  `newton_measure_resolution`, else `mc_resolution`). The "true" measurement is
  `true_descriptors`, which reports **NaN volume for a non-watertight mesh**
  exactly as `mesh_report` reports `None` -- never `convex_hull.volume`, which
  `sdf_sampling.mesh_descriptors` substitutes and which runs 2-4x the solid
  volume on a holed bracket. So a torn row drops out of the volume fit (its
  area pair is kept) and the printed watertight rate says how many rows the
  volume slope really had; `calibration_min_r2` (default 0.5, 0 disables)
  refuses to save a fit too weak to steer with. `mode evaluate` with
  `eval_task descriptor_calibration` writes it to `descriptor_calibration_path`.
- `general_modules/descriptor_refinement.py::newton_correct(vae, z_flat, targets,
  calibration, latent_mean, latent_std, rounds=3, step_cap_rms=0.12,
  line_search_tries=3, measure_resolution=96, resolution=48, tau=0.032) ->
  (z_flat, history)` -- E2. Per round: measure the true descriptors on the res-96
  export path, take the proxy Jacobian `J` (k x D) by autograd w.r.t. the
  **normalized** flat latent, form the residual `a * (target - true)` in proxy
  units, solve the damped minimum-norm step `dz = J^T (J J^T + 1e-6 I)^-1 r`,
  cap its RMS at `step_cap_rms * sqrt(D)`, and backtrack `dz, dz/2, dz/4`,
  accepting a step only if the decoded mesh is valid and the TRUE relative
  residual norm decreases. Because a torn mesh has NaN volume, its residual is
  `inf` and it can never be accepted; `require_watertight` additionally
  defaults to True whenever `volume` is among the corrected names, and
  `latent_clip` (passed from the run config) clamps each candidate before it is
  measured so the corrected latent obeys the same box every other arm does.
  A hybrid quasi-Newton step, not a Newton step on one objective; a round that
  accepts nothing ends the loop. `measure_resolution` follows `mc_resolution`:
  the calibration slope is fitted against the export path at that grid and the
  audit reports there, and MC volume moves 0.02-0.10% between res 96 and 128 --
  10-35% of the accuracy E2 is quoted at.
- `general_modules/descriptor_guidance.py::make_c2_guidance(vae, fm_model, cond,
  cond_mask, targets, calibration, eta=0.1, t_start=0.3, step_mode='velocity_dt',
  resolution=48, tau=0.032, ode_steps_ref=50, latent_mean, latent_std) ->
  callable(z_next, t_next, dt) -> delta_z` -- C2. For `t_start <= t_next < 1`:
  `x1_hat = z_next + (1 - t_next) * fm_model(z_next, t_next, cond, cond_mask)`,
  `loss = sum ((soft(x1_hat) - proxy_target) / proxy_target)^2` on the
  de-normalized latent, `g = grad(loss, z_next)` RMS-normalized per sample, and
  `delta = -eta * (1 - t_next) * g * scale`. `per_step_jump` (`scale = 1`) is the
  pilot's per-step state jump, whose total strength grows with `ode_steps`;
  `velocity_dt` (`scale = dt * ode_steps_ref`) treats the correction as a
  velocity integrated over `dt`, so the total is NFE-invariant and equals the
  pilot at 50 steps. Zero outside the window; runs under `torch.enable_grad()`
  even though the sampler is `no_grad`.
- `sample_latents(model, num_samples, latent_flat_dim, device, cond=None,
  cfg_scale=1.0, ode_steps=50, generator=None, noise=None, cond_mask=None,
  guidance_fn=None)` -- `cond_mask` is passed to every conditional model call
  (CFG's unconditional branch is the all-masked row under `per_dim`);
  `guidance_fn` is called AFTER each Euler update as `delta = guidance_fn(z_next,
  t_next, dt)` and `z_next += delta`. With both `None` the integration is
  bit-identical to the pre-guidance sampler.

`sample.py` keys (defaults = off / legacy): `guidance_enabled False`,
`guidance_t_start 0.3`, `guidance_eta 0.1`, `guidance_step_mode velocity_dt`,
`guidance_targets volume,area`, `soft_descriptor_resolution 48`,
`soft_descriptor_tau 0.032`, `descriptor_calibration_path` (path key; required
when guidance or Newton is on), `newton_rounds 0`, `newton_step_cap_rms 0.12`,
`newton_line_search_tries 3`, `newton_measure_resolution` (defaults to
`mc_resolution`), and `condition_audit geometric|fea|surrogate`. Order of operations: sample (with
guidance) -> decode all candidates -> rank by geometric condition error -> keep
`num_samples` -> Newton-correct each retained latent -> re-decode -> export. The
metadata records the request (specified / unspecified per name), the
`cond_dropout_mode`, per-candidate `actual_conditions` and relative errors, the
Newton history and latent RMS drift, the NFE per candidate, and which audit
backend ran. `condition_audit fea` gmsh-meshes each retained mesh and solves the
load cases the FEA-named conditions need with `design_loop` (`Bracket` with
`opt_length_scale`, `opt_material_*`, `opt_stress_percentile`); `surrogate`
uses `design_loop/surrogate.py`'s HI-MGN bridge (`opt_surrogate_config` /
`opt_surrogate_checkpoint`). **The audit's own unit constants are calibrated
against the labels, not against the design loop**: `opt_length_scale` defaults
to `0.1838 / 1.8` m per unit and `opt_material_rho` to 4470 kg/m3 here (the
183.8 mm mean longest extent and the mean-mass / mean-volume density of
`bracket_labels.csv`), because the optimisation defaults 0.19 m and 4430 kg/m3
put a fixed +8.7% / +9.7% / +6.4% median bias into `mass_kg` / `volume_mm3` /
`surface_area_mm2` -- and `mass_kg` needs no solver, so that bias would be the
entire reported "error" on a perfect decode. The residual floor is the 0.79%
extent CV (2.4% on a volume). `fea_audit` metadata records
`length_scale_source` / `density_source` and the bias note. All `opt_*` value
validators (`SDF-OPT-NU-001`, `SDF-OPT-POSITIVE-001`,
`SDF-OPT-PERCENTILE-001`, `SDF-OPT-SURROGATE-001`) run in `sample` and
`evaluate` too, not only under `mode optimize`. Both report the
raw (MPa / mm / kg) and the stored-space value next to the request, and both
fall back to the geometric audit with one printed message when gmsh/pyamg, the
surrogate checkpoint or `design_loop` are unavailable. Treat their numbers as
**relative only**: tet4 is stiff, the solver reports the 99.5th-percentile von
Mises rather than DeepJEB's max, and the default length scale is 3.3% above the
dataset's mean extent (183.8 mm).

## Interpolation invariants

`inference_profiles/interpolate.py` reproduces an unconditional FM batch from
`seed` and `source_num_samples`. `interpolation_space` selects where the
interpolation happens:

- `slerp_noise` (**default**): the source batch noise is drawn explicitly,
  `eps = torch.randn(source_num_samples, latent_flat_dim, generator=...)` --
  bit-identical to the draw `sample_latents` makes internally -- and the two
  endpoints' noise vectors are spherically interpolated (`omega =
  acos(clamp(cos))`, falling back to lerp when `sin(omega) < 1e-6`). The stacked
  `(eps_a, eps_mid, eps_b)` is then integrated through
  `sample_latents(..., noise=...)`, so the endpoints reproduce the original
  samples exactly and the midpoint is an on-manifold FM sample rather than a
  point on a straight line the FM never trained on.
- `lerp_latent` (legacy): `torch.lerp` in normalized FM latent space, today's
  pre-v3 behaviour, kept for comparison.
- `cond_sweep`: the **controllable morph** for a conditional checkpoint. One
  fixed base noise -- row `sample_index_a` of the seeded batch -- is integrated
  `sweep_steps` times (default 5; `alphas = linspace(0, 1, sweep_steps)`) while
  the condition vector is lerped in **normalized** condition space from
  `cond_values_a` to `cond_values_b` (lists in checkpoint `cond_names` order;
  `nan` = unspecified, `per_dim` checkpoints only). All rows go through ONE
  `sample_latents(noise=eps.repeat, cond=stack)` call, each is decoded and
  written as `sample_<seed>_sweep_<k>.stl`, the triptych plotter is generalised
  to an N-panel strip PNG, and the metadata records per panel the requested and
  measured (geometric audit) conditions and `body_count_raw`. `sample_index_b`
  and `alpha` are not used; the launcher requires `cond_values_a`/`_b` instead
  (`SDF-SWEEP-003`). Read the strip with the metadata: a jump between adjacent
  panels (extra bodies, a topology change) is a property of the flow map, not a
  rendering artefact, and a single seed can misrepresent a condition -- run
  three or more.

`sample_latents(..., noise=None)` is the hook: when `noise` of shape
`[num_samples, latent_flat_dim]` is given it is cloned as the `t=0` state
instead of being drawn. Endpoint indices must be distinct and in range, and
`alpha` must be within `[0, 1]`. Keep `source_num_samples` equal to the original
sampling batch size when reproducing existing endpoint indices (for `cond_sweep`
that is `num_samples x candidate_multiplier` of the sampling run). Conditional
interpolation between two *noises* is still rejected; `cond_sweep` is the
conditional path.

The `slerp_noise` / `lerp_latent` modes write three STLs, a triptych PNG, and
JSON metadata recording `interpolation_space`, the eps-space and z-space
endpoint distances, and each mesh's `body_count_raw`. A missing zero crossing
is a hard failure because all three comparison meshes are required.

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

**Load-unit correction (2026-09).** `problem.py::Bracket.LOAD_CASES` used to hold
the GE challenge's imperial numerals -- 8000 / 8500 / 9500 lbf and 5000 lbf*in --
and `fea.py` read them as N and N*m next to SI material constants. DeepJEB's
labels were produced with the SI values: vertical 35.6 kN (+z), horizontal
37.8 kN, diagonal 42.3 kN at 42 deg, torsion 565 N*m. The table is now
converted explicitly (`LBF_TO_N`, `LBIN_TO_NM`), so the three forces were
**4.448x too small** and the torsion moment **8.85x too large** before the
change. Consequences to keep straight: (a) linear statics makes every stress and
displacement scale exactly with a common force factor, and the optimize loop's
allowables are the median of a population analyzed under the same factor, so the
*relative* ranking of designs from earlier `optimize` runs under the default
`vertical,diagonal` cases is unchanged and those runs keep their ranking; but
their absolute stresses and deflections were understated about 4.4x, and any
run that mixed the torsion case with a force case weighted torsion 39x too
heavily against it -- so no recorded absolute stress/deflection and no
force-vs-torsion trade-off from those runs may be quoted. (b) Even with SI loads,
tet4 stiffness, the 99.5th-percentile stress measure and the default length
scale (190 mm against the dataset's 183.8 mm mean extent) keep every audit and
optimize number **relative only**. The repo's frame is also its own (horizontal
along +y, the diagonal decomposed 42 deg from the horizontal); the DeepJEB paper
labels its horizontal case +x and the challenge reads "42 deg from vertical".
`problem.py`'s docstring flags both as deliberate no-rotation choices, not fixes.
`inference_profiles/sample.py::load_cases_used()` reads the live table at
runtime for the audit metadata, so nothing duplicates these numbers.

**The search mesh is a ranking device.** `opt_target_faces` / `opt_mesh_size_max`
size the per-evaluation mesh; the winner and the baseline are then re-analyzed
together on the finer `opt_verify_*` mesh. `summary.json` records the
same-design shift between the two meshes under `mesh_sensitivity`; treat the
search-phase feasibility flag as relative only.

Artifacts in `output_dir`: `optimized.stl`, `baseline.stl`, `summary.json`,
`history.json` (every evaluation, including failures and timings),
`convergence.png`, `report.md`.

## v3 recipe (`config_train_v3.txt`) and the keys it introduced

Every new key defaults to the pre-v3 behaviour when absent; old configs and
checkpoints are untouched. The reasons behind each setting are in the config's
own header; this section records the contracts.

**Parent-grouped split -- `split_by_parent` (bool, default False).** DeepJEB's
2138 shapes are variants of **263 parent geometries**: the per-shape `source`
attr is `DeepJEB/SurfaceMesh/<parent>_<variant>.stl`, and the parent id is the
basename split at the first `_` (`101_428.stl` -> `101`; a shape with no
`source` or no `_` is its own parent). The seed-42 per-shape random split put a
sibling of **every** val/test shape into train, so validation loss measured
memorization of a parent, not generalization. With `split_by_parent True`,
`sdf_dataset.build_dataset_splits` permutes the *parents* with
`np.random.default_rng(split_seed)` and assigns whole parents to
train/val/test greedily until each reaches ~80/10/10 of the *shape* count (val
and test get at least one parent each), then prints the split sizes and the
parents per split. `train_vae`, `train_fm`, and `evaluate` all call the same
function, so they see identical splits. It is a VAE *and* FM compatibility key
in `train_pipeline.py`, so changing it invalidates checkpoint reuse.

**Encoder point budget -- `num_encoder_points` 6144 of the stored 8192.** The
per-shape draw is `rng.choice(stored, num_encoder_points, replace=False)`, so
setting it equal to the stored cloud size returns a *permutation* of one fixed
point set -- and the cross-attention encoder is permutation-invariant, so the
shape's encoder input is then bit-identical for every epoch of the run. That
subsample is the only surface augmentation this 1713-shape train set has (v2 drew
4096 of 8192). v3 briefly shipped 8192 and silently lost it; the recipe now draws
6144 (75%), which keeps more supervision per step than v2 while the draw stays
stochastic. `config_train_b300.txt` matches. Arm `A8` (4096) is the density axis.
Val/test and the FM latent-cache encode pass subsample deterministically, so they
are unaffected. `num_encoder_points` is a VAE *and* FM compatibility key, and
`evaluate` reads it from the checkpoint, not the run config.

**Run seeding -- `seed` (int, optional; absent = legacy unseeded run).**
`training_profiles/setup.py::seed_stage` seeds `torch`, `numpy`, and `random`
with `seed + rank` -- offset by the distributed rank so no two ranks draw the
same shuffle order or posterior/flow noise -- and `seeded_generator(run_seed)`
gives the train DataLoader a generator at the same value. Module construction is
wrapped in `identical_across_ranks`, which puts torch back on the
rank-independent base seed for that block: `wrap_model` builds FSDP without
`sync_module_states`, so every rank shards whatever it initialized and the
initial weights must be bit-identical. `seed` does NOT seed
`SDFShapeDataset`'s per-item train subsample -- that fresh draw is the surface
augmentation and stays stochastic (val/test and the latent-cache pass are seeded
through `deterministic=True` instead). The key is unprefixed, so
`build_stage_config` passes it to both stage workers untouched. `config_train_v3.txt`, `config_train_b300.txt`, and every arm ship
`seed 0`; `arms/A9.txt` is A0 at `seed 1`, and `|A9 - A0|` is the sweep's
run-to-run noise floor. Without a seed a sweep has no such floor and any arm gap
is unfalsifiable, which is why the launcher lists `seed` as recommended for
`train`, `train_vae`, and `train_fm`.

**Deterministic subsampling.** `SDFShapeDataset(deterministic=False)` gained a
`deterministic` attribute: when set, `__getitem__` draws its encoder/query
subsample from `np.random.default_rng([seed, shape_idx])` instead of the
unseeded module rng. `build_dataset_splits` sets it for the val and test
datasets (train stays stochastic), and `train_fm._encode_split` forces it on
for the latent-cache encode pass and restores it afterwards, so every rank and
every run encodes identical latents.

**Encoder queries -- `encoder_query_type` (`learned` default | `fps`).** `fps`
farthest-point-samples `latent_tokens` input points per sample (starting from
the point farthest from the centroid; deterministic given the point set),
embeds them with the *same* `point_proj` features (Fourier(xyz) + normals), and
uses them as the initial cross-attention queries -- 3DShape2VecSet's
geometry-anchored tokens -- instead of the learned `nn.Parameter` queries. Pair
it with `decoder_type attention`: with `fps` the token order depends on the
input point set, so the flattened-token `mlp` decoder would see an
input-dependent channel layout (launcher warning `SDF-QUERY-002`, an error under
`--strict`).

**Posterior std floor -- `posterior_min_std_rel` (float, default 0.0 = off).**
`reparameterize(mu, logvar, noise_scale=1.0, min_std=None)` uses
`std_eff = max(exp(0.5 * logvar), rel * mu_spread)`, where `mu_spread` is a
per-latent-dim running estimate (momentum 0.99, shape `[latent_dim]`, updated
from the detached batch std of `mu` over batch and tokens during training only).
The buffer is registered **only when rel > 0**, so pre-v3 checkpoints still load
strictly. `SDFVAE.forward(..., posterior_min_std_rel=0.0)` is how the trainer
passes it. v3 pairs `posterior_noise_max_scale 1.0` (real reparameterization;
the legacy 0.1 sampled at a tenth of sigma, so the decoder never learned to
tolerate the width the FM later samples from) with a 2% floor as a
variance-collapse guard.

**KL arithmetic.** The KL term is *summed* over latent elements. ex1's
`kl_weight 1e-5` over 256 dims is a total pressure of 2.56e-3; v3 keeps that
total with `0.0000025 x 1024`, and v2 was corrected from `1e-4 x 2048` (80x
tighter than ex1) to `0.00000125 x 2048`. 3DShape2VecSet's `1e-3` is a
**mean**-KL weight, about 6e-8 per element here, so the old "nudge toward 1e-3"
comment pointed the wrong way. Write these as decimals: `1e-4` has no `.` and
stays a string in both config parsers.

**Training-log contract (`train_vae.py`).** When the hybrid loss is on, the
printed and logged epoch line carries the raw unweighted per-term means
`Surface: x Normal: x Eikonal: x`. Validation lines add:

- `ValidSign: x` -- sign accuracy over the val loader on labels with
  `|sdf| > 0.001`, and `ValidSignBal: x`, the same balanced (the mean of the
  inside-rate and the outside-rate, accumulated from per-class counts over the
  whole split rather than averaged per batch). SDF queries are
  majority-outside, so the raw number has a floor well above 0.5 and a decoder
  that only reproduces the majority class still looks accurate on it.
- two active-unit counts over the latent scalars (tokens x dims):
  `ActiveUnits: k/D`, whose variance of `mu` across the whole validation set
  exceeds an absolute 0.01 (Burda's count), and `ActiveSNR: k/D`, the
  scale-free form `Var_x(mu_d) / mean_x(sigma_d^2) > 1` (per-dim signal over
  posterior noise). Read them together and compare arms on the SNR one: the
  absolute threshold moves with the latent's overall scale, so `ActiveUnits` is
  not comparable across runs that change `latent_dim` or KL pressure, while the
  ratio is. At startup both `train_vae.py` and
`train_fm.py` compute `updates_total = steps_per_epoch * epochs` and, if
`use_ema` and `(1 - ema_decay) * updates_total < 10`, print a WARNING naming the
retained-init fraction `ema_decay ** updates_total` and a suggested decay
`1 - 10 / updates_total`. (The old b300 FM config made 1,200 updates at 0.999
and kept 30% of the random init in its EMA.)

**Latent refinement -- `latent_refine_steps` (int, default 0 = off),
`latent_refine_lr` (0.01), `latent_refine_prior_weight` (0.0).**
`inference_profiles/latent_refine.py::refine_latent(vae, z_flat, surface_points,
surface_normals, query_points, query_sdf, steps, lr, prior_weight,
clamp_dist=0.1)` runs Adam on `z` only, decoder frozen, in fp32, minimizing the
truncated-L1 SDF loss (same as `model.sdf_loss` on the clamped target) `+ 0.1 *
mean|f(surface_points)| + prior_weight * ||z - z0||^2`, and returns the detached
refined `z_flat`. The prior term is `(z - z0).pow(2).sum(dim=-1).mean()`:
**summed over the latent scalars**, averaged over the batch. It used to be a mean
over D, which divided it by 1024 and put it about five orders of magnitude below
the SDF term's gradient (measured 3.0e-6 against 3.7e-1) -- inert at every weight
a user would plausibly write, and silently more inert the larger the latent got.
`config_evaluate.txt` ships `latent_refine_prior_weight 0.0`, i.e. unconstrained,
because that is what this mode is for: refinement measures how much better the
frozen decoder can do than the encoder found, so pulling `z` back toward the
encoder's answer destroys the measurement. A sweep on a smoke checkpoint is
monotone in that direction -- held-out SDF gain +0.66% at `|dz|` 5.17 (weight 0),
+0.19% at 0.51 (1e-4), +0.05% at 0.089 (1e-3), +0.01% at 0.0097 (1e-2) -- and the
fit-half and held-out-half errors move together, so there is no overfitting for
the prior to protect against. Raise it to 1e-4..1e-3 only when the refined
latents will be fed back to the FM stage, where drift off the encoder manifold is
the real risk; calibrate against `latent_shift_l2` in the per-shape rows.

`reconstruct` builds the labels with
`general_modules.sdf_sampling.sample_mesh_sdf` on the normalized input mesh
(`num_surface=num_enc`, `num_near=4096`, `num_uniform=1024`, seeded rng);
`evaluate` uses the shape's stored `sdf_points`/`sdf_values`. Measured on v2,
latent optimization beat the encoder 2.4x -- refinement is the cheap way to
separate encoder error from decoder capacity.

**Refinement is a near-no-op on an undertrained checkpoint, so do not read a
smoke run's numbers as a result.** An undertrained decoder is essentially
z-insensitive: the refinement loss barely moves, `latent_shift_l2` stays small,
and whichever of `ref_` / `enc_` wins on the mesh-space metrics is
marching-cubes noise rather than a measurement of the encoder. The 2.4x figure
above comes from a fully trained v2 VAE. Check `refine_loss_first` against
`refine_loss_last` in the per-shape rows before interpreting any `ref_ - enc_`
gap.

**`evaluate` mode (`inference_profiles/evaluate.py::run_evaluate`).** Keys:
`vae_modelpath`, `dataset_dir`, `output_dir` (required), `eval_split`
(`train|val|test`, default `val`), `eval_num_shapes` (0 = all), `eval_seed`
(0), `mc_resolution` (128), the `latent_refine_*` keys, and `split_seed` /
`split_by_parent`. `eval_task` selects what is scored:

- `reconstruction` (default) -- everything in this section: held-out VAE
  reconstruction metrics, VAE only.
- `descriptor_calibration` -- generate `calibration_num_shapes x
  calibration_samples_per_shape` samples (conditions drawn from `eval_split`'s
  true stored conditions), fit the soft-proxy affine calibration per descriptor
  and save the `DescriptorCalibration` to `descriptor_calibration_path`
  (required; `SDF-EVAL-007`). Calibrate on `val`.
- `conditional` -- for `eval_num_shapes` seeded-random shapes of `eval_split`
  (drawn from the pool left after `eval_exclude_shapes`; ship `2099` on
  DeepJEB, a partial STL carrying full-bracket labels)
  the target is that shape's TRUE stored condition vector restricted to the
  checkpoint's `cond_names`; every method in `eval_methods` (default
  `plain,rejection,e2`; allowed `plain|rejection|c2|e2|c2e2`) starts from the
  SAME base noise per shape (paired `z0`, seeded by `eval_seed` and the shape
  index). `rejection` uses `candidate_multiplier`, `c2` `make_c2_guidance` with
  the `guidance_*` keys, `e2` `newton_correct` with the `newton_*` keys. Per
  method and per condition name it reports the relative error `|actual -
  target| / |target|` in RAW units (`from_stored` for the log names) as median
  and p95, the valid/watertight rate, the latent RMS drift from the plain
  sample, the NFE count and wall time, to `eval_conditional.json` / `.csv` plus
  a printed table. FEA-named conditions are scored only when `condition_audit`
  is `fea` or `surrogate` (else reported as "not measurable geometrically").

Both non-reconstruction tasks need `fm_modelpath` (`SDF-EVAL-004`; the FM is
probed against the VAE like in `sample`). The parent-grouped split is
in-distribution in condition space (test nearest-train distance ratio about
1.0, <= 0.5% of test values outside the train range), so `conditional` on
`test` measures whether seen condition values can be realised with unseen
bracket families -- not condition extrapolation, which needs explicitly
out-of-range targets.

**Where the split comes from.** The architecture, `num_encoder_points`, and
`clamp_dist` are read from the checkpoint's saved config, and so is the split:
`ds_config` starts as a copy of the checkpoint config, and only the `SPLIT_KEYS`
(`split_seed`, `split_by_parent`, `overfit_all_shapes`, `overfit_num_shapes`)
actually **present in the run config** override it. Omitting them therefore
reproduces the training split exactly, which is the safe default; writing them
with different values silently rescores a *different* split, and the "held-out"
shapes are then not held out. `dataset_dir` and the `eval_*` keys always come
from the run config.

Per shape: a deterministic encoder subsample (dataset in deterministic mode with
`eval_seed`) -> `mu` -> optional refinement -> decode at `mc_resolution` ->
`sdf_grid_to_mesh`. Metrics, once per prefix:

- `surface_mean/p95/max` -- exact distance from every stored GT surface point to
  the reconstructed mesh. The backend is chosen at runtime:
  `open3d.t.geometry.RaycastingScene` when it imports (about 30x faster), else
  `trimesh.proximity.closest_point`; the choice is printed once and recorded in
  the JSON summary as `surface_distance_backend`.
- `pred_to_gt_mean/p95` and `chamfer_mean` -- the **other** direction: 8192
  points sampled on the reconstruction (`trimesh.sample.sample_surface`, seeded)
  to their nearest stored GT surface point through a KD-tree, and
  `0.5 * (surface_mean + pred_to_gt_mean)`. One-sided `surface_*` cannot see
  geometry the reconstruction invents where the GT has none -- a noisy
  space-filling field scores well on it -- so compare shapes on `chamfer_mean`.
- `sdf_l1`, `sign_accuracy`, `sign_balanced_accuracy`, `positive_fraction` on
  the stored query points (`|target| > 0.001`, via `vae.decode_flat`). Raw sign
  accuracy has a majority-class floor at the `positive_fraction` each row
  records (~0.64 outside points here); the balanced form averages the inside and
  outside rates, so its trivial baseline is 0.5.
- `body_count_raw`, `watertight`, `valid`, `faces`, `volume`.

With `latent_refine_steps > 0` both the encoder-mu (`enc_*`) and refined
(`ref_*`) rows are reported, and the stored query points are **split in half by
a seeded mask** (`np.random.default_rng([eval_seed, shape_idx]).random(n) <
0.5`): refinement fits one half and BOTH prefixes are scored on the other, so
`ref_ - enc_` is a held-out comparison rather than a measure of fit. The
half-split fields carry a `_heldout` suffix on both prefixes --
`enc_sdf_l1_heldout`, `enc_sign_accuracy_heldout`,
`enc_sign_balanced_accuracy_heldout` and the same three under `ref_` -- so the
pair to quote against each other is `enc_sdf_l1_heldout` vs
`ref_sdf_l1_heldout`; plain `enc_sdf_l1` stays on ALL query points and is the
number comparable across configs with and without refinement. The fit-half
numbers are kept, explicitly labelled, as `ref_sdf_l1_insample` /
`ref_sign_accuracy_insample`; the JSON summary records the arrangement under
`refine_query_split`. The `ref_` rows also carry `refine_seconds`,
`refine_loss_first`, `refine_loss_last`, and `latent_shift_l2`.

Output: `<output_dir>/eval_<split>.json` (aggregate mean/median + per-shape
rows), `eval_<split>.csv`, and a printed aggregate table.

**Mesh and SDF plumbing.** `mesh_extraction.sdf_grid_to_mesh` records the
component count *before* `keep_largest` as `mesh.metadata['body_count_raw']`
(set on the kept mesh after selection, since `split()` drops metadata), and
`mesh_report` exposes it as `body_count_raw`. `sdf_sampling._signed_distance`
tries `igl`, then `open3d.t.geometry.RaycastingScene.compute_signed_distance`
(sign convention verified so inside is negative), then the trimesh fallback,
and prints the backend once per process. `build_dataset.py --near_sigmas
0.01,0.05` (any length >= 1; sigma drawn uniformly per near point) is forwarded
to `sample_mesh_sdf(near_sigmas=...)`, and the builder records `num_surface`,
`max_faces`, `sharp_edge_fraction`, `sharp_edge_angle`, `near_sigmas`, `seed`,
and `sdf_backend` as root attrs.

**Launcher contract.** All of the above keys are in `SDFFLOW_KEYS`
(`cae_suite/specs/sdfflow.py`) and in Studio's `sdfflow` catalog
(`studio/src/constants.js`); `tests/test_studio_spec_key_parity.py` enforces the
parity. Diagnostics: `SDF-QUERY-001/002` (`encoder_query_type`),
`SDF-QUERY-003` (ERROR: `encoder_query_type fps` with `latent_tokens >
num_encoder_points` -- there are not enough input points to anchor one query
each, and the tiling fallback repeats tokens, which is a quality collapse rather
than a degradation), `SDF-NOISE-002` (`posterior_min_std_rel >= 0`),
`SDF-REFINE-001/002/003` (`latent_refine_*`), `SDF-INTERP-006`
(`interpolation_space`, now including `cond_sweep`), `SDF-EVAL-001/005/006`
(`eval_split`, `eval_num_shapes`, `eval_seed`). `seed` is
*recommended* for the three training modes, and `split_by_parent` for
`evaluate`. `reconstruct` and `evaluate` both default `latent_refine_lr` to 0.01
and `latent_refine_prior_weight` to 0.0, so Studio's required rows for those
modes have values `SDF-REFINE-002` accepts (an lr of 0 is rejected).
`cae_suite/preflight.py::_probe_dataset` counts `evaluate` among the modes whose
dataset comes from `dataset_dir`; without that the `sdf_hdf5` schema probe never
ran for it and a wrong-contract dataset preflighted clean, then died on a
`KeyError: 'shapes'` inside the native run. `vae_best_modelpath` and
`descriptor_calibration_path` are in both `PATH_KEYS` sets. `train_pipeline.py`
treats `encoder_query_type`, `posterior_min_std_rel`, and `split_by_parent` as
VAE compatibility keys and `split_by_parent` as an FM compatibility key.

The conditional-generation keys and their diagnostics (all in `SDFFLOW_KEYS`,
Studio's catalog, `CHOICES` and `HELP`): `cond_dropout_mode` --
`SDF-CDROP-001` (`all|per_dim`); `condition_names` with FEA names --
`SDF-COND-FEA-001` NOTICE (sidecar needed) and `SDF-COND-FEA-002` NOTICE when
paired with mode `all`; `cond_values` / `cond_values_a` / `cond_values_b`
entries -- `SDF-COND-003` ERROR unless number or `nan`, `SDF-COND-PARTIAL-001`
NOTICE when a `nan` is present; guidance -- `SDF-GUIDE-001`
(`guidance_step_mode` `velocity_dt|per_step_jump`), `-002` (`guidance_eta > 0`),
`-003` (`soft_descriptor_tau > 0`, `soft_descriptor_resolution > 0`), `-004`
(`guidance_t_start` in `[0, 1)`), `-005` WARNING (a `guidance_targets` name
without a proxy), `-006` WARNING (guidance/Newton without `cond_values`);
Newton -- `SDF-NEWTON-001` (`newton_rounds` nonnegative int), `-002` (cap,
tries, resolution positive); calibration -- `SDF-CALIB-001` ERROR
(`descriptor_calibration_path` missing while guidance or Newton is on),
`SDF-CALIB-002` NOTICE (file does not exist yet), `SDF-CALIB-003` NOTICE (the
calibration task will overwrite an existing file), `SDF-CALIB-004`
(`calibration_num_shapes` / `calibration_samples_per_shape` positive);
`condition_audit` -- `SDF-AUDIT-001` (`geometric|fea|surrogate`); sweep --
`SDF-SWEEP-001` (`sweep_steps` nonnegative int, and an ERROR when it is < 2
under `cond_sweep`, where `interpolate.py` itself raises), `-002` NOTICE
(`sweep_steps` written outside `cond_sweep`, where it is inert),
`-003` ERROR (`cond_values_a`/`_b` missing under `cond_sweep`), `-004` ERROR
(length mismatch), `-005` WARNING (`cond_values` ignored by `cond_sweep`);
`SDF-CDROP-002` ERROR / `SDF-CDROP-003` WARNING (`cond_dropout_all_prob`
range, and 0 under `per_dim` leaving the CFG branch untrained);
`SDF-CALIB-005` (`calibration_min_r2` in [0, 1]); `SDF-EVAL-009`
(`eval_exclude_shapes` nonnegative indices); `SDF-COND-FEA-003` ERROR
(`condition_names` outside the dataset's merged vocabulary, raised from the
dataset layer with the sidecar-builder command as the hint);
`SDF-OPT-PERCENTILE-001` (`opt_stress_percentile` in (0, 100]);
`SDF-INTERP-007` ERROR (`sample_index_b` missing for `slerp_noise` /
`lerp_latent` -- it left `required_by_mode` because `cond_sweep` has no second
endpoint); evaluate -- `SDF-EVAL-002` (`eval_task`), `-003` (`eval_methods`
subset of `plain,rejection,c2,e2,c2e2`), `-004` ERROR (`fm_modelpath` missing
for a non-reconstruction task; the `fm_modelpath` INPUT_FILE PathRule now covers
`evaluate` too), `-007` ERROR (`descriptor_calibration_path` missing for the
calibration task or for `c2`/`e2`/`c2e2`), `-008` NOTICE (`eval_methods` outside
the conditional task). `descriptor_calibration_path` has an OUTPUT_FILE PathRule
for `evaluate` only -- the writability check the calibration task needs -- and
deliberately **no** INPUT_FILE rule for `sample`: the file legitimately does not
exist until the calibration task has run, so a hard input rule would make the
documented order (calibrate, then sample) fail preflight. Its input side is the
validator's job instead (`SDF-CALIB-001` / `-002`). One known artefact of the
output rule: under `eval_task conditional` the file is an *input*, so an existing
one also draws a `PATH-OUTPUT-EXISTS` "may overwrite" warning that does not apply
to that task -- read `SDF-CALIB-002` / `-003` instead. `PATH-OUTPUT-EXISTS` is not
promoted by `--strict`, so it cannot fail a strict check.

The spec's `sample` defaults mirror `inference_profiles/sample.py`'s own
`config.get(key, default)` calls one for one: `cfg_scale 1.0` (the native
default; the table used to claim 2.0), `ode_steps 50`, `mc_resolution 128`,
`guidance_enabled False`, `guidance_t_start 0.3`, `guidance_eta 0.1`,
`guidance_step_mode velocity_dt`, `guidance_targets volume,area`,
`soft_descriptor_resolution 48`, `soft_descriptor_tau 0.032`, `newton_rounds 0`,
`newton_step_cap_rms 0.12`, `newton_line_search_tries 3`,
`newton_measure_resolution 128` (it follows `mc_resolution` natively),
`condition_audit geometric`. `evaluate` adds
`eval_task reconstruction`, `condition_audit geometric`, `eval_methods
plain,rejection,e2`, `calibration_num_shapes 64`,
`calibration_samples_per_shape 4`, `calibration_min_r2 0.5`,
`candidate_multiplier 4` (evaluate's native default, where `sample`'s is 1) and
the same guidance / Newton / soft-proxy
values -- except `newton_rounds`, which `evaluate` `setdefault`s to **3** for its
`e2` method (a benchmark at 0 rounds is not a benchmark) and `sample` defaults to
0; one mode-wide value would misreport one of the two, so the spec lists neither
for `evaluate`. `interpolate` adds `sweep_steps 5`; `train` / `train_fm` add
`cond_dropout_mode all`, the `model/velocity_net.py` default that reproduces the
pre-conditional parameter set exactly.

## The arms sweep (`configs/SDFFlow/arms/`)

Ten `mode train_vae` configs around the v3 recipe, on the same data settings,
split, `seed`, and 500-epoch budget. Most move one axis off A0; **two do not**:

| Arm | Differs from A0 by |
| --- | --- |
| `A0` | nothing -- the v3 control |
| `A1` | **many axes at once**: the whole ex1 recipe (1 x 256 token, MLP decoder, learned query, legacy noise, KL 1e-5, no hybrid, AMP on). An omnibus control -- it answers "is v3 worth it at all", not which change earned it |
| `A2` | legacy posterior noise (`posterior_noise_max_scale 0.1`, `posterior_min_std_rel 0.0`) |
| `A3` | `kl_weight 0.000025` (10x) |
| `A4` | `encoder_query_type learned` |
| `A5` | hybrid terms off (`surface/normal/eikonal_weight 0`; `use_amp` stays False, so the arm changes the loss composition alone) |
| `A6` | `normal_weight 1.0` |
| `A7` | `latent_dim 64` (32 x 64 = 2048 flat) with `kl_weight 0.00000125`, halved so `weight x D` stays at A0's 2.56e-3 and the arm isolates capacity |
| `A8` | `num_encoder_points 4096` (A0: 6144) -- encoder input density; both ends are stochastic per-epoch draws |
| `A9` | `seed 1` and nothing else -- **not a treatment**: `|A9 - A0|` is the sweep's run-to-run noise band, and an arm gap smaller than it is not evidence |

`arms/roster.tsv` feeds `configs/campaigns/benchmarks_all/campaign_runner.py`
(`ex_slot` is `deepjeb`, a label only -- the runner never resolves it for
`--mode train`). `arms/README.md` has the launch and scoring recipe: each arm
is scored with a copy of `config_evaluate.txt` pointed at its checkpoint, the
arms train only the VAE, and the FM is trained once for the winner. The arm
files are named `A<n>.txt`, so `--audit-configs` (which globs `config*.txt`)
does not see them; validate them with per-file `--check`.

## Architecture and checkpoint facts

- SDF sign is negative inside and positive outside. Shapes occupy roughly
  `[-0.9, 0.9]^3`; queries cover `[-1, 1]^3`.
- VAE training supports deterministic, posterior-noise, and KL warmups, plus
  the optional relative posterior-std floor. FM consumes normalized encoder
  means rather than posterior samples.
- The latent is `latent_tokens x latent_dim`. ex1 uses one global 256-d token
  and the MLP decoder; the v3 recipe uses 32 x 32 FPS-anchored VecSet tokens
  and the attention decoder.
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
| `add_fea_conditions.py` | Appends DeepJEB's FEA labels to an HDF5 as the `cond_extra` sidecar (join on `source` basename == `item_name`) |
| `general_modules/condition_names.py` | Registry of geometric and FEA condition names, stored-space transforms (`to_stored` / `from_stored`), units, load cases |
| `general_modules/sdf_sampling.py` | Normalization, SDF samples, descriptors, repair path |
| `general_modules/sdf_dataset.py` | Lazy-open HDF5 dataset, seeded split, condition statistics; merges the `cond_extra` sidecar into `cond` |
| `general_modules/descriptor_proxy.py` | Differentiable soft volume/area on a cell-centre grid; analytic sphere/box SDFs and `MockDecoder` for tests |
| `general_modules/descriptor_calibration.py` | Affine proxy-vs-truth calibration artifact (`DescriptorCalibration`, `calibrate`, `fit_affine`), checkpoint-hash bound |
| `general_modules/descriptor_refinement.py` | E2: proxy-Jacobian Newton correction with RMS cap and true-measure backtracking (`newton_correct`) |
| `general_modules/descriptor_guidance.py` | C2: calibrated endpoint-prediction guidance callback for `sample_latents` (`make_c2_guidance`) |
| `general_modules/mesh_extraction.py` | Latent to SDF grid to Marching Cubes mesh/report |
| `model/sdf_vae.py` | Encoder, latent bottleneck, SDF decoders, reconstruction loss |
| `model/velocity_net.py` | Conditional rectified-flow loss and ODE sampler |
| `training_profiles/train_pipeline.py` | Sequential orchestration and compatibility-based reuse |
| `training_profiles/train_vae.py` | VAE stage worker |
| `training_profiles/train_fm.py` | Latent cache, condition selection, FM stage worker |
| `training_profiles/setup.py` | Device, optimizer/scheduler, EMA, logging, checkpoints |
| `inference_profiles/sample.py` | Sampling, OOD guard, partial (`nan`) requests, candidate ranking, C2 guidance / E2 Newton wiring, geometric / FEA / surrogate condition audit, reconstruction (+ optional latent refinement) |
| `inference_profiles/interpolate.py` | Reproducible noise-space slerp / latent lerp interpolation and triptych output; `cond_sweep` fixed-noise condition strip |
| `inference_profiles/latent_refine.py` | Decoder-frozen Adam refinement of a latent against SDF labels |
| `inference_profiles/evaluate.py` | `evaluate` mode: `eval_task` reconstruction metrics, descriptor calibration, paired conditional benchmark (JSON/CSV) |
| `inference_profiles/optimize.py` | `optimize` mode: config to loop, calibration, verification, reporting |
| `design_loop/generator.py` | Design vector to FM noise to latent to SDF grid to surface mesh |
| `design_loop/mesher.py` | gmsh tetrahedralization (reparametrization-free; see its docstring) |
| `design_loop/fea.py` | Tet4 linear elasticity, AMG solve, von Mises recovery |
| `design_loop/problem.py` | Bracket interfaces, GE load cases, mass objective with penalties |
| `design_loop/loop.py` | Evaluator, baseline calibration, CMA-ES driver, history |

## Validation after changes

At minimum, run from the suite root:

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_interpolate.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_optimize.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3_fea.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_calibrate_descriptors.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample_conditional.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_cond_sweep.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate_conditional.txt --check
for f in configs/SDFFlow/arms/A*.txt; do python AI_CAE4ALL_main.py --config "$f" --check --strict; done
python -m pytest -q tests/test_sdfflow_spec.py tests/test_studio_spec_key_parity.py tests/test_native_config_consumption_parity.py
cd methods/SDFFlow && python -m pytest -q tests/test_conditional_tools.py   # proxy / calibration / E2 / C2 on analytic SDFs, no checkpoint
```

**Boolean keys are read with `bool()`, so the validators are too.** The flat
parser types `true` as a bool but `1` as an int, and every native consumer
spells the read `bool(config.get(key, False))`. A validator that tested
`values.get(key) is True` therefore let `guidance_enabled 1` (guidance really
on) skip `SDF-CALIB-001`, and `use_conditions 1` (a genuinely conditional run)
skip `SDF-COND-001` and the dataset-layer `SDF-COND-FEA-003`. Both now go
through `cae_suite/specs/sdfflow.py::_flag` (and `preflight.py::_truthy` for
the dataset layer), which reads `1`/`yes`/`on` as true and `''`/`0`/`false`/
`no`/`off` as false. Use them for any new boolean key here.

Expect `PATH-INPUT-001` only for checkpoints that have not been trained yet, and
no `CFG-UNKNOWN-001`: a native `config.get('<key>')` whose key is missing from
`SDFFLOW_KEYS` is a launcher bug, not a config bug. `config_train_v3_fea.txt`
fails preflight against a `deepjeb.h5` without the sidecar with
`SDF-COND-FEA-003` naming the `add_fea_conditions.py` command: the dataset
probe now returns the merged condition vocabulary (`cond_names` +
`cond_extra_names`) and `cae_suite/preflight.py::_validate_sdf_against_config`
cross-checks `condition_names` against it. `train_pipeline.py` repeats the
check natively BEFORE stage 1, so even a direct native run fails in seconds
rather than after the whole VAE stage.

> The previously documented `pytest tests/test_sdfflow_pipeline.py
> tests/test_checked_in_configs.py tests/test_required_field_matrix.py` does not
> exist in this checkout (nor anywhere else in the tree) -- as with the stale
> root `testpaths`, per-config `--check` is the live validation. Verified
> 2026-08-30.

The `optimize` mode adds `gmsh`, `pyamg`, and `cma` to the requirement set.

Also run `python -m py_compile` on modified Python files. Documentation changes
must keep the config roster in "Commands and working directories" (the five
training configs, the three `config_evaluate*.txt` / `config_calibrate_descriptors.txt`
evaluate configs, the three sampling configs, the two interpolation configs, the
two `config_optimize*.txt`, and `arms/A0..A9`), the per-config DeepJEB condition
selections (four names for ex1/ex2, `volume,area` for v3, the six-name FEA set
for ex5), and the canonical `output/geometry_generation/ex1` /
`output/geometry_generation/ex4` / `output/geometry_generation/ex5` paths
synchronized (spelled `../../output/...` inside the configs, because the native
process runs in `methods/SDFFlow`; there is no `outputs/` directory anywhere in
this tree).

`GEOMETRY_GENERATION_RESEARCH.md` is design context, not implementation truth;
so is `docs/research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md`, which
records the label analysis and method decisions behind the ex5 track and what
remains unverified until it trains.
