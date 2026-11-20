# 09 — SDFFlow (SDF-VAE + latent flow matching)

## Summary

SDFFlow **generates 3D geometry**. It does not predict a physical field. A shape
is a signed distance function (SDF). A VAE compresses each shape into a set of
latent tokens. A rectified-flow model learns to sample those latents, optionally
conditioned on descriptors such as volume and area or on FEA labels. Marching
Cubes then turns each sampled latent into an STL. It is also the suite's only
closed-loop mode: `mode optimize` runs generation, meshing, a linear-static FEA
solve and CMA-ES over the trained generator.

Use it to sample new shapes, reconstruct or interpolate existing ones, search a
design space under structural constraints, or benchmark how well generated shapes
hit their conditions.

| | |
| --- | --- |
| Route id | `sdfflow` |
| Directory | [`methods/SDFFlow/`](../../methods/SDFFlow/) |
| Entrypoint | [`SDFFlow_main.py`](../../methods/SDFFlow/SDFFlow_main.py) |
| Spec | [`ai_cae4all/specs/sdfflow.py`](../../ai_cae4all/specs/sdfflow.py) |
| Maintainer notes | [`methods/SDFFlow/CLAUDE.md`](../../methods/SDFFlow/CLAUDE.md) (authoritative) |

It is not one paper. The encoder follows 3DShape2VecSet (Zhang et al., SIGGRAPH
2023, [arXiv:2301.11445](https://arxiv.org/abs/2301.11445)). The generator is
flow matching in latent space (Lipman et al., ICLR 2023,
[arXiv:2210.02747](https://arxiv.org/abs/2210.02747)), using the rectified-flow
path and a DiT velocity network with AdaLN-Zero.

## Architecture

Training has two stages. `mode train` runs both in sequence.

### Stage 1: SDF-VAE ([`model/sdf_vae.py`](../../methods/SDFFlow/model/sdf_vae.py))

- **`PointCloudEncoder`**
  - Surface points and normals get Fourier features and a projection to
    `encoder_dim`.
  - `latent_tokens` query tokens cross-attend to these point features over
    `encoder_blocks` blocks. Self-attention among the tokens is optional
    (`encoder_self_attention`).
  - `encoder_query_type` picks the queries. `fps` (every checked-in config)
    farthest-point-samples the tokens from the input points, so each token is
    anchored to a region of the shape. `learned` (the code default) uses free
    parameters.
  - Output: `mu, logvar` of shape `[B, latent_tokens, latent_dim]`.
- **Decoders**
  - `SDFDecoderAttention` (`decoder_type attention`, every checked-in config):
    query points cross-attend to the latent tokens.
  - `SDFDecoderMLP` (`decoder_type mlp`, the code default): a DeepSDF-style MLP
    with a mid-network skip, for the single-token layout.
  - Both initialize the SDF head with std 1e-5.
- **Losses** (`sdf_loss`, `hybrid_geometry_losses`)
  - L1 against a target truncated at `clamp_dist`.
  - KL weighted by `kl_weight`, with deterministic, posterior-noise and KL
    warmups.
  - Optional surface, normal and eikonal terms (`surface_weight`,
    `normal_weight`, `eikonal_weight`). These run in fp32 outside autocast and
    force the math SDPA backend, because they need double backward.
  - `posterior_min_std_rel` optionally floors the posterior std.

### Stage 2: latent flow matching ([`model/velocity_net.py`](../../methods/SDFFlow/model/velocity_net.py))

- **Path:** `z_t = (1 - t) * noise + t * data`, with target velocity
  `data - noise`.
- **Input:** normalized encoder means by default. With `fm_latent_draws K > 1`
  (every checked-in config uses 8), each training shape contributes K posterior
  samples instead. Validation always uses the mean. The latent and condition
  statistics come from the train split and are stored in the FM checkpoint.
- **`VelocityNet`:** `fm_arch dit` is a DiT over the token set; `fm_arch mlp`
  stacks `AdaLNBlock`s. Both use AdaLN-Zero with zero-initialized gates and a
  zero-initialized output projection.
- **Conditioning:** a timestep embedding plus a condition MLP. Condition dropout
  enables classifier-free guidance (`cfg_scale`).
  - `cond_dropout_mode all` drops the whole condition vector and uses a learned
    null condition.
  - `cond_dropout_mode per_dim` drops each entry on its own, so a sample request
    may leave conditions as `nan`.
- **Loss:** `flow_matching_loss`, an MSE on the velocity.
  `fm_time_sampling` (`uniform` or `logit_normal`) sets the timestep
  distribution.
- **Sampling:** `sample_latents` Euler-integrates for `ode_steps` steps. The
  frozen VAE decodes the result to an SDF grid at `mc_resolution`, and
  [`general_modules/mesh_extraction.py`](../../methods/SDFFlow/general_modules/mesh_extraction.py)
  runs Marching Cubes.

### Pipeline and inference

- [`training_profiles/train_pipeline.py`](../../methods/SDFFlow/training_profiles/train_pipeline.py)
  strips the `vae_*` / `fm_*` prefixes for each stage worker
  (`train_vae.py`, `train_fm.py`).
  - It reuses a complete, compatible VAE when `skip_completed_stages` is true.
  - It refuses to start FM unless the VAE checkpoint verifies.
  - A retrained VAE always invalidates the old FM.
- **The FM checkpoint embeds its frozen VAE.** It is the stand-alone inference
  artifact, so inference configs point `fm_modelpath` at `fm_best.pth`.
- [`inference_profiles/`](../../methods/SDFFlow/inference_profiles/) holds the
  inference modes:

  | File | Role |
  | --- | --- |
  | `sample.py` | Sampling, OOD guard, candidate ranking, C2 guidance, E2 Newton, condition audit, reconstruction |
  | `interpolate.py` | Interpolation |
  | `latent_refine.py` | Latent refinement against SDF labels |
  | `evaluate.py` | `mode evaluate` |
  | `optimize.py` | `mode optimize` |
- [`design_loop/`](../../methods/SDFFlow/design_loop/) holds the closed loop:

  | File | Role |
  | --- | --- |
  | `generator.py` | Design vector → FM noise through a gnomonic chart |
  | `mesher.py` | gmsh tetrahedral meshing |
  | `fea.py` | tet4 linear elasticity with an AMG solve |
  | `problem.py` | Load cases and penalized mass objective |
  | `loop.py` | CMA-ES driver |
  | `surrogate.py` | HI-MGN surrogate backend |

### Dataset tools ([`methods/SDFFlow/scripts/`](../../methods/SDFFlow/scripts/))

Standalone tools, run from `methods/SDFFlow` as `python scripts/<name>.py`. The
launcher does not route them.

| Script | Purpose |
| --- | --- |
| `build_dataset.py` | Meshes or synthetic primitives → SDF HDF5 (`--repair`: trimesh, then pymeshfix) |
| `add_fea_conditions.py` | DeepJEB FEA labels → `cond_extra` sidecar |
| `add_drivaerml_conditions.py` | DrivAerML design parameters and coefficients → `cond_extra` |
| `add_mcb_class_conditions.py` | MCB class one-hot (+ `--hole_count`) → `cond_extra` |
| `merge_sdfflow_checkpoint.py` | Merges a legacy separate VAE + FM pair into one FM checkpoint |

## Data contract

SDFFlow does **not** read the mesh HDF5 of
[DATASET_FORMAT.md](../reference/DATASET_FORMAT.md). It reads its own SDF layout,
`dataset_kind sdf_hdf5`:

```text
shapes/{idx:05d}/surface_points    [P, 3]
shapes/{idx:05d}/surface_normals   [P, 3]
shapes/{idx:05d}/sdf_points        [Q, 3]
shapes/{idx:05d}/sdf_values        [Q]       negative inside
shapes/{idx:05d}/cond              geometric descriptors
cond_names (root)                  bbox_x, bbox_y, bbox_z, volume, area
cond_extra (root, optional)        sidecar written by the scripts/add_*_conditions.py tools
```

Shapes are normalized to about `[-0.9, 0.9]^3`, and queries cover `[-1, 1]^3`.
`condition_names` selects and orders the conditions from `cond_names` plus
`cond_extra`. `split_by_parent` keeps sibling shapes in one split, which DeepJEB
needs. See "Data and condition invariants" in
[CLAUDE.md](../../methods/SDFFlow/CLAUDE.md) for the details.

Checked-in slots
([`configs/SDFFlow/geometry_generation/`](../../configs/SDFFlow/geometry_generation/)):

| Slot | Dataset | Conditions |
| --- | --- | --- |
| `ex1/baseline` | `ex1_deepjeb.h5` (`split_by_parent True`) | `volume, area` |
| `ex2/baseline` | `ex2_drivaerml.h5` | 16 DrivAerML design parameters + 5 aero coefficients |
| `ex3/baseline`, `ex3/tokens512` | `ex3_mcb.h5` | 5 geometric descriptors, 5 class one-hots, `hole_count` |

## Modes and keys

Modes: `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`, `interpolate`,
`optimize`, `evaluate`. Run `python AI_CAE4ALL_main.py --describe sdfflow` for the
required keys of each mode.

| Mode | Required beyond the basics | Notes |
| --- | --- | --- |
| `train` | architecture keys (`latent_tokens`, `latent_dim`, `decoder_*`, `encoder_*`, `num_encoder_points`, `num_query_points`, `fm_hidden`, `fm_blocks`, `fm_cond_hidden`), `vae_*` / `fm_*` epochs, batch size and learning rate | VAE then FM; `skip_completed_stages` defaults to true |
| `train_vae` / `train_fm` | the same keys, unprefixed | single stage |
| `sample` | `num_samples`, `seed`, `ode_steps`, `mc_resolution` | defaults: `cfg_scale 1.0`, guidance off, `newton_rounds 0` |
| `reconstruct` | `input_mesh`, `mc_resolution` | `latent_refine_steps` defaults to 0 |
| `interpolate` | `source_num_samples`, `sample_index_a` (+ `sample_index_b`) | `interpolation_space`: `slerp_noise`, `lerp_latent`, `cond_sweep` |
| `evaluate` | `vae_modelpath`, `dataset_dir`, `output_dir` | `eval_task`: `reconstruction`, `descriptor_calibration`, `conditional` |
| `optimize` | `opt_subspace_dim`, `opt_budget`, `opt_popsize`, `opt_baseline_size`, `opt_load_cases` | `opt_analysis fea` (default) or the surrogate backend; `opt_budget 0` screens instead of searching |

The checked-in train configs share one default recipe:

- **VAE:** 128 FPS tokens × 32 dims, attention decoder, `kl_weight 1e-8`.
- **Flow model:** DiT with 512 hidden and 12 blocks, `fm_time_sampling
  logit_normal`, `fm_latent_draws 8`, 250 epochs.

`ex3/tokens512` is the one variant, with 512 tokens.

Sample-time descriptor accuracy has three tools:

- candidate ranking (`candidate_multiplier`)
- C2 guidance (`guidance_enabled`)
- E2 Newton correction (`newton_rounds`)

C2 and E2 read a calibration file written by `eval_task descriptor_calibration`
(`descriptor_calibration_path`). Training scales out with `parallel_mode ddp` or
`fsdp`. Inference uses one GPU. `resume_training` is supported.

## Outputs

All outputs go under `output/`. ex1 writes to
`output/dataset_matrix/geometry_generation/ex1/sdfflow/`. ex2 and ex3 write to
`output/geometry_generation/<exN>_<dataset>/sdfflow/`. The `geometry_generation`
slug is historical.

| Artifact | Written by |
| --- | --- |
| `vae.pth`, `fm.pth`, `fm_best.pth` (stand-alone, VAE embedded), `vae.log`, `fm.log`, `pipeline.log` | `train` |
| `epoch#####_recon.png`, `epoch#####_samples.png` (beside the log) | periodic visualization (`display_testset`) |
| `samples/sample_<seed>_<i>.stl`, `sample_<seed>_meta.json` | `sample` |
| `interpolation/` | `interpolate` |
| `eval/eval_<split>.{json,csv}`, `eval/descriptor_calibration.pth`, `eval_conditional.{json,csv}` | `evaluate` (per `eval_task`) |
| `optimize/`: optimized/baseline/typical STLs, `summary.json`, `history.json`, `report.md`, `designs.h5`, plots, `screening.csv` (`opt_budget 0`), `fea_verified.json` (`opt_fea_verify`) | `optimize` |

## Tests and validation

```bash
python AI_CAE4ALL_main.py --config configs/SDFFlow/geometry_generation/ex1/baseline/config_train_sdfflow.txt --check
cd methods/SDFFlow && python -m pytest -q tests/
```

[`tests/`](../../methods/SDFFlow/tests/) covers conditional inference and tools,
the design loop, `fm_latent_draws`, screening, and surrogate reporting. The
`optimize` tests need `gmsh`, `pyamg` and `cma`.

## Known limitations

- **Two stages, one weld.** A weak VAE caps the FM. The FM depends on the exact
  VAE it was trained on, so changing the VAE means retraining the FM.
- **Marching Cubes can fail.** A latent with no zero crossing gives no mesh.
- **Condition control is approximate.** `cfg_scale` above 1 measurably worsened
  volume error, which is why the default is 1.0. Requests far outside the
  training envelope are guarded (`max_condition_z`), not guaranteed.
- **`optimize` gives relative answers only.** The tet4 FEA is stiff, so its
  numbers support relative comparison, not absolute verification. With
  `opt_analysis` set to the surrogate, results depend on the surrogate's accuracy.
- **Single-process inference.** Only the first GPU id is used.
