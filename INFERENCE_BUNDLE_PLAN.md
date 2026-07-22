# Stand-Alone Inference Bundle — Implementation Plan

Hand-off spec for building a self-contained inference folder that ships to
end-users **without the rest of the repository**. Written to be executable by
an implementer who does not have the author's investigation notes. Every fact
below was verified against the live code at the paths cited; re-verify anything
marked **[VERIFY]** before relying on it.

---

## 1. Goal & Constraints

Build one folder — call it `inference/` — that:

1. Runs inference for a trained model given **only** a `.pth` checkpoint plus a
   small runtime config (paths, device, timesteps). No training data, no
   stats-fitting, no repo checkout required at run time.
2. Is callable **two ways**:
   - **Stand-alone**: `python run_inference.py --checkpoint … --input … --output …`
   - **Library / from the current launcher**: `from cae_infer import infer; infer(...)`
3. Ships to the end-user **without** the training loop, data pipeline,
   augmentation, optimizer/scheduler, DDP/parallelism, config validation, or any
   research/secret machinery.
4. Is packaged as a **self-contained bundle** (PyInstaller frozen exe and/or
   Docker image) so the end-user needs no Python/dependency setup.
5. Supports **all** model families: `point_deeponet`, `deeponet`, `fno`, `gino`
   (Neural_Operator repo), `transolver`, `meshgraphnets`, `meshgraphnets-v`, and
   the geometry generator `sdfflow` (VAE + flow-matching, **one combined `.pth`** — see §5.5).
6. **Runtime target is CPU-only.** No CUDA in the bundle. Device is always
   `torch.device('cpu')`; `torch_cluster` is **dropped** (GINO uses its scipy
   `cKDTree` path); torch / torch_geometric use CPU wheels. This removes the
   biggest bundle-packaging risks and the GPU/CPU decision entirely.

### Secrecy decision (settled with the author)

Ship **readable but minimal model forward code**; exclude all training/research
code. The trained **weights are the IP**, plus the training recipe, data
pipeline, and hyperparameters — none of which ship. Fully *hiding* the
architecture via TorchScript/ONNX is **not pursued** because:

- GINO's neighbor construction uses a **scipy `cKDTree`** (numpy/C, outside the
  torch graph) — impossible to serialize into a traced graph.
- GINO/MGN run **per-graph Python loops over `ptr`**; SDFFlow runs an
  **iterative flow-matching sampler** — none trace/script cleanly.
- It would break "callable from `main.py`" and multiply per-model export work.

**The bundle provides the hiding for free**: a PyInstaller build ships only
compiled bytecode (no `.py` source in the artifact); a Docker image is not
casually browsed. Optionally add a bytecode-only (`.pyc`) build step as a mild
extra deterrent. Most of these architectures are published papers anyway.

---

## 2. Key Architectural Insight — checkpoints are self-describing

Every family's `torch.save` writes a dict that already contains **everything
needed to rebuild the exact network** — so reconstruction needs no dataset
object and no stats fitting. This is the entire reason the bundle is feasible.

| Family | Save site | Rebuild site | Rebuild call |
| --- | --- | --- | --- |
| Neural_Operator | `Neural_Operator/training_profiles/setup.py:250` (`save_checkpoint`) | `Neural_Operator/model/factory.py:60` (`build_model_from_checkpoint`) | `DataSpec.from_dict` + `CoordinateDomain.from_dict` + overlay `model_config` → registry core → `OperatorWrapper` → `load_state_dict(strict=True)` |
| Transolver | `Transolver/training_profiles/setup.py:213` | `Transolver/inference_profiles/rollout.py` (`_build_model_and_load_weights`) | overlay `model_config` → `Transolver(config, device)` → `load_state_dict` |
| MeshGraphNets | `MeshGraphNets/training_profiles/setup.py:192` | `MeshGraphNets/inference_profiles/rollout.py` | overlay `model_config` → `MeshGraphNets(config, device)` → `load_state_dict` |
| MGN-variational | `MeshGraphNets - variational/training_profiles/setup.py` | `.../inference_profiles/rollout.py` | same as MGN + VAE/prior modules |
| SDFFlow (geometry) | `Geometry_generation/training_profiles/train_fm.py:136` (`checkpoint_payload`) — **to be changed to embed the VAE, §5.5** | `Geometry_generation/inference_profiles/sample.py:90` (`run_sample`) | rebuild `VelocityNet(cfg, latent_flat_dim, cond_dim)` + `SDFVAE(vae_cfg)` from the **one** checkpoint's embedded configs → `load_state_dict` |

### Checkpoint key contracts (verified)

**Neural_Operator** (`schema_version = "deeponet_repo_v1"`):
```
schema_version, selected_model, epoch, model_state_dict,
ema_state_dict (optional), model_config, adapter_config, data_config,
normalization = {node_mean, node_std, delta_mean, delta_std,
                 position_scale, node_type_to_idx?, num_node_types?}
optimizer_state_dict / scheduler_state_dict / rng_states / … (IGNORE at inference)
```
Prefer `ema_state_dict` when present; its keys are prefixed `module.` and must be
stripped (`k[len('module.'):]`) before `load_state_dict`. `model_config` is
produced by each core's `export_model_config()` and is overlaid onto `config`
before constructing the core so `__init__`'s `config.get(...)` calls reproduce
the exact shapes.

**Transolver** `model_config` keys: `model, input_var, output_var,
positional_features, use_node_types, num_node_types, latent_dim, num_layers,
num_heads, slice_num, attention_kernel, mlp_ratio, dropout, temperature_init,
temperature_min, temperature_max, small_output_init, use_checkpointing,
num_timesteps`. Plus `data_config` (runtime-only) and `normalization`.

**MeshGraphNets** `model_config` keys: `input_var, output_var, edge_var,
latent_dim, message_passing_num, use_node_types, num_node_types,
positional_features, use_world_edges, use_checkpointing, use_multiscale,
multiscale_levels, mp_per_level, coarsening_type, voronoi_clusters`.
`normalization` additionally carries `world_edge_radius`, `coarse_edge_means`,
`coarse_edge_stds` when the relevant features are enabled.

> ⚠️ **Non-shape keys are NOT all in `model_config`.** Weights still load
> `strict=True`, but a few runtime keys affect *stochastic* behavior and must be
> pinned for reproducibility — see §9 landmines (`split_seed`,
> `point_resample_each_epoch`).

---

## 3. Deliverable Folder Structure

```
inference/                          # <-- the entire deliverable
  README.md                         # end-user run instructions
  PLAN-NOTES.md                     # (optional) how the bundle was assembled
  requirements.txt                  # pinned runtime deps
  Dockerfile                        # self-contained bundle (option A)
  pyinstaller.spec                  # self-contained bundle (option B)
  rebuild_bundle.py                 # repo-side assembler (does NOT ship) — see §10
  run_inference.py                  # stand-alone CLI entrypoint
  cae_infer/
    __init__.py                     # public API: infer(...), detect_family(...)
    cli.py                          # argparse -> infer(...)
    registry.py                     # checkpoint -> family driver dispatch
    common/
      __init__.py
      device.py                     # gpu/cpu selection (from rollout headers)
      hdf5_io.py                    # MGN HDF5 read/write contract (shared)
      config.py                     # minimal runtime-config dict builder
    families/
      neural_operator/              # models: point_deeponet, deeponet, fno, gino
        model/                      # vendored forward cone (see §5.1 manifest)
        general_modules/            # data_spec.py, positional_features.py, normalize.py
        driver.py                   # refactored run_rollout(explicit args)
      transolver/
        model/  general_modules/  driver.py
      meshgraphnets/
        model/  general_modules/  driver.py
      meshgraphnets_v/
        model/  general_modules/  driver.py
      geometry/                     # sdfflow: vae + velocity net + fm sampler
        model/  driver.py
```

### Import-isolation rule (critical)

Each family folder is a self-contained mini-package that keeps its **original
internal package roots** (`model.*`, `general_modules.*`) so vendored files need
**zero import rewriting** (this preserves byte-for-byte parity of the numeric
code). Because `model`/`general_modules` mean different things per family, the
dispatcher must run **one family per process**: prepend that family's folder to
`sys.path` immediately before importing its `driver`, and never import two
families in the same interpreter. Both entrypoints already satisfy this:

- stand-alone CLI: one checkpoint per invocation;
- launcher: `cae_suite/cli.py` dispatches one model per invocation.

Document this constraint in `README.md` and enforce it in `registry.py`
(raise if a second, different family is requested in the same process).

---

## 4. Foundation Components (write these by hand)

### 4.1 `cae_infer/__init__.py` — public API
```python
def detect_family(checkpoint_path) -> str:
    """torch.load(map_location='cpu', weights_only=False) the header and
    classify: 'neural_operator' if schema_version=='deeponet_repo_v1';
    'transolver' if model_config.get('model')=='transolver' or
    checkpoint_version present; 'meshgraphnets'/'meshgraphnets_v' by
    model_config keys (mp/vae presence); 'geometry' for the SDFFlow pair.
    Keep this dumb and explicit — the checkpoint carries the family."""

def infer(checkpoint, input, output, *, device='auto', timesteps=None,
          query_chunk_size=0, **family_opts) -> str:
    """Detect family -> prepend sys.path -> import families.<f>.driver ->
    call driver.run(...). Returns the output path."""
```

### 4.2 `cae_infer/cli.py` / `run_inference.py`
Argparse surface (superset; families ignore what they don't use):
```
--checkpoint PATH        single .pth (geometry too — VAE+FM combined, §5.5)
--input PATH             HDF5 mesh dataset (MGN contract); not used by geometry
--output PATH            output .h5 (rollout) or dir (geometry STLs)
--device cpu             CPU-only bundle; flag accepted but must be cpu
--timesteps N            rollout steps (default: full trajectory)
--query-chunk-size N     memory control for point/query decode (default 0)
--num-samples N          geometry sampler only
--ode-steps N            geometry FM ODE steps (default 50)
--cfg-scale F            geometry classifier-free guidance (default 1.0)
--mc-resolution N        geometry Marching Cubes grid (default 128)
--seed N                 geometry sampler / stochastic reproducibility
```

### 4.3 `cae_infer/registry.py`
Maps family → `(family_dir, driver_module_name)`; enforces one-family-per-process;
provides `load_driver(family)` that does the `sys.path` prepend + import.

### 4.4 `cae_infer/common/`
- `device.py`: **CPU-only** — always return `torch.device('cpu')`, ignore any
  `gpu_ids`/`--device` beyond validating they aren't a hard CUDA request. (The
  repo's `rollout.py` GPU block is not needed.) SDFFlow's `resolve_device` and
  its AMP path collapse to CPU (AMP auto-disables when `device.type != 'cuda'`).
- `hdf5_io.py`: the MGN HDF5 read (`data/{id}/nodal_data`, `data/{id}/mesh_edge`)
  and the atomic writer (`_write_rollout_output` in
  `Neural_Operator/inference_profiles/rollout.py:206`). Shared by NO/Transolver/MGN.
- `config.py`: build the tiny runtime-config dict each driver expects; the
  checkpoint overlay fills in all architecture keys.

---

## 5. Per-Family Build Specs

For each family: **(a) vendor** the listed files verbatim (exclude
`__pycache__`), **(b) apply the listed trims**, **(c) write `driver.py`** as a
refactor of that family's existing `rollout.py`/`sample.py` that takes explicit
args instead of reading a config file.

### 5.1 Neural_Operator (`point_deeponet`, `deeponet`, `fno`, `gino`)

**Vendor into `families/neural_operator/model/`** (verified import cone of the
forward + reconstruction path):
```
base.py  mlp.py  utils.py  operator_wrapper.py  factory.py
deeponet.py  point_deeponet.py  pointnet.py  siren.py
fno.py  spectral.py  gino.py  gno.py  gino_carcfd.py
adapters/coordinate_domain.py  adapters/grid.py
adapters/point_sampling.py  adapters/radius_neighbors.py
__init__.py  adapters/__init__.py
```
**Vendor into `families/neural_operator/general_modules/`**:
```
data_spec.py            # verbatim; from_dict is used, build_* stays dormant
positional_features.py  # verbatim (numpy + scipy.sparse)
normalize.py            # NEW: extract only normalize_positions +
                        # normalize_node_features from mesh_dataset.py
                        # (mesh_dataset.py imports the whole dataset/stats
                        # machinery — do NOT vendor it)
```

**Trims:**
- `model/factory.py`: delete `build_model(...)` and the top import
  `from general_modules.config_validation import validate_model_config` and
  `from general_modules.data_spec import build_data_spec_from_dataset`. Keep
  `build_model_from_checkpoint`, `register_model`, `_resolve_core_class`, and the
  registration calls at the bottom. (`build_model_from_checkpoint` never calls
  the validator.)
- Define `SCHEMA_VERSION = "deeponet_repo_v1"` in the driver (currently imported
  from `training_profiles/setup.py:21`; do not vendor `setup.py`).

**`driver.py`** — refactor of `Neural_Operator/inference_profiles/rollout.py`
(`run_rollout`) verified end-to-end. Key logic to preserve exactly:
- `torch.load(map_location='cpu', weights_only=False)`, validate
  `schema_version`, enforce `config['model'] == checkpoint['selected_model']`.
- `build_model_from_checkpoint(config, checkpoint)` → `model.to(device).eval()`.
- Pull `normalization`: `node_mean/std`, `delta_mean/std`, `position_scale`,
  `node_type_to_idx`, `num_node_types`.
- Per sample: read `nodal_data`,`mesh_edge`; `ref_pos = nodal_data[:3,0,:].T`;
  build bidirectional `edge_index`; `part_ids = nodal_data[-1,0,:]` when
  `use_node_types`; `compute_positional_features` when `positional_dim>0`;
  `normalize_positions`; build a bare `torch_geometric.data.Data` with
  `x, pos, pos_normalized, edge_index, batch, ptr, sample_id`.
- Static (`T==1`): input is zeros, prediction **is** the field. Temporal
  (`T>1`): input is `state[0]`, prediction is a **delta** added to state.
  Denormalize: `pred = pred_norm * delta_std + delta_mean`.
- Optional chunked decode via `encode_operator` + `decode_in_chunks`
  (`inference_profiles/query_decode.py` — vendor it) when
  `query_chunk_size > 0`.
- Write output via the shared `common/hdf5_io.py` writer.

**Runtime config keys the driver needs** (everything else comes from the
checkpoint overlay): `model` (set from `selected_model`), `modelpath`,
`infer_dataset`, `infer_timesteps`, `inference_output_dir`,
`infer_query_chunk_size`, `gpu_ids`.

### 5.2 Transolver

**Vendor `families/transolver/model/`**: `Transolver.py`, `blocks.py`,
`physics_attention.py`, `checkpointing.py`, `__init__.py`. Trace and include any
`general_modules` the forward path imports (positional features, coordinate
normalization, node-type handling) — **[VERIFY import cone]** by grepping
`^from` in `Transolver/model/*.py` and `Transolver/inference_profiles/rollout.py`.
**Reconstruction**: `Transolver(config, str(device))` after overlaying
`model_config`; load `ema_state_dict` (strip `module.`) else `model_state_dict`
(their loader is non-strict — keep it non-strict to match). **Driver**: refactor
`Transolver/inference_profiles/rollout.py`; note it also has `decoupled.py`
(query-decoupled path) — include if the deliverable needs it.

### 5.3 MeshGraphNets

**Vendor `families/meshgraphnets/model/`**: `MeshGraphNets.py`, `blocks.py`,
`encoder_decoder.py`, `mlp.py`, `coarsening.py`, `checkpointing.py`.
`coarsening.py` (`MultiscaleData`) is only needed when `use_multiscale=True` —
include it; it's imported by the rollout. **Reconstruction**:
`MeshGraphNets(config, str(device))` after overlaying `model_config`; load EMA
(strip `module.`) else `model_state_dict`. **Driver**: refactor
`MeshGraphNets/inference_profiles/rollout.py`. MGN consumes **edge attributes**
and (optionally) world edges + multiscale hierarchy — the driver's graph builder
must reproduce these (unlike the Neural_Operator models, which drop edge attrs).
**[VERIFY]** the exact edge-feature + world-edge + coarsening construction in the
MGN rollout and its `general_modules` cone.

### 5.4 MeshGraphNets — variational

Same as §5.3 **plus** `model/vae.py` and `model/conditional_prior.py`. The
variational rollout **resamples the latent per rollout step**; loss composition
is unchanged but inference draws a latent each step (single `time_integration`
flag; AR-RT recipe). Preserve the per-step resample and any seeding so results
are reproducible. **[VERIFY]** the variational rollout entrypoint and whether a
sampling seed is exposed.

### 5.5 Geometry — SDFFlow (generative; **one combined `.pth`**)

Not a mesh-in/mesh-out rollout — it **generates** geometry via
noise → FM ODE → latent → SDF-VAE decoder → Marching Cubes → STL. It uses a
**SDF-VAE** and a **flow-matching velocity net**. Per the author's decision,
**both are stored in a single `.pth`** so the end-user ships one file.

**Current contract (verified).** Two separate checkpoints today:
- VAE (`train_vae.py:81`): `{stage:'vae', epoch, model_state, ema_state,
  config, cond_mean, cond_std, cond_names}`; rebuilt via `SDFVAE(ckpt['config'])`.
- FM (`train_fm.py:136`): `{stage:'fm', epoch, model_state, ema_state, config,
  vae_modelpath, latent_flat_dim, latent_mean, latent_std, cond_dim, cond_mean,
  cond_std, cond_names, condition_* clip/extrema}`; rebuilt via
  `VelocityNet(ckpt['config'], latent_flat_dim, cond_dim)`.
Both recover their architecture from their own embedded `config` dict — so no
dims are missing; the only change needed is to **co-locate** them.

**Training-side change (required, do this first).** In
`Geometry_generation/training_profiles/train_fm.py`, `checkpoint_payload(epoch)`
already runs after the VAE has been loaded for FM training
(`vae_ckpt = load_checkpoint(vae_path, device)` at `train_fm.py:32`). Embed the
VAE into the FM payload so the FM `.pth` is fully self-contained:
```python
'vae': {
    'model_state': _cpu(vae_ckpt['model_state']),
    'ema_state':   _cpu(vae_ckpt.get('ema_state')),
    'config':      vae_ckpt['config'],
    'cond_mean':   vae_ckpt.get('cond_mean'),
    'cond_std':    vae_ckpt.get('cond_std'),
    'cond_names':  vae_ckpt.get('cond_names'),
},
'schema_version': 'sdfflow_infer_v1',
```
(`_cpu` = move tensors to CPU before saving, since the target is CPU-only.)
Keep the existing top-level FM keys unchanged for backward compatibility. The
result is one canonical inference artifact (e.g. `sdfflow.pth`).

**Existing checkpoints** (pre-change, two files) → provide a one-shot merge
helper (repo-side, does not ship): read `sdfflow_fm.pth` + `sdfflow_vae.pth`,
write the combined dict above. Put this in `rebuild_bundle.py` (§10) or a small
`merge_sdfflow_checkpoint.py`.

**Vendor `families/geometry/model/`**: `sdf_vae.py`, `velocity_net.py`,
`mlp.py`, `__init__.py`. Also vendor the forward-path pieces of
`general_modules/mesh_extraction.py` (`decode_sdf_grid`, `sdf_grid_to_mesh`,
`mesh_report`) — Marching Cubes is pure CPU/numpy. **[VERIFY]** the
`mesh_extraction` import cone (it may pull `skimage`/`trimesh`; add to
`requirements.txt`).

**Driver** — refactor `run_sample` (and optionally `run_reconstruct`,
`interpolate.run_interpolate`) to explicit args. Load the **single** checkpoint:
`VelocityNet` from the top-level keys, `SDFVAE` from `ckpt['vae']['config']`
(fall back to `ckpt['vae_modelpath']` only if the `'vae'` key is absent, for old
files). CLI: `--checkpoint`, `--num-samples`, `--ode-steps`, `--cfg-scale`,
`--mc-resolution`, `--seed`, `--output`. The FM **iterative ODE sampler loop**
stays in Python (this is why export-to-graph is off the table). Preserve the
condition audit / OOD guard / candidate ranking exactly (see the repo's
`Geometry_generation/CLAUDE.md` invariants) if the deliverable needs conditioned
sampling; drop them only if the end-user use is unconditional generation.

---

## 6. Self-Contained Bundle Packaging — **CPU-only**

### Dependencies (`requirements.txt`, pin exact versions from the training env)
```
torch            (CPU wheel: --index-url https://download.pytorch.org/whl/cpu)
torch_geometric  (CPU)         # all graph families
scipy                          # cKDTree (gino neighbors) + sparse RWPE (pos. features)
h5py, numpy
scikit-image, trimesh          # SDFFlow Marching Cubes / mesh export  [VERIFY exact set]
```
- **No CUDA, no `torch_cluster`.** GINO runs its scipy `cKDTree` path (the
  `torch_cluster` accelerator is CPU-irrelevant here and is a packaging headache
  — leave it out). Device is always CPU (§4.4).
- Prefer a `torch_geometric` version that does **not** hard-require the compiled
  companions `torch_sparse`/`torch_scatter` (recent PyG makes them optional).
  This keeps the CPU bundle to pure-wheel deps and greatly simplifies PyInstaller.

### Option A — Docker (recommended default)
- Base: a slim Python image (or `pytorch/pytorch:<ver>-cpu`).
- `pip install -r requirements.txt` with the CPU index URL above.
- `COPY inference/ /app`; `ENTRYPOINT ["python", "/app/run_inference.py"]`.
- Source is inside the image, not casually visible — satisfies the secrecy goal.

### Option B — PyInstaller frozen exe
- `pyinstaller.spec` with `run_inference.py` as entry; **ships only bytecode**
  (best source hiding).
- **Hidden-import / data-collection caveats** (these WILL break a naive build):
  `torch_geometric` uses lazy/dynamic imports; `torch` ships compiled extensions
  and data files. Use PyInstaller hooks (`collect_all`, `collect_submodules`) for
  `torch`, `torch_geometric`, `scipy`, `skimage`, `trimesh`. CPU-only removes all
  CUDA/`torch_cluster` collection — the main remaining simplification win.
- Result is large (hundreds of MB) — acceptable for "zero-config for the user".

Provide both if feasible; Docker is lower-risk, PyInstaller hides source best.

---

## 7. Integration with the Current Launcher

"Callable from the current main script" = the repo can `import cae_infer` and
delegate instead of running its own `rollout.py`.

- Simplest: in each native `main.py`'s `mode == 'inference'` branch (e.g.
  `Neural_Operator/main.py:64`, and the Transolver/MGN mains), optionally route
  to `cae_infer.infer(...)` behind a flag/env so behavior is opt-in and the
  existing path stays as the reference for the parity test.
- The top launcher `cae_suite/cli.py` already selects one backend per
  invocation, so the one-family-per-process rule in §3 holds unchanged.
- Keep the bundle importable with `inference/` on `sys.path`; do **not** make the
  bundle depend on anything outside `inference/`.

---

## 8. Parity Test Strategy (the acceptance gate)

For each family, prove the bundle reproduces the repo's output **bit-for-bit**
(or within fp tolerance if any nondeterminism is unavoidable):

1. Take (or train a tiny) checkpoint using the existing repo.
2. Run the repo's native `mode inference` on a fixture HDF5 → `ref.h5`.
3. Run `python inference/run_inference.py --checkpoint … --input … --output out.h5`.
4. Assert `nodal_data` arrays are equal within `atol=1e-6` (ideally exactly
   equal — same code path, same weights).
- Neural_Operator and MGN ship tiny synthetic HDF5 fixtures under `tests/`
  (`tests/conftest.py`) that run in well under a minute — reuse them.
- Add the parity script to CI so bundle drift is caught.

---

## 9. Known Landmines (do not skip)

1. **GINO neighbor cap** (`model/adapters/radius_neighbors.py`): `max_num_neighbors`
   truncation + `nq×cap` preallocation. Inference must reproduce neighbor
   construction **exactly**; a mismatch changes outputs silently. On the CPU-only
   bundle only the **scipy `cKDTree`** path runs (no `torch_cluster`), so parity
   must be checked against a training run that also used the scipy path — if the
   model was **trained** with `torch_cluster` and its results differ from the
   scipy path, resolve that discrepancy before shipping.
2. **Non-shape stochastic keys not in `model_config`**: for `point_deeponet`,
   `split_seed` (→ sampler base seed) and `point_resample_each_epoch` are **not**
   exported. Weights load fine, but sensor sampling at inference depends on them.
   Pin them (add to the runtime config or, better, add to `export_model_config`
   on the training side) so results are reproducible.
3. **EMA key prefix**: EMA state dicts are prefixed `module.`; strip it before
   loading (all families).
4. **Grid axis convention** (FNO/GINO, `model/adapters/grid.py`): tensor dim 2 =
   active axis x, dim 3 = y, dim 4 = z; `F.grid_sample` reverses this. The
   vendored `grid.py` handles it — do not "simplify" it. Verbatim vendoring
   avoids the trap.
5. **Static vs temporal target semantics**: `T==1` → prediction *is* the field
   (input zeros); `T>1` → prediction is a delta added to state. Denormalize with
   `delta_mean/std`. Getting this backwards silently corrupts output.
6. **MGN edge attributes / world edges / multiscale**: unlike the operator
   models, MGN consumes edge features and may need world-edge radius and a
   coarsening hierarchy at inference; the graph builder must reconstruct them
   from `normalization` (which stores `world_edge_radius`, `coarse_edge_*`).
7. **SIREN init**: `point_deeponet`'s SIREN trunk must not receive the generic
   Kaiming init — handled inside the vendored model, don't touch.
8. **`weights_only=False`** is required on `torch.load` (checkpoints contain
   Python objects: numpy arrays, config dicts). Note this for newer torch that
   defaults to `weights_only=True`.

---

## 10. Reproducible Assembly (keep the bundle in sync)

Write `rebuild_bundle.py` (stays in the repo, **does not ship**) that:
- copies the per-family file manifests from §5 into `inference/…` (excluding
  `__pycache__`),
- applies the mechanical trims (factory edit, `normalize.py` extraction,
  `SCHEMA_VERSION` injection),
- leaves the hand-written glue (`cae_infer/*`, drivers, configs) untouched.

This makes "ship inference without the rest of the code" a one-command export
and prevents the vendored copies from drifting from source.

---

## 11. Phasing / Milestones

1. **Foundation**: folder skeleton, `cae_infer` API/CLI/registry/common,
   `requirements.txt`, `README.md`. (No models yet.)
2. **Neural_Operator family** end-to-end + parity test (4 models, one driver).
   This is the reference pattern; smallest, best-documented cone.
3. **Docker bundle** proving the Neural_Operator slice runs zero-config.
4. **Transolver**, then **MeshGraphNets**, then **MGN-variational** — each
   vendored + driver + parity test, reusing `common/`.
5. **Geometry (SDFFlow)** — generative driver. **First** land the training-side
   change so FM checkpoints embed the VAE (§5.5), and add the merge helper for
   existing two-file checkpoints; then build the single-`.pth` driver.
6. **PyInstaller bundle** (source-hiding) once families are stable.
7. **Launcher integration** (§7) + `rebuild_bundle.py` (§10) + CI parity.

Recommended first vertical slice: **Neural_Operator `fno`** (no per-graph loop,
easiest parity) or **`point_deeponet`** (author's primary model).

---

## 12. Open Items to VERIFY before/while coding

- [ ] Transolver forward-path `general_modules` import cone (grep `^from` in
      `Transolver/model/*.py` + its `rollout.py`).
- [ ] MGN rollout graph builder: exact edge-feature, world-edge, and coarsening
      construction + its `general_modules` cone.
- [ ] MGN-variational rollout entrypoint + per-step latent resample seeding.
- [ ] SDFFlow `mesh_extraction` import cone + exact extra deps (`scikit-image`,
      `trimesh`?) for `requirements.txt`.
- [ ] Whether `decoupled.py` (Transolver) and `query_decode.py` chunked paths are
      required in the deliverable or can be dropped for simplicity.

**Resolved by the author:** SDFFlow ships as **one combined `.pth`** (§5.5,
`train_fm.py` embeds the VAE); target is **CPU-only** — no CUDA, no
`torch_cluster` (§1 item 6, §6).

---

### Source references (repo paths, for the implementer)
- Neural_Operator: `model/factory.py:60`, `inference_profiles/rollout.py`,
  `general_modules/{data_spec,positional_features,mesh_dataset}.py`,
  `training_profiles/setup.py:230` (save contract), `main.py:64` (dispatch).
- Transolver: `training_profiles/setup.py:156` (`build_model_config`),
  `inference_profiles/rollout.py`, `model/`.
- MeshGraphNets: `training_profiles/setup.py:158`, `inference_profiles/rollout.py`,
  `model/`.
- MGN-variational: `MeshGraphNets - variational/{model,inference_profiles}/`.
- Geometry: `Geometry_generation/inference_profiles/{sample,interpolate}.py`,
  `model/{sdf_vae,velocity_net,mlp}.py`, `SDFFlow_main.py`.
- Launcher: `cae_suite/cli.py`, `cae_suite/specs/*.py`.
```
