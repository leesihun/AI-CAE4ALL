# AI-CAE4ALL Studio implementation plan

## 1. Objective

Provide one block-pipeline Studio from which the installed AI-CAE4ALL routes,
datasets, configurations, jobs, artifacts, evaluation tools, optimization,
documentation, and portable deployment can be inspected and executed.

The Studio must display repository evidence rather than invented sample values.
Anything not implemented must remain visibly marked `adapter` or `roadmap`.

## 2. Scope boundary

The primary implementation boundary is `frontend/`.

- Browser, local server, launchers, runtime state, reports, and exports live
  under `frontend/`.
- Existing training algorithms and model implementations are invoked rather
  than redesigned. Narrow integration fixes are allowed where the Studio
  exposes their public contracts: cHI-MGNflow checkpoints now record their
  model family, and the portable inference classifier explicitly refuses
  unsupported cHI-MGNflow and SimulGen-VAE checkpoints instead of routing them
  to a similar-looking driver.
- Generated Studio files are isolated under ignored `frontend/runtime/`.
- A remote, multi-user deployment would require authentication, isolation,
  quotas, durable storage, and a separate production-service scope.

## 3. Runtime architecture

```text
Browser index.html
  ├─ typed movable/linkable graph
  ├─ configuration editor
  ├─ HDF5/sample viewers
  └─ Studio workspaces
          │ JSON over localhost
          ▼
frontend/studio_server.py
  ├─ imports live MethodRegistry and preflight
  ├─ invokes AI_CAE4ALL_main.py
  ├─ invokes inference/run_inference.py
  ├─ reads repository HDF5/CSV/docs/artifacts
  └─ writes only frontend/runtime/
          │ existing native contracts
          ▼
AI-CAE4ALL model repositories and datasets
```

`START_STUDIO.bat` starts `start_studio.py`, which binds the real API server
before opening `http://127.0.0.1:8080/index.html`.

## 4. Pipeline execution contract

1. The browser validates required typed links and rejects cycles.
2. Nodes are ordered from graph dependencies, not screen position.
3. Automatic layout groups source blocks by dependency level, centers shorter
   columns against the tallest column, and keeps primary sequential links
   visually stronger than long-range data/parameter buses.
4. Linked HDF5, condition, and checkpoint blocks override the corresponding
   native flat-config fields.
5. A source block accepts a repository file selected in the Studio or a local
   file streamed into the isolated `frontend/runtime/uploads/` area.
6. Every native step is submitted to the suite preflight.
7. The first step receives full filesystem, environment, dataset, checkpoint,
   and native checks. Downstream filesystem/native checks are deferred until
   their upstream artifacts exist.
8. Each step runs through the existing suite launcher.
9. stdout, stderr, PID, exit code, status, timestamps, target node ID, and each
   executable step's exact node ID/type are persisted.
10. Cancellation terminates the launched process tree.
11. The browser persists a versioned graph document containing the pipeline
    name, viewport, exact block configs, artifact/report lineage, and typed
    edges. Import validates block/port types, single-input cardinality, IDs,
    and acyclicity before replacing current state.

Native graph steps:

- Geometry → HDF5: `geometry_ingest` `inspect` or `ingest`
- All eleven current trainable model IDs and every registered mode; the live
  registry also creates a generic GUI block for a future trainable route
- Inference Run via the linked model’s native inference/reconstruction mode
- CAD Generator via SDFFlow `sample`

Decision adapters do not pretend to be model routes:

- Evaluation reads real HDF5 arrays.
- Comparison reads a real CSV.
- Optimization reads a real candidate/evaluation CSV.
- Export copies or archives a real selected artifact.

## 5. Configuration workspace

The live `MethodSpec` is authoritative.

- All accepted keys are displayed.
- Mode-specific required keys are separated.
- Closed, obvious choices use dropdowns.
- Paths, width lists, feature lists, and open-ended values remain editable.
- Checked-in examples can be loaded.
- Presets are limited to meaningful route variants and smoke/low-VRAM cases.
- Raw flat `.txt` input and structured controls stay synchronized.
- Configurations are preflighted before execution and can be saved beneath
  `frontend/runtime/configs/`.

SimulGen-VAE receives four distinct mode contracts:

- `train`: VAE then latent conditioner
- `train_vae`: VAE only
- `train_lc`: conditioner from a compatible VAE checkpoint
- `reconstruct`: VAE + conditioner to HDF5 fields

Studio-specific SimulGen checks additionally verify:

- the condition CSV/image source exists;
- all selected samples have fixed node and timestep counts;
- field ranges fit the rank-3 `nodal_data` arrays;
- CSV row count matches the HDF5 sample count.

## 6. Workspaces

### Models

Read the live registry, modes, known keys, required fields, route health,
repository, entrypoint, dataset kind, native probe, and checked configs.

### Data

Resolve the artifact configured on the selected block, catalog its real
samples, and normalize CAD/mesh files plus mesh-state, SDF-shape,
operator-grid, and table HDF5 contracts through one preview API. The viewer
provides separate Points, Mesh, and Field modes; mesh modes consume real
triangle faces or `mesh_edge` connectivity rather than inferring or inventing
topology.

### Experiments

List persisted jobs, live status, route steps, logs, return codes, and
cancellation state. Parse actual epoch/step metrics, show all series by
default, allow per-series exclusion and visual smoothing, preserve raw CSV
download, and reach metrics directly from a model or job row.

### Evaluation

Inspect both HDF5 contracts before scoring. Match declared sample IDs, enforce
shape/node compatibility, map named output fields exactly, require explicit
confirmation when only positional alignment is available, support embedded
native truth arrays, and keep contiguous mesh-row selection as a deliberate
legacy override. Then compare compatible values and write:

- relative L2;
- MAE;
- RMSE;
- maximum absolute error;
- R²;
- mean, median, p95, min, and max aggregates;
- a per-sample CSV and JSON report.

### Cross-model comparison

Resolve multiple metrics inputs from the open Compare Models block to exact
persisted runs, overlay common metric histories, and rank last raw values.
Training losses without a shared key are not silently treated as comparable.
For qualified cross-family accuracy, read one or more real
comparison/evaluation CSVs, inspect their common schema, select an actual
numeric metric and suggested group column, choose min/max direction, rank
numeric rows, retain source rows, and save JSON. The browser never assumes a
fixed `model` or `mean_relative_l2` field.

### Optimization

Read numeric candidate rows, apply hard inequality constraints, calculate the
feasible non-dominated set, compute crowding distance, select a diverse top-k,
and persist the evidence report. Before execution, inspect the actual CSV
schema, exclude identifier/metadata columns from objective suggestions, and
require explicit objective and direction selection.

The current native engine evaluates a fixed candidate table. Iterative DOE,
NSGA-II, constrained Bayesian optimization, solver verification, and active
learning remain roadmap work.

### Deployment

Run the family-detecting CPU inference bundle for its eight supported model
types through five driver families, track it as a job, and build the Windows
PyInstaller bundle under `frontend/runtime/deploy`. Checkpoint metadata gates
the action before launch. cHI-MGNflow, MLP, and SimulGen-VAE checkpoints are
sent to their native inference/reconstruction paths rather than misclassified
as portable models.

The local `POST /api/inference/run` route is functional. Production remote API
hosting remains outside this local Studio scope.

### Export

Copy a selected file or ZIP a selected directory under
`frontend/runtime/exports`, then expose a browser download link. Source
artifacts remain unchanged.

### Docs and System

Read actual Markdown files, show interpreter and route health, and query the
visible NVIDIA inventory through `nvidia-smi`.

## 7. Verification record

The handoff is based on checks actually run against the live checkout:

1. All frontend JavaScript sources and runners pass `node --check`; 26 backend
   tests pass.
2. `/api/health` reports 12/12 healthy routes, including all 11 trainable model
   routes represented by GUI blocks.
3. The live System audit checks 330 current configs and accurately exposes one
   unrelated failure in the untracked Geometry optimization draft
   (`mode optimize` is not a registered SDFFlow mode); the strict cHI-MGNflow
   ex9 config check still passes.
4. Fourteen Chrome runners exercise graph editing, repository file selection,
    configuration/autofill, history, accessibility, responsive layout, HDF5 and
    geometry viewing, evaluation, comparison, optimization, export, deployment,
    System audit, stale-response races, and process-state behavior. The dedicated
    control-surface runner also clicks every previously uncovered action family;
    the only ID-bearing controls not named directly are the three hidden file
    inputs exercised through their visible Import/Load/Upload buttons.
5. The checkpoint-only cHI-MGNflow runner clicks the real repository picker,
   connects the real checkpoint and held-out HDF5, sets CPU/one-step/mean
   controls in the Inspector, accepts the launch confirmation, passes the exact
   saved-config launch gate, and completes native inference with one real HDF5
   result under `frontend/runtime/`.
6. The focused cHI-MGNflow flow smoke test and history-safety VM test pass.
7. Scoped whitespace validation passes for the Studio, inference bridge, and
   cHI checkpoint-metadata changes; unrelated user worktree changes remain
   untouched.

## 8. Remaining roadmap

These are intentionally not represented as completed:

- production API authentication, OpenAPI governance, quotas, and rollback;
- a persistent sweep scheduler rather than checked child configurations;
- CAD/BRep rendering for every native format in the browser;
- iterative optimization and solver-in-the-loop verification;
- cross-run GPU peak instrumentation when a method does not publish it;
- remote artifact storage and multi-user collaboration.

Implementing the first two items would require a separate production and
security design rather than another local-UI patch.
