# AI-CAE4ALL Studio implementation plan

## 1. Objective

Provide one block-pipeline Studio from which the installed AI-CAE4ALL routes,
datasets, configurations, jobs, artifacts, evaluation tools, optimization,
documentation, and portable deployment can be inspected and executed.

The Studio must display repository evidence rather than invented sample values.
Anything not implemented must remain visibly marked `adapter` or `roadmap`.

## 2. Scope boundary

The user-authorized change boundary is `frontend/`.

- Browser, local server, launchers, runtime state, reports, and exports live
  under `frontend/`.
- Existing suite and method code is imported or invoked, not rewritten.
- Generated files are isolated under ignored `frontend/runtime/`.
- A remote, multi-user deployment would require authentication, isolation,
  quotas, durable storage, and backend changes outside this boundary.

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
9. stdout, stderr, PID, exit code, status, and timestamps are persisted.
10. Cancellation terminates the launched process tree.

Native graph steps:

- Geometry → HDF5: `geometry_ingest` `inspect` or `ingest`
- All ten trainable model IDs and every registered mode
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
cancellation state.

### Evaluation

Match prediction and truth sample IDs, enforce node compatibility, select field
rows, compare overlapping timesteps, and write:

- relative L2;
- MAE;
- RMSE;
- maximum absolute error;
- R²;
- mean, median, p95, min, and max aggregates;
- a per-sample CSV and JSON report.

### Cross-model comparison

Read a real comparison/evaluation CSV, choose a group column, metric column,
and min/max direction, rank numeric rows, retain source rows, and save JSON.

### Optimization

Read numeric candidate rows, apply hard inequality constraints, calculate the
feasible non-dominated set, compute crowding distance, select a diverse top-k,
and persist the evidence report.

The current native engine evaluates a fixed candidate table. Iterative DOE,
NSGA-II, constrained Bayesian optimization, solver verification, and active
learning remain roadmap work.

### Deployment

Run the existing family-detecting CPU inference bundle, track it as a job, and
build the Windows PyInstaller bundle under `frontend/runtime/deploy`.

The local `POST /api/inference/run` route is functional. Production remote API
hosting remains outside the frontend-only boundary.

### Export

Copy a selected file or ZIP a selected directory under
`frontend/runtime/exports`, then expose a browser download link. Source
artifacts remain unchanged.

### Docs and System

Read actual Markdown files, show interpreter and route health, and query the
visible NVIDIA inventory through `nvidia-smi`.

## 7. Verification gates

Before handoff:

1. Parse both Python entrypoints with `ast`.
2. Run `node --check` for browser and smoke-test JavaScript.
3. Confirm `/api/health` and all 11 registered routes.
4. Preflight the generated fixed-geometry SimulGen fixture.
5. Execute a one-epoch SimulGen VAE smoke run.
6. Execute latent-conditioner training and reconstruction.
7. Execute Geometry → HDF5 inspection from the graph.
8. Execute portable checkpoint inference.
9. Evaluate real prediction/truth HDF5 arrays.
10. Rank a real comparison CSV.
11. Run real Pareto selection.
12. Export a real generated report.
13. Drive the browser through config, HDF5, geometry-ingest, evaluation,
    comparison, optimization, and export workspaces.
14. Verify mouse-wheel zoom, alphabetical model ordering, input-first and
    output-first block linking, and source-file selection/upload.
15. Assert that Mesh and Field modes render real `mesh_edge` lines and that
    Points mode renders nodes without topology.
16. Audit Git status and confirm implementation changes stay in `frontend/`.

## 8. Remaining roadmap

These are intentionally not represented as completed:

- production API authentication, OpenAPI governance, quotas, and rollback;
- a persistent sweep scheduler rather than checked child configurations;
- CAD/BRep rendering for every native format in the browser;
- iterative optimization and solver-in-the-loop verification;
- cross-run GPU peak instrumentation when a method does not publish it;
- remote artifact storage and multi-user collaboration.

Implementing the first two items would require widening the authorized change
scope beyond `frontend/`.
