# AI-CAE4ALL Studio Frontend

This folder contains the local AI-CAE4ALL block-pipeline Studio. Its Python
server connects the browser to the existing suite registry, preflight system,
native launcher, repository files, HDF5 datasets, jobs, logs, documentation,
and output artifacts without modifying backend source files.

## Open the studio

On Windows, double-click:

```text
START_STUDIO.bat
```

The launcher opens the Studio in the default browser at
`http://127.0.0.1:8080/index.html` and keeps a local server running in its
console window. It uses `start_studio.py` to open the browser only after the
local server is ready.
Close the window or press `Ctrl+C` to stop it.

If port 8080 is occupied, launch it from a terminal with another port:

```powershell
frontend\START_STUDIO.bat 8081
```

Alternatively, start the same API server manually:

```powershell
python frontend\start_studio.py 8080
```

Do not use `python -m http.server`: it can display the HTML, but it cannot
provide model execution, preflight, repository browsing, or artifact APIs.

If a terminal still prints `Serving HTTP on 127.0.0.1 port 8080`, that is the
wrong static server. Stop it with `Ctrl+C`, then run `START_STUDIO.bat`.
The correct console starts with `AI-CAE4ALL Studio is ready` and the top-right
badge in the browser reads `11/11 routes live`.

Useful review URLs:

- `index.html` — the linked SimulGen-VAE field-reconstruction pipeline
- `index.html?review=config` — the large SimulGen-VAE configuration workspace
- `index.html?review=optimization` — conditional CAD generation and optimization

## Included

- Draggable, linkable, typed pipeline blocks
- Click-in-either-order and drag-to-link ports, compatible-port highlighting,
  larger connection targets, mouse-wheel cursor-centered zoom, and graph fit
- Pipeline-first editor layout with a collapsible inspector, dependency-level
  auto layout, background drag-to-pan, typed connection colors, and subdued
  long-range data/parameter buses
- Socket rows are part of each block instead of floating over the preview, so
  every wire terminates at the exact visual socket center
- Explicit CAD, HDF5, parameter, and saved-model inputs with repository
  browsing or local-file upload directly from each source block
- All eleven live AI-CAE4ALL routes, including the geometry-ingest tool
- SimulGen-VAE as a first-class block with `train`, `train_vae`, `train_lc`,
  and `reconstruct` modes
- All 67 live SimulGen-VAE configuration keys, mode-specific required fields,
  obvious presets, manual values, and synchronized flat `.txt` input/output
- Authoritative suite preflight with real diagnostic codes, paths, environment,
  dataset, checkpoint, and native-probe results
- Actual `AI_CAE4ALL_main.py` execution with sequential pipeline steps, captured
  logs, status polling, exit codes, and process-tree cancellation
- Dependency-ordered graph execution, including the native
  `geometry_ingest` route for the Geometry → HDF5 block
- A repository-wide **Config audit** in the System workspace: runs the same
  parse/spec/route checks as `AI_CAE4ALL_main.py --audit-configs` over every
  checked-in `configs/**/config*.txt` and lists PASS/FAIL with per-file
  diagnostics
- An **Explain config** action in the configuration modal: the same
  configured/required/recommended/inactive/checkpoint-owned/unknown-key
  breakdown as `--explain-config`, without leaving the browser
- Repository-backed Models, Data, Experiments, Benchmarks, Artifacts, System,
  and Docs workspaces
- One configured-artifact viewer for CAD/mesh files and every supported HDF5
  contract. It renders real triangle faces, `mesh_edge` topology, SDFFlow
  `shapes/{id}` surface points, operator grids, table rows, scalar channels,
  and timesteps without substituting another repository file. Open Samples
  starts with no default sample, can switch to another repository dataset or
  upload a local HDF5/CAD/mesh/VTK file in place, and shares left-drag orbit,
  right-drag pan, wheel zoom, keyboard camera controls, and view reset across
  every visualization contract.
- Real HDF5 field evaluation (relative L2, MAE, RMSE, maximum error, and R²),
  real CSV cross-model ranking, Pareto/crowding optimization, and downloadable
  file or ZIP exports
- Portable CPU inference through `POST /api/inference/run`, persistent job
  logs, cancellation, and the existing or newly built Windows executable
- Visible maturity labels (`native`, `adapter`, or `roadmap`) so future
  integration work is not presented as already implemented

## Local API

The browser uses the following real local endpoints:

- `GET /api/health`, `/api/models`, `/api/configs`, `/api/docs`, `/api/files`
- `GET /api/hdf5`, `/api/hdf5/samples`, `/api/hdf5/sample`
- `GET /api/preview/samples`, `/api/preview/sample` — the shared CAD, mesh,
  point-cloud, series, and HDF5 visualization surface
- `GET /api/audit-configs` — structural audit of every checked-in config
- `POST /api/preflight`, `/api/config/explain`, `/api/config/save`, `/api/pipeline/run`
- `POST /api/upload?kind=dataset|geometry|checkpoint`
- `GET /api/jobs`, `/api/jobs/{id}` and `POST /api/jobs/{id}/cancel`
- `POST /api/inference/run`, `/api/build/exe`
- `POST /api/evaluation/run`, `/api/comparison/run`, `/api/optimization/run`
- `POST /api/export`, `/api/simulgen/smoke-fixture`

This is a localhost development API, not a multi-user production deployment.
Authentication, remote hosting, request isolation, quotas, and rollback would
require an explicitly authorized backend change outside `frontend/`.

## Integration boundary

The local bridge is implemented by `studio_server.py`. Runtime configurations,
job metadata, and logs are written only under the ignored `frontend/runtime/`
directory. Existing method repositories and suite Python modules are imported
or launched but are not rewritten by the Studio.

The implementation and verification roadmap is recorded in
`IMPLEMENTATION_PLAN.md`.

The PNG files in this folder are reproducible visual-review captures of the
SimulGen pipeline, full configuration workspace, and optimization pipeline.
