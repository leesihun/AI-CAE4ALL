# The AI-CAE4ALL Studio (GUI)

The Studio is a browser front end for the launcher the CLI uses. You build a
pipeline out of typed blocks. Each executable block becomes a native flat-text
config, is validated by `ai_cae4all`'s real preflight, and runs as the
subprocess `AI_CAE4ALL_main.py` would start. The Studio imports no ML code:
**when the GUI and the launcher disagree, the launcher is right.**

- [`ai_cae4all/specs/`](../ai_cae4all/specs/) is the source of truth for every
  config key the GUI shows.
- [videos/README.md](videos/README.md) has two recorded walkthroughs.

---

## Start it

```bash
python -m pip install -r studio/requirements.txt   # numpy, h5py, trimesh, meshio, gmsh
cd studio
python start_studio.py          # http://127.0.0.1:8080/index.html
python start_studio.py 8090     # another port
```

On Windows, `studio\START_STUDIO.bat [port]` does the same. It uses the first
interpreter it finds: `AI_CAE_STUDIO_PYTHON`, then the active venv
(`VIRTUAL_ENV`), then `python`, `py`, `python3`, `CONDA_PREFIX`.

The console prints `AI-CAE4ALL Studio is ready.` and opens the browser. If the
port is taken, it says so and suggests the next one. Do not serve `studio/`
with `python -m http.server`. That shows the HTML but has no API, so preflight,
runs and file browsing do not work.

The top-right badge (`N/N entrypoints found`) counts registered model IDs whose
repository and entrypoint exist: 10 on a fresh checkout, 12 with the optional
SAR/LDGN routes. It checks discovery only. Environment and data are checked when
you Validate.

**One server per port, and the registry is read once at startup.**
`StudioHTTPServer` sets `allow_reuse_address = False`, so a second instance
fails instead of sharing the port. After editing `ai_cae4all/specs/` or
`studio_backend/`, restart the server.

**Local requests only.** On a specific bind address (the default `127.0.0.1`),
the server answers only a `Host` of `127.0.0.1`, `localhost`, `[::1]` or the
bound host at its own port. Anything else gets 403, which blocks DNS rebinding.
A POST with a foreign `Origin` (including `null`) also gets 403. JSON endpoints
require `Content-Type: application/json` and return 415 otherwise. This is a
single-user development server, not a hosted service.

Review URLs:

| URL | Opens |
| --- | --- |
| `index.html` | the saved workspace, or the HI-MGN template |
| `index.html?review=config` | the SimulGen-VAE template with its config sheet open |
| `index.html?review=optimization` | the Design optimization template with the Optimization workspace open |

---

## The screen

| Region | What it is |
| --- | --- |
| Top bar | Workspace nav, route-health badge, `?` (shortcuts), **Validate** (V), **Run pipeline** (Ctrl+Enter) |
| Canvas toolbar | Library, template picker, Save (Ctrl+S), Export / Import (`.json`), Auto layout (L), Undo (Ctrl+Z), Inspector, zoom −/fit (F)/+ |
| Left | Block library: search, then drag or click to add |
| Centre | Canvas with blocks and typed ports |
| Right | Inspector for the selected block or connection |
| Bottom | Runtime drawer: live log, diagnostics, all jobs, Copy log, Stop process |

The nav opens these workspaces over the canvas: Pipeline, Models, Data,
Geometry, Runs, Optimization, Benchmarks, Artifacts, Deploy, System, Docs.
Evaluation, Compare and Export open from their blocks.

---

## Templates

The picker is generated from `TEMPLATES` in `studio/src/constants.js`. Templates
are grouped by the blocks they contain. Every live route has a default profile
that mirrors a checked-in config.

| Group | Templates |
| --- | --- |
| Mesh field surrogates | HI-MGN multiscale (default), MeshGraphNets flat, MeshGraphNets-V, Transolver, FNO, DeepONet, Point-DeepONet; SAR and LDGN field ensembles when those routes are installed |
| Fixed-geometry and tabular surrogates | SimulGen-VAE reconstruction (ex9), Parametric response estimation (MLP on `ex3_NASA_CRM_mlp_{train,test}.h5`) |
| Generative geometry (SDFFlow) | SDFFlow train (DeepJEB), Design optimization (needs trained SDFFlow) |
| Data preparation | Geometry to HDF5 (ingest) |
| Start from scratch | Untitled pipeline |

- **Mesh templates use ex9 plasticity.** They train on
  `dataset/deterministic/ex9_plasticity.h5` and hold out
  `ex9_plasticity_infer.h5`. The wiring is trainer → Train Metrics, and
  held-out file → Inference → Evaluate → Export. The MeshGraphNets, Transolver
  and operator templates follow their `configs/<Name>/deterministic/ex9/`
  configs. MeshGraphNets-V has no ex9 config, so it reuses those dataset keys
  with its own architecture.
- **The held-out split is its own source block.** If you wire the training
  dataset into Inference, the graph means "predict what you trained on", and the
  graph overrides the trainer's `infer_dataset`.
- **Design optimization fails preflight on a fresh checkout.** It reads
  `output/geometry_generation/studio/sdfflow_{vae,fm}.pth`, which only exist
  after *SDFFlow train* has run.
- **Optional routes are hidden when absent.** `markUnregisteredModels` hides the
  palette blocks and templates of any model the launcher does not report, so SAR
  and LDGN appear only on machines that have `methods/{SAR,LDGN}/`.

---

## Blocks

`native` means a launcher route or backend endpoint. `adapter` means the Studio
combines existing outputs and computes no new physics.

| Category | Blocks |
| --- | --- |
| Sources | CAD, HDF5 Dataset, Design Parameters, Saved ML Model |
| Preparation | Geometry → HDF5 Dataset (the `geometry_ingest` route) |
| Models | one per registered trainable route: MeshGraphNets, MeshGraphNets-V, Transolver, FNO, DeepONet, Point-DeepONet, SimulGen-VAE, SDFFlow, Simple MLP (+ SAR, LDGN when installed) |
| Execution | Inference Run, CAD Generator |
| Optimization | Optimization |
| Evaluation | Evaluate Predictions, Train Metrics, Compare Models |
| Outputs | Export Results |
| Deployment | API Deployment |

MeshGraphNets-V covers HI-MGN-V, HI-MGNflow and HI-MGN-hybrid through
`latent_type` (`global` / `local` / `hybrid`). A route the front end has no
catalog entry for still gets a generic model block (`registerLiveModel`).

**Ports are typed.** A link is offered only when the receiving port accepts the
sending type and the link would not create a loop. Imported pipelines follow the
same rule. `artifact` is a receive-side wildcard: Export accepts anything.
Export's `files` output cannot stand in for a dataset. A required port is
starred only in modes that need it. For example, SDFFlow `sample` reads no
dataset. Optimization has one input, the candidate CSV it ranks. If a saved
pipeline has a link to a removed port, the pipeline still loads, and the dropped
link is named.

**Cards and the inspector.** Card facts are read from real sources. An HDF5
Dataset reads its file through `/api/hdf5/facts`: samples, nodes, timesteps,
state fields, `cond_var` conditions, or shapes/points for SDF and rows/columns
for a table. A model card shows the data file and the keys that identify the
variant. Inference and Evaluate cards name the model and held-out file they
read.

In the inspector, **At a glance** repeats those facts and **Connections** flags
unwired required inputs. A model block shows eight mode-dependent rows: mode,
dataset, checkpoint, epochs/batch/learning rate (or SDFFlow and SimulGen-VAE's
per-stage keys), then the route's distinguishing keys. Everything else is under
**Full config**. Other blocks show every field with one line of help. Inference
hides knobs its linked family cannot read unless you set them.

Row tags:
- *fixed behaviour*: a fact about the block, not a control.
- *auto · &lt;source&gt;*: filled from the graph. Type over it to take control,
  or clear it to follow the graph again.

---

## Full config

The full config is the config modal. It lists every key the live `MethodSpec`
accepts, in sections from Required through Inactive / rejected. Each key shows a
status (required, set, defaulted with the backend default, optional, inactive,
rejected) and help from `HELP`. Dropdowns appear wherever the spec publishes a
closed value set (`CHOICES`). Keys a route knows but does not honour are marked
inactive or rejected instead of looking live.

Controls: mode, preset + **Apply preset**, **Load .txt**, **Export .txt**,
search, *Only values I set*, *Include inactive / rejected*, a raw-text panel
(**Parse & apply**, **Regenerate text**), **Configure via LLM**, **Explain
config**, **Run preflight**, **Save file & close**.

- **Loading or pasting text** marks a key as a manual override only where its
  value differs from what the graph supplies, and names the keys that were
  overridden. The raw panel converts between the form and text. It does not
  keep comments or line order, and it reports how many comment lines it
  dropped.
- **Show field**: a preflight error that names a key gets this button. It
  scrolls to the key and highlights it.
- **Configure via LLM** sends the block's full config and your instruction to
  the LLM endpoint set under System, after a confirmation that lists what will
  be sent. You confirm again before the result is applied. The endpoint is
  OpenAI-compatible, with settings in `studio/runtime/llm_client.local.json`.
  The password comes from `AI_CAE_LLM_PASSWORD` or is entered under System. It is
  kept in server memory only and never written to disk. HTTPS is the default;
  plain HTTP needs an explicit opt-in.

### Presets

| Preset | What it does |
| --- | --- |
| High performance / Low VRAM | Copies the recipe (architecture, optimizer, precision, memory switches) from the largest / smallest checked-in train config named in `PRESET_SOURCES`. The menu names the source, e.g. "High performance (ex9 HI-MGN)" |
| Smoke test | A tiny fixed run |
| Studio defaults | Only for MLP, which has no preset source |
| Route-specific | MGN: Flat MGN / HI-MGN / BSMS-GNN. SDFFlow: VAE only / Flow matching only. SimulGen-VAE: VAE only / LC only / Reconstruct fields. SAR/LDGN: VAE only plus one stage-only preset |

High performance and Low VRAM leave several groups unchanged:
- paths, `opt_*`, seeds, epoch budgets and schedules, `*_interval`;
- the data contract (`input_var`/`output_var`/`cond_var`, node types, rollout);
- dataset-tuned values.

Mesh-sized values are filled only when the block has none. A flat block switched
to a hierarchy gets `voronoi_clusters 512, 64`. Low VRAM also sets
`batch_size 1`, except on SDFFlow and SimulGen-VAE, which keep their per-stage
batch sizes.

---

## SDFFlow, the CAD Generator and optimization

**The SDFFlow model block is the generator being trained.** Its modes are the
training modes plus `evaluate`. `sample`, `reconstruct`, `interpolate` and
`optimize` belong to the **CAD Generator**, whose inspector shows only the rows
its mode reads:
- `num_samples` for sample;
- the interpolation keys for interpolate;
- `input_mesh` for reconstruct;
- `opt_*` for optimize.

The main `opt_*` rows (`opt_analysis`, `opt_budget`, `opt_baseline_size`,
`opt_popsize`, `opt_load_cases`, `opt_vertical_disp_max`, `opt_stress_margin`)
default to
`configs/SDFFlow/geometry_generation/ex1/baseline/config_optimize_sdfflow.txt`.
The rest are under **More optimization settings**, where an empty row means the
native default shown as its placeholder. Older graphs with an `opt_*` value left
on the SDFFlow block still work: the generator uses that value and labels it
"from the SDFFlow block".

- **`opt_budget 0` is a screen.** It analyses `opt_baseline_size` designs once
  and writes `screening.csv`, plus `designs.h5` with every design's surface and
  fields. A budget above 0 runs CMA-ES from that population. The card says which
  (`screening · 12 designs`, `CMA-ES · 120 evaluations`).
- **HI-MGN surrogate port.** Wiring a MeshGraphNets model block or a saved HI-MGN
  checkpoint here does the whole setup:
  - The generator switches to `mode optimize` + `opt_analysis surrogate`.
  - `opt_surrogate_checkpoint` is filled from the wire.
  - From a model block, an inference config is written under
    `studio/runtime/configs/` at Validate and Run, so preflight checks the file
    the run reads. A saved checkpoint has no config, so you type
    `opt_surrogate_config` yourself.
  - If the block uses the ex13 vertical-load contract (`output_var 4`,
    `cond_var 0`), the run is pinned to `opt_load_cases vertical`,
    `opt_length_scale 0.102323`, `opt_material_rho 4470` and
    `opt_fea_verify False`. These are shown read-only, and a deflection limit
    still at 0.2 mm becomes 0.36.
  - In other modes the wire is dormant, not an error.
  - Validate rejects a non-MGN model on the port.
- **FEA re-check** (`opt_fea_verify`): after a surrogate search, the optimized
  design and the best and typical baselines are re-solved with tet4 FEA
  (`design_loop/verify_with_fea.py`). The report gains an FEA section, and the
  Optimization table gains `fea_*` columns. Rank and constrain on those columns.
- **Optimization block / workspace.** It computes a feasible Pareto set and a
  crowding-distance top-k from the candidate CSV. Constraints are `column <=
  value` and are checked against the CSV as you type. An all-infeasible result
  names the constraint that rejected each row. The Pareto plot can use any
  objective or constraint on either axis. Clicking a point, or **View design**,
  opens that design from `designs.h5`.

---

## Validate, then run

**Validate** runs the launcher preflight on every executable step. The first
step's native probe starts the method's interpreter, which can take tens of
seconds. Findings appear in the runtime drawer with step, code, field and hint.
A finding that belongs to a block gets **Fix now**, which opens that block's
config at the field. Graph problems such as a missing input or a type mismatch
appear in the same list.

**Run pipeline** runs the steps in dependency order and re-checks each saved
config just before launch. The drawer streams the native process's stdout. Jobs
live in the backend, so closing the tab does not stop a run.

- **Ownership.** Each canvas has a `canvas_id`, which is recorded on the jobs it
  launches. A job's status is painted only on its own canvas. On load, the canvas
  rebuilds block status from its jobs, including runs that finished while the
  page was closed. A job delivers results only to blocks that have not had a
  newer delivery (`results_from`), so results you chose later are kept. Older
  untagged jobs are handled through `owned_jobs` / `legacy_canvas`.
- **Load pipeline** (Runs) opens a run's graph as a copy that owns only that
  run. Compare Models uses the run this canvas owns. If there is none, it uses
  the newest run of that model and marks it *not this canvas*.
- **Stop** works at every stage, including during the launch preflight. A runner
  error ends the job as *failed* with the traceback, never stuck at *running*.
- **Server restart.** A job whose process is still alive is adopted. Stop
  reaches it by pid, and the job is finalised as *interrupted* when the process
  exits.
- **Logs.** The drawer shows the last 120 000 characters (`log_truncated`); the
  full log is on disk.

Model artifacts go under the repository's `output/`. Studio files (saved
configs, uploads, exports, reports, job records) go under the git-ignored
`studio/runtime/`.

---

## Evaluation and comparison

**Evaluate Predictions** compares a prediction HDF5, or a directory of per-sample
files from `mode inference`, against a ground-truth HDF5. It writes a report and
a per-sample CSV. The contract (sample matching, shapes, field mapping) is
checked before anything is scored.

1. **Fields pair by declared name.** Positional pairing is offered only when the
   counts match. It starts with nothing ticked and scores nothing until you
   confirm.
2. **Coordinate rows are never scored.** They are copied into every rollout, so
   scoring them gives R² = 1. A coordinate-only mapping is refused. Rows named
   `x`/`y`/`z` also count as coordinates.
3. **The trailing node-type row is not a prediction.** Rollout writers record
   `output_var` beside `num_features`, so the scoreable rows are read from the
   file. For older files, the row is dropped by name.
4. **Samples pair by ID.** A file name stands in for an ID only when the file
   holds one record. Two different declared IDs are never paired.

Metrics are relative L2, MAE, RMSE, max absolute error and R², each summarised
as mean, median, p95, min and max. **Relative L2 and R² are macro-averaged over
fields.** Pooled versions are also reported (`relative_l2_pooled`,
`r2_pooled`), along with per-field CSV columns and `aggregate_by_field`.

**Compare Models** ranks the mean of one numeric column across the selected
CSVs. It cross-checks each CSV's array `contract` and sample IDs, and reports any
mismatch above the ranking.

---

## Workspaces

| Workspace | What it gives you |
| --- | --- |
| **Models** | Every registered route: repository, entrypoint, modes, keys, dataset kind, install health, checked-in examples |
| **Data** | HDF5 catalog and sample viewer (points / mesh / field, timestep player, compare, upload with **+ Add dataset**). **Inspect HDF5** shows the file's contract; **Use in pipeline** drops a source block |
| **Geometry** | Geometry Explorer: an SDFFlow slider between two noise endpoints for the car, DeepJEB and MCB profiles (each needs its checkpoints under `output/geometry_generation/`). For DeepJEB, **Predict response** runs the ex13 HI-MGN checkpoint on the generated shape |
| **Runs** | Studio jobs with status, steps and logs; **Load pipeline**; Train Metrics plots every metric parsed from a run log |
| **Optimization** | Pareto / top-k ranking of a candidate CSV (see above); the selection is written back as CSV |
| **Benchmarks** | Every config in `configs/campaigns/dataset_matrix/manifest.json`, each with preflight and loadable into the canvas. A passing config is not a result |
| **Artifacts** | Outputs and checkpoints, searchable, with **Use in pipeline** |
| **Deploy** | Portable CPU inference ([guides/inference-bundle.md](guides/inference-bundle.md)). The run button stays disabled until checkpoint metadata confirms the family is supported. **Build .exe with PyInstaller** is enabled when PyInstaller is installed |
| **System** | Interpreters, CUDA devices, install health, LLM endpoint settings, and a full `--audit-configs` |
| **Docs** | The repository's Markdown, rendered, with working in-repo links |

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| A spec or backend change has no effect | The server holds the old module. Restart it |
| The badge shows fewer routes than expected | A method directory or entrypoint is missing; compare with `python AI_CAE4ALL_main.py --list-models` |
| Validate seems to hang | The native probe is starting the method interpreter |
| A workspace looks empty | It is still loading repository state ("Loading real AI-CAE4ALL state…") |
| Evaluate reports a perfect score | Check the field pairing |
| Inference produced nothing to evaluate | The Inference block needs `inference_output_dir` (auto-filled next to the checkpoint) |
| The Studio will not guess `input_var` / `output_var` | Intended: on mesh routes with `cond_var` rows they are not the feature-row count |
| The Optimization Run button stays disabled | A constraint names a missing column or is not `column <= value`; the note under the field says which |
| A dataset card shows no facts | The path is not `.h5`/`.hdf5`, the file is missing, or the server is offline |

---

## Extending it

| To add | Do this |
| --- | --- |
| A model route | Register its spec in the launcher. `registerLiveModel` installs a generic block from the live spec. Add a `MODEL_CATALOG` entry for labels, defaults and ports, and a `TEMPLATES` entry (every route needs one) |
| A block | A `BLOCK_SPECS` entry in `studio/src/constants.js` (ports, defaults, category, maturity) |
| A template | A `TEMPLATES` entry mirroring a checked-in config that passes Validate on a fresh checkout. Grouping is automatic |
| Key help | `HELP` in `constants.js` |
| A dropdown | `CHOICES`, mirroring the spec validator exactly |
| A preset source | `PRESET_SOURCES` in `constants.js` |

Restart the server after any spec or backend change. Never import
`studio_backend.state` from a helper script: constructing `StudioState()` runs
job recovery and can rewrite a live server's records.

**Front end (`studio/src/`).**

| Module(s) | Role |
| --- | --- |
| `constants.js` | catalog, templates, help, choices |
| `main.js` | boot, template picker, review URLs |
| `graph.js` | canvas |
| `cards.js`, `inspector.js` | cards and the inspector |
| `config.js` | config sheet and presets |
| `validate.js` | graph checks and step construction |
| `autofill.js` | graph-derived values |
| `run.js` | jobs and diagnostics |
| `studio.js` | workspaces |
| `geometry-explorer.js` | Geometry workspace |
| `viewer.js`, `render3d.js` | sample viewer |
| `persistence.js` | saved documents |
| `markdown.js` | docs rendering |

**Back end (`studio/studio_backend/`).**

| Module(s) | Role |
| --- | --- |
| `http_handler.py` | routes and request checks |
| `state.py` | jobs and recovery |
| `suite_bridge.py` | the bridge to `ai_cae4all` |
| `analysis.py` | evaluation, comparison, optimization |
| `training_metrics.py` | log parsing for Train Metrics |
| `hdf5_preview.py`, `geometry_preview.py`, `mesh_topology.py` | sample viewer |
| `geometry_explorer.py`, `geometry_worker.py`, `geometry_analysis.py` | Geometry workspace |
| `llm_configure.py` | LLM client |

### Local API

All endpoints are under `/api/`.

**GET**

| Group | Endpoints |
| --- | --- |
| Registry and configs | `health`, `models`, `configs?model=`, `config?path=`, `audit-configs` |
| Docs and files | `docs`, `doc?path=`, `files?kind=`, `text?path=`, `image?path=` |
| HDF5 and previews | `hdf5`, `hdf5/facts`, `hdf5/samples`, `hdf5/sample`, `preview/samples`, `preview/sample`, `predictions` |
| Runs and checkpoints | `checkpoint?path=`, `benchmarks`, `jobs`, `jobs/{id}`, `training-metrics` (filters `job_id`, `node_id`, `model_id`) |
| Deploy, LLM, geometry | `deploy`, `llm/settings`, `geometry-explorer/profiles`, `geometry-explorer/jobs?id=` |

**POST**

| Group | Endpoints |
| --- | --- |
| Configs and runs | `preflight`, `config/explain`, `config/save`, `pipeline/run`, `jobs/{id}/cancel` |
| Inference and deploy | `inference/run`, `build/exe` |
| Analysis | `evaluation/schema`, `evaluation/run`, `comparison/schema`, `comparison/run`, `optimization/schema`, `optimization/run`, `export` |
| Geometry | `geometry-explorer/generate`, `geometry-explorer/analyze` |
| LLM | `llm/configure`, `llm/settings` |
| Fixtures | `simulgen/smoke-fixture`, `geometry/smoke-fixture`, `viewer/smoke-fixture` (non-scientific usability fixtures) |
| Upload | `upload?kind=dataset\|geometry\|checkpoint` |

- **Upload** takes a raw body and an `X-Filename` header (8 GiB max), and is
  stored under `studio/runtime/uploads/<kind>/`.
- **JSON bodies** are capped at 2 MiB.
- **Preflight** returns 200 with `ok: false` when it finds errors.
- **Static files**: the server serves only `index.html`, the stylesheets,
  `src/*.js` and `runtime/exports/`.

### Tests

```bash
python -m pytest -q studio/studio_backend
```

`studio_backend/conftest.py` points the runtime root at a temporary directory,
because importing `studio_backend.state` recovers jobs, and the tests write
configs and exports. The launcher contracts the GUI depends on are covered by
`python AI_CAE4ALL_main.py --audit-configs` and per-config `--check`. See
[guides/testing.md](guides/testing.md).
