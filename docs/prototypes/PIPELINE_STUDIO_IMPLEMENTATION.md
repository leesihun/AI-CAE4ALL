# AI-CAE4All Pipeline Studio — implementation blueprint

Status: design contract for the interactive prototype in
[`ai-cae4all-studio.html`](./ai-cae4all-studio.html). The HTML is a standalone
simulation; this document describes where the production implementation should
live and how it should call the existing repository.

Companion documents:

- [STUDIO_FUNCTIONAL_COVERAGE_RESEARCH.md](STUDIO_FUNCTIONAL_COVERAGE_RESEARCH.md)
  audits the complete live product surface and external workflow research.
- [STUDIO_COMPLETE_IMPLEMENTATION_PLAN.md](STUDIO_COMPLETE_IMPLEMENTATION_PLAN.md)
  defines the production architecture, adapters, phases, tests, and release
  gates.

## Product decision

The opening route is the **Pipeline** page. A pipeline is a saved directed graph:

- blocks are meaningful sources, transformations, models, runs, evaluations,
  and destinations;
- typed ports define which artifacts may be linked;
- edges bind an output artifact to a downstream input;
- selecting a block opens its configuration;
- every block exposes an artifact/sample viewer;
- “Run selected” executes the selected block plus unresolved upstream
  dependencies;
- “Run pipeline” validates and executes the whole graph in topological order.

Runs, artifact browsing, and settings are supporting views. They must not sit in
front of the graph or turn graph construction into a sequential wizard.

The default graph must also avoid utility-block sprawl. Sample viewers, dataset
statistics/splits, model configuration/preflight/resources/training progress,
rollout and uncertainty controls, candidate galleries, geometry checks, error
maps, and logs are embedded inside their owning block. The reduced default
catalog and ownership map are authoritative in
[STUDIO_COMPLETE_IMPLEMENTATION_PLAN.md](STUDIO_COMPLETE_IMPLEMENTATION_PLAN.md#11-revised-product-surface-capability-rich-blocks-not-utility-block-sprawl).

## Recommended production layout

```text
AI-CAE4ALL/
├─ studio/                         # new React + TypeScript web application
│  ├─ src/app/                     # routes; "/" renders PipelinePage
│  ├─ src/pipeline/                # canvas, blocks, ports, edges, commands
│  ├─ src/inspectors/              # selected-block configuration panels
│  ├─ src/samples/                 # artifact explorer and VTK.js viewers
│  ├─ src/training/                # sessions, curves, GPUs, checkpoints, logs
│  ├─ src/runs/                    # all pipeline-run history and cancellation
│  └─ src/api/                     # generated API client and event stream
├─ cae_suite/
│  ├─ pipeline/                    # new backend graph runtime
│  │  ├─ schema.py                 # versioned graph/block/artifact schemas
│  │  ├─ catalog.py                # BlockSpec registry
│  │  ├─ model_catalog.py          # live MethodRegistry -> per-model BlockSpec
│  │  ├─ validation.py             # type, required-input, cycle validation
│  │  ├─ executor.py               # dependency scheduler and cache checks
│  │  ├─ artifacts.py              # manifests, hashes, sample indices
│  │  └─ adapters/                 # wrappers around existing launch surfaces
│  └─ studio_api/                  # new FastAPI application
│     ├─ app.py
│     ├─ pipelines.py
│     ├─ artifacts.py
│     └─ events.py                 # WebSocket run event stream
└─ output/studio/runs/<run_id>/
   ├─ graph.json                   # immutable submitted graph snapshot
   ├─ run.json                     # state and timestamps
   └─ nodes/<node_id>/
      ├─ artifact.json             # ArtifactManifest
      ├─ stdout.log
      └─ ...
```

Do not put Studio code inside MeshGraphNets, Transolver, Neural Operator,
SDFFlow, or geometry-generation method directories. The suite-root launcher and
registry are the stable integration boundary.

## Frontend stack

- **React + TypeScript** for the production application.
- **React Flow (`@xyflow/react`)** for selectable/draggable custom nodes,
  multiple named handles, edge validation, undo/redo, and save/restore. Its
  custom nodes can contain interactive controls and charts.
- **ELKjs** for “Auto arrange”; its layered algorithm is intended for directed
  node-link diagrams and understands ports.
- **VTK.js** for geometry, mesh, scalar/vector field, tensor, and volume
  visualization in the browser. Keep lightweight thumbnails in the canvas and
  initialize the full renderer only after the artifact explorer opens.
- **FastAPI + WebSocket events** for live state, progress, logs, and artifact
  notifications.

Primary references:

- React Flow custom nodes:
  <https://reactflow.dev/learn/customization/custom-nodes>
- React Flow handles and port IDs:
  <https://reactflow.dev/learn/customization/handles>
- React Flow examples:
  <https://reactflow.dev/examples>
- ELKjs:
  <https://github.com/kieler/elkjs>
- VTK.js:
  <https://kitware.github.io/vtk-js/docs/>
- FastAPI WebSockets:
  <https://fastapi.tiangolo.com/advanced/websockets/>

Rete.js has a useful split between dataflow and control-flow engines, but the
production backend must remain authoritative for this repository. Python jobs,
subprocess isolation, GPU assignment, durable artifacts, and resumable runs
cannot safely depend on a browser-resident execution engine.

## Registered model blocks

The model palette must be generated from the suite's live `MethodRegistry`, not
maintained as an unrelated frontend list. The authoritative development check
is:

```powershell
python AI_CAE4ALL_main.py --list-models
```

As of the 2026-07-24 live audit, that command reports nine installed model IDs:

| Block type | Visible block | Training modes |
|---|---|---|
| `model.meshgraphnets` | MeshGraphNets | `train` |
| `model.meshgraphnets-v` | MeshGraphNets-V | `train` |
| `model.point_deeponet` | Point-DeepONet | `train` |
| `model.deeponet` | DeepONet | `train` |
| `model.fno` | FNO | `train` |
| `model.gino` | GINO | `train` |
| `model.transolver` | Transolver3 (backend ID remains `transolver`) | `train` |
| `model.sdfflow` | SDFFlow | `train`, `train_vae`, `train_fm` |
| `model.mlp` | Simple MLP | `train` |

`geometry_ingest` remains the **Geometry → HDF5 Dataset** preparation block
because it is a registered tool, not a trainable model. An existing checkpoint
is represented by **Saved ML Model**; it does not replace any model block.

Every model block has the same lifecycle-shaped interface while retaining a
model-specific configuration schema:

```text
dataset ────────────────┐
conditions (optional) ─┼─> [ model.<registered_id> ] ─> checkpoint
resume (optional) ──────┘                              └─> metrics
```

- `checkpoint` can feed inference, evaluation, export, or another model's
  optional resume input;
- `metrics` is rendered inside the model session and can feed Compare Models or
  a report/export path;
- the model-specific schema exposes only relevant architecture controls (for
  example FNO modes, Transolver slices, or SDFFlow VAE/FM stages);
- changing the registered model ID changes the block type, rather than changing
  a generic model dropdown inside one block.

The exception is the MeshGraphNets architecture family: Flat MGN, HI-MGN, and
BSMS-GNN are presets inside one `model.meshgraphnets` block because the current
runtime difference is configuration, not a separate suite model ID.

Every model inspector owns an exhaustive, searchable ML configuration
workspace covering every live `known_keys` entry, checked examples, obvious
presets with a confirmed diff, synchronized form/raw `.txt` input and export,
data mapping,
automatic layered preflight, split/statistics context, GPU/VRAM/throughput,
training state, validation samples, checkpoints, and `.pth` download. There is
no separate Config Source, Config Preflight, Resource Probe, or Training Monitor
in the default palette.

## Training sessions

Calling a model block creates a durable training session. Sessions are not
browser-only progress indicators and are not discarded when the graph closes.

Suggested backend schema:

```json
{
  "session_id": "train_mgn_042",
  "run_id": "run_...",
  "node_id": "trainer",
  "model_id": "meshgraphnets",
  "mode": "train",
  "status": "running",
  "process": {"pid": 18420, "host": "worker-01"},
  "resources": {"gpu_ids": [0, 1], "parallel_mode": "ddp"},
  "progress": {"epoch": 342, "total_epochs": 500, "step": 1882},
  "metrics": {"train_loss": 0.00231, "validation_loss": 0.00284},
  "checkpoints": {
    "latest": "nodes/trainer/checkpoints/latest.pt",
    "best": "nodes/trainer/checkpoints/best.pt"
  },
  "resume_token": "..."
}
```

The Training view is a supporting surface reached from the global navigation or
from any model block. It must provide:

1. active, queued, completed, failed, paused, and cancelled sessions;
2. train/validation curves and model-specific metrics;
3. epoch, step, learning rate, throughput, ETA, and wall time;
4. GPU utilization, allocated/reserved memory, temperature, and process rank;
5. structured logs plus raw stdout/stderr;
6. best/latest/periodic checkpoints and downstream artifact publication;
7. pause where supported, graceful cancel, retry, and resume;
8. a link back to the exact pipeline version and owning model block.

Persist session events server-side with monotonically increasing sequence IDs.
WebSocket delivery updates the UI, but reconnect must replay missed events from
the stored event log. A browser disconnect must never terminate training.

## Block contract

The catalog is data, not hard-coded UI conditionals:

```json
{
  "type": "run.inference",
  "version": 1,
  "label": "Inference Run",
  "category": "Execution",
  "inputs": [
    {"id": "data", "artifact_kind": "dataset", "required": true},
    {"id": "model", "artifact_kind": "checkpoint", "required": true}
  ],
  "outputs": [
    {"id": "prediction", "artifact_kind": "field", "required": true}
  ],
  "config_schema": {},
  "executor": "cae_suite.pipeline.adapters.inference:run",
  "visualizer": "field"
}
```

Each `BlockSpec` supplies:

1. stable type and schema version;
2. named, typed input and output ports;
3. JSON Schema for editable configuration;
4. backend executor reference;
5. visualizer ID and empty/running/error states;
6. resource hints such as CPU, GPU, VRAM, and concurrency group.

For model blocks, the catalog additionally supplies `model_id`, supported
training modes, model-specific config schema, session visualizer, and two
standard outputs: `checkpoint` and `metrics`.

Use explicit converters instead of implicit coercion. For example,
`geometry → dataset` is a `Geometry Ingest` block, not a hidden edge
conversion.

## Artifact and sample contract

All block outputs write a small universal manifest even when the payload remains
HDF5, VTK, STL, JSON, CSV, or a checkpoint:

```json
{
  "schema": "ai-cae4all.artifact/v1",
  "artifact_id": "art_...",
  "run_id": "run_...",
  "node_id": "inference",
  "kind": "field",
  "content_hash": "sha256:...",
  "provenance": {
    "block_type": "run.inference",
    "config_hash": "sha256:...",
    "input_artifacts": ["art_dataset", "art_checkpoint"]
  },
  "samples": [
    {
      "sample_id": "bracket_0142",
      "label": "Bracket 0142",
      "preview": {"format": "vtp", "url": "samples/bracket_0142.vtp"},
      "stats": {"nodes": 48216, "elements": 223904},
      "fields": ["von_mises", "displacement"],
      "timesteps": [0, 1, 2, 3]
    }
  ]
}
```

This manifest is why every block can offer the same **Inspect samples** action.
The visualizer switches on `kind`; it does not need to understand the executor.
Large payloads stay on disk and are fetched lazily.

Training adds two artifact kinds:

- `training_session`: the durable mutable job record and event-log reference;
- `metrics`: append-only scalar/series data that can be linked to monitor and
  comparison blocks.

Checkpoint artifacts remain immutable publications. `latest` is a pointer to a
specific immutable checkpoint version, not an overwritten opaque file.

## Execution contract

1. Freeze and store the submitted graph snapshot.
2. Validate block versions, required ports, edge types, and cycles.
3. Resolve the requested targets and their transitive upstream dependencies.
4. Topologically schedule ready blocks, respecting resource/concurrency hints.
5. Compute a cache key from block type/version, normalized config, executor
   version, and input artifact hashes.
6. Spawn the adapter as an isolated process, stream structured events, and
   capture stdout/stderr.
7. Atomically publish the artifact manifest and notify downstream blocks.
8. Persist terminal state so a disconnected browser can reconnect.

The isolation in step 6 is mandatory for inference. The current
`inference/cae_infer/registry.py` guards against loading more than one model
family in a process because vendored packages use conflicting top-level module
names. The Studio API should orchestrate processes; it should not import and
retain multiple drivers.

Suggested event envelope:

```json
{
  "event": "node.progress",
  "run_id": "run_...",
  "node_id": "inference",
  "seq": 37,
  "time": "2026-07-24T00:00:00Z",
  "data": {"progress": 0.64, "message": "sample 64 / 100"}
}
```

## First implementation slice

Build one vertical slice before adding all block types:

1. Pipeline CRUD and versioned graph JSON.
2. `CAD Source → Geometry Ingest → Dataset Inspector`.
3. `Checkpoint + Dataset → Inference Run → Field Viewer`.
4. Artifact manifests and sample indexing for the existing HDF5 helpers.
5. Run event streaming, reconnect, cancellation, and logs.
6. One model block first (`model.meshgraphnets`) with durable session state,
   metrics, best/latest checkpoints, and resume.
7. Generate the other seven model blocks from the live registry and attach
   their model-specific configuration schemas.
8. VTK.js views for geometry/mesh/field plus training-curves and generic
   files/metrics fallbacks.
9. Cache/re-run semantics, templates, undo/redo, then auto-layout.

Acceptance criteria:

- `/` opens Pipeline, with no wizard in front of it;
- a block can be dragged, linked, selected, configured, duplicated, and deleted;
- invalid type links and cycles are rejected before submission;
- each block has inspectable empty/running/complete/error visual states;
- run-selected includes required upstream work;
- page refresh does not lose the graph or an active run;
- every completed output has an `ArtifactManifest`;
- separate model-family nodes execute in separate processes.
- all eight live model IDs have distinct palette block types;
- every model block creates a durable training session and exposes checkpoint
  plus metrics outputs;
- refreshing or closing the browser does not stop or lose a training session;
- a session links back to its exact graph version and owning model block.
