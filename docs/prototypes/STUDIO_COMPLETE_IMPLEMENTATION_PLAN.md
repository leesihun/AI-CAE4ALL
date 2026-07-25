# AI-CAE4ALL Studio: complete implementation plan

Status: proposed production plan  
Snapshot: 2026-07-24  
Product research: [STUDIO_FUNCTIONAL_COVERAGE_RESEARCH.md](STUDIO_FUNCTIONAL_COVERAGE_RESEARCH.md)  
Interactive design prototype: [ai-cae4all-studio.html](ai-cae4all-studio.html)  
Earlier block-contract blueprint: [PIPELINE_STUDIO_IMPLEMENTATION.md](PIPELINE_STUDIO_IMPLEMENTATION.md)

## 1. Objective

Build a local-first Studio in which all current AI-CAE4ALL functionality is:

- discoverable without reading source code;
- callable through movable, linkable, typed pipeline blocks;
- inspectable at the individual sample and artifact level;
- validated through the existing layered launcher semantics;
- repeatable through saved, versioned pipelines and durable runs;
- observable through logs, metrics, checkpoints, and resource telemetry;
- connected to the repository's method, configuration, dataset, benchmark, and
  research documentation;
- honest about whether a capability is **Native**, **Adapter**, or **Roadmap**.

The production Studio must preserve native method repositories and entrypoints.
It is an orchestration and visualization layer, not a replacement training
framework.

### 1.1 Revised product surface: capability-rich blocks, not utility-block sprawl

The 2026-07-24 prototype review changes the catalog strategy. A pipeline block
represents a meaningful engineering object or operation. Controls, diagnostics,
statistics, and viewers that have no independent dataflow meaning belong inside
the object that owns them.

The default visible block catalog is:

| Group | Visible block | What it owns |
|---|---|---|
| Sources | CAD | STEP/IGES/STL/PLY/OBJ selection, geometry preview, units, metadata |
| Sources | HDF5 Dataset | samples, fields, geometry, topology, splits, statistics, normalization, parameter bindings |
| Sources | Design Parameters | typed values, units, ranges, sample scope, HDF5 binding, OOD warnings |
| Sources | Saved ML Model | `.pth`, version, architecture, normalization, lineage, compatibility, download |
| Preparation | Geometry → HDF5 Dataset | meshing, conversion, schema authoring, conversion diagnostics |
| Models | Simple MLP | live tabular `X[S,N] → Y[S,M]` parameter-to-response estimation |
| Models | MeshGraphNets | one entry with Flat MGN, HI-MGN, and BSMS-GNN variants |
| Models | MeshGraphNets-V | training plus uncertainty-aware model output |
| Models | Point-DeepONet | point/condition operator training |
| Models | DeepONet | branch/trunk operator training |
| Models | FNO | spectral operator training |
| Models | GINO | geometry-informed operator training |
| Models | Transolver3 | display name for the compatible `transolver` backend ID |
| Models | SDFFlow | VAE and conditional flow-matching training stages |
| Run | Inference Run | single, batch, rollout, and variational ensemble modes plus result viewers |
| Run | CAD Generator | conditional generation, reconstruction, blending, gallery, and geometry checks |
| Optimization | Optimization | geometry gates, physics evaluators, objectives, constraints, Pareto selection, diversity, search, and verification |
| Experiments | Hyperparameter Sweep | model-family sweep, child runs, ranking, best `.pth` |
| Evaluation | Evaluate Predictions | metrics, per-sample breakdowns, error maps, distributions, exports |
| Evaluation | Compare Models | accuracy, speed, VRAM, size, uncertainty, synchronized sample comparison |
| Deployment | API Deployment | versioned REST endpoint, OpenAPI, health, auth, limits, parity, rollback |
| Outputs | Export Results | HDF5/VTK/STL/CSV/JSON/HTML and provenance |

The following are deliberately **not** standalone default blocks:

| Former block | New owner |
|---|---|
| Config Source / ML Config | model inspector, with form, presets, raw view, examples, resolved command |
| Config Preflight | automatic gate inside every model/run/deployment action |
| Dataset Split / Dataset Statistics / Dataset Inspector / Field Viewer / Geometry Viewer | HDF5 Dataset |
| Resource/VRAM Probe / Training Monitor / logs | ML Model and its durable session |
| Batch/Rollout Inference / Variational Ensemble / Distribution Viewer | Inference Run |
| SDFFlow Sample / Reconstruct / Interpolate / Candidate Gallery / Validity Filter | CAD Generator |
| Error Map | Evaluate Predictions |
| Surrogate Rank | replaced by the visible **Optimization** block; evaluation and search have independent dataflow meaning |
| Benchmark Campaign | named workflow in the Benchmarks workspace, not a vague ordinary node |
| Inference Bundle / Build Executable | top-right **Build .exe** workflow |
| Previous Artifact | typed “Use in pipeline” action that inserts CAD, HDF5 Dataset, Saved ML Model, or another concrete source |

This change reduces visual complexity without removing functionality. The full
capability remains searchable through block inspectors, workspaces, templates,
and contextual documentation.

### 1.2 Dataset parameterization findings and product rule

The shipped HDF5 files do not share one universal parameter contract:

- `dataset/deepjeb.h5` is explicitly conditional. Root `cond_names` declares
  `bbox_x`, `bbox_y`, `bbox_z`, `volume`, and `area`, and each shape stores a
  five-value `cond` vector. Shipped SDFFlow configs use
  `bbox_x,bbox_z,volume,area`.
- Plasticity stores `die_profile_mm` as a static model-state conditioning
  channel and also retains `data/{sample_id}/die_profile`.
- The Point-DeepONet paper data broadcasts five load/material/location
  conditions per point.
- `dataset/ex1.h5` and `dataset/ex2.h5` vary by sample geometry and physical
  response but expose no named global design-parameter vector. Their shared
  `nodal_data` channel convention alone is not enough to declare an editable
  design parameter.
- current common Neural Operator configs explicitly set
  `global_condition_features none`; a generic global-condition loader is not
  live in that path.

Therefore the Studio must not treat every input channel as safely editable.
Design Parameters may link to HDF5 Dataset only through a verified
`ParameterBinding`. Unknown channels remain read-only.

### 1.3 Simple MLP boundary

The live Simple MLP route is intentionally narrow:

- input: a fixed-length vector of named design parameters;
- target: one or more global scalar/vector responses such as peak stress,
  displacement summary, force, energy, mass, or cost;
- output: `ParametricResponse`, not a full nodal field;
- supported actions: train, validate, infer, compare, export `.pth`, and deploy;
- excluded from its initial scope: variable-size mesh-field prediction and
  autoregressive rollout.

As of this snapshot, `mlp` is registered with train and inference modes,
`cae_suite/specs/mlp.py`, `MLP/MLP_main.py`, checked example configs, and
pipeline tests. Its live contract has 33 accepted keys and uses tabular HDF5
datasets with `X[S,N]` and `Y[S,M]`; it must not be linked to a mesh HDF5
dataset merely because both file names end in `.h5`. Portable standalone
inference remains a separate compatibility gate.

## 2. Product principles

### 2.1 Pipeline first

The graph is the primary authoring surface. Data preparation, models, inference,
evaluation, and deployment are pipeline blocks. Visualization, validation,
statistics, configuration, and resource controls are embedded when they are
owned by one block; named multi-stage benchmark workflows live in their
workspace. Supporting workspaces manage durable objects referenced by the
graph.

### 2.2 Native runtime remains authoritative

The Studio may:

- edit and validate configs;
- build native commands;
- choose interpreters;
- launch subprocesses;
- observe outputs;
- index artifacts.

It must not silently reinterpret model semantics, checkpoint loading, dataset
normalization, or training logic. Backend-specific validation remains the final
authority.

### 2.3 Every block is inspectable

Every block publishes one or more typed artifacts, even if the artifact is a
validation report or process log. Clicking a block opens:

1. its input and output samples;
2. configuration and resolved defaults;
3. run state and diagnostics;
4. provenance;
5. contextual documentation.

### 2.4 Durable jobs, immutable evidence

A canvas node is not a process. A node launches a durable `RunRecord`; training
creates a durable `TrainingSession`. Deleting a node does not delete run history
or artifacts. Re-running a node creates a new run version.

### 2.5 Honest maturity

- **Native**: backed by a current launcher route/mode or verified live tool.
- **Adapter**: repository workflow exists, but requires a stable wrapper.
- **Roadmap**: researched product surface with no current callable backend.

Roadmap blocks are hidden by default and cannot be executed.

### 2.6 Local engineering data stays local

The first production target is a local web application bound to
`127.0.0.1`. It must not upload datasets, checkpoints, geometries, logs, or
documentation to an external service.

## 3. Scope

### 3.1 Production v1

Production v1 includes:

- pipeline graph editor;
- complete native block catalog;
- stable adapters for priority repository workflows;
- Data, Experiments, Models, Benchmarks, Artifacts, Deploy, System, and Docs
  workspaces;
- config form/raw editor and layered preflight;
- durable local run database;
- subprocess execution and cancellation;
- sample/artifact viewers;
- training-session monitoring;
- checkpoint catalog and metadata;
- benchmark protocol templates;
- standalone inference-bundle and executable workflows;
- governed API deployment workflow;
- repository documentation catalog and contextual help.

### 3.2 Deferred after v1

The Optimization block is visible in v1, but its layers carry separate
maturity:

- **Native evidence:** SDFFlow zero-crossing/geometry reports and conditional
  descriptor-adherence ranking;
- **Adapter:** candidate-to-HDF5 conversion followed by one or more existing
  Saved ML Model inference routes;
- **Roadmap:** general objective/constraint declarations, Pareto and diversity
  selection, DOE/design-space exploration, iterative evolutionary or Bayesian
  optimization, high-fidelity solver connectors, calibrated confidence/OOD
  gates, robustness/reliability analysis, and active learning.

The following additional items also remain Roadmap:

- custom block/plugin SDK;
- ONNX and container deployment;
- multi-user collaboration, permissions, review, and approvals.

### 3.3 Explicit non-goals

Production v1 will not:

- merge method virtual environments;
- rewrite native training loops;
- convert all configs into a new file format;
- make inert benchmark-intent keys appear functional;
- load multi-gigabyte HDF5/checkpoint files in the browser;
- delete repository outputs when a Studio record is deleted;
- claim paper-equivalent accuracy from smoke runs;
- execute arbitrary user shell strings.

## 4. Proposed repository layout

Add a self-contained production package while leaving the prototype under
`docs/prototypes/`:

```text
studio/
  README.md
  pyproject.toml
  backend/
    ai_cae_studio/
      __init__.py
      app.py
      settings.py
      paths.py
      api/
        catalog.py
        configs.py
        pipelines.py
        runs.py
        artifacts.py
        datasets.py
        models.py
        benchmarks.py
        deploy.py
        docs.py
        system.py
        events.py
      domain/
        block.py
        pipeline.py
        run.py
        artifact.py
        validation.py
        training.py
        docs.py
      blocks/
        registry.py
        sources.py
        preparation.py
        models.py
        execution.py
        experiments.py
        evaluation.py
        visualization.py
        deployment.py
        outputs.py
      adapters/
        launcher.py
        geometry_ingest.py
        sdfflow_dataset.py
        benchmarks/
          elasticity.py
          plasticity.py
          fno_darcy.py
          gino_carcfd.py
          deeponet_fractional2d.py
          point_deeponet.py
        sweeps/
          mgn_variational.py
          transolver.py
        inference_bundle.py
        executable_build.py
      execution/
        scheduler.py
        subprocess_runner.py
        cancellation.py
        event_bus.py
        log_parser.py
        resource_sampler.py
      indexing/
        artifacts.py
        checkpoints.py
        datasets.py
        docs.py
      preview/
        hdf5.py
        geometry.py
        fields.py
        checkpoints.py
        reports.py
      storage/
        database.py
        migrations/
        repositories.py
      tests/
  frontend/
    package.json
    src/
      app/
      components/
      features/
        pipeline/
        data/
        experiments/
        models/
        benchmarks/
        artifacts/
        deploy/
        system/
        docs/
      blocks/
      viewers/
      api/
      state/
      styles/
    tests/
  contracts/
    block-definition.schema.json
    pipeline.schema.json
    run-record.schema.json
    artifact-manifest.schema.json
    validation-report.schema.json
  scripts/
    run_studio.py
    build_frontend.ps1
    smoke_studio.py
```

The exact JavaScript package manager may follow repository preference. The
important boundary is a typed frontend talking only to the local Studio API;
native processes remain behind the Python backend.

## 5. Technology decisions

### 5.1 Backend

Recommended:

- Python 3.11+;
- FastAPI/Starlette for typed HTTP and WebSocket endpoints;
- Pydantic models for API/domain validation;
- SQLite for Studio metadata;
- native filesystem for large artifacts;
- `asyncio` subprocess supervision;
- `psutil` for portable process-tree and resource observation where available.

Rationale:

- the repository is Python-first;
- adapters can import suite metadata without importing heavyweight method
  runtimes;
- training remains in isolated subprocesses and method environments;
- FastAPI makes contracts and a local OpenAPI surface explicit;
- SQLite is sufficient for a single-user local application and avoids adding
  infrastructure.

The backend must not import PyTorch, h5py, gmsh, or method packages at startup.
Heavy inspection runs in the target method interpreter or a dedicated preview
worker.

### 5.2 Frontend

Recommended:

- React + TypeScript;
- React Flow for graph editing;
- Zustand or equivalent small client store;
- TanStack Query for server state;
- vtk.js for mesh/field/geometry visualization;
- uPlot or ECharts for high-volume metrics and sweep comparisons;
- a Markdown renderer with sanitization and heading navigation;
- Playwright for browser integration tests.

The prototype's visual language should be retained: compact engineering UI,
pipeline-first canvas, typed ports, right inspector, dark field viewer, and
strong status distinctions.

### 5.3 Storage

Default local state:

```text
output/studio/
  studio.sqlite3
  pipelines/
  runs/
    <run_id>/
      resolved_config.txt
      command.json
      stdout.log
      stderr.log
      report.json
      artifacts.json
  previews/
  reports/
```

Large native artifacts remain where their configs place them. The Studio stores
manifests and references, not duplicate checkpoint/HDF5 files. Generated
previews may be cached and safely regenerated.

## 6. Core domain contracts

All contracts must be versioned before implementing full UI behavior.

### 6.1 BlockDefinition

Required fields:

```json
{
  "schema_version": "studio.block.v1",
  "type": "run.sdfflow.interpolate",
  "version": "1.0.0",
  "label": "SDFFlow Interpolate",
  "category": "Execution",
  "maturity": "native",
  "backend": {
    "adapter": "suite_launcher",
    "route": "sdfflow",
    "mode": "interpolate"
  },
  "inputs": [],
  "outputs": [],
  "config_schema": {},
  "docs": [],
  "restart_policy": "rerun",
  "cancellation_policy": "graceful_then_tree_stop"
}
```

Rules:

- port IDs are stable across compatible block versions;
- input/output types come from a central type registry;
- a block cannot claim Native without a successful adapter contract test;
- defaults must indicate their provenance: authored, suite default, native
  default, checkpoint-derived, dataset-derived, or runtime-derived;
- unsupported controls are omitted, not disabled without explanation.

### 6.2 PipelineDocument

Contains:

- stable pipeline ID;
- human name and description;
- document version;
- ordered node and edge records;
- block type/version per node;
- authored node config;
- viewport state;
- template provenance;
- created/updated timestamps;
- optional tags.

Saving creates a mutable draft. Running freezes an immutable pipeline version.

### 6.3 ArtifactManifest

Contains:

- artifact ID and version;
- artifact type;
- immutable content hash when practical;
- canonical path;
- producing run/node;
- parent artifact IDs;
- sample count and sample index strategy;
- field names, components, units, ranges, shapes, and timesteps;
- geometry/mesh topology metadata;
- condition names and ranges;
- split metadata;
- checkpoint family, architecture, stage, normalization, EMA, and compatibility;
- available renderers;
- validation status and qualifications;
- preview cache keys.

Artifact types for v1:

- `geometry`
- `dataset`
- `conditions`
- `config`
- `checkpoint`
- `training_metrics`
- `field`
- `candidates`
- `report`
- `log`
- `bundle`
- `files`

### 6.4 RunRecord

Lifecycle:

```text
created
  -> preflighting
  -> queued
  -> starting
  -> running
  -> stopping
  -> succeeded | failed | cancelled
```

Record:

- run ID, pipeline version, node ID, block definition/version;
- parent campaign/sweep/session;
- resolved inputs and output expectations;
- authored and resolved config;
- exact command and working directory;
- selected interpreter;
- environment and hardware snapshot;
- subprocess identifiers;
- timestamps and exit code;
- structured diagnostics;
- log offsets;
- metrics and resource samples;
- produced artifact IDs;
- restart checkpoint and cancellation reason.

### 6.5 ValidationReport

Validation stages:

1. graph
2. block configuration
3. parse
4. route
5. spec
6. filesystem
7. environment
8. dataset
9. checkpoint
10. native
11. command

Each issue:

- severity;
- diagnostic code;
- message;
- stage;
- node ID;
- config key/path/port if applicable;
- suggested action;
- documentation path/anchor.

### 6.6 TrainingSession

A TrainingSession references one or more RunRecords and persists:

- model ID and method variant;
- stage for SDFFlow;
- status;
- epoch/step progress;
- train/validation metrics;
- GPU and memory telemetry;
- best/latest/periodic checkpoints;
- resume lineage;
- stop/pause semantics;
- downstream checkpoint publication state.

### 6.7 ParameterBinding and DatasetVariant

A parameter edit is an immutable overlay. The Studio must never silently modify
the source HDF5 file.

```json
{
  "schema_version": 1,
  "dataset_artifact_id": "art_dataset_ex1_v1",
  "bindings": [
    {
      "parameter_id": "die_profile",
      "display_name": "Die profile",
      "unit": "mm",
      "storage": {
        "kind": "nodal_channel",
        "dataset_path": "data/{sample_id}/nodal_data",
        "channel_index": 6,
        "feature_name": "die_profile_mm",
        "broadcast": "profile_over_y_and_time"
      },
      "role": "input",
      "dtype": "float32",
      "shape": [101],
      "training_range": {"min": 0.0, "max": 1.0},
      "editable": true
    }
  ],
  "overrides": [
    {
      "parameter_id": "die_profile",
      "sample_selector": {"sample_ids": ["42"]},
      "value_artifact_id": "art_parameter_profile_17",
      "range_status": "inside_training_range"
    }
  ],
  "materialization": "run_local_copy_on_write"
}
```

Required binding kinds:

- `global_scalar`;
- `global_vector`;
- `per_sample_scalar`;
- `nodal_channel`;
- `time_channel`;
- `profile`;
- `sdfflow_condition`.

Each binding records semantic role, path/index, feature name, unit, shape,
dtype, broadcast rule, editable flag, train range, and provenance. Roles
`coordinate`, `target`, `identifier`, `node_type`, and `output` are read-only by
default. Materialization creates a run-scoped dataset view or copy-on-write
artifact, hashes it, and records the source plus overlay in lineage.

### 6.8 ModelConfiguration

ML configuration is owned by the model block rather than passed through a
generic config node. It has:

- a large, searchable configuration workspace rather than a narrow inspector;
- every key accepted by the selected live `MethodSpec`, including keys that the
  shared schema marks inactive or a validator rejects for the selected variant;
- sections for required, data/output, architecture, training, resources/runtime,
  inference/evaluation, advanced, and inactive/rejected fields;
- dropdowns only for genuinely closed choices, booleans, and modes;
- validated manual input for numeric, list, path, expression, and open-ended
  family-specific values;
- obvious named presets only: checked repository example, smoke test, low-VRAM,
  MGN Flat/HI/BSMS variants, and SDFFlow stage presets;
- a diff and explicit confirmation before a preset overwrites existing values;
- synchronized raw flat-config editing, `.txt` import/paste, and `.txt` export
  for exact compatibility with checked and user-authored configs;
- an example config for the selected model and mode, sourced from checked
  repository files rather than fabricated generic defaults;
- resolved defaults and repository-root-relative paths;
- the exact native command;
- versioned import/export;
- “set by” provenance for repository preset, imported text, checkpoint,
  inferred default, or manual override;
- validation issues attached to the exact field.

The current `MethodSpec` contract is not sufficient to generate this UI without
heuristics. It exposes known, required, recommended, and default keys plus
validators, but not a field's type, label, section, choices, units, min/max,
help text, visibility conditions, or active/inactive/removed state. Add:

```python
@dataclass(frozen=True)
class ConfigFieldSpec:
    name: str
    label: str
    section: str
    value_type: Literal["bool", "int", "float", "string", "path", "list", "enum"]
    choices: tuple[str, ...] = ()
    units: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    required_when: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    visible_when: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    help: str = ""
    advanced: bool = False
    status: Literal["active", "inactive", "removed", "runtime"] = "active"
```

Publish it through
`GET /catalog/models/{model_id}/config-schema?mode={mode}`. The response must
include the schema version, accepted key count, resolved defaults, required
keys, repository examples, preset definitions, and validator diagnostics. The
production frontend must render from this response; copying validator-local
enum sets into JavaScript is only acceptable in the static prototype.

The automatic preflight runs on **Train**, **Run selected**, **Run pipeline**,
and deployment actions. It performs graph, config, parse, route, spec,
filesystem, environment, dataset, checkpoint, native dry-run, and command
checks. Failure blocks execution and presents the diagnostic code, cause,
affected value, and suggested repair in the model inspector.

### 6.9 DesignOptimization

`Optimization` replaces `Surrogate Rank`. It is not just a sorter and it is not
embedded in CAD Generator because candidate evaluation, selection, iterative
search, and verification have independent inputs, outputs, run budgets, and
lineage.

Inputs:

- `CandidateSet`: generated or imported CAD plus geometry/condition evidence;
- zero or more `SavedModel` evaluators;
- optional `ParameterSpace` defining named design or latent variables, bounds,
  units, and distributions;
- optional solver/truth results for verification.

Configuration:

- mode: evaluate fixed batch, screen/select, optimize, or verify;
- objectives: evaluator response, minimize/maximize direction, units, scaling,
  aggregation over fields/timesteps, and optional target value;
- constraints: hard/soft, equality/inequality, bound, tolerance, and failure
  policy;
- feasibility gates: valid zero crossing, watertightness, connectivity,
  thickness/meshability when available, and condition adherence;
- selection: feasible-only, non-dominated Pareto set, diversity policy, top-k,
  and optional user-authored scalarization;
- uncertainty: ensemble/disagreement, OOD policy, missing-evaluator policy, and
  confidence threshold;
- search: variable space, DOE/optimizer, population/batch size, evaluation
  budget, seed, caching, stopping rule, and parallelism;
- verification: trusted solver adapter, discrepancy threshold, finalist count,
  and active-learning disposition.

Outputs:

- per-candidate evaluation table with status, parameters, geometry evidence,
  objective/constraint responses, evaluator/checkpoint versions, uncertainty,
  feasibility, and provenance;
- feasible and rejected sets with explicit reasons;
- Pareto front plus diversity-aware top-k;
- convergence, hypervolume, feasibility, evaluation-cost, and discrepancy
  histories when an iterative search runs;
- an optimization report and solver-verified finalists.

A weighted score is optional and must never be the only decision view. The
baseline workflow first removes infeasible designs, then exposes the
non-dominated set. Scalarization is permitted only when the user deliberately
provides weights or a target/utility model. Any surrogate result presented as a
decision must retain its training-domain/OOD and verification status.

## 7. Typed graph semantics

### 7.1 Compatibility

An edge is valid when:

- output and input types match;
- input accepts a declared union/supertype;
- cardinality constraints hold;
- no cycle is created unless a future explicit loop/control block supports it;
- the upstream artifact qualification satisfies the downstream gate.

`artifact` may act as a deliberate generic port, but production blocks should
prefer specific types.

### 7.2 Validation timing

Run validation:

- while connecting ports;
- on config change;
- on explicit Validate;
- before a node run;
- before a branch run;
- before a full pipeline run.

Expensive native probes should be debounced and cached by config, environment,
dataset, and checkpoint fingerprints.

### 7.3 Execution plan

Before running:

1. freeze a pipeline version;
2. topologically sort the requested subgraph;
3. resolve cached/pinned upstream artifacts;
4. validate every node;
5. build per-node execution intents;
6. enqueue runnable nodes;
7. publish run events;
8. index artifacts after each successful node;
9. stop dependent nodes when a required predecessor fails;
10. allow independent branches to continue if policy permits.

## 8. Backend adapter matrix

### 8.1 Unified launcher adapter

Use `AI_CAE4ALL_main.py` for all registered route/mode work:

- model training;
- model inference;
- SDFFlow train/train_vae/train_fm/sample/reconstruct/interpolate;
- geometry_ingest ingest/inspect;
- check, strict check, dry-run, explain config, defaults;
- model listing and description;
- JSON diagnostics.

The adapter writes an immutable run config, invokes the launcher with the
selected interpreter policy, and parses the JSON report where available.

### 8.2 Model route catalog

The registry API reads current suite registry/spec data and exposes:

- `meshgraphnets`
- `meshgraphnets-v`
- `point_deeponet`
- `deeponet`
- `fno`
- `gino`
- `transolver`
- `sdfflow`
- `mlp`
- `geometry_ingest`

HI-MGN and BSMS-GNN are model presets over `meshgraphnets`, not new routes.

### 8.3 Geometry and dataset adapters

- Geometry Ingest: launcher route.
- SDF Dataset Builder:
  `Geometry_generation/build_dataset.py`.
- Dataset Probe: reuse launcher probe semantics under a declared interpreter.
- Split/Statistics: first wrap stable existing benchmark/native tools; later
  create a shared implementation only after its contract is validated across
  datasets.

### 8.4 Benchmark adapters

Each benchmark adapter declares:

- preparation command;
- source and license qualification;
- dataset manifest;
- config set;
- protocol label;
- preflight;
- training jobs;
- inference jobs;
- evaluation;
- report outputs;
- smoke/full budgets;
- resume/retry behavior.

Priority order:

1. elasticity;
2. plasticity;
3. FNO Darcy;
4. GINO CarCFD;
5. DeepONet fractional-2D;
6. Point-DeepONet.

The adapter must preserve existing script CLIs rather than copy their logic into
the backend.

### 8.5 Sweep adapters

Initial adapters:

- MGN-V `b8_all_warpage_input` config generation, runs, histogram/CSV/PNG
  comparison;
- Transolver ex2 sweep scheduling and summary.

Represent sweep runs as children of a parent run. Stopping the sweep stops new
children; cancelling may stop active children only after an explicit
destructive-action confirmation.

### 8.6 Deployment adapters

- standalone inference:
  `inference/run_inference.py`;
- family detection and preview of checkpoint metadata;
- SDFFlow legacy VAE/FM merge:
  `Geometry_generation/merge_sdfflow_checkpoint.py`;
- PyInstaller build:
  `inference/pyinstaller.spec`;
- parity/limitation report from `inference/README.md` and build smoke output.

The UI must warn that source re-vendoring is manual until the planned
`rebuild_bundle.py` exists.

## 9. Workspaces

### 9.1 Pipeline

Required features:

- searchable grouped palette;
- Native/Adapter/Roadmap badges;
- drag, click-to-add, move, multi-select, duplicate, delete;
- typed ports and connection validation;
- pan, zoom, fit, auto-arrange;
- undo/redo;
- templates;
- node and branch execution;
- validation overview;
- minimap for large graphs;
- right-side inspector;
- block sample/artifact preview;
- save/version/clone/export pipeline.

Templates for v1:

- Import → Validate → Train → Predict → Evaluate → Export
- Model-zoo training and comparison
- SDFFlow train → sample → inspect
- SDFFlow reconstruct
- SDFFlow interpolate
- MGN-V ensemble → distribution analysis
- Elasticity benchmark
- Plasticity campaign
- Standalone inference bundle

### 9.2 Data

- dataset and geometry catalog;
- recent/pinned datasets;
- schema and compatibility status;
- sample browser;
- fields, units, conditions, shapes, timesteps;
- split manifests;
- preparation history;
- source provenance;
- “use in pipeline” action.

### 9.3 Experiments

- sessions across all pipelines;
- live and archived status;
- loss/metric charts;
- GPU memory, utilization, throughput;
- logs;
- checkpoint timeline;
- pause/resume/stop;
- sweep/campaign parent-child tree;
- comparison table and charts;
- resource-probe results.

### 9.4 Models

- all current model IDs and method variants;
- install health and valid modes;
- method selection guidance;
- checkpoint catalog;
- model/stage/architecture metadata;
- normalization and EMA metadata;
- training run and dataset lineage;
- best/latest/validated aliases;
- compatibility check;
- “use in pipeline” and “run inference” actions.

Promotion is a metadata alias in v1. It must never move or overwrite the
checkpoint file.

### 9.5 Benchmarks

- six repository benchmark families;
- preparation status;
- protocol and qualification;
- smoke/full controls;
- expected budget;
- component job graph;
- leaderboard;
- per-sample metrics;
- failure/retry status;
- generated comparison reports.

### 9.6 Artifacts

- typed filter;
- path and run search;
- lineage;
- sample previews;
- side-by-side compare;
- download/open path;
- “use as block input”;
- validation and qualification badges.

### 9.7 Deploy

- compatible checkpoint selection;
- standalone bundle command editor;
- family-specific options;
- bundle build and smoke test;
- PyInstaller executable build;
- parity and known-limitations view;
- handoff manifest.

### 9.8 System

- route and entrypoint health;
- current launcher interpreter;
- per-model interpreter mappings;
- dependency/import probes;
- CUDA visibility and GPU inventory;
- filesystem/output health;
- config audit;
- known gaps and inert-key warnings;
- recent diagnostic failures.

### 9.9 Docs

The current non-vendored inventory is 54 Markdown documents after this plan is
added. The Docs workspace must:

- discover allowed Markdown files dynamically;
- exclude vendored `source/`, build `dist/`, caches, and generated dependencies;
- group documents:
  - Suite & Studio
  - Methods
  - Data & Benchmarks
  - Geometry Generation
  - MeshGraphNets Research
  - Neural Operators
  - Developer
- search title, path, headings, and body;
- render a table of contents;
- support heading anchors and back/forward navigation;
- show the real repository path and last-modified time;
- open the local file in the IDE/system when requested;
- map blocks, config keys, diagnostic codes, checkpoints, and benchmark protocols
  to relevant documents and anchors.

Do not expose vendored third-party documentation as if it were AI-CAE4ALL's own.
A separate “third-party sources” filter may be added later.

## 10. Documentation indexing design

### 10.1 Discovery

At startup and on manual refresh:

1. enumerate `*.md` under allowlisted repository paths;
2. apply exclusion rules;
3. parse first H1 as title;
4. extract headings and stable slug anchors;
5. index plain text;
6. record path, category, size, and modified time;
7. cache by path + size + modified time.

### 10.2 Context mapping

Create `studio/docs/context-map.json`:

```json
{
  "block:model.transolver": [
    "docs/methods/09_Transolver.md",
    "CONFIGURATION_REFERENCE.md#transolver"
  ],
  "diagnostic:CFG-UNKNOWN-001": [
    "CONFIGURATION_REFERENCE.md"
  ],
  "benchmark:elasticity": [
    "dataset/benchmarks/elasticity/README.md"
  ]
}
```

The map supplements automatic keyword matching. Missing mappings must not block
the main task; they only reduce contextual suggestions.

### 10.3 Rendering security

- sanitize rendered HTML;
- disable raw HTML by default;
- do not execute scripts;
- resolve local links only inside the repository;
- distinguish external links;
- render code blocks without executing them;
- require explicit action before launching a command copied from a document.

## 11. Visualization architecture

### 11.1 Server-side preview extraction

The browser receives compact preview payloads, never raw multi-gigabyte files.

For HDF5:

- list groups/samples lazily;
- return metadata without materializing arrays;
- load one requested sample/field/timestep;
- decimate large point/mesh previews deterministically;
- preserve original indices for selected probes;
- compute cached min/max and histograms;
- cap response bytes.

For geometry:

- use existing mesh readers in an isolated preview worker;
- return indexed triangles/lines/points or a compact VTK/glTF-compatible payload;
- compute bounds, watertight/manifold flags where available;
- keep repair as an explicit pipeline operation.

For checkpoints:

- use safe `weights_only` metadata paths;
- never deserialize arbitrary pickle payloads in the web server process;
- expose architecture and normalization metadata only.

### 11.2 Viewer capabilities

Production viewers:

- surface, wireframe, points, graph edges;
- scalar contours;
- vector glyphs;
- field/component selector;
- time player;
- clipping plane and slices;
- probes and measurements;
- synchronized cameras;
- side-by-side and difference modes;
- candidate gallery;
- training curves;
- histogram/distribution comparison;
- screenshot/export.

### 11.3 Progressive loading

The artifact modal loads:

1. manifest;
2. sample list;
3. selected sample metadata;
4. low-resolution preview;
5. optional higher-detail preview.

Switching samples cancels obsolete preview requests.

## 12. Job execution and safety

### 12.1 Command construction

Adapters return an argv list, working directory, interpreter, and environment
delta. Never execute a shell-concatenated string.

Persist both:

- structured argv for execution;
- shell-escaped display form for the user.

### 12.2 Environment isolation

Interpreter precedence mirrors the suite:

1. explicit run override;
2. exact model mapping;
3. method/spec mapping;
4. activated environment/default.

The backend process must not inherit a method's imports into its own process.

### 12.3 Cancellation

Windows:

- create and record the process tree;
- request graceful termination first when supported;
- stop the resolved child tree with native process APIs/PowerShell-safe logic;
- never construct destructive commands from unresolved strings.

POSIX:

- create a process group;
- send graceful signal;
- escalate after a configured timeout.

The UI distinguishes:

- pause scheduling;
- stop after current work;
- cancel active work;
- force stop.

### 12.4 Recovery

On backend restart:

- mark previously running processes as `unknown`;
- reconcile recorded PIDs against command/start-time fingerprints;
- reattach log tailing if confidently matched;
- otherwise mark `interrupted`;
- retain all checkpoints and partial artifacts;
- offer resume only where the backend supports it.

## 13. API surface

Initial endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/v1/catalog/blocks` | Block definitions and maturity |
| GET | `/api/v1/catalog/models` | Live registry routes, modes, health |
| GET | `/api/v1/catalog/templates` | Pipeline templates |
| POST | `/api/v1/configs/parse` | Parse authored config |
| POST | `/api/v1/configs/validate` | Layered preflight |
| GET | `/api/v1/configs/models/{id}` | Required/default/path rules |
| GET/POST | `/api/v1/pipelines` | List/create pipelines |
| GET/PUT | `/api/v1/pipelines/{id}` | Read/update draft |
| POST | `/api/v1/pipelines/{id}/validate` | Validate graph |
| POST | `/api/v1/pipelines/{id}/runs` | Run graph/branch/node |
| GET | `/api/v1/runs` | Search run history |
| GET | `/api/v1/runs/{id}` | Run detail |
| POST | `/api/v1/runs/{id}/stop` | Graceful stop |
| POST | `/api/v1/runs/{id}/cancel` | Cancel active process |
| GET | `/api/v1/artifacts` | Search artifacts |
| GET | `/api/v1/artifacts/{id}` | Manifest |
| GET | `/api/v1/artifacts/{id}/samples` | Paged samples |
| GET | `/api/v1/artifacts/{id}/preview` | Selected preview |
| GET | `/api/v1/datasets` | Dataset catalog |
| GET | `/api/v1/checkpoints` | Checkpoint catalog |
| GET | `/api/v1/benchmarks` | Benchmark protocols/status |
| GET | `/api/v1/docs` | Documentation catalog |
| GET | `/api/v1/docs/{id}` | Renderable document |
| GET | `/api/v1/docs/search` | Full-text/heading search |
| GET | `/api/v1/system/health` | Environments, routes, GPU, storage |
| WS | `/api/v1/events` | Run/session/artifact/system events |

Mutating endpoints use optimistic concurrency or document revision IDs.

## 14. Frontend state and behavior

### 14.1 Server state

TanStack Query owns:

- catalogs;
- pipeline documents;
- runs and sessions;
- artifacts;
- docs;
- system health.

### 14.2 Client state

The local store owns:

- canvas viewport and selection;
- unsaved draft changes;
- inspector tab;
- open modal/workspace;
- temporary link gesture;
- filters and searches;
- undo/redo stack before save.

### 14.3 Event handling

WebSocket events invalidate or patch relevant queries:

- run state;
- log append;
- metric point;
- resource sample;
- checkpoint created;
- artifact indexed;
- diagnostic emitted;
- system health change.

Events carry monotonically increasing per-run sequence numbers. Reconnect uses
the last seen sequence and falls back to refetch.

## 15. Implementation phases

### Phase 0 — contracts and architectural decisions

Deliverables:

- Studio ADR;
- versioned JSON schemas;
- reduced visible-block catalog and embedded-capability ownership map;
- ParameterBinding, DatasetVariant, ParametricResponse, and ModelConfiguration
  schemas;
- block maturity policy;
- local state/path policy;
- API error contract;
- Native adapter acceptance rules.

Exit gate:

- contracts validate representative MGN, Transolver, SDFFlow, benchmark,
  dataset, and deployment examples.

### Phase 1 — backend foundation and live discovery

Deliverables:

- backend skeleton and settings;
- root path validation;
- SQLite migrations;
- model/route discovery;
- config parse/describe/default/validate APIs;
- layered JSON validation report;
- system interpreter and route health;
- documentation inventory API.

Exit gate:

- live `--list-models` and `--describe` parity;
- representative configs produce the same pass/fail result as the CLI;
- all 54 documentation files are indexed with correct exclusions.

### Phase 2 — pipelines and native launcher blocks

Deliverables:

- block registry;
- pipeline CRUD and versioning;
- graph validation;
- unified launcher adapter;
- node/branch/pipeline execution planning;
- subprocess runner;
- logs, events, cancellation;
- native model, geometry-ingest, unified Inference Run, and CAD Generator
  blocks;
- model-local config and automatic preflight orchestration;
- typed source insertion from prior artifacts.

Exit gate:

- one pipeline can ingest data, train or consume a saved model, run
  inference, and publish artifacts;
- every live route and mode has a tested block path.

### Phase 3 — frontend shell and graph editor

Deliverables:

- production frontend shell;
- Pipeline, Data, Experiments, Models, Benchmarks, Artifacts, Deploy, System,
  Docs navigation;
- graph editor and inspector;
- model-owned form/raw config editor with examples and presets;
- validation badges and diagnostics;
- pipeline templates;
- responsive layout;
- keyboard and accessibility basics.

Exit gate:

- browser tests cover add/move/link/configure/validate/save/run/stop/inspect;
- no route/mode is accessible only through a hidden raw config field.

### Phase 4 — artifact indexing and viewers

Deliverables:

- artifact manifest/indexer;
- dataset/checkpoint indexers;
- HDF5/geometry/checkpoint preview workers;
- vtk.js field/geometry viewer;
- training charts;
- sample browser;
- model-local resource view, HDF5-local sample/statistics/field views,
  Inference-local rollout/distribution views, CAD Generator gallery/validity
  views, and evaluation-local error views;
- lineage panel.

Exit gate:

- representative dataset, rollout, SDFFlow sample/reconstruction/interpolation,
  training session, checkpoint, report, and log are inspectable;
- large files remain server-side and previews respect byte/time limits.

### Phase 5 — experiments and model governance

Deliverables:

- durable TrainingSession;
- checkpoint timeline/catalog;
- resume lineage;
- MGN variant presets;
- integrate the existing Simple MLP live route, config spec, trainer/inference,
  tests, and checked examples as a typed `ParametricResponse` workflow;
- AR-OT/AR-RT controls;
- parallelism/precision/checkpointing controls;
- MGN-V ensemble and distribution controls inside Inference Run;
- sweep adapter and model-local resource probe;
- cross-run comparison.

Exit gate:

- model/session state survives page and backend restart;
- pause/stop/cancel semantics are tested;
- checkpoint metadata and compatibility are visible before inference.

### Phase 6 — benchmarks

Deliverables:

- six benchmark descriptors;
- preparation/campaign/evaluation adapters;
- smoke/full/paper qualification;
- parent-child campaign views;
- leaderboard and per-sample results;
- report generation.

Exit gate:

- elasticity smoke completes end to end;
- plasticity campaign dry-run matches its current scheduler intent;
- no smoke result is presented as paper-equivalent.

### Phase 7 — generated-design evaluation and optimization

Deliverables:

- visible Optimization block and evaluation-table artifact;
- native SDFFlow geometry/condition evidence adapter;
- candidate → evaluator HDF5 conversion with checkpoint compatibility checks;
- multi-evaluator batch inference and lineage;
- objective/constraint schema with units and scaling;
- feasible/rejected partition and Pareto/diversity selection;
- OOD/uncertainty status contract;
- cached, budgeted DOE/evolutionary/Bayesian driver interface;
- high-fidelity solver verification adapter contract;
- convergence, Pareto, hypervolume, feasibility, and discrepancy views.

Exit gate:

- a generated DeepJEB candidate batch can be geometry-gated and exported with
  auditable rejection reasons;
- at least one compatible physics surrogate can evaluate a candidate batch
  through the existing inference route without bypassing its dataset and
  checkpoint contract;
- a two-objective constrained study returns a reproducible feasible Pareto set;
- unverified or OOD surrogate results are never labeled physically valid.

### Phase 8 — deployment and handoff

Deliverables:

- top-right Build `.exe` workflow and deployment workspace;
- checkpoint-family detection;
- SDFFlow merge workflow;
- PyInstaller build;
- target smoke test;
- parity/limitations report;
- handoff manifest.
- API Deployment block and service adapter;
- generated request/response schema and OpenAPI;
- model version pinning, health/readiness, auth, batching/resource limits,
  parity probe, audit log, and rollback.

Exit gate:

- at least one rollout-family and one SDFFlow checkpoint complete portable CPU
  inference from Studio;
- executable build result is inspectable and traceable.
- one supported saved model can be deployed locally, queried through its
  generated API contract, health-checked, version-swapped, and rolled back.

### Phase 9 — hardening and release

Deliverables:

- performance budgets;
- crash recovery;
- path/security audit;
- accessibility review;
- Windows and Ubuntu validation;
- installation/run documentation;
- migration/back-up guidance;
- release checklist.

Exit gate:

- all v1 acceptance criteria pass in clean checkout environments.

## 16. Detailed engineering workstreams

### Workstream A — block catalog

1. Define type registry.
2. Encode all source/preparation/model/execution/experiment/evaluation/inspect/
   deploy/output blocks.
3. Add MGN presets.
4. Add route/mode schemas.
5. Add maturity and docs mapping.
6. Add adapter contract tests.

### Workstream B — config editor

1. Read suite specs.
2. Build field groups by model/mode.
3. Preserve raw config comments/order when possible.
4. Show authored/resolved/derived/default values distinctly.
5. Resolve path previews.
6. Run debounced lightweight validation.
7. Run explicit full preflight.
8. show exact native command.

### Workstream C — execution

1. Run database.
2. queue and dependency planner.
3. argv-safe subprocess runner.
4. event stream.
5. log persistence/tailing.
6. resource sampler.
7. graceful cancellation.
8. restart reconciliation.
9. artifact discovery.

### Workstream D — data and artifact previews

1. metadata-only HDF5 scan.
2. paged sample API.
3. deterministic decimation.
4. field range/histogram cache.
5. geometry preview.
6. checkpoint safe metadata.
7. typed viewer registry.
8. compare/difference synchronization.

### Workstream E — documentation

1. dynamic discovery and exclusions.
2. title/heading/body index.
3. sanitized Markdown rendering.
4. grouped library.
5. contextual mapping.
6. local IDE/system-open action.
7. broken-link check.
8. index refresh.

### Workstream F — quality

1. schema validation.
2. adapter unit tests.
3. CLI parity tests.
4. API tests.
5. browser tests.
6. visual regression screenshots.
7. cancellation/recovery tests.
8. large-artifact performance tests.
9. Windows path/interpreter tests.
10. accessibility tests.

## 17. Test strategy

### 17.1 Unit tests

- type compatibility;
- graph cycle/cardinality validation;
- config serialization;
- maturity rules;
- command argv construction;
- run state transitions;
- artifact manifest parsing;
- docs exclusion/category/anchor logic;
- diagnostic mapping.

### 17.2 Contract tests

For each Native block:

1. build a minimal valid intent;
2. produce a config/argv;
3. run check/dry-run;
4. compare route/mode/command to expected;
5. verify declared outputs can be indexed.

### 17.3 Integration tests

- root launcher check and dry-run;
- geometry ingest smoke;
- SDFFlow dataset builder smoke;
- representative model smoke configs;
- standalone inference smoke;
- PyInstaller descriptor/build preflight;
- benchmark adapter dry-runs.

### 17.4 Browser tests

Minimum flows:

- load every template;
- search all block groups;
- add and connect every port type;
- reject incompatible/cyclic links;
- edit form and raw config;
- validate and jump to an issue;
- start/observe/stop a run;
- open samples from every renderer family;
- search and open Docs;
- inspect model/checkpoint;
- open benchmark protocol;
- build deployment intent;
- restore saved pipeline.

### 17.5 Visual regression

Capture at minimum:

- Pipeline default;
- large model-zoo graph;
- training sessions;
- Data;
- Models;
- Benchmarks;
- Deploy;
- System;
- Docs library;
- field viewer;
- generated candidate gallery;
- mobile/narrow inspector.

## 18. Performance budgets

- backend cold start without native method imports: under 3 seconds;
- first shell render: under 2 seconds on local machine;
- block search: under 50 ms for v1 catalog;
- docs search: under 150 ms for current repository;
- metadata-only artifact list: under 500 ms after index warmup;
- sample switch to low-resolution preview: under 1 second when cached;
- live metric/log event latency: under 500 ms;
- graph interaction: 60 fps for 100 nodes / 150 edges;
- no browser response payload over the configured preview limit;
- no unbounded metric/log arrays in client memory.

## 19. Security and integrity

- bind to loopback by default;
- validate the resolved repository root at startup;
- allowlist readable/writable roots;
- reject traversal outside roots;
- construct argv arrays;
- sanitize Markdown;
- safe checkpoint metadata loading;
- no arbitrary code execution blocks in v1;
- redact configured secret-like environment values from reports;
- do not copy large data without explicit action;
- preserve original native artifacts;
- require confirmation for active-process cancellation and material deletion;
- record the exact action and target in an audit log.

## 20. Migration and compatibility

No native config migration is required. The Studio stores:

- pipeline documents;
- manifests;
- indexes;
- run metadata;
- preview caches.

Existing configs, datasets, checkpoints, and outputs remain valid in place.

Database migrations:

- are forward-only and versioned;
- back up `studio.sqlite3` before destructive schema changes;
- never modify repository datasets/checkpoints;
- rebuild indexes from the filesystem when necessary.

## 21. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Native specs and runtime keys drift | Read live registry/specs, run adapter contract tests, show native-validation result |
| Benchmark scripts have different CLIs | One explicit adapter per benchmark; no generic command guessing |
| Windows subprocess trees are difficult to stop | Record exact process identity; graceful stop then verified tree stop |
| HDF5/checkpoints are too large for UI | Metadata-first, lazy sample preview, deterministic decimation, byte caps |
| Checkpoint pickle safety | Use existing safe probe patterns and isolated worker |
| Method environments conflict | One native subprocess/interpreter per run; no backend imports |
| Prototype implies fake functionality | Maturity badge and adapter tests; Roadmap cannot run |
| Smoke results are overstated | Mandatory protocol qualification and expected-budget display |
| Docs inventory becomes stale | Dynamic discovery, refresh, and broken-link CI |
| Artifact paths move | Store fingerprints and support re-linking; never silently bind wrong file |
| Long sessions outlive UI/backend | Durable database/logs, restart reconciliation, resume state |

## 22. Definition of done

Production v1 is complete when:

1. all nine live route IDs and every valid mode are discoverable and runnable;
2. HI-MGN and BSMS-GNN are discoverable presets;
3. launcher config/introspection/preflight features are accessible;
4. data/geometry preparation and sample inspection are available;
5. AR-OT/AR-RT and supported scaling controls are represented honestly;
6. SDFFlow stage training, sample, reconstruct, and interpolate work;
7. MGN-V ensembles and distribution artifacts work;
8. benchmark adapters expose their preparation, budgets, qualification, and
   results;
9. sweeps and resource probes have durable parent/child runs;
10. every block output opens an appropriate sample/artifact viewer;
11. runs preserve config, command, interpreter, hardware, logs, metrics,
    checkpoints, outputs, and lineage;
12. checkpoints can be searched and compatibility-checked;
13. portable inference and executable build are available;
14. System exposes route/environment/CUDA/filesystem/dataset/checkpoint health;
15. Docs dynamically shows all first-party Markdown documents and opens
    contextual help;
16. Native, Adapter, and Roadmap states cannot be confused;
17. browser, adapter, contract, cancellation, recovery, and large-artifact tests
    pass;
18. the prototype's key interaction—move/link blocks and click any block to see
    samples—survives in the production implementation.

## 23. Immediate next implementation slice

The first pull request should be deliberately vertical:

1. add `studio/backend` skeleton and root validation;
2. expose block/model/docs catalogs;
3. implement config parse/describe/check/dry-run adapter;
4. persist one pipeline and one run record in SQLite;
5. implement safe subprocess execution for a launcher dry-run;
6. add a minimal React shell with Pipeline, System, and Docs;
7. render the complete block palette and the 54-document library;
8. connect one Config Source → Config Preflight graph;
9. show the layered validation report;
10. add contract/API/browser tests.

That slice proves the hardest architectural boundaries—live discovery, typed
blocks, native command parity, durable state, and repository documentation—
before GPU work or heavy visualization is introduced.
