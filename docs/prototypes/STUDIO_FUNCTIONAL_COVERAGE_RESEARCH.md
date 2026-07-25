# AI-CAE4ALL Studio: complete functional coverage and workflow research

Status: product and implementation research  
Snapshot: 2026-07-24  
Prototype: [ai-cae4all-studio.html](ai-cae4all-studio.html)  
Implementation blueprint: [PIPELINE_STUDIO_IMPLEMENTATION.md](PIPELINE_STUDIO_IMPLEMENTATION.md)

## Executive conclusion

The Studio should not be a dashboard wrapped around a few training buttons. It
should be the visual operating system for AI-CAE4ALL.

The pipeline canvas remains the primary interaction: every operation that can be
composed should be represented by a typed, movable, linkable block. The rest of
the Studio exists to manage the durable objects used by those blocks:

- **Data** manages geometry, HDF5 datasets, schemas, splits, statistics, and
  sample-level inspection.
- **Experiments** manages training sessions, sweeps, resource probes, logs,
  metrics, and comparisons.
- **Models** manages the live model catalog, variants, checkpoints, metadata,
  lineage, and promotion.
- **Benchmarks** manages the repository's paper-validation and cross-model
  campaigns as repeatable protocols rather than loose scripts.
- **Artifacts** manages fields, geometries, metrics, reports, checkpoints, and
  their provenance.
- **Deploy** packages checkpoints into the standalone CPU inference bundle and
  executable.
- **System** exposes interpreters, CUDA/GPU health, dependency probes, config
  validation, route resolution, and diagnostics.
- **Learn** makes the method documentation and configuration reference available
  in context.

“Accessible in Studio” does not mean rewriting each backend. The production
Studio should invoke existing entrypoints through thin adapters and publish a
common run/artifact contract. It also must not imply that an aspirational
feature exists. This document uses three explicit maturity levels:

| State | Meaning in the Studio |
| --- | --- |
| **Native** | A live repository entrypoint, mode, or proven workflow exists now. |
| **Adapter** | The underlying code or scripts exist, but need a stable Studio-facing wrapper or common contract. |
| **Roadmap** | A researched product capability that is useful but is not implemented in AI-CAE4ALL today. |

## Audit method and sources of truth

The repository audit started from the live root launcher and current filesystem,
not old diagrams:

- `python AI_CAE4ALL_main.py --help`
- `python AI_CAE4ALL_main.py --list-models`
- `cae_suite/registry.py`, `cae_suite/specs/`, and launcher probes
- `CONFIGURATION_REFERENCE.md`
- `REPOSITORY_OVERVIEW.md`
- `docs/methods/`
- `dataset/DATASET_FORMAT.md`
- `dataset/geometry_ingest/`
- `dataset/benchmarks/`
- `configs/`
- `Geometry_generation/`
- `inference/`
- real files under `output/`

This is important because the repository contains both stable runtime behavior
and benchmark-intent files with keys that are not implemented by every native
backend. The Studio must surface that distinction during validation.

## Complete live AI-CAE4ALL surface

### Launcher, config, and diagnostics

The unified launcher is itself a major product capability. The Studio currently
under-represents it.

| Capability | Live interface | Studio representation | State |
| --- | --- | --- | --- |
| Route a config to its native backend | `--config` | Config Source block + pipeline run | Native |
| Validate without launch | `--check` | Config Preflight block and Validate action | Native |
| Resolve the exact native command | `--dry-run` | Run preview in inspector | Native |
| Reject warnings as errors | `--strict` | Strict-validation toggle | Native |
| Explain resolved config | `--explain-config` | Resolved-config tab | Native |
| Show backend defaults | `--show-defaults` | Defaults diff in config editor | Native |
| List installed models and modes | `--list-models` | Models workspace health table | Native |
| Describe a model contract | `--describe MODEL` | Method card / contextual help | Native |
| Audit configs | `--audit-configs` | Config health view, with known scan-scope warning | Native |
| Skip selected probes | `--skip-*` | Advanced diagnostic escape hatches with warnings | Native |
| Override interpreter | `--python` and local TOML | System workspace and per-run override | Native |
| Machine-readable diagnostics | `--json-report` | Structured preflight report artifact | Native |

Preflight is layered and should be visible as such: parse, route, spec validation,
filesystem checks, environment checks, dataset checks, checkpoint checks, native
validation, then command construction. A red “invalid” badge without the layer
and diagnostic code would discard one of the suite's strongest capabilities.

### Registered routes and modes

The live launcher exposes ten route IDs: nine trainable model IDs and the
geometry-ingest utility.

| Route | Modes | Studio blocks and controls | State |
| --- | --- | --- | --- |
| `mlp` | train, inference | Tabular `X[S,N] → Y[S,M]` trainer, inference, normalization, `.pth`, and global-response evaluation | Native |
| `meshgraphnets` | train, inference | Trainer, inference, AR rollout, flat/HI-MGN/BSMS-GNN variant controls | Native |
| `meshgraphnets-v` | train, inference | Variational trainer, stochastic ensemble inference, histogram/distribution inspection | Native |
| `point_deeponet` | train, inference | Trainer and arbitrary-query inference | Native |
| `deeponet` | train, inference | Trainer and operator inference | Native |
| `fno` | train, inference | Trainer and spectral/grid inference | Native |
| `gino` | train, inference | Trainer and irregular-geometry operator inference | Native |
| `transolver` | train, inference | Trainer and Physics-Attention inference | Native |
| `sdfflow` | train, train_vae, train_fm, sample, reconstruct, interpolate | Separate stage, sample, reconstruct, and interpolate blocks | Native |
| `mlp` | train, inference | Parametric MLP trainer and N→M prediction blocks (tabular X/Y, no mesh) | Native |
| `geometry_ingest` | ingest, inspect | Geometry Ingest and Geometry Inspect blocks | Native |

HI-MGN and BSMS-GNN must be visible even though they are not separate launcher
IDs. They are meaningful MeshGraphNets variants:

- flat MGN: `use_multiscale False`
- HI-MGN: `use_multiscale True` with the hierarchical/Voronoi coarsener
- BSMS-GNN: `use_multiscale True` and `coarsening_type bfs`

A model chooser that shows only “MeshGraphNets” hides two documented
architectures. The preferred UI is one MeshGraphNets block with a prominent
variant selector, plus searchable palette aliases for HI-MGN and BSMS-GNN that
instantiate that block with the correct preset.

### Shared data and geometry preparation

| Capability | Repository evidence | Studio representation | State |
| --- | --- | --- | --- |
| Read the common mesh HDF5 contract | `dataset/DATASET_FORMAT.md`, native datasets | HDF5 Dataset source | Native |
| Inspect graph/point-cloud HDF5 | `geometry_ingest inspect`, launcher dataset probe | HDF5 Dataset inspector | Native |
| Import STEP, IGES, STL, PLY, OBJ | `dataset/geometry_ingest/` | CAD/mesh source | Native |
| Emit graph and/or point cloud | geometry-ingest `emit` modes | Geometry → HDF5 Dataset controls | Native |
| Surface or volume meshing | trimesh/gmsh reader and `volume` controls | Surface/Volume toggle | Native |
| FPS or random resampling | geometry-ingest controls | Resample block or ingest substep | Native |
| Mesh-size control | gmsh size limits | Ingest inspector controls | Native |
| Build SDFFlow SDF datasets | `Geometry_generation/build_dataset.py` | SDFFlow data action | Native |
| Repair and filter mesh input | SDFFlow builder flags | SDFFlow data action | Native |
| Append missing SDFFlow shapes | builder `--append_missing` | SDFFlow data action | Native |
| Benchmark-specific conversion | per-benchmark `prepare_*.py` tools | named Benchmark workflow | Adapter |
| Reusable split creation | native split seeds and benchmark split scripts | HDF5 Dataset split panel | Adapter |
| Dataset statistics and normalization | method dataset-stat modules/checkpoints | HDF5 Dataset statistics panel | Adapter |
| Schema health, feature counts, node types | launcher dataset probe | HDF5 Dataset automatic check | Native |

The HDF5 Dataset inspector must not be a static table. Each sample should be
clickable, with topology, mesh, points, fields, timesteps, conditions, units,
split membership, and provenance views. This matches the core interaction rule
for all blocks: every block publishes inspectable samples or artifacts.

#### Parametric HDF5 audit

The common HDF5 layout is channel-capable, but it is not automatically a
semantic design-parameter schema.

| Data path | Parametric status | Editable binding |
|---|---|---|
| `dataset/ex1.h5` | geometry and responses vary across 100 samples; no named global design vector | none by default |
| `dataset/ex2.h5` | geometry and 50-step responses vary; no named global design vector | none by default |
| `dataset/deepjeb.h5` | explicit `cond_names` and per-shape five-value `cond` | `bbox_x`, `bbox_y`, `bbox_z`, `volume`, `area` |
| Plasticity prepared data | static `die_profile_mm` condition in model state plus `die_profile` dataset | profile binding |
| Point-DeepONet paper data | five load/material/location values broadcast per point | fixed-length MLC vector |

The product consequence is a `ParameterBinding` registry plus immutable
`DatasetVariant` overlays. The Studio must not expose coordinates, targets,
part numbers, node types, or arbitrary channel indices as editable parameters
without a declared input role and broadcast rule.

### Training, inference, and scaling

| Capability | Applies to | Studio representation | State |
| --- | --- | --- | --- |
| Train and resume | all nine model IDs | Model trainer blocks + durable session | Native |
| Native inference | all nine model IDs | Family-aware inference block | Native |
| Autoregressive one-step training | mesh/operator families | AR-OT selector | Native |
| Rollout training | mesh/operator families | AR-RT selector and trajectory view | Native |
| DDP | all trainable families | Parallel strategy in trainer | Native |
| Model-split pipeline parallelism | MGN, MGN-V, FNO, GINO | Parallel strategy and stage diagram | Native |
| Transolver node sharding | Transolver | Parallel strategy and shard telemetry | Native |
| Activation checkpointing | supported backends | Memory/performance control | Native |
| AMP/BF16 and memory controls | backend-specific | Precision/resource controls | Native |
| Checkpoint-led inference architecture | all families | Read-only checkpoint-resolved config | Native |
| Best/latest/periodic checkpoints | training workflows | Checkpoint timeline | Native |
| Stochastic variational trajectories | MGN-V | Inference Run ensemble mode | Native |
| Conditional/unconditional geometry sampling | SDFFlow | CAD Generator generate mode | Native |
| VAE-only reconstruction | SDFFlow | CAD Generator reconstruct mode | Native |
| Latent interpolation | SDFFlow | CAD Generator blend mode | Native |
| Extrapolative condition targets | SDFFlow config | Sampler guardrail warning | Native |
| Resource/VRAM probes | benchmark and method scripts | ML Model resource panel | Adapter |
| Config sweeps and summaries | MGN-V/Transolver checked-in workflows | Sweep block + comparison view | Adapter |

Training is a durable job, not a synchronous node animation. A model block
creates or resumes a session; the pipeline may wait for a published checkpoint,
or downstream blocks may explicitly consume an existing checkpoint. Pause,
stop, and resume semantics must preserve recoverable checkpoints and distinguish
“stop scheduling new work” from “kill an active process.”

### Benchmarks and evaluation

The repository contains product-worthy benchmark workflows that are almost
entirely absent from the original prototype.

| Benchmark/workflow | Existing work | Required Studio access | State |
| --- | --- | --- | --- |
| Elasticity cross-model accuracy | prepare, preflight, train, infer, relative-L2 evaluation, smoke/full comparison | Benchmark campaign template with protocol lock and leaderboard | Adapter |
| Transient plasticity seven-model campaign | prepared/test splits, strict rollout evaluation, comparison, campaign scheduler, resource probes | Campaign, scheduler, result matrix, failure/retry views | Adapter |
| FNO Darcy paper validation | prepare, own-paper and opt-in paper protocols, train/infer/evaluate | Reproducibility template with qualification label | Adapter |
| GINO CarCFD paper validation | prepare, preflight, train, evaluate, source checks | Data provenance and validation-gate template | Adapter |
| DeepONet fractional-2D validation | data generation, training, evaluation workflow | Paper-validation template | Adapter |
| Point-DeepONet benchmark | selective download/preparation, validate/train/evaluate | Download/preparation and protocol template | Adapter |
| Transolver config sweep | run and summary utilities | Sweep definition + result table | Adapter |
| MGN-V distribution study | config generation, inference, histograms, CSV/PNG export | Distribution comparison template | Adapter |

Every benchmark result must carry a protocol label:

- smoke / pipeline verification
- repository baseline
- paper-intent
- faithful paper protocol
- measured result

The UI must never put a one-epoch smoke result in the same unlabeled leaderboard
as a full paper-equivalent run.

### Artifacts, visualization, and analysis

Real outputs include HDF5 fields, checkpoints, STL meshes, JSON/JSONL metadata,
CSV metrics, logs, plots, and text reports. The artifact system therefore needs
typed previews rather than a generic file browser.

Native or immediately adaptable views:

- surface, wireframe, point-cloud, and graph topology
- scalar field contours and vector fields
- time/rollout playback
- generated-geometry gallery
- SDFFlow reconstruction and interpolation pairs
- training/validation curves
- GPU memory and throughput
- variational histograms and generated-sample distributions
- per-sample error and aggregate relative-L2 metrics
- checkpoint metadata, architecture, normalization, EMA, and stage
- log and diagnostic-code view
- CSV/JSON/report preview
- side-by-side and difference comparison

The block viewer should borrow ParaView's data-flow mental model: sources produce
data, filters transform it, and sinks render or write it. Time is part of the
data contract, not an afterthought. See the
[ParaView User's Guide](https://docs.paraview.org/_/downloads/en/latest/pdf/).

### Standalone delivery

The `inference/` tree is a real, CPU-only, repository-independent product
surface:

- checkpoint-family auto-detection
- Point-DeepONet, DeepONet, FNO, GINO, Transolver, MGN, MGN-V, and SDFFlow
- HDF5 rollout output or SDFFlow STL output
- query chunking and rollout length
- SDFFlow samples, ODE steps, CFG scale, Marching Cubes resolution, seed, and
  conditions
- Python library usage
- PyInstaller one-folder executable
- documented parity status and family-specific limitations

Studio needs:

1. **Inference Bundle** block: checkpoint + input to portable output.
2. **Build Executable** block: create and validate the one-folder executable.
3. **Parity Report** artifact: expose family validation and known limitations.
4. **Bundle Health** view: show source snapshot and warn that re-vendoring is
   manual until `rebuild_bundle.py` exists.

### System, environments, and documentation

These are not “admin-only” concerns. They determine whether a pipeline can run:

- activated-venv and per-model interpreter resolution
- `ai_cae4all.local.toml` mapping
- CUDA visibility and import checks
- per-method native validation
- filesystem and output-writability checks
- checkpoint compatibility
- route installation health
- diagnostic codes and exit codes
- method docs and config-key documentation
- known runtime/schema mismatches

The System workspace should provide a one-click health scan and allow every
error to deep-link to the failing pipeline block and the relevant documentation.

## What the first prototype still missed

The first pipeline prototype successfully established the correct center of
gravity—movable, linkable typed blocks with click-through sample viewers—but its
functional coverage was incomplete.

| Area | Original prototype coverage | Required correction |
| --- | --- | --- |
| Model families | Eight trainer blocks | Add the now-live MLP as the ninth; keep MGN variant presets and mode-specific controls |
| Geometry ingest | One ingest block | Expose inspect, volume/surface, graph/point-cloud, resampling, dry-run |
| Config/preflight | Top-level Validate only | Add Config Source and Config Preflight blocks with layered report |
| Dataset lifecycle | HDF5 source + inspector | Add validate, split, statistics, normalization, SDF builder, benchmark preparation |
| Training | Session monitor | Add sweep, resource probe, parallel strategy, precision, resume provenance |
| Inference | Generic inference + SDFFlow sampler | Add batch/rollout, MGN-V ensemble, SDFFlow reconstruct/interpolate |
| Optimization | Surrogate ranking only | Replace it with geometry gates, evaluator contracts, constraints, Pareto/diversity selection, iterative search, and verification |
| Visualization | Dataset, field, candidate, training | Add geometry, distribution, error, time, comparison, logs/reports |
| Checkpoint governance | Source + session files | Add searchable registry, versions, lineage, validation and promotion |
| Benchmarks | Not represented | Add all checked-in benchmark protocols and status labels |
| Deployment | Generic export only | Add portable inference bundle, executable build, parity report |
| System | Not represented | Add environments, interpreters, CUDA, probes, diagnostics |
| Documentation | Not represented | Add contextual method/config documentation |
| Product trust | No maturity labels | Mark Native, Adapter, and Roadmap everywhere |

## Research: what comparable users and products prioritize

### End-to-end, no-code surrogate workflow

[Ansys SimAI](https://www.ansys.com/en-gb/products/ai/simai) emphasizes uploading
simulation data, selecting outputs, generating a model, and predicting new
designs in a browser. [Altair PhysicsAI](https://help.altair.com/simlab/help/en_us/topics/PhysicsAI/physicsAI.htm)
similarly exposes dataset creation, training/testing, and prediction on new
designs or CAD. The lesson is not “hide everything.” It is to make the common
path obvious while preserving expert controls in the block inspector.

Studio response:

- ship templates for Import → Validate → Train → Predict → Compare → Export
- infer sensible defaults from dataset/checkpoint metadata
- keep the resolved config and native command visible
- permit experts to switch from form controls to raw config

### Confidence, domain-of-validity, and solver validation

Ansys SimAI exposes confidence and warns about designs outside the training
distribution. NVIDIA PhysicsNeMo documents
[guardrails](https://docs.nvidia.com/physicsnemo/latest/user-guide/guardrails.html)
and [active learning](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/physicsnemo.active_learning.html).
Siemens describes accuracy-aware AI as a response to model drift.

Studio response:

- **Roadmap:** Confidence/OOD block with calibration evidence
- **Roadmap:** Guardrail gate before a result can be promoted or exported
- **Roadmap:** Solver Validation block that sends selected candidates to a
  high-fidelity solver
- **Roadmap:** Active Learning loop that adds validated samples and retrains
- **Native now:** show training-range and SDFFlow extrapolation warnings
- **Native now:** make checkpoint/data compatibility and known parity
  limitations impossible to miss

### Design exploration, DOE, and multi-objective trade spaces

[Ansys GeomAI](https://www.ansys.com/blog/introducing-ansys-geomai-software)
describes latent-geometry exploration followed by evaluation with SimAI,
high-fidelity solvers, or response surfaces, and closed-loop optimization driven
by optiSLang. [Ansys optiSLang](https://www.ansys.com/de-de/products/connect/ansys-optislang)
groups DOE/sensitivity, surrogate modeling, optimization, and
robustness/reliability rather than treating candidate ranking as the entire
workflow. [Siemens HEEDS](https://www.siemens.com/en-us/products/simcenter/integration-solutions/heeds/)
similarly emphasizes workflow automation, adaptive global/local search,
constraints, resource orchestration, and design-space exploration.

The data contract used by mature optimization tools is also consistent.
[OpenMDAO](https://openmdao.org/newdocs/versions/latest/features/core_features/adding_desvars_cons_objs/adding_design_variables.html)
makes design variables, lower/upper bounds, units, scaling, objectives, and
constraints explicit.
[optiSLang's design table](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/opti_ug/opti_ug_start_designs.html)
retains design IDs, status, parameters, responses, and criteria.
For expensive evaluators,
[BoTorch's constrained multi-objective workflow](https://botorch.org/docs/v0.16.1/tutorials/constrained_multi_objective_bo)
is a closed loop: choose a batch, observe objectives/constraints, update the
surrogate, and track the non-dominated front/hypervolume.

Studio response:

- keep one visible **Optimization** block because candidate evaluation and
  iterative search have their own evaluator, budget, lineage, and output
  contract;
- **Native evidence:** SDFFlow valid-zero-crossing/watertightness reports and
  conditional descriptor-adherence scores;
- **Adapter:** candidate conversion plus existing Saved ML Model inference;
- **Roadmap:** objectives/constraints, feasible/rejected sets, Pareto and
  diversity selection, DOE/evolutionary/Bayesian search, OOD/uncertainty gates,
  robustness/reliability, solver verification, and active learning;
- do not collapse multiple objectives into a weighted score unless the user
  explicitly supplies a utility or weights.

The generated-design path should become:

`parameters → CAD Generator → Optimization {geometry gates → evaluator(s) → constraints → Pareto/diversity → optional search} → engineer approval → solver verification → dataset update`

The fixed-batch evaluation table is the first implementation target. Each row
must keep candidate ID, geometry evidence, parameter values, objective and
constraint responses, feasibility/rejection reasons, evaluator/checkpoint
version, uncertainty/OOD status, evaluation cost, and provenance. This table is
useful before any optimizer exists and becomes the durable history consumed by
later search algorithms.

### Full configuration authoring

The repository audit found 315 unique config keys and 587 key entries across
the seven live specs. Per selected model, the Studio must expose:

- MLP: 33 accepted keys;
- MeshGraphNets: 114;
- MeshGraphNets-V: 105;
- each Neural Operator route: 138 shared-spec keys, with other-variant and
  removed keys clearly marked;
- Transolver: 70;
- SDFFlow: 111.

The existing spec layer knows accepted, required, recommended, and default
keys, path rules, modes, and validators. It does **not** yet publish a
machine-readable field type, label, section, choices, units, bounds, help,
visibility conditions, or active/inactive/removed status. Those metadata must
move out of validator-local constants into `ConfigFieldSpec` and a versioned
config-schema API.

The Studio response is a large model-owned configuration workspace with every
accepted key, search and section filters, closed-choice dropdowns only where
the backend genuinely defines a closed set, manual inputs everywhere else,
checked-example/smoke/low-VRAM/family presets, explicit preset diffs, and
two-way `.txt` import/paste/export. A generic “high accuracy” preset is not
obvious or portable enough to ship without model- and dataset-specific
validation.

### Experiment tracking, sweeps, lineage, and model governance

[MLflow Tracking](https://www.mlflow.org/docs/latest/ml/tracking) organizes work
as experiments and runs with parameters, metrics, and artifacts.
[W&B Artifacts](https://docs.wandb.ai/models/artifacts) versions data and model
inputs/outputs, while the [W&B Registry](https://docs.wandb.ai/models/registry)
adds lifecycle management, lineage, access control, and downstream automation.
W&B sweep views add parameter-importance and parallel-coordinate analysis.

Studio response:

- durable experiment/run identity separate from a canvas node
- dataset, config, code, environment, checkpoint, and output lineage
- parent/child runs for sweeps and benchmark campaigns
- checkpoint aliases such as `best`, `latest`, `candidate`, `validated`
- explicit promote/approve action with evidence
- artifact versions and immutable hashes
- search and compare across pipelines, not just the current canvas

### Data exploration, validation, root cause, and reporting

[Monolith](https://www.monolithai.com/products/core-platform) centers data
exploration/transformation, automatic model evaluation, parameter influence,
test-plan optimization, data validation, root-cause analysis, and collaboration.
Anecdotal engineering discussions repeatedly highlight result validity,
post-processing, vague errors, restartability, automation, and report writing as
larger pain points than pressing “run.” See the
[CFD result interpretation discussion](https://www.reddit.com/r/CFD/comments/1iq8cfe)
and [CFD post-processing discussion](https://www.reddit.com/r/CFD/comments/1ih3dda).

Studio response:

- comparison and error views are first-class blocks
- every diagnostic preserves layer, code, context, and suggested action
- reports are generated from selected, traceable artifacts
- templates and restartable sessions reduce repetitive setup
- validation remains engineer-visible; the UI does not present AI output as
  truth

### Extensibility, custom code, deployment, and collaboration

PhysicsNeMo supports built-in and custom models, physics-informed losses,
distributed training, logging integrations, and deployment including
[ONNX utilities](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo/api/physicsnemo.deploy.html).
SimScale emphasizes shared projects, workflow templates, versioning, review,
open integration, and governed execution. Neural Concept supports SaaS and
private-cloud patterns.

Studio response:

- **Adapter:** Custom Command/Python block executed in a declared environment
- **Roadmap:** plugin/block SDK with versioned contracts
- **Roadmap:** ONNX/API/container deployment targets
- **Roadmap:** comments, approvals, permissions, and shared workspaces
- **Native now:** portable CPU bundle and executable
- **Native now:** per-method interpreters and isolated native processes

## Recommended Studio information architecture

### Top navigation

Keep **Pipeline** selected by default. The supporting workspaces should be:

1. **Pipeline** — graph authoring, validation, execution, and block inspection
2. **Data** — datasets, samples, schemas, splits, preparation, and lineage
3. **Experiments** — sessions, sweeps, comparisons, resources, and logs
4. **Models** — methods, variants, checkpoints, versions, and model cards
5. **Benchmarks** — campaigns, protocols, results, and reports
6. **Artifacts** — all typed outputs and provenance
7. **Deploy** — inference bundle, executable, parity, and future endpoints
8. **System** — environments, GPUs, interpreters, configs, diagnostics, docs

On narrower screens, keep Pipeline/Data/Experiments/Models visible and put the
rest in a Studio switcher. Do not turn the canvas into a cramped row of tabs.

### Complete block catalog

The first complete catalog exposed too many implementation utilities as graph
nodes. User review established a stronger rule: make a separate block only when
it is a meaningful reusable source, transformation, model, run, evaluation, or
destination. Put inspection and safeguards inside the block that owns the
result.

The revised default palette is:

#### Sources and preparation

- CAD
- HDF5 Dataset
- Design Parameters
- Saved ML Model
- Geometry → HDF5 Dataset

`HDF5 Dataset` embeds samples, geometry, topology, fields, timesteps, splits,
statistics, normalization, and verified parameter bindings. Design Parameters
creates a non-destructive overlay; it does not mutate the source HDF5.

#### Models

- Simple MLP — live tabular parameter-to-global-response estimator
- MeshGraphNets — Flat / HI-MGN / BSMS-GNN are variants of one entry
- MeshGraphNets-V
- Point-DeepONet
- DeepONet
- FNO
- GINO
- Transolver3 — display alias; backend ID stays `transolver`
- SDFFlow

Every model embeds:

- simple and advanced ML configuration;
- model-specific examples and presets;
- data/target mapping;
- seeded split and statistics;
- automatic layered preflight;
- GPU, precision, VRAM, throughput, and feasibility;
- training/validation curves and samples;
- checkpoints, resume state, and `.pth` download.

#### Run, evaluate, and output

- Inference Run — single, batch, rollout, and MGN-V ensemble modes
- CAD Generator — conditional generation, reconstruction, blending, gallery,
  and geometric sanity checks
- Optimization — geometry feasibility, linked physics evaluators,
  objectives/constraints, Pareto/diversity selection, search, and verification
- Hyperparameter Sweep
- Evaluate Predictions — compatible metrics, per-sample/channel/timestep
  breakdowns, error maps, distributions, and CSV/JSON/HTML
- Compare Models — accuracy, latency, throughput, VRAM, model size,
  uncertainty, and synchronized sample comparison
- API Deployment
- Export Results

The executable path is a top-right **Build .exe** action, because packaging is
an application-level handoff rather than a normal dataflow transformation.

#### Workspace-only workflows

- Benchmark protocols remain named, locked workflows in **Benchmarks**.
- Run outputs remain immutable typed objects in **Artifacts**.
- “Use in pipeline” creates the concrete compatible source rather than a
  generic Previous Artifact block.
- All repository Markdown remains searchable in **Docs**.

#### Why the removed utilities are still covered

| Removed standalone block | Embedded owner |
|---|---|
| Config Source / Config Preflight | ML Model, Inference Run, or deployment action |
| Dataset Split / Statistics / Inspector / Field Viewer / Geometry Viewer | HDF5 Dataset |
| Training Monitor / Resource Probe / Logs | ML Model session |
| Batch/Rollout / Variational Ensemble / Distribution Viewer | Inference Run |
| SDFFlow Reconstruct / Interpolate / Candidate Gallery / Validity Filter | CAD Generator |
| Error Map | Evaluate Predictions |
| Surrogate Rank | replaced by visible Optimization block |
| Benchmark Campaign | Benchmarks workspace |
| Inference Bundle / Build Executable | top-right Build `.exe` workflow |

The geometric “Validity Filter” means manifoldness, connectivity, bounds, and
minimum-feature checks. It must be labeled **Geometry Checks** and must not be
presented as proof that a design is physically valid. The former “Surrogate
Rank” was too narrow: geometry evidence, one or more physics evaluators,
constraints, Pareto selection, iterative search, and solver verification form
a meaningful independent **Optimization** pipeline stage.

## Core production contracts

### BlockDefinition

Each block definition needs:

- stable type and semantic version
- maturity: native, adapter, roadmap
- backend route and mode, if applicable
- typed input/output ports
- configuration schema and defaults
- preflight adapter
- command builder or job adapter
- artifact renderer capabilities
- documentation links
- environment requirements
- cancellation/restart policy

### ArtifactManifest

Every block execution should publish:

- artifact ID, type, version, and immutable hash
- producing run, pipeline version, node ID, and block version
- source artifacts and lineage edges
- path/URI plus lazy preview payload
- sample count and sample IDs
- fields, units, shape, timesteps, conditions, and split metadata
- checkpoint architecture/stage/normalization/EMA metadata
- renderer capabilities
- warnings, qualifications, and validation state

### RunRecord

Each run needs:

- durable run ID and parent campaign/sweep ID
- pipeline snapshot and resolved config
- exact native command
- model route, mode, interpreter, environment, and hardware
- Git/source snapshot where available
- lifecycle state and timestamps
- logs, diagnostics, metrics, resource telemetry, and checkpoints
- input/output artifact IDs
- cancellation reason and restart checkpoint

### ValidationReport

Preserve the launcher's layered semantics:

- parse
- route
- spec
- filesystem
- environment
- dataset
- checkpoint
- native
- command

Each issue carries severity, diagnostic code, message, affected key/path/port,
and a link to the relevant inspector control or documentation.

## Delivery sequence

### Phase 1 — honest completeness

- expose every live route and mode as a block or block mode
- add the exhaustive model-owned config workspace and automatic layered
  preflight diagnostics
- add `ConfigFieldSpec` metadata and the versioned config-schema API
- add MGN variant presets
- add SDFFlow reconstruct/interpolate and MGN-V ensemble
- add dataset validation and SDF dataset building
- add Data, Models, Artifacts, Deploy, and System workspaces
- connect contextual method/config docs

### Phase 2 — repository workflows

- wrap benchmark preparation/evaluation/campaign scripts
- wrap MGN-V and Transolver sweeps
- wrap resource probes
- add cross-run comparison and report generation
- add artifact/run lineage and checkpoint registry
- add standalone bundle/executable jobs

### Phase 3 — engineering decision loop

- fixed-batch generated-design evaluation, objectives/constraints,
  Pareto/diversity selection, then design-space optimization
- solver connectors and validation gates
- confidence/OOD and active learning
- plugin/custom-code block SDK
- collaboration, review, permissions, and approvals
- ONNX/API/container deployment

## Acceptance criteria for “all functionality is accessible”

The claim is true only when:

1. Every route returned by `--list-models` is visible and runnable.
2. Every valid route mode is selectable without editing source code.
3. HI-MGN and BSMS-GNN presets are discoverable.
4. Every launcher validation/introspection action is accessible.
5. Shared geometry/HDF5 preparation and inspection are accessible.
6. AR-OT/AR-RT, parallelism, checkpointing, precision, and resume controls are
   available where their backends support them.
7. SDFFlow train stages, sampling, reconstruction, interpolation, conditioning,
   and extrapolation warnings are accessible.
8. MGN-V stochastic inference and distribution analysis are accessible.
9. Every checked-in benchmark has a discoverable protocol and correctly labeled
   status.
10. Sweeps, resource probes, and result summaries are accessible through stable
    adapters.
11. Every block output has a clickable sample/artifact view.
12. Runs preserve configs, commands, environments, logs, metrics, checkpoints,
    and lineage.
13. The standalone inference bundle and executable build are accessible.
14. Environment, interpreter, CUDA, dependency, filesystem, dataset, and
    checkpoint health are visible.
15. Method docs, config reference, known limitations, and diagnostic help are
    available in context.
16. Native, Adapter, and Roadmap features are never visually conflated.

The Studio can then truthfully be described as the all-in-one interface for
AI-CAE4ALL—not because it hides the engineering, but because it makes the entire
engineering lifecycle composable, inspectable, repeatable, and governable.
