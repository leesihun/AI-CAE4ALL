# AI-CAE4ALL

**Train simulation models. Generate geometry. Explore design trade-offs.**

A local machine-learning workspace for computer-aided engineering. Build visual
pipelines in **Studio** to connect datasets, native model runtimes, training, and
interactive result viewers. Use the same methods from text configurations with
the command-line launcher.

[Videos](#studio-walkthroughs) | [Features](#features) | [Models](#models) | [Get started](#get-started) | [Documentation](#documentation)

## Studio walkthroughs

### DeepJEB design workflow

Follow a completed SDFFlow + HI-MGN pipeline: inspect a generated bracket, rotate
its mesh, switch predicted stress and displacement fields, and select a design
from the Pareto plot. **2:39 · English captions · continuous recording**

https://github.com/user-attachments/assets/76410429-83e7-4492-aa9f-42115051c48e

### From an empty canvas to training

Inspect a dataset, add SDFFlow, configure a new run, connect the blocks, and start
training. Browse the model catalog and explore results from existing trained
models. **2:39 · English captions**

https://github.com/user-attachments/assets/72b5dfa4-2f81-4ee2-a1af-0ce5f0ca6796

The training recording has one labelled transition after training starts, then
uses previously trained checkpoints. The DeepJEB example screens 100 generated
candidates using HI-MGN predictions. [Recording details and subtitles](docs/videos/README.md).

## Features

Studio provides the visual workflow; the launcher and native method tools provide
the same model operations for scripts and campaigns. The catalog below describes
implemented functions. Required dependencies, compatible data, and checkpoints
must be available in the selected runtime.

| Category | What it covers |
| --- | --- |
| [Pipeline editing](#pipeline-editing) | Blocks, connections, templates, layout, save/import/export |
| [Model configuration and training](#model-configuration-and-training) | Model catalog, settings, presets, stages, validation, resume |
| [Data and parameters](#data-and-parameters) | File inputs, HDF5 inspection, parameter spreadsheets |
| [Geometry and dataset preparation](#geometry-and-dataset-preparation) | CAD meshing, graph/point-cloud output, SDF dataset tools |
| [Geometry and field viewers](#geometry-and-field-viewers) | 3D interaction, channels, timesteps, predictions and errors |
| [Runs and training metrics](#runs-and-training-metrics) | Native jobs, logs, cancellation, recovery, learning curves |
| [Evaluation and model comparison](#evaluation-and-model-comparison) | Field metrics, sample matching, curve/CSV comparisons |
| [Geometry generation and design search](#geometry-generation-and-design-search) | SDFFlow sampling, conditions, reconstruction, FEA/surrogate search |
| [Constraints and Pareto selection](#constraints-and-pareto-selection) | Saved candidate tables, feasibility, diversity, design selection |
| [Benchmark campaigns](#benchmark-campaigns) | Paired configs, audits, sequential campaigns, scoring |
| [Artifacts and export](#artifacts-and-export) | Results, checkpoints, figures, reports, file/ZIP handoff |
| [Portable inference and deployment](#portable-inference-and-deployment) | CPU bundle, Python API, local job API, Windows executable |
| [System and documentation](#system-and-documentation) | Runtime health, GPUs, config audit, LLM setup, local guides |
| [Command-line tools](#use-the-command-line) | Routing, preflight, command preview, JSON diagnostics |

### Pipeline editing

| Feature | What you can do |
| --- | --- |
| Categorized block library | Search blocks by name, description, or model ID; click or drag to add them. |
| Source blocks | Add CAD, HDF5 Dataset, Design Parameters, and Saved ML Model inputs. |
| Model blocks | Add any of the ten learning-model blocks listed under [Models](#models). |
| Execution and analysis blocks | Add Geometry → HDF5 Dataset, Inference Run, CAD Generator, Optimization, Evaluate Predictions, Train Metrics, Compare Models, Export Results, and API Deployment. |
| Pipeline templates | Start from mesh-field, fixed-geometry, tabular, generative-geometry, or geometry-ingest recipes, or an empty canvas. |
| Block inspection | Open a block's settings, actual data/architecture facts, connected inputs, required inputs, and available actions. |
| Block movement | Drag a block; newly added blocks are placed in a visible free position. |
| Typed connections | Click ports or drag wires; valid targets are highlighted and incompatible links or dependency cycles are rejected. |
| Input replacement and fan-in | Replace a single-input connection or combine sources on ports that support multiple inputs. |
| Delete blocks and links | Remove the selected block or wire; deleting a block also removes its connections. |
| Duplicate blocks | Copy configuration into a new block with separate output paths and cleared prior-run results. |
| Undo | Restore graph, configuration, template, and import changes. |
| Auto layout | Arrange blocks by dependencies and fit the resulting graph. |
| Pan, zoom, and Fit | Pan the canvas, zoom around the pointer, use zoom buttons, or fit the whole graph. |
| Panel visibility | Collapse or reopen the library and inspector for more canvas space. |
| Naming and persistence | Rename pipelines; use Save, Ctrl+S, and browser autosave. Reload restores the graph, configuration, viewport, and result references. |
| Pipeline JSON export | Download a versioned graph document with its settings, positions, and references. |
| Pipeline JSON import | Load a graph with node, port, connection, and cycle validation; obsolete ports are reported. |
| Onboarding and shortcuts | Use the first-visit welcome guide and top-bar shortcut reference; search, Fit, layout, validate, run, save, undo, delete, and close panels from the keyboard. |

Pipeline saves are local to the browser. JSON documents reference data and
results by path; they do not embed datasets or model weights. The library has
**23 block types**, including ten model blocks and the geometry-preparation utility.

### Model configuration and training

| Feature | What you can do |
| --- | --- |
| Live model catalog | Browse each route's repository, entrypoint, operations, data kind, accepted keys, and discovery status. |
| Model detail | Inspect all accepted keys and current values, open its canvas block, and find related Studio jobs. |
| Checked-in examples | Browse and load real configuration examples for the selected method. |
| Mode-aware inspector | Change operation and edit relevant dataset, checkpoint, architecture, epoch, batch-size, and learning-rate settings. |
| Full configuration | Inspect the complete key contract and edit active settings in Required, Data & output, Architecture, Training, Resources & runtime, Inference & evaluation, Optimization, Advanced, and Inactive / rejected sections. |
| Key search and filters | Search settings, filter explicitly set keys, and reveal inactive or rejected keys. |
| Setting help and defaults | Inspect descriptions, allowed choices, required/defaulted/runtime-owned status, and native defaults. |
| Graph-derived configuration | Fill data, checkpoint, parameter, and output settings from connected blocks; each automatic value names its source. |
| Manual overrides | Edit an automatic value to take control; clear it to follow the graph again. Explicit held-out input paths are preserved. |
| Checkpoint inputs | Reuse saved models or compatible VAE/FM/LC stages; inspect family, stage, architecture, and normalization compatibility. |
| Flat-text configuration | Import, paste, edit, and export native `.txt` settings; form and raw text stay synchronized. |
| Configuration explanation | Inspect configured/defaulted/inactive/checkpoint-owned values and the selected execution context. |
| Configuration save | Keep settings on the block and write a runtime config copy when the local backend is connected. |
| Architecture and resource recipes | Apply flat MGN, HI-MGN, BSMS, stage, low-VRAM, or performance recipes with a change preview. Availability depends on the method. |
| Training controls | Configure architecture, feature selection/loss weights, normalization, split seeds, epoch/batch/LR schedules, validation cadence, and output paths where supported. |
| Efficiency controls | Select supported AMP, EMA, activation checkpointing, compilation, or query chunking settings. |
| Multi-GPU training | Use each runtime's supported DDP, model/node partitioning, or FSDP modes; the accepted modes differ by family. |
| Multi-stage training | Run SDFFlow VAE → FM, cHI-MGNflow AE → latent flow, or SimulGenVAE VAE → latent conditioner together or as separate stages. |
| Training resume | Supported runtimes restore optimizer, scheduler, RNG, and stage state from resume sidecars in single-process training. MLP and multiprocess resume are outside this support. |
| Smoke runs | Apply short-run settings; the SimulGenVAE smoke recipe can create a real fixed-geometry CPU fixture. |
| Preflight and repair | Validate the effective config in the selected method's interpreter; follow diagnostics to the failing block or field. |
| Optional LLM assistance | Send a block's complete config and an instruction to your configured service, preview its suggestion, and apply or discard it before validation. |

Recipe names describe settings, not measured speed or accuracy. Smoke runs check
execution. LLM assistance asks before sending the configuration and before
applying a suggestion; it does not start training automatically.

### Data and parameters

| Feature | What you can do |
| --- | --- |
| Data catalog | Search repository dataset/parameter paths, inspect file type, size, and modification time, and page through results. |
| Source-file picker | Select compatible local CAD, HDF5, parameter, or checkpoint files for a source block. |
| Local file upload | Copy a chosen file into the local workspace and connect its returned path without modifying the original. |
| HDF5 structure inspector | Inspect root attributes, groups, datasets, shapes, dtypes, feature/condition names, splits, and sample IDs without loading the full numeric file. |
| Dataset facts | Show actual sample/node/timestep/channel counts or SDF/tabular dimensions on the source card. |
| Text and figure preview | Open supported CSV, TSV, JSON, text, log, Markdown, PNG, and JPEG files. |
| Use data in a pipeline | Create a typed source block from a catalog item or the currently viewed artifact. |
| Parameter spreadsheet | Edit named input/output columns and per-sample values; add, remove, or rename columns and select a row. |
| MLP metadata binding | Use input/output column counts and names to configure the tabular model; its numeric training data remain in the X/Y HDF5. |
| SimulGen condition binding | Set compatible CSV/image parameter paths, or use HDF5 dataset-owned condition rows where supported. |
| Explicit generation conditions | Turn one selected row of finite numeric inputs into ordered condition names/values for CAD generation. |

Spreadsheet edits are stored on the pipeline block; they do not rewrite an HDF5
dataset or schedule a DOE sweep. The [dataset reference](docs/reference/DATASET_FORMAT.md)
describes the mesh, SDF, and scalar-table contracts.

### Geometry and dataset preparation

| Feature | What you can do |
| --- | --- |
| Geometry inspection | Inspect a CAD/surface file or directory for mesh counts, bounds, dimensions, and watertightness before conversion. |
| CAD/surface readers | Read STEP/STP, IGES/IGS, BREP, STL, PLY, OBJ, and OFF through supported Trimesh/Gmsh readers. |
| Surface and volume meshing | Choose surface or tetrahedral-volume output and supported minimum/maximum mesh sizes. |
| Graph output | Weld surface vertices and derive graph connectivity from triangle/tetrahedron cells. |
| Point-cloud output | Set point count, seeded FPS/random resampling, and graph, point-cloud, or combined HDF5 output. |
| Batch conversion | Convert collected geometry files, apply an input limit, report failed inputs, and keep valid results. |
| Geometry previews | Generate preview strips; create a small example cube STL from the CAD source inspector. |
| SDF dataset builder (native tools) | Sample surface points, normals, and signed-distance queries from meshes or analytic shapes using SDFFlow's native builder. |
| SDF preparation controls (native tools) | Configure repair, simplification, sharp-edge/near-surface sampling, workers, seeded sampling, append-missing behavior, and held-out/group-aware splits. |
| Condition sidecars (native tools) | Add existing DeepJEB response labels or supported class/hole descriptors to the geometry condition vocabulary. |

GeometryIngest creates geometry and zero-filled placeholder field rows. Solved
response labels must come from your simulation/data pipeline. Reader dependencies
and geometry units must match your setup. See [GeometryIngest](methods/GeometryIngest/)
and [SDFFlow](methods/SDFFlow/) for the native preparation tools.

### Geometry and field viewers

| Feature | What you can do |
| --- | --- |
| Unified artifact opening | Inspect compatible HDF5 datasets, native predictions/rollouts, generated design collections, CAD/mesh files, and geometry folders. |
| Sample search | Find samples by real ID, label, dataset name, or shape and open their actual data. |
| Additional data picker | Add another compatible local or uploaded artifact to the viewer. |
| Mesh mode | Display available topology and coordinates. |
| Points mode | Inspect the returned point cloud when the artifact supports it. |
| Field mode | Color the geometry with the selected sample's actual scalar values. |
| Camera interaction | Rotate, pan, zoom, reset, and use keyboard camera controls without changing source coordinates. |
| Channel selection | Choose a named field from the dropdown or clickable channel list while keeping the camera position. |
| Timestep selection | Select a frame numerically or with the slider. |
| Time playback | Play/pause successive timesteps for transient samples. |
| Prediction/truth/error channels | Inspect available prediction, matching ground truth, and signed error channels for paired results. |
| Field statistics | Read min/max/mean, color legend, sample parameters, timestep, and mesh counts; constant fields are labelled. |
| Provenance and reduction | See the source path, sample, reader, total/returned points, and any preview reduction. |
| Generated design fields | Open matching screening `designs.h5` geometry and structural fields; surface-only artifacts remain viewable without fields. |
| Evaluation/comparison handoff | Send compatible viewed results to prediction evaluation or CSV comparison. |
| Sample JSON download | Export the loaded preview's coordinates, values, topology, statistics, metadata, channel, and timestep. |
| Copy sample identifier | Copy the artifact path and sample ID for a precise local reference. |
| Use viewed data in a pipeline | Insert/reuse a compatible source block and fill missing feature/condition names downstream. |

The viewer uses WebGL with a canvas fallback. A bounded preview and its JSON
download may contain fewer points than the original artifact; original-file
handoff is available through [Artifacts and export](#artifacts-and-export).

### Runs and training metrics

| Feature | What you can do |
| --- | --- |
| Run a block | Launch a selected executable model, inference, generation, or analysis block. |
| Run a pipeline | Execute native and analysis steps in dependency order, resolving artifacts produced upstream. |
| Concurrent jobs | Submit independent pipelines while other jobs run and inspect each job separately. |
| Launch-time validation | Recheck each exact saved native config immediately before its process starts. |
| Runs catalog | Browse Studio jobs with creation time, state, current/total steps, current step, and exit code. |
| Native process logs | Follow actual stdout/stderr and orchestration messages in the runtime drawer. |
| Stop a job | Cancel preflight/launch or terminate the active native process tree. |
| Reopen a run's pipeline | Load its saved launch graph as a separate canvas copy with that run's results and status. |
| Browser reconnection | Close/reopen the page while the backend keeps working; reconcile runs that completed while it was closed. |
| Server recovery | Recover saved jobs/logs and adopt a surviving process so it can be monitored or stopped; unrecoverable continuation is marked interrupted. |
| Metric discovery | Parse numerical training observations from full persisted logs, separating pipeline steps and training stages. |
| Metric selection | Choose a job and plot all, none, or selected discovered series. |
| Curve smoothing | Smooth the display while retaining raw observations for statistics and export. |
| Logarithmic plotting | Toggle log scale for compatible positive series and inspect non-finite/divergent observations. |
| Raw metric CSV export | Download visible metric series, refresh observations, or open the source log. |

The Runs workspace tracks Studio-launched jobs. The log drawer shows a bounded
tail with a truncation notice; the complete log stays on disk. Process recovery
and browser reconnection are separate from checkpoint-based training resume.

### Evaluation and model comparison

| Feature | What you can do |
| --- | --- |
| Prediction/truth selection | Compare a prediction HDF5 or directory of per-sample files against a real ground-truth HDF5. |
| Supported result contracts | Score compatible mesh-state, scalar-table/operator arrays, and native embedded prediction/truth results. |
| Scoring-contract inspection | Inspect sample IDs, shapes, array layouts, and field correspondence before scoring. |
| Field mapping | Pair declared field names; explicitly select and confirm positional mappings when needed. |
| Physical-field protection | Exclude copied reference-coordinate and node-type rows from prediction scores. |
| Error metrics | Compute relative L2, MAE, RMSE, maximum absolute error, and R². |
| Metric aggregation | Report mean, median, p95, minimum, and maximum; retain per-field and pooled L2/R² values. |
| Evaluation reports | Write a JSON report and per-sample metric CSV for later comparison. |
| Connected training comparison | Overlay graph-connected training histories using common stage-aware metric keys. |
| Training observation ranking | Rank the last raw common training metric in a selected min/max direction. |
| Evaluation CSV selection | Compare up to 12 selected CSVs or use graph-connected evaluation outputs. |
| CSV mean ranking | Choose a numeric metric, direction, and optional grouping column; rank finite-value means with counts and ranges. |
| Comparison qualification | Show warnings for different array contracts, disjoint sample IDs, or partial overlap. |
| Saved comparison report | Save the comparison result as JSON with its source evidence. |

Declared sample IDs are checked before scoring; supported unlabeled tables can
use positional matching. Relative L2 and R² use per-field macro-averages; pooled
values remain available. Select the same held-out cases, physical quantities, units,
and preprocessing when comparing methods. Training losses alone do not establish
a held-out accuracy ranking. See [evaluation and comparison](docs/GUI.md#evaluation).

### Geometry generation and design search

| Feature | What you can do |
| --- | --- |
| SDF-VAE and flow training | Train geometry encoding/reconstruction and latent flow matching sequentially as stages or independently. |
| Architecture selection | Select supported SDF decoder and latent flow architectures through the native configuration. |
| Unconditional generation | Sample new shapes from a compatible trained generator. |
| Conditional generation | Supply supported target descriptors, partial conditions, and classifier-free guidance settings. |
| Sampling controls | Set seed, sample count, ODE steps, latent/condition controls, and mesh reconstruction resolution. |
| Geometry reconstruction | Reconstruct an existing geometry from its learned representation. |
| Interpolation and sweeps | Interpolate latent/condition endpoints or use fixed-noise condition sweeps and guarded extrapolation through native tools. |
| Descriptor-guided generation (native tools) | Use supported descriptor agreement ranking, calibrated guidance, and correction tools. |
| Held-out geometry evaluation | Measure surface distance/Chamfer, SDF/sign agreement, geometric validity, watertightness, and components; optionally evaluate latent refinement. |
| Conditional method benchmarks (native tools) | Compare supported plain/rejection/guidance/correction methods with matched base noise and error/validity/NFE/time reports. |
| Fresh population screening | Use CAD Generator `optimize` with `opt_budget = 0` to generate and analyze a baseline candidate population. |
| Iterative structural search | Use a positive budget for CMA-ES search over supported generator latent/condition variables. |
| FEA analysis | Generate candidate structural fields/scores using the implemented tet4 solver path. |
| HI-MGN surrogate analysis | Evaluate compatible generated meshes with a selected HI-MGN checkpoint/configuration, including batched populations. |
| Structural design settings | Set supported load cases, material values, stress/displacement/vertical-deflection limits, baseline size, and search budget. |
| Saved candidate evidence | Persist candidate tables, meshes/fields in `designs.h5`, summaries, and reports. |
| Optional solver re-check | Re-solve selected shapes for compatible surrogate boundary-condition conventions. |

SDFFlow's [native guide](methods/SDFFlow/) describes advanced generation and
evaluation tools. The DeepJEB video shows 100-candidate HI-MGN screening. Its
ex13 vertical-load surrogate has its own scale/material/load contract and cannot
use the older pad/lug FEA re-check; independent verification requires the matching
solver workflow. The generator's surrogate input currently accepts the compatible
MeshGraphNets/HI-MGN family.

### Constraints and Pareto selection

| Feature | What you can do |
| --- | --- |
| Candidate CSV picker | Choose an actual saved output table, grouped by candidate, summary, selected-design, or evaluation role. |
| Numeric schema inspection | Inspect candidate IDs/counts and choose objectives from available numeric columns. |
| Objective directions | Minimize or maximize each selected objective independently. |
| Hard constraints | Enter semicolon-separated numeric `<`, `<=`, `>`, or `>=` thresholds. |
| Constraint validation | Check syntax and column names as you type; invalid settings disable evaluation. |
| Feasibility analysis | Count usable, skipped, feasible, and rejected rows; explain an all-infeasible population with per-constraint counts and closest observed values. |
| Pareto front | Compute the feasible nondominated set from the selected objectives. |
| Diversity-oriented top-k | Select up to 200 Pareto candidates using crowding distance. |
| Interactive plot | Choose plot axes and inspect infeasible, feasible, Pareto, and selected designs with constraint limits. |
| Open a selected design | Click a plotted point or View design to open its matching saved geometry and fields. |
| Result export and restoration | Write selected rows in the source CSV's columns, save report JSON, and reopen a block's last report. |

This workspace ranks the values already in the CSV. Fresh generation, surrogate
inference, and solver execution happen in CAD Generator. Plot display is bounded
to 5,000 points; selection uses the complete input table.

### Benchmark campaigns

| Feature | What you can do |
| --- | --- |
| Manifest roster | Browse every manifest-listed dataset/method train/infer configuration, including missing-file state. |
| Individual config preflight | Validate a selected campaign configuration and inspect its diagnostics. |
| Load into Studio | Open a supported campaign configuration in a model block for editing and execution. |
| Config generation | Generate the paired dataset/model config matrix and check generated-source consistency with native campaign tools. |
| Dataset/topology audits | Run structural/native-parser and optional data audits for campaign inputs. |
| Sequential campaign execution | Use machine-specific CLI plans, dry runs, GPU gates, resumable bookkeeping, and per-stage provenance. |
| Produced-output scoring | Score/rank completed outputs and generate spread reports, CSVs, text summaries, and figures. |
| Provenance qualification | Connect scoring to source/config artifacts and expose missing or invalid evidence. |

The Studio Benchmarks tab browses, validates, and loads configs. Campaign
execution/scoring are separate [CLI tools](configs/campaigns/dataset_matrix/);
a listed or validated config is not a completed benchmark result.

### Artifacts and export

| Feature | What you can do |
| --- | --- |
| Result catalog | Search recognized result/checkpoint files by path and inspect type, size, and modification time. |
| Catalog pagination | Show additional result batches and visible catalog truncation. |
| Checkpoint inspection | Inspect supported saved-model metadata and portable-family support. |
| HDF5 artifact inspection | Open structure and metadata without reading the full numeric result. |
| Reports and figures | Read supported CSV/TSV/JSON/text/log/Markdown reports and view PNG/JPEG figures. |
| Artifact-to-pipeline handoff | Add a typed dataset, checkpoint, or geometry source for a selected result. |
| File export | Copy an existing artifact with an export label while preserving its source. |
| Directory export | Archive a multi-file result directory as ZIP. |
| Export download | Download the copied file or archive from the local Studio export directory. |

The general artifact catalog searches configured result roots and reports its
750-file cap. Export copies/archives artifacts; it does not convert every CAD
format or package every model family.

### Portable inference and deployment

| Feature | What you can do |
| --- | --- |
| Standalone CPU bundle | Copy `inference/` and run supported checkpoints without the main method repositories. |
| Checkpoint family detection | Detect supported families from checkpoint metadata and reject unsupported/unknown checkpoints with guidance. |
| Portable field inference | Run compatible operator, Transolver, MGN/HI-MGN, and MGN-V checkpoints on HDF5 input. |
| Portable geometry generation | Sample SDFFlow geometry from an FM checkpoint with embedded/available VAE state. |
| Portable CLI controls | Set input/output, timesteps, query chunking, or relevant geometry sampling/condition options. |
| Python library API | Call `cae_infer.infer()` and `detect_family()` from the copied bundle. |
| Combined generator checkpoint | Export a self-contained SDFFlow VAE/FM checkpoint or merge compatible legacy stage checkpoints. |
| Studio CPU inference job | Select a checkpoint/input and run portable inference with tracked status, logs, and cancellation. |
| Local HTTP inference API | Submit `POST /api/inference/run` to launch the same tracked local batch job. |
| Windows executable build | Start a PyInstaller build from Deploy and distribute its one-folder output to a Windows machine without Python. |
| Build/readiness inventory | Inspect supported families, PyInstaller availability, and whether an executable exists. |

The portable bundle supports **seven model IDs through five drivers**:
`point_deeponet`, `deeponet`, `fno`, `transolver`, `meshgraphnets`,
`meshgraphnets-v`, and `sdfflow`. It is CPU-only and supports one family per
process. cHI-MGNflow, MLP, and SimulGenVAE use their native routes. The local
inference API starts a batch job; see the [portable inference guide](docs/guides/inference-bundle.md)
for packaging and compatibility requirements.

### System and documentation

| Feature | What you can do |
| --- | --- |
| Runtime health | Inspect the Studio interpreter/version, connection state, and discovered method entrypoints. |
| NVIDIA device inventory | Read detected GPU index/name, driver, used/total memory, utilization, and temperature. |
| Per-method environments | Route methods to separate Python interpreters through local settings or a CLI override. |
| Repository config audit | Run structural checks across checked-in configs, optionally use Strict, and inspect per-file diagnostics. |
| LLM connection setup | Configure the optional config-rewrite service and inspect its endpoint and setup status. |
| Repository documentation catalog | Search and browse Markdown grouped by area, with file details and pagination. |
| Start-here navigation | Open the Studio, architecture, configuration, and dataset guides directly. |
| In-app documentation reading | Render local headings, code, lists, tables, and links; follow repository Markdown links in the pane or HTTP(S) links in a browser tab. |

Studio is a local workspace and process supervisor. `native` identifies native
method/backend execution, `adapter` identifies Studio composition of outputs,
and `roadmap` identifies planned functionality. Discovery and bulk config audits
check availability/structure; actual launch preflight, completed experiments,
and held-out evaluation provide different evidence.

## Get started

### Open Studio

Install **Python 3.10+**, clone the repository, and create an environment:

```bash
git clone https://github.com/leesihun/AI-CAE4ALL.git
cd AI-CAE4ALL
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install and launch:

```bash
python -m pip install -e .
python -m pip install -r studio/requirements.txt
python studio/start_studio.py 8080
```

Open **http://127.0.0.1:8080/index.html**. Keep the terminal running while you use
Studio. Windows users can also use [START_STUDIO.bat](studio/START_STUDIO.bat).

### Run a model

1. Install the selected method's dependencies using its [method guide](docs/methods/).
2. Choose a pipeline template, or add a dataset and model to an empty canvas.
3. Set compatible data, checkpoint/output paths, and GPU selection.
4. Validate the configuration, start the run, and inspect its logs and results.

Each method keeps its own runtime and data contract. For separate environments,
copy [ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` and set each method's Python interpreter.

**Bring your own data and checkpoints.** The datasets, pretrained weights, and
generated results shown in the videos are local artifacts and are not included
in a clone. Design-generation templates need compatible trained checkpoints.

## Models

The launcher exposes **11 model IDs and 32 operation routes across nine native
runtimes**. `geometry_ingest` prepares data; it does not train a model. HI-MGN
is the multiscale configuration of the `meshgraphnets` route.

| Category | Model / ID | Operations |
| --- | --- | --- |
| Mesh-field prediction | MeshGraphNets / HI-MGN (`meshgraphnets`) | `train`, `inference` |
| Mesh-field prediction | Transolver (`transolver`) | `train`, `inference` |
| Neural operators | DeepONet (`deeponet`) | `train`, `inference` |
| Neural operators | Point-DeepONet (`point_deeponet`) | `train`, `inference` |
| Neural operators | FNO (`fno`) | `train`, `inference` |
| Conditional field generation | MeshGraphNets-V (`meshgraphnets-v`) | `train`, `inference` |
| Conditional field generation | [cHI-MGNflow](methods/HI_MGNFlow/) (`chi-mgnflow`) | `train`, `train_ae`, `train_prior`, `inference` |
| Fixed-geometry fields | SimulGenVAE (`simulgenvae`) | `train`, `train_vae`, `train_lc`, `reconstruct` |
| Geometry generation | SDFFlow (`sdfflow`) | `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`, `interpolate`, `optimize`, `evaluate` |
| Tabular response prediction | MLP (`mlp`) | `train`, `inference` |
| CAD/geometry preparation | GeometryIngest (`geometry_ingest`) | `ingest`, `inspect` |

### Architecture and prediction choices

| Family | Supported choices |
| --- | --- |
| MeshGraphNets / HI-MGN / BSMS | Flat or multilevel graph processing; FPS/Voronoi or BFS bi-stride coarsening; per-level depths and cluster budgets; mean/attention pooling, sum/attention lifting, broadcast ablations, world edges, and hierarchy caches. Static fields and compatible temporal rollouts. |
| MeshGraphNets-V | Graph-aware posterior encoding, latent conditioning, multiscale backbones, conditional flow-matching or supported legacy priors, and multiple predicted field samples. |
| cHI-MGNflow | Hierarchical field autoencoding followed by conditional flow matching; separate AE/prior stages and deterministic or ensemble readout. |
| DeepONet | Fixed-sensor branch and coordinate trunk for field prediction, with supported branch/condition choices. |
| Point-DeepONet | PointNet branch, SIREN coordinate trunk, and point/query decoding with supported sampling/refinement choices. |
| FNO | Fourier field operators with mesh/grid adapters and configurable grid, modes, width, and depth. |
| Transolver | Physics-Attention slices, naive/slice-space kernels, tiled or node-sharded execution, decoupled inference, and optional two-stream amortized training. |
| SimulGenVAE | Hierarchical VAE plus parameter-to-latent conditioner for fixed-size static/transient fields; separate VAE/LC training and reconstruction. |
| SDFFlow | SDF-VAE plus latent flow matching, supported MLP/attention decoders and MLP/DiT flow networks, conditional generation and geometry evaluation. |
| MLP | Configurable scalar-input/scalar-output regression with saved input/output normalization and CPU or supported GPU execution. |

The [method guides](docs/methods/) and native repositories document each family's
settings. A shared launcher does not make data layouts, checkpoints, conditions,
or architecture controls interchangeable.

### Data contracts

| Data kind | Stored content | Used by |
| --- | --- | --- |
| Mesh fields | Per-sample `nodal_data[F,T,N]` and mesh connectivity; coordinate rows precede physical fields and input-only conditions. | MeshGraphNets families, cHI-MGNflow, operators, Transolver; fixed-size constraints also apply to SimulGenVAE. |
| SDF geometry | Shape surface points/normals, SDF queries/values, and optional conditions. | SDFFlow |
| Scalar tables | `X[samples,inputs]`, `Y[samples,outputs]`, and optional column names. | MLP |

SimulGenVAE requires matching node and timestep counts across samples.
Compatible temporal workflows distinguish state feedback from input-only
conditioning. See the [dataset reference](docs/reference/DATASET_FORMAT.md)
for exact layouts, feature rows, and metadata.

Inspect the available model IDs and operations:

```bash
python AI_CAE4ALL_main.py --list-models
python AI_CAE4ALL_main.py --describe sdfflow
```

## Use the command line

After setting up the method and updating the example's data paths and GPU
selection, run from the repository root:

```bash
# Validate the configuration.
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/deterministic/ex1/baseline/config_train_himgn.txt --check

# Launch the native method.
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/deterministic/ex1/baseline/config_train_himgn.txt
```

The configuration's `model` and `mode` select the implementation and operation.
See the [configuration reference](docs/CONFIGURATION.md) for accepted keys,
environment selection, and path resolution.

### Launcher options

| Option | Purpose |
| --- | --- |
| `--config PATH` | Parse settings and launch the selected native model/mode. |
| `--list-models` | List registered IDs, modes, runtime paths, and entrypoints. |
| `--describe MODEL` | Inspect one model's route, data kind, and mode requirements. |
| `--check` | Run preflight without starting training or inference. |
| `--dry-run` | Validate and print the interpreter, command, and native working directory. |
| `--explain-config` | Explain explicit, missing, defaulted, inactive, checkpoint-owned, and unknown settings. |
| `--show-defaults` | Include default and notice diagnostics without rewriting the file. |
| `--strict` | Promote selected compatibility warnings to errors. |
| `--python PATH` | Override the method interpreter for this invocation. |
| `--audit-configs` | Structurally check all discovered checked-in config templates. |
| `--json-report PATH` | Write preflight or audit diagnostics as JSON. |
| `--no-color` | Disable colored terminal output. |
| `--skip-native-check` | Bypass the native probe for advanced diagnostics. |
| `--skip-filesystem-check` | Bypass input/output filesystem checks for advanced diagnostics. |
| `--skip-environment-check` | Bypass interpreter/import/CUDA checks for advanced diagnostics. |

Preflight checks configuration/mode contracts, paths, selected environment and
CUDA IDs, dataset dimensions, and supported checkpoint metadata/native probes.
Diagnostics include codes, severity, affected fields, source locations, and
repair hints. Native exit codes propagate; Ctrl+C forwards cancellation to the
process group.

Interpreter selection follows CLI override → model setting → runtime setting →
local default → active Python. Values inside native configs resolve relative to
the selected method directory, so checked-in examples use `../../dataset/...`
and `../../output/...`. The config file itself is passed by absolute path.

Repository-wide audit is structural and skips filesystem, environment, native,
and dataset gates. It is useful for template maintenance; full launch preflight
checks the concrete run.

```bash
# Audit checked-in templates and save diagnostics.
python AI_CAE4ALL_main.py --audit-configs --json-report output/config-audit.json

# Check campaign generation and data contracts.
python configs/campaigns/dataset_matrix/generate.py --check
python configs/campaigns/dataset_matrix/audit.py --data

# Score outputs after comparable runs have actually completed.
python configs/campaigns/dataset_matrix/score_rank.py --root .
```

Installing the launcher also exposes the `ai-cae4all` console command. Portable
inference has its own [CLI and Python API](docs/guides/inference-bundle.md).

## Documentation

| Guide | Contents |
| --- | --- |
| [Studio](docs/GUI.md) | Pipelines, configuration, jobs, and viewers |
| [Methods](docs/methods/) | Model architectures and setup |
| [Dataset formats](docs/reference/DATASET_FORMAT.md) | Input contracts and field layouts |
| [Public datasets](docs/reference/PUBLIC_DATASETS.md) | Dataset sources |
| [Configuration](docs/CONFIGURATION.md) | CLI options and configuration keys |
| [Architecture](docs/ARCHITECTURE.md) | Launcher and native runtime organization |
| [Portable inference](docs/guides/inference-bundle.md) | Exporting inference bundles |
| [Development](docs/guides/testing.md) | Checks and tests |
| [Video notes](docs/videos/README.md) | Recording provenance, subtitles, and next demos |

Core code lives in `cae_suite/` (launcher), `methods/` (native methods), and
`studio/` (browser UI and local server).
