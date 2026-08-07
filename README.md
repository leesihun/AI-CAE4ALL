# AI-CAE4ALL

**The all-in-one SciML platform for AI-driven CAE.** Eight self-contained method
repositories — seven ML methods plus a CAD-to-dataset front end — exposing
**11 routable model IDs across 28 mode routes**, behind one config-driven
launcher that validates everything before a single GPU-second is spent, and a
full browser Studio that turns the whole thing into a drag-and-drop pipeline.

![AI-CAE4ALL Studio — pipeline editor](docs/images/studio-pipeline-editor.png)

Pick a method by writing **one word** in a text config:

```bash
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train1.txt
```

…or never touch a terminal at all:

```powershell
frontend\START_STUDIO.bat
```

---

## Table of contents

- [What this is](#what-this-is)
- [The model zoo — 11 routes, one contract](#the-model-zoo--11-routes-one-contract)
- [The Studio](#the-studio-a-real-gui-over-real-runs)
- [The launcher: fail before you burn GPU hours](#the-launcher-fail-before-you-burn-gpu-hours)
- [One dataset contract, no conversion step](#one-dataset-contract-no-conversion-step)
- [Cross-cutting capabilities](#cross-cutting-capabilities)
- [Paper-fidelity validation](#paper-fidelity-validation)
- [Ship it: the portable inference bundle](#ship-it-the-portable-inference-bundle)
- [Quick start](#quick-start) — [installing](#installing) · [running](#running)
- [Config format](#config-format)
- [Repository layout](#repository-layout)
- [Per-method Python environments](#per-method-python-environments)
- [Testing](#testing)
- [Known gaps — the honest list](#known-gaps--the-honest-list)
- [Documentation map](#documentation-map)
- [Scale](#scale)

---

## What this is

Most ML-for-CAE work dies at the seams: every method wants its own data format,
its own CLI, its own environment, and its own idea of what a checkpoint is.
AI-CAE4ALL removes the seams without merging the code.

Three layers, each usable on its own:

| Layer | What it gives you |
| --- | --- |
| **Studio** (`frontend/`) | A local, zero-install browser workspace: typed drag-and-drop pipeline blocks, a real 3D field/mesh/CAD viewer, live training metrics, authoritative preflight, and real job execution with logs and cancellation. |
| **Launcher** (`cae_suite/`) | `parse → route → layered preflight → subprocess`. One command, every method. Never imports ML code; validates in the *target method's* interpreter. |
| **Method repos** | Eight independent runtimes — each with its own venv, tests, entrypoint, and docs — all runnable standalone. |

The launcher's value is **uniform validation and routing**: it reports *every*
problem with a config before launching, and it always starts the native process
in that method's working directory and Python interpreter.

---

## The model zoo — 11 routes, one contract

Every one of these is selected purely by the `model` field in a flat text config.
No code changes, no format conversion, no per-method CLI to memorize.

| `model` value(s) | Method | Modes | What it does |
| --- | --- | --- | --- |
| `meshgraphnets` | **MeshGraphNets and HI-MGN** | `train`, `inference` | Encode–process–decode GNN mesh simulator with a multiscale V-cycle processor, world edges, and learned attention transfer operators |
| `meshgraphnets-v` | **MeshGraphNets (variational)** | `train`, `inference` | Probabilistic superset: VAE latent path + a **learned conditional prior** (flow-matching or GMM) → a *distribution* of plausible trajectories |
| `point_deeponet` | **Point-DeepONet** | `train`, `inference` | PointNet branch + SIREN trunk with early fusion; arbitrary query points |
| `deeponet` | **DeepONet** | `train`, `inference` | Canonical fixed-sensor branch/trunk operator |
| `fno` | **FNO** | `train`, `inference` | Native spectral (Fourier) convolutions — no `neuraloperator` dependency |
| `gino` | **GINO** | `train`, `inference` | GNO in ↔ latent FNO ↔ GNO out; mesh→grid→query via radius neighborhoods |
| `transolver` | **Transolver** | `train`, `inference` | Transformer surrogate over learned Physics-Attention "slices": `O(N²)` → `O(N·slice_num)` |
| `sdfflow` | **SDFFlow** | `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`, `interpolate` | *Generates new 3D shapes*: SDF-VAE + rectified-flow matching, conditioned on geometric descriptors, meshed with marching cubes |
| `simulgenvae` | **SimulGenVAE** | `train`, `train_vae`, `train_lc`, `reconstruct` | Hierarchical VAE + latent conditioner: conditions → full simulation field, no FOM solve |
| `mlp` | **MLP Surrogate** | `train`, `inference` | Tabular parametric regressor: N scalar inputs → M scalar outputs. CPU-only, seconds to train |
| `geometry_ingest` | **Geometry Ingest** | `ingest`, `inspect` | Non-ML data prep: STEP/IGES/STL/PLY/OBJ → the shared mesh HDF5 contract |

```bash
python AI_CAE4ALL_main.py --list-models   # every route + install health
```

**Four operator architectures live in one repo** (`Neural_Operator/`) sharing a
single split / target / normalization / noise / optimizer / scheduler /
checkpoint / rollout convention. Switching `model fno` → `model gino` must never
require touching dataset, training-loop, loss, checkpoint, or inference code —
and it doesn't.

---

## The Studio: a real GUI over real runs

`frontend/` is not a mockup. Every button is wired to the actual suite: the same
`MethodSpec` validation, the same `AI_CAE4ALL_main.py` subprocess, the same HDF5
files on disk. Blocks carry visible maturity labels (`native` / `adapter` /
`roadmap`) so nothing that isn't finished is presented as if it were.

### Typed, drag-and-drop pipelines

Sources → preparation → models → execution → evaluation → export, with typed
ports that only connect where the data actually flows. Dependency-ordered
execution runs each step through the real launcher, capturing logs, exit codes,
and exact pipeline-node lineage.

![Generative design optimization pipeline](docs/images/studio-optimization-pipeline.png)

**Graph-aware autofill with provenance.** Connect an HDF5 block to a model and
`dataset_dir` populates itself. Connect Design Parameters and MLP's
`input_var`/`output_var` — or SimulGen/SDFFlow condition settings — populate
themselves. Connect a checkpoint and family-appropriate model paths follow.
Edit one value and it becomes a persistent manual override; clear it and the
value resumes following the graph; disconnect and only values still owned by
that connection are removed.

### Every config key, form and text, always in sync

The configuration workspace exposes the complete live key catalog per method —
required, recommended, inactive, and checkpoint-owned — with the flat `.txt`
rendered side-by-side and synchronized in both directions. **Run preflight** and
**Explain config** call the authoritative launcher, not a reimplementation.

![SimulGen-VAE full configuration workspace — 67 live keys](docs/images/studio-config-workspace.png)

### A real 3D viewer for real artifacts

An opaque, depth-buffered WebGL viewport (with a Canvas 2D fallback) renders
**actual repository data** — never a substituted placeholder file. Left-drag
orbit, right-drag pan, wheel zoom, keyboard camera, view reset — identical
across every contract it supports.

| Mesh fields from the shared HDF5 contract | CAD / surface meshes |
| --- | --- |
| ![HDF5 field viewer](docs/images/studio-hdf5-field-viewer.png) | ![CAD mesh viewer](docs/images/studio-cad-viewer.png) |

The shared contract stores **no cells** — only a `mesh_edge` graph. The Studio
reconstructs elements from it: 3-cliques recover triangles, 4-cycles recover
quads, so Field mode is a genuine element-coloured contour rather than a
wireframe fake. Oversized meshes are reduced by **vertex clustering**, never by
striding the edge list, so every surviving element stays connected.

The dataset's own names travel with the data: `metadata/feature_names`, SDFFlow
`cond_names`, and tabular `input_names`/`output_names` drive the channel picker
and the sample-parameter table — and **Use in pipeline** writes them onto the
source block and everything downstream.

| SDFFlow shape point clouds | Tabular MLP input/output pairs |
| --- | --- |
| ![SDF point cloud viewer](docs/images/studio-sdf-pointcloud.png) | ![Design parameters spreadsheet](docs/images/studio-design-parameters.png) |

### Live training metrics, from the actual logs

The **Train Metrics** block discovers every scalar in a persisted training log,
plots all series by default, supports per-series exclusion and visual-only
smoothing (statistics and CSV stay raw), and downloads the selected raw
observations. Multi-run overlays are graph-connected. The floating process log
polls without re-opening a drawer you minimized.

![Train Metrics workspace with live process log](docs/images/studio-training-metrics.png)

### Twelve repository-backed workspaces

Data · Experiments · Optimization · Evaluation · Compare · Export · Models ·
Benchmarks · Artifacts · Deploy · System · Docs — each reading live repository
state. The **System** workspace runs a repository-wide config audit — the same
parse/spec/route checks the CLI's `--audit-configs` performs — over every
checked-in `configs/**/config*.txt`, reporting PASS/FAIL with per-file
diagnostics.

### Real analysis, not decoration

- **Field evaluation** against ground truth: relative L2, MAE, RMSE, maximum
  error, R².
- **Cross-model ranking** from real CSVs, with schema inspection that lists the
  actual finite numeric columns before you rank — identifier and metadata
  columns are excluded from objective suggestions, and every objective requires
  an explicit minimize/maximize choice.
- **Pareto / crowding-distance** optimization with geometry-feasibility gates.
- **Export** to file or ZIP; **Build .exe** produces the portable CPU inference
  bundle; **API Deployment** hands off the endpoint shape.
- **LLM-assisted configuration** (optional): the Studio can ask an
  OpenAI-compatible endpoint to rewrite a block's flat `.txt` config in place —
  spoken over HTTP only, nothing vendored or imported.

Everything the Studio writes lands under the git-ignored `frontend/runtime/`.
Method repositories and suite modules are launched or imported, never rewritten.
It is a localhost development API — not a multi-user production deployment.

---

## The launcher: fail before you burn GPU hours

```
parse config → route on `model` → layered preflight → build native command → subprocess-launch
```

Preflight runs in layers and **short-circuits — each layer runs only if no
errors so far**:

| Layer | What it proves |
| --- | --- |
| **Spec** | Every key is known, required keys present, values in range, mode/model combination legal |
| **Filesystem** | Inputs exist, outputs are writable, `PathRule`s satisfied |
| **Environment** | The resolved interpreter has what this method needs |
| **Dataset** | HDF5 schema, shapes, and feature counts cross-checked *against this config* |
| **Checkpoint** | Safe `weights_only` metadata read: model/stage/normalization consistency |
| **Native probe** | The method's **own** config validator accepts it |

The last three run **as subprocesses under the target method's Python**, so the
launcher validates against the environment the model will actually run in — not
its own.

Diagnostic codes map to exit codes, so tooling can branch on failure class:
`ROUTE-*` → 3, `ENV-*` → 4, `NATIVE-CHECK-*` → 5, any other error → 2, clean → 0.

```bash
python AI_CAE4ALL_main.py --config <cfg> --check           # validate only; report ALL problems together
python AI_CAE4ALL_main.py --config <cfg> --dry-run         # print the exact native command
python AI_CAE4ALL_main.py --config <cfg> --explain-config  # configured / defaulted / inactive / checkpoint-owned
python AI_CAE4ALL_main.py --config <cfg> --strict          # promote flagged warnings to errors
python AI_CAE4ALL_main.py --config <cfg> --json-report r.json
python AI_CAE4ALL_main.py --audit-configs                  # structural lint pass over checked-in configs
python AI_CAE4ALL_main.py --describe transolver            # route, modes, required fields per mode
```

Ctrl-C is forwarded and escalated cleanly across the process group on both
Windows and POSIX.

> **Known gap:** `--audit-configs` scans the wrong root today and reports
> `files=0` — see [Known gaps](#known-gaps--the-honest-list). Use the Studio's
> System workspace, which runs the identical checks over all 127 files.

---

## One dataset contract, no conversion step

MeshGraphNets, MeshGraphNets-V, all four Neural Operators, and Transolver read
the **same** HDF5 layout with zero conversion:

```text
data/{sample_id}/nodal_data   # [num_features, num_timesteps, num_nodes]
                              #   rows 0:3 = reference coordinates
data/{sample_id}/mesh_edge    # graph connectivity
metadata/feature_names        # names that travel with the data
```

The graph methods consume `mesh_edge`; the operators and Transolver read the
same nodes as a point cloud and ignore it. `write_preprocessing` may append
train-derived normalizers to the file.

Two deliberate exceptions:

- **SimulGenVAE** reads the same mesh HDF5 but is a *fixed-geometry dense FOM*
  model — it flattens field rows into `[samples, channels, time]`, so every
  sample must share node and timestep counts.
- **MLP** is tabular: `X[S,N]` / `Y[S,M]` with optional `input_names` /
  `output_names`. No mesh, no GPU.
- **SDFFlow** uses an SDF layout: `shapes/{id}/{surface_points, surface_normals,
  sdf_points, sdf_values, cond}`, sign negative inside.

**And if you only have CAD?** `geometry_ingest` meshes STEP/IGES/STL/PLY/OBJ
straight into the contract (gmsh tet volumes or trimesh surfaces, optional FPS
point-cloud resampling) — so nothing downstream changes.

Full spec: [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md).

### Data that ships with the repo

| File | Used by | Notes |
| --- | --- | --- |
| `dataset/ex1.h5` | mesh methods, SimulGenVAE | **Planar** geometry — `operator_dim` resolves to 2 |
| `dataset/ex2.h5` | mesh methods | Genuinely **3D** geometry (~200k nodes/sample) |
| `dataset/ex*.mscache.*.h5` | MeshGraphNets | Prebuilt multiscale hierarchy caches |
| `dataset/deepjeb.h5` | SDFFlow | Geometry-generation shapes + 5 descriptors |
| `dataset/ex1_infer.h5` | mesh methods | ex1 single-sample hex-mesh inference input; state rows carry the ground-truth field (was `hex_GT.h5`) |
| `dataset/ex2_infer.h5` | mesh methods | ex2 held-out inference set: 5 unseen scenes × 50 timesteps; rollout is seeded from t=0 and scored against t=1..49 |
| `dataset/hex_dataset.h5` | mesh methods | Same mesh with the state rows zeroed — legacy, superseded by `ex1_infer.h5` |
| `dataset/mlp/` | MLP | Tabular `X`/`Y` sample + its generator |
| `dataset/benchmarks/` | Neural_Operator, Transolver | Five per-paper validation suites |

Planar vs. 3D is **discovered from the geometry**, never hardcoded — which is
why `ex1` and `ex2` both work through the same configs. HDF5 and checkpoint
files are git-ignored; builders live under `dataset/` and in the method repos
(e.g. `Geometry_generation/build_dataset.py`).

---

## Cross-cutting capabilities

Features that span methods rather than living in one of them.

### AR-OT vs AR-RT time integration — one config key

Active in both MeshGraphNets variants, Transolver, and all four Neural
Operators. `time_integration ar_ot` (default) or `ar_rt`:

- **AR-OT** — one-step teacher forcing on ground-truth pairs; `std_noise` papers
  over the train/test mismatch.
- **AR-RT** — full-trajectory rollout training (following NVIDIA/GM
  [arXiv:2510.15201](https://arxiv.org/abs/2510.15201)): the model consumes its
  own predictions exactly as inference does and backpropagates through all steps,
  gradient-checkpointed per step. Validation loss *becomes* rollout loss, so
  best-checkpoint selection optimizes what you actually care about. MeshGraphNets
  rebuilds mesh edge features, world edges, and coarse features per step from
  each predicted state; variational MGN resamples the latent every step with
  RNG-preserving checkpointing so the backward draw matches the forward.

### Parallelism, three ways

| Mode | Methods | Meaning |
| --- | --- | --- |
| `ddp` (default) | all | Data-parallel; one full model per GPU |
| `model_split` | MGN, MGN-V, FNO, GINO | 1F1B pipeline across ≥2 GPUs; **merged checkpoints load like single-GPU ones** |
| `node_shard` | Transolver | One mesh's nodes split across ≥2 GPUs; slice aggregates autograd-aware SUM all-reduced, **reproducing single-process results bit-for-bit** |

### Multiscale graph learning — three published architectures, one route

`model meshgraphnets` is not one architecture. Two config keys select between
three, with **identical** encoder, decoder, training loop, and rollout:

| Architecture | Config | Coarsening |
| --- | --- | --- |
| **Flat MGN** | `use_multiscale False` | — (flat message-passing stack) |
| **HI-MGN** | `use_multiscale True`, `coarsening_type voronoi_seedmean` | FPS-Voronoi clustering to a fixed target count per level |
| **BSMS-GNN** | `use_multiscale True`, `coarsening_type bfs` | Bi-stride multi-source BFS (Cao et al., ICML 2023) |

The V-cycle pools *down* to the coarsest level and unpools back *up*, so
long-range interactions cross the whole mesh in a handful of blocks rather than
one layer per hop. Unpooling is a **learned bipartite message-passing step**
conditioned on relative position — not a naive broadcast. World edges can
propagate hierarchically too (`coarse_world_edges`), and the expensive
coarsening is cached **per worker and on disk**, so you build each hierarchy
once. Deep dives: [02_HI-MGN.md](docs/methods/02_HI-MGN.md),
[03_BSMS-GNN.md](docs/methods/03_BSMS-GNN.md).

Recent work adds **learned attention transfer operators** — attention-based
restriction and prolongation (`pool_type` / `unpool_type` / `pool_heads`) that
are provably identical to the fixed mean/sum operators *at initialization*, so
existing behaviour is preserved and the model learns its way out. Score
computation runs in an explicit fp32 island so bf16 autocast is not silently
upgraded around it.

### Training-loop engineering

AMP (bf16), `torch.compile`, EMA (with BatchNorm running-stat copies where the
architecture needs them), gradient accumulation, input-noise regularization,
geometry augmentation, KL warmups, temperature annealing, and activation
checkpointing — available where each is meaningful, and rejected loudly where
it isn't.

### Guardrails that refuse to fail silently

- A **removed-feature guard** hard-rejects VAE/prior keys routed to the
  deterministic MeshGraphNets, so a misrouted variational config fails loudly
  instead of quietly ignoring half its settings.
- `Neural_Operator/config_validation.py` is a strict key registry: unknown or
  removed keys raise *before* any HDF5 is opened.
- SDFFlow's merged pipeline **refuses to start flow matching unless the VAE
  checkpoint verifies complete**, and a newly trained VAE always invalidates
  reuse of an older FM checkpoint. No stale VAE/FM pairings, no idle-GPU gap
  between stages.
- A UTF-8 BOM in a config is a hard error, because the native parsers misread the
  first key.

---

## Paper-fidelity validation

The methods are checked against **their own papers' benchmarks, output
quantities, and metrics** — not against each other, and not against a
substituted dataset. Where a paper protocol cannot be represented without
production changes, the report says so rather than swapping in something easier.

| Method | Own-paper benchmark | Paper result | This suite | Verdict |
| --- | --- | --- | --- | --- |
| **Transolver** | Elasticity, 972 irregular points | rel. L2 **0.0064** | **0.0064211** | passes — 0.33% above paper |
| **FNO** | 2D Darcy flow, 85×85 | rel. L2 **0.0108** | **0.0099181** | passes — 0.918× paper |
| **Point-DeepONet** | Non-parametric 3D structures, variable loads | avg R² **0.897** | **0.892832** | paper-similar — 0.46% low |
| **DeepONet** | 2D fractional Laplacian on the unit disk | ≈**1.2e-3** normalized MSE | **0.00148703** | paper-similar |
| **GINO** | ShapeNet Car pressure | decoder-only rel. L2 **7.12%** | 0.090992 | correction run active |

Paper-reproduction paths are **opt-in and instance-bound**. An independent
default-path audit confirmed that seeded default FNO weights, forward output,
and exported config are **bit-exact to git HEAD**, that default GINO never
imports the Car-CFD paper module, and that the exact-paper Transolver module is
unimported by normal runtime. No paper branch runs inside a hot loop.

Full evidence, protocols, and SHA-256-verified source data:
[dataset/benchmarks/PER_PAPER_VALIDATION_REPORT.md](dataset/benchmarks/PER_PAPER_VALIDATION_REPORT.md).

---

## Ship it: the portable inference bundle

`inference/` is a **stand-alone, CPU-only, zero-dependency-on-this-repo**
inference package. Copy the folder anywhere, install `requirements.txt`, run.
Or build the one-folder `.exe` and run it on a machine with **no Python at all**.

```bash
python run_inference.py --checkpoint model.pth --input scene.h5 --output out/
run_inference.exe --checkpoint model.pth --input scene.h5 --output out/
```

**You never say which model it is.** `detect_family()` inspects the checkpoint's
own top-level keys — the ones each method's native `save_checkpoint` already
writes — and dispatches:

| Checkpoint signature | Family |
| --- | --- |
| `schema_version == 'deeponet_repo_v1'` | Neural Operator (`selected_model` picks the architecture) |
| `schema_version == 'sdfflow_infer_v1'` or `stage in {vae, fm}` | SDFFlow (generative — no `--input` needed) |
| `checkpoint_version` present | Transolver |
| `model_config` has `use_vae` | MeshGraphNets (variational) |
| `model_config` has `message_passing_num`, no `use_vae` | MeshGraphNets |

Each family folder keeps its **original internal module names**, so the vendored
files needed zero import rewriting — the registry enforces one family per
process. SDFFlow checkpoints embed the frozen VAE they were trained against, so
a single `.pth` is a complete, self-contained, hand-it-to-anyone artifact.

---

## Quick start

### Installing

The launcher itself is deliberately tiny: **Python ≥ 3.10** and, on 3.10 only,
`tomli`. It has no ML dependencies at all — that is what lets it validate a
config for a method whose environment it does not share.

```bash
python -m pip install -e .        # optional; also provides the `ai-cae4all` command
python AI_CAE4ALL_main.py --list-models   # confirms which method repos are installable
```

Each method brings its own dependencies, installed into that method's venv —
there is intentionally **no root `requirements.txt`**:

```bash
python -m pip install -r Geometry_generation/requirements.txt   # sdfflow
python -m pip install -r SimulGenVAE/requirements.txt           # simulgenvae
python -m pip install -r MLP/requirements.txt                   # mlp (CPU-only)
python -m pip install -r inference/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The mesh/operator methods need PyTorch matched to your CUDA build; GINO
optionally uses `torch_cluster` for neighbor search and falls back to a scipy
`cKDTree` path without it. `geometry_ingest` needs `trimesh` for surface meshes
and `gmsh` for volume tet meshes. `--list-models` reports install health per
route, and preflight's environment layer tells you what a specific config is
missing before it launches. The Studio needs nothing beyond a browser and the
launcher's own Python.

### Running

`--config` selects the file; **`mode` (train / inference / sample / …) lives
inside the config**, not on the CLI.

```bash
# Validate only — reports every missing or conflicting setting together, no launch:
python AI_CAE4ALL_main.py --config configs/Transolver/ex2/config_train_transolver.txt --check

# Print the exact native command without launching:
python AI_CAE4ALL_main.py --config configs/Neural_Operator/ex1/config_train_fno.txt --dry-run

# A clean preflight auto-launches the native process:
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train1.txt
```

### The Studio

```powershell
frontend\START_STUDIO.bat          # opens http://127.0.0.1:8080/index.html
frontend\START_STUDIO.bat 8081     # if 8080 is taken
python frontend\start_studio.py 8080
```

Do **not** use `python -m http.server` — it will display the HTML but provide no
model execution, preflight, repository browsing, or artifact APIs. The correct
console prints `AI-CAE4ALL Studio is ready` and the badge in the browser reads
`11 routes healthy`.

### SDFFlow (geometry generation)

One production training config trains the VAE, verifies its checkpoint, then
immediately trains flow matching — no idle-GPU gap:

```bash
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_sample.txt
```

Relaunching safely reuses compatible completed stages. Guarded extrapolation and
reproducible interpolation configs ship alongside.

### CAD → dataset

```bash
python AI_CAE4ALL_main.py --config configs/geometry_ingest/config_ingest_volume.txt --check
python AI_CAE4ALL_main.py --config configs/geometry_ingest/config_ingest_surface.txt
```

---

## Config format

All backends read the same flat `key value` text. Keys and string values are
lowercased; `%` starts a comment (**on its own line**); commas or multiple
tokens make a list (`gpu_ids 0,1`); `true`/`false` are booleans. Duplicate keys
are an error. Two quirks worth knowing:

- A **single value parses to a bare scalar, not a one-element list**
  (`test_batch_idx 0` → `0`).
- **Prefer decimal notation** (`0.0001`, not `1e-4`): a token with no `.` fails
  the `int()`/`float()` fast paths and stays a **string**, so consumers convert
  explicitly.

Paths inside a config resolve from the **selected method repository** (its
working directory) — which is why templates use paths like
`../output/meshgraphnets/ex1/model1.pth`. For inference, architecture and
normalization are **checkpoint-led**: config values may be replaced by the
checkpoint once it loads.

Adding or renaming a native config key means editing that method's
`known_keys` in [cae_suite/specs/](cae_suite/specs/) — the specs are the single
source of truth for validation.

---

## Repository layout

```text
AI_CAE4ALL_main.py            # entrypoint → cae_suite.cli.main
cae_suite/                    # the launcher (parse → route → preflight → subprocess); no ML
  ├── cli.py registry.py preflight.py diagnostics.py config_parser.py
  ├── native_probe.py dataset_probe.py checkpoint_probe.py   # run in the METHOD's venv
  └── specs/                  # one MethodSpec per method — validation truth
frontend/                     # the Studio: browser UI + local Python API bridge
inference/                    # stand-alone CPU inference bundle + PyInstaller spec
configs/                      # 127 config templates per method + benchmarks
dataset/                      # shared HDF5 data, format spec, benchmarks, geometry_ingest
docs/                         # per-method deep dives, Studio plans, images
output/                       # run artifacts (checkpoints, rollouts, samples)

MeshGraphNets/                # model = meshgraphnets
MeshGraphNets - variational/  # model = meshgraphnets-v   (note: the name has spaces)
Neural_Operator/              # model = point_deeponet | deeponet | fno | gino
Transolver/                   # model = transolver
Geometry_generation/          # model = sdfflow
SimulGenVAE/                  # model = simulgenvae
MLP/                          # model = mlp
dataset/geometry_ingest/      # model = geometry_ingest
```

> `MeshGraphNets - variational/` literally contains spaces — quote it in shell
> commands and rely on `pathlib`, never string concatenation.

---

## Per-method Python environments

Launching from an activated venv needs no configuration — that venv's Python is
used for everything. For separate backend environments, copy
[ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored):

```toml
[python]
default = ".venv/bin/python"

[python.models]
meshgraphnets   = "MeshGraphNets/.venv/bin/python"
meshgraphnets-v = "MeshGraphNets - variational/.venv/bin/python"
neural_operator = "Neural_Operator/.venv/bin/python"
transolver      = "Transolver/.venv/bin/python"
sdfflow         = "Geometry_generation/.venv/bin/python"
simulgenvae     = "SimulGenVAE/.venv/bin/python"
mlp             = "MLP/.venv/bin/python"
```

Precedence: `--python` → exact model ID → method ID → `python.default` → the
suite's own `sys.executable`.

> `resolve_python` deliberately **never calls `Path.resolve()`** on the
> interpreter path. A venv's `python` is a symlink whose *location* CPython walks
> up from to find `pyvenv.cfg`; dereferencing it would silently drop the venv's
> site-packages.

---

## Testing

Tests live per method repo and run in that repo's venv:

```bash
cd Neural_Operator      && pytest tests/                              # deepest coverage; tiny synthetic HDF5 fixtures
cd MeshGraphNets        && python -m pytest -q tests/                 # AR-rollout, multiscale stats, attention transfer
cd Geometry_generation  && python -m pytest -q tests/test_sdfflow_pipeline.py
cd SimulGenVAE          && python -m pytest -q tests/test_fom_dataset.py
cd MLP                  && python -m pytest -q tests/                 # CPU train → infer smoke test
```

The Studio ships its own browser smoke runners (`frontend/*-smoke-runner.js`)
covering the viewer, autofill, parameter sheet, training metrics, and workflow
UX, plus Python tests for the analysis and metric backends.

There is **no root-level test suite** — the `testpaths = ["tests"]` entry in
[pyproject.toml](pyproject.toml) is stale.

---

## Known gaps — the honest list

A platform this wide has seams, and they are written down rather than left to be
rediscovered the hard way. The short version:

- **`--audit-configs` reports `files=0`.** It walks `suite_root /
  spec.repository`, but every checked-in config lives in the top-level
  `configs/` tree. The Studio's System workspace runs the identical checks
  against the correct root and does cover all 127 files.
- **`num_workers` is runtime-required but not preflight-required** for
  mesh/operator training — a config omitting it can pass validation and fail
  natively. Same story for `infer_timesteps` on static `T=1` data.
- **Benchmark-intent keys** (`split_strategy`, `loss_type`,
  `relative_l2_epsilon`) are accepted but not implemented in the stable runtime.
- **Schema-gap keys** — a handful (Neural_Operator `use_parallel_stats` /
  `train_eval_subset_size`, Transolver `test_batch_idx`, GINO
  `gino_transform_type`) are read by code but missing from a key registry, so
  they cannot currently be authored cleanly.
- **Variational MGN ignores `weight_decay`** (it constructs `Adam`), and its
  training tree lags the deterministic one on several hot-path optimizations.
- The Studio is a **localhost development API** — no authentication, request
  isolation, quotas, or rollback. Not a multi-user deployment.
- Studio blocks labelled `roadmap` (parts of the optimization search layer) are
  design, not implementation — the labels are visible in the UI for exactly this
  reason.

Authoritative and exhaustive:
[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) and
[REPOSITORY_OVERVIEW.md §14](REPOSITORY_OVERVIEW.md).

---

## Documentation map

| Doc | Purpose |
| --- | --- |
| [REPOSITORY_OVERVIEW.md](REPOSITORY_OVERVIEW.md) | Full architecture guide: launcher internals plus a section on every method |
| [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) | Exhaustive, live-code-backed catalog of every config key, its necessity, and known launcher/native mismatches |
| [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) | The shared mesh HDF5 contract (and the tabular/SDF exceptions) |
| [dataset/benchmarks/PER_PAPER_VALIDATION_REPORT.md](dataset/benchmarks/PER_PAPER_VALIDATION_REPORT.md) | Per-paper reproduction protocols, evidence, and results |
| [docs/methods/](docs/methods/) | Per-method deep dives (13 numbered files, incl. HI-MGN, BSMS-GNN, and geometry ingest) |
| [frontend/README.md](frontend/README.md) | Studio capabilities, local API surface, and integration boundary |
| [inference/README.md](inference/README.md) | Portable inference bundle: family detection, CLI, `.exe` build |
| [MeshGraphNets/ATTENTION_TRANSFER_DESIGN.md](MeshGraphNets/ATTENTION_TRANSFER_DESIGN.md) | Learned restriction/prolongation: design, as-built, and what remains design-only |
| [docs/prototypes/STUDIO_COMPLETE_IMPLEMENTATION_PLAN.md](docs/prototypes/STUDIO_COMPLETE_IMPLEMENTATION_PLAN.md) | Studio architecture, adapters, milestones, contracts, and release gates |
| [CLAUDE.md](CLAUDE.md) | Condensed conventions for the root launcher |
| `Neural_Operator/`, `Geometry_generation/`, `SimulGenVAE/`, `MLP/` `CLAUDE.md` | Authoritative notes for those methods |

For any specific config key, `CONFIGURATION_REFERENCE.md` is authoritative; for
a method's internals, that method's own docs and code are authoritative.

---

## Scale

| | |
| --- | --- |
| Tracked Python files | **411** (~79,400 lines) |
| Registered model IDs | **11** across **28** mode routes |
| Config templates | **127**, all lintable in one command |
| Method repositories | **8**, each independently runnable |
| Studio JS + backend | ~11,200 lines across 18 ES modules and 17 Python modules |
| Live Studio API routes | 30+ endpoints over 12 repository-backed workspaces |

---

*The launcher never imports ML code. The Studio never rewrites the method repos.
Each method stays standalone. That separation is the whole point — it's what
lets eight independent research codebases behave like one product.*
