# AI-CAE4ALL — Repository Overview & Architecture Guide

> A comprehensive, code-grounded tour of the whole repository: what it is, how
> it is laid out, how the unified launcher works, and what each of the five ML
> method repositories does. This document is the **map**; three companion docs
> are the **territory**:
>
> - [README.md](README.md) — the 60-second quick start.
> - [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) — the exhaustive,
>   live-code-backed catalog of every config key, its necessity, and every known
>   launcher/native mismatch.
> - [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) — the shared mesh
>   HDF5 data contract.
> - [CLAUDE.md](CLAUDE.md) — condensed agent-facing conventions for the root
>   launcher.
>
> Each method repo additionally has its own authoritative docs; those are
> canonical for that method's data contract and architecture.

---

## 1. What this repository is

**AI-CAE4ALL is a monorepo of five independent machine-learning-for-CAE
(Computer-Aided Engineering) method repositories, unified by a single
config-driven launcher.** The intent is that an engineer picks a method by
writing one word in a text config, and one command validates and runs it — no
matter which of five very different ML codebases actually executes.

The five methods span the major families of ML surrogate modeling for
simulation:

| Family | Method | What it predicts |
| --- | --- | --- |
| Graph neural network simulator | MeshGraphNets (deterministic) | Time evolution of physical fields on a mesh |
| Probabilistic GNN simulator | MeshGraphNets (variational) | A *distribution* of plausible mesh trajectories (VAE + learned prior) |
| Neural operators | Neural_Operator (4 architectures) | Field-to-field mappings that generalize across discretizations |
| Transformer surrogate | Transolver | Mesh fields via learned "physics slices" + attention |
| Generative geometry | Geometry_generation / SDFFlow | *New 3D shapes* via an SDF-VAE and flow matching |

### 1.1 The core idea: one launcher, five native runtimes

The launcher (`cae_suite/`, invoked via `AI_CAE4ALL_main.py` or the `ai-cae4all`
console script) does **not** import any ML code. Its job is:

```
parse config → route on the `model` field → layered preflight validation → build native command → subprocess-launch the chosen repo's own entrypoint
```

Each method repo is fully self-contained: its own (optional) virtual
environment, its own tests, its own `main`-style entrypoint, and — for the two
most complex ones — its own `CLAUDE.md`. Every method is also runnable directly
without the launcher. The launcher's value is **uniform validation and routing**:
it reports every problem with a config *before* a single GPU-second is spent, and
it always launches the native process in *that method's* working directory and
Python interpreter.

### 1.2 Scale

Per the last full audit (see the header of
[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md)) the suite is roughly
**215 Python files / ~37,000 physical lines**, plus ~79 checked-in config
templates and a handful of shell/HTML helpers. Approximate per-area Python size:

| Area | Approx. lines | Role |
| --- | --- | --- |
| `cae_suite/` | ~2,900 | The launcher (no ML) |
| `MeshGraphNets/` | ~9,300 | Deterministic GNN simulator |
| `MeshGraphNets - variational/` | ~9,000+ | Probabilistic GNN simulator |
| `Neural_Operator/` | ~11,600 | Four operator architectures |
| `Transolver/` | ~4,600 | Transformer surrogate |
| `Geometry_generation/` | ~2,600 | SDFFlow generative geometry |

---

## 2. Top-level repository map

```text
AI-CAE4ALL/
├── AI_CAE4ALL_main.py            # 8-line shim → cae_suite.cli.main
├── pyproject.toml                # installs the `ai-cae4all` console script
├── ai_cae4all.local.example.toml # template for per-method interpreter paths
├── README.md                     # quick start
├── CLAUDE.md                     # agent-facing root conventions
├── CONFIGURATION_REFERENCE.md    # exhaustive config-key catalog (source of truth)
├── REPOSITORY_OVERVIEW.md        # ← this document
│
├── cae_suite/                    # THE LAUNCHER (parse→route→preflight→subprocess)
│   ├── cli.py                    #   arg parsing, subcommands, exit-code mapping
│   ├── config_parser.py          #   flat `key value` parser (mirrors native quirks)
│   ├── registry.py               #   model → MethodSpec → repo/entrypoint
│   ├── preflight.py              #   layered, short-circuiting validation pipeline
│   ├── diagnostics.py            #   Severity / Diagnostic / report renderer
│   ├── path_checks.py            #   filesystem path existence/writability rules
│   ├── settings.py               #   local TOML → per-method Python interpreter
│   ├── launcher.py               #   launch_and_wait (process-group Ctrl-C forwarding)
│   ├── native_probe.py           #   runs the method's OWN validator (in its venv)
│   ├── dataset_probe.py          #   HDF5 schema/shape cross-check (in its venv)
│   ├── checkpoint_probe.py       #   safe weights_only checkpoint metadata read
│   └── specs/                    #   one MethodSpec per method — validation truth
│       ├── base.py               #     MethodSpec, PathRule, shared validators
│       ├── meshgraphnets.py
│       ├── meshgraphnets_variational.py
│       ├── neural_operator.py
│       ├── transolver.py
│       └── sdfflow.py
│
├── MeshGraphNets/                # METHOD 1  (model = meshgraphnets)
├── MeshGraphNets - variational/  # METHOD 2  (model = meshgraphnets-v)  ← name has spaces
├── Neural_Operator/              # METHOD 3  (model = point_deeponet|deeponet|fno|gino)
├── Transolver/                   # METHOD 4  (model = transolver)
├── Geometry_generation/          # METHOD 5  (model = sdfflow)
│
├── configs/                      # centralized config templates (see §11)
│   ├── MeshGraphNets/{ex1,ex2}/
│   ├── MeshGraphNets-V/b8_all_warpage_input/
│   ├── Neural_Operator/{ex1,ex2}/
│   ├── Transolver/{ex1,ex2}/
│   ├── Geometry_generation/
│   └── benchmarks/{elasticity,plasticity,fno_darcy,gino_carcfd,deeponet_fractional2d}/
│
├── dataset/                      # shared data + format spec + benchmark data
│   ├── DATASET_FORMAT.md
│   ├── ex1.h5, ex2.h5            # canonical mesh datasets (ex1 planar, ex2 true-3D)
│   ├── deepjeb.h5                # SDFFlow geometry dataset
│   ├── hex_dataset.h5, hex_GT.h5
│   └── benchmarks/…              # per-paper validation datasets
│
├── output/                       # run artifacts (checkpoints, rollouts, samples)
└── git-*.sh                      # auto push/pull/fresh-start helpers
```

The directory name **`MeshGraphNets - variational/` literally contains spaces** —
always quote it in shell commands and rely on `pathlib`, never string
concatenation.

---

## 3. The unified launcher (`cae_suite/`)

The launcher is the piece that makes "one command for five methods" real. It is
pure orchestration and validation — it never imports torch or any ML module in
its own process.

### 3.1 The launch pipeline

`run_preflight()` in [preflight.py](cae_suite/preflight.py) drives the entire
flow. The defining property is that it runs checks **in layers and
short-circuits: each layer runs only if no blocking error has been recorded so
far.** This is why a broken config produces a clean, ordered set of problems
instead of a confusing cascade.

```
parse            config_parser.parse_config → ParsedConfig
  │              (values + per-key source location + duplicate detection)
  ▼
route            registry.resolve(model) → ResolvedMethod (spec + repo + entrypoint)
  │              + mode presence/validity check
  ▼
spec layer       _validate_spec: required fields, unknown keys, recommended keys,
  │              defaults notices, then the spec's custom validators
  ▼  (only if no errors)
filesystem       path_checks.validate_paths: input files/dirs exist,
  │              output dirs writable — per the spec's PathRules
  ▼  (only if no errors)
environment      _probe_environment: import the spec's modules IN THE METHOD'S
  │              PYTHON, report missing deps + CUDA device visibility vs gpu_ids
  ▼  (only if no errors)
dataset          _probe_dataset: HDF5 schema probe + feature-count / node-type /
  │              temporal (input_var==output_var) cross-checks against the config
  ▼  (only if no errors)
checkpoint       _probe_checkpoints: safe weights_only metadata read; model/stage/
  │              normalization consistency vs the config
  ▼  (only if no errors)
native probe     _probe_native: run the METHOD'S OWN config validator as a subprocess
  │
  ▼
command          [python, entrypoint, --config, <original config path>]
```

If every layer passes, `cli.main` calls `launch_and_wait` and the native process
starts. `--check` stops after validation; `--dry-run` additionally prints the
exact native command; `--explain-config` prints a configured/defaulted/inactive
breakdown.

### 3.2 Module-by-module

| Module | Responsibility |
| --- | --- |
| [cli.py](cae_suite/cli.py) | Arg parsing; the standalone subcommands (`--list-models`, `--describe`, `--audit-configs`); rendering the route + report; and the diagnostic-prefix → exit-code mapping. |
| [config_parser.py](cae_suite/config_parser.py) | Parses flat `key value` text into a `ParsedConfig` (values, raw values, per-key `SourceLocation`, duplicate list). **Deliberately mirrors the native parsers' quirks** (§4) while adding stricter diagnostics (duplicate keys, BOM, malformed lines). |
| [registry.py](cae_suite/registry.py) | `MethodRegistry` builds a `model_id → MethodSpec` map from the five `build_*_spec()` functions. Aliased IDs (the four neural-operator names) share one spec. `resolve()` emits `ROUTE-*` errors for missing/unknown models and missing repos/entrypoints, with `difflib` "did you mean" hints. |
| [specs/base.py](cae_suite/specs/base.py) | Defines `MethodSpec` (the per-method validation contract), `PathRule`/`PathKind`, `SpecValidationContext`, and shared value validators (`validate_common_values` — gpu_ids rules, positive-int/number checks, `feature_loss_weights` length). |
| [specs/*.py](cae_suite/specs/) | One spec per method: `known_keys`, required/recommended/default fields (per mode and per model), `PathRule`s, `import_modules`, `dataset_kind`, and custom `validators`. **This is the single source of truth for config validation.** |
| [preflight.py](cae_suite/preflight.py) | `run_preflight` — the layered pipeline above — plus the four probe helpers and the dataset/config cross-checks. Builds the final `command`. |
| [path_checks.py](cae_suite/path_checks.py) | Resolves and checks every `PathRule` (input file/dir must exist; output dir must be creatable/writable), returning the resolved path map used by the dataset/checkpoint probes. |
| [diagnostics.py](cae_suite/diagnostics.py) | `Severity` (ERROR/WARNING/NOTICE), `Diagnostic` (with `promote_in_strict` and `effective_severity`), `DiagnosticReport`, JSON export, and the colored terminal renderer. |
| [settings.py](cae_suite/settings.py) | Loads `ai_cae4all.local.toml` and resolves the Python interpreter per method (§3.5). |
| [launcher.py](cae_suite/launcher.py) | `launch_and_wait`: runs the child in its own process group so Ctrl-C is forwarded and escalated cleanly on both Windows and POSIX. |
| native/dataset/checkpoint `*_probe.py` | Standalone scripts executed **under the method's own Python** (§3.4). |

### 3.3 CLI reference

Everything routes through the launcher. `--config` selects the file; `mode`
(train/inference/sample/…) lives **inside the config**, never on the CLI.

```bash
# Validate all applicable checks and report every problem together (no launch):
python AI_CAE4ALL_main.py --config configs/Transolver/ex2/config_train_transolver.txt --check

# Print the exact native command without launching:
python AI_CAE4ALL_main.py --config configs/Neural_Operator/ex1/config_train_fno.txt --dry-run

# A clean preflight auto-launches the native process:
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train1.txt

# Introspection (no config needed):
python AI_CAE4ALL_main.py --list-models        # registered models + install health
python AI_CAE4ALL_main.py --describe transolver # routing + required fields per mode
python AI_CAE4ALL_main.py --audit-configs       # structural lint of checked-in configs
```

Other flags: `--strict` (promote flagged warnings to errors), `--explain-config`
(key-by-key breakdown), `--show-defaults`, `--json-report PATH`, `--python PATH`
(interpreter override), `--no-color`, and `--skip-{native,filesystem,environment}-check`
to bypass a probe layer.

**A note on `--audit-configs`:** it walks *inside the method repositories* for
`config*.txt`. The current templates live under the root `configs/` tree, so the
audit may report `files=0` for those. Use `--check` per file for the centralized
templates.

### 3.4 Probes run in the *target method's* venv — not the launcher's

Three helper scripts are executed as **subprocesses under the resolved method
Python**, so the launcher validates against the exact environment the model will
run in:

- [native_probe.py](cae_suite/native_probe.py) — imports and runs the method's
  own native config validator, emitting a `__CAE_SUITE_NATIVE_RESULT__` JSON
  line. This is why a config can pass the suite's spec checks yet still be caught
  by the method's stricter native rules (and vice-versa — the two are kept
  deliberately close but not identical).
- [dataset_probe.py](cae_suite/dataset_probe.py) — opens the HDF5 dataset and
  returns its schema + `nodal_shape`, which preflight cross-checks against
  `input_var`/`output_var`/`use_node_types` and static-vs-temporal intent.
- [checkpoint_probe.py](cae_suite/checkpoint_probe.py) — a **safe
  `weights_only` metadata read** (never `weights_only=False`): it surfaces the
  checkpoint's model name, `stage` (vae/fm), normalization presence, and EMA
  presence, so a config pointed at the wrong checkpoint fails preflight instead
  of deep in native inference. The native runtime remains the authoritative
  checkpoint loader.

### 3.5 Per-method Python interpreters

Copy [ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored) to point each method at its own venv:

```toml
[python]
default = ".venv/bin/python"

[python.models]
meshgraphnets   = "MeshGraphNets/.venv/bin/python"
meshgraphnets-v = "MeshGraphNets - variational/.venv/bin/python"
neural_operator = "Neural_Operator/.venv/bin/python"
transolver      = "Transolver/.venv/bin/python"
sdfflow         = "Geometry_generation/.venv/bin/python"
```

Interpreter precedence: `--python` → exact model ID → method/spec ID →
`python.default` → the suite process's own `sys.executable`. When you launch from
an already-activated venv with no local TOML, that venv's Python is used for
everything.

**Critical subtlety** (documented at length in
[settings.py](cae_suite/settings.py)): `resolve_python` **intentionally never
calls `Path.resolve()`** on the interpreter path. A venv's `bin/python` is a
symlink; CPython discovers `pyvenv.cfg` by walking up from the *invoked*
executable path, so dereferencing the symlink would silently drop the venv's
site-packages. The launcher keeps the symlink intact. The legacy filename
`cae_suite.local.toml` is still accepted.

### 3.6 Diagnostic codes → exit codes

`_preflight_exit_code` in [cli.py](cae_suite/cli.py) maps the first matching
error class:

| First matching error prefix | Exit code | Meaning |
| --- | --- | --- |
| `ROUTE-*` | 3 | Bad/unknown `model`, missing repo or entrypoint |
| `ENV-*` | 4 | Interpreter/dependency/CUDA problem |
| `NATIVE-CHECK-*` | 5 | The method's own validator rejected the config |
| any other error | 2 | Spec/filesystem/dataset/checkpoint error |
| (no errors) | 0 | Validated / launched successfully |

**Preserve these prefixes when adding checks** — tooling and CI key off them.

---

## 4. The config system

All five native loaders read the same flat `key value` text format, and the
suite parser mirrors that format's quirks exactly (with stricter diagnostics on
top). Getting these quirks right matters because the same file is passed
unchanged to the native process.

### 4.1 Grammar and quirks

| Rule | Behavior |
| --- | --- |
| Keys | Lowercased. `Training_epochs` == `training_epochs`. |
| String values | Also lowercased (matters for case-sensitive Linux paths). |
| Comments | A line starting with `%` is ignored; text after `#` is stripped; blank lines ignored. |
| Separator | First whitespace splits key from value (tabs or spaces). |
| Lists | Commas **or** multiple space-separated tokens → a list (`gpu_ids 0,1`). |
| Booleans | Only case-insensitive `true`/`false` become `bool`. |
| **Single value → scalar** | A single value parses to a **bare scalar, not a one-element list** (`test_batch_idx 0` → `0`). Consumers that expect a list must normalize. |
| **int vs str for numerics** | `100` → `int`; `1e-4` has no `.`, fails the int/float fast paths, and **stays a `str`** — every numeric consumer converts explicitly (`float(config.get(...))`). Prefer decimal notation like `0.0001`. |
| Quoting | Not syntax; quotes become literal characters. Do not quote paths (even paths with spaces). |
| Duplicate keys | Native loaders silently keep the last; the **suite treats it as a blocking error**. |
| BOM | A UTF-8 BOM is a **hard error** (native parsers misread the first key). |
| `reserved` | A key literally named `reserved` is ignored. |

### 4.2 Path resolution & checkpoint-led architecture

- The suite launches the native process **with the method repo as the working
  directory**, and passes the config file unchanged. Relative paths inside a
  config are therefore relative to the *method repo*, not the config file and not
  the suite root. This is why centralized templates use paths like
  `../output/meshgraphnets/ex1/model1.pth`.
- **For inference, architecture and normalization metadata are
  checkpoint-led.** Deterministic MGN, variational MGN, and Transolver overlay
  `checkpoint['model_config']`; Neural Operator rebuilds from versioned
  `data_config`/`adapter_config`/`model_config`; SDFFlow rebuilds from the
  checkpoint's stored config and prefers `ema_state`. Config-file architecture
  values may be *replaced* after the checkpoint loads. Runtime-only choices
  (inference dataset, output dir, rollout length, sample count, temperature)
  stay config-controlled.

### 4.3 The necessity legend (used throughout CONFIGURATION_REFERENCE.md)

Config keys are classified by whether an authored config should contain them:

`R` required · `R*` runtime-required but missing from the suite's required
check · `C` conditionally required · `O` optional/active · `M` compatibility
marker (accepted but selects no live behavior) · `D` derived/injected · `I`
accepted but inert · `L` legacy/removed & silently ignored · `X`
rejected/unsupported · `G` schema gap (meaningful in code but not safely
authorable through every path).

See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for the full
per-key table for every method.

---

## 5. Conventions shared across the method repos

Four of the five method repos (all but the generative one) share the same
*structural skeleton* and the same **mesh HDF5 data contract** — a deliberate
design so that switching methods rarely means relearning the plumbing.

### 5.1 The common repo skeleton

```text
<Method>/
├── <Method>_main.py  or  main.py   # config load → mode dispatch → (DDP spawn)
├── general_modules/                # data loading, config parsing/validation, stats
│   ├── load_config.py              #   native flat-text parser
│   ├── mesh_dataset.py             #   HDF5 → graph/tensor samples, split, normalize
│   ├── dataset_stats.py            #   train-split moments / normalizers
│   ├── time_integration.py         #   AR-OT vs AR-RT selection (§9.1)
│   └── positional_features.py      #   rotation-invariant node features
├── model/                          # the architecture(s)
├── training_profiles/             # single-GPU + DDP + AR-rollout training loops
├── inference_profiles/            # rollout / decode / sampling
├── parallelism/                    # DDP launch, model-split pipeline, checkpoint I/O
├── tests/                          # per-repo pytest suite (synthetic fixtures)
└── misc/                           # analysis, benchmarking, comparison scripts
```

`Neural_Operator/`, `MeshGraphNets/`, and `MeshGraphNets - variational/` ship
`tests/`; consult each repo's own docs for its exact validation command set.

### 5.2 The shared mesh HDF5 contract

MeshGraphNets, MeshGraphNets-variational, Transolver, and Neural_Operator all
read the **same** layout with **no conversion step** (full spec in
[dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)):

```text
dataset.h5
  attrs: num_samples, num_features, num_timesteps
  data/{sample_id}/
    nodal_data   # shape [num_features, num_timesteps, num_nodes]
    mesh_edge    # shape [2, E]  (unique undirected edges)
    metadata/…   # per-sample source + size + summary stats
  metadata/
    feature_names
    normalization_params/{min,max,mean,std, + train-derived node/edge/delta stats}
    splits/{train,val,test}   # present but NOT consumed by current loaders
```

Key facts:

- `nodal_data` rows: `0:3` are **reference coordinates** (`pos`, not part of
  `input_var`); the standard builder writes 8 rows — `x,y,z, dx,dy,dz, scalar,
  part_no`.
- **Static (`T==1`)** targets are the stored field from a zero physical input.
  **Temporal (`T>1`)** targets are the delta `state[t+1] - state[t]`.
- `mesh_edge` stores topology only; the 8-D edge *attributes*
  (`deformed_{dx,dy,dz,dist}, ref_{dx,dy,dz,dist}`) are recomputed on the fly and
  never stored. The same edge-feature function serves mesh, world, and coarse
  multiscale edges.
- The current loaders **ignore the stored `metadata/splits`** and always
  recompute a deterministic seeded 80/10/10 split (`split_seed`, default 42) from
  sorted sample IDs.
- Training-derived normalizers (node/edge/delta z-scores) are what live in
  `checkpoint['normalization']`; MGN also writes them back into the source HDF5,
  Neural_Operator and Transolver never modify the source file.

---

## 6. Method 1 — MeshGraphNets (deterministic)

`model meshgraphnets` → `MeshGraphNets/MeshGraphNets_main.py`. Modes: `train`,
`inference`.

An encode-process-decode GNN simulator in the classic MeshGraphNets style,
substantially extended.

- **Architecture** ([model/MeshGraphNets.py](MeshGraphNets/model/MeshGraphNets.py),
  `encoder_decoder.py`, `blocks.py`, `mlp.py`): node/edge encoders → a stack of
  message-passing `GnBlock`s → a decoder producing a normalized **delta**. For
  temporal delta prediction the final decode layer is initialized near zero
  (weights ×0.01) to start from "no change".
- **Multiscale V-cycle processor** ([model/coarsening.py](MeshGraphNets/model/coarsening.py),
  `multiscale_helpers.py`, `multiscale_cache.py`): optional hierarchical
  coarsening (`use_multiscale`) with `bfs` or three `voronoi_*` strategies, a
  down-arm/coarsest/up-arm block layout (`mp_per_level` must equal
  `2*multiscale_levels+1`), and per-worker + on-disk hierarchy caching.
- **World edges** ([general_modules/world_edges.py](MeshGraphNets/general_modules/world_edges.py)):
  optional non-mesh radius edges from deformed positions (`use_world_edges`,
  backends `scipy_kdtree`/`torch_cluster`), for contact-like interactions.
- **Training** ([training_profiles/](MeshGraphNets/training_profiles/)):
  single-GPU and DDP loops, AMP (bf16), `torch.compile`, EMA, gradient
  accumulation, input-noise regularization (`std_noise`/`noise_gamma`),
  geometry augmentation, and AR-OT/AR-RT time integration (§9.1).
- **Parallelism** ([parallelism/](MeshGraphNets/parallelism/)): `ddp` or
  `model_split` (a 1F1B pipeline across ≥2 GPUs); merged checkpoints load like
  single-GPU ones.
- **Removed-feature guard**
  ([general_modules/removed_feature_guard.py](MeshGraphNets/general_modules/removed_feature_guard.py)):
  hard-rejects VAE/prior keys, so a variational config accidentally routed here
  fails loudly rather than silently ignoring settings.

---

## 7. Method 2 — MeshGraphNets (variational)

`model meshgraphnets-v` → `MeshGraphNets - variational/MeshGraphNets_main.py`.
Modes: `train`, `inference`.

A probabilistic superset of the deterministic simulator. It keeps all the graph,
hierarchy, world-edge, and runtime machinery from Method 1 and adds a
**variational latent path** so it can model a *distribution* of plausible
trajectories, not a single deterministic one.

- **VAE path** ([model/vae.py](MeshGraphNets%20-%20variational/model/vae.py)):
  `use_vae True` activates a posterior graph-encoder, a stochastic latent `z`,
  and variational losses — reconstruction (`huber`/`mse`), aggregate-posterior
  **MMD** (`lambda_mmd`, `mmd_bandwidth`), an auxiliary latent-stats term
  (`beta_aux`), and KL. Latent width/depth via `vae_latent_dim`/`vae_mp_layers`.
- **Learned conditional prior**
  ([model/conditional_prior.py](MeshGraphNets%20-%20variational/model/conditional_prior.py)):
  instead of sampling `z ~ N(0,I)`, a graph-conditional prior can be jointly
  trained (`prior_type gnn_e2e`) in two families — a **flow-matching** prior
  (`prior_family fm`) or a **GMM** prior (`prior_family gmm`) — each with its own
  set of `prior_*` controls.
- **Stochastic inference**: `num_vae_samples` trajectories per scene, with an
  auto-batching path (`vae_batch_size auto` + `vae_batch_vram_fraction`) and an
  optional inline generated-vs-ground-truth displacement-spread histogram
  (`eval_dataset`, `make_histogram`).
- **Time integration under the VAE** (§9.1): the latent is **resampled at every
  unrolled step**; per-step gradient checkpointing preserves RNG so the backward
  draw matches the forward. Loss composition is unchanged and averaged over the
  trajectory. The posterior encoder conditions on `graph.y`, which the rollout
  writes per step so the encoder sees the correction the model is actually being
  asked to make.
- **Docs**: this repo ships an unusually rich `docs/` folder (architecture,
  distribution-modeling research, VRAM/perf optimization, world edges,
  multiscale coarsening, adaptive-remeshing plan, config reference).

> **Known perf gap** (from prior investigation): the variational training tree
> lags the vanilla one on several hot-path optimizations (per-batch `.item()`
> sync, hardcoded `pin_memory`, older concat-style blocks, per-block `Data`
> objects). It uses `torch.optim.Adam` and never reads `weight_decay`.

---

## 8. Method 3 — Neural_Operator (four architectures)

`model point_deeponet | deeponet | fno | gino` → `Neural_Operator/main.py`.
Modes: `train`, `inference`. This is the most self-documented method (it has its
own [CLAUDE.md](Neural_Operator/CLAUDE.md), `docs/`, and the largest test suite).

**One repo, four selectable operator architectures**, all reading the shared
mesh HDF5 with no conversion and sharing one
split/target/normalization/noise/optimizer/scheduler/checkpoint/rollout
convention. Switching `model` must never require touching dataset, training-loop,
loss, checkpoint, or inference code. The repo is fully self-contained: **FNO and
GINO are implemented natively** (`model/spectral.py`, `model/gno.py`) — no
`neuraloperator` dependency, no network access.

| Architecture | File(s) | Idea |
| --- | --- | --- |
| **Point-DeepONet** (primary) | `model/point_deeponet.py`, `pointnet.py`, `siren.py` | PointNet branch (encodes the geometry as sensor points) + SIREN trunk (query coordinates) with early fusion. |
| **DeepONet** (canonical) | `model/deeponet.py` | Fixed regular sensor grid → branch MLP; trunk MLP over query coords; modal dot-product. |
| **FNO** | `model/fno.py` + `model/spectral.py` | Mesh splatted onto a regular grid; native spectral (Fourier) convolutions; sampled back to query points. |
| **GINO** | `model/gino.py` + `model/gno.py` | GNO kernel-integral in ↔ latent FNO ↔ GNO out; mesh→grid→query via radius neighborhoods. |

Design pillars from the repo's own notes:

- **`model/factory.py` is the only place models are constructed**
  (`MODEL_REGISTRY` + per-model `VALIDATORS`).
- **`model/operator_wrapper.py`** owns the noise contract and batch/ptr
  synthesis — the only thing training/inference calls; every core's
  `forward(graph) -> [sum_N, output_var]` is noise-free.
- **`DataSpec`** (`general_modules/data_spec.py`, immutable) is the single source
  of truth for channel widths / active axes; adapters slice `x` via
  `physical_slice`/`context_slice`/`onehot_slice`, never magic offsets.
- **`config_validation.py`** is a strict key registry: unknown or removed-MGN/VAE
  keys raise *before* any HDF5 is opened.
- The **grid axis-order convention** (adapters) is the single most bug-prone
  spot; its docstring must be read before touching splat/sample code.
- EMA copies BatchNorm running stats after every update (PointNet needs this);
  spectral weights are stored real (fused AdamW rejects complex params).
- **`parallel_mode model_split`** (FNO/GINO only) partitions the sequential
  latent stack into a 1F1B pipeline; DeepONets and `augment_geometry True` are
  rejected there.
- `ex1.h5` is planar (`operator_dim` resolves to 2); `ex2.h5` is genuinely 3D —
  both discovered from geometry, neither hardcoded.

---

## 9. Method 4 — Transolver

`model transolver` → `Transolver/Transolver_main.py`. Modes: `train`,
`inference`.

A transformer surrogate built around **Physics-Attention**
([model/physics_attention.py](Transolver/model/physics_attention.py)): each layer
softly assigns mesh nodes to a small learned set of "physics slices"
(`slice_num`), attends over those slices, and scatters back — turning
`O(N²)` node attention into `O(N·slice_num)`.

- **Two numerically-exact attention kernels** sharing one v1-layout state dict:
  `naive` (project-then-aggregate; default at small/medium meshes) and
  `slice_space` (aggregate-then-project, chunked; required for tiling/node
  sharding). Both operate **per graph, segmented by `ptr`**, so nodes never mix
  across graphs in a batch.
- **Node-shard parallelism** (`parallel_mode node_shard`, alias `model_split`):
  one mesh's nodes are split across ≥2 GPUs and the slice aggregates are
  autograd-aware SUM all-reduced, reproducing single-process results bit-for-bit.
  Requires `attention_kernel slice_space`.
- **Memory characteristic** (from prior investigation): the naive kernel holds
  `[H, N, slice_num]` fp32 per layer, so VRAM scales with `B·L·H·N·slice_num` —
  `slice_num`/`num_layers` (not `latent_dim`) drive memory; deep/wide configs
  need activation checkpointing.
- Uses AdamW; slice-assignment temperature is annealed
  (`temperature_init/min/max`); inference supports `direct` and `decoupled`
  modes.

---

## 10. Method 5 — Geometry_generation (SDFFlow)

`model sdfflow` → `Geometry_generation/SDFFlow_main.py`. Modes: `train`,
`train_vae`, `train_fm`, `sample`, `reconstruct`, `interpolate`. Own docs:
[Geometry_generation/CLAUDE.md](Geometry_generation/CLAUDE.md).

Unlike the other four (which predict *fields on a given mesh*), SDFFlow
**generates new 3D shapes**. It is a two-stage generative model over signed
distance functions (SDFs), plus a different data contract.

**Stage 1 — SDF-VAE** ([model/sdf_vae.py](Geometry_generation/model/sdf_vae.py)):
a transformer encoder consumes surface points and produces a compact latent
(`latent_tokens × latent_dim`; the shipped model uses one global token). MLP or
attention SDF decoders reconstruct the signed distance field. Training supports
deterministic / posterior-noise / KL warmups.

**Stage 2 — Flow Matching** ([model/velocity_net.py](Geometry_generation/model/velocity_net.py)):
a **rectified-flow** velocity network with AdaLN-Zero blocks learns to transport
noise → the VAE latent distribution, optionally conditioned on geometric
descriptors. FM consumes *normalized encoder means* (not posterior samples).

**Data & conditioning**
([general_modules/sdf_sampling.py](Geometry_generation/general_modules/sdf_sampling.py),
`sdf_dataset.py`): HDF5 layout is
`shapes/{index:05d}/{surface_points,surface_normals,sdf_points,sdf_values,cond}`;
`cond` holds five raw descriptors `bbox_x, bbox_y, bbox_z, volume, area`. FM may
select a subset via `condition_names` (the shipped DeepJEB config uses
`bbox_x,bbox_z,volume,area`). SDF sign is **negative inside**, positive outside;
shapes occupy ≈`[-0.9,0.9]³`, queries cover `[-1,1]³`.

**The merged training pipeline** — this is the production path and the
distinctive bit
([training_profiles/train_pipeline.py](Geometry_generation/training_profiles/train_pipeline.py)):
`mode train` builds two derived stage configs (every `vae_<x>`/`fm_<x>` key loses
its prefix for the matching stage), then:

1. Inspects `vae_modelpath` for expected stage/epochs/compat fields.
2. Trains or **reuses** the VAE per `skip_completed_stages`.
3. **Refuses to start FM unless the VAE checkpoint verifies complete.**
4. Frees stage memory before FM.
5. Reuses FM only if the VAE was reused *and* the FM checkpoint is complete and
   compatible — a newly trained VAE always invalidates old FM reuse.

This keeps the GPU busy end-to-end and prevents stale VAE/FM pairings.

**Inference modes**: `sample` (with OOD guarding via `max_condition_z` +
`condition_ood_policy`, candidate ranking, and marching-cubes meshing),
`reconstruct` (mesh → VAE → mesh, no FM needed), and `interpolate` (reproducible
`torch.lerp` in normalized FM latent space → three STLs + a triptych PNG).
Marching cubes via
[general_modules/mesh_extraction.py](Geometry_generation/general_modules/mesh_extraction.py).

---

## 11. Configs & benchmarks

Config templates live under [configs/](configs/), organized by method and
example dataset. Approximate inventory:

| Location | Files | Notes |
| --- | --- | --- |
| `configs/MeshGraphNets/{ex1,ex2}/` | ~25 | Train/inference for both example datasets |
| `configs/MeshGraphNets-V/b8_all_warpage_input/` | ~24 | Variational sweeps (displacement-only, `input_var 3`/`output_var 3`) |
| `configs/Neural_Operator/{ex1,ex2}/` | ~17 | Point-DeepONet/DeepONet/FNO/GINO, incl. smoke configs |
| `configs/Transolver/{ex1,ex2}/` | ~13 | Includes an `ex2_sweep` |
| `configs/Geometry_generation/` | 5 | `config_train`, `config_sample`, `config_sample_extrapolation`, `config_interpolate`, (+reconstruct) |
| `configs/benchmarks/…` | ~34 | Per-paper validation: elasticity, plasticity, fno_darcy, gino_carcfd, deeponet_fractional2d |

The **benchmark** configs (and their data under
[dataset/benchmarks/](dataset/benchmarks/), with a
`PER_PAPER_VALIDATION_REPORT.md`) exist to reproduce published operator-learning
results. Note the caveat from the config reference: several benchmark-only keys
(`split_strategy hdf5`, `loss_type relative_l2`, `relative_l2_epsilon`) are
**intent markers, not runnable** — the native validators reject or ignore them,
so those files capture benchmark *intent* more than a currently-executable
config.

---

## 12. Datasets present

| File | Used by | Notes |
| --- | --- | --- |
| `dataset/ex1.h5` | mesh methods | **Planar** geometry (z≡0 → operator_dim 2) |
| `dataset/ex2.h5` | mesh methods | Genuinely **3D** geometry |
| `dataset/ex2.mscache.*.h5` | MGN | Cached multiscale hierarchy for ex2 |
| `dataset/deepjeb.h5` | SDFFlow | Geometry-generation shapes + descriptors |
| `dataset/hex_dataset.h5`, `hex_GT.h5` | mesh methods | Hex-mesh dataset + ground truth |
| `dataset/benchmarks/…` | Neural_Operator benchmarks | Per-paper validation datasets |

Dataset builders live in the method repos (`Geometry_generation/build_dataset.py`
for SDFFlow) and under `dataset/` for the mesh methods.

---

## 13. Testing

There is **no root-level test suite** — the `testpaths = ["tests"]` line in
[pyproject.toml](pyproject.toml) is stale (no root `tests/` dir exists). Tests
live per method repo and run in that repo's venv:

```bash
cd Neural_Operator && pytest tests/          # ~35 tests, tiny synthetic HDF5 fixtures, <1 min
cd Geometry_generation && python -m pytest -q tests/test_sdfflow_pipeline.py
```

`Neural_Operator/` has by far the deepest coverage (config validation, coordinate
domain, grid adapter, spectral/FNO/GINO/GNO, DeepONet, point sampling, ragged
batching, radius neighbors, checkpoint roundtrip, EMA buffers, model split,
AR-rollout, and per-paper validations). MeshGraphNets and its variational sibling
ship AR-rollout and multiscale-stats tests.

---

## 14. Known gaps & mismatches (the honest list)

These are documented here so they are not rediscovered the hard way; the
authoritative, exhaustive version is in
[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md).

- **`num_workers` is `R*`** for mesh/operator training: the native loop indexes
  it directly, but the suite's required-field check does not enforce it — a
  config omitting it can pass preflight and fail natively.
- **`infer_timesteps` is runtime-required for static `T=1` data** even though the
  suite does not require it.
- **`--audit-configs` searches inside method repos**, so it reports `files=0` for
  the centralized `configs/` templates.
- **Benchmark-intent keys** (`split_strategy`, `loss_type`,
  `relative_l2_epsilon`) are not implemented in the stable runtime.
- **Schema-gap keys** (e.g. Neural_Operator `use_parallel_stats`/
  `train_eval_subset_size`, Transolver `test_batch_idx`,
  GINO `gino_transform_type`) are read by code but omitted from the native key
  registry and/or suite schema — they cannot currently be authored cleanly.
- Variational MGN ignores `weight_decay` (it uses `Adam`); a set of legacy VAE
  keys are silently ignored by its runtime but warned (and, under `--strict`,
  rejected) by the suite.

---

## 15. Development workflow — when you change something

From [CLAUDE.md](CLAUDE.md), the rules that keep the launcher and the native code
in sync:

- **New/renamed native config key** → update that method's spec `known_keys` (and
  required/validator lists) in [cae_suite/specs/](cae_suite/specs/), or the
  launcher will reject a valid config or accept an invalid one. Then re-run the
  relevant `--check`/audit.
- **New method repo** → add a `build_*_spec()` and register it in
  [registry.py](cae_suite/registry.py)'s `MethodRegistry.__init__`.
- **Changing behavior inside a method repo** → follow that repo's own `CLAUDE.md`
  and run its own tests; the launcher change (if any) is usually just the spec.
- **Preserve diagnostic-code prefixes** (`ROUTE-`, `ENV-`, `NATIVE-CHECK-`, …) —
  the exit-code mapping and any tooling depend on them.
- When a method repo's `CLAUDE.md` and its code disagree, the **code is
  authoritative for current behavior**, but treat the mismatch as something to
  reconcile, not ignore.

### Routing quick reference

| `model` value | Repo | Entrypoint | Modes |
| --- | --- | --- | --- |
| `meshgraphnets` | `MeshGraphNets/` | `MeshGraphNets_main.py` | train, inference |
| `meshgraphnets-v` | `MeshGraphNets - variational/` | `MeshGraphNets_main.py` | train, inference |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `Neural_Operator/` | `main.py` | train, inference |
| `transolver` | `Transolver/` | `Transolver_main.py` | train, inference |
| `sdfflow` | `Geometry_generation/` | `SDFFlow_main.py` | train, train_vae, train_fm, sample, reconstruct, interpolate |

---

## Appendix A — Cross-cutting feature: AR-OT vs AR-RT time integration

Active in both MeshGraphNets variants, Transolver, and all four Neural Operator
models; meaningful only for temporal datasets (`num_timesteps > 1`). Selected by
a single config key: `time_integration ar_ot` (default) or `ar_rt`.

- **AR-OT** (Auto-Regressive One-step Teacher-forcing) trains on ground-truth
  consecutive pairs; each `(t, t+1)` pair is its own dataset item, so a `T`-step
  trajectory yields `T-1` optimizer steps. Inference first feeds the model its
  own predictions, and `std_noise` exists to paper over that train/test mismatch.
- **AR-RT** (Auto-Regressive Rollout Training, following NVIDIA/GM
  arXiv:2510.15201) unrolls the model over the whole trajectory during training,
  consuming its own predictions exactly as inference does, and backpropagates
  through all steps (gradient-checkpointed per step; no noise injected). The
  whole trajectory is **one dataset item** → one forward, one averaged loss, one
  backward. Per-epoch node evaluations are unchanged, but an epoch performs ≈`T`×
  fewer optimizer steps — raise `training_epochs` accordingly. Validation loss
  becomes the rollout loss, so best-checkpoint selection optimizes rollout
  accuracy.
- Under `ar_rt`: `std_noise`/`noise_gamma` are inert; MeshGraphNets rebuilds mesh
  edge features, world edges, and coarse features per step from each predicted
  state (`coarse_world_edges True` is rejected); DDP forces `static_graph=True`;
  and variational MGN resamples the latent every step (Appendix, §7).

## Appendix B — Cross-cutting feature: parallelism modes

| Mode | Methods | Meaning |
| --- | --- | --- |
| `ddp` (default) | all | Data-parallel; one full model per GPU. |
| `model_split` | MGN, MGN-v, Neural_Operator (FNO/GINO only) | 1F1B pipeline across ≥2 GPUs; the model is cut into pipeline blocks; merged checkpoints load like single-GPU ones. Rejected for DeepONets and with `augment_geometry True`. |
| `node_shard` (Transolver; `model_split` is an alias) | Transolver | One mesh's nodes are sharded across ≥2 GPUs; slice aggregates are autograd-aware all-reduced. Requires `attention_kernel slice_space`. |

Model-split effective batch size is `batch_size × pipeline_microbatches`
(default microbatches `2 × num_stages`).

---

*Generated from a live read of the repository. For any specific config key,
`CONFIGURATION_REFERENCE.md` is authoritative; for any method's internals, that
method's own `CLAUDE.md` / `docs/` and its code are authoritative.*
