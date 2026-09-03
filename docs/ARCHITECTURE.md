# AI-CAE4ALL — Repository Overview & Architecture Guide

> A comprehensive, code-grounded tour of the whole repository: what it is, how
> it is laid out, how the unified launcher works, and what each ML method
> repository does. This document is the **map**; its companions are the
> **territory**:
>
> - [../README.md](../README.md) — the 60-second quick start.
> - [CONFIGURATION.md](CONFIGURATION.md) — the live configuration grammar,
>   route map, validation workflow, and source pointers.
> - [reference/DATASET_FORMAT.md](reference/DATASET_FORMAT.md) — the shared mesh
>   HDF5 data contract.
> - [guides/testing.md](guides/testing.md) — what to run after a change.
> - [../CLAUDE.md](../CLAUDE.md) — condensed agent-facing conventions for the
>   root launcher.
>
> Each method repo additionally has its own authoritative docs; those are
> canonical for that method's data contract and architecture.

---

## 1. What this repository is

**AI-CAE4ALL is a monorepo of nine independent machine-learning-for-CAE
(Computer-Aided Engineering) method repositories, unified by a single
config-driven launcher.** The intent is that an engineer picks a method by
writing one word in a text config, and one command validates and runs it — no
matter which of eight very different ML codebases actually executes.

The eight methods span the major families of ML surrogate modeling for
simulation:

| Family | Method | What it predicts |
| --- | --- | --- |
| Graph neural network simulator | MeshGraphNets (deterministic) | Time evolution of physical fields on a mesh |
| Probabilistic GNN simulator | MeshGraphNets (variational) | A *distribution* of plausible mesh trajectories (VAE + learned prior) |
| Conditional flow GNN | cHI-MGNflow | Conditional distributions of mesh fields without the retired VAE/prior branch |
| Neural operators | Neural_Operator (4 architectures) | Field-to-field mappings that generalize across discretizations |
| Transformer surrogate | Transolver | Mesh fields via learned "physics slices" + attention |
| Generative geometry | SDFFlow | *New 3D shapes* via an SDF-VAE and flow matching |
| Fixed-geometry latent model | SimulGenVAE | Parametric static/transient fields plus a latent conditioner |
| Parametric regression | MLP | **N scalar inputs → M scalar outputs** (tabular DOE surrogate, no mesh) |

Six of the eight predict **fields on a mesh** and share related HDF5 contracts.
The exceptions are geometry-generating SDFFlow and **MLP**, a tabular
parameters→outputs regressor
(`MLP/`, modes `train`/`inference`) that reads a separate `X[S,N]`/`Y[S,M]` HDF5,
stores normalization in its checkpoint, and needs no GPU. See
[MLP/CLAUDE.md](../methods/MLP/CLAUDE.md),
[docs/methods/12_MLP.md](methods/12_MLP.md), and
[CONFIGURATION.md](CONFIGURATION.md).

Alongside the ML methods, the same launcher routes one **non-ML data-prep
utility**, `geometry_ingest` (`methods/GeometryIngest/`, modes `ingest`/`inspect`):
it meshes CAD/geometry (STEP/IGES/STL) into the shared mesh HDF5 contract — a graph
for MeshGraphNets and a point cloud for the operators/Transolver — so a raw part can
feed any method with no conversion step. It trains nothing and needs no GPU. See
[methods/GeometryIngest/README.md](../methods/GeometryIngest/README.md),
[docs/methods/11_Geometry_Ingest.md](methods/11_Geometry_Ingest.md), and
[CONFIGURATION.md](CONFIGURATION.md).

### 1.1 The core idea: one launcher, eight native runtimes

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

File and line totals drift with every implementation pass, so this guide does
not preserve an old snapshot. Use `git ls-files '*.py'` for tracked Python
files, `git ls-files 'configs/**/config*.txt'` for checked-in templates, and
`python AI_CAE4ALL_main.py --audit-configs --no-color` for the authoritative
live config audit. The route registry currently exposes 12 model IDs across
eight ML repositories plus `geometry_ingest`.

---

## 2. Top-level repository map

```text
AI-CAE4ALL/
├── AI_CAE4ALL_main.py            # 8-line shim → cae_suite.cli.main
├── pyproject.toml                # installs the `ai-cae4all` console script
├── ai_cae4all.local.example.toml # template for per-method interpreter paths
├── README.md                     # quick start and model zoo
├── CLAUDE.md                     # agent-facing root conventions
│
├── cae_suite/                    # THE LAUNCHER (parse→route→preflight→subprocess)
│   ├── cli.py                    #   arg parsing, subcommands, exit-code mapping
│   ├── config_parser.py          #   flat `key value` parser (mirrors native quirks)
│   ├── config_discovery.py       #   locates checked-in configs for the audit
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
│       ├── meshgraphnets.py            meshgraphnets_variational.py
│       ├── neural_operator.py          transolver.py
│       ├── sdfflow.py                  chi_mgnflow.py
│       ├── simulgenvae.py              geometry_ingest.py
│       └── mlp.py
│
├── methods/                      # THE NINE NATIVE RUNTIMES (each standalone)
│   ├── MeshGraphNets/            #   model = meshgraphnets
│   ├── MeshGraphNets_Variational/#   model = meshgraphnets-v
│   ├── HI_MGNFlow/               #   model = chi-mgnflow      (entrypoint CHiMGNFlow_main.py)
│   ├── Neural_Operator/          #   model = point_deeponet | deeponet | fno | gino
│   ├── Transolver/               #   model = transolver
│   ├── SDFFlow/                  #   model = sdfflow          (generative + `optimize` loop)
│   ├── SimulGenVAE/              #   model = simulgenvae
│   ├── MLP/                      #   model = mlp              (tabular, not mesh)
│   └── GeometryIngest/           #   model = geometry_ingest  (non-ML data prep)
│
├── configs/                      # one directory per method, mirroring methods/
│   ├── MeshGraphNets/{ex1..ex9}/       MeshGraphNets_Variational/
│   ├── Neural_Operator/{ex1..ex9}/     Transolver/{ex1..ex9}/
│   ├── HI_MGNFlow/                     SDFFlow/
│   ├── SimulGenVAE/                    MLP/            GeometryIngest/
│   └── campaigns/                #   multi-arm runners: ex1, ex2, ex3, benchmarks_all
│
├── dataset/                      # shared HDF5 data (payloads are git-ignored)
│   ├── ex1.h5 … ex9.h5           #   training datasets, one per experiment slot
│   ├── ex1_infer.h5 … ex9_infer.h5  # the matching held-out inference inputs
│   └── deepjeb.h5, deepjeb_mgn.h5   # SDFFlow geometry + its MGN surrogate bridge
│
├── output/                       # THE ONLY artifact root: checkpoints, logs, rollouts, samples
├── studio/                       # the Studio: browser UI + local Python API bridge
├── inference/                    # stand-alone CPU inference bundle + PyInstaller spec
├── docs/                         # all documentation (this file included)
│   ├── ARCHITECTURE.md CONFIGURATION.md README.md
│   ├── guides/ reference/ methods/ research/ images/
└── tests/                        # launcher / MethodSpec contract tests
```

Two layout invariants are worth stating explicitly:

- **`configs/<Name>/` mirrors `methods/<Name>/`.** Adding a method means adding
  both, under the same name.
- **Nothing writes inside a method directory.** Native config paths resolve from
  the method repository, so every artifact path is spelled `../../output/...`
  and lands in the single root `output/`.

---

## 3. The unified launcher (`cae_suite/`)

The launcher is the piece that makes one command route all supported methods. It is
pure orchestration and validation — it never imports torch or any ML module in
its own process.

### 3.1 The launch pipeline

`run_preflight()` in [preflight.py](../cae_suite/preflight.py) drives the entire
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
| [cli.py](../cae_suite/cli.py) | Arg parsing; the standalone subcommands (`--list-models`, `--describe`, `--audit-configs`); rendering the route + report; and the diagnostic-prefix → exit-code mapping. |
| [config_parser.py](../cae_suite/config_parser.py) | Parses flat `key value` text into a `ParsedConfig` (values, raw values, per-key `SourceLocation`, duplicate list). **Deliberately mirrors the native parsers' quirks** (§4) while adding stricter diagnostics (duplicate keys, BOM, malformed lines). |
| [registry.py](../cae_suite/registry.py) | `MethodRegistry` builds a `model_id → MethodSpec` map from the registered `build_*_spec()` functions. Aliased IDs (the four neural-operator names) share one spec. `resolve()` emits `ROUTE-*` errors for missing/unknown models and missing repos/entrypoints, with `difflib` "did you mean" hints. |
| [specs/base.py](../cae_suite/specs/base.py) | Defines `MethodSpec` (the per-method validation contract), `PathRule`/`PathKind`, `SpecValidationContext`, and shared value validators (`validate_common_values` — gpu_ids rules, positive-int/number checks, `feature_loss_weights` length). |
| [specs/*.py](../cae_suite/specs) | One spec per method: `known_keys`, required/recommended/default fields (per mode and per model), `PathRule`s, `import_modules`, `dataset_kind`, and custom `validators`. **This is the single source of truth for config validation.** |
| [preflight.py](../cae_suite/preflight.py) | `run_preflight` — the layered pipeline above — plus the four probe helpers and the dataset/config cross-checks. Builds the final `command`. |
| [path_checks.py](../cae_suite/path_checks.py) | Resolves and checks every `PathRule` (input file/dir must exist; output dir must be creatable/writable), returning the resolved path map used by the dataset/checkpoint probes. |
| [diagnostics.py](../cae_suite/diagnostics.py) | `Severity` (ERROR/WARNING/NOTICE), `Diagnostic` (with `promote_in_strict` and `effective_severity`), `DiagnosticReport`, JSON export, and the colored terminal renderer. |
| [settings.py](../cae_suite/settings.py) | Loads `ai_cae4all.local.toml` and resolves the Python interpreter per method (§3.5). |
| [launcher.py](../cae_suite/launcher.py) | `launch_and_wait`: runs the child in its own process group so Ctrl-C is forwarded and escalated cleanly on both Windows and POSIX. |
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

**A note on `--audit-configs`:** config discovery is shared by the CLI and
Studio and walks the root `configs/**/config*.txt` tree. Use `--check` for one
file and `--audit-configs` for the full checked-in set.

### 3.4 Probes run in the *target method's* venv — not the launcher's

Three helper scripts are executed as **subprocesses under the resolved method
Python**, so the launcher validates against the exact environment the model will
run in:

- [native_probe.py](../cae_suite/native_probe.py) — imports and runs the method's
  own native config validator, emitting a `__CAE_SUITE_NATIVE_RESULT__` JSON
  line. This is why a config can pass the suite's spec checks yet still be caught
  by the method's stricter native rules (and vice-versa — the two are kept
  deliberately close but not identical).
- [dataset_probe.py](../cae_suite/dataset_probe.py) — opens the HDF5 dataset and
  returns its schema + `nodal_shape`, which preflight cross-checks against
  `input_var`/`output_var`/`use_node_types` and static-vs-temporal intent.
- [checkpoint_probe.py](../cae_suite/checkpoint_probe.py) — a **safe
  `weights_only` metadata read** (never `weights_only=False`): it surfaces the
  checkpoint's model name, `stage` (vae/fm), normalization presence, and EMA
  presence, so a config pointed at the wrong checkpoint fails preflight instead
  of deep in native inference. The native runtime remains the authoritative
  checkpoint loader.

### 3.5 Per-method Python interpreters

Copy [ai_cae4all.local.example.toml](../ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored) to point each method at its own venv:

```toml
[python]
default = ".venv/bin/python"

[python.models]
meshgraphnets   = "MeshGraphNets/.venv/bin/python"
meshgraphnets-v = "methods/MeshGraphNets_Variational/.venv/bin/python"
neural_operator = "Neural_Operator/.venv/bin/python"
transolver      = "Transolver/.venv/bin/python"
sdfflow         = "methods/SDFFlow/.venv/bin/python"
```

Interpreter precedence: `--python` → exact model ID → method/spec ID →
`python.default` → the suite process's own `sys.executable`. When you launch from
an already-activated venv with no local TOML, that venv's Python is used for
everything.

**Critical subtlety** (documented at length in
[settings.py](../cae_suite/settings.py)): `resolve_python` **intentionally never
calls `Path.resolve()`** on the interpreter path. A venv's `bin/python` is a
symlink; CPython discovers `pyvenv.cfg` by walking up from the *invoked*
executable path, so dereferencing the symlink would silently drop the venv's
site-packages. The launcher keeps the symlink intact. The legacy filename
`cae_suite.local.toml` is still accepted.

### 3.6 Diagnostic codes → exit codes

`_preflight_exit_code` in [cli.py](../cae_suite/cli.py) maps the first matching
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

All six native loaders read the same flat `key value` text format, and the
suite parser mirrors that format's quirks exactly (with stricter diagnostics on
top). Getting these quirks right matters because the same file is passed
unchanged to the native process.

### 4.1 Grammar and quirks

| Rule | Behavior |
| --- | --- |
| Keys | Lowercased. `Training_epochs` == `training_epochs`. |
| String values | Also lowercased — **except path-valued keys**, which keep the case you wrote (`dataset_dir`, `infer_dataset`, `modelpath`, the `*_modelpath` variants, `input_mesh`, `sdf_sidecar`, `param_dir`, and the output/log dirs). The exempt set is `cae_suite/config_parser.py::PATH_KEYS`, mirrored per repo in each `load_config.py`. |
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

### 4.3 Live key status

`MethodSpec.known_keys` is a diagnostic catalog, not a promise that every key
is active for every model or mode. Some shared-family keys are retained so the
validator can report “inactive” or “removed” precisely instead of calling them
unknown. Required fields, published defaults, conditional checks, and path
roles live in `cae_suite/specs/`; Studio reads the same contracts and keeps
variant-inactive keys visible but disabled. Use `--check`, `--explain-config`,
or Studio's full configuration sheet for the current status of an authored
key. [CONFIGURATION.md](CONFIGURATION.md) links those live
sources without copying a drift-prone per-key snapshot.

---

## 5. Conventions shared across the method repos

Five method repos — deterministic and variational MeshGraphNets, cHI-MGNflow,
Neural Operator, and Transolver — share the same broad *structural skeleton*
and **mesh HDF5 data contract**. SimulGenVAE consumes a related fixed-geometry
field contract through a different loader; SDFFlow is generative and MLP is
tabular.

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

`Neural_Operator/`, `MeshGraphNets/`, and `methods/MeshGraphNets_Variational/` ship
`tests/`; consult each repo's own docs for its exact validation command set.

### 5.2 The shared mesh HDF5 contract

MeshGraphNets, MeshGraphNets-variational, Transolver, and Neural_Operator all
read the **same** layout with **no conversion step** (full spec in
[dataset/DATASET_FORMAT.md](reference/DATASET_FORMAT.md)):

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

- **Architecture** ([model/MeshGraphNets.py](../methods/MeshGraphNets/model/MeshGraphNets.py),
  `encoder_decoder.py`, `blocks.py`, `mlp.py`): node/edge encoders → a stack of
  message-passing `GnBlock`s → a decoder producing a normalized **delta**. For
  temporal delta prediction the final decode layer is initialized near zero
  (weights ×0.01) to start from "no change".
- **Multiscale V-cycle processor** ([model/coarsening.py](../methods/MeshGraphNets/model/coarsening.py),
  `multiscale_helpers.py`, `multiscale_cache.py`): optional hierarchical
  coarsening (`use_multiscale`) with `bfs` or three `voronoi_*` strategies, a
  down-arm/coarsest/up-arm block layout (`mp_per_level` must equal
  `2*multiscale_levels+1`), and per-worker + on-disk hierarchy caching.
- **Learned inter-level transfer operators** (`pool_type`, `unpool_type`,
  `pool_heads`): the restriction was a fixed mean-pool and the prolongation
  aggregation a fixed sum, while everything around them was learned. Both can
  now be multi-head attention over the cluster / over each fine node's coarse
  sources. Zero-initialized score heads make either reduce to the fixed
  operator *exactly* at step 0, so enabling one starts training from the
  previous model rather than from a different basin.
- **Multi-partition coarse representation** (`voronoi_branches`): a level can
  hold `K` parallel Voronoi partitions of the same node set instead of one,
  merged by a widened `skip_projs`. The motivation is that a single `k`
  controls two things at once — how much field information the coarse level
  retains and how short its graph diameter is — and `K` partitions of `k`
  clusters improve both together relative to one partition of `K*k`. Only the
  deepest configured level may branch; branching an earlier level would fork
  every level beneath it, so it is rejected in the hierarchy builder, the model
  constructor, and the dataset.
  See [ATTENTION_TRANSFER_DESIGN.md](research/meshgraphnets/ATTENTION_TRANSFER_DESIGN.md)
  for the design, the measurements behind the defaults, and the as-built notes.
- **World edges** ([general_modules/world_edges.py](../methods/MeshGraphNets/general_modules/world_edges.py)):
  optional non-mesh radius edges from deformed positions (`use_world_edges`,
  backends `scipy_kdtree`/`torch_cluster`), for contact-like interactions.
- **Training** ([training_profiles/](../methods/MeshGraphNets/training_profiles)):
  single-GPU and DDP loops, AMP (bf16), `torch.compile`, EMA, gradient
  accumulation, input-noise regularization (`std_noise`/`noise_gamma`),
  geometry augmentation, and AR-OT/AR-RT time integration (§9.1).
- **Parallelism** ([parallelism/](../methods/MeshGraphNets/parallelism)): `ddp` or
  `model_split` (a 1F1B pipeline across ≥2 GPUs); merged checkpoints load like
  single-GPU ones.
- **Removed-feature guard**
  ([general_modules/removed_feature_guard.py](../methods/MeshGraphNets/general_modules/removed_feature_guard.py)):
  hard-rejects VAE/prior keys, so a variational config accidentally routed here
  fails loudly rather than silently ignoring settings.
- **Concurrent-job safety.** Training writes normalization statistics back into
  the *source* HDF5 (`write_preprocessing_to_hdf5`). DDP restricts that to rank
  0, but independent jobs on the same dataset — which the ablation sweep runs
  eight of at once — had no guard, and HDF5's own file locking does not block a
  second `'r+'` open here (verified on HDF5 1.14.6). The write is now serialized
  by an exclusive lock file, the same pattern `multiscale_cache.ensure_cache`
  already used for the shared hierarchy cache.

---

## 7. Method 2 — MeshGraphNets (variational)

`model meshgraphnets-v` → `methods/MeshGraphNets_Variational/MeshGraphNets_main.py`.
Modes: `train`, `inference`.

A probabilistic superset of the deterministic simulator. It keeps all the graph,
hierarchy, world-edge, and runtime machinery from Method 1 and adds a
**variational latent path** so it can model a *distribution* of plausible
trajectories, not a single deterministic one.

- **VAE path** ([model/vae.py](../methods/MeshGraphNets_Variational/model/vae.py)):
  `use_vae True` activates a posterior graph-encoder, a stochastic latent `z`,
  and variational losses — reconstruction (`huber`/`mse`), aggregate-posterior
  **MMD** (`lambda_mmd`, `mmd_bandwidth`), an auxiliary latent-stats term
  (`beta_aux`), and KL. Latent width/depth via `vae_latent_dim`/`vae_mp_layers`.
- **Learned conditional prior**
  ([model/conditional_prior.py](../methods/MeshGraphNets_Variational/model/conditional_prior.py)):
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
> objects). It uses `torch.optim.Adam`; configured `weight_decay` is coupled L2
> decay and defaults to zero when omitted.

---

## 8. Method 3 — Neural Operator (four architectures)

`model point_deeponet | deeponet | fno | gino` → `Neural_Operator/main.py`.
Modes: `train`, `inference`. This is the most self-documented method (it has its
own [CLAUDE.md](../methods/Neural_Operator/CLAUDE.md), `docs/`, and the largest test suite).

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
([model/physics_attention.py](../methods/Transolver/model/physics_attention.py)): each layer
softly assigns mesh nodes to a small learned set of "physics slices"
(`slice_num`), attends over those slices, and scatters back — turning
`O(N²)` node attention into `O(N·slice_num)`.

- **Two numerically-exact attention kernels** sharing one v1-layout state dict:
  `slice_space` (Transolver-3's aggregate-then-project; **the default**, and the
  only kernel that can tile, shard, or amortize) and `naive` (v1's
  project-then-aggregate). Both operate **per graph, segmented by `ptr`**, so
  nodes never mix across graphs in a batch, and they agree to fp64 round-off
  (`misc/verify_v3.py` L1), so the choice is memory/speed, never results.
- **Node-shard parallelism** (`parallel_mode node_shard`, alias `model_split`):
  one mesh's nodes are split across ≥2 GPUs and the slice aggregates are
  autograd-aware SUM all-reduced, reproducing single-process results bit-for-bit.
  Requires `attention_kernel slice_space`.
- **Memory characteristic**: the dominant term is the `[H, N, slice_num]` fp32
  slice-weight matrix per layer, so VRAM scales with `B·L·H·N·slice_num` —
  `slice_num`/`num_layers` (not `latent_dim`) drive memory. `chunk_size` tiles
  that matrix and **streams** it: each tile is dropped once folded into the
  aggregates and rebuilt in backward, so the N-scaled attention term leaves
  retained memory (measured 1049 MB → 64 MB) and peak becomes near-independent
  of `slice_num`. This needs no block checkpointing — `use_checkpointing` is
  orthogonal. Measured in `methods/Transolver/misc/verify_v3.py` L5 and CONFIGURATION.md 8.4.
- **Amortized training** (`amortized_training`): builds each layer's physics
  tokens from a subsampled *cache* node stream and computes the loss on a
  smaller decoded *query* stream, cutting activations to
  `O(L·(cache+query)·C)`. Training-only — eval and inference always run the full
  mesh. See CONFIGURATION.md 8.4.
- Uses AdamW; slice-assignment temperature is annealed
  (`temperature_init/min/max`); inference supports `direct` and `decoupled`
  modes.

---

## 10. Method 5 — SDFFlow (generative geometry)

`model sdfflow` → `methods/SDFFlow/SDFFlow_main.py`. Modes: `train`,
`train_vae`, `train_fm`, `sample`, `reconstruct`, `interpolate`. Own docs:
[methods/SDFFlow/CLAUDE.md](../methods/SDFFlow/CLAUDE.md).

Unlike the other four (which predict *fields on a given mesh*), SDFFlow
**generates new 3D shapes**. It is a two-stage generative model over signed
distance functions (SDFs), plus a different data contract.

**Stage 1 — SDF-VAE** ([model/sdf_vae.py](../methods/SDFFlow/model/sdf_vae.py)):
a transformer encoder consumes surface points and produces a compact latent
(`latent_tokens × latent_dim`; the shipped model uses one global token). MLP or
attention SDF decoders reconstruct the signed distance field. Training supports
deterministic / posterior-noise / KL warmups.

**Stage 2 — Flow Matching** ([model/velocity_net.py](../methods/SDFFlow/model/velocity_net.py)):
a **rectified-flow** velocity network with AdaLN-Zero blocks learns to transport
noise → the VAE latent distribution, optionally conditioned on geometric
descriptors. FM consumes *normalized encoder means* (not posterior samples).

**Data & conditioning**
([general_modules/sdf_sampling.py](../methods/SDFFlow/general_modules/sdf_sampling.py),
`sdf_dataset.py`): HDF5 layout is
`shapes/{index:05d}/{surface_points,surface_normals,sdf_points,sdf_values,cond}`;
`cond` holds five raw descriptors `bbox_x, bbox_y, bbox_z, volume, area`. FM may
select a subset via `condition_names` (the shipped DeepJEB config uses
`bbox_x,bbox_z,volume,area`). SDF sign is **negative inside**, positive outside;
shapes occupy ≈`[-0.9,0.9]³`, queries cover `[-1,1]³`.

**The merged training pipeline** — this is the production path and the
distinctive bit
([training_profiles/train_pipeline.py](../methods/SDFFlow/training_profiles/train_pipeline.py)):
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
[general_modules/mesh_extraction.py](../methods/SDFFlow/general_modules/mesh_extraction.py).

---

## 11. Configs & benchmarks

Config templates live under [configs/](../configs), organized by method and
dataset/campaign. Counts drift; use
`python AI_CAE4ALL_main.py --audit-configs --no-color` for the live inventory.

| Location | Coverage |
| --- | --- |
| `configs/MeshGraphNets/ex1` … `ex9` | Deterministic and Hi-MGN train/inference profiles, including the ex1 ablation (§11.1) |
| `configs/MeshGraphNets_Variational/` | B8 and SAOI variational training/inference campaigns |
| `configs/HI_MGNFlow/` | DeepJEB, ex9, SAOI, and wave0 flow campaigns |
| `configs/Neural_Operator/ex1` … `ex9` | Point-DeepONet, DeepONet, FNO, and GINO profiles |
| `configs/Transolver/ex1` … `ex9` | Transolver training/inference profiles |
| `configs/SDFFlow/` | SDFFlow train, sample, reconstruct, interpolate, and optimize profiles |
| `configs/SimulGenVAE/`, `configs/MLP/`, `configs/GeometryIngest/` | Fixed-geometry latent, tabular, and ingestion workflows |
| `configs/campaigns/benchmarks_all/` | Cross-method campaign scheduling, roster, inference, and scoring helpers; no bundled paper dataset or reproduced-result report |

Paper-profile modules and optional tests exist, but the previously described
`dataset/benchmarks/` evidence bundle is not present in this checkout. Do not
infer a reproduced paper metric from those structural tests or campaign tools.

### 11.1 The current Hi-MGN ex1 ablation

The executable study in this checkout is the ex1-only campaign documented by
[its runbook](../configs/MeshGraphNets/ex1/ABLATION.md) and driven by
[`ablation.py`](../configs/MeshGraphNets/ex1/ablation.py). It compares five axes:
hierarchy depth, message-passing placement, coarsening operator, total
message-passing budget, and learned versus broadcast interpolation.

The driver defines 11 run entries representing nine distinct configurations.
The baseline is run three times to expose unseeded training scatter. `gen`
derives one training and one inference file per entry from `config_train1.txt`,
so the generated set contains 22 configs; edit the base or the driver's `ARMS`
table rather than editing a generated arm.

[`run_ablation.sh`](../configs/MeshGraphNets/ex1/run_ablation.sh) deliberately runs
the checked-in, frozen configs through cost, train, inference, evaluation, and
report stages. Generation is a separate explicit step. Reports are written
under `output/meshgraphnets/ex1/ablation/` and include R2, RMSE, peak error,
stress-specific R2, parameter count, and estimated forward FLOPs. No completed
ablation scores are claimed here; consult a generated report artifact for a
particular run.

---

## 12. Datasets present

| File | Used by | Notes |
| --- | --- | --- |
| `dataset/ex1.h5` | mesh methods | **Planar** geometry (z≡0 → operator_dim 2) |
| `dataset/ex2.h5` | mesh methods | Genuinely **3D** geometry |
| `dataset/ex*.mscache.*.h5` | MGN | Transient per-run hierarchy cache — deleted at end of training; leftovers mean a killed run and are pruned on next start |
| `dataset/deepjeb.h5` | SDFFlow | Geometry-generation shapes + descriptors |
| `dataset/ex1_infer.h5` | mesh methods | ex1 single-sample hex-mesh inference input; state rows carry the ground-truth field (was `hex_GT.h5`) |
| `dataset/ex2_infer.h5` | mesh methods | ex2 held-out inference set: 5 unseen scenes × 50 timesteps; rollout is seeded from t=0 and scored against t=1..49 |
| `dataset/hex_dataset.h5` | mesh methods | Same mesh with the state rows zeroed — legacy, superseded by `ex1_infer.h5` |

Dataset builders live in the method repos (`methods/SDFFlow/build_dataset.py`
for SDFFlow) and under `dataset/` for the mesh methods.

---

## 13. Testing

The root `tests/` suite covers launcher and `MethodSpec` contracts. Each method
also has native tests that should run from that method's directory:

```bash
python -m pytest -q
python -m pytest -q studio/studio_backend
cd Neural_Operator && pytest tests/
cd methods/SDFFlow && python -m pytest -q tests/
```

`Neural_Operator/` has by far the deepest coverage (config validation, coordinate
domain, grid adapter, spectral/FNO/GINO/GNO, DeepONet, point sampling, ragged
batching, radius neighbors, checkpoint roundtrip, EMA buffers, model split,
AR-rollout, and paper-profile contract tests). MeshGraphNets and its variational sibling
ship AR-rollout and multiscale-stats tests.

---

## 14. Known gaps & mismatches (the honest list)

These are documented here so they are not rediscovered the hard way;
[CONFIGURATION.md](CONFIGURATION.md) points to the live
validators and native contracts.

- **Benchmark-intent keys** (`split_strategy`, `loss_type`,
  `relative_l2_epsilon`) are not implemented in the stable runtime.
- A set of legacy variational-GNN keys are inactive and warned (or rejected
  under `--strict`) by the suite rather than silently changing training.

---

## 15. Development workflow — when you change something

From [CLAUDE.md](../CLAUDE.md), the rules that keep the launcher and the native code
in sync:

- **New/renamed native config key** → update that method's spec `known_keys` (and
  required/validator lists) in [cae_suite/specs/](../cae_suite/specs), or the
  launcher will reject a valid config or accept an invalid one. Then re-run the
  relevant `--check`/audit.
- **New method repo** → add a `build_*_spec()` and register it in
  [registry.py](../cae_suite/registry.py)'s `MethodRegistry.__init__`.
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
| `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | `MeshGraphNets_main.py` | train, inference |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `Neural_Operator/` | `main.py` | train, inference |
| `transolver` | `Transolver/` | `Transolver_main.py` | train, inference |
| `sdfflow` | `methods/SDFFlow/` | `SDFFlow_main.py` | train, train_vae, train_fm, sample, reconstruct, interpolate |
| `mlp` | `MLP/` | `MLP_main.py` | train, inference |
| `geometry_ingest` | `methods/GeometryIngest/` | `main.py` | ingest, inspect (non-ML data-prep) |

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
`CONFIGURATION.md` points to the executable contracts; for any
method's internals, that method's own `CLAUDE.md` / `docs/` and code are
authoritative.*
