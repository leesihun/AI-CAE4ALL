# Architecture

How AI-CAE4ALL is put together: the repository layout, the launcher that routes
every config, the config grammar every method shares, and the conventions that
cut across methods. What each method does inside is in
[methods/](methods/README.md), one document per method.

Related documents:

- [../README.md](../README.md): quick start.
- [CONFIGURATION.md](CONFIGURATION.md): config workflow, the route map, where
  each key is defined.
- [reference/DATASET_FORMAT.md](reference/DATASET_FORMAT.md): the shared mesh
  HDF5 contract.
- [guides/testing.md](guides/testing.md): what to run after a change.
- [GUI.md](GUI.md): the Studio, a browser front end over the same launcher.

---

## 1. What the repository is

AI-CAE4ALL is a monorepo of independent ML-for-CAE method repositories behind
one config-driven launcher. You name a method in one line of a text config
(`model transolver`). One command then validates the config against that
method's real environment and runs the method's own entrypoint.

| Family | Route(s) | Predicts |
| --- | --- | --- |
| Graph network simulator | `meshgraphnets` | Fields on a mesh, static or rolled out in time. MGN, HI-MGN (`use_multiscale`), BSMS-GNN (`coarsening_type bfs`) |
| Probabilistic graph simulator | `meshgraphnets-v` | A distribution of fields: a conditional graph VAE with a flow-matching prior. `latent_type` global / local / hybrid = HI-MGN-V / HI-MGNflow / HI-MGN-hybrid |
| Neural operators | `deeponet`, `point_deeponet`, `fno` | Field-to-field maps on point clouds or grids |
| Transformer surrogate | `transolver` | Mesh fields via physics-attention slices |
| Generative geometry | `sdfflow` | New 3D shapes (SDF VAE + flow matching), plus the closed-loop `optimize` mode |
| Fixed-geometry latent model | `simulgenvae` | Dense fields on one shared mesh, from a hierarchical VAE and a latent conditioner |
| Tabular regression | `mlp` | `X[S,N]` → `Y[S,M]`, no mesh, no GPU |
| Data prep (non-ML) | `geometry_ingest` | CAD (STEP/IGES/STL) → shared mesh HDF5 |

`sar` and `ldgn` are optional, local-only baselines (git-ignored). They route
only on machines where `methods/SAR/` or `methods/LDGN/` exists.

`--list-models` prints the live route table and the install health of each
route. HI-MGNflow had its own route (`chi-mgnflow`) until 2026-10-19. It is now
`meshgraphnets-v` with `latent_type local`. The old repository is in git
history at commit `9db47d6`.

### The core idea

The launcher (`ai_cae4all/`, run as `AI_CAE4ALL_main.py` or as the
`ai-cae4all` console script) never imports ML code. Its job is:

```text
parse config → route on `model` → layered preflight → build native command → subprocess
```

Each method is self-contained: its own venv, its own tests, its own entrypoint.
Each one also runs directly without the launcher. The launcher adds two things:

- It reports every problem in a config before any GPU time is spent.
- It always launches the native process in that method's directory, under that
  method's Python interpreter.

---

## 2. Repository map

```text
AI-CAE4ALL/
├── AI_CAE4ALL_main.py            # shim → ai_cae4all.cli.main
├── pyproject.toml                # `pip install -e .` adds the `ai-cae4all` command
├── ai_cae4all.local.example.toml # template for per-method interpreters
├── ai_cae4all/                   # the launcher (§3)
│   ├── cli.py  config_parser.py  config_discovery.py  registry.py
│   ├── preflight.py  path_checks.py  diagnostics.py  settings.py  launcher.py
│   ├── native_probe.py  dataset_probe.py  checkpoint_probe.py
│   └── specs/                    # one MethodSpec per method: validation truth
├── methods/                      # the native runtimes, each standalone
│   ├── MeshGraphNets/  MeshGraphNets_Variational/  Neural_Operator/  Transolver/
│   ├── SDFFlow/  SimulGenVAE/  MLP/  GeometryIngest/
│   └── SAR/  LDGN/               # optional, git-ignored
├── configs/                      # mirrors methods/, plus campaigns/
│   ├── <Method>/deterministic|probabilistic|geometry_generation/exN/...
│   ├── MeshGraphNets_Variational/hyperparameter_sweep/   # r8–r13 B300 rounds
│   └── campaigns/dataset_matrix/ # cross-method campaign (manifest.json)
├── dataset/                      # local-only HDF5 inputs (§6)
├── output/                       # the only artifact root
├── studio/                       # browser Studio over the same launcher
├── inference/                    # portable CPU inference bundle
└── docs/                         # this documentation
```

There are two layout rules. Breaking either is a bug:

- **`configs/<Name>/` mirrors `methods/<Name>/`.** A new method adds both
  directories under the same name. The only exception is `configs/campaigns/`.
- **Nothing writes inside `methods/`.** The native process runs with cwd
  `methods/<Name>/`, so every artifact path in a config is `../../output/...`
  and every dataset path is `../../dataset/...`.

The names of the `output/` subdirectories are historical and kept on purpose.
For example, `output/geometry_generation/` holds SDFFlow runs and
`output/chi-mgnflow/` holds runs from the retired route. These names identify
runs; they are not directory names in the tree.

---

## 3. The launcher (`ai_cae4all/`)

### 3.1 The preflight pipeline

`run_preflight()` in [preflight.py](../ai_cae4all/preflight.py) runs checks in
layers. A layer runs only if no earlier layer recorded an error. A broken config
therefore gives one ordered set of problems, not a cascade.

```text
parse        config_parser.parse_config → ParsedConfig
  │          (values, per-key source line, duplicate detection)
route        registry.resolve(model) → spec + repo + entrypoint; mode check
  │
spec         required/unknown/recommended keys, defaults, spec validators
  ▼ (no errors so far)
filesystem   path_checks: inputs exist, outputs writable, case mismatches
  ▼
environment  import the spec's modules in the METHOD's Python; CUDA vs gpu_ids
  ▼
dataset      HDF5 schema probe; feature-count / node-type / temporal checks
  ▼
checkpoint   weights_only metadata: model, stage, normalization consistency
  ▼
native       the method's OWN config validator, as a subprocess
  │
command      [python, entrypoint, --config, <config path>]
```

If every layer passes, `cli.main` calls `launch_and_wait`. `--check` stops after
validation, and `--dry-run` also prints the native command.

### 3.2 Modules

| Module | Responsibility |
| --- | --- |
| [cli.py](../ai_cae4all/cli.py) | Arguments, introspection commands (`--list-models`, `--describe`, `--audit-configs`), report rendering, exit codes |
| [config_parser.py](../ai_cae4all/config_parser.py) | Flat `key value` → `ParsedConfig`. Mirrors the native parsers' quirks (§4) and adds stricter diagnostics |
| [config_discovery.py](../ai_cae4all/config_discovery.py) | Finds checked-in `configs/**/config*.txt` for the audit and the Studio |
| [registry.py](../ai_cae4all/registry.py) | `MethodRegistry`: `model` → `MethodSpec`. Aliased IDs share a spec. `default_specs()` adds SAR/LDGN only if their directories exist. Bad IDs get `ROUTE-*` errors with "did you mean" hints |
| [specs/base.py](../ai_cae4all/specs/base.py) | `MethodSpec`, `PathRule`, shared validators (gpu_ids, positive numbers, `feature_loss_weights` length) |
| [specs/*.py](../ai_cae4all/specs) | Per method: `known_keys`, required/recommended/default fields per mode, `PathRule`s, `import_modules`, `dataset_kind`, validators |
| [preflight.py](../ai_cae4all/preflight.py) | The pipeline above; probe helpers; builds the command |
| [path_checks.py](../ai_cae4all/path_checks.py) | Resolves each `PathRule` against the method repo |
| [diagnostics.py](../ai_cae4all/diagnostics.py) | `Severity`, `Diagnostic` (`promote_in_strict`), report, JSON export |
| [settings.py](../ai_cae4all/settings.py) | Reads `ai_cae4all.local.toml`; picks each method's interpreter |
| [launcher.py](../ai_cae4all/launcher.py) | Runs the child in its own process group, so Ctrl-C is forwarded and escalated on Windows and POSIX |

The spec is the single source of truth for validation. A key that is not in
`known_keys` gets a `CFG-UNKNOWN-001` warning.

### 3.3 CLI

`--config` selects the file. `mode` lives inside the config, never on the CLI.

```bash
python AI_CAE4ALL_main.py --config <cfg> --check     # validate; every problem at once
python AI_CAE4ALL_main.py --config <cfg> --dry-run   # also print the native command
python AI_CAE4ALL_main.py --config <cfg>             # clean preflight → launch

python AI_CAE4ALL_main.py --list-models              # routes + install health
python AI_CAE4ALL_main.py --describe transolver      # routing + required fields per mode
python AI_CAE4ALL_main.py --audit-configs            # structural lint of every config*.txt
```

Other flags:

| Flag | Effect |
| --- | --- |
| `--strict` | Promote flagged warnings to errors |
| `--explain-config` | Key-by-key breakdown: configured / defaulted / inactive / checkpoint-owned |
| `--show-defaults` | Expand default and notice diagnostics |
| `--json-report PATH` | Write the report as JSON |
| `--python PATH` | Override the method's interpreter |
| `--no-color` | Plain report |
| `--skip-native-check`, `--skip-filesystem-check`, `--skip-environment-check` | Skip that layer |

### 3.4 Probes run in the method's venv

Three helper scripts run as subprocesses under the resolved method Python. This
way `--check` validates against the environment the model will actually run in.

- [native_probe.py](../ai_cae4all/native_probe.py) imports the method's own
  config validator and prints one result line. A config can pass the spec and
  still fail the native rules, and the reverse can also happen. The two are kept
  close, but they are not identical.
- [dataset_probe.py](../ai_cae4all/dataset_probe.py) opens the HDF5 and returns
  its schema and `nodal_shape`. Preflight checks these against
  `input_var`/`output_var`/`cond_var`/`use_node_types` and against whether the
  config is static or temporal.
- [checkpoint_probe.py](../ai_cae4all/checkpoint_probe.py) reads metadata with
  `weights_only=True`: model name, `stage`, normalization, EMA. A config
  pointing at the wrong checkpoint then fails preflight, not deep inside
  inference. The native runtime remains the authoritative loader.

### 3.5 Per-method interpreters

Copy [ai_cae4all.local.example.toml](../ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored):

```toml
[python]
default = ".venv/bin/python"

[python.models]
meshgraphnets-v = "methods/MeshGraphNets_Variational/.venv/bin/python"
sdfflow         = "methods/SDFFlow/.venv/bin/python"
```

The interpreter is chosen in this order: `--python`, then the exact model ID,
then the spec ID, then `python.default`, then the launcher's own
`sys.executable`. So with no TOML and an activated venv, that venv's Python
runs everything. The legacy filename `cae_suite.local.toml` is still read.

`resolve_python` never calls `Path.resolve()` on the interpreter. A venv's
`python` is a symlink, and CPython finds `pyvenv.cfg` by walking up from the
path it was invoked as. Dereferencing the symlink would silently drop the venv's
site-packages.

### 3.6 Exit codes

`_preflight_exit_code` in [cli.py](../ai_cae4all/cli.py) returns the code for the
first error class that matches:

| First matching error prefix | Exit | Meaning |
| --- | --- | --- |
| `ROUTE-*` | 3 | Unknown `model`, missing repo or entrypoint |
| `ENV-*` | 4 | Interpreter, dependency or CUDA problem |
| `NATIVE-CHECK-*` | 5 | The method's own validator rejected the config |
| any other error | 2 | Spec, filesystem, dataset or checkpoint error |
| none | 0 | Validated or launched |

Keep these prefixes when you add checks; tooling keys off them.

---

## 4. The config system

Every native loader reads the same flat `key value` text. The launcher parser
mirrors its quirks exactly and adds stricter diagnostics on top. This matters
because the same file is passed unchanged to the native process.

### 4.1 Grammar and quirks

| Rule | Behavior |
| --- | --- |
| Keys | Lowercased |
| String values | Lowercased, except path keys, which keep their case. The set is `PATH_KEYS` in `ai_cae4all/config_parser.py`, mirrored in each native `general_modules/load_config.py`. A new path key means editing both |
| Comments | A line starting with `%` is ignored; text after `#` is stripped |
| Separator | The first whitespace splits the key from the value |
| Lists | Commas or several space-separated tokens make a list (`gpu_ids 0,1`) |
| Single value | A bare scalar, not a one-element list (`test_batch_idx 0` → `0`) |
| Numbers | `100` → `int`. `1e-4` stays a **string**, so every numeric consumer converts explicitly (`float(config.get(...))`) |
| Booleans | Only `true`/`false` (any case) become `bool` |
| Quotes | Not syntax: quotes become literal characters. Do not quote paths |
| Duplicate keys | Native loaders keep the last one; the launcher reports an **error** |
| BOM | A UTF-8 BOM is an **error** (native parsers misread the first key) |

MLP and GeometryIngest store raw strings and need no path exemption.
`PATH-CASE-001` warns when the case in a config differs from the disk. That is
fine on Windows and broken on Linux.

### 4.2 Paths and checkpoint-owned settings

- Relative paths resolve from the **method repository** (the native cwd), not
  from the config file. That is why configs say `../../dataset/...` and
  `../../output/...`.
- At inference, architecture and normalization come from the checkpoint. The
  mesh methods overlay `checkpoint['model_config']`, and SDFFlow rebuilds from
  its stored config, preferring the EMA weights. Architecture values in the
  config file may be replaced once the checkpoint loads. Runtime choices stay
  with the config: inference dataset, output directory, rollout length, sample
  count. `--explain-config` marks which keys are checkpoint-owned.

### 4.3 Key status

`known_keys` is a catalog, not a promise that every key is active for every
model and mode. Some keys are kept so the validator can say "inactive" or
"removed" instead of "unknown". For the status of a key, use `--check`,
`--explain-config`, `--describe <model>` or the Studio's configuration sheet.
Per-method key references are in [reference/config/](reference/config/).

---

## 5. Conventions shared across methods

### 5.1 Repository skeleton

The mesh methods (MGN, MGN-V, Neural_Operator, Transolver) share one skeleton:

```text
<Method>/
├── <Method>_main.py | main.py   # load config → dispatch mode → (DDP spawn)
├── general_modules/              # load_config, mesh_dataset, dataset_stats,
│                                 # time_integration, resume_state, ...
├── model/                        # architectures
├── training_profiles/            # single-GPU, DDP, rollout training loops
├── inference_profiles/           # rollout / decode / sampling
├── parallelism/                  # DDP launch, model split, checkpoint I/O
└── tests/                        # pytest on synthetic fixtures
```

SDFFlow, SimulGenVAE and MLP follow the same naming loosely. Each method's own
`CLAUDE.md`/`README.md` describes its tree.

### 5.2 The shared mesh HDF5

MGN, MGN-V, Transolver and Neural_Operator read the same layout with no
conversion step. SimulGenVAE also reads it, but densely. The full specification
is in [reference/DATASET_FORMAT.md](reference/DATASET_FORMAT.md).

```text
data/{sample_id}/nodal_data   [num_features, num_timesteps, num_nodes]
data/{sample_id}/mesh_edge    [2, E]   unique undirected edges
metadata/...                  feature names, normalization params

rows 0:3                          reference coordinates (never predicted)
rows [3 : 3+input_var]            state: input AND output
rows [3+input_var : +cond_var]    conditions: input only (cond_var, default 0)
last row (if present)             node type / part label, copied through
```

- **Static (`T == 1`)** targets are the stored field. **Temporal (`T > 1`)**
  targets are the delta `state[t+1] − state[t]`. `input_var == output_var` is
  required for T > 1 (the autoregressive loop).
- `mesh_edge` stores topology only. Edge attributes (deformed and reference
  displacement and length) are recomputed on the fly.
- The loaders ignore any stored `metadata/splits` and recompute a seeded
  80/10/10 split from sorted sample IDs (`split_seed`, default 42).
- Train-derived normalizers live in `checkpoint['normalization']`.
  `write_preprocessing` may also append them to the HDF5.
- The MGN multiscale hierarchy cache (`<stem>.mscache.<digest>.h5`, written
  beside the dataset or in `hierarchy_cache_dir`) is per-run and deleted when
  training ends. A leftover file means a run was killed. It is pruned on the
  next start.

### 5.3 Mid-training resume

`general_modules/resume_state.py` is copied byte for byte into every training
repo except MLP. The repos share no imports, so an edit to one copy must go to
all of them. With `resume_training true`:

- The trainer writes `<checkpoint>.resume` at the first epoch boundary after
  every `resume_interval_minutes` (default 15).
- It resumes only from a state with the same config fingerprint.
- It deletes the state once the final checkpoint is written.

New mutable training state must go into that trainer's `resume.save({...})`.
[../CLAUDE.md](../CLAUDE.md) has the RNG-ordering rules.

### 5.4 Periodic visualization

`test_interval` gates a pass that writes a picture into the log file's
directory. `display_testset` (default on) enables it, and `display_trainset`
adds training samples under `train/`. Every plotting backend is imported
lazily: a missing matplotlib or pyvista prints a note and the run continues.
Headless, `offscreen_rendering_available()` probes one render in a child process
and skips the PNGs if that process dies.

### 5.5 Time integration: AR-OT vs AR-RT

This applies to MGN, MGN-V, Transolver and the three Neural_Operator models, and
only to temporal datasets. The key is `time_integration ar_ot` (default) or
`ar_rt`.

- **AR-OT** (one-step teacher forcing) trains on ground-truth `(t, t+1)` pairs,
  so each pair is a dataset item. At inference the model consumes its own
  predictions; `std_noise` narrows that gap.
- **AR-RT** (rollout training, arXiv:2510.15201) unrolls the whole trajectory in
  training on the model's own predictions. It backpropagates through all steps,
  with per-step gradient checkpointing. One trajectory is one item, so an epoch
  makes about `T`× fewer optimizer steps; raise `training_epochs` to match.
  Validation loss becomes the rollout loss.
- Under `ar_rt`:
  - `std_noise` and `noise_gamma` are inert.
  - MGN rebuilds edge, world-edge and coarse features from each predicted state.
  - DDP uses `static_graph=True`.
  - MGN-V resamples the posterior latent every training step and draws it once
    per trajectory at inference.

### 5.6 Parallelism

| `parallel_mode` | Methods | Meaning |
| --- | --- | --- |
| `ddp` (default) | all | One full model per GPU |
| `model_split` | MGN, MGN-V (global latent only), Neural_Operator (FNO only) | 1F1B pipeline across ≥ 2 GPUs. Merged checkpoints load like single-GPU ones. Effective batch = `batch_size × pipeline_microbatches` |
| `node_shard` (alias `model_split`) | Transolver | One mesh's nodes sharded across ≥ 2 GPUs. Needs `attention_kernel slice_space` |

SDFFlow and SimulGenVAE fall back to one GPU when `gpu_ids` lists fewer than
two. SimulGenVAE runs multi-GPU in-process via `mp.spawn`.

---

## 6. Datasets

`dataset/` is local-only and git-ignored. A dataset is one cleaned training
`.h5` plus a matching held-out `_infer.h5`. Raw data stays outside the repo.

| Directory | Contents |
| --- | --- |
| `dataset/deterministic/` | ex1 static thermoelastic, ex2 dynamic contact, ex3 NASA CRM (full/mid, plus the MLP tables), ex4 cylinder flow, ex5 deforming plate, ex6 flag, ex7 AirfRANS, ex8 elasticity, ex9 plasticity, ex10 DeepJEB vertical FEA |
| `dataset/probabilistic/` | ex1 turbulent radiative layer, ex2 crack path, ex3 shell buckling (one-to-many targets for MGN-V) |
| `dataset/geometry_generation/` | SDFFlow SDF datasets: ex1 DeepJEB, ex2 DrivAerML, ex3 MCB |
| `dataset/validation/` | `ellipse*.h5`, the upstream SAR/LDGN validation case |
| `dataset/derived/config_matrix/` | Inputs generated for the dataset_matrix campaign: a canonical CRM copy and ex6/ex9 condition sidecars |

[reference/PUBLIC_DATASETS.md](reference/PUBLIC_DATASETS.md) records where the
public datasets came from. The builders live with their method, for example
`methods/SDFFlow/scripts/build_dataset.py` for SDF data and `geometry_ingest`
for CAD.

---

## 7. Methods

Each method's own `CLAUDE.md`/`README.md` is authoritative for its data
contract, architecture and validation. The documents below explain each one.

| Route | Directory | Entrypoint | Modes | Document |
| --- | --- | --- | --- | --- |
| `meshgraphnets` | `methods/MeshGraphNets/` | `MeshGraphNets_main.py` | train, inference | [01](methods/01_MeshGraphNets_MGN.md), [02](methods/02_HI-MGN.md), [03](methods/03_BSMS-GNN.md) |
| `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | `MeshGraphNets_Variational_main.py` | train, inference | [04](methods/04_MeshGraphNets_Variational.md) |
| `deeponet`, `point_deeponet`, `fno` | `methods/Neural_Operator/` | `main.py` | train, inference | [05](methods/05_DeepONet.md), [06](methods/06_Point-DeepONet.md), [07](methods/07_FNO.md) |
| `transolver` | `methods/Transolver/` | `Transolver_main.py` | train, inference | [08](methods/08_Transolver.md) |
| `sdfflow` | `methods/SDFFlow/` | `SDFFlow_main.py` | train, train_vae, train_fm, sample, reconstruct, interpolate, optimize, evaluate | [09](methods/09_SDFFlow.md) |
| `geometry_ingest` | `methods/GeometryIngest/` | `main.py` | ingest, inspect | [10](methods/10_Geometry_Ingest.md) |
| `mlp` | `methods/MLP/` | `MLP_main.py` | train, inference | [11](methods/11_MLP.md) |
| `simulgenvae` | `methods/SimulGenVAE/` | `SimulGenVAE_main.py` | train, train_vae, train_lc, reconstruct | [12](methods/12_SimulGenVAE.md) |

[methods/00_shared_foundations.md](methods/00_shared_foundations.md) covers the
ideas the mesh methods share.

---

## 8. When you change something

- **A new or renamed native key** goes into the spec's `known_keys` (and its
  validator and required lists). Then run `--audit-configs`. A new **path** key
  also goes into both `PATH_KEYS` copies.
- **A new spec enum** is mirrored in `CHOICES` in `studio/src/constants.js`.
- **A new method** needs `methods/<Name>/` and `configs/<Name>/`, plus a
  `build_*_spec()` with `repository="methods/<Name>"`, registered in
  `default_specs()`. It also needs a Studio `TEMPLATES` entry and `HELP` text.
  Restart the Studio afterwards.
- **A behavior change inside a method** follows that method's `CLAUDE.md` and
  runs its tests. The launcher change, if any, is usually just the spec.
- **Keep the diagnostic prefixes** (`ROUTE-`, `ENV-`, `NATIVE-CHECK-`).
- When a method's docs and its code disagree, the code decides current behavior.
  Fix the docs.
