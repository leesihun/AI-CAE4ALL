# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

AI-CAE4ALL is a **monorepo of nine independent ML-for-CAE method repositories**
unified by one config-driven launcher. The launcher (`cae_suite/`, the
`ai-cae4all` console script, `AI_CAE4ALL_main.py`) reads a native flat-text
config, routes on its `model` field to the right method repo, runs a layered
preflight validation, and then **subprocess-launches that repo's native
entrypoint** — it never imports the ML code.

Every method repo is self-contained (own venv, own tests) and also runnable
directly. All nine live under [methods/](methods/); the specs that route to them
live in [cae_suite/specs/](cae_suite/specs/):

| `model` config value(s) | Directory | Entrypoint | Own CLAUDE.md |
| --- | --- | --- | --- |
| `meshgraphnets` | `methods/MeshGraphNets/` | `MeshGraphNets_main.py` | — |
| `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | `MeshGraphNets_main.py` | — |
| `chi-mgnflow` | `methods/HI_MGNFlow/` | `CHiMGNFlow_main.py` | — (see its README.md) |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `methods/Neural_Operator/` | `main.py` | [yes](methods/Neural_Operator/CLAUDE.md) |
| `transolver` | `methods/Transolver/` | `Transolver_main.py` | — |
| `sdfflow` | `methods/SDFFlow/` | `SDFFlow_main.py` | [yes](methods/SDFFlow/CLAUDE.md) |
| `simulgenvae` | `methods/SimulGenVAE/` | `SimulGenVAE_main.py` | [yes](methods/SimulGenVAE/CLAUDE.md) |
| `mlp` | `methods/MLP/` | `MLP_main.py` | [yes](methods/MLP/CLAUDE.md) |
| `geometry_ingest` | `methods/GeometryIngest/` | `main.py` | — (see its [README.md](methods/GeometryIngest/README.md)) |

**Model ID and directory name do not always match**, because model IDs are
frozen by every checked-in config: `chi-mgnflow` lives in `HI_MGNFlow/`,
`sdfflow` in `SDFFlow/` (formerly `Geometry_generation/`), `meshgraphnets-v` in
`MeshGraphNets_Variational/`. The spec's `repository=` field is the mapping.

The first eight are the ML methods. **`simulgenvae`** is a hierarchical VAE +
latent conditioner for parametric simulation fields; structurally it mirrors
`sdfflow` (a VAE stage + a second stage), with modes `train` (VAE→LC pipeline),
`train_vae`, `train_lc`, and `reconstruct`, in-process multi-GPU via `mp.spawn`,
and stage-prefixed `vae_*`/`lc_*` config keys. It reads the **shared mesh HDF5**
(`dataset_kind=mesh_hdf5`) but is a **fixed-geometry dense FOM** model: it flattens
the physical field rows into a dense `[samples, channels, time]` tensor, so every
sample must share the same node and timestep counts. See
[methods/SimulGenVAE/CLAUDE.md](methods/SimulGenVAE/CLAUDE.md).

**`mlp` is the odd one out among them**: a parametric surrogate (N scalar inputs →
M scalar outputs) that is **tabular, not mesh** — it reads an `X[S,N]`/`Y[S,M]`
HDF5 (`dataset_kind=table_hdf5`, `native_probe=False`), not the shared mesh
contract, and needs no GPU. See [methods/MLP/CLAUDE.md](methods/MLP/CLAUDE.md).

**SDFFlow additionally carries the suite's only closed-loop mode.** Alongside
its train/sample/reconstruct/interpolate modes, `mode optimize` chains
generation, gmsh tetrahedral meshing, a linear-static structural solve, and a
CMA-ES search into one config-driven run over the trained DeepJEB generator --
the one place where a method repo *evaluates* geometry rather than only
producing or predicting it. It trains nothing and needs `gmsh`, `pyamg`, and
`cma`. See [methods/SDFFlow/CLAUDE.md](methods/SDFFlow/CLAUDE.md).

**`geometry_ingest` is a non-ML data-prep utility** routed through the same
launcher: it meshes CAD/geometry (STEP/IGES/STL) into the shared mesh HDF5
contract (graph for MeshGraphNets, point cloud for the operators/Transolver). Its
spec sets `native_probe=False` and `dataset_kind=None`, and its modes are
`ingest`/`inspect`.

**When working inside a method repo, its own `CLAUDE.md` is authoritative** for
that method's data contract, architecture facts, and validation steps. This file
covers only the root-level launcher and the cross-cutting conventions.

## Layout invariants

Two rules hold across the whole tree; breaking either is a bug, not a style
choice:

1. **`configs/<Name>/` mirrors `methods/<Name>/`.** Adding a method means adding
   both directories under the same name. `configs/campaigns/` is the one
   exception: it holds multi-arm runners (`ex1`, `ex2`, `ex3`,
   `benchmarks_all`), not per-method configs.
2. **`output/` at the repo root is the only artifact destination.** Nothing is
   written inside a method directory. Because the native process runs with its
   cwd set to `methods/<Name>/`, every artifact path in a config is spelled
   `../../output/...` and every dataset path `../../dataset/...`.

```text
AI-CAE4ALL/
├── AI_CAE4ALL_main.py   cae_suite/      # launcher (no ML imports)
├── methods/             configs/        # nine runtimes, mirrored config dirs
├── dataset/             output/         # inputs (git-ignored), single artifact root
├── studio/              inference/      # browser Studio, portable CPU bundle
├── docs/                tests/          # all documentation, launcher contract tests
```

> The `output/` **subdirectory** names are historical and deliberately left
> alone: `output/geometry_generation/` (SDFFlow), `output/chi-mgnflow/`,
> `output/meshgraphnets-v/`. They name runs, not directories in the tree.

## Commands

Everything routes through the launcher; `--config` selects the file, and `mode`
(train / inference / sample / …) lives *inside* the config, not on the CLI.

```bash
# Validate only (all applicable checks, reports every problem together):
python AI_CAE4ALL_main.py --config configs/Transolver/ex2/config_train_transolver.txt --check

# Show the exact native command without launching:
python AI_CAE4ALL_main.py --config configs/Neural_Operator/ex1/config_train_fno.txt --dry-run

# A clean preflight auto-launches the native process:
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train_himgn_base.txt

# Introspection (no config needed):
python AI_CAE4ALL_main.py --list-models        # registered models + install health
python AI_CAE4ALL_main.py --describe transolver # routing + required fields per mode
python AI_CAE4ALL_main.py --audit-configs      # structural lint of every checked-in config*.txt
```

Useful flags: `--strict` (promote flagged warnings to errors),
`--explain-config` (configured/defaulted/inactive/checkpoint-owned key
breakdown), `--json-report PATH`, and `--skip-{native,filesystem,environment}-check`
to bypass a probe layer. Installing the root package (`python -m pip install -e .`)
also provides the `ai-cae4all` command.

### Tests

The root suite covers launcher and MethodSpec contracts and runs in the
launcher's own interpreter:

```bash
python -m pytest -q tests/
python -m pytest -q studio/studio_backend
```

Method tests run in that method's venv, from that method's directory:

```bash
cd methods/Neural_Operator && python -m pytest -q tests/
cd methods/SDFFlow         && python -m pytest -q tests/
```

Every method except `GeometryIngest` ships a `tests/` package. See
[docs/guides/testing.md](docs/guides/testing.md) for the full per-layer map.

## Launcher architecture (`cae_suite/`)

The launch pipeline is: **parse → resolve → layered preflight → command → subprocess**.

- [config_parser.py](cae_suite/config_parser.py) — parses the flat `key value`
  text into a `ParsedConfig` (values + per-key source locations + duplicate
  detection), deliberately **mirroring the native parsers' quirks** (see
  gotchas below).
- [registry.py](cae_suite/registry.py) — `MethodRegistry` maps `model` →
  `MethodSpec` → repo root + entrypoint. Aliased model IDs (e.g. the four
  neural-operator names) share one spec.
- [specs/base.py](cae_suite/specs/base.py) + `specs/<method>.py` — each
  `MethodSpec` declares `known_keys`, required/recommended/default fields per
  mode/model, `PathRule`s, `import_modules`, `dataset_kind`, and custom
  `validators`. **This is the single source of truth for config validation**:
  adding or renaming a native config key means editing the spec's `known_keys`
  (an unlisted key becomes a `CFG-UNKNOWN-001` warning) and, if it constrains
  behavior, its validator.
- [preflight.py](cae_suite/preflight.py) — `run_preflight` runs checks in
  layers and **short-circuits: each layer runs only if no errors so far**
  (`spec → filesystem paths → environment → dataset → checkpoint → native
  probe`). It builds the final `command` list.
- [path_checks.py](cae_suite/path_checks.py) — resolves native config paths
  against the method repository and flags missing inputs, unwritable outputs,
  and case mismatches.
- [diagnostics.py](cae_suite/diagnostics.py) — `Severity` (ERROR/WARNING/
  NOTICE), `Diagnostic` (with `promote_in_strict`), and the report renderer.
- [cli.py](cae_suite/cli.py) — arg parsing, the introspection subcommands, and
  the exit-code mapping.
- [launcher.py](cae_suite/launcher.py) — `launch_and_wait`; the child runs in
  its own process group so Ctrl-C is forwarded and escalated cleanly on both
  Windows and POSIX.
- [settings.py](cae_suite/settings.py) — loads `ai_cae4all.local.toml` to pick
  each method's Python interpreter.

### Probes run in the *target method's* venv, not the launcher's

Three helper scripts are executed as **subprocesses under the resolved method
Python**, so the launcher validates against the environment the model will
actually run in: [native_probe.py](cae_suite/native_probe.py) (imports the
method's native config validator), [dataset_probe.py](cae_suite/dataset_probe.py)
(HDF5 schema + shape/feature-count cross-checks against the config), and
[checkpoint_probe.py](cae_suite/checkpoint_probe.py) (safe `weights_only`
metadata inspection — model/stage/normalization consistency). This is why the
per-method venv wiring below matters even for `--check`.

### Diagnostic-code prefix → exit code

`_preflight_exit_code` in [cli.py](cae_suite/cli.py) maps the first matching
error class: `ROUTE-*` → 3, `ENV-*` → 4, `NATIVE-CHECK-*` → 5, any other error
→ 2. `0` = success/validated. Preserve these prefixes when adding checks;
tooling keys off them.

## Cross-cutting conventions

### Per-method Python interpreters

Copy [ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored) to point each method at its own venv.
`resolve_python` in [settings.py](cae_suite/settings.py) **intentionally never
calls `Path.resolve()`** on the interpreter path: a venv's `python` is a symlink
whose location CPython walks up from to find `pyvenv.cfg`; dereferencing it
would silently drop the venv's site-packages. When launched from an already
activated venv with no local TOML, that venv's Python is used for everything.

### Artifact paths in native code

`log_file_dir` is a **plain cwd-relative path**, like every other path key. The
trainers used to prefix it with a literal `'outputs/'`, which configs cancelled
with an extra `../` — that hack is gone from all five repos that had it
(`training_profiles/setup.py::init_log_file`). If you add a new artifact path in
native code, spell its default `../../output/<slug>/...`; do not reintroduce a
method-local `outputs/`.

`init_log_file` records `config['log_dir']`, and the periodic train/test
prediction dumps write under it, so every artifact of one run lands together.

### Config value parsing (shared with the native repos)

The parser in [config_parser.py](cae_suite/config_parser.py) is faithful to the
native `key value` format, quirks included — do not "fix" these without updating
every native call site:

- A **single value parses to a bare scalar, not a one-element list**
  (`test_batch_idx 0` → `0`). Comma- or space-separated values become lists.
- **`int` vs `str` for numerics is significant**: `100` → `int`, but `1e-4` has
  no `.` so it fails `int()`/`float()` fast-paths and stays a **string** — every
  numeric consumer converts explicitly (`float(config.get(...))`).
- **Path-valued keys are exempt from the value lowercasing** and keep the case
  written in the config; every other string value is still lowercased. The set
  lives in `cae_suite/config_parser.py::PATH_KEYS` and is mirrored per repo as
  `PATH_KEYS` in each native `general_modules/load_config.py` — **adding a new
  path key means editing both**, or the launcher's mirror and the native parser
  will disagree about what the model actually opens. `methods/MLP/` and
  `methods/GeometryIngest/` store raw value strings and need no exemption.
  Preflight's `PATH-CASE-001` (warning) flags a config whose path case
  differs from the on-disk name — it resolves on Windows but not on Linux.
- `true`/`false` → `bool`; `%` starts a comment; a UTF-8 **BOM is a hard error**
  (native parsers misread the first key); duplicate keys are an error (native
  would silently take the last).

### Dataset contract

The MeshGraphNets, Transolver, and Neural_Operator methods all read the **same
mesh HDF5 layout** with no conversion step — `data/{sample_id}/{nodal_data,
mesh_edge}` where `nodal_data` is `[num_features, num_timesteps, num_nodes]`,
rows `0:3` are reference coordinates, and `write_preprocessing` may append
train-derived normalizers. SDFFlow uses a different SDF layout. See
[docs/reference/DATASET_FORMAT.md](docs/reference/DATASET_FORMAT.md) for the
full spec.

Row layout past the coordinates is:

```text
rows [3 : 3+input_var]              state:      input AND output
rows [3+input_var : ... +cond_var]  conditions: input ONLY  (cond_var, default 0)
```

**`cond_var`** declares trailing **input-only** rows — known boundary/flight
conditions the model reads but never predicts. They land in `graph.x` as
`[state | conditions | positional | node-type one-hot]`, are read from disk even
in the static (T=1) case where the state block is zeroed, get their own
normalization statistics, and are carried unchanged through an autoregressive
rollout. `cond_var 0` reproduces the pre-conditioning behavior exactly.
`input_var == output_var` is still required for T>1 — that constraint is about
the AR feedback loop, which conditioning rows sit outside of.

SimulGenVAE reads the same rows as a per-sample parameter vector via
`lc_data_type hdf5` + `cond_var`. MLP needs nothing: its tabular `X`/`Y`
contract already separates inputs from outputs.

## The Studio (`studio/`)

A browser front end over the *same* launcher: it builds a native flat-text
config per executable block, runs `cae_suite`'s real preflight on it, and
subprocess-launches the same command the CLI would. It imports no ML code and
reimplements no model. **When the GUI and the launcher disagree, the launcher is
right and the GUI has a bug** — see [docs/GUI.md](docs/GUI.md) for the manual.

```bash
cd studio && python start_studio.py        # http://127.0.0.1:8123/index.html
python -m pytest -q studio/studio_backend  # backend tests
```

Rules that are load-bearing, not style:

- **The live `MethodSpec` owns the config contract.** `/api/models` already
  publishes `known_keys`, `required`, `defaults`, `defaults_by_mode` and
  `modes`; `constants.js` holds only presentation (labels, help, opinionated
  defaults) plus `CHOICES`, which must mirror a spec validator exactly. Never
  invent a dropdown value — a value the sheet offers and the launcher rejects is
  the recurring failure mode here.
- **One server per port.** `allow_reuse_address = False` makes a second instance
  fail loudly instead of splitting requests. The registry is read once at
  startup, so after touching `cae_suite/specs/` or `studio_backend/` you must
  **restart the server** — otherwise the GUI validates against a stale spec and
  reports failures that do not exist.
- **Never import `studio_backend.state` from a helper script**; constructing
  `StudioState()` runs job recovery and can rewrite a live server's records.
- **Templates are default profiles, not demos.** Every live route has one in
  `TEMPLATES`, each mirroring a checked-in config, and each must pass Validate
  on a fresh checkout. A new route needs a template too — a spec added after the
  last GUI audit is invisible to it (this is exactly how `chi-mgnflow` shipped
  with none).
- **The held-out split is its own source block.** Wiring the training dataset
  into an Inference block makes the graph say "predict what you trained on", and
  the graph wins over the trainer's `infer_dataset`.
- **Coordinates are never a prediction, and neither is the node-type row.** Rows
  0:3 of `nodal_data` are copied verbatim into every rollout, so scoring them
  yields R² = 1 for any model, and the trailing part/`node_type` label is copied
  through the same way. The evaluator refuses a coordinate-only mapping;
  `nodal_field` (SimulGen-VAE reconstruct) carries physical rows only and is
  scored in full. The five rollout writers record `output_var` beside
  `num_features` so the scoreable rows are **read**, not inferred — keep that
  attribute when you touch `inference_profiles/rollout.py`.
- **A control the user cannot act on does not ship.** A port that nothing reads,
  a field a workspace overwrites, a preset that duplicates another, a preview
  drawn from invented numbers: each one asserts something the code does not do.
  Prefer removing it, or rendering it read-only with the source of its value
  named ("set in Evaluation"), over leaving it editable.
- **Check the precondition you state.** Compare Models tells the user to pick
  runs from one held-out set; it now cross-checks the `contract` and sample IDs
  in each CSV and reports a mismatch. The same applies to the Optimization
  constraint box, validated against the selected CSV before the run.
- **`input_var`/`output_var` stay the user's to set** on mesh routes. With
  `cond_var` rows present they are not the feature-row count, and deriving them
  reproduces the constant-target class of bug.

## Documentation

All documentation lives under [docs/](docs/) — see [docs/README.md](docs/README.md)
for the index. Method directories keep only their own `CLAUDE.md`/`README.md`.

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full architecture guide, one
  section per method, plus the honest known-gaps list.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — config grammar, routes, and
  pointers to the executable key/default contracts.
- [docs/GUI.md](docs/GUI.md) — the Studio manual: blocks, templates, the config
  sheet, validation, evaluation rules, and how to extend it.
- [docs/reference/](docs/reference/) — dataset format, public dataset
  provenance, per-method config key references.
- [docs/research/](docs/research/) — design notes grouped by method. These are
  design context, not implementation truth; the code wins on current behaviour.

## When you change something

- **New/renamed native config key** → update the method's spec `known_keys`
  (and validator/required lists) in `cae_suite/specs/`, or the launcher will
  reject a valid config or accept an invalid one. Then re-run
  `--audit-configs`.
- **New method repo** → create `methods/<Name>/` and `configs/<Name>/`, add a
  `build_*_spec()` with `repository="methods/<Name>"`, and register it in
  [registry.py](cae_suite/registry.py)'s `MethodRegistry.__init__`.
- **New artifact path** → point it at `../../output/...`; never inside
  `methods/`.
- **Changing behavior inside a method repo** → follow that repo's `CLAUDE.md`
  and run its own tests; the launcher change (if any) is usually just the spec.
- **New model route** → the Studio installs a block for it automatically from
  the live spec, but add a `TEMPLATES` entry (a default profile that passes
  Validate) and `HELP` text for its route-specific keys, or it arrives in the
  GUI as an unwired block with undocumented fields. Restart the Studio server so
  its registry picks the route up.
- **New enum in a spec validator** → mirror it in `CHOICES`
  (`studio/src/constants.js`), or the config sheet renders free text and accepts
  values the launcher rejects.
