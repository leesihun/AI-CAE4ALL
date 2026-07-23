# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

AI-CAE4ALL is a **monorepo of five independent ML-for-CAE method repositories**
unified by one config-driven launcher. The launcher (`cae_suite/`, the
`ai-cae4all` console script, `AI_CAE4ALL_main.py`) reads a native flat-text
config, routes on its `model` field to the right method repo, runs a layered
preflight validation, and then **subprocess-launches that repo's native
entrypoint** — it never imports the ML code.

Each method repo is self-contained (own venv, own tests, own `CLAUDE.md`) and
also runnable directly. The five specs live in [cae_suite/specs/](cae_suite/specs/):

| `model` config value(s) | Repo | Entrypoint | Has own CLAUDE.md |
| --- | --- | --- | --- |
| `meshgraphnets` | `MeshGraphNets/` | `MeshGraphNets_main.py` | — |
| `meshgraphnets-v` | `MeshGraphNets - variational/` | `MeshGraphNets_main.py` | — |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `Neural_Operator/` | `main.py` | [yes](Neural_Operator/CLAUDE.md) |
| `transolver` | `Transolver/` | `Transolver_main.py` | — |
| `sdfflow` | `Geometry_generation/` | `SDFFlow_main.py` | [yes](Geometry_generation/CLAUDE.md) |

**When working inside a method repo, its own `CLAUDE.md` is authoritative** for
that method's data contract, architecture facts, and validation steps. This file
covers only the root-level launcher and the cross-cutting conventions.

## Commands

Everything routes through the launcher; `--config` selects the file, and `mode`
(train / inference / sample / …) lives *inside* the config, not on the CLI.

```bash
# Validate only (all applicable checks, reports every problem together):
python AI_CAE4ALL_main.py --config configs/Transolver/ex2/config_train_transolver.txt --check

# Show the exact native command without launching:
python AI_CAE4ALL_main.py --config configs/Neural_Operator/ex1/config_train_fno.txt --dry-run

# A clean preflight auto-launches the native process:
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train1.txt

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

There is **no root-level test suite** — `[tool.pytest.ini_options] testpaths =
["tests"]` in [pyproject.toml](pyproject.toml) is stale (no root `tests/` dir
exists). Tests live per method repo and run in that repo's venv:

```bash
cd Neural_Operator && pytest tests/        # fast, tiny synthetic HDF5 fixtures
cd Geometry_generation && python -m pytest -q tests/test_sdfflow_pipeline.py
```

`MeshGraphNets/`, `MeshGraphNets - variational/`, and `Neural_Operator/` ship
`tests/`; consult a repo's own CLAUDE.md for its exact validation command set.

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

Note the **directory name `MeshGraphNets - variational/` contains spaces** —
quote it in shell commands and rely on `pathlib`, never string concatenation.

### Config value parsing (shared with the native repos)

The parser in [config_parser.py](cae_suite/config_parser.py) is faithful to the
native `key value` format, quirks included — do not "fix" these without updating
every native call site:

- A **single value parses to a bare scalar, not a one-element list**
  (`test_batch_idx 0` → `0`). Comma- or space-separated values become lists.
- **`int` vs `str` for numerics is significant**: `100` → `int`, but `1e-4` has
  no `.` so it fails `int()`/`float()` fast-paths and stays a **string** — every
  numeric consumer converts explicitly (`float(config.get(...))`).
- `true`/`false` → `bool`; `%` starts a comment; a UTF-8 **BOM is a hard error**
  (native parsers misread the first key); duplicate keys are an error (native
  would silently take the last).

### Dataset contract

The MeshGraphNets, Transolver, and Neural_Operator methods all read the **same
mesh HDF5 layout** with no conversion step — `data/{sample_id}/{nodal_data,
mesh_edge}` where `nodal_data` is `[num_features, num_timesteps, num_nodes]`,
rows `0:3` are reference coordinates, and `write_preprocessing` may append
train-derived normalizers. SDFFlow uses a different SDF layout. See
[dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) for the full spec and
[CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for the exhaustive,
live-code-backed key catalog and current launcher/native mismatches.

## When you change something

- **New/renamed native config key** → update the method's spec `known_keys`
  (and validator/required lists) in `cae_suite/specs/`, or the launcher will
  reject a valid config or accept an invalid one. Then re-run
  `--audit-configs`.
- **New method repo** → add a `build_*_spec()` and register it in
  [registry.py](cae_suite/registry.py)'s `MethodRegistry.__init__`.
- **Changing behavior inside a method repo** → follow that repo's `CLAUDE.md`
  and run its own tests; the launcher change (if any) is usually just the spec.
