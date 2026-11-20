# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this repository is

AI-CAE4ALL is a **monorepo of independent ML-for-CAE method repositories behind
one config-driven launcher**. The launcher (`ai_cae4all/`, the `ai-cae4all`
console script, `AI_CAE4ALL_main.py`) reads a flat-text config, routes on its
`model` field, runs a layered preflight, and then **subprocess-launches that
method's native entrypoint**. It never imports ML code. The package was
`cae_suite/` until 2026-11-20; a machine with an older editable install needs
`python -m pip install -e .` again for the `ai-cae4all` command (running
`AI_CAE4ALL_main.py` or `python -c "from ai_cae4all ..."` from the repo root
needs nothing).

Every method under [methods/](methods/) is self-contained (own venv, own tests)
and also runnable directly. Routing specs live in
[ai_cae4all/specs/](ai_cae4all/specs/); a spec's `repository=` field is the
model-to-directory mapping.

| `model` value(s) | Directory | Entrypoint | Own docs |
| --- | --- | --- | --- |
| `meshgraphnets` | `methods/MeshGraphNets/` | `MeshGraphNets_main.py` | — |
| `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | `MeshGraphNets_Variational_main.py` | — |
| `point_deeponet`, `deeponet`, `fno` | `methods/Neural_Operator/` | `main.py` | [CLAUDE.md](methods/Neural_Operator/CLAUDE.md) |
| `transolver` | `methods/Transolver/` | `Transolver_main.py` | — |
| `sdfflow` | `methods/SDFFlow/` | `SDFFlow_main.py` | [CLAUDE.md](methods/SDFFlow/CLAUDE.md) |
| `simulgenvae` | `methods/SimulGenVAE/` | `SimulGenVAE_main.py` | [CLAUDE.md](methods/SimulGenVAE/CLAUDE.md) |
| `mlp` | `methods/MLP/` | `MLP_main.py` | [CLAUDE.md](methods/MLP/CLAUDE.md) |
| `geometry_ingest` | `methods/GeometryIngest/` | `main.py` | [README.md](methods/GeometryIngest/README.md) |
| `sar` *(optional, local-only)* | `methods/SAR/` | `SAR_main.py` | its README.md |
| `ldgn` *(optional, local-only)* | `methods/LDGN/` | `LDGN_main.py` | its README.md |

**When working inside a method, its own `CLAUDE.md`/`README.md` is
authoritative** for its data contract, architecture and validation. This file
covers the launcher and the cross-cutting conventions.

### Names that do not match

Model IDs are frozen by every checked-in config, so a few differ from the
directory: `sdfflow` → `SDFFlow/` (formerly `Geometry_generation/`),
`meshgraphnets-v` → `MeshGraphNets_Variational/`. Display names follow latent
placement: `meshgraphnets-v` with `use_multiscale` is **HI-MGN-V**
(`latent_type global`, default), **HI-MGNflow** (`latent_type local`) or
**HI-MGN-hybrid** (`latent_type hybrid`). HI-MGNflow was its own route
(`chi-mgnflow`) until 2026-10-19; that ID no longer routes and the old repo is
in git history (commit `9db47d6`). SimulGenVAE's campaign key is `lsh_vae` (its
paper name); it is baked into `output/dataset_matrix/` run paths, so leave it.
The MGN-V entrypoint was `MeshGraphNets_main.py` (the same name as MGN's) until
2026-11-19; the sweeps' `gpu_idle.py` still matches both names, for runs
started before the rename.

### The methods in one line each

- **MeshGraphNets** — MGN, HI-MGN (`use_multiscale`), BSMS-GNN (`coarsening_type bfs`).
- **MeshGraphNets-V** — conditional graph VAE with a flow-matching prior; the
  one-to-many (probabilistic) model. Ensemble scoring (CRPS, W2, spread
  histograms) lives in `inference_profiles/spread.py`; `rollout.py` re-exports
  it, and the sweep plotter loads it directly.
- **Neural_Operator** — DeepONet, Point-DeepONet, FNO.
- **Transolver** — physics-attention transformer (`transolver3` configs).
- **SDFFlow** — SDF VAE + flow matching for geometry. Also the suite's only
  closed-loop mode: `mode optimize` chains generation → gmsh tet meshing →
  linear-static solve → CMA-ES over the trained generator (needs `gmsh`,
  `pyamg`, `cma`; trains nothing). Dataset building and condition sidecars are
  standalone tools in `methods/SDFFlow/scripts/`, run from `methods/SDFFlow`
  as `python scripts/<name>.py`.
- **SimulGenVAE** — hierarchical VAE + latent conditioner. Reads the shared mesh
  HDF5 but is a **fixed-geometry dense** model: every sample must share node and
  timestep counts. Modes `train`/`train_vae`/`train_lc`/`reconstruct`, in-process
  multi-GPU via `mp.spawn`, `vae_*`/`lc_*` keys.
- **MLP** — tabular parametric surrogate, `X[S,N]`→`Y[S,M]`
  (`dataset_kind=table_hdf5`, `native_probe=False`), no GPU, not the mesh contract.
- **GeometryIngest** — non-ML data prep: CAD (STEP/IGES/STL) → shared mesh HDF5.
  `native_probe=False`, `dataset_kind=None`, modes `ingest`/`inspect`.

### Optional routes: SAR and LDGN

`sar` (Lino & Thuerey 2026, scale-autoregressive flow matching) and `ldgn`
(Lino, Pfaff & Thuerey, ICLR 2025, latent diffusion graph net) are upstream
ports kept as one-to-many baselines beside MGN-V. They are **git-ignored**
(`methods/{SAR,LDGN}/`, `configs/{SAR,LDGN}/`) and live only on machines that
already have them. Nothing tracked depends on them:

- `ai_cae4all/registry.py::OPTIONAL_SPEC_BUILDERS` registers each only when its
  `methods/` directory exists; without it the model ID simply does not route.
- The Studio hides palette blocks and templates for any route the launcher does
  not report (`markUnregisteredModels` in `studio/src/constants.js`).
- Their specs (`ai_cae4all/specs/{sar,ldgn}.py`) stay tracked so the routes come
  back as soon as the directories do.

If you touch them: keep `ldgn_diffusion_steps` at 1000 (the linear schedule
clips its betas; below ~50 steps sampling returns NaN). Each README lists every
deviation from upstream.

## Layout

```text
AI-CAE4ALL/
├── AI_CAE4ALL_main.py  ai_cae4all/   launcher (no ML imports) + specs
├── methods/            configs/     method runtimes, mirrored config dirs
├── studio/             inference/   browser Studio, portable CPU inference bundle
├── docs/                            all documentation (index: docs/README.md)
├── dataset/            output/      inputs and artifacts — local-only, git-ignored
└── paper/, "Claude outputs"/        local-only, git-ignored
```

Two invariants; breaking either is a bug:

1. **`configs/<Name>/` mirrors `methods/<Name>/`.** A new method adds both.
   The one exception is `configs/campaigns/`, which holds the cross-method
   campaign (`dataset_matrix/`: `manifest.json` lists every train/infer pair,
   `run_135.sh`/`run_136.sh` drive the two B300 nodes).
2. **`output/` at the repo root is the only artifact destination.** The native
   process runs with cwd `methods/<Name>/`, so every artifact path in a config
   is `../../output/...` and every dataset path `../../dataset/...`. Nothing is
   written inside `methods/`.

`output/` subdirectory names are historical and deliberately kept
(`output/geometry_generation/` = SDFFlow, `output/chi-mgnflow/` = retired route
runs). They name runs, not tree directories.

## Commands

Everything goes through the launcher. `--config` picks the file; `mode` lives
**inside** the config, not on the CLI.

```bash
python AI_CAE4ALL_main.py --config <cfg> --check     # validate only; every problem at once
python AI_CAE4ALL_main.py --config <cfg> --dry-run   # print the native command
python AI_CAE4ALL_main.py --config <cfg>             # clean preflight → launch
python AI_CAE4ALL_main.py --list-models              # routes + install health
python AI_CAE4ALL_main.py --describe transolver      # routing + required fields per mode
python AI_CAE4ALL_main.py --audit-configs            # structural lint of every config*.txt
```

Flags: `--strict` (promote flagged warnings), `--explain-config`
(configured/defaulted/inactive/checkpoint-owned keys), `--json-report PATH`,
`--skip-{native,filesystem,environment}-check`. `python -m pip install -e .`
also installs the `ai-cae4all` command.

### Tests

There is no root `tests/`. The launcher layer is covered by `--audit-configs`
plus per-config `--check`; the Studio backend runs in the launcher interpreter
(`pytest` with no path runs it too, via `pyproject.toml`):

```bash
python AI_CAE4ALL_main.py --audit-configs
python -m pytest -q studio/studio_backend
```

Method tests run in that method's venv, from its directory
(`cd methods/SDFFlow && python -m pytest -q tests/`). Every method except
GeometryIngest has `tests/`. See [docs/guides/testing.md](docs/guides/testing.md).

## Launcher architecture (`ai_cae4all/`)

Pipeline: **parse → resolve → layered preflight → command → subprocess**.

- [config_parser.py](ai_cae4all/config_parser.py) — flat `key value` →
  `ParsedConfig` (values, per-key source lines, duplicate detection),
  **mirroring the native parsers' quirks** (below).
- [registry.py](ai_cae4all/registry.py) — `MethodRegistry`: `model` →
  `MethodSpec` → repo root + entrypoint. Aliased IDs share a spec;
  `default_specs()` adds the optional routes only if present.
- [specs/base.py](ai_cae4all/specs/base.py) + `specs/<method>.py` — each spec
  declares `known_keys`, required/recommended/default fields per mode,
  `PathRule`s, `import_modules`, `dataset_kind`, and validators. **The single
  source of truth for config validation**: an unlisted key is a
  `CFG-UNKNOWN-001` warning.
- [preflight.py](ai_cae4all/preflight.py) — `run_preflight` runs layers and
  **short-circuits on the first layer with errors**: spec → filesystem →
  environment → dataset → checkpoint → native probe. Builds the `command`.
- [path_checks.py](ai_cae4all/path_checks.py) — resolves paths against the
  method repo; missing inputs, unwritable outputs, case mismatches.
- [diagnostics.py](ai_cae4all/diagnostics.py) — `Severity`, `Diagnostic`
  (`promote_in_strict`), report rendering.
- [cli.py](ai_cae4all/cli.py) — args, introspection commands, exit codes.
- [launcher.py](ai_cae4all/launcher.py) — `launch_and_wait`; child in its own
  process group so Ctrl-C forwards and escalates on Windows and POSIX.
- [settings.py](ai_cae4all/settings.py) — reads `ai_cae4all.local.toml` for
  per-method interpreters.

**Probes run in the target method's venv.**
[native_probe.py](ai_cae4all/native_probe.py) (imports the native config
validator), [dataset_probe.py](ai_cae4all/dataset_probe.py) (HDF5 schema and
shape/feature-count cross-checks) and
[checkpoint_probe.py](ai_cae4all/checkpoint_probe.py) (`weights_only` metadata:
model/stage/normalization consistency) are subprocesses under the resolved
method Python, so `--check` validates against the real runtime environment.

**Exit codes** (`_preflight_exit_code` in cli.py, first matching error class):
`ROUTE-*` → 3, `ENV-*` → 4, `NATIVE-CHECK-*` → 5, any other error → 2,
success → 0. Keep these prefixes when adding checks; tooling keys off them.

## Cross-cutting conventions

### Per-method interpreters

Copy [ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored). `resolve_python` **never calls
`Path.resolve()`** on the interpreter: a venv `python` is a symlink CPython
walks up from to find `pyvenv.cfg`, and dereferencing it drops the venv's
site-packages. With no local TOML, an activated venv's Python is used for all.

### Config value parsing (shared with every native repo)

Faithful to the native format, quirks included. Do not "fix" these without
updating every native call site:

- A **single value is a bare scalar**, not a one-element list
  (`test_batch_idx 0` → `0`); comma/space-separated values become lists.
- **`int` vs `str` matters**: `100` → `int`, but `1e-4` stays a **string**;
  every numeric consumer converts explicitly (`float(config.get(...))`).
- **Path keys keep their case**; every other string value is lowercased. The set
  is `PATH_KEYS` in `ai_cae4all/config_parser.py`, mirrored in each native
  `general_modules/load_config.py`. **A new path key means editing both.**
  (MLP and GeometryIngest store raw strings and need no exemption.)
  `PATH-CASE-001` warns when config case differs from disk: fine on Windows,
  broken on Linux.
- `true`/`false` → `bool`; a line starting with `%` is a comment and `#` starts an inline one; a UTF-8 **BOM is an error**;
  **duplicate keys are an error** (native would silently take the last).

### Artifact paths in native code

`log_file_dir` is a plain cwd-relative path like every other path key
(`training_profiles/setup.py::init_log_file` records `config['log_dir']`, and
periodic dumps write under it). A new native artifact path defaults to
`../../output/<slug>/...`; never reintroduce a method-local `outputs/`.

### Dataset contract

MeshGraphNets, MGN-V, Transolver, Neural_Operator (and SimulGenVAE, densely)
read one **mesh HDF5** layout with no conversion: `data/{sample_id}/{nodal_data,
mesh_edge}`, `nodal_data` = `[num_features, num_timesteps, num_nodes]`, rows
`0:3` = reference coordinates; `write_preprocessing` may append train-derived
normalizers. SDFFlow uses its own SDF layout. Full spec:
[docs/reference/DATASET_FORMAT.md](docs/reference/DATASET_FORMAT.md).

```text
rows [3 : 3+input_var]              state:      input AND output
rows [3+input_var : ... +cond_var]  conditions: input ONLY  (cond_var, default 0)
last row (if present)               node type / part label, copied through
```

**`cond_var`** rows are known conditions the model reads but never predicts:
in `graph.x` as `[state | conditions | positional | node-type one-hot]`, read
even when the T=1 state block is zeroed, normalized separately, carried
unchanged through AR rollout. `cond_var 0` is the unconditioned behavior.
`input_var == output_var` is still required for T>1 (the AR loop).
SimulGenVAE reads the same rows via `lc_data_type hdf5` + `cond_var`.

### Mid-training resume (`resume_training`)

`general_modules/resume_state.py` is **copied byte-for-byte** into every
training repo except MLP (six tracked, plus the local SAR/LDGN copies); the
repos share no imports, so an edit to one goes to all. With `resume_training
true` the trainer:

- writes `<checkpoint>.resume` (everything the next epoch reads, RNG included)
  at the first epoch boundary after every `resume_interval_minutes` (15);
- resumes only from a state with the same config fingerprint (`gpu_ids`,
  `log_dir`, resume keys and `_`-keys excluded);
- deletes the state once the final checkpoint is written.

A resumed run appends `==== Resume [TAG] at epoch K` to its epoch log, which
`configs/campaigns/dataset_matrix/score_rank.py` reads as one run.
Multi-process entry points reject the key; off (default) changes nothing. The
dataset_matrix runner appends both keys to each launch copy.

- **New mutable training state** (tracker, stateful schedule) must go into that
  trainer's `resume.save({...})`, or resumed runs silently diverge.
- **Persistent-worker DataLoaders** draw their worker seed when the iterator is
  first built, so call `iter(loader)` on each **before**
  `resume.restore_rng()` and before restoring any loader generator.
- Worker-side RNG is not restored, so two cases resume statistically, not
  bit-identically: SDFFlow VAE surface subsampling with `vae_num_workers > 0`,
  and MGN-V `_pick_hierarchy_variant` with `num_workers > 0` and
  `hierarchy_variants > 1`.

### Periodic visualization

`test_interval` gates a pass that writes a picture into the **log file's
directory**. `display_testset` (default True) turns it on; `display_trainset`
adds a few training samples under `train/` (underfitting vs generalization gap).

| method | picture |
| --- | --- |
| MeshGraphNets, MGN-V, Neural_Operator, Transolver | mesh HDF5 + 2×2 PNG (pred vs truth × normalized vs denormalized), coloured by `plot_feature_idx` |
| MLP | parity plot per output column |
| SimulGenVAE | truth / reconstruction / \|error\| images (VAE), latent parity (LC) |
| SDFFlow | strip of reconstructed or generated shapes beside the STLs |
| SAR, LDGN | VAE parity per channel; spread histogram for one system |
| GeometryIngest | `preview` of the first samples (the only output of `inspect`) |

**The renderer needs triangles**: `edges_to_triangles_*` rebuilds the surface
from `edge_index`; on quad/hex meshes `quads_to_triangles` splits 4-cycles; with
no surface, scalars are drawn per node. **Every plotting backend is imported
lazily**: a missing matplotlib/pyvista prints a note and the run continues.
Headless, an EGL/OSMesa VTK (vtk>=9.4) works; an X-only VTK `abort()`s, so
`mesh_utils_fast.offscreen_rendering_available()` probes one render in a child
process first and skips PNGs if it dies.

## The Studio (`studio/`)

A browser front end over the **same** launcher: it builds a native config per
executable block, runs `ai_cae4all`'s real preflight, and launches the same
command the CLI would. It imports no ML code. **When the GUI and the launcher
disagree, the launcher is right.** Manual: [docs/GUI.md](docs/GUI.md).

```bash
cd studio && python start_studio.py        # http://127.0.0.1:8080/index.html
python -m pytest -q studio/studio_backend
```

Load-bearing rules:

- **The live `MethodSpec` owns the config contract.** `/api/models` publishes
  `known_keys`, `required`, `defaults`, `defaults_by_mode`, `modes`.
  `constants.js` holds presentation only, plus `CHOICES`, which must mirror a
  spec validator exactly. Never invent a dropdown value.
- **One server per port** (`allow_reuse_address = False`). The registry is read
  once at startup: after touching `ai_cae4all/specs/` or `studio_backend/`,
  **restart the server**, or it validates against a stale spec.
- **Never import `studio_backend.state` from a helper script**: constructing
  `StudioState()` runs job recovery and can rewrite a live server's records.
- **Templates are default profiles, not demos.** Every live route has one in
  `TEMPLATES`, mirroring a checked-in config and passing Validate on a fresh
  checkout. A new route needs a template.
- **The held-out split is its own source block.** Wiring the training dataset
  into Inference means "predict what you trained on", and the graph wins over
  the trainer's `infer_dataset`.
- **Coordinates and the node-type row are never predictions.** They are copied
  verbatim into rollouts, so scoring them gives R² = 1. The evaluator refuses a
  coordinate-only mapping. The rollout writers record `output_var` beside
  `num_features` so scoreable rows are read, not inferred; keep it when editing
  `inference_profiles/rollout.py`.
- **A control the user cannot act on does not ship.** Remove it, or render it
  read-only naming the source of its value.
- **Check the precondition you state** (Compare Models cross-checks `contract`
  and sample IDs; the Optimization constraint box validates against its CSV).
- **`input_var`/`output_var` stay the user's to set** on mesh routes: with
  `cond_var` rows they are not the feature-row count.

## Documentation

Everything lives under [docs/](docs/) ([index](docs/README.md)); method
directories keep only their own `CLAUDE.md`/`README.md`.

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — layout, launcher internals, config
  grammar, cross-cutting conventions, the dataset tree.
- [CONFIGURATION.md](docs/CONFIGURATION.md) — config workflow, routes, where each
  contract lives.
- [GUI.md](docs/GUI.md) — Studio manual and how to extend it.
- [methods/](docs/methods/) — one numbered doc per method (`00`–`12`; `12` is
  SimulGenVAE).
- [reference/](docs/reference/) — dataset format, dataset provenance (with the
  commit holding the removed public-dataset converters), per-method config key
  references.
- [research/](docs/research/) — design notes by method. Context, not truth: the
  code wins. Dated report directories (`mgnv_*_2026-09-*`) are frozen snapshots
  whose scripts depend on their depth; do not move them.

MGN-V hyperparameter sweeps are indexed in
[configs/MeshGraphNets_Variational/hyperparameter_sweep/README.md](configs/MeshGraphNets_Variational/hyperparameter_sweep/README.md).

## When you change something

- **New/renamed native config key** → the spec's `known_keys` (and
  validator/required lists), then `--audit-configs`. A new **path** key also
  goes into both `PATH_KEYS` copies.
- **New spec enum** → mirror it in `CHOICES` (`studio/src/constants.js`).
- **New method** → `methods/<Name>/` + `configs/<Name>/`, a `build_*_spec()`
  with `repository="methods/<Name>"`, registered in `default_specs()` in
  [registry.py](ai_cae4all/registry.py); a Studio `TEMPLATES` entry and `HELP`
  text; restart the Studio.
- **New artifact path** → `../../output/...`, never inside `methods/`.
- **Behavior inside a method** → follow its `CLAUDE.md` and run its tests; the
  launcher change is usually just the spec.
- **`resume_state.py`** → edit all copies identically.
