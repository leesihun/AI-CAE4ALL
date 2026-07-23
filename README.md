# AI-CAE4ALL

**A monorepo of five machine-learning-for-CAE method repositories, unified by
one config-driven launcher.** Pick a method by writing one word in a text
config; a single command validates the whole config up front and launches the
right native runtime.

```bash
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train1.txt
```

The launcher reads the `model` field, routes to the matching repository, runs a
layered preflight validation (config schema → paths → environment → dataset →
checkpoint → the method's own native validator), then subprocess-launches that
repo's entrypoint in its own working directory and Python interpreter. It never
imports the ML code, and it will not start a run while any blocking preflight
error remains.

---

## The five methods

Each method is self-contained (own tests, own entrypoint, runnable directly) and
selected purely by the `model` field in the config:

| `model` value(s) | Method | What it does |
| --- | --- | --- |
| `meshgraphnets` | MeshGraphNets (deterministic) | Graph-network mesh simulator: time evolution of physical fields |
| `meshgraphnets-v` | MeshGraphNets (variational) | Probabilistic mesh simulator (VAE + learned prior): a *distribution* of trajectories |
| `point_deeponet`, `deeponet`, `fno`, `gino` | Neural Operators | Four discretization-generalizing field-to-field operators (FNO/GINO implemented natively) |
| `transolver` | Transolver | Transformer surrogate via learned Physics-Attention "slices" |
| `sdfflow` | SDFFlow (Geometry generation) | Generates *new 3D shapes* via an SDF-VAE + flow matching |

Routing map:

| Config value | Native backend |
| --- | --- |
| `meshgraphnets` | `MeshGraphNets/MeshGraphNets_main.py` |
| `meshgraphnets-v` | `MeshGraphNets - variational/MeshGraphNets_main.py` |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `Neural_Operator/main.py` |
| `transolver` | `Transolver/Transolver_main.py` |
| `sdfflow` | `Geometry_generation/SDFFlow_main.py` |

---

## Quick start

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

Introspection (no config needed):

```bash
python AI_CAE4ALL_main.py --list-models         # registered models + install health
python AI_CAE4ALL_main.py --describe transolver  # route, modes, required fields
```

Other useful flags: `--strict` (promote flagged warnings to errors),
`--explain-config` (configured/defaulted/inactive/checkpoint-owned key
breakdown), `--json-report PATH`, `--python PATH` (interpreter override), and
`--skip-{native,filesystem,environment}-check` to bypass a probe layer.

Exit codes encode the failure class: `0` validated/launched, `2` config/data
error, `3` routing error, `4` environment error, `5` the method's native
validator rejected the config.

### SDFFlow (geometry generation)

SDFFlow uses one production training config that trains the VAE first, verifies
its checkpoint, then immediately trains flow matching — no idle-GPU gap between
stages:

```bash
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_sample.txt
```

Relaunching the training config safely reuses compatible completed stages;
retraining the VAE invalidates reuse of an older FM checkpoint. Guarded
extrapolation and reproducible interpolation configs ship alongside the sample
config. SDFFlow additionally supports `train_vae`, `train_fm`, `reconstruct`, and
`interpolate` modes.

---

## Config format

All backends read the same flat `key value` text. Keys and string values are
lowercased; `%` starts a comment; commas or multiple tokens make a list
(`gpu_ids 0,1`); `true`/`false` are booleans. Two quirks worth knowing:

- A **single value parses to a bare scalar, not a one-element list**.
- **Prefer decimal notation** (`0.0001`, not `1e-4`): a token with no `.` stays
  a string, so many consumers convert explicitly.

Paths inside a config are resolved from the **selected method repository** (that
is its working directory), which is why templates use paths like
`../output/meshgraphnets/ex1/model1.pth`. For inference, architecture and
normalization are **checkpoint-led** — values in the config may be replaced by
the checkpoint after it loads.

---

## Repository layout

```text
AI_CAE4ALL_main.py            # entrypoint → cae_suite.cli.main
cae_suite/                    # the launcher (parse → route → preflight → subprocess); no ML
configs/                      # config templates per method + benchmarks
dataset/                      # shared HDF5 data + format spec + benchmark datasets
output/                       # run artifacts (checkpoints, rollouts, samples)
MeshGraphNets/                # model = meshgraphnets
MeshGraphNets - variational/  # model = meshgraphnets-v   (note: name has spaces)
Neural_Operator/              # model = point_deeponet | deeponet | fno | gino
Transolver/                   # model = transolver
Geometry_generation/          # model = sdfflow
```

The mesh methods (all but SDFFlow) share one HDF5 data contract with **no
conversion step** — see [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md).

---

## Per-method Python environments

Launching from an activated venv needs no configuration — that venv's Python is
used for everything. For separate backend environments, copy
[ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` (git-ignored) and set each interpreter path (relative
paths resolve from the repo root):

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

Interpreter precedence: `--python` → exact model ID → method ID →
`python.default` → the suite's own `sys.executable`. Installing the root package
(`python -m pip install -e .`) also provides the `ai-cae4all` command.

---

## Testing

There is no root-level test suite; tests live per method repo and run in that
repo's venv:

```bash
cd Neural_Operator && pytest tests/          # fast, tiny synthetic HDF5 fixtures
cd Geometry_generation && python -m pytest -q tests/test_sdfflow_pipeline.py
```

`Neural_Operator/` has the deepest coverage; `MeshGraphNets/` and its variational
sibling ship AR-rollout and multiscale tests.

---

## Documentation

| Doc | Purpose |
| --- | --- |
| [REPOSITORY_OVERVIEW.md](REPOSITORY_OVERVIEW.md) | Full architecture guide: the launcher internals and a section on every method |
| [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) | Exhaustive, live-code-backed catalog of every config key, its necessity, and known launcher/native mismatches |
| [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md) | The shared mesh HDF5 data contract |
| [CLAUDE.md](CLAUDE.md) | Condensed conventions for the root launcher |
| `Neural_Operator/CLAUDE.md`, `Geometry_generation/CLAUDE.md` | Authoritative notes for those two methods |

For any specific config key, `CONFIGURATION_REFERENCE.md` is authoritative; for a
method's internals, that method's own docs and code are authoritative.
