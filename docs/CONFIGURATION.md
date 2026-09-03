# AI-CAE4ALL configuration reference

This is the suite-wide entry point for configuration behavior. The executable
source of truth is the `MethodSpec` registry under `cae_suite/specs/`, followed
by each method's native loader and validator. This document intentionally does
not duplicate hundreds of key names that would drift from those live sets.

## Validate a configuration

```powershell
python AI_CAE4ALL_main.py --check path\to\config.txt --strict --no-color
python AI_CAE4ALL_main.py --audit-configs --no-color
```

`--check` parses, routes, validates paths and values, probes the native runtime,
and checks the dataset/checkpoint contracts without launching training.
`--audit-configs` discovers the checked-in top-level `configs/**/config*.txt`
tree. The current audit target is 269 files; use the command output, not this
snapshot, as the current count.

## Flat-file grammar

- One `key value` pair per line; tabs and spaces are accepted.
- Blank lines and lines beginning with `%` are ignored.
- `#` starts an inline comment.
- Comma- or space-separated values become lists. A single value stays scalar,
  so native consumers must normalize scalar/list fields such as
  `test_batch_idx`.
- Booleans are `true` or `false` (case-insensitive).
- Paths are resolved relative to the native method repository, matching the
  directory in which the unified launcher starts that method.

## Registered routes

| Model ID | Native repository | Modes |
| --- | --- | --- |
| `meshgraphnets` | `MeshGraphNets/` | `train`, `inference` |
| `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | `train`, `inference` |
| `chi-mgnflow` | `methods/HI_MGNFlow/` | `train`, `inference` |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `Neural_Operator/` | `train`, `inference` |
| `transolver` | `Transolver/` | `train`, `inference` |
| `sdfflow` | `methods/SDFFlow/` | `train`, `train_vae`, `train_fm`, `sample`, `reconstruct`, `interpolate`, `optimize` |
| `simulgenvae` | `SimulGenVAE/` | `train`, `train_vae`, `train_lc`, `reconstruct` |
| `mlp` | `MLP/` | `train`, `inference` |
| `geometry_ingest` | `methods/GeometryIngest/` | `ingest`, `inspect` |

## Exact key catalogs and defaults

The exact current sets live here:

| Contract | Source |
| --- | --- |
| Suite-known keys, required/recommended fields, defaults, paths, and value checks | `cae_suite/specs/*.py` |
| Model-to-spec routing | `cae_suite/registry.py` |
| Shared parsing and path-key behavior | `cae_suite/config_parser.py` |
| Studio parameter catalogs and initial values | `studio/src/constants.js` |
| Native fail-fast validation | each method's `general_modules/config_validation.py` or native loader |
| HDF5 layouts and timestep semantics | `dataset/DATASET_FORMAT.md` plus each method's dataset/rollout code |

Some specs retain removed or method-inactive keys solely to emit a precise
diagnostic instead of an “unknown key” error. Membership in `known_keys` does
not by itself mean a setting is active; the spec validator decides that.

To print a live catalog from the suite registry:

```powershell
python -c "from pathlib import Path; from cae_suite.registry import MethodRegistry; r=MethodRegistry(Path('.')); [print(','.join(s.model_ids), *sorted(s.known_keys), sep='\n  ') for s in r.specs]"
```

## Important cross-method contracts

- Static `T=1` datasets use direct-field targets; temporal datasets use the
  configured autoregressive integration contract. Confirm behavior from the
  HDF5 layout, loader, and rollout path together.
- Inference restores architecture and normalization metadata from the selected
  checkpoint; an inherited training config is not evidence of checkpoint
  compatibility.
- `write_preprocessing` is method-specific. Neural Operator keeps source HDF5
  files read-only; Transolver only permits a dedicated runtime copy.
- Neural Operator `use_parallel_stats` controls train-fit normalization-stat
  computation. Its distributed-only `train_eval_subset_size` must be positive.
- Variational MeshGraphNets uses Adam; `weight_decay` is coupled L2 decay and
  defaults to `0.0` when omitted. Deterministic MeshGraphNets uses AdamW.

## Method-specific detail

Use the method repository's own docs for architecture-specific meaning. In
particular, see `methods/MeshGraphNets_Variational/docs/CONFIG_REFERENCE.md`,
`methods/HI_MGNFlow/docs/CONFIG_REFERENCE.md`, `Neural_Operator/CLAUDE.md`,
`methods/SDFFlow/CLAUDE.md`, and `MLP/CLAUDE.md`.
