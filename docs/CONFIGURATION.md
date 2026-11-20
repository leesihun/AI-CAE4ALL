# Configuration

How to write, validate and look up a config. The executable source of truth is
the `MethodSpec` for each route under [ai_cae4all/specs/](../ai_cae4all/specs),
then each method's native loader and validator. This page does not copy key
lists: they would drift from the specs, which hold 20 to 190 keys per route.

## The workflow

A config is a flat text file. `model` picks the route and `mode` picks what the
method does:

```text
model           transolver
mode            train
gpu_ids         0
dataset_dir     ../../dataset/deterministic/ex2_dynamic_contact.h5
modelpath       ../../output/transolver/ex2/model.pth
log_file_dir    ../../output/transolver/ex2/train.log
```

Then:

```bash
python AI_CAE4ALL_main.py --config path/to/config.txt --check            # every problem at once
python AI_CAE4ALL_main.py --config path/to/config.txt --explain-config   # what each key resolved to
python AI_CAE4ALL_main.py --config path/to/config.txt --dry-run          # the native command
python AI_CAE4ALL_main.py --config path/to/config.txt                    # clean preflight → launch
```

- `--check` runs the whole preflight without launching: spec, paths,
  environment, dataset, checkpoint and native validator (see
  [ARCHITECTURE.md §3](ARCHITECTURE.md#3-the-launcher-ai_cae4all)).
- `--strict` promotes flagged warnings to errors.
- `--audit-configs` lints every checked-in `configs/**/config*.txt`
  structurally. Use its output, not a number written in a document, for the
  current count.

Start from a checked-in config, not from a blank file. Each route has a
`baseline/` train and inference pair under `configs/<Method>/<category>/exN/`.
The Studio's templates mirror those same files.

## Grammar

- One `key value` pair per line. The first whitespace splits the key from the
  value.
- Blank lines and lines starting with `%` are ignored. `#` starts an inline
  comment.
- Comma- or space-separated values become a list. **A single value stays a
  scalar** (`test_batch_idx 0` → `0`), so consumers normalize scalar/list keys.
- `true`/`false` (any case) become booleans.
- `100` becomes an `int`, but **`1e-4` stays a string**. Every numeric consumer
  converts explicitly.
- Keys and string values are lowercased, except **path keys**, which keep their
  case. The set is listed twice and both copies must agree:
  `ai_cae4all/config_parser.py::PATH_KEYS` and each method's
  `general_modules/load_config.py::PATH_KEYS`. A new path key means editing
  both. `PATH-CASE-001` warns when the case in a config differs from the disk:
  fine on Windows, broken on Linux.
- Duplicate keys and a UTF-8 BOM are errors. Native loaders would silently keep
  the last duplicate, or misread the first key after a BOM.
- Quotes are literal characters. Do not quote paths.

## Paths

The native process runs with cwd `methods/<Name>/`. So relative paths resolve
from the method directory, not from the config file:

- dataset paths are `../../dataset/...`
- artifact paths (`modelpath`, `log_file_dir`, output directories) are
  `../../output/...`

Nothing is written inside `methods/`.

## Routes

| `model` | Directory | Modes |
| --- | --- | --- |
| `meshgraphnets` | `methods/MeshGraphNets/` | train, inference |
| `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | train, inference |
| `point_deeponet`, `deeponet`, `fno` | `methods/Neural_Operator/` | train, inference |
| `transolver` | `methods/Transolver/` | train, inference |
| `sdfflow` | `methods/SDFFlow/` | train, train_vae, train_fm, sample, reconstruct, interpolate, optimize, evaluate |
| `simulgenvae` | `methods/SimulGenVAE/` | train, train_vae, train_lc, reconstruct |
| `mlp` | `methods/MLP/` | train, inference |
| `geometry_ingest` | `methods/GeometryIngest/` | ingest, inspect |
| `sar`, `ldgn` | `methods/SAR/`, `methods/LDGN/` | optional, local-only; route only where the directory exists |

`meshgraphnets-v` with `use_multiscale` has three display names, set by
`latent_type`: `global` (default) is HI-MGN-V, `local` is HI-MGNflow, and
`hybrid` is HI-MGN-hybrid. The former `chi-mgnflow` route no longer exists.

`python AI_CAE4ALL_main.py --describe <model>` prints the required and default
fields for each mode of a route.

## Where each contract lives

| Contract | Source |
| --- | --- |
| Known keys, required/recommended fields, defaults, path roles, value checks | `ai_cae4all/specs/<method>.py` |
| Model → spec routing | `ai_cae4all/registry.py` |
| Parsing and path-key behavior | `ai_cae4all/config_parser.py` |
| Native fail-fast validation | the method's `general_modules/config_validation.py` (Neural_Operator, Transolver) or its native loader |
| HDF5 layouts | [reference/DATASET_FORMAT.md](reference/DATASET_FORMAT.md), plus each method's dataset code |
| Studio dropdown values | `CHOICES` in `studio/src/constants.js`, which mirrors the spec validators |
| Per-key meaning | the method docs in [methods/](methods/README.md) and [reference/config/](reference/config/) |

A key being in `known_keys` does not by itself mean it is active. Some specs
keep removed or method-inactive keys only so they can report them precisely
instead of as "unknown". `--explain-config` shows each key as configured,
defaulted, inactive or checkpoint-owned.

To print the live key catalog of every route:

```bash
python -c "from pathlib import Path; from ai_cae4all.registry import MethodRegistry; r=MethodRegistry(Path('.')); [print(','.join(s.model_ids), *sorted(s.known_keys), sep='\n  ') for s in r.specs]"
```

## Cross-method contracts

- **Mesh rows.** The mesh HDF5 holds coordinates, then `input_var` state rows,
  then `cond_var` condition rows, then an optional node-type row.
  - State rows are input and output.
  - Condition rows are input only: read, normalized separately and carried
    through rollout unchanged, never predicted.
  - `input_var`/`output_var` are yours to set; with conditions present they are
    not the row count.
  - `input_var == output_var` is required for temporal data.
- **Static vs temporal.** `T = 1` datasets train on the field itself. `T > 1`
  datasets train on deltas, under `time_integration ar_ot` (default) or `ar_rt`.
- **Inference is checkpoint-led.** Architecture and normalization come from the
  checkpoint. Architecture keys in an inference config may be overridden after
  load. Runtime choices stay with the config: dataset, output, rollout length,
  sample count.
- **The held-out set is its own file.** Every dataset has a `_infer.h5`
  companion. Point `infer_dataset` at it, not at the training file.
- **`write_preprocessing`** is per method. Neural_Operator keeps the source HDF5
  read-only. Transolver allows it only on a dedicated runtime copy.
- **`resume_training true`** checkpoints the full training state every
  `resume_interval_minutes` and resumes a killed run with the same config.
  Multi-process entry points reject it.
