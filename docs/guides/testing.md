# What to run after a change

Tests live in the layer they cover. There is no single root command that runs
everything, because each method repository is validated inside **its own venv**
with its own dependencies.

## 1. Launcher and MethodSpec contracts

**There is no root `tests/` directory any more.** The thirteen modules that used
to live there — config discovery, key contracts, documentation links, native
config-consumption parity, native runtime defaults, the SDFFlow spec, Studio/spec
key parity, the buckling dataset contract, the reflow schedule, and
`tests/campaign/`'s scheduler and layout anchors — were deleted wholesale in
commit `9884fb1` (2026-09-21). They were not moved anywhere; nothing replaced
them. Any instruction to run `python -m pytest -q tests/` from the repo root is
stale, and the command fails with `file or directory not found`.

Until they are restored, the launcher layer is covered by §2 (`--audit-configs`
plus per-config `--check`) and, indirectly, by the Studio backend suite in §3.
The old modules are still recoverable from git:

```bash
git show 9884fb1^:tests/test_config_key_contracts.py   # and the other twelve
```

## 2. Config validation across the whole tree

The fastest way to prove a spec or config change did not break anything:

```bash
python AI_CAE4ALL_main.py --audit-configs        # structural lint of every checked-in config
python AI_CAE4ALL_main.py --config <path> --check # full layered preflight for one config
```

`--audit-configs` parses and route-checks all 315 checked-in configs (measured
2026-09-23: `errors=0, warnings=30`, every warning a `FLOW-COST` advisory on the
deliberately expensive `configs/HI_MGNFlow/{SAOI_run,hyperparameter_sweep}/`
inference arms). A single
`--check` goes further: it also runs the filesystem, environment, dataset, and
native-probe layers, the last three inside the *target method's* interpreter.

## 3. Studio backend

```bash
python -m pytest -q studio/studio_backend
```

Ten modules / 43 tests covering the analysis backends, evaluation contract,
training metrics, checkpoint support, the benchmark roster, HDF5 summaries,
geometry preview, LLM configure, the static allowlist, and the pipeline launch
gate.

## 4. Method repositories

Each runs in that method's environment, from that method's directory:

```bash
cd methods/Neural_Operator          && python -m pytest -q tests/
cd methods/MeshGraphNets            && python -m pytest -q tests/
cd methods/MeshGraphNets_Variational && python -m pytest -q tests/
cd methods/HI_MGNFlow               && python -m pytest -q tests/
cd methods/SDFFlow                  && python -m pytest -q tests/
cd methods/SimulGenVAE              && python -m pytest -q tests/
cd methods/MLP                      && python -m pytest -q tests/
cd methods/Transolver               && python -m pytest -q tests/
```

These run on tiny synthetic HDF5 fixtures, not on `dataset/*.h5`, and finish in
well under a minute each. `methods/GeometryIngest/` ships no test package; it is
covered by its two `--check` configs.

## 5. Compile check

After editing Python across several repositories:

```bash
python -m compileall -q methods cae_suite studio/studio_backend tests
```

## Notes

- `pyproject.toml`'s `[tool.pytest.ini_options] testpaths = ["tests"]` refers to
  the root suite only. Method repositories are never collected from the root.
- A passing structural test is **not** evidence that a published benchmark result
  was reproduced. Report a benchmark number only with the exact dataset,
  checkpoint or log, metric implementation, and run provenance attached.
