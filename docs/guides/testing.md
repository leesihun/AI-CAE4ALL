# What to run after a change

Tests live in the layer they cover. There is no single root command that runs
everything, because each method repository is validated inside **its own venv**
with its own dependencies.

## 1. Launcher and MethodSpec contracts

Runs in the launcher's own interpreter; needs no ML dependencies.

```bash
python -m pytest -q tests/
```

Nine modules covering config discovery, key contracts, documentation links,
native config-consumption parity, native runtime defaults, the SDFFlow spec,
Studio/spec key parity, and — under `tests/campaign/` — the campaign scheduler
and the `methods/` layout anchors.

## 2. Config validation across the whole tree

The fastest way to prove a spec or config change did not break anything:

```bash
python AI_CAE4ALL_main.py --audit-configs        # structural lint of every checked-in config
python AI_CAE4ALL_main.py --config <path> --check # full layered preflight for one config
```

`--audit-configs` parses and route-checks all 269 checked-in configs. A single
`--check` goes further: it also runs the filesystem, environment, dataset, and
native-probe layers, the last three inside the *target method's* interpreter.

## 3. Studio backend

```bash
python -m pytest -q studio/studio_backend
```

Eight modules covering the analysis backends, evaluation contract, training
metrics, checkpoint support, and the pipeline launch gate.

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
