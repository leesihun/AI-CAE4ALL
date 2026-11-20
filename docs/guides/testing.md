# What to run after a change

Each layer has its own tests, and no single command runs them all. Each method
is tested in **its own venv**, with its own dependencies. There is no root
`tests/` directory.

| Layer | Command | Interpreter / cwd |
| --- | --- | --- |
| Launcher and specs, all configs | `python AI_CAE4ALL_main.py --audit-configs` | launcher, repo root |
| One config, full preflight | `python AI_CAE4ALL_main.py --config <path> --check` | launcher, repo root |
| Studio backend | `python -m pytest -q studio/studio_backend` | launcher, repo root |
| A method | `python -m pytest -q tests/` | that method's venv, from `methods/<Name>/` |

## Launcher and configs

```bash
python AI_CAE4ALL_main.py --audit-configs          # parse + spec + route lint of every checked-in config*.txt
python AI_CAE4ALL_main.py --config <path> --check  # every preflight layer for one config
```

`--audit-configs` is structural only and is fast. `--check` goes further: after
the spec, it also runs the filesystem, environment, dataset, checkpoint and
native-probe layers. The probes run inside the target method's interpreter
(resolved through `ai_cae4all.local.toml`). Add `--strict` to promote flagged
warnings to errors.

## Studio backend

```bash
python -m pytest -q studio/studio_backend
```

`pyproject.toml` sets `testpaths = ["studio/studio_backend"]`, so a bare
`pytest` from the repo root runs the same suite. `studio_backend/conftest.py`
points the Studio runtime root at a temporary directory for the run.

## Method repositories

Run each from the method's directory, in that method's environment:

```bash
cd methods/MeshGraphNets             && python -m pytest -q tests/
cd methods/MeshGraphNets_Variational && python -m pytest -q tests/
cd methods/Neural_Operator           && python -m pytest -q tests/
cd methods/Transolver                && python -m pytest -q tests/
cd methods/SDFFlow                   && python -m pytest -q tests/
cd methods/SimulGenVAE               && python -m pytest -q tests/
cd methods/MLP                       && python -m pytest -q tests/
```

These tests use small synthetic fixtures, not `dataset/*.h5`.
`methods/GeometryIngest/` has no `tests/`; use `--check` on its configs instead.
The method repos are never collected from the root.

## Compile check

After editing Python across several packages:

```bash
python -B -m compileall -q methods ai_cae4all studio/studio_backend
```

This writes `__pycache__/` directories. To check without writing into the tree,
run the relevant tests with `python -B`.

## Notes

- `resume_state.py` is copied byte-for-byte into every training repo except MLP.
  After editing it, run the tests of every repo that has a copy.
- A passing structural test does not show that a published benchmark result was
  reproduced. Report a benchmark number only together with its dataset,
  checkpoint or log, metric implementation and run provenance.
