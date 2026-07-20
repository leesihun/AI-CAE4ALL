# AI-CAE4ALL Launcher

Run any installed AI-CAE method from one command. The `model` field in the
native config selects the repository and entrypoint automatically.

```bash
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/ex1/config_train1.txt
```

SDFFlow geometry generation uses one production training config. It trains the
VAE first, verifies its checkpoint, and immediately trains flow matching:

```bash
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt --check
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_train.txt
python AI_CAE4ALL_main.py --config configs/Geometry_generation/config_sample.txt
```

Relaunching the training config safely reuses compatible completed stages;
retraining the VAE invalidates reuse of an older FM checkpoint. The geometry
repository also ships guarded extrapolation and reproducible interpolation
configs beside the main sample config.

The launcher validates the complete applicable config first and reports all
missing or conflicting settings together. It does not start a model while a
blocking preflight error remains.

Supported `model` values and aliases:

| Config value | Native backend |
| --- | --- |
| `meshgraphnets` | `MeshGraphNets/MeshGraphNets_main.py` |
| `meshgraphnets-v` | `MeshGraphNets - variational/MeshGraphNets_main.py` |
| `point_deeponet`, `deeponet`, `fno`, `gino` | `Neural_Operator/main.py` |
| `transolver` | `Transolver/Transolver_main.py` |
| `sdfflow` | `Geometry_generation/SDFFlow_main.py` |

By default, a successful preflight starts the selected native process
automatically. Use `--check` for validation only or `--dry-run` to print the
resolved command without starting it. Paths in native configs continue to be
resolved from the selected method repository, matching direct native runs.

Useful commands:

```bash
python AI_CAE4ALL_main.py --list-models
python AI_CAE4ALL_main.py --describe transolver
python AI_CAE4ALL_main.py --config configs/Transolver/ex2/config_train_transolver.txt --check
python AI_CAE4ALL_main.py --config configs/Neural_Operator/ex1/config_train_fno.txt --dry-run
```

On Ubuntu, activate a venv before launching and no local TOML is required; the
launcher uses that venv's Python. For separate backend venvs, copy
`ai_cae4all.local.example.toml` to `ai_cae4all.local.toml` and set their Python
paths. Relative paths in that file resolve from the repository root. The local
file is ignored by Git. Installing the root package in the venv with
`python -m pip install -e .` also provides the `ai-cae4all` command.

See [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md) for the exhaustive
live-code-backed key catalog, necessity classifications, shipped-config
inventory, script flags, and current launcher/native mismatches. The shared
mesh HDF5 contract is documented in [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md).
