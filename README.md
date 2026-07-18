# CAE ML Suite Launcher

Run any installed AI-CAE method from one command. The `model` field in the
native config selects the repository and entrypoint automatically.

```powershell
python CAE_ML_Suite_main.py --config MeshGraphNets\ex1\config_train1.txt
```

SDFFlow geometry generation uses one production training config. It trains the
VAE first, verifies its checkpoint, and immediately trains flow matching:

```powershell
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_train.txt --check
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_train.txt
python CAE_ML_Suite_main.py --config Geometry_generation\ex1\config_sample.txt
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
| `transolver` | `transolver/Transolver_main.py` |
| `sdfflow` | `Geometry_generation/SDFFlow_main.py` |

By default, a successful preflight starts the selected native process
automatically. Use `--check` for validation only or `--dry-run` to print the
resolved command without starting it. Paths in native configs continue to be
resolved from the selected method repository, matching direct native runs.

Useful commands:

```powershell
python CAE_ML_Suite_main.py --list-models
python CAE_ML_Suite_main.py --describe transolver
python CAE_ML_Suite_main.py --config transolver\ex1\config_train_smoke.txt --check
python CAE_ML_Suite_main.py --config Neural_Operator\ex1\config_train_smoke_fno.txt --dry-run
python CAE_ML_Suite_main.py --audit-configs
```

For method-specific Python environments, copy `cae_suite.local.example.toml`
to `cae_suite.local.toml` and set the desired interpreter paths. This local
file is ignored by Git.

See [DATASET_CONFIG_OUTPUT_REFERENCE.md](DATASET_CONFIG_OUTPUT_REFERENCE.md)
for the current dataset, config, checkpoint, and output contracts. The
[implementation plan](IMPLEMENTATION_PLAN.md) describes the launcher design.
