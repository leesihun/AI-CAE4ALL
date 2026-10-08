# AI-CAE4ALL

**Train simulation models. Generate geometry. Explore design trade-offs.**

A local machine-learning workspace for computer-aided engineering. Build visual
pipelines in **Studio** to connect datasets, native model runtimes, training, and
interactive result viewers. Use the same methods from text configurations with
the command-line launcher.

[Get started](#get-started) | [Models](#models) | [Studio guide](docs/GUI.md) | [Documentation](#documentation)

## Studio walkthroughs

### DeepJEB design workflow

Follow a completed SDFFlow + HI-MGN pipeline: inspect a generated bracket, rotate
its mesh, switch predicted stress and displacement fields, and select a design
from the Pareto plot. **2:39 · English captions · continuous recording**

https://github.com/user-attachments/assets/76410429-83e7-4492-aa9f-42115051c48e

### From an empty canvas to training

Inspect a dataset, add SDFFlow, configure a new run, connect the blocks, and start
training. Browse the model catalog and explore results from existing trained
models. **2:39 · English captions**

https://github.com/user-attachments/assets/72b5dfa4-2f81-4ee2-a1af-0ce5f0ca6796

The training recording has one labelled transition after training starts, then
uses previously trained checkpoints. The DeepJEB example screens 100 generated
candidates using HI-MGN predictions. [Recording details and subtitles](docs/videos/README.md).

## From data to results

- **Inspect data.** Browse geometry, HDF5 samples, and named simulation fields in
  interactive viewers.
- **Build and run pipelines.** Connect compatible blocks, edit model settings,
  validate configurations, and follow training or inference logs.
- **Explore generated designs.** Inspect shapes and predicted fields, apply
  constraints, and compare candidates on a Pareto plot.

## Get started

### Open Studio

Install **Python 3.10+**, clone the repository, and create an environment:

```bash
git clone https://github.com/leesihun/AI-CAE4ALL.git
cd AI-CAE4ALL
python -m venv .venv
```

Activate it with `.\.venv\Scripts\Activate.ps1` on Windows PowerShell, or
`source .venv/bin/activate` on Linux/macOS. Then install and launch:

```bash
python -m pip install -e .
python -m pip install -r studio/requirements.txt
python studio/start_studio.py 8080
```

Open **http://127.0.0.1:8080/index.html**. Keep the terminal running while you use
Studio. Windows users can also use [START_STUDIO.bat](studio/START_STUDIO.bat).

### Run a model

1. Install the selected method's dependencies using its [method guide](docs/methods/).
2. Choose a pipeline template, or add a dataset and model to an empty canvas.
3. Set compatible data, checkpoint/output paths, and GPU selection.
4. Validate the configuration, start the run, and inspect its logs and results.

Each method keeps its own runtime and data contract. For separate environments,
copy [ai_cae4all.local.example.toml](ai_cae4all.local.example.toml) to
`ai_cae4all.local.toml` and set each method's Python interpreter.

**Bring your own data and checkpoints.** The datasets, pretrained weights, and
generated results shown in the videos are local artifacts and are not included
in a clone. Design-generation templates need compatible trained checkpoints.

## Models

| Task | Methods |
| --- | --- |
| Mesh-field prediction | MeshGraphNets, HI-MGN, Transolver |
| Neural operators | DeepONet, Point-DeepONet, FNO |
| Conditional field generation | MeshGraphNets-V, [cHI-MGNflow](methods/HI_MGNFlow/), SimulGenVAE |
| Geometry generation | SDFFlow |
| Tabular response prediction | MLP |
| CAD/geometry preparation | GeometryIngest |

Choose methods with compatible inputs and targets. SDFFlow uses SDF geometry
data; SimulGenVAE requires matching node and timestep counts across samples;
MLP uses tabular inputs and targets. See the [method guides](docs/methods/) and
[dataset reference](docs/reference/DATASET_FORMAT.md).

Inspect the available model IDs and operations:

```bash
python AI_CAE4ALL_main.py --list-models
python AI_CAE4ALL_main.py --describe sdfflow
```

## Use the command line

After setting up the method and updating the example's data paths and GPU
selection, run from the repository root:

```bash
# Validate the configuration.
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/deterministic/ex1/baseline/config_train_himgn.txt --check

# Launch the native method.
python AI_CAE4ALL_main.py --config configs/MeshGraphNets/deterministic/ex1/baseline/config_train_himgn.txt
```

The configuration's `model` and `mode` select the implementation and operation.
See the [configuration reference](docs/CONFIGURATION.md) for accepted keys,
environment selection, and path resolution.

## Documentation

| Guide | Contents |
| --- | --- |
| [Studio](docs/GUI.md) | Pipelines, configuration, jobs, and viewers |
| [Methods](docs/methods/) | Model architectures and setup |
| [Dataset formats](docs/reference/DATASET_FORMAT.md) | Input contracts and field layouts |
| [Public datasets](docs/reference/PUBLIC_DATASETS.md) | Dataset sources |
| [Configuration](docs/CONFIGURATION.md) | CLI options and configuration keys |
| [Architecture](docs/ARCHITECTURE.md) | Launcher and native runtime organization |
| [Portable inference](docs/guides/inference-bundle.md) | Exporting inference bundles |
| [Development](docs/guides/testing.md) | Checks and tests |
| [Video notes](docs/videos/README.md) | Recording provenance, subtitles, and next demos |

Core code lives in `cae_suite/` (launcher), `methods/` (native methods), and
`studio/` (browser UI and local server).
