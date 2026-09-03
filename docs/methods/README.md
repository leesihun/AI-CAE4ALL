# AI-CAE4ALL — Method Documentation

This folder indexes **every ML-for-CAE method** shipped in the AI-CAE4ALL
monorepo and contains the shared foundations plus architecture-specific guides.
The live `MethodSpec` and native validators remain authoritative for exhaustive
config-key contracts.

Everything here is derived directly from the live code (`model/*.py`,
`general_modules/*.py`, `training_profiles/*.py`) and the checked-in configs, not
from the original papers — where the implementation deviates from a paper, the
docs describe the implementation.

## The methods

| # | Doc | `model` config value | Runtime directory | Family |
| --- | --- | --- | --- | --- |
| 0 | [00_shared_foundations.md](00_shared_foundations.md) | — | shared conventions | data + training contract |
| 1 | [01_MeshGraphNets_MGN.md](01_MeshGraphNets_MGN.md) | `meshgraphnets` | `methods/MeshGraphNets/` | GNN simulator |
| 2 | [02_HI-MGN.md](02_HI-MGN.md) | `meshgraphnets` (+`use_multiscale`) | `methods/MeshGraphNets/` | Hierarchical GNN |
| 3 | [03_BSMS-GNN.md](03_BSMS-GNN.md) | `meshgraphnets` (+`coarsening_type bfs`) | `methods/MeshGraphNets/` | Hierarchical GNN |
| 4 | [04_MeshGraphNets_Variational.md](04_MeshGraphNets_Variational.md) | `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | Generative GNN (cVAE) |
| 5 | [05_DeepONet.md](05_DeepONet.md) | `deeponet` | `methods/Neural_Operator/` | Neural operator |
| 6 | [06_Point-DeepONet.md](06_Point-DeepONet.md) | `point_deeponet` | `methods/Neural_Operator/` | Neural operator |
| 7 | [07_FNO.md](07_FNO.md) | `fno` | `methods/Neural_Operator/` | Neural operator |
| 8 | [08_GINO.md](08_GINO.md) | `gino` | `methods/Neural_Operator/` | Neural operator |
| 9 | [09_Transolver.md](09_Transolver.md) | `transolver` | `methods/Transolver/` | Transformer operator |
| 10 | [10_SDFFlow.md](10_SDFFlow.md) | `sdfflow` | `methods/SDFFlow/` | Generative geometry |
| 11 | [11_Geometry_Ingest.md](11_Geometry_Ingest.md) | `geometry_ingest` | `methods/GeometryIngest/` | Data-prep utility (not an ML method) |
| 12 | [12_MLP.md](12_MLP.md) | `mlp` | `methods/MLP/` | Parametric MLP surrogate (tabular, no mesh) |
| 13 | [SimulGenVAE user guide](../../methods/SimulGenVAE/README.md) | `simulgenvae` | `methods/SimulGenVAE/` | Fixed-geometry hierarchical VAE |

> **`geometry_ingest` is a data-prep utility, not an ML method.** It is included
> here because it is a launcher-routed `model`, but it trains nothing: it meshes
> CAD/geometry (STEP/IGES/STL) into the shared HDF5 contract that the ML methods
> above consume. See its doc for details.

> **MGN / HI-MGN / BSMS-GNN share one codebase.** They are not three separate
> repos: MGN is the flat baseline, HI-MGN is the multiscale V-cycle
> (`use_multiscale True` + Voronoi coarsening), and BSMS-GNN is the same V-cycle
> driven by the BFS bi-stride coarsener (`coarsening_type bfs`). Each has its own
> doc because they behave and are configured differently, but they reuse the same
> encoder/processor/decoder building blocks documented in
> [00_shared_foundations.md](00_shared_foundations.md).

## How to run any method

Everything is launched through the root suite launcher; the `model` field inside
the config selects the backend:

```bash
python AI_CAE4ALL_main.py --config <path/to/config.txt> --check     # validate only
python AI_CAE4ALL_main.py --config <path/to/config.txt> --dry-run   # print native command
python AI_CAE4ALL_main.py --config <path/to/config.txt>             # validate + launch
python AI_CAE4ALL_main.py --describe <model>                        # per-mode required keys
```

`mode` (`train` / `inference` / `sample` / …) also lives *inside* the config.

## At-a-glance selection guide

| If you need to… | Use | Why |
| --- | --- | --- |
| Simulate transient physics on a fixed mesh (deformation, stress, crash) | **MGN** | Native mesh message passing; best for local, mesh-resolved fields |
| Same, but on very large meshes with long-range coupling | **HI-MGN / BSMS-GNN** | Multiscale V-cycle propagates information across the mesh in fewer layers |
| Model *manufacturing spread* — many plausible outputs per identical mesh | **MGN-Variational** | cVAE latent + learned conditional prior generate distinct valid variants |
| Learn a solution operator that generalizes across geometries/parameters | **DeepONet / Point-DeepONet / FNO / GINO** | Operator learning; query the field at arbitrary points |
| Best mesh-native operator baseline in this suite | **Point-DeepONet** | PointNet branch + SIREN trunk, no grid projection loss |
| Structured/near-grid domains, spectral efficiency | **FNO** | Global Fourier layers; cheap once splatted to a grid |
| Irregular geometry with a discretization-convergent operator story | **GINO** | Kernel-integral GNO in/out of a latent FNO grid |
| Attention-based operator, one architecture, huge meshes | **Transolver** | Physics-Attention with linear-in-N slice tokens |
| *Generate new 3D shapes* (not simulate a field) | **SDFFlow** | SDF-VAE + latent flow matching, conditioned on shape descriptors |
| Predict a few **scalar** quantities from **scalar** design parameters (no mesh/field) | **MLP** | Plain N→M fully-connected regressor over a tabular DOE / parameter sweep |

See each method's own doc for the detailed strengths/weaknesses and the full
config catalog.
