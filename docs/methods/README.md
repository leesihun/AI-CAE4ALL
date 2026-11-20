# Method documentation

One doc per launcher route, plus shared foundations. Each doc describes the
implementation, not the paper. The live `MethodSpec` (`ai_cae4all/specs/`) and
each method's own `CLAUDE.md`/`README.md` are authoritative for the full key
contract. When a doc and the code disagree, the code is right.

## The docs

| # | Doc | `model` | Directory | Family |
| --- | --- | --- | --- | --- |
| 00 | [00_shared_foundations.md](00_shared_foundations.md) | — | — | shared data and training contract |
| 01 | [01_MeshGraphNets_MGN.md](01_MeshGraphNets_MGN.md) | `meshgraphnets` | `methods/MeshGraphNets/` | GNN simulator |
| 02 | [02_HI-MGN.md](02_HI-MGN.md) | `meshgraphnets` + `use_multiscale` | `methods/MeshGraphNets/` | hierarchical GNN |
| 03 | [03_BSMS-GNN.md](03_BSMS-GNN.md) | `meshgraphnets` + `coarsening_type bfs` | `methods/MeshGraphNets/` | hierarchical GNN |
| 04 | [04_MeshGraphNets_Variational.md](04_MeshGraphNets_Variational.md) | `meshgraphnets-v` | `methods/MeshGraphNets_Variational/` | generative GNN, one-to-many |
| 05 | [05_DeepONet.md](05_DeepONet.md) | `deeponet` | `methods/Neural_Operator/` | neural operator |
| 06 | [06_Point-DeepONet.md](06_Point-DeepONet.md) | `point_deeponet` | `methods/Neural_Operator/` | neural operator |
| 07 | [07_FNO.md](07_FNO.md) | `fno` | `methods/Neural_Operator/` | neural operator |
| 08 | [08_Transolver.md](08_Transolver.md) | `transolver` | `methods/Transolver/` | transformer operator |
| 09 | [09_SDFFlow.md](09_SDFFlow.md) | `sdfflow` | `methods/SDFFlow/` | generative geometry, design loop |
| 10 | [10_Geometry_Ingest.md](10_Geometry_Ingest.md) | `geometry_ingest` | `methods/GeometryIngest/` | data prep (not ML) |
| 11 | [11_MLP.md](11_MLP.md) | `mlp` | `methods/MLP/` | tabular surrogate |
| 12 | [12_SimulGenVAE.md](12_SimulGenVAE.md) | `simulgenvae` | `methods/SimulGenVAE/` | fixed-geometry hierarchical VAE |

MGN, HI-MGN and BSMS-GNN are one codebase with different configuration. They
share the building blocks described in
[00_shared_foundations.md](00_shared_foundations.md).

The optional local-only routes `sar` and `ldgn` have no doc here. They are
documented in their own READMEs, when present.

## Running any method

```bash
python AI_CAE4ALL_main.py --config <config.txt> --check     # validate only
python AI_CAE4ALL_main.py --config <config.txt> --dry-run   # print the native command
python AI_CAE4ALL_main.py --config <config.txt>             # validate + launch
python AI_CAE4ALL_main.py --describe <model>                # required keys per mode
```

`model` and `mode` both live inside the config.

## Selection guide

| If you need to… | Use |
| --- | --- |
| Predict mesh-resolved transient or static fields | MGN |
| The same on large meshes with long-range coupling | HI-MGN, BSMS-GNN |
| Several plausible responses for one input (one-to-many) | MGN-V |
| Learn an operator across geometries or parameters, queryable at any point | DeepONet, Point-DeepONet |
| Structured or near-grid domains | FNO |
| An attention operator on large meshes | Transolver |
| Generate or optimize 3D shapes | SDFFlow |
| Mesh CAD into the shared HDF5 | GeometryIngest |
| Scalar outputs from scalar design parameters | MLP |
| Full-field responses on one fixed mesh, from a few conditions | SimulGenVAE |
