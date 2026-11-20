# AI-CAE4ALL documentation

All suite documentation lives here. A method directory keeps only its own
`CLAUDE.md`/`README.md`, listed at the bottom. Config keys are defined by
[`ai_cae4all/specs/`](../ai_cae4all/specs/); when a document and the code
disagree, the code is right.

## Start here

| Doc | Read it when |
| --- | --- |
| [../README.md](../README.md) | You want the tour: what the suite is, the model zoo, quick start |
| [ARCHITECTURE.md](ARCHITECTURE.md) | You need the layout, the launcher internals, the config grammar, the cross-cutting conventions or the dataset tree |
| [CONFIGURATION.md](CONFIGURATION.md) | You are writing or debugging a config |
| [GUI.md](GUI.md) | You are using or extending the browser Studio |
| [guides/testing.md](guides/testing.md) | You changed something and want to know what to run |

## Guides

| Doc | Covers |
| --- | --- |
| [GUI.md](GUI.md) | The Studio manual: install, workspaces, blocks, templates, presets, Validate/Run, evaluation rules, the local API |
| [guides/inference-bundle.md](guides/inference-bundle.md) | The portable CPU inference bundle: checkpoint family detection, CLI, `.exe` build |
| [guides/testing.md](guides/testing.md) | Which test command belongs to which layer |
| [videos/](videos/README.md) | Two recorded Studio walkthroughs of the DeepJEB design loop |

## Reference

| Doc | Covers |
| --- | --- |
| [reference/DATASET_FORMAT.md](reference/DATASET_FORMAT.md) | The shared mesh HDF5 contract, the rollout output format and the MLP tabular exception |
| [reference/PUBLIC_DATASETS.md](reference/PUBLIC_DATASETS.md) | Where every dataset in `dataset/` comes from, how it was converted, and its traps |
| [reference/config/meshgraphnets_variational.md](reference/config/meshgraphnets_variational.md) | Every MGN-V key, including `latent_type` and the removed/renamed keys |

## Methods

[methods/](methods/README.md) has one write-up per method, plus shared
foundations:

| | |
| --- | --- |
| [00 Shared foundations](methods/00_shared_foundations.md) | [01 MeshGraphNets](methods/01_MeshGraphNets_MGN.md) |
| [02 HI-MGN](methods/02_HI-MGN.md) | [03 BSMS-GNN](methods/03_BSMS-GNN.md) |
| [04 MeshGraphNets-V](methods/04_MeshGraphNets_Variational.md) | [05 DeepONet](methods/05_DeepONet.md) |
| [06 Point-DeepONet](methods/06_Point-DeepONet.md) | [07 FNO](methods/07_FNO.md) |
| [08 Transolver](methods/08_Transolver.md) | [09 SDFFlow](methods/09_SDFFlow.md) |
| [10 Geometry Ingest](methods/10_Geometry_Ingest.md) | [11 MLP](methods/11_MLP.md) |
| [12 SimulGenVAE](methods/12_SimulGenVAE.md) | |

## Research notes

[research/](research/) keeps design notes and investigations by method. They
record why something was built the way it was. They are context, not a
description of current behavior.

- [meshgraphnets/](research/meshgraphnets/): attention transfer operators,
  coarsening and hierarchy ablations, the HI-MGN roadmap
- [meshgraphnets_variational/](research/meshgraphnets_variational/):
  distribution modeling (MGN-V vs VGAE vs LDGN), the latent placement plan
  (HI-MGN-V vs HI-MGNflow vs HI-MGN-hybrid)
- [SAOI_PROBABILISTIC_SWEEP_2026-09.md](research/SAOI_PROBABILISTIC_SWEEP_2026-09.md):
  MGN-V vs HI-MGNflow on SAOI warpage
- `mgnv_conditional_prior_2026-09-16/`: a frozen report snapshot; its scripts
  depend on the directory depth, so do not move it
- [neural_operator/](research/neural_operator/): model capabilities,
  Point-DeepONet parity with the paper
- [transolver/](research/transolver/): foundation-model design
- [sdfflow/](research/sdfflow/): conditional geometry survey, generation
  research, meshing upgrade, guidance mechanisms, conditional generation design,
  foundation-model plan
- [shell_buckling/](research/shell_buckling/): the OpenRadioss shell buckling
  campaign behind `dataset/probabilistic/ex3_shell_buckling*.h5`. Campaign code
  is at `D:\CAE_datasets_raw\shell_buckling\`; raw runs are on aarl at
  `/data/Lee/buckling_radioss/`

## Per-method notes

Authoritative for that method's internals:

- [methods/Neural_Operator/CLAUDE.md](../methods/Neural_Operator/CLAUDE.md)
- [methods/SDFFlow/CLAUDE.md](../methods/SDFFlow/CLAUDE.md), [README.md](../methods/SDFFlow/README.md)
- [methods/SimulGenVAE/CLAUDE.md](../methods/SimulGenVAE/CLAUDE.md), [README.md](../methods/SimulGenVAE/README.md)
- [methods/MLP/CLAUDE.md](../methods/MLP/CLAUDE.md)
- [methods/GeometryIngest/README.md](../methods/GeometryIngest/README.md)
- `methods/SAR/README.md`, `methods/LDGN/README.md`: local-only optional routes
