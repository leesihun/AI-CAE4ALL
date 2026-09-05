# AI-CAE4ALL documentation

Central suite, reference, and research documentation lives here. Method-local
`README.md` and `CLAUDE.md` files, plus a few utility notes, remain beside the
code; their primary entry points are indexed below.

## Start here

| Doc | Read it when |
| --- | --- |
| [../README.md](../README.md) | You want the tour: what the suite is, the model zoo, quick start |
| [ARCHITECTURE.md](ARCHITECTURE.md) | You need the full picture — launcher internals, the config system, a section per method, known gaps |
| [CONFIGURATION.md](CONFIGURATION.md) | You are writing or debugging a config file |
| [GUI.md](GUI.md) | You are using the browser Studio: blocks, templates, validation, evaluation rules |
| [guides/testing.md](guides/testing.md) | You changed something and want to know what to run |

## Guides

| Doc | Covers |
| --- | --- |
| [GUI.md](GUI.md) | The Studio manual: the block library, the shipped pipeline templates, the config sheet, Validate/Run, and the evaluation contract |
| [guides/studio.md](guides/studio.md) | The Studio's front-end notes: capabilities list, local API surface, integration boundary |
| [guides/inference-bundle.md](guides/inference-bundle.md) | The portable CPU inference bundle: checkpoint family detection, CLI, `.exe` build |
| [guides/testing.md](guides/testing.md) | Which test command belongs to which layer |

## Reference

| Doc | Covers |
| --- | --- |
| [reference/DATASET_FORMAT.md](reference/DATASET_FORMAT.md) | The shared mesh HDF5 contract, plus the tabular and SDF exceptions |
| [reference/PUBLIC_DATASETS.md](reference/PUBLIC_DATASETS.md) | Provenance of the public benchmark datasets |
| [reference/config/](reference/config/) | Per-method config key references (HI-MGNflow, MeshGraphNets variational) |

The executable source of truth for config keys is always
[`cae_suite/specs/`](../cae_suite/specs/), not a document.

## Methods

[methods/](methods/) holds thirteen numbered write-ups covering the shared
foundations and the architecture guides listed below:

| | |
| --- | --- |
| [00_shared_foundations.md](methods/00_shared_foundations.md) | [01_MeshGraphNets_MGN.md](methods/01_MeshGraphNets_MGN.md) |
| [02_HI-MGN.md](methods/02_HI-MGN.md) | [03_BSMS-GNN.md](methods/03_BSMS-GNN.md) |
| [04_MeshGraphNets_Variational.md](methods/04_MeshGraphNets_Variational.md) | [05_DeepONet.md](methods/05_DeepONet.md) |
| [06_Point-DeepONet.md](methods/06_Point-DeepONet.md) | [07_FNO.md](methods/07_FNO.md) |
| [08_GINO.md](methods/08_GINO.md) | [09_Transolver.md](methods/09_Transolver.md) |
| [10_SDFFlow.md](methods/10_SDFFlow.md) | [11_Geometry_Ingest.md](methods/11_Geometry_Ingest.md) |
| [12_MLP.md](methods/12_MLP.md) | |

SimulGenVAE's current user guide remains beside its implementation:
[methods/SimulGenVAE/README.md](../methods/SimulGenVAE/README.md).

## Research notes

[research/](research/) collects design documents and investigations, grouped by
the method they belong to. These record *why* something is built the way it is;
when a research note and the code disagree, the code is authoritative for current
behaviour — treat the mismatch as something to reconcile, not to ignore.

- [research/meshgraphnets/](research/meshgraphnets/) — attention transfer, coarsening ablation, HI-MGN roadmap
- [research/meshgraphnets_variational/](research/meshgraphnets_variational/) — distribution modelling, world edges, VRAM and performance work
- [research/neural_operator/](research/neural_operator/) — GINO and Point-DeepONet parity, model capabilities
- [research/transolver/](research/transolver/) — foundation-model design
- [research/sdfflow/](research/sdfflow/) — conditional geometry survey, meshing upgrade, guidance mechanisms, [conditional generation design](research/sdfflow/CONDITIONAL_GENERATION_DESIGN_2026-09.md) (FEA-label conditions, per-dim dropout, C2/E2, the load-unit correction)
- [research/hi_mgnflow/](research/hi_mgnflow/) — deterministic mode, sweep plan
- [research/simulgenvae/](research/simulgenvae/) — technical documentation

## Per-method notes

Authoritative for that method's internals, kept beside the code:

- [methods/Neural_Operator/CLAUDE.md](../methods/Neural_Operator/CLAUDE.md)
- [methods/SDFFlow/CLAUDE.md](../methods/SDFFlow/CLAUDE.md)
- [methods/SDFFlow/README.md](../methods/SDFFlow/README.md)
- [methods/SimulGenVAE/CLAUDE.md](../methods/SimulGenVAE/CLAUDE.md)
- [methods/SimulGenVAE/README.md](../methods/SimulGenVAE/README.md)
- [methods/MLP/CLAUDE.md](../methods/MLP/CLAUDE.md)
- [methods/HI_MGNFlow/README.md](../methods/HI_MGNFlow/README.md)
- [methods/GeometryIngest/README.md](../methods/GeometryIngest/README.md)
