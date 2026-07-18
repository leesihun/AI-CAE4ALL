# CLAUDE.md

Agent-facing notes for this deterministic operator-learning repository. Keep
answers and edits grounded in the live code and `IMPLEMENTATION_PLAN.md`, not
assumptions carried over from MeshGraphNets or the published papers.

## Project Objective

One repository, four selectable operator architectures
(`point_deeponet`, `deeponet`, `fno`, `gino`), all reading the existing
MeshGraphNets HDF5 files with no conversion step, all sharing the same
split/target/normalization/noise/optimizer/scheduler/checkpoint/rollout
conventions. Switching `model` in the config must never require touching
dataset, training-loop, loss, checkpoint, or inference code.

The repository is fully self-contained at runtime: no network access, no
`neuraloperator` dependency. FNO and GINO are implemented natively
(`model/spectral.py`, `model/gno.py`).

## Run Commands

```bash
python main.py --config ex1/config_train_smoke_deeponet.txt
python main.py --config ex1/config_infer_smoke_deeponet.txt
```

`mode` (`train`/`inference`) lives inside the config file; `--config` only
selects which file to read.

## Key Files

| File | Role |
| --- | --- |
| [main.py](main.py) | Config load, mode dispatch, DDP spawn. |
| [model/factory.py](model/factory.py) | The only place models are constructed; `MODEL_REGISTRY`/`VALIDATORS`. |
| [model/operator_wrapper.py](model/operator_wrapper.py) | Noise contract + batch/ptr synthesis; the only thing training/inference calls. |
| [model/base.py](model/base.py) | `OperatorCore` protocol every architecture implements. |
| [model/deeponet.py](model/deeponet.py) | Fixed-sensor DeepONet (splat branch + trunk dot product). |
| [model/point_deeponet.py](model/point_deeponet.py) | Primary model: PointNet branch + SIREN trunk + early fusion. |
| [model/fno.py](model/fno.py) + [model/spectral.py](model/spectral.py) | Mesh-adapted FNO; native spectral convolution. |
| [model/gino.py](model/gino.py) + [model/gno.py](model/gno.py) | GINO; native GNO kernel integral, per-graph loop. |
| [model/adapters/grid.py](model/adapters/grid.py) | Deterministic splat/sample; the axis-order convention is documented at the top of the file. |
| [model/adapters/coordinate_domain.py](model/adapters/coordinate_domain.py) | Active axes, `[0,1]^d` mapping, out-of-bounds policy. |
| [model/adapters/point_sampling.py](model/adapters/point_sampling.py) | Deterministic fixed-size sensor sampling. |
| [model/adapters/radius_neighbors.py](model/adapters/radius_neighbors.py) | scipy KDTree baseline + optional torch_cluster for GINO. |
| [general_modules/mesh_dataset.py](general_modules/mesh_dataset.py) | HDF5 loading, split, normalization, `pos_normalized`, augmentation. |
| [general_modules/dataset_stats.py](general_modules/dataset_stats.py) | Moments, `position_scale`, active axes, grid bounds, rotation-safe radius. |
| [general_modules/config_validation.py](general_modules/config_validation.py) | Full key registry; unknown/legacy keys fail fast. |
| [training_profiles/setup.py](training_profiles/setup.py) | Dataset/model/EMA/optimizer/checkpoint helpers. |
| [training_profiles/training_loop.py](training_profiles/training_loop.py) | Train/validate/test; node-weighted loss; EMA (incl. BatchNorm buffer copy). |
| [parallelism/launcher.py](parallelism/launcher.py) | `parallel_mode model_split`: 1F1B pipeline split (fno/gino only), ported from MGN; merged checkpoints load like single-GPU ones. |
| [parallelism/stages.py](parallelism/stages.py) | Stage = seeded full core pruned to its block range; deterministic cross-stage noise. |
| [inference_profiles/rollout.py](inference_profiles/rollout.py) | Checkpoint-led static inference and autoregressive rollout. |

## Architecture Facts

- Every core's `forward(graph) -> prediction [sum_N, output_var]`; noise and
  batch/ptr synthesis live only in `OperatorWrapper`, never in a core.
- No model consumes MGN edge attributes; `edge_index` is kept only for
  positional-feature computation and output writing.
- `DataSpec` (immutable, `general_modules/data_spec.py`) is the single source
  of truth for channel widths and active axes; adapters/models slice `x` via
  its `physical_slice`/`context_slice`/`onehot_slice`, never magic offsets.
- Grid axis convention: tensor dim 2 = active axis 0 (x), dim 3 = axis 1 (y),
  dim 4 = axis 2 (z). `F.grid_sample`'s convention is the reverse (channel 0
  addresses the *last* spatial dim), so `model/adapters/grid.py::sample`
  permutes the input's spatial dims, never the coordinate channel order.
  This is the single most bug-prone spot in the repo — read that file's
  docstring before touching it.
- Point-DeepONet's SIREN trunk (`model/siren.py`) has its own sine-specific
  init and must never receive the repo's generic Kaiming `init_weights` pass.
- EMA (`training_profiles/training_loop.py::update_ema`) copies BatchNorm
  running-mean/var buffers after every parameter update; `AveragedModel`
  only averages parameters, and PointNet's BatchNorm would otherwise never
  update its running stats.
- Spectral conv weights are stored as real tensors with a trailing size-2
  dim, viewed as complex only inside `forward` — this is required for fused
  AdamW, which rejects complex parameters.
- `parallel_mode model_split` (fno/gino only) cuts the core into pipeline
  blocks: entry (splat / input GNO + lifting) → latent FNO blocks → exit
  (projection / output GNO). Stages are seeded full cores pruned to their
  block range, so state-dict keys never change and merged checkpoints load
  through the normal inference path. The DeepONets and `augment_geometry
  True` are rejected at config validation.

## Data Facts

- `nodal_data[0:3]` are reference coordinates, not part of `input_var`.
- Node type (Part No.) is the *last* row and only exists when the file has
  more than 7 feature rows; `use_node_types True` on a 7-row file is an error.
- Static (`T==1`) targets are the direct stored field from zero input;
  temporal (`T>1`) targets are `state[t+1] - state[t]`.
- `ex1.h5` is planar (z ≡ 0, `operator_dim` resolves to 2); `ex2.h5` is
  genuinely 3D. Neither fact is hardcoded — both come from
  `dataset_stats.resolve_active_axes` on the actual training geometry.
- The stored `metadata/normalization_params`/`metadata/splits` groups are
  ignored; this repo always recomputes its own split/stats and never writes
  back into a source HDF5 (`write_preprocessing` is accepted only as `False`
  and rejected at config validation otherwise).

## Config Parser Gotchas

- A config line with a single value parses to a bare `int`/`float`/`str`,
  **not** a one-element list (`test_batch_idx 0` -> `0`). Any code that
  expects a list from such a key must normalize first
  (`training_profiles/training_loop.py::_as_list` is the existing helper —
  reuse it, don't re-invent it).
- Numeric values with no `.` and no scientific notation stay `int`;
  `1e-4`-style values parse as a **string** (no `.` character), so every
  numeric use site converts explicitly (`float(config.get('weight_decay',
  1e-4))`). This is inherited from MeshGraphNets' parser and is intentional
  — do not "fix" it without updating every call site and its test.
- Unknown or legacy (removed-feature) config keys raise immediately in
  `general_modules/config_validation.py`, before any HDF5 file is opened.

## Testing

`pytest tests/` runs entirely on tiny synthetic HDF5 fixtures
(`tests/conftest.py`) and finishes in well under a minute. It does not touch
`dataset/ex1.h5`/`ex2.h5`. Real-data smoke runs use the `ex1/config_*_smoke_*.txt`
configs directly via `main.py` (see QUICKSTART.md).

## Documentation Notes

The authoritative docs are:

- [README.md](README.md)
- [QUICKSTART.md](QUICKSTART.md)
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- [dataset/DATASET_FORMAT.md](dataset/DATASET_FORMAT.md)
- [docs/POINT_DEEPONET_PARITY.md](docs/POINT_DEEPONET_PARITY.md)
- [docs/GINO_PARITY.md](docs/GINO_PARITY.md)
- [docs/MODEL_CAPABILITIES.md](docs/MODEL_CAPABILITIES.md)

`IMPLEMENTATION_PLAN.md` records the design decisions and their rationale;
when it and the code disagree, the code is authoritative for *current*
behavior, but treat a mismatch as something to reconcile (update whichever
side is stale), not something to ignore.
