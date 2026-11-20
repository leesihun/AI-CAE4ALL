# Model Capabilities

What each Neural-Operator model can chunk or cache, and what "exact" means for
it. Every row is covered by a query-chunk parity test
(`tests/test_query_chunking.py` plus each model's `test_<model>.py`). Nothing
labeled exact may change predictions, loss or gradients beyond the fp32
tolerances asserted in `tests/`. User-facing docs:
[05_DeepONet.md](../../methods/05_DeepONet.md),
[06_Point-DeepONet.md](../../methods/06_Point-DeepONet.md),
[07_FNO.md](../../methods/07_FNO.md).

| Model | `encode_operator` returns | Exact `decode_queries` chunk | Not chunkable |
| --- | --- | --- | --- |
| Point-DeepONet | branch context `[num_graphs, H]` | trunk + fusion + refiners over a node range | PointNet branch (runs once over the sensor set) |
| DeepONet | branch coefficients `[num_graphs, output_var, basis_dim]` | trunk + modal product over a node range | the sensor-grid splat (one grid per graph) |
| FNO | output grid `[num_graphs, output_var, *resolution]` | grid-to-node `grid_sample` over a node range | the spectral core (Fourier layers are global over the grid) |

## Sampling and resolution knobs change the model

`point_sensor_count`, `deeponet_sensor_resolution`, `fno_grid_resolution` and
`fno_modes` change what the model computes: the operator input distribution or
its Fourier capacity. Report them with every accuracy number. Changing them for
speed makes results incomparable across the change.

## Memory controls that do not change results

- `use_checkpointing True`: recomputes each FNO block in backward
  (`torch.utils.checkpoint.checkpoint`, `use_reentrant=False`, in
  `MeshFNO._forward_grid_stack` and `MeshFNO.pipeline_block`). Exact by
  construction. DeepONet and Point-DeepONet ignore the key.
- `time_integration ar_rt` always checkpoints each unrolled step while training
  (`training_profiles/ar_rollout.py`), independent of `use_checkpointing`.
- `train_query_chunk_size` / `infer_query_chunk_size`: split the query range fed
  to `decode_queries`; `0` disables chunking. `test_query_chunking.py` verifies
  exactness.
- `grad_accum_steps`: accumulates gradients over micro-batches before an
  optimizer step. This changes optimizer cadence, not the per-sample forward.

## fp32 islands under bf16 autocast

These ops compute in float32 whatever the ambient AMP setting, because bf16
loses too much precision or the primitive (FFT) does not support it:

- `model/spectral.py::SpectralConvNd.forward` (`torch.fft.rfftn`/`irfftn`,
  autocast disabled).
- `model/deeponet.py::DeepONet._decode` and
  `model/point_deeponet.py::PointDeepONet._decode` (final modal product and
  einsum, autocast disabled).
- `model/adapters/grid.py::splat`: accumulates in explicit fp32 buffers
  regardless of input dtype.

BatchNorm1d (PointNet) already runs in fp32 under autocast and needs no extra
handling.
