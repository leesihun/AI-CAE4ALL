# Model Capabilities

Per-architecture chunking/caching guarantees (IMPLEMENTATION_PLAN.md section
8.5) and what "exact" means for each. Everything in the "exact" column is
covered by a query-chunk-parity test (`tests/test_query_chunking.py` plus
each model's own `test_<model>.py`); nothing labeled exact may change
predictions, loss, or gradients beyond the measured fp32 tolerances in
`tests/`.

| Model | `encode_operator` returns | Exact `decode_queries` chunk | What is NOT chunkable |
| --- | --- | --- | --- |
| Point-DeepONet | branch context `[num_graphs, H]` | trunk + fusion + refiners over a node range | PointNet branch (runs once over the sampled/sensor set) |
| DeepONet | branch coefficients `[num_graphs, output_var, basis_dim]` | trunk + modal product over a node range | the fixed-sensor splat (one grid per graph) |
| FNO | predicted output grid `[num_graphs, output_var, *resolution]` | grid-to-node `grid_sample` over a node range | the spectral core itself (Fourier layers are global over the whole grid) |
| GINO | list of per-graph latent features `[prod(resolution), hidden]` (post input-GNO + FNO) | output-GNO + projection over a node range, crossing graph boundaries by slicing each graph's overlap with `[start, end)` | the input GNO + latent FNO (run once per graph to produce the latent feature) |

## Sampling / resolution knobs are architecture changes, not memory controls

`point_sensor_count`, `deeponet_sensor_resolution`, `fno_grid_resolution`/
`fno_modes`, and `gino_grid_resolution`/`gino_fno_modes`/`gino_in_radius`/
`gino_out_radius` all change what the model actually computes (the operator
input distribution or its Fourier/latent capacity). They must always be
reported alongside accuracy numbers, and changing them for "speed" without
re-running the fair-comparison protocol (`IMPLEMENTATION_PLAN.md` section 17)
invalidates any accuracy comparison across a resolution change.

## Training-time memory controls (also never allowed to change results)

- `use_checkpointing True`: re-runs the forward pass inside the backward
  pass for FNO/GINO's spectral blocks (`torch.utils.checkpoint.checkpoint`,
  `use_reentrant=False`) and Point-DeepONet's/DeepONet's trunk-heavy stages
  where wired. Exact by construction (recomputation, not approximation).
- `train_query_chunk_size` / `infer_query_chunk_size`: splits the query
  range fed to `decode_queries` during training/inference respectively; `0`
  disables chunking. Exactness is what `test_query_chunking.py` verifies.
- `grad_accum_steps`: accumulates gradients over multiple micro-batches
  before an optimizer step (`training_profiles/training_loop.py`); this
  changes optimizer *cadence*, not the per-sample forward computation.

## fp32 islands under bf16 autocast (section 12.4)

Every op below explicitly disables autocast and casts to float32 internally,
regardless of the ambient AMP setting, because bf16 either loses too much
precision or the underlying primitive (FFT) does not support it:

- `model/spectral.py::SpectralConvNd.forward` (`torch.fft.rfftn`/`irfftn`)
- `model/gno.py::GNOLayer.forward` (kernel-integral scatter/reduce)
- `model/deeponet.py::DeepONet._decode` and
  `model/point_deeponet.py::PointDeepONet._decode` (final modal dot
  products / einsum)
- `model/adapters/grid.py::splat` (weighted-mean accumulation; always fp32
  regardless of caller's autocast state, since it never enters an
  autocast-disabled block itself but only ever receives/returns fp32)

BatchNorm1d (PointNet) autocasts itself to fp32 internally by default and
needs no extra handling here.
