# GINO Parity Notes

Source: Li, Z. et al., *Geometry-Informed Neural Operator for Large-Scale 3D
PDEs*, NeurIPS 2023. See IMPLEMENTATION_PLAN.md sections 2.5 and 8.4 for the
research-conclusions summary this document expands.

## Published architecture

GINO maps an irregular input function to a regular latent grid via a graph
neural operator (kernel integral over a radius neighborhood), runs standard
FNO blocks in the latent space, then maps back to arbitrary output query
points via another graph neural operator. The published model additionally
uses a signed distance function (SDF) as part of its input representation
and motivates discretization convergence through the continuous kernel
integral / quadrature-weighted reductions.

## This repository's implementation (`model/gino.py`, `model/gno.py`)

- **Input GNO** (`GNOLayer`, `model/gno.py`): kernel integral over radius
  neighbors found via `model/adapters/radius_neighbors.py` (scipy `cKDTree`
  baseline; optional `torch_cluster.radius`, parity-tested against scipy on
  random fixtures in `tests/test_radius_neighbors.py`). The kernel MLP takes
  `[query_pos, source_pos, source_features]` and the reduction is an
  **unweighted mean** over incoming edges (`index_add_` + count division),
  not a quadrature-weighted sum -- see "Divergences" below.
- **Latent FNO**: the same native `SpectralConvNd` core used by
  `model/fno.py` (section 8.3), operating on the latent grid.
- **Output GNO**: a second `GNOLayer` mapping the latent grid back to
  arbitrary query points (the original mesh nodes at inference/training).
- **Per-graph execution**: official GINO batches input functions only when
  the geometry is shared across the batch. Every scene in the current MGN
  data has different geometry, so `MeshGINO.forward`/`encode_operator`/
  `decode_queries` loop over `ptr` and concatenate results in original node
  order (section 8.4/A.6). `gino_group_shared_geometry` (an optimization
  that would batch timesteps of one scene after a byte-equality check) is
  not implemented -- the per-graph loop is the only path.

## Deliberate divergences (`gino_variant mesh_state`, the only implemented variant)

| Paper | This repo | Why |
| --- | --- | --- |
| SDF as part of the input representation | Coordinates + physical state + context; SDF appended only when `sdf_source != none` | No current dataset provides SDF (section 2.2). |
| Quadrature-weighted kernel reduction, discretization-convergence claim | Unweighted mean reduction over radius neighbors | No dataset provides area/volume integration weights; `integration_weight_source` is always `none` today. |
| — | `gino_variant paper` | Defined in `validate_config` (requires `has_sdf` and `has_integration_weights`) but **not implementable** until a weighted-kernel reduction exists; currently always raises. This is intentional: the gate exists so a future dataset/feature addition is the only thing that needs to change, not silently permissive code. |

Every result from this repository is therefore an **"MGN-data GINO"** and
must never be reported as a reproduction of the paper's discretization-
convergence or SDF-ablation results.

## Coverage preflight (mandatory, section 8.4)

`MeshGINO.coverage_preflight(graph)` reports min/median/max neighbor counts
in both directions and the empty-input/unreachable-output fractions, and
raises before training if:

- the input GNO's empty-latent-cell fraction exceeds
  `gino_max_empty_input_fraction`, or
- any output query has zero neighbors in the latent grid (unreachable).

`model/adapters/radius_neighbors.py::min_reachable_radius` computes the
half-cell-diagonal of the latent grid -- the smallest `gino_out_radius` that
*provably* reaches every grid point from its nearest neighbors, regardless
of mesh density. `validate_config` warns (does not hard-fail, since the
guarantee is direction-specific) when `gino_out_radius` is below this floor.
`training_profiles/setup.py::build_model_and_ema` runs the preflight
automatically on one training batch before training starts.

**Measured on real `ex1.h5` (2026-07-17):** with `augment_geometry True` the
grid box is sized for the worst-case *rotated* footprint (section 4.5), so
any single unrotated sample only occupies the box's central ~35-45% region;
the remaining latent grid points have no nearby mesh node by construction,
not because of a coverage bug. At `gino_in_radius 0.3` (resolution 24x24)
the measured empty-latent fraction is ~0.16-0.20 — `ex1/config_*_gino.txt`
sets `gino_max_empty_input_fraction 0.3`, above the measured value rather
than raised to silently pass. This is an inherent tension between a shared
rotation-safe grid box (needed for augmentation parity across all four
models, section 17) and GINO's latent-grid density; it is not present when
`augment_geometry False` (ex2's policy), where the box tightly matches the
training extent.

## Reference tests

- `tests/test_gno.py`: `GNOLayer` against an independent fp64 Python-loop
  reduction, plus empty-neighbor-query zero-output behavior.
- `tests/test_gino.py`: forward shapes, two-graph isolation/order, exact
  query-chunk parity, and both a passing and a deliberately-too-small-radius
  coverage-preflight case.
- `tests/test_spectral.py`: the shared `SpectralConvNd` core (see
  `docs/POINT_DEEPONET_PARITY.md`'s sibling coverage for FNO).
