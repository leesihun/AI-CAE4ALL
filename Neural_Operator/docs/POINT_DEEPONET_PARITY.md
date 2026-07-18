# Point-DeepONet Parity Notes

Source: Park, J. and Kang, N., *Point-DeepONet: Predicting Nonlinear Fields on
Non-Parametric Geometries under Variable Load Conditions*, Neural Networks 198,
108560 (2026). https://arxiv.org/abs/2412.18362 (verified against the full
HTML text, 2026-07-17). See IMPLEMENTATION_PLAN.md section 2.4 for the
research-conclusions summary this document expands.

## Published architecture (verified facts)

| Component | Published spec |
| --- | --- |
| Branch geometry path | PointNet: shared Conv1D -> BatchNorm -> ReLU blocks, global max pool, output width 128 |
| Branch condition path | force magnitude, direction vector, mass -> fully connected layers with SiLU, output width 128 |
| Branch merge | **elementwise sum**: `B = B_condition + B_geometry` (both width 128) |
| Trunk input | `(x, y, z, SDF)` |
| Trunk | SIREN (sine activations) -> `T in R^{B,N,128}` |
| Fusion | early elementwise multiplication `F[n,h] = B[h] * T[n,h]` |
| Refiners | subsequent MLP layers refine `F` into two feature sets: `B_beta in R^{B,128}` and `T_beta in R^{B,N,128,M}` (M=4 output channels) -- note the asymmetry, only the trunk-side refiner carries the per-output dimension |
| Final synthesis | `pred[n,m] = tanh( sum_h B_beta[h] * T_beta[n,h,m] )` |
| Output activation | tanh |
| Sampling | resample all nodal coordinates/outputs to a fixed N=5000 for training; full-mesh evaluation at inference |
| Training | MSE, AdamW, lr=1e-3, batch=16, 40k iterations, 251,936 parameters, DeepXDE+PyTorch |

## This repository's shape mapping (`model/point_deeponet.py`)

```text
branch_context   B          [num_graphs, H]              geometry_proj(pointnet(...)) [+ condition_proj(condition_mlp(gc))]
trunk_out        T          [chunk_N, H]                 Siren(query_features)
fused            F          [chunk_N, H]                 B[batch] * T
branch_refiner   B_beta     [chunk_N, H]                 MLP(F)
trunk_refiner    T_beta     [chunk_N, H, output_var]      MLP(F).view(-1, H, output_var)
prediction       pred[n,o]  [chunk_N, output_var]         sum_h B_beta[n,h] * T_beta[n,h,o] + bias[o]
```

This is a per-node einsum `'nh,nho->no'` of `B_beta` against `T_beta`,
matching the paper's `sum_h B_beta[h] * T_beta[n,h,m]` exactly (`h` is the
paper's 128-wide latent index, `m`/`o` is the output channel).

## Deliberate divergences (MGN-adapted profile, `point_variant mesh_state`)

| Paper | This repo (`mesh_state`, default) | Why |
| --- | --- | --- |
| Branch condition input: force/direction/mass | Branch geometry input additionally includes current physical state + node context (positional features, node-type one-hot) | No global load conditions exist in the current MGN HDF5 files (section 2.2); the model must still see the current physical state for temporal delta prediction. |
| Trunk input: `(x,y,z,SDF)` | Trunk input: active-axis `pos_normalized` + non-physical context (positional features, node-type one-hot) [+ SDF when available] | No dataset provides SDF; node-type/positional context replace it as the model's only non-coordinate geometric signal. |
| Output activation: tanh | Output activation: **identity** (default) | Normalized MGN deltas are not guaranteed to lie in `[-1, 1]`; `tanh` would clip valid targets. `point_output_activation tanh` remains available as a named ablation. |
| N=5000 fixed resample, geometry-only PointNet input | `point_sensor_count` (default 5000) resample of **[coordinates + state + context]**, `point_variant paper` restores geometry-only sampling | Matches the paper's fixed-size resampling rationale while carrying the state the temporal delta task needs. |

Results produced under `point_variant mesh_state` must be labeled
**"MGN-adapted Point-DeepONet"**, never a reproduction of the published
paper's numbers. `point_variant paper` enforces the published input contract
(validated SDF, declared global conditions, tanh output) and fails
construction when the active dataset does not satisfy it.

## Reference-shape test

`tests/test_point_deeponet.py::test_parity_shapes_match_paper_mapping`
constructs a `PointDeepONet` on a synthetic fixture and asserts:

- `branch_context` has shape `[num_graphs, point_feature_dim]`;
- `trunk_refiner` output reshapes to `[chunk_N, point_feature_dim, output_var]`
  (the paper's asymmetric refiner shapes, not `[N, output_var, basis_dim]`
  like normal DeepONet's symmetric modal product);
- the final prediction einsum contracts exactly the shared latent dimension
  `H`, not a separate `basis_dim`.

If a future correction to this document changes the fusion/refiner placement,
update this file and the reference-shape test together (IMPLEMENTATION_PLAN.md
section 8.1).
