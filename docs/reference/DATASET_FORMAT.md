# Dataset Format

The training and inference code expects HDF5 files with a `data/{sample_id}`
group per sample. The current loader is `general_modules/mesh_dataset.py`; the
dataset builders are `build_dataset.py`, `dataset/generate_inference_dataset.py`,
and `dataset/reduce_dataset.py`.

This same `data/{id}/{nodal_data, mesh_edge}` layout can also be **authored from
CAD/geometry** (STEP/IGES/STL/PLY/OBJ) by the `geometry_ingest` tool
(`methods/GeometryIngest/`, `model geometry_ingest`): it meshes a part (surface
triangles or volume tets) and writes this contract with solution-field rows
zero-filled — i.e. an inference initial condition, not a training pair. See
[geometry_ingest/README.md](../../methods/GeometryIngest/README.md).

**SimulGenVAE** (`model simulgenvae`) also reads this `data/{id}/nodal_data`
layout, but as a **fixed-geometry dense FOM**: it selects the physical field rows
`[field_start_row : field_start_row + num_var]` (default `field_start_row 3`) and
flattens them into a dense `[num_samples, num_var*num_nodes, num_timesteps]` tensor
for its hierarchical VAE. Because that tensor is dense, **every sample must share
the same node count and timestep count** (mismatches are a hard error). `mesh_edge`
is required by the shared contract but ignored by SimulGenVAE. See
[../SimulGenVAE/CLAUDE.md](../../methods/SimulGenVAE/CLAUDE.md).

> The **MLP surrogate** (`model mlp`) is the exception: it is a tabular
> parameters→outputs regressor with no mesh, so it uses a separate **tabular
> `X`/`Y` HDF5** contract, documented in *Tabular Parametric Dataset* at the end
> of this file — not the `data/{id}` layout below.

## File Layout

```text
dataset.h5
  attrs:
    num_samples
    num_features
    num_timesteps
  data/
    1/
      nodal_data
      mesh_edge
      metadata/
        attrs:
          source_filename
          filename_id
          num_nodes
          num_edges
          num_cells
          num_corner_nodes
          num_total_nodes
        feature_min
        feature_max
        feature_mean
        feature_std
    2/
      ...
  metadata/
    feature_names
    normalization_params/
      min
      max
      mean
      std
      optional train-derived arrays written by the loader:
        node_mean
        node_std
        edge_mean
        edge_std
        delta_mean
        delta_std
    splits/
      train
      val
      test
```

The standard builders initialize `metadata/splits/*`, and the elasticity
converter also writes intended benchmark partitions at top-level
`splits/{train,val,test}`. The current training loaders do not consume either
form. Training calls `dataset.split(0.8, 0.1, 0.1, seed=split_seed)` and creates
a deterministic seeded split from sorted sample IDs. The benchmark configs'
`split_strategy hdf5` field is not implemented in the stable runtime snapshot.

## `nodal_data`

Path:

```text
data/{sample_id}/nodal_data
```

Shape:

```text
[num_features, num_timesteps, num_nodes]
```

The standard builder writes 8 features:

| Index | Meaning |
| --- | --- |
| `0` | x coordinate |
| `1` | y coordinate |
| `2` | z coordinate |
| `3` | x displacement |
| `4` | y displacement |
| `5` | z displacement |
| `6` | stress or other scalar field when present |
| `7` | part number |

Current `_b8_all_warpage_input` configs use `input_var 3` and `output_var 3`,
so they train on displacement channels only. The part-number channel is still
kept for visualization and optional node-type features.

For single-timestep datasets, the loader sets the physical input channels to
zeros and uses `nodal_data[3:3 + output_var]` as the target state. For
multi-timestep datasets, the loader uses:

```text
x_phys = nodal_data[3:3 + input_var, t]
target_delta = nodal_data[3:3 + output_var, t + 1] - x_phys
```

`pos` is always the reference coordinate slice `nodal_data[0:3, t]`.

### Input-only conditioning rows (`cond_var`)

Rows after the state block can carry **known conditions** — boundary conditions,
flight parameters, material constants — that the model should *read* but never
*predict*:

```text
rows [0:3]                          reference coordinates
rows [3 : 3+input_var]              state:      input AND output
rows [3+input_var : ... +cond_var]  conditions: input ONLY
```

Set `cond_var N` in the config. `cond_var 0` is the default and reproduces the
pre-conditioning behavior exactly. The resulting node feature vector is:

```text
graph.x = [ state (input_var) | conditions (cond_var) | positional | node-type one-hot ]
```

Two properties make this different from simply widening `input_var`:

- **Conditions are read from disk in the static (T=1) case too.** The state
  block is zeroed there — the model must regress the field from nothing — but
  conditions are *known* at inference time, so zeroing them would make them
  invisible. They also get real normalization statistics, instead of the
  degenerate zero-variance ones the static state head gets.
- **Conditions are never advanced by a rollout.** They sit past `input_var`, so
  the autoregressive loop treats them as static node features and carries them
  forward untouched; only the first `output_var` channels are integrated.

`input_var == output_var` is still required for temporal (T>1) data — that
constraint is about the autoregressive feedback loop, which conditioning rows
are outside of. Supported by `meshgraphnets`, `meshgraphnets-v`, `transolver`,
and the four `Neural_Operator` models.

Two caveats:

- Mesh methods accept either per-sample constants broadcast across nodes or
  spatially varying known fields. SimulGenVAE's `lc_data_type hdf5` latent
  conditioner is narrower: every row it consumes must be a per-sample constant,
  and its reader rejects spatially varying rows.
- Rows `3:6` are the displacement vector by this contract, and geometric
  augmentation rotates them. If a conditioning row is a **vector component**
  (e.g. a freestream direction), keep `augment_geometry False`; scalar
  conditions are rotation-invariant and unaffected.

**SimulGenVAE** reads the same rows, but as one parameter vector per sample
rather than per node: set `lc_data_type hdf5` with `cond_var N` and it takes
rows `[field_start_row + num_var : ... + cond_var]` straight out of
`dataset_dir`, so no separate `param_dir` CSV is needed.

**MLP** needs none of this — its tabular `X`/`Y` contract already keeps inputs
and outputs in separate datasets, so conditions are just extra `X` columns.

### ex2 transient layout

`dataset/ex2.h5` contains 50 samples with 50 stored timesteps and uses:

```text
rows 0:3  x, y, z                                  reference coordinates
rows 3:7  x_disp, y_disp, z_disp, stress           evolving state
row  7    Part No.                                  node-type label
```

The mesh/operator configs use `input_var 4`, `output_var 4`, `cond_var 0`,
`positional_features 4`, and `use_node_types True`. The loader therefore builds
node features as `[state(4) | positional(4) | part one-hot(4)]`; coordinates and
edge geometry are also supplied through the graph/model-specific geometry path.
`Part No.` is a categorical node type, not a conditioning variable.

For AR-OT training, each item uses a ground-truth adjacent pair and learns
`delta_t = state_(t+1) - state_t`. For AR-RT training, the model is unrolled for
all 49 transitions and its predicted state is fed into the next step. Inference
is additive autoregression for either training scheme:

```text
delta_hat_t = network(state_hat_t, static geometry)
state_hat_(t+1) = state_hat_t + delta_hat_t
```

### ex3 full-resolution reordered layout

`dataset/ex3_train_reordered.h5` and `dataset/ex3_test_reordered.h5` use the
transient-safe ordering below. They are reproducibly derived from the original
full-resolution files by `dataset/reorder_ex3_features.py`; the originals are
not modified.

```text
rows  0:3   x, y, z                                      reference coordinates
rows  3:7   Cp, Cf_x, Cf_y, Cf_z                         predicted state
rows  7:13  Mach, AoA, aileron in/out, elevator, HTP     global conditions
rows 13:17  normal_x, normal_y, normal_z, surface_area   spatial conditions
```

Mesh-field methods configure this as `input_var 4`, `output_var 4`, and
`cond_var 10`. Their actual physical node input therefore has 14 channels: the
four zeroed static-state placeholders plus all ten known conditions. Keeping
the conditions outside `input_var` is what makes the same contract safe for
transient AR-OT and AR-RT: only the first four channels are advanced.

SimulGenVAE uses `field_start_row 3`, `num_var 4`, and `cond_var 6`. Its latent
conditioner consumes the six constant global rows and deliberately ignores the
four trailing spatial descriptors. The MLP table uses those same six globals
as `X` and surface-area-weighted Cp/Cf summaries as `Y`.

The current ex3 files have one timestep, so the mesh/operator path is static
regression rather than a rollout. It zeroes the unknown four-channel state and
predicts the absolute field directly:

```text
field_hat = network(zeros(4), conditions(10), static geometry)
```

The full-resolution HI-MGN config matrix contains four AR-OT variants: as-is,
P1 attention transfer, P2 multi-partition, and P12 combined. There is no ex3
AR-RT base config because AR-RT requires more than one stored timestep; selecting
it for the current `T=1` files is a validation error.

If a future ex3 dataset contains multiple timesteps while the ten conditions
remain constant over each trajectory, the existing transient path becomes:

```text
delta_hat_t = network(state_hat_t(4), conditions(10), static geometry)
state_hat_(t+1) = state_hat_t + delta_hat_t
```

The conditions are intentionally reused unchanged at every rollout step. A
dataset with time-varying flight/control conditions would require an explicit
condition sequence; the current rollout contract does not advance or refresh
`cond_var` rows.

## `mesh_edge`

Path:

```text
data/{sample_id}/mesh_edge
```

Shape:

```text
[2, E]
```

`build_dataset.py` extracts unique undirected edges from triangular elements and
writes them once. `MeshGraphDataset` converts loaded mesh edges to the
bidirectional graph representation used by PyG.

Runtime edge attributes are not stored in the dataset. They are recomputed from
reference and deformed positions as 8-D features:

```text
deformed_dx, deformed_dy, deformed_dz, deformed_dist,
ref_dx, ref_dy, ref_dz, ref_dist
```

The same feature function is used for mesh edges, world edges, and coarse
multiscale edges.

## Metadata

Per-sample metadata contains attributes for source tracking and graph size plus
per-feature summary arrays. These arrays are useful for inspection but are not
the training normalizers used by the model.

Global metadata contains:

- `feature_names`
- builder-level `normalization_params/min`, `max`, `mean`, and `std`
- optional split datasets

The loader fits train-split z-score normalizers for nodes, edges, and deltas.
Those are the values saved into checkpoints under `checkpoint['normalization']`.
When `write_preprocessing_to_hdf5()` is called, it writes train-derived arrays
under `metadata/normalization_params`:

```text
node_mean, node_std, edge_mean, edge_std, delta_mean, delta_std
```

The training setup writes preprocessing to HDF5 before training starts, so the
file may contain both builder-level summary stats and train-derived model
normalizers.

## Positional Features

`positional_features` appends rotation-invariant node features to the physical
input channels. The feature order is:

1. distance from the graph centroid
2. mean neighbor edge length
3. remaining features from `positional_encoding`

Supported encodings:

| Encoding | Meaning |
| --- | --- |
| `rwpe` | random-walk return probabilities at powers 2, 4, 8, 16, 32 |
| `lpe` | normalized Laplacian eigenvectors |
| `rwpe+lpe` | split the remaining slots between RWPE and LPE |

The model input size is:

```text
input_var + positional_features + optional num_node_types
```

## Node Types

If `use_node_types True`, the loader reads feature index 7 as a raw node type or
part id, maps observed values to contiguous indices, and appends one-hot node
type vectors to `x`.

The node-type mapping and count are saved in checkpoint normalization:

```text
node_type_to_idx
num_node_types
```

## World Edges

World edges are optional and are not stored in the source dataset. When
`use_world_edges True`, the loader computes radius edges from deformed positions
at each sample access and attaches:

```text
world_edge_index
world_edge_attr
```

World-edge attributes use the same 8-D layout and normalization as mesh-edge
attributes. The computed radius is stored in checkpoint normalization as
`world_edge_radius`.

## Multiscale Data

Multiscale hierarchy tensors are not stored permanently in the default dataset
builder. When `use_multiscale True`, the loader computes and caches them per
worker, then attaches per-level tensors to `MultiscaleData`:

```text
fine_to_coarse_i
coarse_edge_index_i
coarse_edge_attr_i
num_coarse_i
coarse_centroid_i
optional unpool_edge_index_i
optional coarse_seed_idx_i   # only when level i uses voronoi_inherit mode
```

Under `voronoi_inherit` mode the `coarse_centroid_i` tensor stores the FPS
seed's position (a real fine-mesh node) instead of the arithmetic centroid;
the attribute name is retained for compatibility with reader code.

Coarse edge normalizers are saved into checkpoints as:

```text
coarse_edge_means
coarse_edge_stds
```

## Inference Dataset

`dataset/generate_inference_dataset.py` copies selected samples from a source
dataset and keeps only the first timestep:

```text
nodal_data[:, 0:1, :]
mesh_edge
metadata attrs when present
```

Rollout uses this file for initial conditions, then writes one deterministic
output HDF5 per sample.

## Rollout Output Format

`inference_profiles/rollout.py` writes:

```text
outputs/rollout/rollout_sample{sample_id}_steps{steps}.h5
```

Each output file has one sample and this `nodal_data` layout:

```text
x, y, z, predicted output channels..., Part No.
```

Root attributes:

```text
num_samples = 1
num_features = 3 + output_var + 1
num_timesteps = rollout_steps + 1
```

Per-sample metadata stores sample id, node and edge counts, rollout time, model
path, and config file. Global metadata stores feature names and the normalization
arrays used for inference.

## Builder Notes

`build_dataset.py` assumes the local ANSYS/CSV source layout configured at the
top of that file. It writes:

- a last-step static dataset through `build_dataset_last_timestep()`
- a full temporal dataset through `build_dataset()`

`dataset/reduce_dataset.py` can copy a subset of samples, optionally create new
split metadata, and preserve global/per-sample metadata.

## Validation Checklist

Before training, check:

- each sample has `nodal_data` and `mesh_edge`
- `nodal_data` has at least `3 + max(input_var, output_var) + cond_var` feature rows
- feature index 7 exists if `use_node_types True`
- `mesh_edge` uses valid node indices for each sample
- `edge_var` is `8`
- `num_timesteps` is consistent with static vs temporal training intent

## Tabular Parametric Dataset (MLP surrogate)

The `mlp` method (`MLP/`, `model mlp`) is a parametric surrogate — **N scalar
inputs → M scalar outputs**, no mesh. It uses its own tabular HDF5, unrelated to
the `data/{id}` mesh layout above:

```text
table.h5
  X             float [num_samples, N]   input parameters   (must equal input_var)
  Y             float [num_samples, M]   output quantities  (must equal output_var)
  input_names   str   [N]                optional column labels
  output_names  str   [M]                optional column labels
```

- `X` and `Y` are top-level datasets (rank 2) sharing the sample axis. `Y` is
  **optional for inference** — when present, `infer.py` reports per-output
  MAE/RMSE. There are no edges, timesteps, node types, or positional features.
- The launcher validates this with `dataset_kind="table_hdf5"`
  (`cae_suite/dataset_probe.py`) and cross-checks `X`/`Y` widths against
  `input_var`/`output_var` before launch (`DATASET-FEATURES-001/002`).
- **Normalization is not stored in the dataset.** Input/output scaling is fit on
  the train split and saved into the checkpoint, so the source HDF5 stays
  read-only (mirroring the operators' `write_preprocessing False` stance).
- A tiny sample generator is at `dataset/mlp/make_sample.py` (writes `train.h5`
  and `infer.h5`; the shipped `configs/MLP/ex1` templates point at them).

**Before training an MLP, check:** `X` has exactly `input_var` columns, `Y` has
exactly `output_var` columns, and both share the same number of rows.
