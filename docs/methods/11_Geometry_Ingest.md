# Geometry Ingest (`geometry_ingest`)

> **Not an ML method — a data-prep utility.** It trains nothing and has no
> checkpoint. It is documented here because it is a launcher-routed `model`: the
> front door that turns CAD/geometry into the shared data contract every other
> method in this suite already consumes.

- **`model` value:** `geometry_ingest`
- **Repo / entrypoint:** `dataset/geometry_ingest/` · `main.py`
- **Modes:** `ingest` (write HDF5) · `inspect` (mesh + stats, write nothing)
- **Family:** geometry preprocessing / dataset authoring

## What it does

Reads **STEP / IGES / STL / PLY / OBJ**, produces a single welded node set with
connectivity, and writes the shared mesh HDF5 contract
([dataset/DATASET_FORMAT.md](../../dataset/DATASET_FORMAT.md)):

```
data/{id}/nodal_data   [3 + num_fields, 1, N]   rows 0:3 = physical coords, rest zero-filled
data/{id}/mesh_edge    [2, E]                    unique undirected edges (graph emit)
```

One artifact serves every mesh-consuming method, because in this suite the
point-based models already treat mesh nodes as a point cloud:

- **MeshGraphNets / -variational** use `mesh_edge` (the graph).
- **DeepONet / FNO / GINO / Transolver** ignore `mesh_edge` and use the nodes as a
  **point cloud** (optionally resampled to a fixed count).

## The two representations, one pipeline

| Emit | Output | Consumer |
| --- | --- | --- |
| `graph` | `output_dataset` — nodes + `mesh_edge` | GNN methods (operators also read the nodes) |
| `pointcloud` | `<stem>_pointcloud.h5` — nodes only, resampled to `num_points` | operator / transformer methods wanting a fixed point count |

## Surface vs volume (the design fork)

- `mesh_type surface` → triangle mesh (dim 2), via **trimesh** (STL/PLY/OBJ). Runs
  with no extra dependency.
- `mesh_type volume` → **gmsh** tetrahedral mesh (dim 3), via OpenCASCADE. Needed
  for CAD (STEP/IGES) and for volumetric CAE physics. `reader trimesh` +
  `mesh_type volume` is rejected — trimesh cannot tet-mesh.

Volume tets are what real CAE physics (warpage, stress) and the physics-loss plan
want; surface triangles match the datasets shipped today.

## When to use it

| If you need to… | This tool does… |
| --- | --- |
| Run a trained model on a brand-new CAD part | Mesh it into the contract as an inference initial condition |
| Build a dataset skeleton from a directory of geometries | Batch-mesh to one HDF5 (then attach solver fields) |
| Feed both GNN and operator models from one geometry | `emit graph,pointcloud` |
| Inspect mesh sizes/quality before committing | `mode inspect` |

> **It does not create training pairs.** Geometry carries no solution, so field
> rows are zero-filled; a solver or a trained model supplies the fields.

## Config keys

See [CONFIGURATION_REFERENCE.md §9.9](../../CONFIGURATION_REFERENCE.md) for the
full catalog. Minimal volume example:

```
model geometry_ingest
mode ingest
input_geometry incoming_geometry
output_dataset outputs/ingested/parts.h5
reader auto
mesh_type volume
emit graph, pointcloud
num_points 8192
mesh_size_max 5.0
```

## Dependencies

`numpy`, `h5py`, `trimesh` always; **`gmsh`** additionally for volume/CAD (imported
lazily, so surface runs need none). A `volume`/`gmsh` config raises the
`GEOM-GMSH-001` preflight notice. Airgapped install wheels are staged under
`dataset/geometry_ingest/wheels/`.

## Caveats

1. **Inference/authoring only** — no training pairs (see above).
2. **Distributional consistency** — meshing density/ordering must match a model's
   training meshes or its positional encodings drift; tune `mesh_size_*` to match,
   or retrain on ingest-produced meshes.
3. **Element quality** — sliver tets from dirty CAD poison downstream physics; the
   `clean.py` repair path (watertight fix) runs before volume meshing.

See [dataset/geometry_ingest/README.md](../../dataset/geometry_ingest/README.md)
for the standalone CLI and install details.
