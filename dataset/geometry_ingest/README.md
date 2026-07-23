# geometry_ingest

Turn CAD/geometry files (**STEP / IGES / STL / PLY / OBJ**) into the repository's
shared mesh HDF5 contract — one artifact that feeds every mesh-consuming method:

- **MeshGraphNets / -variational** read `data/{id}/mesh_edge` (the graph).
- **Neural_Operator (DeepONet/FNO/GINO) / Transolver** read the same nodes as a
  **point cloud** (they ignore `mesh_edge`).

Because it emits the existing contract (`dataset/DATASET_FORMAT.md`), **nothing
downstream changes**. It is a registered launcher model (`model geometry_ingest`)
*and* a standalone CLI.

> Scope: this is an **inference / dataset-authoring** front end. A geometry has no
> solution fields, so field rows are zero-filled — a solver or a trained model
> fills them. It does not by itself create training pairs.

## Run through the suite launcher (recommended)

```bash
python AI_CAE4ALL_main.py --config configs/geometry_ingest/config_ingest_volume.txt --check     # validate
python AI_CAE4ALL_main.py --config configs/geometry_ingest/config_ingest_volume.txt --dry-run   # show native command
python AI_CAE4ALL_main.py --config configs/geometry_ingest/config_ingest_surface.txt            # run
python AI_CAE4ALL_main.py --describe geometry_ingest                                             # modes + required keys
```

`mode` lives inside the config: **`ingest`** (write HDF5) or **`inspect`** (mesh
each input, print stats, write nothing). Config paths resolve from the tool repo
root (`dataset/geometry_ingest/`), matching the suite's native-path convention.

### Config keys

| Key | Need | Meaning |
| --- | --- | --- |
| `model` | required | `geometry_ingest` |
| `mode` | required | `ingest` or `inspect` |
| `input_geometry` | required | Directory of geometry files (`.step .stp .igs .iges .stl .ply .obj`) |
| `output_dataset` | required for `ingest` | Output HDF5 (graph). Point cloud is written alongside as `<stem>_pointcloud.h5` |
| `reader` | opt (`auto`) | `auto` \| `trimesh` \| `gmsh` |
| `mesh_type` | opt (`volume`) | `volume` → gmsh tet mesh (dim 3); `surface` → triangle mesh (dim 2) |
| `emit` | opt (`graph`) | comma list of `graph`, `pointcloud` |
| `num_points` | opt (`0`) | point-cloud resample size (`0` = keep all nodes) |
| `resample_method` | opt (`fps`) | `fps` \| `random` |
| `num_fields` | opt (`3`) | zero-filled solution-field rows after xyz |
| `mesh_size_max`/`mesh_size_min` | opt (`0`) | gmsh target element size |
| `seed`, `limit` | opt | resample seed; cap input count |

> Comments must be on their own `%` line. An inline `% ...` after a value is **not**
> stripped by the suite parser and becomes part of the value.

Shipped templates in `configs/geometry_ingest/`: `config_ingest_volume.txt`
(CAD → volume, needs gmsh), `config_ingest_surface.txt` (STL/PLY → surface,
trimesh only), `config_inspect_surface.txt` (stats-only).

## Standalone CLI

```bash
# Volume tet mesh from STEP (needs gmsh), graph + resampled point cloud:
python -m geometry_ingest.cli part.step --volume --emit graph,pointcloud \
    --num-points 8192 --mesh-size-max 5.0 --output out/part.h5

# Surface mesh from a directory of STL/PLY (trimesh), stats only:
python -m geometry_ingest.cli ./stl_dir --reader trimesh --dry-run
```

Run from the `dataset/` directory (imported as `geometry_ingest`).

## Install

```bash
pip install gmsh meshio trimesh        # gmsh: STEP/IGES + volume tet meshing
```

- `gmsh` — **required** for STEP/IGES and volume (tet) meshing (OpenCASCADE-backed).
  Imported lazily, so surface (`trimesh`) runs need no gmsh.
- `trimesh` — surface STL/PLY/OBJ reading + repair (already in this repo's envs).
- `meshio` — optional, broader format IO/export.

### Airgapped install

`gmsh` ships one compiled wheel **per platform** (a ctypes wrapper around a
bundled `libgmsh`, so it is *not* Python-version specific). `meshio`/`trimesh`
are pure-Python (`py3-none-any`). `numpy`/`h5py` are already present in the
method venvs. Fetch the closure on a networked box:

```bash
pip download --only-binary=:all: \
    --platform manylinux_2_28_x86_64 \
    gmsh meshio trimesh -d wheels/
# copy wheels/ to the airgapped server, then:
pip install --no-index --find-links wheels/ gmsh trimesh
```

Prefetched Linux/manylinux wheels are **committed** under `wheels/` (all < 50 MB,
GitHub-safe), so an airgapped clone already carries them.

### Automatic offline install

When a required module (`gmsh`, `trimesh`) is not importable at call time, the
tool installs it **offline from the committed `wheels/`** (`pip install --no-index
--find-links wheels/ --no-deps`) and retries — see `deps.py::ensure`. So on the
airgapped Linux server the volume path just works after `git clone`, with no
separate install step. If no bundled wheel matches the platform/Python (e.g. the
Linux gmsh wheel on Windows), it raises a clear error with a manual-install hint
rather than a cryptic traceback.

## Two decisions this tool exposes

1. **Surface vs volume.** `mesh_type volume` (tets) is what CAE physics and the
   physics-loss plan want; `surface` (triangles) matches the current shipped
   datasets. `reader` supports both.
2. **Distributional consistency.** Node density/ordering from meshing must match a
   model's training meshes, or positional encodings (centroid distance, neighbor
   edge length, RWPE/LPE) drift and inference degrades. Match `mesh_size_*` to
   training stats, or retrain on ingest-produced meshes.

## Modules

| File | Role |
| --- | --- |
| `readers.py` | `read_trimesh` (surface) · `read_gmsh` (surface/volume) · `read_auto` |
| `clean.py` | watertight repair for dirty STL before volume meshing |
| `to_graph.py` | elements → unique undirected `mesh_edge [2,E]` (tri & tet) |
| `to_pointcloud.py` | deterministic FPS / random resample to fixed N |
| `writer.py` | the HDF5 contract writer (zero-filled field rows) |
| `pipeline.py` | shared read → graph/pointcloud → write flow |
| `deps.py` | lazy offline install of missing modules from committed `wheels/` |
| `config.py` | flat `key value` config parser for the launcher entrypoint |
| `main.py` | launcher entrypoint (`--config`) invoked by `AI_CAE4ALL_main.py` |
| `cli.py` | standalone argparse CLI (`python -m geometry_ingest.cli`) |
