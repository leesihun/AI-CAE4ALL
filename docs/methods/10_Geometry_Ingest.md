# 10 — Geometry Ingest (CAD → mesh HDF5)

## Summary

GeometryIngest is **data preparation, not an ML method**. It trains nothing and
writes no checkpoint. It turns CAD and mesh files into the shared mesh HDF5 that
the mesh methods read. Use it to:

- mesh a new part so a trained model can run on it
- build the geometry skeleton of a dataset before a solver fills in the fields
- inspect mesh sizes before committing to a dataset

| | |
| --- | --- |
| Route id | `geometry_ingest` |
| Directory | [`methods/GeometryIngest/`](../../methods/GeometryIngest/) |
| Entrypoint | [`main.py`](../../methods/GeometryIngest/main.py) |
| Spec | [`ai_cae4all/specs/geometry_ingest.py`](../../ai_cae4all/specs/geometry_ingest.py) |
| Own docs | [`methods/GeometryIngest/README.md`](../../methods/GeometryIngest/README.md) |

## Architecture

[`pipeline.py`](../../methods/GeometryIngest/pipeline.py) runs these steps for
each input file:

1. **Gather** (`gather_paths`). `input_geometry` is a file or a directory.
   Directories are searched for `stl, ply, obj, off, step, stp, igs, iges, brep`,
   and `limit` caps the count.
2. **Read** ([`readers.py`](../../methods/GeometryIngest/readers.py)).
   `read_auto` sends CAD extensions, and any volume request, to `read_gmsh`. It
   sends everything else to `read_trimesh`, which welds duplicate vertices.
   - gmsh runs single-threaded, so meshing is deterministic.
   - It produces tet4 for `mesh_type volume` and tri3 for `surface`, sized by
     `mesh_size_max` / `mesh_size_min`.
3. **Emit** ([`to_graph.py`](../../methods/GeometryIngest/to_graph.py),
   [`to_pointcloud.py`](../../methods/GeometryIngest/to_pointcloud.py)).
   - `graph` gives unique undirected edges.
   - `pointcloud` gives nodes resampled to `num_points` by `resample_method`.
4. **Write** ([`writer.py`](../../methods/GeometryIngest/writer.py)) and, if
   `preview` is on, render
   ([`preview.py`](../../methods/GeometryIngest/preview.py)).

A file that fails to read or mesh is skipped with a `[skip]` line, and the run
continues. [`deps.py`](../../methods/GeometryIngest/deps.py) installs missing
modules offline from `wheels/`. gmsh is imported only for volume or CAD input.

## Data contract

The output is the mesh HDF5 of
[DATASET_FORMAT.md](../reference/DATASET_FORMAT.md):

```text
data/{i}/nodal_data   [3 + num_fields, 1, N]   rows 0:3 = coordinates, field rows zero-filled
data/{i}/mesh_edge    [2, E]                   graph emit only
metadata/feature_names; root attrs num_samples, num_features, num_timesteps
```

Sample ids start at 1. With `emit pointcloud`, a second file
`<stem>_pointcloud.h5` holds the resampled nodes without edges.

**There are no training pairs.** Geometry carries no solution, so the field rows
are zeros until a solver or a trained model fills them.

## Modes and keys

| Mode | Required | Effect |
| --- | --- | --- |
| `ingest` | `input_geometry`, `output_dataset` | read, mesh, write HDF5 |
| `inspect` | `input_geometry` | read, mesh, print stats, write only the preview |

| Key | Default | Notes |
| --- | --- | --- |
| `reader` | `auto` | `auto`, `trimesh` or `gmsh`; `trimesh` with `mesh_type volume` is rejected |
| `mesh_type` | `volume` | `volume` (gmsh tet) or `surface` |
| `emit` | `graph` | `graph`, `pointcloud`, or both |
| `num_fields` | 3 | zero-filled field rows after the coordinates |
| `num_points` | 0 | point-cloud size (0 keeps every node) |
| `resample_method` | `fps` | `fps` or `random` |
| `seed` | 42 | resampling seed |
| `mesh_size_max`, `mesh_size_min` | — | gmsh element size |
| `limit`, `preview`, `preview_path`, `preview_max_samples`, `plot_dpi` | — | batch cap and preview |

Preflight flags volume or gmsh use with `GEOM-GMSH-001` and point-cloud output
with `GEOM-PC-001` (notices). `dataset_kind` is `None` and `native_probe` is
false, so the spec validator is the only gate.

Checked-in configs:

- [`config_ingest_volume.txt`](../../configs/GeometryIngest/config_ingest_volume.txt)
  (tet, `num_points 8192`)
- [`config_ingest_surface.txt`](../../configs/GeometryIngest/config_ingest_surface.txt)
  (tri, `limit 3`, `num_points 4096`)

Both read `input_geometry incoming_geometry`, which you must supply.

## Outputs

`output_dataset` sets where everything lands:

- the HDF5 itself: `../../output/geometry_ingest/ingested/parts.h5` or
  `parts_surface.h5` in the checked-in configs
- `<stem>_pointcloud.h5`, with `emit pointcloud`
- `<stem>_preview.png`

`inspect` writes only `geometry_ingest_preview.png`, beside the first input.

## Tests and validation

There is no `tests/` directory. Validate with:

```bash
python AI_CAE4ALL_main.py --config configs/GeometryIngest/config_ingest_volume.txt --check
```

Then run `mode inspect` on the target geometry before ingesting.

## Known limitations

- **Geometry only.** The field rows are zeros.
- **Meshing must match the training meshes.** Node density and ordering should
  match the meshes a model was trained on, or positional features drift. Tune
  `mesh_size_*`, or retrain on meshes from this tool.
- **No repair step.** `clean.py::repair_surface` exists but is not called by the
  pipeline. Sliver elements from dirty CAD pass straight through.
- **Static only.** The tool writes T = 1 and no node-type or part-label row.
