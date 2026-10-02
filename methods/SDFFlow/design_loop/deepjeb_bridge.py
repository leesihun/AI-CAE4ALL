"""Turn generated bracket geometry into the contract the DeepJEB surrogate reads.

This is the join between the two AI stages of the design loop. SDFFlow emits a
marching-cubes surface in its own *normalized* frame; HI-MGN was trained on
DeepJEB surface graphs in **millimetres**. This module maps one to the other and
emits the same `data/{id}/{nodal_data, mesh_edge}` layout the training set uses,
with the state rows zeroed -- which is exactly what the model expects at
inference for a static problem.

Two things are deliberate:

**The coarsening is imported, not reimplemented.** `build_deepjeb_mgn.decimate`
does the geodesic vertex clustering for both the training set and this bridge,
so a generated bracket is discretized by the identical procedure the labels were
produced under. A second implementation here would be a train/serve skew waiting
to happen.

**The frame map is measured, not assumed.** DeepJEB brackets occupy nearly
the same envelope (centre 15.79, -71.58, 32.74 mm and longest side 184.18 mm,
measured across the fetched brackets with standard deviations of 0.24 and 0.71
mm), and `normalize_mesh` scales each one's longest side to 1.8.
`normalized_to_millimetres` inverts that with the population numbers. The ex10
labels were written through that same map, so the ex10 layout keeps it (train
and serve share one frame). The ex13 ('ver') labels were solved on the CAD
brackets in their true frames, and a generated shape's own length is not the
population's (ex1 samples come out scaled 0.97-1.01, shifted up to ~3 mm), so
that layout registers each shape on its bores first (`registered_millimetres`).

  python dataset/deepjeb_bridge.py --stl-dir output/.../samples --out infer.h5
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

from design_loop.build_deepjeb_mgn import (
    COND_VAR, FEATURE_NAMES, INPUT_VAR, LOAD_CASES, OUTPUT_VAR,
    decimate, edges_from_faces,
)
from design_loop.interfaces import (
    SERVE_TOL, InterfaceError, node_types, register_to_interfaces, vertex_normals,
)

# The ex13 vertical-load layout (D:/CAE_datasets_raw/deepjeb_ver_fea/src/build_ex13.py):
# one load case, no condition rows, and the boundary conditions carried as a
# trailing node-type row (0 free, 1 bolt bore wall, 2 lug ear bore wall).
VER_FEATURE_NAMES = ['x_coord', 'y_coord', 'z_coord', 'u_x', 'u_y', 'u_z', 'von_mises',
                     'node_type']
VER_INPUT_VAR = VER_OUTPUT_VAR = 4
VER_COND_VAR = 0

# Measured over the fetched DeepJEB brackets; see the module docstring.
DEEPJEB_CENTRE = np.array([15.789, -71.580, 32.743])
DEEPJEB_MAX_SIDE = 184.181
SDF_TARGET_EXTENT = 1.8          # normalize_mesh scales the longest side to this

# The resolution build_deepjeb_fea labels at: every bracket surface is
# decimated to LABEL_SURFACE_FACES by `mesher.prepare_surface`, handed to gmsh
# (which keeps the surface and fills the interior at LABEL_MESH_SIZE_MAX), and
# the solved boundary is clustered to the graph from there. A generated shape
# has to reach the graph through the same surface, or the network is served a
# discretization it never saw.
LABEL_SURFACE_FACES = 12000
LABEL_MESH_SIZE_MAX = 0.05


def normalized_to_millimetres(vertices, centre=DEEPJEB_CENTRE,
                              max_side=DEEPJEB_MAX_SIDE):
    """Invert `normalize_mesh`: normalized coords -> the DeepJEB physical frame."""
    return np.asarray(vertices, dtype=np.float64) * (max_side / SDF_TARGET_EXTENT) + centre


def serving_surface(mesh, surface_faces=LABEL_SURFACE_FACES):
    """The surface a generated shape is bridged from: the label pipeline's.

    The labels were never computed on a raw marching-cubes surface: each one is
    the boundary of a gmsh mesh built from `prepare_surface(mesh,
    LABEL_SURFACE_FACES)`. Clustering the raw MC surface (5-10x more faces, MC
    staircase intact) straight to `target_nodes` gave the surrogate a graph
    drawn from a different surface than any it was trained on. Falsy
    `surface_faces` passes the mesh through unchanged.
    """
    if not surface_faces:
        return mesh
    from design_loop.mesher import prepare_surface
    return prepare_surface(mesh, int(surface_faces))


def mesh_to_records(mesh, load_cases=LOAD_CASES, target_nodes=5000,
                    already_millimetres=False, name='generated'):
    """One record per load case for a single bracket surface.

    `mesh` is clustered as given; pass a generated shape through
    `serving_surface` first.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if not already_millimetres:
        vertices = normalized_to_millimetres(vertices)

    small = decimate({'vertices': vertices,
                      'faces': np.asarray(mesh.faces, dtype=np.int64)}, target_nodes)
    edges = edges_from_faces(small['faces'])
    num_nodes = small['vertices'].shape[0]

    records = []
    for case in load_cases:
        nodal = np.zeros((len(FEATURE_NAMES), 1, num_nodes), dtype=np.float32)
        nodal[0:3, 0, :] = small['vertices'].T
        # rows 3:3+INPUT_VAR stay zero: the static contract feeds a zeroed state
        # block and asks the model to produce it.
        nodal[3 + INPUT_VAR + LOAD_CASES.index(case), 0, :] = 1.0
        records.append({'item': name, 'case': case, 'nodal': nodal, 'edges': edges})
    return records


def registered_millimetres(mesh):
    """A generated surface's vertices in its own DeepJEB frame, and that frame.

    The population map (`normalized_to_millimetres`) is exact only for a
    bracket whose longest side is the population's 184.18 mm; `normalize_mesh`
    gave every training bracket the same normalized length whatever its true
    one. `interfaces.register_to_interfaces` recovers the per-shape scale and
    shift from the bores every DeepJEB bracket shares. Raises InterfaceError
    when they cannot be fitted: a shape without the canonical interface pattern
    is not one the ex13 labels describe.

    frame['mm_per_unit'] is the shape's millimetres per normalized SDF unit
    (DEEPJEB_MAX_SIDE / SDF_TARGET_EXTENT x frame['scale']).
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    population = normalized_to_millimetres(mesh.vertices)
    registered, frame = register_to_interfaces(population, vertex_normals(population, faces))
    if not frame['ok']:
        raise InterfaceError(f"frame registration failed: {frame['reason']}")
    frame['mm_per_unit'] = DEEPJEB_MAX_SIDE / SDF_TARGET_EXTENT * frame['scale']
    return registered, frame


def apply_frame(vertices, frame):
    """Normalized vertices -> the registered DeepJEB mm frame `registered_millimetres` found."""
    return (normalized_to_millimetres(vertices) * frame['scale']
            + np.asarray(frame['shift'], dtype=np.float64))


def mesh_to_ver_record(mesh, target_nodes=5000, already_millimetres=False,
                       name='generated', tol=SERVE_TOL):
    """One ex13-layout record: coordinates, a zeroed state block, node types.

    A generated (normalized) shape is first registered to its own frame
    (`registered_millimetres`); the ex13 labels were solved on the exact CAD
    brackets in theirs, so this is the frame the surrogate was trained in.
    The node types come from the geometric interface rule the labels were made
    with (`design_loop.interfaces`), applied to the full surface and carried
    through `decimate` -- as the labels' types were carried from the tet4
    boundary. The gates are judged there too: on the 5000-node graph a 6.5 mm
    lug ear wall can keep only one ring of nodes inside its band, which refused
    4 of 64 ex1 samples whose full surface has the whole bore (span >= 0.74,
    coverage >= 334 deg). A shape that fails raises InterfaceError, the same
    refusal the labelling run applied.

    The record carries `frame` (None for `already_millimetres` input).
    """
    if already_millimetres:
        vertices, frame = np.asarray(mesh.vertices, dtype=np.float64), None
    else:
        vertices, frame = registered_millimetres(mesh)
    small = decimate({'vertices': vertices,
                      'faces': np.asarray(mesh.faces, dtype=np.int64),
                      'constrained': node_types(vertices, tol)}, target_nodes)
    edges = edges_from_faces(small['faces'])
    num_nodes = small['vertices'].shape[0]
    nodal = np.zeros((len(VER_FEATURE_NAMES), 1, num_nodes), dtype=np.float32)
    nodal[0:3, 0, :] = small['vertices'].T
    nodal[-1, 0, :] = small['constrained']
    return {'item': name, 'case': 'ver', 'nodal': nodal, 'edges': edges, 'frame': frame}


def write_inference_contract(out_path, records, feature_names=FEATURE_NAMES,
                             input_var=INPUT_VAR, output_var=OUTPUT_VAR, cond_var=COND_VAR):
    """Write the shared mesh layout with no labels -- inference input only."""
    n_features = len(feature_names)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    tmp = out_path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        grp = f.create_group('data')
        for sample_id, rec in enumerate(records, start=1):
            nodal, edges = rec['nodal'], rec['edges']
            sg = grp.create_group(str(sample_id))
            sg.create_dataset('nodal_data', data=nodal, compression='gzip', compression_opts=4)
            sg.create_dataset('mesh_edge', data=edges, compression='gzip', compression_opts=4)
            md = sg.create_group('metadata')
            md.attrs['filename_id'] = f"{rec['item']}_{rec['case']}"
            md.attrs['bracket'] = rec['item']
            # The ex13 configs split on `split_group_attr parent`; a generated
            # shape is its own parent.
            md.attrs['parent'] = rec['item']
            md.attrs['load_case'] = rec['case']
            md.attrs['num_nodes'] = nodal.shape[2]
            md.attrs['num_edges'] = edges.shape[1]
            md.attrs['num_timesteps'] = 1
            md.create_dataset('feature_min', data=nodal.min(axis=(1, 2)))
            md.create_dataset('feature_max', data=nodal.max(axis=(1, 2)))
            md.create_dataset('feature_mean', data=nodal.mean(axis=(1, 2)))
            md.create_dataset('feature_std', data=nodal.std(axis=(1, 2)))

        f.attrs['num_samples'] = len(records)
        f.attrs['num_features'] = n_features
        f.attrs['num_timesteps'] = 1
        f.attrs['builder_input_var'] = input_var
        f.attrs['builder_output_var'] = output_var
        f.attrs['builder_cond_var'] = cond_var
        f.attrs['builder_source'] = 'SDFFlow generated geometry via deepjeb_bridge'

        top = f.create_group('metadata')
        top.create_dataset('feature_names', data=np.array(feature_names, dtype='S12'))
        splits = top.create_group('splits')
        ids = np.arange(1, len(records) + 1, dtype=np.int64)
        splits.create_dataset('train', data=np.array([], dtype=np.int64))
        splits.create_dataset('val', data=np.array([], dtype=np.int64))
        splits.create_dataset('test', data=ids)
    os.replace(tmp, out_path)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stl', nargs='*', default=[], help='individual STL files')
    parser.add_argument('--stl-dir', default=None, help='directory of STL files')
    parser.add_argument('--out', required=True, help='output HDF5 in the mesh contract')
    parser.add_argument('--target-nodes', type=int, default=5000)
    parser.add_argument('--surface-faces', type=int, default=LABEL_SURFACE_FACES,
                        help='decimate each surface to this many faces first, as the '
                             'labels were (0 bridges the STL as given)')
    parser.add_argument('--load-cases', default=','.join(LOAD_CASES))
    parser.add_argument('--millimetres', action='store_true',
                        help='input STLs are already in the DeepJEB physical frame')
    args = parser.parse_args(argv)

    import trimesh

    paths = list(args.stl)
    if args.stl_dir:
        paths += sorted(glob.glob(os.path.join(args.stl_dir, '*.stl')))
    if not paths:
        print('no input STLs given', file=sys.stderr)
        return 1

    cases = tuple(c.strip() for c in args.load_cases.split(','))
    unknown = set(cases) - set(LOAD_CASES)
    if unknown:
        print(f'unknown load case(s) {sorted(unknown)}; available {LOAD_CASES}',
              file=sys.stderr)
        return 1

    records = []
    for path in paths:
        mesh = trimesh.load(path, process=False)
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            recs = mesh_to_records(serving_surface(mesh, args.surface_faces),
                                   load_cases=cases,
                                   target_nodes=args.target_nodes,
                                   already_millimetres=args.millimetres, name=name)
        except Exception as exc:
            print(f'  {name}: SKIPPED {type(exc).__name__}: {exc}', flush=True)
            continue
        records += recs
        extent = np.ptp(recs[0]['nodal'][0:3, 0, :], axis=1)
        print(f'  {name}: {len(mesh.vertices)} -> {recs[0]["nodal"].shape[2]} nodes, '
              f'extent {np.round(extent, 1)} mm', flush=True)

    if not records:
        print('no records built', file=sys.stderr)
        return 1
    write_inference_contract(args.out, records)
    print(f'wrote {args.out}: {len(records)} samples '
          f'({len(records) // len(cases)} brackets x {len(cases)} load cases)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
