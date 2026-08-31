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

**The frame map is measured, not assumed.** Every DeepJEB bracket occupies the
same envelope (centre 15.79, -71.58, 32.74 mm and longest side 184.18 mm,
measured across the fetched brackets with standard deviations of 0.24 and 0.71
mm), and `normalize_mesh` scales that longest side to 1.8. Inverting it is
therefore exact to about 0.4%.

  python dataset/deepjeb_bridge.py --stl-dir output/.../samples --out infer.h5
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_deepjeb_mgn import (                                     # noqa: E402
    COND_VAR, FEATURE_NAMES, INPUT_VAR, LOAD_CASES, OUTPUT_VAR,
    decimate, edges_from_faces,
)

# Measured over the fetched DeepJEB brackets; see the module docstring.
DEEPJEB_CENTRE = np.array([15.789, -71.580, 32.743])
DEEPJEB_MAX_SIDE = 184.181
SDF_TARGET_EXTENT = 1.8          # normalize_mesh scales the longest side to this


def normalized_to_millimetres(vertices, centre=DEEPJEB_CENTRE,
                              max_side=DEEPJEB_MAX_SIDE):
    """Invert `normalize_mesh`: normalized coords -> the DeepJEB physical frame."""
    return np.asarray(vertices, dtype=np.float64) * (max_side / SDF_TARGET_EXTENT) + centre


def mesh_to_records(mesh, load_cases=LOAD_CASES, target_nodes=5000,
                    already_millimetres=False, name='generated'):
    """One record per load case for a single generated bracket."""
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


def write_inference_contract(out_path, records):
    """Write the shared mesh layout with no labels -- inference input only."""
    n_features = len(FEATURE_NAMES)
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
        f.attrs['builder_input_var'] = INPUT_VAR
        f.attrs['builder_output_var'] = OUTPUT_VAR
        f.attrs['builder_cond_var'] = COND_VAR
        f.attrs['builder_source'] = 'SDFFlow generated geometry via deepjeb_bridge'

        top = f.create_group('metadata')
        top.create_dataset('feature_names', data=np.array(FEATURE_NAMES, dtype='S12'))
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
            recs = mesh_to_records(mesh, load_cases=cases,
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
