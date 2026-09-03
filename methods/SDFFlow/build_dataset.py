"""
Build an SDFFlow HDF5 dataset from a directory of meshes or from the synthetic
analytic shape family.

Usage:
    python build_dataset.py --output dataset/synthetic64.h5 --synthetic 64
    python build_dataset.py --output dataset/brackets.h5 --mesh_dir ./meshes

HDF5 layout (one group per shape):
    shapes/00000/surface_points   (S, 3) float32   normalized coords in [-0.9, 0.9]
    shapes/00000/surface_normals  (S, 3) float32
    shapes/00000/sdf_points       (Q, 3) float32   near-surface + uniform queries
    shapes/00000/sdf_values       (Q,)   float32   negative inside, positive outside
    shapes/00000/cond             (C,)   float32   geometric descriptors
    root attrs: num_shapes, cond_names, num_near, num_uniform
"""

import os

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse
import glob
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from tqdm import tqdm

from general_modules.sdf_sampling import (
    COND_NAMES,
    mesh_descriptors,
    normalize_mesh,
    sample_mesh_sdf,
    synthetic_sample,
)

MESH_EXTENSIONS = ('*.stl', '*.obj', '*.ply', '*.off')


def main():
    parser = argparse.ArgumentParser(description='SDFFlow dataset builder')
    parser.add_argument('--output', type=str, required=True, help='Output HDF5 path')
    parser.add_argument('--mesh_dir', type=str, default=None, help='Directory of input meshes')
    parser.add_argument('--synthetic', type=int, default=0, help='Number of synthetic shapes')
    parser.add_argument('--num_surface', type=int, default=16384)
    parser.add_argument('--num_near', type=int, default=65536)
    parser.add_argument('--num_uniform', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repair', action='store_true',
                        help='Conservatively repair small holes before watertightness validation')
    parser.add_argument('--max_faces', type=int, default=0,
                        help='Simplify real meshes above this face count (0 disables)')
    parser.add_argument('--sharp_edge_fraction', type=float, default=0.0,
                        help='Fraction of surface points routed onto sharp edges '
                             '(Dora-style Sharp Edge Sampling; 0 disables)')
    parser.add_argument('--sharp_edge_angle', type=float, default=0.5236,
                        help='Dihedral angle (radians) above which an edge is '
                             'treated as sharp (default 30 deg)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel real-mesh workers (0 or 1 runs sequentially)')
    parser.add_argument('--append_missing', action='store_true',
                        help='Append sources not already recorded in an existing HDF5 file')
    args = parser.parse_args()

    if (args.mesh_dir is None) == (args.synthetic == 0):
        raise SystemExit('Specify exactly one of --mesh_dir or --synthetic N')

    rng = np.random.default_rng(args.seed)
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if args.append_missing and args.synthetic > 0:
        raise SystemExit('--append_missing is supported only with --mesh_dir')

    h5_mode = 'a' if args.append_missing and os.path.exists(args.output) else 'w'
    with h5py.File(args.output, h5_mode) as h5:
        shapes_grp = h5.require_group('shapes')
        written = len(shapes_grp)

        if args.synthetic > 0:
            for i in tqdm(range(args.synthetic), desc='Synthetic shapes'):
                sample, cond = synthetic_sample(
                    rng, args.num_surface, args.num_near, args.num_uniform)
                _write_shape(shapes_grp, written, sample, cond, source=f'synthetic_{i}')
                written += 1
        else:
            paths = sorted(p for ext in MESH_EXTENSIONS
                           for p in glob.glob(os.path.join(args.mesh_dir, '**', ext), recursive=True))
            if not paths:
                raise SystemExit(f'No meshes found under {args.mesh_dir}')
            print(f'Found {len(paths)} meshes')
            if args.append_missing:
                existing_sources = {
                    os.path.normcase(str(grp.attrs.get('source', '')))
                    for grp in shapes_grp.values()
                }
                paths = [
                    path for path in paths
                    if os.path.normcase(path) not in existing_sources
                ]
                print(f'Processing {len(paths)} sources missing from the existing dataset')
            tasks = [
                (path, args.num_surface, args.num_near, args.num_uniform,
                 args.seed + i, args.repair, args.max_faces,
                 args.sharp_edge_fraction, args.sharp_edge_angle)
                for i, path in enumerate(paths)
            ]
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    results = executor.map(_process_mesh, tasks, chunksize=1)
                    for result in tqdm(results, total=len(tasks), desc='Meshes'):
                        written += _consume_mesh_result(shapes_grp, written, result)
            else:
                for task in tqdm(tasks, desc='Meshes'):
                    written += _consume_mesh_result(
                        shapes_grp, written, _process_mesh(task))

        h5.attrs['num_shapes'] = written
        h5.attrs['cond_names'] = COND_NAMES
        h5.attrs['num_near'] = args.num_near
        h5.attrs['num_uniform'] = args.num_uniform

    print(f'\nWrote {written} shapes to {args.output}')
    if written == 0:
        raise SystemExit('Dataset is empty.')


def _process_mesh(task):
    """Load, optionally repair/simplify, normalize, and sample one real mesh.

    This function stays module-level so it is picklable by Windows spawn workers.
    HDF5 writes remain in the parent process.
    """
    import trimesh

    (path, num_surface, num_near, num_uniform, seed, repair, max_faces,
     sharp_edge_fraction, sharp_edge_angle) = task
    try:
        mesh = trimesh.load(path, force='mesh')
        if repair and mesh.is_empty:
            mesh = _load_partial_ascii_stl(path, trimesh)
        original_faces = len(mesh.faces)

        if repair and not mesh.is_watertight:
            mesh.remove_unreferenced_vertices()
            mesh.merge_vertices()
            mesh.update_faces(mesh.unique_faces())
            mesh.update_faces(mesh.nondegenerate_faces())
            trimesh.repair.fix_normals(mesh, multibody=True)
            trimesh.repair.fill_holes(mesh)

        if repair and not mesh.is_watertight:
            try:
                import pymeshfix

                meshfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
                meshfix.repair(joincomp=True, remove_smallest_components=False)
                mesh = trimesh.Trimesh(
                    vertices=meshfix.points, faces=meshfix.faces, process=True)
                trimesh.repair.fix_normals(mesh, multibody=True)
            except (ImportError, RuntimeError, ValueError):
                pass

        if not mesh.is_watertight:
            return {'path': path, 'error': 'not watertight after preprocessing'}

        if max_faces > 0 and len(mesh.faces) > max_faces:
            simplified = mesh.simplify_quadric_decimation(face_count=max_faces)
            # Some otherwise valid DeepJEB meshes acquire boundary edges during
            # decimation. Keep the repaired full-resolution mesh in that case.
            if simplified.is_watertight:
                mesh = simplified

        processed_faces = len(mesh.faces)
        mesh, center, scale = normalize_mesh(mesh)
        sample = sample_mesh_sdf(
            mesh, num_surface, num_near, num_uniform,
            rng=np.random.default_rng(seed),
            sharp_edge_fraction=sharp_edge_fraction,
            sharp_edge_angle=sharp_edge_angle)
        cond = mesh_descriptors(mesh)
        return {
            'path': path,
            'sample': sample,
            'cond': cond,
            'center': center,
            'scale': scale,
            'original_faces': original_faces,
            'processed_faces': processed_faces,
        }
    except Exception as exc:
        return {'path': path, 'error': str(exc)}


def _load_partial_ascii_stl(path, trimesh_module):
    """Salvage complete triangles from a truncated ASCII STL.

    A small number of DeepJEB files end part-way through their final facet.
    Both trimesh and VTK reject the complete file, although all preceding
    triangles are valid. MeshFix closes the resulting boundary afterward.
    """
    vertices = []
    with open(path, 'rb') as handle:
        for line in handle:
            stripped = line.lstrip()
            if not stripped.startswith(b'vertex '):
                continue
            point = np.fromstring(stripped[7:], sep=' ', dtype=np.float64)
            if point.size == 3 and np.isfinite(point).all():
                vertices.append(point)

    vertices = np.asarray(vertices, dtype=np.float64)
    vertices = vertices[:len(vertices) // 3 * 3]
    if len(vertices) == 0:
        raise ValueError('STL contains no complete triangles')
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return trimesh_module.Trimesh(vertices=vertices, faces=faces, process=True)


def _consume_mesh_result(shapes_grp, written, result):
    if 'error' in result:
        print(f"  [skip] {result['path']}: {result['error']}")
        return 0

    grp = _write_shape(
        shapes_grp, written, result['sample'], result['cond'],
        source=result['path'])
    grp.attrs['center'] = result['center']
    grp.attrs['scale'] = result['scale']
    grp.attrs['original_faces'] = result['original_faces']
    grp.attrs['processed_faces'] = result['processed_faces']
    return 1


def _write_shape(shapes_grp, index, sample, cond, source=''):
    grp = shapes_grp.create_group(f'{index:05d}')
    for key, arr in sample.items():
        grp.create_dataset(key, data=arr, compression='gzip', compression_opts=4)
    grp.create_dataset('cond', data=cond)
    grp.attrs['source'] = source
    return grp


if __name__ == '__main__':
    main()
