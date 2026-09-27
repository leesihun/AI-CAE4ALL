"""Relabel the DeepJEB brackets with the design loop's own FEA.

Why this exists: DeepJEB's published per-node fields cannot be paired with its
geometry. `nodal_variables` is not in `vertices` order, and it is not in the
order of the `faces` surface either -- even after an exact graph isomorphism
between `faces` and the tet10 boundary rebuilt from `cells`, the vertical
displacement renders as patchy bands and 23% of surface edges jump by more
than 5% of the field range (a linear-static field has essentially none). The
scalar labels in `bracket_labels.csv` match the field maxima, so the values
are real; only the node ordering is lost. A surrogate trained on
`build_deepjeb_mgn.py`'s output therefore learns the marginal of the field,
not the field (edge-smoothness ratio 0.7-0.9 against ~0.05 for a real field).

So the geometry is taken from DeepJEB -- the tet10 `cells` connectivity *is*
consistent with `vertices`, its boundary matches the bracket's own surface-area
label to 0.18% -- and the fields are recomputed with exactly what the design
loop verifies against: `mesher.tet_mesh_with_retries` (the face-budget ladder
over `tet_mesh_from_surface`, which the `fea` backend's Evaluator also uses)
for the tets and `problem.Bracket` for BCs, loads and the linear-static solve. A surrogate
trained on these labels is consistent with the `fea` backend by construction,
which the DeepJEB labels (different solver, tet10, different interface
definitions) never were.

Frames: the surface is normalized per shape exactly as `sdf_sampling.
normalize_mesh` does for SDFFlow's training data, meshed and solved there, and
the surface graph is written in millimetres through `deepjeb_bridge.
normalized_to_millimetres` -- the same map the bridge applies to a generated
shape at inference, so train and serve see one frame.

Outputs per node: von Mises (MPa), |u| (mm) and signed u_z (mm), one sample per
(bracket, load case), the load case one-hot in `cond_var` rows. Row layout and
split policy are `build_deepjeb_mgn`'s. Per-bracket results are cached under
`--cache` so an interrupted run resumes.

  python -m design_loop.build_deepjeb_fea --out ../../dataset/deterministic/ex10_deepjeb_mgn.h5
"""

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np

from design_loop.build_deepjeb_mgn import (
    FEATURE_NAMES, LOAD_CASES, OUTPUT_VAR, RAW_ROOT,
    boundary_faces_from_cells, decimate, edges_from_faces, write_contract,
)
from design_loop.deepjeb_bridge import (
    LABEL_MESH_SIZE_MAX, LABEL_SURFACE_FACES, SDF_TARGET_EXTENT, normalized_to_millimetres,
)

# build_deepjeb_mgn's short case names -> problem.Bracket's load-case keys.
CASE_KEYS = {'ver': 'vertical', 'hor': 'horizontal', 'dia': 'diagonal', 'tor': 'torsion'}


def fieldmesh_surface(path):
    """DeepJEB tet10 boundary (corner triangles) as a consistently wound trimesh, mm."""
    import trimesh
    with h5py.File(path, 'r') as f:
        vertices = f['vertices'][...].astype(np.float64)
        cells = f['cells'][...].astype(np.int64)
    faces = boundary_faces_from_cells(cells)
    used, local = np.unique(faces, return_inverse=True)
    mesh = trimesh.Trimesh(vertices[used], local.reshape(faces.shape), process=True)
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight:
        raise ValueError('FieldMesh boundary is not watertight')
    return mesh


def normalize(mesh):
    """`sdf_sampling.normalize_mesh`: bbox centre to the origin, longest side to 1.8."""
    bounds = mesh.bounds
    out = mesh.copy()
    out.apply_translation(-bounds.mean(axis=0))
    out.apply_scale(SDF_TARGET_EXTENT / float((bounds[1] - bounds[0]).max()))
    return out


def label_surface(nodes, result, target_nodes=5000):
    """A solved bracket's fields as the surrogate's labels store them.

    The tet mesh's boundary nodes, in DeepJEB millimetres, with per load case
    rows (von Mises MPa, |u| mm, signed u_z mm), then geodesically decimated to
    `target_nodes`. `result` is `Bracket.analyze(..., return_fields=True)`.
    Returns (full, small); `verify_with_fea` reads its statistics off `small`
    so that a solver number and a surrogate number are the same measure.
    """
    fields = result['fields']
    faces = fields['faces']
    used, local = np.unique(faces, return_inverse=True)
    constrained = np.zeros(len(nodes), dtype=bool)
    constrained[fields['mount']] = True
    per_case = {}
    for short, key in CASE_KEYS.items():
        if key not in fields['cases']:
            continue
        u = fields['cases'][key]['displacement']
        per_case[short] = np.stack([
            fields['cases'][key]['von_mises_nodal'] / 1e6,     # Pa -> MPa
            np.linalg.norm(u, axis=1) * 1e3,                   # m  -> mm
            u[:, 2] * 1e3,                                     # signed vertical, mm
        ])[:, used]
    full = {'vertices': normalized_to_millimetres(nodes[used]),
            'faces': local.reshape(faces.shape),
            'fields': per_case, 'constrained': constrained[used]}
    return full, decimate(full, target_nodes)


def label_bracket(item, path, cache_dir, target_nodes=5000, mesh_size_max=LABEL_MESH_SIZE_MAX,
                  target_faces=LABEL_SURFACE_FACES):
    """Mesh + solve one bracket; cache and return its per-case surface fields."""
    cache = os.path.join(cache_dir, f'{item}.npz')
    if os.path.exists(cache):
        return item, dict(np.load(cache, allow_pickle=False)), None

    from design_loop.mesher import tet_mesh_with_retries
    from design_loop.problem import Bracket

    started = time.time()
    try:
        surface = normalize(fieldmesh_surface(path))
        nodes, tets, mesh_info = tet_mesh_with_retries(
            surface, mesh_size_max=mesh_size_max, target_faces=target_faces)
        bracket = Bracket(load_cases=tuple(CASE_KEYS.values()))
        result = bracket.analyze(nodes, tets, return_fields=True)
    except Exception as exc:
        return item, None, f'{type(exc).__name__}: {exc}'

    full, small = label_surface(nodes, result, target_nodes)
    per_case = full['fields']

    out = {
        'vertices': small['vertices'].astype(np.float32),
        'faces': small['faces'].astype(np.int32),
        'constrained': small['constrained'].astype(np.uint8),
        'mass': np.float64(result['mass']),
        'num_tets': np.int64(result['num_tets']),
        'min_sicn': np.float64(mesh_info['min_sicn']),
        'seconds': np.float64(time.time() - started),
    }
    for short in LOAD_CASES:
        out[f'field_{short}'] = small['fields'][short].astype(np.float32)
        c = result['cases'][CASE_KEYS[short]]
        out[f'peak_vm_{short}'] = np.float64(c['peak_von_mises'])
        out[f'max_disp_{short}'] = np.float64(c['max_displacement'])
        out[f'max_uz_{short}'] = np.float64(np.abs(fields['cases'][CASE_KEYS[short]]
                                                   ['displacement'][:, 2]).max())
        full_peak = np.abs(per_case[short][0]).max()
        out[f'retention_{short}'] = np.float64(
            np.abs(small['fields'][short][0]).max() / full_peak if full_peak > 0 else 1.0)
    tmp = cache + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, cache)
    return item, out, None


def records_from(item, lab):
    edges = edges_from_faces(lab['faces'].astype(np.int64))
    n = lab['vertices'].shape[0]
    recs = []
    for index, short in enumerate(LOAD_CASES):
        nodal = np.zeros((len(FEATURE_NAMES), 1, n), dtype=np.float32)
        nodal[0:3, 0, :] = lab['vertices'].T
        nodal[3:3 + OUTPUT_VAR, 0, :] = lab[f'field_{short}']
        nodal[3 + OUTPUT_VAR + index, 0, :] = 1.0
        recs.append({'item': item, 'case': short, 'nodal': nodal, 'edges': edges,
                     'constrained': lab['constrained'].astype(bool),
                     'peak_retention': float(lab[f'retention_{short}'])})
    return recs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default=RAW_ROOT)
    parser.add_argument('--out', default='../../dataset/deterministic/ex10_deepjeb_mgn.h5')
    parser.add_argument('--cache', default='../../output/deepjeb_fea_labels')
    parser.add_argument('--target-nodes', type=int, default=5000)
    parser.add_argument('--mesh-size-max', type=float, default=LABEL_MESH_SIZE_MAX)
    parser.add_argument('--target-faces', type=int, default=LABEL_SURFACE_FACES)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--label-only', action='store_true',
                        help='fill the cache, do not write the contract')
    args = parser.parse_args(argv)

    # One solver per process: threaded BLAS in every worker oversubscribes the box.
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    field_dir = os.path.join(args.raw, 'FieldMesh')
    present = sorted(p[:-3] for p in os.listdir(field_dir) if p.endswith('.h5'))
    if args.limit:
        present = present[:args.limit]
    os.makedirs(args.cache, exist_ok=True)
    with open(os.path.join(args.raw, 'Metadata', 'test_split_random.json'),
              encoding='utf-8') as fh:
        test_ids = set(json.load(fh))
    split_of = {i: ('test' if i in test_ids else 'train') for i in present}
    print(f'{len(present)} brackets, {sum(v == "test" for v in split_of.values())} '
          f"in DeepJEB's official random test split; {args.workers} workers", flush=True)

    labels, skipped = {}, {}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(label_bracket, item, os.path.join(field_dir, f'{item}.h5'),
                               args.cache, args.target_nodes, args.mesh_size_max,
                               args.target_faces) for item in present]
        for n, fut in enumerate(as_completed(futures), 1):
            item, lab, err = fut.result()
            if err:
                skipped[item] = err
                print(f'  [{n}/{len(present)}] {item}: SKIPPED {err}', flush=True)
                continue
            labels[item] = lab
            print(f'  [{n}/{len(present)}] {item}: {lab["vertices"].shape[0]} nodes, '
                  f'{int(lab["num_tets"])} tets, peak vm ver {lab["peak_vm_ver"] / 1e6:.0f} MPa, '
                  f'u_z ver {lab["max_uz_ver"] * 1e3:.3f} mm, {float(lab["seconds"]):.0f}s '
                  f'(elapsed {time.time() - started:.0f}s)', flush=True)

    summary = os.path.join(args.cache, 'summary.csv')
    with open(summary, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        cols = ['mass', 'num_tets', 'min_sicn'] + [f'{k}_{c}' for c in LOAD_CASES
                                                   for k in ('peak_vm', 'max_disp', 'max_uz')]
        w.writerow(['bracket', 'split'] + cols)
        for item in sorted(labels):
            w.writerow([item, split_of[item]] + [float(labels[item][c]) for c in cols])
    with open(os.path.join(args.cache, 'skipped.json'), 'w', encoding='utf-8') as fh:
        json.dump(skipped, fh, indent=1)
    print(f'labelled {len(labels)}, skipped {len(skipped)}; summary {summary}')
    if args.label_only or not labels:
        return 0 if labels else 1

    train = [r for i in sorted(labels) if split_of[i] != 'test' for r in records_from(i, labels[i])]
    test = [r for i in sorted(labels) if split_of[i] == 'test' for r in records_from(i, labels[i])]
    split_index, mean, std = write_contract(args.out, train, {i: 'train' for i in labels})
    infer = args.out.replace('.h5', '_infer.h5')
    if test:
        write_contract(infer, test, {i: 'test' for i in labels})
    for path in (args.out, infer):
        if os.path.exists(path):
            with h5py.File(path, 'a') as f:
                f.attrs['builder_source'] = ('DeepJEB FieldMesh geometry (ASME JMD 147(4) '
                                             '041703, ODC-By v1.0), fields relabelled by '
                                             'design_loop FEA (build_deepjeb_fea.py)')
    print(f'wrote {args.out}: {len(train)} samples; {infer}: {len(test)} samples')
    for name, m, s in zip(FEATURE_NAMES, mean, std):
        print(f'    {name:10s} {m:12.4f}  {s:12.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
