"""Convert DeepJEB FieldMesh HDF5 files into the suite's shared mesh contract.

DeepJEB ships one file per bracket holding a ~270k-node second-order tet mesh
plus nodal fields for four load cases (vertical, horizontal, diagonal,
torsional) and two mode shapes. Three conversions happen here:

**Volume to surface.** The tet10 interior is dropped and only the closed surface
triangulation (`faces`) is kept. Peak stress on a bracket lives on the surface,
the design loop only ever asks for surface quantities, and a 270k-node volume
graph does not fit a message-passing model on this hardware.

**Decimation.** The surface is coarsened to `--target-nodes` by geodesic vertex
clustering (see `decimate` for why quadric decimation is not usable here). Each
kept node is an *original* node, so its stress and displacement labels are exact
samples rather than interpolations. Coarsening still misses some of the sharpest
stress peaks -- quantified per sample in `metadata/peak_retention` so the cost is
visible rather than assumed.

**One sample per (bracket, load case).** The load case is written into
`cond_var` rows as a one-hot, which is exactly what that mechanism is for:
input-only channels the model reads but never predicts. Four load cases on one
geometry therefore become four samples that share coordinates and connectivity.

Outputs `stress(MPa)` and `resultant_disp(mm)`. Deliberately *not* the
per-axis displacements: DeepJEB omits `ver_x_disp` entirely, so a uniform
4-component layout would carry a constant-zero row for the vertical case -- the
same silent degeneracy that made an earlier dataset spend half its loss on
constant targets.

  python dataset/build_deepjeb_mgn.py --out dataset/deepjeb_mgn.h5

Dataset: Hong, Kwon, Shin, Park & Kang, ASME JMD 147(4) 041703 (2025), ODC-By v1.0.
"""

import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

LOAD_CASES = ('ver', 'hor', 'dia', 'tor')
FEATURE_NAMES = ['x_coord', 'y_coord', 'z_coord', 'stress', 'disp',
                 'lc_ver', 'lc_hor', 'lc_dia', 'lc_tor']
INPUT_VAR = OUTPUT_VAR = 2
COND_VAR = 4
RAW_ROOT = os.environ.get('DEEPJEB_RAW', 'D:/CAE_datasets_raw/deepjeb')


def surface_from_fieldmesh(path):
    """Closed surface mesh, its nodal fields, and the constrained-node mask."""
    with h5py.File(path, 'r') as f:
        vertices = f['vertices'][...].astype(np.float64)
        faces = f['faces'][...].astype(np.int64)
        nv = f['nodal_variables']
        fields = {}
        for case in LOAD_CASES:
            fields[case] = np.stack([
                nv[f'{case}_stress(MPa)'][...].astype(np.float64),
                nv[f'{case}_resultant_disp(mm)'][...].astype(np.float64),
            ])
        resultant = np.stack([nv[f'{c}_resultant_disp(mm)'][...] for c in LOAD_CASES])

    used, faces_local = np.unique(faces, return_inverse=True)
    return {
        'vertices': vertices[used],
        'faces': faces_local.reshape(faces.shape),
        'fields': {c: v[:, used] for c, v in fields.items()},
        # Zero displacement in every load case AND carrying the model's peak
        # stress: the fully-constrained bolt interface.
        'constrained': (resultant.max(axis=0)[used] < 1e-9),
    }


def _voxel_size_for(vertices, target_nodes, iterations=40):
    """Binary-search the voxel edge length whose occupied-cell count hits the target."""
    span = vertices.max(axis=0) - vertices.min(axis=0)
    lo, hi = span.max() * 1e-4, span.max()
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        occupied = len(np.unique(np.floor(vertices / mid).astype(np.int64), axis=0))
        if occupied > target_nodes:
            lo = mid
        else:
            hi = mid
        if hi - lo < span.max() * 1e-6:
            break
    return 0.5 * (lo + hi)


def decimate(surface, target_nodes, seed=0):
    """Vertex-clustering decimation of the surface, carrying fields exactly.

    Quadric decimation is not usable here: on DeepJEB's second-order surface it
    collapses the mid-side nodes and then refuses to go further, flooring at
    ~23k nodes whatever target is requested (and breaking watertightness on the
    way). Vertex clustering always reaches the target, and because each cluster
    is represented by one *original* node, the stress and displacement labels
    transfer with no interpolation at all -- the retained peak is a real sample
    of the field, not a smoothed one.

    Assignment is *geodesic*, not by voxel membership. A bracket has thin walls,
    and a voxel that spans one merges its front and back surfaces into a single
    node -- which short-circuits the graph through solid material and hands the
    network connectivity the part does not have (measured: 36 edges/node instead
    of the ~6 a surface mesh should have). Seeds are still placed by voxel so
    they stay evenly spread, but every node is then assigned to the nearest seed
    *along mesh edges*, so a cluster can never straddle a wall.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    vertices, faces = surface['vertices'], surface['faces']
    fields = surface.get('fields') or {}
    constrained = surface.get('constrained')
    size = _voxel_size_for(vertices, target_nodes)
    keys = np.floor(vertices / size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    # One seed per occupied voxel: the member closest to that voxel's centroid.
    counts = np.bincount(inverse)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, vertices)
    centroids = sums / counts[:, None]
    offset = ((vertices - centroids[inverse]) ** 2).sum(axis=1)
    seeds = np.zeros(len(counts), dtype=np.int64)
    best = np.full(len(counts), np.inf)
    for node in np.argsort(offset):                    # ascending: first wins
        cluster = inverse[node]
        if offset[node] < best[cluster]:
            best[cluster] = offset[node]
            seeds[cluster] = node

    # Geodesic Voronoi assignment over the surface graph.
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    weights = np.linalg.norm(vertices[e[:, 0]] - vertices[e[:, 1]], axis=1)
    n = len(vertices)
    graph = sp.coo_matrix((np.concatenate([weights, weights]),
                           (np.concatenate([e[:, 0], e[:, 1]]),
                            np.concatenate([e[:, 1], e[:, 0]]))), shape=(n, n)).tocsr()
    _, _, source = dijkstra(graph, indices=seeds, min_only=True,
                            return_predecessors=True)
    reachable = source >= 0
    if not reachable.all():                            # isolated shards: fold into a seed
        source = np.where(reachable, source, seeds[inverse[np.arange(n)]])
    _, inverse = np.unique(source, return_inverse=True)
    rep = np.unique(source)

    remapped = inverse[faces]
    keep = ((remapped[:, 0] != remapped[:, 1]) & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2]))
    tris = np.unique(np.sort(remapped[keep], axis=1), axis=0)
    if len(tris) < 16:
        raise ValueError(f'clustering collapsed the surface to {len(tris)} faces')

    # Drop clusters that lost every incident face, then reindex.
    used, tris_local = np.unique(tris, return_inverse=True)
    kept = rep[used]
    return {
        'vertices': vertices[kept],
        'faces': tris_local.reshape(tris.shape),
        'fields': {c: v[:, kept] for c, v in fields.items()},
        'constrained': None if constrained is None else constrained[kept],
    }


def edges_from_faces(faces):
    """Undirected mesh edges as a [2, E] bidirectional index array."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    e = np.unique(np.sort(e, axis=1), axis=0)
    both = np.concatenate([e, e[:, ::-1]], axis=0)
    return both.T.astype(np.int32)


def build_records(items, target_nodes, verbose=True):
    """One record per (bracket, load case)."""
    records, skipped = [], []
    for n, (item, path) in enumerate(items, 1):
        started = time.time()
        try:
            surface = surface_from_fieldmesh(path)
            small = decimate(surface, target_nodes)
        except Exception as exc:
            skipped.append((item, f'{type(exc).__name__}: {exc}'))
            if verbose:
                print(f'  [{n}/{len(items)}] {item}: SKIPPED {type(exc).__name__}: {exc}',
                      flush=True)
            continue

        edges = edges_from_faces(small['faces'])
        num_nodes = small['vertices'].shape[0]
        retention = {}
        for c, index in zip(LOAD_CASES, range(len(LOAD_CASES))):
            full_peak = np.abs(surface['fields'][c][0]).max()
            kept_peak = np.abs(small['fields'][c][0]).max()
            retention[c] = float(kept_peak / full_peak) if full_peak > 0 else 1.0

            nodal = np.zeros((len(FEATURE_NAMES), 1, num_nodes), dtype=np.float32)
            nodal[0:3, 0, :] = small['vertices'].T
            nodal[3:5, 0, :] = small['fields'][c]
            nodal[5 + index, 0, :] = 1.0                       # load-case one-hot
            records.append({
                'item': item, 'case': c, 'nodal': nodal, 'edges': edges,
                'constrained': small['constrained'],
                'peak_retention': retention[c],
            })
        if verbose:
            print(f'  [{n}/{len(items)}] {item}: {surface["vertices"].shape[0]} -> '
                  f'{num_nodes} nodes, {edges.shape[1]} directed edges, '
                  f'peak kept {np.mean(list(retention.values())) * 100:.0f}%, '
                  f'{time.time() - started:.1f}s', flush=True)
    return records, skipped


def write_contract(out_path, records, split_of):
    """Write the shared `data/{id}/{nodal_data, mesh_edge}` layout."""
    n_features = len(FEATURE_NAMES)
    total = dict(count=0, s=np.zeros(n_features), ss=np.zeros(n_features),
                 lo=np.full(n_features, np.inf), hi=np.full(n_features, -np.inf))
    split_index = {'train': [], 'val': [], 'test': []}

    tmp = out_path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with h5py.File(tmp, 'w') as f:
        grp = f.create_group('data')
        for sample_id, rec in enumerate(records, start=1):
            nodal, edges = rec['nodal'], rec['edges']
            flat = nodal.reshape(n_features, -1).astype(np.float64)
            total['count'] += flat.shape[1]
            total['s'] += flat.sum(axis=1)
            total['ss'] += (flat ** 2).sum(axis=1)
            total['lo'] = np.minimum(total['lo'], flat.min(axis=1))
            total['hi'] = np.maximum(total['hi'], flat.max(axis=1))

            sg = grp.create_group(str(sample_id))
            sg.create_dataset('nodal_data', data=nodal, compression='gzip', compression_opts=4)
            sg.create_dataset('mesh_edge', data=edges, compression='gzip', compression_opts=4)
            md = sg.create_group('metadata')
            md.attrs['source_filename'] = f"{rec['item']}.h5"
            md.attrs['filename_id'] = f"{rec['item']}_{rec['case']}"
            md.attrs['load_case'] = rec['case']
            md.attrs['bracket'] = rec['item']
            md.attrs['num_nodes'] = nodal.shape[2]
            md.attrs['num_edges'] = edges.shape[1]
            md.attrs['num_timesteps'] = 1
            md.attrs['peak_retention'] = rec['peak_retention']
            md.create_dataset('constrained_mask', data=rec['constrained'].astype(np.uint8),
                              compression='gzip')
            md.create_dataset('feature_min', data=nodal.min(axis=(1, 2)))
            md.create_dataset('feature_max', data=nodal.max(axis=(1, 2)))
            md.create_dataset('feature_mean', data=nodal.mean(axis=(1, 2)))
            md.create_dataset('feature_std', data=nodal.std(axis=(1, 2)))
            split_index[split_of.get(rec['item'], 'train')].append(sample_id)

        mean = total['s'] / total['count']
        std = np.sqrt(np.maximum(total['ss'] / total['count'] - mean ** 2, 0.0))

        f.attrs['num_samples'] = len(records)
        f.attrs['num_features'] = n_features
        f.attrs['num_timesteps'] = 1
        f.attrs['builder_input_var'] = INPUT_VAR
        f.attrs['builder_output_var'] = OUTPUT_VAR
        f.attrs['builder_cond_var'] = COND_VAR
        f.attrs['builder_source'] = 'DeepJEB FieldMesh (ASME JMD 147(4) 041703, ODC-By v1.0)'

        top = f.create_group('metadata')
        top.create_dataset('feature_names', data=np.array(FEATURE_NAMES, dtype='S12'))
        norm = top.create_group('normalization_params')
        norm.create_dataset('min', data=total['lo'].astype(np.float32))
        norm.create_dataset('max', data=total['hi'].astype(np.float32))
        norm.create_dataset('mean', data=mean.astype(np.float32))
        norm.create_dataset('std', data=std.astype(np.float32))
        splits = top.create_group('splits')
        for name, ids in split_index.items():
            splits.create_dataset(name, data=np.array(ids, dtype=np.int64))

    os.replace(tmp, out_path)
    return split_index, mean, std


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default=RAW_ROOT)
    parser.add_argument('--out', default='dataset/deepjeb_mgn.h5')
    parser.add_argument('--target-nodes', type=int, default=5000)
    parser.add_argument('--limit', type=int, default=0, help='cap brackets (0 = all present)')
    parser.add_argument('--infer-only', action='store_true',
                        help='write only <out>_infer.h5 (the held-out split)')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--val-frac', type=float, default=0.12)
    parser.add_argument('--test-frac', type=float, default=0.18)
    args = parser.parse_args(argv)

    field_dir = os.path.join(args.raw, 'FieldMesh')
    present = sorted(p[:-3] for p in os.listdir(field_dir) if p.endswith('.h5'))
    if args.limit:
        present = present[:args.limit]
    if not present:
        print(f'no FieldMesh files under {field_dir}', file=sys.stderr)
        return 1

    def load_split(name):
        path = os.path.join(args.raw, 'Metadata', f'{name}_split_random.json')
        with open(path, encoding='utf-8') as fh:
            return set(json.load(fh))

    test_ids = load_split('test')
    official_test = [i for i in present if i in test_ids]
    if len(official_test) >= 5:
        split_of = {i: ('test' if i in test_ids else 'train') for i in present}
        source = "DeepJEB's official random split"
    else:
        # Not enough of the official test brackets were fetched. Fall back to a
        # seeded split *by bracket*: the four load cases of one bracket share a
        # geometry, so splitting by sample would leak it across the boundary and
        # inflate the score.
        rng = np.random.default_rng(args.split_seed)
        shuffled = list(present)
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * args.test_frac)))
        n_val = max(1, int(round(len(shuffled) * args.val_frac)))
        split_of = {}
        for i, item in enumerate(shuffled):
            split_of[item] = 'test' if i < n_test else ('val' if i < n_test + n_val else 'train')
        source = (f'seeded bracket-level split (only {len(official_test)} official '
                  f'test brackets fetched)')
    counts = {name: sum(1 for v in split_of.values() if v == name)
              for name in ('train', 'val', 'test')}
    print(f'{len(present)} brackets present, target {args.target_nodes} nodes')
    print(f'  split by bracket via {source}: ' +
          ', '.join(f'{k}={v}' for k, v in counts.items()), flush=True)

    items = [(i, os.path.join(field_dir, f'{i}.h5')) for i in present]
    records, skipped = build_records(items, args.target_nodes)
    if not records:
        print('no records built', file=sys.stderr)
        return 1

    if args.infer_only:
        # The main file may be open by a running job (Windows refuses to replace
        # a locked file), and re-emitting only the held-out set is cheap.
        infer_path = args.out.replace('.h5', '_infer.h5')
        test_records = [r for r in records if split_of.get(r['item']) == 'test']
        if not test_records:
            print('no test-split brackets present', file=sys.stderr)
            return 1
        write_contract(infer_path, test_records,
                       {r['item']: 'test' for r in test_records})
        print(f'wrote {infer_path}: {len(test_records)} held-out samples '
              f'({len(test_records) // len(LOAD_CASES)} brackets)')
        return 0

    split_index, mean, std = write_contract(args.out, records, split_of)
    # Repo convention: <name>.h5 carries every split, <name>_infer.h5 is the
    # held-out set on its own, which is what the native inference mode reads.
    infer_path = args.out.replace('.h5', '_infer.h5')
    test_records = [r for r in records if split_of.get(r['item']) == 'test']
    if test_records:
        write_contract(infer_path, test_records,
                       {r['item']: 'test' for r in test_records})
        print(f'wrote {infer_path}: {len(test_records)} held-out samples')
    print(f'\nwrote {args.out}: {len(records)} samples '
          f'({len(records) // len(LOAD_CASES)} brackets x {len(LOAD_CASES)} load cases)')
    print(f'  splits: ' + ', '.join(f'{k}={len(v)}' for k, v in split_index.items()))
    if skipped:
        print(f'  skipped {len(skipped)} brackets:')
        for item, why in skipped[:5]:
            print(f'    {item}: {why}')
    print('\n  per-feature mean / std:')
    for name, m, s in zip(FEATURE_NAMES, mean, std):
        print(f'    {name:10s} {m:12.4f}  {s:12.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
