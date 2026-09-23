"""Rebuild ex3_NASA_CRM_mid{,_infer}.h5 as a coarsened copy of _full.

Why this exists
---------------
The shipped mid files carry a `mesh_edge` graph that is not a reconstructable
surface: 27.7% of its edges carry three or more faces, so no signed distance
can be derived from it and the visualisers draw a triangle soup. The cause is
that mid was a *block-wise non-uniform* subsample of full -- 47% of its nodes
keep all four full-mesh neighbours (original resolution) while 13% keep none
(strided) -- and that node set has no manifold surface through it.

Everything else about mid was already exactly full restricted to that node set:
same 105/44 samples, same 17 feature rows, values bit-identical. So the fix is
to choose a *different* node subset, one that does carry a clean surface, and
copy the same values onto it.

Method: polychord collapse. full's surface is a multi-block structured quad
mesh (95.8% of its nodes are regular), and its 909,700 edges partition into
2,213 polychords -- strips of quads running along the grid lines. Collapsing a
polychord merges the two endpoints of each of its edges, which deletes that
strip and leaves every other quad a quad, so the result stays a pure quad mesh
with no holes opened anywhere. Collapsing a node-disjoint set of chords halves
the mesh in one pass; three budgeted passes reach the target size.

Two alternatives were measured and rejected:

* Triangulating full and running `decimate_pro` gives a geometrically better
  surface, but the HDF5 contract stores edges and no faces, so the boundary has
  to be re-derived by counting 3-cycles -- and the decimated surface contains
  ~2,100 chord triangles across the thin trailing edges that are 3-cycles
  without being faces. They flip VTK's normal orientation, and 23% of the
  bounding box then reads as *inside* against full's 6.6%. Quads have no
  diagonals, so this ambiguity does not exist for them at all.
* Keeping every other node by a checkerboard parity needs full's mesh graph to
  be bipartite. It is not: 4.5% of its edges join same-parity nodes even after
  local refinement, and the resulting coarse mesh is 18-25% open.

Run from the suite root:

    python configs/campaigns/dataset_matrix/rebuild_crm_mid.py --check
    python configs/campaigns/dataset_matrix/rebuild_crm_mid.py --write

`--check` builds the surface, grades it, signs a grid against it and writes
nothing. `--write` replaces the two mid files, after saving the node subset and
edges of the outgoing ones to `junk/` -- mid is fully derivable from full plus
those two arrays, so that tiny file is enough to reconstruct what was replaced.
"""
from pathlib import Path
import argparse
import os
import sys

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
FULL = ROOT / 'dataset/deterministic/ex3_NASA_CRM_full{suffix}.h5'
MID = ROOT / 'dataset/deterministic/ex3_NASA_CRM_mid{suffix}.h5'
DERIVED = ROOT / 'dataset/derived/config_matrix'
BACKUP = ROOT / 'junk/ex3_NASA_CRM_mid_previous_topology.npz'

# The node count to coarsen down to; the outgoing files had 122,778.
TARGET_NODES = 123000
# Chord selection is randomised, so the subset has to be pinned to be reproducible.
SEED = 0

# Reconstruction quality the rebuilt surface has to beat, matching the gate in
# methods/Neural_Operator/model/adapters/sdf_from_mesh.py.
MAX_NONMANIFOLD_FRACTION = 0.02
MAX_OPEN_FRACTION = 0.05


def _neural_operator():
    """The reconstruction/signing code lives with the model that consumes it."""
    path = str(ROOT / 'methods/Neural_Operator')
    if path not in sys.path:
        sys.path.insert(0, path)
    from model.adapters import sdf_from_mesh
    return sdf_from_mesh


def sample_ids(f):
    return sorted(f['data'], key=int)


def _reference_geometry(path):
    """Coordinates and edges of the first sample.

    Geometry varies per sample -- the control surfaces deflect -- but the
    topology does not, so the coarsening is computed once here and the same
    node subset and edges are then used for every sample and both files.
    """
    with h5py.File(path, 'r') as f:
        sid = sample_ids(f)[0]
        coords = np.asarray(f[f'data/{sid}/nodal_data'][0:3, 0, :], np.float64).T
        edges = np.asarray(f[f'data/{sid}/mesh_edge'][:], np.int64)
    return coords, edges


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)   # the lower index stays as the survivor


def _polychords(quads, edges, num_nodes):
    """Group edges into polychords, and report which chord pair meets in each quad.

    Two edges are in the same chord when they are opposite sides of some quad;
    following that relation across the mesh traces a strip of quads.
    """
    linear = edges[:, 0].astype(np.int64) * num_nodes + edges[:, 1]
    order = np.argsort(linear, kind='stable')
    sorted_linear = linear[order]

    def edge_id(a, b):
        lo = np.minimum(a, b).astype(np.int64)
        hi = np.maximum(a, b)
        return order[np.searchsorted(sorted_linear, lo * num_nodes + hi)]

    sides = [edge_id(quads[:, i], quads[:, (i + 1) % 4]) for i in range(4)]
    parent = np.arange(len(edges))
    for a, b in zip(sides[0], sides[2]):
        _union(parent, a, b)
    for a, b in zip(sides[1], sides[3]):
        _union(parent, a, b)
    root = np.array([_find(parent, i) for i in range(len(edges))])
    _, chord_of_edge = np.unique(root, return_inverse=True)
    crossing = np.stack([chord_of_edge[sides[0]], chord_of_edge[sides[1]]], axis=1)
    return chord_of_edge, crossing


def _collapse_pass(num_nodes, quads, edges, rng, budget=None):
    """Collapse a maximal set of chords; pure quad mesh in, pure quad mesh out."""
    chord_of_edge, crossing = _polychords(quads, edges, num_nodes)
    n_chords = int(chord_of_edge.max()) + 1
    # Two chords may share nodes when they cross -- they meet inside some quad.
    # Two *parallel* neighbours sharing a node would collapse three grid lines
    # into one, so those are the ones to keep apart.
    crosses = set(map(tuple, np.unique(np.sort(crossing, axis=1), axis=0)))
    order = np.argsort(chord_of_edge, kind='stable')
    run = chord_of_edge[order]
    bounds = np.flatnonzero(np.r_[True, run[1:] != run[:-1], True])
    members = {int(run[bounds[i]]): order[bounds[i]:bounds[i + 1]]
               for i in range(len(bounds) - 1)}

    owner = -np.ones(num_nodes, np.int64)
    picked = []
    for chord in rng.permutation(n_chords):
        chord = int(chord)
        nodes = np.unique(edges[members[chord]])
        claimed = owner[nodes]
        claimed = np.unique(claimed[claimed >= 0])
        if any((min(chord, int(o)), max(chord, int(o))) not in crosses for o in claimed):
            continue
        owner[nodes] = chord
        picked.append(chord)
        if budget is not None:
            budget -= len(members[chord])
            if budget <= 0:
                break

    parent = np.arange(num_nodes)
    for chord in picked:
        for a, b in edges[members[chord]]:
            _union(parent, a, b)
    representative = np.array([_find(parent, i) for i in range(num_nodes)])

    merged = representative[quads]
    distinct = (np.diff(np.sort(merged, axis=1), axis=1) != 0).sum(1) + 1
    merged = merged[distinct == 4]          # the collapsed strip's own quads vanish
    surviving = np.unique(merged)
    remap = -np.ones(num_nodes, np.int64)
    remap[surviving] = np.arange(len(surviving))
    merged = remap[merged]
    new_edges = np.unique(np.sort(np.concatenate(
        [merged[:, [0, 1]], merged[:, [1, 2]],
         merged[:, [2, 3]], merged[:, [3, 0]]]), axis=1), axis=0)
    return surviving, merged, new_edges, len(picked), n_chords


def build_surface(coords, edges, verbose=True):
    """Coarsen full's quad surface; return (full node indices, coarse edges)."""
    mesh = _neural_operator()
    num_nodes = len(coords)
    quads = mesh._quads(mesh._adjacency(mesh.undirected_edges(edges), num_nodes),
                        num_nodes)
    if len(quads) < num_nodes * 0.5:
        raise ValueError(f'full mesh did not reconstruct as quads: {len(quads)} '
                         f'quads for {num_nodes} nodes')
    rng = np.random.default_rng(SEED)
    keep = np.arange(num_nodes)
    current = num_nodes
    current_edges = mesh.undirected_edges(edges)
    for step in range(1, 40):
        # a full pass halves the mesh; the last one is budgeted so it lands on
        # the target instead of overshooting to a third of it
        budget = None if current / 2 > TARGET_NODES * 1.05 else max(current - TARGET_NODES, 0)
        surviving, quads, current_edges, picked, total = _collapse_pass(
            current, quads, current_edges, rng, budget)
        keep = keep[surviving]
        current = len(surviving)
        if verbose:
            print(f'  pass {step}: collapsed {picked}/{total} chords -> '
                  f'{current} nodes, {len(quads)} quads', flush=True)
        if current <= TARGET_NODES * 1.02:
            break
    else:
        raise ValueError('coarsening did not converge to the target node count')
    return keep, current_edges


def report(coords, keep, edges, full_edges):
    """Grade the reconstruction and check the sign against the full mesh."""
    mesh = _neural_operator()
    points = coords[keep]
    faces, quality = mesh.reconstruct_faces(edges.T, len(keep))
    print(f'rebuilt surface: nodes={len(keep)}  {quality}', flush=True)

    low, high = coords.min(0), coords.max(0)
    pad = 0.05 * (high - low)
    axes = [np.linspace(low[i] - pad[i], high[i] + pad[i], 20) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    corners = np.array(np.meshgrid(
        *[[low[i] - pad[i], high[i] + pad[i]] for i in range(3)],
        indexing='ij')).reshape(3, -1).T

    coarse = mesh._signed_distance_3d(grid, points, faces)
    full_faces, _ = mesh.reconstruct_faces(full_edges, len(coords))
    fine = mesh._signed_distance_3d(grid, coords, full_faces)
    bad = int((mesh._signed_distance_3d(corners, points, faces) < 0).sum())
    print(f'  inside fraction: coarse={np.mean(coarse < 0):.2%} '
          f'full={np.mean(fine < 0):.2%}   corners outside the body: '
          f'{8 - bad}/8', flush=True)
    return quality, bad


def _save_previous_topology():
    """The outgoing mid is full plus a node subset plus its edges; keep those."""
    if BACKUP.exists():
        return
    payload = {}
    for suffix in ('', '_infer'):
        path = Path(str(MID).format(suffix=suffix))
        if not path.exists():
            continue
        # per-sample geometry differs (control surfaces deflect), so each mid
        # file has to be matched against the full file it was cut from
        full_coords, _ = _reference_geometry(str(FULL).format(suffix=suffix))
        tree = cKDTree(full_coords)
        with h5py.File(path, 'r') as f:
            sid = sample_ids(f)[0]
            coords = np.asarray(f[f'data/{sid}/nodal_data'][0:3, 0, :], np.float64).T
            payload[f'edges{suffix}'] = np.asarray(f[f'data/{sid}/mesh_edge'][:])
        distance, index = tree.query(coords)
        if distance.max() != 0.0:
            raise ValueError(f'{path.name}: node set is not a subset of full; '
                             'it cannot be reconstructed from the backup.')
        payload[f'nodes{suffix}'] = index.astype(np.int64)
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BACKUP, **payload)
    print(f'saved previous topology -> {BACKUP.relative_to(ROOT)}', flush=True)


def write_mid(suffix, keep, edges):
    source = Path(str(FULL).format(suffix=suffix))
    target = Path(str(MID).format(suffix=suffix))
    temporary = target.with_suffix('.h5.new')
    if temporary.exists():
        temporary.unlink()

    with h5py.File(source, 'r') as src, h5py.File(target, 'r') as old:
        # Keep mid's own row order: prepare.py's CRM_ORDER asserts against it.
        full_names = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in src['metadata/feature_names'][:]]
        mid_names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in old['metadata/feature_names'][:]]
        rows = [full_names.index(n) for n in mid_names]
        ids = sample_ids(src)
        if ids != sample_ids(old):
            raise ValueError(f'{target.name}: sample ids differ from full.')

        with h5py.File(temporary, 'x') as dst:
            for k, v in old.attrs.items():
                dst.attrs[k] = v
            dst.create_dataset('metadata/feature_names',
                               data=old['metadata/feature_names'][:])
            for name in ('test', 'train', 'val'):
                dst.create_dataset(f'metadata/splits/{name}',
                                   data=old[f'metadata/splits/{name}'][:])

            running = []
            for sid in ids:
                arr = np.asarray(src[f'data/{sid}/nodal_data'][:, :, :])[rows][:, :, keep]
                group = dst.create_group(f'data/{sid}')
                group.create_dataset('nodal_data', data=arr.astype(np.float32),
                                     compression='gzip', compression_opts=4)
                # stored uncompressed and undirected, exactly as full/mid do
                group.create_dataset('mesh_edge', data=edges.astype(np.int32))
                meta = group.create_group('metadata')
                flat = arr.reshape(arr.shape[0], -1)
                meta.create_dataset('feature_min', data=flat.min(1).astype(np.float32))
                meta.create_dataset('feature_max', data=flat.max(1).astype(np.float32))
                meta.create_dataset('feature_mean', data=flat.mean(1).astype(np.float32))
                meta.create_dataset('feature_std', data=flat.std(1).astype(np.float32))
                for k, v in old[f'data/{sid}/metadata'].attrs.items():
                    meta.attrs[k] = v
                meta.attrs['num_nodes'] = len(keep)
                meta.attrs['num_edges'] = edges.shape[1]
                running.append(flat)
                print(f'  {target.name} sample {sid}: {arr.shape}', flush=True, end='\r')

            stacked = np.concatenate(running, axis=1)
            for name, value in (('min', stacked.min(1)), ('max', stacked.max(1)),
                                ('mean', stacked.mean(1)), ('std', stacked.std(1))):
                dst.create_dataset(f'metadata/normalization_params/{name}',
                                   data=value.astype(np.float32))
    os.replace(temporary, target)
    print(f'\nwrote {target.relative_to(ROOT)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true',
                        help='replace the mid files (default: report only)')
    args = parser.parse_args()

    coords, full_edges = _reference_geometry(str(FULL).format(suffix=''))
    keep, edges = build_surface(coords, full_edges)
    quality, bad_corners = report(coords, keep, edges, full_edges)
    if quality.nonmanifold_fraction > MAX_NONMANIFOLD_FRACTION:
        raise SystemExit('rebuilt surface is too non-manifold to sign.')
    if quality.open_fraction > MAX_OPEN_FRACTION:
        raise SystemExit('rebuilt surface has too many open edges to sign.')
    if bad_corners:
        raise SystemExit(f'{bad_corners} of 8 bounding-box corners sign as inside the '
                         'body; the reconstruction is not orientable.')
    if not args.write:
        print('--check only; nothing written. Pass --write to replace the files.')
        return

    _save_previous_topology()
    mesh_edge = edges.T.copy()
    for suffix in ('', '_infer'):
        write_mid(suffix, keep, mesh_edge)

    for path in (DERIVED / f'ex3_NASA_CRM_mid_canonical{s}.h5' for s in ('', '_infer')):
        if path.exists():
            path.unlink()
            print(f'removed stale derived view {path.relative_to(ROOT)}', flush=True)
    print('\nNow re-run: python configs/campaigns/dataset_matrix/prepare.py')


if __name__ == '__main__':
    main()
