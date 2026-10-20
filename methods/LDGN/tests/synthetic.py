"""A tiny ex3-shaped mesh HDF5, written into a pytest tmp_path.

Layout (DATASET_FORMAT.md): data/{sid}/nodal_data [9, 1, N] =
x, y, z, ux, uy, uz, thickness, delta, node_type; data/{sid}/mesh_edge
[2, E] unidirectional; data/{sid}/metadata attrs shape_idx (the system) and
the hidden imperfection parameters; a top-level `latent` group (ex3 stores
the per-sample imperfection latents there) and metadata/feature_names.

Each shape is a curved square plate; each draw scales a shape-specific
displacement mode by a hidden amplitude, so draws of one shape differ only
through the hidden parameter.
"""
import h5py
import numpy as np

HIDDEN_ATTRS = ('imp_amplitude', 'imp_mode', 'imp_seed')
FEATURE_NAMES = [b'x_coord', b'y_coord', b'z_coord', b'ux', b'uy', b'uz',
                 b'thickness', b'delta', b'node_type']


def grid_mesh(n):
    """Nodes and unidirectional triangle edges of an n x n grid."""
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij')
    idx = np.arange(n * n).reshape(n, n)
    edges = set()
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i, j + 1], idx[i + 1, j + 1]
            for u, v in ((a, b), (a, c), (b, c), (b, d), (c, d)):
                edges.add((min(u, v), max(u, v)))
    edges = np.asarray(sorted(edges), dtype=np.int64).T
    return xs.ravel(), ys.ravel(), edges


def write_dataset(path, num_shapes=10, draws=4, grid=10, start_sid=0, seed=0):
    rng = np.random.default_rng(seed)
    x, y, edges = grid_mesh(grid)
    n = x.size
    sid = start_sid
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        latent = f.create_group('latent')
        for shape in range(num_shapes):
            curv = 0.1 + 0.05 * shape
            z = curv * ((x - 0.5) ** 2 + (y - 0.5) ** 2)
            thickness = 1.0 + 0.1 * (shape % 3)
            node_type = np.where((x == 0) | (x == 1) | (y == 0) | (y == 1), 1.0, 0.0)
            mode = np.sin(np.pi * x * (1 + shape % 2)) * np.sin(np.pi * y)
            for k in range(draws):
                amp = float(rng.normal())
                delta = 0.5 + 0.1 * (k % 2)
                nodal = np.zeros((9, 1, n), dtype=np.float32)
                nodal[0, 0], nodal[1, 0], nodal[2, 0] = x, y, z
                nodal[3, 0] = 0.01 * amp * mode
                nodal[4, 0] = 0.005 * amp * mode * x
                nodal[5, 0] = amp * mode / thickness
                nodal[6, 0] = thickness
                nodal[7, 0] = delta
                nodal[8, 0] = node_type
                grp = data.create_group(str(sid))
                grp.create_dataset('nodal_data', data=nodal)
                grp.create_dataset('mesh_edge', data=edges)
                meta = grp.create_group('metadata')
                meta.attrs['shape_idx'] = shape
                meta.attrs['imp_amplitude'] = amp
                meta.attrs['imp_mode'] = shape % 2
                meta.attrs['imp_seed'] = sid
                latent.create_dataset(str(sid), data=np.asarray([amp], dtype=np.float32))
                sid += 1
        f.create_group('metadata').create_dataset('feature_names', data=np.asarray(FEATURE_NAMES))
    return path


def write_constant_conditions(path, num_shapes=3, draws=3, grid=8, start_sid=1000, seed=1):
    """Same layout; delta constant per shape (for geometry-scope sampling)."""
    write_dataset(path, num_shapes, draws, grid, start_sid, seed)
    with h5py.File(path, 'r+') as f:
        for sid in f['data']:
            nodal = f[f'data/{sid}/nodal_data']
            block = nodal[...]
            block[7] = 0.5
            nodal[...] = block
    return path
