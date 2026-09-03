"""
HDF5-backed shape dataset for SDF-VAE training.

Each item returns a fixed-size random subsample so the default collate works:
    surface_points  (num_encoder_points, 3)
    surface_normals (num_encoder_points, 3)
    query_points    (num_query_points, 3)
    query_sdf       (num_query_points,)
    cond            (cond_dim,)
    shape_idx       ()
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SDFShapeDataset(Dataset):

    def __init__(self, h5_path, indices, num_encoder_points, num_query_points, seed=0):
        self.h5_path = h5_path
        self.indices = list(indices)
        self.num_encoder_points = num_encoder_points
        self.num_query_points = num_query_points
        self.seed = seed
        self._h5 = None  # opened lazily per worker

        with h5py.File(h5_path, 'r') as h5:
            self.cond_names = [str(n) for n in h5.attrs['cond_names']]
            self.cond_dim = len(self.cond_names)

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __getstate__(self):
        """Drop process-local HDF5 handles before Windows worker spawning."""
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown can tear down h5py before dataset objects.
            pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        shape_idx = self.indices[i]
        grp = self._file()['shapes'][f'{shape_idx:05d}']
        rng = np.random.default_rng()

        # Read full arrays (small per shape), subsample in numpy: h5py fancy
        # indexing requires strictly increasing indices, numpy does not.
        surf = grp['surface_points'][:]
        sel = rng.choice(
            surf.shape[0], size=self.num_encoder_points,
            replace=self.num_encoder_points > surf.shape[0])
        surface_points = surf[sel]
        surface_normals = grp['surface_normals'][:][sel]

        sdf_pts = grp['sdf_points'][:]
        qsel = rng.choice(
            sdf_pts.shape[0], size=self.num_query_points,
            replace=self.num_query_points > sdf_pts.shape[0])
        query_points = sdf_pts[qsel]
        query_sdf = grp['sdf_values'][:][qsel]

        return {
            'surface_points': torch.from_numpy(surface_points),
            'surface_normals': torch.from_numpy(surface_normals),
            'query_points': torch.from_numpy(query_points),
            'query_sdf': torch.from_numpy(query_sdf),
            'cond': torch.from_numpy(grp['cond'][:]),
            'shape_idx': torch.tensor(shape_idx, dtype=torch.long),
        }

    def get_cond(self, shape_idx):
        return self._file()['shapes'][f'{shape_idx:05d}']['cond'][:]


def build_dataset_splits(config, split_seed):
    """Seeded 80/10/10 split over shapes (MeshGraphNets convention)."""
    h5_path = config['dataset_dir']
    with h5py.File(h5_path, 'r') as h5:
        num_shapes = int(h5.attrs['num_shapes'])

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(num_shapes)
    if config.get('overfit_all_shapes', False):
        overfit_count = min(int(config.get('overfit_num_shapes', num_shapes)), num_shapes)
        if overfit_count < 1:
            raise ValueError('overfit_num_shapes must be at least 1')
        overfit_idx = perm[:overfit_count]
        num_enc = int(config.get('num_encoder_points', 4096))
        num_qry = int(config.get('num_query_points', 8192))
        make = lambda: SDFShapeDataset(
            h5_path, overfit_idx, num_enc, num_qry, seed=split_seed)
        train_ds, val_ds, test_ds = make(), make(), make()
        print(f'Dataset: {num_shapes} shapes -> overfit train/val/test {overfit_count}')
        config['cond_dim'] = train_ds.cond_dim
        return train_ds, val_ds, test_ds

    n_train = max(int(round(0.8 * num_shapes)), 1)
    n_val = max(int(round(0.1 * num_shapes)), 1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    if len(test_idx) == 0:
        test_idx = val_idx

    num_enc = int(config.get('num_encoder_points', 4096))
    num_qry = int(config.get('num_query_points', 8192))

    make = lambda idx: SDFShapeDataset(h5_path, idx, num_enc, num_qry, seed=split_seed)
    train_ds, val_ds, test_ds = make(train_idx), make(val_idx), make(test_idx)
    print(f'Dataset: {num_shapes} shapes -> train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}')
    config['cond_dim'] = train_ds.cond_dim
    return train_ds, val_ds, test_ds


def compute_cond_stats(dataset):
    """Per-dimension mean/std of condition vectors over a dataset split."""
    conds = np.stack([dataset.get_cond(idx) for idx in dataset.indices])
    mean = conds.mean(axis=0)
    std = np.maximum(conds.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)
