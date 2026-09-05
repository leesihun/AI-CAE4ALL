"""
HDF5-backed shape dataset for SDF-VAE training.

Each item returns a fixed-size random subsample so the default collate works:
    surface_points  (num_encoder_points, 3)
    surface_normals (num_encoder_points, 3)
    query_points    (num_query_points, 3)
    query_sdf       (num_query_points,)
    cond            (cond_dim,)
    shape_idx       ()

Subsampling draws fresh per __getitem__ by default (the training regime), but
its RNG is seeded from torch's stream rather than OS entropy, so a run with a
``seed`` in its config reproduces exactly while keeping the per-item surface
augmentation. With ``deterministic=True`` the draw for shape ``k`` is seeded
from ``(seed, k)``, so validation, test, latent-cache encoding, and evaluation
see the same encoder point cloud and query set for a shape on every call, on
every rank, and in every run.

Conditions. ``cond`` is the per-shape ``shapes/<i>/cond`` row (the geometric
descriptors named by the root ``cond_names`` attr), optionally followed by the
root ``cond_extra`` sidecar row -- FEA labels appended by
``add_fea_conditions.py`` and named by the root ``cond_extra_names`` attr (see
``general_modules/condition_names.py``). ``cond_names`` is the merged list,
``cond_dim`` its length, and ``get_cond`` / ``__getitem__['cond']`` return the
concatenation, so ``condition_names`` in an FM config selects from both
families transparently. A file without ``cond_extra`` is read exactly as
before (byte-identical outputs).
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from general_modules.condition_names import SIDECAR_DATASET, SIDECAR_NAMES_ATTR


def _decode_str(value):
    return value.decode('utf-8', 'replace') if isinstance(value, bytes) else str(value)


def read_cond_extra(h5):
    """Read the optional ``cond_extra`` sidecar from an open HDF5 file.

    Returns ``(names, values)``: the extra condition names (list of str) and a
    float32 ``[num_shapes, k]`` array, or ``([], None)`` when the file has no
    sidecar. Raises ValueError when the sidecar is present but malformed
    (missing names attr, wrong rank, or a names/columns count mismatch).
    """
    if SIDECAR_DATASET not in h5:
        return [], None
    if SIDECAR_NAMES_ATTR not in h5.attrs:
        raise ValueError(f'HDF5 has {SIDECAR_DATASET!r} but no {SIDECAR_NAMES_ATTR!r} root attr; '
                         'rebuild the sidecar with add_fea_conditions.py --overwrite')
    names = [_decode_str(n) for n in np.atleast_1d(h5.attrs[SIDECAR_NAMES_ATTR])]
    values = np.asarray(h5[SIDECAR_DATASET][()], dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f'{SIDECAR_DATASET!r} must be 2-D [num_shapes, k]; got shape {values.shape}')
    if values.shape[1] != len(names):
        raise ValueError(f'{SIDECAR_DATASET!r} has {values.shape[1]} columns but '
                         f'{SIDECAR_NAMES_ATTR!r} lists {len(names)} names: {names}')
    if len(set(names)) != len(names):
        raise ValueError(f'{SIDECAR_NAMES_ATTR!r} contains duplicates: {names}')
    return names, values


class SDFShapeDataset(Dataset):

    def __init__(self, h5_path, indices, num_encoder_points, num_query_points, seed=0,
                 deterministic=False):
        self.h5_path = h5_path
        self.indices = list(indices)
        self.num_encoder_points = num_encoder_points
        self.num_query_points = num_query_points
        self.seed = int(seed) if seed is not None else 0
        # When True, __getitem__ seeds its rng from (seed, shape_idx). Callers
        # may toggle the attribute (train_fm's latent-cache encode does) and
        # restore it afterwards.
        self.deterministic = bool(deterministic)
        self._h5 = None  # opened lazily per worker

        with h5py.File(h5_path, 'r') as h5:
            self.base_cond_names = [str(n) for n in h5.attrs['cond_names']]
            self.extra_cond_names, cond_extra = read_cond_extra(h5)
            if cond_extra is not None:
                num_shapes = int(h5.attrs.get('num_shapes', len(h5['shapes'])))
                if cond_extra.shape[0] != num_shapes:
                    raise ValueError(
                        f'{SIDECAR_DATASET!r} has {cond_extra.shape[0]} rows but the file holds '
                        f'{num_shapes} shapes (shapes appended after the sidecar was built?); '
                        'rebuild it with add_fea_conditions.py --overwrite')
                clash = [n for n in self.extra_cond_names if n in self.base_cond_names]
                if clash:
                    raise ValueError(f'{SIDECAR_NAMES_ATTR!r} repeats geometric cond_names: {clash}')
        # The sidecar is small ([num_shapes, k] float32) and read-only, so it is
        # held in memory: no per-item HDF5 read, and it pickles with the dataset
        # into DataLoader workers. None when the file has no sidecar.
        self._cond_extra = cond_extra
        self.cond_names = self.base_cond_names + self.extra_cond_names
        self.cond_dim = len(self.cond_names)

    @property
    def has_cond_extra(self):
        return self._cond_extra is not None

    def _cond(self, grp, shape_idx):
        """Stored condition row of one shape: geometric ``cond`` (+ sidecar row)."""
        cond = grp['cond'][:]
        if self._cond_extra is None:
            return cond
        extra = self._cond_extra[int(shape_idx)]
        return np.concatenate([cond, extra.astype(cond.dtype, copy=False)])

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

    def _rng(self, shape_idx):
        if self.deterministic:
            return np.random.default_rng([int(self.seed), int(shape_idx)])
        # Stochastic (train) path. The child seed is drawn from *torch's* RNG,
        # not from OS entropy, so the augmentation stream is reachable from a
        # seed without losing its per-item / per-epoch variety:
        #   * num_workers == 0 -- torch's global RNG, which ``seed_stage`` seeds
        #     from the config's ``seed`` key.
        #   * num_workers > 0  -- the DataLoader seeds each worker's torch RNG
        #     to base_seed + worker_id, drawing base_seed from the loader's
        #     ``generator`` (the trainers pass ``seeded_generator(run_seed)``),
        #     so workers stay distinct from each other and across epochs.
        # With no ``seed`` in the config torch's RNG is itself entropy-seeded,
        # which reproduces the legacy unseeded behaviour exactly.
        return np.random.default_rng(int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))

    def __getitem__(self, i):
        shape_idx = self.indices[i]
        grp = self._file()['shapes'][f'{shape_idx:05d}']
        rng = self._rng(shape_idx)

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
            'cond': torch.from_numpy(self._cond(grp, shape_idx)),
            'shape_idx': torch.tensor(shape_idx, dtype=torch.long),
        }

    def get_cond(self, shape_idx):
        """Stored condition vector of ``shape_idx`` (geometric + sidecar), numpy."""
        return self._cond(self._file()['shapes'][f'{shape_idx:05d}'], shape_idx)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def _as_bool(value, default=False):
    """Config booleans arrive as bool from the native parser, but tolerate strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def parent_key_from_source(source):
    """Parent id of a shape from its builder ``source`` attr.

    ``'DeepJEB/SurfaceMesh/101_428.stl' -> '101'``: the basename split at its
    first underscore. Both path separators are accepted (the DeepJEB file was
    built on Windows and stores backslashes). Returns ``None`` when the source
    is missing/empty or its basename has no underscore -- that shape is then
    its own parent.
    """
    if source is None:
        return None
    if isinstance(source, bytes):
        source = source.decode('utf-8', 'replace')
    name = str(source).replace('\\', '/').rsplit('/', 1)[-1].strip()
    if not name or '_' not in name:
        return None
    head = name.split('_', 1)[0]
    return head if head else None


def _own_parent(shape_idx):
    return f'__shape_{int(shape_idx):05d}'


def _read_parent_keys(h5_path, num_shapes):
    """One parent key per shape index, read from the per-shape 'source' attrs."""
    keys = []
    with h5py.File(h5_path, 'r') as h5:
        shapes = h5['shapes']
        for idx in range(num_shapes):
            grp = shapes.get(f'{idx:05d}')
            source = grp.attrs.get('source') if grp is not None else None
            key = parent_key_from_source(source)
            keys.append(key if key is not None else _own_parent(idx))
    return keys


def _split_indices_by_parent(parent_keys, split_seed):
    """Group-aware 80/10/10 split: whole parents go to one split.

    Parents are permuted with ``np.random.default_rng(split_seed)`` and
    assigned greedily -- to train while train holds fewer than ~80% of the
    shapes, then to val while val holds fewer than ~10%, the rest to test --
    while always leaving at least one parent each for val and test.

    Returns (train_idx, val_idx, test_idx, parents_per_split, num_parents) with
    the index arrays as int64 numpy arrays, parents_per_split a (train, val,
    test) tuple of parent counts, and num_parents the effective number of
    parents actually split (after the per-shape fallback, if it triggered).
    """
    num_shapes = len(parent_keys)
    groups = {}
    for idx, key in enumerate(parent_keys):
        groups.setdefault(key, []).append(idx)
    if len(groups) < 3:
        print(f'WARNING: split_by_parent found only {len(groups)} parent id(s) for '
              f'{num_shapes} shapes; falling back to one parent per shape')
        groups = {_own_parent(idx): [idx] for idx in range(num_shapes)}
    parents = list(groups.keys())  # first-appearance (shape index) order

    rng = np.random.default_rng(split_seed)
    order = rng.permutation(len(parents))

    n_train = max(int(round(0.8 * num_shapes)), 1)
    n_val = max(int(round(0.1 * num_shapes)), 1)
    buckets = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}
    n_parents = {'train': 0, 'val': 0, 'test': 0}
    parents_left = len(parents)
    for pos in order:
        members = groups[parents[pos]]
        parents_left -= 1  # parents still to assign after this one
        if counts['train'] < n_train and parents_left >= 2:
            dest = 'train'
        elif counts['val'] < n_val and parents_left >= 1:
            dest = 'val'
        else:
            dest = 'test'
        buckets[dest].extend(members)
        counts[dest] += len(members)
        n_parents[dest] += 1

    if not buckets['train']:
        raise ValueError('split_by_parent produced an empty train split')
    if not buckets['val']:
        # Only reachable with < 3 parents, which the fallback above excludes.
        buckets['val'] = list(buckets['test'])
        n_parents['val'] = n_parents['test']
    if not buckets['test']:
        buckets['test'] = list(buckets['val'])
        n_parents['test'] = n_parents['val']

    to_arr = lambda lst: np.asarray(lst, dtype=np.int64)
    return (to_arr(buckets['train']), to_arr(buckets['val']), to_arr(buckets['test']),
            (n_parents['train'], n_parents['val'], n_parents['test']), len(parents))


def build_dataset_splits(config, split_seed):
    """Seeded 80/10/10 split over shapes (MeshGraphNets convention).

    With ``split_by_parent`` true, whole parent families (see
    ``parent_key_from_source``) are assigned to one split so DeepJEB variants
    of the same base bracket never straddle train and val/test. The val and
    test datasets are constructed with ``deterministic=True``; train is not.
    train_vae, train_fm, and evaluate all go through this function, so they
    see identical splits for the same (dataset, split_seed, split_by_parent).
    """
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
        # val/test are deterministic here too (same contract as the main path):
        # `vae_best_modelpath` selects on ValidSDF, which must not move with the
        # query subsample between epochs.
        make = lambda det: SDFShapeDataset(
            h5_path, overfit_idx, num_enc, num_qry, seed=split_seed, deterministic=det)
        train_ds, val_ds, test_ds = make(False), make(True), make(True)
        print(f'Dataset: {num_shapes} shapes -> overfit train/val/test {overfit_count}')
        _report_cond_extra(train_ds)
        config['cond_dim'] = train_ds.cond_dim
        return train_ds, val_ds, test_ds

    split_by_parent = _as_bool(config.get('split_by_parent', False))
    parent_note = ''
    if split_by_parent:
        parent_keys = _read_parent_keys(h5_path, num_shapes)
        train_idx, val_idx, test_idx, parents_per_split, num_parents = _split_indices_by_parent(
            parent_keys, split_seed)
        parent_note = (f' (split_by_parent: {num_parents} parents -> '
                       f'{parents_per_split[0]}/{parents_per_split[1]}/{parents_per_split[2]})')
        # With few parents the "leave one parent each for val and test" guard
        # caps train well below its 80% target; say so instead of leaving the
        # reader to infer a degenerate split from the printed counts.
        target_train = max(int(round(0.8 * num_shapes)), 1)
        if len(train_idx) < 0.6 * target_train:
            print(f'WARNING: split_by_parent over only {num_parents} parent(s) put '
                  f'{len(train_idx)}/{num_shapes} shapes ({len(train_idx) / num_shapes:.0%}) in '
                  f'train, far below the 80% target; whole parents cannot be divided more '
                  f'finely. Use split_by_parent False, or a dataset with more parents.')
    else:
        n_train = max(int(round(0.8 * num_shapes)), 1)
        n_val = max(int(round(0.1 * num_shapes)), 1)
        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]
        if len(test_idx) == 0:
            test_idx = val_idx

    num_enc = int(config.get('num_encoder_points', 4096))
    num_qry = int(config.get('num_query_points', 8192))

    make = lambda idx, det: SDFShapeDataset(
        h5_path, idx, num_enc, num_qry, seed=split_seed, deterministic=det)
    train_ds = make(train_idx, False)
    val_ds = make(val_idx, True)
    test_ds = make(test_idx, True)
    print(f'Dataset: {num_shapes} shapes -> train {len(train_ds)} / val {len(val_ds)} '
          f'/ test {len(test_ds)}{parent_note}')
    _report_cond_extra(train_ds)
    config['cond_dim'] = train_ds.cond_dim
    return train_ds, val_ds, test_ds


def _report_cond_extra(dataset):
    """One line naming the merged condition vector when a sidecar is present,
    plus a WARNING when it carries NaN rows (add_fea_conditions.py --allow_missing):
    ``compute_cond_stats`` would propagate them into the FM's cond_mean/std."""
    if not dataset.has_cond_extra:
        return
    extra = dataset._cond_extra
    print(f'Conditions: {len(dataset.base_cond_names)} geometric {dataset.base_cond_names} + '
          f'{len(dataset.extra_cond_names)} from {SIDECAR_DATASET} {dataset.extra_cond_names} '
          f'-> cond_dim {dataset.cond_dim}')
    bad_rows = int((~np.isfinite(extra)).any(axis=1).sum())
    if bad_rows:
        bad_names = [n for j, n in enumerate(dataset.extra_cond_names)
                     if not np.isfinite(extra[:, j]).all()]
        print(f'WARNING: {SIDECAR_DATASET} has non-finite values in {bad_rows} shape row(s) '
              f'(names {bad_names}); exclude those names via condition_names or rebuild the '
              'sidecar without --allow_missing before conditional FM training')


def compute_cond_stats(dataset):
    """Per-dimension mean/std of condition vectors over a dataset split.

    NaN is propagated on purpose rather than skipped (`nanmean`): a NaN column
    comes from a `cond_extra` sidecar written with
    `add_fea_conditions.py --allow_missing`, and silently averaging around the
    hole would hide it. `_report_cond_extra` warns when the dataset is built and
    `train_fm.fm_worker` raises on any non-finite selected condition before the
    first optimizer step -- that is where the contract is enforced. The VAE
    stage only records these numbers in its checkpoint as provenance; nothing
    trains on them.
    """
    conds = np.stack([dataset.get_cond(idx) for idx in dataset.indices])
    mean = conds.mean(axis=0)
    std = np.maximum(conds.std(axis=0), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)
