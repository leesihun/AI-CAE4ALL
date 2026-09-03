"""HDF5-native full-order-model (FOM) field dataset for SimulGenVAE.

SimulGenVAE is a **fixed-geometry dense FOM VAE**: it consumes a dense tensor
``[num_samples, num_channels, num_time]`` (channels are the 1-D conv input dim,
time is the sequence dim). This module builds that tensor from the shared mesh
HDF5 contract used by every other method in the suite
(``data/{sample_id}/nodal_data`` with shape ``[num_features, num_timesteps,
num_nodes]``; rows ``0:3`` are reference coordinates, rows ``3:`` physical
fields). ``mesh_edge`` is part of the shared contract but SimulGenVAE ignores it.

Mapping (see CLAUDE.md):
    nodal_data[F, T, N]
      -> physical rows [field_start_row : field_start_row + num_var]  -> [num_var, T, N]
      -> optional node_start:node_end on N, timesteps_reduced on T
      -> reorder to [T, num_var*N]
    stack over samples -> [num_samples, T, num_var*N]

Because the VAE is dense and fixed-geometry, **every sample must share the same
N and T**; mismatches are a hard error. The scaler (MinMax to [-0.7, 0.7], the
SimulGenVAE convention) is fit here and returned as plain arrays so it can be
folded into the checkpoint ``normalization`` payload.
"""

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

FEATURE_RANGE = (-0.7, 0.7)


def _sample_ids(h5):
    """Sorted integer sample IDs under the shared ``data/`` group."""
    if 'data' not in h5:
        raise ValueError("HDF5 is missing the root 'data' group (shared mesh contract).")
    ids = sorted(int(k) for k in h5['data'].keys() if str(k).isdigit())
    if not ids:
        raise ValueError("Group 'data' has no numeric sample IDs.")
    return ids


def load_fom_from_hdf5(config):
    """Assemble the dense FOM tensor ``[num_samples, num_time, num_channels]``.

    Reads ``dataset_dir`` and returns ``(data, sample_ids)`` where ``data`` is a
    float32 array of shape ``[num_samples, T, num_var*N]``. Enforces the
    fixed-geometry constraint (uniform N and T) and the feature-row bound.
    """
    h5_path = config['dataset_dir']
    num_var = int(config.get('num_var', 1))
    field_start = int(config.get('field_start_row', 3))
    node_start = int(config.get('node_start', 0))
    node_end = config.get('node_end', 0)
    node_end = int(node_end) if node_end not in (None, 0, '') else None
    t_red = int(config.get('timesteps_reduced', 0) or 0)

    if num_var < 1:
        raise ValueError(f"num_var must be >= 1; got {num_var}.")
    if field_start < 0:
        raise ValueError(f"field_start_row must be >= 0; got {field_start}.")

    with h5py.File(h5_path, 'r') as h5:
        ids = _sample_ids(h5)
        ref_shape = None
        mismatched = []
        fields = []
        for sid in ids:
            arr = h5[f'data/{sid}/nodal_data']  # [F, T, N]
            F, T, N = arr.shape
            if F < field_start + num_var:
                raise ValueError(
                    f"Sample {sid}: nodal_data has {F} feature rows but "
                    f"field_start_row + num_var = {field_start + num_var} are required.")
            if ref_shape is None:
                ref_shape = (T, N)
            elif (T, N) != ref_shape:
                mismatched.append((sid, (T, N)))
                continue
            field = arr[field_start:field_start + num_var, :, :]  # [num_var, T, N]
            fields.append(np.asarray(field, dtype=np.float32))

    if mismatched:
        raise ValueError(
            "SimulGenVAE requires a fixed geometry: every sample must share the "
            f"same (T, N)={ref_shape}. Mismatched samples: {mismatched[:8]}"
            + (" ..." if len(mismatched) > 8 else ""))

    T0, N0 = ref_shape
    n_start = max(node_start, 0)
    n_end = node_end if (node_end is not None and node_end <= N0) else N0
    if n_end <= n_start:
        raise ValueError(f"node_end ({n_end}) must be greater than node_start ({n_start}).")
    t_end = t_red if (0 < t_red <= T0) else T0

    samples = []
    for field in fields:                              # [num_var, T, N]
        field = field[:, :t_end, n_start:n_end]       # [num_var, t_end, n]
        field = np.transpose(field, (1, 0, 2))        # [t_end, num_var, n]
        field = field.reshape(field.shape[0], -1)     # [t_end, num_var*n]
        samples.append(field)
    data = np.stack(samples, axis=0)                  # [num_samples, T, C]
    return data, ids


def fit_minmax(data):
    """Per-channel MinMax fit over (samples x time). Returns a normalization dict."""
    C = data.shape[-1]
    flat = data.reshape(-1, C)
    dmin = flat.min(axis=0)
    dmax = flat.max(axis=0)
    return {
        'field_min': dmin.astype(np.float32),
        'field_max': dmax.astype(np.float32),
        'feature_range': np.asarray(FEATURE_RANGE, dtype=np.float32),
    }


def apply_minmax(data, norm):
    """Scale ``data`` into ``feature_range`` using a normalization dict."""
    dmin, dmax = norm['field_min'], norm['field_max']
    span = np.where(dmax > dmin, dmax - dmin, 1.0)
    lo, hi = float(norm['feature_range'][0]), float(norm['feature_range'][1])
    unit = (data - dmin) / span            # -> [0, 1]
    return (unit * (hi - lo) + lo).astype(np.float32)


def invert_minmax(scaled, norm):
    """Inverse of :func:`apply_minmax` (scaled field -> physical field)."""
    dmin, dmax = norm['field_min'], norm['field_max']
    span = np.where(dmax > dmin, dmax - dmin, 1.0)
    lo, hi = float(norm['feature_range'][0]), float(norm['feature_range'][1])
    unit = (scaled - lo) / (hi - lo)
    return (unit * span + dmin).astype(np.float32)


class FomFieldDataset(Dataset):
    """In-memory dense field dataset. Items are ``[num_channels, num_time]`` tensors."""

    def __init__(self, data_nct, indices):
        # data_nct: float32 array [num_samples, num_channels, num_time] (already scaled)
        self.data = data_nct
        self.indices = list(indices)
        self.num_channels = int(data_nct.shape[1])
        self.num_time = int(data_nct.shape[2])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.data[idx]), torch.tensor(idx, dtype=torch.long)


def _split_indices(num_samples, split_seed):
    """Seeded 80/10/10 split over sample order (suite convention)."""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(num_samples)
    n_train = max(int(round(0.8 * num_samples)), 1)
    n_val = max(int(round(0.1 * num_samples)), 1)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    if len(test_idx) == 0:
        test_idx = val_idx
    return train_idx, val_idx, test_idx


def build_dataset_splits(config, split_seed):
    """Load + scale the FOM tensor and return seeded (train, val, test) datasets.

    Derived dimensions (``num_channels``, ``num_time``, ``num_samples``) and the
    field ``normalization`` dict are stashed on the returned datasets and mirrored
    into ``config`` (the SDFFlow convention of writing derived shapes back onto
    ``config``). All three splits share one in-memory array.
    """
    data_ntc, ids = load_fom_from_hdf5(config)          # [N, T, C]
    norm = fit_minmax(data_ntc)
    scaled = apply_minmax(data_ntc, norm)               # [N, T, C]
    data_nct = np.ascontiguousarray(np.transpose(scaled, (0, 2, 1)))  # [N, C, T]

    num_samples = data_nct.shape[0]
    train_idx, val_idx, test_idx = _split_indices(num_samples, split_seed)

    def make(idx):
        ds = FomFieldDataset(data_nct, idx)
        ds.normalization = norm
        ds.sample_ids = ids
        return ds

    train_ds, val_ds, test_ds = make(train_idx), make(val_idx), make(test_idx)
    config['num_channels'] = train_ds.num_channels
    config['num_time'] = train_ds.num_time
    config['num_samples'] = num_samples
    print(f'Dataset: {num_samples} samples '
          f'(channels={train_ds.num_channels}, time={train_ds.num_time}) '
          f'-> train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}')
    return train_ds, val_ds, test_ds


def read_conditions_from_hdf5(config):
    """Read latent-conditioner inputs from the mesh HDF5's conditioning rows.

    Rows ``[field_start_row + num_var : ... + cond_var]`` of ``nodal_data`` hold
    input-only conditions (flight/boundary parameters) broadcast to every node
    -- the same convention the mesh methods consume via ``cond_var``. Because
    they are per-sample constants, one node is enough to recover them; node 0 at
    timestep 0 is read and the row is checked for the constancy that convention
    implies, so pointing ``cond_var`` at a genuinely spatial field fails loudly
    instead of silently training on one arbitrary node's value.

    Returns ``[num_samples, cond_var]`` float32, ordered by sorted sample ID --
    the same order ``load_fom_from_hdf5`` stacks samples in.
    """
    cond_var = int(config.get('cond_var', 0) or 0)
    if cond_var < 1:
        raise ValueError(
            "lc_data_type 'hdf5' needs cond_var >= 1: it names how many "
            "conditioning rows follow the field rows in nodal_data.")

    num_var = int(config.get('num_var', 1))
    field_start = int(config.get('field_start_row', 3))
    cond_start = field_start + num_var
    cond_stop = cond_start + cond_var

    rows = []
    with h5py.File(config['dataset_dir'], 'r') as h5:
        for sid in _sample_ids(h5):
            arr = h5[f'data/{sid}/nodal_data']
            if arr.shape[0] < cond_stop:
                raise ValueError(
                    f"Sample {sid}: nodal_data has {arr.shape[0]} feature rows but "
                    f"conditioning rows {cond_start}:{cond_stop} are required "
                    f"(field_start_row={field_start} + num_var={num_var} + cond_var={cond_var}).")
            block = np.asarray(arr[cond_start:cond_stop, 0, :], dtype=np.float32)  # [cond_var, N]
            spread = np.ptp(block, axis=1)
            scale = np.maximum(np.abs(block[:, 0]), 1.0)
            varying = np.nonzero(spread > 1e-4 * scale)[0]
            if varying.size:
                raise ValueError(
                    f"Sample {sid}: conditioning row(s) "
                    f"{[cond_start + int(i) for i in varying]} vary across nodes "
                    f"(max spread {float(spread.max()):.3g}). cond_var rows must be "
                    "per-sample constants broadcast to every node.")
            rows.append(block[:, 0])

    conditions = np.stack(rows, axis=0).astype(np.float32)  # [num_samples, cond_var]
    return conditions, int(conditions.shape[1])


def read_conditions(config):
    """Load latent-conditioner inputs.

    ``lc_data_type == 'csv'``  -> a headerless CSV of shape [num_samples, features].
    ``lc_data_type == 'image'``-> images under ``param_dir`` (edge-filtered, /255).
    ``lc_data_type == 'hdf5'`` -> the ``cond_var`` conditioning rows of the mesh
    HDF5 named by ``dataset_dir`` (no ``param_dir`` needed) -- see
    :func:`read_conditions_from_hdf5`.

    Returns ``(conditions, input_shape)`` where ``conditions`` is float32
    ``[num_samples, feat]`` and ``input_shape`` is the per-sample feature count.
    """
    data_type = str(config.get('lc_data_type', 'csv')).lower()

    if data_type == 'hdf5':
        return read_conditions_from_hdf5(config)

    param_dir = config['param_dir']

    if data_type == 'csv':
        import pandas as pd
        raw = pd.read_csv(param_dir, header=None).values.astype(np.float32)
        return raw, int(raw.shape[1])

    if data_type == 'image':
        import cv2
        import natsort
        ext = str(config.get('param_data_type', '.png'))
        files = natsort.natsorted(f for f in os.listdir(param_dir) if f.endswith(ext))
        if not files:
            raise ValueError(f"No '{ext}' images found under {param_dir}.")
        im_size = 256
        imgs = np.zeros((len(files), im_size, im_size), dtype=np.float32)
        for i, f in enumerate(files):
            im = cv2.imread(os.path.join(param_dir, f), 0)
            im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
            _, binary = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
            imgs[i] = binary
        flat = imgs.reshape(len(files), -1) / 255.0
        return flat.astype(np.float32), int(flat.shape[1])

    raise ValueError(f"lc_data_type must be 'csv', 'image' or 'hdf5'; got {data_type!r}.")
