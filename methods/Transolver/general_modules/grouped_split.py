"""Leakage-safe grouped partitioning for repeated loads on one geometry."""
import h5py
import numpy as np


def grouped_split_ids(h5_file, sample_ids, attribute, train_ratio, val_ratio, seed):
    groups = {}
    with h5py.File(h5_file, 'r') as f:
        for sid in sample_ids:
            meta = f[f'data/{sid}'].get('metadata')
            if meta is None or attribute not in meta.attrs:
                raise ValueError(f'Sample {sid} lacks metadata attribute {attribute!r} for grouped split')
            key = str(meta.attrs[attribute])
            groups.setdefault(key, []).append(sid)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError('Grouped train/val/test split requires at least three distinct groups')
    np.random.default_rng(seed).shuffle(keys)
    n_train = min(max(int(len(keys) * train_ratio), 1), len(keys) - 2)
    n_val = min(max(int(len(keys) * val_ratio), 1), len(keys) - n_train - 1)
    partitions = (keys[:n_train], keys[n_train:n_train+n_val], keys[n_train+n_val:])
    return tuple([sid for key in part for sid in groups[key]] for part in partitions)
