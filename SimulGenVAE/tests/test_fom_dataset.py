"""Unit tests for the HDF5-native FOM loader and checkpoint round-trip."""

import numpy as np
import pytest

from general_modules.fom_dataset import (
    apply_minmax,
    build_dataset_splits,
    fit_minmax,
    invert_minmax,
    load_fom_from_hdf5,
)


def _cfg(path, **overrides):
    cfg = {'dataset_dir': path, 'num_var': 1, 'field_start_row': 3}
    cfg.update(overrides)
    return cfg


def test_assembly_shape_and_dims(mesh_h5):
    data, ids = load_fom_from_hdf5(_cfg(mesh_h5))
    # 6 samples, T=4, num_var=1 -> channels = 1 * N(10) = 10
    assert data.shape == (6, 4, 10)
    assert ids == [0, 1, 2, 3, 4, 5]


def test_num_var_multi_channel(mesh_h5):
    # rows 3,4 -> num_var=2 -> channels = 2 * 10 = 20
    data, _ = load_fom_from_hdf5(_cfg(mesh_h5, num_var=2))
    assert data.shape == (6, 4, 20)


def test_feature_row_bound_error(mesh_h5):
    # F=5, field_start_row=3, num_var=3 -> needs 6 rows -> error
    with pytest.raises(ValueError, match='feature rows'):
        load_fom_from_hdf5(_cfg(mesh_h5, num_var=3))


def test_non_uniform_geometry_error(write_mesh_h5):
    path = write_mesh_h5(vary_node_sample=2)
    with pytest.raises(ValueError, match='fixed geometry'):
        load_fom_from_hdf5(_cfg(path))


def test_node_and_time_reduction(mesh_h5):
    data, _ = load_fom_from_hdf5(_cfg(mesh_h5, node_start=2, node_end=7, timesteps_reduced=3))
    # nodes 2:7 -> 5, T truncated to 3, num_var=1 -> channels 5
    assert data.shape == (6, 3, 5)


def test_build_dataset_splits(mesh_h5):
    cfg = _cfg(mesh_h5, split_seed=42)
    train, val, test = build_dataset_splits(cfg, split_seed=42)
    assert train.num_channels == 10 and train.num_time == 4
    assert cfg['num_channels'] == 10 and cfg['num_time'] == 4 and cfg['num_samples'] == 6
    assert len(train) + len(val) + len(test) >= 6  # test may reuse val when tiny
    item, idx = train[0]
    assert item.shape == (10, 4)  # [num_channels, num_time]


def test_minmax_roundtrip(mesh_h5):
    data, _ = load_fom_from_hdf5(_cfg(mesh_h5))
    norm = fit_minmax(data)
    scaled = apply_minmax(data, norm)
    assert scaled.min() >= -0.7001 and scaled.max() <= 0.7001
    recovered = invert_minmax(scaled, norm)
    assert np.allclose(recovered, data, atol=1e-4)


def test_checkpoint_roundtrip(tmp_path):
    import torch

    from training_profiles.setup import load_checkpoint, save_checkpoint

    payload = {'stage': 'vae', 'epoch': 3, 'model_state': {'w': torch.zeros(2, 2)},
               'config': {'model': 'simulgenvae'}, 'normalization': {'field_min': np.zeros(4)}}
    path = str(tmp_path / 'vae.pth')
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path, torch.device('cpu'))
    assert loaded['stage'] == 'vae' and loaded['epoch'] == 3
    assert loaded['config']['model'] == 'simulgenvae'
