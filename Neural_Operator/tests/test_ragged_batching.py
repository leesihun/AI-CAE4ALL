"""Shared ragged-batching coverage across all four models (section 16's
"Common model: batch 1/>1, static/temporal" row). Per-model architecture
specifics (branch width invariance, grid shape invariance, chunk parity) are
covered in each model's own test file; this file only asserts that every
model tolerates a batch of graphs with very different node counts, for both
static and temporal data, without crashing and with finite output.
"""

import pytest
import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d

MODEL_EXTRAS = {
    'deeponet': {'deeponet_sensor_resolution': [8, 8]},
    'point_deeponet': {'point_sensor_count': 16},
    'fno': {'fno_grid_resolution': [8, 8], 'fno_modes': [3, 4], 'fno_hidden_channels': 16, 'fno_layers': 2},
    'gino': {'gino_grid_resolution': [6, 6], 'gino_fno_modes': [2, 3], 'gino_fno_hidden_channels': 12,
            'gino_fno_layers': 2, 'gino_in_radius': 0.35, 'gino_out_radius': 0.35},
}


@pytest.mark.parametrize('model_name', ['deeponet', 'point_deeponet', 'fno', 'gino'])
def test_ragged_batch_static(model_name, tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model=model_name, **MODEL_EXTRAS[model_name])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)

    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert pred.shape == target.shape == (batch.x.shape[0], cfg['output_var'])
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize('model_name', ['deeponet', 'point_deeponet', 'fno', 'gino'])
def test_ragged_batch_temporal(model_name, tiny_temporal_3d_h5):
    extras = dict(MODEL_EXTRAS[model_name])
    if model_name == 'deeponet':
        extras['deeponet_sensor_resolution'] = [8, 8, 8]
    elif model_name == 'fno':
        extras['fno_grid_resolution'] = [8, 8, 8]
        extras['fno_modes'] = [3, 3, 4]
    elif model_name == 'gino':
        extras['gino_grid_resolution'] = [6, 6, 6]
        extras['gino_fno_modes'] = [2, 2, 3]

    cfg = base_config_2d(tiny_temporal_3d_h5, model=model_name, **extras)
    ds = MeshGraphDataset(tiny_temporal_3d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=1)
    model, data_spec, domain = build_model(cfg, train)

    loader = DataLoader(train, batch_size=min(4, len(train)), shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert pred.shape == target.shape == (batch.x.shape[0], cfg['output_var'])
    assert torch.isfinite(pred).all()


@pytest.mark.parametrize('model_name', ['deeponet', 'point_deeponet', 'fno', 'gino'])
def test_batch_size_one_and_greater_than_one_agree_in_shape(model_name, tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model=model_name, **MODEL_EXTRAS[model_name])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)

    loader1 = DataLoader(train, batch_size=1, shuffle=False)
    b1 = next(iter(loader1))
    pred1, _ = model(b1, add_noise=False)
    assert pred1.shape[1] == cfg['output_var']

    loaderN = DataLoader(train, batch_size=len(train), shuffle=False)
    bN = next(iter(loaderN))
    predN, _ = model(bN, add_noise=False)
    assert predN.shape[1] == cfg['output_var']
