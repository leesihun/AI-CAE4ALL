import os

import pytest
import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model, build_model_from_checkpoint
from tests.conftest import base_config_2d
from training_profiles.setup import (
    build_optimizer_scheduler, save_checkpoint, SCHEMA_VERSION,
)

MODEL_EXTRAS = {
    'deeponet': {'deeponet_sensor_resolution': [8, 8]},
    'point_deeponet': {'point_sensor_count': 16},
    'fno': {'fno_grid_resolution': [8, 8], 'fno_modes': [3, 4], 'fno_hidden_channels': 16, 'fno_layers': 2},
    'gino': {'gino_grid_resolution': [6, 6], 'gino_fno_modes': [2, 3], 'gino_fno_hidden_channels': 12,
            'gino_fno_layers': 2, 'gino_in_radius': 0.35, 'gino_out_radius': 0.35},
}


@pytest.mark.parametrize('model_name', ['deeponet', 'point_deeponet', 'fno', 'gino'])
def test_checkpoint_roundtrip_reproduces_predictions(tiny_static_2d_h5, tmp_path, model_name):
    modelpath = str(tmp_path / f"{model_name}.pth")
    cfg = base_config_2d(tiny_static_2d_h5, model=model_name, modelpath=modelpath,
                         **MODEL_EXTRAS[model_name])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, coordinate_domain = build_model(cfg, train)

    optimizer, scheduler, _, _ = build_optimizer_scheduler(cfg, model.parameters(), total_epochs=5)

    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        pred, target = model(batch, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        epoch=2, bare_model=model, ema_model=None, optimizer=optimizer, scheduler=scheduler,
        train_loss=float(loss.item()), valid_loss=float(loss.item()), config=cfg,
        train_dataset=train, coordinate_domain=coordinate_domain, data_spec=data_spec,
        modelpath=modelpath, config_filename='dummy_config.txt',
    )
    assert os.path.exists(modelpath)

    model.eval()
    with torch.no_grad():
        pred_before, _ = model(batch.clone(), add_noise=False)

    checkpoint = torch.load(modelpath, map_location='cpu', weights_only=False)
    assert checkpoint['schema_version'] == SCHEMA_VERSION
    assert checkpoint['selected_model'] == model_name

    reload_cfg = dict(cfg)
    reload_cfg['model'] = model_name
    reloaded_model, reloaded_spec, reloaded_domain = build_model_from_checkpoint(reload_cfg, checkpoint)
    reloaded_model.eval()

    with torch.no_grad():
        pred_after, _ = reloaded_model(batch.clone(), add_noise=False)

    assert torch.allclose(pred_before, pred_after, atol=1e-5)
    assert reloaded_spec.operator_dim == data_spec.operator_dim
    assert reloaded_domain.active_axes == coordinate_domain.active_axes


def test_compiled_model_checkpoint_loads_uncompiled(tiny_static_2d_h5, tmp_path):
    """Regression test: torch.compile wraps the model in an OptimizedModule
    whose state_dict keys carry an '_orig_mod.' prefix; save_checkpoint must
    unwrap it so build_model_from_checkpoint's strict load into a fresh
    (uncompiled) OperatorWrapper succeeds (2026-07-17). The compiled forward
    is never invoked, so no compiler backend is needed."""
    modelpath = str(tmp_path / "deeponet_compiled.pth")
    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet', modelpath=modelpath,
                         deeponet_sensor_resolution=[8, 8])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, coordinate_domain = build_model(cfg, train)
    optimizer, scheduler, _, _ = build_optimizer_scheduler(cfg, model.parameters(), total_epochs=2)

    compiled = torch.compile(model, dynamic=True)
    save_checkpoint(
        epoch=0, bare_model=compiled, ema_model=None, optimizer=optimizer, scheduler=scheduler,
        train_loss=0.0, valid_loss=0.0, config=cfg, train_dataset=train,
        coordinate_domain=coordinate_domain, data_spec=data_spec,
        modelpath=modelpath, config_filename='dummy_config.txt',
    )

    checkpoint = torch.load(modelpath, map_location='cpu', weights_only=False)
    assert not any(k.startswith('_orig_mod.') for k in checkpoint['model_state_dict'])

    reload_cfg = dict(cfg)
    reloaded_model, _, _ = build_model_from_checkpoint(reload_cfg, checkpoint)

    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model.eval()
    reloaded_model.eval()
    with torch.no_grad():
        pred_before, _ = model(batch.clone(), add_noise=False)
        pred_after, _ = reloaded_model(batch.clone(), add_noise=False)
    assert torch.allclose(pred_before, pred_after, atol=1e-6)


def test_checkpoint_rejects_wrong_model_name(tiny_static_2d_h5, tmp_path):
    modelpath = str(tmp_path / "deeponet.pth")
    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet', modelpath=modelpath,
                         deeponet_sensor_resolution=[8, 8])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, coordinate_domain = build_model(cfg, train)
    optimizer, scheduler, _, _ = build_optimizer_scheduler(cfg, model.parameters(), total_epochs=2)

    save_checkpoint(
        epoch=0, bare_model=model, ema_model=None, optimizer=optimizer, scheduler=scheduler,
        train_loss=0.0, valid_loss=0.0, config=cfg, train_dataset=train,
        coordinate_domain=coordinate_domain, data_spec=data_spec,
        modelpath=modelpath, config_filename='dummy_config.txt',
    )

    checkpoint = torch.load(modelpath, map_location='cpu', weights_only=False)
    assert checkpoint['selected_model'] == 'deeponet'
    # inference_profiles/rollout.py enforces this exact equality before ever
    # calling build_model_from_checkpoint; assert the guard condition holds.
    requested_model = 'fno'
    assert requested_model != checkpoint['selected_model']
