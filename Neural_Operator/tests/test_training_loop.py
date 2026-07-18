import os
import socket

import pytest
import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d
from training_profiles.training_loop import train_epoch, validate_epoch, _as_list
from training_profiles.training_loop import test_model as run_test_model


def test_as_list_normalizes_bare_scalar():
    # Single-value config lines parse to a bare int/float (section 11.1),
    # e.g. `test_batch_idx 0` -> 0, not [0]. Every membership check on such
    # a config value must normalize it first.
    assert _as_list(0) == [0]
    assert _as_list([0, 1, 2]) == [0, 1, 2]
    assert _as_list(None) is None


def test_test_model_tolerates_scalar_test_batch_idx(tiny_static_2d_h5):
    """Regression test: `test_batch_idx 0` in a real config file parses to
    the bare int 0 (not [0]); test_model must not crash on `batch_idx in
    config['test_batch_idx']` (caught by the ex1 smoke run, 2026-07-17)."""
    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet',
                         deeponet_sensor_resolution=[8, 8], test_batch_idx=0)
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    device = torch.device('cpu')

    test_loader = DataLoader(test, batch_size=1, shuffle=False)
    loss = run_test_model(model, test_loader, device, cfg, epoch=0, dataset=train)
    assert torch.isfinite(torch.tensor(loss))


def test_train_and_validate_epoch_run_and_produce_finite_metrics(tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet',
                         deeponet_sensor_resolution=[8, 8], learningr=0.005,
                         batch_size=2, use_amp=False)
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    device = torch.device('cpu')

    train_loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val, batch_size=cfg['batch_size'], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learningr'], fused=False)

    metrics0 = train_epoch(model, train_loader, optimizer, device, cfg, epoch=0)
    assert 'mean' in metrics0 and 'sum' in metrics0 and 'count' in metrics0
    assert metrics0['count'] > 0
    assert torch.isfinite(torch.tensor(metrics0['mean']))

    val_metrics = validate_epoch(model, val_loader, device, cfg, epoch=0)
    assert torch.isfinite(torch.tensor(val_metrics['mean']))

    # A few more epochs should meaningfully reduce the training loss on this
    # tiny fixture (sanity: gradients actually flow through the full loop).
    losses = [metrics0['mean']]
    for epoch in range(1, 8):
        m = train_epoch(model, train_loader, optimizer, device, cfg, epoch=epoch)
        losses.append(m['mean'])
    assert losses[-1] < losses[0]


def test_train_epoch_runs_under_ddp_wrapper(tiny_static_2d_h5):
    """Regression test: DistributedDataParallel does not delegate custom
    methods, so `model.set_epoch(epoch)` on the DDP wrapper raised
    AttributeError and multi-GPU training crashed on epoch 0 (2026-07-17).
    Exercises the real DDP path with a single-process gloo group on CPU."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_available():
        pytest.skip("torch.distributed is not available in this build")

    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet',
                         deeponet_sensor_resolution=[8, 8], batch_size=2,
                         use_amp=False)
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    device = torch.device('cpu')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group(backend='gloo', rank=0, world_size=1)
    try:
        ddp_model = DDP(model)
        train_loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=False)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3, fused=False)
        metrics = train_epoch(ddp_model, train_loader, optimizer, device, cfg, epoch=0)
        assert torch.isfinite(torch.tensor(metrics['mean']))
    finally:
        dist.destroy_process_group()
        os.environ.pop('MASTER_ADDR', None)
        os.environ.pop('MASTER_PORT', None)


def test_grad_accum_produces_same_batch_count_semantics(tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet',
                         deeponet_sensor_resolution=[8, 8], batch_size=2,
                         grad_accum_steps=2, use_amp=False)
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    device = torch.device('cpu')

    train_loader = DataLoader(train, batch_size=cfg['batch_size'], shuffle=False)
    total_nodes = sum(int(g.x.shape[0]) for g in train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=False)
    metrics = train_epoch(model, train_loader, optimizer, device, cfg, epoch=0)
    assert metrics['count'] == total_nodes  # every node in the train split contributes once
