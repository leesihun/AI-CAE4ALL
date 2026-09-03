"""EMA + BatchNorm buffer copy (section 5.2 item 6 / 12.4). `AveragedModel`
only averages *parameters*; Point-DeepONet's PointNet branch has BatchNorm
running-mean/var *buffers* that would otherwise stay frozen at their
initial values (0 / 1) forever. `update_ema` must copy them from the live
model after every parameter update.
"""

import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d
from training_profiles.training_loop import build_ema_model, update_ema


def test_ema_batchnorm_buffers_track_live_model(tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model='point_deeponet',
                         point_sensor_count=16, use_ema=True, ema_decay=0.9)
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)

    ema_model = build_ema_model(model, cfg)
    assert ema_model is not None

    # EMA buffers start at PyTorch's BatchNorm defaults (mean=0, var=1).
    live_bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
    ema_bn = next(m for m in ema_model.module.modules() if isinstance(m, torch.nn.BatchNorm1d))
    assert torch.allclose(ema_bn.running_mean, torch.zeros_like(ema_bn.running_mean))

    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for _ in range(5):
        opt.zero_grad()
        pred, target = model(batch, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        update_ema(ema_model, model)

    # After several forward passes in train mode, the live BatchNorm's
    # running stats must have moved away from the init defaults, and the
    # EMA copy must match them exactly (copied, not averaged).
    assert not torch.allclose(live_bn.running_mean, torch.zeros_like(live_bn.running_mean))
    assert torch.equal(live_bn.running_mean, ema_bn.running_mean)
    assert torch.equal(live_bn.running_var, ema_bn.running_var)


def test_ema_disabled_when_use_ema_false():
    from model.mlp import build_mlp
    model = build_mlp(4, 8, 4)
    ema = build_ema_model(model, {'use_ema': False})
    assert ema is None
