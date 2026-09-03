import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d


def _build(h5_path, **overrides):
    cfg = base_config_2d(h5_path, model='deeponet', deeponet_sensor_resolution=[8, 8], **overrides)
    ds = MeshGraphDataset(h5_path, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    return model, train, cfg


def test_no_noise_when_std_zero(tiny_static_2d_h5):
    model, train, cfg = _build(tiny_static_2d_h5, std_noise=0.0)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    x_before = batch.x.clone()
    model.train()
    model(batch, add_noise=True)
    assert torch.equal(batch.x, x_before)  # std_noise=0 -> no perturbation despite add_noise=True


def test_noise_perturbs_physical_columns_only(tiny_static_2d_h5):
    model, train, cfg = _build(tiny_static_2d_h5, std_noise=0.5)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    output_var = cfg['output_var']
    x_before = batch.x.clone()
    torch.manual_seed(0)
    model.train()
    model(batch, add_noise=True)
    # physical columns changed
    assert not torch.equal(batch.x[:, :output_var], x_before[:, :output_var])
    # everything after the physical block (positional features + one-hot) unchanged
    assert torch.equal(batch.x[:, output_var:], x_before[:, output_var:])


def test_add_noise_none_follows_training_mode(tiny_static_2d_h5):
    model, train, cfg = _build(tiny_static_2d_h5, std_noise=0.5)
    loader = DataLoader(train, batch_size=2, shuffle=False)

    batch_eval = next(iter(loader))
    x_before = batch_eval.x.clone()
    model.eval()
    model(batch_eval, add_noise=None)
    assert torch.equal(batch_eval.x, x_before)  # eval mode -> no noise by default

    batch_train = next(iter(loader))
    x_before2 = batch_train.x.clone()
    model.train()
    model(batch_train, add_noise=None)
    assert not torch.equal(batch_train.x, x_before2)  # train mode -> noise applied by default


def test_target_correction_applied_when_ratio_configured(tiny_static_2d_h5):
    output_var = 4
    noise_std_ratio = [1.0, 1.0, 1.0, 1.0]
    model, train, cfg = _build(tiny_static_2d_h5, std_noise=0.3, noise_gamma=1,
                               noise_std_ratio=noise_std_ratio)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    y_before = batch.y.clone()
    model.train()
    model(batch, add_noise=True)
    assert not torch.equal(batch.y, y_before)


def test_eval_mode_no_noise_via_validate_style_call(tiny_static_2d_h5):
    model, train, cfg = _build(tiny_static_2d_h5, std_noise=1.0)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    x_before = batch.x.clone()
    model.eval()
    pred, target = model(batch, add_noise=False)
    assert torch.equal(batch.x, x_before)
