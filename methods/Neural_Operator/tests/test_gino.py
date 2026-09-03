import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d


def _small_cfg(**extra):
    cfg = dict(
        gino_grid_resolution=[6, 6], gino_fno_modes=[2, 3],
        gino_fno_hidden_channels=12, gino_fno_layers=2,
        gino_in_radius=0.35, gino_out_radius=0.35, gino_kernel_hidden=16,
        gino_max_empty_input_fraction=0.05,
    )
    cfg.update(extra)
    return cfg


def _build(h5_path, **overrides):
    cfg = base_config_2d(h5_path, model='gino', **overrides)
    ds = MeshGraphDataset(h5_path, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    return model, data_spec, domain, train, val, test, cfg


def test_forward_shapes_static_2d(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(tiny_static_2d_h5, **_small_cfg())
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert pred.shape == target.shape == (batch.x.shape[0], cfg['output_var'])
    assert torch.isfinite(pred).all()


def test_two_graph_isolation_and_order(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(tiny_static_2d_h5, **_small_cfg())
    model.eval()
    loader = DataLoader(train, batch_size=1, shuffle=False)
    items = [b for i, b in zip(range(2), loader)]
    a, b = items[0], items[1]

    with torch.no_grad():
        pred_a_alone, _ = model(a, add_noise=False)
        pred_b_alone, _ = model(b, add_noise=False)

    combined = Batch.from_data_list([a.get_example(0), b.get_example(0)])
    with torch.no_grad():
        pred_combined, _ = model(combined, add_noise=False)

    na = a.x.shape[0]
    assert torch.allclose(pred_a_alone, pred_combined[:na], atol=1e-4)
    assert torch.allclose(pred_b_alone, pred_combined[na:], atol=1e-4)


def test_query_chunking_matches_full_decode(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(tiny_static_2d_h5, **_small_cfg())
    model.eval()
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        full_pred, _ = model(batch, add_noise=False)
        encoded = model.encode_operator(batch)
        n = batch.x.shape[0]
        mid = n // 2
        chunk1 = model.decode_queries(encoded, batch, 0, mid)
        chunk2 = model.decode_queries(encoded, batch, mid, n)
        chunked_pred = torch.cat([chunk1, chunk2], dim=0)
    assert torch.allclose(full_pred, chunked_pred, atol=1e-5)


def test_coverage_preflight_passes_on_generous_radius(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(tiny_static_2d_h5, **_small_cfg())
    loader = DataLoader(train, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    report = model.core.coverage_preflight(batch)
    assert 'reports' in report


def test_coverage_preflight_fails_on_too_small_radius(tiny_static_2d_h5):
    import pytest
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, **_small_cfg(gino_in_radius=0.001, gino_out_radius=0.001))
    loader = DataLoader(train, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    with pytest.raises(ValueError):
        model.core.coverage_preflight(batch)


def test_overfit_tiny_fixture(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, **_small_cfg(), learningr=0.01)
    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    last_loss = None
    for step in range(150):
        opt.zero_grad()
        pred, target = model(batch, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    assert last_loss < 0.2, f"failed to overfit tiny fixture, final loss={last_loss}"
