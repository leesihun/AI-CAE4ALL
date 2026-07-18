import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d


def _build(h5_path, **overrides):
    cfg = base_config_2d(h5_path, model='deeponet', **overrides)
    ds = MeshGraphDataset(h5_path, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    return model, data_spec, domain, train, val, test, cfg


def test_forward_shapes_static_2d(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, deeponet_sensor_resolution=[8, 8])
    loader = DataLoader(train, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert pred.shape == target.shape == (batch.x.shape[0], cfg['output_var'])
    assert torch.isfinite(pred).all()


def test_ragged_batch_produces_identical_branch_width(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, deeponet_sensor_resolution=[8, 8])
    loader = DataLoader(train, batch_size=1, shuffle=False)
    branch_widths = set()
    for i, batch in enumerate(loader):
        if i >= 3:
            break
        branch = model.core._branch_context(batch)
        branch_widths.add(branch.shape[-1] * branch.shape[-2])
    assert len(branch_widths) == 1  # same fixed width regardless of node count


def test_graph_isolation_no_cross_leakage(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, deeponet_sensor_resolution=[8, 8])
    model.eval()
    loader = DataLoader(train, batch_size=1, shuffle=False)
    items = [b for i, b in zip(range(2), loader)]
    a, b = items[0], items[1]

    with torch.no_grad():
        pred_a_alone, _ = model(a, add_noise=False)

    from torch_geometric.data import Batch
    b_scaled = b.get_example(0)
    b_scaled.x = b_scaled.x * 1000.0
    combined = Batch.from_data_list([a.get_example(0), b_scaled])
    with torch.no_grad():
        pred_combined, _ = model(combined, add_noise=False)
    pred_a_in_combined = pred_combined[:a.x.shape[0]]

    assert torch.allclose(pred_a_alone, pred_a_in_combined, atol=1e-4)


def test_query_chunking_matches_full_decode(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, deeponet_sensor_resolution=[8, 8])
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
    assert torch.allclose(full_pred, chunked_pred, atol=1e-6)


def test_overfit_tiny_fixture(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, deeponet_sensor_resolution=[8, 8],
        learningr=0.01, deeponet_hidden_channels=64, deeponet_basis_dim=32,
    )
    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    last_loss = None
    for step in range(300):
        opt.zero_grad()
        pred, target = model(batch, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    assert last_loss < 0.05, f"failed to overfit tiny fixture, final loss={last_loss}"
