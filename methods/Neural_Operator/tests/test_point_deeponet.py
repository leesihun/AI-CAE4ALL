import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from tests.conftest import base_config_2d


def _build(h5_path, **overrides):
    cfg = base_config_2d(h5_path, model='point_deeponet', **overrides)
    ds = MeshGraphDataset(h5_path, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    return model, data_spec, domain, train, val, test, cfg


def test_forward_shapes_static_2d(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=16)
    loader = DataLoader(train, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert pred.shape == target.shape == (batch.x.shape[0], cfg['output_var'])
    assert torch.isfinite(pred).all()


def test_all_points_ablation_forward(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=0)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert pred.shape == target.shape
    assert torch.isfinite(pred).all()


def test_sensor_count_with_replacement_when_fewer_nodes_than_sensors(tiny_static_2d_h5):
    # tiny fixture has 20-35 nodes per sample; force M > any sample's node count.
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=1000)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    pred, target = model(batch, add_noise=False)
    assert torch.isfinite(pred).all()


def test_graph_isolation_no_cross_leakage(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=16)
    model.eval()
    loader = DataLoader(train, batch_size=1, shuffle=False)
    items = [b for i, b in zip(range(2), loader)]
    a, b = items[0], items[1]

    with torch.no_grad():
        pred_a_alone, _ = model(a, add_noise=False)

    b_scaled = b.get_example(0)
    b_scaled.x = b_scaled.x * 1000.0
    combined = Batch.from_data_list([a.get_example(0), b_scaled])
    with torch.no_grad():
        pred_combined, _ = model(combined, add_noise=False)
    pred_a_in_combined = pred_combined[:a.x.shape[0]]

    assert torch.allclose(pred_a_alone, pred_a_in_combined, atol=1e-4)


def test_eval_sampling_is_deterministic(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=16)
    model.eval()
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        pred1, _ = model(batch.clone(), add_noise=False)
        pred2, _ = model(batch.clone(), add_noise=False)
    assert torch.allclose(pred1, pred2, atol=1e-6)


def test_query_chunking_matches_full_decode(tiny_static_2d_h5):
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=16)
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


def test_parity_shapes_match_paper_mapping(tiny_static_2d_h5):
    """docs/POINT_DEEPONET_PARITY.md: trunk_refiner reshapes to
    [chunk_N, H, output_var] (paper's asymmetric refiner), and branch_context
    is [num_graphs, H] -- not normal DeepONet's symmetric [B, O, P] shape."""
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=16, point_feature_dim=24)
    loader = DataLoader(train, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    core = model.core
    branch = core._branch_context(batch)
    assert branch.shape == (2, core.H)

    query = core._query_features(batch)
    t0 = core.trunk(query)
    fused = branch[batch.batch] * t0
    t_beta = core.trunk_refiner(fused)
    assert t_beta.shape == (batch.x.shape[0], core.H * core.output_var)
    assert t_beta.view(-1, core.H, core.output_var).shape == (batch.x.shape[0], core.H, core.output_var)


def test_overfit_tiny_fixture(tiny_static_2d_h5):
    # point_resample_each_epoch=False: the overfit check verifies gradients
    # actually flow and the architecture can fit data; per-epoch sensor
    # resampling (the default for real training) intentionally prevents
    # exact memorization of a fixed batch and is tested separately
    # (test_training_resamples_across_epochs in test_point_sampling.py).
    # lr=0.01 (used by the other three models' overfit tests) overshoots on
    # this bilinear branch*trunk architecture; 0.003 converges smoothly.
    model, data_spec, domain, train, val, test, cfg = _build(
        tiny_static_2d_h5, point_sensor_count=16, point_hidden_channels=32,
        point_feature_dim=32, learningr=0.003, point_resample_each_epoch=False,
    )
    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    model.train()
    last_loss = None
    for step in range(2000):
        model.set_epoch(step)
        opt.zero_grad()
        pred, target = model(batch, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        opt.step()
        last_loss = loss.item()
    assert last_loss < 0.01, f"failed to overfit tiny fixture, final loss={last_loss}"
