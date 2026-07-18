"""Shared exact-chunk-decode parity across all four models (section 14.3 /
16's "Inference: cache/chunk parity" row). Each model's own test file already
covers this individually; this file adds the `inference_profiles.query_decode`
helper on top and checks 3-way splits (not just a single midpoint split).
"""

import pytest
import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from inference_profiles.query_decode import decode_in_chunks
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
def test_decode_in_chunks_matches_full_decode(model_name, tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model=model_name, **MODEL_EXTRAS[model_name])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    model.eval()

    loader = DataLoader(train, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    n = batch.x.shape[0]

    with torch.no_grad():
        full_pred, _ = model(batch, add_noise=False)
        encoded = model.encode_operator(batch)

        # fp32 matmul kernels can pick slightly different reduction orders
        # for different batch/chunk shapes; tolerance is relative + absolute
        # rather than a single tight atol (section 16: "fp32 normally <=1e-5,
        # justified per test" -- Point-DeepONet's deeper composition
        # (SIREN -> two refiner MLPs -> einsum) accumulates more rounding
        # than DeepONet's single dot product, measured up to ~1e-4 here).
        for chunk_size in [0, max(1, n // 5), n]:
            chunked = decode_in_chunks(model, encoded, batch, chunk_size)
            assert torch.allclose(full_pred, chunked, atol=1e-3, rtol=1e-4), (
                f"{model_name} chunk_size={chunk_size} mismatch: "
                f"max abs diff={ (full_pred - chunked).abs().max().item() }"
            )


@pytest.mark.parametrize('model_name', ['deeponet', 'point_deeponet', 'fno', 'gino'])
def test_supports_query_chunking_flag(model_name, tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model=model_name, **MODEL_EXTRAS[model_name])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, domain = build_model(cfg, train)
    assert model.supports_query_chunking() is True
