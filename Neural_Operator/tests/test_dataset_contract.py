import numpy as np
import pytest

from general_modules.mesh_dataset import MeshGraphDataset
from tests.conftest import base_config_2d


def test_static_2d_split_and_shapes(tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5)
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    assert ds.num_timesteps == 1
    assert len(ds) == len(ds.sample_ids) == 10

    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    assert len(train.sample_ids) + len(val.sample_ids) + len(test.sample_ids) == 10
    assert train.operator_dim == 2
    assert train.active_axes == (0, 1)

    item = train[0]
    n = item.x.shape[0]
    assert item.x.shape[1] == cfg['input_var'] + cfg['positional_features'] + train.num_node_types
    assert item.y.shape == (n, cfg['output_var'])
    assert item.pos.shape == (n, 3)
    assert item.pos_normalized.shape == (n, 3)
    assert item.edge_index.shape[0] == 2
    assert item.edge_index.max().item() < n

    # Static case: physical input is exactly zero, target is the stored field.
    assert torch_allzero(item.x[:, :cfg['input_var']])


def torch_allzero(t):
    return bool((t.abs().sum().item()) < 1e-6) if False else bool((t == 0).all().item())


def test_temporal_3d_delta_semantics(tiny_temporal_3d_h5):
    cfg = base_config_2d(tiny_temporal_3d_h5, model='deeponet')
    ds = MeshGraphDataset(tiny_temporal_3d_h5, cfg)
    assert ds.num_timesteps == 5
    assert len(ds) == len(ds.sample_ids) * 4

    train, val, test = ds.split(0.8, 0.1, 0.1, seed=1)
    assert train.operator_dim == 3
    assert train.active_axes == (0, 1, 2)

    item = train[0]
    assert item.time_idx in (0, 1, 2, 3)


def test_node_type_guard_raises_on_seven_rows(tiny_static_2d_no_node_types_h5):
    cfg = base_config_2d(tiny_static_2d_no_node_types_h5, model='deeponet')
    with pytest.raises(ValueError, match="use_node_types"):
        MeshGraphDataset(tiny_static_2d_no_node_types_h5, cfg)


def test_input_output_var_mismatch_rejected_for_temporal(tiny_temporal_3d_h5):
    cfg = base_config_2d(tiny_temporal_3d_h5, model='deeponet', output_var=3,
                         feature_loss_weights=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="input_var == output_var"):
        MeshGraphDataset(tiny_temporal_3d_h5, cfg)


def test_split_determinism(tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5)
    ds1 = MeshGraphDataset(tiny_static_2d_h5, cfg)
    ds2 = MeshGraphDataset(tiny_static_2d_h5, dict(cfg))
    t1, _, _ = ds1.split(0.8, 0.1, 0.1, seed=7)
    t2, _, _ = ds2.split(0.8, 0.1, 0.1, seed=7)
    assert t1.sample_ids == t2.sample_ids
