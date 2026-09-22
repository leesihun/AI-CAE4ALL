"""Regression: physical fields must not silently displace mesh coordinates."""
import h5py
import numpy as np
import pytest
import torch
from general_modules.state_geometry import deformed_positions, displacement_from_state
from general_modules.mesh_dataset import MeshGraphDataset
from general_modules.edge_features import compute_edge_attr
from training_profiles.setup import build_model_config


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_fixed_geometry_ignores_all_physical_fields(width):
    pos = np.arange(12, dtype=np.float32).reshape(4, 3)
    state = np.full((4, width), 1000, dtype=np.float32)
    np.testing.assert_array_equal(deformed_positions(pos, state, {'geometry_state_mode': 'fixed'}), pos)


def test_crack_mapping_and_torch_gradients():
    state = torch.tensor([[999., 0.2, 0.3]], requires_grad=True)
    cfg = {'geometry_state_mode': 'displacement', 'displacement_state_indices': [1, 2, -1]}
    result = displacement_from_state(state, cfg)
    torch.testing.assert_close(result, torch.tensor([[0.2, 0.3, 0.]]))
    result.sum().backward()
    torch.testing.assert_close(state.grad, torch.tensor([[0., 1., 1.]]))
    np.testing.assert_allclose(displacement_from_state(state.detach().numpy(), cfg), result.detach().numpy())


@pytest.mark.parametrize("indices", [[0, 1, 3], [0, 1], [-2, 0, 1]])
def test_invalid_state_mapping_rejected(indices):
    with pytest.raises(ValueError):
        displacement_from_state(np.zeros((4, 3)), {'displacement_state_indices': indices})


def test_stats_graph_and_checkpoint_share_geometry_rule(tmp_path):
    path = tmp_path / 'fields.h5'
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    edges = np.array([[0, 1, 2, 3], [1, 3, 0, 2]])
    with h5py.File(path, 'w') as f:
        for sid in range(10):
            arr = np.zeros((7, 3, 4), np.float32)
            arr[:3] = pos.T[:, None, :]
            arr[3:6] = np.arange(36).reshape(3, 3, 4) + sid
            arr[6] = 123  # condition, never displacement
            f.create_dataset(f'data/{sid}/nodal_data', data=arr)
            f.create_dataset(f'data/{sid}/mesh_edge', data=edges)
    cfg = dict(input_var=3, output_var=3, cond_var=1, edge_var=8,
               positional_features=0, use_node_types=False, use_world_edges=False,
               use_multiscale=False, use_parallel_stats=False, time_integration='ar_ot',
               geometry_state_mode='fixed', write_preprocessing=False)
    train, _, _ = MeshGraphDataset(str(path), cfg).split(.8, .1, .1, seed=42)
    graph = train[1]
    raw = graph.edge_attr.numpy() * train.edge_std + train.edge_mean
    expected = compute_edge_attr(pos, pos, graph.edge_index.numpy())
    np.testing.assert_allclose(raw, expected, atol=1e-6)
    assert build_model_config(cfg)['geometry_state_mode'] == 'fixed'
    train.write_preprocessing_to_hdf5(42)
    with h5py.File(path, 'r') as f:
        assert 'preprocessing' not in f
