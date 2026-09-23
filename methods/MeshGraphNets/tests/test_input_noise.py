"""Training noise must reach the edge features through the geometry, not beside it.

arXiv:2010.03409 adds the noise to the dynamical variable and *then* builds the
graph, so every edge attribute agrees with the perturbed node state. The bug
these tests pin: an independent Gaussian used to be added to all eight
normalized edge channels, which perturbed the exact reference geometry and left
the deformed channels uncorrelated with the node draw.
"""
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from general_modules.edge_features import DEFORMED_FEATURE_DIM, compute_edge_attr
from general_modules.input_noise import InputNoise

POS = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 1.]], np.float32)
EDGES = np.array([[0, 1, 2, 3, 1, 0], [1, 0, 3, 2, 2, 3]])
NODE_STD = np.array([2.0, 0.5, 4.0, 10.0], np.float32)    # 3 displacement + 1 scalar
NODE_MEAN = np.zeros(4, np.float32)
DELTA_STD = np.array([0.1, 0.05, 0.2, 1.0], np.float32)


def _stats(state):
    edge_attr = compute_edge_attr(POS, POS + state[:, :3], EDGES)
    return {
        'node_mean': NODE_MEAN, 'node_std': NODE_STD,
        'edge_mean': edge_attr.mean(0), 'edge_std': np.maximum(edge_attr.std(0), 1e-8),
        'delta_mean': np.zeros(4, np.float32), 'delta_std': DELTA_STD,
    }


def _config(mode='displacement', std_noise=0.05):
    state = np.array([[0., 0., 0., 1.], [0.1, 0., 0., 2.],
                      [0., 0.2, 0., 3.], [0.05, 0.05, 0.3, 4.]], np.float32)
    stats = _stats(state)
    cfg = {
        'input_var': 4, 'output_var': 4, 'std_noise': std_noise,
        'geometry_state_mode': mode, 'displacement_state_indices': [0, 1, 2],
        '_norm_stats': stats,
        'noise_std_ratio': (NODE_STD / DELTA_STD).tolist(),
    }
    return cfg, state, stats


def _graph(cfg, state, stats):
    edge_attr = compute_edge_attr(POS, POS + state[:, :3], EDGES)
    return Data(
        x=torch.from_numpy((state - stats['node_mean']) / stats['node_std']),
        y=torch.zeros(4, 4),
        pos=torch.from_numpy(POS),
        edge_index=torch.from_numpy(EDGES).long(),
        edge_attr=torch.from_numpy((edge_attr - stats['edge_mean']) / stats['edge_std']),
    )


def test_reference_half_is_never_perturbed():
    cfg, state, stats = _config()
    graph = _graph(cfg, state, stats)
    before = graph.edge_attr.clone()
    torch.manual_seed(0)
    InputNoise(cfg)(graph)
    torch.testing.assert_close(
        graph.edge_attr[:, DEFORMED_FEATURE_DIM:], before[:, DEFORMED_FEATURE_DIM:],
        rtol=0, atol=0,
    )
    assert not torch.allclose(graph.edge_attr[:, :DEFORMED_FEATURE_DIM],
                              before[:, :DEFORMED_FEATURE_DIM])


def test_edge_features_match_a_graph_rebuilt_from_the_perturbed_state():
    """The only correctness statement that matters: perturbing the state and
    rebuilding the graph must give the same edge features as this fast path."""
    cfg, state, stats = _config()
    graph = _graph(cfg, state, stats)
    x_before = graph.x.clone()
    torch.manual_seed(7)
    InputNoise(cfg)(graph)

    noisy_state = graph.x.numpy() * stats['node_std'] + stats['node_mean']
    expected = compute_edge_attr(POS, POS + noisy_state[:, :3], EDGES)
    expected = (expected - stats['edge_mean']) / stats['edge_std']
    torch.testing.assert_close(graph.edge_attr, torch.from_numpy(expected),
                               rtol=1e-5, atol=1e-5)
    assert not torch.allclose(graph.x, x_before)


def test_fixed_geometry_leaves_every_edge_feature_alone():
    """CylinderFlow-style: the noise is on momentum, the mesh does not move."""
    cfg, state, stats = _config(mode='fixed')
    graph = _graph(cfg, state, stats)
    before = graph.edge_attr.clone()
    torch.manual_seed(0)
    InputNoise(cfg)(graph)
    torch.testing.assert_close(graph.edge_attr, before, rtol=0, atol=0)


def test_target_correction_is_the_node_draw_in_delta_units():
    cfg, state, stats = _config()
    graph = _graph(cfg, state, stats)
    x_before = graph.x.clone()
    torch.manual_seed(3)
    InputNoise(cfg)(graph)
    drawn = graph.x - x_before
    torch.testing.assert_close(
        graph.y, -drawn * torch.from_numpy(NODE_STD / DELTA_STD), rtol=1e-6, atol=1e-6,
    )


def test_world_edges_follow_the_same_geometry():
    cfg, state, stats = _config()
    graph = _graph(cfg, state, stats)
    world_index = torch.tensor([[0, 3], [3, 0]])
    world = compute_edge_attr(POS, POS + state[:, :3], world_index.numpy())
    graph.world_edge_index = world_index
    graph.world_edge_attr = torch.from_numpy((world - stats['edge_mean']) / stats['edge_std'])
    torch.manual_seed(11)
    InputNoise(cfg)(graph)
    noisy_state = graph.x.numpy() * stats['node_std'] + stats['node_mean']
    expected = compute_edge_attr(POS, POS + noisy_state[:, :3], world_index.numpy())
    expected = (expected - stats['edge_mean']) / stats['edge_std']
    torch.testing.assert_close(graph.world_edge_attr, torch.from_numpy(expected),
                               rtol=1e-5, atol=1e-5)


def test_zero_std_noise_is_a_no_op_and_needs_no_stats():
    cfg, state, stats = _config(std_noise=0.0)
    cfg.pop('_norm_stats')
    graph = _graph(cfg, state, stats)
    before_x, before_e = graph.x.clone(), graph.edge_attr.clone()
    InputNoise(cfg)(graph)
    torch.testing.assert_close(graph.x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(graph.edge_attr, before_e, rtol=0, atol=0)


def test_model_forward_routes_noise_through_the_shared_injector():
    """End-to-end wiring: `MeshGraphNets.forward(add_noise=True)` must use it."""
    from model.MeshGraphNets import MeshGraphNets

    cfg, state, stats = _config()
    cfg.update(edge_var=8, latent_dim=16, message_passing_num=2, cond_var=0,
               positional_features=0, use_node_types=False, use_world_edges=False,
               use_multiscale=False, use_checkpointing=False, num_timesteps=5)
    model = MeshGraphNets(cfg, 'cpu')
    graph = _graph(cfg, state, stats)
    before_x, before_e = graph.x.clone(), graph.edge_attr.clone()

    model(graph, add_noise=False)
    torch.testing.assert_close(graph.x, before_x, rtol=0, atol=0)
    torch.testing.assert_close(graph.edge_attr, before_e, rtol=0, atol=0)

    model(graph, add_noise=True)
    assert not torch.allclose(graph.x, before_x)
    torch.testing.assert_close(graph.edge_attr[:, DEFORMED_FEATURE_DIM:],
                               before_e[:, DEFORMED_FEATURE_DIM:], rtol=0, atol=0)
    assert not torch.allclose(graph.edge_attr[:, :DEFORMED_FEATURE_DIM],
                              before_e[:, :DEFORMED_FEATURE_DIM])


def test_missing_stats_raises_instead_of_silently_skipping_the_geometry():
    cfg, state, stats = _config()
    cfg.pop('_norm_stats')
    graph = _graph(cfg, state, stats)
    with pytest.raises(RuntimeError, match='_norm_stats'):
        InputNoise(cfg)(graph)
