"""AR-OT / AR-RT time integration across all four operator architectures.

The rollout drives `OperatorWrapper`, so a single unroll has to work for
`deeponet`, `point_deeponet`, `fno` and `gino` without branching. These tests
pin that, plus the two invariants the scheme rests on: a one-step rollout must
reproduce AR-OT exactly, and AR-OT must remain the untouched default.
"""

import pytest
import torch

from general_modules.mesh_dataset import MeshGraphDataset
from general_modules.time_integration import (
    resolve_rollout_window,
    resolve_time_integration,
)
from model.factory import build_model
from tests.conftest import _write_dataset as write_dataset, base_config_2d
from training_profiles.ar_rollout import RolloutContext, rollout_loss

NUM_TIMESTEPS = 5

# Smallest architecture settings that build on the 3D temporal fixture.
MODEL_EXTRAS = {
    'deeponet': {'deeponet_sensor_resolution': [6, 6, 6]},
    'point_deeponet': {'point_sensor_count': 16},
    'fno': {'fno_grid_resolution': [6, 6, 6], 'fno_modes': [2, 2, 3],
            'fno_hidden_channels': 12, 'fno_layers': 2},
    'gino': {'gino_grid_resolution': [5, 5, 5], 'gino_fno_modes': [2, 2, 2],
             'gino_fno_hidden_channels': 12, 'gino_fno_layers': 2,
             'gino_in_radius': 0.35, 'gino_out_radius': 0.35},
}


def _temporal_config(h5_path, model="deeponet", **overrides):
    settings = dict(MODEL_EXTRAS[model])
    settings.update(overrides)
    return base_config_2d(h5_path, model=model, infer_timesteps=4, **settings)


def _prepared_dataset(h5_path, config):
    dataset = MeshGraphDataset(str(h5_path), config)
    train, _, _ = dataset.split(0.8, 0.1, 0.1, seed=42)
    return train


def _norm_stats(dataset):
    return {
        'node_mean': dataset.node_mean,
        'node_std': dataset.node_std,
        'delta_mean': dataset.delta_mean,
        'delta_std': dataset.delta_std,
    }


def _loss_fn(prediction, target):
    errors = torch.nn.functional.mse_loss(prediction, target, reduction='none')
    per_node = errors.mean(dim=-1)
    return per_node.mean(), per_node.sum().detach(), per_node.numel()


def test_ar_ot_remains_the_default(tiny_temporal_3d_h5):
    config = _temporal_config(tiny_temporal_3d_h5)
    assert resolve_time_integration(config) == 'ar_ot'
    assert resolve_rollout_window(config, NUM_TIMESTEPS) == 1

    dataset = _prepared_dataset(tiny_temporal_3d_h5, config)
    assert len(dataset) == len(dataset.sample_ids) * (NUM_TIMESTEPS - 1)
    assert getattr(dataset[0], 'y_seq', None) is None


def test_ar_rt_window_and_payload(tiny_temporal_3d_h5):
    config = _temporal_config(tiny_temporal_3d_h5, time_integration='AR-RT')
    dataset = _prepared_dataset(tiny_temporal_3d_h5, config)

    assert dataset.rollout_window == NUM_TIMESTEPS - 1
    assert len(dataset) == len(dataset.sample_ids)

    item = dataset[0]
    num_nodes = item.x.shape[0]
    assert item.y_seq.shape == (num_nodes, NUM_TIMESTEPS - 1, config['output_var'])
    assert item.state0.shape == (num_nodes, config['input_var'])


@pytest.mark.parametrize("model_name", ["deeponet", "point_deeponet", "fno", "gino"])
def test_rollout_trains_every_architecture(tiny_temporal_3d_h5, model_name):
    config = _temporal_config(
        tiny_temporal_3d_h5, model=model_name, time_integration='ar_rt',
    )
    dataset = _prepared_dataset(tiny_temporal_3d_h5, config)
    config['_norm_stats'] = _norm_stats(dataset)

    torch.manual_seed(0)
    model, _, _ = build_model(config, dataset)
    model.train()

    graph = dataset[0]
    ctx = RolloutContext(config, torch.device('cpu'))
    loss, _, loss_count = rollout_loss(model, graph, ctx, _loss_fn, training=True)
    loss.backward()

    # Every step of the trajectory is scored, not just the last.
    assert loss_count == (NUM_TIMESTEPS - 1) * graph.x.shape[0]

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, f"{model_name}: rollout produced no gradients"
    assert any(torch.any(g != 0) for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_single_step_rollout_matches_one_step_loss(tmp_path):
    """A one-step trajectory is AR-OT with extra bookkeeping; losses must agree."""
    # T=2 -> the full-trajectory rollout is a single step.
    path = write_dataset(tmp_path / "two_step.h5", num_samples=8, num_timesteps=2,
                         dim=3, node_count_range=(25, 40), seed=1)
    config = _temporal_config(path, time_integration='ar_rt')
    dataset = _prepared_dataset(path, config)
    config['_norm_stats'] = _norm_stats(dataset)

    torch.manual_seed(0)
    model, _, _ = build_model(config, dataset)
    model.eval()

    graph = dataset[0]
    ctx = RolloutContext(config, torch.device('cpu'))
    with torch.no_grad():
        rollout, _, _ = rollout_loss(model, graph, ctx, _loss_fn, training=False)
        prediction, target = model(graph, add_noise=False)
        one_step, _, _ = _loss_fn(prediction, target)

    torch.testing.assert_close(rollout, one_step, rtol=1e-5, atol=1e-6)
