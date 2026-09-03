"""Input-only conditioning rows (`cond_var`).

Contract under test (dataset/DATASET_FORMAT.md):

    nodal_data rows  [0:3]                              reference coordinates
                     [3 : 3+input_var]                  state: input AND output
                     [3+input_var : ... +cond_var]      conditions: input ONLY

    graph.x columns  [state | conditions | positional | node-type one-hot]

The property that matters is that conditions are read from disk in *both* the
static (T=1) and temporal (T>1) branches. The static branch zeros the state
block -- the model has to regress the field from nothing -- but zeroing the
conditions too would make known boundary/flight parameters invisible, which is
exactly the bug this key exists to fix.
"""
import os

import numpy as np
import pytest
import torch

from general_modules.mesh_dataset import MeshGraphDataset
from model.MeshGraphNets import MeshGraphNets

MODEL_CLS = MeshGraphNets

STATE_ROWS = 8
COND_ROWS = 6
NUM_NODES = 24
NUM_SAMPLES = 12


def _write_fixture(path, num_timesteps):
    """Mesh HDF5 whose state is a function of per-sample condition constants."""
    import h5py

    rng = np.random.default_rng(0)
    total_rows = 3 + STATE_ROWS + COND_ROWS
    src = np.arange(NUM_NODES, dtype=np.int32)
    mesh_edge = np.stack([src, (src + 1) % NUM_NODES], axis=0)
    theta = np.linspace(0, 2 * np.pi, NUM_NODES, endpoint=False)

    with h5py.File(path, 'w') as f:
        f.attrs['num_samples'] = NUM_SAMPLES
        f.attrs['num_features'] = total_rows
        f.attrs['num_timesteps'] = num_timesteps
        data = f.create_group('data')
        for sid in range(1, NUM_SAMPLES + 1):
            nodal = np.zeros((total_rows, num_timesteps, NUM_NODES), dtype=np.float32)
            nodal[0, :, :] = np.cos(theta)
            nodal[1, :, :] = np.sin(theta)
            cond = rng.normal(size=COND_ROWS).astype(np.float32) * 2.0 + 5.0
            for c in range(COND_ROWS):
                nodal[3 + STATE_ROWS + c, :, :] = cond[c]      # broadcast to every node
            for c in range(STATE_ROWS):
                base = cond[c % COND_ROWS] * np.cos(theta * (c + 1))
                # vary over time so temporal deltas are non-trivial
                for t in range(num_timesteps):
                    nodal[3 + c, t, :] = base * (1.0 + 0.3 * t)
            g = data.create_group(str(sid))
            g.create_dataset('nodal_data', data=nodal)
            g.create_dataset('mesh_edge', data=mesh_edge)
    return path


def _config(path, cond_var):
    return {
        'dataset_dir': path, 'input_var': STATE_ROWS, 'output_var': STATE_ROWS,
        'cond_var': cond_var, 'edge_var': 8, 'positional_features': 0,
        'use_node_types': False, 'use_world_edges': False, 'use_multiscale': False,
        'use_parallel_stats': False, 'latent_dim': 16, 'message_passing_num': 2,
        'std_noise': 0.0, 'time_integration': 'ar_ot',
    }


@pytest.fixture(scope='module')
def datasets(tmp_path_factory):
    root = tmp_path_factory.mktemp('cond')
    return {
        1: _write_fixture(str(root / 'static.h5'), 1),
        5: _write_fixture(str(root / 'temporal.h5'), 5),
    }


def _split(path, cond_var):
    train, _, _ = MeshGraphDataset(path, _config(path, cond_var)).split(
        0.8, 0.1, 0.1, seed=0)
    return train


@pytest.mark.parametrize('num_timesteps', [1, 5])
def test_conditions_appear_in_x_after_the_state_block(datasets, num_timesteps):
    train = _split(datasets[num_timesteps], COND_ROWS)
    graph = train[0]
    assert graph.x.shape[1] == STATE_ROWS + COND_ROWS
    assert graph.y.shape[1] == STATE_ROWS
    # conditions are per-sample constants, so every node carries the same value
    cond = graph.x[:, STATE_ROWS:].numpy()
    assert np.allclose(cond, cond[0])


@pytest.mark.parametrize('num_timesteps', [1, 5])
def test_conditions_get_real_statistics(datasets, num_timesteps):
    """The regression this guards: a conditioning column normalized against the
    zero-state statistics would divide by the 1e-8 variance floor."""
    train = _split(datasets[num_timesteps], COND_ROWS)
    assert np.all(train.node_std[STATE_ROWS:] > 1e-3)
    if num_timesteps == 1:
        assert np.all(train.node_std[:STATE_ROWS] == pytest.approx(1e-8))


def test_static_zeros_the_state_but_not_the_conditions(datasets):
    train = _split(datasets[1], COND_ROWS)
    graph = train[0]
    assert np.allclose(graph.x[:, :STATE_ROWS].numpy(), 0.0)
    assert not np.allclose(graph.x[:, STATE_ROWS:].numpy(), 0.0)


@pytest.mark.parametrize('num_timesteps', [1, 5])
def test_conditions_differ_between_samples(datasets, num_timesteps):
    train = _split(datasets[num_timesteps], COND_ROWS)
    # each sample yields (T - window) items, so step by that to cross a sample
    # boundary instead of landing on another window of the same sample
    stride = max(1, num_timesteps - train.rollout_window)
    assert not np.allclose(train[0].x[:, STATE_ROWS:].numpy(),
                           train[stride].x[:, STATE_ROWS:].numpy())
    if stride > 1:  # two windows of ONE sample share its conditions
        assert np.allclose(train[0].x[:, STATE_ROWS:].numpy(),
                           train[1].x[:, STATE_ROWS:].numpy())


@pytest.mark.parametrize('num_timesteps', [1, 5])
def test_cond_var_zero_is_bit_identical_to_the_old_behavior(datasets, num_timesteps):
    path = datasets[num_timesteps]
    with_cond = _split(path, COND_ROWS)[0]
    without = _split(path, 0)[0]
    assert without.x.shape[1] == STATE_ROWS
    assert np.array_equal(without.x.numpy(), with_cond.x[:, :STATE_ROWS].numpy())
    assert np.array_equal(without.y.numpy(), with_cond.y.numpy())


def test_model_widens_its_encoder_and_still_predicts_output_var(datasets):
    train = _split(datasets[1], COND_ROWS)
    config = _config(datasets[1], COND_ROWS)
    config['num_timesteps'] = 1
    model = MeshGraphNets(config, 'cpu')
    graph = train[0]
    predicted, target = model(graph, add_noise=False)
    assert predicted.shape == (graph.x.shape[0], STATE_ROWS)
    assert target.shape == (graph.x.shape[0], STATE_ROWS)


def test_conditions_reach_the_encoder_weights(datasets):
    """A dead input would still produce the right shapes; only a gradient proves
    the conditioning columns are actually wired into the network."""
    train = _split(datasets[1], COND_ROWS)
    config = _config(datasets[1], COND_ROWS)
    config['num_timesteps'] = 1
    model = MeshGraphNets(config, 'cpu')
    predicted, target = model(train[0], add_noise=False)
    torch.nn.functional.mse_loss(predicted, target).backward()
    grad = model.model.encoder.nb_encoder[0].weight.grad
    assert grad[:, STATE_ROWS:].abs().sum().item() > 0


def test_too_few_feature_rows_is_rejected(datasets):
    with pytest.raises(ValueError, match='cond_var'):
        MeshGraphDataset(datasets[1], _config(datasets[1], COND_ROWS + 4))


def test_negative_cond_var_is_rejected(datasets):
    with pytest.raises(ValueError, match='cond_var must be >= 0'):
        MeshGraphDataset(datasets[1], _config(datasets[1], -1))


def test_ar_rt_rollout_carries_conditions_unchanged(datasets):
    """Under AR-RT the state is re-derived on-device at every unrolled step from
    `graph.x[:, input_var:]`. Conditions are in that slice by construction, so
    they must survive the whole unroll -- if they were inside the state block
    instead, the rollout would "integrate" a Mach number."""
    from training_profiles.ar_rollout import RolloutContext, rollout_loss

    path = datasets[5]
    config = _config(path, COND_ROWS)
    config['time_integration'] = 'ar_rt'
    config['num_timesteps'] = 5

    train, _, _ = MeshGraphDataset(path, config).split(0.8, 0.1, 0.1, seed=0)
    config['_norm_stats'] = {
        'node_mean': train.node_mean, 'node_std': train.node_std,
        'delta_mean': train.delta_mean, 'delta_std': train.delta_std,
        'edge_mean': train.edge_mean, 'edge_std': train.edge_std,
        'world_edge_radius': None,
    }

    graph = train[0]
    # state0 is the STATE only; conditions ride along in x as static features
    assert graph.state0.shape[1] == STATE_ROWS
    conditions = graph.x[:, STATE_ROWS:].clone()

    ctx = RolloutContext(config, torch.device('cpu'))
    assert graph.x[:, ctx.input_var:].shape[1] == COND_ROWS

    model = MODEL_CLS(config, 'cpu')

    def loss_fn(predicted, target):
        per_element = torch.nn.functional.mse_loss(predicted, target, reduction='none')
        return per_element.mean(), per_element.sum(), per_element.numel()

    loss, _total, _count = rollout_loss(model, graph, ctx, loss_fn, training=False)
    assert torch.isfinite(loss)
    assert torch.equal(graph.x[:, STATE_ROWS:], conditions)
