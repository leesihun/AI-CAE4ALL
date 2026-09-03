"""Input-only conditioning rows (`cond_var`) across all four architectures.

Contract under test (dataset/DATASET_FORMAT.md):

    nodal_data rows  [0:3]                              reference coordinates
                     [3 : 3+input_var]                  state: input AND output
                     [3+input_var : ... +cond_var]      conditions: input ONLY

    graph.x columns  [state | conditions | positional | node-type one-hot]

`DataSpec` is the single source of truth for those widths, so the checks below
assert the spec and the actual tensor agree -- a model sized from a stale spec
is the failure mode that would otherwise only surface as a shape error deep in
a core's forward.
"""
import numpy as np
import pytest
import torch

from general_modules.data_spec import DataSpec, build_data_spec_from_dataset
from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model

STATE_ROWS = 4
COND_ROWS = 3
NUM_NODES = 24
NUM_SAMPLES = 12

# Minimal per-architecture knobs; the point is the input width, not capacity.
MODEL_EXTRAS = {
    'point_deeponet': {},
    'deeponet': {'deeponet_branch_source': 'fixed_sensors',
                 'deeponet_sensor_resolution': [6, 6]},
    'fno': {'fno_grid_resolution': [8, 8], 'fno_modes': [3, 3],
            'fno_hidden_channels': 8, 'fno_layers': 2},
    'gino': {'gino_variant': 'mesh_state', 'gino_grid_resolution': [8, 8],
             'gino_fno_modes': [3, 3]},
}


def _write_fixture(path, num_timesteps):
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
        'cond_var': cond_var, 'positional_features': 0, 'use_node_types': False,
        'use_parallel_stats': False, 'time_integration': 'ar_ot',
        'global_condition_features': 'none', 'sdf_source': 'none',
        'latent_dim': 16, 'num_layers': 2, 'std_noise': 0.0, 'operator_dim': 'auto',
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
    graph = _split(datasets[1], COND_ROWS)[0]
    assert np.allclose(graph.x[:, :STATE_ROWS].numpy(), 0.0)
    assert not np.allclose(graph.x[:, STATE_ROWS:].numpy(), 0.0)


@pytest.mark.parametrize('num_timesteps', [1, 5])
def test_cond_var_zero_is_bit_identical_to_the_old_behavior(datasets, num_timesteps):
    path = datasets[num_timesteps]
    with_cond = _split(path, COND_ROWS)[0]
    without = _split(path, 0)[0]
    assert without.x.shape[1] == STATE_ROWS
    assert np.array_equal(without.x.numpy(), with_cond.x[:, :STATE_ROWS].numpy())
    assert np.array_equal(without.y.numpy(), with_cond.y.numpy())


def test_data_spec_slices_match_the_tensor(datasets):
    train = _split(datasets[1], COND_ROWS)
    spec = build_data_spec_from_dataset(train, _config(datasets[1], COND_ROWS))
    assert spec.condition_dim == COND_ROWS
    assert spec.total_node_dim == train[0].x.shape[1]
    assert spec.physical_slice == slice(0, STATE_ROWS)
    assert spec.condition_slice == slice(STATE_ROWS, STATE_ROWS + COND_ROWS)
    assert spec.positional_slice.start == STATE_ROWS + COND_ROWS


def test_data_spec_round_trips_through_a_checkpoint_dict(datasets):
    train = _split(datasets[1], COND_ROWS)
    spec = build_data_spec_from_dataset(train, _config(datasets[1], COND_ROWS))
    assert DataSpec.from_dict(spec.to_dict()) == spec


def test_pre_conditioning_checkpoints_still_load():
    """condition_dim is absent from specs written before cond_var existed."""
    payload = {
        'input_var': 4, 'output_var': 4, 'positional_dim': 0, 'node_type_dim': 0,
        'global_condition_dim': 0, 'operator_dim': 2, 'active_axes': [0, 1],
        'has_sdf': False, 'has_integration_weights': False, 'num_timesteps': 1,
    }
    spec = DataSpec.from_dict(payload)
    assert spec.condition_dim == 0
    assert spec.total_node_dim == 4


@pytest.mark.parametrize('model_name', sorted(MODEL_EXTRAS))
def test_every_architecture_builds_and_runs_at_the_wider_input(datasets, model_name):
    train = _split(datasets[1], COND_ROWS)
    config = _config(datasets[1], COND_ROWS)
    config['model'] = model_name
    config.update(MODEL_EXTRAS[model_name])

    model, spec, _domain = build_model(config, train)
    assert spec.total_node_dim == STATE_ROWS + COND_ROWS

    graph = train[0]
    graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)
    graph.ptr = torch.tensor([0, graph.x.shape[0]])
    with torch.no_grad():
        predicted, _ = model(graph, add_noise=False)
    assert predicted.shape == (graph.x.shape[0], STATE_ROWS)


def test_too_few_feature_rows_is_rejected(datasets):
    with pytest.raises(ValueError, match='cond_var'):
        MeshGraphDataset(datasets[1], _config(datasets[1], COND_ROWS + 4))


def test_negative_cond_var_is_rejected(datasets):
    with pytest.raises(ValueError, match='cond_var must be >= 0'):
        MeshGraphDataset(datasets[1], _config(datasets[1], -1))
