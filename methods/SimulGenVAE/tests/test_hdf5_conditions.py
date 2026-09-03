"""`lc_data_type hdf5`: latent-conditioner inputs read from the mesh HDF5.

The other methods consume the conditioning rows per node (`cond_var`); the
latent conditioner wants one parameter vector per sample instead. Because the
rows are per-sample constants broadcast to every node, one node recovers them
exactly -- and the loader enforces that constancy so pointing `cond_var` at a
genuinely spatial field fails loudly rather than training on an arbitrary node.
"""
import numpy as np
import pytest

from general_modules.fom_dataset import read_conditions, read_conditions_from_hdf5

NUM_VAR = 2
COND_VAR = 3
NUM_NODES = 16
NUM_TIME = 4
NUM_SAMPLES = 5


def _write(path, cond_values, spatial_conditions=False):
    import h5py

    total_rows = 3 + NUM_VAR + COND_VAR
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        for i, cond in enumerate(cond_values, start=1):
            nodal = np.zeros((total_rows, NUM_TIME, NUM_NODES), dtype=np.float32)
            nodal[:3] = np.random.default_rng(i).normal(size=(3, NUM_TIME, NUM_NODES))
            nodal[3:3 + NUM_VAR] = float(i)
            for c in range(COND_VAR):
                if spatial_conditions:
                    nodal[3 + NUM_VAR + c, :, :] = np.arange(NUM_NODES, dtype=np.float32)
                else:
                    nodal[3 + NUM_VAR + c, :, :] = cond[c]
            g = data.create_group(str(i))
            g.create_dataset('nodal_data', data=nodal)
            g.create_dataset('mesh_edge', data=np.zeros((2, 1), dtype=np.int32))
    return str(path)


def _config(path, **overrides):
    config = {'dataset_dir': path, 'lc_data_type': 'hdf5',
              'num_var': NUM_VAR, 'field_start_row': 3, 'cond_var': COND_VAR}
    config.update(overrides)
    return config


@pytest.fixture
def conditions():
    return np.arange(NUM_SAMPLES * COND_VAR, dtype=np.float32).reshape(NUM_SAMPLES, COND_VAR)


def test_reads_one_row_per_sample_in_sorted_id_order(tmp_path, conditions):
    path = _write(tmp_path / 'ds.h5', conditions)
    values, input_shape = read_conditions_from_hdf5(_config(path))
    assert values.shape == (NUM_SAMPLES, COND_VAR)
    assert input_shape == COND_VAR
    # sample ids are written 1..N in order, so rows must come back in that order
    assert np.allclose(values, conditions)


def test_dispatches_through_read_conditions(tmp_path, conditions):
    path = _write(tmp_path / 'ds.h5', conditions)
    values, input_shape = read_conditions(_config(path))
    assert values.shape == (NUM_SAMPLES, COND_VAR)
    assert input_shape == COND_VAR
    # and it never touches param_dir
    assert 'param_dir' not in _config(path)


def test_conditions_start_after_the_field_rows(tmp_path, conditions):
    """The block is [field_start_row + num_var : ... + cond_var]; reading it one
    row early would return the constant field rows instead."""
    path = _write(tmp_path / 'ds.h5', conditions)
    values, _ = read_conditions_from_hdf5(_config(path))
    field_values = np.arange(1, NUM_SAMPLES + 1, dtype=np.float32)
    assert not np.allclose(values[:, 0], field_values)


def test_spatially_varying_rows_are_rejected(tmp_path, conditions):
    path = _write(tmp_path / 'ds.h5', conditions, spatial_conditions=True)
    with pytest.raises(ValueError, match='vary across nodes'):
        read_conditions_from_hdf5(_config(path))


def test_cond_var_must_be_set(tmp_path, conditions):
    path = _write(tmp_path / 'ds.h5', conditions)
    with pytest.raises(ValueError, match='cond_var'):
        read_conditions_from_hdf5(_config(path, cond_var=0))


def test_too_few_feature_rows_is_rejected(tmp_path, conditions):
    path = _write(tmp_path / 'ds.h5', conditions)
    with pytest.raises(ValueError, match='conditioning rows'):
        read_conditions_from_hdf5(_config(path, cond_var=COND_VAR + 5))


def test_unknown_lc_data_type_names_all_three(tmp_path, conditions):
    path = _write(tmp_path / 'ds.h5', conditions)
    with pytest.raises(ValueError, match="'csv', 'image' or 'hdf5'"):
        read_conditions(_config(path, lc_data_type='parquet', param_dir='x'))
