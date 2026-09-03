import numpy as np
import pytest

from inference_profiles.reconstruct import _load_fields_with_checkpoint_normalization


def _write_mesh_hdf5(path):
    import h5py

    with h5py.File(path, 'w') as h5:
        data = h5.create_group('data')
        group = data.create_group('1')
        nodal = np.zeros((5, 1, 2), dtype=np.float32)
        nodal[3, 0] = [10.0, 20.0]
        nodal[4, 0] = [30.0, 50.0]
        group.create_dataset('nodal_data', data=nodal)
        group.create_dataset('mesh_edge', data=np.zeros((2, 1), dtype=np.int32))
    return str(path)


def test_held_out_fields_use_checkpoint_normalization(tmp_path):
    path = _write_mesh_hdf5(tmp_path / 'held_out.h5')
    config = {'dataset_dir': path, 'field_start_row': 3, 'num_var': 2}
    checkpoint_norm = {
        'field_min': np.asarray([0.0, 10.0, 20.0, 30.0], dtype=np.float32),
        'field_max': np.asarray([20.0, 30.0, 40.0, 70.0], dtype=np.float32),
        'feature_range': [-1.0, 1.0],
    }

    data_nct, sample_ids = _load_fields_with_checkpoint_normalization(
        config, checkpoint_norm)

    assert sample_ids == [1]
    assert data_nct.shape == (1, 4, 1)
    # Re-fitting one held-out sample would collapse every channel to the lower
    # feature-range bound (-1); checkpoint scaling places these values at 0.
    assert np.allclose(data_nct[0, :, 0], [0.0, 0.0, 0.0, 0.0])


def test_missing_checkpoint_normalization_is_rejected(tmp_path):
    path = _write_mesh_hdf5(tmp_path / 'held_out.h5')
    config = {'dataset_dir': path, 'field_start_row': 3, 'num_var': 2}
    with pytest.raises(ValueError, match='normalization is missing'):
        _load_fields_with_checkpoint_normalization(config, {})
