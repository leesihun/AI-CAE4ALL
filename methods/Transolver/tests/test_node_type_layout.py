"""partNo is the trailing row, not a hard-coded feature index."""
import h5py
import numpy as np
import pytest
from general_modules.mesh_dataset import MeshGraphDataset


@pytest.mark.parametrize("state,conditions", [(1, 0), (3, 0), (4, 2)])
@pytest.mark.parametrize("has_type", [True, False])
def test_node_type_row_follows_field_contract(tmp_path, state, conditions, has_type):
    path = tmp_path / 'types.h5'
    rows = 3 + state + conditions + int(has_type)
    with h5py.File(path, 'w') as f:
        for sid in range(3):
            a = np.zeros((rows, 2, 4), np.float32)
            a[0] = np.arange(4)
            a[1] = [0, 1, 0, 1]
            if has_type:
                a[-1] = [0, 0, 1, 1]
            f.create_dataset(f'data/{sid}/nodal_data', data=a)
            f.create_dataset(f'data/{sid}/mesh_edge', data=np.array([[0, 1, 2], [1, 2, 3]]))
    cfg = dict(input_var=state, output_var=state, cond_var=conditions,
               positional_features=0, use_node_types=True, time_integration='ar_ot',
               use_multiscale=False, use_world_edges=False, augment_geometry=False,
               coordinate_normalization='centered_isotropic', operator_dim='auto')
    if has_type:
        ds = MeshGraphDataset(str(path), cfg)
        assert ds.num_features == rows
    else:
        with pytest.raises(ValueError, match='partNo'):
            MeshGraphDataset(str(path), cfg)
