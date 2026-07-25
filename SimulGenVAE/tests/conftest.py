"""Shared fixtures for SimulGenVAE tests.

Builds tiny synthetic datasets on the shared mesh HDF5 contract
(``data/{id}/{nodal_data[F,T,N], mesh_edge[2,E]}``) so the HDF5-native FOM
loader can be exercised without any real dataset.
"""

import os
import sys

import numpy as np
import pytest

# Make the repo root importable (general_modules, model, training_profiles).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def _write_mesh_h5(path, num_samples=6, F=5, T=4, N=10, vary_node_sample=None):
    """Write a synthetic mesh HDF5. If ``vary_node_sample`` is set, that sample id
    gets a different node count (to exercise the fixed-geometry check)."""
    import h5py

    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        f.attrs['num_samples'] = num_samples
        data = f.create_group('data')
        for sid in range(num_samples):
            n = N
            if vary_node_sample is not None and sid == vary_node_sample:
                n = N + 3
            grp = data.create_group(str(sid))
            grp.create_dataset('nodal_data', data=rng.standard_normal((F, T, n)).astype(np.float32))
            edges = np.stack([np.arange(n), (np.arange(n) + 1) % n]).astype(np.int64)
            grp.create_dataset('mesh_edge', data=edges)
    return path


@pytest.fixture
def mesh_h5(tmp_path):
    return _write_mesh_h5(str(tmp_path / 'fom.h5'))


@pytest.fixture
def write_mesh_h5(tmp_path):
    """Factory: call with kwargs to build a customized fixture under tmp_path."""
    def _factory(name='fom.h5', **kwargs):
        return _write_mesh_h5(str(tmp_path / name), **kwargs)
    return _factory
