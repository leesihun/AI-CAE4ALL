"""Shared pytest fixtures: tiny synthetic HDF5 datasets in the exact MGN schema
(IMPLEMENTATION_PLAN.md section 4.1), used instead of the real ex1/ex2 files
so the full test suite runs in seconds. Real-data smoke tests (Phase 8) use
ex1.h5/hex_dataset.h5 directly and live outside this fixture set.
"""

import h5py
import numpy as np
import pytest
from scipy.spatial import cKDTree


def _make_edges(pos: np.ndarray, k: int = 4) -> np.ndarray:
    """Deterministic k-NN mesh_edge [2, E] (one-way, u < v, no duplicates).

    Every node gets at least one edge, matching the guarantee real meshes
    provide (no isolated nodes).
    """
    n = pos.shape[0]
    k_eff = min(k + 1, n)  # +1 because the query includes the point itself
    tree = cKDTree(pos)
    _, idx = tree.query(pos, k=k_eff)
    edges = set()
    for i in range(n):
        for j in idx[i]:
            j = int(j)
            if j == i:
                continue
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))
    edges = sorted(edges)
    return np.array(edges, dtype=np.int64).T  # [2, E]


def _make_sample(rng: np.random.Generator, num_nodes: int, num_timesteps: int,
                  dim: int, with_node_types: bool) -> tuple:
    """Build one sample's nodal_data [F, T, N] and mesh_edge [2, E]."""
    if dim == 2:
        xy = rng.uniform(-1.0, 1.0, size=(num_nodes, 2)).astype(np.float32)
        pos = np.concatenate([xy, np.zeros((num_nodes, 1), dtype=np.float32)], axis=1)
    else:
        pos = rng.uniform(-1.0, 1.0, size=(num_nodes, 3)).astype(np.float32)

    edge_index = _make_edges(pos, k=4)

    num_features = 8 if with_node_types else 7
    data = np.zeros((num_features, num_timesteps, num_nodes), dtype=np.float32)
    data[0, :, :] = pos[:, 0][None, :]
    data[1, :, :] = pos[:, 1][None, :]
    data[2, :, :] = pos[:, 2][None, :]

    base_disp = 0.05 * np.stack([
        np.sin(pos[:, 0] * 3.0), np.cos(pos[:, 1] * 3.0), 0.02 * pos[:, 0] * pos[:, 1]
    ], axis=1).astype(np.float32)  # [N, 3]
    base_stress = (0.5 + 0.1 * np.linalg.norm(pos, axis=1)).astype(np.float32)  # [N]

    for t in range(num_timesteps):
        growth = float(t + 1) / num_timesteps
        noise = 0.001 * rng.standard_normal((num_nodes, 3)).astype(np.float32)
        data[3:6, t, :] = (base_disp * growth + noise).T
        data[6, t, :] = base_stress * growth + 0.001 * rng.standard_normal(num_nodes).astype(np.float32)

    if with_node_types:
        node_types = (np.arange(num_nodes) % 3).astype(np.float32)
        data[7, :, :] = node_types[None, :]

    return data, edge_index


def _write_dataset(path, num_samples: int, num_timesteps: int, dim: int,
                    node_count_range: tuple, seed: int, with_node_types: bool = True):
    rng = np.random.default_rng(seed)
    with h5py.File(path, 'w') as f:
        data_grp = f.create_group('data')
        for sid in range(num_samples):
            n = int(rng.integers(node_count_range[0], node_count_range[1] + 1))
            data, edge_index = _make_sample(rng, n, num_timesteps, dim, with_node_types)
            sample_grp = data_grp.create_group(str(sid))
            sample_grp.create_dataset('nodal_data', data=data)
            sample_grp.create_dataset('mesh_edge', data=edge_index)
    return str(path)


@pytest.fixture
def tiny_static_2d_h5(tmp_path):
    """10 samples, T=1 (static), planar (z==0), ragged node counts, node types."""
    path = tmp_path / "tiny_static_2d.h5"
    return _write_dataset(path, num_samples=10, num_timesteps=1, dim=2,
                          node_count_range=(20, 35), seed=0)


@pytest.fixture
def tiny_temporal_3d_h5(tmp_path):
    """8 samples, T=5 (temporal), genuinely 3D, ragged node counts, node types."""
    path = tmp_path / "tiny_temporal_3d.h5"
    return _write_dataset(path, num_samples=8, num_timesteps=5, dim=3,
                          node_count_range=(25, 40), seed=1)


@pytest.fixture
def tiny_static_2d_no_node_types_h5(tmp_path):
    """7-feature-row variant (no node-type row) to test the guard rule."""
    path = tmp_path / "tiny_static_2d_no_nt.h5"
    return _write_dataset(path, num_samples=6, num_timesteps=1, dim=2,
                          node_count_range=(15, 20), seed=2, with_node_types=False)


def base_config_2d(h5_path, model="deeponet", **overrides):
    cfg = {
        'model': model, 'mode': 'train', 'gpu_ids': 0, 'parallel_mode': 'ddp',
        'log_file_dir': 'test/train.log', 'modelpath': './test/model.pth',
        'dataset_dir': h5_path, 'infer_dataset': h5_path,
        'inference_output_dir': 'test/rollout', 'infer_timesteps': 1,
        'split_seed': 42,
        'input_var': 4, 'output_var': 4,
        'feature_loss_weights': [1.0, 1.0, 1.0, 1.0],
        'positional_features': 4, 'use_node_types': True,
        'coordinate_normalization': 'centered_isotropic', 'operator_dim': 'auto',
        'dimension_tolerance': 1e-4, 'grid_padding': 0.05,
        'out_of_bounds_policy': 'error',
        'sdf_source': 'none', 'sdf_sidecar': 'none',
        'global_condition_features': 'none', 'integration_weight_source': 'none',
        'training_epochs': 2, 'batch_size': 2, 'learningr': 0.001,
        'weight_decay': 0.0001, 'warmup_epochs': 1, 'num_workers': 0,
        'prefetch_factor': None, 'grad_accum_steps': 1, 'max_grad_norm': 3.0,
        'std_noise': 0.0, 'noise_gamma': 1, 'augment_geometry': False,
        'use_amp': False, 'use_checkpointing': False, 'use_ema': False,
        'ema_decay': 0.99, 'use_compile': False,
        'val_interval': 1, 'test_interval': 1, 'test_max_batches': 10,
        'test_batch_idx': [0], 'plot_feature_idx': -1,
        'display_trainset': False, 'display_testset': False,
        'checkpoint_interval': 0,
        'train_query_chunk_size': 0, 'infer_query_chunk_size': 0,
        'write_preprocessing': False,
        'use_world_edges': False, 'use_multiscale': False,
        'use_parallel_stats': False,
    }
    cfg.update(overrides)
    return cfg
