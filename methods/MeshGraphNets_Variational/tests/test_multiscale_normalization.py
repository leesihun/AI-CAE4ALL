"""Regression tests for multiscale edge normalization geometry."""

from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch_geometric.data import Batch


MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules.edge_features import compute_edge_attr  # noqa: E402
from general_modules.dataset_stats import finalize_moments  # noqa: E402
from general_modules.mesh_dataset import (  # noqa: E402
    MeshGraphDataset,
    _deformed_positions_from_state,
)
from general_modules.multiscale_helpers import (  # noqa: E402
    attach_coarse_levels_to_graph,
    coarse_level_positions,
)
from model.coarsening import MultiscaleData, compute_coarse_centroids  # noqa: E402


class _HierarchyReader:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy

    def get_hierarchy(self, sample_id, variant):
        assert sample_id == 0
        assert variant == 0
        return self.hierarchy


def _write_static_sample(path, ref_pos):
    # ex8-style static layout: xyz plus one nonzero scalar target. Row 3 must
    # never be interpreted (or broadcast) as xyz displacement.
    nodal = np.zeros((4, 1, ref_pos.shape[0]), dtype=np.float32)
    nodal[:3, 0, :] = ref_pos.T
    nodal[3, 0, :] = np.linspace(10.0, 60.0, ref_pos.shape[0], dtype=np.float32)
    mesh_edge = np.stack([
        np.arange(ref_pos.shape[0] - 1),
        np.arange(1, ref_pos.shape[0]),
    ]).astype(np.int64)

    with h5py.File(path, 'w') as handle:
        group = handle.create_group('data/0')
        group.create_dataset('nodal_data', data=nodal)
        group.create_dataset('mesh_edge', data=mesh_edge)


def test_seedmean_coarse_stats_match_forward_seed_anchors_at_every_level(tmp_path):
    ref_pos = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [11.0, 2.0, 0.0],
        [14.0, 8.0, 0.0],
        [20.0, 0.0, 0.0],
        [30.0, 5.0, 0.0],
    ], dtype=np.float32)
    hierarchy = [
        {
            'ftc': np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),
            'c_ei': np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
            'n_c': 3,
            'seeds': np.array([0, 2, 4], dtype=np.int64),
            'mode': 'seedmean',
        },
        {
            'ftc': np.array([0, 0, 1], dtype=np.int64),
            'c_ei': np.array([[0, 1], [1, 0]], dtype=np.int64),
            'n_c': 2,
            'seeds': np.array([0, 2], dtype=np.int64),
            'mode': 'seedmean',
        },
    ]

    dataset_path = tmp_path / 'seedmean_stats.h5'
    _write_static_sample(dataset_path, ref_pos)

    dataset = MeshGraphDataset.__new__(MeshGraphDataset)
    dataset.h5_file = str(dataset_path)
    dataset.sample_ids = [0]
    dataset.multiscale_levels = len(hierarchy)
    dataset.edge_dim = 8
    dataset.input_dim = 1
    dataset.num_timesteps = 1
    dataset.edge_mean = np.zeros(8, dtype=np.float32)
    dataset.edge_std = np.ones(8, dtype=np.float32)
    reader = _HierarchyReader(hierarchy)
    dataset._get_ms_reader = lambda: reader

    dataset._compute_coarse_edge_stats()

    # The forward attachment path receives identity normalization here, so its
    # attached edge features are the exact raw features whose moments the
    # preprocessing pass must have fitted.
    graph = {}
    zeros = [np.zeros(8, dtype=np.float32) for _ in hierarchy]
    ones = [np.ones(8, dtype=np.float32) for _ in hierarchy]
    attach_coarse_levels_to_graph(
        graph, hierarchy, ref_pos, ref_pos, zeros, ones,
    )

    centroid_length_means = []
    cur_pos = ref_pos
    for level, entry in enumerate(hierarchy):
        forward_raw = graph[f'coarse_edge_attr_{level}'].numpy()
        expected_mean, expected_std = finalize_moments(
            forward_raw.sum(axis=0, dtype=np.float64),
            np.square(forward_raw, dtype=np.float64).sum(axis=0),
            forward_raw.shape[0],
        )
        np.testing.assert_allclose(
            dataset.coarse_edge_means[level],
            expected_mean,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            dataset.coarse_edge_stds[level],
            expected_std,
            rtol=1e-6,
            atol=1e-6,
        )

        centroid_pos = compute_coarse_centroids(
            cur_pos, entry['ftc'], entry['n_c'],
        )
        centroid_edges = compute_edge_attr(
            centroid_pos, centroid_pos, entry['c_ei'],
        )
        centroid_length_means.append(centroid_edges[:, 7].mean())
        cur_pos = centroid_pos

    # Both levels are deliberately irregular, so the old centroid-based
    # statistics differ in the reference-length channel and this test cannot
    # pass accidentally with the pre-fix implementation.
    for level, centroid_mean in enumerate(centroid_length_means):
        assert not np.isclose(
            dataset.coarse_edge_means[level][7], centroid_mean,
        )


def test_seed_anchor_without_seed_indices_falls_back_to_centroid():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 1.0, 0.0],
    ], dtype=np.float32)
    fine_to_coarse = np.array([0, 0, 1], dtype=np.int64)

    actual = coarse_level_positions(
        positions, fine_to_coarse, 2, 'seedmean', seeds=None,
    )
    expected = compute_coarse_centroids(positions, fine_to_coarse, 2)

    np.testing.assert_allclose(actual, expected)


def test_deformed_positions_use_only_the_displacement_state_prefix():
    reference = np.zeros((2, 3), dtype=np.float32)
    state = np.array([
        [1.0, 2.0, 3.0, 100.0],
        [4.0, 5.0, 6.0, 200.0],
    ], dtype=np.float32)

    np.testing.assert_allclose(
        _deformed_positions_from_state(reference, state, input_dim=2),
        np.array([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        _deformed_positions_from_state(reference, state, input_dim=4),
        state[:, :3],
    )


def test_multiscale_batch_offsets_ar_rt_anchor_indices_per_level():
    graph_a = MultiscaleData(
        x=torch.zeros((4, 1)),
        num_coarse_0=torch.tensor([3]),
        num_coarse_1=torch.tensor([2]),
        coarse_anchor_idx_0=torch.tensor([0, 2, 3]),
        coarse_anchor_idx_1=torch.tensor([0, 2]),
    )
    graph_b = MultiscaleData(
        x=torch.zeros((5, 1)),
        num_coarse_0=torch.tensor([3]),
        num_coarse_1=torch.tensor([2]),
        coarse_anchor_idx_0=torch.tensor([0, 1, 4]),
        coarse_anchor_idx_1=torch.tensor([1, 2]),
    )

    batch = Batch.from_data_list([graph_a, graph_b])

    assert batch.coarse_anchor_idx_0.tolist() == [0, 2, 3, 4, 5, 8]
    assert batch.coarse_anchor_idx_1.tolist() == [0, 2, 4, 5]
