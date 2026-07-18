import numpy as np
import pytest
import torch

from model.adapters.radius_neighbors import (
    radius_neighbors_scipy, radius_neighbors_torch_cluster, HAS_TORCH_CLUSTER,
    neighbor_stats, min_reachable_radius,
)


def test_scipy_finds_expected_neighbors():
    sources = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 1.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    edge_index = radius_neighbors_scipy(queries, sources, r=0.2)
    assert edge_index.shape == (2, 2)  # source 0 and source 1 are within 0.2
    assert set(edge_index[1].tolist()) == {0, 1}
    assert set(edge_index[0].tolist()) == {0}


def test_scipy_empty_when_no_neighbors():
    sources = np.array([[5.0, 5.0]], dtype=np.float32)
    queries = np.array([[0.0, 0.0]], dtype=np.float32)
    edge_index = radius_neighbors_scipy(queries, sources, r=0.1)
    assert edge_index.shape == (2, 0)


@pytest.mark.skipif(not HAS_TORCH_CLUSTER, reason="torch_cluster not installed")
def test_torch_cluster_matches_scipy_on_random_fixture():
    rng = np.random.default_rng(3)
    for trial in range(5):
        sources = rng.uniform(0, 1, size=(30, 3)).astype(np.float32)
        queries = rng.uniform(0, 1, size=(20, 3)).astype(np.float32)
        r = 0.25
        ei_scipy = radius_neighbors_scipy(queries, sources, r)
        ei_tc = radius_neighbors_torch_cluster(
            torch.from_numpy(queries), torch.from_numpy(sources), r).numpy()
        set_scipy = set(map(tuple, ei_scipy.T.tolist()))
        set_tc = set(map(tuple, ei_tc.T.tolist()))
        assert set_scipy == set_tc, f"trial {trial}: backend mismatch"


def test_neighbor_stats_and_empty_fraction():
    edge_index = np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int64)
    stats = neighbor_stats(edge_index, num_queries=4)
    assert stats['min'] == 0
    assert stats['max'] == 2
    assert stats['empty_fraction'] == 0.5  # queries 2 and 3 have no edges


def test_min_reachable_radius_matches_half_cell_diagonal():
    r = min_reachable_radius((3, 3), dim=2)
    # cell size = 1/(3-1) = 0.5 per axis; diagonal = sqrt(0.5^2+0.5^2); half of that
    expected = np.sqrt(0.5 ** 2 + 0.5 ** 2) / 2.0
    assert abs(r - expected) < 1e-9
