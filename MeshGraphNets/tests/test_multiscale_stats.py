from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules.dataset_stats import finalize_moments  # noqa: E402
from general_modules.edge_features import compute_edge_attr  # noqa: E402
from general_modules import mesh_dataset as mesh_dataset_module  # noqa: E402
from general_modules.multiscale_helpers import attach_coarse_levels_to_graph  # noqa: E402
from model.coarsening import compute_coarse_centroids  # noqa: E402


@pytest.mark.parametrize("mode", ["seedmean", "centroid"])
def test_coarse_edge_stats_use_mode_consistent_positions(tmp_path, monkeypatch, mode):
    ref_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    stored_scalar = np.array([0.0, 0.2, 1.0, 3.0], dtype=np.float32)
    # Static forward starts from a zero state even though the target is stored
    # in row 3. In an ex8-style four-row file, treating row 3 as displacement
    # would broadcast that scalar over all XYZ components.
    deformed_pos = ref_pos.copy()
    fine_to_coarse = np.array([0, 0, 1, 1], dtype=np.int64)
    coarse_edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    seeds = np.array([0, 2], dtype=np.int64)

    dataset_path = tmp_path / "coarse_stats.h5"
    with h5py.File(dataset_path, "w") as handle:
        sample = handle.create_group("data/0")
        nodal_data = np.zeros((4, 1, len(ref_pos)), dtype=np.float32)
        nodal_data[:3, 0, :] = ref_pos.T
        nodal_data[3, 0, :] = stored_scalar
        sample.create_dataset("nodal_data", data=nodal_data)
        sample.create_dataset(
            "mesh_edge", data=np.array([[0, 1], [1, 2]], dtype=np.int64)
        )

    hierarchy = [
        {
            "ftc": fine_to_coarse,
            "c_ei": coarse_edge_index,
            "n_c": 2,
            "seeds": seeds,
            "mode": mode,
        }
    ]
    monkeypatch.setattr(
        mesh_dataset_module,
        "build_multiscale_hierarchy",
        lambda *args, **kwargs: hierarchy,
    )

    dataset = mesh_dataset_module.MeshGraphDataset.__new__(
        mesh_dataset_module.MeshGraphDataset
    )
    dataset.h5_file = str(dataset_path)
    dataset.sample_ids = [0]
    dataset.multiscale_levels = 1
    dataset.input_dim = 1
    dataset.coarsening_types = [mode]
    dataset.voronoi_clusters = [2]
    dataset.edge_dim = 8
    dataset.num_timesteps = 1
    dataset.edge_mean = np.zeros(8, dtype=np.float32)
    dataset.edge_std = np.ones(8, dtype=np.float32)

    dataset._compute_coarse_edge_stats()

    if mode == "seedmean":
        expected_ref = ref_pos[seeds]
        expected_deformed = deformed_pos[seeds]
        alternate_ref = compute_coarse_centroids(ref_pos, fine_to_coarse, 2)
        alternate_deformed = compute_coarse_centroids(
            deformed_pos, fine_to_coarse, 2
        )
    else:
        expected_ref = compute_coarse_centroids(ref_pos, fine_to_coarse, 2)
        expected_deformed = compute_coarse_centroids(
            deformed_pos, fine_to_coarse, 2
        )
        alternate_ref = ref_pos[seeds]
        alternate_deformed = deformed_pos[seeds]

    expected_features = compute_edge_attr(
        expected_ref.astype(np.float32),
        expected_deformed.astype(np.float32),
        coarse_edge_index,
    )
    expected_mean, expected_std = finalize_moments(
        expected_features.sum(axis=0, dtype=np.float64),
        np.square(expected_features, dtype=np.float64).sum(axis=0),
        expected_features.shape[0],
    )
    alternate_features = compute_edge_attr(
        alternate_ref.astype(np.float32),
        alternate_deformed.astype(np.float32),
        coarse_edge_index,
    )
    alternate_mean = alternate_features.mean(axis=0)

    np.testing.assert_allclose(dataset.coarse_edge_means[0], expected_mean)
    np.testing.assert_allclose(dataset.coarse_edge_stds[0], expected_std)
    assert not np.allclose(dataset.coarse_edge_means[0], alternate_mean)

    graph = {}
    attach_coarse_levels_to_graph(
        graph,
        hierarchy,
        ref_pos,
        deformed_pos,
        dataset.coarse_edge_means,
        dataset.coarse_edge_stds,
    )
    expected_normalized = (expected_features - expected_mean) / expected_std
    np.testing.assert_allclose(
        graph["coarse_edge_attr_0"].numpy(),
        expected_normalized.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("input_dim", [1, 2, 4])
def test_temporal_coarse_stats_match_forward_displacement_contract(
    tmp_path, monkeypatch, input_dim,
):
    ref_pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.5, 0.0],
            [10.0, 1.0, 0.0],
            [14.0, 2.0, 0.5],
        ],
        dtype=np.float32,
    )
    all_states = np.array(
        [
            [
                [0.0, 0.0, 0.0, 100.0],
                [0.2, 0.4, 0.1, 200.0],
                [1.0, 0.3, 0.2, 300.0],
                [3.0, 0.7, 0.4, 400.0],
            ],
            [
                [0.1, 0.2, 0.3, 500.0],
                [0.4, 0.1, 0.2, 600.0],
                [1.5, 0.6, 0.1, 700.0],
                [2.5, 0.9, 0.5, 800.0],
            ],
        ],
        dtype=np.float32,
    )
    states = all_states[:, :, :input_dim]
    fine_to_coarse = np.array([0, 0, 1, 1], dtype=np.int64)
    coarse_edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    seeds = np.array([0, 2], dtype=np.int64)
    hierarchy = [
        {
            "ftc": fine_to_coarse,
            "c_ei": coarse_edge_index,
            "n_c": 2,
            "seeds": seeds,
            "mode": "seedmean",
        }
    ]

    dataset_path = tmp_path / f"temporal_coarse_stats_{input_dim}.h5"
    with h5py.File(dataset_path, "w") as handle:
        sample = handle.create_group("data/0")
        nodal_data = np.zeros((3 + input_dim, 2, len(ref_pos)), dtype=np.float32)
        nodal_data[:3, :, :] = ref_pos.T[:, None, :]
        nodal_data[3:, :, :] = states.transpose(2, 0, 1)
        sample.create_dataset("nodal_data", data=nodal_data)
        sample.create_dataset(
            "mesh_edge", data=np.array([[0, 1], [1, 2]], dtype=np.int64)
        )

    monkeypatch.setattr(
        mesh_dataset_module,
        "build_multiscale_hierarchy",
        lambda *args, **kwargs: hierarchy,
    )
    dataset = mesh_dataset_module.MeshGraphDataset.__new__(
        mesh_dataset_module.MeshGraphDataset
    )
    dataset.h5_file = str(dataset_path)
    dataset.sample_ids = [0]
    dataset.multiscale_levels = 1
    dataset.input_dim = input_dim
    dataset.coarsening_types = ["voronoi_seedmean"]
    dataset.voronoi_clusters = [2]
    dataset.edge_dim = 8
    dataset.num_timesteps = 2
    dataset.edge_mean = np.zeros(8, dtype=np.float32)
    dataset.edge_std = np.ones(8, dtype=np.float32)

    dataset._compute_coarse_edge_stats()

    expected_per_timestep = []
    fine_deformed_positions = []
    for state in states:
        displacement = np.zeros_like(ref_pos)
        displacement[:, :min(3, input_dim)] = state[:, :min(3, input_dim)]
        fine_deformed = ref_pos + displacement
        fine_deformed_positions.append(fine_deformed)
        expected_per_timestep.append(
            compute_edge_attr(
                ref_pos[seeds], fine_deformed[seeds], coarse_edge_index,
            )
        )
    expected_features = np.concatenate(expected_per_timestep, axis=0)
    expected_sum = np.zeros(8, dtype=np.float64)
    expected_sumsq = np.zeros(8, dtype=np.float64)
    for features in expected_per_timestep:
        # Match the production accumulator: edge features and their squares
        # are float32 before each timestep's reduction into float64 totals.
        expected_sum += np.sum(features, axis=0)
        expected_sumsq += np.sum(features ** 2, axis=0)
    expected_mean, expected_std = finalize_moments(
        expected_sum,
        expected_sumsq,
        expected_features.shape[0],
    )
    np.testing.assert_allclose(dataset.coarse_edge_means[0], expected_mean)
    np.testing.assert_allclose(dataset.coarse_edge_stds[0], expected_std)

    graph = {}
    attach_coarse_levels_to_graph(
        graph,
        hierarchy,
        ref_pos,
        fine_deformed_positions[0],
        dataset.coarse_edge_means,
        dataset.coarse_edge_stds,
    )
    expected_normalized = (
        expected_per_timestep[0] - expected_mean
    ) / expected_std
    np.testing.assert_allclose(
        graph["coarse_edge_attr_0"].numpy(),
        expected_normalized.astype(np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
