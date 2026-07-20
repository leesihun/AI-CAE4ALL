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
    displacement = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    deformed_pos = ref_pos + displacement
    fine_to_coarse = np.array([0, 0, 1, 1], dtype=np.int64)
    coarse_edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    seeds = np.array([0, 2], dtype=np.int64)

    dataset_path = tmp_path / "coarse_stats.h5"
    with h5py.File(dataset_path, "w") as handle:
        sample = handle.create_group("data/0")
        nodal_data = np.zeros((6, 1, len(ref_pos)), dtype=np.float32)
        nodal_data[:3, 0, :] = ref_pos.T
        nodal_data[3:6, 0, :] = displacement.T
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
