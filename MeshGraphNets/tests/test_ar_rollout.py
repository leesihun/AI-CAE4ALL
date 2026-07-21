"""AR-OT / AR-RT time-integration contract.

The load-bearing claim of AR-RT is that the features it rebuilds from a
predicted state are the same features the dataloader (and therefore inference)
would build from that state. Most of this file tests that equivalence; the
rest pins the scheme's degenerate case (a one-step rollout must reproduce
AR-OT exactly) and its gradient behavior.
"""

from pathlib import Path
import sys

import h5py
import numpy as np
import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules.mesh_dataset import MeshGraphDataset  # noqa: E402
from general_modules.time_integration import (  # noqa: E402
    resolve_rollout_window,
    resolve_time_integration,
)
from training_profiles.ar_rollout import RolloutContext, _apply_state, rollout_loss  # noqa: E402


NUM_NODES = 12
NUM_TIMESTEPS = 6
INPUT_VAR = 4
OUTPUT_VAR = 4


def _write_dataset(path, num_samples=4, num_timesteps=None):
    """A tiny grid-graph trajectory in the repository's HDF5 layout."""
    num_timesteps = num_timesteps or NUM_TIMESTEPS
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        for sample_id in range(num_samples):
            ref_pos = rng.random((NUM_NODES, 3)).astype(np.float32) * 10.0
            nodal = np.zeros((7, num_timesteps, NUM_NODES), dtype=np.float32)
            nodal[:3, :, :] = ref_pos.T[:, None, :]
            for t in range(num_timesteps):
                # Smooth, sample-specific motion so deltas are non-trivial.
                nodal[3:7, t, :] = (
                    rng.random((4, NUM_NODES)).astype(np.float32) * 0.1 + 0.05 * t
                )
            group = handle.create_group(f"data/{sample_id}")
            group.create_dataset("nodal_data", data=nodal)
            edges = np.array(
                [[i for i in range(NUM_NODES - 1)], [i + 1 for i in range(NUM_NODES - 1)]],
                dtype=np.int64,
            )
            group.create_dataset("mesh_edge", data=edges)


def _base_config(dataset_path, **overrides):
    config = {
        "dataset_dir": str(dataset_path),
        "input_var": INPUT_VAR,
        "output_var": OUTPUT_VAR,
        "edge_var": 8,
        "positional_features": 0,
        "use_node_types": False,
        "use_world_edges": False,
        "use_multiscale": False,
        "augment_geometry": False,
        "std_noise": 0.0,
        "latent_dim": 16,
        "message_passing_num": 1,
    }
    config.update(overrides)
    return config


def _prepared_dataset(dataset_path, config):
    dataset = MeshGraphDataset(str(dataset_path), config)
    train, _, _ = dataset.split(0.5, 0.25, 0.25, seed=0)
    config["num_timesteps"] = train.num_timesteps
    return train


def _norm_stats(dataset):
    """Same payload `training_profiles.setup` injects into the config."""
    from training_profiles.setup import build_normalization_dict

    stats = build_normalization_dict(dataset)
    stats["world_edge_radius"] = dataset.world_edge_radius
    return stats


def test_ar_ot_is_the_default(tmp_path):
    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path)
    assert resolve_time_integration(config) == "ar_ot"
    assert resolve_rollout_window(config, NUM_TIMESTEPS) == 1

    dataset = _prepared_dataset(path, config)
    # Unchanged item count: one training pair per consecutive timestep pair.
    assert len(dataset) == len(dataset.sample_ids) * (NUM_TIMESTEPS - 1)
    assert getattr(dataset[0], "y_seq", None) is None


def test_ar_rt_full_trajectory_window(tmp_path):
    path = tmp_path / "traj.h5"
    _write_dataset(path)
    config = _base_config(path, time_integration="AR-RT")
    dataset = _prepared_dataset(path, config)

    assert dataset.rollout_window == NUM_TIMESTEPS - 1
    # One window per sample when the window spans the whole trajectory.
    assert len(dataset) == len(dataset.sample_ids)

    item = dataset[0]
    assert item.y_seq.shape == (NUM_NODES, NUM_TIMESTEPS - 1, OUTPUT_VAR)
    assert item.state0.shape == (NUM_NODES, INPUT_VAR)


GEOMETRY_CASES = {
    "flat": {},
    "world_edges": {"use_world_edges": True, "world_radius_multiplier": 3.0,
                    "world_edge_backend": "scipy_kdtree"},
    "multiscale": {"use_multiscale": True, "multiscale_levels": 1,
                   "coarsening_type": "bfs", "mp_per_level": [1, 1, 1]},
}


@pytest.mark.parametrize("case", sorted(GEOMETRY_CASES))
def test_ar_rt_rebuilds_the_features_the_dataloader_would_produce(tmp_path, case):
    """The core fidelity claim: rebuilt step-k features == dataloader step-k features.

    Covered for each geometry feature that has to be re-derived mid-rollout:
    mesh edge attributes, contact (world) edges, and coarse-level attributes.
    """
    geometry = GEOMETRY_CASES[case]
    path = tmp_path / "traj.h5"
    _write_dataset(path)

    ot_config = _base_config(path, **geometry)
    ot_dataset = _prepared_dataset(path, ot_config)

    rt_config = _base_config(path, time_integration="ar_rt", **geometry)
    rt_dataset = _prepared_dataset(path, rt_config)
    rt_config["_norm_stats"] = _norm_stats(rt_dataset)

    ctx = RolloutContext(rt_config, torch.device("cpu"))
    graph = rt_dataset[0]
    graph.batch = torch.zeros(NUM_NODES, dtype=torch.long)
    graph.ptr = torch.tensor([0, NUM_NODES], dtype=torch.long)

    static_tail = graph.x[:, INPUT_VAR:]
    reference_edge_attr = graph.edge_attr[:, 4:]

    windows = ot_dataset._windows_per_sample()
    for step in range(1, NUM_TIMESTEPS - 1):
        # Ground-truth state at t=step, as the rollout would hold it.
        state = graph.y_seq[:, step - 1, :]
        _apply_state(graph, state, ctx, static_tail, reference_edge_attr)

        # The AR-OT dataset item that starts at the same timestep.
        expected = ot_dataset[0 * windows + step]
        torch.testing.assert_close(graph.x, expected.x, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(graph.edge_attr, expected.edge_attr, rtol=1e-5, atol=1e-5)

        if geometry.get("use_world_edges"):
            # The two backends emit the same contact set in different order.
            assert _edge_key_set(graph.world_edge_index) == _edge_key_set(
                expected.world_edge_index
            )

        if geometry.get("use_multiscale"):
            for level in range(int(geometry["multiscale_levels"])):
                torch.testing.assert_close(
                    graph[f"coarse_edge_attr_{level}"],
                    expected[f"coarse_edge_attr_{level}"],
                    rtol=1e-5, atol=1e-5,
                )


def _edge_key_set(edge_index):
    return {(int(src), int(dst)) for src, dst in zip(edge_index[0], edge_index[1])}


def test_single_step_ar_rt_matches_ar_ot_loss(tmp_path):
    """A one-step trajectory is AR-OT with extra bookkeeping; losses must agree."""
    path = tmp_path / "traj.h5"
    # T=2 -> the full-trajectory rollout is a single step.
    _write_dataset(path, num_timesteps=2)

    from model.MeshGraphNets import MeshGraphNets

    rt_config = _base_config(path, time_integration="ar_rt")
    rt_dataset = _prepared_dataset(path, rt_config)
    rt_config["_norm_stats"] = _norm_stats(rt_dataset)

    torch.manual_seed(0)
    model = MeshGraphNets(rt_config, "cpu")
    model.eval()

    graph = rt_dataset[0]
    graph.batch = torch.zeros(NUM_NODES, dtype=torch.long)
    graph.ptr = torch.tensor([0, NUM_NODES], dtype=torch.long)

    def loss_fn(prediction, target):
        errors = torch.nn.functional.mse_loss(prediction, target, reduction="none")
        per_node = errors.mean(dim=-1)
        return per_node.mean(), per_node.sum().detach(), per_node.numel()

    ctx = RolloutContext(rt_config, torch.device("cpu"))
    with torch.no_grad():
        rollout, _, _ = rollout_loss(model, graph, ctx, loss_fn, training=False)
        predicted, target = model(graph, add_noise=False)
        one_step, _, _ = loss_fn(predicted, target)

    torch.testing.assert_close(rollout, one_step, rtol=1e-5, atol=1e-6)


def test_rollout_backpropagates_through_the_whole_unroll(tmp_path):
    path = tmp_path / "traj.h5"
    _write_dataset(path)

    from model.MeshGraphNets import MeshGraphNets

    config = _base_config(path, time_integration="ar_rt")
    dataset = _prepared_dataset(path, config)
    config["_norm_stats"] = _norm_stats(dataset)

    torch.manual_seed(0)
    model = MeshGraphNets(config, "cpu")
    model.train()

    graph = dataset[0]
    graph.batch = torch.zeros(NUM_NODES, dtype=torch.long)
    graph.ptr = torch.tensor([0, NUM_NODES], dtype=torch.long)

    def loss_fn(prediction, target):
        errors = torch.nn.functional.mse_loss(prediction, target, reduction="none")
        per_node = errors.mean(dim=-1)
        return per_node.mean(), per_node.sum().detach(), per_node.numel()

    ctx = RolloutContext(config, torch.device("cpu"))
    loss, _, loss_count = rollout_loss(model, graph, ctx, loss_fn, training=True)
    loss.backward()

    # Every step of the trajectory is scored, not just the last.
    assert loss_count == (NUM_TIMESTEPS - 1) * NUM_NODES

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "rollout produced no gradients"
    assert any(torch.any(g != 0) for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_coarse_world_edges_batch_into_the_coarse_node_space():
    """Regression: lifted contact edges must be offset by num_coarse, not num_nodes.

    Without an explicit rule in MultiscaleData, PyG's default `'index'`
    heuristic offsets `coarse_world_edge_index_{l}` by the fine node count, so
    at batch_size > 1 sample 1's coarse contacts point past the end of the
    coarse node space and into whatever happens to be there.
    """
    from torch_geometric.data import Batch

    from model.coarsening import MultiscaleData

    num_nodes, num_coarse = 10, 3

    def make_sample():
        data = MultiscaleData(
            x=torch.zeros(num_nodes, 4),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        data.num_coarse_0 = torch.tensor([num_coarse], dtype=torch.long)
        data.fine_to_coarse_0 = torch.zeros(num_nodes, dtype=torch.long)
        data.coarse_edge_index_0 = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        data.coarse_edge_attr_0 = torch.zeros(2, 8)
        data.coarse_world_edge_index_0 = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
        data.coarse_world_edge_attr_0 = torch.zeros(2, 8)
        return data

    batch = Batch.from_data_list([make_sample(), make_sample()])

    # Both the mesh and the contact edges live in the same coarse node space,
    # so they must be offset identically and stay inside it.
    assert batch.coarse_world_edge_index_0.shape == (2, 4)
    assert int(batch.coarse_world_edge_index_0.max()) < 2 * num_coarse
    torch.testing.assert_close(
        batch.coarse_world_edge_index_0[:, 2:] - batch.coarse_world_edge_index_0[:, :2],
        torch.full((2, 2), num_coarse, dtype=torch.long),
    )
