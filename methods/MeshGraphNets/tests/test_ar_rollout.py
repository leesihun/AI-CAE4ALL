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

from general_modules.edge_features import (  # noqa: E402
    DEFORMED_FEATURE_DIM,
    EDGE_FEATURE_DIM,
    REFERENCE_SLICE,
    deformed_edge_attr_torch,
)
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
    # Multi-partition coarsening (ATTENTION_TRANSFER_DESIGN.md Part II): the
    # rollout must refresh every branch of a branched level, each keyed
    # `*_{level}_{branch}` rather than `*_{level}`.
    "multiscale_branched": {"use_multiscale": True, "multiscale_levels": 1,
                            "coarsening_type": "voronoi_seedmean",
                            "voronoi_clusters": [4], "voronoi_branches": [3],
                            "mp_per_level": [1, 1, 1]},
}


def _coarse_attr_keys(geometry):
    """Coarse edge-attr keys a given geometry case is expected to produce."""
    levels = int(geometry["multiscale_levels"])
    branches = geometry.get("voronoi_branches") or [1] * levels
    keys = []
    for level in range(levels):
        k = branches[level] if level < len(branches) else 1
        if k > 1:
            keys.extend(f"coarse_edge_attr_{level}_{b}" for b in range(k))
        else:
            keys.append(f"coarse_edge_attr_{level}")
    return keys


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
    reference_edge_attr = graph.edge_attr[:, REFERENCE_SLICE]

    # Snapshot the dataloader's t=0 coarse features so the branched case can
    # prove the rollout actually overwrote them (see the assertion below).
    baseline_coarse = {
        key: graph[key].clone()
        for key in (_coarse_attr_keys(geometry) if geometry.get("use_multiscale") else [])
    }

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
            keys = _coarse_attr_keys(geometry)
            assert keys, "expected at least one coarse edge-attr key"
            for key in keys:
                assert key in expected, (
                    f"dataloader did not produce {key}; the case's expected "
                    f"branch layout and the dataset disagree"
                )
                torch.testing.assert_close(
                    graph[key], expected[key], rtol=1e-5, atol=1e-5,
                )
            if geometry.get("voronoi_branches"):
                # Guard against this case passing vacuously: a branched level
                # must publish ONLY suffixed keys, and _refresh_multiscale must
                # have actually rewritten each branch (not silently early-returned,
                # which would leave the t=0 values in place and still "match" if
                # the trajectory happened to be static).
                assert "coarse_edge_attr_0" not in expected, (
                    "a branched level must not also publish the unsuffixed key"
                )
                assert len(keys) == geometry["voronoi_branches"][0]
                assert not torch.allclose(graph[keys[0]], baseline_coarse[keys[0]]), (
                    "branch 0 was never refreshed from the rolled-out state"
                )


def _edge_key_set(edge_index):
    return {(int(src), int(dst)) for src, dst in zip(edge_index[0], edge_index[1])}


def test_edge_half_constants_match_the_actual_feature_layout():
    """DEFORMED/REFERENCE slices are only useful if they track the real layout.

    They are consumed as `edge_attr[:, REFERENCE_SLICE]` in the rollout, so a
    constant that drifted from what `deformed_edge_attr_torch` actually emits
    would silently mis-slice every rollout step rather than raise.
    """
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

    deformed = deformed_edge_attr_torch(pos, edge_index)
    assert deformed.shape[1] == DEFORMED_FEATURE_DIM

    from general_modules.edge_features import compute_edge_attr
    full = compute_edge_attr(pos.numpy(), pos.numpy(), edge_index.numpy())
    assert full.shape[1] == EDGE_FEATURE_DIM
    # The two halves must partition the feature, with the deformed half first.
    assert full[:, REFERENCE_SLICE].shape[1] == EDGE_FEATURE_DIM - DEFORMED_FEATURE_DIM
    torch.testing.assert_close(
        torch.from_numpy(full[:, :DEFORMED_FEATURE_DIM]), deformed,
        rtol=1e-6, atol=1e-6,
    )


def test_single_step_ar_rt_is_ar_ot_rescaled_per_channel(tmp_path):
    """A one-step rollout is AR-OT's loss rescaled by (delta_std/node_std)^2.

    AR-RT scores the STATE error normalized by the state spread, AR-OT scores
    the one-step DELTA error normalized by the delta spread. At step 0 the
    rollout has not drifted, so the two differ by exactly the per-channel ratio
    of those two scales -- an identity worth pinning, because the previous
    implementation normalized by delta_std at every step, which made this test
    pass as exact equality while the loss silently grew with the step index on
    a real 49-step trajectory (see the ar_rollout module docstring).
    """
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

        # The exact identity: advanced - gt = delta_std * (pred - target_ot),
        # so the state-normalized residual is that scaled by 1/state_std.
        scale = (ctx.delta_std[:OUTPUT_VAR] / ctx.state_std)
        expected = ((predicted - target) * scale).pow(2).mean(dim=-1).mean()

    torch.testing.assert_close(rollout, expected, rtol=1e-5, atol=1e-7)

    # And the rescale is real, not a no-op: on this fixture the two scales
    # differ, so an implementation that still divided by delta_std would be
    # caught here rather than silently agreeing.
    assert not torch.allclose(scale, torch.ones_like(scale), rtol=1e-3), (
        "fixture has delta_std == node_std, so this test cannot distinguish "
        "the two normalizations -- widen the synthetic trajectory"
    )
    assert not torch.isclose(rollout, one_step, rtol=1e-3), (
        "state-space and delta-space losses coincide; the fix is not in effect"
    )


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
