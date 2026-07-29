"""Multi-partition coarse representation tests (ATTENTION_TRANSFER_DESIGN.md Part II).

Covers what the design doc's own risk list (section 12) flags as highest
risk: batching correctness for branch-suffixed MultiscaleData attributes (a
leak there is silent, not an exception -- see model/coarsening.py's
MultiscaleData docstring). Also covers the config validation guards
(branching only the last level, only voronoi_* methods, not under AR-RT) and
a gradient step showing the branch merge (skip_projs, widened for K
branches) is actually load-bearing, not structurally present but disconnected.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from torch_geometric.loader import DataLoader  # noqa: E402

from general_modules.edge_features import compute_edge_attr  # noqa: E402
from general_modules.multiscale_helpers import (  # noqa: E402
    attach_coarse_levels_to_graph,
    build_multiscale_hierarchy,
)
from model.coarsening import MultiscaleData  # noqa: E402
from model.MeshGraphNets import MeshGraphNets  # noqa: E402
from training_profiles.ar_rollout import RolloutContext  # noqa: E402


def _grid_mesh(n=6):
    """n x n grid mesh -> (bidirectional edge_index [2, E], ref_pos [N, 3])."""
    ids = np.arange(n * n).reshape(n, n)
    edges = []
    for r in range(n):
        for c in range(n):
            if c + 1 < n:
                edges.append((ids[r, c], ids[r, c + 1]))
            if r + 1 < n:
                edges.append((ids[r, c], ids[r + 1, c]))
    edges = np.array(edges, dtype=np.int64).T
    edge_index = np.concatenate([edges, edges[[1, 0]]], axis=1)
    xs, ys = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    ref_pos = np.stack(
        [xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1
    ).astype(np.float32)
    return edge_index, ref_pos


def _make_graph(input_var, multiscale_levels, voronoi_clusters, voronoi_branches, seed=0, n=6):
    edge_index, ref_pos = _grid_mesh(n=n)
    num_nodes = ref_pos.shape[0]
    rng = np.random.default_rng(seed)
    deformed_pos = ref_pos + rng.normal(scale=0.05, size=ref_pos.shape).astype(np.float32)

    hierarchy = build_multiscale_hierarchy(
        edge_index, num_nodes, ref_pos,
        multiscale_levels, ['voronoi_seedmean'] * multiscale_levels,
        list(voronoi_clusters), list(voronoi_branches),
    )

    x = rng.normal(size=(num_nodes, input_var)).astype(np.float32)
    edge_attr_raw = compute_edge_attr(ref_pos, deformed_pos, edge_index)

    graph = MultiscaleData(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(ref_pos),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr_raw),
    )
    means, stds = [], []
    for kb in voronoi_branches[:multiscale_levels]:
        if kb > 1:
            means.append([np.zeros(8, dtype=np.float32)] * kb)
            stds.append([np.ones(8, dtype=np.float32)] * kb)
        else:
            means.append(np.zeros(8, dtype=np.float32))
            stds.append(np.ones(8, dtype=np.float32))
    attach_coarse_levels_to_graph(graph, hierarchy, ref_pos, deformed_pos, means, stds)
    return graph


def _base_config(**overrides):
    config = dict(
        input_var=4, output_var=4, edge_var=8, latent_dim=16,
        message_passing_num=2, use_node_types=False, positional_features=0,
        use_world_edges=False, use_checkpointing=False,
        use_multiscale=True, std_noise=0.0,
    )
    config.update(overrides)
    return config


def test_explicit_single_branch_matches_unset_config():
    """voronoi_branches=[1] must be architecturally and numerically identical
    to omitting it entirely -- unlike Part I's mean-vs-attention comparison,
    the module *shapes* are identical either way, so the same seed gives
    bit-identical weights with no state_dict copying needed."""
    levels, clusters = 1, (6,)
    mp_per_level = [2, 2, 2]
    graph = _make_graph(4, levels, clusters, (1,))

    torch.manual_seed(0)
    model_a = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
    ), device='cpu')
    model_a.eval()

    torch.manual_seed(0)
    model_b = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        voronoi_branches=[1],
    ), device='cpu')
    model_b.eval()

    with torch.no_grad():
        pred_a, _ = model_a(graph, add_noise=False)
        pred_b, _ = model_b(graph, add_noise=False)
    np.testing.assert_allclose(pred_a.numpy(), pred_b.numpy(), rtol=0, atol=0)


def test_multi_partition_forward_runs():
    levels, clusters, branches = 1, (6,), (4,)
    mp_per_level = [2, 2, 2]
    graph = _make_graph(4, levels, clusters, branches)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        voronoi_branches=list(branches),
    ), device='cpu')
    model.eval()

    with torch.no_grad():
        pred, _ = model(graph, add_noise=False)
    assert pred.shape == (graph.x.shape[0], 4)
    assert torch.isfinite(pred).all()


def test_multi_partition_two_level_forward_runs():
    """K branches only on the deeper of two levels; level 0 stays unbranched."""
    levels, clusters, branches = 2, (10, 3), (1, 4)
    mp_per_level = [2, 2, 2, 2, 2]
    graph = _make_graph(4, levels, clusters, branches, n=8)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        voronoi_branches=list(branches),
    ), device='cpu')
    model.eval()

    with torch.no_grad():
        pred, _ = model(graph, add_noise=False)
    assert pred.shape == (graph.x.shape[0], 4)
    assert torch.isfinite(pred).all()


def test_multi_partition_no_cross_sample_leakage_when_batched():
    """The highest-risk item in ATTENTION_TRANSFER_DESIGN.md section 12: a
    wrong batching offset for branch-suffixed attrs silently blends two
    samples' clusters together -- no exception, just wrong numbers."""
    levels, clusters, branches = 1, (6,), (4,)
    mp_per_level = [2, 2, 2]
    g0 = _make_graph(4, levels, clusters, branches, seed=0)
    g1 = _make_graph(4, levels, clusters, branches, seed=42)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        voronoi_branches=list(branches),
    ), device='cpu')
    model.eval()

    loader = DataLoader([g0, g1], batch_size=2, shuffle=False)
    batch = next(iter(loader))

    with torch.no_grad():
        pred_batched, _ = model(batch, add_noise=False)
        pred_g0, _ = model(g0, add_noise=False)
        pred_g1, _ = model(g1, add_noise=False)

    n0 = g0.x.shape[0]
    np.testing.assert_allclose(pred_batched[:n0].numpy(), pred_g0.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(pred_batched[n0:].numpy(), pred_g1.numpy(), rtol=1e-4, atol=1e-5)


def test_multi_partition_combines_with_attention_pool_unpool():
    """Part I (learned transfer operators) and Part II (multi-partition) are
    independent features that must compose: attention pool/unpool run once
    per branch with shared weights, same as mean/sum would."""
    levels, clusters, branches = 1, (6,), (4,)
    mp_per_level = [2, 2, 2]
    graph = _make_graph(4, levels, clusters, branches)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        voronoi_branches=list(branches),
        pool_type='attention', unpool_type='attention', pool_heads=4,
    ), device='cpu')
    model.eval()

    with torch.no_grad():
        pred, _ = model(graph, add_noise=False)
    assert pred.shape == (graph.x.shape[0], 4)
    assert torch.isfinite(pred).all()

    # Batching correctness must hold for the combination too, not just each
    # feature alone -- attention adds a segment-softmax over ftc/dst_fine
    # that Part I's own tests already checked without branches present.
    g1 = _make_graph(4, levels, clusters, branches, seed=42)
    loader = DataLoader([graph, g1], batch_size=2, shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        pred_batched, _ = model(batch, add_noise=False)
        pred_g1, _ = model(g1, add_noise=False)
    n0 = graph.x.shape[0]
    np.testing.assert_allclose(pred_batched[:n0].numpy(), pred.numpy(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(pred_batched[n0:].numpy(), pred_g1.numpy(), rtol=1e-4, atol=1e-5)


def test_multi_partition_gradient_step_uses_all_branches():
    """Loss must drop, and every branch's slice of the widened skip_projs
    input must receive a nonzero gradient every step -- confirms the branch
    outputs actually reach the loss, not just exist structurally."""
    levels, clusters, branches = 1, (6,), (4,)
    latent_dim = 16
    mp_per_level = [2, 2, 2]
    graph = _make_graph(4, levels, clusters, branches)
    target = torch.randn(graph.x.shape[0], 4)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        voronoi_branches=list(branches), latent_dim=latent_dim,
    ), device='cpu')
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        pred, _ = model(graph, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()

        # The load-bearing check: every branch's slice of the merge input
        # gets a nonzero gradient on EVERY step, not just eventually.
        w_grad = model.model.skip_projs[0].weight.grad
        assert w_grad is not None
        for b in range(1, 1 + branches[0]):
            branch_slice = w_grad[:, b * latent_dim:(b + 1) * latent_dim]
            assert branch_slice.abs().max().item() > 0, f"branch {b - 1} got no gradient"

        optimizer.step()
        losses.append(loss.item())

    # Weak sanity check (the gradient-flow assertion above is the real test):
    # training is doing something, not stuck at initialization.
    assert losses[-1] < losses[0] * 0.8, f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_inference_rebuilds_the_branch_layout_the_checkpoint_was_trained_with():
    """A branched checkpoint must be loadable by the inference path.

    `skip_projs` is sized for (1 + K) merge inputs, so if inference rebuilds an
    UNBRANCHED hierarchy for a branched checkpoint the unpool merge feeds a
    2*latent_dim tensor into a (1+K)*latent_dim Linear and dies. This walks the
    same three steps inference_profiles/rollout.py does -- read
    voronoi_branches out of the checkpoint's model_config, rebuild the
    hierarchy with it, run the model -- so the wiring between them is checked
    rather than assumed.
    """
    from training_profiles.setup import build_model_config

    levels, clusters, branches = 1, (6,), (4,)
    mp_per_level = [2, 2, 2]
    train_config = _base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters),
        mp_per_level=mp_per_level, voronoi_branches=list(branches),
        pool_type='attention', unpool_type='attention', pool_heads=4,
    )
    torch.manual_seed(0)
    model = MeshGraphNets(train_config, device='cpu')
    model.eval()

    # What save_checkpoint() would persist, and what rollout.py merges back in.
    model_config = build_model_config(train_config)
    assert model_config['voronoi_branches'] == [1 * b for b in branches], (
        'voronoi_branches must survive into the checkpoint, or inference '
        'cannot know the branch layout'
    )

    # rollout.py's parsing of that value (mirrored here).
    raw_vb = model_config['voronoi_branches']
    infer_branches = [int(v) for v in raw_vb] if isinstance(raw_vb, list) else [int(raw_vb)]
    if len(infer_branches) == 1 and levels > 1:
        infer_branches = infer_branches * levels

    edge_index, ref_pos = _grid_mesh(n=6)
    hierarchy = build_multiscale_hierarchy(
        edge_index, ref_pos.shape[0], ref_pos, levels,
        ['voronoi_seedmean'] * levels, list(clusters), infer_branches,
    )
    assert 'branches' in hierarchy[0] and len(hierarchy[0]['branches']) == branches[0]

    rng = np.random.default_rng(0)
    deformed_pos = ref_pos + rng.normal(scale=0.05, size=ref_pos.shape).astype(np.float32)
    graph = MultiscaleData(
        x=torch.from_numpy(rng.normal(size=(ref_pos.shape[0], 4)).astype(np.float32)),
        pos=torch.from_numpy(ref_pos),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(compute_edge_attr(ref_pos, deformed_pos, edge_index)),
    )
    attach_coarse_levels_to_graph(
        graph, hierarchy, ref_pos, deformed_pos,
        [[np.zeros(8, dtype=np.float32)] * branches[0]],
        [[np.ones(8, dtype=np.float32)] * branches[0]],
    )
    with torch.no_grad():
        pred, _ = model(graph, add_noise=False)
    assert pred.shape == (ref_pos.shape[0], 4)
    assert torch.isfinite(pred).all()


def test_bfs_branching_rejected_at_hierarchy_build():
    edge_index, ref_pos = _grid_mesh(n=6)
    with pytest.raises(ValueError, match="bfs"):
        build_multiscale_hierarchy(
            edge_index, ref_pos.shape[0], ref_pos, 1, ['bfs'], [0], [4],
        )


def test_branching_non_last_level_rejected_at_hierarchy_build():
    edge_index, ref_pos = _grid_mesh(n=8)
    with pytest.raises(ValueError, match="last configured level"):
        build_multiscale_hierarchy(
            edge_index, ref_pos.shape[0], ref_pos, 2,
            ['voronoi_seedmean', 'voronoi_seedmean'], [10, 3], [4, 1],
        )


def test_branching_non_last_level_rejected_at_model_construction():
    with pytest.raises(ValueError, match="last configured level"):
        MeshGraphNets(_base_config(
            multiscale_levels=2, voronoi_clusters=[10, 3],
            mp_per_level=[2, 2, 2, 2, 2], voronoi_branches=[4, 1],
        ), device='cpu')


def test_ar_rt_unwraps_per_branch_coarse_edge_stats():
    """AR-RT supports branched levels; the subtle part is the stats shape.

    A branched level's entry in `coarse_edge_means` is a LIST of per-branch
    arrays, not one array. The deformed-half slice has to be applied to each
    array -- applied to the list it would slice the branch axis and yield a
    silently wrong [K, 8] constant instead of a [4] one. End-to-end rollout
    fidelity for branched levels is covered by test_ar_rollout.py's
    `multiscale_branched` geometry case.
    """
    from general_modules.edge_features import DEFORMED_FEATURE_DIM

    config = {
        'input_var': 4, 'output_var': 4,
        'use_multiscale': True, 'multiscale_levels': 2,
        'voronoi_branches': [1, 4],
        '_norm_stats': {
            'node_mean': np.zeros(4), 'node_std': np.ones(4),
            'edge_mean': np.zeros(8), 'edge_std': np.ones(8),
            'delta_mean': np.zeros(4), 'delta_std': np.ones(4),
            # level 0 unbranched -> one array; level 1 branched -> 4 arrays.
            'coarse_edge_means': [np.zeros(8), [np.zeros(8)] * 4],
            'coarse_edge_stds': [np.ones(8), [np.ones(8)] * 4],
        },
    }
    ctx = RolloutContext(config, torch.device('cpu'))

    mean0, std0 = ctx.coarse_edge_stat(0)
    assert mean0.shape == (DEFORMED_FEATURE_DIM,)
    assert std0.shape == (DEFORMED_FEATURE_DIM,)
    for b in range(4):
        mean1, std1 = ctx.coarse_edge_stat(1, b)
        assert mean1.shape == (DEFORMED_FEATURE_DIM,), (
            f'branch {b} stat has shape {tuple(mean1.shape)}; the branch list '
            f'was sliced instead of each branch array'
        )
        assert std1.shape == (DEFORMED_FEATURE_DIM,)
    # Past the end of the stats list means "skip normalization".
    assert ctx.coarse_edge_stat(9) == (None, None)
