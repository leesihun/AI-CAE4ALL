"""Tests for `learned_interpolation`, the prolongation-operator switch.

`learned_interpolation True` (the default) is the learned bipartite
`UnpoolBlock`; `False` broadcasts each coarse node's state to every fine member
of its cluster. This is the successor to the legacy `bipartite_unpool` key,
which was documented in the removed CONFIGURATION_REFERENCE.md but never
implemented in this checkout.

The two paths differ in SUPPORT as well as weighting -- learned reads a fine
node's own cluster plus every coarse neighbour of it (over `up_ei`), broadcast
reads the cluster alone -- so there is no equivalence property to assert here,
unlike the attention operators in test_attention_transfer.py. What must hold
instead: the default is bit-identical to explicit True, the broadcast path
does not construct dead parameters (DDP runs `find_unused_parameters=False`),
it still trains, and it stays segmented per-sample when batched.
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

from model.MeshGraphNets import MeshGraphNets  # noqa: E402
from tests.test_attention_transfer import _base_config, _make_graph  # noqa: E402


def _build(levels, clusters, seed=0, **overrides):
    mp_per_level = [2] * (2 * levels + 1)
    torch.manual_seed(seed)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters),
        mp_per_level=mp_per_level, **overrides,
    ), device='cpu')
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Default preservation
# ---------------------------------------------------------------------------

def test_default_matches_explicit_true_exactly():
    """Omitting the key must be the pre-existing code path, byte for byte --
    not merely close. There is no softmax rounding on this axis, so anything
    other than exact equality means the default changed."""
    levels, clusters = 2, (6, 2)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)

    unset = _build(levels, clusters, seed=0)
    explicit = _build(levels, clusters, seed=0, learned_interpolation=True)

    with torch.no_grad():
        pred_unset, _ = unset(graph, add_noise=False)
        pred_explicit, _ = explicit(graph, add_noise=False)

    np.testing.assert_array_equal(pred_unset.numpy(), pred_explicit.numpy())


# ---------------------------------------------------------------------------
# Broadcast path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels,clusters", [(1, (6,)), (2, (6, 2))])
def test_broadcast_builds_no_unpool_blocks(levels, clusters):
    """A constructed-but-never-called module aborts the first DDP backward
    under find_unused_parameters=False, so the ModuleList must be absent, not
    merely unused."""
    model = _build(levels, clusters, learned_interpolation=False)
    assert not hasattr(model.model, 'unpool_blocks')

    learned = _build(levels, clusters, learned_interpolation=True)
    expected_drop = sum(
        p.numel() for i in range(levels)
        for p in learned.model.unpool_blocks[i].parameters()
    )
    drop = (sum(p.numel() for p in learned.parameters())
            - sum(p.numel() for p in model.parameters()))
    assert drop == expected_drop, (
        f"broadcast should shed exactly the {levels} UnpoolBlock(s) "
        f"({expected_drop} params), shed {drop}"
    )


@pytest.mark.parametrize("levels,clusters", [(1, (6,)), (2, (6, 2))])
def test_broadcast_forward_runs_and_differs(levels, clusters):
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)
    bcast = _build(levels, clusters, seed=0, learned_interpolation=False)
    learned = _build(levels, clusters, seed=0, learned_interpolation=True)

    with torch.no_grad():
        pred_b, _ = bcast(graph, add_noise=False)
        pred_l, _ = learned(graph, add_noise=False)

    assert pred_b.shape == pred_l.shape == (graph.x.shape[0], 4)
    assert torch.isfinite(pred_b).all()
    # Guards against the broadcast branch silently not being taken.
    assert not torch.allclose(pred_b, pred_l), \
        "broadcast produced the learned path's output -- branch not taken?"


def test_broadcast_still_trains():
    """skip_projs is the only learned coarse->fine path left under broadcast.
    If it stopped receiving gradient the arm would be untrainable, and the
    ablation would measure that instead of the operator."""
    levels, clusters = 2, (6, 2)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)
    model = _build(levels, clusters, learned_interpolation=False)
    model.train()

    pred, _ = model(graph, add_noise=False)
    pred.pow(2).mean().backward()

    for i in range(levels):
        grad = model.model.skip_projs[i].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, \
            f"skip_projs[{i}] received no gradient under broadcast"


def test_broadcast_no_cross_sample_leakage_when_batched():
    """`coarse_x[ftc]` relies on MultiscaleData.__inc__ having offset `ftc`
    into the batched coarse node space. A missed offset blends two samples'
    clusters with no error raised -- the failure mode the multiscale batching
    rules exist to prevent."""
    levels, clusters = 1, (6,)
    g0 = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters, seed=0)
    g1 = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters, seed=42)

    model = _build(levels, clusters, learned_interpolation=False)
    batch = next(iter(DataLoader([g0, g1], batch_size=2, shuffle=False)))

    with torch.no_grad():
        pred_batched, _ = model(batch, add_noise=False)
        pred_g0, _ = model(g0, add_noise=False)
        pred_g1, _ = model(g1, add_noise=False)

    n0 = g0.x.shape[0]
    np.testing.assert_allclose(pred_batched[:n0].numpy(), pred_g0.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(pred_batched[n0:].numpy(), pred_g1.numpy(), rtol=1e-5, atol=1e-6)


def test_broadcast_survives_checkpointing():
    """_unpool_merge_level is wrapped in run_checkpointed; the broadcast early
    return must stay inside that boundary."""
    levels, clusters = 2, (6, 2)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)
    model = _build(levels, clusters, learned_interpolation=False, use_checkpointing=True)
    model.train()
    pred, _ = model(graph, add_noise=False)
    pred.pow(2).mean().backward()
    assert torch.isfinite(pred).all()


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_attention_unpool_rejected_under_broadcast():
    with pytest.raises(ValueError, match="learned_interpolation True"):
        _build(2, (6, 2), learned_interpolation=False, unpool_type='attention')


def test_attention_pool_composes_with_broadcast():
    """pool_type is a separate axis: attention restriction with broadcast
    prolongation must build and run."""
    levels, clusters = 2, (6, 2)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)
    model = _build(levels, clusters, learned_interpolation=False,
                   pool_type='attention', pool_heads=4)
    with torch.no_grad():
        pred, _ = model(graph, add_noise=False)
    assert torch.isfinite(pred).all()
