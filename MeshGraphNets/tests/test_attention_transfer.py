"""Equivalence test for the learned attention pool/unpool transfer operators.

Design: `AttentionPoolBlock` (model/blocks.py) and the attention branch of
`UnpoolBlock` are supposed to reduce EXACTLY to today's fixed operators (mean
pool / plain-sum unpool) when their score heads are zero-initialized -- see
ATTENTION_TRANSFER_DESIGN.md section 4. `MeshGraphNets.__init__` re-zeros
those heads after the generic Kaiming init, so a model built with
`pool_type='attention', unpool_type='attention'` must produce the same output
as one built with `pool_type='mean', unpool_type='sum'`, provided every other
weight is identical.

Matching every other weight via RNG seeding is not reliable here: the
attention variant's extra Linear layers shift `nn.Module.apply()`'s traversal
order, so the same seed does not give the two variants' shared submodules
identical Kaiming draws. Instead, the shared submodules' weights are copied
directly from one model onto the other after construction, isolating the
comparison to exactly the attention-specific code paths.
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


def _grid_mesh(n=4):
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


def _make_graph(input_var, multiscale_levels, voronoi_clusters, seed=0):
    edge_index, ref_pos = _grid_mesh(n=4)
    num_nodes = ref_pos.shape[0]
    rng = np.random.default_rng(seed)
    deformed_pos = ref_pos + rng.normal(scale=0.05, size=ref_pos.shape).astype(np.float32)

    hierarchy = build_multiscale_hierarchy(
        edge_index, num_nodes, ref_pos,
        multiscale_levels, ['voronoi_seedmean'] * multiscale_levels, list(voronoi_clusters),
    )

    x = rng.normal(size=(num_nodes, input_var)).astype(np.float32)
    edge_attr_raw = compute_edge_attr(ref_pos, deformed_pos, edge_index)

    graph = MultiscaleData(
        x=torch.from_numpy(x),
        pos=torch.from_numpy(ref_pos),
        edge_index=torch.from_numpy(edge_index).long(),
        edge_attr=torch.from_numpy(edge_attr_raw),
    )
    means = [np.zeros(8, dtype=np.float32) for _ in range(multiscale_levels)]
    stds = [np.ones(8, dtype=np.float32) for _ in range(multiscale_levels)]
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


def _copy_shared_weights(src, dst):
    """Copy every weight the two model variants share -- everything but the
    attention-specific score heads, which `dst` doesn't have at all."""
    src_m, dst_m = src.model, dst.model
    dst_m.encoder.load_state_dict(src_m.encoder.state_dict())
    dst_m.decoder.load_state_dict(src_m.decoder.state_dict())
    for i in range(src_m.multiscale_levels):
        dst_m.pre_blocks[i].load_state_dict(src_m.pre_blocks[i].state_dict())
        dst_m.post_blocks[i].load_state_dict(src_m.post_blocks[i].state_dict())
        dst_m.coarse_eb_encoders[i].load_state_dict(src_m.coarse_eb_encoders[i].state_dict())
        dst_m.skip_projs[i].load_state_dict(src_m.skip_projs[i].state_dict())
        dst_m.unpool_blocks[i].edge_mlp.load_state_dict(src_m.unpool_blocks[i].edge_mlp.state_dict())
        dst_m.unpool_blocks[i].node_mlp.load_state_dict(src_m.unpool_blocks[i].node_mlp.state_dict())
    dst_m.coarsest_blocks.load_state_dict(src_m.coarsest_blocks.state_dict())


@pytest.mark.parametrize("levels,clusters", [(1, (6,)), (2, (6, 2))])
def test_attention_pool_unpool_equal_mean_sum_at_init(levels, clusters):
    mp_per_level = [2] * (2 * levels + 1)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)

    torch.manual_seed(0)
    attn_model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        pool_type='attention', unpool_type='attention', pool_heads=4,
    ), device='cpu')
    attn_model.eval()

    # Deliberately different seed: the two variants' shared weights are
    # reconciled below via direct state_dict copies, not RNG matching.
    torch.manual_seed(1)
    base_model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        pool_type='mean', unpool_type='sum',
    ), device='cpu')
    base_model.eval()

    _copy_shared_weights(attn_model, base_model)

    with torch.no_grad():
        pred_attn, _ = attn_model(graph, add_noise=False)
        pred_base, _ = base_model(graph, add_noise=False)

    np.testing.assert_allclose(
        pred_attn.numpy(), pred_base.numpy(), rtol=1e-4, atol=1e-5,
    )


@pytest.mark.parametrize("pool_type,unpool_type", [("attention", "sum"), ("mean", "attention")])
def test_attention_pool_or_unpool_alone_equal_baseline(pool_type, unpool_type):
    """Pool-only and unpool-only attention must each independently reduce to
    the baseline too -- the two operators are ablated independently
    (ATTENTION_TRANSFER_DESIGN.md section 10.5), so each must be safe alone."""
    levels, clusters = 1, (6,)
    mp_per_level = [2] * (2 * levels + 1)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)

    torch.manual_seed(0)
    variant_model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        pool_type=pool_type, unpool_type=unpool_type, pool_heads=4,
    ), device='cpu')
    variant_model.eval()

    torch.manual_seed(1)
    base_model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        pool_type='mean', unpool_type='sum',
    ), device='cpu')
    base_model.eval()

    _copy_shared_weights(variant_model, base_model)

    with torch.no_grad():
        pred_variant, _ = variant_model(graph, add_noise=False)
        pred_base, _ = base_model(graph, add_noise=False)

    np.testing.assert_allclose(
        pred_variant.numpy(), pred_base.numpy(), rtol=1e-4, atol=1e-5,
    )


def test_attention_pool_diverges_after_a_gradient_step():
    """Sanity check the other direction: once trained, attention must be able
    to move away from the baseline (the equivalence tests above would also
    pass for a block that's simply broken/inert)."""
    levels, clusters = 1, (6,)
    mp_per_level = [2] * (2 * levels + 1)
    graph = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters)
    target = torch.randn(graph.x.shape[0], 4)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        pool_type='attention', unpool_type='attention', pool_heads=4,
    ), device='cpu')
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        pred, _ = model(graph, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop with attention pool/unpool enabled: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )

    score_weight = model.model.pool_blocks[0].score_mlp[-1].weight
    assert score_weight.abs().max().item() > 1e-6, (
        "pool score head did not move away from its zero-init after training"
    )


def test_attention_pool_unpool_no_cross_sample_leakage_when_batched():
    """The attention pool/unpool softmaxes are keyed by ftc / dst_fine, which
    `MultiscaleData.__inc__` already offsets per-sample for the existing
    non-attention paths. Confirm the same offset machinery keeps the
    attention softmax segmented per-sample too -- a leak here would silently
    blend two different samples' clusters together."""
    levels, clusters = 1, (6,)
    mp_per_level = [2] * (2 * levels + 1)
    g0 = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters, seed=0)
    g1 = _make_graph(input_var=4, multiscale_levels=levels, voronoi_clusters=clusters, seed=42)

    torch.manual_seed(0)
    model = MeshGraphNets(_base_config(
        multiscale_levels=levels, voronoi_clusters=list(clusters), mp_per_level=mp_per_level,
        pool_type='attention', unpool_type='attention', pool_heads=4,
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
