import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.data import Data


def _split_first_linear(net, parts):
    """Apply net's first Linear without materializing the input concatenation.

    Equivalent to ``net[0](torch.cat([t if idx is None else t[idx] for t, idx
    in parts], dim=-1))`` but projects each part through its column block of
    the weight and gathers afterwards. Autograd then saves only the per-part
    inputs (already alive) and the summed output instead of an [E, sum(dims)]
    concat copy, and node-sized parts are projected on N rows instead of E.
    Assumes net is a build_mlp Sequential whose first layer is nn.Linear.

    Args:
        parts: sequence of (tensor, index_or_None). Column blocks are consumed
               left to right; trailing weight columns may stay unused (their
               input is implicitly zero, e.g. absent world edges).
    """
    first = net[0]
    weight = first.weight
    out = None
    col = 0
    for tensor, index in parts:
        width = tensor.shape[-1]
        # Bias rides on the first part only; per-edge gather preserves it.
        bias = first.bias if out is None else None
        proj = F.linear(tensor, weight[:, col:col + width], bias)
        if index is not None:
            proj = proj[index]
        out = proj if out is None else out + proj
        col += width
    return out


def _run_mlp_tail(net, h):
    """Run the layers after the first Linear (see _split_first_linear)."""
    for i in range(1, len(net)):
        h = net[i](h)
    return h


class EdgeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):

        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index):
        """Tensor fast path: update edge features from sender/receiver nodes.

        The first Linear runs in split form so the [E, 3*latent] concat is
        never built and the two node projections run on N rows instead of E.
        """
        senders_idx, receivers_idx = edge_index
        h = _split_first_linear(self.net, [
            (x, senders_idx),
            (x, receivers_idx),
            (edge_attr, None),
        ])
        return _run_mlp_tail(self.net, h)

    def forward(self, graph):
        edge_attr = self.compute(graph.x, graph.edge_attr, graph.edge_index)
        return Data(x=graph.x, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, num_nodes):
        """Tensor fast path: update node features from aggregated edges.

        Sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate): forces and
        stresses from neighbors should add up, not average. The first Linear
        runs in split form to skip the [N, 2*latent] concat.
        """
        _, receivers_idx = edge_index
        agg_received_edges = scatter(edge_attr, receivers_idx, dim=0, dim_size=num_nodes, reduce='sum')
        h = _split_first_linear(self.net, [(x, None), (agg_received_edges, None)])
        return _run_mlp_tail(self.net, h)

    def forward(self, graph):
        x = self.compute(graph.x, graph.edge_attr, graph.edge_index, graph.num_nodes)
        return Data(x=x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)

class HybridNodeBlock(nn.Module):
    """Node block that aggregates from both mesh and world edges."""

    def __init__(self, custom_func: nn.Module):
        super(HybridNodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, world_edge_attr, world_edge_index, num_nodes):
        """Tensor fast path: separate sum aggregation over mesh and world edges.

        With no world edges the world column block of the first Linear is
        simply skipped — identical to projecting an all-zero aggregate.
        """
        _, mesh_receivers = edge_index
        mesh_agg = scatter(edge_attr, mesh_receivers, dim=0, dim_size=num_nodes, reduce='sum')

        parts = [(x, None), (mesh_agg, None)]
        if (world_edge_attr is not None and world_edge_index is not None
                and world_edge_index.shape[1] > 0):
            _, world_receivers = world_edge_index
            world_agg = scatter(world_edge_attr, world_receivers, dim=0, dim_size=num_nodes, reduce='sum')
            parts.append((world_agg, None))

        h = _split_first_linear(self.net, parts)
        return _run_mlp_tail(self.net, h)

    def forward(self, graph):
        world_edge_attr = graph.world_edge_attr if hasattr(graph, 'world_edge_attr') else None
        world_edge_index = graph.world_edge_index if hasattr(graph, 'world_edge_index') else None
        x = self.compute(
            graph.x, graph.edge_attr, graph.edge_index,
            world_edge_attr, world_edge_index, graph.num_nodes,
        )
        return Data(
            x=x,
            edge_attr=graph.edge_attr,
            edge_index=graph.edge_index,
            world_edge_attr=world_edge_attr,
            world_edge_index=world_edge_index
        )


class AttentionPoolBlock(nn.Module):
    """Learned restriction (fine -> coarse) via within-cluster attention.

    Subsumes both fixed operators the dataset can produce: with the score head
    zero-initialized (see MeshGraphNets.__init__), every fine node in a cluster
    gets an equal score, softmax makes the weights uniform, and the output is
    exactly `pool_features` (mean pooling) -- see ATTENTION_TRANSFER_DESIGN.md
    section 4. Training can then depart from that baseline if the gradient
    says to; it can never start below it.

    Multi-head is the point, not a detail: a single head must pick one
    representative statistic per cluster (mean-like or peak-like), the same
    dilemma the fixed operators face. Splitting the latent into H heads lets
    different heads specialize (see ATTENTION_TRANSFER_DESIGN.md section 1-3).
    Each head's "value" is its own slice of h_fine (no separate value
    projection) so that uniform attention reproduces the plain mean exactly,
    per-slice, with no extra learned identity to get right.
    """

    def __init__(self, latent_dim: int, num_heads: int, build_mlp_fn):
        super().__init__()
        if latent_dim % num_heads != 0:
            raise ValueError(
                f"latent_dim ({latent_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        # score_mlp: (h_i, rel_pos_to_anchor[3], log_scale[1]) -> per-head score.
        # No LayerNorm on the score head: it would still be exactly zero at
        # init (LayerNorm of an all-zero vector is zero for default affine
        # params), but a raw Linear keeps that invariant obviously true rather
        # than relying on a LayerNorm identity that's easy to break later.
        self.score_mlp = build_mlp_fn(latent_dim + 4, latent_dim, num_heads, layer_norm=False)

    def forward(self, h_fine, ftc, num_coarse, rel_pos, log_scale):
        """
        Args:
            h_fine:     [N, D] fine node features
            ftc:        [N] long, cluster id in [0, num_coarse)
            num_coarse: int M
            rel_pos:    [N, 3] node position relative to its cluster's anchor
            log_scale:  [N, 1] log local mesh-size proxy
        Returns:
            h_coarse: [M, D] pooled coarse node features
        """
        # Score compute forced to fp32: softmax over ~10-100 cluster members
        # needs more resolution than bf16 autocast gives, and at init the
        # scores are exactly 0 everywhere -- bf16 rounding must not turn that
        # into a non-uniform distribution.
        with torch.autocast(device_type=h_fine.device.type, enabled=False):
            score_in = torch.cat(
                [h_fine.float(), rel_pos.float(), log_scale.float()], dim=-1
            )
            scores = self.score_mlp(score_in)                       # [N, H]
        a = pyg_softmax(scores, ftc, num_nodes=num_coarse, dim=0)    # [N, H], fp32

        v = h_fine.view(h_fine.shape[0], self.num_heads, self.head_dim)  # [N, H, Dh]
        weighted = a.unsqueeze(-1).to(v.dtype) * v                       # [N, H, Dh]
        pooled = scatter(weighted, ftc, dim=0, dim_size=num_coarse, reduce='sum')
        return pooled.reshape(num_coarse, self.num_heads * self.head_dim)


class UnpoolBlock(nn.Module):
    """Bipartite message passing from coarse to fine nodes (learned unpool)."""

    def __init__(self, latent_dim: int, build_mlp_fn, use_attention: bool = False):
        super().__init__()
        # EdgeMLP: (h_coarse, h_fine_skip, rel_pos) → message
        self.edge_mlp = build_mlp_fn(2 * latent_dim + 3, latent_dim, latent_dim)
        # NodeMLP: (h_fine_skip, aggregated_messages) → h_up
        self.node_mlp = build_mlp_fn(2 * latent_dim, latent_dim, latent_dim)
        self.use_attention = use_attention
        if use_attention:
            # Same (h_coarse, h_fine_skip, rel_pos) inputs as edge_mlp, scores
            # one weight per incoming source instead of a message.
            self.attn_mlp = build_mlp_fn(2 * latent_dim + 3, latent_dim, 1, layer_norm=False)

    def forward(self, h_coarse, h_fine_skip, unpool_edge_index, rel_pos):
        """
        Args:
            h_coarse:          [M, D] coarse node features
            h_fine_skip:       [N, D] fine node skip features (from descending arm)
            unpool_edge_index: [2, E_up] row0=coarse src, row1=fine dst
            rel_pos:           [E_up, 3] relative position per edge
        Returns:
            h_up: [N, D] unpooled fine node features

        E_up is ~(1 + coarse degree) * N, so the split-form first Linear
        matters most here: it avoids an [E_up, 2D+3] concat (which autocast
        would additionally promote to fp32 because rel_pos stays fp32) and
        runs the coarse/fine projections on M and N rows instead of E_up.
        """
        src_coarse, dst_fine = unpool_edge_index
        num_fine = h_fine_skip.shape[0]

        h = _split_first_linear(self.edge_mlp, [
            (h_coarse, src_coarse),
            (h_fine_skip, dst_fine),
            (rel_pos, None),
        ])
        messages = _run_mlp_tail(self.edge_mlp, h)

        if self.use_attention:
            # Zero-initialized attn_mlp (see MeshGraphNets.__init__) makes
            # every source score 0 -> softmax uniform at 1/deg_i -> the
            # deg_i-scaled sum below equals the current plain sum exactly.
            with torch.autocast(device_type=h_coarse.device.type, enabled=False):
                hs = _split_first_linear(self.attn_mlp, [
                    (h_coarse.float(), src_coarse),
                    (h_fine_skip.float(), dst_fine),
                    (rel_pos.float(), None),
                ])
                scores = _run_mlp_tail(self.attn_mlp, hs)            # [E_up, 1]
            a = pyg_softmax(scores, dst_fine, num_nodes=num_fine, dim=0)  # [E_up, 1]
            deg = scatter(torch.ones_like(scores), dst_fine, dim=0,
                         dim_size=num_fine, reduce='sum').to(messages.dtype)  # [N, 1]
            agg = deg * scatter(a.to(messages.dtype) * messages, dst_fine, dim=0,
                                dim_size=num_fine, reduce='sum')
        else:
            agg = scatter(messages, dst_fine, dim=0, dim_size=num_fine, reduce='sum')

        h_up = _split_first_linear(self.node_mlp, [(h_fine_skip, None), (agg, None)])
        return _run_mlp_tail(self.node_mlp, h_up)
