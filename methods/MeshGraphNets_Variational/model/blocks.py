import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import Data


def split_linear(linear, parts):
    """`linear(torch.cat([t if idx is None else t[idx] for t, idx in parts], -1))`
    without building the concatenation.

    Each part goes through its own column block of the weight and is gathered
    afterwards, so autograd keeps the per-part inputs (already alive) instead of
    an [rows, sum(widths)] concat copy, and node-sized parts are projected on N
    rows instead of E (the PhysicsNeMo "concat trick"). Columns left over at the
    end are skipped, which is the same as a zero input there (absent world
    edges). The bias rides on the first part, before any gather.
    """
    weight = linear.weight
    out = None
    col = 0
    for tensor, index in parts:
        width = tensor.shape[-1]
        proj = F.linear(tensor, weight[:, col:col + width], linear.bias if out is None else None)
        if index is not None:
            proj = proj[index]
        out = proj if out is None else out + proj
        col += width
    return out


def _split_mlp(net, parts):
    """A build_mlp Sequential applied to the virtual concat of `parts`."""
    h = split_linear(net[0], parts)
    for i in range(1, len(net)):
        h = net[i](h)
    return h


class EdgeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):

        super(EdgeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index):
        """[E, D] edge update from (sender, receiver, edge) features.

        Same state dict as `net(cat([x[s], x[r], e]))`; the [E, 3D] concat is
        never built and the two node projections run on N rows.
        """
        senders_idx, receivers_idx = edge_index
        return _split_mlp(self.net, [(x, senders_idx), (x, receivers_idx), (edge_attr, None)])

    def forward(self, graph):
        edge_attr = self.compute(graph.x, graph.edge_attr, graph.edge_index)
        return Data(x=graph.x, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, num_nodes):
        """[N, D] node update from the sum of incoming edge features.

        Sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate): forces and
        stresses from neighbors add up, not average.
        """
        _, receivers_idx = edge_index
        agg = scatter(edge_attr, receivers_idx, dim=0, dim_size=num_nodes, reduce='sum')
        return _split_mlp(self.net, [(x, None), (agg, None)])

    def forward(self, graph):
        x = self.compute(graph.x, graph.edge_attr, graph.edge_index, graph.num_nodes)
        return Data(x=x, edge_attr=graph.edge_attr, edge_index=graph.edge_index)

class HybridNodeBlock(nn.Module):
    """Node block that aggregates from both mesh and world edges."""

    def __init__(self, custom_func: nn.Module):
        super(HybridNodeBlock, self).__init__()
        self.net = custom_func

    def compute(self, x, edge_attr, edge_index, world_edge_attr, world_edge_index, num_nodes):
        """Separate sum aggregation over mesh and world edges.

        With no world edges the world column block of the first Linear is
        skipped, which is what an all-zero world aggregate contributed.
        """
        _, mesh_receivers = edge_index
        mesh_agg = scatter(edge_attr, mesh_receivers, dim=0, dim_size=num_nodes, reduce='sum')
        parts = [(x, None), (mesh_agg, None)]
        if (world_edge_attr is not None and world_edge_index is not None
                and world_edge_index.shape[1] > 0):
            _, world_receivers = world_edge_index
            parts.append((scatter(world_edge_attr, world_receivers, dim=0,
                                  dim_size=num_nodes, reduce='sum'), None))
        return _split_mlp(self.net, parts)

    def forward(self, graph):
        world_edge_attr = getattr(graph, 'world_edge_attr', None)
        world_edge_index = getattr(graph, 'world_edge_index', None)
        x = self.compute(graph.x, graph.edge_attr, graph.edge_index,
                         world_edge_attr, world_edge_index, graph.num_nodes)
        return Data(
            x=x,
            edge_attr=graph.edge_attr,
            edge_index=graph.edge_index,
            world_edge_attr=world_edge_attr,
            world_edge_index=world_edge_index,
        )


class UnpoolBlock(nn.Module):
    """Bipartite message passing from coarse to fine nodes (learned unpool)."""

    def __init__(self, latent_dim: int, build_mlp_fn):
        super().__init__()
        # EdgeMLP: (h_coarse, h_fine_skip, rel_pos) → message
        self.edge_mlp = build_mlp_fn(2 * latent_dim + 3, latent_dim, latent_dim)
        # NodeMLP: (h_fine_skip, aggregated_messages) → h_up
        self.node_mlp = build_mlp_fn(2 * latent_dim, latent_dim, latent_dim)

    def forward(self, h_coarse, h_fine_skip, unpool_edge_index, rel_pos):
        """
        Args:
            h_coarse:          [M, D] coarse node features
            h_fine_skip:       [N, D] fine node skip features (from descending arm)
            unpool_edge_index: [2, E_up] row0=coarse src, row1=fine dst
            rel_pos:           [E_up, 3] relative position per edge
        Returns:
            h_up: [N, D] unpooled fine node features

        E_up is ~(1 + coarse degree) * N: the split first Linear skips an
        [E_up, 2D+3] concat (which autocast would also promote to fp32, since
        rel_pos is fp32) and projects on M and N rows instead of E_up.
        """
        src_coarse, dst_fine = unpool_edge_index
        messages = _split_mlp(self.edge_mlp, [
            (h_coarse, src_coarse),
            (h_fine_skip, dst_fine),
            (rel_pos, None),
        ])
        agg = scatter(messages, dst_fine, dim=0,
                      dim_size=h_fine_skip.shape[0], reduce='sum')
        return _split_mlp(self.node_mlp, [(h_fine_skip, None), (agg, None)])


class AdaLNZero(nn.Module):
    """DiT-style (shift, scale, gate) conditioning of one GnBlock on the global latent z.

    The name is kept from DiT's AdaLN-Zero, but two things differ: there is no
    LayerNorm before the modulation (the GnBlock's own MLPs normalize), and the
    gate is initialized to 1, not 0, so the block starts as the unconditioned
    block rather than as the identity.

    Replaces the concat-fuser `Linear([x, z])`, which had two defects:
      (a) z is broadcast identically to every node of a graph, so
          `Linear([x, z]) = W_x x + W_z z + b` could only add ONE constant
          vector per graph -- no scaling, no gating, no spatial selection.
      (b) it is a bare, non-residual, un-normalized Linear inserted between all
          ~28 processor blocks. Under the repo's kaiming_uniform(relu) init its
          x-half has gain ~1.33, which compounds to ~10^3 by the decoder and
          drowns the V-cycle's unpool branch (LayerNorm'd to unit scale) under
          the inflated skip branch -- i.e. the multiscale global information is
          attenuated by exactly the mechanism meant to inject z.

    Here z instead emits (shift, scale, gate) and modulates the block
    (Peebles & Xie, DiT 2023):

        x_mod = (1 + scale) * x + shift
        x_out = x + gate * (Block(x_mod) - x_mod)

    The final Linear is zero-initialized with bias (0, 0, 1), so at init
    scale=0, shift=0, gate=1 and this is EXACTLY the unconditioned block: the
    residual highway is preserved bit-for-bit and conditioning can only be
    learned in, never destroy it. `scale`/`gate` are multiplicative, giving z
    the gating power a pure additive offset never had.

    It is also cheaper than what it replaces: Linear(z_dim -> 3*D) is
    3*z_dim*D MACs per node vs the fuser's (D + z_dim)*D -- 0.61 vs 1.84
    GFLOP/block at D=128, z_dim=16, N=100k.
    """

    def __init__(self, latent_dim, z_dim):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(int(z_dim), 3 * self.latent_dim))
        self.reset_identity()

    def reset_identity(self):
        """(Re-)initialize to shift=0, scale=0, gate=1: the unconditioned block.

        Must run AFTER any global `apply(init_weights)`, which would otherwise
        kaiming-fill these and reintroduce the very gain problem being fixed.
        """
        linear = self.net[1]
        nn.init.zeros_(linear.weight)
        with torch.no_grad():
            linear.bias.zero_()
            linear.bias[2 * self.latent_dim:].fill_(1.0)   # gate = 1

    def forward(self, z_per_node):
        """[N, z_dim] -> (shift, scale, gate), each [N, latent_dim]."""
        return self.net(z_per_node).chunk(3, dim=-1)


def adaln_tensors(block, modulation, z_per_node, x, edge_attr, edge_index,
                  world_edge_attr=None, world_edge_index=None):
    """GnBlock.forward_tensors under AdaLN-Zero modulation from `z_per_node`.

    The block runs on the modulated features, then its own update is
    re-attached to the UNMODULATED input through the gate. Returns
    (x, edge_attr, world_edge_attr) like `forward_tensors`.
    """
    shift, scale, gate = modulation(z_per_node)
    x_mod = x * (1.0 + scale) + shift
    x_out, edge_attr, world_edge_attr = block.forward_tensors(
        x_mod, edge_attr, edge_index, world_edge_attr, world_edge_index)
    return x + gate * (x_out - x_mod), edge_attr, world_edge_attr


def apply_adaln(block, graph, modulation, z_per_node):
    """`adaln_tensors` on a Data graph; returns a new Data (the input is not modified)."""
    world_edge_attr = getattr(graph, 'world_edge_attr', None)
    world_edge_index = getattr(graph, 'world_edge_index', None)
    x, edge_attr, world_edge_attr = adaln_tensors(
        block, modulation, z_per_node, graph.x, graph.edge_attr, graph.edge_index,
        world_edge_attr, world_edge_index)
    return block.pack(x, edge_attr, graph.edge_index, world_edge_attr, world_edge_index)
