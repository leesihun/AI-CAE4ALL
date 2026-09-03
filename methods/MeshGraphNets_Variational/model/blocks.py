import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.data import Data


class EdgeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):

        super(EdgeBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        node_attr = graph.x
        senders_idx, receivers_idx = graph.edge_index
        edge_attr = graph.edge_attr

        # Split-linear: project node features at [N, D] scale before gathering to
        # [E, D], instead of gathering first and projecting at [E, 3D] scale.
        # Equivalent because Linear(cat(s, r, e)) = s@W_s + r@W_r + e@W_e.
        # Eliminates the [E, 3D] activation that autograd saves for backward in
        # every GnBlock, cutting activation VRAM roughly in half. State-dict unchanged.
        first = self.net[0]   # nn.Linear(3D, D)
        D = node_attr.shape[-1]
        h_s = F.linear(node_attr, first.weight[:, :D])                    # [N, D]
        h_r = F.linear(node_attr, first.weight[:, D:2 * D])               # [N, D]
        h_e = F.linear(edge_attr, first.weight[:, 2 * D:], first.bias)    # [E, D]
        edge_attr = self.net[1:](h_s[senders_idx] + h_r[receivers_idx] + h_e)

        return Data(x=node_attr, edge_attr=edge_attr, edge_index=graph.edge_index)


class NodeBlock(nn.Module):

    def __init__(self, custom_func:nn.Module):
        super(NodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr # [E, 4] (3D)
        nodes_to_collect = []

        _, receivers_idx = graph.edge_index # [E, 2] (sender, receiver)
        num_nodes = graph.num_nodes
        # Use sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate implementation)
        # For physics: forces/stresses from neighbors should add up, not average
        agg_received_edges = scatter(edge_attr, receivers_idx, dim=0, dim_size=num_nodes, reduce='sum')

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)

class HybridNodeBlock(nn.Module):
    """Node block that aggregates from both mesh and world edges."""

    def __init__(self, custom_func: nn.Module):
        super(HybridNodeBlock, self).__init__()
        self.net = custom_func

    def forward(self, graph):
        # Aggregate mesh edges
        mesh_edge_attr = graph.edge_attr
        _, mesh_receivers = graph.edge_index
        num_nodes = graph.num_nodes
        # Use sum aggregation (matches NVIDIA PhysicsNeMo deforming_plate implementation)
        mesh_agg = scatter(mesh_edge_attr, mesh_receivers, dim=0, dim_size=num_nodes, reduce='sum')

        # Aggregate world edges (if present)
        if (hasattr(graph, 'world_edge_attr') and hasattr(graph, 'world_edge_index')
            and graph.world_edge_attr is not None and graph.world_edge_index.shape[1] > 0):
            world_edge_attr = graph.world_edge_attr
            _, world_receivers = graph.world_edge_index
            # Use sum aggregation for world edges as well
            world_agg = scatter(world_edge_attr, world_receivers, dim=0, dim_size=num_nodes, reduce='sum')
        else:
            world_agg = torch.zeros_like(mesh_agg)

        # Concatenate node features with both aggregations
        collected_nodes = torch.cat([graph.x, mesh_agg, world_agg], dim=-1)
        x = self.net(collected_nodes)

        return Data(
            x=x,
            edge_attr=mesh_edge_attr,
            edge_index=graph.edge_index,
            world_edge_attr=graph.world_edge_attr if hasattr(graph, 'world_edge_attr') else None,
            world_edge_index=graph.world_edge_index if hasattr(graph, 'world_edge_index') else None
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
        """
        src_coarse, dst_fine = unpool_edge_index

        edge_input = torch.cat([
            h_coarse[src_coarse],
            h_fine_skip[dst_fine],
            rel_pos,
        ], dim=-1)
        messages = self.edge_mlp(edge_input)

        agg = scatter(messages, dst_fine, dim=0,
                      dim_size=h_fine_skip.shape[0], reduce='sum')

        h_up = self.node_mlp(torch.cat([h_fine_skip, agg], dim=-1))
        return h_up


class AdaLNZero(nn.Module):
    """AdaLN-Zero conditioning of one GnBlock on the global latent z.

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
        """(Re-)initialize to the identity map.

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


def apply_adaln(block, graph, modulation, z_per_node):
    """Run `block` on `graph` with AdaLN-Zero modulation from `z_per_node`.

    `graph.x` is overwritten with the modulated features (the block consumes
    them and returns a fresh Data), then the block's own update is re-attached
    to the UNMODULATED input through the gate.
    """
    shift, scale, gate = modulation(z_per_node)
    x_in = graph.x
    x_mod = x_in * (1.0 + scale) + shift
    graph.x = x_mod
    out = block(graph)
    out.x = x_in + gate * (out.x - x_mod)
    return out
