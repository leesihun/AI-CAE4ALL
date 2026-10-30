"""
Gradient Checkpointing Utilities for MeshGraphNets

Reduces VRAM usage by recomputing activations during backward pass
instead of storing them. Trades ~20-30% more compute for ~60-70% less memory.

Usage:
    Set `use_checkpointing True` in config.txt to enable.
"""

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint
from torch_geometric.data import Data

from model.blocks import adaln_tensors


def checkpoint(fn, *args):
    """Non-reentrant checkpoint of a region that draws no random numbers.

    `preserve_rng_state=False` (as PhysicsNeMo's processor does): nothing
    checkpointed in this model samples, so stashing and restoring the CPU and
    CUDA RNG around every region only costs time. A region that samples
    (dropout, noise) must not go through here.
    """
    return _checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)


def checkpoint_gn_block(block, x, edge_attr, edge_index, world_edge_attr=None, world_edge_index=None):
    """`block.forward_tensors` under gradient checkpointing: (x, edge_attr, world_edge_attr)."""
    return checkpoint(block.forward_tensors, x, edge_attr, edge_index,
                      world_edge_attr, world_edge_index)


def checkpoint_adaln_block(block, modulation, x, z_per_node, edge_attr, edge_index,
                           world_edge_attr=None, world_edge_index=None):
    """AdaLN-Zero-conditioned GnBlock under gradient checkpointing.

    The modulation heads run INSIDE the checkpointed region on purpose: their
    (shift, scale, gate) are three [N, latent_dim] tensors per block, so keeping
    them out would add ~3x the node-feature activation for all ~28 blocks.
    Recomputing them on backward is a Linear(z_dim -> 3D) — negligible.
    """
    def run_block(modulation, x, z_per_node, edge_attr, edge_index, world_edge_attr, world_edge_index):
        return adaln_tensors(block, modulation, z_per_node, x, edge_attr, edge_index,
                             world_edge_attr, world_edge_index)

    return checkpoint(run_block, modulation, x, z_per_node, edge_attr, edge_index,
                      world_edge_attr, world_edge_index)


def run_blocks(blocks, x, edge_attr, edge_index, world_edge_attr=None, world_edge_index=None,
               z_fusers=None, z_per_node=None, adaln=False, checkpointed=False):
    """GnBlocks on plain tensors, one checkpoint per block when `checkpointed`.

    z_fusers: optional per-block VAE conditioning — Linear([x, z]) fusers when
    adaln=False, AdaLNZero heads when True — with z_per_node [N, z_dim].
    Returns (x, edge_attr, world_edge_attr).
    """
    conditioned = z_fusers is not None and z_per_node is not None
    for i, block in enumerate(blocks):
        if conditioned and adaln:
            if checkpointed:
                x, edge_attr, world_edge_attr = checkpoint_adaln_block(
                    block, z_fusers[i], x, z_per_node, edge_attr, edge_index,
                    world_edge_attr, world_edge_index)
            else:
                x, edge_attr, world_edge_attr = adaln_tensors(
                    block, z_fusers[i], z_per_node, x, edge_attr, edge_index,
                    world_edge_attr, world_edge_index)
            continue
        # Concat-based VAE conditioning keeps z on the reconstruction path.
        if conditioned:
            x = z_fusers[i](torch.cat([x, z_per_node], dim=-1))
        if checkpointed:
            x, edge_attr, world_edge_attr = checkpoint_gn_block(
                block, x, edge_attr, edge_index, world_edge_attr, world_edge_index)
        else:
            x, edge_attr, world_edge_attr = block.forward_tensors(
                x, edge_attr, edge_index, world_edge_attr, world_edge_index)
    return x, edge_attr, world_edge_attr


def run_graph_blocks(blocks, graph, z_fusers=None, z_per_node=None, adaln=False,
                     checkpointed=False):
    """`run_blocks` on a Data graph: one Data in, one Data out, none per block.

    World edges are carried only when the graph has them; the output carries
    them exactly when the input did.
    """
    world_edge_attr = getattr(graph, 'world_edge_attr', None)
    world_edge_index = getattr(graph, 'world_edge_index', None)
    x, edge_attr, world_edge_attr = run_blocks(
        blocks, graph.x, graph.edge_attr, graph.edge_index, world_edge_attr, world_edge_index,
        z_fusers=z_fusers, z_per_node=z_per_node, adaln=adaln, checkpointed=checkpointed)
    out = Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
    if world_edge_index is not None:
        out.world_edge_attr = world_edge_attr
        out.world_edge_index = world_edge_index
    return out


def process_with_checkpointing(processor_list, graph, z_fusers=None, z_per_node=None,
                               adaln=False):
    """Processor blocks with one gradient checkpoint per block (see run_graph_blocks)."""
    return run_graph_blocks(processor_list, graph, z_fusers, z_per_node, adaln, checkpointed=True)


def checkpoint_active(module):
    """True when `module` should checkpoint: flag on, training, autograd recording."""
    return (getattr(module, 'use_checkpointing', False) and module.training
            and torch.is_grad_enabled())


def run_mlp(mlp, x, checkpointed):
    """An encoder MLP, checkpointed when asked.

    On the fine mesh an edge encoder keeps five [E, latent_dim] tensors for
    backward (two Linear outputs, two SiLU outputs, the pre-LayerNorm output);
    checkpointed it keeps only its raw [E, edge_var] input.
    """
    if not checkpointed:
        return mlp(x)
    return checkpoint(mlp, x)


def run_plain_blocks(blocks, graph, checkpointed):
    """Unconditioned GnBlocks without world edges, one checkpoint per block.

    The trunks of the posterior encoder and the conditional prior. Both run on
    the fine mesh outside the processor, so `process_with_checkpointing` never
    covered them.
    """
    return run_graph_blocks(blocks, graph, checkpointed=checkpointed)
