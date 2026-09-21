"""LDGN-style compressor + coarse-latent flow prior for cHI-MGNflow.

Two-stage architecture (Lino, Pfaff & Thuerey, "Learning Distributions of
Complex Fluid Simulations with Diffusion Graph Networks", ICLR 2025,
arXiv:2504.02843; "LDGN" is that paper's own name for its compressed-latent
variant), built on this repo's own HI-MGN
V-cycle instead of the paper's own multiscale backbone:

    Stage 1 -- compressor (HierarchyEncoder + MultiscaleDecoder below):
        a near-unregularized multiscale autoencoder that compresses the TRUE
        field graph.y onto the COARSEST mesh level as a thin per-node latent
        z (`latent_ch` channels per coarse node -- not one pooled vector),
        trained by reconstruction + a near-zero KL against N(0, I).

    Stage 2 -- prior (LatentFlowPrior below):
        a flow-matching velocity field trained ONLY inside that small coarse
        latent space, on a FROZEN compressor, conditioned on geometry/BC
        features that are available even when y is unknown (generation
        time). See CHiMGNFlow.py's `train_prior` mode.

The paper's own ablation (a VGAE built on the same multiscale shape, with a
heavier KL and a coarser latent) still collapses -- proof that hierarchy alone
does not explain LDGN's win. The decisive lever is training a prior instead of
sampling N(0, I) directly. Field-space flow matching (this repo's PREVIOUS
cHI-MGNflow) is exactly the class of approach that ablation loses to.

Two-pass shared encoder (the skip-connection leakage fix)
-----------------------------------------------------------
A naive retrofit that reuses the V-cycle's ordinary fine-level skip states for
an autoencoder's decoder lets the decoder reconstruct the field from the skip
path alone, bypassing z entirely -- precisely the paper's own VGAE-collapse
mechanism. HierarchyEncoder is therefore called TWICE per step with different
node inputs, sharing one set of weights:

    outs_y = hierarchy_encoder(x=[state | graph.y      | cond | pos | ...])
    outs_0 = hierarchy_encoder(x=[state | zeros_like(y) | cond | pos | ...])

outs_y's coarsest output feeds (mu, logvar). ALL of outs_0's per-level outputs
feed the decoder's skip connections and the prior's conditioning, so the
decoder structurally never sees a y-derived feature outside of z -- and the
"blind" pass behaves identically at train time and at generation time, since y
is always absent at generation time anyway.
"""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from model.blocks import AdaLNZero, UnpoolBlock, apply_adaln
from model.checkpointing import process_with_checkpointing
from model.coarsening import pool_features
from model.encoder_decoder import Decoder, Encoder, GnBlock
from model.flow import TimeEmbedding
from model.mlp import build_mlp


def extract_level_data(graph, L):
    """Per-level coarsening topology, read before any encoder pass runs.

    Reads the cached MultiscaleData attributes attach_coarse_levels_to_graph
    wrote onto `graph` (fine_to_coarse_i, coarse_edge_index_i, ...) -- these
    describe MESH topology and are independent of whatever node features a
    caller is about to splice into `graph.x`, so one extraction serves both
    the y-aware and the y-blind HierarchyEncoder pass.
    """
    level_data = {}
    for i in range(L):
        ftc_key = f'fine_to_coarse_{i}'
        if not hasattr(graph, ftc_key):
            break
        centroid = graph[f'coarse_centroid_{i}']
        ld = {
            'ftc': graph[ftc_key],
            'c_ei': graph[f'coarse_edge_index_{i}'],
            'c_ea': graph[f'coarse_edge_attr_{i}'],
            # Read the coarse node count off a shape, not off the GPU: an
            # int(sum()) here forces a CPU<->GPU sync per level per forward,
            # which under K-step integration is paid K times.
            'n_c': centroid.shape[0],
            'c_we_idx': getattr(graph, f'coarse_world_edge_index_{i}', None),
            'c_we_attr': getattr(graph, f'coarse_world_edge_attr_{i}', None),
            'up_ei': graph[f'unpool_edge_index_{i}'],
            'coarse_centroid': centroid,
            'fine_pos': graph.pos if i == 0 else graph[f'coarse_centroid_{i - 1}'],
        }
        seed_key = f'coarse_seed_idx_{i}'
        if hasattr(graph, seed_key) and graph[seed_key] is not None:
            ld['seeds'] = graph[seed_key]
        level_data[i] = ld
    return level_data


def _run_plain(blocks, graph, use_checkpointing, training):
    if use_checkpointing and training:
        return process_with_checkpointing(blocks, graph)
    for block in blocks:
        graph = block(graph)
    return graph


class HierarchyEncoder(nn.Module):
    """V-cycle DESCENDING arm: fine graph -> per-level features -> coarsest.

    Plain -- no time/AdaLN conditioning. Called twice per AE training step
    with the same weights (see module docstring). Returns `(per_level,
    coarse_batch)`: `per_level` has L+1 entries in fine-to-coarse order --
    entries 0..L-1 are the PRE-POOL features at each level (skip source for
    MultiscaleDecoder), entry L is the coarsest-level graph itself (posterior
    head input / prior conditioning). `coarse_batch` is the PyG batch index at
    the coarsest level.
    """

    def __init__(self, edge_input_size, node_input_size, latent_dim,
                 multiscale_levels, mp_per_level, use_world_edges=False,
                 use_coarse_world_edges=False, use_checkpointing=False):
        super().__init__()
        L = multiscale_levels
        self.L = L
        self.use_world_edges = use_world_edges
        self.use_coarse_world_edges = use_coarse_world_edges
        self.use_checkpointing = use_checkpointing

        self.encoder = Encoder(edge_input_size, node_input_size, latent_dim,
                               use_world_edges=use_world_edges)
        self.pre_blocks = nn.ModuleList()
        self.coarse_eb_encoders = nn.ModuleList()
        for i in range(L):
            use_we = use_world_edges if (i == 0 or use_coarse_world_edges) else False
            self.pre_blocks.append(nn.ModuleList([
                GnBlock(latent_dim, use_world_edges=use_we) for _ in range(mp_per_level[i])
            ]))
            self.coarse_eb_encoders.append(build_mlp(edge_input_size, latent_dim, latent_dim))
        self.coarsest_blocks = nn.ModuleList([
            GnBlock(latent_dim, use_world_edges=use_coarse_world_edges)
            for _ in range(mp_per_level[L])
        ])

    def forward(self, work_graph, level_data, batch):
        current_graph = self.encoder(work_graph)
        current_batch = batch
        per_level = []
        L = len(level_data)
        for i in range(L):
            current_graph = _run_plain(self.pre_blocks[i], current_graph,
                                       self.use_checkpointing, self.training)
            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            per_level.append({
                'x': current_graph.x,
                'edge_attr': current_graph.edge_attr,
                'edge_index': current_graph.edge_index,
                'w_attr': getattr(current_graph, 'world_edge_attr', None) if use_we_here else None,
                'w_idx': getattr(current_graph, 'world_edge_index', None) if use_we_here else None,
            })

            ld = level_data[i]
            if 'seeds' in ld:
                h_coarse = current_graph.x[ld['seeds']]
            else:
                h_coarse = pool_features(current_graph.x, ld['ftc'], ld['n_c'])
            e_coarse = self.coarse_eb_encoders[i](ld['c_ea'])
            current_graph = Data(x=h_coarse, edge_attr=e_coarse, edge_index=ld['c_ei'])
            if self.use_coarse_world_edges and ld['c_we_idx'] is not None and ld['c_we_idx'].shape[1] > 0:
                current_graph.world_edge_attr = ld['c_we_attr']
                current_graph.world_edge_index = ld['c_we_idx']

            current_batch = scatter(current_batch, ld['ftc'], dim=0,
                                    dim_size=ld['n_c'], reduce='min')

        current_graph = _run_plain(self.coarsest_blocks, current_graph,
                                   self.use_checkpointing, self.training)
        per_level.append({
            'x': current_graph.x,
            'edge_attr': current_graph.edge_attr,
            'edge_index': current_graph.edge_index,
            'w_attr': getattr(current_graph, 'world_edge_attr', None) if self.use_coarse_world_edges else None,
            'w_idx': getattr(current_graph, 'world_edge_index', None) if self.use_coarse_world_edges else None,
        })
        return per_level, current_batch


class MultiscaleDecoder(nn.Module):
    """V-cycle ASCENDING arm: coarsest z-fused graph -> fine field.

    `skip_levels` must be the y-BLIND HierarchyEncoder pass's per-level list,
    entries 0..L-1 (the coarsest entry is not read here -- the caller already
    fused it with z before invoking this module). Plain -- no AdaLN.
    """

    def __init__(self, latent_dim, node_output_size, multiscale_levels,
                 mp_per_level, use_world_edges=False, use_coarse_world_edges=False,
                 use_checkpointing=False):
        super().__init__()
        L = multiscale_levels
        self.L = L
        self.use_world_edges = use_world_edges
        self.use_coarse_world_edges = use_coarse_world_edges
        self.use_checkpointing = use_checkpointing

        self.unpool_blocks = nn.ModuleList([UnpoolBlock(latent_dim, build_mlp) for _ in range(L)])
        self.skip_projs = nn.ModuleList([nn.Linear(2 * latent_dim, latent_dim) for _ in range(L)])
        self.post_blocks = nn.ModuleList()
        for i in range(L):
            use_we = use_world_edges if (i == 0 or use_coarse_world_edges) else False
            self.post_blocks.append(nn.ModuleList([
                GnBlock(latent_dim, use_world_edges=use_we) for _ in range(mp_per_level[2 * L - i])
            ]))
        self.decoder = Decoder(latent_dim, node_output_size)

    def forward(self, coarsest_graph, level_data, skip_levels):
        current_graph = coarsest_graph
        L = len(level_data)
        for i in range(L - 1, -1, -1):
            ld = level_data[i]
            src, dst = ld['up_ei']
            rel_pos = ld['fine_pos'][dst] - ld['coarse_centroid'][src]
            h_up = self.unpool_blocks[i](
                h_coarse=current_graph.x,
                h_fine_skip=skip_levels[i]['x'],
                unpool_edge_index=ld['up_ei'],
                rel_pos=rel_pos,
            )
            skip = skip_levels[i]
            h_merged = self.skip_projs[i](torch.cat([skip['x'], h_up], dim=-1))
            current_graph = Data(x=h_merged, edge_attr=skip['edge_attr'], edge_index=skip['edge_index'])
            use_we_here = self.use_world_edges and (i == 0 or self.use_coarse_world_edges)
            if use_we_here and skip['w_attr'] is not None:
                current_graph.world_edge_attr = skip['w_attr']
                current_graph.world_edge_index = skip['w_idx']

            current_graph = _run_plain(self.post_blocks[i], current_graph,
                                       self.use_checkpointing, self.training)

        return self.decoder(current_graph)


class LatentFlowPrior(nn.Module):
    """Flow-matching velocity field over the coarsest-level latent z ONLY.

    Small and cheap by construction: it operates on the coarsest mesh level
    (tens-to-hundreds of nodes per graph), not the full mesh -- this is what
    makes the K-step ODE integration at generation time nearly free compared
    to the per-step full-mesh V-cycle the previous field-space flow model
    paid at every one of its K steps.
    """

    def __init__(self, latent_ch, latent_dim, time_freqs, prior_blocks,
                 use_checkpointing=False):
        super().__init__()
        self.latent_ch = latent_ch
        self.use_checkpointing = use_checkpointing
        self.in_proj = nn.Linear(latent_ch + latent_dim, latent_dim)
        self.blocks = nn.ModuleList([GnBlock(latent_dim) for _ in range(prior_blocks)])
        self.time_embed = TimeEmbedding(time_freqs)
        self.t_mod = nn.ModuleList([
            AdaLNZero(latent_dim, self.time_embed.dim) for _ in range(prior_blocks)
        ])
        self.out_head = nn.Linear(latent_dim, latent_ch)

    def reset_time_conditioning(self):
        for m in self.modules():
            if isinstance(m, AdaLNZero):
                m.reset_identity()

    def forward(self, z_t, t, cond_x, edge_attr, edge_index, coarse_batch):
        t_emb = self.time_embed(t).to(z_t.dtype)
        t_per_node = t_emb[coarse_batch]
        x_in = self.in_proj(torch.cat([z_t, cond_x], dim=-1))
        graph = Data(x=x_in, edge_attr=edge_attr, edge_index=edge_index)
        if self.use_checkpointing and self.training:
            graph = process_with_checkpointing(
                self.blocks, graph, z_fusers=self.t_mod, z_per_node=t_per_node, adaln=True)
        else:
            for j, block in enumerate(self.blocks):
                graph = apply_adaln(block, graph, self.t_mod[j], t_per_node)
        return self.out_head(graph.x)
