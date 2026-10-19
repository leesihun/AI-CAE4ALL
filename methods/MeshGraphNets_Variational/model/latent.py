"""Latent placement for MGN-V: one per-graph z, one z per coarse node, or both.

`latent_type` picks where the VAE latent lives on the V-cycle:

  global  z_g [B, num_z, D]   one vector per graph and per level slot (HI-MGN-V).
                              Travels as a PLAIN tensor -- the code path is the
                              historical one, bit for bit.
  local   z_l [n_c, C]        C channels on every coarsest-level node
                              (HI-MGNflow's placement).
  hybrid  both                z_g as above plus z_l (HI-MGN-hybrid).

local and hybrid travel as a `Latent(g, l)`. The coarse-node count n_c differs
from graph to graph, so z_l cannot be stacked into a [B, ...] tensor: it is one
[n_c_total, C] block for the whole PyG batch, in the same row order as the
coarsest level's nodes. A cloud of n draws is `Latent(g=[B, n, num_z, D],
l=[n_c_total, n, C])` and `select_draw` slices one draw out of either form.
"""
from typing import NamedTuple, Optional

import torch
from torch_geometric.utils import scatter

LATENT_TYPES = ('global', 'local', 'hybrid')
LATENT_REGS = ('mmd', 'kl')


class Latent(NamedTuple):
    g: Optional[torch.Tensor]
    l: Optional[torch.Tensor]

    def to(self, *args, **kwargs):
        return Latent(*(None if t is None else t.to(*args, **kwargs) for t in self))


def lmap(fn, *xs):
    """Apply `fn` part-wise. Plain tensors in, plain tensor out."""
    if isinstance(xs[0], Latent):
        return Latent(*(None if parts[0] is None else fn(*parts) for parts in zip(*xs)))
    return fn(*xs)


def select_draw(cloud, s):
    """Draw s of a sample cloud: [B, n, ...] -> [B, ...] (z_l: [rows, n, C])."""
    if isinstance(cloud, Latent):
        return Latent(None if cloud.g is None else cloud.g[:, s],
                      None if cloud.l is None else cloud.l[:, s])
    return cloud[:, s]


def latent_parts(z):
    """[(name, tensor)] of the parts present -- 'z' for a plain global tensor."""
    if isinstance(z, Latent):
        return [(k, t) for k, t in zip(('g', 'l'), z) if t is not None]
    return [('z', z)]


def coarsest_parent_maps(level_data):
    """maps[i]: level-i node -> coarsest-level node, for every level i < L.

    `ftc` of level i sends level-i nodes to level i+1, so the coarsest ancestor
    of a level-i node is the composition ftc_{L-1}[... ftc_i].
    """
    L = len(level_data)
    maps = {}
    for i in range(L - 1, -1, -1):
        ftc = level_data[i]['ftc']
        maps[i] = ftc if i == L - 1 else maps[i + 1][ftc]
    return maps


def coarsest_view(graph, levels):
    """Coarsest-level topology read straight off a batched graph.

    Returns {'parent': [N] fine node -> coarsest node, 'n_c', 'c_ei', 'c_ea',
    'batch': [n_c] coarsest node -> graph}. The posterior encoder and the
    coarse prior both need it and neither runs the V-cycle, so this reads the
    same `fine_to_coarse_{i}` / `coarse_*_{i}` attributes the model does.
    """
    level_data = {}
    for i in range(int(levels)):
        key = f'fine_to_coarse_{i}'
        if not hasattr(graph, key):
            break
        level_data[i] = {'ftc': graph[key]}
    if not level_data:
        raise RuntimeError(
            "latent_type local/hybrid needs the multiscale hierarchy on the graph "
            "(fine_to_coarse_0 ...), but the batch carries none.")
    last = len(level_data) - 1
    parent = coarsest_parent_maps(level_data)[0]
    n_c = graph[f'coarse_centroid_{last}'].shape[0]
    batch = getattr(graph, 'batch', None)
    if batch is None:
        batch = torch.zeros(graph.x.shape[0], dtype=torch.long, device=graph.x.device)
    return {
        'parent': parent,
        'n_c': n_c,
        'c_ei': graph[f'coarse_edge_index_{last}'],
        'c_ea': graph[f'coarse_edge_attr_{last}'],
        'batch': scatter(batch, parent, dim=0, dim_size=n_c, reduce='min'),
    }


def kl_to_standard_normal(mu, logvar):
    """Per-sample KL(q || N(0, I)), averaged over every latent coordinate.

    A per-dimension MEAN rather than the textbook sum, so one `kl_weight` means
    the same pressure per coordinate whether the latent has num_z*D global
    entries or C channels on every coarse node.
    """
    mu = mu.float()
    logvar = logvar.float()
    return (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
