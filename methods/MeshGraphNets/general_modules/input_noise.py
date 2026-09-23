"""Training-input noise, injected in the order the MGN paper specifies.

Pfaff et al., "Learning Mesh-Based Simulation with Graph Networks"
(arXiv:2010.03409): the noise is added to the *dynamical variable* and the
graph -- node **and** edge features -- is then built from the perturbed state,
so the relative-displacement edge attributes see the perturbed geometry. The
targets are corrected by the same noise so that the decoder output, after
integration, cancels the perturbation at the input.

What this module does, in that order:

1.  draw `noise ~ N(0, std_noise)` over the leading `output_var` node channels
    (normalized units -- see below) and add it to `graph.x`;
2.  correct the target, `y -= noise_gamma * noise * noise_std_ratio`, where
    `noise_std_ratio = node_std / delta_std` converts the perturbation from
    node-normalized units into delta-normalized units;
3.  map the same draw through `displacement_state_indices` to a *physical*
    node displacement and rebuild the deformed half of every edge feature
    (mesh edges and world edges) from the perturbed positions.

**`std_noise` is in normalized (z-scored) units, by design.** The physical
sigma of channel `i` is `std_noise * node_std[i]`; step 2 is only dimensionally
correct under that reading, and `noise_std_ratio` exists precisely to perform
the conversion. A single scalar therefore sets a per-field sigma proportional
to each field's own spread, which is not the same thing as the paper's
independently chosen per-field absolute sigmas.

Before this module existed, step 3 was instead an *independent* Gaussian of
the same std added to all eight normalized edge channels. That perturbed the
reference half -- fixed geometry the paper always keeps exact -- applied a
node-normalized scale to edge-normalized channels, and was uncorrelated with
the node draw, so no edge feature agreed with the node state the model saw.

Deliberate deviations, both bounded and both documented rather than hidden:

* World-edge *connectivity* is not re-searched under noise. The radius search
  is the single most expensive operation on a contact-heavy graph (~200 ms per
  call at 200k nodes), and at the noise magnitudes used here it changes the
  contact set only marginally. The world-edge *features* are perturbed.
* Coarse (multiscale) edge features are left untouched, as they were before
  this module existed: coarse positions are pooled from fine positions, so a
  faithful update means re-pooling the whole hierarchy per batch.
"""

import numpy as np
import torch

from general_modules.edge_features import DEFORMED_FEATURE_DIM
from general_modules.state_geometry import displacement_from_state, displacement_indices

# Same guard `edge_features.deformed_edge_attr_torch` uses for d||r||/dr at r=0.
_DIST_EPS = 1e-12
_POS_DIM = DEFORMED_FEATURE_DIM - 1  # dx, dy, dz; the 4th channel is the norm


class InputNoise:
    """Callable that applies the paper's training noise to one graph in place.

    Built once per model (the normalization constants are constants of the
    run) and cached per (device, dtype) so the hot loop does no host-to-device
    copies.
    """

    def __init__(self, config):
        self.config = config
        self.std = float(config.get('std_noise', 0.0) or 0.0)
        self.enabled = self.std > 0.0
        self.output_var = int(config['output_var'])
        self.input_var = int(config['input_var'])
        # Full correction (gamma 1.0) is what a first-order system wants and is
        # what the reference implementation uses for its fluid datasets; the
        # paper's gamma 0.1 is specific to second-order cloth, where the
        # correction is split between position and velocity.
        self.gamma = float(config.get('noise_gamma', 1.0))
        # A state that cannot move geometry (geometry_state_mode fixed, e.g. a
        # fluid on a rigid mesh) perturbs no edge feature at all -- which is
        # exactly what the paper does for CylinderFlow, where the noise is on
        # momentum and the mesh never moves.
        self.moves_geometry = any(
            0 <= idx < self.output_var
            for idx in displacement_indices(config, self.input_var)
        )
        self._cache = {}

    # -- constants -----------------------------------------------------------

    def _constants(self, device, dtype):
        key = (str(device), dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        stats = self.config.get('_norm_stats', None)
        if stats is None:
            raise RuntimeError(
                "std_noise > 0 needs dataset normalization stats in "
                "config['_norm_stats'] so the perturbed state can be mapped back "
                "into physical units; they are injected by "
                "training_profiles.setup.build_dataset_splits."
            )

        def to_tensor(value):
            return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)

        node_std = to_tensor(stats['node_std'])[:self.output_var]
        edge_mean = to_tensor(stats['edge_mean'])
        edge_std = to_tensor(stats['edge_std'])
        ratio = self.config.get('noise_std_ratio', None)
        constants = {
            'node_std': node_std,
            'rel_mean': edge_mean[:_POS_DIM],
            'rel_std': edge_std[:_POS_DIM],
            'dist_mean': edge_mean[_POS_DIM:DEFORMED_FEATURE_DIM],
            'dist_std': edge_std[_POS_DIM:DEFORMED_FEATURE_DIM],
            'ratio': None if ratio is None else to_tensor(ratio),
        }
        self._cache[key] = constants
        return constants

    # -- pieces --------------------------------------------------------------

    def _physical_displacement(self, noise, node_std):
        """Physical [N, 3] node displacement implied by a normalized draw."""
        physical = noise * node_std                       # [N, output_var]
        if self.input_var > self.output_var:
            pad = torch.zeros(
                physical.shape[0], self.input_var - self.output_var,
                device=physical.device, dtype=physical.dtype,
            )
            physical = torch.cat([physical, pad], dim=1)
        return displacement_from_state(physical, self.config, self.input_var)

    @staticmethod
    def _perturb_edges(edge_attr, edge_index, disp, c):
        """Rebuild the deformed half of `edge_attr` from perturbed positions."""
        if edge_attr is None or edge_attr.numel() == 0:
            return edge_attr
        src_idx, dst_idx = edge_index[0], edge_index[1]
        rel = edge_attr[:, :_POS_DIM] * c['rel_std'] + c['rel_mean']
        rel = rel + (disp[dst_idx] - disp[src_idx])
        dist = torch.sqrt((rel * rel).sum(dim=1, keepdim=True) + _DIST_EPS)
        return torch.cat([
            (rel - c['rel_mean']) / c['rel_std'],
            (dist - c['dist_mean']) / c['dist_std'],
            edge_attr[:, DEFORMED_FEATURE_DIM:],   # reference half: exact, never noised
        ], dim=1)

    # -- entry point ---------------------------------------------------------

    def __call__(self, graph):
        if not self.enabled:
            return graph
        c = self._constants(graph.x.device, graph.x.dtype)

        noise = torch.randn(
            graph.x.shape[0], self.output_var,
            device=graph.x.device, dtype=graph.x.dtype,
        ) * self.std
        noise_padded = torch.zeros_like(graph.x)
        noise_padded[:, :self.output_var] = noise
        graph.x = graph.x + noise_padded

        if c['ratio'] is not None and getattr(graph, 'y', None) is not None:
            graph.y = graph.y - self.gamma * noise * c['ratio']

        if not self.moves_geometry:
            return graph

        disp = self._physical_displacement(noise, c['node_std'])
        graph.edge_attr = self._perturb_edges(graph.edge_attr, graph.edge_index, disp, c)
        world_attr = getattr(graph, 'world_edge_attr', None)
        world_index = getattr(graph, 'world_edge_index', None)
        if world_attr is not None and world_index is not None and world_attr.numel() > 0:
            graph.world_edge_attr = self._perturb_edges(world_attr, world_index, disp, c)
        return graph
