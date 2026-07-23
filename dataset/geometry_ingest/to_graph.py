"""Element connectivity -> unique undirected ``mesh_edge`` [2, E].

Matches the dataset contract: ``build_dataset.py`` writes each undirected edge
once (canonical (min, max)); the training loaders make it bidirectional for PyG.
Handles both triangle (surface, 3 edges/elem) and tet (volume, 6 edges/elem).
"""

from __future__ import annotations

import numpy as np

# Undirected edges of a linear element, keyed by node count.
_EDGE_PAIRS = {
    3: [(0, 1), (1, 2), (2, 0)],                          # tri3  (surface)
    4: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],  # tet4  (volume)
}


def edges_from_connectivity(conn: np.ndarray, nodes_per_elem: int) -> np.ndarray:
    """conn [C, k] node indices -> mesh_edge [2, E] int64 (unique, undirected)."""
    pairs = _EDGE_PAIRS[nodes_per_elem]
    stacked = np.vstack([conn[:, [a, b]] for a, b in pairs])
    stacked = np.sort(stacked, axis=1)          # canonical (min, max) per edge
    unique = np.unique(stacked, axis=0)
    return unique.T.astype(np.int64)            # [2, E]
