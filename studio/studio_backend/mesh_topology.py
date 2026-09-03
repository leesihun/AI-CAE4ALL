"""Topology-preserving mesh reduction for the Studio artifact viewer.

The viewer has to show *real* connectivity.  Striding an edge list (keeping
every n-th edge) produces disconnected confetti rather than a mesh, so an
oversized mesh is reduced here by **vertex clustering**: nodes are merged on a
uniform grid and every surviving edge is kept and remapped.  The result is a
coarser but genuinely connected mesh.

The shared mesh HDF5 contract stores no cell connectivity - only
``data/{id}/mesh_edge`` - so elements are reconstructed from the edge graph:
3-cliques recover triangles (the suite's warpage/FE datasets are triangular
surface meshes) and 4-cycles recover quads.  That is what lets the viewer draw
an opaque element-coloured field instead of a wireframe.
"""

from __future__ import annotations

from typing import Any

MAX_GRID = 2048
MAX_SLOT = 24
MIN_BUDGET = 1500


def _np():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("numpy is required for mesh visualization.") from exc
    return np


def sanitize_coordinates(raw: Any) -> Any:
    """Return an ``[N,3]`` float array with every non-finite node recentred."""
    np = _np()
    coords = np.asarray(raw, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        coords = coords.reshape(-1, 3)
    broken = ~np.isfinite(coords).all(axis=1)
    if broken.any():
        healthy = coords[~broken]
        centre = healthy.mean(axis=0) if healthy.size else np.zeros(3)
        coords = coords.copy()
        coords[broken] = centre
    return coords


def normalize_edges(raw: Any, node_count: int) -> Any:
    """Return unique, in-range, undirected edges as an ``[E,2]`` index array."""
    np = _np()
    edges = np.asarray(raw)
    if edges.ndim != 2:
        return np.empty((0, 2), dtype=np.int64)
    if edges.shape[0] == 2 and edges.shape[1] != 2:
        edges = edges.T
    if edges.shape[1] < 2:
        return np.empty((0, 2), dtype=np.int64)
    edges = np.asarray(edges[:, :2], dtype=np.int64)
    edges = edges[np.all((edges >= 0) & (edges < node_count), axis=1)]
    low = np.minimum(edges[:, 0], edges[:, 1])
    high = np.maximum(edges[:, 0], edges[:, 1])
    keep = low != high
    keys = np.unique(low[keep] * node_count + high[keep])
    return np.stack([keys // node_count, keys % node_count], axis=1)


def _cell_keys(coords: Any, low: Any, span: Any, resolution: int) -> Any:
    np = _np()
    cell = np.minimum(((coords - low) / span * resolution).astype(np.int64), resolution - 1)
    cell = np.maximum(cell, 0)
    return (cell[:, 0] * resolution + cell[:, 1]) * resolution + cell[:, 2]


def _cluster_assignment(coords: Any, target: int) -> Any:
    """Map every node to a grid cluster, aiming for ``target`` clusters.

    The occupied-cell count grows like ``resolution ** p`` where ``p`` is near 2
    for surface meshes and near 3 for solid ones, so ``p`` is measured from two
    cheap probes instead of assumed.
    """
    np = _np()
    low = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - low, 1e-12)

    coarse = int(np.unique(_cell_keys(coords, low, span, 8)).size)
    fine = int(np.unique(_cell_keys(coords, low, span, 32)).size)
    if fine <= coarse:
        exponent = 3.0
    else:
        exponent = min(3.0, max(1.2, float(np.log(fine / coarse) / np.log(4.0))))
    resolution = int(round(32 * (target / max(fine, 1)) ** (1.0 / exponent)))
    resolution = int(min(MAX_GRID, max(2, resolution)))

    keys = _cell_keys(coords, low, span, resolution)
    unique, inverse = np.unique(keys, return_inverse=True)
    if unique.size > target * 1.15 and resolution > 2:
        resolution = int(max(2, resolution * (target / unique.size) ** (1.0 / exponent)))
        keys = _cell_keys(coords, low, span, resolution)
        unique, inverse = np.unique(keys, return_inverse=True)
    return np.asarray(inverse, dtype=np.int64).reshape(-1), int(unique.size)


def aggregate_values(inverse: Any, cluster_count: int, values: Any) -> Any:
    """Average node values inside each cluster, ignoring non-finite nodes."""
    np = _np()
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if inverse is None:
        return flat
    finite = np.isfinite(flat)
    weight = np.bincount(inverse, weights=finite.astype(np.float64), minlength=cluster_count)
    total = np.bincount(inverse, weights=np.where(finite, flat, 0.0), minlength=cluster_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / np.maximum(weight, 1e-12), np.nan)


def _adjacency(edges: Any, node_count: int, directed_by_rank: bool) -> tuple[Any, Any, Any]:
    """Build a CSR neighbour table; optionally orient each edge by node degree."""
    np = _np()
    if directed_by_rank:
        degree = np.bincount(edges.reshape(-1), minlength=node_count)
        rank = np.empty(node_count, dtype=np.int64)
        rank[np.argsort(degree, kind="stable")] = np.arange(node_count)
        swap = rank[edges[:, 0]] > rank[edges[:, 1]]
        source = np.where(swap, edges[:, 1], edges[:, 0])
        target = np.where(swap, edges[:, 0], edges[:, 1])
    else:
        source = np.concatenate([edges[:, 0], edges[:, 1]])
        target = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.lexsort((target, source))
    source, target = source[order], target[order]
    counts = np.bincount(source, minlength=node_count)
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return counts, offsets, target


def _edge_lookup(edges: Any, node_count: int):
    np = _np()
    keys = np.sort(edges[:, 0] * node_count + edges[:, 1])

    def exists(left: Any, right: Any) -> Any:
        if keys.size == 0:
            return np.zeros(np.shape(left), dtype=bool)
        low = np.minimum(left, right)
        high = np.maximum(left, right)
        probe = low * node_count + high
        position = np.minimum(np.searchsorted(keys, probe), keys.size - 1)
        return keys[position] == probe

    return exists


def _triangles(edges: Any, node_count: int, limit: int) -> Any:
    """Enumerate 3-cliques - the elements of a triangular surface mesh."""
    np = _np()
    counts, offsets, neighbours = _adjacency(edges, node_count, directed_by_rank=True)
    exists = _edge_lookup(edges, node_count)
    slots = int(min(counts.max(initial=0), MAX_SLOT))
    found: list[Any] = []
    total = 0
    for first in range(slots - 1):
        anchors = np.nonzero(counts > first)[0]
        if anchors.size == 0:
            break
        left = neighbours[offsets[anchors] + first]
        for second in range(first + 1, slots):
            keep = counts[anchors] > second
            if not keep.any():
                break
            apex = anchors[keep]
            right = neighbours[offsets[apex] + second]
            hit = exists(left[keep], right)
            if not hit.any():
                continue
            found.append(np.stack([apex[hit], left[keep][hit], right[hit]], axis=1))
            total += int(hit.sum())
            if total > limit:
                return np.concatenate(found, axis=0)[:limit]
    return np.concatenate(found, axis=0) if found else np.empty((0, 3), dtype=np.int64)


def _quads(edges: Any, node_count: int, limit: int) -> Any:
    """Enumerate 4-cycles as two triangles each - quad and hex mesh elements."""
    np = _np()
    counts, offsets, neighbours = _adjacency(edges, node_count, directed_by_rank=False)
    exists = _edge_lookup(edges, node_count)
    slots = int(min(counts.max(initial=0), MAX_SLOT))
    keys: list[Any] = []
    midpoints: list[Any] = []
    for first in range(slots - 1):
        anchors = np.nonzero(counts > first)[0]
        if anchors.size == 0:
            break
        left = neighbours[offsets[anchors] + first]
        for second in range(first + 1, slots):
            keep = counts[anchors] > second
            if not keep.any():
                break
            apex = anchors[keep]
            right = neighbours[offsets[apex] + second]
            corner = left[keep]
            open_pair = ~exists(corner, right)
            if not open_pair.any():
                continue
            low = np.minimum(corner[open_pair], right[open_pair])
            high = np.maximum(corner[open_pair], right[open_pair])
            keys.append(low * node_count + high)
            midpoints.append(apex[open_pair])
    if not keys:
        return np.empty((0, 3), dtype=np.int64)

    pair_key = np.concatenate(keys)
    midpoint = np.concatenate(midpoints)
    order = np.argsort(pair_key, kind="stable")
    pair_key, midpoint = pair_key[order], midpoint[order]
    # Two diagonal corners sharing consecutive midpoints close one quad.
    shared = pair_key[1:] == pair_key[:-1]
    if not shared.any():
        return np.empty((0, 3), dtype=np.int64)
    diagonal = pair_key[:-1][shared]
    first_mid = midpoint[:-1][shared]
    second_mid = midpoint[1:][shared]
    # Every quad is reachable from either of its two diagonals; keep the one
    # with the smaller key so each element is emitted exactly once.
    mirror = (
        np.minimum(first_mid, second_mid) * node_count
        + np.maximum(first_mid, second_mid)
    )
    unique_quad = (diagonal < mirror) | exists(first_mid, second_mid)
    diagonal = diagonal[unique_quad]
    first_mid = first_mid[unique_quad]
    second_mid = second_mid[unique_quad]
    corner = diagonal // node_count
    opposite = diagonal % node_count
    quads = np.stack(
        [
            np.stack([corner, first_mid, opposite], axis=1),
            np.stack([corner, opposite, second_mid], axis=1),
        ],
        axis=1,
    ).reshape(-1, 3)
    return quads[:limit]


def extract_faces(edges: Any, node_count: int, limit: int) -> tuple[Any, str]:
    """Recover mesh elements from the edge graph, preferring triangles."""
    np = _np()
    if edges.shape[0] == 0 or node_count < 3:
        return np.empty((0, 3), dtype=np.int64), "none"
    triangles = _triangles(edges, node_count, limit)
    if triangles.shape[0] >= max(4, node_count // 8):
        return triangles, "triangle"
    quads = _quads(edges, node_count, limit)
    if quads.shape[0] > triangles.shape[0]:
        return quads, "quad"
    return triangles, "triangle" if triangles.shape[0] else "none"


def build_topology(
    raw_coordinates: Any,
    raw_edges: Any,
    *,
    node_budget: int = 16000,
    edge_limit: int = 60000,
    face_limit: int = 40000,
) -> dict[str, Any]:
    """Reduce a mesh to a drawable budget without ever breaking connectivity.

    Returns the surviving node coordinates, the remapped edge list, the
    reconstructed elements, and the node->cluster map needed to aggregate a
    field onto the reduced nodes.
    """
    np = _np()
    coordinates = sanitize_coordinates(raw_coordinates)
    node_count = int(coordinates.shape[0])
    edges = normalize_edges(raw_edges, node_count) if raw_edges is not None else np.empty((0, 2), np.int64)
    total_edges = int(edges.shape[0])

    budget = int(max(MIN_BUDGET, node_budget))
    for _ in range(3):
        if node_count > budget:
            inverse, cluster_count = _cluster_assignment(coordinates, budget)
            counts = np.bincount(inverse, minlength=cluster_count).astype(np.float64)
            nodes = np.stack(
                [
                    np.bincount(inverse, weights=coordinates[:, axis], minlength=cluster_count) / counts
                    for axis in range(3)
                ],
                axis=1,
            )
            reduced_edges = normalize_edges(inverse[edges], cluster_count) if total_edges else edges
        else:
            inverse, cluster_count = None, node_count
            nodes, reduced_edges = coordinates, edges

        faces, element_kind = extract_faces(reduced_edges, cluster_count, face_limit * 2)
        scale = min(
            1.0,
            face_limit / max(faces.shape[0], 1),
            edge_limit / max(reduced_edges.shape[0], 1),
        )
        if scale > 0.99 or budget <= MIN_BUDGET:
            break
        # Truncating a face list punches holes in the surface, so shrink the
        # node budget instead and rebuild a complete - if coarser - mesh.
        budget = int(max(MIN_BUDGET, min(budget, cluster_count) * scale))

    return {
        "inverse": inverse,
        "node_count": int(cluster_count),
        "total_nodes": node_count,
        "coordinates": nodes,
        "edges": reduced_edges[:edge_limit],
        "total_edges": total_edges,
        "faces": faces[:face_limit],
        "total_faces": int(faces.shape[0]),
        "element_kind": element_kind,
        "reduced": inverse is not None,
    }
