"""Signed distance computed from `mesh_edge` at load time (`sdf_source mesh`).

The SDF input follows GINO (arXiv:2309.00583, "the paper" below; its code is
"the reference implementation"), which feeds the geometry to its operator as
a signed distance field. None of this repository's datasets ship an SDF field,
and the contract in docs/reference/DATASET_FORMAT.md carries no faces or cells
-- only `mesh_edge (2, E)`. This module reconstructs the boundary from that
edge graph (or, where the edge graph cannot carry one, from the node cloud)
and evaluates a true signed distance at arbitrary query points, so an SDF
input needs no change to a single dataset file.

Three boundary routes, tried in that order, because they are ordered by how
much they *assume*:

``planar`` (2-D, exact)
    A 2-D mesh plus its node coordinates is a planar straight-line graph.
    Sorting each vertex's neighbours by angle gives a rotation system, and
    tracing it recovers the true faces of the embedding -- no cycle
    enumeration, so no "chord" triangle that bounds no element. The trace is
    self-verifying: V - E + F = 1 + C holds exactly for a planar embedding and
    for nothing else. The faces that are not elements (the negative-area outer
    face, and any hole) hand over the domain boundary directly.

``faces`` (3-D shell, exact)
    The mesh *is* the surface, so its 3- or 4-cycles are its faces; the shell
    is signed with its own orientation. `reconstruct_faces` is unchanged and
    still gated on edge/face incidence (`BoundaryQuality`).

``alpha`` (volume meshes and point clouds, estimated)
    A 3-D volume mesh stores no cells, so its boundary faces are not
    recoverable from the edge graph -- cycle enumeration returns interior cell
    faces and almost every edge carries three or more of them. The same is true
    of a 2-D dataset whose `mesh_edge` is really a k-nearest-neighbour graph
    (ex8_elasticity is literally k=6, and carries 3,360 > 3N-6 = 2,910 edges,
    so it has no planar embedding at all). There the boundary is *estimated*
    from the node cloud as an alpha complex: Delaunay-tessellate the nodes,
    keep the simplices whose circumradius is below alpha, and take the facets
    used by exactly one kept simplex. alpha is not a magic number here -- it is
    the smallest value whose complex still contains every node (any smaller
    alpha would leave a node outside its own body), widened by
    ALPHA_TIE_MARGIN to clear the floating-point ties a structured grid
    produces. An estimated boundary can pinch, and a pinched surface's normals
    invert the sign over whole regions, so in 2-D the sign is an even-odd ray
    crossing (orientation-free by construction) and in 3-D the surface's
    normals are used only after probes whose side is known by construction
    confirm them -- otherwise the sign falls back to exact Delaunay point
    location. See `_alpha_normal_sign`.

What the routes pick on the shipped arms (first sample of each):

    ex1 ex4 ex6 ex7 ex9   planar, Euler exact, 670/234/128/2,536/260 segments
    ex3_full ex3_mid ex10 faces (3-D shell), 0.03-0.6% non-manifold
    ex8                   alpha 2-D: k=6 neighbour graph, no planar embedding
    ex2 ex5               alpha 3-D: volume meshes, 93%/68% non-manifold

The reconstruction depends only on the reference coordinates (`nodal_data`
rows 0:3), which are static for the whole trajectory, so the result is computed
once per sample and cached rather than per epoch or per timestep.

Sign convention: **negative inside the meshed domain**. On a shell (ex3_full,
ex10) the meshed domain is the solid body, so this matches the paper, where the
SDF is negative inside the car. On a 2-D flow mesh the meshed domain is the
*fluid* and the obstacle is a hole in it, so the values there are the negation
of the paper's body-centred convention -- ex7_airfrans reads negative
throughout the flow and positive inside the airfoil. That is a constant sign
flip, which the lifting layer absorbs, but it is a real difference from the
reference implementation and is recorded rather than quietly normalized away.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


# A mesh whose face reconstruction is worse than this cannot be signed from its
# faces. Tuned against the shipped arms: the accepted ones measure <=0.8%
# non-manifold, the rejected ones 54-93%, so the boundary between them is not a
# fine one. Failing it no longer ends the run -- it hands the sample to the
# alpha route, which does not need faces at all.
MAX_NONMANIFOLD_FRACTION = 0.02
# A 3-D shell is supposed to be closed; a few open edges (a symmetry plane, a
# trailing edge) are tolerable, a quarter of the mesh is a volume mesh.
MAX_OPEN_FRACTION_3D = 0.05
# Below this share of edges covered by faces, the face kind chosen is wrong.
MIN_EDGE_COVERAGE = 0.5

# A traced planar face with this many sides or fewer, and positive (counter-
# clockwise) area, is an element of the mesh. Anything larger is the outer face
# or a hole -- a tri/quad mesh has no 5-sided element, and every 2-D arm here
# traces its holes at 24-1,511 sides, so the separation is not a fine one.
PLANAR_MAX_ELEMENT_SIDES = 4
# ... but if a real share of the bounded faces land just above that, the mesh is
# not a tri/quad mesh and the whole classification is wrong; fall back instead.
PLANAR_MAX_ODD_FACE_FRACTION = 0.01
PLANAR_ODD_FACE_SIDES = 8

# The alpha that just covers every node is, on a structured grid, *exactly* the
# circumradius of the degenerate cospherical simplices, so floating-point noise
# decides which of them survive and the complex fills with spurious voids
# (ex2: 71,952 boundary facets at 1.00x, 48,664 from 1.02x through 1.30x).
# A 5% margin clears the tie and sits in the middle of that plateau.
ALPHA_TIE_MARGIN = 1.05
# Share of the calibration probes that must land on the side their construction
# already puts them on before the alpha boundary's orientation is trusted. A
# closed, consistently oriented boundary scores exactly 1.0 or exactly 0.0
# (0.0 = VTK oriented the shell inward, which is just a global flip); anything
# in between means the surface pinches, the normals cannot be believed, and
# the sign has to come from exact point location instead.
MIN_ALPHA_ORIENTATION_AGREEMENT = 0.999
# Kept simplices sampled for that calibration, and the fewest of them that must
# sit clear of the boundary for the interior half of the check to mean anything.
ALPHA_PROBE_COUNT = 4096
ALPHA_MIN_FAR_PROBES = 8

_QUERY_CHUNK = 4096
# Nearest sub-segments examined per query before the exactness bound is checked.
_SEG_CANDIDATES = (16, 64, 256)
# Cap on the (queries x primitives) block a single vectorized step materializes.
_PAIR_BUDGET = 4000000


class BoundaryQuality(object):
    """Edge/face incidence of a reconstruction -- the evidence for the gate.

    `kind` names the route as well as the face kind (`tri`, `quad`,
    `planar-tri`, `alpha-tri`, `alpha-seg`, ...). For the routes whose "faces"
    are segments, `n_faces` counts the segments and `open`/`manifold`/
    `nonmanifold` count the vertices where one, two or more of them meet.
    """

    def __init__(self, kind, n_faces, n_edges, open_edges, manifold_edges,
                 nonmanifold_edges):
        self.kind = kind
        self.n_faces = int(n_faces)
        self.n_edges = int(n_edges)
        self.open_edges = int(open_edges)
        self.manifold_edges = int(manifold_edges)
        self.nonmanifold_edges = int(nonmanifold_edges)

    @property
    def open_fraction(self):
        return self.open_edges / max(self.n_edges, 1)

    @property
    def nonmanifold_fraction(self):
        return self.nonmanifold_edges / max(self.n_edges, 1)

    def __str__(self):
        return ("{} faces={:,} edges={:,} open={:.2%} non-manifold={:.2%}".format(
            self.kind, self.n_faces, self.n_edges, self.open_fraction,
            self.nonmanifold_fraction))


def undirected_edges(edge_index):
    """Unique undirected edges as an (E, 2) int64 array with column0 < column1."""
    ei = np.asarray(edge_index)
    if ei.shape[0] != 2:
        ei = ei.T
    lo = np.minimum(ei[0], ei[1]).astype(np.int64)
    hi = np.maximum(ei[0], ei[1]).astype(np.int64)
    keep = lo != hi                       # self-loops carry no face
    return np.unique(np.stack([lo[keep], hi[keep]], axis=1), axis=0)


def _adjacency(edges, num_nodes):
    a = sp.csr_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])),
                      shape=(num_nodes, num_nodes))
    a = (a + a.T).tocsr()
    a.data[:] = 1
    return a


def _common_neighbors(adj, pairs, num_nodes):
    """For each pair (a, b), every w adjacent to both. Returns (pair_idx, w).

    Vectorized rather than looped: expand each pair over the neighbours of `a`,
    then keep the ones that are also neighbours of `b` by testing the canonical
    edge key (lo * N + hi) against the sorted edge-key table with searchsorted.
    A Python loop over the 909,700 edges of ex3_NASA_CRM_full takes minutes;
    this takes seconds.
    """
    indptr, indices = adj.indptr, adj.indices
    a, b = pairs[:, 0], pairs[:, 1]
    deg = (indptr[a + 1] - indptr[a]).astype(np.int64)
    total = int(deg.sum())
    if total == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pair_idx = np.repeat(np.arange(len(pairs), dtype=np.int64), deg)
    # gather the neighbours of `a` for every pair, flat
    starts = np.repeat(indptr[a].astype(np.int64), deg)
    ends = np.cumsum(deg)
    offsets = np.arange(total, dtype=np.int64) - np.repeat(ends - deg, deg)
    w = indices[starts + offsets].astype(np.int64)

    b_rep = b[pair_idx]
    lo = np.minimum(b_rep, w)
    hi = np.maximum(b_rep, w)
    keys = lo * num_nodes + hi

    table = adj.tocoo()
    m = table.row < table.col
    edge_keys = np.sort(table.row[m].astype(np.int64) * num_nodes
                        + table.col[m].astype(np.int64))
    pos = np.searchsorted(edge_keys, keys)
    pos[pos >= len(edge_keys)] = 0
    hit = (edge_keys[pos] == keys) & (w != b_rep) & (w != a[pair_idx])
    return pair_idx[hit], w[hit]


def _triangles(adj, edges, num_nodes):
    """3-cycles, each emitted once (w greater than both endpoints)."""
    pair_idx, w = _common_neighbors(adj, edges, num_nodes)
    keep = w > edges[pair_idx, 1]
    pair_idx, w = pair_idx[keep], w[keep]
    return np.stack([edges[pair_idx, 0], edges[pair_idx, 1], w], axis=1)


def _quads(adj, num_nodes):
    """4-cycles: two non-adjacent nodes sharing exactly two neighbours.

    A structured quad or hex mesh has no diagonal edges, so it contains no
    3-cycle at all and the triangle reconstruction returns nothing usable --
    ex7_airfrans and ex9_plasticity have literally zero, and
    ex3_NASA_CRM_full has 1,792 strays against 453,959 real quads.
    """
    prod = (adj @ adj).tocsr()
    prod.setdiag(0)
    prod.eliminate_zeros()
    off = (prod - prod.multiply(adj)).tocoo()
    sel = (off.data == 2) & (off.row < off.col)
    if not np.any(sel):
        return np.zeros((0, 4), np.int64)
    pairs = np.stack([off.row[sel].astype(np.int64),
                      off.col[sel].astype(np.int64)], axis=1)
    pair_idx, w = _common_neighbors(adj, pairs, num_nodes)
    order = np.argsort(pair_idx, kind='stable')
    pair_idx, w = pair_idx[order], w[order]
    counts = np.bincount(pair_idx, minlength=len(pairs))
    keep = (counts == 2)[pair_idx]        # a genuine quad diagonal
    pair_idx, w = pair_idx[keep], w[keep]
    if len(pair_idx) == 0:
        return np.zeros((0, 4), np.int64)
    a = pairs[pair_idx[0::2], 0]
    b = pairs[pair_idx[0::2], 1]
    quads = np.stack([a, w[0::2], b, w[1::2]], axis=1)
    # each quad is found from both of its diagonals -- keep one copy
    _, first = np.unique(np.sort(quads, axis=1), axis=0, return_index=True)
    return quads[np.sort(first)]


def _edge_face_counts(faces):
    """Undirected edges of the face set with the number of faces on each."""
    k = faces.shape[1]
    ring = [(i, (i + 1) % k) for i in range(k)]
    pairs = np.sort(np.concatenate([faces[:, [i, j]] for i, j in ring]), axis=1)
    return np.unique(pairs, axis=0, return_counts=True)


def reconstruct_faces(edge_index, num_nodes):
    """Faces of the mesh from its edge graph, as (faces, BoundaryQuality).

    Picks triangles or quads by which one actually covers the edge graph,
    rather than by whether triangles happen to exist: a quad shell with a
    handful of stray 3-cycles is a quad shell. Among the kinds that do cover
    it, the one with the *cleanest* incidence wins -- coverage alone would pick
    the quad reading of ex3_NASA_CRM_mid (89% non-manifold) over its triangle
    reading (28%), and a face set is only useful here if it bounds something.
    """
    edges = undirected_edges(edge_index)
    if len(edges) == 0:
        raise ValueError("mesh_edge is empty; no boundary can be reconstructed.")
    adj = _adjacency(edges, num_nodes)

    candidates = []
    for kind, faces in (('tri', _triangles(adj, edges, num_nodes)),
                        ('quad', _quads(adj, num_nodes))):
        if len(faces) == 0:
            continue
        uq, cnt = _edge_face_counts(faces)
        coverage = len(uq) / len(edges)
        quality = BoundaryQuality(kind, len(faces), len(edges),
                                  open_edges=int((cnt == 1).sum()),
                                  manifold_edges=int((cnt == 2).sum()),
                                  nonmanifold_edges=int((cnt >= 3).sum()))
        candidates.append((faces, quality, coverage))

    if not candidates:
        raise ValueError("No triangle or quad could be reconstructed from "
                         "mesh_edge; this mesh has no faces to bound a region.")
    covering = [c for c in candidates if c[2] >= MIN_EDGE_COVERAGE]
    if covering:
        faces, quality, coverage = min(
            covering, key=lambda c: (c[1].nonmanifold_fraction, -c[2]))
    else:
        faces, quality, coverage = max(candidates, key=lambda c: c[2])
    if coverage < MIN_EDGE_COVERAGE:
        raise ValueError(
            "Reconstructed {} but its faces cover only {:.1%} of the mesh "
            "edges; the remainder belongs to no face, so no closed boundary "
            "exists.".format(quality, coverage))
    return faces, quality


def _check_quality(quality, dim):
    if quality.nonmanifold_fraction > MAX_NONMANIFOLD_FRACTION:
        raise ValueError(
            "{}: {:.1%} of edges carry three or more faces (limit {:.0%}). "
            "That is a volume mesh, whose interior faces are indistinguishable "
            "from its boundary faces in an edge graph.".format(
                quality, quality.nonmanifold_fraction, MAX_NONMANIFOLD_FRACTION))
    if dim == 3 and quality.open_fraction > MAX_OPEN_FRACTION_3D:
        raise ValueError(
            "{}: {:.1%} of edges carry a single face (limit {:.0%}), so the "
            "surface is not closed and 'inside' is undefined.".format(
                quality, quality.open_fraction, MAX_OPEN_FRACTION_3D))


# --------------------------------------------------------------------------
# Route "planar": the true faces of a 2-D mesh, from its rotation system
# --------------------------------------------------------------------------

def _trace_planar_faces(positions, edges):
    """Faces of the planar embedding the node coordinates induce.

    Half-edge h in [0, 2E) is `edges[h]` read forward for h < E and
    `edges[h - E]` read backward otherwise. Around each source vertex the
    outgoing half-edges are sorted by angle (counter-clockwise); the face
    traversal is `next(h) = clockwise-neighbour-of(twin(h))`, which walks every
    bounded face counter-clockwise and every outer face clockwise. `next` is a
    permutation of the half-edges, so its cycles -- and therefore the faces --
    are exactly the strongly connected components of its functional graph,
    which is why no Python loop over 725,000 half-edges is needed.

    Returns (face_of_half_edge, face_sides, face_signed_area, n_faces).
    """
    num_edges = len(edges)
    num_nodes = len(positions)
    two_e = 2 * num_edges
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    delta = positions[dst] - positions[src]
    angle = np.arctan2(delta[:, 1], delta[:, 0])

    order = np.lexsort((angle, src))          # CCW fan around each source
    degree = np.bincount(src, minlength=num_nodes)
    block = np.concatenate([[0], np.cumsum(degree)])[:num_nodes]
    src_sorted = src[order]
    in_fan = np.arange(two_e) - block[src_sorted]
    prev_sorted = block[src_sorted] + (in_fan - 1) % degree[src_sorted]

    prev_fan = np.empty(two_e, np.int64)
    prev_fan[order] = order[prev_sorted]
    twin = (np.arange(two_e, dtype=np.int64) + num_edges) % two_e
    nxt = prev_fan[twin]

    walk = sp.coo_matrix((np.ones(two_e, np.int8),
                          (np.arange(two_e), nxt)), shape=(two_e, two_e)).tocsr()
    n_faces, face = connected_components(walk, directed=True, connection='strong')

    cross = (positions[src, 0] * positions[dst, 1]
             - positions[dst, 0] * positions[src, 1])
    area = 0.5 * np.bincount(face, weights=cross, minlength=n_faces)
    sides = np.bincount(face, minlength=n_faces)
    return face, sides, area, n_faces


def planar_boundary_segments(positions, edge_index):
    """Domain boundary of a 2-D mesh, as (segments (S, 2), BoundaryQuality).

    Exact where it applies, and it says so for itself: Euler's V - E + F = 1 + C
    holds for a planar embedding and fails for anything else, so a k-nearest-
    neighbour graph (ex8_elasticity, 3,360 edges against a planar maximum of
    3N-6 = 2,910) is rejected here rather than silently traced into nonsense.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[1] != 2:
        raise ValueError("planar tracing needs 2-D positions.")
    edges = undirected_edges(edge_index)
    if len(edges) == 0:
        raise ValueError("mesh_edge is empty; no boundary can be reconstructed.")

    num_nodes = len(positions)
    face, sides, area, n_faces = _trace_planar_faces(positions, edges)

    degree = np.bincount(np.concatenate([edges[:, 0], edges[:, 1]]),
                         minlength=num_nodes)
    n_used = int((degree > 0).sum())
    graph = sp.coo_matrix((np.ones(len(edges), np.int8),
                           (edges[:, 0], edges[:, 1])),
                          shape=(num_nodes, num_nodes))
    n_comp, _ = connected_components(graph, directed=False)
    n_comp -= int((degree == 0).sum())    # isolated nodes are not components here
    euler = n_used - len(edges) + n_faces
    if euler != 1 + n_comp:
        raise ValueError(
            "The node coordinates do not induce a planar embedding of "
            "mesh_edge (V-E+F = {}, a planar embedding of {} component(s) "
            "requires {}). The edge graph is not a 2-D mesh -- most likely a "
            "k-nearest-neighbour graph over a point cloud.".format(
                euler, n_comp, 1 + n_comp))

    bounded = area > 0
    n_bounded = int(bounded.sum())
    if n_bounded == 0:
        raise ValueError("The planar embedding has no bounded face; "
                         "mesh_edge encloses no area.")
    odd = bounded & (sides > PLANAR_MAX_ELEMENT_SIDES) & (sides <= PLANAR_ODD_FACE_SIDES)
    if int(odd.sum()) > PLANAR_MAX_ODD_FACE_FRACTION * n_bounded:
        raise ValueError(
            "{:.1%} of the bounded faces have 5-{} sides, so this is not a "
            "triangle/quadrilateral mesh and an element cannot be told from a "
            "hole.".format(odd.sum() / n_bounded, PLANAR_ODD_FACE_SIDES))

    element = bounded & (sides <= PLANAR_MAX_ELEMENT_SIDES)
    if not np.any(element):
        raise ValueError("No triangle or quadrilateral element was traced; "
                         "mesh_edge bounds no mesh cell.")
    on_element = element[face]
    # exactly one side of the edge is an element -> it is on the domain boundary
    is_boundary = on_element[:len(edges)] != on_element[len(edges):]
    segments = edges[is_boundary]
    if len(segments) == 0:
        raise ValueError(
            "Every edge has an element on both sides, so the mesh has no "
            "boundary at all (a periodic or closed 2-D domain).")

    tri = int((element & (sides == 3)).sum())
    quad = int((element & (sides == 4)).sum())
    kind = 'planar-tri' if quad == 0 else ('planar-quad' if tri == 0 else 'planar-mixed')
    quality = BoundaryQuality(kind, int(element.sum()), len(edges),
                              open_edges=len(segments),
                              manifold_edges=len(edges) - len(segments),
                              nonmanifold_edges=0)
    return segments, quality


# --------------------------------------------------------------------------
# Route "alpha": the boundary a node cloud implies when the edge graph cannot
# --------------------------------------------------------------------------

class AlphaShape(object):
    """An alpha complex of a node cloud and the boundary it bounds."""

    def __init__(self, points, triangulation, keep, facets, alpha, quality):
        self.points = points
        self.triangulation = triangulation
        self.keep = keep
        self.facets = facets
        self.alpha = float(alpha)
        self.quality = quality

    def contains(self, queries):
        """True where a query lies in a kept simplex (exact point location).

        This is the reference answer -- a combinatorial fact about the
        tessellation, with no normal vector and nothing to invert -- but it is
        not what `_signed_distance_alpha_3d` runs, because
        `scipy.spatial.Delaunay.find_simplex`'s directed walk degenerates on a
        cospherical structured grid and silently falls back to scanning every
        simplex per query (ex2_dynamic_contact: 200k queries x 1.34M
        simplices, still running after 70 s). Keep it for verification, and
        for the small complexes where it is instant.
        """
        loc = self.triangulation.find_simplex(
            np.ascontiguousarray(queries, dtype=np.float64))
        return (loc >= 0) & self.keep[np.clip(loc, 0, None)]


def _simplex_circumradii(points, simplices):
    """Circumradius per simplex; +inf where the simplex is degenerate."""
    v = points[simplices]                                   # [S, d+1, d]
    v0 = v[:, 0, :]
    mat = 2.0 * (v[:, 1:, :] - v0[:, None, :])              # [S, d, d]
    rhs = (np.einsum('sij,sij->si', v[:, 1:, :], v[:, 1:, :])
           - np.einsum('si,si->s', v0, v0)[:, None])        # [S, d]
    radius = np.full(len(simplices), np.inf)
    ok = np.abs(np.linalg.det(mat)) > 0.0
    if np.any(ok):
        centre = np.linalg.solve(mat[ok], rhs[ok][..., None])[..., 0]
        radius[ok] = np.linalg.norm(centre - v0[ok], axis=1)
    return radius


def _simplex_longest_edge(points, simplices):
    v = points[simplices]
    k = v.shape[1]
    longest = np.zeros(len(simplices))
    for i in range(k):
        for j in range(i + 1, k):
            longest = np.maximum(longest, np.linalg.norm(v[:, i] - v[:, j], axis=1))
    return longest


def _alpha_radius(points, simplices):
    """Circumradius, with half the longest edge standing in where it is undefined.

    A structured grid is massively cospherical, so Qhull emits flat slivers --
    181,993 of ex2_dynamic_contact's 1,340,670 tetrahedra have no circumsphere
    at all. Dropping them punches thousands of spurious holes through the body
    (71,952 boundary facets instead of 48,664). Half the longest edge is the
    size such a simplex would have had, which puts them back where they belong.
    """
    radius = _simplex_circumradii(points, simplices)
    bad = ~np.isfinite(radius)
    if np.any(bad):
        radius[bad] = 0.5 * _simplex_longest_edge(points, simplices[bad])
    return radius


def _alpha_that_covers_every_node(points, simplices, radius):
    """Smallest alpha whose complex still contains every input node.

    Self-check 1 of the sign -- "no node of the mesh may lie outside the body"
    -- is not something checked after the fact here; it is what *chooses* the
    threshold. A node is covered as soon as one incident simplex survives, so
    the answer is the max over nodes of the min incident circumradius.
    """
    best = np.full(len(points), np.inf)
    np.minimum.at(best, simplices.ravel(),
                  np.repeat(radius, simplices.shape[1]))
    finite = best[np.isfinite(best)]
    if len(finite) == 0:
        raise ValueError("No node is covered by any Delaunay simplex.")
    return float(finite.max())


def _alpha_boundary_facets(triangulation, keep):
    """Facets used by exactly one kept simplex -- the boundary of the complex."""
    neighbours = triangulation.neighbors
    outward = (neighbours < 0) | (~keep[np.clip(neighbours, 0, None)])
    sel = keep[:, None] & outward
    simplex_id, local = np.nonzero(sel)
    verts = triangulation.simplices[simplex_id]
    mask = np.ones(verts.shape, bool)
    mask[np.arange(len(simplex_id)), local] = False
    return verts[mask].reshape(len(simplex_id), verts.shape[1] - 1)


def alpha_shape_boundary(points, alpha=None):
    """Alpha complex of `points` and its boundary facets, as an `AlphaShape`.

    With `alpha=None` the threshold is `ALPHA_TIE_MARGIN` times the smallest
    alpha that still covers every node (see `_alpha_that_covers_every_node`).
    """
    from scipy.spatial import Delaunay
    from scipy.spatial import QhullError

    points = np.ascontiguousarray(points, dtype=np.float64)
    dim = points.shape[1]
    if len(points) < dim + 1:
        raise ValueError(
            "{} points cannot bound a region in {}-D.".format(len(points), dim))
    try:
        triangulation = Delaunay(points)
    except QhullError as exc:
        raise ValueError(
            "The node cloud has no Delaunay tessellation, so no boundary can "
            "be estimated from it ({}). The geometry is degenerate -- every "
            "node lies on one line or plane.".format(
                str(exc).strip().splitlines()[0]))

    simplices = triangulation.simplices
    radius = _alpha_radius(points, simplices)
    if alpha is None:
        alpha = ALPHA_TIE_MARGIN * _alpha_that_covers_every_node(
            points, simplices, radius)
    keep = radius <= alpha
    if not np.any(keep):
        raise ValueError("alpha={:g} keeps no simplex at all.".format(alpha))
    facets = _alpha_boundary_facets(triangulation, keep)
    if len(facets) == 0:
        raise ValueError(
            "The alpha complex has no boundary facet; it fills its own convex "
            "hull, so no point could be outside it.")

    if dim == 3:
        uq, cnt = _edge_face_counts(facets)
        quality = BoundaryQuality('alpha-tri', len(facets), len(uq),
                                  open_edges=int((cnt == 1).sum()),
                                  manifold_edges=int((cnt == 2).sum()),
                                  nonmanifold_edges=int((cnt >= 3).sum()))
    else:
        _, cnt = np.unique(facets.ravel(), return_counts=True)
        quality = BoundaryQuality('alpha-seg', len(facets), len(cnt),
                                  open_edges=int((cnt == 1).sum()),
                                  manifold_edges=int((cnt == 2).sum()),
                                  nonmanifold_edges=int((cnt >= 3).sum()))
    return AlphaShape(points, triangulation, keep, facets, alpha, quality)


# --------------------------------------------------------------------------
# Distance and sign
# --------------------------------------------------------------------------

def _segment_block(q, a0, a1):
    """Brute-force distance from each query to the nearest of every segment."""
    seg = a1 - a0
    den = np.einsum('ij,ij->i', seg, seg)
    den = np.where(den > 0, den, 1.0)
    delta = q[:, None, :] - a0[None, :, :]
    t = np.clip(np.einsum('qsi,si->qs', delta, seg) / den, 0.0, 1.0)
    closest = a0[None, :, :] + t[:, :, None] * seg[None, :, :]
    return np.linalg.norm(q[:, None, :] - closest, axis=2).min(axis=1)


def _segment_candidates(q, a0, a1, ik):
    """Distance to the segments named per-query by `ik` (Q, K)."""
    out = np.empty(len(q), np.float64)
    chunk = max(1, _PAIR_BUDGET // max(ik.shape[1], 1))
    for start in range(0, len(q), chunk):
        qq = q[start:start + chunk]
        idx = ik[start:start + chunk]
        base = a0[idx]
        seg = a1[idx] - base
        den = np.einsum('qki,qki->qk', seg, seg)
        den = np.where(den > 0, den, 1.0)
        t = np.clip(np.einsum('qki,qki->qk', qq[:, None, :] - base, seg) / den,
                    0.0, 1.0)
        closest = base + t[..., None] * seg
        out[start:start + chunk] = np.linalg.norm(
            qq[:, None, :] - closest, axis=2).min(axis=1)
    return out


def _split_segments(p0, p1):
    """Split long segments into pieces no longer than the 75th percentile.

    Splitting a segment does not move it, so the distance field is unchanged;
    it only tightens the pruning bound below, which is `half the longest piece`.
    """
    seg = p1 - p0
    length = np.linalg.norm(seg, axis=1)
    positive = length[length > 0]
    target = float(np.percentile(positive, 75)) if len(positive) else 0.0
    if target <= 0:
        return p0, p1
    parts = np.maximum(1, np.ceil(length / target - 1e-12).astype(np.int64))
    if parts.max() <= 1:
        return p0, p1
    rep = np.repeat(np.arange(len(p0)), parts)
    offset = np.arange(int(parts.sum())) - np.repeat(np.cumsum(parts) - parts, parts)
    n = np.repeat(parts, parts)
    t0 = (offset / n)[:, None]
    t1 = ((offset + 1) / n)[:, None]
    return p0[rep] + t0 * seg[rep], p0[rep] + t1 * seg[rep]


def _point_segment_distance(queries, p0, p1):
    """Exact distance from each query to the nearest of the given segments.

    Brute force is O(Q x S) and ex7_airfrans is 181,794 x 2,536 of it, so the
    search is pruned with a KD-tree over sub-segment midpoints: a segment whose
    midpoint is further away than d + half its length cannot beat a distance d.
    The K-nearest bound is *verified* rather than assumed -- whatever it fails
    to certify is redone with a wider K and finally exhaustively, so the result
    equals the brute-force result exactly.
    """
    from scipy.spatial import cKDTree

    a0, a1 = _split_segments(p0, p1)
    out = np.empty(len(queries), np.float64)
    if len(queries) == 0:
        return out
    if len(a0) <= _SEG_CANDIDATES[0]:
        for start in range(0, len(queries), _QUERY_CHUNK):
            out[start:start + _QUERY_CHUNK] = _segment_block(
                queries[start:start + _QUERY_CHUNK], a0, a1)
        return out

    mid = 0.5 * (a0 + a1)
    half_max = float(0.5 * np.linalg.norm(a1 - a0, axis=1).max())
    tree = cKDTree(mid)

    todo = np.arange(len(queries))
    for k in _SEG_CANDIDATES:
        k = min(k, len(mid))
        dk, ik = tree.query(queries[todo], k=k, workers=-1)
        if ik.ndim == 1:
            dk, ik = dk[:, None], ik[:, None]
        dist = _segment_candidates(queries[todo], a0, a1, ik)
        out[todo] = dist
        if k >= len(mid):
            todo = todo[:0]
            break
        # a segment outside the K-set has midpoint distance >= dk[:, -1], hence
        # true distance >= dk[:, -1] - half_max; below that the answer is certain
        todo = todo[dist > dk[:, -1] - half_max]
        if len(todo) == 0:
            break
    for start in range(0, len(todo), _QUERY_CHUNK):
        idx = todo[start:start + _QUERY_CHUNK]
        out[idx] = _segment_block(queries[idx], a0, a1)
    return out


def _inside_even_odd(queries, p0, p1):
    """Even-odd crossing count of a +x ray, bucketed by y.

    Needs no loop ordering and no consistent orientation -- only that the
    segment set is closed, which every route here establishes before calling.
    A segment can only be crossed by a ray whose y lies inside that segment's
    own y-range, so bucketing on y turns the O(Q x S) scan into O(Q x S/bins)
    without changing a single crossing.
    """
    y0, y1 = p0[:, 1], p1[:, 1]
    y_lo = np.minimum(y0, y1)
    y_hi = np.maximum(y0, y1)
    g_lo, g_hi = float(y_lo.min()), float(y_hi.max())
    inside = np.zeros(len(queries), bool)
    if not (g_hi > g_lo) or len(queries) == 0:
        return inside

    n_bins = int(min(max(int(np.sqrt(len(p0))) * 2, 1), 8192))
    step = (g_hi - g_lo) / n_bins
    b_lo = np.clip(((y_lo - g_lo) / step).astype(np.int64), 0, n_bins - 1)
    b_hi = np.clip(((y_hi - g_lo) / step).astype(np.int64), 0, n_bins - 1)
    span = b_hi - b_lo + 1
    seg_id = np.repeat(np.arange(len(p0)), span)
    offset = np.arange(int(span.sum())) - np.repeat(np.cumsum(span) - span, span)
    bin_id = np.repeat(b_lo, span) + offset
    order = np.argsort(bin_id, kind='stable')
    seg_id, bin_id = seg_id[order], bin_id[order]
    bin_start = np.searchsorted(bin_id, np.arange(n_bins + 1))

    qy = queries[:, 1]
    live = np.nonzero((qy >= g_lo) & (qy <= g_hi))[0]
    if len(live) == 0:
        return inside
    q_bin = np.clip(((qy[live] - g_lo) / step).astype(np.int64), 0, n_bins - 1)
    q_order = np.argsort(q_bin, kind='stable')
    live, q_bin = live[q_order], q_bin[q_order]
    q_start = np.searchsorted(q_bin, np.arange(n_bins + 1))

    for b in range(n_bins):
        qs, qe = q_start[b], q_start[b + 1]
        ss, se = bin_start[b], bin_start[b + 1]
        if qs == qe or ss == se:
            continue
        s = seg_id[ss:se]
        a_y, b_y = y0[s][None, :], y1[s][None, :]
        a_x, b_x = p0[s, 0][None, :], p1[s, 0][None, :]
        den = np.where(np.abs(b_y - a_y) > 0, b_y - a_y, 1.0)
        chunk = max(1, _PAIR_BUDGET // len(s))
        for start in range(qs, qe, chunk):
            idx = live[start:min(start + chunk, qe)]
            q_y = queries[idx, 1:2]
            q_x = queries[idx, 0:1]
            straddles = (a_y > q_y) != (b_y > q_y)
            x_cross = a_x + (q_y - a_y) / den * (b_x - a_x)
            inside[idx] = np.sum(straddles & (q_x < x_cross), axis=1) % 2 == 1
    return inside


def _signed_distance_2d(queries, verts, segments):
    """Distance to the boundary polyline, negative inside."""
    p0 = verts[segments[:, 0]]
    p1 = verts[segments[:, 1]]
    dist = _point_segment_distance(queries, p0, p1)
    inside = _inside_even_odd(queries, p0, p1)
    return np.where(inside, -dist, dist)


def _polydata(verts, faces):
    import pyvista as pv
    k = faces.shape[1]
    cells = np.hstack([np.full((len(faces), 1), k, np.int64), faces]).ravel()
    return pv.PolyData(np.ascontiguousarray(verts, np.float64), faces=cells)


def _vtk_distance(poly, queries):
    """Signed (by `poly`'s normals) distance from every query to `poly`."""
    try:
        from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
        from vtkmodules.util.numpy_support import numpy_to_vtk
    except ImportError as exc:
        raise ImportError(
            "sdf_source=mesh needs PyVista/VTK for the 3-D distance ({}). "
            "Install pyvista, or set sdf_source none.".format(exc))
    implicit = vtkImplicitPolyDataDistance()
    implicit.SetInput(poly)
    out = np.empty(len(queries), np.float64)
    implicit.FunctionValue(numpy_to_vtk(np.ascontiguousarray(queries, np.float64)),
                           numpy_to_vtk(out))
    return out


def _signed_distance_3d(queries, verts, faces):
    """Distance to the reconstructed shell, negative inside, via VTK.

    `vtkImplicitPolyDataDistance` is the same primitive PyVista uses; it is
    imported here rather than at module import so a checkout without PyVista
    still loads every other sdf_source. Validated against a unit sphere to
    3.8e-4, which is the sphere tessellation's own faceting error.
    """
    try:
        import pyvista  # noqa: F401  (PolyData construction needs it)
    except ImportError as exc:
        raise ImportError(
            "sdf_source=mesh needs PyVista/VTK for the 3-D sign ({}). "
            "Install pyvista, or set sdf_source none.".format(exc))
    poly = _polydata(verts, faces)
    poly = poly.compute_normals(auto_orient_normals=True, consistent_normals=True,
                                split_vertices=False)
    return _vtk_distance(poly, queries)


def _alpha_orientation_probes(shape):
    """Points whose side of the alpha boundary is known before it is oriented.

    The centroid of a kept simplex lies strictly inside that simplex, which is
    part of the complex, so it is interior. The corners of a bounding box
    padded by a quarter of its own diagonal lie outside the convex hull of the
    points, so they are outside every subcomplex of their triangulation.
    Neither fact involves a normal vector, which is the point: they are what
    the orientation gets checked *against*.
    """
    kept = np.nonzero(shape.keep)[0]
    step = max(1, len(kept) // ALPHA_PROBE_COUNT)
    inner = shape.points[shape.triangulation.simplices[kept[::step]]].mean(axis=1)
    lo = shape.points.min(axis=0)
    hi = shape.points.max(axis=0)
    pad = 0.25 * float(np.linalg.norm(hi - lo))
    dim = shape.points.shape[1]
    bits = ((np.arange(1 << dim)[:, None] >> np.arange(dim)) & 1).astype(bool)
    outer = np.where(bits, hi + pad + 1.0, lo - pad - 1.0)
    return inner, outer


def _alpha_normal_sign(poly, shape):
    """+1, -1, or None if `poly`'s normals cannot be trusted to sign at all.

    +1 means they point out of the complex and VTK's signed distance is
    already the answer; -1 means they point into it, which is a single global
    flip and equally safe. None means the probes contradict each other, which
    on a boundary that is closed and orientable they cannot do.
    """
    inner, outer = _alpha_orientation_probes(shape)
    # A centroid can sit on the boundary when its simplex is a flat sliver, and
    # then its sign is round-off. Judge the orientation only where there is a
    # side to be on; the padded corners are always far.
    span = float(np.linalg.norm(shape.points.max(axis=0) - shape.points.min(axis=0)))
    d_inner = _vtk_distance(poly, inner)
    far = np.abs(d_inner) > 1e-6 * span
    if int(far.sum()) < ALPHA_MIN_FAR_PROBES:
        return None
    agree = np.concatenate([d_inner[far] < 0, _vtk_distance(poly, outer) > 0])
    score = float(agree.mean())
    if score >= MIN_ALPHA_ORIENTATION_AGREEMENT:
        return 1.0
    if score <= 1.0 - MIN_ALPHA_ORIENTATION_AGREEMENT:
        return -1.0
    return None


def _signed_distance_alpha_3d(queries, shape):
    """Distance to the alpha boundary, negative inside, via VTK.

    The sign is the one the boundary's own oriented normals give, and the
    orientation is *checked* rather than assumed. An alpha boundary can pinch
    at an edge or a vertex, and where it does, a consistently oriented patch
    can end up facing the wrong way -- which would invert the field over a
    whole region without changing its magnitude, the exact failure the quality
    gates exist to prevent. So both families of probes from
    `_alpha_orientation_probes` are evaluated first: a closed, coherently
    oriented boundary puts every one of them on its own side, and a pinched
    one cannot -- and when they disagree the sign falls back to exact point
    location rather than being trusted.

    `AlphaShape.contains` answers the same question exactly and was the first
    implementation, but its point location degenerates on a structured grid
    (see that method); calibrated probes cost a fixed ~4k evaluations instead.
    Measured against `find_simplex(bruteforce=True)` and against VTK's
    ray-crossing test on both alpha arms: 0 disagreements in 208,425 far-field
    queries on ex2_dynamic_contact and 33,121 on ex5_deforming_plate.
    """
    try:
        import pyvista  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "sdf_source=mesh needs PyVista/VTK for the 3-D distance ({}). "
            "Install pyvista, or set sdf_source none.".format(exc))
    poly = _polydata(shape.points, shape.facets).compute_normals(
        auto_orient_normals=True, consistent_normals=True, split_vertices=False)

    sign = _alpha_normal_sign(poly, shape)
    if sign is not None:
        return sign * _vtk_distance(poly, queries)

    # The probes disagree among themselves, so this boundary really does pinch
    # -- a sparse cloud whose alpha complex is a chain of slivers does -- and
    # its normals would invert the field over part of the domain. Only the
    # sign is in doubt; the magnitude is a distance to a surface and is right
    # either way. So take the sign from exact point location instead, at the
    # cost warned about in `AlphaShape.contains`.
    dist = np.abs(_vtk_distance(poly, queries))
    return np.where(shape.contains(queries), -dist, dist)


def _describe(exc):
    return str(exc).replace('\n', ' ')


def signed_distance_from_mesh(positions, edge_index, queries, return_quality=False,
                              method='auto'):
    """Signed distance at `queries` from the boundary implied by `edge_index`.

    Args:
        positions: (N, D) reference node coordinates, D in {2, 3}
        edge_index: (2, E) mesh edges
        queries: (Q, D) points to evaluate -- in production the nodes
            themselves (`model/adapters/sdf.py`). Note that on a 3-D *shell*
            every node lies on the surface, so querying the nodes returns ~0
            and carries no geometry; the paper samples a latent grid instead.
        return_quality: also return the BoundaryQuality of the route taken
        method: 'auto' (default) takes the exact route for the dimension and
            falls back to the alpha complex when it does not apply; 'planar',
            'faces' and 'alpha' force one route and raise instead of falling
            back.

    Returns:
        (Q,) float32 signed distances, negative inside.
    """
    positions = np.asarray(positions, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] not in (2, 3):
        raise ValueError(
            "positions must be (N, 2) or (N, 3), got {}.".format(positions.shape))
    if queries.ndim != 2 or queries.shape[1] != positions.shape[1]:
        raise ValueError("queries must be (Q, {}), got {}.".format(
            positions.shape[1], queries.shape))
    if method not in ('auto', 'planar', 'faces', 'alpha'):
        raise ValueError("Unknown method '{}'.".format(method))

    dim = positions.shape[1]
    exact = 'planar' if dim == 2 else 'faces'
    if method in ('planar', 'faces') and method != exact:
        raise ValueError(
            "method='{}' is not available for {}-D positions.".format(method, dim))

    why = None
    values = quality = None
    if method in ('auto', exact):
        try:
            if dim == 2:
                segments, quality = planar_boundary_segments(positions, edge_index)
                values = _signed_distance_2d(queries, positions, segments)
            else:
                faces, quality = reconstruct_faces(edge_index, len(positions))
                _check_quality(quality, dim)
                values = _signed_distance_3d(queries, positions, faces)
        except ValueError as exc:
            if method != 'auto':
                raise
            quality = None
            why = _describe(exc)

    if values is None:
        try:
            shape = alpha_shape_boundary(positions)
        except ValueError as exc:
            raise ValueError(
                "{}The boundary could not be estimated from the node cloud "
                "either: {} Set sdf_source none for this dataset.".format(
                    (why + ' ') if why else '', _describe(exc)))
        quality = shape.quality
        if dim == 2:
            values = _signed_distance_2d(queries, shape.points, shape.facets)
        else:
            values = _signed_distance_alpha_3d(queries, shape)

    values = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("Computed SDF contains non-finite values.")
    return (values, quality) if return_quality else values
