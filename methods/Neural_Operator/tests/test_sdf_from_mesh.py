"""`sdf_source mesh`: signed distance derived from `mesh_edge` and coordinates.

Every case here has an analytic answer, because the whole point of the module
is that the sign is *derived* rather than guessed -- a test that only checked
it ran would not distinguish it from the occupancy heuristic it replaced.

Three routes are covered, and which one a mesh takes is asserted rather than
left implicit, because the failure that matters is a mesh silently dropping to
a weaker route:

    planar   a 2-D mesh, traced from the rotation system its coordinates induce
    faces    a 3-D shell, whose 3-/4-cycles are its faces
    alpha    a 3-D volume mesh or a neighbour graph, where the edge graph holds
             no boundary and it is estimated from the node cloud instead

The rejection tests still matter: `reconstruct_faces`' gate is *not* relaxed to
let a volume mesh through (`test_volume_mesh_has_no_faces_to_read` pins that),
and a geometry that admits no boundary by any route still fails loudly.
"""

import numpy as np
import pytest

from model.adapters.sdf_from_mesh import (
    BoundaryQuality,
    alpha_shape_boundary,
    planar_boundary_segments,
    reconstruct_faces,
    signed_distance_from_mesh,
    undirected_edges,
    _alpha_normal_sign,
    _alpha_orientation_probes,
    _inside_even_odd,
    _polydata,
    _point_segment_distance,
    _segment_block,
    _signed_distance_alpha_3d,
)


def _edges_from_faces(faces):
    """(2, E) mesh_edge for a face array -- the only input the module gets."""
    k = faces.shape[1]
    ring = [(i, (i + 1) % k) for i in range(k)]
    pairs = np.sort(np.concatenate([faces[:, [i, j]] for i, j in ring]), axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs.T.copy()


def _square_grid(n=9):
    xs = np.linspace(0.0, 1.0, n)
    return np.stack(np.meshgrid(xs, xs, indexing='ij'), -1).reshape(-1, 2)


def _tri_square(n=9):
    """Unit square [0,1]^2 as a triangulated grid: positions and mesh_edge."""
    pos = _square_grid(n)
    idx = np.arange(n * n).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return pos, _edges_from_faces(faces)


def _quad_cube(m=5):
    """Closed unit-cube shell [0,1]^3 built from six quad patches."""
    t = np.linspace(0.0, 1.0, m)
    u, v = np.meshgrid(t, t, indexing='ij')
    patches = []
    for axis in range(3):
        for value in (0.0, 1.0):
            p = np.empty((m, m, 3))
            other = [k for k in range(3) if k != axis]
            p[..., axis] = value
            p[..., other[0]] = u
            p[..., other[1]] = v
            patches.append(p.reshape(-1, 3))
    pts = np.concatenate(patches)
    # merge the seams: the six patches share their edge and corner vertices
    keys = np.round(pts, 9)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)

    faces = []
    grid = np.arange(m * m).reshape(m, m)
    for patch in range(6):
        g = grid + patch * m * m
        faces.append(np.stack([g[:-1, :-1].ravel(), g[1:, :-1].ravel(),
                               g[1:, 1:].ravel(), g[:-1, 1:].ravel()], 1))
    faces = inverse[np.concatenate(faces)]
    return uniq, _edges_from_faces(faces)


def _tet_block(n=5):
    """A 3-D VOLUME mesh: a grid of cubes, fully connected within each cube.

    Its interior faces are indistinguishable from its boundary faces once only
    the edges survive (86% of its edges carry three or more reconstructed
    faces), so it has to be signed from the node cloud instead. This is the
    ex2_dynamic_contact / ex5_deforming_plate shape of problem, small enough to
    know the answer: the body is exactly [0,1]^3.
    """
    xs = np.linspace(0.0, 1.0, n)
    pos = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), -1).reshape(-1, 3)
    idx = np.arange(n ** 3).reshape(n, n, n)
    edges = set()
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                if di == dj == dk == 0:
                    continue
                a = idx[:n - di or None, :n - dj or None, :n - dk or None].ravel()
                b = idx[di:, dj:, dk:].ravel()
                for u, w in zip(a.tolist(), b.tolist()):
                    edges.add((min(u, w), max(u, w)))
    return pos, np.asarray(sorted(edges), np.int64).T.copy()


def _knn_edges(pos, k):
    """k-nearest-neighbour graph, the way ex8_elasticity's `mesh_edge` is built.

    It is not a mesh: it carries no faces, its diagonals cross, and for k >= 6
    it has more edges than a planar embedding of that many nodes can hold.
    """
    from scipy.spatial import cKDTree
    _, idx = cKDTree(pos).query(pos, k=k + 1)
    src = np.repeat(np.arange(len(pos)), k)
    return np.stack([src, idx[:, 1:].ravel()]).astype(np.int64)


# --------------------------------------------------------------------------
# 2-D, route "planar": the true faces of the embedding the coordinates induce
# --------------------------------------------------------------------------

def test_2d_square_signed_distance_is_analytic():
    pos, edges = _tri_square(n=9)
    queries = np.array([[0.5, 0.5],     # centre: 0.5 from every side
                        [0.1, 0.5],     # 0.1 from the left side
                        [0.5, 0.875],   # 0.125 from the top
                        [0.0, 0.5],     # on the boundary
                        [1.5, 0.5]])    # 0.5 outside
    d = signed_distance_from_mesh(pos, edges, queries)
    assert d == pytest.approx([-0.5, -0.1, -0.125, 0.0, 0.5], abs=1e-6)


def test_2d_mesh_takes_the_planar_route():
    """Not an implementation detail: the alpha fallback is an estimate, and a
    mesh that quietly stopped tracing its own faces would lose the boundary's
    concavities without losing a single test."""
    pos, edges = _tri_square(n=9)
    _, quality = signed_distance_from_mesh(pos, edges, pos[:4],
                                           return_quality=True)
    assert quality.kind.startswith('planar')


def test_planar_route_traces_the_true_boundary():
    pos, edges = _tri_square(n=9)
    segments, quality = planar_boundary_segments(pos, edges)
    # an n x n grid has 4 (n-1) perimeter edges and nothing else on the boundary
    assert len(segments) == 4 * 8
    assert quality.kind == 'planar-tri'
    assert quality.n_faces == 2 * 8 * 8           # two triangles per cell
    on_edge = np.isclose(pos[segments], 0.0) | np.isclose(pos[segments], 1.0)
    assert np.all(on_edge.any(axis=2))            # every segment lies on a side


def test_2d_sign_is_negative_inside_the_meshed_domain():
    pos, edges = _tri_square(n=9)
    inside = np.array([[0.5, 0.5], [0.25, 0.75]])
    outside = np.array([[-0.2, 0.5], [0.5, 1.4], [2.0, 2.0]])
    assert np.all(signed_distance_from_mesh(pos, edges, inside) < 0)
    assert np.all(signed_distance_from_mesh(pos, edges, outside) > 0)


def test_2d_hole_counts_as_outside_the_domain():
    """A hole is not part of the meshed domain, so its interior is positive.

    This is the ex7_airfrans case: the airfoil is a hole in the flow mesh.
    """
    n = 9
    pos, _ = _tri_square(n=n)
    idx = np.arange(n * n).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    # drop every face touching the central node -> a hole around it
    centre = int(idx[n // 2, n // 2])
    faces = faces[~np.any(faces == centre, axis=1)]
    edges = _edges_from_faces(faces)

    segments, _ = planar_boundary_segments(pos, edges)
    # the outer perimeter plus the hexagonal ring around the hole: this
    # triangulation puts six triangles on every interior node
    assert len(segments) == 4 * 8 + 6

    values = signed_distance_from_mesh(pos, edges, np.array([[0.5, 0.5],
                                                             [0.25, 0.25]]))
    assert values[0] > 0      # inside the hole -> outside the domain
    assert values[1] < 0      # ordinary interior point


def test_planar_route_rejects_a_non_planar_neighbour_graph():
    """Euler's V - E + F = 1 + C is the gate, and it needs no tuning.

    ex8_elasticity's `mesh_edge` is literally a k=6 neighbour graph: 3,360
    edges over 972 nodes, against a planar maximum of 3N - 6 = 2,910. Tracing
    a rotation system over crossing edges yields faces that bound nothing, so
    the trace has to notice it is not looking at a mesh.
    """
    pos = _square_grid(n=9)
    edges = _knn_edges(pos, k=8)
    with pytest.raises(ValueError, match='planar embedding'):
        planar_boundary_segments(pos, edges)


# --------------------------------------------------------------------------
# 3-D, route "faces": the mesh is the shell, signed by its own orientation
# --------------------------------------------------------------------------

def test_3d_cube_shell_signed_distance_is_analytic():
    pytest.importorskip('pyvista')
    pos, edges = _quad_cube(m=5)
    queries = np.array([[0.5, 0.5, 0.5],    # centre: 0.5 from every face
                        [0.1, 0.5, 0.5],    # 0.1 from the x=0 face
                        [0.5, 0.5, 1.25]])  # 0.25 outside the top
    d, quality = signed_distance_from_mesh(pos, edges, queries,
                                           return_quality=True)
    assert quality.kind == 'quad'           # the shell route, not the fallback
    assert d == pytest.approx([-0.5, -0.1, 0.25], abs=1e-5)


def test_3d_cube_is_reconstructed_as_quads():
    pos, edges = _quad_cube(m=5)
    faces, quality = reconstruct_faces(edges, len(pos))
    assert quality.kind == 'quad'
    assert faces.shape[1] == 4
    # a closed cube shell: every edge carries exactly two faces
    assert quality.open_edges == 0
    assert quality.nonmanifold_edges == 0


# --------------------------------------------------------------------------
# Route "alpha": what the node cloud implies when the edge graph holds nothing
# --------------------------------------------------------------------------

def test_volume_mesh_has_no_faces_to_read():
    """The face gate is not relaxed -- a volume mesh still fails it.

    Pinning this separately from the fallback is the point: if the gate were
    widened instead of routed around, the shell route would return a boundary
    made of interior faces and the sign would invert over whole regions with
    nothing raising.
    """
    pos, edges = _tet_block(n=5)
    _, quality = reconstruct_faces(edges, len(pos))
    assert quality.nonmanifold_fraction > 0.5
    with pytest.raises(ValueError, match='three or more faces'):
        signed_distance_from_mesh(pos, edges, np.zeros((4, 3)), method='faces')


def test_volume_mesh_is_signed_by_the_alpha_fallback():
    pytest.importorskip('pyvista')
    pos, edges = _tet_block(n=5)
    queries = np.array([[0.5, 0.5, 0.5],     # centre of [0,1]^3
                        [0.1, 0.5, 0.5],     # 0.1 from the x=0 face
                        [0.5, 0.5, 1.25],    # 0.25 above the top
                        [0.5, 0.5, 0.0]])    # on the boundary
    d, quality = signed_distance_from_mesh(pos, edges, queries,
                                           return_quality=True)
    assert quality.kind == 'alpha-tri'
    assert d == pytest.approx([-0.5, -0.1, 0.25, 0.0], abs=1e-6)


def test_alpha_recovers_the_cube_boundary_exactly():
    pos, _ = _tet_block(n=5)
    shape = alpha_shape_boundary(pos)
    # six faces, each a 4x4 grid of cells split into two triangles
    assert len(shape.facets) == 6 * 2 * 16
    # and it is a closed manifold surface
    assert shape.quality.open_edges == 0
    assert shape.quality.nonmanifold_edges == 0


def test_alpha_threshold_covers_every_input_node():
    """Self-check 1 -- "no node of the mesh lies outside the body" -- is what
    chooses alpha here, not something checked afterwards and hoped for."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 1.0, size=(300, 3))
    shape = alpha_shape_boundary(pos)
    covered = np.zeros(len(pos), bool)
    covered[shape.triangulation.simplices[shape.keep].ravel()] = True
    assert covered.all()

    # ...and it is the *smallest* such alpha, not a comfortable one: shrinking
    # it below the tie margin starts leaving nodes out of their own body
    tight = alpha_shape_boundary(pos, alpha=shape.alpha * 0.8)
    still = np.zeros(len(pos), bool)
    still[tight.triangulation.simplices[tight.keep].ravel()] = True
    assert not still.all()


def test_alpha_signs_a_neighbour_graph_point_cloud():
    """The ex8_elasticity case: no faces, no planar embedding, a real boundary.

    The cloud is a square lattice, so the alpha complex is the unit square
    exactly and the answer is the same analytic one the meshed square gives.
    """
    pos = _square_grid(n=9)
    edges = _knn_edges(pos, k=8)
    queries = np.array([[0.5, 0.5], [0.1, 0.5], [0.5, 0.875],
                        [0.0, 0.5], [1.5, 0.5]])
    d, quality = signed_distance_from_mesh(pos, edges, queries,
                                           return_quality=True)
    assert quality.kind == 'alpha-seg'
    assert quality.n_faces == 4 * 8           # the perimeter, nothing else
    assert d == pytest.approx([-0.5, -0.1, -0.125, 0.0, 0.5], abs=1e-6)


def test_alpha_route_passes_the_three_self_checks():
    """The checks the shipped arms are audited with, on a shape with an answer.

    The body is [0,1]^3 inside a bounding box padded by 10% of its extent, so
    the occupancy of a 32^3 grid over that box is exactly the fraction of grid
    points that land in the cube -- 26 of 32 per axis.
    """
    pytest.importorskip('pyvista')
    pos, edges = _tet_block(n=5)
    sdf = lambda q: signed_distance_from_mesh(pos, edges, q)

    assert np.all(sdf(pos) <= 1e-6)                        # 1: no node outside

    corners = np.stack(np.meshgrid(*[[-0.1, 1.1]] * 3, indexing='ij'),
                       -1).reshape(-1, 3)
    assert np.all(sdf(corners) > 0)                        # 2: bbox corners out

    axis = np.linspace(-0.1, 1.1, 32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'),
                    -1).reshape(-1, 3)
    occupancy = float((sdf(grid) < 0).mean())              # 3: plausible volume
    expected = ((axis > 0.0) & (axis < 1.0)).sum() ** 3 / 32 ** 3
    assert occupancy == pytest.approx(expected, abs=1e-9)


def test_alpha_orientation_probes_know_their_side_without_a_normal():
    """What the 3-D alpha sign is calibrated against, checked independently.

    Simplex centroids and padded bounding-box corners are claimed to be
    interior and exterior by construction; exact point location says whether
    that claim is true, and nothing here consults a normal vector.
    """
    rng = np.random.default_rng(1)
    shape = alpha_shape_boundary(rng.uniform(0.0, 1.0, size=(400, 3)))
    inner, outer = _alpha_orientation_probes(shape)
    assert shape.contains(inner).all()
    assert not shape.contains(outer).any()


@pytest.mark.parametrize('cloud', ['grid', 'random'])
def test_alpha_3d_sign_agrees_with_exact_point_location(cloud):
    """The shipped sign is normal-based for speed; this pins it to the exact
    answer it replaced (`AlphaShape.contains`), on every query far enough from
    the boundary for "inside" to be a fact rather than round-off.

    Both paths are covered on purpose. A solid block's alpha boundary is
    closed and orientable, so the calibration accepts its normals; a sparse
    random cloud's is a chain of slivers that pinches, so it does not, and
    that case has to reach the same answer the slow way.
    """
    pytest.importorskip('pyvista')
    if cloud == 'grid':
        pos, _ = _tet_block(n=5)
    else:
        pos = np.random.default_rng(2).uniform(0.0, 1.0, size=(400, 3))
    shape = alpha_shape_boundary(pos)
    poly = _polydata(shape.points, shape.facets).compute_normals(
        auto_orient_normals=True, consistent_normals=True, split_vertices=False)
    assert (_alpha_normal_sign(poly, shape) is not None) == (cloud == 'grid')

    axis = np.linspace(-0.2, 1.2, 24)
    queries = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'),
                       -1).reshape(-1, 3)
    values = _signed_distance_alpha_3d(queries, shape)
    span = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0)))
    far = np.abs(values) > 1e-6 * span
    assert far.sum() > 0.9 * len(queries)
    assert np.array_equal(values[far] < 0, shape.contains(queries[far]))


def test_alpha_3d_sign_survives_a_boundary_wound_inside_out():
    """Reversing every facet's winding reverses its normals. The calibration
    exists so that this changes nothing: it is a global flip, and the probes
    detect it rather than the field silently inverting."""
    pytest.importorskip('pyvista')
    pos, _ = _tet_block(n=5)
    shape = alpha_shape_boundary(pos)
    flipped = alpha_shape_boundary(pos)
    flipped.facets = np.ascontiguousarray(flipped.facets[:, ::-1])

    axis = np.linspace(-0.2, 1.2, 12)
    queries = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'),
                       -1).reshape(-1, 3)
    assert np.allclose(_signed_distance_alpha_3d(queries, shape),
                       _signed_distance_alpha_3d(queries, flipped))


# --------------------------------------------------------------------------
# Rejection: what cannot be signed by any route must fail loudly
# --------------------------------------------------------------------------

def test_rejection_names_the_way_out():
    """The message has to tell the operator what to set, not just complain.

    A collinear node set bounds no area as a mesh and has no Delaunay
    tessellation as a cloud, so both routes are genuinely out of options.
    """
    pos = np.stack([np.linspace(0.0, 1.0, 6), np.zeros(6)], axis=1)
    edges = np.stack([np.arange(5), np.arange(1, 6)])
    with pytest.raises(ValueError, match='sdf_source none'):
        signed_distance_from_mesh(pos, edges, np.zeros((2, 2)))


def test_rejection_reports_both_routes():
    """...and why each one failed, not only the last."""
    pos = np.stack([np.linspace(0.0, 1.0, 6), np.zeros(6)], axis=1)
    edges = np.stack([np.arange(5), np.arange(1, 6)])
    with pytest.raises(ValueError) as excinfo:
        signed_distance_from_mesh(pos, edges, np.zeros((2, 2)))
    message = str(excinfo.value)
    assert 'no bounded face' in message          # the mesh route's diagnosis
    assert 'node cloud' in message               # the fallback's diagnosis


def test_a_forced_route_does_not_fall_back():
    """`method=` is for diagnosing which route a dataset takes, so it must not
    quietly answer with a different one."""
    pos = _square_grid(n=9)
    edges = _knn_edges(pos, k=8)
    with pytest.raises(ValueError, match='planar embedding'):
        signed_distance_from_mesh(pos, edges, pos[:4], method='planar')
    # ...while 'auto' on the same input succeeds through the fallback
    assert signed_distance_from_mesh(pos, edges, pos[:4]).shape == (4,)


def test_empty_edges_are_rejected():
    with pytest.raises(ValueError, match='mesh_edge is empty'):
        signed_distance_from_mesh(np.zeros((4, 2)), np.zeros((2, 0), np.int64),
                                  np.zeros((1, 2)))


def test_query_dimension_must_match_positions():
    pos, edges = _tri_square(n=5)
    with pytest.raises(ValueError, match=r'queries must be \(Q, 2\)'):
        signed_distance_from_mesh(pos, edges, np.zeros((3, 3)))


# --------------------------------------------------------------------------
# Face-kind selection
# --------------------------------------------------------------------------

def test_stray_triangles_do_not_hide_a_quad_shell():
    """ex3_NASA_CRM_full has 1,792 stray 3-cycles against 453,959 real quads.

    Selecting on "are there any triangles?" reads that mesh as 1,792 triangles
    and throws away 99.6% of it; selecting on coverage reads it as quads.
    """
    pos, edges = _quad_cube(m=5)
    # add one diagonal, creating a pair of stray 3-cycles
    corner = np.argmin(np.linalg.norm(pos - np.array([0.0, 0.0, 0.0]), axis=1))
    far = np.argmin(np.linalg.norm(pos - np.array([0.0, 0.25, 0.25]), axis=1))
    extra = np.array([[corner], [far]], np.int64)
    edges = np.concatenate([edges, extra], axis=1)

    faces, quality = reconstruct_faces(edges, len(pos))
    assert quality.kind == 'quad'
    assert quality.n_faces > 90


def test_quality_reports_the_evidence_for_the_gate():
    q = BoundaryQuality('tri', 10, 100, open_edges=2, manifold_edges=95,
                        nonmanifold_edges=3)
    assert q.open_fraction == pytest.approx(0.02)
    assert q.nonmanifold_fraction == pytest.approx(0.03)
    assert 'non-manifold=3.00%' in str(q)


def test_undirected_edges_drops_duplicates_and_self_loops():
    ei = np.array([[0, 1, 1, 2, 3], [1, 0, 2, 1, 3]], np.int64)
    out = undirected_edges(ei)
    assert out.tolist() == [[0, 1], [1, 2]]


# --------------------------------------------------------------------------
# The 2-D accelerations must be accelerations, not approximations
# --------------------------------------------------------------------------

def _brute_inside(queries, p0, p1):
    a_y, b_y = p0[:, 1][None, :], p1[:, 1][None, :]
    a_x, b_x = p0[:, 0][None, :], p1[:, 0][None, :]
    den = np.where(np.abs(b_y - a_y) > 0, b_y - a_y, 1.0)
    q_y, q_x = queries[:, 1:2], queries[:, 0:1]
    straddles = (a_y > q_y) != (b_y > q_y)
    crossing = a_x + (q_y - a_y) / den * (b_x - a_x)
    return np.sum(straddles & (q_x < crossing), axis=1) % 2 == 1


def _wiggly_polygon(seed, n):
    rng = np.random.default_rng(seed)
    theta = np.sort(rng.random(n)) * 2.0 * np.pi
    radius = 1.0 + 0.3 * rng.random(n)
    v = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    return v, np.roll(v, -1, axis=0)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_pruned_distance_equals_brute_force(seed):
    """ex7_airfrans is 181,794 queries against 2,536 segments; brute force is
    40 s of that. The KD-tree prune *verifies* its bound rather than assuming
    it, so the two must agree to round-off on shapes chosen to break it."""
    p0, p1 = _wiggly_polygon(seed, 250)
    rng = np.random.default_rng(seed + 100)
    queries = rng.uniform(-2.0, 2.0, size=(3000, 2))
    fast = _point_segment_distance(queries, p0, p1)
    slow = _segment_block(queries, p0, p1)
    assert fast == pytest.approx(slow, rel=0, abs=1e-12)


def test_pruned_distance_survives_wildly_unequal_segment_lengths():
    """The prune bound is `half the longest piece`, so a segment set mixing
    1e-3 and 20.0 lengths is where an unverified bound would go wrong."""
    rng = np.random.default_rng(7)
    p0 = rng.uniform(-1.0, 1.0, size=(300, 2))
    scale = rng.choice([0.001, 0.01, 1.0, 20.0], size=(300, 1))
    p1 = p0 + rng.normal(0.0, 1.0, size=(300, 2)) * scale
    queries = rng.uniform(-5.0, 5.0, size=(3000, 2))
    fast = _point_segment_distance(queries, p0, p1)
    slow = _segment_block(queries, p0, p1)
    assert fast == pytest.approx(slow, rel=0, abs=1e-12)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_bucketed_crossing_count_equals_the_unbucketed_one(seed):
    p0, p1 = _wiggly_polygon(seed, 250)
    rng = np.random.default_rng(seed + 200)
    queries = rng.uniform(-2.0, 2.0, size=(3000, 2))
    assert np.array_equal(_inside_even_odd(queries, p0, p1),
                          _brute_inside(queries, p0, p1))
