"""Surface reconstruction for the periodic train/test visualization.

The renderer draws triangles, so a mesh built only from quads -- iFEM dense48 is
a structured 9x9 grid with no diagonals -- used to reconstruct zero faces. The
dump still wrote its HDF5, plot_mesh_comparison returned early, render_plot_data
reported success, and no PNG was ever produced: a silent no-op that the training
log described as "All visualizations complete!".

These tests pin the quad fallback and, just as importantly, that a mesh which
already has triangles is left byte-identical.
"""
import numpy as np
import pytest

from general_modules.mesh_utils_fast import (
    _edges_to_3cycles_cpu,
    edges_to_triangles_optimized,
    quads_to_triangles,
)


def _grid_edges(nx, ny):
    """Axis-aligned edges of an nx-by-ny quad grid, bidirectional."""
    idx = np.arange(nx * ny).reshape(ny, nx)
    src, dst = [], []
    for a, b in ((idx[:, :-1].ravel(), idx[:, 1:].ravel()),
                 (idx[:-1, :].ravel(), idx[1:, :].ravel())):
        src.append(a)
        dst.append(b)
    u = np.concatenate(src)
    v = np.concatenate(dst)
    return np.stack([np.concatenate([u, v]), np.concatenate([v, u])])


def _edge_set(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


def test_quad_grid_yields_two_triangles_per_cell():
    nx = ny = 9
    edges = _grid_edges(nx, ny)
    assert _edges_to_3cycles_cpu(edges).shape[0] == 0, "grid should have no 3-cycles"

    faces = edges_to_triangles_optimized(edges)
    assert faces.shape == (2 * (nx - 1) * (ny - 1), 3)
    assert faces.min() >= 0 and faces.max() < nx * ny


def test_fallback_triangles_are_not_invented():
    """Each emitted triangle splits a real quad: two of its three sides are
    real mesh edges and the third is the quad diagonal."""
    edges = _grid_edges(6, 5)
    known = _edge_set(edges)
    for tri in edges_to_triangles_optimized(edges):
        sides = ((tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2]))
        real = sum((int(a), int(b)) in known for a, b in sides)
        assert real >= 2, f"triangle {tri} has only {real} real edges"


def test_triangular_mesh_is_untouched():
    """A mesh with 3-cycles must not reach the fallback at all."""
    # two triangles sharing an edge
    tri_edges = np.array([[0, 1, 2, 0, 1, 3, 1, 0, 2, 1, 3, 1],
                          [1, 2, 0, 2, 3, 1, 0, 1, 1, 3, 2, 2]])
    base = _edges_to_3cycles_cpu(tri_edges)
    assert base.shape[0] > 0
    assert np.array_equal(edges_to_triangles_optimized(tri_edges), base)


def test_single_quad():
    quad = np.array([[0, 1, 2, 3, 1, 2, 3, 0],
                     [1, 2, 3, 0, 0, 1, 2, 3]])
    faces = quads_to_triangles(quad)
    assert faces.shape == (2, 3)
    assert set(faces.ravel().tolist()) == {0, 1, 2, 3}


def test_degenerate_inputs_return_empty():
    assert quads_to_triangles(np.zeros((2, 0), dtype=np.int64)).shape == (0, 3)
    # a path has neither 3-cycles nor 4-cycles
    path = np.array([[0, 1, 2, 1, 2, 3], [1, 2, 3, 0, 1, 2]])
    assert edges_to_triangles_optimized(path).shape == (0, 3)


def test_high_degree_nodes_are_skipped():
    """A hub node joined to many leaves must not explode the O(d^3) scan."""
    hub = np.array([[0] * 40 + list(range(1, 41)),
                    list(range(1, 41)) + [0] * 40])
    assert quads_to_triangles(hub, max_degree=16).shape == (0, 3)


@pytest.mark.parametrize("shape", [(2, 2), (3, 7), (10, 4)])
def test_grid_face_count_scales(shape):
    nx, ny = shape
    faces = edges_to_triangles_optimized(_grid_edges(nx, ny))
    assert faces.shape[0] == 2 * (nx - 1) * (ny - 1)
