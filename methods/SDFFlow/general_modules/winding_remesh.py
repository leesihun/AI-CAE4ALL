"""Rebuild a watertight surface from a broken triangle soup by its winding number.

MeshFix closes holes by stitching boundary loops and, with ``joincomp``, keeps
only what it can stitch. On a mesh whose faces are all there but do not share
vertices along the seams (every raw MCB castle nut), it throws away most of the
surface -- a median 24% of the faces survived -- and the fragment it returns is
then normalized to fill the box as if it were the whole part.

The generalized winding number (Jacobson et al. 2013) does not need a closed or
manifold surface: it is the solid angle the faces subtend at a point divided by
4 pi, close to 1 inside and 0 outside even through seams and small holes. This
module evaluates it on a grid, combines its sign with the exact unsigned
distance to the faces, and runs Marching Cubes at the zero level. The result is
watertight by construction and keeps every face's geometry to within one grid
cell.

Only the cells near the surface need the exact winding number: everywhere else
the sign is taken from a coarse grid, which is safe because a band of several
fine cells separates the two.
"""

import math

import numpy as np


def winding_number(vertices, faces, points, chunk=None):
    """Generalized winding number of ``points`` [Q, 3] w.r.t. a triangle soup.

    Van Oosterom-Strackee solid angle per triangle, summed. Exact (not the
    fast multipole approximation), so the cost is Q x F; callers keep Q small.
    Evaluated with torch (float64) because the Q x F broadcast is ~50x faster
    there than in numpy.
    """
    import torch

    v = torch.as_tensor(np.asarray(vertices, dtype=np.float64))
    f = torch.as_tensor(np.asarray(faces, dtype=np.int64))
    p = torch.as_tensor(np.asarray(points, dtype=np.float64))
    a0, b0, c0 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    if chunk is None:
        chunk = max(16, int(4_000_000 // max(len(f), 1)))
    out = torch.empty(len(p), dtype=torch.float64)
    with torch.no_grad():
        for i in range(0, len(p), chunk):
            q = p[i:i + chunk, None, :]
            a, b, c = a0[None] - q, b0[None] - q, c0[None] - q
            la, lb, lc = a.norm(dim=2), b.norm(dim=2), c.norm(dim=2)
            det = (a * torch.cross(b, c, dim=2)).sum(dim=2)
            den = (la * lb * lc + (a * b).sum(dim=2) * lc
                   + (b * c).sum(dim=2) * la + (c * a).sum(dim=2) * lb)
            out[i:i + chunk] = (2.0 * torch.atan2(det, den)).sum(dim=1) / (4.0 * math.pi)
    return out.numpy()


def _unsigned_distance(vertices, faces, points):
    """Exact point-to-triangle distance; open3d when present, trimesh otherwise."""
    try:
        import open3d as o3d

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, dtype=np.float32)),
                            o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)))
        out = np.empty(len(points), dtype=np.float64)
        for i in range(0, len(points), 1_000_000):
            q = o3d.core.Tensor(np.asarray(points[i:i + 1_000_000], dtype=np.float32))
            out[i:i + 1_000_000] = scene.compute_distance(q).numpy()
        return out
    except ImportError:
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        _, dist, _ = trimesh.proximity.closest_point(mesh, points)
        return np.asarray(dist, dtype=np.float64)


def winding_remesh(mesh, resolution=192, band_cells=4.0, coarse_factor=4):
    """Watertight Marching Cubes surface of the winding-number solid of ``mesh``.

    The grid spans the mesh bounds plus two cells of padding, so the surface
    never touches the grid edge and always closes. Returns ``(mesh, info)``
    where ``info`` holds the winding volume and how many points were evaluated
    exactly; the mesh is None when the winding number finds no solid at all.
    """
    import trimesh
    from skimage import measure

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    cell = float((hi - lo).max()) / (resolution - 5)
    lo = lo - 2.0 * cell
    dims = np.ceil((hi + 2.0 * cell - lo) / cell).astype(int) + 1
    axes = [lo[d] + cell * np.arange(dims[d]) for d in range(3)]

    # Coarse sign: winding number on every coarse_factor-th node.
    coarse_axes = [ax[::coarse_factor] for ax in axes]
    coarse = np.stack(np.meshgrid(*coarse_axes, indexing='ij'), axis=-1).reshape(-1, 3)
    coarse_inside = (winding_number(vertices, faces, coarse) > 0.5).reshape(
        [len(ax) for ax in coarse_axes])
    index = [np.minimum(np.round(np.arange(dims[d]) / coarse_factor).astype(int),
                        coarse_inside.shape[d] - 1) for d in range(3)]
    inside = coarse_inside[np.ix_(*index)]

    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    dist = _unsigned_distance(vertices, faces, grid)
    # A coarse node decides the fine nodes within coarse_factor/2 cells of it
    # (sqrt(3) times that along a diagonal); outside a band that wide the two
    # cannot sit on opposite sides of the surface.
    band = dist < max(band_cells, 0.5 * math.sqrt(3.0) * coarse_factor + 1.0) * cell
    flat_inside = inside.reshape(-1).copy()
    flat_inside[band] = winding_number(vertices, faces, grid[band]) > 0.5
    # A node lying exactly on a face would give Marching Cubes a zero value and
    # it then emits zero-area triangles that leave the merged surface open.
    dist = np.maximum(dist, 1e-4 * cell)
    sdf = np.where(flat_inside, -dist, dist).reshape(dims)
    info = {'winding_resolution': int(resolution), 'winding_cell': cell,
            'winding_band_points': int(band.sum()), 'winding_grid_points': int(len(grid))}
    if sdf.min() >= 0 or sdf.max() <= 0:
        return None, info
    verts, tris, _, _ = measure.marching_cubes(sdf, level=0.0, spacing=(cell,) * 3)
    out = trimesh.Trimesh(vertices=verts + lo, faces=tris, process=True)
    # Marching Cubes orients faces by the gradient, which points outward for a
    # negative-inside field; fix_normals only guards against a flipped body.
    trimesh.repair.fix_normals(out, multibody=True)
    info['winding_volume'] = float(abs(out.volume)) if out.is_watertight else None
    return out, info
