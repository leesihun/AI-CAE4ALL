"""
Mesh -> SDF sample generation and synthetic analytic shape families.

Sign convention everywhere: SDF is NEGATIVE inside the solid, POSITIVE outside
(DeepSDF convention).

Signed-distance backends, tried in this order once per process (see
``sdf_backend_name``):

    igl      libigl winding-number signed distance; negative inside already.
    open3d   ``open3d.t.geometry.RaycastingScene.compute_signed_distance``;
             verified negative inside on a unit icosphere and a rotated box
             (2026-09), so it is used as-is.
    trimesh  ``trimesh.proximity.signed_distance`` returns POSITIVE inside and
             is flipped here (requires rtree, slowest).
"""

import numpy as np
import trimesh


# ---------------------------------------------------------------------------
# Signed-distance backend selection
# ---------------------------------------------------------------------------

_SDF_BACKEND = None


def _resolve_sdf_backend():
    # Every probe swallows Exception, not just ImportError: a partially working
    # install (open3d with `t` but no populated `t.geometry`, an igl build that
    # raises on import) must fall through to the trimesh fallback rather than
    # take down every caller of sdf_backend_name().
    try:
        import igl  # noqa: F401
        return 'igl'
    except Exception:
        pass
    try:
        import open3d  # noqa: F401
        tensor_geometry = getattr(getattr(open3d, 't', None), 'geometry', None)
        if tensor_geometry is not None and hasattr(tensor_geometry, 'RaycastingScene'):
            return 'open3d'
    except Exception:
        pass
    return 'trimesh'


def sdf_backend_name():
    """Name of the signed-distance backend this process uses: 'igl', 'open3d',
    or 'trimesh'. Resolved lazily once per process and announced on stdout the
    first time it is asked for (dataset builders record it as provenance)."""
    global _SDF_BACKEND
    if _SDF_BACKEND is None:
        _SDF_BACKEND = _resolve_sdf_backend()
        print(f'SDF backend: {_SDF_BACKEND}')
    return _SDF_BACKEND


# ---------------------------------------------------------------------------
# Mesh normalization and SDF sampling (real meshes)
# ---------------------------------------------------------------------------

def normalize_mesh(mesh, target_half_extent=0.9):
    """Center the mesh at the origin and uniformly scale the longest bbox side
    to fit inside [-target_half_extent, target_half_extent]^3.

    Returns (mesh, center, scale) where original = normalized / scale + center.
    """
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = (bounds[1] - bounds[0]).max()
    scale = (2.0 * target_half_extent) / max(extent, 1e-12)
    mesh = mesh.copy()
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh, center, scale


def _signed_distance_igl(mesh, points):
    import igl
    sd, _, _ = igl.signed_distance(
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64),
    )
    return np.asarray(sd, dtype=np.float32)  # igl: negative inside already


def _signed_distance_open3d(mesh, points, chunk):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    tmesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.ascontiguousarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(mesh.faces, dtype=np.int32)))
    scene.add_triangles(tmesh)
    out = np.empty(len(points), dtype=np.float32)
    for i in range(0, len(points), chunk):
        query = o3d.core.Tensor(np.ascontiguousarray(points[i:i + chunk], dtype=np.float32))
        try:
            # Odd ray count: a majority vote guards against a ray grazing an
            # edge or vertex exactly. Older Open3D builds lack the kwarg.
            sd = scene.compute_signed_distance(query, nsamples=3)
        except TypeError:
            sd = scene.compute_signed_distance(query)
        # Open3D: negative inside (verified), matches the module convention.
        out[i:i + chunk] = sd.numpy().reshape(-1)
    return out


def _signed_distance_trimesh(mesh, points, chunk):
    out = np.empty(len(points), dtype=np.float32)
    for i in range(0, len(points), chunk):
        # trimesh: positive inside -> flip to negative inside
        out[i:i + chunk] = -trimesh.proximity.signed_distance(mesh, points[i:i + chunk])
    return out


def _signed_distance(mesh, points, chunk=32768):
    """Signed distance to mesh surface, negative inside (chunked).

    Backend order: igl -> open3d RaycastingScene -> trimesh (see module doc).
    """
    backend = sdf_backend_name()
    if backend == 'igl':
        return _signed_distance_igl(mesh, points)
    if backend == 'open3d':
        return _signed_distance_open3d(mesh, points, chunk)
    return _signed_distance_trimesh(mesh, points, chunk)


def _near_sigmas_array(near_sigmas):
    """Validate ``near_sigmas`` (scalar or any sequence of length >= 1)."""
    sig = np.atleast_1d(np.asarray(near_sigmas, dtype=np.float64)).reshape(-1)
    if sig.size < 1 or not np.all(np.isfinite(sig)) or np.any(sig <= 0):
        raise ValueError(
            f'near_sigmas must hold at least one positive finite value, got {near_sigmas!r}')
    return sig


def _choose_near_sigmas(rng, near_sigmas, num_near):
    """One sigma per near point, chosen uniformly at random among ``near_sigmas``.

    Implemented as a single uniform draw binned into ``len(near_sigmas)`` equal
    intervals, so the legacy two-scale case (``u < 0.5 -> sigma[0]``)
    reproduces the historical random stream bit-for-bit.
    """
    sig = _near_sigmas_array(near_sigmas)
    u = rng.random(num_near)
    idx = np.minimum((u * sig.size).astype(np.int64), sig.size - 1)
    return sig[idx]


def _sharp_face_ids(mesh, angle_threshold):
    """Face indices adjacent to a sharp edge (dihedral angle > threshold rad)."""
    angles = mesh.face_adjacency_angles
    if angles is None or len(angles) == 0:
        return np.empty(0, dtype=np.int64)
    sharp_pairs = mesh.face_adjacency[angles > angle_threshold]
    if len(sharp_pairs) == 0:
        return np.empty(0, dtype=np.int64)
    return np.unique(sharp_pairs.reshape(-1)).astype(np.int64)


def _sample_faces(mesh, face_ids, count, rng):
    """Area-weighted point sampling restricted to a subset of faces.

    Returns (points, face_normals). Uses uniform barycentric coordinates so the
    density within each chosen face is uniform.
    """
    areas = mesh.area_faces[face_ids]
    total = areas.sum()
    if total <= 0:
        probs = np.full(len(face_ids), 1.0 / len(face_ids))
    else:
        probs = areas / total
    chosen = rng.choice(face_ids, size=count, p=probs)
    tris = mesh.triangles[chosen]  # (count, 3, 3)
    u = rng.random(count)
    v = rng.random(count)
    over = u + v > 1.0
    u[over] = 1.0 - u[over]
    v[over] = 1.0 - v[over]
    w = 1.0 - u - v
    points = (tris[:, 0] * w[:, None] + tris[:, 1] * u[:, None] + tris[:, 2] * v[:, None])
    return points.astype(np.float32), mesh.face_normals[chosen].astype(np.float32)


def sample_mesh_sdf(mesh, num_surface, num_near, num_uniform,
                    near_sigmas=(0.01, 0.05), bound=1.0, rng=None,
                    sharp_edge_fraction=0.0, sharp_edge_angle=0.5236):
    """Sample surface points/normals and SDF query points from a watertight mesh.

    Near-surface queries are surface samples perturbed by Gaussian noise whose
    scale is drawn uniformly at random per point from ``near_sigmas`` (any
    sequence of length >= 1; the default two scales are used half each);
    uniform queries fill [-bound, bound]^3.

    `sharp_edge_fraction` (Dora-style Sharp Edge Sampling) routes that fraction
    of surface points onto faces adjacent to sharp edges (dihedral angle above
    `sharp_edge_angle` radians, default 30 deg), so high-curvature features are
    over-represented in the encoder point cloud and near-surface queries. It
    falls back to uniform area sampling when a mesh has no sharp edges.

    Returns dict with surface_points, surface_normals, sdf_points, sdf_values.
    """
    rng = rng or np.random.default_rng()

    num_sharp = int(round(num_surface * float(sharp_edge_fraction)))
    if num_sharp > 0:
        sharp_ids = _sharp_face_ids(mesh, sharp_edge_angle)
        num_sharp = num_sharp if len(sharp_ids) > 0 else 0
    num_area = num_surface - num_sharp

    area_points, area_face_idx = trimesh.sample.sample_surface(
        mesh, num_area, seed=int(rng.integers(2**31)))
    area_normals = mesh.face_normals[area_face_idx]
    if num_sharp > 0:
        sharp_points, sharp_normals = _sample_faces(mesh, sharp_ids, num_sharp, rng)
        surface_points = np.concatenate([area_points, sharp_points], axis=0)
        surface_normals = np.concatenate([area_normals, sharp_normals], axis=0)
    else:
        surface_points = area_points
        surface_normals = area_normals

    base_idx = rng.integers(0, num_surface, size=num_near)
    base = surface_points[base_idx]
    sigmas = _choose_near_sigmas(rng, near_sigmas, num_near)
    near_pts = base + rng.normal(size=(num_near, 3)) * sigmas[:, None]

    uni_pts = rng.uniform(-bound, bound, size=(num_uniform, 3))

    sdf_points = np.concatenate([near_pts, uni_pts], axis=0).astype(np.float32)
    sdf_values = _signed_distance(mesh, sdf_points)

    return {
        'surface_points': surface_points.astype(np.float32),
        'surface_normals': surface_normals.astype(np.float32),
        'sdf_points': sdf_points,
        'sdf_values': sdf_values.astype(np.float32),
    }


def mesh_descriptors(mesh):
    """Automatic geometric condition vector: bbox extents, volume, area.

    These are 'free' labels available for any shape; used for conditional FM.
    """
    extents = mesh.extents
    try:
        volume = float(abs(mesh.volume)) if mesh.is_watertight else float(mesh.convex_hull.volume)
    except Exception:
        volume = float(np.prod(extents))
    area = float(mesh.area)
    return np.array([extents[0], extents[1], extents[2], volume, area], dtype=np.float32)


COND_NAMES = ['bbox_x', 'bbox_y', 'bbox_z', 'volume', 'area']


# ---------------------------------------------------------------------------
# Synthetic analytic shape family (for pipeline validation / smoke tests)
# ---------------------------------------------------------------------------

def _sdf_box(p, half):
    q = np.abs(p) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def _sdf_sphere(p, r):
    return np.linalg.norm(p, axis=-1) - r


def _sdf_cylinder(p, r, h):
    d_xy = np.linalg.norm(p[..., :2], axis=-1) - r
    d_z = np.abs(p[..., 2]) - h
    outside = np.linalg.norm(np.maximum(np.stack([d_xy, d_z], axis=-1), 0.0), axis=-1)
    inside = np.minimum(np.maximum(d_xy, d_z), 0.0)
    return outside + inside


def _sdf_torus(p, R, r):
    q = np.stack([np.linalg.norm(p[..., :2], axis=-1) - R, p[..., 2]], axis=-1)
    return np.linalg.norm(q, axis=-1) - r


def _random_rotation(rng):
    a = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def synthetic_sdf(rng):
    """Random union of 1-3 rotated/translated primitives; returns sdf(points)->values."""
    prims = []
    for _ in range(int(rng.integers(1, 4))):
        kind = rng.choice(['box', 'sphere', 'cylinder', 'torus'])
        rot = _random_rotation(rng)
        offset = rng.uniform(-0.35, 0.35, size=3)
        if kind == 'box':
            half = rng.uniform(0.15, 0.45, size=3)
            fn = lambda p, half=half: _sdf_box(p, half)
        elif kind == 'sphere':
            r = rng.uniform(0.2, 0.45)
            fn = lambda p, r=r: _sdf_sphere(p, r)
        elif kind == 'cylinder':
            r, h = rng.uniform(0.12, 0.3), rng.uniform(0.2, 0.45)
            fn = lambda p, r=r, h=h: _sdf_cylinder(p, r, h)
        else:
            R_, r_ = rng.uniform(0.25, 0.4), rng.uniform(0.08, 0.15)
            fn = lambda p, R_=R_, r_=r_: _sdf_torus(p, R_, r_)
        prims.append((rot, offset, fn))

    def sdf(points):
        vals = None
        for rot, offset, fn in prims:
            local = (points - offset) @ rot
            v = fn(local)
            vals = v if vals is None else np.minimum(vals, v)
        return vals

    return sdf


def synthetic_sample(rng, num_surface, num_near, num_uniform,
                     near_sigmas=(0.01, 0.05), bound=1.0, mc_resolution=96):
    """Build one synthetic shape: analytic SDF -> Marching Cubes mesh for surface
    points, exact analytic SDF for query labels. Returns (sample_dict, cond)."""
    from skimage import measure

    sdf = synthetic_sdf(rng)

    xs = np.linspace(-bound, bound, mc_resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, xs, xs, indexing='ij'), axis=-1)
    volume = sdf(grid.reshape(-1, 3)).reshape(mc_resolution, mc_resolution, mc_resolution)

    spacing = 2.0 * bound / (mc_resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=(spacing,) * 3)
    verts -= bound
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    surface_points, face_idx = trimesh.sample.sample_surface(mesh, num_surface, seed=int(rng.integers(2**31)))
    surface_normals = mesh.face_normals[face_idx]

    base = surface_points[rng.integers(0, num_surface, size=num_near)]
    sigmas = _choose_near_sigmas(rng, near_sigmas, num_near)
    near_pts = base + rng.normal(size=(num_near, 3)) * sigmas[:, None]
    uni_pts = rng.uniform(-bound, bound, size=(num_uniform, 3))

    sdf_points = np.concatenate([near_pts, uni_pts], axis=0).astype(np.float32)
    sdf_values = sdf(sdf_points).astype(np.float32)

    sample = {
        'surface_points': surface_points.astype(np.float32),
        'surface_normals': surface_normals.astype(np.float32),
        'sdf_points': sdf_points,
        'sdf_values': sdf_values,
    }
    return sample, mesh_descriptors(mesh)
