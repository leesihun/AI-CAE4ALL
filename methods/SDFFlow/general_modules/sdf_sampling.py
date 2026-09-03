"""
Mesh -> SDF sample generation and synthetic analytic shape families.

Sign convention everywhere: SDF is NEGATIVE inside the solid, POSITIVE outside
(DeepSDF convention). trimesh.signed_distance returns positive inside, so it is
flipped here.
"""

import numpy as np
import trimesh


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


def _signed_distance(mesh, points, chunk=32768):
    """Signed distance to mesh surface, negative inside (chunked).

    Uses libigl when available (fast, robust winding number); falls back to
    trimesh.proximity.signed_distance (requires rtree, slower).
    """
    try:
        import igl
        sd, _, _ = igl.signed_distance(
            points.astype(np.float64),
            mesh.vertices.astype(np.float64),
            mesh.faces.astype(np.int64),
        )
        return sd.astype(np.float32)  # igl: negative inside already
    except ImportError:
        pass

    out = np.empty(len(points), dtype=np.float32)
    for i in range(0, len(points), chunk):
        # trimesh: positive inside -> flip to negative inside
        out[i:i + chunk] = -trimesh.proximity.signed_distance(mesh, points[i:i + chunk])
    return out


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

    Near-surface queries are surface samples perturbed by Gaussian noise at two
    scales (half each); uniform queries fill [-bound, bound]^3.

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
    sigmas = np.where(rng.random(num_near) < 0.5, near_sigmas[0], near_sigmas[1])
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
    sigmas = np.where(rng.random(num_near) < 0.5, near_sigmas[0], near_sigmas[1])
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
