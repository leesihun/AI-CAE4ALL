"""Preview render of what was ingested.

This method trains nothing, so there is no periodic test to hang a picture on;
the equivalent check is "did I actually mesh the geometry I meant to". One
strip figure over the first few samples answers it -- a reader that silently
produced a degenerate mesh, a unit mix-up between files, or a point-cloud
resample that collapsed shows up immediately, while the numeric summary line
(nodes/cells/bbox) does not make any of them obvious.

Surface meshes (3-node cells) are drawn as shaded triangles; anything else --
volume meshes, mixed connectivity -- falls back to a node scatter, because
extracting a boundary from a tet mesh is work this preview does not need to do.

matplotlib is imported lazily and a missing install degrades to a printed note.
"""

from __future__ import annotations

import os

import numpy as np

_MISSING_BACKEND_WARNED = False


def _load_backend():
    global _MISSING_BACKEND_WARNED
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        return plt, Poly3DCollection
    except Exception as exc:  # pragma: no cover - environment dependent
        if not _MISSING_BACKEND_WARNED:
            _MISSING_BACKEND_WARNED = True
            print(f"  [preview] matplotlib unavailable ({exc}); skipping preview render.")
        return None, None


def _shade(faces_xyz, base_rgb, light=(0.3, 0.45, 0.84)):
    """Lambertian shading so a surface reads as a solid, not a flat silhouette."""
    a = faces_xyz[:, 1] - faces_xyz[:, 0]
    b = faces_xyz[:, 2] - faces_xyz[:, 0]
    normals = np.cross(a, b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    direction = np.asarray(light, dtype=float)
    direction /= np.linalg.norm(direction)
    intensity = 0.55 + 0.45 * np.abs(normals @ direction)
    colors = np.tile(np.asarray(base_rgb, dtype=float), (faces_xyz.shape[0], 1))
    return np.clip(colors * intensity[:, None], 0.0, 1.0)


def _draw_sample(ax, sample, Poly3DCollection, max_faces, max_points, rng):
    coords = np.asarray(sample["coords"], dtype=float)
    conn = sample.get("conn")
    nodes_per_elem = int(sample.get("nodes_per_elem", 0) or 0)
    drawn_as = "points"

    if conn is not None and nodes_per_elem == 3:
        faces = np.asarray(conn, dtype=np.int64).reshape(-1, 3)
        if max_faces > 0 and faces.shape[0] > max_faces:
            faces = faces[np.linspace(0, faces.shape[0] - 1, max_faces, dtype=np.int64)]
        triangles = coords[faces]
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolors=_shade(triangles, (0.23, 0.51, 0.77)),
            edgecolors="none", linewidth=0.0))
        drawn_as = f"{faces.shape[0]:,} triangles"
    else:
        points = coords
        if max_points > 0 and points.shape[0] > max_points:
            points = points[rng.choice(points.shape[0], max_points, replace=False)]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.2,
                   c="#3B82C4", depthshade=False, linewidths=0)
        drawn_as = f"{points.shape[0]:,} nodes"

    # Each panel frames its own geometry: these files are not pre-normalized and
    # a shared box would hide everything but the largest part.
    low, high = coords.min(axis=0), coords.max(axis=0)
    span = high - low
    pad = 0.02 * float(span.max()) if float(span.max()) > 0 else 1.0
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_zlim(low[2] - pad, high[2] + pad)
    ax.set_box_aspect(tuple(np.where(span > 0, span, 1.0)))
    ax.view_init(elev=24, azim=-58)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    return drawn_as, span


def render_geometry_preview(samples, path, *, max_panels=6, max_faces=60000,
                            max_points=20000, dpi=150, seed=0, title=None):
    """Write one panel per ingested sample. Returns the path, or None."""
    plt, Poly3DCollection = _load_backend()
    if plt is None:
        return None

    shown = list(samples)[:max(1, int(max_panels))]
    if not shown:
        return None

    rng = np.random.default_rng(seed)
    columns = min(3, len(shown))
    rows = (len(shown) + columns - 1) // columns
    fig = plt.figure(figsize=(5.2 * columns, 5.0 * rows), dpi=dpi, facecolor="white")

    for index, sample in enumerate(shown, start=1):
        ax = fig.add_subplot(rows, columns, index, projection="3d")
        drawn_as, span = _draw_sample(ax, sample, Poly3DCollection, max_faces, max_points, rng)
        watertight = sample.get("watertight")
        watertight_text = "n/a" if watertight is None else ("watertight" if watertight else "NOT watertight")
        ax.set_title(f"{sample.get('source', f'sample {index}')}\n"
                     f"{drawn_as}, {watertight_text}\n"
                     f"bbox=[{span[0]:.3g}, {span[1]:.3g}, {span[2]:.3g}]",
                     fontsize=10, pad=0, y=0.93)

    total = len(list(samples))
    header = title or "geometry_ingest preview"
    if total > len(shown):
        header += f"  (first {len(shown)} of {total})"
    fig.suptitle(header, fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.02, hspace=0.08)

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
