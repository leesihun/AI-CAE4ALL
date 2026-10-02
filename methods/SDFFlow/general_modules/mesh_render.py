"""Headless mesh rendering shared by the training loop and the inference modes.

Drawn with matplotlib's 3-D polygon collection rather than a VTK/pyvista
pipeline: this runs on boxes with no usable GL context, and a marching-cubes
surface is few enough triangles to raster directly (``max_faces`` thins it
further when it is not).

``plot_mesh_strip`` was factored out of ``inference_profiles/interpolate.py``
so the periodic train/test renders (train_vae's reconstructions, train_fm's
samples) draw the same picture the interpolation figures do, on the same axes
and camera, instead of a second look-alike implementation.

matplotlib is imported lazily: a missing install degrades to a printed note and
a ``None`` return, never a dead training run.
"""

import os

import numpy as np

ENDPOINT_COLOR = '#3B82C4'
MIDDLE_COLOR = '#E68A2E'
FLOATER_COLOR = '#D64545'

_MISSING_BACKEND_WARNED = False


def _load_backend():
    """Return (pyplot, Poly3DCollection), or (None, None) once, quietly."""
    global _MISSING_BACKEND_WARNED
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        return plt, Poly3DCollection
    except Exception as exc:  # pragma: no cover - environment dependent
        if not _MISSING_BACKEND_WARNED:
            _MISSING_BACKEND_WARNED = True
            print(f'  [viz] matplotlib unavailable ({exc}); skipping mesh renders.')
        return None, None


def _face_colors(mesh, color, view_dir, floater_color=FLOATER_COLOR):
    """Per-face RGBA: Lambert shading toward the camera, minor bodies recoloured.

    Flat single-colour faces hide every hole and edge of a marching-cubes
    surface (a nut reads as a disc), so each face is darkened by the angle
    between its normal and a light placed at the camera. Bodies other than the
    largest are drawn in ``floater_color`` so a stray body is visible instead of
    blending into the shape.
    """
    from matplotlib.colors import to_rgb
    import trimesh

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    light = np.asarray(view_dir, dtype=np.float64)
    light = light / np.linalg.norm(light)
    lambert = np.abs(normals @ light)
    shade = 0.30 + 0.70 * lambert
    base = np.tile(np.asarray(to_rgb(color)), (len(normals), 1))
    if len(mesh.faces) > 1:
        labels = trimesh.graph.connected_component_labels(
            mesh.face_adjacency, node_count=len(mesh.faces))
        counts = np.bincount(labels)
        if len(counts) > 1:
            base[labels != int(np.argmax(counts))] = to_rgb(floater_color)
    rgb = np.clip(base * shade[:, None], 0.0, 1.0)
    return np.concatenate([rgb, np.ones((len(rgb), 1))], axis=1)


def _view_direction(elev, azim):
    elev, azim = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])


def plot_mesh_strip(meshes, labels, reports, path, dpi=180, max_faces=0, title=None,
                    colors=None, ncols=None):
    """Render N meshes with identical axes and camera settings.

    `meshes` entries may be None (a decode without a zero crossing); that
    panel is drawn empty with its label so the strip keeps one panel per step.
    `colors` defaults to endpoints blue / interior panels orange. Faces are
    shaded toward the camera, and bodies other than the largest are drawn red.
    `ncols` wraps the panels into a grid (default: one row).

    Returns the written path, or None when matplotlib is unavailable.
    """
    plt, Poly3DCollection = _load_backend()
    if plt is None:
        return None

    n = len(meshes)
    if n == 0:
        raise ValueError('plot_mesh_strip needs at least one panel')
    if colors is None:
        colors = tuple(ENDPOINT_COLOR if i in (0, n - 1) else MIDDLE_COLOR for i in range(n))
    ncols = n if not ncols else max(1, min(int(ncols), n))
    nrows = (n + ncols - 1) // ncols
    fig = plt.figure(figsize=(max(5.3 * ncols, 6.0), 5.8 * nrows), dpi=dpi, facecolor='white')
    elev, azim = 24, -58
    view_dir = _view_direction(elev, azim)

    for index, (mesh, label, report, color) in enumerate(
            zip(meshes, labels, reports, colors), start=1):
        ax = fig.add_subplot(nrows, ncols, index, projection='3d')
        if mesh is not None and len(mesh.faces):
            face_ids = np.arange(len(mesh.faces))
            if max_faces > 0 and len(face_ids) > max_faces:
                face_ids = np.linspace(0, len(face_ids) - 1, max_faces, dtype=np.int64)
            facecolors = _face_colors(mesh, color, view_dir)[face_ids]
            surface = Poly3DCollection(
                mesh.triangles[face_ids], facecolors=facecolors, edgecolor='none',
                linewidth=0.0, alpha=1.0)
            ax.add_collection3d(surface)
        else:
            ax.text(0.0, 0.0, 0.0, 'no zero crossing', ha='center', va='center', fontsize=11)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
        volume = report.get('volume')
        volume_text = f'{volume:.4f}' if volume is not None else 'n/a'
        faces = report.get('faces')
        faces_text = f'{faces:,}' if faces is not None else 'n/a'
        bodies = report.get('body_count_kept')
        raw = report.get('body_count_raw')
        bodies_text = ''
        if bodies is not None:
            bodies_text = f', bodies={bodies}' + (f' (raw {raw})' if raw not in (None, bodies) else '')
        ax.set_title(f'{label}\nvolume={volume_text}, faces={faces_text}{bodies_text}',
                     fontsize=12, pad=0, y=0.90)

    fig.suptitle(title or 'SDFFlow latent interpolation', fontsize=16, y=0.99 if nrows > 1 else 0.97)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1.0 - 0.16 / nrows,
                        wspace=0.01, hspace=0.05)

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path
