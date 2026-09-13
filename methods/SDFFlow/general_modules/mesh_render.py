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


def plot_mesh_strip(meshes, labels, reports, path, dpi=180, max_faces=0, title=None,
                    colors=None):
    """Render N meshes side by side with identical axes and camera settings.

    `meshes` entries may be None (a decode without a zero crossing); that
    panel is drawn empty with its label so the strip keeps one panel per step.
    `colors` defaults to endpoints blue / interior panels orange.

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
    fig = plt.figure(figsize=(max(5.3 * n, 6.0), 5.8), dpi=dpi, facecolor='white')

    for index, (mesh, label, report, color) in enumerate(
            zip(meshes, labels, reports, colors), start=1):
        ax = fig.add_subplot(1, n, index, projection='3d')
        if mesh is not None:
            triangles = mesh.triangles
            if max_faces > 0 and len(triangles) > max_faces:
                selected = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
                triangles = triangles[selected]
            surface = Poly3DCollection(
                triangles, facecolor=color, edgecolor='none', linewidth=0.0, alpha=1.0)
            ax.add_collection3d(surface)
        else:
            ax.text(0.0, 0.0, 0.0, 'no zero crossing', ha='center', va='center', fontsize=11)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
        volume = report.get('volume')
        volume_text = f'{volume:.4f}' if volume is not None else 'n/a'
        faces = report.get('faces')
        faces_text = f'{faces:,}' if faces is not None else 'n/a'
        ax.set_title(f'{label}\nvolume={volume_text}, faces={faces_text}',
                     fontsize=12, pad=0, y=0.90)

    fig.suptitle(title or 'SDFFlow latent interpolation', fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path
