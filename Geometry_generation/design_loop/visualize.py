"""Side-by-side stress rendering of the baseline and the optimized bracket.

Drawn with matplotlib's 3-D polygon collection rather than a VTK/pyvista
pipeline: the loop has to run headless on machines with no usable GL context,
and the outer triangles of a ~30k-tet part are few enough to raster directly.
"""

import numpy as np


def _shade(faces_xyz, base_rgba, light=(0.3, 0.45, 0.84)):
    """Lambertian shading so the geometry reads as a solid, not a flat blob."""
    a = faces_xyz[:, 1] - faces_xyz[:, 0]
    b = faces_xyz[:, 2] - faces_xyz[:, 0]
    normals = np.cross(a, b)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    direction = np.asarray(light, dtype=float)
    direction /= np.linalg.norm(direction)
    intensity = 0.55 + 0.45 * np.abs(normals @ direction)
    shaded = base_rgba.copy()
    shaded[:, :3] *= intensity[:, None]
    return np.clip(shaded, 0.0, 1.0)


def _draw(ax, fields, vmax, cmap, title):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nodes = fields['nodes_norm']
    faces = fields['faces']
    vm = fields['von_mises_nodal'] / 1e6                      # MPa
    tri = nodes[faces]
    face_values = vm[faces].mean(axis=1)

    colors = _shade(tri, cmap(np.clip(face_values / vmax, 0.0, 1.0)))
    collection = Poly3DCollection(tri, facecolors=colors, edgecolors='none')
    ax.add_collection3d(collection)

    # Interface markers: what is held and where the load goes in.
    ax.scatter(*nodes[fields['mount']].T, s=1.2, c='#111111', alpha=0.55,
               depthshade=False, label='fixed (mount pads)')
    ax.scatter(*nodes[fields['lug']].T, s=1.6, c='#d62728', alpha=0.9,
               depthshade=False, label='loaded (lug)')

    # Frame each axis on its own range and carry the shape into box_aspect: a
    # cubic box around an elongated bracket is undistorted but mostly empty.
    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    pad = 0.02 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(lo[2] - pad, hi[2] + pad)
    ax.set_box_aspect(tuple(hi - lo))
    ax.view_init(elev=24, azim=-62)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=0)
    ax.legend(loc='lower center', fontsize=8, frameon=False, ncol=2,
              bbox_to_anchor=(0.5, 0.04))


def render_comparison(path, baseline_fields, baseline_stats,
                      optimized_fields, optimized_stats, load_case, dpi=150):
    """Write a two-panel von Mises comparison on a shared colour scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = cm.get_cmap('turbo') if hasattr(cm, 'get_cmap') else plt.get_cmap('turbo')
    vmax = max(np.percentile(baseline_fields['von_mises_nodal'], 99.5),
               np.percentile(optimized_fields['von_mises_nodal'], 99.5)) / 1e6

    fig = plt.figure(figsize=(14, 5.6))
    panels = (
        (baseline_fields, baseline_stats, 'Baseline'),
        (optimized_fields, optimized_stats, 'Optimized'),
    )
    for i, (fields, stats, name) in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        title = (f"{name}\n{stats['mass']:.4f} kg   "
                 f"peak {stats['peak_von_mises'] / 1e6:.1f} MPa   "
                 f"{stats['max_displacement'] * 1e3:.4f} mm")
        _draw(ax, fields, vmax, cmap, title)

    mappable = cm.ScalarMappable(norm=Normalize(0.0, vmax), cmap=cmap)
    bar = fig.colorbar(mappable, ax=fig.axes, fraction=0.022, pad=0.06)
    bar.set_label('von Mises (MPa)')
    fig.suptitle(f'DeepJEB bracket, {load_case} load case', fontsize=13)
    fig.subplots_adjust(left=0.0, right=0.88, top=0.90, bottom=0.0, wspace=0.0)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return path
