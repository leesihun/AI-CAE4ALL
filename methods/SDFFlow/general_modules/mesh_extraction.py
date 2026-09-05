"""Latent -> SDF grid -> Marching Cubes -> trimesh mesh / STL export."""

import numpy as np
import torch
import trimesh
from skimage import measure


@torch.no_grad()
def decode_sdf_grid(vae, z_flat, resolution=128, bound=1.0, chunk=65536, device='cpu'):
    """Evaluate the decoder on a dense grid. z_flat: (1, D). Returns (R, R, R) numpy."""
    xs = torch.linspace(-bound, bound, resolution, device=device)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing='ij'), dim=-1).reshape(-1, 3)
    values = torch.empty(grid.shape[0], device=device)
    for i in range(0, grid.shape[0], chunk):
        pts = grid[i:i + chunk].unsqueeze(0)
        values[i:i + chunk] = vae.decode_flat(z_flat, pts).squeeze(0).float()
    return values.reshape(resolution, resolution, resolution).cpu().numpy()


def sdf_grid_to_mesh(volume, bound=1.0, keep_largest=True):
    """Marching Cubes at the zero level set. Returns trimesh.Trimesh or None.

    The connected-component count of the raw Marching Cubes surface -- before
    ``keep_largest`` discards the smaller bodies -- is stored on the returned
    mesh as ``mesh.metadata['body_count_raw']`` (also when it is 1), so callers
    can report how many stray bodies the decoded SDF produced.
    """
    if volume.min() > 0 or volume.max() < 0:
        return None  # no zero crossing
    resolution = volume.shape[0]
    spacing = 2.0 * bound / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=(spacing,) * 3)
    verts = verts - bound
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    body_count_raw = int(mesh.body_count)
    if keep_largest and body_count_raw > 1:
        parts = mesh.split(only_watertight=False)
        mesh = max(parts, key=lambda m: len(m.faces))
    # split() builds new Trimesh objects, so set the metadata on the kept mesh
    # after selection rather than on the pre-split mesh.
    mesh.metadata['body_count_raw'] = body_count_raw
    return mesh


def body_count_raw(mesh):
    """``mesh.metadata['body_count_raw']`` as int, or None when not recorded."""
    if mesh is None:
        return None
    value = getattr(mesh, 'metadata', {}).get('body_count_raw')
    return int(value) if value is not None else None


def mesh_report(mesh):
    if mesh is None:
        return {'valid': False}
    return {
        'valid': True,
        'watertight': bool(mesh.is_watertight),
        'vertices': int(len(mesh.vertices)),
        'faces': int(len(mesh.faces)),
        'volume': float(abs(mesh.volume)) if mesh.is_watertight else None,
        'area': float(mesh.area),
        'extents': [float(e) for e in mesh.extents],
        'body_count_raw': body_count_raw(mesh),
    }
