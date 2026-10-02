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


# A Marching Cubes body smaller than this share of the surface's faces is a
# floater (a stray SDF sign flip) and is dropped; anything larger is real
# geometry and is kept. Keeping only the single largest body instead cut real
# parts off multi-part shapes: on the Thingi10K held-out set it raised the
# reconstruction Chamfer p90 from 0.011 to 0.063.
MIN_BODY_FRACTION = 0.005


def sdf_grid_to_mesh(volume, bound=1.0, keep_largest=False, min_body_fraction=MIN_BODY_FRACTION):
    """Marching Cubes at the zero level set. Returns trimesh.Trimesh or None.

    Bodies with fewer than ``min_body_fraction`` of the faces are dropped as
    floaters (0 keeps every body). ``keep_largest=True`` keeps only the single
    largest body instead; the design loop needs that, because gmsh meshes one
    solid.

    The connected-component count of the raw Marching Cubes surface is stored
    on the returned mesh as ``mesh.metadata['body_count_raw']`` (also when it is
    1), and the count that survived as ``mesh.metadata['body_count_kept']``, so
    callers can report how many stray bodies the decoded SDF produced.
    """
    if volume.min() > 0 or volume.max() < 0:
        return None  # no zero crossing
    resolution = volume.shape[0]
    spacing = 2.0 * bound / (resolution - 1)
    verts, faces, _, _ = measure.marching_cubes(volume, level=0.0, spacing=(spacing,) * 3)
    verts = verts - bound
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    body_count_raw = int(mesh.body_count)
    if body_count_raw > 1:
        parts = mesh.split(only_watertight=False)
        if keep_largest:
            mesh = max(parts, key=lambda m: len(m.faces))
        else:
            floor = float(min_body_fraction) * sum(len(m.faces) for m in parts)
            largest = max(parts, key=lambda m: len(m.faces))
            kept = [m for m in parts if m is largest or len(m.faces) >= floor]
            mesh = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)
    # split()/concatenate() build new Trimesh objects, so set the metadata on
    # the kept mesh after selection rather than on the pre-split mesh.
    mesh.metadata['body_count_raw'] = body_count_raw
    mesh.metadata['body_count_kept'] = int(mesh.body_count) if body_count_raw > 1 else 1
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
        'body_count_kept': mesh.metadata.get('body_count_kept'),
    }
