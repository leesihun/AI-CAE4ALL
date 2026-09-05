"""
Differentiable geometric descriptors of a decoded SDF (the C2/E2 "proxy").

Every guidance and correction mechanism in `descriptor_guidance.py` /
`descriptor_refinement.py` needs a scalar descriptor of the decoded shape that
is differentiable with respect to the latent. Marching Cubes is not, so the
proxy is a smooth occupancy integral on a fixed grid:

    occ(x)  = sigmoid(-SDF(x) / tau)          # soft inside indicator (inside = SDF < 0)
    volume  = sum_cells occ(x) * h^3
    area    = sum_cells ||grad occ(x)|| * h^3   # co-area formula on the occupancy field

with CELL-CENTER quadrature: the grid has `resolution` cells per axis over
[-bound, bound], cell centres at ``-bound + h * (i + 0.5)``, ``h = 2 * bound /
resolution``. (The pilot in GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md used
node-centred `linspace` samples weighted by an inconsistent `h^3`; the cell
form is the one whose Riemann sum is unbiased for a smooth integrand.) The
gradient is a finite difference on the occupancy grid (`torch.gradient`:
central differences inside, one-sided at the boundary), so everything stays a
plain autograd graph through `vae.decode_flat`.

The proxy is BIASED relative to the Marching Cubes measurement the export path
reports (the sigmoid smears the boundary over ~4 tau, and the decoder is not
an exact metric SDF), which is why `descriptor_calibration.py` fits an affine
map proxy ~= a * true + b per descriptor and every consumer works in proxy
units. For an exact sphere SDF the smearing alone overestimates the volume by
``8 pi r tau^2 pi^2 / 6`` (about 4% at r = 0.5, tau = 0.032) and the area by
``4 pi tau^2 pi^2 / 3``; the tests pin the proxy against these analytic values.

Only `volume` and `area` have soft proxies (`SUPPORTED_SOFT_NAMES`). The bbox
extents are not proxied: they are near-constant in DeepJEB (bbox_y has zero
train-split std) and a differentiable extent has no clean occupancy form.
Callers filter their target names through `supported_soft_names`.

`MockDecoder` is a stand-in for the VAE whose latent parametrises an analytic
sphere (``radius = z_flat[:, 0]``), so the refinement and guidance code can be
exercised without a trained checkpoint.
"""

import math

import torch
import torch.nn as nn

SUPPORTED_SOFT_NAMES = ('volume', 'area')

# Below this squared gradient magnitude a cell contributes no area and its
# gradient is zeroed rather than blowing up through sqrt at 0.
_AREA_EPS = 1e-20


def supported_soft_names(names):
    """The subset of `names` that `soft_descriptors` can compute, in order."""
    return tuple(n for n in names if n in SUPPORTED_SOFT_NAMES)


def cell_center_grid(resolution, bound=1.0, device='cpu', dtype=torch.float32):
    """Cell-centre quadrature grid over [-bound, bound]^3.

    Returns (points [resolution^3, 3] in ij order, h) where h is the cell side.
    """
    resolution = int(resolution)
    if resolution < 2:
        raise ValueError(f'resolution must be >= 2, got {resolution}')
    h = 2.0 * float(bound) / resolution
    xs = -float(bound) + h * (torch.arange(resolution, device=device, dtype=dtype) + 0.5)
    grid = torch.stack(torch.meshgrid(xs, xs, xs, indexing='ij'), dim=-1).reshape(-1, 3)
    return grid, h


def soft_occupancy(sdf, tau):
    """sigmoid(-sdf / tau): ~1 inside (sdf < 0), ~0 outside."""
    tau = float(tau)
    if tau <= 0:
        raise ValueError(f'tau must be > 0, got {tau}')
    return torch.sigmoid(-sdf / tau)


def decode_sdf_cells(vae, z_flat, resolution=48, bound=1.0, chunk=32768):
    """Evaluate `vae.decode_flat` on the cell-centre grid, keeping the graph.

    Returns (sdf [B, R, R, R], h). Unlike `mesh_extraction.decode_sdf_grid`
    this is NOT under no_grad and is batched over the latent rows.
    """
    z_flat = torch.as_tensor(z_flat)
    if z_flat.dim() == 1:
        z_flat = z_flat.unsqueeze(0)
    batch = z_flat.shape[0]
    grid, h = cell_center_grid(resolution, bound, device=z_flat.device, dtype=torch.float32)
    chunk = max(1, int(chunk))
    pieces = []
    for i in range(0, grid.shape[0], chunk):
        pts = grid[i:i + chunk].unsqueeze(0).expand(batch, -1, -1)
        pieces.append(vae.decode_flat(z_flat, pts).float())
    sdf = torch.cat(pieces, dim=1)
    return sdf.reshape(batch, resolution, resolution, resolution), h


def soft_descriptors_from_sdf(sdf, h, names=SUPPORTED_SOFT_NAMES, tau=0.032):
    """Soft volume/area from an SDF grid [B, R, R, R] with cell side h."""
    names = tuple(names)
    unknown = [n for n in names if n not in SUPPORTED_SOFT_NAMES]
    if unknown:
        raise ValueError(f'no soft proxy for {unknown}; supported: {SUPPORTED_SOFT_NAMES}')
    occ = soft_occupancy(sdf, tau)
    cell = float(h) ** 3
    out = {}
    if 'volume' in names:
        out['volume'] = occ.sum(dim=(1, 2, 3)) * cell
    if 'area' in names:
        gx, gy, gz = torch.gradient(occ, spacing=float(h), dim=(1, 2, 3))
        mag = (gx * gx + gy * gy + gz * gz).clamp_min(_AREA_EPS).sqrt()
        out['area'] = mag.sum(dim=(1, 2, 3)) * cell
    return {n: out[n] for n in names}


def soft_descriptors(vae, z_flat, names=('volume', 'area'), resolution=48, tau=0.032,
                     bound=1.0, chunk=32768):
    """Differentiable soft descriptors of the shapes decoded from `z_flat`.

    Args:
        vae: anything with ``decode_flat(z_flat [B, D], points [B, N, 3]) -> sdf [B, N]``
            (an `SDFVAE`, or `MockDecoder`). `z_flat` is the VAE-space latent,
            i.e. what `decode_flat` expects -- de-normalize an FM latent first.
        z_flat: [B, D] (or [D]) latent; gradients flow back into it.
        names: subset of `SUPPORTED_SOFT_NAMES`.
        resolution, bound: cell-centre grid (see module doc).
        tau: sigmoid temperature in SDF units.
        chunk: decoder query points per call.

    Returns dict name -> tensor [B].
    """
    sdf, h = decode_sdf_cells(vae, z_flat, resolution=resolution, bound=bound, chunk=chunk)
    return soft_descriptors_from_sdf(sdf, h, names=names, tau=tau)


# ---------------------------------------------------------------------------
# Analytic SDFs and a mock decoder for tests
# ---------------------------------------------------------------------------

def sphere_sdf(points, radius):
    """Exact SDF of a sphere centred at the origin. `radius` may be a scalar or
    a tensor broadcastable to points[..., 0] (e.g. [B, 1] against [B, N, 3])."""
    return points.norm(dim=-1) - radius


def box_sdf(points, half_extents):
    """Exact SDF of an axis-aligned box centred at the origin (Inigo Quilez form)."""
    half = torch.as_tensor(half_extents, dtype=points.dtype, device=points.device)
    q = points.abs() - half
    outside = q.clamp_min(0.0).norm(dim=-1)
    inside = q.max(dim=-1).values.clamp_max(0.0)
    return outside + inside


def sphere_volume(radius):
    return 4.0 / 3.0 * math.pi * float(radius) ** 3


def sphere_area(radius):
    return 4.0 * math.pi * float(radius) ** 2


def box_volume(half_extents):
    hx, hy, hz = (float(v) for v in half_extents)
    return 8.0 * hx * hy * hz


def box_area(half_extents):
    hx, hy, hz = (float(v) for v in half_extents)
    return 8.0 * (hx * hy + hy * hz + hz * hx)


class MockDecoder(nn.Module):
    """Analytic stand-in for the VAE: the latent's first coordinate is a sphere radius.

    ``decode_flat(z_flat [B, D], points [B or 1, N, 3]) -> sdf [B, N]`` with
    ``sdf = ||p|| - z_flat[:, 0]``; the remaining latent coordinates are
    ignored, so the soft-descriptor Jacobian is exactly zero outside column 0
    and the tests can reason about the Newton step in closed form.

    `latent_tokens` / `latent_dim` / `latent_flat_dim` mirror `SDFVAE` so code
    that reads the latent geometry off the VAE works unchanged.
    """

    def __init__(self, latent_flat_dim=8):
        super().__init__()
        self.latent_tokens = 1
        self.latent_dim = int(latent_flat_dim)
        self.latent_flat_dim = int(latent_flat_dim)
        # A parameter so `.parameters()` is non-empty (refinement code toggles
        # requires_grad on it); it does not influence the SDF.
        self.unused = nn.Parameter(torch.zeros(1))

    def decode_flat(self, z_flat, query_points):
        z_flat = torch.as_tensor(z_flat)
        if z_flat.dim() == 1:
            z_flat = z_flat.unsqueeze(0)
        radius = z_flat[:, 0].to(query_points.dtype)  # [B]
        pts = query_points
        if pts.dim() == 2:
            pts = pts.unsqueeze(0)
        if pts.shape[0] == 1 and z_flat.shape[0] > 1:
            pts = pts.expand(z_flat.shape[0], -1, -1)
        return sphere_sdf(pts, radius[:, None])

    def decode(self, z_tokens, query_points):
        return self.decode_flat(z_tokens.flatten(1), query_points)

    def forward(self, z_flat, query_points):
        return self.decode_flat(z_flat, query_points)
