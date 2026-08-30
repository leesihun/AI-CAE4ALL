"""Design vector -> SDFFlow latent -> SDF grid -> marching-cubes surface mesh.

The design vector parameterizes the flow-matching *noise* (t=0 state of the ODE),
restricted to a low-dimensional orthonormal subspace, plus optional shape-descriptor
conditions. Every point of the design space therefore maps to an on-manifold
DeepJEB-like bracket, which is what makes a black-box optimizer usable here.
"""

import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh  # noqa: E402
from model.sdf_vae import SDFVAE  # noqa: E402
from model.velocity_net import VelocityNet  # noqa: E402
from training_profiles.setup import load_checkpoint  # noqa: E402


def _model_state(ckpt):
    """Prefer EMA weights; strip the AveragedModel 'module.' prefix."""
    state = ckpt.get('ema_state') or ckpt['model_state']
    if ckpt.get('ema_state') is not None:
        state = {k.replace('module.', '', 1): v for k, v in state.items() if k != 'n_averaged'}
    return state


class SDFFlowGenerator:
    """Wraps a trained VAE + FM pair as a differentiable-free shape parameterization."""

    def __init__(self, vae_path, fm_path, device='cuda', subspace_dim=12,
                 subspace_seed=0, base_seed=0, ode_steps=50, mc_resolution=128,
                 cond_dims=('volume', 'area'), shell_scale=1.25, latent_range=None,
                 cond_range=1.5):
        self.device = torch.device(device if torch.cuda.is_available() or device == 'cpu'
                                   else 'cpu')
        self.fm_ckpt = load_checkpoint(fm_path, self.device)
        vae_path = vae_path or self.fm_ckpt['vae_modelpath']
        vae_ckpt = load_checkpoint(vae_path, self.device)
        self.vae = SDFVAE(vae_ckpt['config']).to(self.device)
        self.vae.load_state_dict(_model_state(vae_ckpt))
        self.vae.eval()

        self.latent_flat_dim = int(self.fm_ckpt['latent_flat_dim'])
        self.cond_dim = int(self.fm_ckpt['cond_dim'])
        self.cond_names = list(self.fm_ckpt['cond_names'])
        self.fm = VelocityNet(self.fm_ckpt['config'], self.latent_flat_dim,
                              cond_dim=self.cond_dim).to(self.device)
        self.fm.load_state_dict(_model_state(self.fm_ckpt))
        self.fm.eval()

        self.ode_steps = int(ode_steps)
        self.mc_resolution = int(mc_resolution)
        self.shell_scale = float(shell_scale)

        # Orthonormal search subspace of the 256-d FM noise space.
        rng = np.random.default_rng(subspace_seed)
        basis, _ = np.linalg.qr(rng.standard_normal((self.latent_flat_dim, subspace_dim)))
        self.basis = torch.tensor(basis.T.copy(), dtype=torch.float32, device=self.device)
        self.subspace_dim = int(subspace_dim)

        # Base noise draw; the search replaces only its in-subspace component.
        g = torch.Generator(device='cpu').manual_seed(int(base_seed))
        z0 = torch.randn(self.latent_flat_dim, generator=g).to(self.device)
        self.z0_perp = z0 - (z0 @ self.basis.T) @ self.basis

        # Condition axes actually searched over (normalized units, i.e. train-split sigmas).
        self.cond_dims = [c for c in cond_dims if c in self.cond_names]
        self.cond_index = [self.cond_names.index(c) for c in self.cond_dims]
        self.n_design = self.subspace_dim + len(self.cond_dims)
        # Per-coordinate bound. The default puts the box *corner* on the shell:
        # corner norm = latent_range * sqrt(d), shell radius = shell_scale * sqrt(d).
        self.latent_range = float(latent_range) if latent_range is not None             else self.shell_scale
        self.cond_range = float(cond_range)

    # ------------------------------------------------------------------ #

    def bounds(self):
        """Box bounds for the search, inscribed in the Gaussian shell by default.

        Pure function of construction-time state: `bounds()` is called from
        several places (baseline sampling, the CMA-ES setup) and must return the
        same box every time, so the range is fixed in `__init__` rather than
        defaulted here.

        With `latent_range == shell_scale` the box corner sits exactly on the
        shell. A wider box is mostly *degenerate*, not merely generous: `_noise`
        rescales anything outside the shell back onto it, keeping direction and
        discarding magnitude, so every point on a ray beyond the radius decodes
        to the identical shape.
        """
        lo = [-self.latent_range] * self.subspace_dim             + [-self.cond_range] * len(self.cond_dims)
        hi = [self.latent_range] * self.subspace_dim             + [self.cond_range] * len(self.cond_dims)
        return np.asarray(lo), np.asarray(hi)

    def _noise(self, x_latent):
        """Compose the full 256-d ODE start state, kept near the Gaussian shell."""
        v = torch.tensor(np.asarray(x_latent, dtype=np.float32), device=self.device)
        radius = float(np.sqrt(self.subspace_dim)) * self.shell_scale
        norm = torch.linalg.norm(v)
        if norm > radius:
            v = v * (radius / norm)
        return (self.z0_perp + v @ self.basis).unsqueeze(0)

    def _conditions(self, x_cond):
        if self.cond_dim == 0:
            return None
        # Unsearched condition channels stay at the train-split mean (normalized 0).
        cond = torch.zeros(1, self.cond_dim, device=self.device)
        for slot, value in zip(self.cond_index, np.asarray(x_cond, dtype=np.float32)):
            cond[0, slot] = float(value)
        return cond

    @torch.no_grad()
    def _integrate(self, z, cond):
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t = torch.full((z.shape[0],), i * dt, device=self.device)
            z = z + self.fm(z, t, cond=cond) * dt
        return z

    @torch.no_grad()
    def generate(self, x, mc_resolution=None):
        """Design vector -> (trimesh.Trimesh | None, info dict)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size != self.n_design:
            raise ValueError(f'design vector must have {self.n_design} entries, got {x.size}')
        x_latent, x_cond = x[:self.subspace_dim], x[self.subspace_dim:]

        z_n = self._integrate(self._noise(x_latent), self._conditions(x_cond))
        z = z_n * self.fm_ckpt['latent_std'].to(self.device) + \
            self.fm_ckpt['latent_mean'].to(self.device)
        volume = decode_sdf_grid(self.vae, z,
                                 resolution=mc_resolution or self.mc_resolution,
                                 device=self.device)
        mesh = sdf_grid_to_mesh(volume)
        info = {
            'latent_norm': float(torch.linalg.norm(z_n).item()),
            'requested_conditions': dict(zip(self.cond_dims, x_cond.tolist())),
            'sdf_min': float(volume.min()),
            'sdf_max': float(volume.max()),
        }
        if mesh is not None:
            info.update(watertight=bool(mesh.is_watertight),
                        faces=int(len(mesh.faces)),
                        surface_volume=float(abs(mesh.volume)) if mesh.is_watertight else None)
        return mesh, info
