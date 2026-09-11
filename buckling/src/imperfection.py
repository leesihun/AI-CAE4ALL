"""The imperfection law.

Two components (docs/BENCHMARK_DESIGN.md section 2):

  A -- deterministic process signature, per family, geometry-derived.
       Ovalization (l=2,3) plus an N_p-panel weld pattern. Total RMS 0.30*t.
       Does NOT vary across draws. This is what breaks the SO(2) symmetry, so
       the buckle location is non-uniform AND predictable from geometry.

  B -- band-limited stochastic residual, Matern kernel, the mode selector.
       This is the withheld variable.

Both are stored as SPECTRAL COEFFICIENTS, not nodal values. That is what makes
a realization re-samplable onto a different mesh: the field is evaluated in
closed form at whatever node coordinates you pass, so it is exactly
mesh-independent rather than approximately so.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma as gammafn


# ------------------------------------------------------------------ component B
def matern_psd(kappa_sq, sigma, ell, nu):
    """2-D Matern spectral density, normalised so that integral S d^2kappa = sigma^2."""
    pref = (sigma ** 2) * gammafn(nu + 1.0) * (2.0 * nu) ** nu / (
        gammafn(nu) * np.pi * ell ** (2.0 * nu))
    return pref * (2.0 * nu / ell ** 2 + kappa_sq) ** (-(nu + 1.0))


class SpectralImperfection:
    """Band-limited Gaussian random field on the unrolled mid-surface.

    Circumferential coordinate s = R*theta is periodic with period 2*pi*R, so the
    circumferential wavenumbers are the integers p (kappa_1 = p/R). The axial
    direction is embedded in a padded length L_p = 2L and cropped (circulant
    embedding) so no spurious axial periodicity is imposed.
    """

    def __init__(self, R, L, t, sigma_hat, a_bar, nu=1.5, kappa_cut_factor=8.0,
                 rng=None):
        self.R, self.L, self.t = float(R), float(L), float(t)
        self.sigma = float(sigma_hat) * self.t          # physical RMS
        self.ell = float(a_bar) * np.sqrt(self.R * self.t)
        self.nu = float(nu)
        self.Lp = 2.0 * self.L

        # truncate at kappa_cut = kappa_cut_factor / ell (well past the
        # correlation scale); the mesh rule h <= ell/4 resolves this.
        kcut = kappa_cut_factor / self.ell
        self.P = int(np.ceil(kcut * self.R))                 # |p| <= P
        self.Q = int(np.ceil(kcut * self.Lp / (2.0 * np.pi)))  # |q| <= Q

        rng = np.random.default_rng() if rng is None else rng
        self._draw(rng)

    def _draw(self, rng):
        p = np.arange(-self.P, self.P + 1)
        q = np.arange(-self.Q, self.Q + 1)
        k1 = p / self.R
        k2 = 2.0 * np.pi * q / self.Lp
        dk1 = 1.0 / self.R
        dk2 = 2.0 * np.pi / self.Lp

        K1, K2 = np.meshgrid(k1, k2, indexing="ij")
        S = matern_psd(K1 ** 2 + K2 ** 2, self.sigma, self.ell, self.nu)
        amp = np.sqrt(2.0 * S * dk1 * dk2)

        phi = rng.uniform(0.0, 2.0 * np.pi, size=amp.shape)
        # Hermitian symmetry -> real field
        self.k1, self.k2 = k1, k2
        self.C = amp * np.exp(1j * phi)          # spectral coefficients (stored)

        # exact RMS of the truncated realization, for reporting
        self.rms_nominal = float(np.sqrt(0.5 * np.sum(amp ** 2)))

    def evaluate(self, theta, z):
        """Closed-form evaluation at arbitrary node coordinates.

        xi(s,z) = Re( sum_pq C_pq exp(i k1_p s) exp(i k2_q z) ),  s = R*theta

        Exact at any point, so the SAME realization transfers to a different
        mesh or a different geometry without interpolation error.
        """
        s = self.R * np.asarray(theta, dtype=float)
        z = np.asarray(z, dtype=float)
        Es = np.exp(1j * np.outer(s, self.k1))       # (N, 2P+1)
        Ez = np.exp(1j * np.outer(z, self.k2))       # (N, 2Q+1)
        # per-node: sum_pq C_pq Es[n,p] Ez[n,q]  ->  einsum over both axes
        val = np.einsum("np,pq,nq->n", Es, self.C, Ez, optimize=True)
        return np.real(val)

    def coefficients(self):
        """The artifact that gets persisted alongside the field."""
        return dict(k1=self.k1, k2=self.k2, C_real=self.C.real, C_imag=self.C.imag,
                    sigma=self.sigma, ell=self.ell, nu=self.nu,
                    R=self.R, L=self.L, t=self.t, rms_nominal=self.rms_nominal)


# ------------------------------------------------------------------ component A
def process_signature(theta, z, L, t, n_panels, rms_over_t=0.30, seed=0):
    """Deterministic per-family signature: ovalization + weld pattern.

    One axial half-wave only (NASA SP-8007 Rev 2 section 4.3: components with
    k > 4 are typically small in cylinders without circumferential joints).
    Amplitudes decay monotonically in l. Phases are fixed per family.
    """
    ls = [2, 3] + [j * n_panels for j in (1, 2, 3, 4)]
    rng = np.random.default_rng(seed)
    psi = rng.uniform(0.0, 2.0 * np.pi, size=len(ls))

    w = np.zeros_like(np.asarray(theta, dtype=float))
    raw = np.array([1.0 / (l ** 1.5) for l in ls])       # monotone decay in l
    raw = raw / np.sqrt(np.sum(raw ** 2) * 0.5 * 0.5)     # unit RMS over the surface
    for a, l, ps in zip(raw, ls, psi):
        w += a * np.sin(np.pi * z / L) * np.cos(l * theta + ps)

    # normalise to the requested RMS
    cur = np.sqrt(np.mean(w ** 2))
    if cur > 0:
        w *= (rms_over_t * t) / cur
    return w, dict(l_list=ls, psi=psi.tolist(), rms=float(rms_over_t * t))


def apply_imperfection(mesh, sigma_hat, a_bar, nu, n_panels, seed,
                       comp_a_rms_over_t=0.30, with_component_a=True):
    """Returns (perturbed_coords, spectral_coeffs, info)."""
    rng = np.random.default_rng(seed)
    fieldB = SpectralImperfection(mesh.R, mesh.L, mesh.t, sigma_hat, a_bar, nu, rng=rng)
    wB = fieldB.evaluate(mesh.theta, mesh.z)

    if with_component_a:
        wA, infoA = process_signature(mesh.theta, mesh.z, mesh.L, mesh.t,
                                      n_panels, comp_a_rms_over_t, seed=n_panels)
    else:
        wA, infoA = np.zeros_like(wB), dict(l_list=[], psi=[], rms=0.0)

    w = wA + wB
    coords = mesh.coords + w[:, None] * mesh.normal
    info = dict(rms_A=float(np.sqrt(np.mean(wA ** 2))),
                rms_B=float(np.sqrt(np.mean(wB ** 2))),
                rms_B_over_t=float(np.sqrt(np.mean(wB ** 2)) / mesh.t),
                max_abs_over_t=float(np.max(np.abs(w)) / mesh.t),
                n_modes=int(fieldB.C.size), P=fieldB.P, Q=fieldB.Q,
                component_a=infoA)
    return coords, fieldB.coefficients(), info
