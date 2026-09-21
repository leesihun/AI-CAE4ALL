"""The imperfection law -- main.tex Section 2 ("Imperfection: how, and by what
randomness, does it enter").

ONE mechanism, no separate deterministic/stochastic split: a handful of
localized Gaussian dimples, placed by Random Sequential Adsorption (RSA) on
the unrolled cylinder surface, each independently signed and log-normally
sized. Every draw of the same geometry re-samples all of it -- count,
positions, signs, magnitudes. Only the dimple width sigma = l_c(R, t) is
fixed per geometry, not per draw.

This is NOT the same law as the old buckling/ project's imperfection.py
(Matern-spectral field + deterministic ovalization/weld harmonics) -- that
law predates the paper's pivot to closely match the reference Butterfly
Defect paper's Sec. VI stochastic multi-defect mechanism (arXiv:2607.19245).
The only thing carried over is critical_wavenumber(), kept here for
reference/QA only; this law has no harmonic-exclusion step.

Storage: a draw's imperfection is the dimple list {(theta_k, z_k, a_k)} plus
the per-geometry sigma -- NOT nodal displacement values (main.tex Sec 2.3).
`evaluate()` recomputes the field in closed form at whatever node
coordinates are passed, so a realization is exactly mesh-independent.
"""
from __future__ import annotations

import numpy as np


def l_c(R, t, nu=0.3):
    """Dimple width == classical buckling half-wavelength (main.tex Eq., Sec
    2.1, citing the reference paper's Eq. 3): l_c = pi*[12(1-nu^2)]^(-1/4)*
    sqrt(R t). At nu=0.3, l_c ~= 1.73*sqrt(R t).
    """
    return np.pi * (12.0 * (1.0 - nu ** 2)) ** (-0.25) * np.sqrt(R * t)


def critical_wavenumber(r_over_t):
    """Empirical critical circumferential wavenumber n ~= 0.86*sqrt(R/t).
    Carried over from the old law for reference/QA only -- this dimple law
    does not use it (no harmonic exclusion; see module docstring).
    """
    return 0.86 * float(r_over_t) ** 0.5


def sample_K(rng):
    """K ~ round(clip(Normal(20, 7^2), 0, 40)), independent every draw
    (main.tex Sec 2.2). K=0 (a perfect cylinder) is an allowed boundary case.
    """
    raw = rng.normal(20.0, 7.0)
    return int(round(float(np.clip(raw, 0.0, 40.0))))


def sample_signed_amplitude(rng, size):
    """delta_bar_k ~ LogNormal(ln(0.142), 0.325), independent +-1 sign each
    with probability 1/2 (main.tex Sec 2.2). Returns the signed *normalized*
    amplitude delta_bar_k; multiply by t for the physical amplitude a_k.
    """
    sign = rng.choice([-1.0, 1.0], size=size)
    mag = rng.lognormal(mean=np.log(0.142), sigma=0.325, size=size)
    return sign * mag


def rsa_place(rng, K, R, L, sigma, edge_margin=None, max_tries=2000):
    """RSA placement of K dimple centers on the unrolled (theta, z) surface.

    Candidates are drawn uniformly in (theta, z) and accepted only if their
    surface (geodesic) distance to every already-placed center is >= the
    minimum separation 2*sigma (= 2*l_c, main.tex Sec 2.2). A 3*sigma-wide
    band at each end (z < edge_margin or z > L - edge_margin) is excluded so
    dimples don't mix with the clamped/loaded boundary condition itself.

    Raises RuntimeError if a center can't be placed within max_tries -- that
    means K is too large for this geometry's placeable band, a configuration
    problem to fix (smaller K, or check L/sigma), not something to retry past.
    """
    if K == 0:
        return []
    if edge_margin is None:
        edge_margin = 3.0 * sigma
    if L - 2.0 * edge_margin <= 0:
        raise ValueError(
            f"edge margins (2*{edge_margin:.4g}={2*edge_margin:.4g}) exceed L ({L:.4g})"
        )
    min_sep = 2.0 * sigma
    centers = []
    for k in range(K):
        for _try in range(max_tries):
            theta = rng.uniform(0.0, 2.0 * np.pi)
            z = rng.uniform(edge_margin, L - edge_margin)
            ok = True
            for (theta2, z2) in centers:
                dtheta = abs(theta - theta2)
                dtheta = min(dtheta, 2.0 * np.pi - dtheta) * R
                dz = z - z2
                if dtheta * dtheta + dz * dz < min_sep * min_sep:
                    ok = False
                    break
            if ok:
                centers.append((theta, z))
                break
        else:
            raise RuntimeError(
                f"RSA placement failed after {max_tries} tries for dimple "
                f"{k + 1}/{K} -- K too dense for this geometry's placeable "
                f"band (L={L:.4g}, sigma={sigma:.4g}, edge_margin={edge_margin:.4g})"
            )
    return centers


def draw_imperfection(rng, R, t, L, nu=0.3, k_override=None):
    """One full independent draw of the imperfection law for one geometry.

    k_override: if given, forces K to this exact value instead of sampling
    it via sample_K(rng) -- for controlled verification cases only (e.g. a
    single-dimple K=1 arm). Leave None for the real production stochastic
    law (default everywhere in production code paths).

    Returns (centers, amps_normalized, sigma, K):
      centers          -- list of (theta_k, z_k), len K
      amps_normalized  -- array of signed delta_bar_k (a_k = t * this)
      sigma            -- l_c(R, t, nu), fixed per geometry not per draw
      K                -- realized dimple count for this draw
    """
    sigma = l_c(R, t, nu)
    K = sample_K(rng) if k_override is None else int(k_override)
    centers = rsa_place(rng, K, R, L, sigma)
    amps_normalized = sample_signed_amplitude(rng, size=len(centers))
    return centers, amps_normalized, sigma, K


def evaluate(theta, z, R, centers, amps_physical, sigma):
    """Closed-form field w(theta, z) = sum_k a_k * exp(-d_k^2 / sigma^2),
    d_k^2 = (R*Delta_theta_k)^2 + (z - z_k)^2 (main.tex Sec 2.1), evaluated at
    arbitrary node coordinates -- exactly mesh-independent since the dimple
    list, not a nodal field, is the stored artifact (Sec 2.3).
    """
    theta = np.asarray(theta, dtype=float)
    z = np.asarray(z, dtype=float)
    w = np.zeros_like(theta)
    for (theta_k, z_k), a_k in zip(centers, amps_physical):
        dtheta = np.abs(theta - theta_k)
        dtheta = np.minimum(dtheta, 2.0 * np.pi - dtheta) * R
        dz = z - z_k
        d2 = dtheta ** 2 + dz ** 2
        w += a_k * np.exp(-d2 / sigma ** 2)
    return w


def apply_imperfection(mesh, rng, nu=0.3, k_override=None):
    """Draw + evaluate on `mesh` (a geometry.ShellMesh), pushing every node
    along its outward normal. Returns (coords, record) where `record` is the
    persistable dimple list plus per-draw summary stats.
    """
    centers, amps_normalized, sigma, K = draw_imperfection(
        rng, mesh.R, mesh.t, mesh.L, nu, k_override=k_override
    )
    amps_physical = mesh.t * amps_normalized
    w = evaluate(mesh.theta, mesh.z, mesh.R, centers, amps_physical, sigma)
    coords = mesh.coords + w[:, None] * mesh.normal
    record = dict(
        centers=np.asarray(centers, dtype=float).reshape(-1, 2),  # (K, 2): theta_k, z_k
        amps_normalized=np.asarray(amps_normalized, dtype=float),  # delta_bar_k
        amps_physical=np.asarray(amps_physical, dtype=float),      # a_k
        sigma=float(sigma),
        K=int(K),
        max_abs_over_t=float(np.max(np.abs(w)) / mesh.t) if K > 0 else 0.0,
        rms_over_t=float(np.sqrt(np.mean(w ** 2)) / mesh.t),
    )
    return coords, record
