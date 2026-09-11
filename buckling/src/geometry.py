"""Analytic mapped shell meshes, generated in float64.

Deliberately NOT gmsh: gmsh writes text meshes at limited precision, and a
6-significant-digit coordinate file injects ~3e-4*t of symmetry-breaking noise,
which is above the physical imperfection signal (see docs/BENCHMARK_DESIGN.md
section 5). Everything here is closed-form in float64 and written at %.15e.

Element type is S8R (8-node serendipity shell). The node lattice is
(2*n_circ) x (2*n_axial + 1) with element-centre nodes omitted; the
circumferential index wraps.
"""
from __future__ import annotations

import numpy as np

NU = 0.3
E_MOD = 1.0          # non-dimensional; all results scale
R_NOM = 1.0          # reference radius


def mesh_counts(r_over_t: float, l_over_r: float, elems_per_half_wave: float = 4.0):
    """Circumferential and axial element counts from the buckling half-wavelength.

    lambda_half = 1.728*sqrt(R t) at nu=0.3, and we want `elems_per_half_wave`
    elements across it in both directions.
    """
    srt = np.sqrt(r_over_t)
    n_circ = int(round(elems_per_half_wave * 2.0 * np.pi / 1.728 * srt / 4.0 * 4.0))
    # 2*pi*R / lambda_half = 2*pi/1.728 * sqrt(R/t) = 3.6357*sqrt(R/t)
    n_circ = int(round(elems_per_half_wave * 3.6357 * srt))
    n_axial = int(round(elems_per_half_wave * 0.5787 * l_over_r * srt))
    n_circ = max(n_circ, 24)
    n_axial = max(n_axial, 6)
    return n_circ, n_axial


def avoid_resonant_ncirc(n_circ: int, n_crit_lo: int = 6, n_crit_hi: int = 26) -> int:
    """A structured mesh keeps a discrete C_N symmetry and is not neutral among
    modes. Nudge n_circ so it is not a small multiple of any plausible critical
    circumferential wavenumber."""
    def bad(n):
        for k in range(n_crit_lo, n_crit_hi + 1):
            if n % k == 0:
                return True
        return False
    n = n_circ
    for delta in range(0, 40):
        for cand in (n + delta, n - delta):
            if cand >= 24 and not bad(cand):
                return cand
    return n_circ


class ShellMesh:
    """Structured S8R mesh on a surface of revolution.

    shape='cylinder' : radius R constant
    shape='cone'     : radius varies linearly, semi-vertex angle alpha (deg)
    thickness_fn     : optional callable z_norm -> thickness multiplier
    """

    def __init__(self, r_over_t, l_over_r, shape="cylinder", alpha_deg=0.0,
                 thickness_gamma=0.0, elems_per_half_wave=4.0, avoid_resonance=True):
        self.r_over_t = float(r_over_t)
        self.l_over_r = float(l_over_r)
        self.shape = shape
        self.alpha_deg = float(alpha_deg)
        self.thickness_gamma = float(thickness_gamma)

        self.R = R_NOM
        self.t = self.R / self.r_over_t
        self.L = self.l_over_r * self.R

        nc, na = mesh_counts(self.r_over_t, self.l_over_r, elems_per_half_wave)
        if avoid_resonance:
            nc = avoid_resonant_ncirc(nc)
        self.n_circ, self.n_axial = nc, na

        self._build()

    # ---------------------------------------------------------------- geometry
    def radius_at(self, z):
        if self.shape == "cone":
            # R at z=0 is self.R; radius grows with the semi-vertex angle
            return self.R + z * np.tan(np.deg2rad(self.alpha_deg))
        return np.full_like(np.asarray(z, dtype=float), self.R)

    def thickness_at(self, z):
        """Axial thickness variation family (structural-OOD tier T3)."""
        if self.thickness_gamma == 0.0:
            return np.full_like(np.asarray(z, dtype=float), self.t)
        return self.t * (1.0 + self.thickness_gamma * np.sin(np.pi * z / self.L))

    def _build(self):
        NI = 2 * self.n_circ          # circumferential lattice, wraps
        NJ = 2 * self.n_axial + 1     # axial lattice

        # serendipity: drop nodes where both indices are odd (element centres)
        keep = np.ones((NI, NJ), dtype=bool)
        keep[1::2, 1::2] = False
        idx = -np.ones((NI, NJ), dtype=np.int64)
        ii, jj = np.nonzero(keep)
        idx[ii, jj] = np.arange(ii.size, dtype=np.int64)
        self.node_index = idx

        theta = 2.0 * np.pi * ii / NI                       # exact in float64
        zeta = jj / (NJ - 1)
        z = zeta * self.L
        r = self.radius_at(z)

        self.theta = theta
        self.z = z
        self.r0 = r
        self.n_nodes = ii.size

        # outward normal of a surface of revolution
        if self.shape == "cone":
            ca = np.cos(np.deg2rad(self.alpha_deg))
            sa = np.sin(np.deg2rad(self.alpha_deg))
            nr, nz = ca, -sa
        else:
            nr, nz = 1.0, 0.0
        self.normal = np.stack([nr * np.cos(theta), nr * np.sin(theta),
                                np.full_like(theta, nz)], axis=1)

        self.coords = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)

        # ---- elements (S8R node order: 4 corners CCW then 4 midsides)
        els = []
        for i in range(self.n_circ):
            for j in range(self.n_axial):
                a, b = (2 * i) % NI, (2 * i + 2) % NI
                m = (2 * i + 1) % NI
                c, d, e = 2 * j, 2 * j + 1, 2 * j + 2
                els.append([idx[a, c], idx[b, c], idx[b, e], idx[a, e],
                            idx[m, c], idx[b, d], idx[m, e], idx[a, d]])
        self.elements = np.asarray(els, dtype=np.int64)
        assert (self.elements >= 0).all(), "serendipity indexing produced a hole"

        self.bottom_nodes = np.nonzero(jj == 0)[0]
        self.top_nodes = np.nonzero(jj == NJ - 1)[0]

    # ---------------------------------------------------------------- physics
    @property
    def lambda_half(self):
        return 1.728 * np.sqrt(self.R * self.t)

    @property
    def batdorf_Z(self):
        return (self.L ** 2) / (self.R * self.t) * np.sqrt(1.0 - NU ** 2)

    @property
    def sigma_cr_classical(self):
        return E_MOD * self.t / (self.R * np.sqrt(3.0 * (1.0 - NU ** 2)))

    @property
    def end_shortening_cr(self):
        return self.sigma_cr_classical / E_MOD * self.L

    def summary(self):
        return dict(shape=self.shape, r_over_t=self.r_over_t, l_over_r=self.l_over_r,
                    alpha_deg=self.alpha_deg, thickness_gamma=self.thickness_gamma,
                    n_circ=self.n_circ, n_axial=self.n_axial,
                    n_nodes=int(self.n_nodes), n_elements=int(len(self.elements)),
                    t=self.t, L=self.L, Z=self.batdorf_Z,
                    lambda_half=self.lambda_half,
                    elems_per_half_wave=float(2 * np.pi * self.R / self.n_circ
                                              and self.lambda_half /
                                              (2 * np.pi * self.R / self.n_circ)),
                    d_cr=self.end_shortening_cr)
