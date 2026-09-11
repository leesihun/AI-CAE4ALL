"""Extract the buckling signature from a terminal displacement field.

The label we care about is the circumferential wavenumber n of the post-buckled
pattern, plus its SO(2) phase. Phase must be quotiented out before any
clustering: on a nominally axisymmetric shell the phase is uniform by symmetry,
so a field-level metric that keeps it measures phase mismatch, not physics.
"""
from __future__ import annotations

import numpy as np


def radial(mesh, disp):
    n = mesh.normal[:, :2]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return disp[:mesh.n_nodes, 0] * n[:, 0] + disp[:mesh.n_nodes, 1] * n[:, 1]


def to_grid(mesh, w):
    """Scatter nodal values onto the (I,J) lattice; element-centre slots are NaN."""
    NI, NJ = 2 * mesh.n_circ, 2 * mesh.n_axial + 1
    g = np.full((NI, NJ), np.nan)
    ii, jj = np.nonzero(mesh.node_index >= 0)
    g[ii, jj] = w[mesh.node_index[ii, jj]]
    return g


def circumferential_spectrum(mesh, w, use_even_rows=True):
    """Mean |FFT| over axial stations, on the circumferential index.

    Only even lattice rows are complete (odd rows have the serendipity holes),
    so restrict to those to get a uniformly sampled ring.
    """
    g = to_grid(mesh, w)
    NI, NJ = g.shape
    rows = range(0, NJ, 2) if use_even_rows else range(NJ)
    acc, cnt = None, 0
    for j in rows:
        col = g[:, j]
        if np.isnan(col).any():
            continue
        s = np.abs(np.fft.rfft(col - col.mean()))
        acc = s if acc is None else acc + s
        cnt += 1
    return (acc / max(cnt, 1)), cnt


def mode_label(mesh, w, nmax=None):
    """Dominant circumferential wavenumber and its share of the spectrum."""
    sp, _ = circumferential_spectrum(mesh, w)
    sp = sp[1:]                                   # drop the mean
    if nmax:
        sp = sp[:nmax]
    tot = sp.sum()
    order = np.argsort(sp)[::-1]
    n1 = int(order[0]) + 1
    return dict(n=n1,
                share=float(sp[order[0]] / tot) if tot > 0 else 0.0,
                top5=[int(k) + 1 for k in order[:5]],
                top5_w=[float(sp[k] / tot) for k in order[:5]] if tot > 0 else [])


def expected_n(mesh):
    """Classical critical circumferential wavenumber: half the number of
    buckling half-wavelengths that fit around the circumference."""
    return 0.5 * 2.0 * np.pi * mesh.R / mesh.lambda_half


def summarise(mesh, disp, label=""):
    w = radial(mesh, disp)
    ml = mode_label(mesh, w)
    return dict(label=label,
                rms_over_t=float(np.sqrt(np.mean(w ** 2)) / mesh.t),
                max_over_t=float(np.abs(w).max() / mesh.t),
                n_dominant=ml["n"], n_share=round(ml["share"], 3),
                top5=ml["top5"], top5_w=[round(x, 3) for x in ml["top5_w"]],
                n_expected=round(expected_n(mesh), 1))
