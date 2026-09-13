"""Where, circumferentially, did the shell buckle?

Component A's job is to break the SO(2) symmetry so the buckle LOCATION is
predictable from geometry, while leaving the mode NUMBER to the withheld field.
Both halves need a measurement, and the location half is the one that was never
instrumented -- Component A's amplitude was tuned against mode counts alone.

The naive metric fails: averaging the complex circumferential FFT over axial
stations cancels, because the axial mode shape changes sign along z. Take the
phase at the axial station with the largest response instead.
"""
from __future__ import annotations

import numpy as np

from analyse import to_grid, mode_label


def buckle_phase(mesh, w, n=None):
    """Return (n, phase, angle, concentration).

    `phase`   angle of the dominant circumferential harmonic, radians
    `angle`   physical buckle angle in [0, 2*pi/n), i.e. phase folded by n
    `conc`    |mean resultant| of the per-row phases, 0..1; low means the
              pattern does not hold a single circumferential phase and the
              location is not well defined for this draw.
    """
    if n is None:
        n = mode_label(mesh, w)["n"]
    g = to_grid(mesh, w)
    rows, coefs = [], []
    for j in range(0, g.shape[1], 2):
        col = g[:, j]
        if np.isnan(col).any():
            continue
        c = np.fft.rfft(col - col.mean())
        if n < len(c):
            rows.append(j)
            coefs.append(c[n])
    if not coefs:
        return n, float("nan"), float("nan"), 0.0
    coefs = np.asarray(coefs)
    k = int(np.argmax(np.abs(coefs)))
    phase = float(np.angle(coefs[k]))

    # concentration over rows, after removing the axial sign flip: square the
    # unit phasors so +pi and 0 coincide, then halve the resultant angle.
    u = coefs / np.maximum(np.abs(coefs), 1e-30)
    conc = float(np.abs(np.mean(u ** 2)))
    angle = (phase / n) % (2.0 * np.pi / n)
    return n, phase, angle, conc


def rayleigh_p(R, n):
    """P(resultant >= R) for n i.i.d. uniform angles -- the null this must beat.

    Without this, a small sample looks pinned when it is not: 8 uniform draws
    give R = 0.47 about 17% of the time, which is exactly the value the
    inert Component A produced.
    """
    import math
    K = n * R * R
    return math.exp(-K) * (1.0 + (2.0 * K - K * K) / (4.0 * n))


def circ_stats(angles, period):
    """Circular mean and resultant length of angles living on [0, period)."""
    a = np.asarray(angles, dtype=float) * (2.0 * np.pi / period)
    z = np.exp(1j * a).mean()
    Rbar = float(np.abs(z))
    mean = float((np.angle(z) % (2.0 * np.pi)) * period / (2.0 * np.pi))
    # circular standard deviation, in the same units as `angles`
    std = float(np.sqrt(max(-2.0 * np.log(max(Rbar, 1e-12)), 0.0))
                * period / (2.0 * np.pi))
    return dict(mean=mean, R=Rbar, std=std)


def location_pinned(mesh, ws, n_expected=None):
    """Across draws: is the buckle in the SAME place every time?

    R near 1 means pinned (Component A is doing its job); R near 0 means the
    location is uniform on the circle, which is what a shell with no
    deterministic signature does.
    """
    rec = []
    for w in ws:
        n, ph, ang, conc = buckle_phase(mesh, w, n_expected)
        rec.append((n, ang, conc))
    out = {}
    for n in sorted({r[0] for r in rec}):
        angs = [r[1] for r in rec if r[0] == n]
        if len(angs) < 3:
            continue
        st = circ_stats(angs, 2.0 * np.pi / n)
        st["n"] = n
        st["draws"] = len(angs)
        st["conc"] = float(np.mean([r[2] for r in rec if r[0] == n]))
        out[n] = st
    return out
