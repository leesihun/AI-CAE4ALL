"""Proper scoring rules for the buckling distributional benchmark.

What is being scored is a DISTRIBUTION over fields at a geometry, not a field.
Three properties make the naive choices wrong here:

  1. The supplied Fourier-magnitude descriptor removes circumferential phase.
     It scores a descriptor distribution, not the complete field law. Fixed
     non-axisymmetric Component A does not imply exact SO(2) symmetry, and
     proper scores on raw ensembles remain valid if phase is part of the task.

  2. Ensembles are small. The plain energy-score and CRPS estimators are biased
     upward at small m, and the bias depends on m -- so a model that emits 8
     draws would be compared unfairly against one that emits 64. The FAIR
     estimators (Ferro 2014) divide the self-term by m(m-1) rather than m^2,
     which removes that dependence.

  3. A non-degenerate true distribution has positive expected energy score
     even with infinite samples. A reference half-split estimates that intrinsic
     baseline with sampling noise; it is not a hard lower bound. Finite-sample
     fair scores may fall below this estimated baseline (but remain nonnegative).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------- descriptors
def descriptor_matrix(mesh, ws, n_keep=40):
    """Stack SO(2)-invariant descriptors, one row per draw."""
    from gates import so2_invariant
    X = np.stack([so2_invariant(mesh, w, n_keep=n_keep) for w in ws])
    return X.astype(np.float64)


# --------------------------------------------------------------- energy score
def energy_score(fc, obs, fair=True):
    """Energy score of ensemble `fc` (m, d) against one observation `obs` (d,).

    ES = mean||x_i - y|| - 0.5 * mean||x_i - x_j||
    With fair=True the second term uses the unbiased m(m-1) normalisation.
    Lower is better; the population score is strictly proper on this vector
    space. Fairness assumes independent, identically distributed ensemble draws.
    A singleton is interpreted as a deterministic point forecast.
    """
    fc = np.atleast_2d(np.asarray(fc, dtype=float))
    obs = np.asarray(obs, dtype=float).ravel()
    m = fc.shape[0]
    if fc.ndim != 2 or m == 0 or fc.shape[1] != obs.size or not np.isfinite(fc).all() or not np.isfinite(obs).all():
        raise ValueError("fc must be a nonempty finite (m,d) array matching obs")
    t1 = float(np.mean(np.linalg.norm(fc - obs[None, :], axis=1)))
    if m < 2:
        return t1
    D = np.linalg.norm(fc[:, None, :] - fc[None, :, :], axis=2)
    denom = m * (m - 1) if fair else m * m
    t2 = float(D.sum() / denom)
    return t1 - 0.5 * t2


def energy_score_vs_sample(fc, obs_sample, fair=True):
    """Mean energy score over every member of a reference sample."""
    obs_sample = np.atleast_2d(obs_sample)
    if obs_sample.ndim != 2 or not len(obs_sample):
        raise ValueError("reference sample must be a nonempty (n,d) array")
    return float(np.mean([energy_score(fc, y, fair=fair) for y in obs_sample]))


# ---------------------------------------------------------------------- CRPS
def crps(fc, y, fair=True):
    """Fair CRPS of a 1-D ensemble against a scalar. Same estimator family."""
    fc = np.asarray(fc, dtype=float).ravel()
    m = fc.size
    if m == 0 or not np.isfinite(fc).all() or not np.isfinite(y):
        raise ValueError("CRPS requires a nonempty finite ensemble and finite observation")
    t1 = float(np.mean(np.abs(fc - y)))
    if m < 2:
        return t1
    d = float(np.abs(fc[:, None] - fc[None, :]).sum())
    denom = m * (m - 1) if fair else m * m
    return t1 - 0.5 * d / denom


def crps_vs_sample(fc, ys, fair=True):
    ys = np.ravel(ys)
    if not ys.size:
        raise ValueError("reference sample must not be empty")
    return float(np.mean([crps(fc, y, fair=fair) for y in ys]))


# ------------------------------------------------------------- the self floor
def self_score(sample, fair=True, splits=20, rng=None):
    """Estimate the intrinsic expected-score baseline by reference half-splits.

    This is a noisy estimate, not an unbeatable floor; scores can fall below it.
    """
    X = np.atleast_2d(sample)
    m = X.shape[0]
    if splits < 1:
        raise ValueError("splits must be positive")
    if m < 4:
        return float("nan")
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for _ in range(splits):
        idx = rng.permutation(m)
        a, b = X[idx[: m // 2]], X[idx[m // 2:]]
        out.append(energy_score_vs_sample(a, b, fair=fair))
    return float(np.mean(out))


def skill(score, floor, ref):
    """Relative skill: 1 = at estimated baseline, 0 = at ref; values can exceed 1."""
    if not np.isfinite(floor) or not np.isfinite(ref) or abs(ref - floor) < 1e-12:
        return float("nan")
    return float((ref - score) / (ref - floor))


# ------------------------------------------------------------ spread / calib
def spread_skill(fc, obs_sample):
    """Ensemble spread over RMSE, a diagnostic mixing width and mean error.

    Even for calibrated iid ensembles the finite-ensemble mean adds sampling
    error to RMSE; the squared-moment reference ratio is sqrt(m/(m+1)), not 1.

    Both SAOI ensembles this project has measured came out ~2x too narrow, so
    this is reported alongside every energy score rather than instead of it:
    ES alone cannot tell under-dispersion from bias.
    """
    fc = np.atleast_2d(fc)
    obs = np.atleast_2d(obs_sample)
    if fc.ndim != 2 or obs.ndim != 2 or not len(fc) or not len(obs) or fc.shape[1] != obs.shape[1] or not np.isfinite(fc).all() or not np.isfinite(obs).all():
        raise ValueError("ensembles must be nonempty finite arrays with equal feature counts")
    mu = fc.mean(axis=0)
    spread = (float(np.sqrt(np.mean(np.var(fc, axis=0, ddof=1))))
              if fc.shape[0] > 1 else 0.0)
    rmse = float(np.sqrt(np.mean((obs - mu[None, :]) ** 2)))
    return dict(spread=spread, rmse=rmse,
                iid_moment_reference=float(np.sqrt(len(fc) / (len(fc) + 1))),
                ratio=float(spread / rmse) if rmse > 1e-12 else float("nan"))


def rank_histogram(fc, obs_sample, bins=None, projection=None, rng=None):
    """Marginal ranks on an independently fixed projection (default: coordinate 0).

    A PC estimated from forecasts alone breaks exchangeability. If a custom
    projection is supplied, fit it independently. Randomize ties uniformly.
    Flat marginal ranks do not establish multivariate or conditional calibration.
    """
    fc = np.atleast_2d(fc)
    obs = np.atleast_2d(obs_sample)
    if fc.ndim != 2 or obs.ndim != 2 or not len(fc) or not len(obs) or fc.shape[1] != obs.shape[1] or not np.isfinite(fc).all() or not np.isfinite(obs).all():
        raise ValueError("ensembles must be nonempty finite arrays with equal feature counts")
    v = np.eye(1, fc.shape[1], 0).ravel() if projection is None else np.asarray(projection, dtype=float)
    if v.shape != (fc.shape[1],) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError("projection must be a nonzero finite vector with one entry per feature")
    rng = np.random.default_rng(0) if rng is None else rng
    f = fc @ v
    ranks = [int(rng.integers(np.count_nonzero(f < y), np.count_nonzero(f <= y) + 1)) for y in obs @ v]
    m = fc.shape[0]
    h = np.bincount(ranks, minlength=m + 1).astype(float)
    if bins is not None:
        if not isinstance(bins, (int, np.integer)) or not 1 <= bins <= m + 1:
            raise ValueError("bins must be an integer in [1, ensemble size + 1]")
        h = np.array([part.sum() for part in np.array_split(h, bins)])
    return h / max(h.sum(), 1.0)


# ------------------------------------------------------- discrete mode target
def mode_distribution(ns, nmax=40):
    h = np.zeros(nmax + 1)
    for n in ns:
        if 0 <= n <= nmax:
            h[int(n)] += 1
    return h / max(h.sum(), 1.0)


def tv_distance(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def js_divergence(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / np.maximum(b[mask], 1e-300))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def evaluate(mesh, pred_ws, true_ws, pred_ns=None, true_ns=None, n_keep=40):
    """Full report card for one geometry.

    `pred_ws` model draws, `true_ws` reference draws -- radial fields in units
    of t on the SAME mesh. Returns descriptor scores and a reference half-split
    baseline (legacy key self_floor). mean_only_ref uses the observed mean and
    is an in-sample diagnostic, not an independently fitted forecast baseline.
    """
    P = descriptor_matrix(mesh, pred_ws, n_keep)
    T = descriptor_matrix(mesh, true_ws, n_keep)
    es = energy_score_vs_sample(P, T)
    floor = self_score(T)
    # reference: predict the reference MEAN every time (a zero-spread model)
    ref = energy_score_vs_sample(T.mean(axis=0, keepdims=True), T)
    out = dict(energy_score=es, self_floor=floor, mean_only_ref=ref,
               skill=skill(es, floor, ref),
               n_pred=len(pred_ws), n_true=len(true_ws))
    out.update({"spread_" + k: v for k, v in spread_skill(P, T).items()})
    if pred_ns is not None and true_ns is not None:
        p, q = mode_distribution(pred_ns), mode_distribution(true_ns)
        out["mode_tv"] = tv_distance(p, q)
        out["mode_js"] = js_divergence(p, q)
    return out
