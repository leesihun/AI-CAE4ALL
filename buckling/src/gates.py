"""Pilot gates G2, G3, G5 (docs/BENCHMARK_DESIGN.md section 6).

G2 multimodal        >=6 branches, effective K >= 4.0, top-branch weight <= 0.45
G3 geometry-varying  pairwise TV >= 0.35 on >=5 of 6 corner pairs, >=0.15 on all
G5 honesty           non-convergence < 5%, failure uncorrelated with branch

SO(2) must be quotiented before clustering. On a nominally axisymmetric shell
the buckle phase is uniform by symmetry, so any representation that keeps the
phase measures phase mismatch rather than physics -- and a model would score
well by predicting the symmetry orbit and nothing else. The magnitude of the
circumferential FFT is exactly the rotation-invariant we need.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh          # noqa: E402
from analyse import to_grid             # noqa: E402


def so2_invariant(mesh, w, n_keep=40):
    """Rotation-invariant descriptor: |FFT_circumferential| per axial station."""
    g = to_grid(mesh, w)
    NI, NJ = g.shape
    feats = []
    for j in range(0, NJ, 2):                 # even rows are complete
        col = g[:, j]
        if np.isnan(col).any():
            continue
        s = np.abs(np.fft.rfft(col - col.mean()))[1:n_keep + 1]
        feats.append(s)
    return np.concatenate(feats) if feats else np.zeros(1)


def pca(X, d=32):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    d = min(d, Vt.shape[0])
    return Xc @ Vt[:d].T, S


def kmeans(X, k, iters=120, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == lab).all():
            break
        lab = new
        for c in range(k):
            if (lab == c).any():
                C[c] = X[lab == c].mean(0)
    inertia = float(((X - C[lab]) ** 2).sum())
    return lab, inertia


def silhouette(X, lab):
    """Mean silhouette; -1 if any cluster is a singleton set of all points."""
    ks = sorted(set(lab.tolist()))
    if len(ks) < 2:
        return -1.0
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    out = []
    for i in range(len(X)):
        same = (lab == lab[i])
        same[i] = False
        if not same.any():
            out.append(0.0)
            continue
        a = D[i, same].mean()
        b = min(D[i, lab == c].mean() for c in ks if c != lab[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out))


def choose_k(X, kmax=10, seed=0):
    """Select k by mean silhouette.

    The previous elbow-on-inertia rule was unreliable and gave physically
    impossible answers: at Component A = 0.30 t every one of 12 draws pinned to
    the same circumferential wavenumber, yet the rule reported 7 branches with
    effective K = 6.45. Silhouette is scored against the actual cluster
    geometry and does not reward splitting a single blob.

    Note the sample-size floor: k is capped at N/4, because branch structure
    cannot be estimated from a handful of draws however the k is chosen.
    """
    kmax = int(min(kmax, max(2, len(X) // 4)))
    best, best_s, scores = 1, -2.0, {}
    for k in range(2, kmax + 1):
        lab, _ = kmeans(X, k, seed=seed)
        sc = silhouette(X, lab)
        scores[k] = round(sc, 3)
        if sc > best_s:
            best, best_s = k, sc
    # a weak best silhouette means there is no cluster structure at all
    if best_s < 0.15:
        best = 1
    return best, scores


def effective_k(labels):
    c = np.array(list(Counter(labels).values()), dtype=float)
    p = c / c.sum()
    H = -(p * np.log(p)).sum()
    return float(np.exp(H)), p


def mode_hist(records, nmax=40):
    h = np.zeros(nmax)
    for r in records:
        n = r.get("n_dominant")
        if n and 1 <= n <= nmax:
            h[n - 1] += 1
    return h / max(h.sum(), 1)


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def g2(mesh, ws, seed=0):
    """ws: list of radial fields (already /t) for one geometry."""
    X = np.stack([so2_invariant(mesh, w) for w in ws])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)   # shape, not amplitude
    Z, _ = pca(X, min(32, len(X) - 1))         # PCA rank cannot exceed N-1
    k, sil = choose_k(Z, seed=seed)
    lab = (np.zeros(len(Z), int) if k <= 1
           else kmeans(Z, k, seed=seed)[0])
    keff, p = effective_k(lab)
    # spread: mean pairwise relative L2 on the raw fields
    W = np.stack(ws)
    rel = []
    for i in range(len(W)):
        for j in range(i + 1, len(W)):
            rel.append(np.linalg.norm(W[i] - W[j]) / np.linalg.norm(W[i]))
    return dict(n_branches=int(len(set(lab.tolist()))), effective_k=round(keff, 2),
                top_weight=round(float(p.max()), 3),
                mean_pairwise_relL2=round(float(np.mean(rel)), 3),
                labels=lab.tolist(), silhouette=sil,
                PASS=bool(len(set(lab.tolist())) >= 6 and keff >= 4.0 and p.max() <= 0.45))


def load_batch(base):
    with open(os.path.join(base, "results.json")) as f:
        recs = json.load(f)
    for r in recs:
        wf = os.path.join(base, r["key"], "w.npy")
        r["_w"] = np.load(wf) if os.path.exists(wf) else None
    return recs
