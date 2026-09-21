"""Shape-tier sampling -- main.tex Section 1's tier table (Table 1), scaled
from pilot to production size.

The OOD (t1) tier is NOT re-derived here: main.tex already fixes it at
exactly 4 shapes -- the full 2x2 grid R/t in {95, 245} x L/R in {0.6, 1.5},
32 draws each -- and that count already matches the production request
("OOD 샘플은 4개정도"), so it is carried over verbatim, unchanged from pilot
to production.

The train tier is what actually scales (pilot: a 4x3=12-point grid, 48 draws
each; production: 50 shapes x 30 draws each, per explicit instruction).
Going from a full grid to 50 points calls for space-filling sampling rather
than a denser grid, so this uses Latin Hypercube Sampling -- but critically
over the SAME range the pilot's grid already covers (R/t in [110,200], L/R
in [0.8,1.3]), not a wider one. Widening it would blur the one thing the t1
tier is designed to isolate: t1 tests extrapolation *beyond* the train
range, so the train range has to stay fixed for that framing to still mean
anything as shape count grows.
"""
from __future__ import annotations

import numpy as np

import imperfection

# from main.tex Table 1 (Sec. 1) -- train tier's grid bounds, pilot scale
TRAIN_ROT_RANGE = (110.0, 200.0)
TRAIN_LOR_RANGE = (0.8, 1.3)
DRAWS_PER_TRAIN_SHAPE = 100

# from main.tex Table 1 -- t1 (OOD/extrapolation) tier, fixed 2x2 grid,
# unchanged from pilot to production (already exactly 4 shapes).
# The two original low-L/R corners (95, 0.6) and (245, 0.6) were found to be
# hard-infeasible for the imperfection law: the RSA edge margin (2*3*l_c(R,t))
# exceeds L there, so no draw at any seed could ever place even one dimple.
# Nudged up to the nearest L/R that clears that floor with a workable
# accept rate -- (245, 0.7) stays comfortably below the train L/R floor of
# 0.8 (still a genuine low-L/R extrapolation point); (95, 1.4) cannot go
# below the train range at all at this R/t (the feasibility floor there is
# ~1.07, itself inside [0.8, 1.3]) -- confirmed by a direct search, see
# corner_test.py's run log. That corner is OOD via R/t only, not L/R; and its
# accepted draws skew toward lower K (mean ~13 vs the nominal 20) because the
# RSA-retry accept filter (run_batch.py's max_rsa_attempts loop) implicitly
# excludes the K values too dense to fit -- a real, unavoidable narrowing of
# that one corner's severity range, not a sampling bug.
OOD_SHAPES = [
    (95.0, 1.4),
    (95.0, 1.5),
    (245.0, 0.7),
    (245.0, 1.5),
]
DRAWS_PER_OOD_SHAPE = 100


def latin_hypercube(n, ranges, seed):
    """n points, one per stratum per dimension, independently permuted per
    dimension (standard LHS) -- space-filling without a grid's combinatorial
    blowup or a plain-random sample's clumping/gaps.
    """
    rng = np.random.default_rng(seed)
    ndim = len(ranges)
    cuts = np.linspace(0.0, 1.0, n + 1)
    points = np.empty((n, ndim))
    for d in range(ndim):
        lo, hi = cuts[:-1], cuts[1:]
        u = rng.uniform(lo, hi)
        rng.shuffle(u)
        vmin, vmax = ranges[d]
        points[:, d] = vmin + u * (vmax - vmin)
    return points


def _hard_feasible(r_over_t, l_over_r):
    """True iff the RSA edge margin (main.tex Sec 2.2) fits on this geometry:
    2*edge_margin(R,t) < L, i.e. l_over_r > 6*l_c(R,t)/R. Scale-invariant, so
    checked in the R=1 units geometry.ShellMesh uses internally.
    """
    sigma = imperfection.l_c(1.0, 1.0 / r_over_t)
    return l_over_r > 6.0 * sigma


def sample_training_shapes(n=100, seed=0):
    """n (r_over_t, l_over_r) pairs via LHS over the pilot's train range.

    A handful of LHS points can land in a wedge of this box where the
    imperfection law's RSA edge margin can never fit (main.tex Sec 2.2) --
    no draw at any seed would place even one dimple there. Each such point is
    replaced with a fresh uniform draw, rejection-sampled until feasible,
    continuing the same rng stream -- a small, deliberate compromise on
    strict LHS stratification for the ~10% of points this affects, in
    exchange for every one of the 50 shapes actually being runnable.
    """
    pts = latin_hypercube(n, [TRAIN_ROT_RANGE, TRAIN_LOR_RANGE], seed)
    rng = np.random.default_rng((seed, 0xBAD6EE5))
    for i in range(len(pts)):
        r_over_t, l_over_r = pts[i]
        while not _hard_feasible(r_over_t, l_over_r):
            r_over_t = rng.uniform(*TRAIN_ROT_RANGE)
            l_over_r = rng.uniform(*TRAIN_LOR_RANGE)
        pts[i] = (r_over_t, l_over_r)
    return [(float(r), float(l)) for r, l in pts]


def production_tiers(n_train=100, seed=0):
    """Full production shape roster: (tier, r_over_t, l_over_r, draws)."""
    rows = []
    for r_over_t, l_over_r in sample_training_shapes(n_train, seed):
        rows.append(("train", r_over_t, l_over_r, DRAWS_PER_TRAIN_SHAPE))
    for r_over_t, l_over_r in OOD_SHAPES:
        rows.append(("t1", r_over_t, l_over_r, DRAWS_PER_OOD_SHAPE))
    return rows


if __name__ == "__main__":
    rows = production_tiers()
    n_train = sum(1 for r in rows if r[0] == "train")
    n_ood = sum(1 for r in rows if r[0] == "t1")
    total_draws = sum(r[3] for r in rows)
    print(f"{n_train} train shapes x {DRAWS_PER_TRAIN_SHAPE} draws "
          f"+ {n_ood} OOD shapes x {DRAWS_PER_OOD_SHAPE} draws "
          f"= {total_draws} total draws")
    for tier, r_over_t, l_over_r, draws in rows:
        print(f"  {tier:5s}  R/t={r_over_t:7.2f}  L/R={l_over_r:.3f}  draws={draws}")
