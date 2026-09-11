"""Gate G1 -- signal above noise.

On the PERFECT shell, nominally degenerate cos/sin eigenvalue pairs must split
by less than 1e-4 relative (target 1e-5). That single number is the total
symmetry-breaking floor of mesh + assembly + solver, and it must sit well below
the physical per-mode imperfection signal (~0.39% knockdown at sigma_hat=1e-4).

Also require >= 8 eigenvalues within 0.4% of lambda_1 -- the near-critical
cluster the imperfection is supposed to reorder.

Run:  python run_g1.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh                     # noqa: E402
from ccx import write_buckle_inp, run_ccx, read_buckle_eigenvalues  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work", "g1")

CORNERS = [(110, 1.0), (110, 1.6), (230, 1.0), (230, 1.6)]


def pair_splitting(ev, rel_tol=5e-3):
    """Group eigenvalues into near-degenerate clusters and report the relative
    splitting within each nominally-degenerate cos/sin pair."""
    ev = np.sort(np.asarray(ev, dtype=float))
    ev = ev[ev > 0]
    if ev.size < 2:
        return np.array([]), ev
    splits, i = [], 0
    while i < ev.size - 1:
        rel = (ev[i + 1] - ev[i]) / ev[i]
        if rel < rel_tol:
            splits.append(rel)
            i += 2
        else:
            i += 1
    return np.asarray(splits), ev


WINDOWS = [4e-3, 1.56e-2, 4.86e-2, 1.456e-1]   # knockdowns at sigma_hat 1e-4..1e-2

def cluster_count(ev, window=4e-3):
    ev = np.sort(np.asarray(ev, dtype=float))
    ev = ev[ev > 0]
    if ev.size == 0:
        return 0
    return int(np.sum(ev <= ev[0] * (1.0 + window)))


def cluster_profile(ev):
    """Modes within each candidate knockdown window. The window is set by the
    imperfection amplitude via Koiter, so this table decides sigma_hat."""
    return {("n_within_%.3g" % w): cluster_count(ev, w) for w in WINDOWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="one small geometry only, for pipeline validation")
    ap.add_argument("--neig", type=int, default=60)
    ap.add_argument("--eph", type=float, default=4.0,
                    help="elements per buckling half-wavelength")
    args = ap.parse_args()

    os.makedirs(WORK, exist_ok=True)
    corners = [(110, 1.0)] if args.quick else CORNERS
    neig = 20 if args.quick else args.neig

    results, passed = [], True
    for (rt, lr) in corners:
        m = ShellMesh(rt, lr, elems_per_half_wave=args.eph)
        tag = "g1_rt%d_lr%03d" % (rt, round(lr * 100))
        inp = os.path.join(WORK, tag + ".inp")
        write_buckle_inp(inp, m, m.coords, n_eig=neig)

        ok, out, secs = run_ccx(inp, timeout=7200)
        ev = read_buckle_eigenvalues(os.path.join(WORK, tag + ".dat"))

        splits, evs = pair_splitting(ev)
        nclust = cluster_count(ev)
        max_split = float(np.max(splits)) if splits.size else float("nan")
        med_split = float(np.median(splits)) if splits.size else float("nan")

        # Ratio of the computed critical load to the classical value: a check
        # that the mesh and BCs are behaving. Clamped ends raise it above 1.
        lam1 = float(evs[0]) if evs.size else float("nan")

        g1a = splits.size > 0 and max_split < 1e-4
        g1b = nclust >= 8
        passed = passed and ok and g1a and g1b

        r = dict(r_over_t=rt, l_over_r=lr, ok=bool(ok), seconds=round(secs, 1),
                 n_nodes=int(m.n_nodes), n_elems=int(len(m.elements)),
                 n_circ=m.n_circ, n_axial=m.n_axial, Z=round(m.batdorf_Z, 1),
                 n_eig_found=int(evs.size), lambda1=lam1,
                 n_pairs=int(splits.size), median_split=med_split,
                 max_split=max_split, n_within_0p4pct=nclust,
                 **cluster_profile(ev),
                 G1a_split_lt_1e4=bool(g1a), G1b_cluster_ge_8=bool(g1b))
        results.append(r)
        print(json.dumps(r, indent=None))
        if not ok:
            print("---- ccx tail ----\n" + out[-1200:])

    with open(os.path.join(WORK, "g1_results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("\nG1 %s" % ("PASSED" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
