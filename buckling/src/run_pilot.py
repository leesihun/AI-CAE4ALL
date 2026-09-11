"""The pilot -- gates G2..G5 (docs/BENCHMARK_DESIGN.md section 6).

G2 multimodal        : >=6 branches, effective K >= 4.0, top weight <= 0.45
G3 geometry-varying  : pairwise TV >= 0.35 on >=5 of 6 corner pairs
G5 cost and failure  : non-convergence < 5%, failure uncorrelated with branch

Usage:
  python run_pilot.py --smoke                       one solve, validate the path
  python run_pilot.py --sigma 1e-3 --draws 80       full corner sweep
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh                                   # noqa: E402
from imperfection import apply_imperfection                      # noqa: E402
from ccx import write_static_inp, run_ccx, read_frd_displacements  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "work", "pilot")
CORNERS = [(110, 1.0), (110, 1.6), (230, 1.0), (230, 1.6)]
N_PANELS = 8          # weld-panel count for component A (per-family property)


def radial_field(mesh, disp):
    """Radial component of the displacement field -- the buckling signature."""
    n = mesh.normal[:, :2]
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return disp[:mesh.n_nodes, 0] * n[:, 0] + disp[:mesh.n_nodes, 1] * n[:, 1]


def one_draw(mesh, tag, sigma_hat, a_bar, nu, seed, shortening_factor=1.5,
             n_inc=60, keep_files=False, timeout=5400):
    os.makedirs(WORK, exist_ok=True)
    coords, coeffs, info = apply_imperfection(
        mesh, sigma_hat=sigma_hat, a_bar=a_bar, nu=nu,
        n_panels=N_PANELS, seed=seed)
    inp = os.path.join(WORK, tag + ".inp")
    write_static_inp(inp, mesh, coords, shortening_factor * mesh.end_shortening_cr,
                     n_inc=n_inc)
    ok, out, secs = run_ccx(inp, timeout=timeout)
    frd = os.path.join(WORK, tag + ".frd")
    disp = read_frd_displacements(frd) if ok else None
    w = radial_field(mesh, disp) if disp is not None else None
    if not keep_files:
        for ext in (".inp", ".frd", ".dat", ".sta", ".cvg", ".12d"):
            f = os.path.join(WORK, tag + ext)
            if os.path.exists(f) and ext != ".sta":
                try:
                    os.remove(f)
                except OSError:
                    pass
    return dict(ok=ok, seconds=secs, w=w, info=info, coeffs=coeffs, tail=out[-800:])


def smoke():
    """One nonlinear solve on the smallest corner. Validates the whole path."""
    m = ShellMesh(110, 1.0)
    print("mesh: %d nodes, %d elems, Z=%.0f, d_cr=%.4e"
          % (m.n_nodes, len(m.elements), m.batdorf_Z, m.end_shortening_cr))
    for sig in (1e-2, 1e-3):
        t0 = time.time()
        r = one_draw(m, "smoke_s%g" % sig, sigma_hat=sig, a_bar=0.70, nu=1.5,
                     seed=1, n_inc=40, keep_files=True)
        w = r["w"]
        line = ("sigma_hat=%-6g ok=%-5s %6.1fs  rmsB/t=%.3e" %
                (sig, r["ok"], r["seconds"], r["info"]["rms_B_over_t"]))
        if w is not None:
            line += ("  |w|max/t=%.3f  radial rms/t=%.4f"
                     % (np.max(np.abs(w)) / m.t, np.sqrt(np.mean(w ** 2)) / m.t))
        print(line)
        if not r["ok"]:
            print("---- tail ----\n" + r["tail"])
        print("   wall %.1fs" % (time.time() - t0))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--a-bar", type=float, default=0.70)
    ap.add_argument("--nu", type=float, default=1.5)
    ap.add_argument("--draws", type=int, default=80)
    ap.add_argument("--n-inc", type=int, default=60)
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    raise SystemExit("full pilot: use run_pilot_parallel.py once smoke passes")


if __name__ == "__main__":
    sys.exit(main())
