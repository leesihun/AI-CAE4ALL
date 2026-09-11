"""Parallel draw runner.

Each job gets its own directory: CalculiX writes <jobname>.* plus a shared
`spooles.out` into the cwd, so running many jobs in one directory races.

Results land in one HDF5 per batch:
    draws/<key>/w          radial displacement at the terminal state, /t
    draws/<key>/force      (step_time, |Fz|) history
    draws/<key>/coeffs_*   the imperfection spectral coefficients (the withheld
                           variable -- stored for reproducibility, never fed to
                           a model)
    plus per-draw attrs: geometry, seed, sigma_hat, mode label, timings
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh                       # noqa: E402
from imperfection import apply_imperfection          # noqa: E402
from ccx import write_twostep_inp, run_ccx, read_frd_displacements  # noqa: E402
from analyse import radial, mode_label, expected_n   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PANELS = 8


def _forces(dat_path):
    if not os.path.exists(dat_path):
        return np.zeros((0, 2))
    txt = open(dat_path, errors="replace").read()
    out = []
    for m in re.finditer(r"time\s+([0-9.E+-]+)\s*\n\s*\n\s*([-0-9.E+ ]+)", txt):
        try:
            out.append((float(m.group(1)), abs(float(m.group(2).split()[2]))))
        except (ValueError, IndexError):
            pass
    return np.asarray(out, dtype=float)


def one(job):
    """Run a single draw in an isolated directory. Returns a plain dict."""
    (workdir, rt, lr, eph, sigma_hat, a_bar, nu, comp_a, seed,
     shape, alpha_deg, gamma, shorten, timeout) = job
    key = os.path.basename(workdir)
    os.makedirs(workdir, exist_ok=True)
    t0 = time.time()
    try:
        m = ShellMesh(rt, lr, shape=shape, alpha_deg=alpha_deg,
                      thickness_gamma=gamma, elems_per_half_wave=eph)
        coords, coeffs, info = apply_imperfection(
            m, sigma_hat=sigma_hat, a_bar=a_bar, nu=nu, n_panels=N_PANELS,
            seed=seed, comp_a_rms_over_t=comp_a,
            with_component_a=(comp_a > 0))
        inp = os.path.join(workdir, key + ".inp")
        write_twostep_inp(inp, m, coords, shorten * m.end_shortening_cr)
        ok, tail, secs = run_ccx(inp, timeout=timeout)
        d = read_frd_displacements(os.path.join(workdir, key + ".frd")) if ok else None

        res = dict(key=key, ok=bool(ok), seconds=round(time.time() - t0, 1),
                   solve_seconds=round(secs, 1), r_over_t=rt, l_over_r=lr,
                   shape=shape, alpha_deg=alpha_deg, gamma=gamma,
                   sigma_hat=sigma_hat, a_bar=a_bar, nu_matern=nu,
                   comp_a=comp_a, seed=seed, n_nodes=int(m.n_nodes),
                   Z=float(m.batdorf_Z), n_expected=float(expected_n(m)))
        f = _forces(os.path.join(workdir, key + ".dat"))
        A = 2.0 * np.pi * m.R * m.t
        Flin = (shorten * m.end_shortening_cr / m.L) * A
        res["f_end_over_lin"] = float(f[-1, 1] / Flin) if len(f) else float("nan")
        res["f_min_over_lin"] = float(np.min(f[:, 1]) / Flin) if len(f) else float("nan")
        res["buckled"] = bool(res["f_end_over_lin"] < 0.85)

        if d is not None and d.shape[0] >= m.n_nodes:
            w = radial(m, d) / m.t
            ml = mode_label(m, w)
            res.update(n_dominant=ml["n"], n_share=float(ml["share"]),
                       top5=ml["top5"], rms_over_t=float(np.sqrt(np.mean(w ** 2))),
                       max_over_t=float(np.abs(w).max()))
            np.save(os.path.join(workdir, "w.npy"), w.astype(np.float32))
            np.savez_compressed(os.path.join(workdir, "coeffs.npz"), **coeffs)
            np.save(os.path.join(workdir, "force.npy"), f.astype(np.float32))
        else:
            res["ok"] = False
            res["tail"] = tail[-400:]
        # keep only the small artifacts
        for fn in os.listdir(workdir):
            if fn.endswith((".frd", ".inp", ".12d", ".cvg", ".log")):
                try:
                    os.remove(os.path.join(workdir, fn))
                except OSError:
                    pass
        return res
    except Exception as exc:                      # noqa: BLE001
        return dict(key=key, ok=False, error=repr(exc)[:300],
                    seconds=round(time.time() - t0, 1))


def run_jobs(jobs, lanes, out_json):
    done, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=lanes) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            done.append(r)
            msg = ("[%3d/%3d] %-26s %-5s %6.1fs" %
                   (i, len(jobs), r["key"], r.get("ok"), r.get("seconds", 0)))
            if r.get("ok"):
                msg += ("  n=%-3s F/Flin=%.3f rms/t=%.3f"
                        % (r.get("n_dominant"), r.get("f_end_over_lin", float("nan")),
                           r.get("rms_over_t", float("nan"))))
            else:
                msg += "  " + str(r.get("error", r.get("tail", "")))[:90]
            print(msg, flush=True)
            with open(out_json, "w") as f:
                json.dump(done, f, indent=1)
    print("\n%d/%d ok in %.1f min (%d lanes)"
          % (sum(1 for r in done if r.get("ok")), len(done),
             (time.time() - t0) / 60.0, lanes))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compa", "cost", "pilot"], required=True)
    ap.add_argument("--lanes", type=int, default=19,
                    help="60%% of the 32 logical cores, leaving headroom")
    ap.add_argument("--draws", type=int, default=80)
    ap.add_argument("--eph", type=float, default=3.0)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--shorten", type=float, default=1.5)
    ap.add_argument("--comp-a", type=float, default=0.01,
                    dest="comp_a", help="Component A RMS / t; 0.01 from the sweep")
    args = ap.parse_args()

    base = os.path.join(ROOT, "work", args.mode)
    if os.path.exists(base):
        shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    corners = [(110, 1.0), (110, 1.6), (230, 1.0), (230, 1.6)]
    jobs = []

    if args.mode == "compa":
        # Component A amplitude sweep on one corner: find the largest A that
        # breaks SO(2) symmetry without dominating mode selection.
        for ca in (0.0, 0.01, 0.03, 0.10, 0.30):
            for s in range(1, 13):
                k = "ca%03d_s%02d" % (round(ca * 100), s)
                jobs.append((os.path.join(base, k), 110, 1.0, args.eph,
                             args.sigma, 0.70, 1.5, ca, s, "cylinder", 0.0, 0.0,
                             args.shorten, 7200))
    elif args.mode == "cost":
        # box-spanning cost: 2 draws per corner, both mesh densities
        for (rt, lr) in corners:
            for eph in (3.0, 4.0):
                for s in (1, 2):
                    k = "rt%d_lr%03d_e%d_s%d" % (rt, round(lr * 100), int(eph), s)
                    jobs.append((os.path.join(base, k), rt, lr, eph, args.sigma,
                                 0.70, 1.5, 0.0, s, "cylinder", 0.0, 0.0,
                                 args.shorten, 14400))
    else:  # pilot
        # comp_a = 0.01 chosen by the sweep: it breaks SO(2) symmetry while
        # leaving the mode distribution as diverse as A=0 (4 modes, spread
        # 0.350 vs 0.352). At 0.03 Component A starts pinning the mode (n=8
        # wins 7/12) and by 0.10 it dictates it outright.
        for (rt, lr) in corners:
            for s in range(1, args.draws + 1):
                k = "rt%d_lr%03d_s%03d" % (rt, round(lr * 100), s)
                jobs.append((os.path.join(base, k), rt, lr, args.eph, args.sigma,
                             0.70, 1.5, args.comp_a, s, "cylinder", 0.0, 0.0,
                             args.shorten, 14400))

    print("mode=%s  %d jobs  %d lanes" % (args.mode, len(jobs), args.lanes))
    run_jobs(jobs, args.lanes, os.path.join(base, "results.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
