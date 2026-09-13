"""Does a longer settle window actually fix the high-Z geometries?

The 896-draw run showed convergence failure correlating with the Batdorf
parameter (corr(Z, median drift) = +0.81): every Z <= 125 geometry settled
completely, while Z >= 226 ran 52-100% over the retention threshold. The
proposed remedy is to scale the settle window with Z.

That is a hypothesis, not a result. Before spending days of compute on an
expanded grid built around it, run the worst-converging geometry at several
settle lengths and see whether drift actually falls. If it does not, the
remedy is wrong and the grid must avoid high Z instead.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from imperfection import apply_imperfection
from diag_settle import write_diag, read_all_frd_blocks
from ccx import run_ccx
from analyse import radial, mode_label

NU = 0.3


def batdorf(rt, lr):
    return lr ** 2 * rt * np.sqrt(1.0 - NU ** 2)


def one(job):
    rt, lr, seed, t_settle, n_inc, outdir = job
    key = "ss_rt%d_lr%03d_s%03d_t%03d" % (rt, round(lr * 100), seed, round(t_settle))
    d = os.path.join(outdir, key)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    try:
        m = ShellMesh(rt, lr, elems_per_half_wave=3.0)
        coords, _, _ = apply_imperfection(m, sigma_hat=1e-3, a_bar=0.70, nu=1.5,
                                          n_panels=8, seed=seed,
                                          comp_a_rms_over_t=0.15)
        inp = os.path.join(d, key + ".inp")
        write_diag(inp, m, coords, 1.5 * m.end_shortening_cr,
                   t_settle=t_settle, n_inc_settle=n_inc, damp_alpha=0.3)
        ok, tail, secs = run_ccx(inp, timeout=36000)
        if not ok:
            return dict(key=key, ok=False, t_settle=t_settle, seed=seed,
                        seconds=round(time.time() - t0, 1), tail=str(tail)[-200:])
        blocks = read_all_frd_blocks(os.path.join(d, key + ".frd"))
        good = [b for b in blocks if b.shape[0] >= m.n_nodes]
        if len(good) < 3:
            return dict(key=key, ok=False, t_settle=t_settle, seed=seed,
                        seconds=round(time.time() - t0, 1), tail="few frd blocks")
        final = good[-1][:m.n_nodes]
        w = radial(m, final) / m.t
        tail_blocks = good[-max(3, len(good) // 3):]
        rms_tail = [float(np.sqrt(np.mean((radial(m, b[:m.n_nodes]) / m.t) ** 2)))
                    for b in tail_blocks]
        drift = (max(rms_tail) - min(rms_tail)) / max(float(np.mean(rms_tail)), 1e-12)
        ml = mode_label(m, w)
        for fn in os.listdir(d):
            if fn.endswith((".frd", ".inp", ".12d", ".cvg", ".sta", ".dat")):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass
        return dict(key=key, ok=True, rt=rt, lr=lr, seed=seed, t_settle=t_settle,
                    n_inc=n_inc, drift=round(float(drift), 5),
                    n_dominant=int(ml["n"]), n_share=round(float(ml["share"]), 4),
                    rms_over_t=round(float(np.sqrt(np.mean(w ** 2))), 5),
                    seconds=round(time.time() - t0, 1))
    except Exception as exc:                                    # noqa: BLE001
        return dict(key=key, ok=False, t_settle=t_settle, seed=seed,
                    seconds=round(time.time() - t0, 1), tail=repr(exc)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", type=int, default=170)
    ap.add_argument("--lr", type=float, default=1.3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[3001, 3002, 3003])
    ap.add_argument("--settles", type=float, nargs="+", default=[40, 80, 160])
    ap.add_argument("--lanes", type=int, default=9)
    ap.add_argument("--outdir", default="settle")
    ap.add_argument("--out", default="settle_results.json")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    Z = batdorf(a.rt, a.lr)
    print("settle scaling test: R/t=%d L/R=%.1f  Z=%.0f" % (a.rt, a.lr, Z), flush=True)
    print("this geometry ran %s over the 0.15 threshold in the production run"
          % ("81%" if (a.rt, a.lr) == (170, 1.3) else "?"), flush=True)
    print("settle lengths: %s  (increments scale with the window)"
          % a.settles, flush=True)

    jobs = []
    for ts in a.settles:
        n_inc = int(round(200 * ts / 40.0))          # keep the step size fixed
        for s in a.seeds:
            jobs.append((a.rt, a.lr, s, float(ts), n_inc, a.outdir))

    done = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.lanes) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            done.append(r)
            if r.get("ok"):
                print("[%d/%d] t_settle=%-5.0f seed=%d  drift=%.4f  n=%d  %.0fs"
                      % (i, len(jobs), r["t_settle"], r["seed"], r["drift"],
                         r["n_dominant"], r["seconds"]), flush=True)
            else:
                print("[%d/%d] t_settle=%-5.0f seed=%d  FAILED %s"
                      % (i, len(jobs), r["t_settle"], r["seed"],
                         str(r.get("tail"))[:90]), flush=True)
            json.dump(done, open(a.out, "w"), indent=1)

    ok = [r for r in done if r.get("ok")]
    print("\n%-10s %-8s %-10s %-10s %s" % ("t_settle", "n", "med drift",
                                           "max drift", "mean cost [s]"))
    for ts in a.settles:
        g = [r for r in ok if r["t_settle"] == ts]
        if g:
            d = [r["drift"] for r in g]
            print("%-10.0f %-8d %-10.4f %-10.4f %.0f"
                  % (ts, len(g), np.median(d), max(d),
                     np.mean([r["seconds"] for r in g])))
    print("\nverdict: the remedy works if median drift falls below 0.15 as the")
    print("window lengthens, and the mode label stays put. If drift is flat,")
    print("the settle window is not the binding constraint.")
    print("total wall: %.2f h" % ((time.time() - t0) / 3600.0))


if __name__ == "__main__":
    main()
