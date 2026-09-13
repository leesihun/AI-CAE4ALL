"""Run several seeds at the CONVERGED settle settings, keeping full 3-D displacement.

The production pilot used a settle window that is too short (see DATASET_CARD.md
section 7). This runs the same geometry at the settle length that was verified to
converge on this corner, and keeps the whole displacement vector rather than only
the radial component, so the deformed geometry can be reconstructed.
"""
import argparse, os, sys, json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from imperfection import apply_imperfection
from ccx import run_ccx
from diag_settle import write_diag, read_all_frd_blocks
from analyse import radial, mode_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work", "converged")


def one(job):
    rt, lr, seed, t_settle, n_inc_settle, damp, tag, with_a, ca = job
    key = "cv%s_rt%d_lr%03d_s%03d_d%g" % (tag, rt, round(lr * 100), seed, damp)
    d = os.path.join(W, key)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    try:
        m = ShellMesh(rt, lr, elems_per_half_wave=3.0)
        coords, _, _ = apply_imperfection(m, sigma_hat=1e-3, a_bar=0.70, nu=1.5,
                                          n_panels=8, seed=seed,
                                          comp_a_rms_over_t=ca,
                                          with_component_a=with_a)
        inp = os.path.join(d, key + ".inp")
        write_diag(inp, m, coords, 1.5 * m.end_shortening_cr,
                   t_settle=t_settle, n_inc_settle=n_inc_settle, damp_alpha=damp)
        ok, tail, secs = run_ccx(inp, timeout=10800)
        if not ok:
            return dict(key=key, ok=False, tail=tail[-400:])

        blocks = read_all_frd_blocks(os.path.join(d, key + ".frd"))
        good = [b for b in blocks if b.shape[0] >= m.n_nodes]
        if not good:
            return dict(key=key, ok=False, tail="no usable frd block")
        final = good[-1][:m.n_nodes]
        np.save(os.path.join(d, "disp3d.npy"), final.astype(np.float32))
        w = radial(m, final) / m.t
        ml = mode_label(m, w)
        np.save(os.path.join(d, "w.npy"), w.astype(np.float32))
        for fn in os.listdir(d):
            if fn.endswith((".frd", ".inp", ".12d", ".cvg")):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass
        return dict(key=key, ok=True, seed=seed, r_over_t=rt, l_over_r=lr,
                    n_dominant=ml["n"], n_share=round(ml["share"], 3),
                    rms_over_t=float(np.sqrt(np.mean(w ** 2))),
                    max_over_t=float(np.abs(w).max()),
                    seconds=round(time.time() - t0, 1), n_nodes=int(m.n_nodes))
    except Exception as exc:                                   # noqa: BLE001
        return dict(key=key, ok=False, tail=repr(exc)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", type=int, default=110)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--t-settle", type=float, default=40.0)
    ap.add_argument("--n-inc-settle", type=int, default=200)
    ap.add_argument("--damp", type=float, default=0.3)
    ap.add_argument("--lanes", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-comp-a", action="store_true")
    ap.add_argument("--comp-a", type=float, default=0.01)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    os.makedirs(W, exist_ok=True)
    jobs = [(a.rt, a.lr, s, a.t_settle, a.n_inc_settle, a.damp, a.tag,
             not a.no_comp_a, a.comp_a) for s in a.seeds]
    print("running %d seeds on R/t=%d L/R=%.1f, %d lanes" % (len(jobs), a.rt, a.lr, a.lanes),
          flush=True)
    done = []
    with ProcessPoolExecutor(max_workers=a.lanes) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            done.append(r)
            if r.get("ok"):
                print("[%d/%d] %s  n=%d rms/t=%.4f  %.0fs"
                      % (i, len(jobs), r["key"], r["n_dominant"], r["rms_over_t"],
                         r["seconds"]), flush=True)
            else:
                print("[%d/%d] %s FAILED %s" % (i, len(jobs), r["key"],
                                                str(r.get("tail"))[:120]), flush=True)
            json.dump(done, open(os.path.join(W, a.out), "w"), indent=1)
    print("done: %d/%d ok" % (sum(1 for r in done if r.get("ok")), len(done)), flush=True)


if __name__ == "__main__":
    main()
