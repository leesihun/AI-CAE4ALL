"""Production generator for the buckling one-to-many benchmark.

One "draw" = one geometry + one realization of the withheld imperfection field
(Component B). The geometry and the deterministic process signature (Component A)
are GIVEN to the model; Component B is the withheld latent. That is the entire
source of spread, and it is the same law at every geometry -- which is what makes
a distribution learned on one family transferable to another.

Differences from the earlier pilot, all of them load-bearing (DATASET_CARD.md s7):
  * Component A no longer contains a harmonic inside the critical band, and its
    amplitude was re-picked at 0.15*t against a direct sweep. It contributes a
    geometry-dependent input. Phase tests did not detect pinning in the pilot;
    the non-axisymmetric signature does not imply exact SO(2) invariance.
  * The settle window is t=40, 200 inc. RMS drift is a diagnostic, not a
    certificate of static equilibrium or of unchanged field shape.
  * The full 3-D displacement is kept, not only the radial component, and the
    full circumferential spectrum is kept alongside the argmax mode label --
    the label alone is not damping-invariant on near-degenerate draws.
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
from imperfection import apply_imperfection, critical_wavenumber
from diag_settle import write_diag, read_all_frd_blocks, THICKNESS_SCHEMA
from ccx import run_ccx, read_total_force
from analyse import radial, mode_label, circumferential_spectrum

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work", "production")

N_PANELS = 8
SIGMA_HAT = 1e-3      # withheld-field RMS, in units of t
A_BAR = 0.70          # correlation length, in units of sqrt(R*t)
NU_MATERN = 1.5
COMP_A = 0.15         # process-signature RMS, in units of t (see BUILD_LOG "Component A")
EPH = 3.0             # elements per buckling half-wave
DAMP = 0.3
T_SETTLE = 40.0
N_INC_SETTLE = 200


def run_spec(g, seed, shorten):
    """Settings affecting the solve; persisted so changed jobs cannot hit cache."""
    return dict(geometry=g, seed=int(seed), shorten=float(shorten),
                sigma_hat=SIGMA_HAT, a_bar=A_BAR, nu_matern=NU_MATERN,
                comp_a=COMP_A, n_panels=N_PANELS, elems_per_half_wave=EPH,
                damp_alpha=DAMP, t_settle=T_SETTLE, n_inc_settle=N_INC_SETTLE,
                thickness_schema=THICKNESS_SCHEMA)


def cache_problem(z, g, seed, shorten):
    if "run_spec_json" in z:
        if json.loads(str(z["run_spec_json"])) != run_spec(g, seed, shorten):
            return "cached run settings differ from requested solve"
        return None
    if float(g.get("thickness_gamma", 0)) != 0:
        return "legacy taper was solved with uniform thickness; use a new --work-dir"
    # Legacy cylinders/cones are retained for the documented original recipe.
    # They have no complete source provenance, so do not reuse them under any
    # different settings or target shortening.
    if (SIGMA_HAT, A_BAR, NU_MATERN, COMP_A, N_PANELS, EPH, DAMP, T_SETTLE,
            N_INC_SETTLE, float(shorten)) != (1e-3, .70, 1.5, .15, 8, 3., .3, 40., 200, 1.5):
        return "unversioned cache only supports the original uniform-thickness recipe"
    return None


def geom_key(g):
    return "%s_rt%d_lr%03d_a%02d_g%02d" % (
        g["shape"][:3], g["r_over_t"], round(g["l_over_r"] * 100),
        round(g["alpha_deg"] * 10), round(g["thickness_gamma"] * 100))


def one(job):
    g, seed, shorten, timeout = job[:4]
    work_dir = job[4] if len(job) > 4 else W
    key = geom_key(g) + "_s%03d" % seed
    d = os.path.join(work_dir, key)
    os.makedirs(d, exist_ok=True)
    done = os.path.join(d, "draw.npz")
    t0 = time.time()
    if os.path.exists(done):                       # resumable
        try:
            with np.load(done) as z:
                problem = cache_problem(z, g, seed, shorten)
                if problem:
                    return _fail(key, g, seed, t0, problem)
                r = dict(key=key, ok=True, cached=True, seed=seed, seconds=0.0,
                         n_dominant=int(z["n_dominant"]), n_share=float(z["n_share"]),
                         rms_over_t=float(z["rms_over_t"]), drift=float(z["drift"]))
            r.update(g)
            return r
        except Exception as exc:
            return _fail(key, g, seed, t0, "unreadable cache (preserved): " + repr(exc))

    t0 = time.time()
    try:
        m = ShellMesh(g["r_over_t"], g["l_over_r"], shape=g["shape"],
                      alpha_deg=g["alpha_deg"],
                      thickness_gamma=g["thickness_gamma"],
                      elems_per_half_wave=EPH)
        coords, coeffs, info = apply_imperfection(
            m, sigma_hat=SIGMA_HAT, a_bar=A_BAR, nu=NU_MATERN, n_panels=N_PANELS,
            seed=seed, comp_a_rms_over_t=COMP_A, with_component_a=True)

        inp = os.path.join(d, key + ".inp")
        write_diag(inp, m, coords, shorten * m.end_shortening_cr,
                   t_settle=T_SETTLE, n_inc_settle=N_INC_SETTLE, damp_alpha=DAMP)
        ok, tail, secs = run_ccx(inp, timeout=timeout)
        if not ok:
            return _fail(key, g, seed, t0, str(tail)[-300:])

        blocks = read_all_frd_blocks(os.path.join(d, key + ".frd"))
        good = [b for b in blocks if b.shape[0] >= m.n_nodes]
        if len(good) < 3:
            return _fail(key, g, seed, t0, "only %d usable frd blocks" % len(good))

        final = good[-1][:m.n_nodes]
        w = radial(m, final) / m.t
        ml = mode_label(m, w)
        spec, _ = circumferential_spectrum(m, w)

        # Settle residual: how much the field still moved over the last third of
        # the window, relative to its own amplitude. This is the honest
        # convergence flag -- a draw still in motion is not an equilibrium, and
        # the pilot shipped 320 of those because nothing measured it.
        tail_blocks = good[-max(3, len(good) // 3):]
        rms_tail = [float(np.sqrt(np.mean((radial(m, b[:m.n_nodes]) / m.t) ** 2)))
                    for b in tail_blocks]
        drift = (max(rms_tail) - min(rms_tail)) / max(float(np.mean(rms_tail)), 1e-12)

        # Reaction-force history at the loaded end. The deck already requests
        # it; earlier runs deleted the .dat before reading it and threw the
        # engineering scalar away. Keep it: measured on the pilot, the
        # knockdown factor varies only 0.33-0.62% between draws of ONE geometry
        # while spanning 13.3% across geometries, whereas the buckling pattern
        # is broadly distributed within every geometry. That gives the dataset
        # a near-deterministic target and a genuinely distributional one, and
        # a calibrated model has to be right about both -- an ensemble that is
        # uniformly too narrow looks good on the first and fails the second.
        f_t, f_z = read_total_force(os.path.join(d, key + ".dat"))
        f_peak = float(np.abs(f_z).max()) if f_z.size else float("nan")
        f_end = float(f_z[-1]) if f_z.size else float("nan")

        temporary = done + ".tmp.npz"
        np.savez_compressed(
            temporary,
            run_spec_json=json.dumps(run_spec(g, seed, shorten), sort_keys=True),
            force_t=f_t.astype(np.float32), force_z=f_z.astype(np.float32),
            disp=final.astype(np.float32), w=w.astype(np.float32),
            imp_C_real=coeffs["C_real"].astype(np.float32),
            imp_C_imag=coeffs["C_imag"].astype(np.float32),
            k1=coeffs["k1"].astype(np.float32),
            k2=coeffs["k2"].astype(np.float32),
            spectrum=spec.astype(np.float32),
            n_dominant=ml["n"], n_share=ml["share"],
            rms_over_t=float(np.sqrt(np.mean(w ** 2))), drift=drift)
        os.replace(temporary, done)

        for fn in os.listdir(d):
            if fn.endswith((".frd", ".inp", ".12d", ".cvg", ".sta", ".dat")):
                try:
                    os.remove(os.path.join(d, fn))
                except OSError:
                    pass

        r = dict(key=key, ok=True, cached=False, seed=seed,
                 n_dominant=int(ml["n"]), n_share=round(float(ml["share"]), 4),
                 top5=ml["top5"], top5_w=[round(x, 4) for x in ml["top5_w"]],
                 rms_over_t=round(float(np.sqrt(np.mean(w ** 2))), 5),
                 max_over_t=round(float(np.abs(w).max()), 5),
                 drift=round(float(drift), 5),
                 f_peak=f_peak, f_end=f_end, n_force=int(f_z.size),
                 n_crit_pred=round(critical_wavenumber(g["r_over_t"]), 2),
                 n_nodes=int(m.n_nodes), Z=round(float(m.batdorf_Z), 1),
                 rms_B_over_t=round(float(info["rms_B_over_t"]), 6),
                 seconds=round(time.time() - t0, 1))
        r.update(g)
        return r
    except Exception as exc:                                       # noqa: BLE001
        return _fail(key, g, seed, t0, repr(exc)[:300])


def _fail(key, g, seed, t0, tail):
    r = dict(key=key, ok=False, seed=seed, tail=tail,
             seconds=round(time.time() - t0, 1))
    r.update(g)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True,
                    help="json file: list of geometry dicts, each with n_draws")
    ap.add_argument("--lanes", type=int, default=19,
                    help="60%% of 32 logical cores, per the standing constraint")
    ap.add_argument("--shorten", type=float, default=1.5)
    ap.add_argument("--timeout", type=int, default=9000)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--work-dir", default=W,
                    help="raw solve/cache directory; use a new directory to regenerate invalid legacy tapers")
    a = ap.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    plan = json.load(open(a.plan))

    jobs = []
    for entry in plan:
        e = dict(entry)
        n = e.pop("n_draws")
        s0 = e.pop("seed0", 1000)
        sh = e.pop("shorten", a.shorten)
        e.pop("tier", None)
        for k in range(n):
            jobs.append((dict(e), s0 + k, sh, a.timeout, a.work_dir))

    outp = os.path.join(a.work_dir, a.out)
    done = []
    if os.path.exists(outp):
        try:
            done = json.load(open(outp))
        except Exception:
            done = []
    total = len(jobs)
    # A results.json entry alone cannot establish a usable artifact. Recheck
    # every cache in one(); that also prevents stale taper successes being skipped.
    done = {r["key"]: r for r in done}
    print("%d draws over %d geometries; %d already done, %d to run, %d lanes"
          % (total, len(plan), total - len(jobs), len(jobs), a.lanes), flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.lanes) as ex:
        futs = {ex.submit(one, j): j for j in jobs}
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            done[r["key"]] = r
            el = time.time() - t0
            eta = el / i * (len(jobs) - i) / 3600.0
            if r.get("ok"):
                print("[%4d/%d] %-36s n=%-3d sh=%.2f rms=%.3f drift=%.4f %5.0fs  ETA %.1fh"
                      % (i, len(jobs), r["key"], r["n_dominant"], r["n_share"],
                         r["rms_over_t"], r.get("drift", -1.0), r["seconds"], eta),
                      flush=True)
            else:
                print("[%4d/%d] %-36s FAILED %s"
                      % (i, len(jobs), r["key"], str(r.get("tail"))[:110]), flush=True)
            temporary = outp + ".tmp"
            with open(temporary, "w") as f:
                json.dump(list(done.values()), f, indent=1)
            os.replace(temporary, outp)

    nok = sum(1 for r in done.values() if r.get("ok"))
    print("done: %d/%d ok in %.2f h" % (nok, len(done), (time.time() - t0) / 3600.0),
          flush=True)


if __name__ == "__main__":
    main()
