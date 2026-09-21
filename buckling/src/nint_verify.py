"""Does the shell have any bending stiffness? -- /PROP/SHELL N sweep.

`load_verify.py` established that raising the load fixes nothing on its own:
at delta = 1.0 d_cr the shell absorbs 0.287 J where linear elasticity predicts
30.8 J, i.e. it carries ~1% of P_cr = sigma_cl * 2*pi*R*t. The energy log says
where that goes -- IE tracks the linear-elastic prediction exactly
(IE/linear = 1.007) up to delta = 0.012 mm = 3.9% of d_cr, then the shell
loses essentially all axial stiffness and IE flatlines while the imposed
shortening grows 25x further.

Collapse at ~5% of the classical stress is not a knockdown. The seeded
imperfection here is 0.248 t, for which the empirical knockdown at R/t = 197
is ~0.5, not 0.05. The deck's `/PROP/SHELL` writes `N = 1` -- one
through-thickness integration point, on the mid-surface -- which leaves the
element with membrane stress only and no through-thickness stress gradient to
resist bending. Shell buckling is sigma_cr ~ sqrt(membrane * bending), so
losing the bending term is exactly the failure signature observed: a perfectly
linear EA/L response until the first fold forms, then near-zero stiffness, with
the folds localized in the two clamped-edge bending boundary layers (measured
mean dr = -0.20 t there against ~0 at mid-wall).

This sweeps N at one fixed load on one fixed geometry and seed, and adds
perfect-cylinder arms as the calibration point: a perfect shell should peak
near P_cr, an imperfect one at a knockdown factor of it. Everything else is
held at the production settings.

Run on aarl only, and from a copy of `src/` that the live production campaign
does not import.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import geometry
import radioss
from load_verify import (E, L_OVER_R, LENGTH_SCALE, N_ANIM_FRAMES, NU, R_OVER_T,
                         RHO, SEED, TEND, draw_imperfection, frame_radial_profile,
                         reaction_history)
from run_batch import (ENGINE_EXE, STARTER_EXE, _parse_engine_out,
                       _parse_starter_out, _radioss_env, _run_exe)

FRAC = 1.50       # main.tex Sec 2.3's specified target, 1.5 * d_cr
T_RAMP = 0.05

# (label, n_integ, imperfect). N=1 repeats the production deck; N=3/5 are the
# standard shell settings. The perfect arms calibrate the peak against the
# classical P_cr with no imperfection knockdown in the way.
ARMS = [
    ("N1_imperf", 1, True),
    ("N3_imperf", 3, True),
    ("N5_imperf", 5, True),
    ("N1_perfect", 1, False),
    ("N5_perfect", 5, False),
]


def smooth(xs, w=5):
    """Centered moving average -- the reaction force is a finite difference of
    the log's external-work column, so a single spike is noise, not a peak."""
    if len(xs) < w:
        return list(xs)
    k = np.ones(w) / w
    return list(np.convolve(np.asarray(xs, dtype=float), k, mode="same"))


def run_arm(root, label, n_integ, imperfect):
    mesh = geometry.ShellMesh(r_over_t=R_OVER_T, l_over_r=L_OVER_R)
    if imperfect:
        coords, record = draw_imperfection(mesh, SEED)
        imperf = record["max_abs_over_t"]
        kwaves = record["K"]
    else:
        coords, imperf, kwaves = mesh.coords, 0.0, 0

    d_cr = mesh.end_shortening_cr * LENGTH_SCALE
    v = FRAC * d_cr / (TEND - 0.5 * T_RAMP)
    R = mesh.R * LENGTH_SCALE
    t = mesh.t * LENGTH_SCALE
    eps_cr = float(mesh.sigma_cr_classical)
    p_cr = eps_cr * E * 2.0 * np.pi * R * t      # classical axial load, kN
    k_axial = E * 2.0 * np.pi * R * t / (mesh.L * LENGTH_SCALE)

    draw_dir = root / label
    draw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = draw_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    env = _radioss_env()
    t0 = time.time()

    radioss.write_starter(
        draw_dir / "draw_0000.rad", mesh, coords, radioss.DeckIds(mesh),
        E=E, nu=NU, rho=RHO, thickness=mesh.t, length_scale=LENGTH_SCALE,
        impose_velocity=v, t_ramp=T_RAMP, title=f"nint {label}",
        n_integ=n_integ,
    )
    radioss.write_engine(draw_dir / "draw_0001.rad", "draw", 1,
                          tend=TEND, anim_dt=TEND / N_ANIM_FRAMES)

    res = dict(label=label, n_integ=n_integ, imperfect=imperfect, frac=FRAC,
               seed=SEED, K=kwaves, imperf_max_over_t=imperf, R=R, t=t,
               L=mesh.L * LENGTH_SCALE, d_cr=d_cr, v=v, eps_cr=eps_cr,
               p_cr=float(p_cr), k_axial=float(k_axial))

    rc0 = _run_exe(STARTER_EXE, draw_dir / "draw_0000.rad", draw_dir, env,
                    timeout=3600)
    n_err, n_warn = _parse_starter_out(draw_dir / "draw_0000.out")
    res.update(starter_rc=rc0, n_err=n_err, n_warn=n_warn)
    if rc0 != 0 or n_err != 0:
        res["ok"] = False
        return res

    rc1 = _run_exe(ENGINE_EXE, draw_dir / "draw_0001.rad", draw_dir, env,
                    timeout=7200)
    normal_term, rows = _parse_engine_out(draw_dir / "draw_0001.out")
    res.update(engine_rc=rc1, normal_term=normal_term, n_rows=len(rows),
               wall_s=round(time.time() - t0, 1))
    if not rows:
        res["ok"] = False
        return res

    final = rows[-1]
    rx = reaction_history(rows, v, T_RAMP)
    res["reaction"] = rx
    res["radial"] = frame_radial_profile(draw_dir, "draw", t, tmp_dir)
    res["ok"] = bool(normal_term and final["time"] >= 0.9 * TEND)

    # Where does IE leave the linear-elastic curve? That crossing is the load
    # the shell actually fails at, independent of the noisy force estimate.
    # Rows inside the ramp have a tiny delta and an IE that has not caught up,
    # so IE/IE_linear dips below any threshold there for every arm regardless of
    # what the shell does -- skip them or this metric reports the ramp, not the
    # shell (it read 0.012 for all five arms, including one that stayed linear
    # to delta/d_cr = 1.1).
    post = [q for q in rx if q["time"] > T_RAMP]
    dep = next((q for q in post
                if 0.5 * k_axial * q["delta"] ** 2 > 0 and
                q["ie"] / (0.5 * k_axial * q["delta"] ** 2) < 0.75), None)
    fs = smooth([q["force"] for q in post])
    ipk = int(np.argmax(fs)) if fs else -1
    res.update(
        ke_ie=final["ke_t"] / final["i_energy"] if final["i_energy"] > 0 else float("nan"),
        ie_end=final["i_energy"],
        ie_linear_end=0.5 * k_axial * (post[-1]["delta"] ** 2) if post else float("nan"),
        depart_delta_over_dcr=dep["delta"] / d_cr if dep else float("nan"),
        peak_force=fs[ipk] if fs else float("nan"),
        peak_over_pcr=(fs[ipk] / p_cr) if fs else float("nan"),
        peak_delta_over_dcr=(post[ipk]["delta"] / d_cr) if fs else float("nan"),
    )
    for pat in ("draw_0000_0001.rst", "draw_0001_0001.rst"):
        (draw_dir / pat).unlink(missing_ok=True)
    tmp_dir.rmdir()
    (draw_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=len(ARMS)) as ex:
        futs = {ex.submit(run_arm, root, *a): a[0] for a in ARMS}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as exc:
                print(f"{futs[fut]}: EXCEPTION {exc!r}", flush=True)
                continue
            results.append(r)
            print(f"{r['label']}: ok={r.get('ok')} wall={r.get('wall_s')}s "
                  f"peak/P_cr={r.get('peak_over_pcr', float('nan')):.3f}",
                  flush=True)
    results.sort(key=lambda r: (not r["imperfect"], r["n_integ"]))
    (root / "nint_summary.json").write_text(json.dumps(results, indent=1))

    r0 = results[0] if results else {}
    print(f"\n=== N sweep at delta={FRAC} d_cr, R/t={R_OVER_T} L/R={L_OVER_R} "
          f"seed={SEED} ===")
    print(f"P_cr = {r0.get('p_cr', float('nan')):.1f} kN   "
          f"d_cr = {r0.get('d_cr', float('nan')):.4f} mm   "
          f"EA/L = {r0.get('k_axial', float('nan')):.1f} kN/mm   "
          f"imperfection = {r0.get('imperf_max_over_t', 0):.3f} t")
    print(f"{'arm':12s} {'N':>2s} {'imp':>4s} {'ok':>5s} {'peakF':>9s} "
          f"{'pk/P_cr':>8s} {'pk@d/dcr':>9s} {'lin->d/dcr':>11s} "
          f"{'IE/IElin':>9s} {'max|dr|/t':>10s}")
    for r in results:
        rad = (r.get("radial") or [{}])[-1]
        iel = r.get("ie_linear_end", float("nan"))
        print(f"{r['label']:12s} {r['n_integ']:2d} "
              f"{'yes' if r['imperfect'] else 'no':>4s} {str(r.get('ok')):>5s} "
              f"{r.get('peak_force', float('nan')):9.3g} "
              f"{r.get('peak_over_pcr', float('nan')):8.3f} "
              f"{r.get('peak_delta_over_dcr', float('nan')):9.3f} "
              f"{r.get('depart_delta_over_dcr', float('nan')):11.3f} "
              f"{(r.get('ie_end', float('nan')) / iel if iel else float('nan')):9.4f} "
              f"{rad.get('max_abs', float('nan')):10.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../runs/nint_verify")
