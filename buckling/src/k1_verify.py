"""K=1 single-imperfection verification.

The interactive viewer's N5_imperf arm used the production stochastic draw's
seed=0 result, which happened to place K=21 dimples -- realistic for the
production law (main.tex Sec 2.2: K ~ round(clip(Normal(20,7^2),0,40))) but
visually busy for a first, easy-to-interpret buckling case. This script forces
K=1 via imperfection.py's new k_override parameter (the production law itself
is untouched -- k_override stays None everywhere else) and reruns the already
corrected deck (N=5 integration points, load = 1.5 x d_cr; see
buckling-rbody-rim-boundary-layer memory) on the identical reference
geometry/seed used throughout this verification work.

Ten arms on the same geometry: nine independent K=1 draws (seeds 0-8, each
exactly one RSA-placed dimple at its own random position/sign/amplitude) plus
one perfect (K=0) calibration baseline already established in
nint_verify.py: peak/P_cr = 1.046. The nine K=1 draws are the point -- they
show the outcome diversity a single hidden random dimple location produces on
an otherwise identical shell, which is what the interactive viewer is for.

Run on aarl only (see the campaign's aarl-only rule).
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
                         RHO, TEND, draw_imperfection, frame_radial_profile,
                         reaction_history)
from run_batch import (ENGINE_EXE, STARTER_EXE, _parse_engine_out,
                       _parse_starter_out, _radioss_env, _run_exe)

FRAC = 1.50       # main.tex Sec 2.3's specified target, 1.5 * d_cr
T_RAMP = 0.05
N_INTEG = 5       # corrected deck
N_SEEDS = 9       # nine independent K=1 draws + one perfect baseline = 10 arms

# (label, seed, k_override or None, imperfect)
ARMS = [(f"K1_seed{s}", s, 1, True) for s in range(N_SEEDS)] + [
    ("perfect", 0, None, False),
]


def smooth(xs, w=5):
    """Centered moving average -- the reaction force is a finite difference of
    the log's external-work column, so a single spike is noise, not a peak."""
    if len(xs) < w:
        return list(xs)
    k = np.ones(w) / w
    return list(np.convolve(np.asarray(xs, dtype=float), k, mode="same"))


def run_arm(root, label, seed, k_override, imperfect):
    mesh = geometry.ShellMesh(r_over_t=R_OVER_T, l_over_r=L_OVER_R)
    if imperfect:
        coords, record = draw_imperfection(mesh, seed, k_override=k_override)
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
        impose_velocity=v, t_ramp=T_RAMP, title=f"k1 {label}",
        n_integ=N_INTEG,
    )
    radioss.write_engine(draw_dir / "draw_0001.rad", "draw", 1,
                          tend=TEND, anim_dt=TEND / N_ANIM_FRAMES)

    res = dict(label=label, n_integ=N_INTEG, imperfect=imperfect, frac=FRAC,
               seed=seed, k_override=k_override, K=kwaves, imperf_max_over_t=imperf,
               R=R, t=t, L=mesh.L * LENGTH_SCALE, d_cr=d_cr, v=v, eps_cr=eps_cr,
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

    post = [q for q in rx if q["time"] > T_RAMP]
    fs = smooth([q["force"] for q in post])
    ipk = int(np.argmax(fs)) if fs else -1
    res.update(
        ke_ie=final["ke_t"] / final["i_energy"] if final["i_energy"] > 0 else float("nan"),
        ie_end=final["i_energy"],
        ie_linear_end=0.5 * k_axial * (post[-1]["delta"] ** 2) if post else float("nan"),
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
                  f"K={r.get('K')} peak/P_cr={r.get('peak_over_pcr', float('nan')):.3f}",
                  flush=True)
    results.sort(key=lambda r: (r["imperfect"], r.get("seed", 0)), reverse=True)
    (root / "k1_summary.json").write_text(json.dumps(results, indent=1))

    r0 = results[0] if results else {}
    print(f"\n=== K=1 verification at delta={FRAC} d_cr, R/t={R_OVER_T} L/R={L_OVER_R} "
          f"({N_SEEDS} seeds + 1 perfect) ===")
    print(f"P_cr = {r0.get('p_cr', float('nan')):.1f} kN   "
          f"d_cr = {r0.get('d_cr', float('nan')):.4f} mm   "
          f"EA/L = {r0.get('k_axial', float('nan')):.1f} kN/mm")
    print(f"{'arm':12s} {'seed':>4s} {'K':>2s} {'ok':>5s} {'peakF':>9s} "
          f"{'pk/P_cr':>8s} {'pk@d/dcr':>9s} {'imperf':>7s} {'max|dr|/t':>10s}")
    for r in results:
        rad = (r.get("radial") or [{}])[-1]
        print(f"{r['label']:12s} {r.get('seed', 0):4d} {r['K']:2d} {str(r.get('ok')):>5s} "
              f"{r.get('peak_force', float('nan')):9.3g} "
              f"{r.get('peak_over_pcr', float('nan')):8.3f} "
              f"{r.get('peak_delta_over_dcr', float('nan')):9.3f} "
              f"{r.get('imperf_max_over_t', 0):7.3f} "
              f"{rad.get('max_abs', float('nan')):10.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../runs/k1_verify")
