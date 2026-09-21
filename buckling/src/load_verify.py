"""Single-geometry load-level verification.

Purpose: the production campaign imposes only 0.0975 * d_cr of end shortening
(run_batch.py's `v = 0.1 * d_cr_actual / tend`, delivered over
`tend - t_ramp/2`), i.e. ~10% of the classical critical end shortening. No
shell in the dataset can buckle at that level, so the exported "modes" are the
seeded imperfection plus a clamped-edge bending boundary layer, amplified
~300x by the viewer's warp scale. This script tests that claim directly by
sweeping the load level on ONE geometry with ONE fixed imperfection seed,
changing nothing else.

What it adds over run_batch.run_one_draw, and why each addition is needed to
see buckling at all:

  * 41 ANIM frames instead of 2. The production deck's `anim_dt = tend`
    records only t=0 and t=tend, so there is no way to tell a shell that grew
    smoothly from one that snapped. Buckling is an event in time.
  * A load-displacement curve reconstructed from the energy log. The starter
    writes no /TH card of any kind, so no reaction force is saved anywhere.
    But the imposed end shortening is a known function of time, and the log
    prints external work, so F(t) = dW/dt / v(t) recovers the axial reaction
    without touching the deck's card set. A buckling run shows F rising,
    peaking, and dropping; a purely elastic run shows F rising monotonically.
  * max|dr|/t per frame. Real post-buckling radial displacement is 1-5x the
    shell thickness; the exported dataset sits at 0.063-0.078.

Everything else -- mesh, imperfection, material, BCs, RBODY ends, t_ramp,
tend -- is taken unchanged from the production path, so the only variable is
the load level. Run on aarl only (see the campaign's aarl-only rule); the
arms are independent and run concurrently.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import geometry
import imperfection
import radioss
from run_batch import (ENGINE_EXE, RADIOSS_ROOT, STARTER_EXE, _parse_engine_out,
                       _parse_starter_out, _radioss_env, _run_exe)

# The geometry the interactive viewer's shape #1 was exported from
# (R/t = 196.6 -> R = 100.0 mm, t = 0.508647 mm, L = 96.8224 mm).
R_OVER_T = 196.6
L_OVER_R = 0.968
SEED = 0

E, NU, RHO = 210.0, 0.3, 7.85e-6
LENGTH_SCALE = 100.0
TEND = 1.0
N_ANIM_FRAMES = 40

ANIM_TO_VTK = RADIOSS_ROOT / "exec" / "anim_to_vtk_linux64_gf"

# (label, load fraction of d_cr, t_ramp). f=0.0975 reproduces the production
# campaign exactly and is the baseline every other arm is read against; the
# last arm varies t_ramp instead of the load, to separate "too abrupt a start"
# from "too small a load" as explanations for the ringing in the snapshots.
ARMS = [
    ("f0.0975_prod", 0.0975, 0.05),
    ("f0.50", 0.50, 0.05),
    ("f1.00", 1.00, 0.05),
    ("f1.50", 1.50, 0.05),
    ("f2.50", 2.50, 0.05),
    ("f1.50_slowramp", 1.50, 0.20),
]


def draw_imperfection(mesh, seed, k_override=None):
    """Same RSA retry loop as run_batch.run_one_draw, so the arms all carry
    the identical imperfection field and differ only in load level.
    k_override forces the dimple count K instead of sampling it -- for
    controlled verification cases (e.g. a single-dimple K=1 arm)."""
    for attempt in range(50):
        rng = np.random.default_rng((seed, attempt))
        try:
            return imperfection.apply_imperfection(mesh, rng, k_override=k_override)
        except RuntimeError:
            continue
    raise RuntimeError(f"RSA never placed a dimple for seed={seed}")


def read_vtk_points(path):
    """Minimal ASCII-VTK point reader -- aarl's bare python3 has numpy but no
    pyvista, and only the point coordinates are needed here."""
    with open(path) as f:
        tok = f.read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    vals = np.array(tok[i + 3: i + 3 + 3 * n], dtype=np.float64)
    return vals.reshape(n, 3)


def anim_frames(draw_dir, run_name):
    return sorted(draw_dir.glob(f"{run_name}A[0-9][0-9][0-9]"))


def frame_radial_profile(draw_dir, run_name, thickness, tmp_dir):
    """max/rms radial displacement per ANIM frame, in units of shell
    thickness, measured against the frame-1 (t=0) geometry."""
    frames = anim_frames(draw_dir, run_name)
    if not frames:
        return []
    ref = None
    out = []
    for fr in frames:
        vtk_path = tmp_dir / f"{fr.name}.vtk"
        # The converter resolves its argument against `cwd`, so it must be the
        # bare frame name -- passing the script-relative path silently makes
        # the frame unfindable and the converter exits 1.
        with open(vtk_path, "w") as fh:
            subprocess.run([str(ANIM_TO_VTK), fr.name], cwd=fr.parent,
                            stdout=fh, stderr=subprocess.DEVNULL, check=True)
        pts = read_vtk_points(vtk_path)
        vtk_path.unlink(missing_ok=True)
        r = np.hypot(pts[:, 0], pts[:, 1])
        if ref is None:
            ref = r
        dr = (r - ref) / thickness
        out.append(dict(frame=fr.name, n=int(len(r)),
                        max_abs=float(np.abs(dr).max()),
                        rms=float(np.sqrt((dr ** 2).mean())),
                        p99=float(np.percentile(dr, 99)),
                        p01=float(np.percentile(dr, 1))))
    return out


def reaction_history(rows, v, t_ramp):
    """Axial reaction force and end shortening vs time, reconstructed from the
    engine log's external-work column. W(t) = integral F d(delta), so
    F = dW/dt / v(t); v(t) follows the same /FUNCT ramp the deck imposes."""
    out = []
    for a, b in zip(rows[:-1], rows[1:]):
        dt = b["time"] - a["time"]
        if dt <= 0:
            continue
        t_mid = 0.5 * (a["time"] + b["time"])
        v_mid = v * min(1.0, t_mid / t_ramp) if t_ramp > 0 else v
        if v_mid <= 0:
            continue
        delta = v * (b["time"] - 0.5 * min(b["time"], t_ramp))
        out.append(dict(time=b["time"], delta=delta,
                        force=(b["ext_work"] - a["ext_work"]) / dt / v_mid,
                        ie=b["i_energy"], ke=b["ke_t"] + b["ke_r"],
                        ext_work=b["ext_work"]))
    return out


def run_arm(root, label, frac, t_ramp):
    mesh = geometry.ShellMesh(r_over_t=R_OVER_T, l_over_r=L_OVER_R)
    coords, record = draw_imperfection(mesh, SEED)

    d_cr = mesh.end_shortening_cr * LENGTH_SCALE
    # Deliver exactly frac*d_cr of total shortening by tend, accounting for the
    # area lost under the /FUNCT ramp. (Production instead sets v = 0.1*d_cr/tend
    # and therefore delivers 0.0975*d_cr, not 0.1*d_cr -- the f0.0975 arm
    # reproduces that delivered value.)
    v = frac * d_cr / (TEND - 0.5 * t_ramp)

    draw_dir = root / label
    draw_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = draw_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    env = _radioss_env()
    run_name = "draw"
    t0 = time.time()

    radioss.write_starter(
        draw_dir / f"{run_name}_0000.rad", mesh, coords, radioss.DeckIds(mesh),
        E=E, nu=NU, rho=RHO, thickness=mesh.t, length_scale=LENGTH_SCALE,
        impose_velocity=v, t_ramp=t_ramp, title=f"verify {label}",
    )
    radioss.write_engine(draw_dir / f"{run_name}_0001.rad", run_name, 1,
                          tend=TEND, anim_dt=TEND / N_ANIM_FRAMES)

    res = dict(label=label, frac=frac, t_ramp=t_ramp, seed=SEED,
               r_over_t=R_OVER_T, l_over_r=L_OVER_R, K=record["K"],
               imperf_max_over_t=record["max_abs_over_t"],
               R=mesh.R * LENGTH_SCALE, L=mesh.L * LENGTH_SCALE,
               t=mesh.t * LENGTH_SCALE, d_cr=d_cr, v=v,
               eps_cr=float(mesh.sigma_cr_classical), n_elem=len(mesh.elements))

    rc0 = _run_exe(STARTER_EXE, draw_dir / f"{run_name}_0000.rad", draw_dir, env,
                    timeout=3600)
    n_err, n_warn = _parse_starter_out(draw_dir / f"{run_name}_0000.out")
    res.update(starter_rc=rc0, n_err=n_err, n_warn=n_warn)
    if rc0 != 0 or n_err != 0:
        res["ok"] = False
        return res

    rc1 = _run_exe(ENGINE_EXE, draw_dir / f"{run_name}_0001.rad", draw_dir, env,
                    timeout=7200)
    normal_term, rows = _parse_engine_out(draw_dir / f"{run_name}_0001.out")
    res.update(engine_rc=rc1, normal_term=normal_term, n_rows=len(rows),
               wall_s=round(time.time() - t0, 1))
    if not rows:
        res["ok"] = False
        return res

    final = rows[-1]
    res.update(final_time=final["time"],
               ke_ie=final["ke_t"] / final["i_energy"] if final["i_energy"] > 0
                     else float("nan"))
    res["reaction"] = reaction_history(rows, v, t_ramp)
    res["radial"] = frame_radial_profile(draw_dir, run_name,
                                          mesh.t * LENGTH_SCALE, tmp_dir)
    res["ok"] = bool(normal_term and final["time"] >= 0.9 * TEND)

    for pat in (f"{run_name}_0000_0001.rst", f"{run_name}_0001_0001.rst"):
        (draw_dir / pat).unlink(missing_ok=True)
    tmp_dir.rmdir()
    (draw_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


def post_arm(root, label, frac, t_ramp):
    """Re-derive one arm's diagnostics from the ANIM frames and .out log that
    a previous solve already left on disk, without re-running Radioss."""
    mesh = geometry.ShellMesh(r_over_t=R_OVER_T, l_over_r=L_OVER_R)
    _, record = draw_imperfection(mesh, SEED)
    d_cr = mesh.end_shortening_cr * LENGTH_SCALE
    v = frac * d_cr / (TEND - 0.5 * t_ramp)

    draw_dir = root / label
    tmp_dir = draw_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    normal_term, rows = _parse_engine_out(draw_dir / "draw_0001.out")
    final = rows[-1]
    res = dict(label=label, frac=frac, t_ramp=t_ramp, seed=SEED,
               r_over_t=R_OVER_T, l_over_r=L_OVER_R, K=record["K"],
               imperf_max_over_t=record["max_abs_over_t"],
               R=mesh.R * LENGTH_SCALE, L=mesh.L * LENGTH_SCALE,
               t=mesh.t * LENGTH_SCALE, d_cr=d_cr, v=v,
               eps_cr=float(mesh.sigma_cr_classical), n_elem=len(mesh.elements),
               normal_term=normal_term, n_rows=len(rows),
               final_time=final["time"],
               ke_ie=final["ke_t"] / final["i_energy"] if final["i_energy"] > 0
                     else float("nan"))
    res["reaction"] = reaction_history(rows, v, t_ramp)
    res["radial"] = frame_radial_profile(draw_dir, "draw",
                                          mesh.t * LENGTH_SCALE, tmp_dir)
    res["ok"] = bool(normal_term and final["time"] >= 0.9 * TEND)
    tmp_dir.rmdir()
    (draw_dir / "result.json").write_text(json.dumps(res, indent=1))
    return res


def main(root, post_only=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    worker = post_arm if post_only else run_arm
    with ThreadPoolExecutor(max_workers=len(ARMS)) as ex:
        futs = {ex.submit(worker, root, *a): a[0] for a in ARMS}
        for fut in as_completed(futs):
            label = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                print(f"{label}: EXCEPTION {exc!r}", flush=True)
                continue
            results.append(r)
            rad = r.get("radial") or [{}]
            print(f"{label}: ok={r.get('ok')} frac={r['frac']} "
                  f"wall={r.get('wall_s')}s KE/IE={r.get('ke_ie'):.4g} "
                  f"final_max|dr|/t={rad[-1].get('max_abs', float('nan')):.4g}",
                  flush=True)
    results.sort(key=lambda r: (r["frac"], r["t_ramp"]))
    (root / "verify_summary.json").write_text(json.dumps(results, indent=1))

    print("\n=== load sweep, R/t=%.1f L/R=%.3f seed=%d ===" % (R_OVER_T, L_OVER_R, SEED))
    print(f"{'arm':16s} {'frac':>6s} {'ramp':>5s} {'ok':>3s} {'KE/IE':>9s} "
          f"{'max|dr|/t':>10s} {'peakF':>10s} {'F@end':>10s} {'F drop':>8s}")
    for r in results:
        rad = r.get("radial") or [{}]
        rx = r.get("reaction") or []
        f_series = [q["force"] for q in rx if q["time"] > r["t_ramp"]]
        peak = max(f_series) if f_series else float("nan")
        fend = f_series[-1] if f_series else float("nan")
        drop = (1.0 - fend / peak) if f_series and peak > 0 else float("nan")
        print(f"{r['label']:16s} {r['frac']:6.4f} {r['t_ramp']:5.2f} "
              f"{str(r.get('ok')):>3s} {r.get('ke_ie', float('nan')):9.4g} "
              f"{rad[-1].get('max_abs', float('nan')):10.4g} "
              f"{peak:10.4g} {fend:10.4g} {drop:8.2%}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--post"]
    main(argv[0] if argv else "../runs/load_verify",
         post_only="--post" in sys.argv)
