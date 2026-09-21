"""Isolate whether train_tight_2's (R/t=158.59, L/R=0.849) NaN divergence
is caused by the imperfection at all, or happens even for the PERFECT
(unperturbed) cylinder -- i.e. is it a base geometry/mesh/BC/loading
problem independent of imperfection.py entirely."""
from __future__ import annotations
import json, time
import geometry, radioss
from run_batch import _radioss_env, _run_exe, _parse_starter_out, _parse_engine_out, _purge, STARTER_EXE, ENGINE_EXE
from pathlib import Path

def run_perfect(name, r_over_t, l_over_r, E=210.0, nu=0.3, rho=7.85e-6,
                 length_scale=100.0, tend=1.0):
    mesh = geometry.ShellMesh(r_over_t=r_over_t, l_over_r=l_over_r)
    draw_dir = Path(name)
    draw_dir.mkdir(parents=True, exist_ok=True)
    env = _radioss_env()
    coords = mesh.coords  # no imperfection at all
    d_cr_actual = mesh.end_shortening_cr * length_scale
    v = 0.1 * d_cr_actual / tend
    ids = radioss.DeckIds(mesh)
    run_name = "draw"
    starter_path = draw_dir / f"{run_name}_0000.rad"
    radioss.write_starter(starter_path, mesh, coords, ids, E=E, nu=nu, rho=rho,
                           thickness=mesh.t, length_scale=length_scale,
                           impose_velocity=v, t_ramp=0.05, title=f"perfect {name}")
    radioss.write_engine(draw_dir / f"{run_name}_0001.rad", run_name, 1, tend=tend)
    t0 = time.time()
    try:
        rc0 = _run_exe(STARTER_EXE, starter_path, draw_dir, env)
    except Exception as e:
        print(f"{name}: STARTER EXC {e!r}", flush=True); return
    n_err, n_warn = _parse_starter_out(draw_dir / f"{run_name}_0000.out")
    print(f"{name}: starter rc={rc0} n_err={n_err} n_warn={n_warn}", flush=True)
    if rc0 != 0 or n_err != 0:
        return
    try:
        rc1 = _run_exe(ENGINE_EXE, draw_dir / f"{run_name}_0001.rad", draw_dir, env)
    except Exception as e:
        print(f"{name}: ENGINE TIMEOUT/EXC after {time.time()-t0:.1f}s {e!r}", flush=True); return
    normal_term, rows = _parse_engine_out(draw_dir / f"{run_name}_0001.out")
    final = rows[-1] if rows else None
    print(f"{name}: engine rc={rc1} normal_term={normal_term} n_rows={len(rows)} "
          f"final={final} wall_s={time.time()-t0:.1f}", flush=True)

if __name__ == "__main__":
    run_perfect("perfect_train_tight_2", 158.59, 0.849)
    run_perfect("perfect_train_tight_1", 120.64, 0.959)  # known-good control
