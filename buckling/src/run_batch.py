"""Batch driver: geometry + N imperfection draws -> OpenRadioss Starter/Engine
runs, throttled to a worker cap, with immediate process-and-purge.

Process-and-purge is not optional at this dataset's scale (1628 draws at
production size, per sampling.py). Two restart files exist per draw and
neither is needed afterwards -- `<run>_0000_0001.rst` (the Starter's
initial-state dump the Engine reads to start, ~62MB on the ~91k-node nominal
geometry) and `<run>_0001_0001.rst` (the Engine's own end-of-run restart,
unneeded since no draw is ever resumed, ~58MB) -- plus the ASCII `.rad`
decks themselves (~16MB each, fully regeneratable from the geometry params +
imperfection seed). Left in place, that is ~136MB/draw of pure waste on top
of the ~8.45MB actually needed (2 ANIM frames + .out logs + summary.json,
confirmed via the 8-draw smoke pilot). Every draw is purged down to that
~8.45MB right after it finishes, not batched at the end of the whole
campaign -- at 1628 draws the real total is ~13.4GB, comfortably within
reach even on aarl's /data mount (390GB free; its root filesystem is
separately near-full and must not be used).

Runs on both Windows (local) and Linux (aarl) -- RADIOSS_ROOT and the
exe/env differences (PATH+.exe vs LD_LIBRARY_PATH+no-extension) are picked
by platform below; override RADIOSS_ROOT via the env var of the same name
if a install lives somewhere else.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import geometry
import imperfection
import radioss

_IS_WINDOWS = os.name == "nt"
RADIOSS_ROOT = Path(os.environ.get(
    "RADIOSS_ROOT",
    "C:/tmp/OpenRadioss/win64/OpenRadioss" if _IS_WINDOWS
    else "/opt/OpenRadioss/OpenRadioss",
))
STARTER_EXE = "starter_win64.exe" if _IS_WINDOWS else "starter_linux64_gf"
ENGINE_EXE = "engine_win64.exe" if _IS_WINDOWS else "engine_linux64_gf"

_ENERGY_ROW_RE = re.compile(
    r"^\s*(\d+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+(\S+)\s+(\d+)\s+([\d.+-]+)%"
    r"\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)"
    r"\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s*$"
)


def _radioss_env():
    rr = RADIOSS_ROOT
    env = dict(os.environ)
    if _IS_WINDOWS:
        env["PATH"] = os.pathsep.join([
            str(rr / "extlib/intelOneAPI_runtime/win64"),
            str(rr / "extlib/hm_reader/win64"),
            str(rr / "extlib/h3d/lib/win64"),
            env.get("PATH", ""),
        ])
    else:
        env["LD_LIBRARY_PATH"] = os.pathsep.join([
            str(rr / "extlib/hm_reader/linux64"),
            str(rr / "extlib/h3d/lib/linux64"),
            env.get("LD_LIBRARY_PATH", ""),
        ])
    env["RAD_CFG_PATH"] = str(rr / "hm_cfg_files")
    return env


def _run_exe(exe, deck_path, cwd, env, timeout=1200):
    result = subprocess.run(
        [str(RADIOSS_ROOT / "exec" / exe), "-i", deck_path.name],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode


def _parse_starter_out(out_path):
    text = out_path.read_text(errors="replace")
    m_err = re.search(r"(\d+)\s+ERROR\(S\)", text)
    m_warn = re.search(r"(\d+)\s+WARNING\(S\)", text)
    n_err = int(m_err.group(1)) if m_err else -1
    n_warn = int(m_warn.group(1)) if m_warn else -1
    return n_err, n_warn


def _parse_engine_out(out_path):
    text = out_path.read_text(errors="replace")
    normal_term = "NORMAL TERMINATION" in text
    rows = []
    for line in text.splitlines():
        m = _ENERGY_ROW_RE.match(line)
        if m:
            rows.append(dict(
                cycle=int(m.group(1)), time=float(m.group(2)),
                i_energy=float(m.group(7)), ke_t=float(m.group(8)),
                ke_r=float(m.group(9)), ext_work=float(m.group(10)),
            ))
    return normal_term, rows


def _purge(draw_dir, run_name):
    for pat in (f"{run_name}_0000_0001.rst", f"{run_name}_0001_0001.rst",
                f"{run_name}_0000.rad", f"{run_name}_0001.rad",
                f"{run_name}T01"):
        p = draw_dir / pat
        if p.exists():
            p.unlink()


def run_one_draw(draw_dir, mesh, seed, E, nu, rho, length_scale, tend,
                  run_name="draw"):
    """One full geometry+imperfection realization: draw -> decks -> Starter ->
    Engine -> parse -> purge. Returns a JSON-able summary dict."""
    draw_dir.mkdir(parents=True, exist_ok=True)
    env = _radioss_env()
    t0 = time.time()

    max_rsa_attempts = 50
    for attempt in range(max_rsa_attempts):
        rng = np.random.default_rng((seed, attempt))
        try:
            coords, record = imperfection.apply_imperfection(mesh, rng)
            break
        except RuntimeError:
            continue
    else:
        return dict(seed=seed, ok=False, stage="imperfection",
                    n_rsa_attempts=max_rsa_attempts)

    d_cr_actual = mesh.end_shortening_cr * length_scale
    # main.tex Sec 2.3 specifies 1.5 * d_cr of end shortening. The ramp costs
    # half its own duration in delivered displacement (FUNCT ramps 0 -> 1 over
    # t_ramp), so divide by the effective time, not by tend. The old
    # `0.1 * d_cr / tend` delivered 0.0975 d_cr -- below even the severest
    # imperfection knockdown, so nothing in that dataset ever buckled.
    t_ramp = 0.05
    v = 1.5 * d_cr_actual / (tend - 0.5 * t_ramp)

    ids = radioss.DeckIds(mesh)
    starter_path = draw_dir / f"{run_name}_0000.rad"
    radioss.write_starter(
        starter_path, mesh, coords, ids, E=E, nu=nu, rho=rho,
        thickness=mesh.t, length_scale=length_scale, impose_velocity=v,
        t_ramp=t_ramp, title=f"draw seed={seed}",
    )
    radioss.write_engine(draw_dir / f"{run_name}_0001.rad", run_name, 1, tend=tend)

    diag = dict(K=record["K"], max_abs_over_t=record["max_abs_over_t"])

    try:
        rc0 = _run_exe(STARTER_EXE, starter_path, draw_dir, env)
    except subprocess.TimeoutExpired:
        return dict(seed=seed, ok=False, stage="starter", timeout=True, **diag)
    n_err, n_warn = _parse_starter_out(draw_dir / f"{run_name}_0000.out")
    if rc0 != 0 or n_err != 0:
        return dict(seed=seed, ok=False, stage="starter",
                    returncode=rc0, n_err=n_err, n_warn=n_warn, **diag)

    try:
        rc1 = _run_exe(ENGINE_EXE, draw_dir / f"{run_name}_0001.rad", draw_dir, env)
    except subprocess.TimeoutExpired:
        # Observed failure mode: energies go NaN, the auto time step collapses
        # to a near-zero floor (~1e-14), and the run grinds cycles forever
        # with no time progress instead of aborting -- this is the only way
        # to bound that case's wall time.
        return dict(seed=seed, ok=False, stage="engine", timeout=True, **diag)
    normal_term, rows = _parse_engine_out(draw_dir / f"{run_name}_0001.out")
    if rc1 != 0 or not normal_term or not rows:
        return dict(seed=seed, ok=False, stage="engine",
                    returncode=rc1, normal_term=normal_term, **diag)

    final = rows[-1]
    if final["time"] < 0.9 * tend:
        # Observed failure mode: energy blows up to a nonphysical magnitude
        # within a handful of cycles and the engine exits claiming normal
        # termination anyway -- rc1==0 and normal_term==True don't catch
        # this, only checking how much of tend was actually reached does.
        return dict(seed=seed, ok=False, stage="engine",
                    reason="early_termination", final_time=final["time"], **diag)
    ke_ie = final["ke_t"] / final["i_energy"] if final["i_energy"] > 0 else float("nan")

    summary = dict(
        seed=seed, ok=True, wall_s=time.time() - t0,
        K=record["K"], sigma=record["sigma"],
        max_abs_over_t=record["max_abs_over_t"], rms_over_t=record["rms_over_t"],
        final_time=final["time"], final_i_energy=final["i_energy"],
        final_ke_t=final["ke_t"], final_ext_work=final["ext_work"],
        ke_ie_ratio=ke_ie, n_energy_rows=len(rows),
    )
    (draw_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _purge(draw_dir, run_name)
    return summary


def run_pilot(run_root, r_over_t, l_over_r, n_draws, seeds=None,
              max_workers=None, E=210.0, nu=0.3, rho=7.85e-6,
              length_scale=100.0, tend=1.0):
    """One geometry, N independent imperfection draws, throttled concurrency."""
    if seeds is None:
        seeds = list(range(n_draws))
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) // 2)

    mesh = geometry.ShellMesh(r_over_t=r_over_t, l_over_r=l_over_r)
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(run_one_draw, run_root / f"draw_{i:03d}", mesh, seed,
                      E, nu, rho, length_scale, tend): (i, seed)
            for i, seed in enumerate(seeds)
        }
        for fut in as_completed(futs):
            i, seed = futs[fut]
            r = fut.result()
            print(f"draw {i} (seed={seed}): "
                  + (f"ok, K={r['K']}, KE/IE={r['ke_ie_ratio']:.4f}" if r["ok"]
                     else f"FAILED at {r['stage']}"))
            results.append(r)
    return mesh, results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    mesh, results = run_pilot("smoke_pilot", r_over_t=150.0, l_over_r=2.0, n_draws=n)
    ok = [r for r in results if r["ok"]]
    print(f"\n{len(ok)}/{len(results)} draws OK")
    if ok:
        ratios = [r["ke_ie_ratio"] for r in ok]
        print(f"KE/IE range: {min(ratios):.4f} - {max(ratios):.4f}")
        keys = [(r["K"], round(r["final_i_energy"], 10), round(r["final_ext_work"], 10)) for r in ok]
        print(f"distinct (K, I-ENERGY, EXT-WORK) tuples: {len(set(keys))}/{len(ok)}")
