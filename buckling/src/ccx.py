"""CalculiX .inp writer and runner.

Two settings here are load-bearing, not defaults (docs/BENCHMARK_DESIGN.md s.5):

  * Coordinates at %.13e (14 significant digits; CalculiX caps a
    free-format field at 20 characters, so %.15e silently corrupts the deck).
    A 6-significant-digit mesh file injects ~3e-4*t of
    symmetry-breaking noise, which is ABOVE the physical imperfection signal.

  * *CONTROLS with Rn=1e-6, Cn=1e-8. CalculiX defaults (Rn=0.005, Cn=0.01) sit
    ~4 orders above the signal; left at default every draw returns the same
    mode and the dataset is silently deterministic.

CalculiX 2.22 has NO Riks / arc-length method, so continuation is displacement
control on the end shortening. That is the correct substitute: with the
imperfection present the bifurcation is unfolded into a smooth limit-point
path, and dP/du = 0 is a regular point of the displacement-controlled system.
"""
from __future__ import annotations

import os
import re
import subprocess
import numpy as np

from geometry import E_MOD, NU

CCX = os.environ.get("CCX_BIN", "$HOME/bin/ccx")
LDPATH = os.environ.get("CCX_LD", "$HOME/lib")


def _win_to_wsl(p: str) -> str:
    p = os.path.abspath(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def _nodes_block(coords):
    out = ["*NODE, NSET=NALL"]
    for i, (x, y, z) in enumerate(coords, start=1):
        out.append("%d, %.13e, %.13e, %.13e" % (i, x, y, z))
    return "\n".join(out)


def _elems_block(elements):
    out = ["*ELEMENT, TYPE=S8R, ELSET=EALL"]
    for e, nd in enumerate(elements, start=1):
        out.append("%d, " % e + ", ".join(str(int(n) + 1) for n in nd))
    return "\n".join(out)


def _nset(name, ids):
    lines = ["*NSET, NSET=%s" % name]
    ids = [int(i) + 1 for i in ids]
    for k in range(0, len(ids), 12):
        lines.append(", ".join(str(i) for i in ids[k:k + 12]) + ("," if k + 12 < len(ids) else ""))
    return "\n".join(lines)


CONTROLS = """*CONTROLS, PARAMETERS=FIELD
1.000000e-06, 1.000000e-08, , , 1.000000e-06, , , 1.000000e-12
"""


def write_buckle_inp(path, mesh, coords, n_eig=60):
    """Linear eigenvalue buckling of the (possibly perfect) shell -- gate G1."""
    body = [
        "** G1: perfect-shell eigenvalue census",
        _nodes_block(coords),
        _elems_block(mesh.elements),
        _nset("NBOT", mesh.bottom_nodes),
        _nset("NTOP", mesh.top_nodes),
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        "%.13e, %.13e" % (E_MOD, NU),
        "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL",
        "%.13e" % mesh.t,
        "*BOUNDARY",
        "NBOT, 1, 6, 0.0",
        "NTOP, 1, 2, 0.0",
        "NTOP, 4, 6, 0.0",
        "*STEP",
        "*BUCKLE",
        "%d" % n_eig,
        "*BOUNDARY",
        "NTOP, 3, 3, %.13e" % (-mesh.end_shortening_cr),
        "*NODE FILE",
        "U",
        "*END STEP",
        "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(body))


def write_static_inp(path, mesh, coords, target_shortening, n_inc=200):
    """Nonlinear displacement-controlled compression past the limit point."""
    d = target_shortening
    body = [
        "** displacement-controlled post-buckling",
        _nodes_block(coords),
        _elems_block(mesh.elements),
        _nset("NBOT", mesh.bottom_nodes),
        _nset("NTOP", mesh.top_nodes),
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        "%.13e, %.13e" % (E_MOD, NU),
        "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL",
        "%.13e" % mesh.t,
        "*BOUNDARY",
        "NBOT, 1, 6, 0.0",
        "NTOP, 1, 2, 0.0",
        "NTOP, 4, 6, 0.0",
        "*STEP, NLGEOM, INC=%d" % (n_inc * 20),
        "*STATIC",
        "%.6e, 1.0, 1.0e-9, %.6e" % (1.0 / n_inc, 4.0 / n_inc),
        CONTROLS.rstrip(),
        "*BOUNDARY",
        "NTOP, 3, 3, %.13e" % (-d),
        "*NODE FILE, FREQUENCY=%d" % max(1, n_inc // 4),
        "U",
        "*EL FILE, FREQUENCY=%d" % max(1, n_inc // 4),
        "S",
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY",
        "RF",
        "*END STEP",
        "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(body))


def write_dynamic_inp(path, mesh, coords, target_shortening, t_ramp=20.0,
                      n_inc=400, rho=1.0, damp_alpha=8.0, hht_alpha=-0.3,
                      rn=1.0e-5, cn=1.0e-7, direct=True, dt0_frac=None,
                      dt_max_frac=None):
    """Quasi-static implicit dynamics -- the route through the snap.

    CalculiX has no arc-length method and pure displacement control diverges at
    the limit point ("too many cutbacks"), because the shell releases energy
    faster than a static solve can follow. Implicit dynamics regularises that:
    inertia absorbs the snap and mass-proportional damping bleeds the kinetic
    energy off so the terminal state is a genuine static equilibrium.

    HHT alpha adds numerical dissipation of the high-frequency content that the
    snap excites; alpha=-0.3 is close to the maximum CalculiX allows (-1/3).
    """
    d = target_shortening
    dt = t_ramp / n_inc
    body = [
        "** quasi-static implicit dynamics through the limit point",
        _nodes_block(coords),
        _elems_block(mesh.elements),
        _nset("NBOT", mesh.bottom_nodes),
        _nset("NTOP", mesh.top_nodes),
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        "%.13e, %.13e" % (E_MOD, NU),
        "*DENSITY",
        "%.13e" % rho,
        "*DAMPING, ALPHA=%.13e" % damp_alpha,
        "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL",
        "%.13e" % mesh.t,
        "*BOUNDARY",
        "NBOT, 1, 6, 0.0",
        "NTOP, 1, 2, 0.0",
        "NTOP, 4, 6, 0.0",
        "*AMPLITUDE, NAME=RAMP",
        "0.0, 0.0, %.13e, 1.0" % t_ramp,
        "*STEP, NLGEOM, INC=%d" % (n_inc * 40),
        ("*DYNAMIC, DIRECT, ALPHA=%.4f" % hht_alpha) if direct
        else ("*DYNAMIC, ALPHA=%.4f" % hht_alpha),
        ("%.13e, %.13e" % (dt, t_ramp)) if direct
        else ("%.13e, %.13e, %.13e, %.13e" % (
            (dt0_frac or 0.02) * t_ramp, t_ramp,
            1.0e-8 * t_ramp, (dt_max_frac or 0.02) * t_ramp)),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY, AMPLITUDE=RAMP",
        "NTOP, 3, 3, %.13e" % (-d),
        "*NODE FILE, FREQUENCY=%d" % n_inc,
        "U",
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY",
        "RF",
        "*END STEP",
        "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(body))


def write_twostep_inp(path, mesh, coords, target_shortening, handover=0.70,
                      n_inc_static=10, n_inc_dyn=80, t_dyn=8.0, rho=1.0,
                      n_inc_settle=20, t_settle=4.0, damp_alpha=0.3, hht_alpha=-0.3, rn=1.0e-5, cn=1.0e-7):
    """Static pre-buckling, then implicit dynamics through the snap.

    The pre-buckling path is essentially linear and a static solve walks it in
    a handful of increments; it only fails AT the limit point. Dynamics is
    ~10x more expensive per increment, so spending it on the linear phase is
    pure waste -- measured, the fixed-increment dynamic run burned 129 of its
    increments before reaching 42% of the ramp, all of it pre-buckling.

    `handover` is the fraction of the classical critical shortening at which we
    switch. It must sit BELOW the imperfect shell's limit point: Koiter gives
    gamma = 0.855 at sigma_hat=1e-2 and 0.951 at 1e-3, so 0.70 is safe for both.

    Boundary values in CalculiX are total, and ramp linearly across a step, so
    step 2 continues from wherever step 1 ended without an *AMPLITUDE.
    """
    d = target_shortening
    dcr = mesh.end_shortening_cr
    body = [
        "** step 1 static pre-buckling, step 2 implicit dynamics through the snap",
        _nodes_block(coords),
        _elems_block(mesh.elements),
        _nset("NBOT", mesh.bottom_nodes),
        _nset("NTOP", mesh.top_nodes),
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        "%.13e, %.13e" % (E_MOD, NU),
        "*DENSITY",
        "%.13e" % rho,
        "*DAMPING, ALPHA=%.13e" % damp_alpha,
        "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL",
        "%.13e" % mesh.t,
        "*BOUNDARY",
        "NBOT, 1, 6, 0.0",
        "NTOP, 1, 2, 0.0",
        "NTOP, 4, 6, 0.0",
        # ---- step 1: cheap static walk up to the handover
        "*STEP, NLGEOM, INC=%d" % (n_inc_static * 20),
        "*STATIC",
        "%.6e, 1.0, 1.0e-9, %.6e" % (1.0 / n_inc_static, 2.0 / n_inc_static),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY",
        "NTOP, 3, 3, %.13e" % (-handover * dcr),
        # CalculiX requires output requests to appear in the FIRST step of a
        # nonlinear calculation, even though we only want the step-2 result.
        "*NODE FILE, OUTPUT=2D, FREQUENCY=%d" % (n_inc_static * 20),
        "U",
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY",
        "RF",
        # A *DYNAMIC step implicitly requests energy output, and CalculiX
        # refuses that in a nonlinear calculation unless the FIRST step also
        # requests it. Without this line step 2 aborts with
        # "energy output requests, if any, must be specified in the first step".
        "*EL FILE",
        "ENER",
        "*END STEP",
        # ---- step 2: dynamics through the limit point
        "*STEP, NLGEOM, INC=%d" % (n_inc_dyn * 40),
        "*DYNAMIC, ALPHA=%.4f" % hht_alpha,
        "%.13e, %.13e, %.13e, %.13e" % (t_dyn / n_inc_dyn, t_dyn,
                                        1.0e-8 * t_dyn, t_dyn / n_inc_dyn),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY",
        "NTOP, 3, 3, %.13e" % (-d),
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=2",
        "RF",
        "*END STEP",
        # ---- step 3: hold the displacement, damped dynamics, to bleed off the
        # kinetic energy of the snap.
        #
        # A *STATIC* settle was tried and is WRONG here: a deep post-buckled
        # cylinder state is not an isolated stable equilibrium, so the static
        # solve wanders off (measured rms/t of 8.0 and 12.7 against a physical
        # 0.45). Damped dynamics is the only thing that holds the branch.
        #
        # Note F_end/F_linear is NOT a converged quantity -- it is a sensitive
        # scalar that still reflects residual ringing. It is used only as a
        # buckled / not-buckled indicator. The QoI is the FIELD, whose rms
        # converges to 2.7% over n_inc_dyn = 20 -> 80.
        "*STEP, NLGEOM, INC=%d" % (n_inc_settle * 40),
        "*DYNAMIC, ALPHA=%.4f" % hht_alpha,
        "%.13e, %.13e, %.13e, %.13e" % (t_settle / n_inc_settle, t_settle,
                                        1.0e-8 * t_settle,
                                        t_settle / n_inc_settle),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY",
        "NTOP, 3, 3, %.13e" % (-d),
        "*NODE FILE, OUTPUT=2D, FREQUENCY=%d" % (n_inc_settle * 40),
        "U",
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=2",
        "RF",
        "*END STEP",
        "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(body))


def run_ccx(inp_path, timeout=3600, nthreads=1, scratch=True):
    """Run ccx under WSL. Returns (ok, stdout_tail, seconds).

    `scratch=True` copies the deck into the WSL-native filesystem, solves there
    and copies the results back. This is not a micro-optimisation: /mnt/c is the
    Windows-filesystem bridge, and CalculiX does a great many small reads and
    writes. Measured, a 200 MB bulk write is 5.8x slower on /mnt/c (3.67 s vs
    0.63 s), and with 24 concurrent jobs the small-I/O penalty was far worse
    still -- 82 s per increment against ~4 s solo, a 20x slowdown that looked
    like CPU contention but was not.
    """
    import time
    job = os.path.splitext(os.path.abspath(inp_path))[0]
    wsl_job = _win_to_wsl(job)
    wsl_dir = os.path.dirname(wsl_job)
    name = os.path.basename(wsl_job)
    if scratch:
        sd = "$HOME/ccxrun/%s" % name
        inner = ("rm -rf %s && mkdir -p %s && cp '%s.inp' %s/ && cd %s && "
                 "%s -i %s; st=$?; "
                 "for e in frd dat sta; do [ -f %s.$e ] && cp %s.$e '%s/'; done; "
                 "cd / && rm -rf %s; exit $st"
                 % (sd, sd, wsl_job, sd, sd, CCX, name, name, name, wsl_dir, sd))
    else:
        inner = "cd '%s' && %s -i '%s'" % (wsl_dir, CCX, name)
    cmd = ("export LD_LIBRARY_PATH=%s; export OMP_NUM_THREADS=%d; %s"
           % (LDPATH, nthreads, inner))
    t0 = time.time()
    try:
        p = subprocess.run(["wsl", "-e", "bash", "-lc", cmd],
                           capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
        ok = "Job finished" in out
    except subprocess.TimeoutExpired:
        out, ok = "TIMEOUT", False
    return ok, out[-3000:], time.time() - t0


# ------------------------------------------------------------------- readers
def read_buckle_eigenvalues(dat_path):
    """Parse the BUCKLING FACTOR block out of a .dat file."""
    if not os.path.exists(dat_path):
        return np.array([])
    txt = open(dat_path, "r", errors="replace").read()
    m = re.search(r"B\s*U\s*C\s*K\s*L\s*I\s*N\s*G\s+F\s*A\s*C\s*T\s*O\s*R", txt)
    if not m:
        m = re.search(r"BUCKLING FACTOR", txt)
    if not m:
        return np.array([])
    vals = []
    for line in txt[m.end():].splitlines():
        s = line.strip()
        if not s:
            if vals:
                break
            continue
        parts = s.split()
        if len(parts) == 2:
            try:
                vals.append(float(parts[1]))
            except ValueError:
                if vals:
                    break
        elif vals:
            break
    return np.asarray(vals, dtype=float)


def read_frd_displacements(frd_path):
    """Last displacement block from a .frd file -> (n_nodes, 3) float array."""
    if not os.path.exists(frd_path):
        return None
    blocks, cur, in_disp = [], None, False
    with open(frd_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith(" -4") and "DISP" in line:
                cur, in_disp = {}, True
                continue
            if in_disp:
                if line.startswith(" -3"):
                    blocks.append(cur)
                    cur, in_disp = None, False
                    continue
                if line.startswith(" -1"):
                    try:
                        nid = int(line[3:13])
                        vals = [float(line[13 + 12 * k: 25 + 12 * k]) for k in range(3)]
                        cur[nid] = vals
                    except ValueError:
                        pass
    if not blocks:
        return None
    last = blocks[-1]
    n = max(last)
    arr = np.zeros((n, 3))
    for k, v in last.items():
        arr[k - 1] = v
    return arr
