"""Natural-frequency extraction for each corner of the geometry box.

This settles a question the settle-step design silently assumed away: the solve
uses mass-proportional damping (C = alpha*M) with a SINGLE fixed alpha and a
SINGLE fixed settle duration for every geometry. For mass-proportional damping
the damping ratio of mode i is

    zeta_i = alpha / (2 * omega_i)

so a fixed alpha only gives the same damping behaviour if omega_1 is the same
across geometries. If omega_1 varies, then both the damping ratio AND the number
of oscillation periods contained in a fixed settle window vary with geometry --
which would explain why one corner settles cleanly and another does not.
"""
import argparse, os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh, E_MOD, NU
from ccx import _nodes_block, _elems_block, _nset, run_ccx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work", "freq")


def write_freq_inp(path, mesh, n_modes=12, rho=1.0):
    body = [
        "** natural frequency extraction, undeformed clamped shell",
        _nodes_block(mesh.coords),
        _elems_block(mesh.elements),
        _nset("NBOT", mesh.bottom_nodes),
        _nset("NTOP", mesh.top_nodes),
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        "%.13e, %.13e" % (E_MOD, NU),
        "*DENSITY",
        "%.13e" % rho,
        "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL",
        "%.13e" % mesh.t,
        "*BOUNDARY",
        "NBOT, 1, 6, 0.0",
        "NTOP, 1, 2, 0.0",
        "NTOP, 3, 3, 0.0",
        "NTOP, 4, 6, 0.0",
        "*STEP",
        "*FREQUENCY",
        "%d" % n_modes,
        "*END STEP",
        "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(body))


def read_frequencies(dat_path):
    """CalculiX prints a table of eigenvalue / frequency(rad/time) / frequency(cycles/time)."""
    if not os.path.exists(dat_path):
        return np.array([])
    txt = open(dat_path, errors="replace").read()
    lo = txt.lower()
    key = lo.find("eigenvalue")
    if key < 0:
        return np.array([])
    out = []
    for line in txt[key:].splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                # mode_no, eigenvalue, omega(rad/time), freq(cycles/time)...
                vals = [float(p) for p in parts[1:]]
                out.append(vals)
            except ValueError:
                if out:
                    break
        elif out:
            break
    return np.asarray(out, dtype=float) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--damp", type=float, default=0.3,
                    help="the mass-proportional alpha actually used in the solves")
    ap.add_argument("--t-settle", type=float, default=40.0)
    a = ap.parse_args()

    os.makedirs(W, exist_ok=True)
    corners = [(110, 1.0), (110, 1.6), (230, 1.0), (230, 1.6)]
    results = []

    for rt, lr in corners:
        m = ShellMesh(rt, lr, elems_per_half_wave=3.0)
        tag = "freq_rt%d_lr%03d" % (rt, round(lr * 100))
        inp = os.path.join(W, tag + ".inp")
        write_freq_inp(inp, m, n_modes=a.modes)
        ok, tail, secs = run_ccx(inp, timeout=7200)
        tab = read_frequencies(os.path.join(W, tag + ".dat"))
        if not ok or tab.size == 0:
            print("%s FAILED  %s" % (tag, tail[-300:]))
            continue

        # columns: eigenvalue, omega(rad), freq(cycles) -- take omega column
        omega = tab[:, 1] if tab.shape[1] >= 2 else np.sqrt(np.abs(tab[:, 0]))
        w1 = float(omega[0])
        T1 = 2 * np.pi / w1 if w1 > 0 else float("nan")
        zeta1 = a.damp / (2 * w1) if w1 > 0 else float("nan")
        periods = a.t_settle / T1 if T1 > 0 else float("nan")

        r = dict(r_over_t=rt, l_over_r=lr, n_nodes=int(m.n_nodes),
                 Z=round(m.batdorf_Z, 1), t=m.t, L=m.L,
                 omega1=round(w1, 5), T1=round(T1, 4),
                 zeta1=round(zeta1, 4), settle_periods=round(periods, 2),
                 omega_first5=[round(float(x), 4) for x in omega[:5]],
                 seconds=round(secs, 1))
        results.append(r)
        print("R/t=%-4d L/R=%-4.1f  omega1=%8.4f  T1=%8.3f  zeta1=%7.3f  "
              "settle=%6.2f periods  (%.0fs)"
              % (rt, lr, w1, T1, zeta1, periods, secs), flush=True)

    json.dump(results, open(os.path.join(W, "freq_results.json"), "w"), indent=1)

    if len(results) > 1:
        w = [r["omega1"] for r in results]
        z = [r["zeta1"] for r in results]
        p = [r["settle_periods"] for r in results]
        print()
        print("omega1 spread across box : %.4f .. %.4f  (%.2fx)" % (min(w), max(w), max(w) / min(w)))
        print("zeta1  spread across box : %.3f .. %.3f  (%.2fx)" % (min(z), max(z), max(z) / min(z)))
        print("settle periods           : %.2f .. %.2f  (%.2fx)" % (min(p), max(p), max(p) / min(p)))
        print()
        print("zeta1 = 1.0 would be critical damping. alpha for critical damping")
        print("of the fundamental mode, per corner:")
        for r in results:
            print("   R/t=%-4d L/R=%-4.1f -> alpha_crit = 2*omega1 = %.4f"
                  % (r["r_over_t"], r["l_over_r"], 2 * r["omega1"]))


if __name__ == "__main__":
    main()
