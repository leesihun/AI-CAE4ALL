"""Diagnostic: is the recorded terminal field a converged equilibrium, or a
snapshot mid-oscillation? Writes a variant of the two-step solve with dense
displacement output through step 3, tracks dominant-n and rms/t and F/Flin
over time, and reports whether the SHAPE is stable while only the amplitude
rings (recoverable by more damping) or whether the pattern itself is still
selecting (a much worse problem).
"""
import sys, os, re, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from imperfection import apply_imperfection
from ccx import _nodes_block, _elems_block, _nset, E_MOD, NU, run_ccx, CONTROLS
from analyse import radial, mode_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work", "diag")
os.makedirs(W, exist_ok=True)


def write_diag(path, mesh, coords, target_shortening, handover=0.70,
              n_inc_static=10, n_inc_dyn=80, t_dyn=8.0,
              n_inc_settle=200, t_settle=40.0, damp_alpha=0.3, hht_alpha=-0.3,
              rn=1.0e-5, cn=1.0e-7, rho=1.0):
    d = target_shortening
    dcr = mesh.end_shortening_cr
    body = [
        "** diagnostic: dense output through a long settle",
        _nodes_block(coords), _elems_block(mesh.elements),
        _nset("NBOT", mesh.bottom_nodes), _nset("NTOP", mesh.top_nodes),
        "*MATERIAL, NAME=STEEL", "*ELASTIC", "%.13e, %.13e" % (E_MOD, NU),
        "*DENSITY", "%.13e" % rho,
        "*DAMPING, ALPHA=%.13e" % damp_alpha,
        "*SHELL SECTION, ELSET=EALL, MATERIAL=STEEL", "%.13e" % mesh.t,
        "*BOUNDARY", "NBOT, 1, 6, 0.0", "NTOP, 1, 2, 0.0", "NTOP, 4, 6, 0.0",
        "*STEP, NLGEOM, INC=%d" % (n_inc_static * 20), "*STATIC",
        "%.6e, 1.0, 1.0e-9, %.6e" % (1.0 / n_inc_static, 2.0 / n_inc_static),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY", "NTOP, 3, 3, %.13e" % (-handover * dcr),
        "*NODE FILE, OUTPUT=2D, FREQUENCY=%d" % (n_inc_static * 20), "U",
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=2", "RF",
        "*EL FILE", "ENER",
        "*END STEP",
        "*STEP, NLGEOM, INC=%d" % (n_inc_dyn * 40),
        "*DYNAMIC, ALPHA=%.4f" % hht_alpha,
        "%.13e, %.13e, %.13e, %.13e" % (t_dyn / n_inc_dyn, t_dyn, 1.0e-8 * t_dyn, t_dyn / n_inc_dyn),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY", "NTOP, 3, 3, %.13e" % (-d),
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=2", "RF",
        "*END STEP",
        # dense field output through the settle -- this is what we need
        "*STEP, NLGEOM, INC=%d" % (n_inc_settle * 40),
        "*DYNAMIC, ALPHA=%.4f" % hht_alpha,
        "%.13e, %.13e, %.13e, %.13e" % (t_settle / n_inc_settle, t_settle, 1.0e-8 * t_settle, t_settle / n_inc_settle),
        "*CONTROLS, PARAMETERS=FIELD",
        "%.6e, %.6e, , , %.6e, , , 1.000000e-12" % (rn, cn, rn),
        "*BOUNDARY", "NTOP, 3, 3, %.13e" % (-d),
        "*NODE FILE, OUTPUT=2D, FREQUENCY=%d" % max(1, n_inc_settle // 40), "U",
        "*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=2", "RF",
        "*END STEP", "",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(body))


def read_all_frd_blocks(frd_path):
    """Every DISP block in order (not just the last)."""
    blocks = []
    cur, in_disp = None, False
    with open(frd_path, "r", errors="replace") as f:
        for line in f:
            if line.startswith(" -4") and "DISP" in line:
                cur, in_disp = {}, True
                continue
            if in_disp:
                if line.startswith(" -3"):
                    blocks.append(cur); cur, in_disp = None, False; continue
                if line.startswith(" -1"):
                    try:
                        nid = int(line[3:13])
                        vals = [float(line[13 + 12 * k: 25 + 12 * k]) for k in range(3)]
                        cur[nid] = vals
                    except ValueError:
                        pass
    out = []
    for b in blocks:
        n = max(b)
        arr = np.zeros((n, 3))
        for k, v in b.items():
            arr[k - 1] = v
        out.append(arr)
    return out


def forces(dat_path):
    txt = open(dat_path, errors="replace").read()
    out = []
    for m in re.finditer(r"time\s+([0-9.E+-]+)\s*\n\s*\n\s*([-0-9.E+ ]+)", txt):
        try:
            out.append((float(m.group(1)), abs(float(m.group(2).split()[2]))))
        except (ValueError, IndexError):
            pass
    return np.asarray(out)


def run_one(rt, lr, seed=1, sigma=1e-3, comp_a=0.01, t_settle=40.0, n_inc_settle=200,
           damp_alpha=0.3):
    m = ShellMesh(rt, lr, elems_per_half_wave=3.0)
    coords, _, _ = apply_imperfection(m, sigma_hat=sigma, a_bar=0.70, nu=1.5,
                                      n_panels=8, seed=seed, comp_a_rms_over_t=comp_a,
                                      with_component_a=(comp_a > 0))
    key = "diag_rt%d_lr%03d_a%g" % (rt, round(lr * 100), damp_alpha)
    d = os.path.join(W, key)
    os.makedirs(d, exist_ok=True)
    inp = os.path.join(d, key + ".inp")
    write_diag(inp, m, coords, 1.5 * m.end_shortening_cr,
              t_settle=t_settle, n_inc_settle=n_inc_settle, damp_alpha=damp_alpha)
    ok, tail, secs = run_ccx(inp, timeout=7200)
    print("  ok=%s  %.0fs" % (ok, secs))
    if not ok:
        print(tail[-1500:])
        return None

    A = 2 * np.pi * m.R * m.t
    Flin = (1.5 * m.end_shortening_cr / m.L) * A
    f = forces(os.path.join(d, key + ".dat"))
    blocks = read_all_frd_blocks(os.path.join(d, key + ".frd"))
    print("  %d displacement snapshots recorded" % len(blocks))

    hist = []
    for b in blocks:
        if b.shape[0] < m.n_nodes:
            continue
        w = radial(m, b) / m.t
        ml = mode_label(m, w)
        hist.append((ml["n"], ml["share"], float(np.sqrt(np.mean(w ** 2)))))
    ns = [h[0] for h in hist]
    rms = [h[2] for h in hist]
    print("  n_dominant sequence (last 15): %s" % ns[-15:])
    print("  rms/t sequence (last 15): %s" % [round(x, 4) for x in rms[-15:]])
    if len(rms) >= 10:
        q = len(rms) // 4
        print("  rms/t at 0/25/50/75/100%%: %.4f %.4f %.4f %.4f %.4f"
              % (rms[0], rms[q], rms[2 * q], rms[3 * q], rms[-1]))
        last_third = rms[-len(rms) // 3:]
        print("  rms/t last-third range: %.4f (max-min)/mean=%.1f%%"
              % (max(last_third) - min(last_third),
                 100 * (max(last_third) - min(last_third)) / (np.mean(last_third) + 1e-12)))
    if len(f):
        tail_f = f[-len(f) // 3:, 1] / Flin
        print("  F/Flin last-third range: %.3f (max-min)/mean=%.1f%%"
              % (tail_f.max() - tail_f.min(), 100 * (tail_f.max() - tail_f.min()) / tail_f.mean()))
    return dict(ns=ns, hist=hist, f=f, Flin=Flin)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", type=int, default=110)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t-settle", type=float, default=40.0)
    ap.add_argument("--n-inc-settle", type=int, default=200)
    ap.add_argument("--damp-alpha", type=float, default=0.3)
    a = ap.parse_args()
    print("=== R/t=%d L/R=%.1f seed=%d  t_settle=%.0f n_inc=%d damp=%.2f ==="
          % (a.rt, a.lr, a.seed, a.t_settle, a.n_inc_settle, a.damp_alpha))
    run_one(a.rt, a.lr, a.seed, t_settle=a.t_settle, n_inc_settle=a.n_inc_settle,
           damp_alpha=a.damp_alpha)
