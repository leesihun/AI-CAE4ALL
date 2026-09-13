"""Pick Component A's amplitude against both of its jobs at once.

A is supposed to do exactly two things:

  PIN     make the buckle LOCATION predictable from geometry (design goal F3),
          measured as the circular resultant of the buckle angle across draws,
          tested against the Rayleigh uniform null.

  DEFER   leave the MODE NUMBER to the withheld field, measured as the mode
          histogram staying as wide as the A-off control.

The amplitude was previously tuned against mode counts alone, against a version
of A whose weld harmonic sat on the critical wavenumber. Both halves get
measured here, and the accepted amplitude is the largest one that still defers.
"""
import glob
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from analyse import mode_label
from phase_metric import buckle_phase, circ_stats, rayleigh_p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CV = os.path.join(ROOT, "work", "converged")

# top mode share above this = the distribution has collapsed
MAX_TOP_SHARE = 0.60
MIN_DISTINCT = 3
PIN_ALPHA = 0.05          # Rayleigh p below this = location genuinely pinned


def load(tag, rt=110, lr=1.0):
    out = {}
    pat = os.path.join(CV, "cv%s_rt%d_lr%03d_s*_d0.3" % (tag, rt, round(lr * 100)))
    for d in sorted(glob.glob(pat)):
        f = os.path.join(d, "w.npy")
        if os.path.exists(f):
            s = int(re.search(r"_s(\d+)", os.path.basename(d)).group(1))
            out[s] = np.load(f)
    return out


def summarise(m, ws):
    ns, angs = [], []
    for w in ws:
        n, ph, ang, conc = buckle_phase(m, w)
        ns.append(n)
        angs.append((n, ang))
    cnt = Counter(ns)
    top = max(cnt.values()) / len(ns)

    # location: pool draws by mode, since the angle period depends on n
    best = None
    for n in sorted(cnt):
        a = [x[1] for x in angs if x[0] == n]
        if len(a) < 4:
            continue
        st = circ_stats(a, 2.0 * np.pi / n)
        p = rayleigh_p(st["R"], len(a))
        if best is None or p < best["p"]:
            best = dict(n=n, R=st["R"], p=p, draws=len(a))
    return dict(n_draws=len(ns), hist=dict(sorted(cnt.items())),
                distinct=len(cnt), top_share=top, loc=best)


def main():
    m = ShellMesh(110, 1.0, elems_per_half_wave=3.0)
    sets = [("A off", "N", 0.0), ("0.01 t", "A", 0.01),
            ("0.05 t", "C005", 0.05), ("0.15 t", "C015", 0.15),
            ("0.40 t", "C040", 0.40)]

    rows = []
    print("R/t = 110, L/R = 1.0, converged settle, damping 0.3\n")
    print("%-8s %-6s %-26s %-9s %-8s %s"
          % ("comp A", "draws", "mode histogram", "distinct", "top", "location"))
    print("-" * 92)
    for label, tag, amp in sets:
        ws = load(tag)
        if not ws:
            print("%-8s  (no data)" % label)
            continue
        s = summarise(m, list(ws.values()))
        loc = s["loc"]
        locs = ("n=%d R=%.2f p=%.3f" % (loc["n"], loc["R"], loc["p"])
                if loc else "insufficient")
        print("%-8s %-6d %-26s %-9d %-8.0f%% %s"
              % (label, s["n_draws"], str(s["hist"]), s["distinct"],
                 100 * s["top_share"], locs))
        s["label"], s["amp"] = label, amp
        rows.append(s)

    print()
    print("Acceptance: distinct >= %d AND top share <= %d%% (spread preserved)"
          % (MIN_DISTINCT, int(100 * MAX_TOP_SHARE)))
    print("            AND Rayleigh p < %.2f (location genuinely pinned)" % PIN_ALPHA)
    print()
    ok = []
    for s in rows:
        if s["amp"] == 0.0:
            continue
        defers = (s["distinct"] >= MIN_DISTINCT and s["top_share"] <= MAX_TOP_SHARE)
        pins = bool(s["loc"]) and s["loc"]["p"] < PIN_ALPHA
        verdict = ("ACCEPT" if (defers and pins) else
                   "defers but does not pin" if defers else
                   "pins but collapses the distribution" if pins else
                   "neither")
        print("   %-8s defer=%-5s pin=%-5s  -> %s"
              % (s["label"], defers, pins, verdict))
        if defers and pins:
            ok.append(s)

    print()
    if ok:
        best = max(ok, key=lambda r: r["amp"])
        print("CHOOSE comp_a_rms_over_t = %.2f  (largest amplitude that still defers)"
              % best["amp"])
    else:
        anyd = [s for s in rows if s["amp"] > 0
                and s["distinct"] >= MIN_DISTINCT
                and s["top_share"] <= MAX_TOP_SHARE]
        print("NO amplitude satisfies both.")
        if anyd:
            print("Largest that preserves spread: %s."
                  % max(anyd, key=lambda r: r["amp"])["label"])
        print("Then the honest options are:")
        print("  (a) drop Component A and state that the buckle location is")
        print("      uniform on the circle -- the benchmark is about the mode")
        print("      distribution, not the location;")
        print("  (b) keep the largest spread-preserving amplitude and report")
        print("      that location is only weakly pinned.")
        print("Do NOT keep an amplitude that collapses the mode histogram --")
        print("that is the defect this rebuild exists to remove.")


if __name__ == "__main__":
    main()
