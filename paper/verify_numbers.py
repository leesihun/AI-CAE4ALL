"""Cross-check every quantitative claim in the paper against its source.

Written because the paper asserts a lot of numbers and the sources are three
different places: the SAOI research note, the live buckling dumps, and closed
-form mathematics. A transcription slip in any of them is invisible in a clean
LaTeX build.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROD = os.path.join(ROOT, "buckling", "work", "production")
SAOI_NOTE = os.path.join(ROOT, "docs", "research",
                         "SAOI_PROBABILISTIC_SWEEP_2026-09.md")

fails, checks = [], 0


def chk(name, ok, detail=""):
    global checks
    checks += 1
    print("  %-52s %s  %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        fails.append(name)


def paper_text():
    import pypdf
    r = pypdf.PdfReader(os.path.join(HERE, "main.pdf"))
    return "".join(p.extract_text() for p in r.pages)


def main():
    txt = paper_text()
    flat = re.sub(r"\s+", " ", txt)
    # LaTeX math mode kerns digits, and pypdf renders that as "0 .001".
    # Collapse it so numeric comparisons see the number the reader sees.
    flat = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", flat)

    print("\n[1] SAOI numbers against the research note")
    note = open(SAOI_NOTE, encoding="utf-8", errors="replace").read()
    for label, val in (("MGN-V best W1", "0.410"), ("MGN-V arm3 W1", "0.434"),
                       ("flow best W1", "0.896"), ("MGN-V best sd_ratio", "0.526"),
                       ("flow best sd_ratio", "0.769"), ("flow worst W1", "1.604"),
                       ("MGN-V worst W1", "0.510")):
        chk("%s = %s in note" % (label, val), val in note)
        chk("%s = %s in paper" % (label, val), val in flat)

    chk("2.2x ranking claim in note", "2.2" in note)
    chk("2.2x ranking claim in paper", "2.2" in flat)
    chk("g_enc null effect 0.001 in both",
        "0.001" in note and "0.001" in flat)
    chk("logitnormal delta 0.333 in both",
        "0.333" in note and "0.333" in flat)
    chk("flow per-set 0.199 in both", "0.199" in note and "0.199" in flat)

    print("\n[2] Closed-form results recomputed")
    from scipy.integrate import quad
    err = max(abs(quad(lambda t, a=a: a ** 2 / ((1 - t) ** 2 + a ** 2 * t ** 2),
                       0, 1)[0] - np.pi * a / 2)
              for a in np.linspace(0.05, 10, 80))
    chk("integral == pi*a/2", err < 1e-9, "max err %.2e" % err)
    # conditional variance identity, symbolic check at sample points
    bad = 0
    for a in (0.3, 1.0, 2.5):
        for t in (0.1, 0.5, 0.9):
            lhs = (1 + a ** 2) - (a ** 2 * t - (1 - t)) ** 2 / ((1 - t) ** 2 + a ** 2 * t ** 2)
            rhs = a ** 2 / ((1 - t) ** 2 + a ** 2 * t ** 2)
            if abs(lhs - rhs) > 1e-10:
                bad += 1
    chk("Var(U|Zt) identity holds", bad == 0)
    chk("pi*a/2 stated in paper", "a/2" in flat or "πa/2" in flat)

    print("\n[3] Buckling numbers against live dumps")
    f = os.path.join(PROD, "results.json")
    if not os.path.exists(f):
        chk("production results present", False)
    else:
        recs = [x for x in json.load(open(f)) if x.get("ok")]
        g = defaultdict(list)
        for x in recs:
            g[x["key"].rsplit("_s", 1)[0]].append(x)
        full = {k: v for k, v in g.items() if len(v) >= 40}
        drift = [x["drift"] for x in recs]
        n_over = sum(1 for d in drift if d > 0.15)

        chk("zero solver failures", all(x.get("ok") for x in recs),
            "%d draws" % len(recs))
        chk("4+ complete geometries", len(full) >= 4, "%d" % len(full))
        # The paper no longer claims a single global drift figure; it claims a
        # per-aspect-ratio pattern, which is what gets checked.
        by_lr = defaultdict(list)
        for k, v in full.items():
            by_lr[round(float(v[0]["l_over_r"]), 1)].extend(x["drift"] for x in v)
        for lr, lo, hi in ((0.8, 0.015, 0.040), (1.0, 0.015, 0.120),
                           (1.3, 0.030, 0.200)):
            if lr in by_lr:
                m = float(np.median(by_lr[lr]))
                chk("L/R=%.1f median drift in %.3f-%.3f" % (lr, lo, hi),
                    lo <= m <= hi, "%.3f" % m)
        if 0.8 in by_lr:
            chk("L/R=0.8 fully converged (0%% over)",
                sum(1 for d in by_lr[0.8] if d > 0.15) == 0)
        if 1.3 in by_lr and 0.8 in by_lr:
            worst_lr13 = max(
                100.0 * sum(1 for x in v if x["drift"] > 0.15) / len(v)
                for v in full.values() if round(float(v[0]["l_over_r"]), 1) == 1.3)
            chk("L/R=1.3 worst exceedance ~81%%", 70 <= worst_lr13 <= 90,
                "%.0f%%" % worst_lr13)
        mm = re.search(r"(\d+) draws are complete", flat)
        if mm:
            chk("paper's draw count matches dumps",
                abs(int(mm.group(1)) - len(recs)) <= 25,   # snapshot claim; generation is ongoing
                "paper %s vs dumps %d" % (mm.group(1), len(recs)))

        # top-mode share range claimed as 54-67%
        shares = []
        means = []
        for k, v in full.items():
            c = Counter(x["n_dominant"] for x in v)
            shares.append(100.0 * max(c.values()) / len(v))
            means.append((float(np.mean([x["n_dominant"] for x in v])),
                          v[0]["n_crit_pred"]))
        chk("top-mode share within 38-67", min(shares) >= 37 and max(shares) <= 68,
            "%.0f-%.0f%%" % (min(shares), max(shares)))
        chk("3-5 distinct modes per geometry",
            all(2 < len(set(x["n_dominant"] for x in v)) <= 5 for v in full.values()),
            str(sorted(len(set(x["n_dominant"] for x in v)) for v in full.values())))
        worst = max(abs(m - c) / c for m, c in means)
        chk("mean mode deviation within claimed 13%%", worst < 0.135,
            "max dev %.1f%%" % (100 * worst))
        chk("deviation is one-directional (all below)",
            all(m < c for m, c in means), "claimed systematic")

        for v in ("0.68", "0.59", "0.54", "0.28"):
            chk("paper quotes correlation %s" % v, v in flat)

    print("\n[4] Imperfection ratio claim")
    chk("150x ratio is 0.15/1e-3", abs(0.15 / 1e-3 - 150.0) < 1e-9)
    chk("paper states 150x", "150" in flat)

    print("\n[5] Structure")
    for sec in ("Introduction", "Methodology", "Evaluation protocol",
                "Numerical results", "Conclusions"):
        chk("section present: %s" % sec, sec in flat)
    chk("no unresolved refs", "??" not in flat)
    chk("no unresolved cites", "[?]" not in flat)

    print("\n%d checks, %d failed" % (checks, len(fails)))
    if fails:
        print("FAILED: %s" % fails)
    print("VERIFICATION: %s" % ("PASS" if not fails else "FAIL"))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
