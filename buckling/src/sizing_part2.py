"""Follow-up: effective condition count done properly, field-level check, cost.

Part 1 used single-linkage to group geometries, which chains through
overlapping pairs and collapsed all twelve into one group. That is an artefact
of the linkage rule, not a measurement. This redoes it with complete linkage
(a group must have EVERY internal pair inside the noise floor), adds the
field-level separation the discrete label cannot see, and prices the options.
"""
import collections
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROD = os.path.join(ROOT, "work", "production")
sys.path.insert(0, HERE)

from sizing_analysis import load, mode_hist, tv, noise_floor   # noqa: E402
from geometry import ShellMesh                                  # noqa: E402
from scoring import energy_score_vs_sample, self_score          # noqa: E402

NU = 0.3


def main():
    g = load()
    ks = sorted(g)
    H = {k: mode_hist([x["n_dominant"] for x in g[k]]) for k in ks}
    thr = float(np.percentile(noise_floor(g, 24), 95))

    # ---- complete-linkage grouping -------------------------------------
    print("[3b] effective conditions, complete linkage (every internal pair "
          "inside %.3f)" % thr)
    remaining = list(ks)
    groups = []
    while remaining:
        seed = remaining.pop(0)
        grp = [seed]
        for k in list(remaining):
            if all(tv(H[k], H[m]) <= thr for m in grp):
                grp.append(k)
                remaining.remove(k)
        groups.append(grp)
    print("     %d nominal geometries -> %d groups" % (len(ks), len(groups)))
    for i, grp in enumerate(groups, 1):
        tag = [m.replace("cyl_", "").replace("_a00_g00", "") for m in grp]
        print("       group %d (%d): %s" % (i, len(grp), ", ".join(tag)))

    # per-R/t view, which part 1 showed is where the signal lives
    by_rt = collections.defaultdict(list)
    for k in ks:
        by_rt[g[k][0]["r_over_t"]].append(k)
    print("     for comparison, grouping by R/t alone gives %d groups: %s"
          % (len(by_rt), sorted(by_rt)))

    # ---- field-level separation on ALL 12 ------------------------------
    print("\n[5] field-level separation, all %d pairs (SO(2)-invariant spectrum)"
          % (len(ks) * (len(ks) - 1) // 2))
    NK = 40
    S = {}
    for k in ks:
        M = []
        for x in g[k]:
            z = np.load(os.path.join(PROD, x["key"], "draw.npz"))
            sp = z["spectrum"][1:NK + 1].astype(float)
            M.append(sp / max(np.linalg.norm(sp), 1e-12))
        S[k] = np.stack(M)
    floors = {k: self_score(S[k]) for k in ks}
    ratios, weak = [], []
    for a, b in itertools.combinations(ks, 2):
        r = 0.5 * (energy_score_vs_sample(S[a], S[b]) / floors[b]
                   + energy_score_vs_sample(S[b], S[a]) / floors[a])
        ratios.append(r)
        if r < 1.10:
            weak.append((a, b, r))
    ratios = np.asarray(ratios)
    print("     ratio to self floor: min %.2f  median %.2f  max %.2f"
          % (ratios.min(), np.median(ratios), ratios.max()))
    print("     pairs above 1.10: %d / %d" % ((ratios > 1.10).sum(), len(ratios)))
    print("     weakest pairs (field can barely tell them apart):")
    for a, b, r in sorted(weak, key=lambda x: x[2])[:6]:
        print("       %-14s vs %-14s %.3f"
              % (a.replace("cyl_", "").replace("_a00_g00", ""),
                 b.replace("cyl_", "").replace("_a00_g00", ""), r))

    # ---- cost of the options -------------------------------------------
    print("\n[6] what expansion costs, priced on the measured run")
    recs = [x for x in json.load(open(os.path.join(PROD, "results.json")))
            if x.get("ok") and not x.get("cached")]
    sec = [x["seconds"] for x in recs]
    per_draw = float(np.mean(sec))
    LANES, EFF = 19, 0.72
    print("     measured mean cost per draw: %.0f s (%d draws)" % (per_draw, len(sec)))

    def wall(n_draws):
        return n_draws * per_draw / 3600.0 / LANES / EFF

    print("     current train tier: 12 geom x 48 = 576 draws -> %.1f h" % wall(576))
    print()
    print("     %-44s %-8s %-9s" % ("option", "draws", "wall-h"))
    opts = [
        ("A  more draws: 12 geom x 128", 12 * 128),
        ("B  more geometries: 36 geom x 48", 36 * 48),
        ("C  more geometries: 60 geom x 48", 60 * 48),
        ("D  both: 36 geom x 96", 36 * 96),
        ("E  both: 60 geom x 128", 60 * 128),
    ]
    for lbl, n in opts:
        print("     %-44s %-8d %-9.1f" % (lbl, n, wall(n)))

    print("\n     TV noise at each per-geometry sample size (0.863/sqrt(n)):")
    for n in (48, 96, 128, 192):
        print("       n=%-4d -> %.3f" % (n, 0.863 / np.sqrt(n)))


if __name__ == "__main__":
    main()
