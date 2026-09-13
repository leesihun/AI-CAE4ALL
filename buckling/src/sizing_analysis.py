"""How large does the training set actually need to be?

Two axes trade off at fixed compute: more geometries (better conditioning map)
versus more draws per geometry (better resolved conditional). They are not
interchangeable, and which one binds is a measurement, not a preference.

The question this answers: at the current 48 draws per geometry, how many of
the 12 training conditions are statistically distinguishable from each other?
A pair that sits inside the sampling noise is, for learning purposes, one
condition seen twice.
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


def mode_hist(ns, lo=5, hi=16):
    h = np.zeros(hi - lo + 1)
    for n in ns:
        if lo <= n <= hi:
            h[int(n) - lo] += 1
    return h / max(h.sum(), 1)


def tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def load():
    recs = [x for x in json.load(open(os.path.join(PROD, "results.json")))
            if x.get("ok")]
    plan = {}
    for e in json.load(open(os.path.join(ROOT, "work", "plan_full.json"))):
        k = "%s_rt%d_lr%03d_a%02d_g%02d" % (
            e["shape"][:3], e["r_over_t"], round(e["l_over_r"] * 100),
            round(e["alpha_deg"] * 10), round(e["thickness_gamma"] * 100))
        plan[k] = e["tier"]
    g = collections.defaultdict(list)
    for x in recs:
        gk = x["key"].rsplit("_s", 1)[0]
        if plan.get(gk) == "train":
            g[gk].append(x)
    return {k: v for k, v in g.items() if len(v) >= 40}


def noise_floor(g, n_sub, reps=400, rng=None):
    """TV between two independent halves of the SAME geometry at sample size n_sub.

    This is the resolution limit: two geometries closer than this cannot be
    told apart with n_sub draws each, no matter how good the model is.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for v in g.values():
        ns = [x["n_dominant"] for x in v]
        if len(ns) < 2 * n_sub:
            continue
        for _ in range(reps // max(len(g), 1) + 1):
            idx = rng.permutation(len(ns))
            a = [ns[i] for i in idx[:n_sub]]
            b = [ns[i] for i in idx[n_sub:2 * n_sub]]
            out.append(tv(mode_hist(a), mode_hist(b)))
    return np.asarray(out)


def main():
    g = load()
    ks = sorted(g)
    print("training geometries: %d, draws each: %s"
          % (len(ks), sorted({len(v) for v in g.values()})))
    H = {k: mode_hist([x["n_dominant"] for x in g[k]]) for k in ks}

    # ---- resolution limit at several sample sizes ------------------------
    print("\n[1] sampling noise floor on the mode histogram")
    print("    (TV between two independent halves of the same geometry)")
    print("    %-8s %-9s %-9s" % ("n/geom", "mean TV", "p95 TV"))
    floors = {}
    for n_sub in (12, 24):
        f = noise_floor(g, n_sub)
        floors[n_sub] = f
        print("    %-8d %-9.3f %-9.3f" % (n_sub, f.mean(), np.percentile(f, 95)))
    # extrapolate: TV noise scales ~ 1/sqrt(n)
    base = floors[24].mean() * np.sqrt(24)
    print("    TV noise ~ %.3f / sqrt(n):" % base)
    for n in (48, 96, 192, 384):
        print("       n=%-4d -> mean TV %.3f" % (n, base / np.sqrt(n)))

    # ---- how many training pairs actually separate? ----------------------
    thr = np.percentile(floors[24], 95)          # p95 floor at n=24 per half
    print("\n[2] between-geometry separation, all %d pairs" % (len(ks) * (len(ks) - 1) // 2))
    print("    threshold = p95 of the within-geometry floor at n=24 -> %.3f" % thr)
    sep, tot, below = 0, 0, []
    for a, b in itertools.combinations(ks, 2):
        d = tv(H[a], H[b])
        tot += 1
        if d > thr:
            sep += 1
        else:
            below.append((a, b, d))
    print("    separated: %d / %d  (%.0f%%)" % (sep, tot, 100 * sep / tot))
    print("    NOT separated: %d pairs" % len(below))
    for a, b, d in sorted(below, key=lambda x: x[2])[:12]:
        ra = g[a][0]
        rb = g[b][0]
        same_rt = "same R/t" if ra["r_over_t"] == rb["r_over_t"] else ""
        print("       %-14s vs %-14s TV=%.3f  %s"
              % (a.replace("cyl_", "").replace("_a00_g00", ""),
                 b.replace("cyl_", "").replace("_a00_g00", ""), d, same_rt))

    # ---- effective number of distinct conditions ------------------------
    print("\n[3] effective number of distinct conditions")
    # greedy clustering: merge any pair closer than the noise floor
    parent = {k: k for k in ks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(ks, 2):
        if tv(H[a], H[b]) <= thr:
            parent[find(a)] = find(b)
    groups = collections.defaultdict(list)
    for k in ks:
        groups[find(k)].append(k)
    print("    %d nominal geometries collapse to %d distinguishable groups"
          % (len(ks), len(groups)))
    for i, (_, mem) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
        rts = sorted({g[m][0]["r_over_t"] for m in mem})
        lrs = sorted({g[m][0]["l_over_r"] for m in mem})
        print("      group %d: %d geometries  R/t=%s  L/R=%s"
              % (i, len(mem), rts, lrs))

    # ---- where does the variation actually live? ------------------------
    print("\n[4] which axis carries the conditioning signal?")
    by_rt = collections.defaultdict(list)
    by_lr = collections.defaultdict(list)
    for k in ks:
        by_rt[g[k][0]["r_over_t"]].append(k)
        by_lr[g[k][0]["l_over_r"]].append(k)
    for lbl, grp in (("R/t", by_rt), ("L/R", by_lr)):
        within, between = [], []
        for _, mem in grp.items():
            for a, b in itertools.combinations(mem, 2):
                within.append(tv(H[a], H[b]))
        allp = [tv(H[a], H[b]) for a, b in itertools.combinations(ks, 2)]
        print("    fixing %s: mean TV among geometries sharing it = %.3f "
              "(all pairs %.3f)" % (lbl, np.mean(within), np.mean(allp)))


if __name__ == "__main__":
    main()
