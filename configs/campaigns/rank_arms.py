#!/usr/bin/env python3
"""Rank sweep arms from their spread dumps. Nothing else is read.

    python configs/campaigns/rank_arms.py output/meshgraphnets-v/saoi_sweep3
    python configs/campaigns/rank_arms.py output/chi-mgnflow/saoi_sweepB
    python configs/campaigns/rank_arms.py output/meshgraphnets-v/saoi_sweep3 output/chi-mgnflow/saoi_sweepB

Finds every spread_values.npz under the given directories, computes the
metrics from the two arrays inside (gt = ground-truth spreads, gen = generated
spreads), and prints one ranked table. No configs, no scene labels, no
score_sweep.py -- if the npz exists this works.

The SAOI eval design: each eval set is ONE geometry with 125 realizations of
it, so gt is the true conditional distribution for that part and gen is the
model's. The comparison is exact and sd_ratio's target is 1.

Columns (ideal value in the header):

  W1/sd     1-Wasserstein(gt, gen) / sd(gt).        0 = identical distributions
  dmean/sd  (mean gen - mean gt) / sd(gt).          0 = unbiased
  sd_ratio  sd(gen) / sd(gt).                       1 = right width; <1 too narrow
  PITtails  share of gt realizations landing in the outer 2% of the ensemble
            on either side.                         0.04 expected; >>0.04 too narrow
  PIT_KS    KS distance of the gt ranks from uniform. 0 = calibrated

Ranking is by mean W1/sd over eval sets (lower is better). sd_ratio and
PITtails are two independent views of the same width defect and should agree.
"""
import glob
import os
import sys

import numpy as np


def w1(a, b, n=512):
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def ks_uniform(u):
    u = np.sort(u)
    n = u.size
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def metrics(gt, gen):
    gt = np.asarray(gt, float).ravel()
    gen = np.asarray(gen, float).ravel()
    sd = float(gt.std())
    if gt.size < 2 or sd == 0.0:
        return {"error": f"gt has {gt.size} value(s) -- nothing to normalize by"}
    order = np.sort(gen)
    pit = np.searchsorted(order, gt, side="left") / order.size
    return {
        "n_gt": int(gt.size), "n_gen": int(gen.size),
        "w1": w1(gt, gen) / sd,
        "dmean": (float(gen.mean()) - float(gt.mean())) / sd,
        "sd_ratio": float(gen.std()) / sd,
        "pit_tails": float(np.mean((pit <= 0.02) | (pit >= 0.98))),
        "pit_ks": ks_uniform(pit),
    }


def arm_tag(path):
    """.../infer/<arm>/<tag>/spread_values.npz  or  .../infer/det/<arm>/<tag>/..."""
    parts = os.path.normpath(path).split(os.sep)
    # tag = parent dir, arm = grandparent dir
    return parts[-3], parts[-2], ("det" in parts[-5:-3])


def collect(roots):
    rows = {}
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "spread_values.npz"),
                                  recursive=True)):
            arm, tag, det = arm_tag(f)
            sweep = os.path.basename(root.rstrip("/\\"))
            mode = "det" if det else "full"
            try:
                with np.load(f, allow_pickle=False) as z:
                    gt, gen = z["gt"], z["gen"]
                    lam = z["gen_lam"] if "gen_lam" in z else None
            except Exception as exc:
                rows.setdefault((sweep, mode, arm), {})[tag] = {"error": str(exc)}
                continue
            if lam is not None and lam.size == gen.size and np.unique(lam).size > 1:
                # One inference pass swept several inflation factors; each is a
                # different model variant, so rank them as separate arms.
                for lv in np.unique(lam):
                    m = metrics(gt, gen[lam == lv])
                    rows.setdefault((sweep, mode, f"{arm}@lam{lv:g}"), {})[tag] = m
            else:
                rows.setdefault((sweep, mode, arm), {})[tag] = metrics(gt, gen)
    return rows



def lam_response(arms):
    """sd_ratio vs lambda per (base arm, eval set), with gain and the lam that
    would reach sd_ratio 1.

    Returns [] unless the dumps carry more than one lambda.
    """
    by_base = {}
    for arm, tags in arms.items():
        if "@lam" not in arm:
            continue
        base, lam = arm.split("@lam")
        for tag, m in tags.items():
            if "error" in m:
                continue
            by_base.setdefault((base, tag), {})[float(lam)] = m["sd_ratio"]
    return [(b, t, dict(sorted(d.items()))) for (b, t), d in sorted(by_base.items())
            if len(d) > 1]


def render_lam_response(arms):
    rows = lam_response(arms)
    if not rows:
        return
    lams = sorted({l for _, _, d in rows for l in d})
    print()
    print("LAMBDA RESPONSE   sd_ratio per inflation factor   (target 1.000)")
    head = f"{'arm':<6}{'eval set':<16}" + "".join(f"{('lam=' + f'{l:g}'):>9}" for l in lams)
    print(head + f"{'gain':>8}{'linear':>8}{'lam*':>8}")
    print("-" * len(head + " " * 24))
    star = {}
    for base, tag, d in rows:
        cells = "".join(f"{d[l]:>9.3f}" if l in d else f"{'-':>9}" for l in lams)
        lo, hi = min(d), max(d)
        gain = d[hi] / d[lo] if d[lo] else float("nan")
        # local slope through the two outermost points, extrapolated to 1.0
        slope = (d[hi] - d[lo]) / (hi - lo) if hi != lo else 0.0
        lam_star = (lo + (1.0 - d[lo]) / slope) if slope > 1e-9 else float("nan")
        star.setdefault(base, []).append((tag, lam_star))
        ls = f"{lam_star:>8.2f}" if lam_star == lam_star else f"{'n/a':>8}"
        print(f"{base:<6}{tag:<16}{cells}{gain:>8.2f}{hi / lo:>8.2f}{ls}")
    print()
    print("gain   = sd_ratio(lam_max)/sd_ratio(lam_min) actually observed")
    print("linear = what it would be if the decoder passed width through "
          "proportionally")
    print("lam*   = the lam that set would need to reach sd_ratio 1")
    for base, pairs in star.items():
        vals = [v for _, v in pairs if v == v]
        if len(vals) < 2:
            continue
        spread = max(vals) / min(vals)
        print()
        if spread > 1.4:
            print(f"arm {base}: the eval sets need DIFFERENT lam "
                  f"({', '.join(f'{t} {v:.2f}' for t, v in pairs if v == v)}) -- "
                  f"a {spread:.1f}x disagreement.")
            print(f"  Inflation is a single global multiplier, so no one value "
                  f"calibrates them all. The width error is geometry-dependent, "
                  f"not a scale constant: inflation is ruled out as the fix.")
        else:
            print(f"arm {base}: the eval sets agree on lam ~ "
                  f"{sum(vals) / len(vals):.2f} (within {spread:.2f}x) -- "
                  f"a single global factor calibrates them. Use it.")

def fmt(v, w=8, p=3):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def main():
    roots = sys.argv[1:]
    if not roots:
        print(__doc__)
        return 2
    rows = collect(roots)
    if not rows:
        print("No spread_values.npz found under:")
        for r in roots:
            print(f"  {r}")
            if os.path.isdir(r):
                print(f"    contains: {sorted(os.listdir(r))[:15]}")
            else:
                print("    (directory does not exist)")
        return 1

    # group by (sweep, mode)
    groups = {}
    for (sweep, mode, arm), tags in rows.items():
        groups.setdefault((sweep, mode), {})[arm] = tags

    for (sweep, mode), arms in groups.items():
        tags_all = sorted({t for tg in arms.values() for t in tg})
        print()
        print("=" * 100)
        print(f"{sweep}   [{mode} run]   {len(arms)} arm(s), eval sets: {', '.join(tags_all)}")
        print("=" * 100)

        # ---- per arm x eval set ----
        print(f"{'arm':<12}{'eval set':<16}{'n_gt':>6}{'n_gen':>7}"
              f"{'W1/sd':>9}{'dmean/sd':>10}{'sd_ratio':>10}{'PITtails':>10}{'PIT_KS':>8}")
        print(f"{'':<12}{'':<16}{'':>6}{'':>7}"
              f"{'(0)':>9}{'(0)':>10}{'(1)':>10}{'(0.04)':>10}{'(0)':>8}")
        print("-" * 100)
        summary = []
        for arm in sorted(arms, key=lambda a: (a.split('@')[0].zfill(3), a)):
            vals = []
            for tag in tags_all:
                m = arms[arm].get(tag)
                if m is None:
                    print(f"{arm:<12}{tag:<16}{'(no dump)':>13}")
                    continue
                if "error" in m:
                    print(f"{arm:<12}{tag:<16}   INVALID: {m['error']}")
                    continue
                print(f"{arm:<12}{tag:<16}{m['n_gt']:>6}{m['n_gen']:>7}"
                      f"{fmt(m['w1'],9)}{fmt(m['dmean'],10)}{fmt(m['sd_ratio'],10)}"
                      f"{fmt(m['pit_tails'],10)}{fmt(m['pit_ks'],8)}")
                vals.append(m)
            if vals:
                summary.append((arm, {
                    "n": len(vals),
                    "w1": np.mean([v["w1"] for v in vals]),
                    "dmean": np.mean([abs(v["dmean"]) for v in vals]),
                    "sd_ratio": np.mean([v["sd_ratio"] for v in vals]),
                    "pit_tails": np.mean([v["pit_tails"] for v in vals]),
                    "pit_ks": np.mean([v["pit_ks"] for v in vals]),
                }))

        # ---- lambda response (inflation sweeps only) ----
        render_lam_response(arms)

        # ---- ranking ----
        if not summary:
            continue
        summary.sort(key=lambda s: s[1]["w1"])
        best_width = min(summary, key=lambda s: abs(s[1]["sd_ratio"] - 1.0))[0]
        print()
        print(f"RANKING by mean W1/sd  (lower = better; {len(tags_all)} eval sets averaged)")
        print(f"{'rank':<6}{'arm':<12}{'sets':>5}{'W1/sd':>9}{'|dmean|/sd':>12}"
              f"{'sd_ratio':>10}{'PITtails':>10}{'PIT_KS':>8}   note")
        print("-" * 100)
        for i, (arm, s) in enumerate(summary, 1):
            notes = []
            if arm == best_width:
                notes.append("closest to sd_ratio 1")
            if s["sd_ratio"] < 0.7:
                notes.append("too narrow")
            if s["pit_tails"] > 0.10:
                notes.append("truths fall outside ensemble")
            if s["dmean"] > 1.0:
                notes.append("BIASED >1 sd")
            if s["n"] < len(tags_all):
                notes.append(f"only {s['n']}/{len(tags_all)} sets")
            print(f"{i:<6}{arm:<12}{s['n']:>5}{fmt(s['w1'],9)}{fmt(s['dmean'],12)}"
                  f"{fmt(s['sd_ratio'],10)}{fmt(s['pit_tails'],10)}{fmt(s['pit_ks'],8)}"
                  f"   {'; '.join(notes)}")
        print()
        print("read: W1/sd ranks accuracy+distribution together. sd_ratio<1 and PITtails>>0.04 "
              "both mean the ensemble is too narrow. |dmean|/sd>1 means a mean offset "
              "that no width fix can repair.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
