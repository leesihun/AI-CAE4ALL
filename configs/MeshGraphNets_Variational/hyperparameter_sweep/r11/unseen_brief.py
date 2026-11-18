#!/usr/bin/env python3
"""Round 11, second block: the prior on unseen geometries inside the training range, and the seeds pooled.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/unseen_brief.py
    python .../r11/unseen_brief.py --long      # every number, not for typing
    python .../r11/unseen_brief.py --figures   # also one histogram per geometry, the seeds overlaid

Standard library only (--figures needs matplotlib and is skipped without it).
It reads what posterior_vs_prior_groups.py left for three sets of geometries:

    tr  16 of the 120 training geometries x 64 FE draws        (diagnose_135.sh)
    ts  the 15 geometries of the test split x 64 FE draws:     (unseen_135.sh)
        never trained on, drawn from the same ranges as the training geometries
    ho  the 8 held-out geometries x 256 FE draws, outside those ranges   (diagnose_135.sh)

and prints one line per arm. Two numbers are computed here from the draws those
files keep, with the same split protocol (half of the FE results is the
reference, 200 random splits, median; Python's permutations, not numpy's, so a
score recomputed here can differ by about 1 from the files'):

    p   the seeds pooled: each draw comes from one seed's model picked at
        random, as many draws as one seed has. Never above the RMS of the seeds;
        how far below says how much of the error the seeds do not share.
    s   seed against seed: the same distance between two seeds' draws, over the
        FE sd. Near the floor = the seeds draw the same distribution.

    ==== TYPE THIS BACK ====
    R11U 135 D21/21 F0                     ts parts done / runs, failed
    fl tr56 ts45 ho27                      the floor of each set
    bd tr9 ts8 ho6                         half the 5-95% band of one run's prior score over the splits
    a0 tr 82/85/80 s30 ts 48 99/104/97 p90 s40 ho 78/95/75 p70 s55
    T a0 n15 b4 x210 md60                  ts per geometry: count, above 100, worst, median
    G a0 1:154/60/88p70 2:...              ho per geometry: prior per seed, p pooled
    ==== END ====
"""
import argparse
import math
import os
import random
import statistics
import sys

sys.dont_write_bytecode = True   # importing brief.py must not leave a __pycache__ beside the configs
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brief as B   # noqa: E402  (REFS, ARMS, the config reader: one copy)

PARTS = ("tr", "ts", "ho")
DETAIL = ("base", "tfk8")   # the arms with a T and a G line
SPLITS = 200                # posterior_vs_prior_groups.py SPLITS
OUT_ROOT = os.path.join(B.REPO, "output", f"b300_{B.ROUND}")
FAIL_FILE = os.path.join(OUT_ROOT, "_diag", "unseen_fail.txt")

LEGEND = """\
COLUMNS (do not type)   every number x100, rounded ("27" = 0.27)
  header  ts parts done / runs (this round's jobs and R1-R5), failed parts
  fl      the floor of each set: half of a geometry's FE results against the other half
  bd      half the 5-95% band of one run's prior score over the 200 splits (mean over the runs):
          what one checkpoint's score moves by when only the draws and the FE half change
  aN      one line per arm (ARMS below), per set of geometries:
          tr  16 training geometries      prior per seed (42/43(/44))
          ts  15 unseen geometries inside the training ranges: posterior sample (mean over the
              seeds), then the prior per seed
          ho  8 held-out geometries outside the ranges      prior per seed
          p   the seeds pooled (one model per draw, picked at random), the same number of draws
              (ts and ho; --long prints tr's)
          s   seed against seed, over the FE sd: near the floor = the seeds agree;
              near the prior scores = the seeds differ from each other as much as from FE
          (p and s need two seeds; "-" = that part is not there yet)
  T aN    the ts geometries one by one (prior, mean over the seeds): n geometries,
          b how many are above 100, x the worst, md the median geometry
  G aN    the ho geometries one by one (GEOMETRIES below): prior per seed, p pooled
  d       a ts part that failed (run:part); the reason is in output/b300_r11/_diag/unseen_fail.txt
"""
READING_KO = """\
읽는 법 (do not type)
  ts의 prior가 ho만큼 나쁘다      = 범위 안의 새 geometry에서도 틀린다: 학습 geometry 수가 모자란다
  ts의 prior가 floor 근처다       = 남은 오차는 범위 밖(ho)으로 나갈 때만 생긴다
  p가 seed들보다 한참 낮다, s가 크다 = seed마다 다른 쪽으로 틀린다: 여러 모델을 섞으면 줄어든다
  p가 seed들과 비슷하다, s가 작다  = seed들이 같은 쪽으로 틀린다: 섞어도 소용없다
  s가 tr < ts < ho로 커진다       = 학습 geometry에서 멀수록 prior가 seed에 따라 달라진다
"""
READING_EN = """\
HOW TO READ (do not type)
  ts prior as bad as ho          = wrong on new geometries inside the ranges too: too few training geometries
  ts prior near the floor        = the remaining error appears only outside the ranges (ho)
  p far below the seeds, s large = each seed errs its own way: mixing models removes that part
  p near the seeds, s small      = the seeds share the error: mixing does not help
  s grows tr < ts < ho           = the farther from the training geometries, the more the prior depends on the seed
"""


def reading():
    try:
        READING_KO.encode(getattr(sys.stdout, "encoding", None) or "ascii")
    except (UnicodeError, LookupError):
        return READING_EN
    return READING_KO


def rel(path):
    return os.path.relpath(path, B.REPO).replace(os.sep, "/")


def runs():
    """R1-R5, then this round's jobs: num, label, arm, seed, infer config, diag dir, checkpoint."""
    out = []
    for num, rnd, name, arm, seed, ddir in B.REFS:
        out.append({"num": num, "label": f"{rnd}_{name}", "arm": arm, "seed": seed,
                    "infer": os.path.join(HERE, "..", rnd, f"config_infer_{name}.txt"),
                    "ddir": os.path.join(B.REPO, "output", *ddir.split("/"))})
    table = os.path.join(HERE, "jobs.txt")
    if not os.path.isfile(table):
        raise SystemExit(f"no jobs.txt ({table}); generate it with make_sweep.py")
    k = 0
    with open(table, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5 or parts[0] != B.NODE:
                continue
            k += 1
            m = B.JID_RE.match(parts[1])
            if not m:
                continue
            out.append({"num": str(k), "label": parts[1], "arm": m.group(1), "seed": int(m.group(2)),
                        "infer": os.path.join(B.REPO, parts[4]),
                        "ddir": os.path.join(OUT_ROOT, parts[1], "diag")})
    for r in out:
        r["infer"] = os.path.normpath(r["infer"])
        try:
            r["model"] = B.method_path(B.cfg(r["infer"])["modelpath"])
        except (OSError, KeyError):
            r["model"] = None
    return out


# ------------------------------------------------------------------ scoring

def finite(vals):
    return all(v == v and abs(v) != float("inf") for v in vals)


def sd(vals):
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def rms(vals):
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def w2sq(x, y):
    """Exact 1-D W2^2 of two sorted samples (the quantile coupling), any two sizes."""
    n, m = len(x), len(y)
    if n == m:
        return sum((a - b) ** 2 for a, b in zip(x, y)) / n
    i = j = t = 0   # t: the quantile level in units of 1/(n*m)
    acc = 0.0
    while i < n and j < m:
        nx, ny = (i + 1) * m, (j + 1) * n
        t2 = min(nx, ny)
        acc += (t2 - t) * (x[i] - y[j]) ** 2
        i += nx == t2
        j += ny == t2
        t = t2
    return acc / (n * m)


def score(d, ensemble, key="p50"):
    return ((((d or {}).get("scores") or {}).get(ensemble) or {}).get("w2") or {}).get(key)


def pooled(datas, splits=SPLITS):
    """The seeds of one arm on one set of geometries, scored together.

    datas: their diagnosis JSONs. Returns None or
    {pool, pair, seeds: [own score per seed], geo: {geometry: pooled median}}.
    """
    spreads = [d.get("spreads") or {} for d in datas]
    geos = []
    for v, s0 in spreads[0].items():
        truth = s0.get("truth") or []
        if len(truth) >= 4 and finite(truth) and all(
                v in s and len(s[v].get("prior") or []) == len(truth) and finite(s[v]["prior"])
                for s in spreads):
            geos.append(v)
    if not geos or len(datas) < 2:
        return None
    n_seed = len(datas)
    k = min(len(spreads[0][v]["truth"]) // 2 for v in geos)
    rng = random.Random(0)
    runs_ = {"pool": [], "pair": [], "seeds": [[] for _ in range(n_seed)]}
    per_geo = {v: [] for v in geos}
    for _ in range(splits):
        rows = {"pool": [], "pair": [], "seeds": [[] for _ in range(n_seed)]}
        for v in geos:
            truth = spreads[0][v]["truth"]
            prior = [s[v]["prior"] for s in spreads]
            perm = rng.sample(range(len(truth)), len(truth))
            ia, ib = perm[:k], perm[k:]
            ref = sorted(truth[i] for i in ib)
            sy = sd(ref)
            if sy <= 0.0:
                continue
            w = math.sqrt(w2sq(sorted(prior[rng.randrange(n_seed)][i] for i in ia), ref)) / sy
            rows["pool"].append(w)
            per_geo[v].append(w)
            first = [sorted(p[i] for i in ia) for p in prior]
            second = [sorted(p[i] for i in ib) for p in prior]
            for s in range(n_seed):
                rows["seeds"][s].append(math.sqrt(w2sq(first[s], ref)) / sy)
            pairs = [math.sqrt(w2sq(first[a], second[b])) / sy
                     for a in range(n_seed) for b in range(a + 1, n_seed)]
            rows["pair"].append(sum(pairs) / len(pairs))
        if not rows["pool"]:
            continue
        runs_["pool"].append(rms(rows["pool"]))
        runs_["pair"].append(rms(rows["pair"]))
        for s in range(n_seed):
            runs_["seeds"][s].append(rms(rows["seeds"][s]))
    if not runs_["pool"]:
        return None
    return {"pool": statistics.median(runs_["pool"]), "pair": statistics.median(runs_["pair"]),
            "seeds": [statistics.median(r) for r in runs_["seeds"]],
            "geo": {v: statistics.median(w) for v, w in per_geo.items() if w}}


def per_geometry(d, ensemble="prior"):
    """{geometry: that ensemble's median W2} of one diagnosis, in the file's order."""
    return {v: e.get(ensemble) for v, e in ((d or {}).get("per_group_w2_median") or {}).items()}


# ------------------------------------------------------------------ figures

def figures(arms, data, arm_num):
    """One PNG per arm and set: per geometry, the FE results and every seed's prior draws."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:   # no matplotlib here: the block does not need it
        print(f"  [figures skipped: {exc}]")
        return
    out_dir = os.path.join(OUT_ROOT, "_diag")
    os.makedirs(out_dir, exist_ok=True)
    for arm, rs in arms.items():
        for part in PARTS:
            ds = [(r["seed"], data[r["label"], part]) for r in rs if data.get((r["label"], part))]
            if not ds:
                continue
            geos = list(ds[0][1].get("spreads") or {})
            if not geos:
                continue
            cols = min(4, len(geos))
            nrow = -(-len(geos) // cols)
            fig, axes = plt.subplots(nrow, cols, figsize=(4.2 * cols, 3.0 * nrow), squeeze=False)
            for ax in axes.flat[len(geos):]:
                ax.axis("off")
            for ax, v in zip(axes.flat, geos):
                truth = ds[0][1]["spreads"][v]["truth"]
                lo, hi = min(truth), max(truth)
                span = (hi - lo) or 1.0
                lo, hi = lo - 0.5 * span, hi + 0.5 * span   # draws beyond pile up in the end bins
                edges = [lo + (hi - lo) * i / 40 for i in range(41)]
                ax.hist(truth, bins=edges, density=True, color="0.78", label="FE")
                for seed, d in ds:
                    draws = (d["spreads"].get(v) or {}).get("prior") or []
                    if not draws or not finite(draws):
                        continue
                    w = per_geometry(d).get(v)
                    ax.hist([min(max(x, lo), hi) for x in draws], bins=edges, density=True,
                            histtype="step", linewidth=1.3, label=f"seed {seed}: {B.x100(w)}")
                ax.set_title(f"{ds[0][1].get('attr', 'geometry')} = {v}", fontsize=9)
                ax.set_yticks([])
                ax.legend(fontsize=7, frameon=False)
            fig.suptitle(f"a{arm_num[arm]} {arm}, {part}: FE results and the prior's draws per seed "
                         f"(legend: W2 x100)", fontsize=10)
            fig.tight_layout()
            path = os.path.join(out_dir, f"unseen_{arm}_{part}.png")
            fig.savefig(path, dpi=110)
            plt.close(fig)
            print(f"  figure {rel(path)}")


# ------------------------------------------------------------------ the block

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", action="store_true", help="every run and every geometry, not for typing")
    ap.add_argument("--figures", action="store_true", help="per-geometry histograms under output/b300_r11/_diag")
    ap.add_argument("--runs", action="store_true",
                    help="for unseen_135.sh: label, infer config, diag dir, checkpoint of every run")
    args = ap.parse_args()

    all_runs = runs()
    if args.runs:
        for r in all_runs:
            print(r["label"], rel(r["infer"]), rel(r["ddir"]), rel(r["model"]) if r["model"] else "-")
        return 0

    data = {}
    for r in all_runs:
        for part in PARTS:
            data[r["label"], part] = B.load_json(
                os.path.join(r["ddir"], f"posterior_vs_prior_groups_{part}.json"))
    arm_num = {arm: k for k, arm in enumerate(B.ARMS)}
    arms = {arm: sorted((r for r in all_runs if r["arm"] == arm), key=lambda r: r["seed"]) for arm in B.ARMS}

    print(LEGEND)
    print(reading())
    print("RUNS (do not type)")
    for r in all_runs:
        print(f"  {r['num']:>2} {r['label']:<24} a{arm_num.get(r['arm'], '?')} {r['arm']}, seed {r['seed']}")
    print()
    geo_names = []
    for r in all_runs:
        d = data[r["label"], "ho"]
        if d:
            geo_names = list(d.get("spreads") or {})
            print("GEOMETRIES of the G lines (do not type)   " + (d.get("attr") or "group") + ": "
                  + " ".join(f"{i}={v}" for i, v in enumerate(geo_names, 1)))
            print()
            break

    lines, tlines, glines = [], [], []
    for arm, rs in arms.items():
        if not any(data[r["label"], part] for r in rs for part in PARTS):
            continue
        line = f"a{arm_num[arm]}"
        pools = {}
        for part in PARTS:
            ds = [data[r["label"], part] for r in rs]
            line += f" {part} "
            if part == "ts":
                line += B.x100(B.mean([score(d, "posterior_sample") for d in ds])) + " "
            line += "/".join(B.x100(score(d, "prior")) for d in ds)
            good = [d for d in ds if score(d, "prior") is not None and finite([score(d, "prior")])]
            pools[part] = pooled(good) if len(good) > 1 else None
            if pools[part]:   # tr: s only, to keep the line short
                line += (f" p{B.x100(pools[part]['pool'])}" if part != "tr" else "") \
                    + f" s{B.x100(pools[part]['pair'])}"
            if args.long and pools[part]:
                print(f"  a{arm_num[arm]} {part}: p{B.x100(pools[part]['pool'])}, recomputed here per seed "
                      + "/".join(B.x100(v) for v in pools[part]["seeds"])
                      + "   (the files: " + "/".join(B.x100(score(d, "prior")) for d in good) + ")")
        lines.append(line)
        if args.long:
            for part in PARTS:
                per = [per_geometry(data[r["label"], part]) for r in rs]
                for v in next((list(p) for p in per if p), []):
                    print(f"  a{arm_num[arm]} {part} {v}: "
                          + "/".join(B.x100(p.get(v)) for p in per)
                          + (f" p{B.x100(pools[part]['geo'].get(v))}" if pools[part] else ""))
        if arm not in DETAIL:
            continue
        per = [p for p in (per_geometry(data[r["label"], "ts"]) for r in rs) if p]
        if per:
            means = [B.mean([p.get(v) for p in per]) for v in per[0]]
            means = [m for m in means if m is not None]
            if means:
                tlines.append(f"T a{arm_num[arm]} n{len(means)} b{sum(m > 1.0 for m in means)} "
                              f"x{B.x100(max(means))} md{B.x100(statistics.median(means))}")
        per = [per_geometry(data[r["label"], "ho"]) for r in rs]
        if any(per):
            order = geo_names or next(list(p) for p in per if p)
            cells = []
            for i, v in enumerate(order, 1):
                cell = f"{i}:" + "/".join(B.x100(p.get(v)) for p in per)
                if pools["ho"]:
                    cell += f"p{B.x100(pools['ho']['geo'].get(v))}"
                cells.append(cell)
            glines.append(f"G a{arm_num[arm]} " + " ".join(cells))

    if args.long:
        print()
        for r in all_runs:
            cells = []
            for part in PARTS:
                d = data[r["label"], part]
                cells.append(f"{part} " + "/".join(B.x100(score(d, e)) for e in ("floor", "posterior_sample", "prior"))
                             + f" [{B.x100(score(d, 'prior', 'p5'))}-{B.x100(score(d, 'prior', 'p95'))}]")
            print(f"  {r['num']:>2} {r['label']:<24} " + "   ".join(cells))
        print()
    if args.figures:
        figures(arms, data, arm_num)
        print()

    def over_runs(part, fn):
        return B.mean([fn(data[r["label"], part]) for r in all_runs if data[r["label"], part]])

    def half_band(d):
        lo, hi = score(d, "prior", "p5"), score(d, "prior", "p95")
        return None if lo is None or hi is None else (hi - lo) / 2

    fails = []
    nums = {r["label"]: r["num"] for r in all_runs}
    for row in (l.split() for l in (B.read_text(FAIL_FILE) or "").splitlines()):
        if len(row) > 2 and row[0] == "FAIL":
            fails.append(f"{nums.get(row[1], row[1])}:{row[2]}")
    done = sum(1 for r in all_runs if data[r["label"], "ts"])
    print("==== TYPE THIS BACK ====")
    print(f"{B.ROUND.upper()}U {B.NODE} D{done}/{len(all_runs)} F{len(fails)}")
    print("fl " + " ".join(f"{p}{B.x100(over_runs(p, lambda d: score(d, 'floor')))}" for p in PARTS))
    print("bd " + " ".join(f"{p}{B.x100(over_runs(p, half_band))}" for p in PARTS))
    if fails:
        print("d " + " ".join(fails))
    for line in lines + tlines + glines:
        print(line)
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
