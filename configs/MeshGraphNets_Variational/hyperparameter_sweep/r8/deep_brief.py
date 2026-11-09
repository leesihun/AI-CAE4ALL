#!/usr/bin/env python3
"""The block to type back from diagnose_deep.sh (B300 round 8): where the prior loses, and why.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/deep_brief.py
    python .../r8/deep_brief.py --list     # the job ids diagnose_deep.sh runs (one per line)

Standard library only. It reads the JSONs posterior_vs_prior_groups.py wrote
for each job (output/b300_r8/<job>/diag/posterior_vs_prior_groups_deep_<split>.json)
and each job's train.log, and prints the legend, then the block:

    ==== TYPE THIS BACK ====
    DEEP R8 crps x1e-4
    2 ho 27/32/87/140 jt 33/92 ts 40/44/80 tr 40/41/52 c 123/110/112 fm 2.5/1.9
    2 geom 150:+120/60/90 151:...
    2 lat g 43/21/94 31/20/90 12/22/101 l 80/104 70/101 30/99
    ==== END ====
"""
import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
ROUND = os.path.basename(HERE)   # r8
NODE = "135"
OUT_ROOT = os.path.join(REPO, "output", f"b300_{ROUND}")
# base, the best (e360), the longest (e720), the other winners, the prior-capacity arm, the outlier
JOBS = ["hybrid_base", "hybrid_e360", "hybrid_e720", "hybrid_lr2e-4", "hybrid_pms0.1",
        "hybrid_fmbig", "hybrid_mmd100"]
DETAIL = ["hybrid_base", "hybrid_e360"]   # the geom and lat lines
SPLITS = ["ho", "jt", "ts", "tr"]
EPOCH_RE = re.compile(r"^Elapsed: [\d.]+s Epoch (\d+) ")
FM_RE = re.compile(r"\| Train [^|]*? prior=([-\d.e+]+)")
CRPS_RE = re.compile(r"\| CRPS ([-\d.e+]+)")
LEGEND = """\
COLUMNS (do not type)   w2 numbers x100 ("27" = 0.27), floor/post/prior as in brief.py's diag line
  #     job number (brief.py's JOBS)
  ho    the 8 held-out geometries x 256 FE results (the scored file), final checkpoint:
        floor / posterior sample / prior / N(0,I) (z drawn from N(0,I) instead of the prior)
  jt    the same, the joint-phase checkpoint (<model>.joint.pth, before the prior-only tail): post/prior
  ts    the training file's 15 TEST geometries (never seen, not used for selection) x 64: floor/post/prior
  tr    16 of its 120 TRAINING geometries x 64 (evenly spaced): floor/post/prior
        -> tr prior near tr post: the prior learned what it was shown; ts/ho far: it does not
           carry over to new geometries. tr far too: the prior is under-fitted.
  c     validation CRPS from train.log (z from the prior, the val geometries), in the unit of the
        header: best of the joint phase / best of the prior-only tail / last
  fm    the prior's flow-matching loss (train, as logged): last joint epoch / last epoch
  geom  (base, e360) per held-out geometry, the prior's W2 parts: bias (signed, in FE sd) /
        sd ratio (100 = right width) / shape (modes; 0 = same shape)
  lat   (base, e360) latent, for ho ts tr in turn. g (per-graph part): centroid offset (prior vs
        posterior, over the posterior cloud's radius; >100 = outside it) / between-geometry over
        within (100 = the geometries' clouds overlap as much as they spread) / prior sd over
        posterior sd. l (per coarse node): offset / sd ratio
  sig   (base, e360, held-out) per latent part, the encoder's sigma over the spread of the posterior
        means across one geometry's draws (>>1: the sampled z is mostly noise around the mean)
  off   (base, e360, held-out) per geometry, the prior's centroid offset x100 per part (g/l order as listed)"""


def x(v):
    return "-" if v is None or v != v else str(int(round(100 * v)))


def load(job, split):
    path = os.path.join(OUT_ROOT, job, "diag", f"posterior_vs_prior_groups_deep_{split}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def w2(d, ens):
    return None if d is None else d["scores"][ens]["w2"]["p50"]


def cfg_value(path, key):
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split("%", 1)[0].split()
                if len(parts) >= 2 and parts[0].lower() == key:
                    return parts[1]
    except OSError:
        pass
    return None


def curves(job):
    """(crps joint best, crps tail best, crps last, fm last joint, fm last) from train.log."""
    freeze = int(cfg_value(os.path.join(HERE, f"config_train_{job}.txt"), "prior_freeze_epoch") or 0)
    crps, fm = {}, {}
    try:
        with open(os.path.join(OUT_ROOT, job, "train.log"), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = EPOCH_RE.match(line)
                if not m:
                    continue
                e = int(m.group(1))
                f, c = FM_RE.search(line), CRPS_RE.search(line)
                if f:
                    fm[e] = float(f.group(1))
                if c:
                    crps[e] = float(c.group(1))
    except OSError:
        return None
    joint = [v for e, v in crps.items() if not freeze or e < freeze]
    tail = [v for e, v in crps.items() if freeze and e >= freeze]
    fm_joint = [fm[e] for e in sorted(fm) if not freeze or e < freeze]
    return (min(joint) if joint else None, min(tail) if tail else None,
            crps[max(crps)] if crps else None,
            fm_joint[-1] if fm_joint else None, fm[max(fm)] if fm else None)


def numbers():
    nums = {}
    try:
        with open(os.path.join(HERE, "jobs.txt"), encoding="utf-8") as fh:
            k = 0
            for line in fh:
                p = line.split()
                if len(p) >= 5 and not p[0].startswith("#") and p[0] == NODE:
                    k += 1
                    nums[p[1]] = str(k)
    except OSError:
        pass
    return nums


def lat_part(d, part, keys):
    if d is None:
        return "-"
    r = d.get("latent_geomean_over_groups", {}).get(part)
    if not r:
        return "-"
    return "/".join(x(r.get(k)) for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the job ids and exit")
    a = ap.parse_args()
    if a.list:
        print("\n".join(JOBS))
        return 0

    nums = numbers()
    rows, scale_vals = [], []
    for job in JOBS:
        c = curves(job)
        rows.append((job, {s: load(job, s) for s in SPLITS}, c))
        scale_vals += [v for v in (c or ())[:3] if v]
    # one unit for every crps number: the largest prints with 3 digits
    exp = 0
    if scale_vals:
        top = max(abs(v) for v in scale_vals)
        while top * 10 ** exp >= 1000:
            exp -= 1
        while top * 10 ** exp < 100:
            exp += 1
    unit = lambda v: "-" if v is None else str(int(round(v * 10 ** exp)))   # noqa: E731

    print(LEGEND)
    print()
    print("JOBS (do not type)")
    for job in JOBS:
        print(f"  {nums.get(job, '?'):>2} {job}")
    print()
    print("==== TYPE THIS BACK ====")
    print(f"DEEP {ROUND.upper()} crps x1e{-exp}")
    fails = os.path.join(OUT_ROOT, "_diag", "deep_fail.txt")
    if os.path.isfile(fails):
        with open(fails, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    print(line.rstrip())
    for job, d, c in rows:
        n = nums.get(job, job)
        if not any(d.values()) and not c:
            print(f"{n} none")
            continue
        ho, jt, ts, tr = (d[s] for s in SPLITS)
        line = f"{n} ho {x(w2(ho, 'floor'))}/{x(w2(ho, 'posterior_sample'))}/{x(w2(ho, 'prior'))}/{x(w2(ho, 'std_normal'))}"
        line += f" jt {x(w2(jt, 'posterior_sample'))}/{x(w2(jt, 'prior'))}"
        for s, dd in (("ts", ts), ("tr", tr)):
            line += f" {s} {x(w2(dd, 'floor'))}/{x(w2(dd, 'posterior_sample'))}/{x(w2(dd, 'prior'))}"
        if c:
            fm = lambda v: "-" if v is None else f"{v:.2g}"   # noqa: E731
            line += f" c {unit(c[0])}/{unit(c[1])}/{unit(c[2])} fm {fm(c[3])}/{fm(c[4])}"
        print(line)
    for job, d, _c in rows:
        if job not in DETAIL:
            continue
        n = nums.get(job, job)
        ho = d["ho"]
        if ho and ho.get("per_group_parts_median"):
            print(f"{n} geom " + " ".join(
                f"{g}:{100 * p['prior']['bias']:+.0f}/{x(p['prior']['sd_ratio'])}/{x(p['prior']['shape'])}"
                for g, p in ho["per_group_parts_median"].items()))
        if any(d[s] for s in ("ho", "ts", "tr")):
            gp = next((p for p in ("g", "z") if any(d[s] and p in d[s].get("latent_geomean_over_groups", {})
                                                     for s in ("ho", "ts", "tr"))), "g")
            g = " ".join(lat_part(d[s], gp, ("centroid_offset_mean", "between_over_within", "prior_over_post_sd"))
                         for s in ("ho", "ts", "tr"))
            l_ = " ".join(lat_part(d[s], "l", ("centroid_offset_mean", "prior_over_post_sd"))
                          for s in ("ho", "ts", "tr"))
            print(f"{n} lat {gp} {g} l {l_}")
        if ho and ho.get("latent"):
            sig = " ".join(f"{p} {r['sigma_over_sd_means']:.2g}"
                           for p, r in ho.get("latent_geomean_over_groups", {}).items()
                           if r.get("sigma_over_sd_means") is not None)
            off = " ".join(f"{v}:" + "/".join(x(ho["latent"][v][p].get("centroid_offset"))
                                             for p in sorted(ho["latent"][v]))
                           for v in ho["latent"])
            print(f"{n} sig {sig} off {off}")
    print("==== END ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
