#!/usr/bin/env python3
"""The block to type back from diagnose_anatomy.sh (B300 round 8): the hybrid latent taken apart.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/anatomy_brief.py
    python .../r8/anatomy_brief.py --list     # the job ids diagnose_anatomy.sh runs (one per line)

Standard library only. It reads the JSONs latent_anatomy.py wrote
(output/b300_r8/<job>/diag/latent_anatomy_anat_<part>.json); each carries its
own brief lines, and this prints them under the job number and the part.
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
ROUND = os.path.basename(HERE)   # r8
NODE = "135"
OUT_ROOT = os.path.join(REPO, "output", f"b300_{ROUND}")
JOBS = ["hybrid_base", "hybrid_e360"]
# which brief lines each (job, part) prints; the first job gets the most
LINES = {
    ("hybrid_base", "ho"): ["w2", "pc", "grid", "dir", "fm"],
    ("hybrid_base", "jt"): ["w2", "dir"],
    ("hybrid_base", "ts"): ["w2", "grid", "dir"],
    ("hybrid_base", "tr"): ["w2", "grid", "dir"],
    ("hybrid_e360", "ho"): ["w2", "grid", "dir"],
    ("hybrid_e360", "jt"): ["w2"],
    ("hybrid_e360", "ts"): ["w2"],
    ("hybrid_e360", "tr"): ["w2", "dir"],
}
GEOM = ("hybrid_base", "ho")     # the per-geometry lines
PARTS = ["ho", "jt", "ts", "tr"]
LEGEND = """\
COLUMNS (do not type)   numbers x100; g = per-graph latent (3 slots x 32, the prior's token
stream), l = per coarse node latent (256 nodes x 4)
  #  job number (brief.py's JOBS); then the part:
     ho the 8 held-out geometries x 256 FE results | jt the same, joint-phase checkpoint
     ts the training file's 15 test geometries x 64 | tr 16 of its training geometries x 64
  w2   W2 (like brief.py; floor = FE half vs half): floor / mu (decode the posterior mean) /
       post (a posterior sample)
       g  g_shuffle (g of ANOTHER FE result, own l) / g_mean (g = the geometry's mean g: l only) /
          g_normal (g ~ N(0,I)) / g_prior (g from a prior draw, own l)
       l  l_shuffle (l of another FE result) / l_mean (l = the mean: g only) / l_prior (l from the prior)
       p  prior / nowrite (same noise, the token's write-back into the nodes cut) / g0 (no g noise) /
          l0 (no l noise) / N(0,I) for both / grid (K x K: K g noises x K l noises)
       -> g_shuffle ~ mu: g carries none of the draw. l_mean ~ floor-far: l carries it.
          g0 ~ prior: the g dice does not matter; l0 ~ prior: the l dice does not.
  pc   correlation with the FE value within a geometry, of: mu/post/g_shuffle/g_mean/g_normal/g_prior/
       l_shuffle/l_mean/l_prior;  ph = P(upper mode) of the FE results (2-means cut);
       d = |P(upper) - FE's| for post/prior/nowrite/g0/l0/grid;  bc = bimodality coefficient of
       the output (> 56: two modes) for FE/post/prior/nowrite
  grid share of the output's variance from the g-noise rows / the l-noise columns (null = 1/(K+1));
       wl = the same for the l latent along the decisive direction; phsd = sd of P(upper) across
       g-noise rows / its binomial null (rows >> null: the g dice picks the mode);
       wb block:early rms/late rms/early draw-sd/late draw-sd of the token's write-back over the
       node stream's (t < .5 / t >= .5; ~0 = the token does not talk to the nodes)
  dir  r2 = cross-validated R2 of the FE value from the posterior means' top PCs, g / l / joint /
       joint for the DECODED mu value (-99 = no fit);  wl per set (mu/prior/nowrite/g0/l0/N(0,I)):
       |offset| along the l direction in posterior-sample sd / sd ratio (100 = right) / 1-D W2;
       bc along it, posterior samples / prior;  yhat set:|bias predicted from the latent| /
       |actual output bias| (in FE sd) for prior/nowrite/g0
  fm   the prior's FM error on this geometry's posterior samples: l / g, l by t bin (.125 .375 .625
       .875); wl / wj = error along the decisive direction over the target's (100 = no extra miss);
       grad = the decoder's gradient: output variance from g / from l over the FE variance, under
       posterior samples (ps) and prior draws (pr); cos = |cos(gradient, decisive direction)|;
       pr l / nw / g = k-NN precision/recall vs posterior samples (prior l, nowrite l, prior g);
       PR l = participation ratio (effective dims) of posterior means / prior draws
  geom (base ho) per geometry: w2 mu/prior/nowrite/g0/l0/g_prior/l_prior | ph FE/mu/prior/nowrite/
       g0/l0 | r2 l/joint | wl prior offset (signed)/sd ratio | nw nowrite offset | grid g/l share"""


def load(job, part):
    path = os.path.join(OUT_ROOT, job, "diag", f"latent_anatomy_anat_{part}.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the job ids and exit")
    a = ap.parse_args()
    if a.list:
        print("\n".join(JOBS))
        return 0

    nums = numbers()
    print(LEGEND)
    print()
    print("JOBS (do not type)")
    for job in JOBS:
        print(f"  {nums.get(job, '?'):>2} {job}")
    print()
    print("==== TYPE THIS BACK ====")
    print(f"ANAT {ROUND.upper()}")
    fails = os.path.join(OUT_ROOT, "_diag", "anatomy_fail.txt")
    if os.path.isfile(fails):
        with open(fails, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.strip():
                    print(line.rstrip())
    for job in JOBS:
        n = nums.get(job, job)
        got = {p: load(job, p) for p in PARTS}
        if not any(got.values()):
            print(f"{n} none")
            continue
        for p in PARTS:
            d = got[p]
            if d is None:
                continue
            for key in LINES.get((job, p), ["w2"]):
                line = d.get("brief", {}).get(key)
                if line:
                    print(f"{n} {p} {line}")
        if (job, "ho") == GEOM and got["ho"]:
            for line in got["ho"].get("brief_geom", []):
                print(f"{n} {line}")
    print("==== END ====")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
