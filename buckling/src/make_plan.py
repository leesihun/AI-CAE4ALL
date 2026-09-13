"""Emit the production geometry plan as JSON.

Four tiers, and the split between them is the actual research claim:

  train  -- cylinders on a grid inside the tractable box. The model sees many
            draws per geometry, so it can learn p(field | geometry) here.
  t1     -- cylinders with R/t or L/R OUTSIDE the training range. Same family,
            extrapolated parameters.
  t2     -- cones. A different shape entirely; the imperfection law is
            unchanged, so a model that learned the LAW rather than memorising
            the training modes should still place the distribution correctly.
  t3     -- axially varying thickness. The critical wavenumber now varies along
            the shell, which no training geometry exhibits.

Tiers t1..t3 are inference-only. They are the transferability test the whole
benchmark exists to pose: a distribution learned on AA must predict the
distribution on BB.
"""
import argparse
import json
import sys

import numpy as np


def cyl(rt, lr, n, seed0, tier, shorten=1.5):
    return dict(shape="cylinder", r_over_t=rt, l_over_r=lr, alpha_deg=0.0,
                thickness_gamma=0.0, n_draws=n, seed0=seed0, tier=tier,
                shorten=shorten)


def cone(rt, lr, alpha, n, seed0, tier, shorten=1.5):
    return dict(shape="cone", r_over_t=rt, l_over_r=lr, alpha_deg=alpha,
                thickness_gamma=0.0, n_draws=n, seed0=seed0, tier=tier,
                shorten=shorten)


def taper(rt, lr, gamma, n, seed0, tier, shorten=1.5):
    return dict(shape="cylinder", r_over_t=rt, l_over_r=lr, alpha_deg=0.0,
                thickness_gamma=gamma, n_draws=n, seed0=seed0, tier=tier,
                shorten=shorten)


def build(n_train, n_ood, lr_max, shorten_slender):
    """lr_max caps the training box; shorten_slender is the reduced target
    applied to the slender geometries if the hard-corner test said we need it."""
    plan, s = [], 1000

    RT_TRAIN = [110, 140, 170, 200]
    LR_TRAIN = [x for x in (0.8, 1.0, 1.3) if x <= lr_max]

    for rt in RT_TRAIN:
        for lr in LR_TRAIN:
            sh = shorten_slender if (rt >= 200 and lr >= 1.3) else 1.5
            plan.append(cyl(rt, lr, n_train, s, "train", sh))
            s += n_train

    # t1: parameter extrapolation, both directions, still cylinders
    for rt, lr in ((95, 1.0), (245, 1.0), (140, 0.6), (140, lr_max + 0.2)):
        sh = shorten_slender if (rt >= 200 or lr > 1.3) else 1.5
        plan.append(cyl(rt, lr, n_ood, s, "t1", sh))
        s += n_ood

    # t2: cones -- the headline out-of-distribution family.
    # Angles and base R/t are chosen so the LOCAL R/t stays inside the training
    # range at every axial station (G4, docs/BUILD_LOG.md). That makes this a
    # pure structural-novelty test: the model has seen every local R/t value it
    # meets here, it has just never seen one VARY along the axis. Confounding
    # it with parameter extrapolation would make a failure unattributable.
    for alpha, rt in ((10.0, 115), (20.0, 115), (10.0, 140), (20.0, 140)):
        plan.append(cone(rt, 1.0, alpha, n_ood, s, "t2"))
        s += n_ood

    # t3: axially varying thickness. Same rationale as t2 -- local R/t spans
    # 112..140 at gamma=0.25, entirely inside the training range.
    for gamma in (0.15, 0.25):
        plan.append(taper(140, 1.0, gamma, n_ood, s, "t3"))
        s += n_ood

    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-ood", type=int, default=32)
    ap.add_argument("--lr-max", type=float, default=1.3)
    ap.add_argument("--shorten-slender", type=float, default=1.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    plan = build(a.n_train, a.n_ood, a.lr_max, a.shorten_slender)
    json.dump(plan, open(a.out, "w"), indent=1)

    tot = sum(e["n_draws"] for e in plan)
    print("wrote %s" % a.out)
    print("%d geometries, %d draws" % (len(plan), tot))
    for t in ("train", "t1", "t2", "t3"):
        es = [e for e in plan if e["tier"] == t]
        print("  %-6s %2d geometries  %4d draws" % (t, len(es),
                                                    sum(e["n_draws"] for e in es)))
    print()
    print("%-8s %-10s %-7s %-7s %-7s %s" % ("tier", "shape", "R/t", "L/R",
                                            "extra", "draws"))
    for e in plan:
        extra = ("alpha=%.0f" % e["alpha_deg"] if e["alpha_deg"] else
                 ("gamma=%.2f" % e["thickness_gamma"] if e["thickness_gamma"]
                  else "-"))
        print("%-8s %-10s %-7d %-7.1f %-7s %d%s"
              % (e["tier"], e["shape"], e["r_over_t"], e["l_over_r"], extra,
                 e["n_draws"], "" if e["shorten"] == 1.5 else
                 "   (shorten %.2f)" % e["shorten"]))


if __name__ == "__main__":
    main()
