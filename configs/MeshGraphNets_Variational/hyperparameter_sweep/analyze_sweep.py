#!/usr/bin/env python
"""Read the MGN-V SAOI hyperparameter sweep and say what it actually showed.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/analyze_sweep.py \
        [--out-root output/meshgraphnets-v/saoi_sweep] [--arms ...] [--tags ...] \
        [--json report.json] [--figure]

WHAT IT READS  (nothing is recomputed from the model; both are run artifacts)

    <out-root>/infer/<arm>/<tag>/spread_values.npz
        gt        peak-to-valley spread of each real realization of the part
        gen       one spread per generated rollout
        gen_lam   the latent_inflation factor that produced each gen value
        *_scene   which part each value came from
    <out-root>/diag/posterior_vs_prior_<arm>_<tag>.json
        ensembles.{truth,posterior_mean,posterior_sample,prior}.sd_ratio
        latent.pca, latent.overall_sd_ratio_rms, verdict

THE THREE QUESTIONS, IN THE ORDER THEY HAVE TO BE ASKED

    1. Is anything here bigger than noise?  Arm `seed` differs from `base` only
       in training_seed, so base-vs-seed is a pure replicate. The largest
       base-vs-seed gap across the eval sets is the floor. An arm that moves
       sd_ratio by less than that has shown nothing, however tidy the ordering
       looks. The retired 8-arm ablation had no such arm, which is why its
       sub-2% W1/sd result could never be read.

    2. Is the defect in the prior or in the decoder?  sd_ratio alone cannot
       say. The PVP decomposition can: if decoding a posterior SAMPLE already
       loses the width, no amount of prior work can put it back, and every
       prior-side arm in this sweep is answering the wrong question. Read
       table 3 before table 2 means anything.

    3. How much can post-hoc rescaling buy?  latent_inflation widens the latent
       cloud around its per-graph center after integration. Prior art on this
       model: 2.5x on z bought 1.35x on the field -- sub-linear, because the
       prior puts variance in directions the decoder does not read. The
       efficiency column is that number per arm, and an arm that raises it has
       improved the ALIGNMENT of prior and decoder, which is the mechanism this
       sweep is actually hunting.

Target for every sd_ratio here is exactly 1.0: each `_compare_` file holds the
physical realizations of the one part in the matching `_infer_` file, so the
generated ensemble should be exactly as wide as the real one. This is a
calibration measurement, not an accuracy measurement.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_ARMS = ["base", "seed", "zdim8", "zdim4", "pmin15", "pmin30",
                "g2e", "mmd10", "arecon"]
DEFAULT_TAGS = ["s26fe_main", "s26fe_sec", "sm_l345u_main"]
BASE, SEED = "base", "seed"

# What each arm changed, so the report stands on its own in a log file.
WHAT = {
    "base":   "SAOI_run recipe verbatim (reference)",
    "seed":   "training_seed only -> THE NOISE FLOOR",
    "zdim8":  "vae_latent_dim 16 -> 8",
    "zdim4":  "vae_latent_dim 16 -> 4",
    "pmin15": "posterior_min_std 0.05 -> 0.15",
    "pmin30": "posterior_min_std 0.05 -> 0.30",
    "g2e":    "prior_grad_to_encoder 0.0 -> 1.0",
    "mmd10":  "lambda_mmd 1 -> 10",
    "arecon": "alpha_recon 1000 -> 100",
    # sweep2 (run_sweep2.sh) -- read against sweep1's base/seed
    "aux0":   "beta_aux 10 -> 0",
    "aux3":   "beta_aux 10 -> 3",
    "aux30":  "beta_aux 10 -> 30",
    "aux100": "beta_aux 10 -> 100",
    "mmd100": "lambda_mmd 1 -> 100",
    "vel512": "prior_velocity_hidden_dim 256 -> 512",
    "fmmom":  "prior_fm_moments False -> True",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def pooled_sd(values, scenes):
    """Within-scene standard deviation.

    Each eval file is one part, so scenes is normally constant and this is the
    plain sd. Pooling is kept because between-scene variation of the
    conditional mean is a different quantity from ensemble width, and silently
    mixing the two is the failure mode the scene labels exist to prevent.
    """
    values = np.asarray(values, dtype=np.float64)
    if scenes is None or len(np.unique(scenes)) <= 1:
        return float(values.std(ddof=0)), 1
    uniq = np.unique(scenes)
    var = np.mean([values[scenes == s].var(ddof=0) for s in uniq])
    return float(math.sqrt(var)), int(uniq.size)


def load_spreads(out_root: Path, arm: str, tag: str):
    path = out_root / "infer" / arm / tag / "spread_values.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    gt = np.asarray(d["gt"], dtype=np.float64)
    gen = np.asarray(d["gen"], dtype=np.float64)
    lam = np.asarray(d["gen_lam"], dtype=np.float64) if "gen_lam" in d else np.ones_like(gen)
    gen_scene = d["gen_scene"] if "gen_scene" in d else None
    gt_scene = d["gt_scene"] if "gt_scene" in d else None

    gt_sd, gt_scenes = pooled_sd(gt, gt_scene)
    out = {"path": str(path), "gt_sd": gt_sd, "gt_mean": float(gt.mean()),
           "gt_n": int(gt.size), "gt_scenes": gt_scenes, "by_lam": {}}
    for value in np.unique(lam):
        sel = lam == value
        sub = gen[sel]
        sub_scene = gen_scene[sel] if gen_scene is not None else None
        sd, _ = pooled_sd(sub, sub_scene)
        out["by_lam"][float(value)] = {
            "n": int(sub.size),
            "sd_ratio": sd / gt_sd if gt_sd > 0 else float("nan"),
            "bias": (float(sub.mean()) - out["gt_mean"]) / gt_sd if gt_sd > 0 else float("nan"),
        }
    return out


def load_pvp(out_root: Path, arm: str, tag: str):
    path = out_root / "diag" / f"posterior_vs_prior_{arm}_{tag}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    ens, lat = d.get("ensembles", {}), d.get("latent", {})
    pca = lat.get("pca", {})
    return {
        "path": str(path),
        "post_mu": ens.get("posterior_mean", {}).get("sd_ratio", float("nan")),
        "post_z": ens.get("posterior_sample", {}).get("sd_ratio", float("nan")),
        "prior": ens.get("prior", {}).get("sd_ratio", float("nan")),
        "prior_bias": ens.get("prior", {}).get("dmean_over_sd", float("nan")),
        "z_sd_ratio": lat.get("overall_sd_ratio_rms", float("nan")),
        "post_sigma": lat.get("posterior_sigma_rms", float("nan")),
        "pca_post_frac": pca.get("post_var_frac_in_top_k", float("nan")),
        "pca_prior_frac": pca.get("prior_var_frac_in_top_k", float("nan")),
        "verdict": d.get("verdict", []),
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def classify_pvp(post_mu, post_z, prior):
    """The same thresholds posterior_vs_prior.py uses, applied to tag means."""
    if not all(np.isfinite([post_mu, post_z, prior])):
        return "n/a"
    post = max(post_mu, post_z)
    if post >= 0.8 and prior <= 0.7 * post:
        return "PRIOR-BOUND"
    if post < 0.8 and prior <= post + 0.1:
        return "DECODER-BOUND"
    if prior > post + 0.1:
        return "PRIOR-WIDER"
    return "MIXED"


def mean_or_nan(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(value, width=7, prec=3):
    if value is None or not np.isfinite(value):
        return "-".rjust(width)
    return f"{value:{width}.{prec}f}"


def rule(char="-", n=78):
    return char * n


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default="output/meshgraphnets-v/saoi_sweep")
    ap.add_argument("--arms", default=" ".join(DEFAULT_ARMS),
                    help="space-separated arm names")
    ap.add_argument("--tags", default=" ".join(DEFAULT_TAGS),
                    help="space-separated eval-set tags")
    ap.add_argument("--json", default=None, help="also dump every number here")
    ap.add_argument("--figure", action="store_true",
                    help="write sweep_report.png beside the JSON (needs matplotlib)")
    a = ap.parse_args()

    out_root = Path(a.out_root)
    arms = a.arms.split()
    tags = a.tags.split()

    spreads = {arm: {t: load_spreads(out_root, arm, t) for t in tags} for arm in arms}
    pvps = {arm: {t: load_pvp(out_root, arm, t) for t in tags} for arm in arms}

    print()
    print(rule("="))
    print(" MGN-V SAOI hyperparameter sweep -- report")
    print(rule("="))
    print(f"  root : {out_root}")
    print(f"  arms : {len(arms)}   eval sets : {len(tags)}")
    print("  target sd_ratio is exactly 1.000 (same part, same condition)")

    # ---------------------------------------------------------------- coverage
    missing_s = [(arm, t) for arm in arms for t in tags if spreads[arm][t] is None]
    missing_p = [(arm, t) for arm in arms for t in tags if pvps[arm][t] is None]
    print()
    print("1. COVERAGE")
    print(f"   spread_values.npz : {len(arms) * len(tags) - len(missing_s)} / {len(arms) * len(tags)}")
    print(f"   posterior_vs_prior: {len(arms) * len(tags) - len(missing_p)} / {len(arms) * len(tags)}")
    for label, miss in (("spreads", missing_s), ("pvp", missing_p)):
        if miss:
            joined = ", ".join(f"{arm}/{t}" for arm, t in miss[:10])
            more = f" (+{len(miss) - 10} more)" if len(miss) > 10 else ""
            print(f"   MISSING {label}: {joined}{more}")

    have_base = any(spreads.get(BASE, {}).get(t) for t in tags)
    have_seed = any(spreads.get(SEED, {}).get(t) for t in tags)

    # per-lam count balance: the list is cycled per BATCH, so an unlucky batch
    # count can leave one lam with far fewer draws than the others.
    for arm in arms:
        for t in tags:
            s = spreads[arm][t]
            if not s:
                continue
            counts = [v["n"] for v in s["by_lam"].values()]
            if counts and min(counts) < 0.5 * max(counts):
                print(f"   NOTE {arm}/{t}: inflation draws are unbalanced "
                      f"(min {min(counts)}, max {max(counts)}) -- lam cycles per batch, "
                      f"so a small batch count skews it; read those sd_ratios with care.")

    # -------------------------------------------------------------- noise floor
    print()
    print("2. NOISE FLOOR  (base vs seed at lam=1.0; nothing smaller is an effect)")
    floor_per_tag, floor = {}, float("nan")
    if have_base and have_seed:
        print(f"   {'eval set':<16}{'base':>9}{'seed':>9}{'|diff|':>9}")
        print("   " + rule("-", 43))
        for t in tags:
            sb, ss = spreads[BASE][t], spreads[SEED][t]
            if not sb or not ss:
                continue
            b = sb["by_lam"].get(1.0, {}).get("sd_ratio", float("nan"))
            s = ss["by_lam"].get(1.0, {}).get("sd_ratio", float("nan"))
            d = abs(b - s) if np.isfinite(b) and np.isfinite(s) else float("nan")
            floor_per_tag[t] = d
            print(f"   {t:<16}{fmt(b, 9)}{fmt(s, 9)}{fmt(d, 9)}")
        finite = [v for v in floor_per_tag.values() if np.isfinite(v)]
        if finite:
            floor = max(finite)
            print("   " + rule("-", 43))
            print(f"   floor = max over eval sets = {floor:.3f}  "
                  f"(mean {np.mean(finite):.3f})")
            print("   Conservative on purpose: one replicate gives a range, not a")
            print("   standard error, so the widest observed gap is the honest bar.")
    else:
        print("   UNAVAILABLE -- arm `base` and/or `seed` has no results.")
        print("   Without both, table 3 below is still readable but table 4 is not:")
        print("   there is no way to tell a 5% ordering from 5% seed scatter.")

    # ------------------------------------------------------- PVP decomposition
    print()
    print("3. WHERE THE WIDTH IS LOST  (posterior_vs_prior, mean over eval sets)")
    print("   post_mu / post_z: decode the encoder output. prior: decode p(z|g).")
    print("   If post_z is already far below 1.0 the decoder is the bottleneck")
    print("   and NO prior-side arm in this sweep can help.")
    print()
    print(f"   {'arm':<9}{'post_mu':>9}{'post_z':>9}{'prior':>9}{'z sd':>8}"
          f"{'pca p':>7}{'pca q':>7}  verdict")
    print("   " + rule("-", 72))
    pvp_mean = {}
    for arm in arms:
        rows = [pvps[arm][t] for t in tags if pvps[arm][t]]
        if not rows:
            print(f"   {arm:<9}{'-':>9}{'-':>9}{'-':>9}{'-':>8}{'-':>7}{'-':>7}  (no data)")
            continue
        m = {k: mean_or_nan([r[k] for r in rows])
             for k in ("post_mu", "post_z", "prior", "z_sd_ratio",
                       "pca_post_frac", "pca_prior_frac", "post_sigma")}
        m["class"] = classify_pvp(m["post_mu"], m["post_z"], m["prior"])
        pvp_mean[arm] = m
        print(f"   {arm:<9}{fmt(m['post_mu'], 9)}{fmt(m['post_z'], 9)}"
              f"{fmt(m['prior'], 9)}{fmt(m['z_sd_ratio'], 8, 2)}"
              f"{fmt(m['pca_post_frac'], 7, 2)}{fmt(m['pca_prior_frac'], 7, 2)}"
              f"  {m['class']}")
    print()
    print("   pca p / pca q: fraction of the POSTERIOR cloud variance, and of the")
    print("   PRIOR variance, lying in the posterior top principal directions.")
    print("   pca q well below pca p is the alignment defect -- the prior spends")
    print("   its variance where the decoder does not look, which is exactly why")
    print("   isotropic inflation is sub-linear (table 5).")

    # ------------------------------------------------------------ arm ranking
    print()
    print("4. ARM RANKING at lam=1.0  (the honest, un-inflated sample)")
    print(f"   {'arm':<9}" + "".join(f"{t[:11]:>12}" for t in tags)
          + f"{'mean':>8}{'d vs base':>11}  effect")
    print("   " + rule("-", 76))
    base_per_tag = {}
    if have_base:
        for t in tags:
            s = spreads[BASE][t]
            base_per_tag[t] = (s["by_lam"].get(1.0, {}).get("sd_ratio", float("nan"))
                               if s else float("nan"))
    ranking = {}
    for arm in arms:
        per_tag, deltas = [], []
        for t in tags:
            s = spreads[arm][t]
            v = s["by_lam"].get(1.0, {}).get("sd_ratio", float("nan")) if s else float("nan")
            per_tag.append(v)
            b = base_per_tag.get(t, float("nan"))
            deltas.append(v - b if np.isfinite(v) and np.isfinite(b) else float("nan"))
        mean_v = mean_or_nan(per_tag)
        mean_d = mean_or_nan(deltas)
        finite_d = [d for d in deltas if np.isfinite(d)]
        same_sign = bool(finite_d) and (all(d > 0 for d in finite_d) or all(d < 0 for d in finite_d))

        if arm == BASE:
            note = "(reference)"
        elif arm == SEED:
            note = "(this IS the floor)"
        elif not np.isfinite(mean_d) or not np.isfinite(floor):
            note = "no floor"
        elif len(finite_d) < len(tags):
            # Replication across the three parts IS the test. An arm missing a
            # part has not passed a weaker version of it -- it has not taken it.
            note = f"incomplete ({len(finite_d)}/{len(tags)} parts)"
        elif not same_sign:
            note = "inconsistent sign"
        elif abs(mean_d) > floor:
            note = "EFFECT " + ("(wider)" if mean_d > 0 else "(narrower)")
        else:
            note = "within noise"

        ranking[arm] = {"per_tag": per_tag, "mean": mean_v, "delta": mean_d,
                        "same_sign": same_sign, "note": note}
        print(f"   {arm:<9}" + "".join(fmt(v, 12) for v in per_tag)
              + fmt(mean_v, 8) + fmt(mean_d, 11) + f"  {note}")
    print()
    print("   An effect must clear the floor AND move the same direction on all")
    print("   three parts. One part out of three disagreeing is not a weak effect;")
    print("   it is the absence of one.")

    # -------------------------------------------------------- inflation curve
    print()
    print("5. INFLATION CURVE  (free: one inference pass produced every column)")
    lams = sorted({l for arm in arms for t in tags
                   if spreads[arm][t] for l in spreads[arm][t]["by_lam"]})
    if lams:
        print(f"   {'arm':<9}" + "".join(f"{('lam ' + f'{l:g}'):>9}" for l in lams)
              + f"{'eff':>7}{'lam->1.0':>10}")
        print("   " + rule("-", 76))
        for arm in arms:
            curve = []
            for l in lams:
                vals = [spreads[arm][t]["by_lam"][l]["sd_ratio"]
                        for t in tags
                        if spreads[arm][t] and l in spreads[arm][t]["by_lam"]]
                curve.append(mean_or_nan(vals))
            good = [(l, v) for l, v in zip(lams, curve) if np.isfinite(v)]
            eff, lam1 = float("nan"), float("nan")
            if len(good) >= 2:
                (l0, v0), (lN, vN) = good[0], good[-1]
                if v0 > 0 and lN > 0:
                    eff = (vN / v0) / (lN / l0)
                xs = np.array([g[0] for g in good])
                ys = np.array([g[1] for g in good])
                slope, intercept = np.polyfit(xs, ys, 1)
                if slope > 0:
                    lam1 = (1.0 - intercept) / slope
            flag = "*" if np.isfinite(lam1) and lam1 > max(lams) else " "
            ranking.setdefault(arm, {})["curve"] = curve
            ranking[arm]["efficiency"] = eff
            ranking[arm]["lam_to_unity"] = lam1
            print(f"   {arm:<9}" + "".join(fmt(v, 9) for v in curve)
                  + fmt(eff, 7, 2) + fmt(lam1, 9, 2) + flag)
        print()
        print("   eff = (sd_ratio gain) / (lam gain). 1.00 would mean the field")
        print("   widens exactly as fast as the latent. Prior art on this model is")
        print("   ~0.54 (2.5x on z bought 1.35x on the field). An arm that RAISES")
        print("   eff has aligned the prior with the directions the decoder reads,")
        print("   which is the mechanism -- not the ranking -- worth keeping.")
        print("   * = lam->1.0 is extrapolated past the largest lam tested, so it")
        print("   is an order-of-magnitude statement, not a setting to adopt.")
    else:
        print("   no spread data")

    # -------------------------------------------------------------- the verdict
    print()
    print(rule("="))
    print(" READING")
    print(rule("="))
    base_cls = pvp_mean.get(BASE, {}).get("class", "n/a")
    if base_cls == "DECODER-BOUND":
        print(" Baseline is DECODER-BOUND: decoding a posterior sample already")
        print(" loses the width, so the prior merely inherits a deficit it did not")
        print(" create. Every prior-side arm here (g2e, mmd10, and the prior half")
        print(" of arecon) was aimed at the wrong stage. Weight for the next round:")
        print(" decoder capacity, a peak-to-valley loss term, spatial latents.")
        print(" Note the objective has NO term penalizing a narrow ensemble --")
        print(" crps is only checkpoint selection and carries no gradient.")
    elif base_cls == "PRIOR-BOUND":
        print(" Baseline is PRIOR-BOUND: the encoder and decoder do carry the")
        print(" width and the prior does not reach it. The prior-side arms are")
        print(" answering the right question; take the ranking in table 4 at face")
        print(" value and follow the winning axis further.")
    elif base_cls == "MIXED":
        print(" Baseline is MIXED: both stages lose some. Read the latent columns")
        print(" in table 3 -- if pca q sits well below pca p, the loss is an")
        print(" alignment problem rather than a scale problem in either stage.")
    elif base_cls == "PRIOR-WIDER":
        print(" The prior path is WIDER than the posterior path. Check the bias")
        print(" column before reading that as good: a prior that wanders off the")
        print(" conditional mean also widens the histogram.")
    else:
        print(" No baseline PVP result, so the prior-vs-decoder question is still")
        print(" open and table 4 cannot be acted on. Run the PVP stage first.")

    effects = [arm for arm in arms
               if arm not in (BASE, SEED) and ranking.get(arm, {}).get("note", "").startswith("EFFECT")]
    print()
    if not np.isfinite(floor):
        print(" No noise floor was measured, so no ranking claim is supported.")
    elif effects:
        print(f" Cleared the floor on all three parts: {', '.join(effects)}")
        for arm in effects:
            r = ranking[arm]
            print(f"   {arm:<8} {WHAT.get(arm, ''):<38} d={r['delta']:+.3f}")
    else:
        print(" NO arm cleared the noise floor on all three parts. With the floor")
        print(f" at {floor:.3f}, this sweep says the tested axes do not move")
        print(" calibration -- which is a result, not a failed run. Do not rerun it")
        print(" with more arms on the same axes.")

    # ------------------------------------------------------------------ dumps
    payload = {"out_root": str(out_root), "arms": arms, "tags": tags,
               "floor": floor, "floor_per_tag": floor_per_tag,
               "ranking": ranking, "pvp_mean": pvp_mean,
               "baseline_class": base_cls,
               "missing": {"spreads": missing_s, "pvp": missing_p}}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=float)
        print(f"\n wrote {a.json}")

    if a.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:                       # lazy by repo convention
            print(f"\n figure skipped: {exc}")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            for arm in arms:
                curve = ranking.get(arm, {}).get("curve")
                if curve and any(np.isfinite(v) for v in curve):
                    ax1.plot(lams, curve, marker="o",
                             lw=2.5 if arm == BASE else 1.4,
                             ls="--" if arm == SEED else "-", label=arm)
            ax1.axhline(1.0, color="k", lw=1, ls=":")
            ax1.set_xlabel("latent_inflation")
            ax1.set_ylabel("sd_ratio (target 1.0)")
            ax1.set_title("Inflation curve")
            ax1.legend(fontsize=8, ncol=2)
            names = [arm for arm in arms if arm in pvp_mean]
            width = 0.27
            xs = np.arange(len(names))
            for off, key, lbl in ((-width, "post_mu", "posterior mean"),
                                  (0.0, "post_z", "posterior sample"),
                                  (width, "prior", "prior")):
                ax2.bar(xs + off, [pvp_mean[n][key] for n in names], width, label=lbl)
            ax2.axhline(1.0, color="k", lw=1, ls=":")
            ax2.set_xticks(xs)
            ax2.set_xticklabels(names, rotation=45, ha="right")
            ax2.set_ylabel("sd_ratio")
            ax2.set_title("Where the width is lost")
            ax2.legend(fontsize=8)
            fig.tight_layout()
            png = out_root / "sweep_report.png"
            fig.savefig(png, dpi=150)
            print(f"\n figure {png}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
