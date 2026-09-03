#!/usr/bin/env python
"""Score the SAOI "all input" TOP/BOT models against their 3 held-out
extrapolation geometries (S26FE-MAIN, S26FE-SEC, SM-L345U-MAIN) and report
which section generalizes better where.

This directory is NOT a hyperparameter sweep -- every config shares one fixed
recipe. The only axes here are:
    section   top | bot     (which mesh half / which checkpoint)
    test_set  which held-out geometry was inferred on
so "which is better" is decided by accuracy on unseen geometries, not by
comparing training recipes.

Each infer config's `inference_output_dir` holds one HDF5 per rollout
(`rollout_sample<id>_steps<N>.h5`, or `rollout_sample<id>_vaesample<k>_steps<N>.h5`
per draw when `num_vae_samples > 1` -- see inference_profiles/rollout.py). Per
held-out geometry this script:
  * averages the draws into an ensemble mean and scores THAT against ground
    truth with relative L2 and R2 -- the point-accuracy signal "which is
    better" mostly hinges on.
  * also reports a rank_frac calibration signal: where the true "max |disp|
    across nodes" scalar falls (0..1) among the draws' values for the same
    scalar. ~0.5 and scattered = calibrated; clustered near 0 or 1 across
    geometries = biased.

Always reads the LAST stored timestep (index -1) of nodal_data, both for the
rollout output and for ground truth. rollout.py's own save convention packs
static (T=1) trajectories as [zero seed, prediction] -- i.e. 2 timesteps, not
1 -- so a T=1-vs-T>1 branch like configs/campaigns/benchmarks_all/score_rollouts.py's
(built for different models with a different on-disk convention) would score
the zero seed instead of the prediction here. Always-last sidesteps that:
correct whether ground truth stores 1 timestep (the true state alone) or 2
(mirroring the same zero-seed convention).

Usage:
    python configs/MeshGraphNets_Variational/SAOI_all_input/score_generalization.py
    python .../score_generalization.py --out-dir outputs/saoi_all_input
"""
import argparse
import glob
import json
import pathlib
import re
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
METHOD_REPO = REPO_ROOT / "methods" / "MeshGraphNets_Variational"

ARM_RE = re.compile(r"^config_infer_(?P<test_set>.+)_(?P<section>top|bot)$")
ROLLOUT_RE = re.compile(
    r"^rollout_sample(?P<sample>\d+)_(?:vaesample(?P<vae>\d+)_)?steps(?P<steps>\d+)\.h5$"
)


def parse_config(path):
    """Minimal reader for the native flat `key value` format: `%` starts a
    comment line, `#` an inline one, key/value split on the first whitespace
    run -- mirrors cae_suite/config_parser.py's own quirks."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("%") or line == "'":
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0].strip().lower()] = parts[1].strip()
    return out


def resolve(value):
    """Config paths are relative to the METHOD REPO dir, a sibling of
    dataset/output/configs at the suite root (see root CLAUDE.md)."""
    return (METHOD_REPO / value).resolve()


def discover_arms():
    arms = []
    for cfg_path in sorted(HERE.glob("config_infer_*.txt")):
        m = ARM_RE.match(cfg_path.stem)
        if not m:
            continue
        arms.append((cfg_path.stem[len("config_"):], cfg_path,
                     m.group("test_set"), m.group("section")))
    return arms


def last_state(nodal_data, num_state):
    """[F, T, N] -> (state[N, C] at the LAST stored timestep, coords[N, 3])."""
    state = nodal_data[3:3 + num_state, -1, :].T.astype(np.float64)
    coords = nodal_data[:3, 0, :].T.astype(np.float64)
    return state, coords


def load_ground_truth(gt_path, num_state):
    gt = {}
    with h5py.File(gt_path, "r") as f:
        for key in f["data"].keys():
            nodal = f[f"data/{key}/nodal_data"][:]
            state, coords = last_state(nodal, num_state)
            gt[int(key)] = (state, coords)
    return gt


def load_rollout_draws(rollout_dir, num_state):
    """sample_id -> list of state[N, C] arrays, one per VAE draw found."""
    draws = {}
    for path in glob.glob(str(rollout_dir / "rollout_sample*_steps*.h5")):
        m = ROLLOUT_RE.match(pathlib.Path(path).name)
        if not m:
            continue
        sample_id = int(m.group("sample"))
        with h5py.File(path, "r") as f:
            key = next(iter(f["data"].keys()))
            nodal = f[f"data/{key}/nodal_data"][:]
        state, coords = last_state(nodal, num_state)
        draws.setdefault(sample_id, []).append((state, coords))
    return draws


def rel_l2(pred, truth):
    denom = np.linalg.norm(truth)
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(pred - truth) / denom)


def r2(pred, truth):
    ss_res = float(np.sum((pred - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rank_frac(draw_vals, truth_val):
    """Fraction of draws below the true value -- a cheap per-geometry
    verification-rank stat (0..1). Needs >= 2 draws to mean anything."""
    if len(draw_vals) < 2:
        return None
    return float(np.mean(np.asarray(draw_vals) < truth_val))


def score_arm(arm, cfg_path, test_set, section):
    cfg = parse_config(cfg_path)
    if "infer_dataset" not in cfg or "inference_output_dir" not in cfg:
        return {"arm": arm, "test_set": test_set, "section": section,
                "error": "config missing infer_dataset/inference_output_dir"}
    num_state = int(cfg.get("output_var", cfg.get("input_var", 0)))
    if num_state <= 0:
        return {"arm": arm, "test_set": test_set, "section": section,
                "error": f"could not determine state channel count from {cfg_path}"}

    gt_path = resolve(cfg["infer_dataset"])
    rollout_dir = resolve(cfg["inference_output_dir"])
    if not gt_path.exists():
        return {"arm": arm, "test_set": test_set, "section": section,
                "error": f"ground truth not found: {gt_path}"}
    if not rollout_dir.exists():
        return {"arm": arm, "test_set": test_set, "section": section,
                "error": f"no rollout output at {rollout_dir} (run inference first)"}

    gt = load_ground_truth(gt_path, num_state)
    draws = load_rollout_draws(rollout_dir, num_state)
    if not draws:
        return {"arm": arm, "test_set": test_set, "section": section,
                "error": f"no rollout_sample*.h5 files found in {rollout_dir}"}

    geometries = []
    for sample_id, (truth_state, gt_coords) in sorted(gt.items()):
        if sample_id not in draws:
            continue
        pred_states = []
        for state, coords in draws[sample_id]:
            if state.shape[0] != truth_state.shape[0]:
                continue  # node count mismatch -- likely a stale/foreign rollout file
            pred_states.append(state)
        if not pred_states:
            continue
        stacked = np.stack(pred_states, axis=0)  # [K, N, C]
        ens_mean = stacked.mean(axis=0)

        truth_stat = float(np.abs(truth_state).max())
        draw_stats = [float(np.abs(s).max()) for s in pred_states]

        geometries.append({
            "sample_id": sample_id,
            "n_draws": len(pred_states),
            "rel_l2": rel_l2(ens_mean, truth_state),
            "r2": r2(ens_mean, truth_state),
            "rank_frac": rank_frac(draw_stats, truth_stat),
        })

    if not geometries:
        return {"arm": arm, "test_set": test_set, "section": section,
                "error": "no ground-truth/rollout sample_id overlap"}

    return {
        "arm": arm, "test_set": test_set, "section": section,
        "n_geometries": len(geometries),
        "mean_rel_l2": float(np.nanmean([g["rel_l2"] for g in geometries])),
        "mean_r2": float(np.nanmean([g["r2"] for g in geometries])),
        "mean_rank_frac": (
            float(np.nanmean([g["rank_frac"] for g in geometries if g["rank_frac"] is not None]))
            if any(g["rank_frac"] is not None for g in geometries) else None
        ),
        "geometries": geometries,
    }


def fmt(v, spec=".4f", dash="-"):
    return dash if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def render_markdown(rows):
    L = []
    L.append("# SAOI all-input -- TOP vs BOT generalization to held-out geometries")
    L.append("")
    L.append("Lower relL2 / higher R2 = better. mean_rank_frac ~ 0.5 = calibrated;")
    L.append("near 0 or 1 = the model is consistently over/under the truth across")
    L.append("its own generated draws (needs num_vae_samples > 1 to be meaningful).")
    L.append("")
    L.append("| arm | test set | section | n geometries | mean relL2 | mean R2 | mean rank_frac |")
    L.append("|---|---|---|---|---|---|---|")
    ranked = sorted((r for r in rows if not r.get("error")),
                    key=lambda r: r["mean_rel_l2"])
    for r in ranked:
        L.append(f"| {r['arm']} | {r['test_set']} | {r['section']} | {r['n_geometries']} | "
                 f"{fmt(r['mean_rel_l2'])} | {fmt(r['mean_r2'])} | {fmt(r['mean_rank_frac'], '.2f')} |")
    for r in rows:
        if r.get("error"):
            L.append(f"| {r['arm']} | {r['test_set']} | {r['section']} | **{r['error']}** | | | |")

    L.append("")
    L.append("## Section verdict per held-out test set")
    L.append("")
    by_test_set = {}
    for r in rows:
        if r.get("error"):
            continue
        by_test_set.setdefault(r["test_set"], {})[r["section"]] = r
    for test_set in sorted(by_test_set):
        pair = by_test_set[test_set]
        top, bot = pair.get("top"), pair.get("bot")
        if top and bot:
            winner = "top" if top["mean_rel_l2"] < bot["mean_rel_l2"] else "bot"
            L.append(f"- **{test_set}**: top relL2={fmt(top['mean_rel_l2'])}, "
                     f"bot relL2={fmt(bot['mean_rel_l2'])} -> **{winner} generalizes better**")
        else:
            have = "top" if top else "bot"
            L.append(f"- **{test_set}**: only `{have}` scored (the other section's run "
                     f"is missing or failed)")

    L.append("")
    L.append("## Overall")
    if ranked:
        best, worst = ranked[0], ranked[-1]
        L.append(f"- Best: `{best['arm']}` (relL2={fmt(best['mean_rel_l2'])})")
        L.append(f"- Worst: `{worst['arm']}` (relL2={fmt(worst['mean_rel_l2'])})")
        by_section = {}
        for r in rows:
            if not r.get("error"):
                by_section.setdefault(r["section"], []).append(r["mean_rel_l2"])
        if "top" in by_section and "bot" in by_section:
            top_avg = float(np.mean(by_section["top"]))
            bot_avg = float(np.mean(by_section["bot"]))
            winner = "top" if top_avg < bot_avg else "bot"
            L.append(f"- Averaged across all held-out test sets: top relL2={top_avg:.4f}, "
                     f"bot relL2={bot_avg:.4f} -> **{winner} generalizes better overall**")
    else:
        L.append("No arm scored successfully -- see the error rows above.")

    L.append("")
    L.append("Full per-geometry numbers are in generalization_results.json.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "saoi_all_input"))
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for arm, cfg_path, test_set, section in discover_arms():
        print(f"[{arm}] scoring...", flush=True)
        row = score_arm(arm, cfg_path, test_set, section)
        if row.get("error"):
            print(f"  SKIP: {row['error']}", flush=True)
        else:
            print(f"  {row['n_geometries']} geometries, mean relL2={row['mean_rel_l2']:.4f}, "
                  f"mean R2={row['mean_r2']:.4f}", flush=True)
        rows.append(row)

    json_path = out_dir / "generalization_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md = render_markdown(rows)
    md_path = out_dir / "generalization_report.md"
    md_path.write_text(md, encoding="utf-8")

    try:
        print("\n" + md)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print("\n" + md.encode(enc, "replace").decode(enc))
    print(f"\nWrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
