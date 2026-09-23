#!/usr/bin/env python3
"""One table for sweep 2, read against sweep 1 wherever sweep 1 has results.

    python configs/HI_MGNFlow/hyperparameter_sweep/report_sweep2.py
    python configs/HI_MGNFlow/hyperparameter_sweep/report_sweep2.py --half bot

Sweep 2 moves the two compressor axes sweep 1 holds fixed: the coarsest-level
resolution (voronoi_clusters) and the coarsest stack depth (mp_per_level[L]).
Per board half it prints, for every sweep-2 arm and every sweep-1 compressor
that has a log:

    B            latent numbers per part = coarsest nodes x latent_ch
    best recon   lowest Valid recon in the Phase-A log (ae_report.py's reader)
    ceiling      worst ae_ceiling_check ratio over the three parts (< 0.30 is
                 green, >= 1.00 means the compressor cannot represent the
                 spread at all)
    sd / bias    per part, from spread_values.npz (rank_arms.py's reader),
                 prior p8u for every row so the prior is held fixed
    score        mean |log sd ratio| -- rank_arms.py's ranking number

HOW TO READ IT. Rows with the same B are the controlled comparisons:
n200c4 vs sweep-1 c8 (800), n400c4 vs c16 (1600), c8m16 vs c8 (800, depth
only). n400c8 (3200) is the headroom check. The recon column answers the
reconstruction question on its own; the score column says whether a better
compressor also gave a better-calibrated generator once a prior sat on it.

Missing pieces print "-" instead of failing, so this runs at any point in the
campaign.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import ae_report as aer   # noqa: E402  (local module, path just inserted)
import rank_arms as ra    # noqa: E402

try:
    import ae_ceiling_check as aec  # noqa: E402  (needs h5py)
except SystemExit:
    aec = None

REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep2"
DEFAULT_SWEEP1_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "SAOI"

HALVES = ("bot", "top")
TAGS = ra.TAGS

# name: (coarsest nodes, latent_ch, coarsest mp, what it tests)
SWEEP2 = {
    "n200c4": (200, 4, 8, "space, iso-budget with c8"),
    "n400c4": (400, 4, 8, "space, iso-budget with c16"),
    "n400c8": (400, 8, 8, "headroom past c16"),
    "c8m16": (100, 8, 16, "coarsest depth only, vs c8"),
}
SWEEP1 = {
    "c4": (100, 4, 8, "sweep 1"),
    "c8": (100, 8, 8, "sweep 1"),
    "c16": (100, 16, 8, "sweep 1"),
    "c8kl": (100, 8, 8, "sweep 1, ae_kl 1e-4"),
}
PRIOR = "p8u"


def best_recon(log_path: Path):
    if not log_path.is_file():
        return None
    curve, _cfg = aer.read_log(log_path)
    if not curve:
        return None
    return min(v for _e, v in curve)


def ceiling(test_root: Path, half: str, dataset_dir: Path):
    """Worst ceiling ratio over the eval parts, or None if anything is missing."""
    if aec is None:
        return None
    try:
        epoch_dir = aec.latest_epoch_dir(test_root)
    except SystemExit:
        return None
    fractions, _abs, _truth, _n = aec.recon_fractions(epoch_dir)
    if fractions.size == 0:
        return None
    rel_recon = float(np.sqrt(np.mean(fractions ** 2)))
    worst = None
    for stem in aec.STEMS:
        path = dataset_dir / f"test_{stem}_compare_{half}.h5"
        if not path.is_file():
            continue
        spreads = aec.compare_spreads(path)
        if spreads.size < 2 or spreads.mean() == 0.0:
            continue
        rel_spread = float(spreads.std() / abs(spreads.mean()))
        if rel_spread > 0:
            ratio = rel_recon / rel_spread
            worst = ratio if worst is None else max(worst, ratio)
    return worst


def cells(infer_root: Path):
    out = []
    for tag in TAGS:
        path = infer_root / tag / "spread_values.npz"
        out.append(ra.load_cell(path) if path.is_file() else None)
    return out


def sweep1_gpu(cfg_dir: Path, half: str, arm: str):
    """Sweep 1's AE configs pin physical cards, and its dumps are keyed by them."""
    cfg = cfg_dir / f"config_train_ae_{half}_{arm}.txt"
    if aec is None or not cfg.is_file():
        return None
    try:
        return aec.parse_gpu_ids(cfg)
    except SystemExit:
        return None


def row(name, spec, recon, ceil, cs):
    n, ch, mp, note = spec
    r_s = f"{recon:.4e}" if recon is not None else "-"
    c_s = f"{ceil:.3f}" if ceil is not None else "-"
    sc = ra.score(cs)
    s_s = f"{sc:.3f}" if math.isfinite(sc) else "-"
    body = "  ".join(ra.fmt_cell(c) for c in cs)
    flag = ""
    if any(c is not None and abs(c[1]) > ra.BIAS_FLAG for c in cs):
        flag = "  BIAS"
    return (f"  {name:<7} {n:>4} {ch:>4} {mp:>3} {n * ch:>5}  {r_s:>11} {c_s:>7}  "
            f"{body}  {s_s:>6}{flag}   {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--sweep1-root", default=str(DEFAULT_SWEEP1_ROOT))
    ap.add_argument("--config-dir", default=str(SCRIPT_DIR))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--half", choices=HALVES)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    s1_root = Path(args.sweep1_root).resolve()
    cfg_dir = Path(args.config_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    halves = (args.half,) if args.half else HALVES

    if aec is None:
        print("note: h5py is not importable here, so the ceiling column is blank.")

    tag_head = "  ".join(f"{t[:15]:>15}" for t in TAGS)
    for half in halves:
        print(f"\n=== half {half}  (prior {PRIOR} on every row; per part: sd ratio, bias)")
        print(f"  {'arm':<7} {'n_c':>4} {'ch':>4} {'mp':>3} {'B':>5}  "
              f"{'best recon':>11} {'ceil':>7}  {tag_head}  {'score':>6}")
        for arm, spec in SWEEP2.items():
            recon = best_recon(out_root / "ae" / f"{half}_{arm}" / "ae.log")
            ceil = ceiling(out_root / "ae" / f"{half}_{arm}" / "test" / "0", half, dataset_dir)
            cs = cells(out_root / "infer" / half / arm)
            print(row(arm, spec, recon, ceil, cs))
        print("  -- sweep 1 --")
        for arm, spec in SWEEP1.items():
            recon = best_recon(s1_root / "ae" / f"{half}.ae_{arm}.log")
            gpu = sweep1_gpu(cfg_dir, half, arm)
            ceil = ceiling(s1_root / "ae" / "test" / gpu, half, dataset_dir) if gpu else None
            # Sweep 1 trains its priors on ONE promoted compressor, so its
            # calibration columns belong to whichever arm select_ae.py chose.
            cs = [None, None, None]
            print(row(arm, spec, recon, ceil, cs))
        s1_cells = cells(s1_root / "infer" / half / PRIOR)
        sc = ra.score(s1_cells)
        body = "  ".join(ra.fmt_cell(c) for c in s1_cells)
        s_s = f"{sc:.3f}" if math.isfinite(sc) else "-"
        print(f"  {'promoted':<7} {'':>4} {'':>4} {'':>3} {'':>5}  {'':>11} {'':>7}  "
              f"{body}  {s_s:>6}   sweep 1 {PRIOR} on its selected compressor")

    print("\nSame-B pairs are the controlled reads: n200c4|c8, n400c4|c16, c8m16|c8.")
    print(f"Scores inside {ra.TIE} of each other are tied (2000-draw noise).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
