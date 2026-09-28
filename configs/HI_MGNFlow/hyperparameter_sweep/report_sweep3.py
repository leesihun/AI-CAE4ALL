#!/usr/bin/env python3
"""Sweep 3 (KL ladder): one table per half, one row per arm, end to end.

    python configs/HI_MGNFlow/hyperparameter_sweep/report_sweep3.py
    python configs/HI_MGNFlow/hyperparameter_sweep/report_sweep3.py --half bot

Columns, left to right, follow one arm down its chain:

    ch, kl        latent_ch and ae_kl_weight
    AE ep/recon   the stage-1 epoch best_by recon kept, and its validation recon
    AE kl         the per-element KL at that same epoch -- how much the latent
                  still carries. Recon going up while KL comes down is the trade
                  this sweep buys; recon going up with KL already ~0 means the
                  arm has passed the collapse point.
    ceiling       ae_ceiling_check's worst rel_recon / rel_spread on the last
                  stage-1 dump (below ~0.3 the compressor can represent the
                  spread at all; above 1 it cannot)
    PR ep/spread  the stage-2 epoch best_by recon kept and the validation
                  ensemble spread/gt printed at that epoch
    sd / bias     per evaluation part, from spread_values.npz: std(gen)/std(gt)
                  (target 1) and (mean(gen)-mean(gt))/std(gt) (target 0)
    score         mean |log sd ratio| over the parts -- rank_arms.py's score;
                  arms inside 0.02 of each other are tied

Sweep 1's promoted p8u is printed under each half as the reference row. The
stage-1/stage-2 columns come from the runner's stdout captures in run_logs/
(A.<half>_<arm>.out, B.<half>_<arm>.out), since only stdout carries KL and the
sampling pass.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import rank_arms as ra

try:
    import ae_ceiling_check as aec
except SystemExit:  # h5py missing: every column but `ceiling` still works
    aec = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep3"
SWEEP1_INFER = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep" / "infer"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "SAOI"

HALVES = ("bot", "top")
# name: (latent_ch, ae_kl_weight)
ARMS = {
    "c8k1e6": (8, "1e-6"),
    "c8k1e4": (8, "1e-4"),
    "c8k3e4": (8, "3e-4"),
    "c8k1e3": (8, "1e-3"),
    "c8k3e3": (8, "3e-3"),
    "c8k1e2": (8, "1e-2"),
    "c8k3e2": (8, "3e-2"),
    "c16k3e3": (16, "3e-3"),
}
TAGS = ra.TAGS

NUM = r"([0-9.eE+-]+|nan|inf)"
AE_EPOCH_RE = re.compile(rf"Epoch\s+(\d+)/\d+\b.*?Valid recon={NUM} kl={NUM}")
PR_EPOCH_RE = re.compile(rf"Epoch\s+(\d+)/\d+\b.*?Valid fm={NUM}(?: \| CRPS {NUM} spread {NUM})?")
SAVED_RE = re.compile(r"Model saved at epoch (\d+)")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_saved(text: str):
    saved = SAVED_RE.findall(text)
    return int(saved[-1]) if saved else None


def ae_row(log: Path):
    """(saved epoch, valid recon, valid kl) of the kept stage-1 checkpoint."""
    text = _read(log)
    curve = {int(e): (float(r), float(k)) for e, r, k in AE_EPOCH_RE.findall(text)}
    ep = last_saved(text)
    if ep is None or ep not in curve:
        if not curve:
            return None
        ep = min(curve, key=lambda e: curve[e][0])
    return (ep,) + curve[ep]


def prior_row(log: Path):
    """(saved epoch, validation spread/gt at that epoch) of the kept prior."""
    text = _read(log)
    curve = {}
    for e, _fm, _crps, spread in PR_EPOCH_RE.findall(text):
        curve[int(e)] = float(spread) if spread else None
    ep = last_saved(text)
    if ep is None:
        return None
    return ep, curve.get(ep)


def ceiling(test_root: Path, half: str, dataset_dir: Path):
    """Worst rel_recon / rel_spread over the three eval parts, or None."""
    if aec is None:
        return None
    try:
        epoch_dir = aec.latest_epoch_dir(test_root)
    except SystemExit:
        return None
    fractions, _abs, _truths, _n = aec.recon_fractions(epoch_dir)
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


def f(value, spec, width):
    return f"{'-':>{width}}" if value is None else f"{value:>{width}{spec}}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--half", choices=HALVES, action="append",
                    help="restrict to one half (repeatable)")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    run_logs = out_root / "run_logs"
    any_cell = False

    part_head = " ".join(f"{t[:13]:>15}" for t in TAGS)
    for half in args.half or HALVES:
        print(f"\n== half {half} " + "=" * 110)
        print(f"{'arm':<9} {'ch':>3} {'kl':>5} | {'AE ep':>5} {'recon':>9} {'AE kl':>9} "
              f"{'ceiling':>7} | {'PR ep':>5} {'spread':>6} | {part_head} | {'score':>6}")
        print(f"{'':<9} {'':>3} {'':>5} | {'':>5} {'':>9} {'':>9} {'':>7} | {'':>5} {'':>6} | "
              + " ".join(f"{'sd':>7} {'bias':>7}" for _ in TAGS) + f" | {'':>6}")
        rows = []
        for arm, (ch, kl) in ARMS.items():
            key = f"{half}_{arm}"
            ae = ae_row(run_logs / f"A.{key}.out")
            pr = prior_row(run_logs / f"B.{key}.out")
            ceil = ceiling(out_root / "ae" / key / "test" / "0", half, dataset_dir)
            cs = cells(out_root / "infer" / half / arm)
            any_cell = any_cell or any(c is not None for c in cs)
            rows.append((arm, ch, kl, ae, ceil, pr, cs, ra.score(cs)))

        best = min((r[7] for r in rows), default=float("inf"))
        for arm, ch, kl, ae, ceil, pr, cs, sc in rows:
            ae_ep, ae_rec, ae_kl = ae if ae else (None, None, None)
            pr_ep, pr_sp = pr if pr else (None, None)
            flags = []
            if sc != float("inf") and sc - best <= ra.TIE:
                flags.append("BEST" if sc == best else "tie")
            biases = [abs(c[1]) for c in cs if c is not None]
            if biases and max(biases) > ra.BIAS_FLAG:
                flags.append("BIAS")
            score_s = f"{sc:>6.3f}" if sc != float("inf") else f"{'-':>6}"
            print(f"{arm:<9} {ch:>3} {kl:>5} | {f(ae_ep, 'd', 5)} {f(ae_rec, '.2e', 9)} "
                  f"{f(ae_kl, '.2e', 9)} {f(ceil, '.3f', 7)} | {f(pr_ep, 'd', 5)} "
                  f"{f(pr_sp, '.3f', 6)} | {' '.join(ra.fmt_cell(c) for c in cs)} | "
                  f"{score_s} {' '.join(flags)}")

        ref = cells(SWEEP1_INFER / half / "p8u")
        if any(c is not None for c in ref):
            sc = ra.score(ref)
            score_s = f"{sc:>6.3f}" if sc != float("inf") else f"{'-':>6}"
            print(f"{'sweep1':<9} {'':>3} {'':>5} | {'':>5} {'':>9} {'':>9} {'':>7} | "
                  f"{'p8u':>5} {'':>6} | {' '.join(ra.fmt_cell(c) for c in ref)} | {score_s}"
                  "  (promoted compressor, best_by crps)")

    if not any_cell:
        print("\nNo spread_values.npz under this sweep yet -- inference has not "
              "finished for any arm. The AE/prior columns fill in as stages end.")
    if aec is None:
        print("note: h5py is not importable here, so the ceiling column is blank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
