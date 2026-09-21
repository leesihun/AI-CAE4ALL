#!/usr/bin/env python3
"""Rank the Phase-C arms by how well their draws reproduce the real spread.

    python configs/HI_MGNFlow/hyperparameter_sweep/rank_arms.py
    python configs/HI_MGNFlow/hyperparameter_sweep/rank_arms.py --half bot

Reads the spread_values.npz that rollout.py writes beside each histogram and
tabulates, per arm and evaluation part:

    sd ratio = std(gen) / std(gt)          target 1
    bias     = (mean(gen) - mean(gt)) / std(gt)   target 0, in GT sigmas

This is a CALIBRATION measurement, not an accuracy one. Each _infer_ file is
one part and its _compare_ file holds that same part's 125 physical
realizations, so the condition is fixed and the only question is whether the
2000 draws spread the way the real boards do. The previous campaign found both
cHI-MGNflow and MGN-V about 2x too narrow -- an sd ratio near 0.5 -- which is
the number this sweep exists to move.

RANKING. Arms are ordered by the mean of |log(sd ratio)| over the three parts.
The log is what makes 2x-too-narrow and 2x-too-wide count equally; a plain
difference would quietly prefer the narrow side. Bias is reported but not
ranked on, because a prior can be well centred and still useless if its width
is wrong -- though an arm winning on width while carrying a bias above about
one GT sigma is flagged, since it is then matching the width of a distribution
sitting somewhere else.

READING THE MARGINS. std(gt) is estimated from the same 125 realizations for
every arm, so its sampling error (about 6% on a log scale) shifts the whole
column together and cancels out of the ranking. What does not cancel is the
generated side, 2000 draws, worth about 1.6%. Two arms inside roughly 0.02 of
each other in mean |log sd ratio| are tied.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep"
DEFAULT_BASELINE = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_run" / "infer"

HALVES = ("bot", "top")
ARMS = ("p4u", "p8u", "p4x", "p8x")
TAGS = ("s26fe_main", "s26fe_sec", "sm_l345u_main")

ARM_DESC = {
    "p4u": "prior_blocks 4, flow_loss_weighting uniform",
    "p8u": "prior_blocks 8, flow_loss_weighting uniform",
    "p4x": "prior_blocks 4, flow_loss_weighting x0",
    "p8x": "prior_blocks 8, flow_loss_weighting x0",
}

# Below this the two arms are inside the 2000-draw sampling noise on std(gen).
TIE = 0.02
BIAS_FLAG = 1.0


def load_cell(path: Path):
    """(sd_ratio, bias, n_gt, n_gen) from one spread_values.npz."""
    with np.load(path, allow_pickle=False) as data:
        if "gt" not in data or "gen" not in data:
            return None
        gt = np.asarray(data["gt"], dtype=np.float64)
        gen = np.asarray(data["gen"], dtype=np.float64)
    if gt.size < 2 or gen.size < 2:
        return None
    sd_gt = float(gt.std())
    if not math.isfinite(sd_gt) or sd_gt <= 0.0:
        return None
    sd_ratio = float(gen.std()) / sd_gt
    bias = (float(gen.mean()) - float(gt.mean())) / sd_gt
    return sd_ratio, bias, gt.size, gen.size


def score(cells) -> float:
    """Mean |log sd ratio| over the parts that produced a value."""
    logs = [abs(math.log(c[0])) for c in cells if c is not None and c[0] > 0]
    return sum(logs) / len(logs) if logs else float("inf")


def fmt_cell(cell) -> str:
    if cell is None:
        return f"{'-':>7} {'-':>7}"
    return f"{cell[0]:>7.3f} {cell[1]:>+7.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--half", choices=HALVES, action="append",
                    help="restrict to one half (repeatable)")
    ap.add_argument("--baseline", default=str(DEFAULT_BASELINE),
                    help="SAOI_run infer directory to compare against, or '' "
                         "to skip (default: the un-swept recipe)")
    args = ap.parse_args()

    infer_root = Path(args.out_root).resolve() / "infer"
    halves = args.half or list(HALVES)

    print("Phase C -- spread calibration by arm")
    print(f"  results: {infer_root}")
    print("  sd ratio target 1 (>1 too wide, <1 too narrow); "
          "bias in GT sigmas, target 0\n")

    if not infer_root.is_dir():
        print(f"FAIL: {infer_root} does not exist. Has Phase C run?")
        return 1

    found_any = False
    winners = {}

    for half in halves:
        table = {}
        for arm in ARMS:
            cells = []
            for tag in TAGS:
                path = infer_root / half / arm / tag / "spread_values.npz"
                cells.append(load_cell(path) if path.is_file() else None)
            table[arm] = cells
            if any(c is not None for c in cells):
                found_any = True

        header = f"    {'arm':<6}"
        for tag in TAGS:
            header += f"  {tag[:13]:>15}"
        print(f"  half {half}")
        print(header + f"  {'mean|log|':>10}")
        print(f"    {'':<6}" + "".join(f"  {'sd':>7} {'bias':>7}" for _ in TAGS))

        ranked = sorted(ARMS, key=lambda a: score(table[a]))
        for arm in ARMS:
            cells = table[arm]
            s = score(cells)
            row = f"    {arm:<6}" + "".join(f"  {fmt_cell(c)}" for c in cells)
            row += f"  {s:>10.3f}" if math.isfinite(s) else f"  {'-':>10}"
            if math.isfinite(s) and arm == ranked[0]:
                row += "  <-- best"
            print(row)

        # The baseline is the fixed recipe from SAOI_run: one number to say
        # whether the sweep bought anything at all.
        if args.baseline:
            base_cells = []
            for tag in TAGS:
                path = Path(args.baseline).resolve() / half / tag / "spread_values.npz"
                base_cells.append(load_cell(path) if path.is_file() else None)
            if any(c is not None for c in base_cells):
                s = score(base_cells)
                row = f"    {'SAOI_run':<6}" + "".join(f"  {fmt_cell(c)}" for c in base_cells)
                print(row + f"  {s:>10.3f}  (baseline)")

        best = ranked[0]
        best_score = score(table[best])
        if math.isfinite(best_score):
            winners[half] = (best, best_score, table[best])
            tied = [a for a in ranked[1:]
                    if math.isfinite(score(table[a]))
                    and score(table[a]) - best_score < TIE]
            print(f"    best: {best} ({ARM_DESC[best]})")
            if tied:
                print(f"    tied with {', '.join(tied)} -- inside the "
                      f"2000-draw noise floor; prefer the cheaper arm "
                      f"(fewer prior_blocks).")
            biases = [abs(c[1]) for c in table[best] if c is not None]
            if biases and max(biases) > BIAS_FLAG:
                print(f"    WARNING: this arm's draws sit up to "
                      f"{max(biases):.1f} GT sigmas off centre. It matches the "
                      f"WIDTH of a distribution centred somewhere else; check "
                      f"the histogram PNG before adopting it.")
        print()

    if not found_any:
        print("No spread_values.npz was found. Either Phase C has not run, or")
        print("make_histogram / eval_dataset was dropped from the configs.")
        return 1

    if winners:
        print("Summary")
        for half, (arm, s, cells) in winners.items():
            sds = [c[0] for c in cells if c is not None]
            span = f"{min(sds):.2f}-{max(sds):.2f}" if sds else "-"
            print(f"  {half}: {arm:<4} mean|log sd ratio| {s:.3f}  "
                  f"(sd ratio {span} across parts)")
        print()
        print("  An sd ratio still near 0.5 after the best arm wins means the")
        print("  prior is not the binding constraint -- run ae_ceiling_check.py")
        print("  on the promoted compressor, because a compressor that cannot")
        print("  reproduce peak-to-valley caps every arm at the same width.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
