"""Analyze the ex2 Transolver capacity sweep (OFAT star, 8 cells).

Reads the per-epoch logs each cell's config writes via log_file_dir
(output/transolver/ex2_sweep/train{n}_<tag>.log), NOT the run_train_all.sh
transcripts.

The sweep is a one-factor-at-a-time star around

    ANCHOR = num_layers 10, latent_dim 256, slice_num 128

so every cell differs from the anchor in exactly one hyperparameter. This
script walks each axis in turn and reports the effect of moving along it,
measured against the anchor -- which is the direct answer to "what does each
axis do on ex2?".

Cell 8 (the ex2 bridge, L8 C256 M128) differs from the anchor only in
num_layers, so it is picked up automatically as a 4th point on the layers axis.

Reported per cell:
  - min_valid   : best validation loss ever reached
  - final_valid : validation loss at the last epoch
  - overfit_gap : final - min. With only 40 training trajectories, a large gap
                  means the cell peaked early and then degraded; that is a real
                  result about capacity vs this dataset, not a tuning failure.

Usage (from this directory):
    python summarize_sweep.py
    python summarize_sweep.py --metric final      # use last instead of best
    python summarize_sweep.py --log-dir /path/to/output/transolver/ex2_sweep
"""
import argparse
import glob
import math
import os
import re

VALID_RE = re.compile(r"Epoch (\d+) TrainOpt [0-9.eE+-]+ Valid ([0-9.eE+-]+)")
FILENAME_RE = re.compile(r"train(\d+)_(L(\d+)_C(\d+)_M(\d+)_mlp(\d+))\.log$")

# must match gen_sweep_configs.py
ANCHOR_LAYERS, ANCHOR_DIM, ANCHOR_SLICES = 10, 256, 128

AXES = [
    ("layers", "num_layers", ("dim", "slices")),
    ("dim", "latent_dim", ("layers", "slices")),
    ("slices", "slice_num", ("layers", "dim")),
]
ANCHOR_OF = {"layers": ANCHOR_LAYERS, "dim": ANCHOR_DIM, "slices": ANCHOR_SLICES}


def parse_log(path):
    """Returns (final_epoch, final_valid, min_valid, min_epoch) or None.

    Two quirks of the trainer's log format are handled here:

    - The DDP path (distributed_training.py, which is what this sweep runs)
      writes a `Valid <x>` field on EVERY epoch, repeating the last real
      validation value on non-validation epochs instead of marking them
      skipped. Using a strict `<` for the min means the FIRST occurrence wins,
      so min_epoch reports the true validation epoch rather than a stale
      carry-over. final_valid is correct because the last epoch always validates.
    - NaN/inf are skipped. A diverged cell writing NaN would otherwise sort to
      the top as the "best" result, since every comparison against NaN is False
      and it would slip in as the initial min.
    """
    final_epoch = final_valid = min_valid = min_epoch = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = VALID_RE.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            try:
                valid = float(m.group(2))
            except ValueError:
                continue
            if not math.isfinite(valid):
                continue
            final_epoch, final_valid = epoch, valid
            if min_valid is None or valid < min_valid:
                min_valid, min_epoch = valid, epoch
    if final_valid is None:
        return None
    return final_epoch, final_valid, min_valid, min_epoch


def collect(log_dir, expect_epochs):
    """A cell is 'complete' only if its last finite validation reached the final
    expected epoch. This catches still-running cells AND cells that diverged to
    NaN partway (their last finite epoch stops early)."""
    rows = []
    final_expected = expect_epochs - 1
    for path in sorted(glob.glob(os.path.join(log_dir, "train*_*.log"))):
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        rec = dict(n=int(m.group(1)), tag=m.group(2), layers=int(m.group(3)),
                   dim=int(m.group(4)), slices=int(m.group(5)))
        rec["is_anchor"] = (rec["layers"] == ANCHOR_LAYERS
                            and rec["dim"] == ANCHOR_DIM
                            and rec["slices"] == ANCHOR_SLICES)
        parsed = parse_log(path)
        if parsed is None:
            rec.update(final_epoch=None, final=None, min=None, min_epoch=None,
                       complete=False)
        else:
            fe, fv, mv, me = parsed
            rec.update(final_epoch=fe, final=fv, min=mv, min_epoch=me,
                       complete=fe >= final_expected)
        rows.append(rec)
    return rows


def on_axis(rec, axis, held):
    """A cell is on `axis` if the other two knobs sit at their anchor values."""
    return all(rec[h] == ANCHOR_OF[h] for h in held)


def print_axes(rows, metric):
    anchor = next((r for r in rows if r["is_anchor"] and r[metric] is not None), None)
    print("\n" + "=" * 78)
    print(f"PER-AXIS EFFECTS  (metric = {metric}_valid, lower is better)")
    print("=" * 78)
    if anchor is None:
        print("\nAnchor cell (L10 C256 M128) has no results yet -- percentages vs the")
        print("anchor are unavailable until it finishes. Raw values still shown.")

    for axis, label, held in AXES:
        held_desc = ", ".join(f"{ANCHOR_OF[h]}" for h in held)
        held_names = ", ".join(dict(layers="num_layers", dim="latent_dim",
                                    slices="slice_num")[h] for h in held)
        members = [r for r in rows if on_axis(r, axis, held) and r[metric] is not None]
        print(f"\n{label}   (holding {held_names} at {held_desc})")
        if not members:
            print("    no completed cells on this axis yet")
            continue
        print(f"    {'value':>6} {'cell':>5} {metric + '_valid':>13} "
              f"{'vs anchor':>10} {'overfit_gap':>12}  {'':<3}")
        for r in sorted(members, key=lambda r: r[axis]):
            if anchor is not None and anchor[metric]:
                delta = (r[metric] - anchor[metric]) / anchor[metric] * 100
                vs = "anchor" if r["is_anchor"] else f"{delta:+.1f}%"
            else:
                vs = "--"
            gap = r["final"] - r["min"]
            flags = ("" if r["complete"] else "!")
            print(f"    {r[axis]:>6} {r['n']:>5} {r[metric]:>13.4e} "
                  f"{vs:>10} {gap:>12.4e}  {flags:<3}")
        # The verdict is what gets acted on, so base it only on finished cells:
        # an unfinished cell's loss reflects how long it trained, not its config.
        finished = [r for r in members if r["complete"]]
        dropped = len(members) - len(finished)
        if not finished:
            print("    -> no finished cells on this axis yet; no verdict.")
            continue
        best = min(finished, key=lambda r: r[metric])
        note = f"  (ignoring {dropped} unfinished cell(s))" if dropped else ""
        if best[axis] == ANCHOR_OF[axis]:
            print(f"    -> best on this axis is the anchor value "
                  f"({ANCHOR_OF[axis]}); no excursion beat it.{note}")
        else:
            direction = "increasing" if best[axis] > ANCHOR_OF[axis] else "decreasing"
            print(f"    -> best on this axis is {best[axis]} "
                  f"({direction} from {ANCHOR_OF[axis]}), cell {best['n']}.{note}")


def print_table(rows, metric):
    hdr = (f"{'cell':>4}  {'config':<24} {'L':>3} {'C':>4} {'M':>4} "
           f"{'ep':>5} {'final_valid':>12} {'min_valid':>12} {'min_ep':>6} "
           f"{'overfit_gap':>12}")
    print("\n" + "=" * len(hdr))
    print(f"ALL CELLS, ranked by {metric}_valid")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    done = [r for r in rows if r[metric] is not None]
    pending = [r for r in rows if r[metric] is None]
    for r in sorted(done, key=lambda r: r[metric]):
        marks = ("*" if r["is_anchor"] else "") + ("" if r["complete"] else "!")
        print(f"{r['n']:>4}  {r['tag']:<24} {r['layers']:>3} {r['dim']:>4} "
              f"{r['slices']:>4} {r['final_epoch']:>5} {r['final']:>12.4e} "
              f"{r['min']:>12.4e} {r['min_epoch']:>6} "
              f"{r['final'] - r['min']:>12.4e}{'  ' + marks if marks else ''}")
    for r in sorted(pending, key=lambda r: r["n"]):
        print(f"{r['n']:>4}  {r['tag']:<24} {r['layers']:>3} {r['dim']:>4} "
              f"{r['slices']:>4}   (no completed validation epochs yet)")
    if any(r["is_anchor"] for r in done):
        print("\n  * = anchor (star center, L10 C256 M128)")
    if any(not r["complete"] for r in done):
        print("  ! = INCOMPLETE (still running, crashed, or diverged to NaN).")
        print("      Its loss is not comparable to a finished cell -- treat any")
        print("      axis conclusion resting on it as provisional.")


def main():
    parser = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_log_dir = os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "output", "transolver", "ex2_sweep"))
    parser.add_argument("--log-dir", default=default_log_dir)
    parser.add_argument("--metric", choices=["min", "final"], default="min",
                        help="use best-ever (min) or last (final) validation loss")
    parser.add_argument("--expect-epochs", type=int, default=500,
                        help="training_epochs the cells were configured with; a cell whose "
                             "last finite validation falls short is flagged incomplete")
    args = parser.parse_args()

    rows = collect(args.log_dir, args.expect_epochs)
    if not rows:
        print(f"No sweep logs found under {args.log_dir}")
        print("(cells still running, or point --log-dir at the right output/transolver/ex2_sweep dir)")
        return

    print(f"ex2 Transolver capacity sweep (OFAT star) -- {len(rows)}/8 cell logs found")
    print_axes(rows, args.metric)
    print_table(rows, args.metric)

    print("\n\nHOW TO READ THIS")
    print("  Each axis moves ONE hyperparameter with the other two held at the anchor,")
    print("  so 'vs anchor' is the isolated effect of that knob. What this design")
    print("  CANNOT show is interactions -- e.g. depth paying off only at latent_dim")
    print("  512 -- because no cell moves two axes at once. If one axis shows a strong")
    print("  effect, the follow-up is a small 2x2 crossing it with the next best axis.")
    print("  overfit_gap = final - min. Gaps concentrated in the high-capacity cells")
    print("  are capacity outrunning ex2's 40-trajectory data budget.")


if __name__ == "__main__":
    main()
