#!/usr/bin/env python3
"""Size a training budget from a timestamped probe log.

    <probe stdout, each line prefixed with a unix timestamp> | probe_epochs.py \
        --budget-s 51840 --val-interval 30 [--baseline-s-per-epoch 113]

WHY NOT JUST DIVIDE WALL TIME BY EPOCHS
    That was the first version and it was wrong. A probe's wall clock is

        start-up  +  E * t_epoch  +  V * t_val

    and start-up here is not small: process launch, dataset load, and a cold
    multiscale hierarchy cache, which rebuilds whenever the source HDF5's mtime
    moves -- which training itself causes, by writing normalizers back into it.
    Charging all of that to three epochs made the derived budget absurd (29
    epochs for a run that had already done 1000).

    So this reads the interval BETWEEN consecutive epoch lines instead. Those
    intervals contain no start-up at all. With a probe that validates on some
    epochs and not others, the shortest interval is a bare epoch and the longest
    is an epoch plus a validation, which separates the two costs exactly.

WHY THE VALIDATION COST GETS ITS OWN LINE
    In both trees the sampling validation runs on RANK 0 ONLY, unsharded, while
    every other rank blocks on the broadcast that follows. Adding GPUs does not
    make it faster -- it makes more of them wait. For the flow tree it also
    integrates the ODE (val_num_samples x flow_steps x 2 forwards per graph
    under heun), so it can cost many epochs' worth. If it is a large share of
    the run, the fix is a bigger val_interval, and this prints the interval that
    holds it under --val-share.
"""
import argparse
import re
import sys

STAMPED = re.compile(r"^(\d+)\s+(.*)$")
EPOCH = re.compile(r"^Epoch\s+(\d+)\s*/")


def hms(seconds):
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', nargs='?', help='timestamped probe log (default: stdin)')
    ap.add_argument('--budget-s', type=float, required=True)
    ap.add_argument('--val-interval', type=int, default=30,
                    help='the interval the real run will use')
    ap.add_argument('--val-share', type=float, default=0.10,
                    help='largest share of wall time validation may take')
    ap.add_argument('--baseline-s-per-epoch', type=float, default=None,
                    help='known single-GPU cost, to report the actual DDP speed-up')
    a = ap.parse_args()

    text = (open(a.log, encoding='utf-8', errors='replace').read()
            if a.log else sys.stdin.read())

    marks, first_stamp = [], None
    for line in text.splitlines():
        m = STAMPED.match(line)
        if not m:
            continue
        t, body = int(m.group(1)), m.group(2)
        if first_stamp is None:
            first_stamp = t
        if EPOCH.match(body.strip()):
            marks.append(t)

    if len(marks) < 3:
        print(f"need at least 3 timestamped 'Epoch N/M' lines, found {len(marks)}. "
              f"The probe did not get far enough to measure anything.", file=sys.stderr)
        return 1

    deltas = [b - a_ for a_, b in zip(marks, marks[1:])]
    t_epoch = float(min(deltas))
    t_slow = float(max(deltas))
    t_val = t_slow - t_epoch
    startup = float(marks[0] - first_stamp) - t_slow

    if t_epoch <= 0:
        print("epoch intervals are zero -- the log has no usable timing.", file=sys.stderr)
        return 1

    per_epoch = t_epoch + t_val / max(a.val_interval, 1)
    epochs = max(1, int(a.budget_s / per_epoch))

    # An interval that keeps validation under --val-share of the run.
    if t_val > 0:
        need = int(-(-t_val / (a.val_share * t_epoch) // 1))   # ceil
        val_interval = max(a.val_interval, need)
    else:
        val_interval = a.val_interval

    out = sys.stderr
    print("", file=out)
    print("  probe decomposition (start-up excluded from the per-epoch figure)", file=out)
    print(f"    start-up (load + cache) : {hms(startup)}   ONE-TIME -- not charged to epochs", file=out)
    print(f"    per epoch               : {hms(t_epoch)}", file=out)
    if t_val > 0:
        print(f"    per validation          : {hms(t_val)}"
              f"  = {t_val / t_epoch:.1f} epochs, on rank 0 alone", file=out)
    else:
        print(f"    per validation          : not separable "
              f"(every probe epoch validated, or none did)", file=out)
    if a.baseline_s_per_epoch:
        speedup = a.baseline_s_per_epoch / t_epoch
        print(f"    vs 1 GPU ({a.baseline_s_per_epoch:g}s/epoch)  : {speedup:.2f}x", file=out)
        if speedup < 1.0:
            print(f"    WARNING: slower than a single GPU. The per-rank batch may be too "
                  f"small to fill a card -- 100% nvidia-smi utilisation means 'a kernel "
                  f"was resident', not 'the SMs are busy'.", file=out)
    print(f"    -> {epochs} epochs in {hms(a.budget_s)} at val_interval {a.val_interval}", file=out)
    if val_interval != a.val_interval:
        print(f"    -> validation is {t_val / (a.val_interval * t_epoch):.0%} of that run; "
              f"val_interval {val_interval} would hold it under {a.val_share:.0%}", file=out)
    print("", file=out)

    print(epochs)
    return 0


if __name__ == '__main__':
    sys.exit(main())
