"""Wave 0 report: how many more epochs does flow matching need?

Reads the two training logs (deterministic HI-MGN and cHI-MGNflow, trained on
the SAME backbone, data and budget) and reports where each curve flattens.

The two loss VALUES are not comparable -- one regresses the field, the other a
velocity, and they live on different scales. What IS comparable is the SHAPE:
the fraction of the run each arm needs to reach a given fraction of its own
total improvement. Their ratio is the multiple that sizes `training_epochs` for
the Wave B sweep.

    python misc/wave0_report.py --det ../../output/chi-mgnflow/wave0/det.log \n        --flow ../../output/chi-mgnflow/wave0/flow.log

Both `--det` and `--flow` also accept the raw stdout transcript, because
[FlowDiag] lines are emitted with tqdm.write and reach stdout only -- they never
land in the epoch log file. Pass whichever file has the lines you need; the
parser takes what it finds in each.
"""
import argparse
import os
import re
import sys


EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/")
# deterministic tree:  "Epoch 3/24 TrainOpt: 1.15e+00 Valid: 1.29e+00 LR: ..."
DET_TRAIN_RE = re.compile(r"Train(?:Opt)?[:=]\s*([0-9.eE+-]+)")
DET_VALID_RE = re.compile(r"Valid(?:ation)?[:=]\s*([0-9.eE+-]+)")
# flow tree:           "Epoch 3/24 LR: ... | Train fm=1.34e+00 | Valid fm=1.45e+00 | CRPS ... spread ..."
FLOW_TRAIN_RE = re.compile(r"Train\s+fm=([0-9.eE+-]+)")
FLOW_VALID_RE = re.compile(r"Valid\s+fm=([0-9.eE+-]+)")
CRPS_RE = re.compile(r"CRPS\s+([0-9.eE+-]+)")
SPREAD_RE = re.compile(r"spread\s+([0-9.eE+-]+)")


def as_float(m):
    """Guarded float(). `m` is None on every line missing the field, and
    None.group raises AttributeError, which a (TypeError, ValueError) guard
    would sail straight past -- the exact bug that crashed the wave-3 scorer."""
    if m is None:
        return None
    try:
        return float(m.group(1))
    except (TypeError, ValueError):
        return None


def parse(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if 'Epoch' not in line:
                continue
            ep = as_float(EPOCH_RE.search(line))
            if ep is None:
                continue
            rows.append({
                'epoch': int(ep),
                'train': as_float(FLOW_TRAIN_RE.search(line)) or as_float(DET_TRAIN_RE.search(line)),
                'valid': as_float(FLOW_VALID_RE.search(line)) or as_float(DET_VALID_RE.search(line)),
                'crps': as_float(CRPS_RE.search(line)),
                'spread': as_float(SPREAD_RE.search(line)),
            })
    return rows


def flatten_epoch(rows, key, frac):
    """First epoch reaching `frac` of the run's total improvement in `key`."""
    vals = [(r['epoch'], r[key]) for r in rows if r.get(key) is not None]
    if len(vals) < 3:
        return None, None
    first, best = vals[0][1], min(v for _, v in vals)
    if first <= best:
        return None, None
    target = first - frac * (first - best)
    for ep, v in vals:
        if v <= target:
            return ep, v
    return None, None


def summarise(name, rows, total):
    print(f"\n{name}")
    print("-" * 58)
    if not rows:
        print("  no epoch lines found -- check the path")
        return {}
    out = {}
    for key in ('train', 'valid'):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        print(f"  {key:<6} first={vals[0]:.4e}  last={vals[-1]:.4e}  best={min(vals):.4e}")
        for frac in (0.5, 0.8, 0.9):
            ep, v = flatten_epoch(rows, key, frac)
            if ep is not None:
                print(f"     {int(frac*100)}% of total improvement at epoch {ep:>4} "
                      f"({ep/max(total,1):.0%} of the run)")
                out[(key, frac)] = ep
    crps = [(r['epoch'], r['crps']) for r in rows if r.get('crps') is not None]
    if crps:
        ep, v = min(crps, key=lambda t: t[1])
        print(f"  crps   best={v:.4e} at epoch {ep}")
    spread = [r['spread'] for r in rows if r.get('spread') is not None]
    if spread:
        print(f"  spread first={spread[0]:.3f}  last={spread[-1]:.3f}   "
              f"(1.0 = ensemble spread matches the ground-truth std)")
        if spread[-1] < 0.05:
            print("     WARNING: spread collapsed -- the noise channel is being ignored")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--det', required=True)
    ap.add_argument('--flow', required=True)
    ap.add_argument('--total-epochs', type=int, default=0)
    args = ap.parse_args()

    det, flow = parse(args.det), parse(args.flow)
    total = args.total_epochs or max(
        [r['epoch'] for r in det + flow] or [1]) + 1

    print("=" * 58)
    print(f"WAVE 0 REPORT   (budget = {total} epochs per arm)")
    print("=" * 58)
    d = summarise("DET  - deterministic HI-MGN", det, total)
    f = summarise("FLOW - cHI-MGNflow", flow, total)

    print("\n" + "=" * 58)
    print("EPOCH MULTIPLE")
    print("=" * 58)
    any_ratio = False
    for frac in (0.5, 0.8, 0.9):
        de, fe = d.get(('valid', frac)), f.get(('valid', frac))
        if de and fe:
            any_ratio = True
            print(f"  to reach {int(frac*100)}% of own improvement: "
                  f"det={de:>3}  flow={fe:>3}   ratio = {fe/de:.2f}x")
    if not any_ratio:
        print("  not enough validation points -- lower val_interval or run longer")
    else:
        print("\n  Use the LARGEST ratio to size training_epochs for Wave B; the")
        print("  90% figure is the honest one because the tail is where the two")
        print("  objectives diverge most.")
        print("\n  CAVEAT: with a short budget neither arm may have flattened, in")
        print("  which case this reports how fast each one is EARLY, not how long")
        print("  each one NEEDS. Re-run at 3-5x this budget before committing.")


if __name__ == '__main__':
    main()
