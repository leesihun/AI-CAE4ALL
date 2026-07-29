#!/usr/bin/env python
"""Compare the Hi-MGN ablation arms from their training logs.

Reads the per-epoch lines that `training_profiles/single_training.py` (and the
DDP twin) append to each run's log:

    Elapsed: 123.45s Epoch 5 TrainOpt 1.2345e-03 Valid 2.3456e-03 LR: 1.0000e-04

`Valid` is written as the literal `Valid skipped` on non-validation epochs
(val_interval), so those lines contribute a train loss but no val loss.

Usage:
    python compare_ablation.py ex1
    python compare_ablation.py ex2 --at-epoch 150   # rank at a common epoch
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# arm key -> (log basename, human label)
ARMS = {
    'base': ('train_himgn_base.log', 'baseline (mean pool / sum unpool, 1 partition)'),
    'p1':   ('train_himgn_p1.log',   'Part I   (attention pool + unpool, 4 heads)'),
    'p2':   ('train_himgn_p2.log',   'Part II  (4 partitions of coarsest level)'),
    'p12':  ('train_himgn_p12.log',  'Both     (Part I + Part II)'),
}

# Time integration is forced by each dataset, not chosen: ex2 has 50 timesteps
# so it trains with AR-RT; ex1 has 1 (static), so AR-RT is impossible there.
# Arms are only comparable within an ex.
EX_SCHEME = {'ex1': 'AR-OT (ex1 is static: num_timesteps=1)',
             'ex2': 'AR-RT (49-step rollout)'}

LINE_RE = re.compile(
    r'Elapsed:\s*([\d.]+)s\s+Epoch\s+(\d+)\s+TrainOpt\s+([\d.eE+-]+)\s+'
    r'(?:Valid\s+([\d.eE+-]+)|Valid skipped)'
)


def parse_log(path: Path):
    """Return [(epoch, elapsed_s, train, valid_or_None), ...] in file order."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors='replace').splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue  # header / embedded config / test output
        elapsed, epoch, train, valid = m.groups()
        rows.append((
            int(epoch), float(elapsed), float(train),
            float(valid) if valid is not None else None,
        ))
    return rows


def summarize(rows, at_epoch=None):
    if not rows:
        return None
    if at_epoch is not None:
        rows = [r for r in rows if r[0] <= at_epoch]
        if not rows:
            return None
    vals = [(e, v) for e, _, _, v in rows if v is not None]
    best_epoch, best_val = min(vals, key=lambda t: t[1]) if vals else (None, None)
    return {
        'epochs': rows[-1][0] + 1,          # epoch index is 0-based in the log
        'hours': rows[-1][1] / 3600.0,
        'final_train': rows[-1][2],
        'final_val': vals[-1][1] if vals else None,
        'best_val': best_val,
        'best_epoch': best_epoch,
    }


def fmt(x, spec='.4e'):
    return format(x, spec) if x is not None else '--'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ex', choices=['ex1', 'ex2'])
    ap.add_argument('--at-epoch', type=int, default=None,
                    help='truncate every arm at this epoch before comparing '
                         '(use when arms have run for different lengths -- '
                         'comparing a 2000-epoch run against a 300-epoch one '
                         'measures run length, not the feature)')
    args = ap.parse_args()

    log_dir = ROOT / 'output' / 'meshgraphnets' / args.ex
    results = {}
    for arm, (basename, _label) in ARMS.items():
        results[arm] = summarize(parse_log(log_dir / basename), args.at_epoch)

    print(f'\n=== {args.ex} ablation'
          + (f' (truncated at epoch {args.at_epoch})' if args.at_epoch else '')
          + ' ===')
    print(f'logs:   {log_dir}')
    print(f'scheme: {EX_SCHEME[args.ex]}\n')

    missing = [a for a, r in results.items() if r is None]
    if missing:
        print(f'not yet run (or no epoch lines yet): {", ".join(missing)}\n')

    header = f'{"arm":<6}{"epochs":>8}{"hours":>8}{"best val":>13}{"@ep":>7}{"final val":>13}{"vs base":>10}'
    print(header)
    print('-' * len(header))

    base = results.get('base')
    base_best = base['best_val'] if base else None

    for arm, (_basename, label) in ARMS.items():
        r = results[arm]
        if r is None:
            print(f'{arm:<6}{"--":>8}{"--":>8}{"--":>13}{"--":>7}{"--":>13}{"--":>10}')
            continue
        if base_best and r['best_val']:
            delta = f'{100 * (r["best_val"] / base_best - 1):+.1f}%'
        else:
            delta = '--'
        print(f'{arm:<6}{r["epochs"]:>8}{r["hours"]:>8.1f}'
              f'{fmt(r["best_val"]):>13}{str(r["best_epoch"]):>7}'
              f'{fmt(r["final_val"]):>13}{delta:>10}')

    print()
    for arm, (_b, label) in ARMS.items():
        print(f'  {arm:<5} {label}')

    # Guard against the most likely misreading of this table.
    lengths = {a: r['epochs'] for a, r in results.items() if r}
    if len(set(lengths.values())) > 1 and args.at_epoch is None:
        print(f'\nWARNING: arms ran for different epoch counts {lengths}.')
        print('         "vs base" is not a fair comparison across different run')
        print(f'         lengths -- rerun with --at-epoch {min(lengths.values()) - 1}.')

    print('\nNote: best val is the selection metric the trainer itself uses, but it is a')
    print('node-averaged MSE. For the peak-stress question, evaluate the saved')
    print('checkpoints separately -- that quantity is not in these logs.')


if __name__ == '__main__':
    main()
