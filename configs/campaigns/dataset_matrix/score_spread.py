#!/usr/bin/env python3
"""Collect the probabilistic arms' spread scores into one comparable table.

Every probabilistic inference run writes, into its own
`output/dataset_matrix/probabilistic/<example>/<method>/infer/`:

    spread_values.npz    gt / gen values with the scene each came from
    spread_metrics.json  the scores computed from them
    histogram_compare.png

That is the per-arm picture. This script is the cross-arm one: it reads every
`spread_metrics.json` under the campaign output root, prints one row per arm,
and -- when matplotlib is available -- redraws the ground-truth distribution
once per example with every method's generated distribution overlaid on it, so
the routes are compared on the same axis instead of across separate PNGs.

    python configs/campaigns/dataset_matrix/score_spread.py
    python configs/campaigns/dataset_matrix/score_spread.py --csv report.csv

Run it after the campaign (configs/run_all_135.sh / run_all_136.sh) finishes,
or at any point during it: arms that have not produced metrics yet are listed
as missing rather than skipped silently, because a quietly absent arm is how a
broken route survives a sweep. run_matrix.py also runs it at the end of every
run that was not stopped; there the other machine's example shows as missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'output' / 'dataset_matrix' / 'probabilistic'
MANIFEST = Path(__file__).resolve().parent / 'manifest.json'

# (key, target, how to read it). The first four are the calibration scores; the
# last three describe the marginal, which is what the histogram draws.
SCORES = (
    ('crps_norm', 0.0, 'per-scene CRPS / sd(gt); lower is better'),
    ('spread_skill', 1.0, 'ensemble sd / RMSE of the ensemble mean; <1 too narrow'),
    ('pit_ks', 0.0, 'KS distance of the truth ranks from uniform'),
    ('pit_tails', 0.04, 'share of truths in the outer 2%'),
    ('sd_ratio', 1.0, 'pooled sd(gen) / sd(gt)'),
    ('dmean_norm', 0.0, 'pooled mean shift / sd(gt)'),
    ('w1_norm', 0.0, 'pooled 1-Wasserstein / sd(gt)'),
)


def expected_arms():
    """The probabilistic (example, method) arms the matrix declares."""
    if not MANIFEST.exists():
        return []
    pairs = json.loads(MANIFEST.read_text(encoding='utf-8'))['pairs']
    return sorted({(p['example'], p['method']) for p in pairs
                   if p['category'] == 'probabilistic'})


def load_arm(out_root: Path, example: str, method: str):
    d = out_root / example / method / 'infer'
    metrics = d / 'spread_metrics.json'
    if not metrics.exists():
        return None
    row = json.loads(metrics.read_text(encoding='utf-8'))
    row['example'], row['method'], row['dir'] = example, method, str(d)
    return row


def print_table(rows):
    if not rows:
        return
    head = ['example', 'method', 'field', 'stat', 'scenes']
    head += [k for k, _, _ in SCORES]
    widths = [max(len(head[i]), *(len(_cell(r, head[i])) for r in rows))
              for i in range(len(head))]
    print('  '.join(h.ljust(w) for h, w in zip(head, widths)))
    print('  '.join('-' * w for w in widths))
    for r in rows:
        print('  '.join(_cell(r, h).ljust(w) for h, w in zip(head, widths)))
    print()
    print('targets: ' + '  '.join(f'{k}->{t:g}' for k, t, _ in SCORES))
    for k, _, why in SCORES:
        print(f'  {k:<14} {why}')


def _cell(row, key):
    if key == 'field':
        return str(row.get('label', '?'))
    if key == 'stat':
        return str(row.get('stat', 'range'))
    if key == 'scenes':
        return str(row.get('n_scenes', '?'))
    value = row.get(key)
    if isinstance(value, (int, float)):
        return f'{value:.4f}'
    return str(value if value is not None else '-')


def overlay(out_root: Path, rows, dest: Path):
    """One figure per example: GT once, every method's generated set over it."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:  # plotting is never required to get the table
        print(f'(overlay skipped: {exc})')
        return []
    written = []
    by_example = {}
    for r in rows:
        by_example.setdefault(r['example'], []).append(r)
    for example, arms in sorted(by_example.items()):
        series, gt = [], None
        for r in arms:
            npz = Path(r['dir']) / 'spread_values.npz'
            if not npz.exists():
                print(f'  (no spread_values.npz for {example}/{r["method"]})')
                continue
            data = np.load(npz, allow_pickle=True)
            gt = data['gt'] if gt is None else gt
            series.append((r['method'], data['gen'], r))
        if not series or gt is None:
            continue
        pool = np.concatenate([gt] + [g for _, g, _ in series])
        edges = np.histogram_bin_edges(pool, bins=40)
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.hist(gt, bins=edges, density=True, alpha=0.5, color='steelblue',
                label=f'ground truth (n={gt.size:,})')
        for (method, gen, r), colour in zip(series, ('darkorange', 'seagreen',
                                                     'crimson', 'purple')):
            ax.hist(gen, bins=edges, density=True, histtype='step', lw=2,
                    color=colour,
                    label=(f'{method} (n={gen.size:,})  '
                           f'crps={_cell(r, "crps_norm")} '
                           f'skill={_cell(r, "spread_skill")}'))
        first = series[0][2]
        stat, label = first.get('stat', 'range'), first.get('label', 'field')
        expr = f'max({label}) - min({label})' if stat == 'range' else f'{stat}({label})'
        ax.set_title(f'probabilistic/{example}: {expr} per realization, final timestep')
        ax.set_xlabel(expr)
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = dest / f'spread_overlay_{example}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
        print(f'  overlay -> {path}')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-root', default=str(DEFAULT_OUT),
                    help='campaign probabilistic output root (default: %(default)s)')
    ap.add_argument('--csv', help='also write the table to this CSV path')
    ap.add_argument('--no-overlay', action='store_true', help='skip the figures')
    args = ap.parse_args()

    out_root = Path(args.output_root)
    arms = expected_arms()
    if not arms:
        print(f'No probabilistic arms in {MANIFEST}.', file=sys.stderr)
        return 2

    rows, missing = [], []
    for example, method in arms:
        row = load_arm(out_root, example, method)
        (rows.append(row) if row else missing.append(f'{example}/{method}'))

    print(f'Probabilistic spread scores  ({len(rows)}/{len(arms)} arm(s) scored)')
    print(f'output root: {out_root}\n')
    print_table(rows)
    if missing:
        print('\nNot yet scored (no spread_metrics.json):')
        for m in missing:
            print(f'  {m}')

    if args.csv:
        fields = (['example', 'method', 'label', 'stat', 'n_scenes']
                  + [k for k, _, _ in SCORES] + ['n_gt', 'n_gen', 'eval_dataset'])
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'\ncsv -> {path}')

    if not args.no_overlay and rows:
        print()
        overlay(out_root, rows, out_root / '_campaign')

    # A missing arm is a finding, not a silent gap.
    return 1 if missing else 0


if __name__ == '__main__':
    raise SystemExit(main())
