"""Score the DeepJEB surrogate against the real FEA labels on held-out brackets.

Two families of number, because they answer different questions:

*Field accuracy* (R^2, relative L2) says whether the predicted stress and
displacement maps look like the solver's.

*Design-quantity accuracy* is what actually decides whether the surrogate can
drive an optimizer: the loop never consumes a field, it consumes a scalar per
candidate -- peak stress and peak displacement. A surrogate can have a decent
field R^2 and still be useless for screening if it flattens the peaks, and the
rank correlation across candidates is what determines whether the search moves
in the right direction at all.

  python dataset/score_deepjeb_surrogate.py \
      --truth dataset/deepjeb_mgn_infer.h5 \
      --rollout-dir cHI-MGNflow/outputs/rollout
"""

import argparse
import glob
import json
import os
import re
import sys

import h5py
import numpy as np

FIELDS = ('stress', 'disp')
UNITS = ('MPa', 'mm')
STRESS_PERCENTILE = 99.5


def load_truth(path):
    samples = {}
    with h5py.File(path, 'r') as f:
        for sid in f['data']:
            g = f['data'][sid]
            samples[int(sid)] = {
                'nodal': g['nodal_data'][...],
                'bracket': g['metadata'].attrs['bracket'],
                'case': g['metadata'].attrs['load_case'],
            }
    return samples


def load_predictions(rollout_dir):
    preds = {}
    for path in glob.glob(os.path.join(rollout_dir, 'rollout_sample*_steps*.h5')):
        m = re.search(r'rollout_sample(\d+)_', os.path.basename(path))
        if not m:
            continue
        with h5py.File(path, 'r') as f:
            key = next(iter(f['data']))
            # t=0 is the (zeroed) input state; t=-1 is the prediction.
            preds[int(m.group(1))] = f['data'][key]['nodal_data'][:, -1, :]
    return preds


def r2(truth, pred):
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def rel_l2(truth, pred):
    denom = float(np.linalg.norm(truth))
    return float(np.linalg.norm(truth - pred) / denom) if denom > 0 else float('nan')


def spearman(a, b):
    if len(a) < 3:
        return float('nan')
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float('nan')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', default='dataset/deepjeb_mgn_infer.h5')
    parser.add_argument('--rollout-dir', default='cHI-MGNflow/outputs/rollout')
    parser.add_argument('--json-out', default=None)
    args = parser.parse_args(argv)

    truth = load_truth(args.truth)
    preds = load_predictions(args.rollout_dir)
    shared = sorted(set(truth) & set(preds))
    if not shared:
        print('no overlapping samples between truth and rollouts', file=sys.stderr)
        return 1
    print(f'scoring {len(shared)} held-out samples '
          f'({len(truth)} truth, {len(preds)} predictions)\n')

    rows, per_case = [], {}
    for sid in shared:
        t, p = truth[sid]['nodal'], preds[sid]
        if t.shape[2] != p.shape[1]:
            print(f'  sample {sid}: node-count mismatch {t.shape[2]} vs {p.shape[1]}; skipped')
            continue
        row = {'sample': sid, 'bracket': truth[sid]['bracket'], 'case': truth[sid]['case']}
        for k, name in enumerate(FIELDS):
            gt, pr = t[3 + k, 0, :], p[3 + k, :]
            row[f'{name}_r2'] = r2(gt, pr)
            row[f'{name}_rel_l2'] = rel_l2(gt, pr)
            if name == 'stress':
                row['peak_true'] = float(np.percentile(np.abs(gt), STRESS_PERCENTILE))
                row['peak_pred'] = float(np.percentile(np.abs(pr), STRESS_PERCENTILE))
            else:
                row['maxdisp_true'] = float(gt.max())
                row['maxdisp_pred'] = float(pr.max())
        rows.append(row)
        per_case.setdefault(row['case'], []).append(row)

    def col(rs, key):
        return np.array([r[key] for r in rs], dtype=float)

    print('--- field accuracy (all held-out samples) ---')
    for name, unit in zip(FIELDS, UNITS):
        r2v, l2v = col(rows, f'{name}_r2'), col(rows, f'{name}_rel_l2')
        print(f'  {name:7s} ({unit:3s})  R2 median {np.median(r2v):+.3f}  mean {r2v.mean():+.3f}'
              f'   | rel-L2 median {np.median(l2v):.3f}  mean {l2v.mean():.3f}')

    print('\n--- by load case ---')
    for case in ('ver', 'hor', 'dia', 'tor'):
        rs = per_case.get(case)
        if not rs:
            continue
        print(f'  {case}: n={len(rs):2d}  stress R2 {np.median(col(rs, "stress_r2")):+.3f}'
              f'   disp R2 {np.median(col(rs, "disp_r2")):+.3f}'
              f'   stress rel-L2 {np.median(col(rs, "stress_rel_l2")):.3f}')

    print('\n--- design quantities (what the optimizer actually reads) ---')
    for label, tk, pk, unit in (('peak |stress| (p99.5)', 'peak_true', 'peak_pred', 'MPa'),
                                ('max displacement', 'maxdisp_true', 'maxdisp_pred', 'mm')):
        gt, pr = col(rows, tk), col(rows, pk)
        mape = float(np.mean(np.abs(pr - gt) / np.maximum(np.abs(gt), 1e-12)) * 100)
        print(f'  {label:22s}: MAPE {mape:6.1f}%   R2 {r2(gt, pr):+.3f}   '
              f'Spearman {spearman(gt, pr):+.3f}   '
              f'(true {gt.min():.4g}-{gt.max():.4g} {unit})')

    summary = {
        'samples': len(rows),
        'field': {name: {'r2_median': float(np.median(col(rows, f'{name}_r2'))),
                         'rel_l2_median': float(np.median(col(rows, f'{name}_rel_l2')))}
                  for name in FIELDS},
        'design_quantities': {
            'peak_stress': {'mape_pct': float(np.mean(np.abs(col(rows, 'peak_pred')
                                                             - col(rows, 'peak_true'))
                                                      / col(rows, 'peak_true')) * 100),
                            'spearman': spearman(col(rows, 'peak_true'),
                                                 col(rows, 'peak_pred'))},
            'max_disp': {'mape_pct': float(np.mean(np.abs(col(rows, 'maxdisp_pred')
                                                          - col(rows, 'maxdisp_true'))
                                                   / np.maximum(col(rows, 'maxdisp_true'), 1e-12)) * 100),
                         'spearman': spearman(col(rows, 'maxdisp_true'),
                                              col(rows, 'maxdisp_pred'))},
        },
        'rows': rows,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or '.', exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2, default=float)
        print(f'\nwrote {args.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
