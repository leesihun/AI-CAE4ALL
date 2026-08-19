#!/usr/bin/env python3
"""Score configs/benchmarks_all rollouts against their exN_infer.h5 ground
truth: relative L2 and R2 (coefficient of determination), per channel and
aggregate.

Reads configs/benchmarks_all/roster.tsv for the (label, canonical train
config) pairs, then pulls `infer_dataset` (ground truth), `inference_output_dir`
(where infer_all.sh's rollout.py wrote `rollout_sample<id>_steps<N>.h5`), and
`output_var`/`input_var` (state channel count -- NOT `num_features - 4`, since
AirfRANS has cond_var=5 trailing rows that read()-ex2's simpler heuristic
would silently fold into "state") straight out of each canonical config, so
nothing here can drift from what train_all.sh/infer_all.sh actually ran.

Static (T=1: ex7 AirfRANS, ex8 elasticity) and temporal (T>1: ex4/ex5/ex6,
ex9 plasticity) rollouts are scored differently: T=1 has no seeded initial
condition to exclude, so the single stored step is scored directly; T>1
rollouts seed from the ground-truth t=0 state (identical by construction), so
scoring starts at t=1 like configs/ex2/score_rollouts.py already does.

Usage:
    python configs/benchmarks_all/score_rollouts.py
    python configs/benchmarks_all/score_rollouts.py --labels meshgraphnets_ex8 transolver_ex8
    python configs/benchmarks_all/score_rollouts.py --csv output/benchmarks_all/scores.csv
"""

import argparse
import csv
import glob
import os
import re
import sys

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROSTER = os.path.join(os.path.dirname(__file__), 'roster.tsv')


def load_roster(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        next(f)  # header
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            label, cfg, ex_slot, light = line.split('\t')
            rows.append((label, cfg, ex_slot, light))
    return rows


def parse_config(path):
    """Minimal key/value reader -- good enough for the scalar fields used here."""
    vals = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('%', 1)[0].rstrip('\n')
            if not line.strip() or line.strip() == "'":
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            key, val = parts
            vals[key.strip().lower()] = val.strip()
    return vals


def resolve(repo_root_relative_to_config, value):
    """Config paths are relative to the METHOD REPO dir (e.g. MeshGraphNets/),
    a sibling of dataset/output/configs at the suite root -- see CLAUDE.md."""
    return os.path.normpath(os.path.join(REPO_ROOT, repo_root_relative_to_config, value))


def method_repo_for(config_path):
    # configs/<MethodDir>/exN/file.txt -> <MethodDir>
    parts = config_path.replace('\\', '/').split('/')
    return parts[1]


def rel_l2(pred, truth):
    denom = np.linalg.norm(truth)
    if denom == 0.0:
        return float('nan')
    return float(np.linalg.norm(pred - truth) / denom)


def r2(pred, truth):
    ss_res = float(np.sum((pred - truth) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    if ss_tot == 0.0:
        return float('nan')
    return 1.0 - ss_res / ss_tot


def read_rollout(path, num_state):
    with h5py.File(path, 'r') as f:
        sample_key = list(f['data'].keys())[0]
        nodal = f[f'data/{sample_key}/nodal_data'][:]  # [F, T, N]
        pred = np.transpose(nodal[3:3 + num_state], (1, 2, 0)).astype(np.float64)  # [T, N, C]
        coords = nodal[:3, 0, :].T.astype(np.float64)
    return pred, coords


def load_gt(path, num_state):
    gt = {}
    with h5py.File(path, 'r') as f:
        for key in sorted(f['data'].keys(), key=int):
            nodal = f[f'data/{key}/nodal_data']
            state = nodal[3:3 + num_state, :, :]
            gt[int(key)] = (
                np.transpose(state, (1, 2, 0)).astype(np.float64),  # [T, N, C]
                nodal[:3, 0, :].T.astype(np.float64),                # [N, 3]
            )
    return gt


def score_label(label, cfg_path, skip_coord_check):
    cfg = parse_config(cfg_path)
    repo = method_repo_for(cfg_path)
    if 'infer_dataset' not in cfg or 'inference_output_dir' not in cfg:
        print(f"[{label}] SKIP: config missing infer_dataset/inference_output_dir")
        return []
    num_state = int(cfg.get('output_var', cfg.get('input_var', 0)))
    if num_state <= 0:
        print(f"[{label}] SKIP: could not determine state channel count from {cfg_path}")
        return []

    gt_path = resolve(repo, cfg['infer_dataset'])
    rollout_dir = resolve(repo, cfg['inference_output_dir'])
    if not os.path.exists(gt_path):
        print(f"[{label}] SKIP: ground truth not found: {gt_path}")
        return []

    files = sorted(
        glob.glob(os.path.join(rollout_dir, 'rollout_sample*_steps*.h5')),
        key=lambda p: int(re.search(r'rollout_sample(\d+)_', os.path.basename(p)).group(1)),
    )
    if not files:
        print(f"[{label}] no rollouts found in {rollout_dir} (run infer_all.sh first)")
        return []

    gt = load_gt(gt_path, num_state)
    rows = []
    for path in files:
        sample_id = int(re.search(r'rollout_sample(\d+)_', os.path.basename(path)).group(1))
        if sample_id not in gt:
            print(f"  [{label}] sample {sample_id}: no ground truth -- skipped")
            continue
        truth_all, gt_coords = gt[sample_id]
        pred_all, pred_coords = read_rollout(path, num_state)

        if pred_coords.shape[0] != gt_coords.shape[0]:
            print(f"  [{label}] sample {sample_id}: node count {pred_coords.shape[0]} != "
                  f"GT {gt_coords.shape[0]} -- skipped")
            continue
        if not skip_coord_check:
            drift = float(np.abs(pred_coords - gt_coords).max())
            if drift > 1e-3:
                print(f"  [{label}] sample {sample_id}: reference coords differ by {drift:.3g} "
                      "-- node order may not match; skipped")
                continue

        steps = min(pred_all.shape[0], truth_all.shape[0])
        if steps == 1:
            # Static (T=1): the single stored step IS the prediction, not a seed.
            pred, truth = pred_all[0:1], truth_all[0:1]
        else:
            # Temporal AR rollout: t=0 is the seeded ground-truth IC, excluded.
            pred, truth = pred_all[1:steps], truth_all[1:steps]

        per_channel_rel = [rel_l2(pred[:, :, c], truth[:, :, c]) for c in range(num_state)]
        per_channel_r2 = [r2(pred[:, :, c], truth[:, :, c]) for c in range(num_state)]
        rows.append({
            'label': label,
            'sample': sample_id,
            'steps': pred.shape[0],
            'rel_l2_all': rel_l2(pred, truth),
            'r2_all': r2(pred, truth),
            'rel_l2': per_channel_rel,
            'r2': per_channel_r2,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--roster', default=ROSTER)
    ap.add_argument('--labels', nargs='*', default=None,
                    help='subset of roster labels (default: all present)')
    ap.add_argument('--csv', default=None, help='also write per-sample scores here')
    ap.add_argument('--skip-coord-check', action='store_true')
    args = ap.parse_args()

    roster = load_roster(args.roster)
    if args.labels:
        wanted = set(args.labels)
        roster = [r for r in roster if r[0] in wanted]
        missing = wanted - {r[0] for r in roster}
        if missing:
            sys.exit(f"Unknown label(s): {', '.join(sorted(missing))}")

    all_rows = []
    for label, cfg_path, ex_slot, _light in roster:
        rows = score_label(label, cfg_path, args.skip_coord_check)
        if not rows:
            continue
        all_rows.extend(rows)
        print(f"\n=== {label} ({ex_slot}) ===")
        header = f"  {'sample':>8} {'steps':>6} {'relL2':>9} {'R2':>9}"
        print(header)
        for r in rows:
            print(f"  {r['sample']:>8} {r['steps']:>6} {r['rel_l2_all']:>9.4f} {r['r2_all']:>9.4f}")
        mean_rel = float(np.nanmean([r['rel_l2_all'] for r in rows]))
        mean_r2 = float(np.nanmean([r['r2_all'] for r in rows]))
        print(f"  {'MEAN':>8} {'':>6} {mean_rel:>9.4f} {mean_r2:>9.4f}")

    if not all_rows:
        print("\nNo rollouts scored. Run configs/benchmarks_all/train_all.sh then infer_all.sh first.")
        return

    print("\n" + "=" * 60)
    print(f"{'label':<22} {'n':>4} {'mean relL2':>12} {'mean R2':>10}")
    print("-" * 60)
    by_label = {}
    for r in all_rows:
        by_label.setdefault(r['label'], []).append(r)
    for label in sorted(by_label):
        rows = by_label[label]
        print(f"{label:<22} {len(rows):>4} "
              f"{float(np.nanmean([r['rel_l2_all'] for r in rows])):>12.4f} "
              f"{float(np.nanmean([r['r2_all'] for r in rows])):>10.4f}")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or '.', exist_ok=True)
        num_channels = max(len(r['rel_l2']) for r in all_rows)
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['label', 'sample', 'steps', 'rel_l2_all', 'r2_all']
                            + [f'rel_l2_ch{i}' for i in range(num_channels)]
                            + [f'r2_ch{i}' for i in range(num_channels)])
            for r in all_rows:
                writer.writerow([r['label'], r['sample'], r['steps'],
                                 f"{r['rel_l2_all']:.6g}", f"{r['r2_all']:.6g}"]
                                + [f"{v:.6g}" for v in r['rel_l2']]
                                + [f"{v:.6g}" for v in r['r2']])
        print(f"\nWrote {len(all_rows)} rows to {args.csv}")


if __name__ == '__main__':
    main()
