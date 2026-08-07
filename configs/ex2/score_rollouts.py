#!/usr/bin/env python3
"""Score ex2 rollouts against the held-out ground truth in dataset/ex2_infer.h5.

The inference paths (MeshGraphNets/Transolver/Neural_Operator
`inference_profiles/rollout.py`) seed the rollout from the stored t=0 state and
write predictions only -- nothing in the repo compares them back to the stored
trajectory. This does that: for every method that produced
`output/<method>/rollout/ex2/<arm>/rollout_sample{id}_steps{N}.h5`, it lines the
prediction up with `data/{id}/nodal_data[3:3+C, t, :]` in the GT file and
reports per-channel relative L2 and RMSE, plus the error-vs-time curve.

Node order is guaranteed to match: the writers copy `ref_pos` straight from the
GT file's t=0 coordinate rows without reordering. That is verified here anyway
(--skip-coord-check turns it off).

Usage:
    python configs/ex2/score_rollouts.py
    python configs/ex2/score_rollouts.py --methods meshgraphnets-hi transolver
    python configs/ex2/score_rollouts.py --csv output/ex2_head_to_head/scores.csv
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
GT_PATH = os.path.join(REPO_ROOT, 'dataset', 'ex2_infer.h5')

# method label -> rollout directory, mirroring configs/ex2/infer_all.sh.
ROLLOUT_DIRS = {
    'meshgraphnets':    'output/meshgraphnets/rollout/ex2/model_vanilla',
    'meshgraphnets-hi': 'output/meshgraphnets/rollout/ex2/model_himgn',
    'himgn-base':       'output/meshgraphnets/rollout/ex2/model_himgn_base',
    'himgn-p1':         'output/meshgraphnets/rollout/ex2/model_himgn_p1',
    'himgn-p2':         'output/meshgraphnets/rollout/ex2/model_himgn_p2',
    'himgn-p12':        'output/meshgraphnets/rollout/ex2/model_himgn_p12',
    'deeponet':         'output/neural_operator/rollout/ex2/deeponet',
    'fno':              'output/neural_operator/rollout/ex2/fno',
    'gino':             'output/neural_operator/rollout/ex2/gino',
    'point_deeponet':   'output/neural_operator/rollout/ex2/point_deeponet',
    'transolver':       'output/transolver/rollout/ex2/transolver',
}

CHANNEL_NAMES = ['x_disp', 'y_disp', 'z_disp', 'stress']


def load_gt(path):
    """{sample_id: (state[T, N, C], coords[N, 3], scene_name)} for the whole GT file."""
    gt = {}
    with h5py.File(path, 'r') as f:
        for key in sorted(f['data'].keys(), key=int):
            nodal = f[f'data/{key}/nodal_data']
            num_features = nodal.shape[0]
            # rows 0:3 coords, last row node types -> everything between is state.
            num_state = num_features - 4
            state = nodal[3:3 + num_state, :, :]          # [C, T, N]
            gt[int(key)] = (
                np.transpose(state, (1, 2, 0)).astype(np.float64),   # [T, N, C]
                nodal[:3, 0, :].T.astype(np.float64),                # [N, 3]
                dict(f[f'data/{key}/metadata'].attrs).get('source_filename', key),
            )
    return gt


def read_rollout(path):
    """(pred[T, N, C], coords[N, 3]) from one rollout_sample*.h5."""
    with h5py.File(path, 'r') as f:
        sample_key = list(f['data'].keys())[0]
        nodal = f[f'data/{sample_key}/nodal_data'][:]     # [3 + C + 1, T, N]
        num_state = nodal.shape[0] - 4
        pred = np.transpose(nodal[3:3 + num_state], (1, 2, 0)).astype(np.float64)
        coords = nodal[:3, 0, :].T.astype(np.float64)
    return pred, coords


def rel_l2(pred, truth):
    """Relative L2 over all nodes and timesteps; NaN when the truth is all zeros."""
    denom = np.linalg.norm(truth)
    if denom == 0.0:
        return float('nan')
    return float(np.linalg.norm(pred - truth) / denom)


def score_method(label, rollout_dir, gt, skip_coord_check):
    files = sorted(
        glob.glob(os.path.join(rollout_dir, 'rollout_sample*_steps*.h5')),
        key=lambda p: int(re.search(r'rollout_sample(\d+)_', os.path.basename(p)).group(1)),
    )
    if not files:
        return []

    rows = []
    for path in files:
        sample_id = int(re.search(r'rollout_sample(\d+)_', os.path.basename(path)).group(1))
        if sample_id not in gt:
            print(f"  [{label}] sample {sample_id} has no ground truth in {GT_PATH} -- skipped")
            continue
        truth_all, gt_coords, scene = gt[sample_id]
        pred_all, pred_coords = read_rollout(path)

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
        channels = min(pred_all.shape[2], truth_all.shape[2])
        # t=0 is the seeded initial condition and is identical by construction,
        # so it is excluded -- scoring it would dilute the rollout error.
        pred = pred_all[1:steps, :, :channels]
        truth = truth_all[1:steps, :, :channels]

        per_channel_rel = [rel_l2(pred[:, :, c], truth[:, :, c]) for c in range(channels)]
        per_channel_rmse = [float(np.sqrt(np.mean((pred[:, :, c] - truth[:, :, c]) ** 2)))
                            for c in range(channels)]
        # error vs rollout horizon, on the displacement magnitude
        disp_c = min(3, channels)
        per_step = [
            float(np.sqrt(np.mean(np.sum((pred[t, :, :disp_c] - truth[t, :, :disp_c]) ** 2, axis=1))))
            for t in range(pred.shape[0])
        ]

        rows.append({
            'method': label,
            'sample': sample_id,
            'scene': scene,
            'steps': pred.shape[0],
            'rel_l2_all': rel_l2(pred, truth),
            'rel_l2': per_channel_rel,
            'rmse': per_channel_rmse,
            'per_step_disp_rmse': per_step,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt', default=GT_PATH, help='ground-truth HDF5 (default: dataset/ex2_infer.h5)')
    ap.add_argument('--methods', nargs='*', default=None,
                    help=f'subset of {sorted(ROLLOUT_DIRS)} (default: all present)')
    ap.add_argument('--csv', default=None, help='also write per-sample scores here')
    ap.add_argument('--skip-coord-check', action='store_true',
                    help='do not verify that rollout ref coords match the GT node order')
    args = ap.parse_args()

    if not os.path.exists(args.gt):
        sys.exit(f"Ground-truth file not found: {args.gt}")
    gt = load_gt(args.gt)
    print(f"Ground truth: {args.gt} ({len(gt)} scenes)")

    labels = args.methods if args.methods else list(ROLLOUT_DIRS)
    all_rows = []
    for label in labels:
        if label not in ROLLOUT_DIRS:
            print(f"Unknown method '{label}' -- known: {', '.join(ROLLOUT_DIRS)}")
            continue
        rollout_dir = os.path.join(REPO_ROOT, ROLLOUT_DIRS[label])
        rows = score_method(label, rollout_dir, gt, args.skip_coord_check)
        if not rows:
            if args.methods:
                print(f"\n[{label}] no rollouts found in {ROLLOUT_DIRS[label]}")
            continue
        all_rows.extend(rows)

        print(f"\n=== {label}  ({ROLLOUT_DIRS[label]}) ===")
        header = f"  {'scene':<32} {'steps':>5} {'relL2':>9} " + \
                 ' '.join(f"{n:>10}" for n in CHANNEL_NAMES[:len(rows[0]['rel_l2'])])
        print(header)
        for r in rows:
            chans = ' '.join(f"{v:>10.4f}" for v in r['rel_l2'])
            print(f"  {str(r['scene']):<32} {r['steps']:>5} {r['rel_l2_all']:>9.4f} {chans}")
        mean_all = float(np.mean([r['rel_l2_all'] for r in rows]))
        print(f"  {'MEAN':<32} {'':>5} {mean_all:>9.4f}")
        first, last = rows[0]['per_step_disp_rmse'][0], rows[0]['per_step_disp_rmse'][-1]
        print(f"  displacement RMSE growth on {rows[0]['scene']}: "
              f"step 1 = {first:.4g} -> step {rows[0]['steps']} = {last:.4g}")

    if not all_rows:
        print("\nNo rollouts scored. Run configs/ex2/infer_all.sh first.")
        return

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or '.', exist_ok=True)
        num_channels = len(all_rows[0]['rel_l2'])
        names = CHANNEL_NAMES[:num_channels]
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['method', 'sample', 'scene', 'steps', 'rel_l2_all']
                            + [f'rel_l2_{n}' for n in names]
                            + [f'rmse_{n}' for n in names])
            for r in all_rows:
                writer.writerow([r['method'], r['sample'], r['scene'], r['steps'],
                                 f"{r['rel_l2_all']:.6g}"]
                                + [f"{v:.6g}" for v in r['rel_l2']]
                                + [f"{v:.6g}" for v in r['rmse']])
        print(f"\nWrote {len(all_rows)} rows to {args.csv}")


if __name__ == '__main__':
    main()
