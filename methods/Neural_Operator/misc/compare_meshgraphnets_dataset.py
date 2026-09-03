#!/usr/bin/env python3
"""Independently check this repository's dataset loader against the pinned
MeshGraphNets checkout (IMPLEMENTATION_PLAN.md section 4.7).

Workflow:
    1. Copy misc/_mgn_dump_helper.py into ..\\MeshGraphNets and run it there
       (it needs MGN's own general_modules.mesh_dataset), producing an .npz.
    2. Run this script in this repository, pointing at that .npz plus the
       same config this repo would use for the same dataset file.

Checks: identical train/val/test scene ID split; matching raw reference
positions, physical input, and target for the same (sample_id, time_idx)
pairs; and node/delta normalization statistics within a documented tolerance
(the two repos compute the same moments differently -- MGN's is a global
online accumulator, section 4.4's port -- so exact float equality is not
expected, but agreement should be tight).

Usage:
    python misc/compare_meshgraphnets_dataset.py --config ex1/config_train_deeponet.txt \
        --mgn-dump mgn_dump_ex1.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mgn-dump', required=True)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()

    config = load_config(args.config)
    dump = np.load(args.mgn_dump)

    dataset = MeshGraphDataset(config['dataset_dir'], dict(config))
    split_seed = int(config.get('split_seed', 42))
    train, val, test = dataset.split(0.8, 0.1, 0.1, seed=split_seed)

    issues = []

    def check_split(name, ours, theirs_key):
        theirs = set(int(s) for s in dump[theirs_key])
        ours_set = set(ours)
        if ours_set != theirs:
            issues.append(
                f"{name} split IDs differ: ours has {len(ours_set)}, MGN dump has "
                f"{len(theirs)}, symmetric difference size "
                f"{len(ours_set.symmetric_difference(theirs))}"
            )

    check_split('train', train.sample_ids, 'train_sample_ids')
    check_split('val', val.sample_ids, 'val_sample_ids')
    check_split('test', test.sample_ids, 'test_sample_ids')

    num_records = int(dump['num_records'])
    by_sample = {sid: idx for idx, sid in enumerate(train.sample_ids)}
    for i in range(num_records):
        sid = int(dump[f'record_{i}_sample_id'])
        time_idx = int(dump[f'record_{i}_time_idx'])
        if sid not in by_sample:
            issues.append(f"MGN dump sample {sid} not found in our train split")
            continue

        our_item = None
        for j in range(len(train)):
            item = train[j]
            item_time = int(item.time_idx) if item.time_idx is not None else -1
            if int(item.sample_id) == sid and item_time == time_idx:
                our_item = item
                break
        if our_item is None:
            issues.append(f"sample {sid} time_idx {time_idx}: not found via direct scan")
            continue

        their_pos = dump[f'record_{i}_pos']
        their_x = dump[f'record_{i}_x']
        their_y = dump[f'record_{i}_y']

        if not np.allclose(our_item.pos.numpy(), their_pos, atol=args.atol):
            issues.append(f"sample {sid} t={time_idx}: raw position mismatch "
                          f"(max diff {np.abs(our_item.pos.numpy() - their_pos).max():.4g})")
        if not np.allclose(our_item.x.numpy()[:, :their_x.shape[1]], their_x, atol=args.atol):
            issues.append(f"sample {sid} t={time_idx}: normalized x mismatch")
        if not np.allclose(our_item.y.numpy(), their_y, atol=args.atol):
            issues.append(f"sample {sid} t={time_idx}: normalized target mismatch")

    node_diff = np.abs(train.node_mean[:len(dump['node_mean'])] - dump['node_mean']).max()
    delta_diff = np.abs(train.delta_mean - dump['delta_mean']).max()
    print(f"node_mean max abs diff: {node_diff:.6g}")
    print(f"delta_mean max abs diff: {delta_diff:.6g}")
    if node_diff > args.atol or delta_diff > args.atol:
        issues.append(f"normalization stats differ beyond atol={args.atol}")

    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)

    print("\nAll checks passed: dataset loader matches the pinned MeshGraphNets checkout.")


if __name__ == '__main__':
    main()
