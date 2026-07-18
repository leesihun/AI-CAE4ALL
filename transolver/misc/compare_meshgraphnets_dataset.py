#!/usr/bin/env python3
"""Phase 1 parity gate (IMPLEMENTATION_PLAN.md section 5.6): with identical
sample IDs, split seed, input/output widths, positional-feature count, and
node-type setting, compare this loader against MeshGraphNets'
general_modules/mesh_dataset.py:

    - exact sample_id, time_idx, and mirrored edge_index;
    - equal raw pos and part_ids;
    - close node_mean/std and delta_mean/std;
    - close normalized x and y; and
    - pos_normalized against an independently calculated reference.

Edge attributes, world edges, and multiscale tensors are intentionally outside
the parity surface (baseline Transolver does not consume them).

Usage:
    python misc/compare_meshgraphnets_dataset.py \\
      --dataset ..\\MeshGraphNets\\dataset\\ex1.h5 \\
      --meshgraphnets-root ..\\MeshGraphNets \\
      --config ex1\\config_train_smoke.txt
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset, normalize_positions


def _build_mgn_config(our_config):
    """Translate our config into the minimal MGN config needed to reproduce
    the same node/positional/node-type/target composition. edge_var/world/
    multiscale are MGN-required or MGN-only keys with no Transolver
    counterpart; fixed here to the values that keep MGN's node composition
    identical to ours."""
    return dict(
        input_var=our_config['input_var'],
        output_var=our_config['output_var'],
        positional_features=our_config.get('positional_features', 0),
        edge_var=8,
        use_node_types=our_config.get('use_node_types', False),
        use_world_edges=False,
        use_multiscale=False,
        augment_geometry=False,
        use_parallel_stats=True,
    )


def _dump_mgn_side(dataset_path, mgn_root, mgn_config, split_seed, n_items):
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_mgn_dump_helper.py')
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, 'mgn_dump.pkl')
        cmd = [
            sys.executable, helper,
            '--dataset', os.path.abspath(dataset_path),
            '--config-json', json.dumps(mgn_config),
            '--split-seed', str(split_seed),
            '--n-items', str(n_items),
            '--out', out_path,
        ]
        result = subprocess.run(cmd, cwd=mgn_root, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"MGN-side dump subprocess failed (exit {result.returncode})")
        with open(out_path, 'rb') as f:
            return pickle.load(f)


def compare(dataset_path, mgn_root, our_config, split_seed=42, n_items=5):
    failures = []

    def check(name, cond):
        status = 'OK' if cond else 'FAIL'
        print(f"  [{status}] {name}")
        if not cond:
            failures.append(name)

    print(f"Dataset: {dataset_path}")
    print(f"MeshGraphNets root: {mgn_root}")
    print(f"split_seed={split_seed}, input_var={our_config['input_var']}, "
          f"output_var={our_config['output_var']}, "
          f"positional_features={our_config.get('positional_features', 0)}, "
          f"use_node_types={our_config.get('use_node_types', False)}\n")

    our_ds = MeshGraphDataset(dataset_path, dict(our_config, augment_geometry=False))
    our_train, our_val, our_test = our_ds.split(0.8, 0.1, 0.1, seed=split_seed)

    mgn_config = _build_mgn_config(our_config)
    mgn = _dump_mgn_side(dataset_path, mgn_root, mgn_config, split_seed, n_items)

    print("--- split ---")
    check('train sample_ids match', list(our_train.sample_ids) == mgn['train_sample_ids'])
    check('val sample_ids match', list(our_val.sample_ids) == mgn['val_sample_ids'])
    check('test sample_ids match', list(our_test.sample_ids) == mgn['test_sample_ids'])

    print("--- statistics (close) ---")
    node_width = min(len(our_train.node_mean), len(mgn['node_mean']))
    check('node_mean close',
         np.allclose(our_train.node_mean[:node_width], mgn['node_mean'][:node_width], atol=1e-2, rtol=1e-2))
    check('node_std close',
         np.allclose(our_train.node_std[:node_width], mgn['node_std'][:node_width], atol=1e-2, rtol=1e-2))
    check('delta_mean close', np.allclose(our_train.delta_mean, mgn['delta_mean'], atol=1e-2, rtol=1e-2))
    check('delta_std close', np.allclose(our_train.delta_std, mgn['delta_std'], atol=1e-2, rtol=1e-2))

    print(f"--- per-item ({len(mgn['items'])} items) ---")
    for i, mgn_item in enumerate(mgn['items']):
        our_item = our_train[i]
        check(f'item {i}: sample_id exact', int(our_item.sample_id) == mgn_item['sample_id'])
        our_time_idx = getattr(our_item, 'time_idx', None)
        our_t = int(our_time_idx) if our_time_idx is not None else None
        check(f'item {i}: time_idx exact', our_t == mgn_item['time_idx'])
        check(f'item {i}: pos equal',
             np.allclose(our_item.pos.numpy(), mgn_item['pos'], atol=1e-4))
        check(f'item {i}: edge_index exact',
             np.array_equal(our_item.edge_index.numpy(), mgn_item['edge_index']))
        our_part_ids = getattr(our_item, 'part_ids', None)
        if mgn_item['part_ids'] is not None and our_part_ids is not None:
            check(f'item {i}: part_ids exact',
                 np.array_equal(our_part_ids.numpy(), mgn_item['part_ids']))
        check(f'item {i}: normalized x close',
             np.allclose(our_item.x.numpy(), mgn_item['x'], atol=5e-2, rtol=5e-2))
        check(f'item {i}: normalized y close',
             np.allclose(our_item.y.numpy(), mgn_item['y'], atol=5e-2, rtol=5e-2))

        expected_pos_norm = normalize_positions(our_item.pos.numpy(), our_train.position_scale)
        check(f'item {i}: pos_normalized matches independent reference',
             np.allclose(our_item.pos_normalized.numpy(), expected_pos_norm, atol=1e-5))

    print(f"\n{len(failures)} failure(s) out of the checks above.")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--meshgraphnets-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-items', type=int, default=5)
    args = parser.parse_args()

    our_config = load_config(args.config)
    split_seed = int(our_config.get('split_seed', 42))
    failures = compare(args.dataset, args.meshgraphnets_root, our_config, split_seed, args.n_items)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
