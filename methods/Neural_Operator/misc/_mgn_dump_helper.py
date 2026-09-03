#!/usr/bin/env python3
"""Dump reference values from the pinned MeshGraphNets checkout for parity
comparison (IMPLEMENTATION_PLAN.md section 4.7's compare_meshgraphnets_dataset
tool). This file is NOT run from this repository -- it has no dependency on
`..\\MeshGraphNets`'s code at import time here; it is meant to be **copied
into** the MeshGraphNets checkout and run there, where its own
`general_modules.mesh_dataset.MeshGraphDataset` is importable.

Usage (run from inside ..\\MeshGraphNets, after copying this file there):
    python _mgn_dump_helper.py --dataset dataset/ex1.h5 --split-seed 42 \
        --input-var 4 --output-var 4 --positional-features 4 \
        --use-node-types --out mgn_dump_ex1.npz

Produces an .npz with, for the first few train-split samples: sample_id,
time_idx, raw reference positions, raw physical input, raw target delta, and
the fitted node/delta normalization stats. `compare_meshgraphnets_dataset.py`
in this repository reads that .npz and compares it against this repo's own
MeshGraphDataset output for the same samples/config.
"""

import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--input-var', type=int, required=True)
    parser.add_argument('--output-var', type=int, required=True)
    parser.add_argument('--positional-features', type=int, default=0)
    parser.add_argument('--use-node-types', action='store_true')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    # Imports deliberately local and MGN-specific: this script only works
    # when copied into the MeshGraphNets checkout.
    from general_modules.mesh_dataset import MeshGraphDataset

    config = {
        'input_var': args.input_var,
        'output_var': args.output_var,
        'positional_features': args.positional_features,
        'use_node_types': args.use_node_types,
        'edge_var': 8,
        'use_world_edges': False,
        'use_multiscale': False,
    }
    dataset = MeshGraphDataset(args.dataset, config)
    train, val, test = dataset.split(0.8, 0.1, 0.1, seed=args.split_seed)

    records = []
    for i in range(min(args.num_samples, len(train))):
        item = train[i]
        records.append({
            'sample_id': int(item.sample_id),
            'time_idx': int(item.time_idx) if item.time_idx is not None else -1,
            'pos': item.pos.numpy(),
            'x': item.x.numpy(),
            'y': item.y.numpy(),
        })

    np.savez(
        args.out,
        train_sample_ids=np.array(sorted(train.sample_ids)),
        val_sample_ids=np.array(sorted(val.sample_ids)),
        test_sample_ids=np.array(sorted(test.sample_ids)),
        node_mean=train.node_mean, node_std=train.node_std,
        delta_mean=train.delta_mean, delta_std=train.delta_std,
        num_records=len(records),
        **{f'record_{i}_sample_id': r['sample_id'] for i, r in enumerate(records)},
        **{f'record_{i}_time_idx': r['time_idx'] for i, r in enumerate(records)},
        **{f'record_{i}_pos': r['pos'] for i, r in enumerate(records)},
        **{f'record_{i}_x': r['x'] for i, r in enumerate(records)},
        **{f'record_{i}_y': r['y'] for i, r in enumerate(records)},
    )
    print(f"Wrote {args.out} with {len(records)} sample records "
          f"({len(train)} train / {len(val)} val / {len(test)} test total).")


if __name__ == '__main__':
    main()
