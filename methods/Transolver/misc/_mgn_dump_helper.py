#!/usr/bin/env python3
"""Runs INSIDE the MeshGraphNets checkout (invoked as a subprocess with
cwd=meshgraphnets_root) to dump a few split/dataset artifacts for
compare_meshgraphnets_dataset.py to diff against.

Isolated as a subprocess deliberately: both repos have a top-level
`general_modules` package, so importing MGN's dataset code in the same
process as this one risks silently shadowing one package with the other.
A subprocess with its own sys.path avoids that entirely (IMPLEMENTATION_PLAN.md
section 7: the sibling checkout is never imported in production, only as an
optional, isolated reference in parity tests).
"""
import argparse
import json
import os
import pickle
import sys

# cwd is the MeshGraphNets root (subprocess launched with cwd=meshgraphnets_root);
# sys.path[0] is this script's own directory by default, which is NOT enough.
sys.path.insert(0, os.getcwd())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config-json', required=True)
    parser.add_argument('--split-seed', type=int, required=True)
    parser.add_argument('--n-items', type=int, default=5)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    from general_modules.mesh_dataset import MeshGraphDataset  # MGN's own

    config = json.loads(args.config_json)
    dataset = MeshGraphDataset(args.dataset, config)
    train, val, test = dataset.split(0.8, 0.1, 0.1, seed=args.split_seed)

    items = []
    for i in range(min(args.n_items, len(train))):
        g = train[i]
        time_idx = getattr(g, 'time_idx', None)
        part_ids = getattr(g, 'part_ids', None)
        items.append(dict(
            sample_id=int(g.sample_id),
            time_idx=(int(time_idx) if time_idx is not None else None),
            pos=g.pos.numpy(),
            edge_index=g.edge_index.numpy(),
            part_ids=(part_ids.numpy() if part_ids is not None else None),
            x=g.x.numpy(),
            y=g.y.numpy(),
        ))

    result = dict(
        train_sample_ids=list(train.sample_ids),
        val_sample_ids=list(val.sample_ids),
        test_sample_ids=list(test.sample_ids),
        node_mean=train.node_mean,
        node_std=train.node_std,
        delta_mean=train.delta_mean,
        delta_std=train.delta_std,
        items=items,
    )
    with open(args.out, 'wb') as f:
        pickle.dump(result, f)


if __name__ == '__main__':
    sys.exit(main())
