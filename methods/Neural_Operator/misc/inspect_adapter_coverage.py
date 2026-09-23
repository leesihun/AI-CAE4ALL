#!/usr/bin/env python3
"""Grid coverage statistics across a train split (IMPLEMENTATION_PLAN.md
sections 8.3/9). More thorough than the single-sample check inside
misc/audit_input_identifiability.py: scans every training sample and reports
min/median/max occupancy of the FNO/DeepONet grid.

Usage:
    python misc/inspect_adapter_coverage.py --config ex1/config_train_fno.txt
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset
from model.adapters.coordinate_domain import CoordinateDomain
from model.adapters.grid import splat
from model.utils import parse_int_tuple


def inspect_grid(train_dataset, resolution, max_samples: int):
    domain = CoordinateDomain.from_dataset(train_dataset, out_of_bounds_policy='clamp')
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    occupancies = []
    for i, batch in enumerate(loader):
        if i >= max_samples:
            break
        c01, oob = domain.to_unit_box(batch.pos_normalized)
        values = torch.ones(c01.shape[0], 1)
        _, occ, _ = splat(values, c01, batch.batch, 1, resolution)
        occupancies.append(occ.mean().item())
        if oob > 0:
            print(f"  sample {i}: {oob} out-of-bounds point(s)")
    occupancies = np.array(occupancies)
    print(f"  Occupancy over {len(occupancies)} samples: "
          f"min={occupancies.min():.3f} median={np.median(occupancies):.3f} max={occupancies.max():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--max-samples', type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = MeshGraphDataset(config['dataset_dir'], dict(config))
    train, val, test = dataset.split(0.8, 0.1, 0.1, seed=int(config.get('split_seed', 42)))
    d = train.operator_dim

    model_name = config.get('model')
    print(f"=== Adapter coverage for model='{model_name}' ===")

    if model_name in ('fno', 'deeponet') or 'fno_grid_resolution' in config:
        res_key = 'fno_grid_resolution' if model_name == 'fno' else 'deeponet_sensor_resolution'
        if res_key in config:
            resolution = parse_int_tuple(config[res_key], d, res_key)
            print(f"\n-- {res_key}={resolution} --")
            inspect_grid(train, resolution, args.max_samples)


if __name__ == '__main__':
    main()
