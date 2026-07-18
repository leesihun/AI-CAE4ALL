#!/usr/bin/env python3
"""Identifiability and geometry audit (IMPLEMENTATION_PLAN.md section 4.7).

Run before long training. Reports:
  - duplicate model-visible inputs with different targets;
  - repeated static geometries whose responses differ without a visible condition;
  - temporal states with conflicting next deltas;
  - SDF/condition/quadrature availability;
  - active-axis and grid-bound resolution;
  - graph/grid coverage statistics for candidate FNO/GINO resolutions.

Usage:
    python misc/audit_input_identifiability.py --config ex1/config_train_deeponet.txt
"""

import argparse
import hashlib
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_modules.load_config import load_config
from general_modules.mesh_dataset import MeshGraphDataset
from model.adapters.radius_neighbors import radius_neighbors_scipy, neighbor_stats, min_reachable_radius


def _round_key(arr: np.ndarray, decimals: int = 4) -> bytes:
    return np.round(arr.astype(np.float64), decimals).tobytes()


def audit_static_duplicates(h5_file: str, sample_ids, output_var: int) -> list:
    """Flag pairs of static samples whose stored fields are near-identical
    reference geometry hashes but materially different targets, and vice
    versa (section 4.7's "repeated geometry ... responses differ")."""
    issues = []
    seen = {}
    with h5py.File(h5_file, 'r') as f:
        for sid in sample_ids:
            data = f[f'data/{sid}/nodal_data'][:]
            if data.shape[1] != 1:
                return issues  # temporal dataset; static-duplicate check does not apply
            ref_pos = data[:3, 0, :]
            target = data[3:3 + output_var, 0, :]
            geom_key = _round_key(ref_pos, decimals=2)
            geom_hash = hashlib.sha1(geom_key).hexdigest()
            target_summary = float(np.mean(np.abs(target)))
            if geom_hash in seen:
                other_sid, other_summary = seen[geom_hash]
                if abs(target_summary - other_summary) > 0.1 * max(abs(target_summary), abs(other_summary), 1e-8):
                    issues.append(
                        f"samples {sid} and {other_sid}: near-identical reference geometry "
                        f"but target magnitudes differ ({target_summary:.4g} vs {other_summary:.4g}) "
                        "with no visible condition to explain it."
                    )
            else:
                seen[geom_hash] = (sid, target_summary)
    return issues


def audit_temporal_conflicts(h5_file: str, sample_ids, input_var: int, output_var: int) -> list:
    """Flag (state_t -> delta) pairs that repeat with conflicting deltas
    within the same sample's trajectory."""
    issues = []
    with h5py.File(h5_file, 'r') as f:
        for sid in sample_ids:
            data = f[f'data/{sid}/nodal_data'][:]
            T = data.shape[1]
            if T <= 1:
                continue
            seen = {}
            for t in range(T - 1):
                state = data[3:3 + input_var, t, :]
                delta = data[3:3 + output_var, t + 1, :] - state[:output_var]
                key = hashlib.sha1(_round_key(state, decimals=3)).hexdigest()
                delta_summary = float(np.mean(np.abs(delta)))
                if key in seen and abs(seen[key] - delta_summary) > 0.1 * max(abs(seen[key]), abs(delta_summary), 1e-8):
                    issues.append(
                        f"sample {sid}: state at t={t} repeats an earlier state with a "
                        f"materially different next delta ({delta_summary:.4g} vs {seen[key]:.4g})."
                    )
                seen[key] = delta_summary
    return issues


def report_grid_coverage(train_dataset, resolutions: dict) -> None:
    """Print occupancy/density diagnostics for candidate FNO/GINO resolutions."""
    from model.adapters.coordinate_domain import CoordinateDomain
    from model.adapters.grid import splat
    import torch

    domain = CoordinateDomain.from_dataset(train_dataset, out_of_bounds_policy='clamp')
    sample_id = train_dataset.sample_ids[0]
    item = train_dataset[0]
    c01, oob = domain.to_unit_box(item.pos_normalized)
    batch = torch.zeros(c01.shape[0], dtype=torch.long)

    for name, resolution in resolutions.items():
        values = torch.ones(c01.shape[0], 1)
        grid, occ, dens = splat(values, c01, batch, 1, resolution)
        occ_frac = occ.mean().item()
        print(f"  [{name}] resolution={resolution}: occupancy={occ_frac:.3f}, "
              f"min_reachable_radius={min_reachable_radius(resolution, len(resolution)):.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    h5_file = config['dataset_dir']
    input_var = config['input_var']
    output_var = config['output_var']

    with h5py.File(h5_file, 'r') as f:
        sample_ids = sorted(int(k) for k in f['data'].keys())

    print(f"=== Identifiability audit: {h5_file} ({len(sample_ids)} samples) ===\n")

    print("-- Static geometry-duplicate check --")
    issues = audit_static_duplicates(h5_file, sample_ids, output_var)
    if issues:
        for issue in issues[:20]:
            print(f"  CONFLICT: {issue}")
    else:
        print("  none found")

    print("\n-- Temporal delta-conflict check --")
    issues = audit_temporal_conflicts(h5_file, sample_ids, input_var, output_var)
    if issues:
        for issue in issues[:20]:
            print(f"  CONFLICT: {issue}")
    else:
        print("  none found (or dataset is static)")

    print("\n-- Optional-field availability --")
    sdf_source = config.get('sdf_source', 'none')
    gc_features = config.get('global_condition_features', 'none')
    print(f"  sdf_source: {sdf_source}")
    print(f"  global_condition_features: {gc_features}")
    print(f"  integration_weight_source: {config.get('integration_weight_source', 'none')}")

    print("\n-- Fitting dataset (active axes / grid bounds) --")
    dataset = MeshGraphDataset(h5_file, dict(config))
    train, val, test = dataset.split(0.8, 0.1, 0.1, seed=int(config.get('split_seed', 42)))
    print(f"  active_axes={train.active_axes}  operator_dim={train.operator_dim}")
    print(f"  grid_bound_min={train.grid_bound_min}  grid_bound_max={train.grid_bound_max}")
    print(f"  rot_invariant_radius={train.rot_invariant_radius:.4f}")

    print("\n-- Candidate grid coverage --")
    d = train.operator_dim
    candidates = {
        'fno': tuple(config.get('fno_grid_resolution', [32] * d)) if 'fno_grid_resolution' in config
               else tuple([32] * d),
        'gino': tuple(config.get('gino_grid_resolution', [32] * d)) if 'gino_grid_resolution' in config
                else tuple([32] * d),
    }
    report_grid_coverage(train, candidates)

    print("\nAudit complete.")


if __name__ == '__main__':
    main()
