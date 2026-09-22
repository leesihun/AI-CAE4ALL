"""Read-only matrix checks. No training, HDF5 mutation, or fake checkpoints."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def audit_native(repository):
    """One fresh interpreter per independent method namespace."""
    from generate import build
    from cae_suite.config_parser import parse_config
    method_root = ROOT / 'methods' / repository
    os.chdir(method_root)
    sys.path.insert(0, str(method_root))
    from general_modules.load_config import load_config
    count = 0
    deferred = 0
    for name in build():
        if not name.startswith(f'configs/{repository}/') or not name.endswith('.txt'):
            continue
        parsed = parse_config(ROOT / name).values
        try:
            native = load_config(str(ROOT / name))
        except FileNotFoundError as exc:
            if repository == 'Transolver' and parsed['mode'] == 'inference' and 'modelpath does not exist' in str(exc):
                deferred += 1
                continue
            raise
        for key, value in parsed.items():
            actual = native.get(key)
            # Some native loaders normalize gpu_ids / indices to lists.
            if actual != value and actual != [value]:
                raise AssertionError((name, key, value, actual))
        count += 1
    print(f'{repository}: {count} native configurations parsed and values preserved.')
    if deferred:
        print(f'{deferred} native inference checks deferred: paired training checkpoints do not exist yet.')


def audit_data():
    import h5py
    import numpy as np
    from generate import CASES
    for c in CASES:
        expected = 3 + len(c['output']) + len(c['conditioner']) + bool(c['partNo'])
        for split in ('train', 'infer'):
            with h5py.File(ROOT / 'dataset' / c[split], 'r') as f:
                for sid in f['data']:
                    a = f[f'data/{sid}/nodal_data']
                    assert a.shape[0] == expected, (c['key'], sid, a.shape, expected)
                    assert max(a.shape[1]-1, 1) == c['steps'], (c['key'], sid, a.shape)
                    probe = a[:, 0, ::max(1, a.shape[-1] // 97)]
                    assert np.isfinite(probe).all(), (c['key'], sid, 'nonfinite probe')
                    # Inactive coordinate axes must be genuinely constant.
                    if c['dim'] == 2:
                        assert np.ptp(probe[2]) < 0.0001, (c['key'], sid, 'not planar')
                print(f"{c['category']}/{c['key']} {split}: {len(f['data'])} sample layouts checked")
    # This supplements, not replaces, the exact fixed-topology check in prepare.py.
    print('All mesh layouts checked; finite-value checks above are sampled, not exhaustive field certification.')


def audit_specs():
    from cae_suite.preflight import run_preflight, PreflightOptions
    from cae_suite.registry import MethodRegistry
    from cae_suite.settings import LocalSettings
    from cae_suite.diagnostics import Severity
    from generate import build
    registry = MethodRegistry(ROOT)
    problems = {}
    for name in build():
        if not name.endswith('.txt'):
            continue
        result = run_preflight(ROOT / name, suite_root=ROOT, registry=registry,
                               settings=LocalSettings(base_dir=ROOT),
                               options=PreflightOptions(skip_filesystem=True, skip_native=True,
                                                        skip_environment=True, skip_dataset=True))
        for d in result.report.diagnostics:
            if d.severity in (Severity.ERROR, Severity.WARNING):
                key = f'{d.severity.value} {d.code}: {d.message}'
                problems.setdefault(key, []).append(name)
    for problem, paths in problems.items():
        print(f'{problem}\n  {len(paths)} files; first: {paths[0]}')
    if problems:
        raise SystemExit(f'{len(problems)} distinct warnings/errors')
    print('172 configs: launcher schemas clean (filesystem/checkpoints/GPU intentionally not claimed).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', choices=['MeshGraphNets', 'MeshGraphNets_Variational', 'HI_MGNFlow',
                                           'Neural_Operator', 'Transolver', 'SimulGenVAE', 'SDFFlow'])
    parser.add_argument('--data', action='store_true')
    args = parser.parse_args()
    if args.native:
        audit_native(args.native)
    elif args.data:
        audit_data()
    else:
        audit_specs()
