"""Prepare derived inputs without modifying any source dataset.

Run from the suite root: python configs/campaigns/dataset_matrix/prepare.py
Existing artifacts are verified, never silently overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'dataset/derived/config_matrix'
CRM_ORDER = [0, 1, 2, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 3, 4, 5, 7]
CRM_FIELDS = ['x_coord', 'y_coord', 'z_coord', 'cp', 'cf_x', 'cf_y', 'cf_z',
              'mach', 'aoa_deg', 'aileron_inboard_deg', 'aileron_outboard_deg',
              'elevator_deg', 'htp_deg', 'normal_x', 'normal_y', 'normal_z', 'surface_area']


def sample_ids(f):
    return sorted(f['data'], key=int)


def reorder_crm(source, destination):
    """Relative VDS + external topology links: zero-copy, relocatable with dataset/."""
    import os
    relative_source = os.path.relpath(source, destination.parent).replace('\\', '/')
    with h5py.File(source, 'r') as src:
        old_names = [x.decode() if isinstance(x, bytes) else str(x)
                     for x in src['metadata/feature_names'][:]]
        assert [old_names[i] for i in CRM_ORDER] == CRM_FIELDS
        if destination.exists():
            with h5py.File(destination, 'r') as dst:
                assert dst.attrs['matrix_source'] == relative_source
                assert sample_ids(dst) == sample_ids(src)
                for sid in sample_ids(src):
                    # All channels, a deterministic node subset, every sample.
                    a = src[f'data/{sid}/nodal_data'][:, :, ::4096]
                    np.testing.assert_array_equal(dst[f'data/{sid}/nodal_data'][:, :, ::4096], a[CRM_ORDER])
            return
        with h5py.File(destination, 'x', libver='latest') as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs['matrix_source'] = relative_source
            dst.attrs['field_order'] = 'input output conditioner partNo'
            dst.create_dataset('metadata/feature_names', data=np.array(CRM_FIELDS, dtype=h5py.string_dtype()))
            for sid in sample_ids(src):
                arr = src[f'data/{sid}/nodal_data']
                layout = h5py.VirtualLayout(shape=arr.shape, dtype=arr.dtype)
                origin = h5py.VirtualSource(relative_source, arr.name, shape=arr.shape)
                for row, old in enumerate(CRM_ORDER):
                    layout[row, :, :] = origin[old, :, :]
                group = dst.create_group(f'data/{sid}')
                group.create_virtual_dataset('nodal_data', layout, fillvalue=np.nan)
                for key in src[f'data/{sid}']:
                    if key != 'nodal_data':
                        group[key] = h5py.ExternalLink(relative_source, f'/data/{sid}/{key}')
    reorder_crm(source, destination)  # read-back validation


def make_conditions(source, destination, kind):
    rows, ids = [], []
    with h5py.File(source, 'r') as f:
        for sid in sample_ids(f):
            arr = f[f'data/{sid}/nodal_data']
            if kind == 'flag':
                # Frame-major, then ux/uy/uz, then node index. No future frame.
                row = arr[3:6, :2, :].transpose(1, 0, 2).reshape(-1)
            else:
                # Spatial conditions must not be replaced by an arbitrary node.
                # Known reference xy, initial ux/uy and die profile; no future state.
                first = arr[:, 0, :]
                row = first[[0, 1, 3, 4, 6], :].reshape(-1)
            rows.append(row)
            ids.append(int(sid))
    values = np.stack(rows).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite conditioner: {source}')
    if destination.exists():
        np.testing.assert_allclose(np.loadtxt(destination, delimiter=',', ndmin=2), values, rtol=1e-7, atol=1e-8)
    else:
        with destination.open('x', encoding='utf-8', newline='') as stream:
            np.savetxt(stream, values, delimiter=',', fmt='%.9g')
    sidecar = destination.with_suffix('.json')
    payload = {'source': str(source.relative_to(ROOT)).replace('\\', '/'),
               'sample_ids': ids, 'shape': list(values.shape),
               'order': ('frame(0,1), displacement(x,y,z), node' if kind == 'flag'
                         else 'field(x,y,ux_initial,uy_initial,die_profil), node')}
    if sidecar.exists():
        assert json.loads(sidecar.read_text(encoding='utf-8')) == payload
    else:
        sidecar.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'{destination.relative_to(ROOT)}: {values.shape}', flush=True)


def check_dense_topology(paths):
    reference = None
    for path in paths:
        with h5py.File(path, 'r') as f:
            for sid in sample_ids(f):
                grp = f[f'data/{sid}']
                signature = (grp['nodal_data'].shape[1:],
                             hashlib.sha256(grp['mesh_edge'][:].tobytes()).hexdigest())
                if reference is None:
                    reference = signature
                if signature != reference:
                    raise ValueError(f'Dense topology differs: {path}, sample {sid}')
    print('Dense topology verified: ' + ', '.join(p.name for p in paths), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-topology-audit', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in ('', '_infer'):
        reorder_crm(ROOT / f'dataset/deterministic/ex3_NASA_CRM_mid{s}.h5',
                    OUT / f'ex3_NASA_CRM_mid_canonical{s}.h5')
    for name, kind in [('ex6_flag_simple', 'flag'), ('ex9_plasticity', 'plasticity')]:
        for suffix in ('', '_infer'):
            make_conditions(ROOT / f'dataset/deterministic/{name}{suffix}.h5',
                            OUT / f'{name}{suffix}_conditions.csv', kind)
    if not args.skip_topology_audit:
        for resolution in ('full', 'mid'):
            check_dense_topology([ROOT / f'dataset/deterministic/ex3_NASA_CRM_{resolution}{s}.h5'
                                  for s in ('', '_infer')])
        for name in ('ex6_flag_simple', 'ex9_plasticity'):
            check_dense_topology([ROOT / f'dataset/deterministic/{name}{s}.h5' for s in ('', '_infer')])


if __name__ == '__main__':
    main()
