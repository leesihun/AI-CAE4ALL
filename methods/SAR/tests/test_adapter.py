"""The mesh-HDF5 -> SAR adapter: shapes, normalization, split parity with
MeshGraphNets_Variational, and that the hidden imperfection parameters never
reach the model."""
import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch

from conftest import REPO
from general_modules import mesh_dataset as md
from general_modules.load_config import load_config

import synthetic

MGNV = os.path.join(os.path.dirname(REPO), 'MeshGraphNets_Variational')


def _setup(case, **overrides):
    config = load_config(case.write_config(**overrides))
    train_ids, val_ids, _ = md.split_ids(config)
    attr = md.group_attr(config)
    train = md.group_ids(config['dataset_dir'], train_ids, attr)
    val = md.group_ids(config['dataset_dir'], val_ids, attr)
    stats = md.compute_normalization(config, train, log=lambda *_: None)
    return config, train, val, stats


def test_static_graph_and_fields_shapes(case):
    config, train, _, stats = _setup(case)
    key, sids = next(iter(train.items()))
    with h5py.File(config['dataset_dir'], 'r') as f:
        n = f[f'data/{sids[0]}/nodal_data'].shape[2]
        e = f[f'data/{sids[0]}/mesh_edge'].shape[1]
        static = md.build_static(f, sids[0], stats, 3, config)
        target, glob = md.read_fields(f, sids[0], stats, config)
    assert static['_num_nodes'] == n
    assert static['pos'].shape == (n, 3)
    assert static['edge_index'].shape == (2, 2 * e)          # bidirectional, no self loops
    assert static['edge_attr'].shape == (2 * e, 3)
    assert static['loc'].shape == (n, 2)                     # node-type one-hot (2 types)
    assert static['scale'].shape == (n,) and int(static['scale'].max()) == 3
    for k in (2, 3):
        assert f'pos_{k}' in static and f'edge_index_{k}' in static and f'idx{k - 1}_to_idx{k}' in static
    assert target.shape == (n, 3) and glob.shape == (n, 2)
    assert float(target.abs().max()) <= 1.0 + 1e-6
    assert md.cond_features(stats) == 2 + 2

    g = md.make_graph(static, target=target, glob=glob)
    assert g.edge_index.dtype == torch.long and g.edge_index.data_ptr() != static['edge_index'].data_ptr()


def test_normalization_roundtrip_and_train_only(case):
    config, train, val, stats = _setup(case)
    with h5py.File(config['dataset_dir'], 'r') as f:
        sid = next(iter(train.values()))[0]
        raw = f[f'data/{sid}/nodal_data'][3:6, -1, :].T
    back = md.denormalize_target(md.normalize_target(raw, stats), stats)
    np.testing.assert_allclose(back, raw, rtol=1e-6, atol=1e-6)
    # Statistics come from the train systems only.
    lo = np.full(3, np.inf)
    with h5py.File(config['dataset_dir'], 'r') as f:
        for sids in train.values():
            for s in sids:
                lo = np.minimum(lo, f[f'data/{s}/nodal_data'][3:6, -1, :].min(axis=1))
    np.testing.assert_allclose(stats['target_min'], lo, rtol=1e-6)


def test_target_scaling_standard(case):
    """`target_scaling standard`: train mean/std z-scores; minmax stays the default."""
    from training_profiles.common import check_normalization_matches
    _, _, _, default = _setup(case)
    assert default['target_scaling'] == 'minmax'
    config, train, _, stats = _setup(case, target_scaling='standard')
    assert stats['target_scaling'] == 'standard'
    rows = []
    with h5py.File(config['dataset_dir'], 'r') as f:
        for sids in train.values():
            for s in sids:
                rows.append(f[f'data/{s}/nodal_data'][3:6, -1, :].astype(np.float64))
        sid = next(iter(train.values()))[0]
        raw = f[f'data/{sid}/nodal_data'][3:6, -1, :].T
    pooled = np.concatenate(rows, axis=1)
    np.testing.assert_allclose(stats['target_mean'], pooled.mean(axis=1), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(stats['target_std'], pooled.std(axis=1), rtol=1e-5, atol=1e-9)
    z = md.normalize_target(raw, stats)
    np.testing.assert_allclose(z, (raw - pooled.mean(axis=1)) / pooled.std(axis=1), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(md.denormalize_target(z, stats), raw, rtol=1e-6, atol=1e-6)
    # A checkpoint without the key is min/max.
    legacy = {k: v for k, v in default.items() if k not in ('target_scaling', 'target_mean', 'target_std')}
    np.testing.assert_allclose(md.normalize_target(raw, legacy), md.normalize_target(raw, default))
    # The second stage and the sampler refuse a different scaling.
    check_normalization_matches(stats, config, 'vae.pth')
    with pytest.raises(ValueError, match='target_scaling'):
        check_normalization_matches(stats, dict(config, target_scaling='minmax'), 'vae.pth')
    with pytest.raises(ValueError, match='target_scaling'):
        md.target_scaling({'target_scaling': 'tanh'})


def test_dataset_items(case):
    config, train, val, stats = _setup(case)
    statics = md.build_statics(config, train, stats, 3, log=lambda *_: None)
    ds = md.SystemDrawDataset(config['dataset_dir'], train, statics, stats, config, mode='random',
                              draws_per_epoch=2)
    assert len(ds) == 2 * len(train)
    g = ds[0]
    assert int(g.sample_id[0]) in train[ds.keys[0]]
    fixed = md.SystemDrawDataset(config['dataset_dir'], val, statics, stats, config, mode='fixed',
                                 draws_per_system=2)
    assert len(fixed) == 2 * len(val)


@pytest.mark.parametrize('group', ['shape_idx', None])
def test_split_matches_meshgraphnets_variational(case, group):
    """Same ids as MGN-V's own MeshGraphDataset._resolve_split_ids (run in a
    child process with MGN-V's package on the path)."""
    config = load_config(case.write_config(split_group_attr=group))
    ours = md.split_ids(config)
    code = (
        'import json, sys\n'
        'sys.dont_write_bytecode = True\n'
        'import h5py\n'
        'from general_modules.mesh_dataset import MeshGraphDataset\n'
        'ds = MeshGraphDataset.__new__(MeshGraphDataset)\n'
        'ds.h5_file = sys.argv[1]\n'
        'with h5py.File(ds.h5_file, "r") as f:\n'
        '    ds.sample_ids = sorted(int(k) for k in f["data"].keys())\n'
        'ds.config = {"split_group_attr": sys.argv[2]} if sys.argv[2] else {}\n'
        'tr, va, te = ds._resolve_split_ids(0.8, 0.1, 0.1, 42)\n'
        'print("SPLIT" + json.dumps([[int(i) for i in tr], [int(i) for i in va], [int(i) for i in te]]))\n'
    )
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    res = subprocess.run([sys.executable, '-B', '-c', code, case.train, group or ''], cwd=MGNV,
                         env=env, capture_output=True, text=True, timeout=300)
    if res.returncode != 0 and 'ModuleNotFoundError' in res.stderr:
        pytest.skip(f'MGN-V dependencies missing: {res.stderr.strip().splitlines()[-1]}')
    assert res.returncode == 0, res.stderr
    line = [ln for ln in res.stdout.splitlines() if ln.startswith('SPLIT')][0]
    theirs = json.loads(line[5:])
    assert [list(map(int, s)) for s in ours] == theirs


def test_hidden_parameters_never_read(case, tmp_path):
    """A copy of the file whose hidden metadata and `latent` group are changed
    gives bit-identical statistics, graphs and training items."""
    config, train, val, stats = _setup(case)
    other = str(tmp_path / 'train_hidden_changed.h5')
    synthetic.write_dataset(other, num_shapes=10, draws=4, grid=10)
    with h5py.File(other, 'r+') as f:
        for sid in f['data']:
            meta = f[f'data/{sid}/metadata']
            for attr in synthetic.HIDDEN_ATTRS:
                meta.attrs[attr] = -999
            f[f'latent/{sid}'][...] = 1e6
    config2 = dict(config, dataset_dir=other)
    stats2 = md.compute_normalization(config2, train, log=lambda *_: None)
    assert stats == stats2
    s1 = md.build_statics(config, train, stats, 3, log=lambda *_: None)
    s2 = md.build_statics(config2, train, stats2, 3, log=lambda *_: None)
    d1 = md.SystemDrawDataset(config['dataset_dir'], train, s1, stats, config, mode='fixed', draws_per_system=4)
    d2 = md.SystemDrawDataset(other, train, s2, stats2, config2, mode='fixed', draws_per_system=4)
    for i in range(len(d1)):
        a, b = d1[i], d2[i]
        assert sorted(a.keys()) == sorted(b.keys())
        for k in a.keys():
            if isinstance(a[k], torch.Tensor):
                assert torch.equal(a[k], b[k]), k
    names = set(d1[0].keys())
    assert not names & set(synthetic.HIDDEN_ATTRS) and 'latent' not in names
