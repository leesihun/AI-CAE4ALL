"""CPU tests for `fm_latent_draws` (train_fm._encode_split with draws > 1).

A randomly initialised tiny VAE (the checked-in ex1 training config shrunk to
CPU sizes, as in test_conditional_inference.py) encodes a 6-shape synthetic
HDF5. Checked: draws=1 is the plain encoder-mean cache; draws=K is K
draw-major blocks with conditions tiled, reproducible from the seed, different
across seeds and across draws, and leaves the dataset's flags as it found them.

Run from `methods/SDFFlow`:  python -m pytest -q tests/test_fm_latent_draws.py
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
h5py = pytest.importorskip('h5py')

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.dirname(os.path.dirname(REPO))
sys.path.insert(0, REPO)

from general_modules.load_config import load_config                   # noqa: E402
from general_modules.sdf_dataset import build_dataset_splits          # noqa: E402
from general_modules.sdf_sampling import COND_NAMES, synthetic_sample  # noqa: E402
from model.sdf_vae import SDFVAE                                      # noqa: E402
from training_profiles.train_fm import _encode_split                 # noqa: E402

CONFIG_TRAIN = os.path.join(SUITE, 'configs', 'SDFFlow', 'geometry_generation',
                            'ex1', 'baseline', 'config_train_sdfflow.txt')
NUM_SHAPES = 6


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    path = str(tmp_path_factory.mktemp('draws') / 'synthetic.h5')
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as h5:
        shapes = h5.create_group('shapes')
        for i in range(NUM_SHAPES):
            sample, cond = synthetic_sample(rng, 512, 512, 256, mc_resolution=32)
            grp = shapes.create_group(f'{i:05d}')
            for key, arr in sample.items():
                grp.create_dataset(key, data=arr)
            grp.create_dataset('cond', data=cond)
            grp.attrs['source'] = f'synthetic_{i}'
        h5.attrs['num_shapes'] = NUM_SHAPES
        h5.attrs['cond_names'] = COND_NAMES

    config = load_config(CONFIG_TRAIN)
    config.update({
        'dataset_dir': path, 'split_seed': 42, 'encode_batch_size': 4,
        'latent_tokens': 1, 'latent_dim': 16, 'decoder_type': 'mlp', 'decoder_hidden': 32,
        'decoder_layers': 2, 'encoder_dim': 32, 'encoder_heads': 2, 'encoder_blocks': 1,
        'fourier_bands': 2, 'num_encoder_points': 256, 'num_query_points': 256,
    })
    torch.manual_seed(0)
    vae = SDFVAE(config).eval()
    train_ds, val_ds, test_ds = build_dataset_splits(dict(config), 42)
    yield vae, train_ds, config
    for ds in (train_ds, val_ds, test_ds):
        ds.close()


def test_one_draw_is_the_encoder_mean_cache(world):
    vae, ds, config = world
    z, c = _encode_split(vae, ds, 'cpu', config)
    ds.deterministic = True
    try:
        with torch.no_grad():
            ref = []
            for i in range(len(ds)):
                item = ds[i]
                mu, _ = vae.encode(item['surface_points'][None], item['surface_normals'][None])
                ref.append(mu.flatten(1))
    finally:
        ds.deterministic = False
    assert torch.allclose(z, torch.cat(ref), atol=1e-6)
    assert c.shape[0] == len(ds)


def test_k_draws_are_draw_major_seeded_and_distinct(world):
    vae, ds, config = world
    n, k = len(ds), 3
    z_mean, c_mean = _encode_split(vae, ds, 'cpu', config)
    ds.deterministic, ds.draw = False, 0
    a, ca = _encode_split(vae, ds, 'cpu', config, draws=k, draw_seed=42)
    b, _ = _encode_split(vae, ds, 'cpu', config, draws=k, draw_seed=42)
    d, _ = _encode_split(vae, ds, 'cpu', config, draws=k, draw_seed=7)

    assert a.shape == (k * n, z_mean.shape[1])
    assert torch.equal(ca, c_mean.repeat(k, 1))           # conditions tiled draw-major
    assert torch.equal(a, b)                              # reproducible from the seed
    assert not torch.equal(a, d)                          # the seed matters
    blocks = a.view(k, n, -1)
    for j in range(k):
        assert not torch.allclose(blocks[j], z_mean)      # a sample, not the mean
    assert not torch.allclose(blocks[0], blocks[1])       # draws differ from each other
    assert ds.deterministic is False and ds.draw == 0     # flags restored


def test_draw_index_changes_the_fixed_subsample(world):
    _, ds, _ = world
    ds.deterministic = True
    try:
        ds.draw = 0
        p0 = ds[0]['surface_points']
        ds.draw = 1
        p1 = ds[0]['surface_points']
        p1_again = ds[0]['surface_points']
    finally:
        ds.deterministic, ds.draw = False, 0
    assert torch.equal(p1, p1_again)
    assert not torch.equal(p0, p1)
