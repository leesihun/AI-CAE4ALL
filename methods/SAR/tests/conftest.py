"""Shared fixtures: synthetic ex3-shaped HDF5 files and a tiny CPU config.

Everything is written under pytest's tmp_path; nothing lands in the repo.
Run from `methods/SAR`:  python -B -m pytest -q -p no:cacheprovider tests/
"""
import os
import sys

import pytest

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for path in (REPO, TESTS):
    if path not in sys.path:
        sys.path.insert(0, path)

import synthetic  # noqa: E402

TINY = {
    'model': 'sar',
    'gpu_ids': -1,
    'split_seed': 42,
    'split_group_attr': 'shape_idx',
    'output_var': 3,
    'cond_var': 2,
    'use_node_types': 'True',
    'training_seed': 42,
    'vae_num_layers': 1,
    'vae_d_model': 16,
    'vae_latent_features': 4,
    'vae_training_epochs': 3,
    'vae_batch_size': 2,
    'vae_learningr': '1e-3',
    'vae_lr_patience': 10,
    'sar_num_iters': 3,
    'sar_num_cond_layers': 1,
    'sar_num_ar_layers': 1,
    'sar_num_sampler_layers': 1,
    'sar_d_model': 16,
    'sar_num_slices': 4,
    'sar_num_heads': 2,
    'sar_training_epochs': 3,
    'sar_batch_size': 2,
    'sar_learningr': '1e-3',
    'sar_lr_patience': 10,
    'sar_sampler_batch_size': 2,
    'val_interval': 1,
    'test_interval': 2,
    'display_testset': 'True',
    'display_trainset': 'True',
    'viz_num_samples': 4,
    'num_samples': 5,
    'sample_batch_size': 2,
    'sample_steps': '2, 2, 1',
    'spread_channel': '0, 1, 2',
    'spread_stat': 'mag_max',
    'spread_label': 'u',
    'show_histogram': 'False',
    'histogram_bins': 10,
}


@pytest.fixture
def case(tmp_path):
    """Synthetic train / infer files and a `write_config(**overrides)` helper."""
    train = synthetic.write_dataset(str(tmp_path / 'train.h5'), num_shapes=10, draws=4, grid=10)
    infer = synthetic.write_dataset(str(tmp_path / 'infer.h5'), num_shapes=2, draws=3, grid=10,
                                    start_sid=500, seed=3)
    out = tmp_path / 'output'

    def write_config(mode='train', name='config.txt', **overrides):
        values = dict(TINY)
        values.update({
            'mode': mode,
            'dataset_dir': train,
            'infer_dataset': infer,
            'eval_dataset': infer,
            'modelpath': str(out / 'model.pth'),
            'vae_modelpath': str(out / 'vae.pth'),
            'inference_output_dir': str(out / 'infer'),
            'log_file_dir': str(out / 'train.log'),
        })
        values.update(overrides)
        path = tmp_path / name
        with open(path, 'w', encoding='utf-8') as fh:
            for key, value in values.items():
                if value is not None:
                    fh.write(f'{key} {value}\n')
        return str(path)

    class Case:
        pass
    c = Case()
    c.tmp = tmp_path
    c.train = train
    c.infer = infer
    c.out = out
    c.write_config = write_config
    return c


@pytest.fixture
def run_main(monkeypatch):
    """Run SAR_main.main with cwd at the repo, as the launcher does."""
    def run(config_path):
        monkeypatch.chdir(REPO)
        import SAR_main
        return SAR_main.main(['--config', config_path])
    return run
