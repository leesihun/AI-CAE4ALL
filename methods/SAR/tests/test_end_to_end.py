"""Tiny CPU runs of every mode on the synthetic file: train (VAE then SAR),
inference in MGN-V's rollout format with the spread tail, the separate
stage modes, and mid-training resume."""
import json
import os
import re

import h5py
import numpy as np
import torch

from general_modules.resume_state import TrainingResume

# configs/campaigns/dataset_matrix/brief.py reads epoch lines with this.
EPOCH_RE = re.compile(r'^Elapsed: ([\d.]+)s Epoch (\d+)')


def _epochs(path):
    with open(path, encoding='utf-8') as fh:
        return [int(m.group(2)) for m in (EPOCH_RE.match(ln) for ln in fh) if m]


def test_train_then_inference(case, run_main):
    assert run_main(case.write_config('train')) == 0
    out = case.out
    vae = torch.load(out / 'vae.pth', weights_only=True)
    sar = torch.load(out / 'model.pth', weights_only=True)
    assert vae['stage'] == 'vae' and sar['stage'] == 'sar'
    assert vae['normalization'] == sar['normalization']
    assert sar['num_scales'] == 3 and sar['vae_scales'] == 1
    assert _epochs(out / 'train_vae.log') == [0, 1, 2]
    assert _epochs(out / 'train_sar.log') == [0, 1, 2]
    pics = sorted(p.name for p in out.glob('*.png'))
    assert 'vae_parity_ep1.png' in pics and 'sar_samples_ep2.png' in pics
    assert (out / 'train' / 'sar_samples_ep2.png').exists()

    assert run_main(case.write_config('inference', name='infer.txt', save_rollouts='True')) == 0
    infer = out / 'infer'
    with h5py.File(case.infer, 'r') as f:
        scenes = sorted(int(k) for k in f['data'].keys())
        n_nodes = f[f'data/{scenes[0]}/nodal_data'].shape[2]
        n_edges = f[f'data/{scenes[0]}/mesh_edge'].shape[1]
    num_samples = 5
    files = sorted(infer.glob('rollout_sample*_vaesample*_steps1.h5'))
    assert len(files) == len(scenes) * num_samples
    with h5py.File(infer / f'rollout_sample{scenes[0]}_vaesample4_steps1.h5', 'r') as f:
        assert int(f.attrs['output_var']) == 3
        assert int(f.attrs['num_features']) == 3 + 3 + 1
        nodal = f[f'data/{scenes[0]}/nodal_data']
        assert nodal.shape == (7, 2, n_nodes)
        assert f[f'data/{scenes[0]}/mesh_edge'].shape == (2, n_edges)
        assert np.all(nodal[3:6, 0, :] == 0) and np.any(nodal[3:6, 1, :] != 0)
        assert f[f'data/{scenes[0]}/metadata'].attrs['vae_sample_idx'] == 4
        assert len(f['metadata/feature_names'][()]) == 7

    values = np.load(infer / 'spread_values.npz', allow_pickle=False)
    assert values['gt'].size == len(scenes)
    # sample_n returns exactly num_samples per ensemble (upstream returned
    # batch_size * ceil(n / batch_size)); every scene gets num_samples values.
    assert values['gen'].size == len(scenes) * num_samples
    counts = {s: int((values['gen_scene'] == str(s)).sum()) for s in scenes}
    assert set(counts.values()) == {num_samples}
    metrics = json.loads((infer / 'spread_metrics.json').read_text(encoding='utf-8'))
    for key in ('crps_norm', 'spread_skill', 'pit_ks', 'w1_norm', 'sd_ratio', 'n_scenes'):
        assert key in metrics
    assert metrics['n_scenes'] == len(scenes) and metrics['stat'] == 'mag_max'
    assert (infer / 'histogram_compare.png').exists()


def test_stage_modes_and_seeded_sampling(case, run_main):
    assert run_main(case.write_config('train_vae', name='vae.txt', test_interval=0)) == 0
    assert (case.out / 'vae.pth').exists() and not (case.out / 'model.pth').exists()
    assert run_main(case.write_config('train_sar', name='sar.txt', test_interval=0)) == 0
    assert (case.out / 'model.pth').exists()

    gens = []
    for name in ('a', 'b'):
        cfg = case.write_config('inference', name=f'infer_{name}.txt', sample_scope='scene',
                                inference_output_dir=str(case.out / f'infer_{name}'))
        assert run_main(cfg) == 0
        gens.append(np.load(case.out / f'infer_{name}' / 'spread_values.npz')['gen'])
    np.testing.assert_array_equal(gens[0], gens[1])      # sample_seed makes a run reproducible
    assert np.std(gens[0]) > 0


def test_resume_continues_a_cut_off_stage(case, run_main, monkeypatch):
    cfg = case.write_config('train_vae', name='resume.txt', test_interval=0, val_interval=0,
                            vae_training_epochs=4, resume_training='True',
                            resume_interval_minutes=0)
    # Simulate a run cut off after its last periodic state: keep the state
    # that finish() would delete, and drop the final checkpoint.
    monkeypatch.setattr(TrainingResume, 'finish', lambda self: None)
    assert run_main(cfg) == 0
    state_path = case.out / 'vae.pth.resume'
    assert state_path.exists()
    assert torch.load(state_path, weights_only=False)['epoch'] == 2
    os.remove(case.out / 'vae.pth')
    monkeypatch.undo()

    assert run_main(cfg) == 0
    assert not state_path.exists()
    log = (case.out / 'train_vae.log').read_text(encoding='utf-8')
    assert '==== Resume' in log
    assert _epochs(case.out / 'train_vae.log') == [0, 1, 2, 3, 3]
