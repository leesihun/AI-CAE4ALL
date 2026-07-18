import os

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.loader import DataLoader

from general_modules.mesh_dataset import MeshGraphDataset
from inference_profiles.rollout import run_rollout
from model.factory import build_model
from tests.conftest import base_config_2d
from training_profiles.setup import build_optimizer_scheduler, save_checkpoint


def _train_briefly_and_checkpoint(h5_path, model_name, modelpath, extra_cfg, steps=5):
    cfg = base_config_2d(h5_path, model=model_name, modelpath=modelpath,
                         gpu_ids=-1, **extra_cfg)
    ds = MeshGraphDataset(h5_path, cfg)
    train, val, test = ds.split(0.8, 0.1, 0.1, seed=42)
    model, data_spec, coordinate_domain = build_model(cfg, train)

    optimizer, scheduler, _, _ = build_optimizer_scheduler(cfg, model.parameters(), total_epochs=steps)
    loader = DataLoader(train, batch_size=len(train), shuffle=False)
    batch = next(iter(loader))
    model.train()
    loss = None
    for _ in range(steps):
        optimizer.zero_grad()
        pred, target = model(batch, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

    save_checkpoint(
        epoch=steps - 1, bare_model=model, ema_model=None, optimizer=optimizer,
        scheduler=scheduler, train_loss=float(loss.item()), valid_loss=float(loss.item()),
        config=cfg, train_dataset=train, coordinate_domain=coordinate_domain,
        data_spec=data_spec, modelpath=modelpath, config_filename='dummy_config.txt',
    )
    return cfg


def test_static_inference_writes_expected_schema(tiny_static_2d_h5, tmp_path):
    modelpath = str(tmp_path / "model.pth")
    train_cfg = _train_briefly_and_checkpoint(
        tiny_static_2d_h5, 'deeponet', modelpath, {'deeponet_sensor_resolution': [8, 8]},
    )

    output_dir = str(tmp_path / "rollout_out")
    infer_cfg = dict(train_cfg)
    infer_cfg['mode'] = 'inference'
    infer_cfg['infer_dataset'] = tiny_static_2d_h5
    infer_cfg['inference_output_dir'] = output_dir
    infer_cfg['infer_timesteps'] = 1
    infer_cfg['gpu_ids'] = -1

    run_rollout(infer_cfg, config_filename='dummy_infer_config.txt')

    files = os.listdir(output_dir)
    assert len(files) == 10  # one output file per sample in the static fixture

    sample_file = [f for f in files if f.startswith('rollout_sample')][0]
    with h5py.File(os.path.join(output_dir, sample_file), 'r') as f:
        assert 'data' in f
        sid = list(f['data'].keys())[0]
        nodal_data = f[f'data/{sid}/nodal_data'][:]
        mesh_edge = f[f'data/{sid}/mesh_edge'][:]
        assert nodal_data.shape[0] == 3 + 4 + 1  # xyz + 4 outputs + part-no row
        assert nodal_data.shape[1] == 2  # steps+1 = 0+1... static infer_timesteps=1 -> 2 states written
        assert np.isfinite(nodal_data).all()
        assert mesh_edge.shape[0] == 2
        assert 'metadata' in f
        assert f['metadata/feature_names'] is not None


def test_temporal_rollout_autoregresses_and_writes_all_steps(tiny_temporal_3d_h5, tmp_path):
    modelpath = str(tmp_path / "model_temporal.pth")
    # out_of_bounds_policy=clamp: the tiny synthetic fixture's val/test-split
    # geometry can fall slightly outside a box fit from only 6 train samples
    # (expected per section 4.4); this test exercises rollout mechanics, not
    # the reject-vs-clamp policy itself (see test_coordinate_domain.py).
    train_cfg = _train_briefly_and_checkpoint(
        tiny_temporal_3d_h5, 'deeponet', modelpath,
        {'deeponet_sensor_resolution': [8, 8, 8], 'out_of_bounds_policy': 'clamp'}, steps=3,
    )

    output_dir = str(tmp_path / "rollout_out_temporal")
    infer_cfg = dict(train_cfg)
    infer_cfg['mode'] = 'inference'
    infer_cfg['infer_dataset'] = tiny_temporal_3d_h5
    infer_cfg['inference_output_dir'] = output_dir
    infer_cfg['infer_timesteps'] = 3
    infer_cfg['gpu_ids'] = -1

    run_rollout(infer_cfg, config_filename='dummy_infer_config.txt')

    files = [f for f in os.listdir(output_dir) if f.startswith('rollout_sample')]
    assert len(files) == 8  # one per sample in the temporal fixture

    with h5py.File(os.path.join(output_dir, files[0]), 'r') as f:
        sid = list(f['data'].keys())[0]
        nodal_data = f[f'data/{sid}/nodal_data'][:]
        assert nodal_data.shape[1] == 4  # 3 rollout steps + initial state
        assert np.isfinite(nodal_data).all()


def test_rollout_rejects_wrong_model_in_config(tiny_static_2d_h5, tmp_path):
    modelpath = str(tmp_path / "model.pth")
    train_cfg = _train_briefly_and_checkpoint(
        tiny_static_2d_h5, 'deeponet', modelpath, {'deeponet_sensor_resolution': [8, 8]},
    )
    infer_cfg = dict(train_cfg)
    infer_cfg['mode'] = 'inference'
    infer_cfg['infer_dataset'] = tiny_static_2d_h5
    infer_cfg['inference_output_dir'] = str(tmp_path / "out2")
    infer_cfg['infer_timesteps'] = 1
    infer_cfg['gpu_ids'] = -1
    infer_cfg['model'] = 'fno'  # wrong: checkpoint was trained as deeponet

    with pytest.raises(ValueError, match="Loading a checkpoint under a different model"):
        run_rollout(infer_cfg, config_filename='dummy_infer_config.txt')
