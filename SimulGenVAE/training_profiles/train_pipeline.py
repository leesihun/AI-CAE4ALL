"""Sequential production training: hierarchical VAE first, then latent conditioner.

Mirrors ``Geometry_generation/training_profiles/train_pipeline.py``: per-stage
settings are carried with ``vae_``/``lc_`` prefixes and stripped to the unprefixed
names each worker reads; completed stages can be skipped via ``skip_completed_stages``.
"""

import gc
import os
import time

import torch

from general_modules import distributed as D

_STAGE_SETTING_SUFFIXES = (
    'log_file_dir', 'training_epochs', 'batch_size', 'learningr', 'weight_decay',
    'warmup_epochs', 'num_workers', 'use_amp', 'use_ema', 'ema_decay',
)

_VAE_COMPATIBILITY_KEYS = (
    'dataset_dir', 'split_seed', 'num_var', 'field_start_row', 'node_start',
    'node_end', 'timesteps_reduced', 'latent_dim', 'latent_dim_end',
    'num_filter_enc', 'network_size', 'loss_type', 'alpha', 'training_epochs',
    'batch_size', 'learningr',
)

_LC_COMPATIBILITY_KEYS = (
    'dataset_dir', 'split_seed', 'vae_modelpath', 'param_dir', 'lc_data_type',
    'lc_filter', 'lc_dropout', 'latent_dim', 'latent_dim_end', 'training_epochs',
    'batch_size', 'learningr',
)


def build_stage_config(config, stage):
    """Convert merged pipeline settings into a native stage configuration."""
    if stage not in ('vae', 'lc'):
        raise ValueError(f'Unknown pipeline stage: {stage}')
    stage_keys = {f'{prefix}_{suffix}'
                  for prefix in ('vae', 'lc') for suffix in _STAGE_SETTING_SUFFIXES}
    stage_config = {
        key: value for key, value in config.items()
        if key not in stage_keys and key not in ('pipeline_log_file', 'skip_completed_stages')
    }
    for suffix in _STAGE_SETTING_SUFFIXES:
        source_key = f'{stage}_{suffix}'
        if source_key in config:
            stage_config[suffix] = config[source_key]
    stage_config['mode'] = f'train_{stage}'
    return stage_config


def checkpoint_status(path, stage, expected_config):
    """Return whether a checkpoint is complete and compatible with this run."""
    if not os.path.exists(path):
        return False, 'checkpoint does not exist'
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as exc:
        return False, f'checkpoint could not be read: {exc}'
    if checkpoint.get('stage') != stage:
        return False, f'checkpoint stage is {checkpoint.get("stage")!r}, expected {stage!r}'
    expected_epoch = int(expected_config['training_epochs']) - 1
    epoch = int(checkpoint.get('epoch', -1))
    if epoch < expected_epoch:
        return False, f'checkpoint epoch {epoch} is below required epoch {expected_epoch}'
    saved_config = checkpoint.get('config', {})
    keys = _VAE_COMPATIBILITY_KEYS if stage == 'vae' else _LC_COMPATIBILITY_KEYS
    mismatched = [k for k in keys
                  if k in expected_config and saved_config.get(k) != expected_config.get(k)]
    if mismatched:
        return False, f'incompatible config fields: {mismatched}'
    return True, f'complete at epoch {epoch}'


def _append_pipeline_log(path, message):
    if not path or not D.is_main_process():
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(message + '\n')


def _release_stage_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_pipeline(config, config_filename='config.txt'):
    """Train the VAE, verify its checkpoint, then train the latent conditioner."""
    from training_profiles.train_lc import lc_worker
    from training_profiles.train_vae import vae_worker

    vae_config = build_stage_config(config, 'vae')
    lc_config = build_stage_config(config, 'lc')
    vae_path = vae_config.get('vae_modelpath', './outputs/simulgenvae_vae.pth')
    lc_path = lc_config.get('lc_modelpath', './outputs/simulgenvae_lc.pth')
    if lc_config.get('vae_modelpath') != vae_path:
        raise ValueError('Merged pipeline must use the same vae_modelpath for both stages')

    skip_completed = bool(config.get('skip_completed_stages', True))
    pipeline_log = config.get('pipeline_log_file')
    started = time.time()
    rank0 = D.is_main_process()
    _append_pipeline_log(pipeline_log,
                         f'==== Pipeline {time.strftime("%Y-%m-%d %H:%M:%S")} config={config_filename} ====')
    if rank0:
        print('\n' + '=' * 60)
        print('Starting sequential training pipeline: VAE -> LC')
        print('=' * 60)

    vae_complete, vae_reason = checkpoint_status(vae_path, 'vae', vae_config)
    vae_trained = not (skip_completed and vae_complete)
    if vae_trained:
        if rank0:
            print(f'\n[Pipeline 1/2] Training VAE ({vae_reason})')
        vae_worker(vae_config, config_filename)
        D.barrier()
        vae_complete, vae_reason = checkpoint_status(vae_path, 'vae', vae_config)
        if not vae_complete:
            raise RuntimeError(f'VAE stage did not complete; LC will not start: {vae_reason}')
    elif rank0:
        print(f'\n[Pipeline 1/2] Reusing VAE: {vae_reason}')

    _release_stage_memory()

    lc_complete, lc_reason = checkpoint_status(lc_path, 'lc', lc_config)
    reuse_lc = skip_completed and not vae_trained and lc_complete
    if reuse_lc:
        if rank0:
            print(f'\n[Pipeline 2/2] Reusing LC: {lc_reason}')
    else:
        if vae_trained and lc_complete:
            lc_reason = 'VAE was retrained, so the existing LC is stale'
        if rank0:
            print(f'\n[Pipeline 2/2] Training LC ({lc_reason})')
        lc_worker(lc_config, config_filename)
        D.barrier()

    elapsed = time.time() - started
    message = f'Pipeline complete in {elapsed:.2f}s: VAE={vae_path}, LC={lc_path}'
    if rank0:
        print(f'\n{message}')
    _append_pipeline_log(pipeline_log, message)
