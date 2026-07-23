"""Sequential production training: SDF-VAE first, then latent flow matching."""

import gc
import os
import time

import torch


_STAGE_SETTING_SUFFIXES = (
    'log_file_dir',
    'training_epochs',
    'batch_size',
    'learningr',
    'weight_decay',
    'warmup_epochs',
    'num_workers',
    'use_amp',
    'use_ema',
    'ema_decay',
    'val_interval',
    'test_interval',
    'num_test_shapes',
    'mc_resolution_test',
)

_VAE_COMPATIBILITY_KEYS = (
    'dataset_dir', 'split_seed', 'num_encoder_points', 'num_query_points',
    'latent_tokens', 'latent_dim', 'decoder_type', 'decoder_hidden',
    'decoder_layers', 'decoder_heads', 'encoder_dim', 'encoder_heads',
    'encoder_blocks', 'encoder_self_attention', 'fourier_bands', 'kl_weight',
    'surface_weight', 'normal_weight', 'eikonal_weight',
    'deterministic_warmup_epochs', 'posterior_noise_warmup_epochs',
    'posterior_noise_max_scale', 'kl_warmup_epochs', 'clamp_dist',
    'training_epochs', 'batch_size', 'learningr', 'weight_decay',
    'warmup_epochs', 'use_amp', 'use_ema', 'ema_decay',
)

_FM_COMPATIBILITY_KEYS = (
    'dataset_dir', 'split_seed', 'num_encoder_points', 'encode_batch_size',
    'vae_modelpath', 'use_conditions', 'condition_names', 'condition_clip',
    'min_condition_std', 'cond_dropout', 'fm_hidden', 'fm_blocks',
    'fm_cond_hidden', 'fm_arch', 'fm_heads', 'fm_time_sampling',
    'fm_time_logit_mean', 'fm_time_logit_std', 'training_epochs', 'batch_size',
    'learningr', 'weight_decay', 'warmup_epochs', 'use_amp', 'use_ema', 'ema_decay',
)


def build_stage_config(config, stage):
    """Convert merged pipeline settings into a native stage configuration."""
    if stage not in ('vae', 'fm'):
        raise ValueError(f'Unknown pipeline stage: {stage}')

    stage_keys = {
        f'{prefix}_{suffix}'
        for prefix in ('vae', 'fm')
        for suffix in _STAGE_SETTING_SUFFIXES
    }
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
    keys = _VAE_COMPATIBILITY_KEYS if stage == 'vae' else _FM_COMPATIBILITY_KEYS
    mismatched = [
        key for key in keys
        if key in expected_config and saved_config.get(key) != expected_config.get(key)
    ]
    if mismatched:
        return False, f'incompatible config fields: {mismatched}'
    return True, f'complete at epoch {epoch}'


def _append_pipeline_log(path, message):
    if not path:
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
    """Train VAE, verify its checkpoint, then immediately train FM."""
    from training_profiles.train_fm import fm_worker
    from training_profiles.train_vae import vae_worker

    vae_config = build_stage_config(config, 'vae')
    fm_config = build_stage_config(config, 'fm')
    vae_path = vae_config.get('vae_modelpath', './outputs/sdfflow_vae.pth')
    fm_path = fm_config.get('fm_modelpath', './outputs/sdfflow_fm.pth')
    if fm_config.get('vae_modelpath') != vae_path:
        raise ValueError('Merged pipeline must use the same vae_modelpath for both stages')

    skip_completed = bool(config.get('skip_completed_stages', True))
    pipeline_log = config.get('pipeline_log_file', 'ex1/train.log')
    started = time.time()
    banner = f'==== Pipeline {time.strftime("%Y-%m-%d %H:%M:%S")} config={config_filename} ===='
    _append_pipeline_log(pipeline_log, banner)
    print('\n' + '=' * 60)
    print('Starting sequential training pipeline: VAE -> FM')
    print('=' * 60)

    vae_complete, vae_reason = checkpoint_status(vae_path, 'vae', vae_config)
    vae_trained = not (skip_completed and vae_complete)
    if vae_trained:
        print(f'\n[Pipeline 1/2] Training VAE ({vae_reason})')
        _append_pipeline_log(pipeline_log, f'VAE start: {vae_reason}')
        vae_worker(vae_config, config_filename)
        vae_complete, vae_reason = checkpoint_status(vae_path, 'vae', vae_config)
        if not vae_complete:
            raise RuntimeError(f'VAE stage did not complete; FM will not start: {vae_reason}')
        _append_pipeline_log(pipeline_log, f'VAE complete: {vae_reason}')
    else:
        print(f'\n[Pipeline 1/2] Reusing VAE: {vae_reason}')
        _append_pipeline_log(pipeline_log, f'VAE reused: {vae_reason}')

    _release_stage_memory()

    fm_complete, fm_reason = checkpoint_status(fm_path, 'fm', fm_config)
    reuse_fm = skip_completed and not vae_trained and fm_complete
    if reuse_fm:
        print(f'\n[Pipeline 2/2] Reusing FM: {fm_reason}')
        _append_pipeline_log(pipeline_log, f'FM reused: {fm_reason}')
    else:
        if vae_trained and fm_complete:
            fm_reason = 'VAE was retrained, so the existing FM is stale'
        print(f'\n[Pipeline 2/2] Training FM ({fm_reason})')
        _append_pipeline_log(pipeline_log, f'FM start: {fm_reason}')
        fm_worker(fm_config, config_filename)
        fm_complete, fm_reason = checkpoint_status(fm_path, 'fm', fm_config)
        if not fm_complete:
            raise RuntimeError(f'FM stage did not complete: {fm_reason}')
        _append_pipeline_log(pipeline_log, f'FM complete: {fm_reason}')

    elapsed = time.time() - started
    message = f'Pipeline complete in {elapsed:.2f}s: VAE={vae_path}, FM={fm_path}'
    print(f'\n{message}')
    _append_pipeline_log(pipeline_log, message)

