"""Sequential production training: SDF-VAE first, then latent flow matching."""

import gc
import os
import time

import numpy as np
import torch

from general_modules import distributed as D


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
# `vae_best_modelpath` is deliberately NOT a stage-suffixed key: it reaches the
# VAE worker unchanged (train_vae.py reads it by its full name) and the FM
# worker simply ignores it.

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
    'encoder_query_type', 'posterior_min_std_rel', 'split_by_parent',
)

_FM_COMPATIBILITY_KEYS = (
    'dataset_dir', 'split_seed', 'num_encoder_points', 'encode_batch_size',
    'vae_modelpath', 'use_conditions', 'condition_names', 'condition_clip',
    'min_condition_std', 'cond_dropout', 'cond_dropout_mode',
    'fm_hidden', 'fm_blocks',
    'fm_cond_hidden', 'fm_arch', 'fm_heads', 'fm_time_sampling',
    'fm_time_logit_mean', 'fm_time_logit_std', 'training_epochs', 'batch_size',
    'learningr', 'weight_decay', 'warmup_epochs', 'use_amp', 'use_ema', 'ema_decay',
    'split_by_parent',
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


def check_condition_names(fm_config):
    """Fail in seconds when `condition_names` is not in the dataset, not hours later.

    Reads only the HDF5 root attrs (`cond_names` and the optional
    `cond_extra_names` sidecar written by `add_fea_conditions.py`), so it costs
    one file open and needs no split, no encode pass, and no VAE.
    """
    if not bool(fm_config.get('use_conditions', False)):
        return
    requested = fm_config.get('condition_names')
    if requested is None:
        return
    if not isinstance(requested, list):
        requested = [requested]
    requested = [str(name) for name in requested]
    dataset_dir = fm_config.get('dataset_dir')
    if not dataset_dir or not os.path.exists(str(dataset_dir)):
        return
    import h5py
    try:
        with h5py.File(str(dataset_dir), 'r') as h5:
            available = [str(n) for n in np.atleast_1d(h5.attrs.get('cond_names', []))]
            available += [str(n) for n in np.atleast_1d(h5.attrs.get('cond_extra_names', []))]
    except OSError:
        return
    if not available:
        return
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f'Unknown condition_names {unknown}; {dataset_dir} carries {available}. '
            'FEA-named conditions live in the cond_extra sidecar -- run '
            'methods/SDFFlow/add_fea_conditions.py --h5 <dataset> --csv <bracket_labels.csv> '
            'first. (Checked before the VAE stage so the mistake costs seconds, not a full '
            'VAE training run.)')


def train_pipeline(config, config_filename='config.txt'):
    """Train VAE, verify its checkpoint, then immediately train FM."""
    from training_profiles.train_fm import fm_worker
    from training_profiles.train_vae import vae_worker

    vae_config = build_stage_config(config, 'vae')
    fm_config = build_stage_config(config, 'fm')
    vae_path = vae_config.get('vae_modelpath', '../../output/geometry_generation/sdfflow_vae.pth')
    fm_path = fm_config.get('fm_modelpath', '../../output/geometry_generation/sdfflow_fm.pth')
    if fm_config.get('vae_modelpath') != vae_path:
        raise ValueError('Merged pipeline must use the same vae_modelpath for both stages')

    # Check the FM stage's condition_names against the dataset BEFORE the VAE
    # stage runs. train_fm validates them too, but only after hours of VAE
    # training have already been spent -- and the usual failure is an ex5-style
    # config naming FEA conditions against a deepjeb.h5 that has never had
    # add_fea_conditions.py run on it.
    check_condition_names(fm_config)

    skip_completed = bool(config.get('skip_completed_stages', True))
    pipeline_log = config.get('pipeline_log_file', 'ex1/train.log')
    started = time.time()
    rank0 = D.is_main_process()
    banner = f'==== Pipeline {time.strftime("%Y-%m-%d %H:%M:%S")} config={config_filename} ===='
    _append_pipeline_log(pipeline_log, banner)
    if rank0:
        print('\n' + '=' * 60)
        print('Starting sequential training pipeline: VAE -> FM')
        print('=' * 60)

    vae_complete, vae_reason = checkpoint_status(vae_path, 'vae', vae_config)
    vae_trained = not (skip_completed and vae_complete)
    if vae_trained:
        if rank0:
            print(f'\n[Pipeline 1/2] Training VAE ({vae_reason})')
        _append_pipeline_log(pipeline_log, f'VAE start: {vae_reason}')
        vae_worker(vae_config, config_filename)
        # Barrier so every rank observes rank 0's final VAE checkpoint write
        # before the compatibility re-check reads it.
        D.barrier()
        vae_complete, vae_reason = checkpoint_status(vae_path, 'vae', vae_config)
        if not vae_complete:
            raise RuntimeError(f'VAE stage did not complete; FM will not start: {vae_reason}')
        _append_pipeline_log(pipeline_log, f'VAE complete: {vae_reason}')
    else:
        if rank0:
            print(f'\n[Pipeline 1/2] Reusing VAE: {vae_reason}')
        _append_pipeline_log(pipeline_log, f'VAE reused: {vae_reason}')

    _release_stage_memory()

    fm_complete, fm_reason = checkpoint_status(fm_path, 'fm', fm_config)
    reuse_fm = skip_completed and not vae_trained and fm_complete
    if reuse_fm:
        if rank0:
            print(f'\n[Pipeline 2/2] Reusing FM: {fm_reason}')
        _append_pipeline_log(pipeline_log, f'FM reused: {fm_reason}')
    else:
        if vae_trained and fm_complete:
            fm_reason = 'VAE was retrained, so the existing FM is stale'
        if rank0:
            print(f'\n[Pipeline 2/2] Training FM ({fm_reason})')
        _append_pipeline_log(pipeline_log, f'FM start: {fm_reason}')
        fm_worker(fm_config, config_filename)
        D.barrier()
        fm_complete, fm_reason = checkpoint_status(fm_path, 'fm', fm_config)
        if not fm_complete:
            raise RuntimeError(f'FM stage did not complete: {fm_reason}')
        _append_pipeline_log(pipeline_log, f'FM complete: {fm_reason}')

    elapsed = time.time() - started
    message = f'Pipeline complete in {elapsed:.2f}s: VAE={vae_path}, FM={fm_path}'
    if rank0:
        print(f'\n{message}')
    _append_pipeline_log(pipeline_log, message)

