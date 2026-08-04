"""Shared setup helpers for the SimulGenVAE training stages.

Mirrors ``Geometry_generation/training_profiles/setup.py`` (fused AdamW, LinearLR
warmup -> cosine restarts, optional EMA, text log, dict checkpoints) and adds the
``build_vae`` / ``build_lc`` factories that turn a flat config into the
hierarchical VAE and the (MLP or CNN) latent conditioner.
"""

import os
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

_LOSS_BY_TYPE = {1: 'MSE', 2: 'MAE', 3: 'smoothL1', 4: 'Huber'}


def as_int_list(value):
    """Config lists may parse as a scalar or a list; always return list[int]."""
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def resolve_device(config):
    from general_modules import distributed as D

    # In a spawned distributed run the process group is already initialized and
    # this rank's GPU is pinned; use the current device rather than gpu_ids[0].
    if D.is_dist():
        if D.cuda_enabled():
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
        else:
            device = torch.device('cpu')
        if D.is_main_process():
            print(f'Using device: {device} (distributed: {D.parallel_mode(config)}, '
                  f'world_size={D.get_world_size()})')
        return device

    gpu_ids = config.get('gpu_ids', 0)
    if isinstance(gpu_ids, list):
        if len(gpu_ids) > 1:
            print('NOTE: parallel_mode=single ignores extra GPU IDs; using the first. '
                  'Set parallel_mode ddp to use all listed GPUs.')
        gpu_ids = gpu_ids[0]
    if int(gpu_ids) >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_ids))
        device = torch.device(f'cuda:{int(gpu_ids)}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device


def build_vae(config, num_channels, num_time, batch_size):
    """Construct the hierarchical VAE from config + data-derived dims.

    Returns ``(model, num_filter_enc)``; the number of hierarchical latent levels
    is ``len(num_filter_enc) - 1``.
    """
    from model.vae import VAE

    small = str(config.get('network_size', 'small')).lower() != 'large'
    latent_main = int(config['latent_dim_end'])     # main latent dim
    latent_hier = int(config['latent_dim'])         # per-level hierarchical dim
    num_filter_enc = as_int_list(config['num_filter_enc'])
    num_filter_dec = num_filter_enc[::-1]
    lossfun = _LOSS_BY_TYPE.get(int(config.get('loss_type', 1)), 'MSE')
    model = VAE(latent_main, latent_hier, num_filter_enc, num_filter_dec,
                num_channels, num_time, lossfun=lossfun,
                batch_size=batch_size, small=small)
    return model, num_filter_enc


def build_lc(config, input_shape, num_levels, data_shape=None):
    """Construct the latent conditioner (MLP for csv, CNN for image)."""
    latent_main = int(config['latent_dim_end'])
    latent_hier = int(config['latent_dim'])
    lc_filter = as_int_list(config['lc_filter'])
    dropout = float(config.get('lc_dropout', 0.3))
    data_type = str(config.get('lc_data_type', 'csv')).lower()

    # 'hdf5' conditions are a scalar parameter vector like 'csv', just sourced
    # from the mesh file's conditioning rows -- same MLP conditioner.
    if data_type in ('csv', 'hdf5'):
        from model.latent_conditioner import LatentConditioner
        return LatentConditioner(lc_filter, latent_main, input_shape,
                                 latent_hier, num_levels, dropout_rate=dropout)
    from model.latent_conditioner_cnn import LatentConditionerImg
    use_attention = bool(int(config.get('use_spatial_attention', 1)))
    return LatentConditionerImg(lc_filter, latent_main, input_shape, latent_hier,
                                num_levels, data_shape or (256, 256),
                                dropout_rate=dropout, use_attention=use_attention)


def build_ema_model(model, config):
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))


def build_optimizer_scheduler(config, params, total_epochs):
    """Fused AdamW + SequentialLR (linear warmup -> cosine warm restarts)."""
    learning_rate = float(config.get('learningr', 1e-3))
    weight_decay = float(config.get('weight_decay', 0.0))
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(params, lr=learning_rate,
                                  weight_decay=weight_decay, fused=use_fused)

    warmup_epochs = min(int(config.get('warmup_epochs', max(1, total_epochs // 20))),
                        max(1, total_epochs - 1))
    cosine_t0 = max(total_epochs - warmup_epochs, 1)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_t0, T_mult=1, eta_min=1e-8)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    print(f'Optimizer: AdamW (fused={use_fused}, lr={learning_rate}, weight_decay={weight_decay})')
    print(f'Scheduler: LinearLR warmup ({warmup_epochs}) -> CosineAnnealingWarmRestarts (T_0={cosine_t0})')
    return optimizer, scheduler


def log_model_summary(model, config, ema_model=None):
    print('\nModel initialized successfully')
    if config.get('use_ema', False):
        print(f"EMA: ENABLED (decay={config.get('ema_decay', 0.999)})")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total:,}')
    print(f'Trainable parameters: {trainable:,}')


def init_log_file(config, config_filename):
    log_file = config.get('log_file_dir')
    if not log_file:
        return None
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"\n==== Run {time.strftime('%Y-%m-%d %H:%M:%S')} config={config_filename} ====\n")
    return log_file


def append_log(log_file, text):
    if log_file:
        with open(log_file, 'a') as f:
            f.write(text + '\n')


def save_checkpoint(path, payload):
    ckpt_dir = os.path.dirname(path)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    return torch.load(path, map_location=device, weights_only=False)
