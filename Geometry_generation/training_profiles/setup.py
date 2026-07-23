"""
Shared setup helpers for the SDFFlow training stages (MeshGraphNets conventions:
fused AdamW, LinearLR warmup -> cosine warm restarts, optional EMA, text log).
"""

import os
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def resolve_device(config):
    from general_modules import distributed as D

    # In a spawned distributed run the process group is already initialized and
    # this rank's GPU is pinned; use the current device rather than gpu_ids[0].
    if D.is_dist():
        if torch.cuda.is_available():
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
                  'Set parallel_mode ddp (or fsdp) to use all listed GPUs.')
        gpu_ids = gpu_ids[0]
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_ids))
        device = torch.device(f'cuda:{int(gpu_ids)}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    return device


def build_ema_model(model, config):
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))


def build_optimizer_scheduler(config, params, total_epochs):
    """Fused AdamW + SequentialLR (linear warmup -> cosine warm restarts)."""
    learning_rate = float(config.get('learningr', 1e-4))
    weight_decay = float(config.get('weight_decay', 1e-4))
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, fused=use_fused)

    warmup_epochs = int(config.get('warmup_epochs', 3))
    cosine_T0 = max(total_epochs - warmup_epochs, 1)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_T0, T_mult=1, eta_min=1e-8)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    print(f'Optimizer: AdamW (fused={use_fused}, lr={learning_rate}, weight_decay={weight_decay})')
    print(f'Scheduler: LinearLR warmup ({warmup_epochs}) -> CosineAnnealingWarmRestarts (T_0={cosine_T0})')
    return optimizer, scheduler


def log_model_summary(model, config, ema_model=None):
    print('\nModel initialized successfully')
    if config.get('use_amp', False):
        amp_dtype = ('bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                     else 'float16')
        print(f'Mixed precision (AMP): ENABLED ({amp_dtype})')
    if ema_model is not None:
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
