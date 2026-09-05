"""
Shared setup helpers for the SDFFlow training stages (MeshGraphNets conventions:
fused AdamW, LinearLR warmup -> cosine warm restarts, optional EMA, text log).
"""

import contextlib
import os
import time

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


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


def ema_horizon_warning(config, steps_per_epoch, total_epochs, stage=''):
    """Return a WARNING string when the EMA horizon does not fit the run, else None.

    An EMA with decay ``d`` averages over roughly ``1 / (1 - d)`` optimizer
    updates. When the whole run makes fewer than ten such horizons of updates,
    the shadow weights still carry a visible fraction ``d ** updates`` of their
    random initialization -- and the EMA model is what validation, the periodic
    test, and inference prefer. Returns None when EMA is off or the budget is
    adequate; the caller decides whether (rank 0) and where to print it.
    """
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    updates_total = int(steps_per_epoch) * int(total_epochs)
    if updates_total <= 0 or (1.0 - decay) * updates_total >= 10.0:
        return None
    retained = decay ** updates_total
    label = f'{stage} ' if stage else ''
    head = (f'WARNING: {label}EMA decay {decay:g} over only {updates_total} optimizer updates '
            f'({int(steps_per_epoch)} steps/epoch x {int(total_epochs)} epochs) leaves '
            f'{retained:.1%} of the random initialization in the EMA weights; ')
    if updates_total <= 10:
        # `1 - 10 / updates_total` is <= 0 here, so quoting it would print the
        # useless 'set ema_decay <= 0.000000'. No decay in (0, 1) gives an EMA
        # horizon that fits ten times into a run this short.
        return (head + 'the run is too short for a meaningful EMA at any decay -- '
                'train longer, or set use_ema False and validate the raw weights.')
    suggested = 1.0 - 10.0 / updates_total
    return (head + f'set ema_decay <= {suggested:.6f} (or train longer) so the EMA horizon '
            f'1/(1-decay) fits at least 10 times into the run.')


def seed_stage(config, stage='', offset=0, verbose=True):
    """Seed torch / numpy / python from the config's ``seed`` key, if present.

    Without this, nothing in the training path seeds anything: model init, the
    DataLoader shuffle order, and the reparameterization noise all differ
    between runs, so a sweep cannot tell an arm gap from run-to-run noise. The
    key is optional -- absent, the legacy unseeded behaviour is kept exactly.

    ``offset`` is the caller's distributed rank: every rank seeds from
    ``seed + rank`` so the ranks do not draw identical posterior noise or
    shuffle orders. Returns the seed actually used, or None. Module
    construction must stay bit-identical across ranks -- see
    ``identical_across_ranks`` below, which the trainers wrap it in.

    ``SDFShapeDataset``'s per-item train subsample stays a fresh draw (it is the
    surface-sampling augmentation), but its RNG is a child of torch's stream, so
    seeding torch here makes that augmentation reproducible without freezing it;
    see ``SDFShapeDataset._rng``. val/test/latent-cache reads are pinned
    separately through ``deterministic=True``.
    """
    raw = config.get('seed')
    if raw is None or str(raw).strip() == '':
        return None
    seed = int(raw) + int(offset)
    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if verbose:
        label = f'{stage} ' if stage else ''
        print(f'{label}seeded from config seed={raw} (effective {seed})')
    return seed


def seeded_generator(seed):
    """A ``torch.Generator`` at ``seed`` for DataLoader shuffling, or None."""
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


@contextlib.contextmanager
def identical_across_ranks(run_seed, rank):
    """Seed torch identically on every rank inside the block, then restore.

    ``seed_stage`` offsets each rank's stream by its rank, which is what you
    want for posterior noise and shuffling -- but NOT for module construction:
    ``distributed.wrap_model`` builds FSDP without ``sync_module_states``, so
    every rank shards whatever it happened to initialize. Wrapping the model
    construction in this context keeps the initial weights bit-identical.
    A no-op for an unseeded run (``run_seed`` None) or rank 0.
    """
    if run_seed is None or int(rank) == 0:
        yield
        return
    base = int(run_seed) - int(rank)
    state = torch.get_rng_state()
    torch.manual_seed(base)
    try:
        yield
    finally:
        torch.set_rng_state(state)
