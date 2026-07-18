"""
Shared setup helpers for training launchers (mirrors MeshGraphNets'
training_profiles/setup.py). Both single_training.py and
distributed_training.py use these builders.
"""

import os
import time

import numpy as np
import torch

from general_modules.data_loader import load_data
from model.Transolver import Transolver
from training_profiles.training_loop import build_ema_model

CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset_splits(config, split_seed: int):
    """
    Load dataset, split 80/10/10, inject metadata into config, and return the
    three split datasets. Writing normalization stats to HDF5 (opt-in) and
    barrier synchronization (for DDP) are left to the caller.
    """
    dataset = load_data(config)

    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1, seed=split_seed)

    config['num_timesteps'] = train_dataset.num_timesteps
    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        config['num_node_types'] = train_dataset.num_node_types

    # Noise target-correction ratio (node_std / delta_std) -- used in forward pass.
    if train_dataset.node_std is not None and train_dataset.delta_std is not None:
        output_var = config['output_var']
        config['noise_std_ratio'] = (
            train_dataset.node_std[:output_var] / np.maximum(train_dataset.delta_std, 1e-8)
        ).tolist()

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_and_ema(config, device):
    """
    Instantiate Transolver, wrap with EMA if configured, and optionally
    compile with torch.compile. Returns (model, ema_model); ema_model is None
    when use_ema is not set.
    """
    model = Transolver(config, str(device)).to(device)

    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    if config.get('use_compile', False):
        model = torch.compile(model, dynamic=True)

    return model, ema_model


def log_model_summary(model, config, ema_model=None):
    """Print a one-time summary of enabled model features and parameter counts."""
    print('\n' * 2)
    print("Model initialized successfully")
    if config.get('use_checkpointing', False):
        print("Gradient checkpointing: ENABLED")
    if config.get('use_amp', True):
        print("Mixed precision (AMP): ENABLED (bfloat16)")
    if config.get('use_compile', False):
        print("torch.compile: ENABLED (dynamic=True)")
    if ema_model is not None:
        print(f"EMA: ENABLED (decay={config.get('ema_decay', 0.999)})")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


# ---------------------------------------------------------------------------
# Optimizer / Scheduler
# ---------------------------------------------------------------------------

def build_optimizer_scheduler(config, params, total_epochs: int):
    """
    Build fused AdamW and a SequentialLR: linear warmup then cosine warm
    restarts, matching MeshGraphNets' composition exactly (section 9):

        LinearLR(start_factor=0.01, total_iters=warmup_epochs)
        CosineAnnealingWarmRestarts(T_0=total_epochs - warmup_epochs, T_mult=1, eta_min=1e-8)
        SequentialLR([warmup, cosine], milestones=[warmup_epochs])

    Because T_0 equals the remaining epochs, training ends at the first
    restart -- this is a single cosine decay in practice, not literal warm
    restarts, but the composition must be reproduced exactly since changing
    total_epochs alone changes the schedule shape.
    """
    learning_rate = config.get('learningr')
    weight_decay = float(config.get('weight_decay', 1e-4))
    use_fused = torch.cuda.is_available()
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, fused=use_fused)

    warmup_epochs = int(config.get('warmup_epochs', 3))
    remaining_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_T0 = remaining_epochs

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_T0, T_mult=1, eta_min=1e-8
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    return optimizer, scheduler, warmup_epochs, cosine_T0


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def build_normalization_dict(train_dataset) -> dict:
    """Collect normalization stats and optional extras into a serialisable dict."""
    norm = {
        'node_mean': train_dataset.node_mean,
        'node_std': train_dataset.node_std,
        'delta_mean': train_dataset.delta_mean,
        'delta_std': train_dataset.delta_std,
        'position_scale': train_dataset.position_scale,
        'coordinate_normalization': 'centered_isotropic',
    }
    if train_dataset.use_node_types and train_dataset.node_type_to_idx is not None:
        norm['node_type_to_idx'] = train_dataset.node_type_to_idx
        norm['num_node_types'] = train_dataset.num_node_types
    return norm


def build_model_config(config) -> dict:
    """Collect architecture hyper-parameters into a serialisable dict
    (section 10 -- every shape/behavior choice, so inference can reconstruct
    the exact architecture from the checkpoint alone)."""
    return {
        'model': 'transolver',
        'input_var': config.get('input_var'),
        'output_var': config.get('output_var'),
        'positional_features': config.get('positional_features', 0),
        'use_node_types': config.get('use_node_types', False),
        'num_node_types': config.get('num_node_types', 0),
        'latent_dim': config.get('latent_dim'),
        'num_layers': config.get('num_layers'),
        'num_heads': config.get('num_heads'),
        'slice_num': config.get('slice_num'),
        'attention_kernel': config.get('attention_kernel', 'naive'),
        'mlp_ratio': config.get('mlp_ratio', 1),
        'dropout': config.get('dropout', 0.0),
        'temperature_init': config.get('temperature_init', 0.5),
        'temperature_min': config.get('temperature_min', 0.1),
        'temperature_max': config.get('temperature_max', 5.0),
        'small_output_init': config.get('small_output_init'),
        'use_checkpointing': config.get('use_checkpointing', False),
        'num_timesteps': config.get('num_timesteps'),
    }


def build_data_config(config) -> dict:
    """Run metadata that changes memory/scheduling but never results
    (section 8/10): chunk_size, infer_mode, infer_chunk_size, split_seed."""
    return {
        'split_seed': config.get('split_seed'),
        'coordinate_normalization': config.get('coordinate_normalization', 'centered_isotropic'),
        'num_timesteps': config.get('num_timesteps'),
        'chunk_size': config.get('chunk_size', 0),
        'infer_mode': config.get('infer_mode', 'direct'),
        'infer_chunk_size': config.get('infer_chunk_size', 0),
        'feature_loss_weights': config.get('feature_loss_weights'),
        'std_noise': config.get('std_noise', 0.0),
        'noise_gamma': config.get('noise_gamma', 1),
        'noise_std_ratio': config.get('noise_std_ratio'),
    }


def save_checkpoint(
    epoch: int,
    bare_model,          # unwrapped model (no DDP wrapper)
    ema_model,
    optimizer,
    scheduler,
    train_loss: float,
    valid_loss: float,
    config,
    train_dataset,
    modelpath: str,
) -> None:
    """Build and write a checkpoint dict to modelpath."""
    save_dict = {
        'checkpoint_version': CHECKPOINT_VERSION,
        'epoch': epoch,
        'model_state_dict': bare_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'normalization': build_normalization_dict(train_dataset),
        'model_config': build_model_config(config),
        'data_config': build_data_config(config),
    }
    if ema_model is not None:
        save_dict['ema_state_dict'] = ema_model.state_dict()
    model_dir = os.path.dirname(modelpath)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(save_dict, modelpath)
    print(f"Checkpoint saved: {modelpath} (epoch {epoch})")


# ---------------------------------------------------------------------------
# Post-training helpers
# ---------------------------------------------------------------------------

def cleanup_dataloaders(*loaders) -> None:
    """Explicitly shut down DataLoader persistent workers before process exit."""
    import gc
    for loader in loaders:
        if loader is None:
            continue
        it = getattr(loader, '_iterator', None)
        if it is not None and hasattr(it, '_shutdown_workers'):
            try:
                it._shutdown_workers()
            except Exception:
                pass
        try:
            loader._iterator = None
        except Exception:
            pass
    gc.collect()


def init_log_file(config, config_filename: str):
    """Create the epoch log file (config embedded) and return its path, or None.

    Also records config['log_dir'], which the training profiler uses as the
    destination for its chrome trace. `log_file_dir` is NOT itself a path: the
    real file lands at 'outputs/' + log_file_dir (section 8).
    """
    log_file_dir = config.get('log_file_dir')
    if not log_file_dir:
        return None

    log_file = 'outputs/' + log_file_dir
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir'] = log_dir

    with open(log_file, 'w') as f:
        f.write("Transolver training epoch log file\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file absolute path: {os.path.abspath(log_file)}\n")
        with open(config_filename, 'r') as fc:
            f.write(fc.read())
    return log_file
