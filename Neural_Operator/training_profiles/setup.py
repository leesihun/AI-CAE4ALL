"""Shared setup helpers for training launchers (IMPLEMENTATION_PLAN.md
sections 12-13), adapted from MeshGraphNets' training_profiles/setup.py.
Both single_training.py and distributed_training.py use these builders so
dataset/model/optimizer/checkpoint logic is written exactly once.
"""

import hashlib
import os
import time

import h5py
import numpy as np
import torch
import scipy
import torch_geometric

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model
from training_profiles.training_loop import build_ema_model

SCHEMA_VERSION = "deeponet_repo_v1"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset_splits(config, split_seed: int):
    """
    Load dataset, split 80/10/10, inject metadata into config, and return the
    three split datasets. Normalization stats are never written back into the
    source HDF5 (`write_preprocessing True` is rejected at config validation).
    """
    dataset = MeshGraphDataset(config.get('dataset_dir'), config)

    train_dataset, val_dataset, test_dataset = dataset.split(0.8, 0.1, 0.1, seed=split_seed)

    config['num_timesteps'] = train_dataset.num_timesteps
    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        config['num_node_types'] = train_dataset.num_node_types

    # Noise target-correction ratio (node_std / delta_std) -- used in the
    # model wrapper's forward pass (section 4.6). Computed once here so no
    # model rereads HDF5 statistics.
    if train_dataset.node_std is not None and train_dataset.delta_std is not None:
        output_var = config['output_var']
        config['noise_std_ratio'] = (
            train_dataset.node_std[:output_var] / np.maximum(train_dataset.delta_std, 1e-8)
        ).tolist()

    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_and_ema(config, train_dataset, device):
    """
    Build the selected OperatorWrapper via the factory, run GINO's mandatory
    coverage preflight when applicable, wrap with EMA if configured, and
    optionally compile with torch.compile.
    """
    model, data_spec, coordinate_domain = build_model(config, train_dataset)
    model = model.to(device)

    if model.model_name == 'gino':
        from torch_geometric.loader import DataLoader
        probe_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        probe_batch = next(iter(probe_loader)).to(device)
        print("[gino] Running mandatory coverage preflight (section 8.4)...")
        report = model.core.coverage_preflight(probe_batch)
        for r in report['reports']:
            print(f"  graph {r['graph']}: input_gno={r['input_gno']}  output_gno={r['output_gno']}")
        print("[gino] Coverage preflight passed.")

    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    if config.get('use_compile', False):
        model = torch.compile(model, dynamic=True)

    return model, ema_model, data_spec, coordinate_domain


def log_model_summary(model, config, ema_model=None):
    """Print a one-time summary of enabled model features and parameter counts."""
    print('\n' * 2)
    print(f"Model '{model.model_name}' initialized successfully")
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
    restarts. Exact port of MeshGraphNets' scheduler policy (section 12.3).

    Optimizer hyper-parameters:
        weight_decay   (config key, default 1e-4; use decimal form in config
                        files -- the parser reads `1e-4` as a string)

    Scheduler hyper-parameters:
        warmup_epochs  (config key, default 3)
        cosine_T0 = total_epochs - warmup_epochs
        cosine_T_mult = 1
        eta_min = 1e-8
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
# Checkpoint (section 13)
# ---------------------------------------------------------------------------

def build_normalization_dict(train_dataset) -> dict:
    """Collect normalization stats into a serialisable dict."""
    norm = {
        'node_mean': train_dataset.node_mean,
        'node_std': train_dataset.node_std,
        'delta_mean': train_dataset.delta_mean,
        'delta_std': train_dataset.delta_std,
        'position_scale': float(train_dataset.position_scale),
        'grid_bound_min': train_dataset.grid_bound_min,
        'grid_bound_max': train_dataset.grid_bound_max,
        'active_axes': list(train_dataset.active_axes),
        'operator_dim': train_dataset.operator_dim,
        'rot_invariant_radius': float(train_dataset.rot_invariant_radius),
    }
    if train_dataset.use_node_types and train_dataset.node_type_to_idx is not None:
        norm['node_type_to_idx'] = train_dataset.node_type_to_idx
        norm['num_node_types'] = train_dataset.num_node_types
    if train_dataset.has_sdf:
        norm['sdf_mean'] = float(train_dataset.sdf_mean)
        norm['sdf_std'] = float(train_dataset.sdf_std)
    return norm


def _lightweight_file_fingerprint(path: str) -> dict:
    """Fast fingerprint (size + mtime + hash of the first 1 MB), not a full
    cryptographic hash: ex2.h5 is over 8 GB and hashing it in full on every
    checkpoint save would dominate wall-clock time."""
    if not path or not os.path.exists(path):
        return {'path': path, 'size': None, 'mtime': None, 'head_sha1': None}
    stat = os.stat(path)
    with open(path, 'rb') as f:
        head = f.read(1024 * 1024)
    return {
        'path': os.path.abspath(path),
        'size': stat.st_size,
        'mtime': stat.st_mtime,
        'head_sha1': hashlib.sha1(head).hexdigest(),
    }


def build_dependency_versions() -> dict:
    return {
        'torch': torch.__version__,
        'torch_geometric': torch_geometric.__version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'h5py': h5py.__version__,
    }


def save_checkpoint(
    epoch: int,
    bare_model,          # unwrapped OperatorWrapper (no DDP wrapper)
    ema_model,
    optimizer,
    scheduler,
    train_loss: float,
    valid_loss: float,
    config,
    train_dataset,
    coordinate_domain,
    data_spec,
    modelpath: str,
    config_filename: str,
) -> None:
    """Build and write a checkpoint dict to `modelpath` (section 13's full contract)."""
    # torch.compile wraps the model in an OptimizedModule whose state_dict keys
    # carry an '_orig_mod.' prefix; unwrap so `build_model_from_checkpoint` can
    # load the weights into a fresh (uncompiled) OperatorWrapper with strict=True.
    bare_model = getattr(bare_model, '_orig_mod', bare_model)
    save_dict = {
        'schema_version': SCHEMA_VERSION,
        'selected_model': bare_model.model_name,
        'epoch': epoch,
        'model_state_dict': bare_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'model_config': bare_model.export_model_config(),
        'adapter_config': coordinate_domain.to_dict(),
        'data_config': data_spec.to_dict(),
        'normalization': build_normalization_dict(train_dataset),
        'dependency_versions': build_dependency_versions(),
        'rng_states': {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
        },
        'source_reference': {
            'config_file': os.path.abspath(config_filename) if config_filename else None,
            'dataset': _lightweight_file_fingerprint(config.get('dataset_dir')),
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    }
    if ema_model is not None:
        save_dict['ema_state_dict'] = ema_model.state_dict()
    model_dir = os.path.dirname(modelpath)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(save_dict, modelpath)


# ---------------------------------------------------------------------------
# Post-training helpers
# ---------------------------------------------------------------------------

def cleanup_dataloaders(*loaders) -> None:
    """Explicitly shut down DataLoader persistent workers before process exit.

    Without this, `persistent_workers=True` keeps worker processes alive until
    the Python interpreter tears down, at which point
    `multiprocessing.resource_tracker` reports leaked semaphore warnings.
    """
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

    Also records `config['log_dir']`, used as the destination for the
    training profiler's chrome trace.
    """
    log_file_dir = config.get('log_file_dir')
    if not log_file_dir:
        return None

    log_file = 'outputs/' + log_file_dir
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    config['log_dir'] = log_dir

    with open(log_file, 'w') as f:
        f.write("Training epoch log file\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file absolute path: {os.path.abspath(log_file)}\n")
        with open(config_filename, 'r') as fc:
            f.write(fc.read())
    return log_file
