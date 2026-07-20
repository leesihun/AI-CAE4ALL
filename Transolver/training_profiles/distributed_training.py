"""DDP training launcher (IMPLEMENTATION_PLAN.md section 12 Phase 6).

Mirrors single_training.py's structure under torch.nn.parallel.DistributedDataParallel,
with a DistributedSampler on train/val and rank-local graph segmentation (ptr
is computed per-batch inside the model, so it is automatically rank-local --
no cross-rank state is ever shared except gradient/EMA all-reduce).

NOTE: this module has not been exercised at world_size > 1 in this
implementation pass (single-GPU hardware only). Correctness at 2+ ranks is a
Phase 6 gate (section 12) that still needs to be run before this path is
trusted for a real multi-GPU training job.
"""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from training_profiles.setup import (
    build_dataset_splits,
    build_model_and_ema,
    build_optimizer_scheduler,
    cleanup_dataloaders,
    init_log_file,
    log_model_summary,
    save_checkpoint,
)
from training_profiles.training_loop import (
    log_training_config,
    run_periodic_test,
    train_epoch,
    validate_epoch,
)


def _setup_process_group(rank, world_size, port):
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(port)
    # NCCL only when it is actually built (Linux). On a Windows CUDA box NCCL is
    # unavailable, so fall back to gloo instead of crashing.
    backend = 'nccl' if (torch.cuda.is_available() and dist.is_nccl_available()) else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def _all_reduce_metrics(metrics, device):
    """Combine a per-rank {'sum', 'count'} loss into the true global mean.

    train_epoch/validate_epoch each reduce over ONE rank's DistributedSampler
    shard, so their 'mean' is only that shard's mean. Summing 'sum' and 'count'
    across ranks and re-dividing yields the exact dataset-wide mean (identical
    to what single-process training would report). Returns a metrics dict with
    corrected 'mean'/'total_mean'/'sum'/'count'.
    """
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return metrics
    packed = torch.tensor([float(metrics['sum']), float(metrics['count'])],
                          dtype=torch.float64, device=device)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    total_sum = packed[0].item()
    total_count = packed[1].item()
    mean = total_sum / total_count if total_count > 0 else float('nan')
    return {'mean': mean, 'total_mean': mean, 'sum': total_sum, 'count': int(total_count)}


def train_worker(rank, world_size, config, gpu_ids, config_filename='config.txt'):
    """Entry point for torch.multiprocessing.spawn (one process per GPU)."""
    port = config.get('_ddp_port', '29500')
    _setup_process_group(rank, world_size, port)

    if torch.cuda.is_available():
        gpu_id = gpu_ids[rank]
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'[rank {rank}] Using physical GPU {gpu_id}, device: {device}')
    else:
        device = torch.device('cpu')
        print(f'[rank {rank}] Using device: {device}')

    is_main = rank == 0

    # ---- Dataset ----
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)

    if is_main and config.get('write_preprocessing', False):
        train_dataset.write_preprocessing_to_hdf5(split_seed)
    dist.barrier()

    # ---- DataLoaders (DistributedSampler: no duplicate samples across ranks) ----
    num_workers = config['num_workers']
    pin_memory = torch.cuda.is_available()
    config['_pin_memory'] = pin_memory

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], sampler=val_sampler,
        num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0,
    )
    test_loader = None
    if is_main:
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory)

    # ---- Model ----
    model, ema_model = build_model_and_ema(config, device)
    bare_model = model
    ddp_model = DDP(model, device_ids=[device.index] if device.type == 'cuda' else None)

    if is_main:
        log_model_summary(model, config, ema_model)

    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, ddp_model.parameters(), total_epochs
    )

    if is_main:
        log_training_config(config)
    log_file = init_log_file(config, config_filename) if is_main else None
    modelname = config.get('modelpath')
    val_interval = int(config.get('val_interval', 1))
    train_loss = float('nan')
    valid_loss = float('nan')
    start_time = time.time()

    try:
        for epoch in range(total_epochs):
            train_sampler.set_epoch(epoch)
            train_metrics = train_epoch(
                ddp_model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
            )
            # All ranks must enter the collective; each holds only its own shard's
            # sum/count until this reduces them to the dataset-wide mean.
            train_metrics = _all_reduce_metrics(train_metrics, device)
            train_loss = train_metrics['mean']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = ema_model.module if ema_model is not None else ddp_model
            if do_val:
                valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
                valid_metrics = _all_reduce_metrics(valid_metrics, device)
                valid_loss = valid_metrics['mean']

            if is_main:
                msg = f"Epoch {epoch}/{total_epochs} TrainOpt: {train_loss:.2e} LR: {current_lr:.2e}"
                if do_val:
                    msg += f" Valid: {valid_loss:.2e}"
                print(msg)
                if log_file:
                    with open(log_file, 'a') as f:
                        elapsed = time.time() - start_time
                        f.write(f"Elapsed: {elapsed:.2f}s Epoch {epoch} TrainOpt {train_loss:.4e} "
                                f"Valid {valid_loss:.4e} LR: {current_lr:.4e}\n")

                test_interval = int(config.get('test_interval', 10))
                last_epoch = epoch == total_epochs - 1
                if test_loader is not None and (epoch % test_interval == 0 or last_epoch):
                    run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)

            dist.barrier()

        if is_main:
            save_checkpoint(
                epoch, bare_model, ema_model, optimizer, scheduler,
                train_loss, valid_loss, config, train_dataset, modelname,
            )
            print(f"\nTraining finished. Final model saved at epoch {epoch}.")
    except KeyboardInterrupt:
        if is_main:
            print("\nTraining interrupted by user. No checkpoint saved.")

    cleanup_dataloaders(train_loader, val_loader, test_loader)
    dist.destroy_process_group()
