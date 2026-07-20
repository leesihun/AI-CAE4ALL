"""DDP training worker (IMPLEMENTATION_PLAN.md section 5.2 item 14), ported
from MeshGraphNets' training_profiles/distributed_training.py. Pipeline
model parallelism lives in parallelism/launcher.py (`parallel_mode
model_split`, fno/gino only); this file is standard data-parallel DDP.
"""

import os
import signal
import threading
import time
import datetime
import traceback

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

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

# Per-process shutdown flag, set by signal handler
_stop_event = threading.Event()

_FORCED_EXIT_DELAY_SECONDS = 10


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM by setting the stop flag and scheduling a forced exit."""
    _stop_event.set()

    def _force_exit():
        time.sleep(_FORCED_EXIT_DELAY_SECONDS)
        os._exit(1)
    t = threading.Thread(target=_force_exit, daemon=True)
    t.start()


def train_worker(rank, world_size, config, gpu_ids, config_filename='config.txt'):
    """Training worker for distributed training."""
    try:
        _train_worker_inner(rank, world_size, config, gpu_ids, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train_worker_inner(rank, world_size, config, gpu_ids, config_filename):
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    os.environ.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', '1000')

    gpu_id = gpu_ids[rank]
    port = config['_ddp_port']
    setup_distributed(rank, world_size, gpu_id, port)

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        if rank == 0:
            print(f'[Rank {rank}] Using physical GPU {gpu_id}, device: {device}')
            print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        if rank == 0:
            print(f'Using device: {device}')

    # ---- Dataset ----
    if rank == 0:
        print("\nLoading dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0 and config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")
    dist.barrier()

    if rank == 0:
        print("\nCreating dataloaders (distributed train, rank-0 eval)...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    num_workers = config['num_workers']
    pin_memory = torch.cuda.is_available()
    config['_pin_memory'] = pin_memory
    mp_context = 'spawn' if num_workers > 0 else None
    prefetch_factor = int(config.get('prefetch_factor', 4)) if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_context,
    )

    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=mp_context,
        )
    else:
        val_loader = None

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory)
    if rank == 0:
        train_eval_subset_size = min(len(train_dataset), int(config.get('train_eval_subset_size', 128)))
        train_eval_rng = np.random.default_rng(split_seed)
    else:
        train_eval_subset_size = None
        train_eval_rng = None
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # ---- Model ----
    if rank == 0:
        print("\nInitializing model...")
    model, ema_model, data_spec, coordinate_domain = build_model_and_ema(config, train_dataset, device)

    if torch.cuda.is_available():
        ddp_model = DDP(
            model, device_ids=[gpu_id], broadcast_buffers=True,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )
    else:
        ddp_model = DDP(
            model, broadcast_buffers=True,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )

    if torch.cuda.is_available() and rank == 0:
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_model_summary(ddp_model, config, ema_model)

    # ---- Optimizer / Scheduler ----
    if rank == 0:
        print("\nInitializing optimizer...")
    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, ddp_model.parameters(), total_epochs
    )
    if rank == 0:
        paper_darcy = (str(config.get('model', '')).lower() == 'fno'
                       and str(config.get('fno_variant', 'mesh')).lower() == 'paper_darcy')
        if paper_darcy:
            print(f"Optimizer: Adam (coupled L2 weight_decay={float(config.get('weight_decay', 1e-4))})")
            print("Scheduler: StepLR (step_size=100, gamma=0.5); no warmup")
        else:
            use_fused = torch.cuda.is_available()
            print(f"Optimizer: AdamW (fused={use_fused}, weight_decay={float(config.get('weight_decay', 1e-4))})")
            print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
                  f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=1, eta_min=1e-8)")

    if torch.cuda.is_available() and rank == 0:
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_training_config(config)
        print("\n" + "=" * 60)
        print("Starting training loop...")
        print("=" * 60 + "\n")

    start_time = time.time()

    log_file = None
    if rank == 0:
        log_file = init_log_file(config, config_filename)

    dist.barrier(device_ids=[gpu_id]) if torch.cuda.is_available() else dist.barrier()

    modelname = config.get('modelpath')

    interrupted = False
    for epoch in range(config.get('training_epochs')):
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(ddp_model, train_loader, optimizer, device, config, epoch, ema_model=ema_model)

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after train_epoch).")
            break

        train_totals = torch.tensor(
            [train_metrics['sum'], float(train_metrics['count'])],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss = (train_totals[0] / train_totals[1]).item()

        if rank == 0:
            eval_model = ema_model.module if ema_model is not None else model

            train_eval_indices = train_eval_rng.choice(
                len(train_dataset), size=train_eval_subset_size, replace=False
            ).tolist()
            train_eval_loader = DataLoader(
                Subset(train_dataset, train_eval_indices),
                batch_size=config['batch_size'], shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                prefetch_factor=prefetch_factor, multiprocessing_context=mp_context,
            )
            train_eval_metrics = validate_epoch(model, train_eval_loader, device, config, epoch)
            valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
            train_eval_loss = train_eval_metrics['mean']
            valid_loss = valid_metrics['mean']
        else:
            train_eval_loss = 0.0
            valid_loss = 0.0
        train_eval_loss_tensor = torch.tensor([train_eval_loss], device=device)
        valid_loss_tensor = torch.tensor([valid_loss], device=device)
        dist.broadcast(train_eval_loss_tensor, src=0)
        dist.broadcast(valid_loss_tensor, src=0)
        train_eval_loss = train_eval_loss_tensor.item()
        valid_loss = valid_loss_tensor.item()

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after validate_epoch).")
            break

        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0:
            print(
                f"Epoch {epoch}/{config['training_epochs']} "
                f"TrainOpt: {train_loss:.2e} "
                f"TrainEval: {train_eval_loss:.2e} "
                f"Valid: {valid_loss:.2e} "
                f"LR: {current_lr:.2e}"
            )

        if log_file and rank == 0:
            with open(log_file, 'a') as f:
                f.write(
                    f"Elapsed: {time.time() - start_time:.2f}s "
                    f"Epoch {epoch} TrainOpt {train_loss:.4e} "
                    f"TrainEval {train_eval_loss:.4e} "
                    f"Valid {valid_loss:.4e} LR: {current_lr:.4e}\n"
                )

        test_interval = int(config.get('test_interval', 10))
        last_epoch = epoch == config.get('training_epochs') - 1
        if epoch % test_interval == 0 or last_epoch:
            if rank == 0:
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)
            dist.barrier(device_ids=[gpu_id]) if torch.cuda.is_available() else dist.barrier()

        checkpoint_interval = int(config.get('checkpoint_interval', 0))
        if rank == 0 and checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
            save_checkpoint(
                epoch, ddp_model.module, ema_model, optimizer, scheduler,
                train_loss, valid_loss, config, train_dataset, coordinate_domain,
                data_spec, modelname, config_filename,
            )
            print(f"  Periodic checkpoint saved at epoch {epoch}")

    if rank == 0:
        if interrupted:
            print("\nTraining interrupted. No checkpoint saved.")
        else:
            save_checkpoint(
                epoch, ddp_model.module, ema_model, optimizer, scheduler,
                train_loss, valid_loss, config, train_dataset, coordinate_domain,
                data_spec, modelname, config_filename,
            )
            print(f"\nTraining finished. Final model saved at epoch {epoch} with validation loss {valid_loss:.2e}")

    cleanup_dataloaders(train_loader, val_loader, test_loader)


def setup_distributed(rank, world_size, gpu_id, port):
    """Initialize distributed training process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=60),
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
