import os
import signal
import threading
import time
import datetime
import traceback

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from training_profiles.setup import (
    build_dataset_splits,
    build_model_and_ema,
    build_optimizer_scheduler,
    cleanup_dataloaders,
    dump_memory_snapshot,
    init_log_file,
    log_model_summary,
    release_hierarchy_cache,
    save_checkpoint,
    start_memory_history,
)
from training_profiles.ar_rollout import ar_rt_enabled
from training_profiles.training_loop import (
    evaluate_flow_sampling_epoch,
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
    # Start a daemon thread that force-kills the process after a grace period.
    # This covers the case where the main thread is stuck inside a blocking C++
    # NCCL call and cannot check _stop_event.
    def _force_exit():
        time.sleep(_FORCED_EXIT_DELAY_SECONDS)
        os._exit(1)
    t = threading.Thread(target=_force_exit, daemon=True)
    t.start()

def train_worker(rank, world_size, config, gpu_ids, config_filename='config.txt'):
    """Training worker for distributed training.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        config: Configuration dictionary
        gpu_ids: List of GPU IDs to use
        config_filename: Path to the config file (default: config.txt)
    """
    try:
        _train_worker_inner(rank, world_size, config, gpu_ids, config_filename)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _train_worker_inner(rank, world_size, config, gpu_ids, config_filename):
    """Actual training logic, called inside the error-handling wrapper."""
    # Register signal handlers so Ctrl+C (SIGINT on main → SIGTERM on workers)
    # sets the stop flag instead of killing the process mid-collective.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


    # Enable NCCL flight recorder for debugging collective mismatches
    os.environ.setdefault('TORCH_NCCL_TRACE_BUFFER_SIZE', '1000')

    # Get the physical GPU ID for this rank
    gpu_id = gpu_ids[rank]
    port = config['_ddp_port']
    setup_distributed(rank, world_size, gpu_id, port)

    # Set device
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

    if rank == 0:
        print("Writing train-derived normalization stats to HDF5...")
        train_dataset.write_preprocessing_to_hdf5(split_seed)
        if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
            print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")
    # device_ids pins the barrier to this rank's GPU. Without it NCCL guesses,
    # and every rank guessing device 0 leaves four extra CUDA contexts on GPU 0
    # (a few hundred MB each) for the rest of the run -- GPU 0 is already the
    # heaviest rank since it alone runs validation and the periodic test.
    dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

    # Create distributed samplers
    if rank == 0:
        print("\nCreating dataloaders (distributed train, rank-0 eval)...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create dataloaders
    num_workers = config['num_workers']
    pin_memory = torch.cuda.is_available()
    config['_pin_memory'] = pin_memory
    mp_context = 'spawn' if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
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
            prefetch_factor=2 if num_workers > 0 else None,
            multiprocessing_context=mp_context,
        )
    else:
        val_loader = None

    # Test loader only needed on rank 0 (no DDP forward, uses unwrapped model)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=pin_memory
    )
    if torch.cuda.is_available() and rank == 0:
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # ---- Model ----
    if rank == 0:
        print("\nInitializing model...")
    model, ema_model = build_model_and_ema(config, device)

    # Wrap with DistributedDataParallel
    if torch.cuda.is_available():
        # AR-RT runs one forward per unrolled step before a single backward, so
        # every parameter's gradient hook fires once per step. DDP's reducer
        # rejects that ("marked ready twice") unless the graph is declared
        # static, which also legalizes the per-step checkpointing.
        static_graph = ar_rt_enabled(config)
        if static_graph:
            print("  DDP: static_graph=True (required by AR-RT unrolling)")
        ddp_model = DDP(
            model,
            device_ids=[gpu_id],
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=static_graph,
        )
    else:
        ddp_model = DDP(
            model,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True
        )

    if torch.cuda.is_available() and rank == 0:
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_model_summary(ddp_model, config, ema_model)

    best_valid_loss = float('inf')
    last_valid_loss = float('inf')
    last_saved_epoch = -1

    # ---- Optimizer / Scheduler ----
    if rank == 0:
        print("\nInitializing optimizer...")
    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, ddp_model.parameters(), total_epochs
    )
    use_fused = torch.cuda.is_available()
    if rank == 0:
        print(f"Optimizer: Adam (fused={use_fused})")
        print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
              f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

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

    mem_recording = start_memory_history(rank)

    # Synchronize all processes before starting training
    dist.barrier(device_ids=[gpu_id])

    modelname = config.get('modelpath')
    # Members drawn per graph for the sampling-based validation score.
    val_num_samples = int(config.get('val_num_samples', 8))
    if val_num_samples < 1:
        raise ValueError("val_num_samples must be >= 1")

    interrupted = False
    for epoch in range(config.get('training_epochs')):
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        train_metrics = train_epoch(ddp_model, train_loader, optimizer, device, config, epoch, ema_model=ema_model)

        # Synchronize stop decision across all ranks — if ANY rank wants to stop,
        # ALL ranks must stop together to avoid NCCL collective mismatches.
        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after train_epoch).")
            break

        train_totals = torch.tensor(
            [train_metrics['sum'], float(train_metrics['count'])],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss = (train_totals[0] / train_totals[1]).item()


        if rank == 0:
            eval_model = ema_model.module if ema_model is not None else model

            valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
            # Inference-mirroring eval: integrate the ODE and score the ensemble.
            sample_metrics = evaluate_flow_sampling_epoch(
                eval_model, val_loader, device, config, epoch, progress_name='Sample'
            )
            valid_loss = valid_metrics['mean']
        else:
            valid_loss = 0.0
        valid_loss_tensor = torch.tensor([valid_loss], device=device)
        dist.broadcast(valid_loss_tensor, src=0)
        valid_loss = valid_loss_tensor.item()

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print("\nTraining interrupted by user (after validate_epoch).")
            break

        # Step scheduler on all ranks (valid_loss is identical after all_reduce)
        scheduler.step()

        # Per epoch, node-weighted optimization and evaluation losses.
        current_lr = optimizer.param_groups[0]['lr']
        # Rank 0 only: it is the rank that also runs validation and the periodic
        # test, so its footprint is the one that OOMs first. A peak that keeps
        # climbing means reshuffling is still drawing heavier batches; a reserved
        # far above peak means allocator fragmentation.
        vram_str = (f" | VRAM peak={train_metrics.get('peak_gb', 0.0):.2f}GB "
                    f"reserved={train_metrics.get('reserved_gb', 0.0):.2f}GB")
        sample_str = ''
        if rank == 0 and sample_metrics is not None:
            sample_str = (f" | CRPS {sample_metrics['crps']:.2e}"
                          f" spread {sample_metrics['spread']:.3f}")
        if rank == 0:
            print(
                f"Epoch {epoch}/{config['training_epochs']} LR: {current_lr:.2e} | "
                f"Train fm={train_loss:.2e} | Valid fm={valid_loss:.2e}"
                f"{sample_str}{vram_str}"
            )

        # Only rank 0 saves checkpoints — only when validation improves or on last epoch.
        if rank == 0:
            last_epoch = epoch == config.get('training_epochs') - 1
            # best_by crps: rank checkpoints by the inference-mirroring CRPS
            # (z from the learned prior) instead of posterior reconstruction —
            # recon can improve while the generative path degrades.
            select_loss = valid_loss
            if (str(config.get('best_by', 'recon')).lower().strip() == 'crps'
                    and sample_metrics is not None and 'crps' in sample_metrics):
                select_loss = float(sample_metrics['crps'])
            elif (str(config.get('best_by', 'recon')).lower().strip() == 'det'
                    and sample_metrics is not None and 'det' in sample_metrics):
                # Select on the 1-forward deterministic readout: the right
                # criterion when the product is a single prediction, not an
                # ensemble. CRPS and det can disagree -- a checkpoint can get
                # better at covering the distribution while its conditional
                # mean drifts.
                select_loss = float(sample_metrics['det'])
            is_best = select_loss < best_valid_loss
            # `modelname` holds ONE checkpoint, so saving unconditionally on the
            # final epoch overwrote the best one — which left `best_by` mattering
            # only for runs that were killed early. Keep the best; fall back to
            # the last epoch only when validation never saved anything at all.
            if is_best or (last_epoch and last_saved_epoch < 0):
                if is_best:
                    best_valid_loss = select_loss
                last_valid_loss = valid_loss
                last_saved_epoch = epoch
                save_checkpoint(
                    epoch, ddp_model.module, ema_model, optimizer, scheduler,
                    train_loss, valid_loss, config, train_dataset, modelname,
                )
                reason = []
                if is_best:
                    crit = str(config.get("best_by", "recon")).lower().strip()
                    reason.append(f"new best {crit}={select_loss:.2e}")
                if last_epoch and not is_best:
                    reason.append("last epoch; no validated checkpoint existed")
                print(f"  -> Model saved at epoch {epoch}: {', '.join(reason)}")

        if log_file and rank == 0: 
            with open(log_file, 'a') as f:
                f.write(
                    f"Elapsed: {time.time() - start_time:.2f}s "
                    f"Epoch {epoch} LR: {current_lr:.4e} | "
                    f"Train fm={train_loss:.4e} | Valid fm={valid_loss:.4e}"
                    f"{sample_str}{vram_str}\n"
                )

        # Periodically test the model on the test set
        # Use unwrapped model to avoid DDP deadlock (only rank 0 runs this)
        # Barrier ensures all ranks wait so rank 1+ don't race into next epoch's
        # DDP forward/backward while rank 0 is still running test_model
        test_interval = int(config.get('test_interval', 10))
        last_epoch = epoch == config.get('training_epochs') - 1
        if epoch % test_interval == 0 or last_epoch:
            if rank == 0:
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)
            dist.barrier(device_ids=[gpu_id])

        if rank == 0:
            dump_memory_snapshot(epoch, mem_recording, config)

    if rank == 0:
        if interrupted:
            criterion = str(config.get("best_by", "recon")).lower().strip()
            print(f"\nTraining interrupted. Kept checkpoint: epoch "
                  f"{last_saved_epoch} (best by {criterion}), validation "
                  f"loss {last_valid_loss:.2e}")
        else:
            # Kept checkpoint is the BEST one, not the final epoch.
            criterion = str(config.get("best_by", "recon")).lower().strip()
            print(f"\nTraining finished. Kept checkpoint: epoch "
                  f"{last_saved_epoch} (best by {criterion}), validation "
                  f"loss {last_valid_loss:.2e}")

    cleanup_dataloaders(train_loader, val_loader, test_loader)

    # Every rank drops its hierarchy-cache handle first; only then may rank 0
    # delete the file (Windows refuses a delete while any rank holds it open).
    # Skipped after an interrupt: a rank may already be gone, and collectives on
    # a half-dead group hang. The next run's _prune_siblings collects the file.
    release_hierarchy_cache(config, train_dataset, val_dataset, test_dataset, delete=False)
    if not interrupted:
        try:
            dist.barrier(device_ids=[gpu_id])
        except Exception:
            return
        if rank == 0:
            release_hierarchy_cache(config, train_dataset)

def setup_distributed(rank, world_size, gpu_id, port):
    """Initialize distributed training process group.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        gpu_id: Physical GPU ID to use for this rank
        port: TCP port for the rendezvous store
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=60)
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
