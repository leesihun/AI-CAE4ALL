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
from training_profiles.training_loop import (
    _unwrap,
    evaluate_prior_sampling_epoch,
    log_training_config,
    run_periodic_test,
    train_ae_epoch,
    train_prior_epoch,
    validate_ae_epoch,
    validate_prior_epoch,
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


def _run_ae_stage_ddp(ddp_model, model, ema_model, optimizer, scheduler, train_sampler,
                      train_loader, val_loader, test_loader, device, config, gpu_id, rank,
                      train_dataset, modelname, log_file, start_time, mem_recording,
                      total_epochs, tag='AE'):
    """Reconstruction + KL loop: all ranks train, rank 0 validates/checkpoints/logs."""
    test_interval = int(config.get('test_interval', 10))
    best_valid_loss = float('inf')
    last_valid_loss = float('inf')
    last_saved_epoch = -1
    interrupted = False

    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = train_ae_epoch(
            ddp_model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
        )

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print(f"\n[{tag}] stage interrupted by user (after train_ae_epoch).")
            break

        # One all_reduce carries loss/recon/kl together instead of three round trips.
        train_totals = torch.tensor(
            [train_metrics['sum'], float(train_metrics['count']),
             train_metrics['recon'] * train_metrics['count'],
             train_metrics['kl'] * train_metrics['count']],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss = (train_totals[0] / train_totals[1]).item()
        train_recon = (train_totals[2] / train_totals[1]).item()
        train_kl = (train_totals[3] / train_totals[1]).item()

        if rank == 0:
            eval_model = ema_model.module if ema_model is not None else model
            valid_metrics = validate_ae_epoch(eval_model, val_loader, device, config, epoch)
            valid_loss = valid_metrics['mean']
        else:
            valid_metrics = {}
            valid_loss = 0.0
        valid_loss_tensor = torch.tensor([valid_loss], device=device)
        dist.broadcast(valid_loss_tensor, src=0)
        valid_loss = valid_loss_tensor.item()

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print(f"\n[{tag}] stage interrupted by user (after validate_ae_epoch).")
            break

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        vram_str = (f" | VRAM peak={train_metrics.get('peak_gb', 0.0):.2f}GB "
                    f"reserved={train_metrics.get('reserved_gb', 0.0):.2f}GB")

        if rank == 0:
            print(
                f"[{tag}] Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                f"Train recon={train_recon:.2e} kl={train_kl:.2e} | "
                f"Valid recon={valid_loss:.2e} kl={valid_metrics.get('kl', 0.0):.2e}{vram_str}"
            )

            last_epoch = (epoch == total_epochs - 1)
            is_best = valid_loss < best_valid_loss
            if is_best or (last_epoch and last_saved_epoch < 0):
                if is_best:
                    best_valid_loss = valid_loss
                last_valid_loss = valid_loss
                last_saved_epoch = epoch
                save_checkpoint(
                    epoch, ddp_model.module, ema_model, optimizer, scheduler,
                    train_loss, valid_loss, config, train_dataset, modelname,
                )
                reason = (f"new best recon={valid_loss:.2e}" if is_best
                         else "last epoch; no validated checkpoint existed")
                print(f"  -> Model saved at epoch {epoch}: {reason}")

            if log_file:
                with open(log_file, 'a') as f:
                    elapsed = time.time() - start_time
                    f.write(
                        f"[{tag}] Elapsed: {elapsed:.2f}s Epoch {epoch} LR: {current_lr:.4e} | "
                        f"Train recon={train_recon:.4e} | Valid recon={valid_loss:.4e}{vram_str}\n"
                    )

        last_epoch = (epoch == total_epochs - 1)
        if epoch % test_interval == 0 or last_epoch:
            if rank == 0:
                eval_model = ema_model.module if ema_model is not None else model
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)
            dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

        if rank == 0:
            dump_memory_snapshot(epoch, mem_recording, config)

    if rank == 0:
        status = "interrupted" if interrupted else "finished"
        print(f"\n[{tag}] stage {status}. Kept checkpoint: epoch {last_saved_epoch} "
              f"(best recon), validation loss {last_valid_loss:.2e}")
    return last_saved_epoch, last_valid_loss, interrupted


def _run_prior_stage_ddp(ddp_model, model, ema_model, optimizer, scheduler, train_sampler,
                         train_loader, val_loader, test_loader, device, config, gpu_id, rank,
                         train_dataset, modelname, log_file, start_time, mem_recording,
                         total_epochs, tag='Prior'):
    """Latent flow-matching loop against the frozen AE: all ranks train, rank 0
    validates/samples/checkpoints/logs."""
    test_interval = int(config.get('test_interval', 10))
    best_valid_loss = float('inf')
    last_valid_loss = float('inf')
    last_saved_epoch = -1
    interrupted = False

    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = train_prior_epoch(
            ddp_model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
        )

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print(f"\n[{tag}] stage interrupted by user (after train_prior_epoch).")
            break

        train_totals = torch.tensor(
            [train_metrics['sum'], float(train_metrics['count'])],
            device=device, dtype=torch.float64,
        )
        dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        train_loss = (train_totals[0] / train_totals[1]).item()

        if rank == 0:
            eval_model = ema_model.module if ema_model is not None else model
            valid_metrics = validate_prior_epoch(eval_model, val_loader, device, config, epoch)
            # Inference-mirroring eval: integrate the coarse-latent ODE, decode,
            # score the resulting field ensemble.
            sample_metrics = evaluate_prior_sampling_epoch(
                eval_model, val_loader, device, config, epoch, progress_name='Sample'
            )
            valid_loss = valid_metrics['mean']
        else:
            sample_metrics = None
            valid_loss = 0.0
        valid_loss_tensor = torch.tensor([valid_loss], device=device)
        dist.broadcast(valid_loss_tensor, src=0)
        valid_loss = valid_loss_tensor.item()

        stop_flag = torch.tensor([1.0 if _stop_event.is_set() else 0.0], device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
            interrupted = True
            if rank == 0:
                print(f"\n[{tag}] stage interrupted by user (after validate_prior_epoch).")
            break

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        vram_str = (f" | VRAM peak={train_metrics.get('peak_gb', 0.0):.2f}GB "
                    f"reserved={train_metrics.get('reserved_gb', 0.0):.2f}GB")
        sample_str = ''
        if rank == 0 and sample_metrics is not None:
            sample_str = (f" | CRPS {sample_metrics['crps']:.2e}"
                          f" spread {sample_metrics['spread']:.3f}")

        if rank == 0:
            print(
                f"[{tag}] Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                f"Train fm={train_loss:.2e} | Valid fm={valid_loss:.2e}"
                f"{sample_str}{vram_str}"
            )

            last_epoch = (epoch == total_epochs - 1)
            # best_by crps/det: rank checkpoints by the inference-mirroring metric
            # instead of posterior reconstruction -- recon can keep improving while
            # the generative path degrades.
            select_loss = valid_loss
            best_by = str(config.get('best_by', 'recon')).lower().strip()
            if best_by == 'crps' and sample_metrics is not None and 'crps' in sample_metrics:
                select_loss = float(sample_metrics['crps'])
            elif best_by == 'det' and sample_metrics is not None and 'det' in sample_metrics:
                select_loss = float(sample_metrics['det'])

            is_best = select_loss < best_valid_loss
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
                    reason.append(f"new best {best_by}={select_loss:.2e}")
                if last_epoch and not is_best:
                    reason.append("last epoch; no validated checkpoint existed")
                print(f"  -> Model saved at epoch {epoch}: {', '.join(reason)}")

            if log_file:
                with open(log_file, 'a') as f:
                    elapsed = time.time() - start_time
                    f.write(
                        f"[{tag}] Elapsed: {elapsed:.2f}s Epoch {epoch} LR: {current_lr:.4e} | "
                        f"Train fm={train_loss:.4e} | Valid fm={valid_loss:.4e}"
                        f"{sample_str}{vram_str}\n"
                    )

        last_epoch = (epoch == total_epochs - 1)
        if epoch % test_interval == 0 or last_epoch:
            if rank == 0:
                eval_model = ema_model.module if ema_model is not None else model
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)
            dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

        if rank == 0:
            dump_memory_snapshot(epoch, mem_recording, config)

    if rank == 0:
        status = "interrupted" if interrupted else "finished"
        crit = str(config.get("best_by", "recon")).lower().strip()
        print(f"\n[{tag}] stage {status}. Kept checkpoint: epoch {last_saved_epoch} "
              f"(best by {crit}), validation loss {last_valid_loss:.2e}")
    return last_saved_epoch, last_valid_loss, interrupted


def _train_worker_inner(rank, world_size, config, gpu_ids, config_filename):
    """Actual training logic, called inside the error-handling wrapper.

    mode 'train_ae':    stage 1 alone (reconstruction + KL).
    mode 'train_prior':  stage 2 alone, against a frozen ae_checkpoint (loaded
                        inside CHiMGNFlow.__init__ -- see model/CHiMGNFlow.py).
    mode 'train':       stage 1 (ae_epochs) then stage 2 (training_epochs) in
                        one process, no checkpoint round-trip: freeze_ae() is
                        called on the same in-memory model between stages.
    """
    # Register signal handlers so Ctrl+C (SIGINT on main → SIGTERM on workers)
    # sets the stop flag instead of killing the process mid-collective.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = str(config.get('mode', 'train')).lower().strip()

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
    num_workers = int(config.get('num_workers', 0))
    # Page-locking is serialized in the CUDA driver and these batches are large,
    # variable-size graphs, so the pinned-buffer cache does not get reused. With
    # several ranks x several workers all pinning, that lock can cost more than
    # the faster host-to-device copy buys -- measured at 0.17x a single GPU on an
    # 8-rank run. Default is the historical behaviour; set pin_memory False in the
    # config to take it out of the path.
    pin_memory = bool(config.get('pin_memory', torch.cuda.is_available()))
    config['_pin_memory'] = pin_memory
    mp_context = 'spawn' if num_workers > 0 else None
    train_prefetch = int(config.get('prefetch_factor', 4)) if num_workers > 0 else None
    eval_prefetch = int(config.get('prefetch_factor', 2)) if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_prefetch,
        multiprocessing_context=mp_context,
    )

    # The validation loader is rank-0 only and NOT sharded: every other rank
    # blocks on the broadcast below while it runs, so its cost does not fall
    # with world_size. `batch_size` is PER RANK, so a DDP run that lowers it to
    # hold the global batch would also shrink this loader and multiply the
    # number of validation batches. val_batch_size keeps the two independent.
    val_batch_size = int(config.get('val_batch_size', config['batch_size']))
    if rank == 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            prefetch_factor=eval_prefetch,
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
    # For mode 'train_prior', CHiMGNFlow.__init__ already loads ae_checkpoint
    # and freezes everything but `prior` -- see model/CHiMGNFlow.py.
    model, ema_model = build_model_and_ema(config, device)

    # Wrap with DistributedDataParallel.
    # find_unused_parameters=True (unlike the sibling MGN/MGN-V trees, which use
    # False for speed): this model trains only ONE of {compressor, prior} per
    # phase (mode 'train_ae' / 'train_prior', and each half of combined mode
    # 'train'), so the OTHER half's parameters get no gradient that phase --
    # either because forward_ae/forward_prior_step never touches them, or
    # because freeze_ae() has set requires_grad_(False) on them mid-run. With
    # find_unused_parameters=False the reducer hangs waiting for a bucket that
    # never arrives; True makes DDP tolerate it. This is also why static_graph
    # (the old AR-RT requirement) is gone: there is no more per-step unrolled
    # multi-forward/single-backward pattern to legalize.
    if torch.cuda.is_available():
        ddp_model = DDP(
            model,
            device_ids=[gpu_id],
            broadcast_buffers=True,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    else:
        ddp_model = DDP(
            model,
            broadcast_buffers=True,
            find_unused_parameters=True,
            gradient_as_bucket_view=True
        )

    if torch.cuda.is_available() and rank == 0:
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    if rank == 0:
        log_model_summary(ddp_model, config, ema_model)

    if mode == 'train':
        ae_epochs = int(config.get('ae_epochs', 0))
        if ae_epochs <= 0:
            raise ValueError("mode 'train' (combined) requires ae_epochs > 0")
        first_stage, first_epochs = 'train_ae', ae_epochs
        first_params = _unwrap(model).ae_parameters()
    elif mode == 'train_ae':
        first_stage, first_epochs = 'train_ae', int(config['training_epochs'])
        first_params = _unwrap(model).ae_parameters()
    elif mode == 'train_prior':
        first_stage, first_epochs = 'train_prior', int(config['training_epochs'])
        first_params = _unwrap(model).prior_parameters()
    else:
        raise ValueError(f"_train_worker_inner: unsupported mode '{mode}' (expected "
                         f"'train_ae', 'train_prior', or 'train')")

    # ---- Optimizer / Scheduler (stage 1, or the only stage) ----
    if rank == 0:
        print("\nInitializing optimizer...")
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, first_params, first_epochs
    )
    use_fused = torch.cuda.is_available()
    if rank == 0:
        print(f"Optimizer: Adam (fused={use_fused}) over "
              f"{'AE' if first_stage == 'train_ae' else 'prior'} parameters")
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
    dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

    modelname = config.get('modelpath')
    if mode in ('train_prior', 'train'):
        # Members drawn per graph for the sampling-based validation score. The
        # CRPS estimator is unbiased at any S; S buys variance only, and the
        # noise floor is set by the number of validation GRAPHS rather than S.
        val_num_samples = int(config.get('val_num_samples', 8))
        if val_num_samples < 1:
            raise ValueError("val_num_samples must be >= 1")

    stage_args = dict(
        ddp_model=ddp_model, model=model, ema_model=ema_model, optimizer=optimizer,
        scheduler=scheduler, train_sampler=train_sampler, train_loader=train_loader,
        val_loader=val_loader, test_loader=test_loader, device=device, config=config,
        gpu_id=gpu_id, rank=rank, train_dataset=train_dataset, modelname=modelname,
        log_file=log_file, start_time=start_time, mem_recording=mem_recording,
        total_epochs=first_epochs,
    )
    config['mode'] = first_stage
    if first_stage == 'train_ae':
        last_saved_epoch, last_valid_loss, interrupted = _run_ae_stage_ddp(**stage_args)
    else:
        last_saved_epoch, last_valid_loss, interrupted = _run_prior_stage_ddp(**stage_args)

    if mode == 'train' and not interrupted:
        if rank == 0:
            print("\n" + "=" * 60)
            print("AE stage complete -- freezing compressor, starting prior stage")
            print("=" * 60 + "\n")
        _unwrap(model).freeze_ae()
        dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)

        prior_epochs = int(config['training_epochs'])
        optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
            config, _unwrap(model).prior_parameters(), prior_epochs
        )
        if rank == 0:
            print(f"Optimizer: Adam (fused={use_fused}) over prior parameters")
            print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
                  f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

        val_num_samples = int(config.get('val_num_samples', 8))
        if val_num_samples < 1:
            raise ValueError("val_num_samples must be >= 1")

        config['mode'] = 'train_prior'
        last_saved_epoch, last_valid_loss, interrupted = _run_prior_stage_ddp(
            ddp_model=ddp_model, model=model, ema_model=ema_model, optimizer=optimizer,
            scheduler=scheduler, train_sampler=train_sampler, train_loader=train_loader,
            val_loader=val_loader, test_loader=test_loader, device=device, config=config,
            gpu_id=gpu_id, rank=rank, train_dataset=train_dataset, modelname=modelname,
            log_file=log_file, start_time=start_time, mem_recording=mem_recording,
            total_epochs=prior_epochs,
        )
        config['mode'] = 'train'

    cleanup_dataloaders(train_loader, val_loader, test_loader)

    # Every rank drops its hierarchy-cache handle first; only then may rank 0
    # delete the file (Windows refuses a delete while any rank holds it open).
    # Skipped after an interrupt: a rank may already be gone, and collectives on
    # a half-dead group hang. The next run's _prune_siblings collects the file.
    release_hierarchy_cache(config, train_dataset, val_dataset, test_dataset, delete=False)
    if not interrupted:
        try:
            dist.barrier(device_ids=[gpu_id] if torch.cuda.is_available() else None)
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
