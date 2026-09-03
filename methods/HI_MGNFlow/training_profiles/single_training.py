import time

import torch
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
    evaluate_flow_sampling_epoch,
    log_training_config,
    run_periodic_test,
    train_epoch,
    validate_epoch,
)


def single_worker(config, config_filename='config.txt'):
    """Single GPU/CPU training entry point."""
    gpu_ids = config.get('gpu_ids')
    print("Starting single-process training...")


    if torch.cuda.is_available():
        gpu_id = gpu_ids
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
        print(f'Using physical GPU {gpu_id}, device: {device}')
        print(f'Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')
    else:
        device = torch.device('cpu')
        print(f'Using device: {device}')

    # ---- Dataset ----
    print("\nLoading dataset...")
    split_seed = int(config.get('split_seed', 42))
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    if torch.cuda.is_available():
        print(f'After dataset load: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    print("Writing train-derived normalization stats to HDF5...")
    train_dataset.write_preprocessing_to_hdf5(split_seed)

    if config.get('use_node_types', False) and train_dataset.num_node_types is not None:
        print(f"  Node types enabled: {train_dataset.num_node_types} types will be added to input")

    # ---- DataLoaders ----
    print("\nCreating dataloaders...")
    num_workers = int(config.get('num_workers', 0))
    pin_memory = torch.cuda.is_available()
    config['_pin_memory'] = pin_memory
    mp_context = 'spawn' if num_workers > 0 else None
    train_prefetch = int(config.get('prefetch_factor', 4)) if num_workers > 0 else None
    eval_prefetch = int(config.get('prefetch_factor', 2)) if num_workers > 0 else None
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_prefetch,
        multiprocessing_context=mp_context,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=eval_prefetch,
        multiprocessing_context=mp_context,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=pin_memory)

    if torch.cuda.is_available():
        print(f'After dataloader creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    # ---- Model ----
    print("\nInitializing model...")
    model, ema_model = build_model_and_ema(config, device)
    if torch.cuda.is_available():
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    log_model_summary(model, config, ema_model)

    # ---- Optimizer / Scheduler ----
    print("\nInitializing optimizer...")
    total_epochs = config.get('training_epochs')
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, model.parameters(), total_epochs
    )
    use_fused = torch.cuda.is_available()
    print(f"Optimizer: Adam (fused={use_fused})")
    print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
          f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    log_training_config(config)
    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")
    start_time = time.time()

    # ---- Logging ----
    log_file = init_log_file(config, config_filename)

    modelname = config.get('modelpath')
    # Members drawn per graph for the sampling-based validation score. The
    # CRPS estimator is unbiased at any S; S buys variance only, and the noise
    # floor is set by the number of validation GRAPHS rather than by S.
    val_num_samples = int(config.get('val_num_samples', 8))
    if val_num_samples < 1:
        raise ValueError("val_num_samples must be >= 1")

    best_valid_loss = float('inf')
    last_valid_loss = float('inf')
    last_saved_epoch = -1
    val_interval = int(config.get('val_interval', 1))
    mem_recording = start_memory_history()

    try:
        for epoch in range(total_epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
            )

            train_loss = train_metrics['mean']
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            # A peak that keeps climbing across epochs means reshuffling is still
            # drawing heavier batches; a reserved far above peak is allocator
            # fragmentation. Neither is visible from memory_allocated().
            vram_str = (f" | VRAM peak={train_metrics.get('peak_gb', 0.0):.2f}GB "
                        f"reserved={train_metrics.get('reserved_gb', 0.0):.2f}GB")

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)

            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                # One-step velocity regression on held-out graphs: cheap and
                # low-variance, but it says nothing about sample quality.
                valid_metrics = validate_epoch(eval_model, val_loader, device, config, epoch)
                # The metric that mirrors inference: integrate the ODE and score
                # the resulting ensemble. Costs flow_steps forwards per member.
                sample_metrics = evaluate_flow_sampling_epoch(
                    eval_model, val_loader, device, config, epoch, progress_name='Sample'
                )
                valid_loss = valid_metrics['mean']
            else:
                valid_loss = last_valid_loss  # reuse last known for checkpoint metadata
                valid_metrics = {}
                sample_metrics = None

            # best_by crps: rank checkpoints by the inference-mirroring CRPS
            # (z from the learned prior) instead of posterior reconstruction.
            # Posterior recon can keep improving while the generative path
            # degrades; for a model whose product is the generated distribution,
            # CRPS is the metric that matches the objective.
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

            sample_str = ''
            if sample_metrics is not None:
                sample_str = (f" | CRPS {sample_metrics['crps']:.2e}"
                              f" spread {sample_metrics['spread']:.3f}")
            if do_val:
                print(
                    f"Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                    f"Train fm={train_loss:.2e} | Valid fm={valid_loss:.2e}"
                    f"{sample_str}{vram_str}"
                )
            else:
                print(
                    f"Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                    f"Train fm={train_loss:.2e}{vram_str}"
                )

            last_epoch = (epoch == total_epochs - 1)
            is_best = do_val and select_loss < best_valid_loss
            # `modelname` holds ONE checkpoint, so saving unconditionally on the
            # final epoch overwrote the best one — which left `best_by` mattering
            # only for runs that were killed early. Keep the best; fall back to
            # the last epoch only when validation never saved anything at all.
            if is_best or (last_epoch and last_saved_epoch < 0):
                if is_best:
                    best_valid_loss = select_loss
                if do_val:
                    last_valid_loss = valid_loss
                save_checkpoint(
                    epoch, model, ema_model, optimizer, scheduler,
                    train_loss, valid_loss, config, train_dataset, modelname,
                )
                last_saved_epoch = epoch
                reason = []
                if is_best:
                    crit = str(config.get("best_by", "recon")).lower().strip()
                    reason.append(f"new best {crit}={select_loss:.2e}")
                if last_epoch and not is_best:
                    reason.append("last epoch; no validated checkpoint existed")
                print(f"  -> Model saved at epoch {epoch}: {', '.join(reason)}")

            if log_file:
                with open(log_file, 'a') as f:
                    elapsed = time.time() - start_time
                    val_str = (f"Valid fm={valid_loss:.4e}" if do_val
                               else "Valid skipped")
                    f.write(
                        f"Elapsed: {elapsed:.2f}s Epoch {epoch} LR: {current_lr:.4e} | "
                        f"Train fm={train_loss:.4e} | {val_str}"
                        f"{sample_str}{vram_str}\n"
                    )

            test_interval = int(config.get('test_interval', 10))
            last_epoch = epoch == total_epochs - 1
            if epoch % test_interval == 0 or last_epoch:
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset)

            dump_memory_snapshot(epoch, mem_recording, config)

        # The kept checkpoint is the BEST one, not the final epoch, so
        # name the criterion: an epoch well below total_epochs here is
        # the expected outcome, not a sign the run stopped early.
        criterion = str(config.get("best_by", "recon")).lower().strip()
        print(f"\nTraining finished. Kept checkpoint: epoch {last_saved_epoch} "
              f"(best by {criterion}), validation loss {last_valid_loss:.2e}")
    except KeyboardInterrupt:
        criterion = str(config.get("best_by", "recon")).lower().strip()
        print(f"\nTraining interrupted by user. Kept checkpoint: epoch "
              f"{last_saved_epoch} (best by {criterion}), validation loss "
              f"{last_valid_loss:.2e}")

    cleanup_dataloaders(train_loader, val_loader, test_loader)
    release_hierarchy_cache(config, train_dataset, val_dataset, test_dataset)
