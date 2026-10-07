import os
import time

import torch
from torch_geometric.loader import DataLoader

from general_modules.resume_state import TrainingResume
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


def _stage_state(phase, epoch, model, ema_model, optimizer, scheduler,
                 best_valid_loss, last_valid_loss, last_saved_epoch, start_time):
    """A resume state of stage `phase` ('AE' / 'Prior'): everything its next epoch reads."""
    return {
        'phase': phase, 'epoch': epoch, 'model': model.state_dict(),
        'ema': ema_model.state_dict() if ema_model is not None else None,
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'best_valid_loss': best_valid_loss, 'last_valid_loss': last_valid_loss,
        'last_saved_epoch': last_saved_epoch, 'elapsed': time.time() - start_time,
    }


def _restore_stage(state, model, ema_model, optimizer, scheduler, loaders):
    """Load a _stage_state. Returns (best_valid_loss, last_valid_loss,
    last_saved_epoch, start_epoch)."""
    model.load_state_dict(state['model'])
    if ema_model is not None:
        ema_model.load_state_dict(state['ema'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    # A persistent-worker loader draws its worker base seed only when its
    # iterator is first built; build it now so that draw does not come
    # out of the restored RNG stream.
    for loader in loaders:
        if loader.persistent_workers:
            iter(loader)
    return (state['best_valid_loss'], state['last_valid_loss'],
            state['last_saved_epoch'], state['epoch'] + 1)


def _resume_note(resume):
    return (f" Resume state: {resume.path}"
            if resume.enabled and os.path.exists(resume.path) else "")


def _run_ae_stage(model, ema_model, optimizer, scheduler, train_loader, val_loader,
                  test_loader, device, config, train_dataset, modelname, log_file,
                  start_time, mem_recording, total_epochs, resume, state=None, tag='AE'):
    """Reconstruction + KL loop, continued from a resume `state` when one is given.
    Returns (last_saved_epoch, last_valid_loss, interrupted)."""
    val_interval = int(config.get('val_interval', 1))
    test_interval = int(config.get('test_interval', 10))
    best_valid_loss = float('inf')
    last_valid_loss = float('inf')
    last_saved_epoch = -1
    start_epoch = 0
    if state is not None:
        best_valid_loss, last_valid_loss, last_saved_epoch, start_epoch = _restore_stage(
            state, model, ema_model, optimizer, scheduler, (train_loader, val_loader))

    try:
        resume.restore_rng()
        for epoch in range(start_epoch, total_epochs):
            train_metrics = train_ae_epoch(
                model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
            )
            train_loss = train_metrics['mean']
            # `mean` is the optimized recon + ae_kl_weight*kl; the line prints
            # the reconstruction term alone, as the DDP stage does.
            train_recon = train_metrics['recon']
            # Read before step(): afterwards it is the next epoch's LR, and after the
            # last epoch a warm restart reports the peak LR for an epoch that never runs.
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            vram_str = (f" | VRAM peak={train_metrics.get('peak_gb', 0.0):.2f}GB "
                        f"reserved={train_metrics.get('reserved_gb', 0.0):.2f}GB")

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                valid_metrics = validate_ae_epoch(eval_model, val_loader, device, config, epoch)
                valid_loss = valid_metrics['mean']
            else:
                valid_metrics = {}
                valid_loss = last_valid_loss

            if do_val:
                print(
                    f"[{tag}] Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                    f"Train recon={train_recon:.2e} kl={train_metrics['kl']:.2e} | "
                    f"Valid recon={valid_loss:.2e} kl={valid_metrics.get('kl', 0.0):.2e}{vram_str}"
                )
            else:
                print(
                    f"[{tag}] Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                    f"Train recon={train_recon:.2e} kl={train_metrics['kl']:.2e}{vram_str}"
                )

            last_epoch = (epoch == total_epochs - 1)
            is_best = do_val and valid_loss < best_valid_loss
            if is_best or (last_epoch and last_saved_epoch < 0):
                if is_best:
                    best_valid_loss = valid_loss
                if do_val:
                    last_valid_loss = valid_loss
                save_checkpoint(
                    epoch, model, ema_model, optimizer, scheduler,
                    train_loss, valid_loss, config, train_dataset, modelname,
                )
                last_saved_epoch = epoch
                reason = ([f"new best recon={valid_loss:.2e}"] if is_best
                         else ["last epoch; no validated checkpoint existed"])
                print(f"  -> Model saved at epoch {epoch}: {', '.join(reason)}")

            if log_file:
                with open(log_file, 'a') as f:
                    elapsed = time.time() - start_time
                    val_str = (f"Valid recon={valid_loss:.4e}" if do_val else "Valid skipped")
                    f.write(
                        f"[{tag}] Elapsed: {elapsed:.2f}s Epoch {epoch} LR: {current_lr:.4e} | "
                        f"Train recon={train_recon:.4e} | {val_str}{vram_str}\n"
                    )

            if epoch % test_interval == 0 or last_epoch:
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset,
                                  tag=tag)

            dump_memory_snapshot(epoch, mem_recording, config)

            if resume.due(epoch, total_epochs):
                resume.save(_stage_state(tag, epoch, model, ema_model, optimizer, scheduler,
                                         best_valid_loss, last_valid_loss, last_saved_epoch,
                                         start_time))

        print(f"\n[{tag}] stage finished. Kept checkpoint: epoch {last_saved_epoch} "
              f"(best recon), validation loss {last_valid_loss:.2e}")
        return last_saved_epoch, last_valid_loss, False
    except KeyboardInterrupt:
        print(f"\n[{tag}] stage interrupted by user. Kept checkpoint: epoch "
              f"{last_saved_epoch} (best recon), validation loss {last_valid_loss:.2e}"
              + _resume_note(resume))
        return last_saved_epoch, last_valid_loss, True


def _run_prior_stage(model, ema_model, optimizer, scheduler, train_loader, val_loader,
                     test_loader, device, config, train_dataset, modelname, log_file,
                     start_time, mem_recording, total_epochs, resume, state=None, tag='Prior'):
    """Latent flow-matching loop against the frozen AE, continued from a resume
    `state` when one is given. Returns (last_saved_epoch, last_valid_loss, interrupted)."""
    val_interval = int(config.get('val_interval', 1))
    test_interval = int(config.get('test_interval', 10))
    best_valid_loss = float('inf')
    last_valid_loss = float('inf')
    last_saved_epoch = -1
    start_epoch = 0
    if state is not None:
        best_valid_loss, last_valid_loss, last_saved_epoch, start_epoch = _restore_stage(
            state, model, ema_model, optimizer, scheduler, (train_loader, val_loader))

    try:
        resume.restore_rng()
        for epoch in range(start_epoch, total_epochs):
            train_metrics = train_prior_epoch(
                model, train_loader, optimizer, device, config, epoch, ema_model=ema_model,
            )
            train_loss = train_metrics['mean']
            # Read before step(): afterwards it is the next epoch's LR, and after the
            # last epoch a warm restart reports the peak LR for an epoch that never runs.
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            vram_str = (f" | VRAM peak={train_metrics.get('peak_gb', 0.0):.2f}GB "
                        f"reserved={train_metrics.get('reserved_gb', 0.0):.2f}GB")

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                # One-step velocity regression on held-out graphs: cheap and
                # low-variance, but it says nothing about sample quality.
                valid_metrics = validate_prior_epoch(eval_model, val_loader, device, config, epoch)
                # The metric that mirrors inference: integrate the ODE and score
                # the resulting ensemble. Costs flow_steps forwards per member,
                # but over the coarse mesh only.
                sample_metrics = evaluate_prior_sampling_epoch(
                    eval_model, val_loader, device, config, epoch, progress_name='Sample'
                )
                valid_loss = valid_metrics['mean']
            else:
                valid_loss = last_valid_loss
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
                # det is the 1-forward readout `best_by det` selects on; on the
                # epoch line it is charted, not only in the [FlowDiag] line.
                sample_str = (f" | CRPS {sample_metrics['crps']:.2e}"
                              f" spread {sample_metrics['spread']:.3f}"
                              f" det mse {sample_metrics['det']:.2e}")
            if do_val:
                print(
                    f"[{tag}] Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
                    f"Train fm={train_loss:.2e} | Valid fm={valid_loss:.2e}"
                    f"{sample_str}{vram_str}"
                )
            else:
                print(
                    f"[{tag}] Epoch {epoch}/{total_epochs} LR: {current_lr:.2e} | "
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
                        f"[{tag}] Elapsed: {elapsed:.2f}s Epoch {epoch} LR: {current_lr:.4e} | "
                        f"Train fm={train_loss:.4e} | {val_str}"
                        f"{sample_str}{vram_str}\n"
                    )

            if epoch % test_interval == 0 or last_epoch:
                run_periodic_test(eval_model, test_loader, device, config, epoch, train_dataset,
                                  tag=tag)

            dump_memory_snapshot(epoch, mem_recording, config)

            if resume.due(epoch, total_epochs):
                resume.save(_stage_state(tag, epoch, model, ema_model, optimizer, scheduler,
                                         best_valid_loss, last_valid_loss, last_saved_epoch,
                                         start_time))

        criterion = str(config.get("best_by", "recon")).lower().strip()
        print(f"\n[{tag}] stage finished. Kept checkpoint: epoch {last_saved_epoch} "
              f"(best by {criterion}), validation loss {last_valid_loss:.2e}")
        return last_saved_epoch, last_valid_loss, False
    except KeyboardInterrupt:
        criterion = str(config.get("best_by", "recon")).lower().strip()
        print(f"\n[{tag}] stage interrupted by user. Kept checkpoint: epoch "
              f"{last_saved_epoch} (best by {criterion}), validation loss "
              f"{last_valid_loss:.2e}" + _resume_note(resume))
        return last_saved_epoch, last_valid_loss, True


def single_worker(config, config_filename='config.txt'):
    """Single GPU/CPU training entry point.

    mode 'train_ae':    stage 1 alone (reconstruction + KL).
    mode 'train_prior':  stage 2 alone, against a frozen ae_checkpoint (loaded
                        inside CHiMGNFlow.__init__ -- see model/CHiMGNFlow.py).
    mode 'train':       stage 1 (ae_epochs) then stage 2 (training_epochs) in
                        one process, no checkpoint round-trip: freeze_ae() is
                        called on the same in-memory model between stages.
    """
    gpu_ids = config.get('gpu_ids')
    mode = str(config.get('mode', 'train')).lower().strip()
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
    # For mode 'train_prior', CHiMGNFlow.__init__ already loads ae_checkpoint
    # and freezes everything but `prior` -- see model/CHiMGNFlow.py.
    model, ema_model = build_model_and_ema(config, device)
    if torch.cuda.is_available():
        print(f'After model initialization: {torch.cuda.memory_allocated()/1e9:.2f}GB')

    log_model_summary(model, config, ema_model)

    if mode == 'train':
        ae_epochs = int(config.get('ae_epochs', 0))
        if ae_epochs <= 0:
            raise ValueError("mode 'train' (combined) requires ae_epochs > 0")
        first_stage, first_epochs, first_params = 'train_ae', ae_epochs, _unwrap(model).ae_parameters()
    elif mode == 'train_ae':
        first_stage, first_epochs, first_params = 'train_ae', int(config['training_epochs']), _unwrap(model).ae_parameters()
    elif mode == 'train_prior':
        first_stage, first_epochs, first_params = 'train_prior', int(config['training_epochs']), _unwrap(model).prior_parameters()
    else:
        raise ValueError(f"single_worker: unsupported mode '{mode}' (expected "
                         f"'train_ae', 'train_prior', or 'train')")

    # ---- Optimizer / Scheduler (stage 1, or the only stage) ----
    print("\nInitializing optimizer...")
    optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
        config, first_params, first_epochs
    )
    use_fused = torch.cuda.is_available()
    print(f"Optimizer: Adam (fused={use_fused}) over "
          f"{'AE' if first_stage == 'train_ae' else 'prior'} parameters")
    print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
          f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

    if torch.cuda.is_available():
        print(f'After optimizer creation: {torch.cuda.memory_allocated()/1e9:.2f}GB')
        print(f'Peak memory so far: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')

    log_training_config(config)

    modelname = config.get('modelpath')
    # One state beside modelpath for both stages of a combined run; its 'phase'
    # says which. train_prior trains over ae_checkpoint, so a state written on
    # top of an older one is stale.
    upstream = [config['ae_checkpoint']] if mode == 'train_prior' and config.get('ae_checkpoint') else []
    resume = TrainingResume(config, modelname, upstream=upstream)
    state = resume.load()
    first_state, prior_state = state, None
    if mode == 'train' and state is not None and state['phase'] == 'Prior':
        # Cut off in the prior stage: the trained compressor comes back with
        # the state, so the AE stage is not run again.
        first_state, prior_state = None, state

    print("\n" + "=" * 60)
    print("Starting training loop...")
    print("=" * 60 + "\n")
    start_time = time.time() - (state['elapsed'] if state is not None else 0.0)

    # ---- Logging ----
    log_file = init_log_file(config, config_filename,
                             resumed_at=state['epoch'] + 1 if state is not None else None,
                             tag=state['phase'] if state is not None else None)

    if mode in ('train_prior', 'train'):
        # Members drawn per graph for the sampling-based validation score. The
        # CRPS estimator is unbiased at any S; S buys variance only, and the
        # noise floor is set by the number of validation GRAPHS rather than S.
        val_num_samples = int(config.get('val_num_samples', 8))
        if val_num_samples < 1:
            raise ValueError("val_num_samples must be >= 1")

    mem_recording = start_memory_history()

    stage_args = dict(
        model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, config=config, train_dataset=train_dataset, modelname=modelname,
        log_file=log_file, start_time=start_time, mem_recording=mem_recording,
        total_epochs=first_epochs, resume=resume, state=first_state,
    )
    config['mode'] = first_stage
    if prior_state is not None:
        interrupted = False
    elif first_stage == 'train_ae':
        last_saved_epoch, last_valid_loss, interrupted = _run_ae_stage(**stage_args)
    else:
        last_saved_epoch, last_valid_loss, interrupted = _run_prior_stage(**stage_args)

    if mode == 'train' and not interrupted:
        print("\n" + "=" * 60)
        print("AE stage complete -- freezing compressor, starting prior stage")
        print("=" * 60 + "\n")
        _unwrap(model).freeze_ae()

        prior_epochs = int(config['training_epochs'])
        optimizer, scheduler, warmup_epochs, cosine_T0 = build_optimizer_scheduler(
            config, _unwrap(model).prior_parameters(), prior_epochs
        )
        print(f"Optimizer: Adam (fused={use_fused}) over prior parameters")
        print(f"Scheduler: LinearLR warmup ({warmup_epochs} epochs) -> "
              f"CosineAnnealingWarmRestarts (T_0={cosine_T0}, T_mult=2, eta_min=1e-8)")

        val_num_samples = int(config.get('val_num_samples', 8))
        if val_num_samples < 1:
            raise ValueError("val_num_samples must be >= 1")

        if resume.enabled and prior_state is None:
            # The stage boundary (due() never fires on the AE's last epoch): a run
            # cut off before the prior's first state resumes here, not in the AE.
            resume.save(_stage_state('Prior', -1, model, ema_model, optimizer, scheduler,
                                     float('inf'), float('inf'), -1, start_time))

        config['mode'] = 'train_prior'
        last_saved_epoch, last_valid_loss, interrupted = _run_prior_stage(
            model=model, ema_model=ema_model, optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            device=device, config=config, train_dataset=train_dataset, modelname=modelname,
            log_file=log_file, start_time=start_time, mem_recording=mem_recording,
            total_epochs=prior_epochs, resume=resume, state=prior_state,
        )
        config['mode'] = 'train'

    if not interrupted:
        resume.finish()

    cleanup_dataloaders(train_loader, val_loader, test_loader)
    release_hierarchy_cache(config, train_dataset, val_dataset, test_dataset)
