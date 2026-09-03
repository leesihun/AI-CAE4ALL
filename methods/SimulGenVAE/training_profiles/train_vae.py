"""Stage 1: hierarchical VAE training (reconstruction + warmup-KL).

Runs single-process or, under ``parallel_mode`` ddp, as one rank of a spawned
distributed job (gradients shared across ranks; rank 0 owns validation, logging,
and checkpoint writes). The loss follows the SimulGenVAE convention
``alpha * recon + beta * sum(KL)`` with a linear KL beta warmup.
"""

import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from general_modules import distributed as D
from general_modules.fom_dataset import build_dataset_splits
from model.common import add_sn, initialize_weights_He
from training_profiles.setup import (
    append_log,
    build_ema_model,
    build_optimizer_scheduler,
    build_vae,
    init_log_file,
    log_model_summary,
    resolve_device,
    save_checkpoint,
)


def _kl_beta(epoch, total_epochs, init_beta, beta_target, start_frac, warmup_epochs):
    """Linear beta warmup: init_beta until `start`, ramp to beta_target over `warmup`."""
    start = int(total_epochs * start_frac)
    end = start + warmup_epochs if warmup_epochs > 0 else int(total_epochs * 0.8)
    end = max(end, start + 1)
    if epoch < start:
        return init_beta
    if epoch >= end:
        return beta_target
    return (epoch - start) * (beta_target - init_beta) / (end - start) + init_beta


def vae_worker(config, config_filename='config.txt'):
    device = resolve_device(config)
    split_seed = int(config.get('split_seed', 42))
    rank0 = D.is_main_process()
    world_size = D.get_world_size()

    if rank0:
        print('\nLoading dataset...')
    train_dataset, val_dataset, _test_dataset = build_dataset_splits(config, split_seed)
    normalization = train_dataset.normalization
    num_channels = train_dataset.num_channels
    num_time = train_dataset.num_time

    num_workers = int(config.get('num_workers', 0))
    pin_memory = torch.cuda.is_available()
    batch_size = int(config.get('batch_size', 16))

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=D.get_rank(), shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0)
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory) if rank0 else None

    if rank0:
        print('\nInitializing model...')
    model, num_filter_enc = build_vae(config, num_channels, num_time, batch_size)
    model.apply(initialize_weights_He)
    model.apply(add_sn)   # spectral norm registered in place
    model = model.to(device)

    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    train_model, _is_fsdp = D.wrap_model(model, config, device)
    if rank0:
        log_model_summary(model, config, ema_model)

    total_epochs = int(config.get('training_epochs', 500))
    optimizer, scheduler = build_optimizer_scheduler(config, train_model.parameters(), total_epochs)

    alpha = float(config.get('alpha', 1.0))
    init_beta = 10.0 ** (-int(config.get('init_beta_divisor', 4)))
    beta_target = float(config.get('beta_target', 1.0))
    kl_warmup_epochs = int(config.get('kl_warmup_epochs', 0))
    kl_start_frac = float(config.get('kl_warmup_start_frac', 0.3))

    use_amp = bool(config.get('use_amp', False))
    amp_enabled = use_amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)

    val_interval = int(config.get('val_interval', 20))
    modelpath = config.get('vae_modelpath', '../../output/simulgenvae/simulgenvae_vae.pth')
    log_file = init_log_file(config, config_filename) if rank0 else None

    if rank0:
        print('\n' + '=' * 60)
        print('Starting hierarchical VAE training loop...')
        print('=' * 60 + '\n')
    start_time = time.time()
    valid_loss = float('nan')

    def checkpoint_payload(epoch):
        return {
            'stage': 'vae',
            'epoch': epoch,
            'model_state': D.full_state_dict(train_model, False),
            'ema_state': (D.unwrap_model(ema_model).state_dict() if ema_model is not None else None),
            'config': config,
            'normalization': normalization,
            'num_channels': num_channels,
            'num_time': num_time,
            'num_filter_enc': num_filter_enc,
        }

    def maybe_save(epoch):
        payload = checkpoint_payload(epoch)
        if rank0:
            save_checkpoint(modelpath, payload)

    try:
        for epoch in range(total_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_model.train()
            beta = _kl_beta(epoch, total_epochs, init_beta, beta_target,
                            kl_start_frac, kl_warmup_epochs)
            recon_sum, kl_sum, batches = 0.0, 0.0, 0
            for field, _idx in train_loader:
                field = field.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    _out, recon_loss, kl_losses, _recon_mse = train_model(field)
                    kl_total = sum(kl_losses)
                    loss = alpha * recon_loss + beta * kl_total
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    ema_model.update_parameters(model)
                recon_sum += float(recon_loss.item())
                kl_sum += float(kl_total.item())
                batches += 1

            scheduler.step()
            train_recon = D.reduce_epoch_mean(recon_sum, batches, device)
            train_kl = D.reduce_epoch_mean(kl_sum, batches, device)
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = D.unwrap_model(ema_model) if ema_model is not None else model
            if do_val and rank0:
                valid_loss = _validate(eval_model, val_loader, device, alpha)
                print(f'Epoch {epoch}/{total_epochs} Recon: {train_recon:.4e} '
                      f'KL: {train_kl:.4e} Beta: {beta:.2e} ValRecon: {valid_loss:.4e} LR: {current_lr:.2e}')
            elif rank0:
                print(f'Epoch {epoch}/{total_epochs} Recon: {train_recon:.4e} '
                      f'KL: {train_kl:.4e} Beta: {beta:.2e} LR: {current_lr:.2e}')
            if rank0:
                elapsed = time.time() - start_time
                val_str = f'ValRecon {valid_loss:.4e}' if do_val else 'Val skipped'
                append_log(log_file, f'Elapsed: {elapsed:.2f}s Epoch {epoch} '
                                     f'Recon {train_recon:.4e} KL {train_kl:.4e} Beta {beta:.4e} {val_str} '
                                     f'LR {current_lr:.4e}')

        maybe_save(total_epochs - 1)
        D.barrier()
        if rank0:
            print(f'\nTraining finished. VAE saved to {modelpath} (val recon {valid_loss:.2e})')
    except KeyboardInterrupt:
        if rank0:
            print('\nTraining interrupted by user. Saving checkpoint...')
        maybe_save(-1)


@torch.no_grad()
def _validate(model, val_loader, device, alpha):
    model.eval()
    total, batches = 0.0, 0
    for field, _idx in val_loader:
        field = field.to(device)
        _out, recon_loss, _kl, _mse = model(field)
        total += alpha * float(recon_loss.item())
        batches += 1
    model.train()
    return total / max(batches, 1)
