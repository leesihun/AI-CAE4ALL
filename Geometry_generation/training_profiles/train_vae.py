"""Stage 1: SDF-VAE training (reconstruction + KL + optional hybrid losses).

Runs single-process or, under `parallel_mode` ddp/fsdp, as one rank of a spawned
distributed job. Gradients are shared across ranks; rank 0 owns validation,
logging, the periodic reconstruction test, and checkpoint writes.
"""

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from general_modules import distributed as D
from general_modules.sdf_dataset import build_dataset_splits, compute_cond_stats
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE, sdf_loss
from training_profiles.setup import (
    append_log,
    build_ema_model,
    build_optimizer_scheduler,
    init_log_file,
    load_checkpoint,
    log_model_summary,
    resolve_device,
    save_checkpoint,
)


def _clip_grads(train_model, is_fsdp, params, max_norm=1.0):
    if is_fsdp:
        train_model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def vae_worker(config, config_filename='config.txt'):
    device = resolve_device(config)
    split_seed = int(config.get('split_seed', 42))
    rank0 = D.is_main_process()
    world_size = D.get_world_size()

    if rank0:
        print('\nLoading dataset...')
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    cond_mean, cond_std = compute_cond_stats(train_dataset)

    num_workers = int(config.get('num_workers', 0))
    pin_memory = torch.cuda.is_available()
    mp_context = 'spawn' if (num_workers > 0 and D.is_dist()) else None
    batch_size = int(config.get('batch_size', 8))

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=D.get_rank(), shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0, multiprocessing_context=mp_context)
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0)
    # Validation runs on rank 0 only (no distributed forward), on the raw model.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=num_workers > 0) if rank0 else None

    if rank0:
        print('\nInitializing model...')
    model = SDFVAE(config).to(device)
    init_modelpath = config.get('init_vae_modelpath')
    if init_modelpath:
        checkpoint = load_checkpoint(init_modelpath, device)
        state = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state)
        if rank0:
            print(f'Initialized VAE weights from {init_modelpath}')

    # EMA is built from the raw module. FSDP shards parameters in place, which
    # AveragedModel cannot mirror cleanly, so EMA is disabled under FSDP.
    is_fsdp = D.is_dist() and D.parallel_mode(config) == 'fsdp'
    ema_config = config
    if is_fsdp and config.get('use_ema', False):
        if rank0:
            print('NOTE: EMA is not supported under parallel_mode=fsdp; disabling it.')
        ema_config = dict(config); ema_config['use_ema'] = False
    ema_model = build_ema_model(model, ema_config)
    if ema_model is not None:
        ema_model = ema_model.to(device)

    train_model, is_fsdp = D.wrap_model(model, config, device)
    if rank0:
        log_model_summary(model, config, ema_model)

    total_epochs = int(config.get('training_epochs', 500))
    optimizer, scheduler = build_optimizer_scheduler(config, train_model.parameters(), total_epochs)

    kl_weight = float(config.get('kl_weight', 1e-4))
    deterministic_warmup_epochs = int(config.get('deterministic_warmup_epochs', 0))
    posterior_noise_warmup_epochs = int(config.get('posterior_noise_warmup_epochs', 0))
    posterior_noise_max_scale = float(config.get('posterior_noise_max_scale', 1.0))
    kl_warmup_epochs = int(config.get('kl_warmup_epochs', 0))
    clamp_dist = float(config.get('clamp_dist', 0.1))

    # Hybrid geometry losses (TripoSG-style). Any positive weight enables a
    # dedicated fp32 path because the normal/eikonal terms need second-order
    # gradients that are unstable under AMP.
    surface_weight = float(config.get('surface_weight', 0.0))
    normal_weight = float(config.get('normal_weight', 0.0))
    eikonal_weight = float(config.get('eikonal_weight', 0.0))
    hybrid_grad_points = int(config.get('hybrid_grad_points', 2048))
    use_hybrid = (surface_weight > 0 or normal_weight > 0 or eikonal_weight > 0)
    if use_hybrid and rank0:
        print(f'Hybrid VAE losses enabled (surface={surface_weight:g} '
              f'normal={normal_weight:g} eikonal={eikonal_weight:g}); '
              f'stage runs in fp32 (AMP bypassed for the gradient terms).')

    # FSDP performs its own mixed precision; the trainer's autocast is only for
    # single/DDP. Hybrid needs fp32 and never autocasts.
    use_amp = bool(config.get('use_amp', False))
    amp_enabled = use_amp and device.type == 'cuda' and not is_fsdp and not use_hybrid
    amp_dtype = (torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    val_interval = int(config.get('val_interval', 5))
    test_interval = int(config.get('test_interval', 100))
    modelpath = config.get('vae_modelpath', './outputs/sdfflow_vae.pth')

    log_file = init_log_file(config, config_filename) if rank0 else None
    if rank0:
        print('\n' + '=' * 60)
        print('Starting SDF-VAE training loop...')
        print('=' * 60 + '\n')
    start_time = time.time()
    valid_loss = float('nan')
    params = [p for p in train_model.parameters() if p.requires_grad]

    def checkpoint_payload(epoch):
        return {
            'stage': 'vae',
            'epoch': epoch,
            'model_state': D.full_state_dict(train_model, is_fsdp),
            'ema_state': (D.unwrap_model(ema_model).state_dict()
                          if ema_model is not None else None),
            'config': config,
            'cond_mean': cond_mean,
            'cond_std': cond_std,
            'cond_names': train_dataset.cond_names,
        }

    def maybe_save(epoch):
        payload = checkpoint_payload(epoch)  # collective under FSDP; call on all ranks
        if rank0:
            save_checkpoint(modelpath, payload)

    try:
        for epoch in range(total_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_model.train()
            recon_sum, kl_sum, batches = 0.0, 0.0, 0
            hybrid_sum = 0.0
            posterior_noise_scale = posterior_noise_max_scale * _warmup_scale(
                epoch, deterministic_warmup_epochs, posterior_noise_warmup_epochs)
            effective_kl_weight = kl_weight * _warmup_scale(
                epoch, deterministic_warmup_epochs, kl_warmup_epochs)
            for batch in train_loader:
                surface_points = batch['surface_points'].to(device, non_blocking=True)
                surface_normals = batch['surface_normals'].to(device, non_blocking=True)
                query_points = batch['query_points'].to(device, non_blocking=True)
                query_sdf = batch['query_sdf'].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                if use_hybrid:
                    losses = train_model(
                        surface_points, surface_normals, query_points, query_sdf,
                        posterior_noise_scale=posterior_noise_scale, clamp_dist=clamp_dist,
                        surface_weight=surface_weight, normal_weight=normal_weight,
                        eikonal_weight=eikonal_weight, hybrid_grad_points=hybrid_grad_points)
                    recon, kl = losses['recon'], losses['kl']
                    hybrid_l = (surface_weight * losses['surface']
                                + normal_weight * losses['normal']
                                + eikonal_weight * losses['eikonal'])
                    loss = recon + effective_kl_weight * kl + hybrid_l
                    loss.backward()
                    _clip_grads(train_model, is_fsdp, params, 1.0)
                    optimizer.step()
                    hybrid_sum += float(hybrid_l.item())
                else:
                    with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                        losses = train_model(
                            surface_points, surface_normals, query_points, query_sdf,
                            posterior_noise_scale=posterior_noise_scale, clamp_dist=clamp_dist)
                        recon, kl = losses['recon'], losses['kl']
                        loss = recon + effective_kl_weight * kl
                    scaler.scale(loss).backward()
                    if amp_enabled and amp_dtype == torch.float16:
                        scaler.unscale_(optimizer)
                    _clip_grads(train_model, is_fsdp, params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                if ema_model is not None:
                    ema_model.update_parameters(model)

                recon_sum += recon.item()
                kl_sum += kl.item()
                batches += 1

            scheduler.step()
            train_loss = D.reduce_epoch_mean(recon_sum, batches, device)
            train_kl = D.reduce_epoch_mean(kl_sum, batches, device)
            train_hybrid = D.reduce_epoch_mean(hybrid_sum, batches, device)
            hybrid_str = f' Hybrid: {train_hybrid:.2e}' if use_hybrid else ''
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = D.unwrap_model(ema_model) if ema_model is not None else model
            if do_val and rank0:
                valid_loss = _validate(eval_model, val_loader, device, clamp_dist)
                print(f'Epoch {epoch}/{total_epochs} TrainSDF: {train_loss:.2e} '
                      f'KL: {train_kl:.2e}{hybrid_str} ValidSDF: {valid_loss:.2e} LR: {current_lr:.2e} '
                      f'KLWeight: {effective_kl_weight:.2e} PosteriorNoise: {posterior_noise_scale:.2f}')
            elif rank0:
                print(f'Epoch {epoch}/{total_epochs} TrainSDF: {train_loss:.2e} '
                      f'KL: {train_kl:.2e}{hybrid_str} LR: {current_lr:.2e} '
                      f'KLWeight: {effective_kl_weight:.2e} PosteriorNoise: {posterior_noise_scale:.2f}')

            if rank0:
                elapsed = time.time() - start_time
                val_str = f'Valid {valid_loss:.4e}' if do_val else 'Valid skipped'
                append_log(log_file, f'Elapsed: {elapsed:.2f}s Epoch {epoch} '
                                     f'TrainSDF {train_loss:.4e} KL {train_kl:.4e} {val_str} '
                                     f'LR: {current_lr:.4e} KLWeight: {effective_kl_weight:.4e} '
                                     f'PosteriorNoise: {posterior_noise_scale:.4f}')

            if epoch % test_interval == 0 or epoch == total_epochs - 1:
                if rank0:
                    run_reconstruction_test(eval_model, test_dataset, device, config, epoch)
                maybe_save(epoch)
                D.barrier()

        maybe_save(total_epochs - 1)
        if rank0:
            print(f'\nTraining finished. VAE saved to {modelpath} (val SDF loss {valid_loss:.2e})')
    except KeyboardInterrupt:
        if rank0:
            print('\nTraining interrupted by user. Saving checkpoint...')
        maybe_save(-1)


def _warmup_scale(epoch, start_epoch, warmup_epochs):
    """Return a 0..1 linear ramp after an initial deterministic period."""
    if epoch < start_epoch:
        return 0.0
    if warmup_epochs <= 0:
        return 1.0
    return min((epoch - start_epoch + 1) / warmup_epochs, 1.0)


@torch.no_grad()
def _validate(model, val_loader, device, clamp_dist):
    model.eval()
    total, batches = 0.0, 0
    for batch in val_loader:
        surface_points = batch['surface_points'].to(device)
        surface_normals = batch['surface_normals'].to(device)
        query_points = batch['query_points'].to(device)
        query_sdf = batch['query_sdf'].to(device)
        mu, _ = model.encode(surface_points, surface_normals)
        sdf_pred = model.decode(mu, query_points)
        total += sdf_loss(sdf_pred.float(), query_sdf, clamp_dist).item()
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def run_reconstruction_test(model, test_dataset, device, config, epoch):
    """Reconstruct a few test shapes through mu -> Marching Cubes -> STL."""
    model.eval()
    out_dir = os.path.join(config.get('output_dir', './outputs'), 'vae_recon')
    os.makedirs(out_dir, exist_ok=True)
    resolution = int(config.get('mc_resolution_test', 96))
    num_shapes = min(int(config.get('num_test_shapes', 2)), len(test_dataset))

    for i in range(num_shapes):
        item = test_dataset[i]
        surface_points = item['surface_points'].unsqueeze(0).to(device)
        surface_normals = item['surface_normals'].unsqueeze(0).to(device)
        mu, _ = model.encode(surface_points, surface_normals)
        volume = decode_sdf_grid(model, mu.flatten(1), resolution=resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        if report['valid']:
            path = os.path.join(out_dir, f'epoch{epoch:05d}_shape{int(item["shape_idx"])}.stl')
            mesh.export(path)
            print(f'  [test] recon shape {int(item["shape_idx"])}: watertight={report["watertight"]} '
                  f'faces={report["faces"]} -> {path}')
        else:
            print(f'  [test] recon shape {int(item["shape_idx"])}: NO ZERO CROSSING')
