"""Stage 1: SDF-VAE training (reconstruction + KL)."""

import os
import time

import torch
from torch.utils.data import DataLoader

from general_modules.sdf_dataset import build_dataset_splits, compute_cond_stats
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE, sdf_loss, hybrid_geometry_losses
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


def vae_worker(config, config_filename='config.txt'):
    device = resolve_device(config)
    split_seed = int(config.get('split_seed', 42))

    print('\nLoading dataset...')
    train_dataset, val_dataset, test_dataset = build_dataset_splits(config, split_seed)
    cond_mean, cond_std = compute_cond_stats(train_dataset)

    num_workers = int(config.get('num_workers', 0))
    loader_kwargs = dict(
        batch_size=int(config.get('batch_size', 8)),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    print('\nInitializing model...')
    model = SDFVAE(config).to(device)
    init_modelpath = config.get('init_vae_modelpath')
    if init_modelpath:
        checkpoint = load_checkpoint(init_modelpath, device)
        state = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state)
        print(f'Initialized VAE weights from {init_modelpath}')
    ema_model = build_ema_model(model, config)
    if ema_model is not None:
        ema_model = ema_model.to(device)
    log_model_summary(model, config, ema_model)

    total_epochs = int(config.get('training_epochs', 500))
    optimizer, scheduler = build_optimizer_scheduler(config, model.parameters(), total_epochs)

    kl_weight = float(config.get('kl_weight', 1e-4))
    deterministic_warmup_epochs = int(config.get('deterministic_warmup_epochs', 0))
    posterior_noise_warmup_epochs = int(config.get('posterior_noise_warmup_epochs', 0))
    posterior_noise_max_scale = float(config.get('posterior_noise_max_scale', 1.0))
    kl_warmup_epochs = int(config.get('kl_warmup_epochs', 0))
    clamp_dist = float(config.get('clamp_dist', 0.1))

    # Hybrid geometry losses (TripoSG-style). Any positive weight enables a
    # dedicated fp32 training path because the normal/eikonal terms need
    # second-order gradients that are unstable under AMP.
    surface_weight = float(config.get('surface_weight', 0.0))
    normal_weight = float(config.get('normal_weight', 0.0))
    eikonal_weight = float(config.get('eikonal_weight', 0.0))
    hybrid_grad_points = int(config.get('hybrid_grad_points', 2048))
    use_hybrid = (surface_weight > 0 or normal_weight > 0 or eikonal_weight > 0)
    if use_hybrid:
        print(f'Hybrid VAE losses enabled (surface={surface_weight:g} '
              f'normal={normal_weight:g} eikonal={eikonal_weight:g}); '
              f'stage runs in fp32 (AMP bypassed for the gradient terms).')
    use_amp = bool(config.get('use_amp', False))
    amp_enabled = use_amp and device.type == 'cuda'
    amp_dtype = (torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    val_interval = int(config.get('val_interval', 5))
    test_interval = int(config.get('test_interval', 100))
    modelpath = config.get('vae_modelpath', './outputs/sdfflow_vae.pth')

    log_file = init_log_file(config, config_filename)
    print('\n' + '=' * 60)
    print('Starting SDF-VAE training loop...')
    print('=' * 60 + '\n')
    start_time = time.time()
    valid_loss = float('nan')

    def checkpoint_payload(epoch):
        return {
            'stage': 'vae',
            'epoch': epoch,
            'model_state': model.state_dict(),
            'ema_state': ema_model.state_dict() if ema_model is not None else None,
            'config': config,
            'cond_mean': cond_mean,
            'cond_std': cond_std,
            'cond_names': train_dataset.cond_names,
        }

    try:
        for epoch in range(total_epochs):
            model.train()
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
                    # fp32 path: reconstruction + KL + surface/normal/eikonal.
                    mu, logvar = model.encode(surface_points, surface_normals)
                    z = model.reparameterize(mu, logvar, posterior_noise_scale)
                    sdf_pred = model.decode(z, query_points)
                    recon = sdf_loss(sdf_pred, query_sdf, clamp_dist)
                    kl = model.kl_divergence(mu, logvar)
                    surface_l, normal_l, eikonal_l = hybrid_geometry_losses(
                        model, z, surface_points, surface_normals, query_points,
                        subsample=hybrid_grad_points)
                    hybrid_l = (surface_weight * surface_l + normal_weight * normal_l
                                + eikonal_weight * eikonal_l)
                    loss = recon + effective_kl_weight * kl + hybrid_l
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    hybrid_sum += float(hybrid_l.item())
                else:
                    with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                        sdf_pred, kl, _, _ = model(
                            surface_points, surface_normals, query_points,
                            posterior_noise_scale=posterior_noise_scale)
                        recon = sdf_loss(sdf_pred.float(), query_sdf, clamp_dist)
                        loss = recon + effective_kl_weight * kl.float()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                if ema_model is not None:
                    ema_model.update_parameters(model)

                recon_sum += recon.item()
                kl_sum += kl.item()
                batches += 1

            scheduler.step()
            train_loss = recon_sum / max(batches, 1)
            train_kl = kl_sum / max(batches, 1)
            train_hybrid = hybrid_sum / max(batches, 1)
            hybrid_str = f' Hybrid: {train_hybrid:.2e}' if use_hybrid else ''
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = ema_model.module if ema_model is not None else model
            if do_val:
                valid_loss = _validate(eval_model, val_loader, device, clamp_dist)
                print(f'Epoch {epoch}/{total_epochs} TrainSDF: {train_loss:.2e} '
                      f'KL: {train_kl:.2e}{hybrid_str} ValidSDF: {valid_loss:.2e} LR: {current_lr:.2e} '
                      f'KLWeight: {effective_kl_weight:.2e} PosteriorNoise: {posterior_noise_scale:.2f}')
            else:
                print(f'Epoch {epoch}/{total_epochs} TrainSDF: {train_loss:.2e} '
                      f'KL: {train_kl:.2e}{hybrid_str} LR: {current_lr:.2e} '
                      f'KLWeight: {effective_kl_weight:.2e} PosteriorNoise: {posterior_noise_scale:.2f}')

            elapsed = time.time() - start_time
            val_str = f'Valid {valid_loss:.4e}' if do_val else 'Valid skipped'
            append_log(log_file, f'Elapsed: {elapsed:.2f}s Epoch {epoch} '
                                 f'TrainSDF {train_loss:.4e} KL {train_kl:.4e} {val_str} '
                                 f'LR: {current_lr:.4e} KLWeight: {effective_kl_weight:.4e} '
                                 f'PosteriorNoise: {posterior_noise_scale:.4f}')

            if epoch % test_interval == 0 or epoch == total_epochs - 1:
                run_reconstruction_test(eval_model, test_dataset, device, config, epoch)
                save_checkpoint(modelpath, checkpoint_payload(epoch))

        save_checkpoint(modelpath, checkpoint_payload(total_epochs - 1))
        print(f'\nTraining finished. VAE saved to {modelpath} (val SDF loss {valid_loss:.2e})')
    except KeyboardInterrupt:
        print('\nTraining interrupted by user. Saving checkpoint...')
        save_checkpoint(modelpath, checkpoint_payload(-1))


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
