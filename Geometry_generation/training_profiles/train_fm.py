"""Stage 2: flow matching over frozen-VAE latents (optionally conditional + CFG).

Runs single-process or, under `parallel_mode` ddp/fsdp, as one rank of a spawned
distributed job. Each rank deterministically encodes the dataset to the same
frozen latents (identical normalization), shards the latent batch, and shares
gradients. Rank 0 owns validation, the generation test, and checkpoints. FSDP is
the intended "model split" for a large velocity DiT.
"""

import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from general_modules import distributed as D
from general_modules.sdf_dataset import build_dataset_splits
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE
from model.velocity_net import VelocityNet, flow_matching_loss, sample_latents
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
from training_profiles.train_vae import _clip_grads


def _state_dict_to_cpu(state_dict):
    """Move every tensor in a (possibly None) state dict to CPU before it is
    embedded in another checkpoint -- the source VAE checkpoint may have been
    loaded onto a CUDA device."""
    if state_dict is None:
        return None
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state_dict.items()}


def fm_worker(config, config_filename='config.txt'):
    device = resolve_device(config)
    split_seed = int(config.get('split_seed', 42))
    rank0 = D.is_main_process()
    world_size = D.get_world_size()

    # ---- Frozen VAE (loaded identically on every rank) ----
    vae_path = config.get('vae_modelpath', './outputs/sdfflow_vae.pth')
    if rank0:
        print(f'\nLoading frozen VAE from {vae_path}')
    vae_ckpt = load_checkpoint(vae_path, device)
    vae = SDFVAE(vae_ckpt['config']).to(device)
    state = vae_ckpt['ema_state'] or vae_ckpt['model_state']
    if vae_ckpt['ema_state'] is not None:
        # AveragedModel state dict prefixes parameters with 'module.'
        state = {k.replace('module.', '', 1): v for k, v in state.items() if k != 'n_averaged'}
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # ---- Encode all shapes to latents (deterministic: eval + mu) ----
    if rank0:
        print('\nEncoding dataset to latents...')
    train_dataset, val_dataset, _ = build_dataset_splits(config, split_seed)
    z_train, c_train = _encode_split(vae, train_dataset, device, config)
    z_val, c_val = _encode_split(vae, val_dataset, device, config)

    latent_mean = z_train.mean(dim=0, keepdim=True)
    latent_std = z_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    z_train_n = (z_train - latent_mean) / latent_std
    z_val_n = (z_val - latent_mean) / latent_std

    use_conditions = bool(config.get('use_conditions', False))
    cond_names = []
    cond_min = cond_max = None
    cond_clip = float(config.get('condition_clip', 5.0))
    if use_conditions:
        requested_names = config.get('condition_names', train_dataset.cond_names)
        if not isinstance(requested_names, list):
            requested_names = [requested_names]
        requested_names = [str(name) for name in requested_names]
        unknown = [name for name in requested_names if name not in train_dataset.cond_names]
        if unknown:
            raise ValueError(f'Unknown condition_names {unknown}; available: '
                             f'{train_dataset.cond_names}')
        if len(set(requested_names)) != len(requested_names):
            raise ValueError(f'condition_names contains duplicates: {requested_names}')

        selected = [train_dataset.cond_names.index(name) for name in requested_names]
        c_train = c_train[:, selected]
        c_val = c_val[:, selected]
        cond_names = requested_names
        cond_dim = len(cond_names)
        if cond_dim == 0:
            raise ValueError('use_conditions True requires at least one condition_name')

        cond_mean = c_train.mean(dim=0, keepdim=True)
        raw_cond_std = c_train.std(dim=0, keepdim=True)
        min_condition_std = float(config.get('min_condition_std', 1e-5))
        constant = [cond_names[i] for i, value in enumerate(raw_cond_std.squeeze(0))
                    if float(value) < min_condition_std]
        if constant:
            raise ValueError(
                f'Condition descriptors have near-zero training variance: {constant}. '
                'Remove them with condition_names instead of normalizing by an epsilon.')
        cond_std = raw_cond_std
        cond_min = c_train.amin(dim=0, keepdim=True)
        cond_max = c_train.amax(dim=0, keepdim=True)
        c_train_n = ((c_train - cond_mean) / cond_std).clamp(-cond_clip, cond_clip)
        c_val_n = ((c_val - cond_mean) / cond_std).clamp(-cond_clip, cond_clip)
        if rank0:
            print(f'Conditional FM: cond_dim={cond_dim} ({cond_names})')
            print(f'Condition normalization clipped to +/-{cond_clip:g} sigma')
    else:
        cond_dim = 0
        cond_mean = cond_std = None
        c_train_n = torch.zeros(len(z_train_n), 0)
        c_val_n = torch.zeros(len(z_val_n), 0)
        if rank0:
            print('Unconditional FM (use_conditions False)')

    latent_flat_dim = z_train.shape[1]
    if rank0:
        print(f'Latents: train {z_train.shape}, val {z_val.shape}')

    batch_size = int(config.get('batch_size', 64))
    train_ds = TensorDataset(z_train_n, c_train_n)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=D.get_rank(), shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # ---- Model ----
    if rank0:
        print('\nInitializing velocity network...')
    model = VelocityNet(config, latent_flat_dim, cond_dim=cond_dim).to(device)

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

    total_epochs = int(config.get('training_epochs', 2000))
    optimizer, scheduler = build_optimizer_scheduler(config, train_model.parameters(), total_epochs)

    cond_dropout = float(config.get('cond_dropout', 0.1))
    time_sampling = str(config.get('fm_time_sampling', 'uniform')).lower()
    if time_sampling not in ('uniform', 'logit_normal'):
        raise ValueError("fm_time_sampling must be 'uniform' or 'logit_normal'")
    logit_mean = float(config.get('fm_time_logit_mean', 0.0))
    logit_std = float(config.get('fm_time_logit_std', 1.0))
    if time_sampling == 'logit_normal' and rank0:
        print(f'FM timestep sampling: logit-normal (mean={logit_mean:g}, std={logit_std:g})')
    use_amp = bool(config.get('use_amp', False))
    amp_enabled = use_amp and device.type == 'cuda' and not is_fsdp
    amp_dtype = (torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported()
                 else torch.float16)
    scaler = torch.amp.GradScaler(
        'cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    val_interval = int(config.get('val_interval', 10))
    test_interval = int(config.get('test_interval', 250))
    modelpath = config.get('fm_modelpath', './outputs/sdfflow_fm.pth')
    params = [p for p in train_model.parameters() if p.requires_grad]

    log_file = init_log_file(config, config_filename) if rank0 else None
    if rank0:
        print('\n' + '=' * 60)
        print('Starting flow-matching training loop...')
        print('=' * 60 + '\n')
    start_time = time.time()
    valid_loss = float('nan')

    def checkpoint_payload(epoch):
        return {
            'schema_version': 'sdfflow_infer_v1',
            'stage': 'fm',
            'epoch': epoch,
            'model_state': D.full_state_dict(train_model, is_fsdp),
            'ema_state': (D.unwrap_model(ema_model).state_dict()
                          if ema_model is not None else None),
            'config': config,
            'vae_modelpath': vae_path,
            # The FM checkpoint is the one canonical inference artifact: it
            # embeds the frozen VAE it was trained against (co-located, both
            # moved to CPU) so a stand-alone inference bundle needs only this
            # one file (INFERENCE_BUNDLE_PLAN.md section 5.5). `vae_modelpath`
            # above is kept for backward compatibility / provenance only.
            'vae': {
                'model_state': _state_dict_to_cpu(vae_ckpt['model_state']),
                'ema_state': _state_dict_to_cpu(vae_ckpt.get('ema_state')),
                'config': vae_ckpt['config'],
                'cond_mean': vae_ckpt.get('cond_mean'),
                'cond_std': vae_ckpt.get('cond_std'),
                'cond_names': vae_ckpt.get('cond_names'),
            },
            'latent_flat_dim': latent_flat_dim,
            'latent_mean': latent_mean.cpu(),
            'latent_std': latent_std.cpu(),
            'cond_dim': cond_dim,
            'cond_mean': cond_mean.cpu() if cond_mean is not None else None,
            'cond_std': cond_std.cpu() if cond_std is not None else None,
            'cond_min': cond_min.cpu() if cond_min is not None else None,
            'cond_max': cond_max.cpu() if cond_max is not None else None,
            'cond_clip': cond_clip if cond_dim > 0 else None,
            'cond_names': cond_names,
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
            loss_sum, batches = 0.0, 0
            for z_batch, c_batch in train_loader:
                z_batch = z_batch.to(device, non_blocking=True)
                cond = c_batch.to(device, non_blocking=True) if cond_dim > 0 else None
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                    loss = flow_matching_loss(
                        train_model, z_batch, cond=cond, cond_dropout=cond_dropout,
                        time_sampling=time_sampling, logit_mean=logit_mean,
                        logit_std=logit_std)
                scaler.scale(loss).backward()
                if amp_enabled and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                _clip_grads(train_model, is_fsdp, params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    ema_model.update_parameters(model)
                loss_sum += loss.item()
                batches += 1

            scheduler.step()
            train_loss = D.reduce_epoch_mean(loss_sum, batches, device)
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = D.unwrap_model(ema_model) if ema_model is not None else model
            if do_val and rank0:
                valid_loss = _validate(eval_model, z_val_n, c_val_n, device, cond_dim)
                print(f'Epoch {epoch}/{total_epochs} TrainFM: {train_loss:.2e} '
                      f'ValidFM: {valid_loss:.2e} LR: {current_lr:.2e}')
            elif rank0:
                print(f'Epoch {epoch}/{total_epochs} TrainFM: {train_loss:.2e} LR: {current_lr:.2e}')

            if rank0:
                elapsed = time.time() - start_time
                val_str = f'Valid {valid_loss:.4e}' if do_val else 'Valid skipped'
                append_log(log_file, f'Elapsed: {elapsed:.2f}s Epoch {epoch} '
                                     f'TrainFM {train_loss:.4e} {val_str} LR: {current_lr:.4e}')

            if epoch % test_interval == 0 or epoch == total_epochs - 1:
                if rank0:
                    run_generation_test(eval_model, vae, device, config, epoch,
                                        latent_flat_dim, latent_mean, latent_std,
                                        cond_dim=cond_dim)
                maybe_save(epoch)
                D.barrier()

        maybe_save(total_epochs - 1)
        if rank0:
            print(f'\nTraining finished. FM saved to {modelpath} (val FM loss {valid_loss:.2e})')
    except KeyboardInterrupt:
        if rank0:
            print('\nTraining interrupted by user. Saving checkpoint...')
        maybe_save(-1)


@torch.no_grad()
def _encode_split(vae, dataset, device, config):
    """Encode every shape in a split to (mu latents, conditions)."""
    latents, conds = [], []
    batch, batch_c = [], []
    batch_size = int(config.get('encode_batch_size', 16))

    def flush():
        if not batch:
            return
        pts = torch.stack([b[0] for b in batch]).to(device)
        nrm = torch.stack([b[1] for b in batch]).to(device)
        mu, _ = vae.encode(pts, nrm)
        latents.append(mu.flatten(1).cpu())
        conds.extend(batch_c)
        batch.clear()
        batch_c.clear()

    for i in range(len(dataset)):
        item = dataset[i]
        batch.append((item['surface_points'], item['surface_normals']))
        batch_c.append(item['cond'])
        if len(batch) == batch_size:
            flush()
    flush()
    return torch.cat(latents, dim=0), torch.stack(conds, dim=0)


@torch.no_grad()
def _validate(model, z_val_n, c_val_n, device, cond_dim):
    model.eval()
    g = torch.Generator(device='cpu').manual_seed(0)
    z = z_val_n.to(device)
    cond = c_val_n.to(device) if cond_dim > 0 else None
    noise = torch.randn(z.shape, generator=g).to(device)
    t = torch.rand(z.shape[0], generator=g).to(device)
    z_t = (1 - t[:, None]) * noise + t[:, None] * z
    v_pred = model(z_t, t, cond=cond)
    return (v_pred - (z - noise)).pow(2).mean().item()


@torch.no_grad()
def run_generation_test(model, vae, device, config, epoch, latent_flat_dim,
                        latent_mean, latent_std, cond_dim=0):
    """Sample mean-conditioned latents (or unconditional latents) for a smoke test."""
    model.eval()
    out_dir = os.path.join(config.get('output_dir', './outputs'), 'fm_samples')
    os.makedirs(out_dir, exist_ok=True)
    num_samples = min(int(config.get('num_test_shapes', 2)), 8)
    resolution = int(config.get('mc_resolution_test', 96))

    cond = torch.zeros(num_samples, cond_dim, device=device) if cond_dim > 0 else None
    z_n = sample_latents(model, num_samples, latent_flat_dim, device, cond=cond,
                         ode_steps=int(config.get('ode_steps', 50)))
    z = z_n * latent_std.to(device) + latent_mean.to(device)
    for i in range(num_samples):
        volume = decode_sdf_grid(vae, z[i:i + 1], resolution=resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        if report['valid']:
            path = os.path.join(out_dir, f'epoch{epoch:05d}_sample{i}.stl')
            mesh.export(path)
            print(f'  [test] sample {i}: watertight={report["watertight"]} faces={report["faces"]} -> {path}')
        else:
            print(f'  [test] sample {i}: NO ZERO CROSSING')
