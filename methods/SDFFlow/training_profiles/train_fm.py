"""Stage 2: flow matching over frozen-VAE latents (optionally conditional + CFG).

Runs single-process or, under `parallel_mode` ddp/fsdp, as one rank of a spawned
distributed job. Every rank encodes the full train/val splits itself; the
dataset is switched to deterministic (seeded per-shape) encoder subsampling for
that pass, so all ranks -- and repeated runs of the same checkpoint and seed --
hold bit-identical frozen latents and normalization statistics. The latent batch
is then sharded and gradients are shared. Rank 0 owns validation, the generation
test, and checkpoints. FSDP is the intended "model split" for a large velocity
DiT.

With `fm_best_modelpath` set, the best-so-far validation model is additionally
checkpointed there, carrying the same complete payload (the frozen VAE embedded)
as the final save; the final save to `fm_modelpath` is unchanged and remains the
pipeline's completeness signal.
"""

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from general_modules import distributed as D
from general_modules.sdf_dataset import build_dataset_splits
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from general_modules.mesh_render import ENDPOINT_COLOR, MIDDLE_COLOR, plot_mesh_strip
from general_modules.resume_state import TrainingResume
from model.sdf_vae import SDFVAE
from model.velocity_net import VelocityNet, flow_matching_loss, sample_latents
from training_profiles.setup import (
    append_log,
    build_ema_model,
    build_optimizer_scheduler,
    ema_horizon_warning,
    identical_across_ranks,
    init_log_file,
    load_checkpoint,
    log_model_summary,
    resolve_device,
    save_checkpoint,
    seed_stage,
    seeded_generator,
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
    # Optional global seeding (model init, shuffle order, flow-matching noise),
    # offset by the rank so no two ranks draw the same t / noise pairs; model
    # construction is put back on the rank-independent base seed below.
    run_seed = seed_stage(config, stage='FM', offset=D.get_rank(), verbose=rank0)

    # ---- Frozen VAE (loaded identically on every rank) ----
    vae_path = config.get('vae_modelpath', '../../output/geometry_generation/sdfflow_vae.pth')
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

    # ---- Encode all shapes to latents (deterministic: eval, fixed subsample) ----
    # `fm_latent_draws` K > 1 replaces each TRAIN shape's single encoder-mean
    # latent with K posterior samples (a different fixed point subsample + mu +
    # std * eps, std floored as in VAE training). Val keeps the mean, so ValidFM
    # stays comparable across K. One epoch then covers K x N latents, i.e. K x
    # the optimizer steps: lower `training_epochs` to hold the step budget.
    latent_draws = int(config.get('fm_latent_draws', 1))
    if latent_draws < 1:
        raise ValueError(f'fm_latent_draws must be >= 1, got {latent_draws}')
    if rank0:
        print('\nEncoding dataset to latents...')
    train_dataset, val_dataset, _ = build_dataset_splits(config, split_seed)
    # The draw noise follows the run seed (rank-independent: `run_seed` carries
    # the rank offset), falling back to split_seed for an unseeded run.
    raw_seed = config.get('seed')
    draw_seed = split_seed if raw_seed is None or str(raw_seed).strip() == '' else int(raw_seed)
    z_train, c_train = _encode_split(vae, train_dataset, device, config,
                                     draws=latent_draws, draw_seed=draw_seed)
    z_val, c_val = _encode_split(vae, val_dataset, device, config)
    if rank0 and latent_draws > 1:
        print(f'FM latent cache: {latent_draws} posterior draws per train shape -> '
              f'{z_train.shape[0]} train latents ({len(train_dataset)} shapes); val keeps the mean')

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

        # Finiteness FIRST: `add_fea_conditions.py --allow_missing` writes NaN
        # rows by design (an unmatched shape, or a non-positive value under a
        # log transform), and NaN is invisible to the near-zero-variance guard
        # below -- `float('nan') < 1e-5` is False. One NaN row makes cond_mean
        # and cond_std NaN for that column, which makes the normalized column
        # NaN for EVERY sample, which makes the loss and then every weight NaN
        # from the first optimizer step. The run then "completes" and writes a
        # checkpoint that decodes nothing.
        nonfinite = []
        for split_name, tensor in (('train', c_train), ('val', c_val)):
            if tensor.numel() == 0:
                continue
            finite = torch.isfinite(tensor)
            for i in range(cond_dim):
                bad = int((~finite[:, i]).sum())
                if bad:
                    nonfinite.append(f'{cond_names[i]} ({bad} non-finite {split_name} row(s))')
        if nonfinite:
            raise ValueError(
                f'Condition values are not finite: {nonfinite}. The usual source is '
                'add_fea_conditions.py --allow_missing, which writes NaN rows for shapes the '
                'CSV does not cover and for non-positive values under a log transform. Rebuild '
                'the cond_extra sidecar without --allow_missing, or drop those names from '
                'condition_names.')
        cond_mean = c_train.mean(dim=0, keepdim=True)
        raw_cond_std = c_train.std(dim=0, keepdim=True)
        min_condition_std = float(config.get('min_condition_std', 1e-5))
        # `not (value >= min)` rather than `value < min` so a NaN that somehow
        # survives the check above is still reported instead of passing.
        constant = [cond_names[i] for i, value in enumerate(raw_cond_std.squeeze(0))
                    if not (float(value) >= min_condition_std)]
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

    # Real shapes whose TRUE conditions the periodic generation picture samples
    # from (val; train's first posterior draw only when val is empty).
    generation_reference = None
    if rank0 and cond_dim > 0:
        binary = ((c_train == 0) | (c_train == 1)).all(dim=0).tolist()
        if len(z_val_n):
            generation_reference = _pick_generation_shapes(
                config, 'val', z_val_n, c_val, c_val_n, val_dataset.indices, cond_names, binary)
        else:
            n_shapes = len(train_dataset)
            generation_reference = _pick_generation_shapes(
                config, 'train', z_train_n[:n_shapes], c_train[:n_shapes], c_train_n[:n_shapes],
                train_dataset.indices, cond_names, binary)

    batch_size = int(config.get('batch_size', 64))
    train_ds = TensorDataset(z_train_n, c_train_n)
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=D.get_rank(), shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  generator=seeded_generator(run_seed))

    # ---- Model ----
    if rank0:
        print('\nInitializing velocity network...')
    with identical_across_ranks(run_seed, D.get_rank()):
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
    if rank0:
        ema_warning = ema_horizon_warning(ema_config, len(train_loader), total_epochs, stage='FM')
        if ema_warning:
            print(ema_warning)

    cond_dropout = float(config.get('cond_dropout', 0.1))
    cond_dropout_mode = str(config.get('cond_dropout_mode', 'all')).lower()
    cond_dropout_all_prob = float(config.get('cond_dropout_all_prob', 0.1))
    if not 0.0 <= cond_dropout_all_prob < 1.0:
        raise ValueError('cond_dropout_all_prob must lie in [0, 1)')
    if rank0 and cond_dim > 0 and cond_dropout > 0 and cond_dropout_mode == 'per_dim':
        implicit = cond_dropout ** cond_dim
        print(f'Condition dropout: per_dim p={cond_dropout:g} over {cond_dim} conditions -> the '
              f'all-masked (unconditional) row would appear with probability {implicit:.2g} by '
              f'chance alone; cond_dropout_all_prob={cond_dropout_all_prob:g} draws it '
              'explicitly so the CFG branch is trained. At 0 the branch is starved and '
              'cfg_scale must stay 1.0.')
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
    modelpath = config.get('fm_modelpath', '../../output/geometry_generation/sdfflow_fm.pth')
    # Optional best-validation checkpoint (rank 0 writes; the decision is
    # broadcast so the FSDP state-dict gather below stays collective).
    #
    # No warmup guard here, unlike `vae_best_modelpath`: the FM objective is the
    # SAME quantity at every epoch -- there is no KL/beta ramp, and
    # `fm_warmup_epochs` moves the learning rate only -- so every ValidFM value
    # is comparable and its minimum is a genuine minimum rather than an artifact
    # of a partially applied loss term.
    best_modelpath = config.get('fm_best_modelpath')
    best_valid_loss = float('inf')
    params = [p for p in train_model.parameters() if p.requires_grad]
    valid_loss = float('nan')

    # Opt-in mid-training resume (`resume_training`, single process only). The
    # latent cache and its statistics are re-encoded deterministically above;
    # a state written before the VAE checkpoint was rewritten goes stale.
    start_epoch, elapsed_before = 0, 0.0
    resume = TrainingResume(config, modelpath, upstream=[vae_path])
    resumed = resume.load()
    if resumed is not None:
        model.load_state_dict(resumed['model'])
        if ema_model is not None:
            ema_model.load_state_dict(resumed['ema'])
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        if scaler.is_enabled():
            scaler.load_state_dict(resumed['scaler'])
        if train_loader.generator is not None:
            train_loader.generator.set_state(resumed['loader_generator'])
        best_valid_loss, valid_loss = resumed['best_valid_loss'], resumed['valid_loss']
        start_epoch, elapsed_before = resumed['epoch'] + 1, resumed['elapsed']
        del resumed

    log_file = init_log_file(config, config_filename, resumed_at=start_epoch or None) if rank0 else None
    if rank0:
        print('\n' + '=' * 60)
        print('Starting flow-matching training loop...')
        print('=' * 60 + '\n')
    start_time = time.time() - elapsed_before

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
        resume.restore_rng()
        for epoch in range(start_epoch, total_epochs):
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
                        logit_std=logit_std, cond_dropout_all_prob=cond_dropout_all_prob)
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

            # Read before step(): afterwards it is the next epoch's LR, and after the
            # last epoch a warm restart reports the peak LR for an epoch that never runs.
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            train_loss = D.reduce_epoch_mean(loss_sum, batches, device)

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

            # Best-validation checkpoint. `do_val` is identical on every rank;
            # only rank 0 knows the loss, so its verdict is broadcast before the
            # (FSDP-collective) state-dict gather. The payload is the same
            # `checkpoint_payload` the final save uses, so the best file is a
            # complete, self-contained inference artifact with the frozen VAE
            # embedded -- and the FM is the terminal stage, so nothing
            # downstream is coupled to which epoch it comes from.
            if best_modelpath and do_val:
                improved = 1.0 if (rank0 and valid_loss < best_valid_loss) else 0.0
                if D.is_dist():
                    improved = D.broadcast_scalar(improved, device)
                if improved > 0.5:
                    best_payload = checkpoint_payload(epoch)
                    if rank0:
                        best_valid_loss = valid_loss
                        save_checkpoint(best_modelpath, best_payload)
                        print(f'  [best] ValidFM {valid_loss:.4e} at epoch {epoch} -> {best_modelpath}')
                    del best_payload

            if epoch % test_interval == 0 or epoch == total_epochs - 1:
                if rank0:
                    run_generation_test(eval_model, vae, device, config, epoch,
                                        latent_flat_dim, latent_mean, latent_std,
                                        cond_dim=cond_dim, reference=generation_reference)
                maybe_save(epoch)
                D.barrier()

            if resume.due(epoch, total_epochs):
                resume.save({
                    'epoch': epoch, 'model': model.state_dict(),
                    'ema': ema_model.state_dict() if ema_model is not None else None,
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'loader_generator': (train_loader.generator.get_state()
                                         if train_loader.generator is not None else None),
                    'best_valid_loss': best_valid_loss, 'valid_loss': valid_loss,
                    'elapsed': time.time() - start_time,
                })

        maybe_save(total_epochs - 1)
        resume.finish()
        if rank0:
            print(f'\nTraining finished. FM saved to {modelpath} (val FM loss {valid_loss:.2e})')
            if best_modelpath:
                print(f'Best-validation FM (val FM loss {best_valid_loss:.2e}) at {best_modelpath}')
    except KeyboardInterrupt:
        if rank0:
            print('\nTraining interrupted by user. Saving checkpoint...'
                  + (f' Resume state: {resume.path}'
                     if resume.enabled and os.path.exists(resume.path) else ''))
        maybe_save(-1)


@torch.no_grad()
def _encode_split(vae, dataset, device, config, draws=1, draw_seed=0):
    """Encode every shape in a split to (latents, conditions).

    The dataset is switched to deterministic subsampling for the duration of
    the pass (and restored afterwards): `SDFShapeDataset.__getitem__` then seeds
    its rng from (split seed, shape index), so every rank -- and every run of
    the same checkpoint and seed -- caches bit-identical latents. Without this,
    each rank would normalize by its own private latent statistics.

    `draws` 1 caches the encoder mean. `draws` K > 1 caches K posterior samples
    per shape, draw-major (all shapes for draw 1, then draw 2, ...): draw k
    reads the subsample seeded from (split seed, shape index, k) and adds
    std * eps with eps from a CPU generator seeded from (draw_seed, k), where
    std is floored at `posterior_min_std_rel` x mu_spread exactly as in VAE
    training. Both seeds are rank-independent, so the cache stays identical
    across ranks.
    """
    latents, conds = [], []
    batch, batch_c = [], []
    batch_size = int(config.get('encode_batch_size', 16))
    min_std = vae._posterior_min_std(getattr(vae, 'posterior_min_std_rel', 0.0)) if draws > 1 else None
    generator = None

    def flush():
        if not batch:
            return
        pts = torch.stack([b[0] for b in batch]).to(device)
        nrm = torch.stack([b[1] for b in batch]).to(device)
        mu, logvar = vae.encode(pts, nrm)
        if generator is not None:
            std = torch.exp(0.5 * logvar)
            if min_std is not None:
                std = torch.maximum(std, min_std.to(dtype=std.dtype, device=std.device))
            eps = torch.randn(mu.shape, generator=generator).to(device=mu.device, dtype=mu.dtype)
            mu = mu + eps * std
        latents.append(mu.flatten(1).cpu())
        conds.extend(batch_c)
        batch.clear()
        batch_c.clear()

    previous_deterministic = getattr(dataset, 'deterministic', False)
    previous_draw = getattr(dataset, 'draw', 0)
    dataset.deterministic = True
    try:
        for k in range(1, draws + 1) if draws > 1 else (0,):
            if k:
                dataset.draw = k
                seed = int(np.random.SeedSequence([int(draw_seed), k]).generate_state(1)[0])
                generator = torch.Generator().manual_seed(seed)
            for i in range(len(dataset)):
                item = dataset[i]
                batch.append((item['surface_points'], item['surface_normals']))
                batch_c.append(item['cond'])
                if len(batch) == batch_size:
                    flush()
            flush()
    finally:
        dataset.deterministic = previous_deterministic
        dataset.draw = previous_draw
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


def _pick_generation_shapes(config, split, z_n, c_raw, c_n, shape_ids, cond_names, binary):
    """Evenly spaced shapes of one split for the periodic generation picture.

    Returns (normalized latents [n, D], normalized TRUE condition rows [n, C],
    panel labels). A label names the h5 shape and its active one-hot columns
    (`binary`: columns that hold only 0/1 in the training set), prefix dropped:
    cat_household -> household, class_locknuts -> locknuts. Above four shapes
    the count is rounded down to even, so the source/generated row pairs of the
    picture stay column-aligned.
    """
    n = min(int(config.get('num_test_shapes', 2)), 8, len(shape_ids))
    if n > 4:
        n -= n % 2
    positions = np.unique(np.linspace(0, len(shape_ids) - 1, n).round().astype(np.int64))
    labels = []
    for p in positions:
        tags = [cond_names[j].split('_', 1)[-1] for j in range(len(cond_names))
                if binary[j] and float(c_raw[p, j]) > 0.5]
        labels.append(f'{split} shape {int(shape_ids[p])}' + (f' ({", ".join(tags)})' if tags else ''))
    index = torch.as_tensor(positions)
    return z_n[index], c_n[index], labels


@torch.no_grad()
def run_generation_test(model, vae, device, config, epoch, latent_flat_dim,
                        latent_mean, latent_std, cond_dim=0, reference=None):
    """Periodic generation picture (and STLs) for the FM stage.

    A conditional FM samples from the TRUE conditions of a few real shapes
    (`reference` from `_pick_generation_shapes`), and each sample is drawn
    directly under the VAE reconstruction of the shape it took its conditions
    from, so the picture shows whether the generator follows its conditions.
    The all-zeros "mean condition" used before is a request no real shape
    makes -- with one-hot class columns it is a fractional member of every
    class at once -- and drew the same blob in every panel. An unconditional FM
    (or no reference) draws plain samples.
    """
    model.eval()
    out_dir = os.path.join(config.get('output_dir', './outputs'), 'fm_samples')
    os.makedirs(out_dir, exist_ok=True)
    resolution = int(config.get('mc_resolution_test', 96))
    latent_std, latent_mean = latent_std.to(device), latent_mean.to(device)

    if cond_dim > 0 and reference is not None:
        z_ref_n, cond, sample_labels = reference
        cond = cond.to(device)
        num_samples = len(cond)
    else:
        z_ref_n = cond = None
        num_samples = min(int(config.get('num_test_shapes', 2)), 8)
        sample_labels = [f'sample {i}' for i in range(num_samples)]
    # The same starting noise every epoch (drawn on the CPU so it does not
    # depend on the device): panels then change only because the network did,
    # and one epoch's picture can be compared with the next.
    noise = torch.randn(num_samples, latent_flat_dim,
                        generator=torch.Generator().manual_seed(int(config.get('seed', 0))))
    z_n = sample_latents(model, num_samples, latent_flat_dim, device, cond=cond,
                         ode_steps=int(config.get('ode_steps', 50)), noise=noise.to(device))
    z = z_n * latent_std + latent_mean

    def decode(z_row):
        mesh = sdf_grid_to_mesh(decode_sdf_grid(vae, z_row, resolution=resolution, device=device))
        report = mesh_report(mesh)
        return (mesh if report['valid'] else None), report

    # Collected for one figure: samples on shared axes show mode collapse
    # (every panel the same shape) at a glance, which per-shape STL files do not.
    source = ([f' [{label}]' for label in sample_labels] if z_ref_n is not None
              else [''] * num_samples)
    samples, sample_reports = [], []
    for i in range(num_samples):
        mesh, report = decode(z[i:i + 1])
        if mesh is not None:
            path = os.path.join(out_dir, f'epoch{epoch:05d}_sample{i}.stl')
            mesh.export(path)
            print(f'  [test] sample {i}{source[i]}: watertight={report["watertight"]} '
                  f'faces={report["faces"]} -> {path}')
        else:
            print(f'  [test] sample {i}{source[i]}: NO ZERO CROSSING')
        samples.append(mesh)
        sample_reports.append(report)

    if not (config.get('display_testset', True) and samples):
        return
    if z_ref_n is None:
        meshes, labels, reports, colors = samples, sample_labels, sample_reports, None
        ncols = 4
        title = f'SDFFlow samples -- epoch {epoch}'
    else:
        # Row pairs: the source shapes' reconstructions (blue), then the samples
        # generated from their conditions (orange) directly below them.
        z_ref = z_ref_n.to(device) * latent_std + latent_mean
        refs = [decode(z_ref[i:i + 1]) for i in range(num_samples)]
        ncols = num_samples if num_samples <= 4 else num_samples // 2
        meshes, labels, reports, colors = [], [], [], []
        for start in range(0, num_samples, ncols):
            chunk = range(start, start + ncols)
            for i in chunk:
                meshes.append(refs[i][0])
                reports.append(refs[i][1])
                labels.append(f'{sample_labels[i]}: reconstruction')
                colors.append(ENDPOINT_COLOR)
            for i in chunk:
                meshes.append(samples[i])
                reports.append(sample_reports[i])
                labels.append('generated from its conditions')
                colors.append(MIDDLE_COLOR)
        title = (f'SDFFlow samples from real shape conditions -- epoch {epoch} '
                 '(blue: source shape reconstruction, orange: generated)')
    written = plot_mesh_strip(
        meshes, labels, reports,
        os.path.join(out_dir, f'epoch{epoch:05d}_samples.png'),
        dpi=int(config.get('plot_dpi', 180)),
        max_faces=int(config.get('plot_max_faces', 0)),
        title=title, colors=colors, ncols=ncols,
    )
    if written:
        print(f'  [viz] {written}')
