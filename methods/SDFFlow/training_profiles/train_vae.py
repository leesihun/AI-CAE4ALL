"""Stage 1: SDF-VAE training (reconstruction + KL + optional hybrid losses).

Runs single-process or, under `parallel_mode` ddp/fsdp, as one rank of a spawned
distributed job. Gradients are shared across ranks; rank 0 owns validation,
logging, the periodic reconstruction test, and checkpoint writes.

Validation reports, besides the truncated-L1 SDF loss:

  ValidSign     sign accuracy over the val query points with |target| > 1e-3
  ValidSignBal  the same, balanced: the mean of the inside-rate and the
                outside-rate, so a decoder that only reproduces the majority
                class cannot look accurate (SDF queries are majority-outside)
  ActiveUnits   latent scalars whose encoder-mean variance across the val split
                exceeds `ACTIVE_UNIT_VAR_THRESHOLD` (Burda's absolute count)
  ActiveSNR     the scale-free form, `Var_x(mu_d) / mean_x(sigma_d^2) > 1` --
                compare arms on THIS one, since the absolute count moves with
                `kl_weight` and `latent_dim` for reasons unrelated to how many
                latent directions are actually used

With `vae_best_modelpath` set, the best-so-far validation model is additionally
checkpointed there; the final save to `vae_modelpath` is unchanged and remains
the pipeline's completeness signal.
"""

import inspect
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from general_modules import distributed as D
from general_modules.sdf_dataset import build_dataset_splits, compute_cond_stats
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE, describe_state_key_flag, load_vae_state_dict, sdf_loss
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

try:  # the model module owns the canonical implementation
    from model.sdf_vae import sign_accuracy
except ImportError:  # older model module: local fallback with the same contract
    def sign_accuracy(sdf_pred, sdf_target, eps=0.001):
        """Fraction of query points whose predicted SDF sign matches the target.

        Points within `eps` of the surface are excluded (their sign is not
        meaningful). Returns NaN when no point survives the mask.
        """
        mask = sdf_target.abs() > eps
        if int(mask.sum().item()) == 0:
            return float('nan')
        agree = torch.sign(sdf_pred[mask]) == torch.sign(sdf_target[mask])
        return float(agree.float().mean().item())


# A latent scalar counts as "active" when the variance of its encoder mean over
# the validation shapes exceeds this (Burda et al. 2015 convention).
#
# CAVEAT: that absolute threshold is only calibrated when the KL actually pins
# the latent to N(0, I). At the sum-KL weights this repo uses (2.5e-6) the
# latent scale is nearly free, so the count mostly measures |mu| -- an arm with
# a 10x KL weight reports fewer active units for the same number of used
# directions, and an arm with a different latent_dim uses a different
# denominator. `ActiveSNR` below is the scale-free version and is the one
# to compare ACROSS arms; the raw count is kept for continuity.
ACTIVE_UNIT_VAR_THRESHOLD = 0.01
# Scale-free active-unit test: a unit is active when the spread of its encoder
# mean across shapes exceeds the posterior width it is encoded with, i.e.
# Var_x(mu_d) / mean_x(exp(logvar_d)) > 1.
ACTIVE_UNIT_SNR_THRESHOLD = 1.0
SIGN_ACCURACY_EPS = 0.001


def _clip_grads(train_model, is_fsdp, params, max_norm=1.0):
    if is_fsdp:
        train_model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(params, max_norm)


def _forward_accepts(module, name):
    """True when `module.forward` takes a keyword argument called `name`."""
    try:
        return name in inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return False


def count_active_units(mu_all, threshold=ACTIVE_UNIT_VAR_THRESHOLD):
    """Active latent units from encoder means over a whole split.

    `mu_all` is `(num_shapes, D)` (tokens already flattened). Returns `(k, D)`
    with `k` the number of latent scalars whose population variance across the
    shapes exceeds `threshold`.
    """
    mu_all = mu_all.float()
    dim = int(mu_all.shape[1]) if mu_all.dim() == 2 else 0
    if mu_all.dim() != 2 or mu_all.shape[0] < 2:
        return 0, dim
    var = mu_all.var(dim=0, unbiased=False)
    return int((var > threshold).sum().item()), dim


def count_active_units_snr(mu_all, logvar_all, threshold=ACTIVE_UNIT_SNR_THRESHOLD):
    """Scale-free active units: Var_x(mu_d) / mean_x(sigma_d^2) > threshold.

    Both arguments are `(num_shapes, D)`. Unlike the absolute-variance count
    this is invariant to a rescaling of the latent, so it is comparable across
    arms with different `kl_weight` or `latent_dim`. Returns `(k, D)`.
    """
    mu_all = mu_all.float()
    dim = int(mu_all.shape[1]) if mu_all.dim() == 2 else 0
    if mu_all.dim() != 2 or mu_all.shape[0] < 2 or logvar_all is None:
        return 0, dim
    logvar_all = logvar_all.float()
    if logvar_all.shape != mu_all.shape:
        return 0, dim
    var_mu = mu_all.var(dim=0, unbiased=False)
    mean_post_var = logvar_all.exp().mean(dim=0).clamp_min(1e-12)
    return int((var_mu / mean_post_var > threshold).sum().item()), dim


def sign_class_counts(sdf_pred, sdf_target, eps=SIGN_ACCURACY_EPS):
    """Per-class sign agreement counts over the points with |target| > eps.

    Returns `(inside_correct, inside_total, outside_correct, outside_total)` as
    python ints so a caller can accumulate them across batches and only then
    form a balanced accuracy (a batch-mean of balanced accuracies is not the
    split's balanced accuracy). Inside is `target < -eps`, outside `> eps`.
    """
    pred = torch.as_tensor(sdf_pred).detach().float().flatten()
    target = torch.as_tensor(sdf_target).detach().float().flatten().to(pred.device)
    agree = torch.sign(pred) == torch.sign(target)
    inside = target < -float(eps)
    outside = target > float(eps)
    return (int((agree & inside).sum().item()), int(inside.sum().item()),
            int((agree & outside).sum().item()), int(outside.sum().item()))


def balanced_sign_accuracy(inside_correct, inside_total, outside_correct, outside_total):
    """Mean of the inside-rate and the outside-rate (class-balanced accuracy).

    An empty class is skipped rather than scored zero; with both empty the
    result is NaN. A majority-class predictor scores the class fraction on the
    raw accuracy but exactly 0.5 here, which is the point of reporting it.
    """
    rates = []
    if inside_total > 0:
        rates.append(inside_correct / inside_total)
    if outside_total > 0:
        rates.append(outside_correct / outside_total)
    if not rates:
        return float('nan')
    return float(sum(rates) / len(rates))


# Warm-start notes: which config key explains a tolerated state-dict difference,
# and in which direction. Keyed by (state key, 'missing' | 'unexpected'), where
# 'missing' means this model has the entry and the init checkpoint does not.
_INIT_FLAG_NOTES = {
    ('encoder.queries', 'unexpected'): (
        'init checkpoint was trained with encoder_query_type=learned; its '
        'encoder.queries parameter is dropped because this run uses fps'),
    ('encoder.queries', 'missing'): (
        'init checkpoint was trained with encoder_query_type=fps, which stores no '
        'encoder.queries parameter; this run uses learned queries and keeps them at '
        'their fresh initialization'),
    ('mu_spread', 'unexpected'): (
        'init checkpoint was trained with posterior_min_std_rel > 0; its mu_spread '
        'buffer is dropped because this run has the posterior std floor off'),
    ('mu_spread', 'missing'): (
        'init checkpoint was trained without a posterior std floor '
        "(posterior_min_std_rel 0), so this run's mu_spread buffer starts fresh"),
}


def describe_init_state_diff(missing, unexpected):
    """Rank-0 notes explaining each tolerated warm-start state-dict difference.

    Every note names the CONFIG KEY responsible, not just the tensor:
    `encoder.queries` and `mu_spread` exist only for one setting of
    `encoder_query_type` / `posterior_min_std_rel`.
    """
    notes = []
    for kind, keys in (('missing', missing), ('unexpected', unexpected)):
        for key in sorted(keys):
            note = _INIT_FLAG_NOTES.get((key, kind))
            if note is None:  # a flag-dependent key the model module added later
                note = f'{describe_state_key_flag([key])} ({kind} in the init checkpoint)'
            notes.append(note)
    return notes


def vae_worker(config, config_filename='config.txt'):
    device = resolve_device(config)
    split_seed = int(config.get('split_seed', 42))
    rank0 = D.is_main_process()
    world_size = D.get_world_size()
    # Optional global seeding (model init, shuffle order, posterior noise),
    # offset by the rank so no two ranks draw the same posterior noise; model
    # construction is put back on the rank-independent base seed below.
    run_seed = seed_stage(config, stage='VAE', offset=D.get_rank(), verbose=rank0)

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
            persistent_workers=num_workers > 0,
            generator=seeded_generator(run_seed))
    # Validation runs on rank 0 only (no distributed forward), on the raw model.
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            persistent_workers=num_workers > 0) if rank0 else None

    if rank0:
        print('\nInitializing model...')
    with identical_across_ranks(run_seed, D.get_rank()):
        model = SDFVAE(config).to(device)
    init_modelpath = config.get('init_vae_modelpath')
    if init_modelpath:
        checkpoint = load_checkpoint(init_modelpath, device)
        state = checkpoint.get('model_state', checkpoint)
        # Tolerant only for the two flag-dependent state keys (`mu_spread`,
        # `encoder.queries`): a warm start that flips posterior_min_std_rel or
        # encoder_query_type legitimately adds/removes one, and a strict load
        # would die with a bare torch error naming an internal buffer. Any other
        # difference is a real architecture mismatch, reported as what it is --
        # a config error, named by config key rather than by tensor.
        try:
            missing, unexpected = load_vae_state_dict(model, state, source=init_modelpath)
        except (ValueError, RuntimeError) as exc:
            raise ValueError(
                f'init_vae_modelpath ({init_modelpath}) was trained with a different VAE '
                f'architecture than this config asks for. Reconcile the architecture keys '
                f'with the ones stored in that checkpoint: latent_tokens, latent_dim, '
                f'decoder_type, encoder_query_type, num_encoder_points, hidden_dim, '
                f'num_heads, num_layers (and posterior_min_std_rel, which owns the '
                f'mu_spread buffer). Underlying state-dict error: {exc}') from exc
        if rank0:
            expected = len(model.state_dict())
            print(f'Initialized VAE weights from {init_modelpath} '
                  f'({expected - len(missing)}/{expected} tensors loaded)')
            for note in describe_init_state_diff(missing, unexpected):
                print(f'  NOTE: {note}')

    # Posterior std floor (relative to the running spread of mu). The model's
    # forward applies it; only pass the kwarg when this model version takes it
    # so a legacy model module keeps working with the default (off).
    posterior_min_std_rel = float(config.get('posterior_min_std_rel', 0.0))
    if posterior_min_std_rel < 0:
        raise ValueError(f'posterior_min_std_rel must be >= 0, got {posterior_min_std_rel}')
    forward_kwargs = {}
    if _forward_accepts(model, 'posterior_min_std_rel'):
        forward_kwargs['posterior_min_std_rel'] = posterior_min_std_rel
    elif posterior_min_std_rel > 0:
        raise ValueError('posterior_min_std_rel > 0 requires model.sdf_vae.SDFVAE.forward '
                         'to accept the posterior_min_std_rel keyword; this model module '
                         'does not.')
    if posterior_min_std_rel > 0 and rank0:
        print(f'Posterior std floor enabled: posterior_min_std_rel={posterior_min_std_rel:g}')

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
    if rank0:
        ema_warning = ema_horizon_warning(ema_config, len(train_loader), total_epochs, stage='VAE')
        if ema_warning:
            print(ema_warning)

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
    modelpath = config.get('vae_modelpath', '../../output/geometry_generation/sdfflow_vae.pth')
    # Optional best-validation checkpoint (rank 0 writes; the decision is
    # broadcast so the FSDP state-dict gather below stays collective).
    best_modelpath = config.get('vae_best_modelpath')
    best_valid_loss = float('inf')

    log_file = init_log_file(config, config_filename) if rank0 else None
    if rank0:
        print('\n' + '=' * 60)
        print('Starting SDF-VAE training loop...')
        print('=' * 60 + '\n')
    start_time = time.time()
    valid_loss = float('nan')
    valid_sign = float('nan')
    valid_sign_balanced = float('nan')
    active_units, active_units_snr, latent_dim_total = 0, 0, int(model.latent_flat_dim)
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
            surface_sum, normal_sum, eikonal_sum = 0.0, 0.0, 0.0
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
                        eikonal_weight=eikonal_weight, hybrid_grad_points=hybrid_grad_points,
                        **forward_kwargs)
                    recon, kl = losses['recon'], losses['kl']
                    hybrid_l = (surface_weight * losses['surface']
                                + normal_weight * losses['normal']
                                + eikonal_weight * losses['eikonal'])
                    loss = recon + effective_kl_weight * kl + hybrid_l
                    loss.backward()
                    _clip_grads(train_model, is_fsdp, params, 1.0)
                    optimizer.step()
                    hybrid_sum += float(hybrid_l.item())
                    # Raw (unweighted) per-term means, so the log shows what each
                    # geometry term is doing independently of its weight.
                    surface_sum += float(losses['surface'].item())
                    normal_sum += float(losses['normal'].item())
                    eikonal_sum += float(losses['eikonal'].item())
                else:
                    with torch.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled):
                        losses = train_model(
                            surface_points, surface_normals, query_points, query_sdf,
                            posterior_noise_scale=posterior_noise_scale, clamp_dist=clamp_dist,
                            **forward_kwargs)
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
            hybrid_str, hybrid_log = '', ''
            if use_hybrid:
                # Collectives: every rank calls these (use_hybrid is config-wide).
                train_surface = D.reduce_epoch_mean(surface_sum, batches, device)
                train_normal = D.reduce_epoch_mean(normal_sum, batches, device)
                train_eikonal = D.reduce_epoch_mean(eikonal_sum, batches, device)
                hybrid_str = (f' Hybrid: {train_hybrid:.2e} Surface: {train_surface:.2e} '
                              f'Normal: {train_normal:.2e} Eikonal: {train_eikonal:.2e}')
                hybrid_log = (f'Hybrid {train_hybrid:.4e} Surface: {train_surface:.4e} '
                              f'Normal: {train_normal:.4e} Eikonal: {train_eikonal:.4e} ')
            current_lr = optimizer.param_groups[0]['lr']

            do_val = (epoch % val_interval == 0) or (epoch == total_epochs - 1)
            eval_model = D.unwrap_model(ema_model) if ema_model is not None else model
            if do_val and rank0:
                val_stats = _validate(eval_model, val_loader, device, clamp_dist)
                valid_loss = val_stats['sdf']
                valid_sign = val_stats['sign']
                valid_sign_balanced = val_stats['sign_balanced']
                active_units, latent_dim_total = val_stats['active_units'], val_stats['latent_dim']
                active_units_snr = val_stats['active_units_snr']
                print(f'Epoch {epoch}/{total_epochs} TrainSDF: {train_loss:.2e} '
                      f'KL: {train_kl:.2e}{hybrid_str} ValidSDF: {valid_loss:.2e} '
                      f'ValidSign: {valid_sign:.4f} ValidSignBal: {valid_sign_balanced:.4f} '
                      f'ActiveUnits: {active_units}/{latent_dim_total} '
                      f'ActiveSNR: {active_units_snr}/{latent_dim_total} '
                      f'LR: {current_lr:.2e} '
                      f'KLWeight: {effective_kl_weight:.2e} PosteriorNoise: {posterior_noise_scale:.2f}')
            elif rank0:
                print(f'Epoch {epoch}/{total_epochs} TrainSDF: {train_loss:.2e} '
                      f'KL: {train_kl:.2e}{hybrid_str} LR: {current_lr:.2e} '
                      f'KLWeight: {effective_kl_weight:.2e} PosteriorNoise: {posterior_noise_scale:.2f}')

            if rank0:
                elapsed = time.time() - start_time
                val_str = (f'Valid {valid_loss:.4e} ValidSign: {valid_sign:.4f} '
                           f'ValidSignBal: {valid_sign_balanced:.4f} '
                           f'ActiveUnits: {active_units}/{latent_dim_total} '
                           f'ActiveSNR: {active_units_snr}/{latent_dim_total}'
                           if do_val else 'Valid skipped')
                append_log(log_file, f'Elapsed: {elapsed:.2f}s Epoch {epoch} '
                                     f'TrainSDF {train_loss:.4e} KL {train_kl:.4e} {hybrid_log}{val_str} '
                                     f'LR: {current_lr:.4e} KLWeight: {effective_kl_weight:.4e} '
                                     f'PosteriorNoise: {posterior_noise_scale:.4f}')

            # Best-validation checkpoint. `do_val` is identical on every rank;
            # only rank 0 knows the loss, so its verdict is broadcast before the
            # (FSDP-collective) state-dict gather.
            if best_modelpath and do_val:
                improved = 1.0 if (rank0 and valid_loss < best_valid_loss) else 0.0
                if D.is_dist():
                    improved = D.broadcast_scalar(improved, device)
                if improved > 0.5:
                    best_payload = checkpoint_payload(epoch)
                    if rank0:
                        best_valid_loss = valid_loss
                        save_checkpoint(best_modelpath, best_payload)
                        print(f'  [best] ValidSDF {valid_loss:.4e} at epoch {epoch} -> {best_modelpath}')
                    del best_payload

            if epoch % test_interval == 0 or epoch == total_epochs - 1:
                if rank0:
                    run_reconstruction_test(eval_model, test_dataset, device, config, epoch)
                maybe_save(epoch)
                D.barrier()

        maybe_save(total_epochs - 1)
        if rank0:
            print(f'\nTraining finished. VAE saved to {modelpath} (val SDF loss {valid_loss:.2e})')
            if best_modelpath:
                print(f'Best-validation VAE (val SDF loss {best_valid_loss:.2e}) at {best_modelpath}')
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
def _validate(model, val_loader, device, clamp_dist, sign_eps=SIGN_ACCURACY_EPS):
    """Validation pass on the encoder mean (no posterior noise).

    Returns a dict:
      sdf          mean truncated-L1 SDF loss over batches (the historical metric)
      sign         sign accuracy over every val query point with |target| > sign_eps
      sign_balanced the same, class-balanced: mean of the inside-rate and the
                   outside-rate over split-wide counts (SDF queries are
                   majority-outside, so the raw number has a high floor)
      active_units number of latent scalars with var(mu) > ACTIVE_UNIT_VAR_THRESHOLD
      active_units_snr scale-free count: var(mu) / mean(sigma^2) > ACTIVE_UNIT_SNR_THRESHOLD
      latent_dim   D = latent_tokens * latent_dim
    """
    model.eval()
    total, batches = 0.0, 0
    sign_weighted, sign_count = 0.0, 0
    in_correct, in_total, out_correct, out_total = 0, 0, 0, 0
    mu_chunks, logvar_chunks = [], []
    for batch in val_loader:
        surface_points = batch['surface_points'].to(device)
        surface_normals = batch['surface_normals'].to(device)
        query_points = batch['query_points'].to(device)
        query_sdf = batch['query_sdf'].to(device)
        mu, logvar = model.encode(surface_points, surface_normals)
        sdf_pred = model.decode(mu, query_points).float()
        total += sdf_loss(sdf_pred, query_sdf, clamp_dist).item()
        batches += 1
        # Weight each batch's accuracy by its number of sign-defined points so
        # the aggregate equals the accuracy over the whole split.
        n_valid = int((query_sdf.abs() > sign_eps).sum().item())
        if n_valid > 0:
            sign_weighted += float(sign_accuracy(sdf_pred, query_sdf, eps=sign_eps)) * n_valid
            sign_count += n_valid
        counts = sign_class_counts(sdf_pred, query_sdf, eps=sign_eps)
        in_correct += counts[0]
        in_total += counts[1]
        out_correct += counts[2]
        out_total += counts[3]
        mu_chunks.append(mu.flatten(1).float().cpu())
        logvar_chunks.append(logvar.flatten(1).float().cpu())
    latent_dim = int(getattr(model, 'latent_flat_dim', mu_chunks[0].shape[1] if mu_chunks else 0))
    mu_all = torch.cat(mu_chunks, dim=0) if mu_chunks else torch.zeros(0, latent_dim)
    logvar_all = torch.cat(logvar_chunks, dim=0) if logvar_chunks else torch.zeros(0, latent_dim)
    active_units, latent_dim = count_active_units(mu_all)
    active_units_snr, _ = count_active_units_snr(mu_all, logvar_all)
    return {
        'sdf': total / max(batches, 1),
        'sign': (sign_weighted / sign_count) if sign_count > 0 else float('nan'),
        'sign_balanced': balanced_sign_accuracy(in_correct, in_total, out_correct, out_total),
        'active_units': active_units,
        'active_units_snr': active_units_snr,
        'latent_dim': latent_dim,
    }


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
