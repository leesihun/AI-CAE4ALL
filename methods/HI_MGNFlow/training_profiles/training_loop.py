"""Training and evaluation for cHI-MGNflow, an LDGN-style two-stage model.

Stage 1 (AE): a near-unregularized multiscale autoencoder compresses the field
onto a coarse mesh level. Objective:

    loss = MSE(y_hat, y) + ae_kl_weight * KL(q(z|y) || N(0, I))

with ae_kl_weight tiny (LDGN: ~1e-6) -- this is a compressor, not a generative
bottleneck by itself; KL only keeps the latent well-scaled for stage 2.

Stage 2 (prior): a conditional flow-matching prior is trained ON FROZEN
LATENTS drawn from stage 1's posterior, entirely within the small coarse
latent space:

    loss = MSE(v_theta(z_t, t, ctx), z1 - s*z0),   z1 ~ q(z|y) from the frozen AE

(t, z0) freshly drawn every step; there is no reconstruction term here and no
gradient into the AE (see prior_loss / train_prior_epoch). This is the exact
field-space objective from the old flow.py, transplanted onto the coarse
latent -- sample_path and loss_weight do not care what tensor they are handed.

Evaluation mirrors this split:

  * `validate_ae_epoch` scores stage 1's reconstruction+KL on held-out graphs.
  * `validate_prior_epoch` scores stage 2's one-step regression on frozen
    latents. Cheap, low-variance, runs every epoch.
  * `evaluate_prior_sampling_epoch` integrates the latent ODE, decodes, and
    scores the resulting FIELD ensemble (CRPS + spread/skill + a deterministic
    read-out). This is the metric that mirrors inference and is what
    `best_by crps` selects on -- and because the ODE now runs over the coarse
    mesh instead of the fine one, it is far cheaper per member than the old
    field-space version.
"""
import os
import contextlib
import time

import tqdm
import torch
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from training_profiles.amp import build_grad_scaler, resolve_amp_dtype
from general_modules.mesh_utils_fast import (
    save_inference_results_fast,
    render_plot_data,
    edges_to_triangles_gpu,
    edges_to_triangles_optimized,
)
from model.flow import loss_weight, resolve_flow_config, sample_path


# ── small helpers ───────────────────────────────────────────────────────────

def _unwrap(model):
    """Peel DDP / AveragedModel / torch.compile wrappers off the real module."""
    m = model
    for attr in ('module', '_orig_mod'):
        inner = getattr(m, attr, None)
        if inner is not None:
            m = inner
    return m


def build_ema_model(model, config):
    if not config.get('use_ema', False):
        return None
    decay = float(config.get('ema_decay', 0.999))
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=decay))
    for p in ema_model.parameters():
        p.requires_grad_(False)
    return ema_model


def _build_loss_weights(config, device):
    """Per-feature weights normalized to sum to 1 (a weighted mean).

    Physical-feature weights only -- they size to `output_var` and apply to
    the AE's field-space reconstruction term. The prior's loss lives in latent
    space, whose channels carry no such per-feature meaning, so prior_loss
    never takes this argument.
    """
    w = config.get('feature_loss_weights', None)
    if w is None:
        return None
    if not isinstance(w, list):
        w = [w]
    w = torch.tensor(w, dtype=torch.float32, device=device)
    return w / w.sum()


def _weighted_mse(pred, target, loss_weights):
    err = (pred.float() - target.float()).pow(2)
    if loss_weights is not None:
        return torch.sum(err * loss_weights, dim=-1).mean()
    return err.mean()


def _num_graphs(graph):
    batch = getattr(graph, 'batch', None)
    if batch is None:
        return 1
    return int(batch.max().item()) + 1


def _mem_gb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.max_memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9)


def _mem_str():
    peak, reserved = _mem_gb()
    return f'{peak:.1f}/{reserved:.1f}GB'


def _move(graph, device, config):
    non_blocking = bool(config.get('_pin_memory', False)) and getattr(device, 'type', None) == 'cuda'
    return graph.to(device, non_blocking=non_blocking)


def _accum_window_size(batch_idx, total_batches, actual_accum):
    start = (batch_idx // actual_accum) * actual_accum
    return min(start + actual_accum, total_batches) - start


def _delta_stats(dataloader, device):
    ds = getattr(dataloader, 'dataset', None)
    for obj in (ds, getattr(ds, 'dataset', None)):
        dm = getattr(obj, 'delta_mean', None)
        st = getattr(obj, 'delta_std', None)
        if dm is not None and st is not None:
            return (torch.as_tensor(dm, dtype=torch.float32, device=device),
                    torch.as_tensor(st, dtype=torch.float32, device=device))
    return None, None


def _sample_posterior_latent(model, graph):
    """Draw z1 ~ q(z|y) from the (frozen, eval-mode) compressor.

    No-grad: stage 2 never backpropagates into the AE (see freeze_ae /
    CHiMGNFlow.forward's DDP-safe dispatch) -- this is data preparation for the
    prior's training pair, not part of the computation the optimizer steps on.
    """
    with torch.no_grad():
        mu, logvar, ctx = _unwrap(model).encode_both_auto(graph)
        std = torch.exp(0.5 * logvar)
        z1 = mu + std * torch.randn_like(std)
    return z1, ctx


# ── stage 1: the compressor objective ───────────────────────────────────────

def ae_loss(model, graph, loss_weights, use_amp, amp_dtype, kl_weight):
    """Reconstruction + KL for the compressor stage. One call to model('ae', ...)
    runs both the y-aware posterior pass and the y-blind decoder-skip pass
    (see LatentDiffusionGraphNet.encode_both) inside a single autocast block.
    """
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        y_hat, mu, logvar = model('ae', graph)
    recon = _weighted_mse(y_hat, graph.y, loss_weights)
    mu_f, logvar_f = mu.float(), logvar.float()
    # Per-element KL to N(0,1), mean (not summed) over coarse-nodes x channels
    # so kl_weight's useful range does not move with latent_ch or mesh size.
    kl = (-0.5 * (1 + logvar_f - mu_f.pow(2) - logvar_f.exp())).mean()
    return recon + kl_weight * kl, recon.detach(), kl.detach()


def train_ae_epoch(model, dataloader, optimizer, device, config, epoch, ema_model=None):
    model.train()

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = resolve_amp_dtype(device)
    scaler = build_grad_scaler(amp_dtype, use_amp)
    kl_weight = float(config.get('ae_kl_weight', 1e-6))

    grad_accum_steps = config.get('grad_accum_steps', 1)
    total_batches = len(dataloader)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps

    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
    total_recon_gpu = torch.zeros((), device=device, dtype=torch.float32)
    total_kl_gpu = torch.zeros((), device=device, dtype=torch.float32)
    n_batches = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm.tqdm(dataloader, total=total_batches)
    for batch_idx, graph in enumerate(pbar):
        graph = _move(graph, device, config)
        loss, recon, kl = ae_loss(model, graph, loss_weights, use_amp, amp_dtype, kl_weight)

        scaled = loss / _accum_window_size(batch_idx, total_batches, actual_accum)
        is_step = (batch_idx + 1) % actual_accum == 0 or (batch_idx == total_batches - 1)
        sync_ctx = (contextlib.nullcontext() if (is_step or not hasattr(model, 'no_sync'))
                    else model.no_sync())
        with sync_ctx:
            scaler.scale(scaled).backward()

        # GPU accumulators: .item() deferred to epoch end so the loop never syncs per batch.
        total_loss_gpu += loss.detach().float()
        total_recon_gpu += recon
        total_kl_gpu += kl
        n_batches += 1

        if is_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            pbar.set_postfix({'recon': f'{float(recon):.2e}', 'kl': f'{float(kl):.2e}',
                              'mem': _mem_str()})

    peak_gb, reserved_gb = _mem_gb()
    n_batches = max(n_batches, 1)
    mean = total_loss_gpu.item() / n_batches
    recon_mean = total_recon_gpu.item() / n_batches
    kl_mean = total_kl_gpu.item() / n_batches
    return {'mean': mean, 'total_mean': mean, 'sum': mean * n_batches,
            'count': n_batches, 'recon': recon_mean, 'kl': kl_mean,
            'peak_gb': peak_gb, 'reserved_gb': reserved_gb}


def validate_ae_epoch(model, dataloader, device, config, epoch=0):
    """Reconstruction + KL on held-out graphs.

    `mean` reports pure reconstruction error (what the compressor is FOR);
    the composite recon+kl_weight*kl objective is available as 'total_mean'.
    """
    model.eval()
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = resolve_amp_dtype(device)
    kl_weight = float(config.get('ae_kl_weight', 1e-6))

    total = torch.zeros((), device=device, dtype=torch.float32)
    total_recon = torch.zeros((), device=device, dtype=torch.float32)
    total_kl = torch.zeros((), device=device, dtype=torch.float32)
    n = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc='Validation')
        for i, graph in enumerate(pbar):
            graph = _move(graph, device, config)
            loss, recon, kl = ae_loss(model, graph, loss_weights, use_amp, amp_dtype, kl_weight)
            total += loss.detach().float()
            total_recon += recon
            total_kl += kl
            n += 1
            if i % 10 == 0:
                pbar.set_postfix({'recon': f'{total_recon.item() / max(n, 1):.2e}'})
    n = max(n, 1)
    recon_mean = total_recon.item() / n
    return {'mean': recon_mean, 'total_mean': total.item() / n, 'sum': total.item(),
            'count': n, 'recon': recon_mean, 'kl': total_kl.item() / n}


# ── stage 2: the latent flow prior objective ────────────────────────────────

def prior_loss(model, ctx, z1, coarse_batch, use_amp, amp_dtype, flow_cfg):
    """One flow-matching training term, transplanted onto the coarse latent.

    Identical in structure to the old field-space flow_loss: y1 is now a
    posterior draw z1 ~ q(z|y) from the frozen AE, and `batch` is the coarse
    mesh's graph index -- sample_path / loss_weight are agnostic to what
    tensor they are handed. No feature_loss_weights here: latent channels
    carry no physical per-feature meaning.
    """
    cfg = flow_cfg or {}
    B = int(coarse_batch.max().item()) + 1 if coarse_batch.numel() else 1
    t, z_t, u = sample_path(z1, coarse_batch, B,
                            cfg.get('t_sampling', 'uniform'),
                            cfg.get('logit_scale', 1.0),
                            cfg.get('det_prob', 0.0))
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        v = model('prior', ctx, z_t, t)

    per_node = (v.float() - u.float()).pow(2).mean(dim=-1)
    w = loss_weight(t.float(), cfg.get('weighting', 'uniform'))
    if w is not None:
        w_n = w[coarse_batch].squeeze(-1)
        return (per_node * w_n).sum() / w_n.sum().clamp_min(1e-8)
    return per_node.mean()


def train_prior_epoch(model, dataloader, optimizer, device, config, epoch, ema_model=None):
    model.train()

    use_amp = config.get('use_amp', True)
    amp_dtype = resolve_amp_dtype(device)
    scaler = build_grad_scaler(amp_dtype, use_amp)
    flow_cfg = resolve_flow_config(config)

    grad_accum_steps = config.get('grad_accum_steps', 1)
    total_batches = len(dataloader)
    actual_accum = total_batches if grad_accum_steps == 0 else grad_accum_steps

    total_loss_gpu = torch.zeros((), device=device, dtype=torch.float32)
    n_batches = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm.tqdm(dataloader, total=total_batches)
    for batch_idx, graph in enumerate(pbar):
        graph = _move(graph, device, config)
        z1, ctx = _sample_posterior_latent(model, graph)
        loss = prior_loss(model, ctx, z1, ctx['coarse_batch'], use_amp, amp_dtype, flow_cfg)

        scaled = loss / _accum_window_size(batch_idx, total_batches, actual_accum)
        is_step = (batch_idx + 1) % actual_accum == 0 or (batch_idx == total_batches - 1)
        sync_ctx = (contextlib.nullcontext() if (is_step or not hasattr(model, 'no_sync'))
                    else model.no_sync())
        with sync_ctx:
            scaler.scale(scaled).backward()

        total_loss_gpu += loss.detach().float()
        n_batches += 1

        if is_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_model.update_parameters(model)
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            pbar.set_postfix({'fm': f'{float(loss):.2e}', 'mem': _mem_str()})

    peak_gb, reserved_gb = _mem_gb()
    n_batches = max(n_batches, 1)
    mean = total_loss_gpu.item() / n_batches
    return {'mean': mean, 'total_mean': mean, 'sum': mean * n_batches,
            'count': n_batches, 'peak_gb': peak_gb, 'reserved_gb': reserved_gb}


def validate_prior_epoch(model, dataloader, device, config, epoch=0):
    """One-step flow-matching regression loss on frozen latents from held-out graphs.

    Cheap and low-variance, but it measures the velocity field, not the
    decoded samples. Use `evaluate_prior_sampling_epoch` for sample quality.
    """
    model.eval()
    use_amp = config.get('use_amp', True)
    amp_dtype = resolve_amp_dtype(device)
    flow_cfg = resolve_flow_config(config)

    total = torch.zeros((), device=device, dtype=torch.float32)
    n = 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc='Validation')
        for i, graph in enumerate(pbar):
            graph = _move(graph, device, config)
            z1, ctx = _sample_posterior_latent(model, graph)
            total += prior_loss(model, ctx, z1, ctx['coarse_batch'], use_amp, amp_dtype,
                                flow_cfg).detach().float()
            n += 1
            if i % 10 == 0:
                pbar.set_postfix({'fm': f'{total.item() / max(n, 1):.2e}'})
    mean = total.item() / max(n, 1)
    return {'mean': mean, 'total_mean': mean, 'sum': mean * n, 'count': n}


# ── sampling (decoded fields) ────────────────────────────────────────────────

@torch.no_grad()
def _generate_fields(model, graph, flow_cfg, num_samples, use_amp, amp_dtype):
    """Draw `num_samples` DECODED fields for one batch of graphs.

    Each draw integrates the prior's ODE over the small coarse latent, then
    decodes -- far cheaper per member than the old field-space integration.
    generate() itself refuses flow_cfg['predict']=='ensemble_mean' (that mode
    means averaging DECODED fields over several calls, which is what the S
    loop here already does), so callers that want an ensemble mean pass
    'sample' or 'mean' and average `_generate_fields`'s own output instead.
    """
    inner = _unwrap(model)
    out = []
    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        for _ in range(int(num_samples)):
            out.append(inner.generate(graph, flow_cfg))
    return torch.stack(out, dim=0)


def _crps(samples, target, loss_weights):
    """Fair (unbiased) per-element CRPS, reduced to a scalar.

    CRPS ~ mean_s |x_s - y| - 1/(2 S (S-1)) sum_s sum_j |x_s - x_j|

    Accumulated pair-by-pair: the vectorized [S,S,N,F] form is the largest
    single allocation in a validation pass on 50k-node meshes.
    """
    S = samples.shape[0]
    acc = torch.zeros_like(target)
    for i in range(S):
        acc += (samples[i] - target).abs()
    acc /= S
    if S >= 2:
        spread = torch.zeros_like(acc)
        for i in range(S - 1):
            for j in range(i + 1, S):
                spread += (samples[i] - samples[j]).abs()
        acc = acc - spread / (S * (S - 1))
    if loss_weights is not None:
        return torch.sum(acc * loss_weights, dim=-1).mean()
    return acc.mean()


def evaluate_prior_sampling_epoch(model, dataloader, device, config, epoch=0,
                                  progress_name='ValidationSample'):
    """Integrate the latent ODE, decode, and score the resulting field ensemble.

    Reports:
        crps       fair CRPS of the decoded ensemble against the ground truth.
                   The estimator is unbiased at any S; S buys variance only,
                   and the noise floor is set by the number of validation
                   GRAPHS, not by S.
        recon      error of member 0 -- a single draw, NOT a reconstruction.
                   Expect it to be worse than the AE's own recon (stage 1):
                   a calibrated ensemble member is not supposed to sit on the
                   mean, and it also carries the AE's own reconstruction error.
        spread     mean over graphs of std over members, in units of the
                   target std. Near 0 means the noise channel is being ignored.
    """
    model.eval()
    flow_cfg = resolve_flow_config(config)
    flow_cfg['steps'] = int(config.get('val_flow_steps', flow_cfg['steps']))
    if flow_cfg['predict'] == 'ensemble_mean':
        flow_cfg['predict'] = 'sample'
    S = int(config.get('val_num_samples', 8))

    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = resolve_amp_dtype(device)

    crps_sum, recon_sum, spread_sum, det_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc=progress_name)
        for graph in pbar:
            graph = _move(graph, device, config)
            samples = _generate_fields(model, graph, flow_cfg, S, use_amp, amp_dtype)
            target = graph.y.float()

            crps_sum += float(_crps(samples, target, loss_weights))
            recon_sum += float(_weighted_mse(samples[0], target, loss_weights))
            # The deterministic readout, one prior forward at t=0 (predict_mean):
            # tracked every validation so det quality is visible DURING training.
            mean_cfg = dict(flow_cfg)
            mean_cfg['predict'] = 'mean'
            predicted = _generate_fields(model, graph, mean_cfg, 1, use_amp, amp_dtype)[0]
            det_sum += float(_weighted_mse(predicted, target, loss_weights))
            if S >= 2:
                gt_std = target.std().clamp_min(1e-8)
                spread_sum += float(samples.std(dim=0).mean() / gt_std)
            n += 1
            pbar.set_postfix({'crps': f'{crps_sum / max(n, 1):.3e}', 'mem': _mem_str()})

    n = max(n, 1)
    result = {'crps': crps_sum / n, 'mean': recon_sum / n,
              'sum': recon_sum, 'count': n, 'spread': spread_sum / n,
              'det': det_sum / n}
    tqdm.tqdm.write(
        f"  [FlowDiag] crps={result['crps']:.4e}  det(1fwd) mse={result['det']:.4e}  "
        f"1-draw mse={result['mean']:.4e}  spread/gt={result['spread']:.3f}  "
        f"(steps={flow_cfg['steps']}, S={S})")
    if result['spread'] < 0.02:
        tqdm.tqdm.write("  [FlowDiag] WARNING: ensemble spread is ~0 -- the model is "
                        "ignoring the noise channel.")
    return result


def log_training_config(config):
    mode = str(config.get('mode', 'train')).lower().strip()
    lw = config.get('feature_loss_weights', None)
    if lw is not None:
        if not isinstance(lw, list):
            lw = [lw]
        w = torch.tensor(lw, dtype=torch.float32)
        print(f"Per-feature loss weights (raw):        {lw}")
        print(f"Per-feature loss weights (normalized): {[f'{v:.4f}' for v in (w / w.sum()).tolist()]}")
    else:
        print("Per-feature loss weights: equal (default)")

    L = int(config.get('multiscale_levels', 1))
    mp = config.get('mp_per_level', [])
    mp = [int(mp)] if not isinstance(mp, list) else [int(x) for x in mp]
    print(f"Compressor V-cycle: {L} levels, {sum(mp)} GnBlocks total")
    for i in range(L):
        print(f"  Level {i} pre:  {mp[i]} blocks")
    print(f"  Coarsest:    {mp[L]} blocks")
    for i in range(L - 1, -1, -1):
        print(f"  Level {i} post: {mp[2 * L - i]} blocks")
    print(f"  Coarse latent: {config.get('latent_ch', 4)} channels/coarse-node "
          f"(norm_latents, ae_kl_weight={float(config.get('ae_kl_weight', 1e-6)):g})")

    if mode in ('train_ae', 'train'):
        print("Stage 1 (compressor): reconstruction + KL, near-unregularized")

    if mode in ('train_prior', 'train'):
        flow_cfg = resolve_flow_config(config)
        print(f"Stage 2 (prior): {config.get('prior_blocks', 4)} AdaLN-Zero GnBlocks, "
              f"conditional flow matching on FROZEN coarse latents")
        print(f"  parameterization:      {flow_cfg['weighting']}"
              + ("  (velocity prediction)" if flow_cfg['weighting'] == 'uniform'
                 else "  (data prediction -- (1-s*t)^2 weight, favours the deterministic end)"))
        if flow_cfg['det_prob'] > 0:
            print(f"  deterministic slice:   {flow_cfg['det_prob']:.0%} of graphs pinned to t=0 "
                  f"(pure regression on E[z|g])")
        print(f"  inference integration: {flow_cfg['steps']} steps, {flow_cfg['solver']} solver")
        print(f"  validation sampling:   {config.get('val_flow_steps', flow_cfg['steps'])} steps, "
              f"{config.get('val_num_samples', 8)} members")
        print(f"  time embedding:        Fourier({flow_cfg['time_freqs']}) -> {2 * flow_cfg['time_freqs']} dims")
        print(f"  t schedule:            {flow_cfg['t_sampling']}"
              + (f" (scale={flow_cfg['logit_scale']:g})"
                 if flow_cfg['t_sampling'] == 'logitnormal' else ""))
    if mode == 'train':
        print(f"  Combined run: {config.get('ae_epochs', 0)} AE epochs, then "
              f"{config.get('training_epochs', 0)} prior epochs on the frozen AE")


# ── periodic visual test ────────────────────────────────────────────────────

def run_periodic_test(model, test_loader, device, config, epoch, train_dataset):
    start = time.time()
    test_loss = test_model(model, test_loader, device, config, epoch, train_dataset)
    print(f"  Test loss: {test_loss:.2e} ({time.time() - start:.1f}s)")

    if config.get('display_trainset', True):
        viz_indices = config.get('test_batch_idx', [0, 1, 2, 3])
        if not isinstance(viz_indices, list):
            viz_indices = [viz_indices]
        viz_indices = [i for i in viz_indices if i < len(train_dataset)]
        if viz_indices:
            viz_loader = DataLoader(
                Subset(train_dataset, viz_indices), batch_size=1, shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )
            viz_config = dict(config)
            viz_config['test_batch_idx'] = list(range(len(viz_indices)))
            viz_loss = test_model(model, viz_loader, device, viz_config, epoch,
                                  train_dataset, output_prefix='train')
            print(f"  Train-split sample loss: {viz_loss:.2e}")
    return test_loss


def _scalar_attr(graph, name):
    value = getattr(graph, name, None)
    if value is None:
        return None
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'item'):
        return value.item()
    if hasattr(value, '__getitem__') and len(value) > 0:
        return int(value[0])
    return int(value)


def test_model(model, dataloader, device, config, epoch, dataset=None, output_prefix='test'):
    """Draw ONE prediction per test graph and write it out for visual inspection.

    In 'train_ae' mode this is the AE's reconstruction (posterior mean sample,
    no prior exists yet). Otherwise it is a single draw from the trained prior
    through the frozen decoder -- a different call produces a different field,
    which is the point: the pictures are for judging whether individual
    samples look like physical fields.
    """
    model.eval()
    loss_weights = _build_loss_weights(config, device)
    use_amp = config.get('use_amp', True)
    amp_dtype = resolve_amp_dtype(device)
    mode = str(config.get('mode', 'train')).lower().strip()
    is_ae_only = mode == 'train_ae'
    flow_cfg = None
    if not is_ae_only:
        flow_cfg = resolve_flow_config(config)
        flow_cfg['steps'] = int(config.get('val_flow_steps', flow_cfg['steps']))

    use_gpu = device.type == 'cuda' if hasattr(device, 'type') else (device != 'cpu')
    mesh_device = device if use_gpu else 'cpu'
    faces_cache = {}

    total_test = len(dataloader)
    max_test_batches = int(config.get('test_max_batches', 200))
    effective_total = min(max_test_batches, total_test)

    delta_mean = getattr(dataset, 'delta_mean', None) if dataset is not None else None
    delta_std = getattr(dataset, 'delta_std', None) if dataset is not None else None

    test_idx = config.get('test_batch_idx', [0, 1, 2, 3])
    if not isinstance(test_idx, list):
        test_idx = [test_idx]

    total_loss, n = 0.0, 0
    plot_data_queue = []

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, total=effective_total)
        for batch_idx, graph in enumerate(pbar):
            if batch_idx >= max_test_batches:
                break
            graph = _move(graph, device, config)
            if is_ae_only:
                with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                    y_hat, _, _ = model('ae', graph)
                predicted = y_hat.float()
            else:
                predicted = _generate_fields(model, graph, flow_cfg, 1, use_amp, amp_dtype)[0]
            target = graph.y.float()
            loss = _weighted_mse(predicted, target, loss_weights)
            total_loss += float(loss)
            n += 1
            pbar.set_postfix({'loss': f'{float(loss):.2e}', 'mem': _mem_str()})

            if batch_idx in test_idx:
                gpu_ids = str(config.get('gpu_ids'))
                sample_id = _scalar_attr(graph, 'sample_id')
                time_idx = _scalar_attr(graph, 'time_idx')
                if sample_id is not None and time_idx is not None:
                    filename = f'sample{sample_id}_t{time_idx}'
                elif sample_id is not None:
                    filename = f'sample{sample_id}'
                else:
                    filename = f'batch{batch_idx}'

                viz_base = config.get('log_dir') or '../../output/chi-mgnflow'
                output_path = os.path.join(viz_base, output_prefix, gpu_ids,
                                           str(epoch), f'{filename}.h5')

                predicted_np = predicted.cpu().numpy()
                target_np = target.cpu().numpy()
                if delta_mean is not None and delta_std is not None:
                    predicted_denorm = predicted_np * delta_std + delta_mean
                    target_denorm = target_np * delta_std + delta_mean
                else:
                    predicted_denorm, target_denorm = predicted_np, target_np

                cached_faces = faces_cache.get(sample_id)
                if cached_faces is None and sample_id is not None:
                    if use_gpu and torch.cuda.is_available():
                        cached_faces = edges_to_triangles_gpu(
                            graph.edge_index.to(mesh_device), device=mesh_device)
                    else:
                        ei_np = (graph.edge_index.cpu().numpy()
                                 if hasattr(graph.edge_index, 'cpu')
                                 else np.array(graph.edge_index))
                        cached_faces = edges_to_triangles_optimized(ei_np)
                    faces_cache[sample_id] = cached_faces

                plot_data = save_inference_results_fast(
                    output_path, graph,
                    predicted_norm=predicted_np, target_norm=target_np,
                    predicted_denorm=predicted_denorm, target_denorm=target_denorm,
                    skip_visualization=not config.get('display_testset', True),
                    device=mesh_device,
                    feature_idx=config.get('plot_feature_idx', -1),
                    precomputed_faces=cached_faces,
                )
                if plot_data:
                    plot_data_queue.append(plot_data)

        if plot_data_queue:
            print(f"\nRendering {len(plot_data_queue)} visualizations...")
            failed = sum(0 if render_plot_data(pd) else 1 for pd in plot_data_queue)
            print("All visualizations complete!" if not failed
                  else f"Visualization done with {failed}/{len(plot_data_queue)} failures.")

    return total_loss / max(n, 1)
