"""
Test-time latent refinement for the SDF-VAE.

Starting from the encoder mean (or any flat latent), run Adam on the latent
alone against SDF labels with the decoder frozen -- DeepSDF-style
auto-decoding warm-started by the encoder. Measured 2026-08-24, latent
optimization beats the encoder by ~2.4x in reconstruction error on DeepJEB, so
this is the lever `reconstruct` and `evaluate` expose through
`latent_refine_steps` / `latent_refine_lr` / `latent_refine_prior_weight`.

The objective is the training reconstruction loss plus a surface anchor and an
optional pull toward the starting latent:

    L(z) = truncated-L1(f_z(query_points), clamp(query_sdf))      (== model.sdf_loss)
         + 0.1 * mean |f_z(surface_points)|
         + prior_weight * ||z - z0||^2      (SUM over latent dims, mean over batch)

The prior term SUMS over the latent rather than averaging over it. With the mean
form its gradient measured 8.1e-06 of the SDF term's at D = 1024 -- about five
orders of magnitude down -- so every `prior_weight` a user would plausibly write
was inert and the refined latent drifted without bound as `latent_refine_steps`
grew. Summed, `prior_weight 0.01` (the shipped `config_evaluate.txt` value) is a
real constraint; `latent_shift_l2` in the evaluate rows is the quantity to
calibrate it against.

Everything runs in fp32 with autocast disabled; no gradient reaches the VAE
parameters. Given identical inputs the result is deterministic up to GEMM
kernel numerics.
"""

import torch

from model.sdf_vae import sdf_loss

SURFACE_TERM_WEIGHT = 0.1


def _batched(tensor, batch, name):
    """Accept [N, C] / [N] or [B, N, C] / [B, N]; return a float32 tensor with a batch dim."""
    t = torch.as_tensor(tensor).float()
    if t.dim() == 1 or (t.dim() == 2 and t.shape[-1] == 3 and name != 'query_sdf'):
        t = t.unsqueeze(0)
    if t.shape[0] != batch:
        if t.shape[0] == 1:
            t = t.expand(batch, *t.shape[1:])
        else:
            raise ValueError(f'{name} batch {t.shape[0]} does not match the latent batch {batch}')
    return t


def refine_latent(vae, z_flat, surface_points, surface_normals, query_points, query_sdf,
                  steps, lr, prior_weight, clamp_dist=0.1, history=None):
    """Refine a flat latent against SDF labels with the decoder frozen.

    Args:
        vae: SDFVAE (any mode; switched to eval for the duration and restored).
        z_flat: [B, D] or [D] starting latent (the encoder mean).
        surface_points: [B, S, 3] surface samples; the level set is pulled onto them.
        surface_normals: [B, S, 3]; accepted for interface symmetry with the
            trainer, not used by the objective.
        query_points: [B, Q, 3] SDF query positions.
        query_sdf: [B, Q] SDF labels (truncated to +/- clamp_dist, like training).
        steps: Adam iterations (<= 0 returns z_flat.detach() untouched).
        lr: Adam learning rate on z.
        prior_weight: weight of the per-sample squared L2 distance ||z - z0||^2
            (summed over the latent, averaged over the batch); 0 disables the pull.
        clamp_dist: truncation band of the SDF loss (the checkpoint's value).
        history: optional list; receives (step, total, sdf_l1, surface_abs) per step.

    Returns the refined latent, detached, same shape as z_flat (batch dim kept).
    """
    steps = int(steps)
    z_start = torch.as_tensor(z_flat).detach()
    if steps <= 0:
        return z_start
    device = z_start.device
    z0 = z_start.to(dtype=torch.float32)
    squeeze = z0.dim() == 1
    if squeeze:
        z0 = z0.unsqueeze(0)
    batch = z0.shape[0]

    surf = _batched(surface_points, batch, 'surface_points').to(device)
    qp = _batched(query_points, batch, 'query_points').to(device)
    qs = _batched(query_sdf, batch, 'query_sdf').to(device)
    if qs.shape[:2] != qp.shape[:2]:
        raise ValueError(f'query_sdf {tuple(qs.shape)} does not match query_points {tuple(qp.shape)}')

    was_training = vae.training
    grad_flags = [p.requires_grad for p in vae.parameters()]
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    try:
        z = z0.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=float(lr))
        prior_weight = float(prior_weight)
        with torch.autocast(device_type=device.type, enabled=False):
            for step in range(steps):
                optimizer.zero_grad(set_to_none=True)
                sdf_l1 = sdf_loss(vae.decode_flat(z, qp).float(), qs, clamp_dist)
                surface_abs = vae.decode_flat(z, surf).float().abs().mean()
                loss = sdf_l1 + SURFACE_TERM_WEIGHT * surface_abs
                if prior_weight > 0:
                    loss = loss + prior_weight * (z - z0).pow(2).sum(dim=-1).mean()
                loss.backward()
                optimizer.step()
                if history is not None:
                    history.append((step, float(loss.item()), float(sdf_l1.item()),
                                    float(surface_abs.item())))
            # The losses above are all computed BEFORE their optimizer step, so
            # history[-1] would describe the second-to-last latent rather than
            # the one returned. Score the final latent once more.
            if history is not None:
                with torch.no_grad():
                    sdf_l1 = sdf_loss(vae.decode_flat(z, qp).float(), qs, clamp_dist)
                    surface_abs = vae.decode_flat(z, surf).float().abs().mean()
                    final = sdf_l1 + SURFACE_TERM_WEIGHT * surface_abs
                    if prior_weight > 0:
                        final = final + prior_weight * (z - z0).pow(2).sum(dim=-1).mean()
                history.append((steps, float(final.item()), float(sdf_l1.item()),
                                float(surface_abs.item())))
    finally:
        for p, flag in zip(vae.parameters(), grad_flags):
            p.requires_grad_(flag)
        vae.train(was_training)

    out = z.detach()
    return out.squeeze(0) if squeeze else out
