"""
C2: calibrated endpoint-prediction guidance for the FM sampler.

Pilot recipe (GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md section 2.2; volume
median error 7.6% -> 1.7% on the ex1 checkpoint), written as the callback
`sample_latents(..., guidance_fn=...)` calls AFTER each Euler update:

    delta = guidance_fn(z_next, t_next, dt);  z_next = z_next + delta

    if t_start <= t_next < 1:
        v_next  = fm_model(z_next, t_next, cond, cond_mask)
        x1_hat  = z_next + (1 - t_next) * v_next            # one-step endpoint prediction
        loss    = sum_k ((soft_k(denorm(x1_hat)) - proxy_target_k) / proxy_target_k)^2
        g       = grad(loss, z_next) / RMS_per_sample(g)     # through the velocity net
        delta   = -eta * (1 - t_next) * g * scale
    else:
        delta   = 0

`scale` is what `step_mode` decides. The pilot applied the correction as a
per-step STATE JUMP at 50 steps, so its total strength grew with the step
count. `per_step_jump` (`scale = 1`) matches that schedule; `velocity_dt`
(`scale = dt * ode_steps_ref`) treats the correction as a velocity integrated
over dt, so the total is NFE-invariant and coincides with the pilot at
`ode_steps_ref` steps. Guidance targets are mapped to PROXY units with
`calibration.proxy_target` (the "2" in C2). The whole computation runs under
`torch.enable_grad()` because `sample_latents` is decorated `@torch.no_grad()`.

Only `volume` / `area` targets are guided (no bbox proxy); other names are
dropped with a printed note. The returned callable carries a `stats` dict
(`calls`, `active_calls`, `fm_evaluations`, `decoder_grids`) for NFE accounting.

TWO DELIBERATE DEVIATIONS from the pilot code, so the pilot's percentages are
an expectation to re-measure rather than a number this file reproduces:
  1. the loss denominator is the PROXY target `a * true + b`, not the raw true
     target. `g` is RMS-normalized per sample, so the loss scale itself is
     irrelevant, but the volume:area WEIGHT RATIO is not: at the pilot's own
     coefficients and typical DeepJEB values that is a ~11x shift in the
     balance of the guidance direction.
  2. `descriptor_proxy.soft_descriptors` integrates on CELL CENTRES
     (`h = 2 / R`), where the pilot summed node-centred `linspace` samples
     (`h = 2 / (R - 1)`), which over-integrates the domain by `(48/47)^3` =
     6.5%. The pilot's `a = 0.86` therefore does not transfer; fit the
     calibration on this quadrature (`eval_task descriptor_calibration`).
"""

import torch

from general_modules.descriptor_proxy import soft_descriptors, supported_soft_names

STEP_MODES = ('velocity_dt', 'per_step_jump')


def call_velocity(fm_model, z, t, cond=None, cond_mask=None):
    """`fm_model(z, t, cond=cond, cond_mask=cond_mask)`, retrying without
    `cond_mask` on a model that predates the keyword (only when it is None or
    all-True, i.e. carries no information)."""
    try:
        return fm_model(z, t, cond=cond, cond_mask=cond_mask)
    except TypeError as exc:
        if 'cond_mask' not in str(exc):
            raise
        if cond_mask is not None and not bool(torch.as_tensor(cond_mask).all()):
            raise TypeError('the velocity model does not accept cond_mask but a partial '
                            'condition mask was requested') from exc
        return fm_model(z, t, cond=cond)


def _scalar_time(t_next):
    """Float time from a Python number, a 0-d tensor, or a [B] tensor of equal entries."""
    if torch.is_tensor(t_next):
        flat = t_next.reshape(-1)
        return float(flat[0].item())
    return float(t_next)


def make_c2_guidance(vae, fm_model, cond, cond_mask, targets, calibration, eta=0.1,
                     t_start=0.3, step_mode='velocity_dt', resolution=48, tau=0.032,
                     ode_steps_ref=50, latent_mean=None, latent_std=None, bound=1.0,
                     chunk=32768, rms_eps=1e-8, verbose=True):
    """Build the C2 guidance callback for `sample_latents(..., guidance_fn=...)`.

    Args:
        vae: decoder with `decode_flat` (an `SDFVAE` or `MockDecoder`).
        fm_model: velocity net `fm_model(z, t, cond=..., cond_mask=...) -> v`.
        cond, cond_mask: the SAME tensors the sampler integrates with ([B,
            cond_dim] normalized conditions or None; mask None / [B] / [B,
            cond_dim] bool) -- the lookahead velocity must be the conditional one.
        targets: dict name -> TRUE target (raw geometric units); names without
            a soft proxy are ignored.
        calibration: `DescriptorCalibration` (forward map to proxy units).
        eta: guidance strength; t_start: window start (guides t_start <= t < 1).
        step_mode: 'velocity_dt' (NFE-invariant) or 'per_step_jump' (pilot-exact).
        resolution, tau: soft-proxy settings (must match `calibration`).
        ode_steps_ref: step count at which both modes coincide (pilot: 50).
        latent_mean, latent_std: [1, D] / [D] FM normalization statistics used
            to DE-NORMALIZE the endpoint prediction before decoding (keyword;
            the sampler's latent is normalized). None means the FM latent is
            already in VAE space (tests, mocks).
        bound, chunk: soft-proxy grid extent and decoder query chunk.

    Returns `guidance(z_next, t_next, dt) -> delta_z` (same shape/dtype as
    z_next, detached, exact zeros outside the window).
    """
    step_mode = str(step_mode).lower()
    if step_mode not in STEP_MODES:
        raise ValueError(f'step_mode must be one of {STEP_MODES}, got {step_mode!r}')
    eta = float(eta)
    if eta <= 0:
        raise ValueError(f'eta must be > 0, got {eta}')
    t_start = float(t_start)
    if not (0.0 <= t_start < 1.0):
        raise ValueError(f't_start must be in [0, 1), got {t_start}')
    names = list(supported_soft_names(targets.keys()))
    ignored = [n for n in targets if n not in names]
    if ignored and verbose:
        print(f'C2 guidance: no soft proxy for {ignored}; guiding on {names} only', flush=True)
    if not names:
        raise ValueError(f'make_c2_guidance: none of the targets {list(targets)} has a soft proxy')
    proxy_targets = {}
    for name in names:
        if name not in calibration:
            raise ValueError(f'calibration has no coefficients for {name!r}')
        pt = calibration.proxy_target(name, float(targets[name]))
        if abs(pt) < 1e-12:
            raise ValueError(f'proxy target for {name!r} is ~0 ({pt:g}); the relative loss is undefined')
        proxy_targets[name] = pt
    stats = {'calls': 0, 'active_calls': 0, 'fm_evaluations': 0, 'decoder_grids': 0,
             'names': names, 'ignored_targets': ignored, 'step_mode': step_mode,
             'proxy_targets': dict(proxy_targets)}
    scale_ref = float(ode_steps_ref)

    def guidance(z_next, t_next, dt):
        stats['calls'] += 1
        t = _scalar_time(t_next)
        if not (t_start <= t < 1.0):
            return torch.zeros_like(z_next)
        stats['active_calls'] += 1
        batch = z_next.shape[0]
        device = z_next.device
        with torch.enable_grad():
            z = z_next.detach().clone().requires_grad_(True)
            tt = torch.full((batch,), t, device=device, dtype=z.dtype)
            c = cond.to(device) if torch.is_tensor(cond) else cond
            m = cond_mask.to(device) if torch.is_tensor(cond_mask) else cond_mask
            v = call_velocity(fm_model, z, tt, cond=c, cond_mask=m)
            stats['fm_evaluations'] += 1
            x1_hat = z + (1.0 - t) * v
            if latent_mean is not None and latent_std is not None:
                mean = torch.as_tensor(latent_mean).float().reshape(1, -1).to(device)
                std = torch.as_tensor(latent_std).float().reshape(1, -1).to(device)
                x1_vae = x1_hat * std + mean
            else:
                x1_vae = x1_hat
            soft = soft_descriptors(vae, x1_vae, names=names, resolution=resolution, tau=tau,
                                    bound=bound, chunk=chunk)
            stats['decoder_grids'] += 1
            loss = 0.0
            for name in names:
                pt = proxy_targets[name]
                loss = loss + (((soft[name] - pt) / pt) ** 2).sum()
            g, = torch.autograd.grad(loss, z)
        g = g.detach()
        rms = g.pow(2).mean(dim=1, keepdim=True).add(rms_eps).sqrt()
        g = g / rms
        scale = 1.0 if step_mode == 'per_step_jump' else float(dt) * scale_ref
        delta = -eta * (1.0 - t) * scale * g
        return delta.to(dtype=z_next.dtype)

    guidance.stats = stats
    guidance.names = names
    guidance.proxy_targets = proxy_targets
    return guidance


def guidance_window_steps(ode_steps, t_start=0.3):
    """How many of `ode_steps` Euler updates land inside [t_start, 1) -- the
    number of extra FM evaluations (lookaheads) C2 costs at that step count."""
    dt = 1.0 / int(ode_steps)
    return sum(1 for i in range(int(ode_steps)) if t_start <= (i + 1) * dt < 1.0 - 1e-12)


def total_guidance_strength(ode_steps, t_start=0.3, eta=0.1, step_mode='velocity_dt',
                            ode_steps_ref=50):
    """sum over guided steps of eta * (1 - t) * scale -- the nominal total
    displacement (in RMS-normalized units) a constant-direction gradient would
    receive.

    `velocity_dt` makes this APPROXIMATELY independent of `ode_steps`: it is a
    Riemann sum of a smooth integrand whose window boundary `t_start <= t` also
    snaps to the step grid. Measured at the shipped defaults (eta 0.1,
    t_start 0.3, ode_steps_ref 50): 1.4000 / 1.2240 / 1.2600 / 1.2425 / 1.2338
    at 10 / 25 / 50 / 100 / 200 steps against the analytic limit
    `eta * ode_steps_ref * (1 - t_start)^2 / 2` = 1.2250 (the `ode_steps_ref`
    factor is the `scale` this mode carries) -- within 3% from 25 steps up, but
    +11% at 10, so a cheap 10-step preview is guided harder than the 50-step
    run it is compared against. `per_step_jump` grows with the step count by
    construction (pilot behaviour)."""
    dt = 1.0 / int(ode_steps)
    scale = 1.0 if step_mode == 'per_step_jump' else dt * float(ode_steps_ref)
    total = 0.0
    for i in range(int(ode_steps)):
        t = (i + 1) * dt
        if t_start <= t < 1.0 - 1e-12:
            total += float(eta) * (1.0 - t) * scale
    return total


__all__ = ['make_c2_guidance', 'call_velocity', 'guidance_window_steps',
           'total_guidance_strength', 'STEP_MODES']
