"""
E2: true-measure / proxy-Jacobian Newton correction of a sampled latent.

Pilot recipe (GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md section 2.4), which
on the ex1 checkpoint took the plain conditional sample's volume median error
from 7.6% to 0.28% in three rounds:

    for round in range(rounds):
        true      = measure_MC(z)                                # export path, res 96
        J         = d soft_descriptors(z) / d z_normalized       # k x D, autograd
        r         = a_name * (target_name - true_name)           # residual in PROXY units
        dz        = J^T (J J^T + 1e-6 I)^-1 r                    # damped minimum-norm step
        dz        = cap ||dz||_2 at step_cap_rms * sqrt(D)       # coordinate RMS <= cap
        for step in (dz, dz/2, dz/4):                            # backtracking
            if measure_MC(z + step) is valid and the TRUE relative residual norm
               sqrt(sum_k ((true_k - target_k) / target_k)^2) DECREASES: accept, break

Direction and scale come from the differentiable proxy; acceptance is judged
on the real export-path measurement, so a proxy that is merely correlated
with the truth is enough. That measurement is `true_descriptors`, which
reports NaN volume for a non-watertight mesh rather than the convex hull, so a
candidate that tears open is rejected (residual `inf`) instead of scored
against a hull that is monotonically larger than the solid it replaced. It is
a hybrid quasi-Newton step, not a Newton step on one consistent objective.
Once a round accepts no step the loop stops:
with the same z, J and r the next round would propose the same rejected dz.

Latent space: the step, the Jacobian, and the RMS cap all live in the
NORMALIZED (FM-space) flat latent, `z_vae = z_n * latent_std + latent_mean`.
`newton_correct` takes and returns the NORMALIZED latent by default
(`normalized=True`, i.e. what `sample_latents` produced, before `latent_clip`
and de-normalization); pass `normalized=False` to hand it the VAE-space latent
and get the VAE-space latent back.

Only `volume` / `area` targets are correctable (the proxy has no bbox form);
other names are ignored with a note in the history. FEA-named targets are the
caller's business to filter out.

History contract: `newton_correct` returns `(z_out, history)` where `history`
is ONE FLAT LIST of per-round dicts (JSON-friendly plain types), ordered by
latent row and then round, every dict tagged with `row`. Round `-1` is the
initial export-path measurement; `step_accepted` says whether that round moved
the latent. A consumer that handles one latent at a time can therefore count
`sum(1 for h in history if h.get('step_accepted'))` directly;
`split_history_by_row` / `summarize_history` serve the batched case.
"""

import math
import time

import torch

from general_modules.descriptor_calibration import true_descriptors
from general_modules.descriptor_proxy import soft_descriptors, supported_soft_names

DAMPING = 1e-6


def _to_normalized(z, latent_mean, latent_std, normalized):
    if normalized:
        return z
    return (z - latent_mean) / latent_std


def _to_vae(z_n, latent_mean, latent_std):
    return z_n * latent_std + latent_mean


def relative_residual(true, targets, names):
    """sqrt(sum_k ((true_k - target_k) / target_k)^2); inf when a value is missing/NaN."""
    total = 0.0
    for name in names:
        value = true.get(name)
        target = float(targets[name])
        if value is None or not math.isfinite(float(value)):
            return float('inf')
        denom = abs(target) if abs(target) > 1e-12 else 1e-12
        total += ((float(value) - target) / denom) ** 2
    return math.sqrt(total)


def proxy_jacobian(vae, z_n, names, latent_mean, latent_std, resolution=48, tau=0.032,
                   bound=1.0, chunk=32768):
    """Jacobian [k, D] of the soft descriptors w.r.t. ONE normalized latent [1, D].

    Also returns the proxy values [k] (detached). Runs under enable_grad and
    never touches the decoder's parameter gradients.
    """
    with torch.enable_grad():
        z = z_n.detach().clone().requires_grad_(True)
        z_vae = _to_vae(z, latent_mean, latent_std)
        soft = soft_descriptors(vae, z_vae, names=names, resolution=resolution, tau=tau,
                                bound=bound, chunk=chunk)
        rows = []
        values = []
        for i, name in enumerate(names):
            out = soft[name].sum()
            grad, = torch.autograd.grad(out, z, retain_graph=i < len(names) - 1)
            rows.append(grad.reshape(-1))
            values.append(out.detach())
    return torch.stack(rows, dim=0).detach(), torch.stack(values)


def damped_least_squares_step(J, residual, damping=DAMPING):
    """dz = J^T (J J^T + damping I)^-1 r  (minimum-norm solution for k << D)."""
    J = J.double()
    r = residual.double().reshape(-1)
    k = J.shape[0]
    gram = J @ J.T + damping * torch.eye(k, dtype=J.dtype, device=J.device)
    coeff = torch.linalg.solve(gram, r)
    return (J.T @ coeff)


def cap_step(dz, step_cap_rms):
    """Scale dz so its coordinate RMS is at most `step_cap_rms` (||dz|| <= cap * sqrt(D))."""
    cap = float(step_cap_rms) * math.sqrt(dz.numel())
    norm = float(dz.norm())
    if cap > 0 and norm > cap:
        return dz * (cap / norm), norm, True
    return dz, norm, False


def _measure(vae, z_n, latent_mean, latent_std, measure_resolution, bound, require_watertight):
    truth = true_descriptors(vae, _to_vae(z_n, latent_mean, latent_std),
                             measure_resolution=measure_resolution, bound=bound)
    ok = bool(truth['valid']) and (not require_watertight or bool(truth['watertight']))
    return truth, ok


def newton_correct(vae, z_flat, targets, calibration, latent_mean, latent_std, rounds=3,
                   step_cap_rms=0.12, line_search_tries=3, measure_resolution=96,
                   resolution=48, tau=0.032, bound=1.0, chunk=32768, normalized=True,
                   require_watertight=None, latent_clip=0.0, damping=DAMPING, verbose=False):
    """Pilot E2 on each row of `z_flat`.

    Args:
        vae: decoder (`SDFVAE` or `MockDecoder`), evaluated through `decode_flat`.
        z_flat: [B, D] or [D] latent; NORMALIZED FM-space when `normalized`
            (default), VAE-space otherwise. Returned in the same space/shape.
        targets: dict name -> TRUE target value (raw geometric units, the
            normalized-mesh frame). Names without a soft proxy are ignored.
        calibration: `DescriptorCalibration`; only its slopes `a` are used
            (residual scaling into proxy units).
        latent_mean, latent_std: [1, D] / [D] FM normalization statistics.
        rounds: correction rounds (0 returns z_flat unchanged, bitwise).
        step_cap_rms: cap on the coordinate RMS of one normalized step.
        line_search_tries: backtracking halvings tried per round (dz, dz/2, ...).
        measure_resolution: Marching Cubes grid for the true measurements.
        resolution, tau, bound, chunk: soft-proxy settings for the Jacobian.
        require_watertight: also reject steps whose mesh is not watertight.
            None (default) resolves to True whenever `volume` is one of the
            corrected names -- a torn mesh has no volume to score the step on
            (`true_descriptors` reports NaN, `relative_residual` inf), so
            accepting one would move the latent on an unmeasured quantity.
        latent_clip: when > 0 (and `normalized`), every candidate is clamped to
            +/- this magnitude BEFORE it is measured, so the accepted latent
            obeys the same box the sampler's `latent_clip` applies and the
            residual is measured on the latent that will actually be decoded.

    Returns (z_out, history): `history` is one flat list of per-round dicts,
    ordered by row then round, each with keys `row`, `round`, `true_before`,
    `true_after` (dict of the five COND_NAMES + valid/watertight/body_count_raw
    as measured after the round), `residual_before`, `residual_after`
    (relative residual norms; `residual_after` == `residual_before` when no
    step was accepted), `step_accepted` (bool; None on the initial entry),
    `tries` (backtracking measurements made), `step_rms` (coordinate RMS of
    the step taken, or of the rejected proposal), `step_norm_raw` (L2 norm of
    the uncapped proposal), `capped`, `proxy_before` (soft values at the start
    of the round), `seconds`. The first dict of each row (`round` = -1) is the
    initial measurement and carries `ignored_targets`; a row whose initial mesh
    is invalid has only that dict, with `skipped` set. `rounds == 0` returns
    `(z_flat, [])` with `z_flat` untouched (the same object).
    """
    z_in = torch.as_tensor(z_flat)
    squeeze = z_in.dim() == 1
    z_batch = z_in.unsqueeze(0) if squeeze else z_in
    rounds = int(rounds)
    if rounds < 0:
        raise ValueError(f'rounds must be >= 0, got {rounds}')
    tries = max(1, int(line_search_tries))
    if rounds == 0:
        return z_in, []
    latent_clip = float(latent_clip or 0.0)
    if latent_clip < 0:
        raise ValueError(f'latent_clip must be >= 0, got {latent_clip}')
    if latent_clip > 0 and not normalized:
        raise ValueError('latent_clip applies to the NORMALIZED latent; call with normalized=True')

    device = z_batch.device
    latent_mean = torch.as_tensor(latent_mean).float().reshape(1, -1).to(device)
    latent_std = torch.as_tensor(latent_std).float().reshape(1, -1).to(device)
    names = list(supported_soft_names(targets.keys()))
    ignored = [n for n in targets if n not in names]
    if require_watertight is None:
        require_watertight = 'volume' in names
    if not names:
        raise ValueError(f'newton_correct: none of the targets {list(targets)} has a soft proxy')
    for name in names:
        if name not in calibration:
            raise ValueError(f'calibration has no coefficients for {name!r}')
    slopes = torch.tensor([calibration.slope(n) for n in names], dtype=torch.float64, device=device)
    target_vec = torch.tensor([float(targets[n]) for n in names], dtype=torch.float64, device=device)

    was_training = getattr(vae, 'training', False)
    if hasattr(vae, 'eval'):
        vae.eval()
    out_rows = []
    histories = []
    try:
        for b in range(z_batch.shape[0]):
            z_n = _to_normalized(z_batch[b:b + 1].detach().float(), latent_mean, latent_std, normalized)
            history = []
            t0 = time.time()
            true, ok = _measure(vae, z_n, latent_mean, latent_std, measure_resolution, bound,
                                require_watertight)
            residual = relative_residual(true, targets, names) if ok else float('inf')
            history.append({'row': b, 'round': -1, 'true_before': None, 'true_after': dict(true),
                            'residual_before': None, 'residual_after': residual,
                            'step_accepted': None, 'tries': 0, 'step_rms': 0.0,
                            'step_norm_raw': 0.0, 'capped': False, 'proxy_before': None,
                            'ignored_targets': list(ignored), 'seconds': time.time() - t0})
            if not ok:
                history[-1]['skipped'] = 'initial mesh invalid; nothing to correct against'
                out_rows.append(z_n)
                histories.extend(history)
                continue
            for rnd in range(rounds):
                t0 = time.time()
                J, proxy = proxy_jacobian(vae, z_n, names, latent_mean, latent_std,
                                          resolution=resolution, tau=tau, bound=bound, chunk=chunk)
                true_vec = torch.tensor([float(true[n]) for n in names], dtype=torch.float64,
                                        device=device)
                r = slopes * (target_vec - true_vec)
                dz = damped_least_squares_step(J.reshape(len(names), -1), r, damping=damping)
                dz, raw_norm, capped = cap_step(dz, step_cap_rms)
                dz = dz.to(torch.float32).reshape(1, -1)
                accepted = False
                used = 0
                new_true = None
                new_residual = residual
                for k in range(tries):
                    used = k + 1
                    step = dz / (2.0 ** k)
                    cand = z_n + step
                    if latent_clip > 0:
                        cand = cand.clamp(-latent_clip, latent_clip)
                    cand_true, cand_ok = _measure(vae, cand, latent_mean, latent_std,
                                                  measure_resolution, bound, require_watertight)
                    cand_residual = relative_residual(cand_true, targets, names) if cand_ok else float('inf')
                    if cand_ok and cand_residual < residual:
                        step = (cand - z_n).detach()
                        z_n = cand.detach()
                        accepted = True
                        new_true, new_residual = cand_true, cand_residual
                        step_rms = float(step.pow(2).mean().sqrt())
                        break
                if not accepted:
                    step_rms = float(dz.pow(2).mean().sqrt())
                entry = {'row': b, 'round': rnd, 'true_before': dict(true),
                         'true_after': dict(new_true) if new_true is not None else dict(true),
                         'residual_before': residual, 'residual_after': new_residual,
                         'step_accepted': bool(accepted), 'tries': used, 'step_rms': step_rms,
                         'step_norm_raw': float(raw_norm), 'capped': bool(capped),
                         'proxy_before': {n: float(v) for n, v in zip(names, proxy.tolist())},
                         'seconds': time.time() - t0}
                history.append(entry)
                if verbose:
                    print(f'  newton round {rnd}: residual {residual:.4g} -> {new_residual:.4g} '
                          f'accepted={accepted} tries={used} step_rms={step_rms:.4f}', flush=True)
                if not accepted:
                    break
                true, residual = new_true, new_residual
            out_rows.append(z_n)
            histories.extend(history)
    finally:
        if hasattr(vae, 'train'):
            vae.train(was_training)

    z_out_n = torch.cat(out_rows, dim=0)
    z_out = z_out_n if normalized else _to_vae(z_out_n, latent_mean, latent_std)
    z_out = z_out.to(dtype=z_in.dtype)
    if squeeze:
        z_out = z_out.squeeze(0)
    return z_out, histories


def split_history_by_row(history):
    """Group a flat `newton_correct` history into one list per latent row
    (ordered by `row`). An empty history gives []."""
    rows = {}
    for entry in history:
        rows.setdefault(int(entry.get('row', 0)), []).append(entry)
    return [rows[key] for key in sorted(rows)]


def summarize_history(history, row=None):
    """Compact summary of one latent row's history: rounds run, steps accepted,
    initial/final relative residual, `skipped` reason, export-path
    measurements made (1 initial + one per backtracking try).

    `history` is the flat list `newton_correct` returns. With `row` None the
    history must describe a single row (the one-latent-at-a-time consumer);
    pass `row` to pick one row out of a batched history, or use
    `split_history_by_row`.
    """
    if not history:
        return {'rounds': 0, 'accepted': 0, 'residual_initial': None, 'residual_final': None,
                'skipped': None, 'measurements': 0}
    if row is None:
        present = sorted({int(h.get('row', 0)) for h in history})
        if len(present) > 1:
            raise ValueError(f'history holds rows {present}; pass row=<index> or use '
                             'split_history_by_row')
        entries = list(history)
    else:
        entries = [h for h in history if int(h.get('row', 0)) == int(row)]
        if not entries:
            raise ValueError(f'history has no entries for row {row}')
    rounds = [h for h in entries if h['round'] >= 0]
    return {
        'rounds': len(rounds),
        'accepted': sum(1 for h in rounds if h['step_accepted']),
        'residual_initial': entries[0]['residual_after'],
        'residual_final': entries[-1]['residual_after'],
        'skipped': entries[0].get('skipped'),
        'measurements': 1 + sum(h['tries'] for h in rounds),
    }
