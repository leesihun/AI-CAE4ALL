"""Interpolation between two reproducible flow-matching samples.

`interpolation_space` selects where the mixing happens:

  * `slerp_noise` (default): re-draw the seeded source batch's t=0 noise
    explicitly (bit-identical to the draw `sample_latents` makes internally),
    spherically interpolate the two endpoint noise rows, and integrate the
    three-row stack (eps_a, eps_mid, eps_b) through the FM ODE. The endpoints
    reproduce the original samples; the midpoint stays on the Gaussian shell
    the velocity net was trained on, so it decodes to an on-manifold shape.
  * `lerp_latent` (legacy): reproduce the whole source batch through the ODE
    and `torch.lerp` the two normalized FM latents. Straight-line latent
    midpoints can leave the data manifold (multi-body / blended decodes).
  * `cond_sweep`: hold the t=0 noise FIXED (row `sample_index_a` of the seeded
    batch) and sweep the CONDITION instead: `cond_values_a` -> `cond_values_b`
    in `sweep_steps` equal steps of the NORMALIZED condition space (an affine
    map of the raw one, so raw values sweep linearly too). All rows are
    integrated in one `sample_latents` call (noise repeated, conditions
    stacked), each is decoded and audited against its requested geometric
    conditions, and the strip PNG shows the shape family one condition
    direction produces. Entries written as the literal `nan` are unspecified
    (needs a `cond_dropout_mode per_dim` FM; both endpoints must leave the
    same entries unspecified). `cfg_scale` applies; `cond_values` is not read
    -- the two endpoint keys are.

The `slerp_noise` / `lerp_latent` code paths are unchanged by the sweep.
"""

import json
import os
import time

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from general_modules.mesh_extraction import decode_sdf_grid, mesh_report, sdf_grid_to_mesh
from inference_profiles.sample import (
    VelocityCallCounter,
    _audit_report,
    _jsonable,
    _model_state,
    cond_dropout_mode_of,
    load_vae,
    normalize_condition_request,
    parse_condition_values,
)
from model.velocity_net import VelocityNet, sample_latents
from training_profiles.setup import load_checkpoint, resolve_device

INTERPOLATION_SPACES = ('slerp_noise', 'lerp_latent', 'cond_sweep')
DEFAULT_SWEEP_STEPS = 5
_ENDPOINT_COLOR = '#3B82C4'
_MIDDLE_COLOR = '#E68A2E'


def slerp(a, b, alpha, eps=1e-6):
    """Spherical linear interpolation between two flat vectors.

    omega = acos(clamp(cos<a, b>)); result = sin((1-alpha) omega)/sin omega * a
    + sin(alpha omega)/sin omega * b. Computed in float64 and cast back to the
    input dtype; falls back to torch.lerp when sin(omega) < eps (parallel or
    anti-parallel inputs). alpha=0 / alpha=1 return a / b exactly. For equal
    norms the norm is preserved, which is what keeps an interpolated Gaussian
    noise row on the shell the flow was trained from.
    """
    alpha = float(alpha)
    a64 = a.detach().double()
    b64 = b.detach().double()
    a_unit = a64 / a64.norm().clamp_min(1e-12)
    b_unit = b64 / b64.norm().clamp_min(1e-12)
    cos = (a_unit * b_unit).sum().clamp(-1.0, 1.0)
    omega = torch.acos(cos)
    sin_omega = torch.sin(omega)
    if float(sin_omega.abs()) < eps:
        return torch.lerp(a, b, alpha)
    out = (torch.sin((1.0 - alpha) * omega) / sin_omega) * a64 \
        + (torch.sin(alpha * omega) / sin_omega) * b64
    return out.to(a.dtype)


def _plot_strip(meshes, labels, reports, path, dpi=180, max_faces=0, title=None, colors=None):
    """Render N meshes side by side with identical axes and camera settings.

    `meshes` entries may be None (a decode without a zero crossing); that
    panel is drawn empty with its label so the strip keeps one panel per step.
    `colors` defaults to endpoints blue / interior panels orange.
    """
    n = len(meshes)
    if n == 0:
        raise ValueError('_plot_strip needs at least one panel')
    if colors is None:
        colors = tuple(_ENDPOINT_COLOR if i in (0, n - 1) else _MIDDLE_COLOR for i in range(n))
    fig = plt.figure(figsize=(max(5.3 * n, 6.0), 5.8), dpi=dpi, facecolor='white')

    for index, (mesh, label, report, color) in enumerate(
            zip(meshes, labels, reports, colors), start=1):
        ax = fig.add_subplot(1, n, index, projection='3d')
        if mesh is not None:
            triangles = mesh.triangles
            if max_faces > 0 and len(triangles) > max_faces:
                selected = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
                triangles = triangles[selected]
            surface = Poly3DCollection(
                triangles, facecolor=color, edgecolor='none', linewidth=0.0, alpha=1.0)
            ax.add_collection3d(surface)
        else:
            ax.text(0.0, 0.0, 0.0, 'no zero crossing', ha='center', va='center', fontsize=11)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
        volume = report.get('volume')
        volume_text = f'{volume:.4f}' if volume is not None else 'n/a'
        faces = report.get('faces')
        faces_text = f'{faces:,}' if faces is not None else 'n/a'
        ax.set_title(f'{label}\nvolume={volume_text}, faces={faces_text}',
                     fontsize=12, pad=0, y=0.90)

    fig.suptitle(title or 'SDFFlow latent interpolation', fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_triptych(meshes, labels, reports, path, dpi=180, max_faces=0, title=None):
    """Render three meshes with identical axes and camera settings (the
    slerp_noise / lerp_latent figure; a 3-panel `_plot_strip`)."""
    _plot_strip(meshes, labels, reports, path, dpi=dpi, max_faces=max_faces, title=title,
                colors=(_ENDPOINT_COLOR, _MIDDLE_COLOR, _ENDPOINT_COLOR))


def _l2(x):
    return float(torch.linalg.vector_norm(x.float()).item())


# ---------------------------------------------------------------------------
# cond_sweep
# ---------------------------------------------------------------------------

def sweep_alphas(sweep_steps):
    """`sweep_steps` equally spaced alphas in [0, 1] (endpoints included)."""
    sweep_steps = int(sweep_steps)
    if sweep_steps < 2:
        raise ValueError(f'sweep_steps must be an integer >= 2 (both endpoints are always '
                         f'included), got {sweep_steps}')
    return np.linspace(0.0, 1.0, sweep_steps)


def sweep_conditions(cond_a_n, cond_b_n, alphas):
    """Stack of lerp'd NORMALIZED conditions, [len(alphas), cond_dim] float32.

    Row k is `(1 - alpha_k) * cond_a + alpha_k * cond_b`; rows 0 and -1 equal
    the endpoints exactly (alpha 0 and 1).
    """
    a = torch.as_tensor(cond_a_n).float().reshape(-1)
    b = torch.as_tensor(cond_b_n).float().reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f'endpoint conditions differ in length: {tuple(a.shape)} vs {tuple(b.shape)}')
    rows = []
    for alpha in alphas:
        alpha = float(alpha)
        if alpha == 0.0:
            rows.append(a.clone())
        elif alpha == 1.0:
            rows.append(b.clone())
        else:
            rows.append(torch.lerp(a, b, alpha))
    return torch.stack(rows, dim=0)


def integrate_cond_sweep(model, eps_row, cond_stack, cond_mask, device, ode_steps=50,
                         cfg_scale=1.0):
    """Integrate ONE noise row under every condition of `cond_stack` in a
    single `sample_latents` call.

    `eps_row` is the shared t=0 state ([D] or [1, D]); it is repeated to
    `cond_stack.shape[0]` rows. `cond_mask` is None (all specified) or a
    [cond_dim] bool mask broadcast to every row (a partial request, which
    needs a `cond_dropout_mode per_dim` FM). Returns the normalized latents
    [K, D]. Row k equals `sample_latents(model, 1, D, device, cond=cond_k,
    noise=eps_row, ...)` up to batched-kernel rounding.
    """
    cond_stack = torch.as_tensor(cond_stack).float().to(device)
    k = int(cond_stack.shape[0])
    eps_row = torch.as_tensor(eps_row).float().reshape(1, -1).to(device)
    noise = eps_row.repeat(k, 1)
    kwargs = {}
    if cond_mask is not None:
        mask = torch.as_tensor(cond_mask).bool().reshape(1, -1).to(device)
        kwargs['cond_mask'] = mask.repeat(k, 1)
    return sample_latents(model, k, eps_row.shape[1], device, cond=cond_stack,
                          cfg_scale=cfg_scale, ode_steps=int(ode_steps), noise=noise, **kwargs)


def _raw_from_normalized(cond_n, fm_ckpt, mask_np):
    """De-normalize a NORMALIZED condition row; NaN in unspecified dims."""
    mean = fm_ckpt['cond_mean'].squeeze(0).cpu().double().numpy()
    std = fm_ckpt['cond_std'].squeeze(0).cpu().double().numpy()
    raw = np.asarray(cond_n, dtype=np.float64) * std + mean
    raw = np.where(mask_np, raw, np.nan)
    return raw


def _run_cond_sweep(config, fm_ckpt, fm_path, vae, vae_path, model, device, out_dir,
                    latent_flat_dim, cond_dim, seed, index_a, source_num_samples, ode_steps,
                    resolution):
    cond_names = [str(n) for n in fm_ckpt.get('cond_names', [])]
    if cond_dim == 0:
        raise ValueError('interpolation_space cond_sweep needs a CONDITIONAL FM checkpoint '
                         '(this one has cond_dim 0)')
    if config.get('cond_values') is not None:
        raise ValueError('interpolation_space cond_sweep reads cond_values_a and cond_values_b, '
                         'not cond_values; remove cond_values from the config')
    values_a = config.get('cond_values_a')
    values_b = config.get('cond_values_b')
    if values_a is None or values_b is None:
        raise ValueError('interpolation_space cond_sweep requires both cond_values_a and '
                         f'cond_values_b (lists of {cond_dim} entries in checkpoint order '
                         f'{cond_names}; write nan for an unspecified entry)')
    alphas = sweep_alphas(config.get('sweep_steps', DEFAULT_SWEEP_STEPS))
    cfg_scale = float(config.get('cfg_scale', 1.0))
    if cfg_scale < 0:
        raise ValueError('cfg_scale must be a nonnegative number')

    raw_a = parse_condition_values(values_a, cond_names, label='cond_values_a')
    raw_b = parse_condition_values(values_b, cond_names, label='cond_values_b')
    cond_a_n, mask_a, request_a = normalize_condition_request(raw_a, fm_ckpt, config,
                                                              label='cond_values_a')
    cond_b_n, mask_b, request_b = normalize_condition_request(raw_b, fm_ckpt, config,
                                                              label='cond_values_b')
    if not torch.equal(mask_a, mask_b):
        differ = [cond_names[i] for i in range(cond_dim) if bool(mask_a[i]) != bool(mask_b[i])]
        raise ValueError(f'cond_values_a and cond_values_b must leave the SAME entries '
                         f"unspecified ('nan'); they differ on {differ}. A sweep between a "
                         'specified and an unspecified value of one condition is undefined.')
    mask = mask_a
    mask_np = mask.numpy()
    partial = bool(not mask_np.all())
    cond_std_np = fm_ckpt['cond_std'].squeeze(0).cpu().numpy().astype(np.float64)
    k = len(alphas)
    shown = lambda req: {name: ('unspecified' if not req['specified'][name] else req['raw'][name])
                         for name in cond_names}
    print(f'Condition sweep: {k} steps from {shown(request_a)} to {shown(request_b)} '
          f'(cfg_scale={cfg_scale:g}, fixed noise row {index_a} of the seed-{seed} '
          f'{source_num_samples}-row batch'
          + (f", cond_dropout_mode={cond_dropout_mode_of(fm_ckpt)}, partial request" if partial else '')
          + ')')

    # Fixed base noise: bit-identical to the draw sample_latents() makes for the
    # seeded batch, so row `index_a` is the very noise `mode sample` used.
    generator = torch.Generator(device=device).manual_seed(seed)
    eps = torch.randn(source_num_samples, latent_flat_dim, device=device, generator=generator)
    eps_a = eps[index_a]
    cond_stack = sweep_conditions(cond_a_n, cond_b_n, alphas)

    counter = VelocityCallCounter(model)
    t0 = time.time()
    try:
        z_n = integrate_cond_sweep(model, eps_a, cond_stack, mask if partial else None, device,
                                   ode_steps=ode_steps, cfg_scale=cfg_scale)
    finally:
        counter.remove()
    sample_seconds = time.time() - t0
    print(f'Integrated the {k}-row sweep in {sample_seconds:.2f}s '
          f'({counter.calls} velocity-net calls)')
    z = z_n * fm_ckpt['latent_std'].to(device) + fm_ckpt['latent_mean'].to(device)

    meshes, reports, steps = [], [], []
    for i, alpha in enumerate(alphas):
        volume = decode_sdf_grid(vae, z[i:i + 1], resolution=resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        report.setdefault('body_count_raw', None)
        requested_raw = _raw_from_normalized(cond_stack[i].numpy(), fm_ckpt, mask_np)
        _audit_report(report, cond_names, requested_raw, cond_std_np, mask_np)
        label = f'alpha={float(alpha):.3f}'
        report['label'] = label
        report['alpha'] = float(alpha)
        report['step'] = i
        report['requested_normalized'] = {
            name: (float(cond_stack[i, j]) if mask_np[j] else None)
            for j, name in enumerate(cond_names)}
        report['requested_raw'] = {
            name: (float(requested_raw[j]) if mask_np[j] else None)
            for j, name in enumerate(cond_names)}
        if report['valid']:
            mesh_path = os.path.join(out_dir, f'sample_{seed}_sweep_{i}.stl')
            mesh.export(mesh_path)
            report['path'] = mesh_path
            volume_text = (f'{report["volume"]:.6f}' if report.get('volume') is not None else 'n/a')
            score_text = (f' score={report["condition_score"]:.3f}'
                          if report.get('condition_score') is not None else '')
            print(f'  step {i} {label}: watertight={report["watertight"]} faces={report["faces"]} '
                  f'volume={volume_text} body_count_raw={report["body_count_raw"]}{score_text} '
                  f'-> {mesh_path}')
            meshes.append(mesh)
        else:
            report['path'] = None
            print(f'  step {i} {label}: NO ZERO CROSSING')
            meshes.append(None)
        reports.append(report)
        # A compact per-panel view of what was ASKED for and what came back --
        # the strip PNG's caption in JSON form. `results` below keeps the full
        # mesh reports; this is the same dicts filtered, not a second copy of
        # them, so the metadata does not serialize every report twice.
        steps.append({
            'step': i,
            'alpha': report['alpha'],
            'label': label,
            'valid': bool(report['valid']),
            'watertight': report.get('watertight'),
            'body_count_raw': report.get('body_count_raw'),
            'faces': report.get('faces'),
            'volume': report.get('volume'),
            'area': report.get('area'),
            'path': report.get('path'),
            'requested_normalized': report['requested_normalized'],
            'requested_raw': report['requested_raw'],
            'actual_conditions': report.get('actual_conditions'),
            'condition_rel_error': report.get('condition_rel_error'),
            'condition_score': report.get('condition_score'),
            'audited_names': report.get('audited_names'),
            'not_measurable_geometrically': report.get('not_measurable_geometrically'),
        })

    plot_path = os.path.join(out_dir, f'cond_sweep_{seed}_{index_a:03d}.png')
    labels = [f'Step {i} ({r["label"]})' for i, r in enumerate(reports)]
    _plot_strip(meshes, labels, reports, plot_path,
                dpi=int(config.get('plot_dpi', 180)),
                max_faces=int(config.get('plot_max_faces', 0)),
                title=f'SDFFlow condition sweep (noise row {index_a}, seed {seed}): '
                      f'{k} steps, cfg_scale={cfg_scale:g}')

    consecutive = [_l2(z_n[i + 1] - z_n[i]) for i in range(k - 1)]
    metadata = {
        'fm_modelpath': fm_path,
        'vae_modelpath': vae_path,
        'seed': seed,
        'source_num_samples': source_num_samples,
        'sample_index_a': index_a,
        'interpolation_space': 'cond_sweep',
        'sweep_steps': k,
        'alphas': [float(a) for a in alphas],
        'cfg_scale': cfg_scale,
        'ode_steps': ode_steps,
        'mc_resolution': resolution,
        'cond_names': cond_names,
        'cond_values_a': values_a,
        'cond_values_b': values_b,
        'condition_request_a': request_a,
        'condition_request_b': request_b,
        'cond_mask': mask_np.tolist(),
        'partial': partial,
        'cond_dropout_mode': cond_dropout_mode_of(fm_ckpt),
        'noise_norm': _l2(eps_a),
        'velocity_net_calls': counter.calls,
        'sample_seconds': sample_seconds,
        'latent_distances': {
            'endpoint_l2': _l2(z_n[-1] - z_n[0]),
            'consecutive_l2': consecutive,
            'from_first_l2': [_l2(z_n[i] - z_n[0]) for i in range(k)],
        },
        'body_count_raw': [r['body_count_raw'] for r in reports],
        'valid': [bool(r['valid']) for r in reports],
        'steps': steps,
        'results': reports,
        'plot_path': plot_path,
    }
    metadata_path = os.path.join(out_dir, f'cond_sweep_{seed}_{index_a:03d}_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(_jsonable(metadata), f, indent=2)
    print(f'Plot: {plot_path}')
    print(f'Metadata: {metadata_path}')
    return metadata


# ---------------------------------------------------------------------------
# mode interpolate
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_interpolate(config, config_filename='config.txt'):
    device = resolve_device(config)

    fm_path = config.get('fm_modelpath', '../../output/geometry_generation/sdfflow_fm.pth')
    print(f'Loading FM checkpoint from {fm_path}')
    fm_ckpt = load_checkpoint(fm_path, device)
    vae_path = config.get('vae_modelpath', fm_ckpt['vae_modelpath'])
    print(f'Loading VAE checkpoint from {vae_path}')
    vae, _ = load_vae(vae_path, device)

    latent_flat_dim = int(fm_ckpt['latent_flat_dim'])
    cond_dim = int(fm_ckpt['cond_dim'])
    model = VelocityNet(fm_ckpt['config'], latent_flat_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(_model_state(fm_ckpt))
    model.eval()

    seed = int(config.get('seed', 0))
    index_a = int(config.get('sample_index_a', 0))
    index_b = int(config.get('sample_index_b', 1))
    alpha = float(config.get('alpha', 0.5))
    source_num_samples = int(config.get('source_num_samples', max(index_a, index_b) + 1))
    ode_steps = int(config.get('ode_steps', 50))
    resolution = int(config.get('mc_resolution', 128))
    out_dir = config.get('output_dir', '../../output/geometry_generation/interpolation')
    space = str(config.get('interpolation_space', 'slerp_noise')).lower()

    if space not in INTERPOLATION_SPACES:
        raise ValueError(f"interpolation_space must be one of {INTERPOLATION_SPACES}, got '{space}'")
    if space == 'cond_sweep':
        # The sweep uses ONE noise row (sample_index_a); sample_index_b and alpha
        # play no role, so only index_a is range-checked here.
        if index_a < 0:
            raise ValueError('sample_index_a must be a non-negative index')
        if index_a >= source_num_samples:
            raise ValueError('source_num_samples must exceed sample_index_a')
        os.makedirs(out_dir, exist_ok=True)
        return _run_cond_sweep(config, fm_ckpt, fm_path, vae, vae_path, model, device, out_dir,
                               latent_flat_dim, cond_dim, seed, index_a, source_num_samples,
                               ode_steps, resolution)
    if index_a < 0 or index_b < 0 or index_a == index_b:
        raise ValueError('sample_index_a and sample_index_b must be distinct non-negative indices')
    if max(index_a, index_b) >= source_num_samples:
        raise ValueError('source_num_samples must exceed both endpoint indices')
    if not 0.0 <= alpha <= 1.0:
        raise ValueError('alpha must be within [0, 1]')
    if config.get('cond_values') is not None:
        raise ValueError('This interpolation mode currently reproduces unconditional samples only')

    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    noise_distances = None
    if space == 'slerp_noise':
        # Bit-identical to the draw sample_latents() makes for the seeded batch.
        eps = torch.randn(source_num_samples, latent_flat_dim, device=device, generator=generator)
        eps_a = eps[index_a]
        eps_b = eps[index_b]
        eps_mid = slerp(eps_a, eps_b, alpha)
        selected_noise = torch.stack((eps_a, eps_mid, eps_b))
        selected_z_n = sample_latents(
            model, 3, latent_flat_dim, device, ode_steps=ode_steps, noise=selected_noise)
        z_a_n, z_mid_n, z_b_n = selected_z_n[0], selected_z_n[1], selected_z_n[2]
        noise_distances = {
            'endpoint_l2': _l2(eps_b - eps_a),
            'a_to_interpolation_l2': _l2(eps_mid - eps_a),
            'interpolation_to_b_l2': _l2(eps_b - eps_mid),
            'norms': {'a': _l2(eps_a), 'interpolation': _l2(eps_mid), 'b': _l2(eps_b)},
        }
        print(f'Re-drew the {source_num_samples}-row seed-{seed} noise batch, slerp-mixed rows '
              f'{index_a}/{index_b} at alpha={alpha:g}, and integrated the 3-row stack '
              f'in {time.time() - t0:.2f}s')
    else:
        source_z_n = sample_latents(
            model, source_num_samples, latent_flat_dim, device,
            ode_steps=ode_steps, generator=generator)
        z_a_n = source_z_n[index_a]
        z_b_n = source_z_n[index_b]
        z_mid_n = torch.lerp(z_a_n, z_b_n, alpha)
        selected_z_n = torch.stack((z_a_n, z_mid_n, z_b_n))
        print(f'Reproduced {source_num_samples} source latents and lerped alpha={alpha:g} '
              f'in {time.time() - t0:.2f}s')
    selected_z = (selected_z_n * fm_ckpt['latent_std'].to(device)
                  + fm_ckpt['latent_mean'].to(device))

    alpha_tag = f'{alpha:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')
    names = (
        f'sample_{seed}_{index_a:03d}',
        f'sample_{seed}_{index_a:03d}_{index_b:03d}_alpha{alpha_tag}',
        f'sample_{seed}_{index_b:03d}',
    )
    mid_label = ('Noise slerp' if space == 'slerp_noise' else 'Latent midpoint')
    labels = (
        f'Sample {index_a} (alpha=0.0)',
        f'{mid_label} (alpha={alpha:g})',
        f'Sample {index_b} (alpha=1.0)',
    )

    meshes = []
    reports = []
    for i, name in enumerate(names):
        volume = decode_sdf_grid(
            vae, selected_z[i:i + 1], resolution=resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        if not report['valid']:
            raise RuntimeError(f'Interpolation decode failed for {labels[i]}: no zero crossing')
        mesh_path = os.path.join(out_dir, f'{name}.stl')
        mesh.export(mesh_path)
        report['label'] = labels[i]
        report['path'] = mesh_path
        # Raw component count before keep_largest (set by sdf_grid_to_mesh on
        # mesh.metadata and surfaced by mesh_report); None on older builds.
        report.setdefault('body_count_raw', None)
        meshes.append(mesh)
        reports.append(report)
        volume_text = (f'{report["volume"]:.6f}' if report.get('volume') is not None else 'n/a')
        print(f'  {labels[i]}: watertight={report["watertight"]} '
              f'faces={report["faces"]} volume={volume_text} '
              f'body_count_raw={report["body_count_raw"]} -> {mesh_path}')

    plot_path = os.path.join(
        out_dir, f'interpolation_{index_a:03d}_{index_b:03d}_alpha{alpha_tag}.png')
    _plot_triptych(
        meshes, labels, reports, plot_path,
        dpi=int(config.get('plot_dpi', 180)),
        max_faces=int(config.get('plot_max_faces', 0)),
        title=(f'SDFFlow interpolation ({space}): sample {index_a} to sample {index_b}'))

    metadata = {
        'fm_modelpath': fm_path,
        'vae_modelpath': vae_path,
        'seed': seed,
        'source_num_samples': source_num_samples,
        'sample_index_a': index_a,
        'sample_index_b': index_b,
        'alpha': alpha,
        'interpolation_space': space,
        'ode_steps': ode_steps,
        'mc_resolution': resolution,
        'noise_distances': noise_distances,
        'latent_distances': {
            'endpoint_l2': _l2(z_b_n - z_a_n),
            'a_to_interpolation_l2': _l2(z_mid_n - z_a_n),
            'interpolation_to_b_l2': _l2(z_b_n - z_mid_n),
        },
        'body_count_raw': [r['body_count_raw'] for r in reports],
        'results': reports,
        'plot_path': plot_path,
    }
    metadata_path = os.path.join(
        out_dir, f'interpolation_{index_a:03d}_{index_b:03d}_alpha{alpha_tag}_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Plot: {plot_path}')
    print(f'Metadata: {metadata_path}')
    return metadata
