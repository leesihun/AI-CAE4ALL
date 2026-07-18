"""
Inference: generate geometries from the trained SDF-VAE + FM stack, or
round-trip reconstruct an input mesh through the VAE.

Modes (config `mode`):
    sample      noise -> FM ODE (optional conditions + CFG) -> SDF -> STL
    reconstruct input mesh -> encoder mu -> SDF -> STL
"""

import json
import os
import time

import numpy as np
import torch

from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE
from model.velocity_net import VelocityNet, sample_latents
from training_profiles.setup import load_checkpoint, resolve_device


def _actual_conditions(report, cond_names):
    """Return decoded-mesh descriptors in checkpoint condition order."""
    if not report.get('valid'):
        return None
    extents = report.get('extents')
    values = {
        'bbox_x': extents[0] if extents else None,
        'bbox_y': extents[1] if extents else None,
        'bbox_z': extents[2] if extents else None,
        'volume': report.get('volume'),
        'area': report.get('area'),
    }
    actual = [values.get(name) for name in cond_names]
    return None if any(value is None for value in actual) else np.asarray(actual, dtype=np.float64)


def _audit_report(report, cond_names, target, cond_std):
    actual = _actual_conditions(report, cond_names)
    if actual is None:
        report['condition_score'] = None
        return
    error = actual - target
    normalized_error = error / cond_std
    report['actual_conditions'] = dict(zip(cond_names, actual.tolist()))
    report['condition_abs_error'] = dict(zip(cond_names, np.abs(error).tolist()))
    report['condition_rel_error'] = dict(zip(
        cond_names, (np.abs(error) / np.maximum(np.abs(target), 1e-8)).tolist()))
    report['condition_score'] = float(np.sqrt(np.mean(normalized_error ** 2)))


def _condition_summary(results, cond_names, target):
    audited = [r for r in results if r.get('actual_conditions') is not None]
    if not audited:
        return {'audited_meshes': 0}
    actual = np.asarray([[r['actual_conditions'][name] for name in cond_names]
                         for r in audited], dtype=np.float64)
    error = np.abs(actual - target[None, :])
    scores = [r['condition_score'] for r in audited]
    return {
        'audited_meshes': len(audited),
        'target': dict(zip(cond_names, target.tolist())),
        'median_actual': dict(zip(cond_names, np.median(actual, axis=0).tolist())),
        'median_abs_error': dict(zip(cond_names, np.median(error, axis=0).tolist())),
        'median_rel_error': dict(zip(
            cond_names,
            np.median(error / np.maximum(np.abs(target[None, :]), 1e-8), axis=0).tolist())),
        'best_condition_score': float(min(scores)),
        'median_condition_score': float(np.median(scores)),
    }


def _model_state(ckpt):
    """Prefer EMA weights; strip AveragedModel 'module.' prefix."""
    state = ckpt.get('ema_state') or ckpt['model_state']
    if ckpt.get('ema_state') is not None:
        state = {k.replace('module.', '', 1): v for k, v in state.items() if k != 'n_averaged'}
    return state


def load_vae(path, device):
    ckpt = load_checkpoint(path, device)
    vae = SDFVAE(ckpt['config']).to(device)
    vae.load_state_dict(_model_state(ckpt))
    vae.eval()
    return vae, ckpt


def run_sample(config, config_filename='config.txt'):
    device = resolve_device(config)

    fm_path = config.get('fm_modelpath', './outputs/sdfflow_fm.pth')
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

    num_samples = int(config.get('num_samples', 8))
    ode_steps = int(config.get('ode_steps', 50))
    cfg_scale = float(config.get('cfg_scale', 1.0))
    resolution = int(config.get('mc_resolution', 128))
    seed = int(config.get('seed', 0))
    candidate_multiplier = int(config.get('candidate_multiplier', 1))
    if candidate_multiplier < 1:
        raise ValueError('candidate_multiplier must be at least 1')
    out_dir = config.get('output_dir', './outputs/samples')
    os.makedirs(out_dir, exist_ok=True)

    # ---- Conditions ----
    cond = None
    cond_values = config.get('cond_values', None)
    condition_request = None
    target = cond_std_np = None
    total_candidates = num_samples
    if cond_values is not None and cond_dim > 0:
        if not isinstance(cond_values, list):
            cond_values = [cond_values]
        if len(cond_values) != cond_dim:
            raise ValueError(f'cond_values must have {cond_dim} entries '
                             f'({fm_ckpt["cond_names"]}), got {len(cond_values)}')
        raw = torch.tensor([float(v) for v in cond_values], dtype=torch.float32)
        cond_mean = fm_ckpt['cond_mean'].squeeze(0).cpu()
        cond_std = fm_ckpt['cond_std'].squeeze(0).cpu()
        cond_n = (raw - cond_mean) / cond_std
        max_condition_z = float(config.get(
            'max_condition_z', fm_ckpt.get('cond_clip') or 5.0))
        ood_policy = str(config.get('condition_ood_policy', 'error')).lower()
        if ood_policy not in ('error', 'warn', 'clamp'):
            raise ValueError('condition_ood_policy must be error, warn, or clamp')
        excessive = cond_n.abs() > max_condition_z
        if excessive.any():
            details = ', '.join(
                f'{fm_ckpt["cond_names"][i]}={float(cond_n[i]):.2f} sigma'
                for i in torch.where(excessive)[0].tolist())
            message = (f'Condition request exceeds max_condition_z={max_condition_z:g}: '
                       f'{details}')
            if ood_policy == 'error':
                raise ValueError(message)
            print(f'WARNING: {message}')
            if ood_policy == 'clamp':
                cond_n = cond_n.clamp(-max_condition_z, max_condition_z)
                print('Condition input was clamped; requested raw values remain in metadata.')

        extrapolated = []
        cond_min = fm_ckpt.get('cond_min')
        cond_max = fm_ckpt.get('cond_max')
        if cond_min is not None and cond_max is not None:
            lo = cond_min.squeeze(0).cpu()
            hi = cond_max.squeeze(0).cpu()
            extrapolated = [fm_ckpt['cond_names'][i] for i in range(cond_dim)
                            if raw[i] < lo[i] or raw[i] > hi[i]]
            if extrapolated:
                print(f'Extrapolated dimensions: {extrapolated}')

        total_candidates = num_samples * candidate_multiplier
        cond = cond_n.unsqueeze(0).repeat(total_candidates, 1).to(device)
        target = raw.numpy().astype(np.float64)
        cond_std_np = cond_std.numpy().astype(np.float64)
        condition_request = {
            'normalized': dict(zip(fm_ckpt['cond_names'], cond_n.tolist())),
            'max_condition_z': max_condition_z,
            'ood_policy': ood_policy,
            'extrapolated_dimensions': extrapolated,
        }
        print(f'Conditional generation: {dict(zip(fm_ckpt["cond_names"], cond_values))} '
              f'(cfg_scale={cfg_scale}, candidates={total_candidates})')
    else:
        print('Unconditional generation')

    # ---- Sample latents ----
    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    z_n = sample_latents(model, total_candidates, latent_flat_dim, device,
                         cond=cond, cfg_scale=cfg_scale, ode_steps=ode_steps,
                         generator=generator)
    latent_clip = float(config.get('latent_clip', 0.0))
    clipped_latent_fraction = 0.0
    if latent_clip > 0:
        clipped_latent_fraction = float((z_n.abs() > latent_clip).float().mean().item())
        z_n = z_n.clamp(-latent_clip, latent_clip)
    z = z_n * fm_ckpt['latent_std'].to(device) + fm_ckpt['latent_mean'].to(device)
    print(f'Sampled {total_candidates} latents in {time.time() - t0:.2f}s '
          f'({ode_steps} ODE steps)')

    # ---- Decode and export ----
    candidates = []
    for candidate_index in range(total_candidates):
        volume = decode_sdf_grid(
            vae, z[candidate_index:candidate_index + 1], resolution=resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        report['candidate_index'] = candidate_index
        if target is not None:
            _audit_report(report, fm_ckpt['cond_names'], target, cond_std_np)
        candidates.append((mesh, report))
        if total_candidates > num_samples and ((candidate_index + 1) % 16 == 0
                                               or candidate_index + 1 == total_candidates):
            print(f'  decoded candidates: {candidate_index + 1}/{total_candidates}')

    if target is not None and candidate_multiplier > 1:
        candidates.sort(key=lambda item: (
            item[1].get('condition_score') is None,
            item[1].get('condition_score') if item[1].get('condition_score') is not None
            else float('inf')))
    selected = candidates[:num_samples]

    results = []
    for i, (mesh, report) in enumerate(selected):
        report['index'] = i
        if report['valid']:
            path = os.path.join(out_dir, f'sample_{seed}_{i:03d}.stl')
            mesh.export(path)
            report['path'] = path
            score_text = (f' score={report["condition_score"]:.3f}'
                          if report.get('condition_score') is not None else '')
            print(f'  sample {i:03d}: watertight={report["watertight"]} '
                  f'faces={report["faces"]} extents={np.round(report["extents"], 3).tolist()}'
                  f'{score_text}'
                  f' -> {path}')
        else:
            print(f'  sample {i:03d}: NO ZERO CROSSING (rejected)')
        results.append(report)

    condition_audit = (_condition_summary(results, fm_ckpt['cond_names'], target)
                       if target is not None else None)

    meta = {
        'fm_modelpath': fm_path,
        'vae_modelpath': vae_path,
        'seed': seed,
        'cfg_scale': cfg_scale,
        'ode_steps': ode_steps,
        'mc_resolution': resolution,
        'cond_names': fm_ckpt['cond_names'],
        'cond_values': cond_values,
        'condition_request': condition_request,
        'candidate_multiplier': candidate_multiplier,
        'num_candidates': total_candidates,
        'latent_clip': latent_clip,
        'clipped_latent_fraction': clipped_latent_fraction,
        'condition_audit': condition_audit,
        'results': results,
    }
    meta_path = os.path.join(out_dir, f'sample_{seed}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    valid = sum(1 for r in results if r['valid'])
    watertight = sum(1 for r in results if r.get('watertight'))
    print(f'\nDone: {valid}/{num_samples} valid, {watertight}/{num_samples} watertight. '
          f'Metadata: {meta_path}')
    if condition_audit:
        print(f'Condition audit: median actual={condition_audit.get("median_actual")}')
        print(f'Condition audit: median relative error={condition_audit.get("median_rel_error")}')


def run_reconstruct(config, config_filename='config.txt'):
    import trimesh
    from general_modules.sdf_sampling import normalize_mesh

    device = resolve_device(config)
    vae_path = config.get('vae_modelpath', './outputs/sdfflow_vae.pth')
    vae, vae_ckpt = load_vae(vae_path, device)

    input_path = config.get('input_mesh')
    if not input_path:
        raise ValueError("reconstruct mode requires 'input_mesh' in the config")
    resolution = int(config.get('mc_resolution', 128))
    num_enc = int(vae_ckpt['config'].get('num_encoder_points', 4096))
    out_dir = config.get('output_dir', './outputs/recon')
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(input_path, force='mesh')
    mesh, _, _ = normalize_mesh(mesh)
    points, face_idx = trimesh.sample.sample_surface(mesh, num_enc, seed=0)
    normals = mesh.face_normals[face_idx]

    surface_points = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(device)
    surface_normals = torch.from_numpy(normals.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = vae.encode(surface_points, surface_normals)
        volume = decode_sdf_grid(vae, mu.flatten(1), resolution=resolution, device=device)

    recon = sdf_grid_to_mesh(volume)
    report = mesh_report(recon)
    if report['valid']:
        base = os.path.splitext(os.path.basename(input_path))[0]
        path = os.path.join(out_dir, f'{base}_recon.stl')
        recon.export(path)
        print(f'Reconstruction: watertight={report["watertight"]} faces={report["faces"]} -> {path}')
    else:
        print('Reconstruction failed: NO ZERO CROSSING')
