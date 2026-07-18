"""Latent interpolation between two reproducible flow-matching samples."""

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
from inference_profiles.sample import _model_state, load_vae
from model.velocity_net import VelocityNet, sample_latents
from training_profiles.setup import load_checkpoint, resolve_device


def _plot_triptych(meshes, labels, reports, path, dpi=180, max_faces=0):
    """Render three meshes with identical axes and camera settings."""
    fig = plt.figure(figsize=(16, 5.8), dpi=dpi, facecolor='white')
    colors = ('#3B82C4', '#E68A2E', '#3B82C4')

    for index, (mesh, label, report, color) in enumerate(
            zip(meshes, labels, reports, colors), start=1):
        ax = fig.add_subplot(1, 3, index, projection='3d')
        triangles = mesh.triangles
        if max_faces > 0 and len(triangles) > max_faces:
            selected = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
            triangles = triangles[selected]
        surface = Poly3DCollection(
            triangles, facecolor=color, edgecolor='none', linewidth=0.0, alpha=1.0)
        ax.add_collection3d(surface)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_zlim(-1.0, 1.0)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
        ax.set_proj_type('ortho')
        ax.set_axis_off()
        ax.set_title(
            f'{label}\nvolume={report["volume"]:.4f}, faces={report["faces"]:,}',
            fontsize=12, pad=0, y=0.90)

    fig.suptitle('SDFFlow latent interpolation: sample 0 to sample 1', fontsize=16, y=0.97)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.84, wspace=0.01)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)


@torch.no_grad()
def run_interpolate(config, config_filename='config.txt'):
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

    seed = int(config.get('seed', 0))
    index_a = int(config.get('sample_index_a', 0))
    index_b = int(config.get('sample_index_b', 1))
    alpha = float(config.get('alpha', 0.5))
    source_num_samples = int(config.get('source_num_samples', max(index_a, index_b) + 1))
    ode_steps = int(config.get('ode_steps', 50))
    resolution = int(config.get('mc_resolution', 128))
    out_dir = config.get('output_dir', './outputs/interpolation')

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
    source_z_n = sample_latents(
        model, source_num_samples, latent_flat_dim, device,
        ode_steps=ode_steps, generator=generator)
    z_a_n = source_z_n[index_a]
    z_b_n = source_z_n[index_b]
    z_mid_n = torch.lerp(z_a_n, z_b_n, alpha)
    selected_z_n = torch.stack((z_a_n, z_mid_n, z_b_n))
    selected_z = (selected_z_n * fm_ckpt['latent_std'].to(device)
                  + fm_ckpt['latent_mean'].to(device))
    print(f'Reproduced {source_num_samples} source latents and interpolated alpha={alpha:g} '
          f'in {time.time() - t0:.2f}s')

    alpha_tag = f'{alpha:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')
    names = (
        f'sample_{seed}_{index_a:03d}',
        f'sample_{seed}_{index_a:03d}_{index_b:03d}_alpha{alpha_tag}',
        f'sample_{seed}_{index_b:03d}',
    )
    labels = (
        f'Sample {index_a} (alpha=0.0)',
        f'Latent midpoint (alpha={alpha:g})',
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
        meshes.append(mesh)
        reports.append(report)
        print(f'  {labels[i]}: watertight={report["watertight"]} '
              f'faces={report["faces"]} volume={report["volume"]:.6f} -> {mesh_path}')

    plot_path = os.path.join(
        out_dir, f'interpolation_{index_a:03d}_{index_b:03d}_alpha{alpha_tag}.png')
    _plot_triptych(
        meshes, labels, reports, plot_path,
        dpi=int(config.get('plot_dpi', 180)),
        max_faces=int(config.get('plot_max_faces', 0)))

    endpoint_distance = float(torch.linalg.vector_norm(z_b_n - z_a_n).item())
    metadata = {
        'fm_modelpath': fm_path,
        'vae_modelpath': vae_path,
        'seed': seed,
        'source_num_samples': source_num_samples,
        'sample_index_a': index_a,
        'sample_index_b': index_b,
        'alpha': alpha,
        'interpolation_space': 'normalized_fm_latent',
        'ode_steps': ode_steps,
        'mc_resolution': resolution,
        'latent_distances': {
            'endpoint_l2': endpoint_distance,
            'a_to_interpolation_l2': float(torch.linalg.vector_norm(z_mid_n - z_a_n).item()),
            'interpolation_to_b_l2': float(torch.linalg.vector_norm(z_b_n - z_mid_n).item()),
        },
        'results': reports,
        'plot_path': plot_path,
    }
    metadata_path = os.path.join(
        out_dir, f'interpolation_{index_a:03d}_{index_b:03d}_alpha{alpha_tag}_meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f'Plot: {plot_path}')
    print(f'Metadata: {metadata_path}')
