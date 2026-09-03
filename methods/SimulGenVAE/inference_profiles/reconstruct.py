"""Inference: conditions -> latent conditioner -> hierarchical VAE decode.

Loads the trained VAE and latent-conditioner checkpoints, maps each sample's
physical conditions to the dual latent (main + hierarchical) via the LC, decodes
through the hierarchical VAE (``mode='fix'`` deterministic path), un-scales the
field, and writes one reconstruction HDF5 plus per-sample reconstruction MSE.
"""

import os

import h5py
import numpy as np
import torch

from general_modules.fom_dataset import (
    apply_minmax,
    invert_minmax,
    load_fom_from_hdf5,
    read_conditions,
)
from model.common import add_sn
from training_profiles.setup import build_lc, build_vae, load_checkpoint, resolve_device


def _load_fields_with_checkpoint_normalization(config, field_norm):
    """Scale held-out fields with the VAE training normalization."""
    required = {'field_min', 'field_max', 'feature_range'}
    missing = required.difference(field_norm or {})
    if missing:
        raise ValueError(
            f"VAE checkpoint normalization is missing {sorted(missing)}; "
            "held-out reconstruction cannot be scaled safely."
        )
    raw_ntc, sample_ids = load_fom_from_hdf5(config)       # [N, T, C]
    scaled_ntc = apply_minmax(raw_ntc, field_norm)
    data_nct = np.ascontiguousarray(np.transpose(scaled_ntc, (0, 2, 1)))
    return data_nct, sample_ids


def run_reconstruct(config, config_filename='config.txt'):
    device = resolve_device(config)

    # The checkpoint must be loaded before the held-out fields: re-fitting
    # min/max on inference data changes the decoder scale and leaks test stats.
    vae_ckpt = load_checkpoint(config.get('vae_modelpath', '../../output/simulgenvae/simulgenvae_vae.pth'), device)
    lc_ckpt = load_checkpoint(config.get('lc_modelpath', '../../output/simulgenvae/simulgenvae_lc.pth'), device)
    if vae_ckpt.get('stage') != 'vae':
        raise ValueError("vae_modelpath is not a 'vae' checkpoint.")
    if lc_ckpt.get('stage') != 'lc':
        raise ValueError("lc_modelpath is not an 'lc' checkpoint.")

    print('\nLoading dataset (ground truth + condition ordering)...')
    field_norm = vae_ckpt.get('normalization')
    data_nct, sample_ids = _load_fields_with_checkpoint_normalization(config, field_norm)
    num_channels, num_time = int(data_nct.shape[1]), int(data_nct.shape[2])
    if int(vae_ckpt.get('num_channels', num_channels)) != num_channels:
        raise ValueError(
            f"VAE checkpoint expects {vae_ckpt.get('num_channels')} channels, "
            f"but inference data provides {num_channels}."
        )
    if int(vae_ckpt.get('num_time', num_time)) != num_time:
        raise ValueError(
            f"VAE checkpoint expects {vae_ckpt.get('num_time')} timesteps, "
            f"but inference data provides {num_time}."
        )

    vae, num_filter_enc = build_vae(config, num_channels, num_time, int(config.get('batch_size', 16)))
    vae.apply(add_sn)   # match the spectral-norm parametrization saved by the VAE stage
    vae.load_state_dict(vae_ckpt['model_state'])
    vae = vae.to(device).eval()
    num_levels = len(num_filter_enc) - 1

    conditions, input_shape = read_conditions(config)
    lc = build_lc(config, input_shape, num_levels)
    lc.load_state_dict(lc_ckpt['model_state'])
    lc = lc.to(device).eval()

    lc_norm = lc_ckpt['normalization']
    if lc_norm.get('input') is not None:
        conditions = apply_minmax(conditions, lc_norm['input'])

    num_var = int(config.get('num_var', 1))
    n = data_nct.shape[0]
    batch_size = int(config.get('batch_size', 16))
    out_dir = config.get('output_dir', '../../output/simulgenvae/reconstruct')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'reconstructions.h5')

    mse_sum = 0.0
    with h5py.File(out_path, 'w') as out, torch.no_grad():
        out.attrs['num_samples'] = n
        out.attrs['num_var'] = num_var
        grp = out.create_group('data')
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cond = torch.from_numpy(np.float32(conditions[start:end])).to(device)
            pred_main_s, pred_hier_s = lc(cond)                     # [b,Dm], [b,L,Dh]
            main = invert_minmax(pred_main_s.cpu().numpy(), lc_norm['main'])
            hier = invert_minmax(pred_hier_s.cpu().numpy(), lc_norm['hier'])
            z = torch.from_numpy(np.float32(main)).to(device)
            xs = [torch.from_numpy(np.float32(hier[:, i, :])).to(device) for i in range(num_levels)]
            field_s, _ = vae.decoder(z, xs, mode='fix')             # [b, C, T]
            field_s = field_s.cpu().numpy()
            field = invert_minmax(np.transpose(field_s, (0, 2, 1)), field_norm)  # [b, T, C]

            # Ground truth in physical units for MSE.
            gt = invert_minmax(np.transpose(data_nct[start:end], (0, 2, 1)), field_norm)  # [b, T, C]
            mse_sum += float(np.mean((field - gt) ** 2)) * (end - start)

            for j in range(end - start):
                sid = sample_ids[start + j]
                # [T, C] -> [T, num_var, N] -> [num_var, T, N] to mirror nodal_data field rows.
                f = field[j].reshape(num_time, num_var, -1).transpose(1, 0, 2)
                grp.create_dataset(f'{sid}/nodal_field', data=np.float32(f))

    print(f'\nReconstruction complete. Mean field MSE: {mse_sum / max(n, 1):.6e}')
    print(f'Wrote {out_path}')
