# SDFFlow
import os

# Must be set before h5py is imported transitively by data loading modules.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse

from general_modules.load_config import load_config


def main():
    parser = argparse.ArgumentParser(description='SDFFlow: SDF-VAE + flow-matching geometry generator')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to config file (default: config.txt)')
    args = parser.parse_args()

    print('\n' * 2)
    print("""
    SDFFlow
    SDF-VAE + flow-matching geometry generation
    """)
    print(' ' * 64 + 'Version 0.1.0, 2026-07-17')
    print(' ' * 50 + 'Developed by SiHun Lee, Ph. D., MX, SEC')
    print()

    config = load_config(args.config)
    run_mode = config.get('mode')
    valid_modes = ('train', 'train_vae', 'train_fm', 'sample', 'reconstruct', 'interpolate')
    if run_mode not in valid_modes:
        raise ValueError(f"Unsupported mode '{run_mode}'. Supported: {valid_modes}. "
                         f"(Datasets are built with build_dataset.py, not a mode.)")

    print(f'           Config file   : {args.config}')
    print(f"           Selected Model: {config.get('model', 'SDFFlow')}")
    print(f'           Running in    : {run_mode} mode')
    print(f"Current absolute path: {os.path.abspath('.')}")
    print()

    if run_mode == 'train':
        from training_profiles.train_pipeline import train_pipeline
        train_pipeline(config, args.config)
    elif run_mode == 'train_vae':
        from training_profiles.train_vae import vae_worker
        vae_worker(config, args.config)
    elif run_mode == 'train_fm':
        from training_profiles.train_fm import fm_worker
        fm_worker(config, args.config)
    elif run_mode == 'sample':
        from inference_profiles.sample import run_sample
        run_sample(config, args.config)
    elif run_mode == 'reconstruct':
        from inference_profiles.sample import run_reconstruct
        run_reconstruct(config, args.config)
    elif run_mode == 'interpolate':
        from inference_profiles.interpolate import run_interpolate
        run_interpolate(config, args.config)


if __name__ == '__main__':
    main()
