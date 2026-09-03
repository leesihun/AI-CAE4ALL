# SimulGenVAE
import os

# Must be set before h5py is imported transitively by data loading modules.
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import argparse

from general_modules.load_config import load_config


def _train_dispatch(rank, world_size, gpu_ids, config, config_filename):
    """Run the requested training mode. Under distributed launch this executes
    inside each spawned rank (process group already initialized); single-process
    it is called once with rank 0 / world_size 1."""
    run_mode = config.get('mode')
    if run_mode == 'train':
        from training_profiles.train_pipeline import train_pipeline
        train_pipeline(config, config_filename)
    elif run_mode == 'train_vae':
        from training_profiles.train_vae import vae_worker
        vae_worker(config, config_filename)
    elif run_mode == 'train_lc':
        from training_profiles.train_lc import lc_worker
        lc_worker(config, config_filename)


def main():
    parser = argparse.ArgumentParser(
        description='SimulGenVAE: hierarchical VAE + latent conditioner for parametric simulation fields')
    parser.add_argument('--config', type=str, default='config.txt',
                        help='Path to config file (default: config.txt)')
    args = parser.parse_args()

    print('\n' * 2)
    print("""
    SimulGenVAE
    Hierarchical physics-aware VAE + latent conditioner
    """)
    print(' ' * 52 + 'Developed by SiHun Lee, Ph. D.')
    print()

    config = load_config(args.config)
    run_mode = config.get('mode')
    valid_modes = ('train', 'train_vae', 'train_lc', 'reconstruct')
    if run_mode not in valid_modes:
        raise ValueError(f"Unsupported mode '{run_mode}'. Supported: {valid_modes}.")

    print(f'           Config file   : {args.config}')
    print(f"           Selected Model: {config.get('model', 'SimulGenVAE')}")
    print(f'           Running in    : {run_mode} mode')
    print(f"Current absolute path: {os.path.abspath('.')}")
    print()

    # Multi-GPU: when parallel_mode is ddp and >1 GPU is requested, the training
    # modes self-spawn one worker per GPU (each runs _train_dispatch).
    from general_modules import distributed as D
    if run_mode in ('train', 'train_vae', 'train_lc') and D.should_distribute(config):
        D.spawn_workers(_train_dispatch, config, args.config)
        return

    if run_mode in ('train', 'train_vae', 'train_lc'):
        _train_dispatch(0, 1, D.resolve_gpu_ids(config), config, args.config)
    elif run_mode == 'reconstruct':
        from inference_profiles.reconstruct import run_reconstruct
        run_reconstruct(config, args.config)


if __name__ == '__main__':
    main()
