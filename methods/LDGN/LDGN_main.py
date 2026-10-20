"""LDGN (latent diffusion graph net; Lino, Pfaff & Thuerey, ICLR 2025) entrypoint.

    python LDGN_main.py --config <flat config>

Run from this directory (the launcher sets cwd to methods/LDGN/), so every path
in the config is spelled ../../dataset/... and ../../output/....

Modes (the `mode` key):
    train       VGAE stage, then the LDGN stage over it
    train_vae   the variational graph autoencoder   -> vae_modelpath
    train_ldgn  the latent diffusion model over a
                trained VGAE                        -> modelpath
    inference  num_samples draws per held-out system, MGN-V rollout format
"""
import argparse
import os
import sys
import time

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from general_modules.load_config import load_config  # noqa: E402
from general_modules.resume_state import reject_multi_process  # noqa: E402

MODES = ('train', 'train_vae', 'train_ldgn', 'inference')


def _vae_stage_done(config):
    """`mode train` resumed after its VAE stage finished: keep that VAE.

    Only with `resume_training true`, and only when vae_modelpath is a finished
    VAE checkpoint (no `.resume` beside it) written under this same config.
    Otherwise the VAE is retrained, as without the key.
    """
    if not bool(config.get('resume_training', False)):
        return False
    path = config.get('vae_modelpath')
    if not path or not os.path.exists(path) or os.path.exists(f'{path}.resume'):
        return False
    import torch
    from general_modules.resume_state import config_fingerprint
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return False
    if ckpt.get('stage') != 'vae' or config_fingerprint(ckpt.get('config', {})) != config_fingerprint(config):
        return False
    print(f'resume_training: {path} is the finished VAE of this run; continuing with the LDGN stage')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='LDGN: latent diffusion graph net on mesh HDF5')
    parser.add_argument('--config', default='config.txt', help='flat key/value config file')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    mode = str(config.get('mode', 'train')).lower()
    if str(config.get('model', 'ldgn')).lower() != 'ldgn':
        raise SystemExit(f"model must be 'ldgn' for LDGN_main.py, got {config.get('model')!r}")
    if mode not in MODES:
        raise SystemExit(f'Unsupported mode {mode!r}; expected one of {MODES}')
    gpu_ids = config.get('gpu_ids', 0)
    if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) > 1:
        reject_multi_process(config, 'LDGN with several gpu_ids')

    print('LDGN - latent diffusion graph net (Lino, Pfaff & Thuerey; dgn4cfd)')
    print(f'Config: {os.path.abspath(args.config)}')
    print(f'Mode  : {mode}')
    t0 = time.time()
    if mode in ('train', 'train_vae', 'train_ldgn'):
        from training_profiles.train import train_ldgn, train_vae
        if mode == 'train_vae' or (mode == 'train' and not _vae_stage_done(config)):
            train_vae(config, args.config)
        if mode in ('train', 'train_ldgn'):
            train_ldgn(config, args.config)
    else:
        from inference_profiles.sampler import run_inference
        run_inference(config, args.config)
    print(f'\nDone in {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
