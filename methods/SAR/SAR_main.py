"""SAR (scale-autoregressive flow matching, Lino & Thuerey 2026) entrypoint.

    python SAR_main.py --config <flat config>

Run from this directory (the launcher sets cwd to methods/SAR/), so every path
in the config is spelled ../../dataset/... and ../../output/....

Modes (the `mode` key):
    train      VAE stage, then the SAR stage over it
    train_vae  the graph VAE only                  -> vae_modelpath
    train_sar  the SAR model over a trained VAE    -> modelpath
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

MODES = ('train', 'train_vae', 'train_sar', 'inference')


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
    print(f'resume_training: {path} is the finished VAE of this run; continuing with the SAR stage')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description='SAR: scale-autoregressive flow matching on mesh HDF5')
    parser.add_argument('--config', default='config.txt', help='flat key/value config file')
    args = parser.parse_args(argv)

    config = load_config(args.config)
    mode = str(config.get('mode', 'train')).lower()
    if str(config.get('model', 'sar')).lower() != 'sar':
        raise SystemExit(f"model must be 'sar' for SAR_main.py, got {config.get('model')!r}")
    if mode not in MODES:
        raise SystemExit(f'Unsupported mode {mode!r}; expected one of {MODES}')
    gpu_ids = config.get('gpu_ids', 0)
    if isinstance(gpu_ids, (list, tuple)) and len(gpu_ids) > 1:
        reject_multi_process(config, 'SAR with several gpu_ids')

    print('SAR - scale-autoregressive flow matching (Lino & Thuerey 2026)')
    print(f'Config: {os.path.abspath(args.config)}')
    print(f'Mode  : {mode}')
    t0 = time.time()
    if mode in ('train', 'train_vae', 'train_sar'):
        from training_profiles.train import train_sar, train_vae
        if mode == 'train_vae' or (mode == 'train' and not _vae_stage_done(config)):
            train_vae(config, args.config)
        if mode in ('train', 'train_sar'):
            train_sar(config, args.config)
    else:
        from inference_profiles.sampler import run_inference
        run_inference(config, args.config)
    print(f'\nDone in {time.time() - t0:.1f}s')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
