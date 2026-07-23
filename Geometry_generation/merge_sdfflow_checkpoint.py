"""One-shot repo-side helper: merge a pre-existing two-file SDFFlow checkpoint
pair (separate VAE + FM `.pth`) into the single combined FM checkpoint that
`training_profiles/train_fm.py` now writes directly (see
INFERENCE_BUNDLE_PLAN.md section 5.5). Does not ship in the inference bundle.

Usage:
    python merge_sdfflow_checkpoint.py --vae outputs/ex1/sdfflow_vae.pth \
        --fm outputs/ex1/sdfflow_fm.pth --output outputs/ex1/sdfflow.pth

New checkpoints produced by the current train_fm.py already embed the VAE and
do not need this script.
"""

import argparse

import torch


def _state_dict_to_cpu(state_dict):
    if state_dict is None:
        return None
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in state_dict.items()}


def merge(vae_path: str, fm_path: str, output_path: str) -> None:
    vae_ckpt = torch.load(vae_path, map_location='cpu', weights_only=False)
    fm_ckpt = torch.load(fm_path, map_location='cpu', weights_only=False)

    if fm_ckpt.get('stage') != 'fm':
        raise ValueError(f"'{fm_path}' does not look like an FM checkpoint (stage={fm_ckpt.get('stage')!r}).")
    if vae_ckpt.get('stage') != 'vae':
        raise ValueError(f"'{vae_path}' does not look like a VAE checkpoint (stage={vae_ckpt.get('stage')!r}).")

    merged = dict(fm_ckpt)
    merged['schema_version'] = 'sdfflow_infer_v1'
    merged['vae'] = {
        'model_state': _state_dict_to_cpu(vae_ckpt['model_state']),
        'ema_state': _state_dict_to_cpu(vae_ckpt.get('ema_state')),
        'config': vae_ckpt['config'],
        'cond_mean': vae_ckpt.get('cond_mean'),
        'cond_std': vae_ckpt.get('cond_std'),
        'cond_names': vae_ckpt.get('cond_names'),
    }

    torch.save(merged, output_path)
    print(f"Merged checkpoint written to {output_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vae', required=True, help='Path to sdfflow_vae.pth')
    p.add_argument('--fm', required=True, help='Path to sdfflow_fm.pth')
    p.add_argument('--output', required=True, help='Path to write the combined .pth')
    args = p.parse_args()
    merge(args.vae, args.fm, args.output)


if __name__ == '__main__':
    main()
