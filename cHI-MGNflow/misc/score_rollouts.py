"""Score autoregressive rollouts against ground truth, PER STEP.

Every metric reported before this script was one-step (teacher-forced): each
graph was a single (sample, timestep) pair and the model predicted one step from
the true previous state. That measures the model's local accuracy and is blind
to the thing that actually decides a transient surrogate -- compounding error,
where a model is fed its OWN previous prediction and small errors accumulate.

A model can win at one step and lose badly over a rollout. This script is what
distinguishes the two.

    python misc/score_rollouts.py \\
        --gt ../dataset/ex9_infer.h5 \\
        --roll det=../output/ex9flow/roll_det fmx0=../output/ex9flow/roll_fmx0_mean \\
        [--out rollout_error.png] [--channels 0 1]

Rollout HDF5 layout (written by inference_profiles/rollout.py):
    /data/<sample_id>/nodal_data  [features, steps+1, nodes]
        rows 0:3      reference coordinates
        rows 3:3+F    the predicted state at each step; index 0 is the initial
                      condition, so predictions start at index 1.
"""
import argparse
import glob
import os
import re

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import h5py
import numpy as np


def load_rollout_dir(path, output_var):
    """{sample_id: [steps+1, nodes, F]} from one rollout output directory."""
    out = {}
    files = glob.glob(os.path.join(path, '**', '*.h5'), recursive=True)
    for f in files:
        m = re.search(r'rollout_sample(\d+)', os.path.basename(f))
        if not m:
            continue
        sid = int(m.group(1))
        with h5py.File(f, 'r') as h:
            grp = h['data']
            key = list(grp.keys())[0]
            nd = np.asarray(grp[key]['nodal_data'])      # [F_all, steps+1, N]
        out[sid] = np.transpose(nd[3:3 + output_var], (1, 2, 0))   # [steps+1, N, F]
    return out


def load_gt(path, output_var, input_var):
    """{sample_id: [T, nodes, F]} of the true state trajectory."""
    out = {}
    with h5py.File(path, 'r') as f:
        for k in f['data'].keys():
            nd = np.asarray(f['data'][k]['nodal_data'])   # [F_all, T, N]
            out[int(k)] = np.transpose(nd[3:3 + output_var], (1, 2, 0))
    return out


def per_step_mse(roll, gt, channels=None):
    """MSE at each rollout step, averaged over samples and nodes.

    Both sides are in PHYSICAL units (the rollout file stores denormalised
    state), so the numbers are not comparable to the normalised one-step MSEs --
    only across models on this same scale.
    """
    sids = sorted(set(roll) & set(gt))
    if not sids:
        return None, None, []
    steps = min(min(roll[s].shape[0] for s in sids), min(gt[s].shape[0] for s in sids))
    acc = np.zeros(steps)
    ref = np.zeros(steps)
    for s in sids:
        r, g = roll[s][:steps], gt[s][:steps]
        if channels is not None:
            r, g = r[..., channels], g[..., channels]
        acc += ((r - g) ** 2).mean(axis=(1, 2))
        ref += (g ** 2).mean(axis=(1, 2))
    return acc / len(sids), ref / len(sids), sids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--roll', nargs='+', required=True, help='name=dir ...')
    ap.add_argument('--output-var', type=int, default=2)
    ap.add_argument('--input-var', type=int, default=2)
    ap.add_argument('--channels', type=int, nargs='*', default=None)
    ap.add_argument('--out', default=None, help='write a PNG of error vs step')
    args = ap.parse_args()

    gt = load_gt(args.gt, args.output_var, args.input_var)
    print(f'ground truth: {len(gt)} samples, {list(gt.values())[0].shape[0]} timesteps')

    curves = {}
    for spec in args.roll:
        name, path = spec.split('=', 1)
        roll = load_rollout_dir(path, args.output_var)
        if not roll:
            print(f'  {name}: no rollout files under {path} -- skipped')
            continue
        mse, ref, sids = per_step_mse(roll, gt, args.channels)
        if mse is None:
            print(f'  {name}: no overlapping sample ids -- skipped')
            continue
        curves[name] = (mse, ref)
        print(f'  {name}: {len(sids)} samples, {len(mse)} steps')

    if not curves:
        print('nothing to score')
        return

    steps = min(len(m) for m, _ in curves.values())
    print(f"\n{'step':>6}" + ''.join(f'{n:>14}' for n in curves))
    print('-' * (6 + 14 * len(curves)))
    # Step 0 is the shared initial condition, so it is identically zero for
    # every model and carries no information; start the table at step 1.
    for t in list(range(1, min(6, steps))) + [steps // 2, steps - 1]:
        if t < 1 or t >= steps:
            continue
        print(f'{t:>6}' + ''.join(f'{curves[n][0][t]:>14.4e}' for n in curves))
    print('-' * (6 + 14 * len(curves)))
    print(f"{'mean':>6}" + ''.join(
        f'{curves[n][0][1:steps].mean():>14.4e}' for n in curves))
    print(f"{'final':>6}" + ''.join(f'{curves[n][0][steps-1]:>14.4e}' for n in curves))

    print('\ngrowth from step 1 to the last step (compounding factor):')
    for n, (m, _) in curves.items():
        g = m[steps - 1] / max(m[1], 1e-30)
        print(f'  {n:<12} {m[1]:.4e} -> {m[steps-1]:.4e}   x{g:.1f}')

    if args.out:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        colors = {'det': '#A2650D', 'fmx0': '#0B7B85', 'fmv': '#7A5AA8',
                  'fmx0_sample': '#B5476E'}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        for n, (m, ref) in curves.items():
            xs = np.arange(1, steps)
            c = colors.get(n.split('_')[0] if n not in colors else n)
            ax1.plot(xs, m[1:steps], 'o-', label=n, lw=2, ms=4, color=c)
            ax2.plot(xs, m[1:steps] / np.maximum(ref[1:steps], 1e-30),
                     'o-', label=n, lw=2, ms=4, color=c)
        ax1.set_yscale('log'); ax1.set_xlabel('rollout step'); ax1.set_ylabel('MSE (physical units)')
        ax1.set_title('Compounding error over the rollout'); ax1.grid(alpha=.3); ax1.legend()
        ax2.set_yscale('log'); ax2.set_xlabel('rollout step')
        ax2.set_ylabel('MSE / mean(GT^2)   (relative)')
        ax2.set_title('Same, normalised by the field magnitude at each step')
        ax2.grid(alpha=.3); ax2.legend()
        fig.suptitle('ex9 autoregressive rollout -- the product path, not one-step')
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(args.out, dpi=150)
        print(f'\nsaved {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
