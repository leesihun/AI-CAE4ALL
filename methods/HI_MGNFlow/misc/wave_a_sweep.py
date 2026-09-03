"""Wave A: sweep the integration axes on ONE trained checkpoint. Costs no training.

`flow_steps` (K) and `flow_solver` are sampling-time choices. What training
produces is a continuous velocity field v(y_t, t, g); K is only the quadrature
resolution applied to it afterwards, so the same weights integrate correctly at
K=4 or K=100. Sweeping them over training runs -- as the variational tree's
wave-3 had to do with every one of its axes -- would burn the entire arm budget
on a question that inference answers for free.

This script answers, for one checkpoint:

    * the smallest K whose CRPS is indistinguishable from the largest K
      (below that point the discretisation error dominates);
    * whether Heun's 2 evaluations per step buy more than doubling Euler's K;
    * what `val_flow_steps` should be for the Wave B sweep, which directly
      drives Wave B's wall clock because validation integrates every
      `val_interval`.

    python misc/wave_a_sweep.py --config configs/HI_MGNFlow/wave0/config_wave0_flow.txt \\
        [--split val] [--samples 8] [--max-graphs 0] [--ks 4 8 12 20 30 50]
"""
import argparse
import os
import sys
import time

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch
from torch_geometric.loader import DataLoader

from general_modules.load_config import load_config
from model.flow import integrate, predict_mean
from training_profiles.setup import build_dataset_splits

from eval_prediction_modes import load_flow_model, resolve_device, _metrics  # noqa: E402


def fair_crps(samples, target):
    """Fair (unbiased) CRPS, averaged over elements.

    Accumulated pair by pair; the vectorised [S,S,N,F] form is the largest
    single allocation anywhere in an evaluation pass on large meshes.
    """
    S = samples.shape[0]
    acc = torch.zeros_like(target)
    for i in range(S):
        acc += (samples[i] - target).abs()
    acc /= S
    if S >= 2:
        spread = torch.zeros_like(acc)
        for i in range(S - 1):
            for j in range(i + 1, S):
                spread += (samples[i] - samples[j]).abs()
        acc = acc - spread / (S * (S - 1))
    return float(acc.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    ap.add_argument('--samples', type=int, default=8, help='ensemble members per graph')
    ap.add_argument('--max-graphs', type=int, default=0)
    ap.add_argument('--ks', type=int, nargs='+', default=[4, 8, 12, 20, 30, 50])
    ap.add_argument('--solvers', nargs='+', default=['heun', 'euler'])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    config = load_config(args.config)
    device = resolve_device(config)
    splits = dict(zip(('train', 'val', 'test'),
                      build_dataset_splits(config, int(config.get('split_seed', 42)))))
    dataset = splits[args.split]
    loader = list(DataLoader(dataset, batch_size=1, shuffle=False))
    if args.max_graphs:
        loader = loader[:args.max_graphs]

    print("\nLoading checkpoint...")
    model = load_flow_model(config, device)
    print(f"\nsplit={args.split}  graphs={len(loader)}  members={args.samples}")

    rows = []
    with torch.no_grad():
        for solver in args.solvers:
            for K in args.ks:
                # Same seed for every cell: the cells then differ only by the
                # integrator, not by which noise fields happened to be drawn.
                torch.manual_seed(args.seed)
                crps_sum = spread_sum = mean_mse = 0.0
                t0 = time.time()
                for graph in loader:
                    graph = graph.to(device)
                    target = graph.y.float()

                    def velocity(y_cur, t_scalar):
                        t = torch.full((1, 1), float(t_scalar), device=y_cur.device,
                                       dtype=y_cur.dtype)
                        return model(graph, y_cur, t).float()

                    draws = torch.stack([
                        integrate(velocity, torch.randn_like(target), K, solver)
                        for _ in range(args.samples)
                    ])
                    crps_sum += fair_crps(draws, target)
                    spread_sum += float(draws.std(0).mean() / target.std().clamp_min(1e-8))
                    mean_mse += _metrics(draws.mean(0), target)[0]
                n = max(len(loader), 1)
                rows.append({
                    'solver': solver, 'K': K,
                    'fwd': K * (2 if solver == 'heun' else 1),
                    'crps': crps_sum / n, 'spread': spread_sum / n,
                    'ens_mse': mean_mse / n, 'sec': (time.time() - t0) / n,
                })
                r = rows[-1]
                print(f"  {solver:<6} K={K:<3} fwd/draw={r['fwd']:<3} "
                      f"crps={r['crps']:.4e}  spread={r['spread']:.3f}  "
                      f"ens_mse={r['ens_mse']:.4e}  {r['sec']:.2f}s/graph")

    # ── the deterministic mode, for reference: one forward, no integration ──
    torch.manual_seed(args.seed)
    det_mse = 0.0
    t0 = time.time()
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(device)
            target = graph.y.float()

            def velocity(y_cur, t_scalar):
                t = torch.full((1, 1), float(t_scalar), device=y_cur.device,
                               dtype=y_cur.dtype)
                return model(graph, y_cur, t).float()

            det_mse += _metrics(predict_mean(velocity, torch.randn_like(target)), target)[0]
    det_mse /= max(len(loader), 1)

    best = min(rows, key=lambda r: r['crps'])
    print("\n" + "=" * 66)
    print(f"best CRPS  : {best['crps']:.4e}  at {best['solver']} K={best['K']} "
          f"({best['fwd']} forwards/draw)")
    # 2% of the best is comfortably inside the estimator's own noise at these
    # graph counts, so anything within it is "the same answer, cheaper".
    ok = [r for r in rows if r['crps'] <= best['crps'] * 1.02]
    cheap = min(ok, key=lambda r: r['fwd'])
    print(f"cheapest within 2%: {cheap['solver']} K={cheap['K']} "
          f"({cheap['fwd']} forwards/draw, {best['fwd']/max(cheap['fwd'],1):.1f}x cheaper)")
    print(f"\nSet flow_steps to the production value and val_flow_steps to the")
    print(f"cheapest cell above -- validation integrates every val_interval, so")
    print(f"it lands directly in the Wave B wall clock.")
    print(f"\nmean-1step (1 forward, no integration): mse={det_mse:.4e}")
    print("=" * 66)


if __name__ == '__main__':
    main()
