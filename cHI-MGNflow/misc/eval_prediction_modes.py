"""Compare every prediction mode a cHI-MGNflow checkpoint supports, on one split.

A flow-matching checkpoint is not only a sampler. Four different predictions can
be read out of it, and they cost wildly different amounts:

    mean-1step   z0 + v(z0, 0)                    1 forward     E[y|g], analytic
    mean-ens-M   average of M integrated draws    M*K*2 fwd     E[y|g], estimated
    draw-1       one integrated draw              K*2 forwards  a SAMPLE
    (baseline)   a deterministic HI-MGN checkpoint 1 forward    E[y|g], trained for it

`mean-1step` is exact in the optimum: at t=0 the path point is the noise field
itself, so the regression optimum is E[y|g] - s*z0 and one Euler step of size 1
returns E[y|g] to within sigma_min. See model/flow.py::predict_mean.

`draw-1` is EXPECTED to score worse on MSE than either mean. That is not a
defect -- a calibrated ensemble member is not supposed to sit on the mean. It is
reported so the gap between it and the means is visible; that gap IS the
conditional spread the model believes in.

Usage:
    python misc/eval_prediction_modes.py --config path/to/config_train.txt \\
        [--split val] [--steps 20] [--ensemble 8] [--max-graphs 0]
"""
import argparse
import os
import sys
import time

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from general_modules.load_config import load_config
from model.flow import integrate, predict_mean, resolve_flow_config
from training_profiles.setup import build_dataset_splits


def resolve_device(cfg):
    """Use the GPU the config actually asked for.

    Nothing in this repo sets CUDA_VISIBLE_DEVICES and the trainer treats
    gpu_ids as PHYSICAL indices, so a hardcoded cuda:0 would score on a
    different card than the run trained on.
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    gid = cfg.get('gpu_ids', 0)
    if isinstance(gid, list):
        gid = gid[0] if gid else 0
    try:
        gid = int(gid)
    except (TypeError, ValueError):
        gid = 0
    if not 0 <= gid < torch.cuda.device_count():
        gid = 0
    torch.cuda.set_device(gid)
    return torch.device(f'cuda:{gid}')


def load_flow_model(config, device):
    from model.CHiMGNFlow import CHiMGNFlow
    ckpt = torch.load(config['modelpath'], map_location=device, weights_only=False)
    if 'model_config' in ckpt:
        for k, v in ckpt['model_config'].items():
            # flow_steps / flow_solver are sampling-time; the CLI wins.
            if k not in ('flow_steps', 'flow_solver'):
                config[k] = v
    model = CHiMGNFlow(config, str(device)).to(device)
    if 'ema_state_dict' in ckpt:
        sd = {k[len('module.'):]: v for k, v in ckpt['ema_state_dict'].items()
              if k.startswith('module.')}
        model.load_state_dict(sd)
        src = 'EMA'
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        src = 'training'
    model.eval()
    print(f"  loaded {src} weights, epoch {ckpt.get('epoch', '?')}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model


def _metrics(pred, target):
    """MSE, per-channel R^2, and correlation. All on the normalized field."""
    p, t = pred.reshape(-1).double(), target.reshape(-1).double()
    mse = torch.mean((p - t) ** 2).item()
    var = torch.var(t, unbiased=False).item()
    r2 = 1.0 - mse / var if var > 1e-12 else float('nan')
    pc = p - p.mean()
    tc = t - t.mean()
    denom = (pc.norm() * tc.norm()).item()
    corr = (torch.dot(pc, tc).item() / denom) if denom > 1e-12 else float('nan')
    return mse, r2, corr



def _load_dataset(config, split, dataset_override=None, ckpt=None):
    """Build the evaluation graphs, with the inference normalization contract.

    Without an override this is the plain train/val/test split of `dataset_dir`,
    whose statistics are fit on the train portion -- i.e. exactly what the model
    saw. With an override (a separate inference file) the split statistics would
    be fit on the TEST data, which inference never does; rollout.py reads them
    from the checkpoint instead. So the checkpoint block is injected here to
    reproduce that contract, and the whole file is used rather than a sub-split.
    """
    from training_profiles.setup import build_dataset_splits
    import numpy as np

    if dataset_override:
        config = dict(config)
        config['dataset_dir'] = dataset_override

    tr, va, te = build_dataset_splits(config, int(config.get('split_seed', 42)))

    if dataset_override and ckpt is not None and 'normalization' in ckpt:
        norm = ckpt['normalization']
        for ds in (tr, va, te):
            for k in ('node_mean', 'node_std', 'edge_mean', 'edge_std',
                      'delta_mean', 'delta_std'):
                if k in norm:
                    setattr(ds, k, np.asarray(norm[k]).copy())
            if 'coarse_edge_means' in norm:
                ds.coarse_edge_means = [np.asarray(m).copy()
                                        for m in norm['coarse_edge_means']]
                ds.coarse_edge_stds = [np.asarray(s).copy()
                                       for s in norm['coarse_edge_stds']]
        print("  normalization: taken from the CHECKPOINT (inference contract)")

    if split == 'all':
        from torch.utils.data import ConcatDataset
        out = ConcatDataset([tr, va, te])
        # delta stats are read off the dataset by some consumers
        out.delta_mean, out.delta_std = tr.delta_mean, tr.delta_std
        return out
    return {'train': tr, 'val': va, 'test': te}[split]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='val',
                    choices=['train', 'val', 'test', 'all'])
    ap.add_argument('--dataset', default=None,
                    help='score a different HDF5 (e.g. the inference file); the '
                         'checkpoint normalization is then used, as inference does')
    ap.add_argument('--steps', type=int, default=None, help='K for the integrated modes')
    ap.add_argument('--solver', default=None, choices=['heun', 'euler'])
    ap.add_argument('--ensemble', type=int, default=8, help='M for mean-ens')
    ap.add_argument('--max-graphs', type=int, default=0, help='0 = whole split')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    device = resolve_device(config)
    print(f"device: {device}")

    ckpt_peek = torch.load(config['modelpath'], map_location='cpu', weights_only=False)
    dataset = _load_dataset(config, args.split, args.dataset, ckpt_peek)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("\nLoading checkpoint...")
    model = load_flow_model(config, device)

    flow_cfg = resolve_flow_config(config)
    K = int(args.steps or flow_cfg['steps'])
    solver = args.solver or flow_cfg['solver']
    M = int(args.ensemble)
    per_eval = 2 if solver == 'heun' else 1
    print(f"\nsplit={args.split}  graphs={len(dataset)}  K={K}  solver={solver}  M={M}")
    print(f"forward cost per graph: mean-1step 1 | draw-1 {K*per_eval} | "
          f"mean-ens {M*K*per_eval}")

    acc = {k: [0.0, 0.0, 0.0] for k in ('mean-1step', 'mean-ens', 'draw-1')}
    timing = {k: 0.0 for k in acc}
    n = 0

    with torch.no_grad():
        for gi, graph in enumerate(loader):
            if args.max_graphs and gi >= args.max_graphs:
                break
            graph = graph.to(device)
            target = graph.y.float()

            def velocity(y_cur, t_scalar):
                t = torch.full((1, 1), float(t_scalar), device=y_cur.device,
                               dtype=y_cur.dtype)
                return model(graph, y_cur, t).float()

            # ── mean-1step: one forward ─────────────────────────────────────
            z0 = torch.randn_like(target)
            t0 = time.time()
            pred = predict_mean(velocity, z0)
            timing['mean-1step'] += time.time() - t0
            for i, v in enumerate(_metrics(pred, target)):
                acc['mean-1step'][i] += v

            # ── integrated draws: reused for draw-1 and mean-ens ────────────
            t0 = time.time()
            draws = []
            for _ in range(M):
                draws.append(integrate(velocity, torch.randn_like(target), K, solver))
            elapsed = time.time() - t0
            timing['mean-ens'] += elapsed
            timing['draw-1'] += elapsed / max(M, 1)

            for i, v in enumerate(_metrics(draws[0], target)):
                acc['draw-1'][i] += v
            for i, v in enumerate(_metrics(torch.stack(draws).mean(0), target)):
                acc['mean-ens'][i] += v

            n += 1

    print(f"\n{'mode':<14}{'fwd/graph':>10}{'MSE':>12}{'R2':>9}{'corr':>9}{'s/graph':>10}")
    print('-' * 64)
    order = [('mean-1step', 1), ('mean-ens', M * K * per_eval), ('draw-1', K * per_eval)]
    for name, cost in order:
        mse, r2, corr = (v / n for v in acc[name])
        print(f"{name:<14}{cost:>10}{mse:>12.4e}{r2:>9.4f}{corr:>9.4f}"
              f"{timing[name] / n:>10.3f}")
    print('-' * 64)
    print(f"{n} graphs from the {args.split} split.")
    print("\ndraw-1 scoring worse than the means is expected, not a defect: a")
    print("calibrated ensemble member is not supposed to sit on the mean. The")
    print("gap between draw-1 and mean-ens is the spread the model believes in.")


if __name__ == '__main__':
    main()
