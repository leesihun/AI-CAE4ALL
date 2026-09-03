"""Score a DETERMINISTIC MeshGraphNets checkpoint with the exact metric code
eval_prediction_modes.py uses, so the two numbers are comparable.

The deterministic tree reports its own validation loss, but a head-to-head
against cHI-MGNflow has to go through one metric implementation on one split --
otherwise a difference in reduction, weighting or normalisation shows up as a
difference in method. This script therefore duplicates `_metrics` verbatim
rather than importing it: the two repos have colliding module names
(`model/`, `general_modules/`), so it must run as its own process with the
MeshGraphNets repo at the front of sys.path.

    python methods/HI_MGNFlow/misc/eval_det_baseline.py \\
        --config configs/HI_MGNFlow/wave0/config_wave0_det.txt --split val
"""
import argparse
import os
import sys

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

SUITE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DET_REPO = os.path.join(SUITE, 'methods', 'MeshGraphNets')
sys.path.insert(0, DET_REPO)
os.chdir(DET_REPO)

import torch                                                   # noqa: E402
from torch_geometric.loader import DataLoader                   # noqa: E402

from general_modules.load_config import load_config             # noqa: E402
from model.MeshGraphNets import MeshGraphNets                   # noqa: E402
from training_profiles.setup import build_dataset_splits        # noqa: E402


def _metrics(pred, target):
    """Byte-for-byte the same reduction as eval_prediction_modes._metrics."""
    p, t = pred.reshape(-1).double(), target.reshape(-1).double()
    mse = torch.mean((p - t) ** 2).item()
    var = torch.var(t, unbiased=False).item()
    r2 = 1.0 - mse / var if var > 1e-12 else float('nan')
    pc, tc = p - p.mean(), t - t.mean()
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
    ap.add_argument('--max-graphs', type=int, default=0)
    args = ap.parse_args()

    config = load_config(os.path.join(SUITE, args.config)
                         if not os.path.isabs(args.config) else args.config)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        gid = config.get('gpu_ids', 0)
        gid = (gid[0] if isinstance(gid, list) and gid else gid)
        try:
            gid = int(gid)
        except (TypeError, ValueError):
            gid = 0
        if not 0 <= gid < torch.cuda.device_count():
            gid = 0
        torch.cuda.set_device(gid)
        device = torch.device(f'cuda:{gid}')
    print(f"device: {device}")

    ckpt = torch.load(config['modelpath'], map_location=device, weights_only=False)
    dataset = _load_dataset(config, args.split, args.dataset, ckpt)
    loader = list(DataLoader(dataset, batch_size=1, shuffle=False))
    if args.max_graphs:
        loader = loader[:args.max_graphs]
    if 'model_config' in ckpt:
        config.update(ckpt['model_config'])
    model = MeshGraphNets(config, str(device)).to(device)
    if 'ema_state_dict' in ckpt:
        sd = {k[len('module.'):]: v for k, v in ckpt['ema_state_dict'].items()
              if k.startswith('module.')}
        model.load_state_dict(sd)
        src = 'EMA'
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        src = 'training'
    model.eval()
    print(f"loaded {src} weights, epoch {ckpt.get('epoch', '?')}, "
          f"{sum(p.numel() for p in model.parameters()):,} params")

    acc = [0.0, 0.0, 0.0]
    n = 0
    with torch.no_grad():
        for graph in loader:
            graph = graph.to(device)
            out = model(graph, add_noise=False)
            pred = out[0] if isinstance(out, (tuple, list)) else out
            for i, v in enumerate(_metrics(pred.float(), graph.y.float())):
                acc[i] += v
            n += 1

    mse, r2, corr = (v / max(n, 1) for v in acc)
    print(f"\n{'mode':<14}{'fwd/graph':>10}{'MSE':>12}{'R2':>9}{'corr':>9}")
    print('-' * 54)
    print(f"{'HI-MGN (det)':<14}{1:>10}{mse:>12.4e}{r2:>9.4f}{corr:>9.4f}")
    print('-' * 54)
    print(f"{n} graphs from the {args.split} split.")


if __name__ == '__main__':
    main()
