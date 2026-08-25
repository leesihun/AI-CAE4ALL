"""Generative-distribution diagnostic for a trained MeshGraphNets-V VAE checkpoint.

Real data has exactly ONE ground-truth field per geometry (the simulation only ran
once), so there is no way to directly compare a "generated distribution" to a "true
distribution" per sample. The rank histogram sidesteps this: for each held-out
geometry, draw K samples from the model, reduce each to a scalar statistic (default:
max z-displacement), and record where the true value RANKS among the K generated
values (0..K). If the model's distribution matches reality, that rank is uniformly
distributed over held-out geometries -- this is the standard verification-rank /
PIT-histogram diagnostic used for ensemble calibration. It needs no synthetic label
and works on any dataset.

Also reports the "wild rate": the fraction of (geometry, draw) pairs whose min/max
falls outside a margin around the observed data envelope -- a direct measurement of
the "totally unreasonable" complaint, independent of calibration shape.

Usage:
    python misc/eval_distribution.py --config path/to/config_train9.txt --split test \
        [--k 50] [--stat max_z] [--feature-idx 2] [--n-graphs 0]

    --stat max_z (default) or mean or l2: reduction applied to the requested feature
        column over each graph's nodes.
    --n-graphs 0 (default) uses every graph in the split; cap it for a quick check.

Interpreting the histogram (K+1 bins):
    flat / uniform  -> calibrated (this is the target)
    U-shaped        -> under-dispersed: truth is often outside the generated range
                       (raise gamma_es / es_samples, check gamma_es warm-start epoch,
                       or check for the exact failure this tool was built to catch:
                       a conditional-prior/N(0,I) mismatch between train and inference)
    dome-shaped     -> over-dispersed: generated spread is too wide (lower gamma_es,
                       check for a poorly-conditioned prior amplifying noise --
                       see MeshGraphNets - variational note on es_noise_source)
    skewed to one side -> biased location, independent of spread
"""
import argparse
import os
import sys

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import numpy as np
import torch
from torch_geometric.data import Batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from general_modules.load_config import load_config
from general_modules.data_loader import load_data
from inference_profiles.rollout import _load_model_from_checkpoint, _load_conditional_prior


STATS = {
    'max_z': lambda f: f[:, -1].max(),
    'mean':  lambda f: f.mean(),
    'l2':    lambda f: (f ** 2).sum() ** 0.5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--k', type=int, default=50)
    ap.add_argument('--stat', default='max_z', choices=list(STATS))
    ap.add_argument('--feature-idx', type=int, default=-1,
                     help='column of the model OUTPUT (not raw nodal_data) to reduce; '
                          'default -1 = last output channel (e.g. z_disp when '
                          'output_var is [x_disp,y_disp,z_disp]).')
    ap.add_argument('--n-graphs', type=int, default=0, help='0 = use every graph')
    ap.add_argument('--sampler', default='auto', choices=['auto', 'prior', 'normal'],
                     help="auto = use the conditional prior if the checkpoint has one, "
                          "else N(0,I); force with 'prior' or 'normal'.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['num_timesteps'] = 1
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ds = load_data(cfg)
    tr, va, te = ds.split(0.8, 0.1, 0.1, seed=int(cfg['split_seed']))
    sub = {'train': tr, 'val': va, 'test': te}[args.split]
    for s in (tr, va, te):
        s.augment_geometry = False
    dm = torch.tensor(tr.delta_mean, device=dev)
    dst = torch.tensor(tr.delta_std, device=dev)

    ck = torch.load(cfg['modelpath'], map_location=dev, weights_only=False)
    c2 = dict(cfg)
    for k, v in ck['model_config'].items():
        c2[k] = v
    model = _load_model_from_checkpoint(c2, ck, dev)

    has_prior = str(c2.get('prior_type', '')).lower().strip() == 'gnn_e2e'
    # A checkpoint can have prior_type=gnn_e2e (module built) yet never have
    # trained it (prior_nll_weight 0 / use_conditional_prior False) -- e.g. a
    # recipe that scores gamma_es against N(0,I) instead. Trust the flag that
    # says whether rollout.py actually samples from it, not module presence.
    prior_was_used = has_prior and bool(c2.get('use_conditional_prior', False))
    use_prior = prior_was_used if args.sampler == 'auto' else (args.sampler == 'prior')
    if use_prior and not has_prior:
        raise SystemExit("--sampler prior requested but this checkpoint has no conditional prior")
    if args.sampler == 'auto' and has_prior and not prior_was_used:
        print("  [note] checkpoint has a prior_type=gnn_e2e module but "
              "use_conditional_prior=False -- it was never trained; sampling "
              "N(0,I) instead (pass --sampler prior to force the untrained module).")
    prior = _load_conditional_prior(c2, ck, model, dev) if use_prior else None
    nz, zd = model.model.num_z, int(c2['vae_latent_dim'])

    n = len(sub) if args.n_graphs <= 0 else min(args.n_graphs, len(sub))
    graphs = [sub[i] for i in range(n)]
    g = Batch.from_data_list(graphs).to(dev)
    nn_per_graph = g.x.shape[0] // n
    n_feat = int(c2['output_var'])
    fi = args.feature_idx if args.feature_idx >= 0 else n_feat + args.feature_idx
    gt = (g.y * dst + dm).reshape(n, nn_per_graph, n_feat)
    stat_fn = STATS[args.stat]
    gt_stat = np.array([float(stat_fn(gt[i].cpu().numpy()[:, [fi]] if args.stat != 'l2'
                                       else gt[i].cpu().numpy()))
                         for i in range(n)])
    gt_lo, gt_hi = float(gt[:, :, fi].min()), float(gt[:, :, fi].max())
    margin = 0.5 * (gt_hi - gt_lo)

    K = args.k
    torch.manual_seed(1234)
    with torch.no_grad():
        zc = (prior.sample_n(g, K, temperature=1.0) if use_prior
              else torch.randn(n, K, nz, zd, device=dev))

    ranks = np.zeros(n, dtype=int)
    wild = 0
    gen_stats = np.zeros((K, n))
    for j in range(K):
        with torch.no_grad():
            gg = Batch.from_data_list(graphs).to(dev)
            gg.y = None
            p, *_ = model(gg, add_noise=False, use_posterior=False, fixed_z=zc[:, j])
            v = (p.float() * dst + dm).reshape(n, nn_per_graph, n_feat)
        vf = v[:, :, fi]
        vlo = vf.min(-1).values.cpu().numpy()
        vhi = vf.max(-1).values.cpu().numpy()
        wild += int(((vlo < gt_lo - margin) | (vhi > gt_hi + margin)).sum())
        for i in range(n):
            s = float(stat_fn(v[i].cpu().numpy()[:, [fi]] if args.stat != 'l2'
                               else v[i].cpu().numpy()))
            gen_stats[j, i] = s
    for i in range(n):
        ranks[i] = int((gen_stats[:, i] < gt_stat[i]).sum())

    hist = np.bincount(ranks, minlength=K + 1)
    expected = n / (K + 1)
    chi2 = float(((hist - expected) ** 2 / max(expected, 1e-9)).sum())
    dof = K  # K+1 bins, 1 constraint
    # Rough uniform-chi2 5% critical value via Wilson-Hilferty approx (no scipy dep).
    crit = dof * (1 - 2 / (9 * dof) + 1.645 * (2 / (9 * dof)) ** 0.5) ** 3

    print("DIAGNOSTIC cfg=%s split=%s n_graphs=%d K=%d sampler=%s stat=%s feature_idx=%d" %
          (os.path.basename(args.config), args.split, n, K,
           'prior' if use_prior else 'N(0,I)', args.stat, fi))
    print("  GT envelope [%.1f, %.1f]  margin +/-%.1f" % (gt_lo, gt_hi, margin))
    print("  WILD RATE = %d / %d (%.1f%%)" % (wild, K * n, 100.0 * wild / (K * n)))
    print("  RANK HISTOGRAM (%d bins, expect ~%.1f each): %s" %
          (K + 1, expected, hist.tolist()))
    print("  chi2 = %.1f  (uniform-fit 5%% critical ~%.1f; below = can't reject uniform)" %
          (chi2, crit))
    shape = ('U-shaped (under-dispersed)' if hist[0] + hist[-1] > 2 * expected * 2
             else 'dome-shaped (over-dispersed)' if hist[K // 2] > 2 * expected
             else 'skewed' if abs(hist[:len(hist) // 2].sum() - hist[len(hist) // 2:].sum()) > n * 0.3
             else 'roughly flat')
    print("  shape: %s" % shape)


if __name__ == '__main__':
    main()
