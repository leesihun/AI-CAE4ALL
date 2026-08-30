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
                       (raise posterior_min_std so the decoder is trained at the
                       noise level the prior actually delivers, or check for the
                       exact failure this tool was built to catch: a conditional-
                       prior/N(0,I) mismatch between train and inference)
    dome-shaped     -> over-dispersed: generated spread is too wide (raise
                       lambda_mmd, or check for a poorly-conditioned prior
                       amplifying noise -- try prior_fm_solver heun, since Euler
                       overshoots into the z tails)
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


def resolve_device(cfg):
    """Pick the GPU the config actually asked for.

    The trainer uses `gpu_ids` as PHYSICAL device indices
    (`torch.cuda.set_device(gpu_id)` in training_profiles/single_training.py) and
    nothing in this repo sets CUDA_VISIBLE_DEVICES, so a hardcoded `cuda:0` here
    scores on a different card than the one the arm trained on -- fighting
    whatever else lives on GPU 0 and misreporting where the numbers came from.
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
        print(f"  [note] gpu_ids={gid} is not visible ({torch.cuda.device_count()} "
              f"device(s)); falling back to cuda:0")
        gid = 0
    torch.cuda.set_device(gid)
    print(f"  device: cuda:{gid}  ({torch.cuda.get_device_name(gid)})")
    return torch.device(f'cuda:{gid}')


def graph_slices(flat, batch, n):
    """Split a batched [total_nodes, F] tensor into one numpy array per graph.

    Real CAE meshes do NOT share a node count -- SAOI's warpage geometries all
    differ -- so `total_nodes // n` is not a per-graph node count, and reshaping
    the batch to [n, nodes, F] raises a size error (or, when the counts happen
    to divide evenly, silently mixes two graphs' nodes into one row). Use the
    offsets PyG already recorded when it concatenated the batch.

    One host transfer per call, not one per graph.
    """
    ptr = getattr(batch, 'ptr', None)
    if ptr is None:
        counts = torch.bincount(batch.batch, minlength=n)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    ptr = ptr.tolist()
    arr = flat.detach().cpu().numpy()
    return [arr[ptr[i]:ptr[i + 1]] for i in range(n)]


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
    ap.add_argument('--chunk-size', type=int, default=8,
                     help='graphs materialized and scored at a time. Bounds peak '
                          'HOST and device memory; results are identical either '
                          'way. 0 = the whole split at once (old behaviour).')
    ap.add_argument('--sampler', default='auto', choices=['auto', 'prior', 'normal'],
                     help="auto = use the conditional prior if the checkpoint has one, "
                          "else N(0,I); force with 'prior' or 'normal'.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg['num_timesteps'] = 1
    dev = resolve_device(cfg)
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
    n_feat = int(c2['output_var'])
    fi = args.feature_idx if args.feature_idx >= 0 else n_feat + args.feature_idx
    stat_fn = STATS[args.stat]
    reduce = lambda f: float(stat_fn(f if args.stat == 'l2' else f[:, [fi]]))
    K = args.k
    chunk = args.chunk_size if args.chunk_size > 0 else n

    # Score in chunks of graphs. Materializing the WHOLE split at once was the
    # expensive part, and most of that cost is not on the GPU at all:
    # `sub[i]` reads the sample from HDF5, rebuilds its edge features and its
    # cached hierarchy, and normalizes it -- all on the CPU -- and every graph
    # so produced is held in host RAM until the batch is collated, which then
    # makes a second full copy. On SAOI-sized meshes a full split is tens of GB
    # of host memory before a single kernel launches, which is why the process
    # can sit at near-zero GPU utilization for a long time and still grow.
    #
    # Chunking bounds both: peak memory scales with `chunk`, not with split
    # size. The only cross-graph quantity is the ground-truth envelope, and that
    # is accumulated as a running min/max and applied to the stored per-draw
    # extrema once every chunk is in, so `wild` and the rank histogram are what
    # they would have been in one pass.
    gt_stat = np.zeros(n)
    gen_stats = np.zeros((K, n))
    gen_lo = np.zeros((K, n))
    gen_hi = np.zeros((K, n))
    gt_lo, gt_hi = float('inf'), float('-inf')
    torch.manual_seed(1234)
    # Draw the N(0,I) latents for the whole split up front -- they are tiny --
    # so `--sampler normal` gives the same numbers at any --chunk-size. The
    # conditional prior draws its own noise inside sample_n, so for
    # `--sampler prior` the specific draws do depend on the chunk size (they
    # stay reproducible for a given one).
    zc_all = None if use_prior else torch.randn(n, K, nz, zd, device=dev)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        graphs = [sub[i] for i in range(start, stop)]
        m = len(graphs)
        g = Batch.from_data_list(graphs).to(dev)
        del graphs                            # collated; the Data list is dead

        gt = g.y * dst + dm                   # [chunk_nodes, n_feat], physical
        for k, f in enumerate(graph_slices(gt, g, m)):
            gt_stat[start + k] = reduce(f)
        gt_lo = min(gt_lo, float(gt[:, fi].min()))
        gt_hi = max(gt_hi, float(gt[:, fi].max()))
        del gt

        with torch.no_grad():
            zc = (prior.sample_n(g, K, temperature=1.0) if use_prior
                  else zc_all[start:stop])

        # One collated batch serves every draw in this chunk: only `fixed_z`
        # differs, and the model never writes into its input (`add_noise=False`
        # skips the sole in-place block, and the encoder returns a fresh Data
        # that each processor step rebinds).
        g.y = None                            # gt already extracted
        for j in range(K):
            with torch.no_grad():
                p, *_ = model(g, add_noise=False, use_posterior=False,
                              fixed_z=zc[:, j])
                v = p.float() * dst + dm      # [chunk_nodes, n_feat], physical
            for k, f in enumerate(graph_slices(v, g, m)):
                col = f[:, fi]
                gen_lo[j, start + k] = col.min()
                gen_hi[j, start + k] = col.max()
                gen_stats[j, start + k] = reduce(f)
            del p, v                          # release before the next forward
        del g, zc
        if dev.type == 'cuda':
            torch.cuda.empty_cache()          # give the chunk's blocks back
        print("  scored graphs %d-%d of %d" % (start, stop - 1, n), flush=True)

    # `margin` is a FRACTION OF THE WHOLE OBSERVED RANGE, so the historical
    # 0.5 asks whether a draw leaves [min - 50% of range, max + 50% of range]
    # -- twice the data's own width. That is a blow-up detector, and a model in
    # roughly the right ballpark scores 0 on it no matter how badly calibrated
    # it is. Report a ladder so the number can actually separate arms: at 0.00
    # it asks the sharp question, "did this draw step outside the data envelope
    # at all?".
    span = gt_hi - gt_lo

    def wild_at(frac):
        m = frac * span
        return int(((gen_lo < gt_lo - m) | (gen_hi > gt_hi + m)).sum())

    MARGINS = (0.0, 0.1, 0.25, 0.5)
    margin = 0.5 * span
    wild = wild_at(0.5)
    ranks = np.array([int((gen_stats[:, i] < gt_stat[i]).sum()) for i in range(n)],
                     dtype=int)

    hist = np.bincount(ranks, minlength=K + 1)
    expected = n / (K + 1)
    chi2 = float(((hist - expected) ** 2 / max(expected, 1e-9)).sum())
    dof = K  # K+1 bins, 1 constraint
    # Rough uniform-chi2 5% critical value via Wilson-Hilferty approx (no scipy dep).
    crit = dof * (1 - 2 / (9 * dof) + 1.645 * (2 / (9 * dof)) ** 0.5) ** 3

    # The K+1-bin histogram has as many bins as there are DRAWS per graph, but
    # only `n` observations to fill them -- at K=50 with a 100-graph split that
    # is 1.96 expected per bin, well under the >=5 a chi-square test needs.
    # Collapsing to 5 bins fixes it: measured over 600 synthetic replays at
    # (n=100, K=50) the 5-bin test rejects a calibrated forecast 6% of the time
    # and an over-dispersed one 100%, where the 51-bin `shape` label calls
    # over-dispersion "roughly flat" about half the time.
    edges = [round(i * (K + 1) / 5) for i in range(6)]
    five = np.array([hist[edges[i]:edges[i + 1]].sum() for i in range(5)])
    e5 = n / 5.0
    chi5 = float(((five - e5) ** 2 / max(e5, 1e-9)).sum())
    CRIT5 = 9.488  # chi2(4), 5%

    print("DIAGNOSTIC cfg=%s split=%s n_graphs=%d K=%d sampler=%s stat=%s feature_idx=%d" %
          (os.path.basename(args.config), args.split, n, K,
           'prior' if use_prior else 'N(0,I)', args.stat, fi))
    print("  GT envelope [%.1f, %.1f]  margin +/-%.1f" % (gt_lo, gt_hi, margin))
    print("  WILD RATE = %d / %d (%.1f%%)" % (wild, K * n, 100.0 * wild / (K * n)))
    print("  WILD LADDER (margin x GT range): " + "  ".join(
        "%.2f=%.1f%%" % (f, 100.0 * wild_at(f) / max(K * n, 1)) for f in MARGINS))
    print("  [note] the envelope is taken from these %d graphs only, so wild%% "
          "is comparable across arms only at equal n_graphs." % n)
    print("  RANK HISTOGRAM (%d bins, expect ~%.1f each): %s" %
          (K + 1, expected, hist.tolist()))
    print("  chi2 = %.1f  (uniform-fit 5%% critical ~%.1f; below = can't reject uniform)" %
          (chi2, crit))
    print("  RANK5 = %s %%  (5 bins, expect 20%% each)" %
          [round(100.0 * float(v) / max(n, 1), 1) for v in five])
    print("  chi2_5 = %.1f  (5-bin critical %.1f; THIS is the reliable test)" %
          (chi5, CRIT5))
    if n < 50:
        # Measured: at n=40 the shape label still recovers a genuine U only ~66%
        # of the time (Poisson noise on ~13-count tails); at n>=100 it is ~90%+.
        print("  [warn] only %d graphs (%.1f per 5-bin): the shape label and "
              "chi2_5 are indicative here, not conclusive." % (n, e5))

    # Shape from the 5-bin view. The old test asked `hist[K//2] > 2*expected`
    # -- a SINGLE middle bin out of K+1. An over-dispersed forecast spreads its
    # mass across many middle bins, so that bin often fails the threshold and
    # the label came back "roughly flat"; measured miss rate ~50% at K=50 and
    # ~85% at K=9. Comparing the outer fifths against the middle fifth is the
    # same question asked at a resolution the data can actually support.
    # A U needs BOTH outer fifths heavy. Testing their SUM (as the old rule did
    # via hist[0]+hist[-1]) cannot tell a genuine U from a one-sided pile-up,
    # which is what made pure bias read as under-dispersion; requiring both
    # separately lets the U test run first without swallowing the skew case.
    lo5, mid5, hi5 = five[0] / max(n, 1), five[2] / max(n, 1), five[-1] / max(n, 1)
    shape = ('U-shaped (under-dispersed)' if lo5 > 0.25 and hi5 > 0.25
             else 'skewed' if abs(five[:2].sum() - five[-2:].sum()) > n * 0.3
             else 'dome-shaped (over-dispersed)' if mid5 > 0.30
             else 'roughly flat')
    print("  shape: %s" % shape)


if __name__ == '__main__':
    main()
