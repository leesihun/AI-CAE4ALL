"""Latent anatomy of a hybrid MGN-V model, per geometry: which latent part the
decoder reads the draw from, where the prior's draw-to-draw variety comes
from, and along which latent directions the prior misses.

    cd methods/MeshGraphNets_Variational
    python misc/latent_anatomy.py \
        --config ../../configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/config_infer_hybrid_base.txt \
        [--gpu 0] [--batch 8] [--split heldout|train|val|test] [--n-groups N] [--out DIR] [--tag NAME]

posterior_vs_prior_groups.py splits the W2 gap into encode/decode and prior.
This one takes the hybrid latent (g = per-graph slots, l = coarsest-node
channels) apart. Geometries, splits and scoring are the same (one geometry per
`split_group_attr` value, W2 over the reference half's sd, rms over the
geometries, median over plots.py's 200 permutations, x100). Nothing trains.

DECODER SWAPS  (one decode per FE result, paired with it)

    mu           decode (mu_g, mu_l) of the FE result
    post_sample  decode z ~ q(z | y)
    g_shuffle    (mu_g of another FE result of the geometry, mu_l)
    g_mean       (the geometry's mean mu_g, mu_l)          -- "l only"
    g_normal     (N(0, I), mu_l)
    g_prior      (a prior draw's g, mu_l)
    l_shuffle    (mu_g, mu_l of another FE result)
    l_mean       (mu_g, the geometry's mean mu_l)          -- "g only"
    l_prior      (mu_g, a prior draw's l)

The paired correlation with the FE value (within the geometry) says which part
carries the draw: g_shuffle ~ mu means g carries none, l_shuffle ~ 0 means l
carries all of it. g_prior / l_prior's W2 says which part of the prior's draw
costs what.

PRIOR NOISE  (the prior's draw is an ODE from noise (eps_g, eps_l))

    prior        the sampler, iid noise
    nowrite      the same noise, the token's write-back into the node stream
                 (g_out) zeroed: the l flow no longer sees the g stream
    g0           eps_g = 0, eps_l as `prior`: no g dice
    l0           eps_l = 0, eps_g as `prior`: no l dice
    std_normal   z ~ N(0, I) straight to the decoder
    grid         K x K draws, eps_g from K values (rows) x eps_l from K (columns)

The grid's two-way ANOVA splits the output's variance (and the l latent's
along the decisive direction) into the g-noise rows, the l-noise columns and
the rest; with no effect a share is about 1/(K+1). P(high) is the share of
draws above the FE results' 2-means threshold (the upper mode on ex3).

DECISIVE DIRECTIONS  (standardized prior space: u = (z - shift) / scale)

The FE value regressed on the top-8 principal components of the posterior
means, per part (g, l, joint), 2-fold cross-validated R^2; its coefficients
give a direction w. Along w_l the prior's draws (and each ablation's) against
the posterior samples': offset in posterior sd, sd ratio, 1-D W2, bimodality
coefficient (> 0.555: two modes). yhat: the regression applied to the
prior's latents predicts the prior's output bias; next to the actual bias.
grad: the decoder's gradient of the statistic at posterior means, the
first-order output variance it implies from g and from l (posterior samples
and prior draws), over the FE variance, and |cos| between the mean gradient
and the regression direction. prec/rec: k-NN (k=3) precision and recall of
the prior's draws against the posterior samples in the whitened top-8 PCA
space of each part.

FM LOSS ON THIS GEOMETRY  the prior's flow-matching MSE on posterior samples,
l and g, at t in 4 bins; the share of the l error along w_l over its share of
the target (1 = the error is no more concentrated on the decisive direction
than the signal is).

TOKEN WRITE-BACK  rms of g_out(token) over the rms of the node stream after
in_proj, and the same for their draw-to-draw sd, per block, t < 0.5 / t >= 0.5.

Writes latent_anatomy_<tag>.json (with `brief` lines for the round's brief)
and ends with a ==== TYPE THIS BACK ==== block. hybrid models only.
"""
import argparse
import json
import math
import os
import sys

ORIG_CWD = os.getcwd()
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import h5py                                                       # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from torch_geometric.data import Batch                            # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'misc'))

from general_modules.grouped_split import grouped_split_ids       # noqa: E402
from general_modules.load_config import load_config               # noqa: E402
from general_modules.renamed_keys import migrate_checkpoint_config  # noqa: E402
from inference_profiles.rollout import (                          # noqa: E402
    _SampleContext, _eval_dataset_groups, _group_order,
    _load_conditional_prior, _load_model_from_checkpoint,
    _spread_index, _spread_stat,
)
from model.conditional_prior import _autocast_disabled_for        # noqa: E402
from model.latent import Latent                                   # noqa: E402
from posterior_vs_prior import resolve_device                     # noqa: E402
from posterior_vs_prior_groups import (                           # noqa: E402
    SPLIT_SEED, SPLITS, geometry_row, paired, summarize,
)

os.chdir(REPO_ROOT)   # config paths are relative to the method directory

ALIGNED = ('mu', 'post_sample', 'g_shuffle', 'g_mean', 'g_normal', 'g_prior',
           'l_shuffle', 'l_mean', 'l_prior')
FREE = ('prior', 'nowrite', 'g0', 'l0', 'std_normal', 'grid')
ENSEMBLES = ('floor',) + ALIGNED + FREE
LATENT_SETS = ('post_sample', 'mu', 'prior', 'nowrite', 'g0', 'l0', 'std_normal')
TOP_PC = 8
BC_BIMODAL = 5.0 / 9.0


# ─────────────────────────────────────────────────────────────────────────────
# statistics
# ─────────────────────────────────────────────────────────────────────────────

def split_scores(spreads, order, splits=SPLITS):
    """posterior_vs_prior_groups.split_scores over this script's ensembles.

    The same permutations (SPLIT_SEED), so the same floor. An ensemble whose
    size differs from the FE results' (the grid, when n is not a square) is
    left out.
    """
    rng = np.random.default_rng(SPLIT_SEED)
    perms = {v: [rng.permutation(spreads[v]['truth'].size) for _ in range(splits)] for v in order}
    k = min(spreads[v]['truth'].size // 2 for v in order)
    pooled_sd = float(np.concatenate([spreads[v]['truth'] for v in order]).std()) or 1.0
    runs = {e: [] for e in ENSEMBLES}
    per_group = {v: {e: [] for e in ENSEMBLES} for v in order}
    for i in range(splits):
        rows = {e: {} for e in ENSEMBLES}
        for v in order:
            s = spreads[v]
            ia, ib = perms[v][i][:k], perms[v][i][k:]
            ref = s['truth'][ib]
            for e in ENSEMBLES:
                src = s['truth'] if e == 'floor' else s.get(e)
                if src is None or src.size != s['truth'].size:
                    continue
                row = geometry_row(ref, src[ia], pooled_sd)
                if row is not None:
                    rows[e][v] = row
                    per_group[v][e].append(row)
        for e in ENSEMBLES:
            if rows[e]:
                runs[e].append(summarize(rows[e]))
    out = {}
    for e in ENSEMBLES:
        if not runs[e]:
            continue
        vals = np.asarray([r['w2'] for r in runs[e]])
        out[e] = {'p50': float(np.median(vals)), 'p5': float(np.percentile(vals, 5)),
                  'p95': float(np.percentile(vals, 95)),
                  'sd_ratio_geomean': float(np.median([r['sd_ratio_geomean'] for r in runs[e]])),
                  'bias_rms': float(np.median([r['bias'] for r in runs[e]]))}
    med = {v: {e: float(np.median([r['w2'] for r in per_group[v][e]])) if per_group[v][e]
               else float('nan') for e in ENSEMBLES} for v in order}
    parts = {v: {e: {key: float(np.median([r[key] for r in per_group[v][e]]))
                     for key in ('bias', 'sd_ratio', 'shape')}
                 for e in ENSEMBLES if per_group[v][e]} for v in order}
    return out, med, parts, k


def two_means_threshold(x):
    """The 1-D 2-means cut of x (the midpoint between the two clusters)."""
    s = np.sort(np.asarray(x, np.float64))
    n = s.size
    if n < 4 or s[-1] <= s[0]:
        return float(np.median(s)) if n else float('nan')
    c1, c2 = np.cumsum(s), np.cumsum(s ** 2)
    best, cut = None, 1
    for i in range(1, n):
        left = c2[i - 1] - c1[i - 1] ** 2 / i
        right = (c2[-1] - c2[i - 1]) - (c1[-1] - c1[i - 1]) ** 2 / (n - i)
        if best is None or left + right < best:
            best, cut = left + right, i
    return float(0.5 * (s[cut - 1] + s[cut]))


def bimodality(x):
    """Sarle's bimodality coefficient (> 5/9 suggests two modes)."""
    x = np.asarray(x, np.float64)
    n = x.size
    if n < 4 or x.std() == 0:
        return float('nan')
    d = x - x.mean()
    m2 = np.mean(d ** 2)
    g1 = np.mean(d ** 3) / m2 ** 1.5
    g2 = np.mean(d ** 4) / m2 ** 2 - 3.0
    return float((g1 ** 2 + 1.0) / (g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def w2_1d(a, b):
    """1-D W2 between two samples (quantile matching on a common grid)."""
    q = (np.arange(200) + 0.5) / 200
    return float(np.sqrt(np.mean((np.quantile(a, q) - np.quantile(b, q)) ** 2)))


def along(proj, ref):
    """proj against ref (both 1-D): offset / sd ratio / W2, in ref sd."""
    sd = float(np.std(ref)) or float('nan')
    return {'offset': float((np.mean(proj) - np.mean(ref)) / sd),
            'sd_ratio': float(np.std(proj) / sd),
            'w2': w2_1d(proj, ref) / sd,
            'bc': bimodality(proj)}


def participation_ratio(X):
    lam = np.linalg.svd(X - X.mean(axis=0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if lam.sum() > 0 else float('nan')


def direction(U, y, seed):
    """Regress y on the top principal components of U (posterior means).

    Returns the 2-fold cross-validated R^2, the unit direction in U's space,
    and a predictor yhat(U') = (U' - centre) @ coef + mean(y).
    """
    U = np.asarray(U, np.float64)
    y = np.asarray(y, np.float64)
    n, d = U.shape
    c = U.mean(axis=0)
    X = U - c
    k = int(min(TOP_PC, n // 4, d))   # 2-fold CV: >= 2 samples per coefficient
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    var = sv ** 2
    out = {'pc_var_frac': (var[:k] / var.sum()).tolist() if var.sum() > 0 else [],
           'pr_post': float(var.sum() ** 2 / (var ** 2).sum()) if var.sum() > 0 else float('nan')}
    if k < 1 or y.std() == 0:
        out.update({'r2_cv': float('nan'), 'w': np.zeros(d), 'coef': np.zeros(d), 'centre': c,
                    'y_mean': float(y.mean()), 'pc_corr': []})
        return out
    S = X @ vt[:k].T
    out['pc_corr'] = [float(np.corrcoef(S[:, j], y)[0, 1]) if S[:, j].std() > 0 else 0.0
                      for j in range(k)]
    perm = np.random.default_rng(seed).permutation(n)
    folds = (perm[: n // 2], perm[n // 2:])
    pred = np.empty(n)
    for a, b in (folds, folds[::-1]):
        A = np.c_[S[a], np.ones(len(a))]
        beta = np.linalg.lstsq(A, y[a], rcond=None)[0]
        pred[b] = np.c_[S[b], np.ones(len(b))] @ beta
    out['r2_cv'] = float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
    beta = np.linalg.lstsq(S, y - y.mean(), rcond=None)[0]
    coef = vt[:k].T @ beta
    out.update({'coef': coef, 'w': coef / (np.linalg.norm(coef) or 1.0), 'centre': c,
                'y_mean': float(y.mean())})
    return out


def prec_recall(P, Q, k=3):
    """k-NN precision (Q inside P's balls) and recall (P inside Q's balls) in
    the whitened top-8 PCA space of P."""
    P = np.asarray(P, np.float64)
    Q = np.asarray(Q, np.float64)
    c = P.mean(axis=0)
    _, sv, vt = np.linalg.svd(P - c, full_matrices=False)
    m = int(min(TOP_PC, P.shape[0] - 1, P.shape[1]))
    scale = np.maximum(sv[:m] / math.sqrt(P.shape[0]), 1e-12)
    p = (P - c) @ vt[:m].T / scale
    q = (Q - c) @ vt[:m].T / scale

    def radii(a):
        d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
        return np.sort(d, axis=1)[:, min(k, a.shape[0] - 1)]

    rp, rq = radii(p), radii(q)
    dqp = np.sqrt(((q[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    return {'precision': float(np.mean((dqp <= rp[None, :]).any(axis=1))),
            'recall': float(np.mean((dqp.T <= rq[None, :]).any(axis=1)))}


def anova_shares(Y):
    """Two-way (rows x columns) variance shares of a K x K table."""
    Y = np.asarray(Y, np.float64)
    gm = Y.mean()
    sst = float(((Y - gm) ** 2).sum())
    if sst <= 0:
        return {'row': float('nan'), 'col': float('nan')}
    K1, K2 = Y.shape
    return {'row': float(K2 * ((Y.mean(axis=1) - gm) ** 2).sum() / sst),
            'col': float(K1 * ((Y.mean(axis=0) - gm) ** 2).sum() / sst)}


# ─────────────────────────────────────────────────────────────────────────────
# one geometry
# ─────────────────────────────────────────────────────────────────────────────

class Geometry:
    """Encode / decode / prior flow for one geometry's graph."""

    def __init__(self, model, prior, ctx, ch, stat, batch, steps, temperature, inflation, grad_batch=2):
        self.model, self.prior, self.ctx = model, prior, ctx
        self.inner = getattr(model, 'model', model)
        self.ch, self.stat, self.batch = ch, stat, batch
        # backward through the decoder keeps every activation (~17 GiB per 86k-node
        # graph in fp32), so it runs in its own, smaller chunks; the prior is coarse-only
        self.grad_batch, self.flow_chunk = grad_batch, 256
        self.stage = 'setup'
        self.steps, self.amp, self.inflation = steps, math.sqrt(max(temperature, 1e-6)), inflation
        self.dm = np.asarray(ctx.delta_mean, np.float32)
        self.dsd = np.asarray(ctx.delta_std, np.float32)
        self.base = ctx.build_step_graph(np.zeros((ctx.num_nodes, ctx.input_dim), dtype=np.float32))
        self.N = ctx.num_nodes
        self.dev = self.base.x.device
        self._cache = {}
        self.S, self.D = prior.num_z, prior.z_dim
        self.C = prior.latent_ch
        with torch.no_grad():
            self.cond1 = prior.condition(self.graphs(1))
        self.n_c = int(self.cond1['x'].shape[0])
        self.z_shift = prior.z_shift.double().cpu().numpy()
        self.z_scale = prior.z_scale.double().cpu().numpy()
        self.zl_shift = prior.zl_shift.double().cpu().numpy()
        self.zl_scale = prior.zl_scale.double().cpu().numpy()

    # graphs --------------------------------------------------------------

    def graphs(self, m, fields=None):
        if fields is None:
            if m not in self._cache:
                gs = []
                for _ in range(m):
                    gi = self.base.clone()
                    gi.y = torch.zeros(self.N, self.dm.size, device=gi.x.device)
                    gs.append(gi)
                self._cache[m] = Batch.from_data_list(gs)
            return self._cache[m].clone()
        gs = []
        for f in fields:
            gi = self.base.clone()
            gi.y = torch.from_numpy((f - self.dm) / self.dsd).to(gi.x.device)
            gs.append(gi)
        return Batch.from_data_list(gs)

    def _stats_np(self, pred, m):
        phys = pred.float().cpu().numpy().reshape(m, self.N, -1) * self.dsd + self.dm
        return [_spread_stat(phys[j][:, self.ch], self.stat) for j in range(m)]

    def _stats_torch(self, pred, m):
        phys = (pred.float().view(m, self.N, -1) * torch.as_tensor(self.dsd, device=pred.device)
                + torch.as_tensor(self.dm, device=pred.device))
        v = phys[:, :, self.ch]
        if v.dim() == 2:
            v = v.unsqueeze(-1)
        if self.stat == 'mag_max':
            return v.norm(dim=-1).amax(dim=1)
        v = v[..., 0]
        if self.stat == 'range':
            return v.amax(dim=1) - v.amin(dim=1)
        if self.stat == 'mean':
            return v.mean(dim=1)
        return v.std(dim=1, unbiased=False)

    # chunking ------------------------------------------------------------

    def _chunks(self, n, attr, body):
        """[body(s, m)] over [0, n) in chunks of self.<attr>; a CUDA OOM halves
        the chunk for the rest of the geometry and retries it."""
        out, s = [], 0
        while s < n:
            m = min(getattr(self, attr), n - s)
            try:
                out.append(body(s, m))
            except torch.cuda.OutOfMemoryError:
                if m <= 1:
                    raise
                oom = True
            else:
                oom = False
                s += m
            if oom:   # outside the handler, so the failed chunk's tensors are released
                torch.cuda.empty_cache()
                setattr(self, attr, max(1, m // 2))
                print(f'  CUDA OOM in {self.stage} at {attr}={m}: now {getattr(self, attr)}', flush=True)
        return out

    # encode / decode -----------------------------------------------------

    def encode(self, fields):
        def body(s, m):
            with torch.no_grad():
                _, mu, logvar = self.inner.posterior(self.graphs(m, fields[s:s + m]))
            return (mu.g.double().cpu().numpy(), mu.l.double().cpu().numpy().reshape(m, -1, self.C),
                    np.exp(0.5 * logvar.g.double().cpu().numpy()),
                    np.exp(0.5 * logvar.l.double().cpu().numpy()).reshape(m, -1, self.C))
        mg, ml, sg, sl = zip(*self._chunks(len(fields), 'batch', body))
        mu_g, mu_l = np.concatenate(mg), np.concatenate(ml)
        if mu_l.shape[1] != self.n_c:
            raise RuntimeError(f'posterior has {mu_l.shape[1]} coarse nodes, the prior {self.n_c}')
        return mu_g, mu_l, np.concatenate(sg), np.concatenate(sl)

    def _latent(self, G, L):
        m = G.shape[0]
        return Latent(torch.from_numpy(np.ascontiguousarray(G, np.float32)).to(self.dev),
                      torch.from_numpy(np.ascontiguousarray(L, np.float32).reshape(m * self.n_c, self.C))
                      .to(self.dev))

    def decode(self, G, L):
        """Spread statistic of each decoded (g [M, S, D], l [M, n_c, C]) in z space."""
        def body(s, m):
            with torch.no_grad():
                pred = self.model(self.graphs(m), add_noise=False, use_posterior=False,
                                  fixed_z=self._latent(G[s:s + m], L[s:s + m]))[0]
            return self._stats_np(pred, m)
        return np.asarray([v for c in self._chunks(len(G), 'batch', body) for v in c], np.float64)

    def gradients(self, G, L):
        """d statistic / d u (standardized) at each (g, l): ([M, S*D], [M, n_c*C])."""
        def body(s, m):
            z = self._latent(G[s:s + m], L[s:s + m])
            zg, zl = z.g.requires_grad_(True), z.l.requires_grad_(True)
            with torch.enable_grad():
                pred = self.model(self.graphs(m), add_noise=False, use_posterior=False,
                                  fixed_z=Latent(zg, zl))[0]
                self._stats_torch(pred, m).sum().backward()
            return (zg.grad.double().cpu().numpy().reshape(m, -1) * self.z_scale,
                    (zl.grad.double().cpu().numpy().reshape(m, self.n_c, self.C)
                     * self.zl_scale).reshape(m, -1))
        gg, gl = zip(*self._chunks(len(G), 'grad_batch', body))
        return np.concatenate(gg), np.concatenate(gl)

    # standardization -----------------------------------------------------

    def u_of(self, G, L):
        """z -> u = (z - shift) / scale, flattened: ([M, S*D], [M, n_c*C])."""
        M = G.shape[0]
        ug = (G.reshape(M, -1) - self.z_shift) / self.z_scale
        ul = ((L - self.zl_shift) / self.zl_scale).reshape(M, -1)
        return ug, ul

    def z_of(self, ug, ul):
        M = ug.shape[0]
        return ((ug * self.z_scale + self.z_shift).reshape(M, self.S, self.D),
                ul.reshape(M, self.n_c, self.C) * self.zl_scale + self.zl_shift)

    # prior ---------------------------------------------------------------

    def flow(self, eg, el):
        """Integrate the prior from noise eg [M, S*D], el [M, n_c, C] (already
        temperature-scaled); returns u ([M, S*D], [M, n_c*C])."""
        p = self.prior

        def body(s, m):
            with torch.no_grad(), _autocast_disabled_for(self.cond1['x']):
                ctx = p._replicate(p._float_ctx(self.cond1), m)
                zl = torch.from_numpy(np.ascontiguousarray(el[s:s + m], np.float32)
                                      .reshape(m * self.n_c, self.C)).to(self.dev)
                zg = torch.from_numpy(np.ascontiguousarray(eg[s:s + m], np.float32)).to(self.dev)
                zl1, zg1 = p._integrate(zl, zg, ctx, self.steps)
                if self.inflation != 1.0:
                    cl, cg = p._integrate(torch.zeros_like(zl), torch.zeros_like(zg), ctx, self.steps)
                    zl1 = cl + self.inflation * (zl1 - cl)
                    zg1 = cg + self.inflation * (zg1 - cg)
            return zg1.double().cpu().numpy(), zl1.double().cpu().numpy().reshape(m, -1)
        ug, ul = zip(*self._chunks(len(eg), 'flow_chunk', body))
        return np.concatenate(ug), np.concatenate(ul)

    def fm_errors(self, Ug, Ul, rng, bins=4, reps=2):
        """The prior's FM regression errors on standardized posterior samples, per t bin.

        Returns [(t, e_g [n, S*D], e_l [n, n_c*C], tgt_g, tgt_l)].
        """
        p = self.prior
        s_ = 1.0 - p.sigma_min
        n = Ug.shape[0]
        out = []
        for _ in range(reps):
            for b in range(bins):
                t = (b + 0.5) / bins
                z0g = rng.standard_normal(Ug.shape)
                z0l = rng.standard_normal(Ul.shape)
                tgt_g, tgt_l = Ug - s_ * z0g, Ul - s_ * z0l
                xg, xl = (1 - s_ * t) * z0g + t * Ug, (1 - s_ * t) * z0l + t * Ul
                def body(s, m, t=t, xg=xg, xl=xl, tgt_g=tgt_g, tgt_l=tgt_l):
                    with torch.no_grad(), _autocast_disabled_for(self.cond1['x']):
                        ctx = p._replicate(p._float_ctx(self.cond1), m)
                        tt = torch.full((m, 1), t, device=self.dev)
                        v_l, v_g = p.velocity(
                            torch.from_numpy(xl[s:s + m].reshape(m * self.n_c, self.C)).float().to(self.dev),
                            torch.from_numpy(xg[s:s + m]).float().to(self.dev), tt, ctx)
                    return (v_g.double().cpu().numpy() - tgt_g[s:s + m],
                            v_l.double().cpu().numpy().reshape(m, -1) - tgt_l[s:s + m])
                eg, el = zip(*self._chunks(n, 'flow_chunk', body))
                out.append((t, np.concatenate(eg), np.concatenate(el), tgt_g, tgt_l))
        return out


class TokenProbe:
    """Hooks on the prior's token write-back (g_out) and node stream (in_proj).

    `nowrite` zeroes every g_out output; `record` keeps rms and draw-to-draw sd.
    """

    def __init__(self, prior):
        self.prior = prior
        self.nowrite = False
        self.record = False
        self.t = 0.0
        self.m = 1
        self.rows = []          # (kind, block, t, rms, draw_sd)
        self.handles = [prior.in_proj.register_forward_hook(self._x0)]
        for j, lin in enumerate(prior.g_out):
            self.handles.append(lin.register_forward_hook(self._wb(j)))
        self._orig_velocity = prior.velocity

        def velocity(zl_t, zg_t, t, ctx):
            self.t = float(t.reshape(-1)[0])
            self.m = int(ctx['B'])
            return self._orig_velocity(zl_t, zg_t, t, ctx)
        prior.velocity = velocity

    @staticmethod
    def _rms_sd(o):
        o = o.float()
        return float(o.pow(2).mean().sqrt()), float(o.std(dim=0, unbiased=False).pow(2).mean().sqrt())

    def _x0(self, mod, inp, out):
        if self.record and self.m > 1:
            rms, sd = self._rms_sd(out.view(self.m, -1, out.shape[-1]))
            self.rows.append(('x0', -1, self.t, rms, sd))

    def _wb(self, j):
        def hook(mod, inp, out):
            if self.record and self.m > 1:
                rms, sd = self._rms_sd(out)
                self.rows.append(('wb', j, self.t, rms, sd))
            if self.nowrite:
                return torch.zeros_like(out)
            return None
        return hook

    def summary(self):
        """Per block j: write-back rms / node-stream rms, and the same for the
        draw-to-draw sd, for t < 0.5 and t >= 0.5."""
        out = {}
        for half, sel in (('early', lambda t: t < 0.5), ('late', lambda t: t >= 0.5)):
            x0 = [(r[3], r[4]) for r in self.rows if r[0] == 'x0' and sel(r[2])]
            if not x0:
                continue
            x_rms, x_sd = np.mean([a for a, _ in x0]), np.mean([b for _, b in x0])
            for j in range(len(self.prior.g_out)):
                wb = [(r[3], r[4]) for r in self.rows if r[0] == 'wb' and r[1] == j and sel(r[2])]
                if wb:
                    out.setdefault(str(j), {})[half] = {
                        'rms_over_x0': float(np.mean([a for a, _ in wb]) / x_rms) if x_rms else float('nan'),
                        'drawsd_over_x0': float(np.mean([b for _, b in wb]) / x_sd) if x_sd else float('nan')}
        self.rows = []
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.prior.velocity = self._orig_velocity


def run_group(geo, probe, truths, fields, seed, grad_draws):
    """Every ensemble and latent statistic of one geometry."""
    rng = np.random.default_rng(seed)
    n = len(fields)
    S, D, C, n_c = geo.S, geo.D, geo.C, geo.n_c
    sp = {'truth': np.asarray(truths, np.float64)}
    geo.stage = 'encode'
    mu_g, mu_l, sg, sl = geo.encode(fields)
    ps_g = mu_g + sg * rng.standard_normal(mu_g.shape)
    ps_l = mu_l + sl * rng.standard_normal(mu_l.shape)

    # derangement: draw i takes the latent of draw shuf[i] != i
    perm = rng.permutation(n)
    shuf = np.empty(n, dtype=int)
    shuf[perm] = np.roll(perm, n // 2 if n > 1 else 0)

    # prior: iid, no write-back, no g dice, no l dice (one shared noise set)
    eg = rng.standard_normal((n, S * D)) * geo.amp
    el = rng.standard_normal((n, n_c, C)) * geo.amp
    geo.stage = 'prior flow'
    probe.record = True
    U = {'prior': geo.flow(eg, el)}
    probe.record = False
    token = probe.summary()
    probe.nowrite = True
    U['nowrite'] = geo.flow(eg, el)
    probe.nowrite = False
    U['g0'] = geo.flow(np.zeros_like(eg), el)
    U['l0'] = geo.flow(eg, np.zeros_like(el))
    Z = {k: geo.z_of(*u) for k, u in U.items()}
    zn_g = rng.standard_normal((n, S, D))
    zn_l = rng.standard_normal((n, n_c, C))
    Z['std_normal'] = (zn_g, zn_l)
    Z['mu'] = (mu_g, mu_l)
    Z['post_sample'] = (ps_g, ps_l)
    for k in ('std_normal', 'mu', 'post_sample'):
        U[k] = geo.u_of(*Z[k])

    mean_g = np.broadcast_to(mu_g.mean(axis=0, keepdims=True), mu_g.shape)
    mean_l = np.broadcast_to(mu_l.mean(axis=0, keepdims=True), mu_l.shape)
    pairs = {
        'mu': (mu_g, mu_l), 'post_sample': (ps_g, ps_l),
        'g_shuffle': (mu_g[shuf], mu_l), 'g_mean': (mean_g, mu_l),
        'g_normal': (rng.standard_normal(mu_g.shape), mu_l), 'g_prior': (Z['prior'][0], mu_l),
        'l_shuffle': (mu_g, mu_l[shuf]), 'l_mean': (mu_g, mean_l), 'l_prior': (mu_g, Z['prior'][1]),
        'prior': Z['prior'], 'nowrite': Z['nowrite'], 'g0': Z['g0'], 'l0': Z['l0'],
        'std_normal': Z['std_normal'],
    }
    for k, (G, L) in pairs.items():
        geo.stage = f'decode {k}'
        sp[k] = geo.decode(G, L)

    # grid: K g-noises (rows) x K l-noises (columns)
    K = int(math.isqrt(n))
    ag = rng.standard_normal((K, S * D)) * geo.amp
    al = rng.standard_normal((K, n_c, C)) * geo.amp
    geo.stage = 'grid'
    ug_grid, ul_grid = geo.flow(np.repeat(ag, K, axis=0), np.tile(al, (K, 1, 1)))
    sp['grid'] = geo.decode(*geo.z_of(ug_grid, ul_grid))

    # decisive directions, fitted on the posterior means (u space)
    y = sp['truth']
    dirs = {}
    for part, idx in (('g', 0), ('l', 1)):
        dirs[part] = direction(U['mu'][idx], y, seed)
    dirs['j'] = direction(np.c_[U['mu'][0], U['mu'][1]], y, seed)
    dirs['j_dec'] = direction(np.c_[U['mu'][0], U['mu'][1]], sp['mu'], seed)

    def proj(part, key):
        u = U[key]
        X = u[1] if part == 'l' else (u[0] if part == 'g' else np.c_[u[0], u[1]])
        return (X - dirs[part]['centre']) @ dirs[part]['w']

    def yhat(key):
        u = U[key]
        return (np.c_[u[0], u[1]] - dirs['j']['centre']) @ dirs['j']['coef'] + dirs['j']['y_mean']

    sd_y = float(y.std()) or float('nan')
    lat = {}
    for part in ('l', 'j'):
        ref = proj(part, 'post_sample')
        lat[part] = {k: along(proj(part, k), ref) for k in LATENT_SETS if k != 'post_sample'}
        lat[part]['post_sample'] = {'bc': bimodality(ref)}
    proj_l = {k: proj('l', k).tolist() for k in ('post_sample', 'prior', 'nowrite', 'g0', 'l0')}
    yh = {k: {'pred_bias': float((yhat(k).mean() - y.mean()) / sd_y),
              'actual_bias': float((sp[k].mean() - y.mean()) / sd_y)}
          for k in ('prior', 'nowrite', 'g0', 'l0', 'post_sample')}

    # grid ANOVA: output, and l along w_l
    grid = {'K': K, 'null_share': 1.0 / (K + 1) if K > 0 else float('nan')}
    thr = two_means_threshold(y)
    if K >= 2:
        Y = sp['grid'][:K * K].reshape(K, K)
        grid['out'] = anova_shares(Y)
        Pl = ((ul_grid - dirs['l']['centre']) @ dirs['l']['w']).reshape(K, K)
        grid['wl'] = anova_shares(Pl)
        grid['table'] = Y.tolist()
        ph_rows = (Y > thr).mean(axis=1)
        p = float((Y > thr).mean())
        grid['p_high_row_sd'] = float(ph_rows.std())
        grid['p_high_row_sd_null'] = math.sqrt(p * (1 - p) / K)

    # P(high) and bimodality of the output
    p_high = {k: float(np.mean(sp[k] > thr)) for k in sp}
    bc_out = {k: bimodality(sp[k]) for k in sp}

    # supports: posterior samples vs prior draws, per part
    pr = {}
    for part, idx in (('g', 0), ('l', 1)):
        pr[part] = {k: prec_recall(U['post_sample'][idx], U[k][idx])
                    for k in ('prior', 'nowrite')}
        pr[part]['pr_post_mu'] = participation_ratio(U['mu'][idx])
        pr[part]['pr_post_sample'] = participation_ratio(U['post_sample'][idx])
        pr[part]['pr_prior'] = participation_ratio(U['prior'][idx])

    # FM regression error on posterior samples
    geo.stage = 'fm errors'
    errs = geo.fm_errors(U['post_sample'][0], U['post_sample'][1], rng)
    wl = dirs['l']['w']
    wj = dirs['j']['w']
    fm = {'bins': {}}
    for t, e_g, e_l, tg, tl in errs:
        b = fm['bins'].setdefault(f'{t:.3f}', {'mse_l': [], 'mse_g': []})
        b['mse_l'].append(float(np.mean(e_l ** 2)))
        b['mse_g'].append(float(np.mean(e_g ** 2)))
    for b in fm['bins'].values():
        b['mse_l'], b['mse_g'] = float(np.mean(b['mse_l'])), float(np.mean(b['mse_g']))
    E_l = np.concatenate([e[2] for e in errs])
    E_g = np.concatenate([e[1] for e in errs])
    T_l = np.concatenate([e[4] for e in errs])
    T_g = np.concatenate([e[3] for e in errs])
    E_j, T_j = np.c_[E_g, E_l], np.c_[T_g, T_l]

    def share(E, w):
        return float(np.mean((E @ w) ** 2) / np.mean(E ** 2)) if np.mean(E ** 2) > 0 else float('nan')

    fm.update({'mse_l': float(np.mean(E_l ** 2)), 'mse_g': float(np.mean(E_g ** 2)),
               'err_along_wl': share(E_l, wl), 'tgt_along_wl': share(T_l, wl),
               'err_along_wj': share(E_j, wj), 'tgt_along_wj': share(T_j, wj)})

    # decoder gradient at posterior means
    grad = {}
    if grad_draws > 0:
        pick = np.unique(np.linspace(0, n - 1, min(grad_draws, n)).round().astype(int))
        geo.stage = 'decoder gradients'
        gg, gl = geo.gradients(mu_g[pick], mu_l[pick])
        var_y = float(y.var()) or float('nan')

        def contrib(key):
            ug, ul = U[key]
            vg = np.mean([np.var(ug @ gg[i]) for i in range(len(pick))]) / var_y
            vl = np.mean([np.var(ul @ gl[i]) for i in range(len(pick))]) / var_y
            return {'g': float(vg), 'l': float(vl)}

        grad = {k: contrib(k) for k in ('post_sample', 'mu', 'prior')}
        gj = np.c_[gg, gl].mean(axis=0)
        grad['cos_wj'] = float(abs(gj @ wj) / (np.linalg.norm(gj) or 1.0))
        grad['g_norm_share'] = float(np.mean(np.sum(gg ** 2, axis=1)
                                             / (np.sum(gg ** 2, axis=1) + np.sum(gl ** 2, axis=1))))
        # the decoder's own sensitive l direction (mean gradient): no linear fit needed
        wgl = gl.mean(axis=0)
        wgl = wgl / (np.linalg.norm(wgl) or 1.0)
        grad['coherence'] = float(np.mean(np.abs(gl @ wgl) / (np.linalg.norm(gl, axis=1) + 1e-30)))
        c_l = U['mu'][1].mean(axis=0)
        ref = (U['post_sample'][1] - c_l) @ wgl
        grad['along'] = {k: along((U[k][1] - c_l) @ wgl, ref) for k in ('prior', 'nowrite', 'g0', 'l0')}
        # first-order output shift from the latent centroid's move, posterior samples -> k, in FE sd
        sd_y = float(y.std()) or float('nan')
        grad['lin_bias'] = {k: {p: float(np.mean((U[k][i].mean(axis=0) - U['post_sample'][i].mean(axis=0))
                                                 @ G_.T) / sd_y)
                                for p, i, G_ in (('g', 0, gg), ('l', 1, gl))}
                            for k in ('prior', 'nowrite', 'g0')}

    clean = {k: {kk: vv for kk, vv in d.items() if kk not in ('w', 'coef', 'centre')}
             for k, d in dirs.items()}
    return sp, {'directions': clean, 'along': lat, 'yhat': yh, 'grid': grid,
                'p_high': p_high, 'threshold': thr, 'bc_out': bc_out, 'supports': pr,
                'fm': fm, 'grad': grad, 'token': token, 'n': n, 'n_c': n_c,
                'proj_l': proj_l}


# ─────────────────────────────────────────────────────────────────────────────
# the run
# ─────────────────────────────────────────────────────────────────────────────

def _abs(path):
    return path if path is None or os.path.isabs(path) else os.path.normpath(os.path.join(ORIG_CWD, path))


def _med(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.median(vals)) if vals else float('nan')


def _x(v):
    return '-' if v is None or not np.isfinite(v) else str(int(round(100 * v)))


def _sx(v):
    return '-' if v is None or not np.isfinite(v) else f'{100 * v:+.0f}'


def _r2(v):
    """R^2 x100, floored at -99 (a CV R^2 far below 0 only says 'no fit')."""
    return _x(max(v, -0.99) if v is not None and np.isfinite(v) else v)


def brief_lines(scores, pair, groups, order):
    """The type-back lines (a round's brief prints them with job and split)."""
    w = lambda e: _x(scores.get(e, {}).get('p50'))   # noqa: E731
    med = lambda f: _med([f(groups[v]) for v in order])   # noqa: E731
    L = {}
    L['w2'] = ('w2 ' + '/'.join(w(e) for e in ('floor', 'mu', 'post_sample')) + ' g '
               + '/'.join(w(e) for e in ('g_shuffle', 'g_mean', 'g_normal', 'g_prior')) + ' l '
               + '/'.join(w(e) for e in ('l_shuffle', 'l_mean', 'l_prior')) + ' p '
               + '/'.join(w(e) for e in ('prior', 'nowrite', 'g0', 'l0', 'std_normal', 'grid')))
    ph = lambda e: med(lambda r: abs(r['p_high'][e] - r['p_high']['truth']))   # noqa: E731
    L['pc'] = ('pc ' + '/'.join(_x(pair[e]['corr']) for e in ALIGNED)
               + f" ph {_x(med(lambda r: r['p_high']['truth']))} d "
               + '/'.join(_x(ph(e)) for e in ('post_sample', 'prior', 'nowrite', 'g0', 'l0', 'grid'))
               + ' bc ' + '/'.join(_x(med(lambda r, e=e: r['bc_out'][e]))
                                   for e in ('truth', 'post_sample', 'prior', 'nowrite')))
    gr = lambda a, b: med(lambda r: r['grid'].get(a, {}).get(b, float('nan')))   # noqa: E731
    tok = {}
    for v in order:
        for j, d in groups[v]['token'].items():
            for half, q in d.items():
                tok.setdefault((j, half), []).append(q)
    L['grid'] = (f"grid y {_x(gr('out', 'row'))}/{_x(gr('out', 'col'))} wl {_x(gr('wl', 'row'))}/"
                 f"{_x(gr('wl', 'col'))} null {_x(med(lambda r: r['grid']['null_share']))} phsd "
                 f"{_x(med(lambda r: r['grid'].get('p_high_row_sd', float('nan'))))}/"
                 f"{_x(med(lambda r: r['grid'].get('p_high_row_sd_null', float('nan'))))} wb "
                 + ' '.join(f"{j}:{_x(_med([q['rms_over_x0'] for q in tok.get((j, 'early'), [])]))}/"
                            f"{_x(_med([q['rms_over_x0'] for q in tok.get((j, 'late'), [])]))}/"
                            f"{_x(_med([q['drawsd_over_x0'] for q in tok.get((j, 'early'), [])]))}/"
                            f"{_x(_med([q['drawsd_over_x0'] for q in tok.get((j, 'late'), [])]))}"
                            for j in sorted({k[0] for k in tok}, key=int)))
    al = lambda k, key: med(lambda r: r['along']['l'][k][key])   # noqa: E731
    L['dir'] = ('dir r2 ' + '/'.join(_r2(med(lambda r, p=p: r['directions'][p]['r2_cv']))
                                     for p in ('g', 'l', 'j', 'j_dec'))
                + ' wl ' + ' '.join(f"{k}:{_x(med(lambda r, k=k: abs(r['along']['l'][k]['offset'])))}/"
                                    f"{_x(al(k, 'sd_ratio'))}/{_x(al(k, 'w2'))}"
                                    for k in ('mu', 'prior', 'nowrite', 'g0', 'l0', 'std_normal'))
                + f" bc {_x(med(lambda r: r['along']['l']['post_sample']['bc']))}/{_x(al('prior', 'bc'))}"
                + ' yhat ' + ' '.join(
                    f"{k}:{_x(med(lambda r, k=k: abs(r['yhat'][k]['pred_bias'])))}/"
                    f"{_x(med(lambda r, k=k: abs(r['yhat'][k]['actual_bias'])))}"
                    for k in ('prior', 'nowrite', 'g0')))
    fm = lambda key: med(lambda r: r['fm'][key])   # noqa: E731
    bins = sorted({b for v in order for b in groups[v]['fm']['bins']}, key=float)
    g = lambda k, p: med(lambda r: r['grad'].get(k, {}).get(p, float('nan')) if r['grad'] else float('nan'))  # noqa: E731
    sup = lambda p, k, key: med(lambda r: r['supports'][p][k][key])   # noqa: E731
    L['fm'] = (f"fm {_x(fm('mse_l'))}/{_x(fm('mse_g'))} t "
               + '/'.join(_x(med(lambda r, b=b: r['fm']['bins'][b]['mse_l'])) for b in bins)
               + f" wl {_x(fm('err_along_wl') / fm('tgt_along_wl'))} wj {_x(fm('err_along_wj') / fm('tgt_along_wj'))}"
               + f" grad ps {_x(g('post_sample', 'g'))}/{_x(g('post_sample', 'l'))} pr "
               f"{_x(g('prior', 'g'))}/{_x(g('prior', 'l'))} cos "
               f"{_x(med(lambda r: r['grad'].get('cos_wj', float('nan')) if r['grad'] else float('nan')))}"
               + f" pr l {_x(sup('l', 'prior', 'precision'))}/{_x(sup('l', 'prior', 'recall'))} nw "
               f"{_x(sup('l', 'nowrite', 'precision'))}/{_x(sup('l', 'nowrite', 'recall'))} g "
               f"{_x(sup('g', 'prior', 'precision'))}/{_x(sup('g', 'prior', 'recall'))}"
               + f" PR l {med(lambda r: r['supports']['l']['pr_post_mu']):.1f}/"
               f"{med(lambda r: r['supports']['l']['pr_prior']):.1f}")
    ga = lambda k, key: med(lambda r: r['grad']['along'][k][key] if r['grad'] else float('nan'))   # noqa: E731
    lb = lambda k, p: med(lambda r: abs(r['grad']['lin_bias'][k][p]) if r['grad'] else float('nan'))   # noqa: E731
    L['gd'] = (f"gd coh {_x(med(lambda r: r['grad']['coherence'] if r['grad'] else float('nan')))} "
               + ' '.join(f"{k}:{_x(med(lambda r, k=k: abs(r['grad']['along'][k]['offset']) if r['grad'] else float('nan')))}"
                          f"/{_x(ga(k, 'sd_ratio'))}/{_x(ga(k, 'w2'))}"
                          for k in ('prior', 'nowrite', 'g0', 'l0'))
               + ' lin ' + ' '.join(f"{k}:{_x(lb(k, 'g'))}/{_x(lb(k, 'l'))}/"
                                    f"{_x(med(lambda r, k=k: abs(r['yhat'][k]['actual_bias'])))}"
                                    for k in ('prior', 'nowrite', 'g0')))
    return L


def geom_lines(med, groups, order):
    out = []
    for v in order:
        r, m = groups[v], med[v]
        ph = r['p_high']
        out.append(
            f"{v}: w2 {'/'.join(_x(m[e]) for e in ('mu', 'prior', 'nowrite', 'g0', 'l0', 'g_prior', 'l_prior'))}"
            f" ph {'/'.join(_x(ph[e]) for e in ('truth', 'mu', 'prior', 'nowrite', 'g0', 'l0'))}"
            f" r2 {_r2(r['directions']['l']['r2_cv'])}/{_r2(r['directions']['j']['r2_cv'])}"
            f" wl {_sx(r['along']['l']['prior']['offset'])}/{_x(r['along']['l']['prior']['sd_ratio'])}"
            f" nw {_sx(r['along']['l']['nowrite']['offset'])}"
            f" grid {_x(r['grid'].get('out', {}).get('row', float('nan')))}/"
            f"{_x(r['grid'].get('out', {}).get('col', float('nan')))}"
            + (f" gd {_sx(r['grad']['along']['prior']['offset'])}/{_x(r['grad']['along']['prior']['sd_ratio'])}"
               f" lin {_sx(r['grad']['lin_bias']['prior']['l'])}/{_sx(r['yhat']['prior']['actual_bias'])}"
               if r['grad'] else ''))
    return out


FIG_OUT = ('post_sample', 'g_mean', 'l_mean', 'prior', 'nowrite', 'g0', 'l0', 'l_prior')
FIG_LAT = ('prior', 'nowrite', 'g0', 'l0')


def figures(spreads, groups, med, order, stem):
    """<stem>_out.png: per geometry (rows), the FE results' histogram against
    each ensemble's. <stem>_lat.png: the l latent along the geometry's decisive
    direction, posterior samples against the prior and its ablations, and the
    grid's decoded statistic (rows: g noise, columns: l noise)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:                                   # noqa: BLE001
        print(f'  (no figures: {exc})')
        return []
    R = len(order)
    written = []
    fig, axes = plt.subplots(R, len(FIG_OUT), figsize=(2.2 * len(FIG_OUT), 1.7 * R), squeeze=False)
    for i, v in enumerate(order):
        s = spreads[v]
        vals = np.concatenate([s['truth']] + [s[e] for e in FIG_OUT])
        bins = np.linspace(vals.min(), vals.max(), 31) if vals.max() > vals.min() else 10
        for j, e in enumerate(FIG_OUT):
            ax = axes[i, j]
            ax.hist(s['truth'], bins=bins, color='0.6', alpha=0.7, density=True)
            ax.hist(s[e], bins=bins, histtype='step', color='C3', lw=1.3, density=True)
            ax.set_title(f'{e} {_x(med[v].get(e))}' if i == 0 else _x(med[v].get(e)), fontsize=7)
            ax.tick_params(labelsize=5)
            ax.set_yticks([])
        axes[i, 0].set_ylabel(str(v), fontsize=8)
    fig.suptitle('grey: FE results, red: ensemble (title: W2 x100)', fontsize=8)
    fig.tight_layout()
    written.append(f'{stem}_out.png')
    fig.savefig(written[-1], dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(R, len(FIG_LAT) + 1, figsize=(2.2 * (len(FIG_LAT) + 1), 1.7 * R),
                             squeeze=False)
    for i, v in enumerate(order):
        pj = {k: np.asarray(a) for k, a in groups[v]['proj_l'].items()}
        vals = np.concatenate(list(pj.values()))
        bins = np.linspace(vals.min(), vals.max(), 31) if vals.max() > vals.min() else 10
        for j, e in enumerate(FIG_LAT):
            ax = axes[i, j]
            ax.hist(pj['post_sample'], bins=bins, color='0.6', alpha=0.7, density=True)
            ax.hist(pj[e], bins=bins, histtype='step', color='C0', lw=1.3, density=True)
            a = groups[v]['along']['l'][e]
            ax.set_title(f"{e} off {a['offset']:+.2f} sd {a['sd_ratio']:.2f}", fontsize=6)
            ax.tick_params(labelsize=5)
            ax.set_yticks([])
        axes[i, 0].set_ylabel(str(v), fontsize=8)
        ax = axes[i, -1]
        tab = groups[v]['grid'].get('table')
        if tab:
            im = ax.imshow(np.asarray(tab), cmap='viridis', aspect='auto')
            fig.colorbar(im, ax=ax, fraction=0.05).ax.tick_params(labelsize=5)
            o = groups[v]['grid']['out']
            ax.set_title(f"grid g {_x(o['row'])} l {_x(o['col'])}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('l along w_l (grey: posterior samples, R2 per geometry in the brief); '
                 'grid: rows = g noise, columns = l noise', fontsize=8)
    fig.tight_layout()
    written.append(f'{stem}_lat.png')
    fig.savefig(written[-1], dpi=110)
    plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True, help='the INFERENCE config the sweep scored')
    ap.add_argument('--modelpath', default=None, help="another checkpoint than the config's")
    ap.add_argument('--gpu', type=int, default=None, help="card; default the config's gpu_ids")
    ap.add_argument('--batch', type=int, default=8, help='graphs per forward (memory only)')
    ap.add_argument('--grad-batch', type=int, default=2,
                    help='graphs per decoder backward (memory only; ~17 GiB each at 86k nodes)')
    ap.add_argument('--splits', type=int, default=SPLITS)
    ap.add_argument('--grad-draws', type=int, default=16,
                    help='posterior means per geometry for the decoder gradient (0 = skip)')
    ap.add_argument('--out', default=None, help='default: <inference_output_dir>/../diag')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--split', default='heldout', choices=('heldout', 'train', 'val', 'test'))
    ap.add_argument('--dataset', default=None, help='another HDF5 than the split reads')
    ap.add_argument('--n-groups', type=int, default=None, help='N geometries, evenly spaced')
    ap.add_argument('--max-groups', type=int, default=None, help='smoke tests only')
    ap.add_argument('--max-per-group', type=int, default=None, help='smoke tests only')
    a = ap.parse_args()

    cfg = load_config(_abs(a.config))
    if a.modelpath:
        cfg['modelpath'] = _abs(a.modelpath)
    dev = resolve_device(cfg, a.gpu)
    ch, stat = _spread_index(cfg)
    torch.manual_seed(1234)
    np.random.seed(1234)

    ck = torch.load(cfg['modelpath'], map_location=dev, weights_only=False)
    norm = ck['normalization']
    for k, v in migrate_checkpoint_config(ck.get('model_config', {})).items():
        cfg[k] = v
    if str(cfg.get('latent_type', 'global')).lower() != 'hybrid':
        raise SystemExit(f"latent_anatomy takes a hybrid model apart; this one is "
                         f"latent_type {cfg.get('latent_type', 'global')}")
    if cfg.get('use_node_types') and norm.get('num_node_types'):
        cfg['num_node_types'] = norm['num_node_types']
    model = _load_model_from_checkpoint(cfg, ck, dev)
    prior = _load_conditional_prior(cfg, ck, model, dev)
    if prior is None or not getattr(prior, 'hybrid', False):
        raise SystemExit('this checkpoint has no hybrid coarse FM prior')
    model.eval()
    prior.eval()
    temperature = float(cfg.get('prior_temperature', 1.0))
    infl = cfg.get('latent_inflation', 1.0)
    lam = float(infl[0] if isinstance(infl, (list, tuple)) else infl)
    if isinstance(cfg.get('hierarchy_seed'), (list, tuple)):
        cfg['hierarchy_seed'] = cfg['hierarchy_seed'][0]

    attr = cfg.get('split_group_attr')
    if a.dataset:
        h5_path = _abs(a.dataset)
    elif a.split == 'heldout':
        h5_path = cfg.get('eval_dataset') or cfg.get('infer_dataset')
    else:
        h5_path = cfg['dataset_dir']
    if not attr:
        raise SystemExit('the config has no split_group_attr')
    groups_attr = _eval_dataset_groups(h5_path, attr)
    if not groups_attr:
        raise SystemExit(f'{h5_path}: not every scene carries metadata attr {attr!r}')
    order = _group_order(groups_attr.values())
    if a.split != 'heldout':
        part = grouped_split_ids(h5_path, sorted(groups_attr, key=int), attr, 0.8, 0.1,
                                 int(cfg.get('split_seed', 42)))[('train', 'val', 'test').index(a.split)]
        keep = {str(groups_attr[s]) for s in part}
        order = [v for v in order if v in keep]
    if a.n_groups and len(order) > a.n_groups:
        order = [order[i] for i in np.linspace(0, len(order) - 1, a.n_groups).round().astype(int)]
    order = order[:a.max_groups]
    input_var, out_var = int(cfg['input_var']), int(cfg['output_var'])
    cond_var = int(cfg.get('cond_var', 0) or 0)
    rows = 3 + np.asarray(ch)

    probe = TokenProbe(prior)
    spreads, groups = {}, {}
    try:
        with h5py.File(h5_path, 'r') as f:
            for gi, v in enumerate(order):
                ids = sorted((s for s, g in groups_attr.items() if g == v), key=int)[:a.max_per_group]
                nd0 = f[f'data/{ids[0]}/nodal_data'][:, -1, :]
                edge0 = f[f'data/{ids[0]}/mesh_edge'][:]
                fields, truths = [], []
                for s in ids:
                    nd = f[f'data/{s}/nodal_data'][:, -1, :]
                    same = (nd.shape == nd0.shape and np.array_equal(nd[0:3], nd0[0:3])
                            and np.array_equal(nd[3 + input_var:3 + input_var + cond_var],
                                               nd0[3 + input_var:3 + input_var + cond_var])
                            and np.array_equal(nd[-1], nd0[-1])
                            and np.array_equal(f[f'data/{s}/mesh_edge'][:], edge0))
                    if not same:
                        raise SystemExit(f'{attr}={v}: scene {s} differs from scene {ids[0]}; '
                                         f'this diagnostic needs one graph per group')
                    fields.append(nd[3:3 + out_var].T.astype(np.float32))
                    truths.append(_spread_stat(nd[rows].T if rows.ndim else nd[int(rows)], stat))
                part_ids = (nd0[-1].astype(np.int32)
                            if cfg.get('use_node_types') and nd0.shape[0] > 7 else None)
                cond = (nd0[3 + input_var:3 + input_var + cond_var].T.astype(np.float32)
                        if cond_var else None)
                ctx = _SampleContext(cfg, norm, nd0[0:3].T, np.concatenate([edge0, edge0[[1, 0]]], axis=1),
                                     part_ids, dev, cond_feat=cond)
                geo = Geometry(model, prior, ctx, ch, stat, a.batch, prior.num_steps, temperature, lam,
                               a.grad_batch)
                print(f'  {attr}={v}: {len(ids)} FE results, {ctx.num_nodes} nodes, '
                      f'{geo.n_c} coarse nodes', flush=True)
                try:
                    spreads[v], groups[v] = run_group(geo, probe, truths, fields, 1000 + gi, a.grad_draws)
                except torch.cuda.OutOfMemoryError:
                    gib = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
                    raise SystemExit(f'CUDA OOM {attr}={v} {ctx.num_nodes} nodes, in {geo.stage} at one '
                                     f'graph (card {gib:.0f} GiB, peak '
                                     f'{torch.cuda.max_memory_allocated(dev) / 2 ** 30:.0f} GiB)')
                del geo
                torch.cuda.empty_cache()
    finally:
        probe.close()

    scores, med, w2parts, k = split_scores(spreads, order, a.splits)
    pair = {}
    for e in ALIGNED:
        pair[e] = paired(np.concatenate([spreads[v][e] - spreads[v]['truth'].mean() for v in order]),
                         np.concatenate([spreads[v]['truth'] - spreads[v]['truth'].mean() for v in order]))
        pair[e]['vs_mu_rmse_over_sd'] = float(np.sqrt(np.mean(np.concatenate(
            [(spreads[v][e] - spreads[v]['mu']) / (spreads[v]['truth'].std() or 1.0) for v in order]) ** 2)))
    lines = brief_lines(scores, pair, groups, order)
    glines = geom_lines(med, groups, order)

    tag = a.tag or os.path.splitext(os.path.basename(a.config))[0].replace('config_infer_', '')
    out_dir = _abs(a.out) if a.out else os.path.join(
        os.path.dirname(os.path.normpath(cfg.get('inference_output_dir', '.'))), 'diag')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'latent_anatomy_{tag}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'config': a.config, 'modelpath': cfg['modelpath'], 'epoch': ck.get('epoch'),
                   'split': a.split, 'dataset': h5_path, 'attr': attr, 'k': k, 'splits': a.splits,
                   'prior_temperature': temperature, 'latent_inflation': lam,
                   'scores': scores, 'per_group_w2_median': med, 'per_group_parts_median': w2parts,
                   'paired_within_geometry': pair, 'groups': groups,
                   'brief': lines, 'brief_geom': glines,
                   'spreads': {v: {e: s.tolist() for e, s in spreads[v].items()} for v in order}},
                  fh, indent=1)

    pngs = figures(spreads, groups, med, order, os.path.join(out_dir, f'latent_anatomy_{tag}'))

    n_fe = sorted({spreads[v]['truth'].size for v in order})
    print(f'\n  wrote {out}' + ''.join(f'\n  wrote {p}' for p in pngs) + '\n')
    print('==== TYPE THIS BACK ====')
    print(f"anatomy {tag} ep={ck.get('epoch')} split={a.split} groups={len(order)} "
          f"n={n_fe[0] if len(n_fe) == 1 else n_fe} k={k}")
    for key in ('w2', 'pc', 'grid', 'dir', 'fm', 'gd'):
        print(lines[key])
    for ln in glines:
        print(ln)
    print('==== END ====')
    return 0


if __name__ == '__main__':
    sys.exit(main())
