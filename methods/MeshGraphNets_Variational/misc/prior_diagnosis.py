"""What is wrong with the coarse FM prior of a local / hybrid MGN-V model, and
how much: its input, its output and its error, per geometry. Nothing trains.

    cd methods/MeshGraphNets_Variational
    python misc/prior_diagnosis.py \
        --config ../../configs/MeshGraphNets_Variational/hyperparameter_sweep/r9/config_infer_hybrid_base_s43.txt \
        [--gpu 0] [--batch 8] [--split heldout|train|test] [--n-groups N] [--out DIR] [--tag NAME]

Prints a verdict in plain English sentences, then a CAUSES table that marks
every candidate cause YES / no / ? with the number behind it
(misc/prior_verdict.py), and ends with a ==== TYPE THIS BACK ==== block.
Writes prior_diagnosis_<tag>.json
(tag: ho / tr / ts). Geometries, splits and scoring are those of
posterior_vs_prior_groups.py: one geometry per `split_group_attr` value, W2 of
the spread statistic over the FE half's sd, rms over the geometries, median
over 200 half-vs-half splits.

THE REFERENCE  For one geometry the prior's target is the mixture of its FE
results' posteriors, (1/n) sum_i N(mu_i, sigma_i^2). Its flow-matching optimum
has a closed form (the `oracle` field below), so the model's field is compared
with the exact answer for THIS geometry, not with a loss value. Everything is
in the prior's standardized space u = (z - shift) / scale, one vector per draw:
[g slots | l, node by node].

ENSEMBLES  (each as many draws as the geometry has FE results, same noise)

    floor  FE half against FE half
    P      posterior samples, decoded                 encoder + decoder alone
    O      the oracle field through the model's sampler and steps   (= P when
           the steps resolve the target; O - P is what a perfect field loses)
    Q      the prior: the model's field, the model's sampler
    Qf     Q with 4x the steps                        Q - Qf: step count
    R5     oracle field for t < 0.5, the model's after   Q - R5: early half
    Qc     Q, each coordinate moved to P's mean       Q - Qc: mean position
    Qcs    ... and rescaled to P's sd                 Qc - Qcs: width
    Qm     ... and rank-mapped onto P's values        Qcs - Qm: marginal shape
           (P's exact per-coordinate distributions,   Qm - P: joint structure
           Q's dependence between coordinates)
    S      no decode: every Q draw read off as the decoded value of the nearest
           posterior component.  Q - S: draws off the posterior's support (the
           decoder's share);  S - P: the wrong share per component
    c10..c90  the model's field until t, the oracle after, read off like S:
           when the draw commits
    Qg     hybrid: the prior's z_g with a posterior sample's z_l
    Ql     hybrid: a posterior sample's z_g with the prior's z_l
    Px     hybrid: a posterior sample's z_g with ANOTHER posterior sample's
           z_l. Qg and Ql pair two unrelated draws, so they are read against
           Px, not P:  Qg - Px = what the prior's z_g costs,  Ql - Px = its z_l
    Qx     hybrid: the prior's z_g with ANOTHER prior draw's z_l. Px - P is
           what the pairing of z_g with z_l is worth; Qx = Q says the prior's
           own pairing carries nothing
    Pr Qr  a checkpoint with EMA weights also holds the trained ones. Pr, Qr
           are P and Q with those (encoder, prior and decoder all non-EMA):
           (Q - P) - (Qr - Pr) is what the EMA costs the prior
    M      the point halfway between two posteriors of this geometry (the mean
           of FE result i and of FE result i + n/2, with the first one's
           noise), decoded: what the decoder does with a draw that lies
           BETWEEN two outcomes. It is one direction off the posterior, the
           one a prior that averages two outcomes takes; see mid_out, mid_snap

NUMBERS PER GEOMETRY

    ve     |v_model - v_oracle|^2 / |v_oracle|^2 at t = .1 .3 .5 .7 .9 on the
           path from noise to posterior samples: no irreducible part, 0 is a
           perfect field. Next to it the same for a field that ignores the
           geometry (target N(0, I)) and for the geometry's best diagonal
           Gaussian. `bias` = the share of the error that is one constant
           vector (a mean error) rather than draw-dependent.
    off    distance of a draw from its nearest posterior component, in that
           component's sigma (1 = a posterior sample).   cov = components
           reached over what n exact draws reach.   rd = reliability of S:
           held-out posterior samples read off the same way, correlation with
           their own decoded value.
    far    distance of the geometry's input from the training geometries',
           over the training geometries' own nearest-neighbour distance:
           far_emb on the prior's conditioning embedding (mean and sd over the
           coarse nodes), far_raw on the raw normalized node / edge features.
           raw_out = how far a raw feature statistic lies outside the training
           range, in training sd.
    track  (per part) R^2 of the prior cloud's centre and width against the
           posterior cloud's, across the geometries: does the output follow
           the input at all.
    pit / oor / share   FE results outside the prior's 5-95% band, prior draws
           outside the FE range, the upper-mode share on two-mode geometries.
    ov     one posterior's sigma over the spread of the means: below ~0.3 the
           target is isolated points.
    enc_shift  posterior means of the trained encoder against the EMA
           encoder's, in posterior sigma: 0 = the encoder the prior was
           trained on is the one this script reads the posterior from.
    mid_out / mid_snap   from M. With a, b the decoded values of the two
           posterior samples and m the decoded value of their midpoint:
           mid_out = rms of how far m lies OUTSIDE [a, b], over the FE sd
           (0 = always between the two: no holes between posteriors);
           mid_snap = median of min(1, 2 |(m - a) / (b - a) - 0.5|) over the
           pairs with |b - a| > 0.5 FE sd: 0 = m is the average of the two
           (the decoder blends, a draw between two outcomes gives an
           in-between result), 1 = m equals one of them (the decoder snaps:
           only which posterior a draw is nearest to matters).

TRAINING  The training config beside the inference config
(config_infer_X.txt -> config_train_X.txt, or --train-config) gives the epoch
count, the prior freeze, `best_by` and the epoch log. From the log: where the
kept epoch lies among the epochs it was chosen from, and whether validation
CRPS was still falling at the end (stopped too early) or had turned up
(overfit). Without that config or log those two causes print `?`.

The % in the verdict are differences of two scores over the gap Q - P. They
are by construction (the order of a ladder's steps is a choice); there is no
threshold from the literature in here, the unit of "noticeable" is the floor's
own p95 - p50 over the splits.

WHAT ONE CHECKPOINT CANNOT TELL  (printed as `?`, never as "no")
  * what training the prior and the encoder together would give instead of a
    prior fitted to a frozen posterior: that needs such a run;
  * how another epoch's weights would score on these geometries: only the
    validation curve is known. One other epoch can be run: `<stem>.joint.pth`
    (the joint phase's best, kept at the freeze). `--tag jt --split heldout
    --modelpath <stem>.joint.pth --train-ref 0 --no-raw` writes the `jt` part,
    and with it beside `ho` the verdict says what the prior-only epochs
    bought (cause 38).
"""
import argparse
import json
import math
import os
import sys
import time

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
from model.conditional_prior import (                             # noqa: E402
    CoarseFMPrior, _autocast_disabled_for, _fm_time_steps,
)
from model.latent import Latent                                   # noqa: E402
from posterior_vs_prior import resolve_device                     # noqa: E402
from posterior_vs_prior_groups import (                           # noqa: E402
    SPLIT_SEED, SPLITS, _abs, geometry_row, summarize,
)
import prior_verdict as PV                                        # noqa: E402

os.chdir(REPO_ROOT)   # config paths are relative to the method directory

TAUS = (0.1, 0.25, 0.5, 0.75, 0.9)
COMMIT = tuple(f'c{int(round(100 * t))}' for t in TAUS)
DECODED = ('P', 'O', 'Q', 'Qf', 'R5', 'Qc', 'Qcs', 'Qm', 'Qg', 'Ql', 'Px', 'Qx', 'M')   # with the EMA weights
MID_PAIRS = 4                # mid_snap needs this many pairs whose two ends differ by > 0.5 FE sd
DECODED_RAW = ('Pr', 'Qr')                                                    # with the trained ones
ENSEMBLES = ('floor',) + DECODED[:4] + ('S',) + DECODED[4:] + DECODED_RAW + COMMIT
STD_BUFFERS = ('prior.z_shift', 'prior.z_scale', 'prior.zl_shift', 'prior.zl_scale')
VE_T = (0.1, 0.3, 0.5, 0.7, 0.9)
FINE = 4                     # Qf: this many times the model's sampler steps
BC_BIMODAL = 5.0 / 9.0
PART_TAG = {'heldout': 'ho', 'train': 'tr', 'test': 'ts'}


# ─────────────────────────────────────────────────────────────────────────────
# statistics
# ─────────────────────────────────────────────────────────────────────────────

def split_scores(spreads, order, splits=SPLITS):
    """posterior_vs_prior_groups.split_scores over this script's ensembles (the
    same permutations, so the same floor)."""
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


def rms(x):
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float('nan')


def corr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def rank_map(Q, P):
    """Q with every coordinate's values replaced by P's of the same rank: P's
    exact per-coordinate distributions, Q's dependence between coordinates."""
    ranks = np.argsort(np.argsort(Q, axis=0), axis=0)
    return np.take_along_axis(np.sort(P, axis=0), ranks, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# the exact field of a Gaussian mixture, and the nearest component
# ─────────────────────────────────────────────────────────────────────────────

def oracle_field(Mu, S2, s_):
    """The flow-matching optimum for the target (1/n) sum_i N(Mu_i, diag S2_i)
    on the path x_t = (1 - s_ t) z0 + t z1, z0 ~ N(0, I):

        v(x, t) = sum_i w_i(x, t) [Mu_i + coef_i (x - t Mu_i)]
        var_i = (1 - s_ t)^2 + t^2 S2_i,  coef_i = (t S2_i - s_ (1 - s_ t)) / var_i
        w_i ~ N(x; t Mu_i, var_i)

    Mu, S2 [n, d] double on the device; returns field(X [M, d], t) -> [M, d].
    """
    n, d = Mu.shape
    chunk = max(1, int(3e7 // (n * d)))

    def field(X, t):
        a = 1.0 - s_ * t
        var = a * a + (t * t) * S2
        coef = (t * S2 - s_ * a) / var
        logdet = var.log().sum(1)
        out = []
        for i in range(0, X.shape[0], chunk):
            diff = X[i:i + chunk, None, :] - t * Mu[None]
            w = torch.softmax(-0.5 * ((diff * diff / var[None]).sum(-1) + logdet[None]), dim=1)
            out.append(torch.einsum('mn,mnd->md', w, Mu[None] + coef[None] * diff))
        return torch.cat(out)
    return field


def snap(X, Mu, S2, sigma_min):
    """The most responsible component of each x at t = 1 and its distance in
    that component's sigma (rms over the coordinates; 1 for a posterior sample)."""
    n, d = Mu.shape
    var = sigma_min ** 2 + S2
    logdet = var.log().sum(1)
    chunk = max(1, int(3e7 // (n * d)))
    idx, off = [], []
    for i in range(0, X.shape[0], chunk):
        d2 = ((X[i:i + chunk, None, :] - Mu[None]) ** 2 / var[None]).sum(-1)
        j = (-0.5 * (d2 + logdet[None])).argmax(1)
        idx.append(j)
        off.append((d2.gather(1, j[:, None])[:, 0] / d).sqrt())
    return torch.cat(idx).cpu().numpy(), torch.cat(off).cpu().numpy()


def coverage(idx, n):
    """Distinct components hit over what as many exact draws would hit."""
    m = len(idx)
    return float(len(set(int(i) for i in idx)) / (n * (1.0 - (1.0 - 1.0 / n) ** m)))


# ─────────────────────────────────────────────────────────────────────────────
# one geometry
# ─────────────────────────────────────────────────────────────────────────────

class Geo:
    """Encode / decode / prior field for one geometry's graph, in u space."""

    def __init__(self, model, prior, ctx, ch, stat, batch, temperature, inflation, raw=None):
        """raw: (model, prior) built from the checkpoint's trained (non-EMA)
        weights, or None; the main pair is what inference loads."""
        self.model, self.prior, self.ctx = model, prior, ctx
        self.inner = getattr(model, 'model', model)
        self.ch, self.stat, self.batch = ch, stat, batch
        self.flow_chunk = 256
        self.stage = 'setup'
        self.amp, self.inflation = math.sqrt(max(temperature, 1e-6)), inflation
        self.dm = np.asarray(ctx.delta_mean, np.float32)
        self.dsd = np.asarray(ctx.delta_std, np.float32)
        self.base = ctx.build_step_graph(np.zeros((ctx.num_nodes, ctx.input_dim), dtype=np.float32))
        self.N = ctx.num_nodes
        self.dev = self.base.x.device
        self._cache, self._ctx = {}, {}
        self.hybrid = bool(getattr(prior, 'hybrid', False))
        self.C = prior.latent_ch
        self.S, self.D = prior.num_z, prior.z_dim
        with torch.no_grad():
            self.cond1 = prior.condition(self.graphs(1))
            self.cond_f = prior._float_ctx(self.cond1)
        self.n_c = int(self.cond1['x'].shape[0])
        self.dg = int(prior.flat_dim) if self.hybrid else 0
        self.dl = self.n_c * self.C
        self.d = self.dg + self.dl
        self.shift, self.scale = self._standard(prior)
        self.v_model = self._velocity('ema', prior, self.cond1, self.cond_f)
        self.raw = None
        if raw is not None:
            raw_model, raw_prior = raw
            with torch.no_grad():
                cond1 = raw_prior.condition(self.graphs(1))
                cond_f = raw_prior._float_ctx(cond1)
            shift, scale = self._standard(raw_prior)
            self.raw = {'model': raw_model, 'inner': getattr(raw_model, 'model', raw_model),
                        'shift': shift, 'scale': scale,
                        'field': self._velocity('raw', raw_prior, cond1, cond_f)}

    def _standard(self, prior):
        """(shift, scale) of a prior's standardized space, one entry per coordinate."""
        shift = np.tile(prior.zl_shift.double().cpu().numpy(), self.n_c)
        scale = np.tile(prior.zl_scale.double().cpu().numpy(), self.n_c)
        if self.hybrid:
            shift = np.concatenate([prior.z_shift.double().cpu().numpy(), shift])
            scale = np.concatenate([prior.z_scale.double().cpu().numpy(), scale])
        return shift, scale

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
                self._ctx = {}
                torch.cuda.empty_cache()
                setattr(self, attr, max(1, m // 2))
                print(f'  CUDA OOM in {self.stage} at {attr}={m}: now {getattr(self, attr)}', flush=True)
        return out

    # encode / decode -----------------------------------------------------

    def encode(self, fields, raw=False):
        """Posterior means and sigmas of the FE results in z space, [n, d] each
        (raw: from the trained weights' encoder)."""
        inner = self.raw['inner'] if raw else self.inner

        def body(s, m):
            with torch.no_grad():
                _, mu, logvar = inner.posterior(self.graphs(m, fields[s:s + m]))
            ml = mu.l.double().cpu().numpy().reshape(m, -1)
            sl = np.exp(0.5 * logvar.l.double().cpu().numpy()).reshape(m, -1)
            if not self.dg:
                return ml, sl
            return (np.concatenate([mu.g.double().cpu().numpy().reshape(m, -1), ml], axis=1),
                    np.concatenate([np.exp(0.5 * logvar.g.double().cpu().numpy()).reshape(m, -1), sl], axis=1))
        mu, sg = zip(*self._chunks(len(fields), 'batch', body))
        mu, sg = np.concatenate(mu), np.concatenate(sg)
        if mu.shape[1] != self.d:
            raise RuntimeError(f'the posterior has {mu.shape[1]} latent coordinates, the prior {self.d}')
        return mu, sg

    def decode(self, U, raw=False):
        """Spread statistic of each decoded draw, U [M, d] in u space (raw: U in
        the trained weights' u space, decoded by the trained weights)."""
        src = self.raw if raw else {'model': self.model, 'shift': self.shift, 'scale': self.scale}
        model = src['model']
        Z = np.ascontiguousarray(U * src['scale'] + src['shift'], np.float32)

        def body(s, m):
            z = torch.from_numpy(Z[s:s + m]).to(self.dev)
            lat = Latent(z[:, :self.dg].reshape(m, self.S, self.D) if self.dg else None,
                         z[:, self.dg:].reshape(m * self.n_c, self.C))
            with torch.no_grad():
                pred = model(self.graphs(m), add_noise=False, use_posterior=False, fixed_z=lat)[0]
            return self._stats_np(pred, m)
        return np.asarray([v for c in self._chunks(len(Z), 'batch', body) for v in c], np.float64)

    # the model's field -----------------------------------------------------

    def _velocity(self, name, p, cond1, cond_f):
        """field(X [M, d] double on the device, t) -> a prior network's velocity."""
        def field(X, t):
            def body(s, m):
                held = self._ctx.get(name)
                if held is None or held[0] != m:
                    held = self._ctx[name] = (m, p._replicate(cond_f, m))
                x = X[s:s + m].float()
                zl = x[:, self.dg:].reshape(m * self.n_c, self.C)
                zg = x[:, :self.dg] if self.dg else None
                tt = torch.full((m, 1), float(t), device=self.dev)
                with torch.no_grad(), _autocast_disabled_for(cond1['x']):
                    vl, vg = p.velocity(zl, zg, tt, held[1])
                vl = vl.double().reshape(m, -1)
                return vl if vg is None else torch.cat([vg.double().reshape(m, -1), vl], dim=1)
            return torch.cat(self._chunks(X.shape[0], 'flow_chunk', body))
        return field

    def flow(self, X, field, grid, k0=0, k1=None, save=()):
        """The prior's own solver over steps [k0, k1) of `grid` with any field.
        With `save`, also returns {step index: state}."""
        k1 = len(grid) if k1 is None else k1
        kept = {k0: X} if k0 in save else {}
        euler = self.prior.solver == 'euler'
        for k in range(k0, k1):
            t, dt = grid[k]
            v1 = field(X, t)
            if euler:
                X = X + dt * v1
            else:
                X = X + 0.5 * dt * (v1 + field(X + dt * v1, t + dt))
            if k + 1 in save:
                kept[k + 1] = X
        return (X, kept) if save else X

    def integrator_gap(self, X0, grid):
        """max |this script's model flow - CoarseFMPrior._integrate| on a few draws."""
        p = self.prior
        m = X0.shape[0]
        mine = self.flow(X0, self.v_model, grid)
        with torch.no_grad(), _autocast_disabled_for(self.cond1['x']):
            x = X0.float()
            zl, zg = p._integrate(x[:, self.dg:].reshape(m * self.n_c, self.C),
                                  x[:, :self.dg] if self.dg else None,
                                  p._replicate(self.cond_f, m), len(grid))
        ref = zl.double().reshape(m, -1)
        if zg is not None:
            ref = torch.cat([zg.double().reshape(m, -1), ref], dim=1)
        return float((mine - ref).abs().max())

    # input descriptors -----------------------------------------------------

    def descriptors(self):
        """(raw, embedding): what the prior is given, as one vector each."""
        def cols(t):
            t = t.double()
            return [t.mean(0), t.std(0)]
        raw = torch.cat(cols(self.base.x) + cols(self.base.edge_attr)).cpu().numpy()
        emb = cols(self.cond1['x'])
        if self.cond1.get('pooled') is not None:
            emb.append(self.cond1['pooled'].double().reshape(-1))
        return raw, torch.cat(emb).cpu().numpy()


def run_group(geo, truths, fields, seed, check=False):
    """Every ensemble and number of one geometry: (spreads, group dict)."""
    p = geo.prior
    rng = np.random.default_rng(seed)
    truth = np.asarray(truths, np.float64)
    n, d, dg = truth.size, geo.d, geo.dg
    s_ = 1.0 - float(p.sigma_min)
    dev = geo.dev

    def T(a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float64)).to(dev)

    geo.stage = 'encode'
    mu_z, sg_z = geo.encode(fields)
    Mu, Sg = (mu_z - geo.shift) / geo.scale, sg_z / geo.scale
    tMu, tS2 = T(Mu), T(Sg ** 2)
    eps = rng.standard_normal((n, d))
    U = {'P': Mu + Sg * eps}
    X0 = T(rng.standard_normal((n, d)) * geo.amp)

    fm, fo = geo.v_model, oracle_field(tMu, tS2, s_)
    K = int(p.num_steps)
    grid = _fm_time_steps(K, p.time_power)
    times = [g[0] for g in grid] + [1.0]
    ks = {name: int(np.argmin([abs(t - tau) for t in times])) for name, tau in zip(COMMIT, TAUS)}
    k5 = ks['c50']

    out = {'n': int(n), 'nodes': int(geo.N), 'n_c': int(geo.n_c), 'd_g': int(dg), 'd_l': int(geo.dl)}
    Ur = {}   # the trained (non-EMA) weights: their own posterior samples and prior draws, same noise
    if geo.raw is not None:
        r = geo.raw
        mu_r, sg_r = geo.encode(fields, raw=True)
        out['enc_shift'] = float(np.median(np.sqrt((((mu_r - mu_z) / sg_z) ** 2).mean(1))))
        Ur['Pr'] = (mu_r - r['shift']) / r['scale'] + sg_r / r['scale'] * eps
        geo.stage = 'flow (trained weights)'
        XQr = geo.flow(X0, r['field'], grid)
        if geo.inflation != 1.0:
            XQr = (c := geo.flow(torch.zeros_like(X0[:1]), r['field'], grid)) + geo.inflation * (XQr - c)
        Ur['Qr'] = XQr.cpu().numpy()
        del XQr
        geo._ctx.pop('raw', None)

    geo.stage = 'flow'
    if check:
        out['integrator_gap'] = geo.integrator_gap(X0[:4], grid)
        x = X0[:4]
        out['oracle_gap_t0'] = float((fo(x, 0.0) - (tMu.mean(0) - s_ * x)).abs().max())
    XQ, kept = geo.flow(X0, fm, grid, save=set(ks.values()))
    fine = _fm_time_steps(FINE * K, p.time_power)
    XQf = geo.flow(X0, fm, fine)
    if geo.inflation != 1.0:
        zero = torch.zeros_like(X0[:1])
        XQ = (c := geo.flow(zero, fm, grid)) + geo.inflation * (XQ - c)
        XQf = (c := geo.flow(zero, fm, fine)) + geo.inflation * (XQf - c)
    XO = geo.flow(X0, fo, grid)
    XR = geo.flow(geo.flow(X0, fo, grid, 0, k5), fm, grid, k5)
    U.update({'Q': XQ.cpu().numpy(), 'Qf': XQf.cpu().numpy(), 'O': XO.cpu().numpy(),
              'R5': XR.cpu().numpy()})
    mP, mQ = U['P'].mean(0), U['Q'].mean(0)
    sP, sQ = U['P'].std(0), U['Q'].std(0)
    U['Qc'] = U['Q'] - mQ + mP
    U['Qcs'] = (U['Q'] - mQ) * (sP / np.maximum(sQ, 1e-6)) + mP
    U['Qm'] = rank_map(U['Q'], U['P'])
    other = np.roll(np.arange(n), n // 2)   # another FE result's posterior sample
    if n > 1:   # halfway between two posteriors of this geometry
        U['M'] = 0.5 * (Mu + Mu[other]) + Sg * eps
    if dg:   # which latent: one from the prior, the other from a posterior sample
        U['Qg'] = np.concatenate([U['Q'][:, :dg], U['P'][:, dg:]], axis=1)
        U['Ql'] = np.concatenate([U['P'][:, :dg], U['Q'][:, dg:]], axis=1)
        U['Px'] = np.concatenate([U['P'][other, :dg], U['P'][:, dg:]], axis=1)
        U['Qx'] = np.concatenate([U['Q'][other, :dg], U['Q'][:, dg:]], axis=1)

    geo.stage = 'decode'
    spreads = {'truth': truth}
    for e in DECODED:
        if e in U:
            spreads[e] = geo.decode(U[e])
    for e in DECODED_RAW:
        if e in Ur:
            spreads[e] = geo.decode(Ur[e], raw=True)

    # between two posteriors: does the decoder blend them, snap to one, or break
    if 'M' in spreads:
        a, b, m = spreads['P'], spreads['P'][other], spreads['M']
        sd = float(truth.std())
        if sd > 0:
            out['mid_out'] = rms(np.maximum(0.0, np.maximum(np.minimum(a, b) - m, m - np.maximum(a, b)))) / sd
            apart = np.abs(b - a) > 0.5 * sd
            out['mid_pairs'] = int(apart.sum())
            if out['mid_pairs'] >= MID_PAIRS:
                lam = (m[apart] - a[apart]) / (b[apart] - a[apart])
                out['mid_snap'] = float(np.median(np.minimum(1.0, 2.0 * np.abs(lam - 0.5))))

    # nearest posterior component: no decode ------------------------------
    geo.stage = 'snap'
    sm = float(p.sigma_min)
    idx, off, cov = {}, {}, {}
    for e, X in (('P', T(U['P'])), ('Q', XQ), ('O', XO)):
        idx[e], o = snap(X, tMu, tS2, sm)
        off[e], cov[e] = float(np.median(o)), coverage(idx[e], n)
    spreads['S'] = spreads['P'][idx['Q']]
    for name in COMMIT:
        k = ks[name]
        spreads[name] = spreads['P'][snap(geo.flow(kept[k], fo, grid, k), tMu, tS2, sm)[0]]
    half = rng.permutation(n)
    A, B = half[: n // 2], half[n // 2:]
    jb, _ = snap(T(U['P'][B]), tMu[torch.as_tensor(A, device=dev)], tS2[torch.as_tensor(A, device=dev)], sm)
    out.update({'off': off, 'cov': cov, 'rd': corr(spreads['P'][B], spreads['P'][A][jb]),
                'self_hit': float(np.mean(idx['P'] == np.arange(n)))})

    # the field's error against the exact one ------------------------------
    geo.stage = 'velocity'
    one = oracle_field(torch.zeros(1, d, dtype=torch.float64, device=dev),
                       torch.ones(1, d, dtype=torch.float64, device=dev), s_)
    gauss = oracle_field(tMu.mean(0, keepdim=True), (tMu.var(0, unbiased=False) + tS2.mean(0))[None], s_)
    ve = {key: [] for key in ('model', 'model_g', 'model_l', 'bias', 'g0', 'g1')}
    up = T(np.tile(U['P'], (2, 1)))
    for t in VE_T:
        x = (1.0 - s_ * t) * T(rng.standard_normal((2 * n, d))) + t * up
        vo = fo(x, t)
        e = fm(x, t) - vo
        den = float((vo ** 2).sum())
        ve['model'].append(float((e ** 2).sum()) / den)
        ve['model_l'].append(float((e[:, dg:] ** 2).sum()) / float((vo[:, dg:] ** 2).sum()))
        ve['model_g'].append(float((e[:, :dg] ** 2).sum()) / float((vo[:, :dg] ** 2).sum()) if dg else None)
        ve['bias'].append(float((e.mean(0) ** 2).sum()) / float((e ** 2).mean(0).sum()))
        ve['g0'].append(float(((one(x, t) - vo) ** 2).sum()) / den)
        ve['g1'].append(float(((gauss(x, t) - vo) ** 2).sum()) / den)
    out['ve'] = dict(ve, t=list(VE_T))

    # latent clouds ----------------------------------------------------------
    def part(lo, hi, tag):
        if hi <= lo:
            return
        cP, cQ = mP[lo:hi], mQ[lo:hi]
        wP = rms(sP[lo:hi])
        out[f'ov_{tag}'] = rms(Sg[:, lo:hi]) / (rms(Mu[:, lo:hi].std(0)) or float('nan'))
        out[f'u_rms_{tag}'] = rms(U['P'][:, lo:hi])
        out[f'u_between_{tag}'] = rms(cP)
        out[f'u_within_{tag}'] = wP
        out[f'cen_{tag}'] = rms(cQ - cP) / (wP or float('nan'))
        out[f'sdr_{tag}'] = rms(sQ[lo:hi]) / (wP or float('nan'))
    part(0, dg, 'g')
    part(dg, d, 'l')

    def cloud(m, s):
        """A cloud as numbers comparable across geometries: g as it is; l per
        channel over the nodes (the nodes of two geometries do not correspond)."""
        ml, sl = m[dg:].reshape(geo.n_c, geo.C), s[dg:].reshape(geo.n_c, geo.C)
        return {'g': np.concatenate([m[:dg], s[:dg]]).tolist(),
                'l': np.concatenate([ml.mean(0), np.sqrt((ml ** 2).mean(0)),
                                     np.sqrt((sl ** 2).mean(0))]).tolist()}
    out['cloud'] = {'P': cloud(mP, sP), 'Q': cloud(mQ, sQ)}

    # the output statistic ----------------------------------------------------
    q = spreads['Q']
    lo, hi = np.percentile(q, [5, 95])
    out.update({'pit_lo': float(np.mean(truth < lo)), 'pit_hi': float(np.mean(truth > hi)),
                'oor': float(np.mean((q < truth.min()) | (q > truth.max()))),
                'fe_bc': bimodality(truth), 'fe_mean': float(truth.mean()), 'fe_sd': float(truth.std())})
    out['fe_bimodal'] = bool(out['fe_bc'] > BC_BIMODAL)
    cut = two_means_threshold(truth)
    out['share'] = {e: float(np.mean((truth if e == 'truth' else spreads[e]) > cut))
                    for e in ('truth', 'P', 'Q', 'S')}
    return spreads, out


# ─────────────────────────────────────────────────────────────────────────────
# input: how far from the training geometries
# ─────────────────────────────────────────────────────────────────────────────

def far_from(ref, x, skip=None):
    """Nearest-reference distance of x over the reference's own median
    nearest-neighbour distance, in reference-standardized coordinates."""
    ref = np.asarray(ref, np.float64)
    sd = ref.std(0)
    keep = sd > 1e-9 * max(float(sd.max()), 1e-30)
    if ref.shape[0] < 3 or not keep.any():
        return float('nan')
    R = (ref[:, keep] - ref[:, keep].mean(0)) / sd[keep]
    D = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    base = float(np.median(D.min(1)))
    dx = np.sqrt((((np.asarray(x, np.float64)[keep] - ref[:, keep].mean(0)) / sd[keep] - R) ** 2).sum(-1))
    if skip is not None:
        dx[skip] = np.inf
    return float(dx.min() / base) if base > 0 else float('nan')


def outside(ref, x):
    """(largest distance of a coordinate of x outside the reference's range, in
    reference sd; its index)."""
    ref, x = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    sd = ref.std(0)
    keep = sd > 1e-9 * max(float(sd.max()), 1e-30)
    ex = np.maximum(np.maximum(x - ref.max(0), ref.min(0) - x), 0.0)
    ex = np.where(keep, ex / np.where(keep, sd, 1.0), 0.0)
    return float(ex.max()), int(ex.argmax())


def scene(f, sid, cfg):
    """(nodal data, edges, part ids, conditions) of one scene, for a _SampleContext."""
    input_var, cond_var = int(cfg['input_var']), int(cfg.get('cond_var', 0) or 0)
    nd = f[f'data/{sid}/nodal_data'][:, -1, :]
    edge = f[f'data/{sid}/mesh_edge'][:]
    part_ids = nd[-1].astype(np.int32) if cfg.get('use_node_types') and nd.shape[0] > 7 else None
    cond = nd[3 + input_var:3 + input_var + cond_var].T.astype(np.float32) if cond_var else None
    return nd, edge, part_ids, cond


def reference(cfg, norm, model, prior, dev, n_ref, cache, stamp):
    """Input descriptors of n_ref evenly spaced TRAINING geometries:
    (group values, raw [R, .], embedding [R, .])."""
    if cache and os.path.exists(cache):
        try:
            z = np.load(cache, allow_pickle=False)
            if str(z['stamp']) == stamp:
                return [str(v) for v in z['values']], z['raw'], z['emb']
        except (OSError, ValueError, KeyError):
            pass
    path = cfg['dataset_dir']
    attr = cfg.get('eval_group_attr') or cfg['split_group_attr']
    ga = _eval_dataset_groups(path, attr)
    if not ga:
        return [], None, None
    train = grouped_split_ids(path, sorted(ga, key=int), cfg.get('split_group_attr') or attr, 0.8, 0.1,
                              int(cfg.get('split_seed', 42)))[0]
    values = _group_order({str(ga[s]) for s in train})
    if len(values) > n_ref:
        values = [values[i] for i in np.linspace(0, len(values) - 1, n_ref).round().astype(int)]
    raw, emb = [], []
    t0 = time.time()
    with h5py.File(path, 'r') as f:
        for v in values:
            sid = min((s for s, g in ga.items() if str(g) == v), key=int)
            nd, edge, part_ids, cond = scene(f, sid, cfg)
            ctx = _SampleContext(cfg, norm, nd[0:3].T, np.concatenate([edge, edge[[1, 0]]], axis=1),
                                 part_ids, dev, cond_feat=cond)
            r, e = Geo(model, prior, ctx, 0, 'range', 1, 1.0, 1.0).descriptors()
            raw.append(r)
            emb.append(e)
            del ctx
    print(f'  reference: {len(values)} training geometries, {time.time() - t0:.0f} s', flush=True)
    raw, emb = np.asarray(raw), np.asarray(emb)
    if cache:
        try:
            np.savez(cache, values=np.asarray(values), raw=raw, emb=emb, stamp=np.asarray(stamp))
        except OSError:
            pass
    return values, raw, emb


def ema_record(ck, raw_loaded):
    """How far the checkpoint's EMA weights are from its trained ones: relative
    L2 distance over the prior's tensors and over all the others, and whether
    the prior's standardization buffers agree."""
    ema = ck.get('ema_state_dict')
    if ema is None:
        return {'has': False, 'raw': False}
    num, den = {'prior': 0.0, 'rest': 0.0}, {'prior': 0.0, 'rest': 0.0}
    buffers = None
    for key, a in ck['model_state_dict'].items():
        b = ema.get('module.' + key)
        if (b is None or not torch.is_tensor(a) or not torch.is_floating_point(a) or a.shape != b.shape
                or key.startswith('aux_decoder.')):
            continue
        a, b = a.double(), b.double().to(a.device)
        if key in STD_BUFFERS:
            buffers = (buffers is not False) and bool(torch.allclose(a, b, rtol=1e-5, atol=1e-8))
            continue
        part = 'prior' if key.startswith('prior.') else 'rest'
        num[part] += float(((a - b) ** 2).sum())
        den[part] += float((a ** 2).sum())
    return {'has': True, 'raw': bool(raw_loaded), 'buffers_equal': buffers,
            'prior_diff': math.sqrt(num['prior'] / den['prior']) if den['prior'] > 0 else None,
            'rest_diff': math.sqrt(num['rest'] / den['rest']) if den['rest'] > 0 else None}


def training_record(config_path, train_config, ck):
    """What the training run says about the kept checkpoint: the epoch counts and
    `best_by` from the training config, validation CRPS and the training prior
    loss per epoch from its log. None without a training config."""
    path = train_config
    if not path:
        name = os.path.basename(config_path)
        if not name.startswith('config_infer_'):
            return None
        path = os.path.join(os.path.dirname(config_path), 'config_train_' + name[len('config_infer_'):])
    if not os.path.isfile(path):
        print(f'  no training config ({path}): the checkpoint-selection causes stay open', flush=True)
        return None
    try:
        tc = load_config(path)
    except Exception as err:   # an unreadable config only leaves two causes open
        print(f'  {path}: {str(err)[:100]}', flush=True)
        return None

    def number(key):
        try:
            return int(float(tc.get(key)))
        except (TypeError, ValueError):
            return None
    try:   # > 0: the prior loss also trains the encoder (with no freeze, a joint run)
        grad_enc = float(tc.get('prior_grad_to_encoder', 0.0) or 0.0)
    except (TypeError, ValueError):
        grad_enc = None
    rec = {'config': path, 'kept': ck.get('epoch'), 'freeze': number('prior_freeze_epoch') or 0,
           'grad_enc': grad_enc,
           'total': number('training_epochs'), 'best_by': str(tc.get('best_by', 'recon')).lower(),
           'val_interval': number('val_interval'), 'log': None, 'val': [], 'loss': []}
    log = tc.get('log_file_dir')
    hist = PV.read_history(os.path.abspath(str(log))) if log else None
    if hist:
        rec.update(hist, log=os.path.abspath(str(log)))
    else:
        print(f'  no epoch log ({log}): the checkpoint-selection causes stay open', flush=True)
    return rec


def track(groups, order, part):
    """R^2 of the prior cloud's descriptor against the posterior cloud's, across
    the geometries: 1 = follows it, 0 = no better than one constant cloud."""
    P = np.asarray([groups[v]['cloud']['P'][part] for v in order], np.float64)
    Q = np.asarray([groups[v]['cloud']['Q'][part] for v in order], np.float64)
    if P.size == 0 or len(order) < 3:
        return None
    den = float(((P - P.mean(0)) ** 2).sum())
    return float(1.0 - ((Q - P) ** 2).sum() / den) if den > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True, help='the INFERENCE config the sweep scored')
    ap.add_argument('--modelpath', default=None, help="another checkpoint than the config's")
    ap.add_argument('--gpu', type=int, default=None, help="card; default the config's gpu_ids")
    ap.add_argument('--batch', type=int, default=8, help='graphs per forward (memory only)')
    ap.add_argument('--splits', type=int, default=SPLITS)
    ap.add_argument('--out', default=None, help='default: <inference_output_dir>/../diag')
    ap.add_argument('--tag', default=None, help='default: ho / tr / ts by --split')
    ap.add_argument('--split', default='heldout', choices=('heldout', 'train', 'test'))
    ap.add_argument('--dataset', default=None, help='another HDF5 than the split reads')
    ap.add_argument('--n-groups', type=int, default=None, help='N geometries, evenly spaced')
    ap.add_argument('--train-ref', type=int, default=32,
                    help='training geometries the input is compared with (0 = skip)')
    ap.add_argument('--train-config', default=None,
                    help='the training config; default config_train_<name>.txt beside --config')
    ap.add_argument('--no-raw', action='store_true',
                    help='skip the trained (non-EMA) weights: no EMA test, less memory')
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
    latent_type = str(cfg.get('latent_type', 'global')).lower()
    if cfg.get('use_node_types') and norm.get('num_node_types'):
        cfg['num_node_types'] = norm['num_node_types']
    model = _load_model_from_checkpoint(cfg, ck, dev)
    prior = _load_conditional_prior(cfg, ck, model, dev)
    if not isinstance(prior, CoarseFMPrior):
        raise SystemExit(f'prior_diagnosis reads the coarse FM prior of latent_type local / hybrid; '
                         f'this checkpoint is latent_type {latent_type}')
    model.eval()
    prior.eval()
    trained = None   # the same checkpoint's trained weights, when inference loaded the EMA ones
    if 'ema_state_dict' in ck and not a.no_raw:
        try:
            ck_trained = {key: v for key, v in ck.items() if key != 'ema_state_dict'}
            trained_model = _load_model_from_checkpoint(cfg, ck_trained, dev)
            trained_prior = _load_conditional_prior(cfg, ck_trained, trained_model, dev)
            trained_model.eval()
            trained = (trained_model, trained_prior)
        except (RuntimeError, KeyError, AttributeError) as err:
            print(f'  the trained (non-EMA) weights do not load ({str(err)[:100]}): no EMA test', flush=True)
    ema = ema_record(ck, trained is not None)
    training = training_record(_abs(a.config), _abs(a.train_config) if a.train_config else None, ck)
    temperature = float(cfg.get('prior_temperature', 1.0))
    infl = cfg.get('latent_inflation', 1.0)
    lam = float(infl[0] if isinstance(infl, (list, tuple)) else infl)
    if isinstance(cfg.get('hierarchy_seed'), (list, tuple)):
        cfg['hierarchy_seed'] = cfg['hierarchy_seed'][0]

    # geometries are grouped by eval_group_attr; the train/val/test split is made by
    # split_group_attr (the two differ when one split group holds several geometries)
    attr = cfg.get('eval_group_attr') or cfg.get('split_group_attr')
    split_attr = cfg.get('split_group_attr') or attr
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
        part = grouped_split_ids(h5_path, sorted(groups_attr, key=int), split_attr, 0.8, 0.1,
                                 int(cfg.get('split_seed', 42)))[('train', 'val', 'test').index(a.split)]
        keep = {str(groups_attr[s]) for s in part}
        order = [v for v in order if v in keep]
    if a.n_groups and len(order) > a.n_groups:
        order = [order[i] for i in np.linspace(0, len(order) - 1, a.n_groups).round().astype(int)]
    order = order[:a.max_groups]
    input_var, out_var = int(cfg['input_var']), int(cfg['output_var'])
    cond_var = int(cfg.get('cond_var', 0) or 0)
    rows = 3 + np.asarray(ch)

    tag = a.tag or PART_TAG[a.split]
    out_dir = _abs(a.out) if a.out else os.path.join(
        os.path.dirname(os.path.normpath(cfg.get('inference_output_dir', '.'))), 'diag')
    os.makedirs(out_dir, exist_ok=True)

    ref_values, ref_raw, ref_emb = [], None, None
    if a.train_ref > 0:
        stamp = f"{os.path.abspath(cfg['modelpath'])}|{os.path.getmtime(cfg['modelpath']):.0f}|{a.train_ref}"
        ref_values, ref_raw, ref_emb = reference(cfg, norm, model, prior, dev, a.train_ref,
                                                 os.path.join(out_dir, 'prior_diagnosis_ref.npz'), stamp)
    same_file = os.path.abspath(h5_path) == os.path.abspath(cfg['dataset_dir'])

    spreads, groups = {}, {}
    with h5py.File(h5_path, 'r') as f:
        for gi, v in enumerate(order):
            t0 = time.time()
            ids = sorted((s for s, g in groups_attr.items() if g == v), key=int)[:a.max_per_group]
            nd0, edge0, part_ids, cond = scene(f, ids[0], cfg)
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
            ctx = _SampleContext(cfg, norm, nd0[0:3].T, np.concatenate([edge0, edge0[[1, 0]]], axis=1),
                                 part_ids, dev, cond_feat=cond)
            geo = Geo(model, prior, ctx, ch, stat, a.batch, temperature, lam, raw=trained)
            try:
                spreads[v], groups[v] = run_group(geo, truths, fields, 1000 + gi, check=gi == 0)
            except torch.cuda.OutOfMemoryError:
                gib = torch.cuda.get_device_properties(dev).total_memory / 2 ** 30
                raise SystemExit(f'CUDA OOM {attr}={v} {ctx.num_nodes} nodes, in {geo.stage} at one '
                                 f'graph (card {gib:.0f} GiB, peak '
                                 f'{torch.cuda.max_memory_allocated(dev) / 2 ** 30:.0f} GiB)')
            g = groups[v]
            if gi == 0 and g['integrator_gap'] > 1e-3:
                raise SystemExit(f"integrator mismatch {g['integrator_gap']:.2e}: this script's flow is "
                                 f"not the prior's sampler")
            if ref_raw is not None:
                raw, emb = geo.descriptors()
                skip = ref_values.index(v) if same_file and v in ref_values else None
                g['far_raw'], g['far_emb'] = far_from(ref_raw, raw, skip), far_from(ref_emb, emb, skip)
                g['raw_out'], g['raw_out_dim'] = outside(ref_raw, raw)
            print(f'  {attr}={v}: {len(ids)} FE results, {ctx.num_nodes} nodes, {geo.n_c} coarse nodes, '
                  f'{time.time() - t0:.0f} s', flush=True)
            del geo, ctx
            torch.cuda.empty_cache()

    scores, med, w2parts, k = split_scores(spreads, order, a.splits)
    doc = {'tag': tag, 'config': a.config, 'modelpath': cfg['modelpath'], 'epoch': ck.get('epoch'),
           'split': a.split, 'dataset': h5_path, 'attr': attr, 'k': k, 'splits': a.splits,
           'latent_type': latent_type, 'steps': int(prior.num_steps), 'fine': FINE,
           'solver': prior.solver, 'time_power': float(prior.time_power),
           'prior_temperature': temperature, 'latent_inflation': lam,
           'ema': ema, 'training': training,
           'scores': scores, 'per_group': med,
           'parts': {v: w2parts[v].get('Q') for v in order},
           'track': {'g': track(groups, order, 'g'), 'l': track(groups, order, 'l')},
           'groups': groups}
    summary = PV.summarize(doc)
    doc['summary'] = summary
    doc['spreads'] = {v: {e: s.tolist() for e, s in spreads[v].items()} for v in order}
    out = os.path.join(out_dir, f'prior_diagnosis_{tag}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)

    def w(x):
        x = PV.whole(x)
        return '-' if x is None else str(x)

    print(f'\n  wrote {out}\n')
    print(f"  per geometry (x100)   check: integrator {groups[order[0]]['integrator_gap']:.1e}, "
          f"oracle at t=0 {groups[order[0]]['oracle_gap_t0']:.1e}, posterior sample finds itself "
          f"{w(np.mean([groups[v]['self_hit'] for v in order]))}%")
    shown = ('floor', 'P', 'O', 'Q', 'S', 'R5', 'Qm', 'Qg', 'Ql', 'Qr')
    print(f"  {attr:>10} {'n':>4} " + ' '.join(f'{e:>5}' for e in shown)
          + f" {'far':>5} {'off':>6} {'cov':>4} {'bias':>5} {'sdr':>5}")
    for v in order:
        g, m, pt = groups[v], med[v], w2parts[v].get('Q') or {}
        print(f"  {v:>10} {g['n']:>4} " + ' '.join(f'{w(m.get(e)):>5}' for e in shown)
              + f" {w(g.get('far_emb')):>5} {g['off']['Q']:>6.1f} {w(g['cov']['Q']):>4} "
                f"{w(pt.get('bias')):>5} {w(pt.get('sd_ratio')):>5}")
    print('  when the draw commits (model until t, exact after; x100): '
          + ' '.join(f"{name[1:]}%={w((scores.get(name) or {}).get('p50'))}" for name in COMMIT) + '\n')

    parts = {tag if tag in PV.PARTS else PART_TAG[a.split]: summary}
    for line in PV.verdict(parts, os.path.splitext(os.path.basename(a.config))[0].replace('config_infer_', '')):
        print(line)
    n_fe = sorted({spreads[v]['truth'].size for v in order})
    print('\n' + PV.BLOCK_LEGEND)
    print('==== TYPE THIS BACK ====')
    print(f"PRIOR {tag} ep={ck.get('epoch')} g={len(order)} n={n_fe[0] if len(n_fe) == 1 else n_fe[-1]} k={k}")
    for line in PV.block(parts):
        print(line)
    print('==== END ====')
    return 0


if __name__ == '__main__':
    sys.exit(main())
