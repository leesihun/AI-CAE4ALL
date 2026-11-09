"""Posterior vs. prior, per held-out geometry: where does the W2 gap come from?

    cd methods/MeshGraphNets_Variational
    python misc/posterior_vs_prior_groups.py \
        --config ../../configs/MeshGraphNets_Variational/hyperparameter_sweep/r6/config_infer_hybrid_e240.txt \
        [--gpu 0] [--batch 8] [--splits 200] [--out DIR] [--tag NAME]

`posterior_vs_prior.py` asks this question for one geometry (the SAOI files).
This one asks it for a grouped held-out file like ex3's: G unseen geometries
(`split_group_attr`) x n FE results each, every latent placement (global,
local, hybrid). Nothing is trained; everything comes from the checkpoint and
the inference config the sweep already scored.

FOUR ENSEMBLES PER GEOMETRY (one value per FE result of that geometry)

    truth             the FE result itself
    posterior mean    encode the FE result, decode mu                 (no sampling)
    posterior sample  encode the FE result, decode a z ~ q(z | y, g)
    prior             decode a z ~ p(z | g), the deployed sampler
    std_normal        decode a z ~ N(0, I), the same shape as the prior's draw: the
                      learned conditional prior ablated without retraining (the
                      MMD/KL regularizer pulls the aggregate posterior toward it)

SCORE  (the same numbers as the sweeps' plots.py noise block)

Each geometry's n FE results are cut at k = n // 2 by plots.py's own 200
permutations (SPLIT_SEED 0). The second half is the reference. The first half,
and the three ensembles built FROM the first half's FE results, are scored
against it: W2 over the reference's sd, root mean square over the geometries,
median over the splits, x100. So

    floor             = plots.py's floor (27 on ex3): two FE halves, sampling noise only
    posterior sample  = what the decoder makes of encoded FE results it never
                        saw in the reference -- a perfect encode/decode scores
                        the floor
    prior             = the deployed path at the same sample size

and the gap splits as  floor -> posterior sample (encode/decode)  and
posterior sample -> prior (the prior). The posterior ensembles are encoded from
held-out FE results, so they exist only here: deployment has no FE result.

LATENT  (per geometry, per latent part: g = per-graph slots, l = coarse nodes)

The posterior means of the n FE results form a cloud; the prior's n draws for
the same geometry form another. Their sd ratio, and how wide the prior is along
the posterior cloud's top principal directions (posterior_vs_prior.latent_table).
Beside it, the posterior's own sigma against the sd of its means: sigma >> sd
means the encoder barely tells the FE results of one geometry apart.

Per geometry the W2 also splits into its parts (bias, sd ratio, shape), and
the latent line adds where the prior's cloud sits: the distance between the
prior's and the posterior's centroids over the posterior cloud's radius
(> 1: the prior is centred outside it), and for the per-graph part how far
the geometries' posterior centroids lie apart over their own radius (>> 1:
the latent carries the geometry, and the prior has to predict where).

WHICH GEOMETRIES  (--split)

    heldout   the config's infer_dataset (default): the unseen geometries scored
    train     the training file's geometries the model trained on
    val       ... the validation geometries (checkpoint selection saw them)
    test      ... the test geometries (never seen)

train/val/test are setup.py's split, dataset.split(0.8, 0.1, 0.1, split_seed)
grouped by split_group_attr; --n-groups N takes N of them evenly spaced. The
same geometries scored on train and test separate a prior that has not
learned what it was shown (train bad) from one that does not carry over to
new geometries (train good, test bad). Fewer FE results per geometry (64 on
ex3's training file) give a higher floor; compare gaps within one split.

Writes posterior_vs_prior_groups_<tag>.json, one histogram figure, and ends
with a ==== TYPE THIS BACK ==== block.
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
from model.latent import latent_parts, lmap                        # noqa: E402
from posterior_vs_prior import latent_table, resolve_device       # noqa: E402

os.chdir(REPO_ROOT)   # config paths are relative to the method directory

ENSEMBLES = ('floor', 'posterior_mean', 'posterior_sample', 'prior', 'std_normal')
SPLITS = 200          # plots.py SPLITS
SPLIT_SEED = 0        # plots.py SPLIT_SEED: the same permutations, so the same floor


# ─────────────────────────────────────────────────────────────────────────────
# scoring (plots.py's w2_parts / geometry_row / summarize, copied: the sweep
# directory is not importable from here)
# ─────────────────────────────────────────────────────────────────────────────

def w2_parts(gt, gen):
    """Exact 1-D W2^2 and its parts: (w2^2, mean gap, sd_gen, sd_gt, shape)."""
    x = np.sort(np.asarray(gen, dtype=np.float64))
    y = np.sort(np.asarray(gt, dtype=np.float64))
    n, m = x.size, y.size
    cuts = np.union1d(np.arange(n + 1) / n, np.arange(m + 1) / m)
    w = np.diff(cuts)
    mid = 0.5 * (cuts[:-1] + cuts[1:])
    qx = x[np.minimum((mid * n).astype(int), n - 1)]
    qy = y[np.minimum((mid * m).astype(int), m - 1)]
    mx, my = float((w * qx).sum()), float((w * qy).sum())
    sx = math.sqrt(float((w * (qx - mx) ** 2).sum()))
    sy = math.sqrt(float((w * (qy - my) ** 2).sum()))
    cov = float((w * (qx - mx) * (qy - my)).sum())
    return float((w * (qx - qy) ** 2).sum()), mx - my, sx, sy, max(2.0 * (sx * sy - cov), 0.0)


def geometry_row(g, p, pooled_sd):
    w2sq, gap, sx, sy, shape = w2_parts(g, p)
    if g.size < 2 or sy <= 1e-9 * pooled_sd:
        return None
    return {'w2': math.sqrt(w2sq) / sy, 'bias': gap / sy, 'sd_ratio': sx / sy,
            'shape': math.sqrt(shape) / sy}


def summarize(rows):
    def rms(key, shift=0.0):
        return math.sqrt(sum((r[key] - shift) ** 2 for r in rows.values()) / len(rows))
    return {'w2': rms('w2'), 'bias': rms('bias'), 'sd_gap': rms('sd_ratio', 1.0),
            'shape': rms('shape'),
            'sd_ratio_geomean': math.exp(sum(math.log(max(r['sd_ratio'], 1e-12))
                                             for r in rows.values()) / len(rows))}


def split_scores(spreads, order, splits=SPLITS):
    """Every ensemble against the second half of each geometry's FE results.

    spreads[group] = {'truth', 'posterior_mean', 'posterior_sample', 'prior', ...},
    each [n] and aligned to the same FE results. Returns
    ({ensemble: {key: {p50, p5, p95}}}, {group: {ensemble: median w2}},
     {group: {ensemble: {bias, sd_ratio, shape}: median}}, k).
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
                src = s['truth'] if e == 'floor' else s[e]
                row = geometry_row(ref, src[ia], pooled_sd)
                if row is not None:
                    rows[e][v] = row
                    per_group[v][e].append(row)
        for e in ENSEMBLES:
            if rows[e]:
                runs[e].append(summarize(rows[e]))
    out = {}
    for e in ENSEMBLES:
        out[e] = {}
        for key in ('w2', 'bias', 'sd_gap', 'shape', 'sd_ratio_geomean'):
            vals = np.asarray([r[key] for r in runs[e]])
            out[e][key] = {'p50': float(np.median(vals)), 'p5': float(np.percentile(vals, 5)),
                           'p95': float(np.percentile(vals, 95))}
    def median(rows, key):
        return float(np.median([r[key] for r in rows])) if rows else float('nan')
    med = {v: {e: median(per_group[v][e], 'w2') for e in ENSEMBLES} for v in order}
    parts = {v: {e: {key: median(per_group[v][e], key) for key in ('bias', 'sd_ratio', 'shape')}
                 for e in ENSEMBLES} for v in order}
    return out, med, parts, k


def paired(est, truth):
    """Reconstruction of each FE result by its own encode/decode."""
    est, truth = np.asarray(est, np.float64), np.asarray(truth, np.float64)
    sd = float(truth.std())
    r = float(np.corrcoef(est, truth)[0, 1]) if sd > 0 and est.std() > 0 else float('nan')
    return {'corr': r, 'rmse_over_sd': float(np.sqrt(np.mean((est - truth) ** 2))) / sd
            if sd > 0 else float('nan')}


def latent_summary(post_mu, prior_z, post_sigma):
    """latent_table for one part, plus the posterior's sigma against its means' sd.

    post_mu, post_sigma : [n, S, D]; prior_z : [n', S, D]. The per-slot rows are
    reduced to quantiles: for the l part there is one slot per coarse node.
    """
    lat = latent_table(post_mu, prior_z)
    ratios = np.asarray([r['ratio'] for r in lat.pop('per_slot')], dtype=np.float64)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size:
        lat['slot_ratio_q'] = {q: float(np.percentile(ratios, p))
                               for q, p in (('p5', 5), ('p50', 50), ('p95', 95))}
    sd_mu = np.asarray(post_mu, np.float64).reshape(len(post_mu), -1).std(axis=0)
    sig = np.asarray(post_sigma, np.float64).reshape(len(post_sigma), -1)
    lat['sd_post_means_rms'] = float(np.sqrt(np.mean(sd_mu ** 2)))
    lat['posterior_sigma_rms'] = float(np.sqrt(np.mean(sig ** 2)))
    lat['sigma_over_sd_means'] = (lat['posterior_sigma_rms'] / lat['sd_post_means_rms']
                                  if lat['sd_post_means_rms'] > 0 else float('nan'))
    lat['dims'] = int(sd_mu.size)
    # Where the prior's cloud sits: centroid distance over the radius of the
    # posterior cloud the decoder trains on (spread of the means + sigma).
    mu2 = np.asarray(post_mu, np.float64).reshape(len(post_mu), -1)
    pz2 = np.asarray(prior_z, np.float64).reshape(len(prior_z), -1)
    within = float((sd_mu ** 2).sum() + (sig ** 2).mean(axis=0).sum())
    lat['centroid_offset'] = (float(np.linalg.norm(pz2.mean(axis=0) - mu2.mean(axis=0))) / math.sqrt(within)
                              if within > 0 else float('nan'))
    lat['_centroid'], lat['_within'] = mu2.mean(axis=0), within   # main() pops both
    return lat


def verdict(sc):
    """A reading of the medians, stated with its thresholds."""
    f, pm = sc['floor']['w2']['p50'], sc['posterior_mean']['w2']['p50']
    pz, pr = sc['posterior_sample']['w2']['p50'], sc['prior']['w2']['p50']
    lines = []
    total = pr - f
    if total <= 0:
        return [f"the prior path ({100 * pr:.0f}) is already at the floor ({100 * f:.0f}): "
                f"nothing to split"]
    share_post = (pz - f) / total
    if pr < pz - 0.02:
        lines.append(f"PRIOR BEATS POSTERIOR: prior {100 * pr:.0f} < posterior sample "
                     f"{100 * pz:.0f} (floor {100 * f:.0f}). Decoding encoded FE results is "
                     f"worse than the deployed path: encode/decode is the loss, the prior is not.")
    elif share_post < 1 / 3:
        lines.append(f"PRIOR: of the {100 * total:.0f} points between floor ({100 * f:.0f}) and "
                     f"prior ({100 * pr:.0f}), encode/decode costs {100 * (pz - f):.0f} and the "
                     f"prior {100 * (pr - pz):.0f}. The decoder can reproduce the FE spread from "
                     f"the right latents; the prior does not deliver them. Lever: the prior "
                     f"(longer prior-only training on a frozen simulator, its size, its FM loss), "
                     f"not the decoder or the latent size.")
    elif share_post > 2 / 3:
        lines.append(f"ENCODE/DECODE: of the {100 * total:.0f} points between floor "
                     f"({100 * f:.0f}) and prior ({100 * pr:.0f}), encode/decode already costs "
                     f"{100 * (pz - f):.0f}; the prior adds {100 * (pr - pz):.0f}. A better prior "
                     f"cannot recover it. Lever: the decoder / latent capacity / regularizer.")
    else:
        lines.append(f"BOTH: floor {100 * f:.0f} -> posterior sample {100 * pz:.0f} -> prior "
                     f"{100 * pr:.0f}; encode/decode and the prior each cost a comparable share "
                     f"(thresholds: posterior share < 1/3 = prior, > 2/3 = encode/decode).")
    lines.append(f"posterior mean {100 * pm:.0f} vs posterior sample {100 * pz:.0f}: the "
                 f"posterior's own noise {'widens' if pz < pm else 'does not help'} the decoded spread.")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# the run
# ─────────────────────────────────────────────────────────────────────────────

def _abs(path):
    return path if path is None or os.path.isabs(path) else os.path.normpath(os.path.join(ORIG_CWD, path))


def _np_parts(z):
    return {name: t.float().cpu().numpy() for name, t in latent_parts(z)}


def _per_graph(arr, name, m):
    """[m, S, D] from a part: g is [m, S, D] already; l is [m * n_c, C]."""
    if name == 'l':
        if arr.shape[0] % m:
            raise RuntimeError(f'l part has {arr.shape[0]} rows for {m} graphs')
        return arr.reshape(m, arr.shape[0] // m, -1)
    return arr.reshape(m, -1, arr.shape[-1]) if arr.ndim == 2 else arr


def run_group(model, prior, ctx, truths, fields, ch, stat, batch, temperature, lam):
    """Spreads of the three model ensembles for one geometry, plus its latents."""
    out_dim = fields[0].shape[1]
    dm = np.asarray(ctx.delta_mean, np.float32)
    dsd = np.asarray(ctx.delta_std, np.float32)
    base = ctx.build_step_graph(np.zeros((ctx.num_nodes, ctx.input_dim), dtype=np.float32))
    N = ctx.num_nodes
    sp = {'posterior_mean': [], 'posterior_sample': [], 'prior': [], 'std_normal': []}
    lat = {'mu': {}, 'sigma': {}, 'prior': {}}

    def make_batch(idx):
        graphs = []
        for i in idx:
            gi = base.clone()
            gi.y = torch.from_numpy((fields[i] - dm) / dsd).to(gi.x.device)
            graphs.append(gi)
        return Batch.from_data_list(graphs)

    def spreads(pred, m):
        phys = pred.float().cpu().numpy().reshape(m, N, out_dim) * dsd + dm
        return [_spread_stat(phys[j][:, ch], stat) for j in range(m)]

    def keep(store, z, m):
        for name, arr in _np_parts(z).items():
            store.setdefault(name, []).append(_per_graph(arr, name, m))

    for start in range(0, len(fields), batch):
        idx = list(range(start, min(start + batch, len(fields))))
        m = len(idx)
        with torch.no_grad():
            pred_z, _, vl, _, _ = model(make_batch(idx), add_noise=False, use_posterior=True)
            mu, logvar = vl['mu'], vl['logvar']
            pred_mu = model(make_batch(idx), add_noise=False, use_posterior=False, fixed_z=mu)[0]
            b = make_batch(idx)
            zp = prior.sample(b, temperature=temperature, inflation=lam)
            pred_p = model(b, add_noise=False, use_posterior=False, fixed_z=zp)[0]
            zn = lmap(torch.randn_like, zp)
            pred_n = model(make_batch(idx), add_noise=False, use_posterior=False, fixed_z=zn)[0]
        sp['posterior_sample'] += spreads(pred_z, m)
        sp['posterior_mean'] += spreads(pred_mu, m)
        sp['prior'] += spreads(pred_p, m)
        sp['std_normal'] += spreads(pred_n, m)
        keep(lat['mu'], mu, m)
        keep(lat['prior'], zp, m)
        sig = {n: np.exp(0.5 * a) for n, a in _np_parts(logvar).items()}
        for name, arr in sig.items():
            lat['sigma'].setdefault(name, []).append(_per_graph(arr, name, m))
    spreads_out = {'truth': np.asarray(truths, np.float64)}
    spreads_out.update({k: np.asarray(v, np.float64) for k, v in sp.items()})
    latents = {name: latent_summary(np.concatenate(lat['mu'][name]),
                                    np.concatenate(lat['prior'][name]),
                                    np.concatenate(lat['sigma'][name]))
               for name in lat['mu']}
    return spreads_out, latents


def write_figure(path, order, spreads, med, tag, label):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f'  [note] no figure ({exc})')
        return None
    cols = min(4, len(order))
    rows = int(math.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
    style = {'posterior_sample': ('#2878B5', '-', 'posterior sample'),
             'prior': ('#D65F32', '-', 'prior (deployed)'),
             'posterior_mean': ('#6E7781', '--', 'posterior mean'),
             'std_normal': ('#4C9A2A', ':', 'N(0, I)')}
    for ax, v in zip(axes.flat, order):
        s = spreads[v]
        allv = np.concatenate([s[e] for e in ('truth', 'posterior_mean', 'posterior_sample', 'prior',
                                              'std_normal')])
        allv = allv[np.isfinite(allv)]
        bins = np.linspace(allv.min(), allv.max() if allv.max() > allv.min() else allv.min() + 1, 31)
        ax.hist(s['truth'], bins=bins, density=True, color='#C9CED6', label='FE')
        for e, (c, ls, lab) in style.items():
            ax.hist(s[e], bins=bins, density=True, histtype='step', color=c, linestyle=ls,
                    linewidth=1.8, label=lab)
        m = med[v]
        ax.set_title(f"{v}: W2x100 floor {100 * m['floor']:.0f} | post {100 * m['posterior_sample']:.0f}"
                     f" | prior {100 * m['prior']:.0f} | N(0,I) {100 * m['std_normal']:.0f}", fontsize=9)
        ax.set_xlabel(label, fontsize=8)
    for ax in list(axes.flat)[len(order):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f'Posterior vs. prior per geometry: {tag}  (W2 = median over splits, half vs half)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return os.path.abspath(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True, help='the INFERENCE config the sweep scored')
    ap.add_argument('--modelpath', default=None, help="another checkpoint than the config's")
    ap.add_argument('--gpu', type=int, default=None, help="card; default the config's gpu_ids")
    ap.add_argument('--batch', type=int, default=8, help='graphs per forward (memory only)')
    ap.add_argument('--splits', type=int, default=SPLITS)
    ap.add_argument('--out', default=None, help='default: <inference_output_dir>/../diag')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--split', default='heldout', choices=('heldout', 'train', 'val', 'test'),
                    help="heldout = the config's infer_dataset; train/val/test = that part of "
                         "the training file (setup.py's grouped 0.8/0.1/0.1 split)")
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
    if cfg.get('use_node_types') and norm.get('num_node_types'):
        cfg['num_node_types'] = norm['num_node_types']
    model = _load_model_from_checkpoint(cfg, ck, dev)
    prior = _load_conditional_prior(cfg, ck, model, dev)
    if prior is None:
        raise SystemExit('this checkpoint has no conditional prior; the comparison needs one')
    temperature = float(cfg.get('prior_temperature', 1.0))
    infl = cfg.get('latent_inflation', 1.0)
    lam = float(infl[0] if isinstance(infl, (list, tuple)) else infl)
    if isinstance(cfg.get('hierarchy_seed'), (list, tuple)):
        print('  [note] hierarchy_seed lists several seeds; this uses the first for every geometry')
        cfg['hierarchy_seed'] = cfg['hierarchy_seed'][0]

    attr = cfg.get('split_group_attr')
    if a.dataset:
        h5_path = _abs(a.dataset)
    elif a.split == 'heldout':
        h5_path = cfg.get('eval_dataset') or cfg.get('infer_dataset')
    else:
        h5_path = cfg['dataset_dir']
    if not attr:
        raise SystemExit('the config has no split_group_attr; use misc/posterior_vs_prior.py')
    groups = _eval_dataset_groups(h5_path, attr)
    if not groups:
        raise SystemExit(f'{h5_path}: not every scene carries metadata attr {attr!r}')
    order = _group_order(groups.values())
    if a.split != 'heldout':
        # training_profiles/setup.py: dataset.split(0.8, 0.1, 0.1, seed=split_seed)
        part = grouped_split_ids(h5_path, sorted(groups, key=int), attr, 0.8, 0.1,
                                 int(cfg.get('split_seed', 42)))[('train', 'val', 'test').index(a.split)]
        keep = {str(groups[s]) for s in part}
        order = [v for v in order if v in keep]
    if a.n_groups and len(order) > a.n_groups:
        order = [order[i] for i in np.linspace(0, len(order) - 1, a.n_groups).round().astype(int)]
    order = order[:a.max_groups]
    input_var, out_var = int(cfg['input_var']), int(cfg['output_var'])
    cond_var = int(cfg.get('cond_var', 0) or 0)
    rows = 3 + np.asarray(ch)

    spreads, latents, nodes = {}, {}, {}
    with h5py.File(h5_path, 'r') as f:
        for v in order:
            ids = sorted((s for s, g in groups.items() if g == v), key=int)[:a.max_per_group]
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
                    raise SystemExit(f'{attr}={v}: scene {s} differs from scene {ids[0]} in '
                                     f'coordinates, conditions, node types or edges -- not one '
                                     f'geometry; this diagnostic needs one graph per group')
                fields.append(nd[3:3 + out_var].T.astype(np.float32))
                truths.append(_spread_stat(nd[rows].T if rows.ndim else nd[int(rows)], stat))
            part_ids = (nd0[-1].astype(np.int32)
                        if cfg.get('use_node_types') and nd0.shape[0] > 7 else None)
            cond = (nd0[3 + input_var:3 + input_var + cond_var].T.astype(np.float32)
                    if cond_var else None)
            ctx = _SampleContext(cfg, norm, nd0[0:3].T, np.concatenate([edge0, edge0[[1, 0]]], axis=1),
                                 part_ids, dev, cond_feat=cond)
            print(f'  {attr}={v}: {len(ids)} FE results, {ctx.num_nodes} nodes', flush=True)
            spreads[v], latents[v] = run_group(model, prior, ctx, truths, fields, ch, stat,
                                               a.batch, temperature, lam)
            nodes[v] = int(ctx.num_nodes)

    scores, med, w2parts, k = split_scores(spreads, order, a.splits)
    pair = {e: paired(np.concatenate([spreads[v][e] for v in order]) -
                      np.concatenate([np.full(spreads[v]['truth'].size, spreads[v]['truth'].mean())
                                      for v in order]),
                      np.concatenate([spreads[v]['truth'] - spreads[v]['truth'].mean() for v in order]))
            for e in ('posterior_mean', 'posterior_sample')}
    lines = verdict(scores)

    parts = sorted({p for v in order for p in latents[v]})
    lat_rms = {}
    for p in parts:
        vals = [latents[v][p] for v in order if p in latents[v]]
        def gm(key, sub=None):
            xs = [x['pca'][key] if sub == 'pca' else x[key] for x in vals]
            xs = [np.mean(x) if isinstance(x, list) else x for x in xs]
            xs = np.asarray([x for x in xs if x is not None and np.isfinite(x)], np.float64)
            return float(np.exp(np.mean(np.log(np.maximum(xs, 1e-12))))) if xs.size else float('nan')
        lat_rms[p] = {'dims': vals[0]['dims'],
                      'prior_over_post_sd': gm('overall_sd_ratio_rms'),
                      'prior_over_post_sd_top_k': gm('prior_over_post_sd_along_top_k', 'pca'),
                      'post_var_frac_top_k': gm('post_var_frac_in_top_k', 'pca'),
                      'prior_var_frac_top_k': gm('prior_var_frac_in_top_k', 'pca'),
                      'sigma_over_sd_means': gm('sigma_over_sd_means'),
                      'centroid_offset_mean': float(np.mean([x['centroid_offset'] for x in vals])),
                      'centroid_offset_max': float(np.max([x['centroid_offset'] for x in vals]))}
        cents = [latents[v][p].pop('_centroid') for v in order if p in latents[v]]
        within = [latents[v][p].pop('_within') for v in order if p in latents[v]]
        # Between-geometry spread of the posterior centroids over the clouds'
        # own radius. Per-graph parts only: an l slot is a coarse node, and
        # coarse nodes of two meshes do not correspond.
        if p != 'l' and len(cents) > 1 and len({c.size for c in cents}) == 1:
            c = np.stack(cents)
            between = float(np.sqrt(((c - c.mean(axis=0)) ** 2).sum(axis=1).mean()))
            lat_rms[p]['between_over_within'] = between / math.sqrt(float(np.mean(within)))

    tag = a.tag or os.path.splitext(os.path.basename(a.config))[0].replace('config_infer_', '')
    out_dir = _abs(a.out) if a.out else os.path.join(
        os.path.dirname(os.path.normpath(cfg.get('inference_output_dir', '.'))), 'diag')
    os.makedirs(out_dir, exist_ok=True)
    label = str(cfg.get('spread_label', 'spread'))
    fig = write_figure(os.path.join(out_dir, f'posterior_vs_prior_groups_{tag}.png'),
                       order, spreads, med, tag, f'{stat}({label})')
    out = os.path.join(out_dir, f'posterior_vs_prior_groups_{tag}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'config': a.config, 'modelpath': cfg['modelpath'], 'epoch': ck.get('epoch'),
                   'split': a.split, 'dataset': h5_path,
                   'latent_type': cfg.get('latent_type', 'global'), 'attr': attr, 'k': k,
                   'splits': a.splits, 'prior_temperature': temperature, 'latent_inflation': lam,
                   'scores': scores, 'per_group_w2_median': med, 'per_group_parts_median': w2parts,
                   'paired_within_geometry': pair,
                   'latent': latents, 'latent_geomean_over_groups': lat_rms, 'verdict': lines,
                   'figure': fig, 'nodes': nodes,
                   'spreads': {v: {e: s.tolist() for e, s in spreads[v].items()} for v in order}},
                  fh, indent=2)

    def band(e, key='w2'):
        s = scores[e][key]
        return f"{100 * s['p50']:.0f} [{100 * s['p5']:.0f}-{100 * s['p95']:.0f}]"

    n_fe = sorted({spreads[v]['truth'].size for v in order})
    print(f'\n  wrote {out}\n  figure {fig}\n')
    print('==== TYPE THIS BACK ====')
    print(f"posterior_vs_prior {tag} ep={ck.get('epoch')} {cfg.get('latent_type', 'global')} split={a.split} groups={len(order)} "
          f"n={n_fe[0] if len(n_fe) == 1 else n_fe} k={k}")
    print('w2x100 p50[p5-p95]: ' + ' | '.join(f'{e} {band(e)}' for e in ENSEMBLES))
    print('sd_ratio geomean: ' + ' | '.join(
        f"{e} {scores[e]['sd_ratio_geomean']['p50']:.2f}" for e in ENSEMBLES))
    print('bias rms x100: ' + ' | '.join(
        f"{e} {100 * scores[e]['bias']['p50']:.0f}" for e in ENSEMBLES))
    print('per geom w2x100 floor/mu/post/prior/N(0,I): ' + ' '.join(
        f"{v}:{100 * med[v]['floor']:.0f}/{100 * med[v]['posterior_mean']:.0f}/"
        f"{100 * med[v]['posterior_sample']:.0f}/{100 * med[v]['prior']:.0f}/"
        f"{100 * med[v]['std_normal']:.0f}" for v in order))
    print('paired (within geom) mu r={:.2f} rmse/sd={:.2f} | post r={:.2f} rmse/sd={:.2f}'.format(
        pair['posterior_mean']['corr'], pair['posterior_mean']['rmse_over_sd'],
        pair['posterior_sample']['corr'], pair['posterior_sample']['rmse_over_sd']))
    for p in parts:
        r = lat_rms[p]
        print(f"latent {p} ({r['dims']}d): prior/post sd {r['prior_over_post_sd']:.2f}, "
              f"top-k {r['prior_over_post_sd_top_k']:.2f}, var in top-k post "
              f"{r['post_var_frac_top_k']:.2f} prior {r['prior_var_frac_top_k']:.2f}, "
              f"sigma/sd(mu) {r['sigma_over_sd_means']:.2f}, centroid offset "
              f"{r['centroid_offset_mean']:.2f} (max {r['centroid_offset_max']:.2f})"
              + (f", between/within {r['between_over_within']:.2f}" if 'between_over_within' in r else ''))
    print('per geom prior bias/sd/shape x100: ' + ' '.join(
        f"{v}:{100 * w2parts[v]['prior']['bias']:+.0f}/{100 * w2parts[v]['prior']['sd_ratio']:.0f}/"
        f"{100 * w2parts[v]['prior']['shape']:.0f}" for v in order))
    for ln in lines:
        print(ln)
    print('==== END ====')
    return 0


if __name__ == '__main__':
    sys.exit(main())
