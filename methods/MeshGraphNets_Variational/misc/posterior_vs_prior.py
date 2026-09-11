"""Where does MeshGraphNets-V lose its variability -- in the prior, or in the decoder?

    cd methods/MeshGraphNets_Variational
    python misc/posterior_vs_prior.py \
        --config ../../configs/MeshGraphNets_Variational/SAOI_sweep3/config_infer_pv_top_a100_s26fe_main.txt \
        [--n-prior 500] [--chunk 16] [--out ../../output/meshgraphnets-v/saoi_sweep3/diag]

THE QUESTION
    Posterior reconstruction is excellent while the generated distribution is
    ~2x too narrow (sd_ratio 0.45-0.53 on every sweep arm). Training decodes
    z ~ q(z | y, g); deployment decodes z ~ p(z | g). If the encoder spreads the
    125 realizations of one part over some region of latent space and the prior
    concentrates on a smaller or differently shaped region for that SAME part,
    the decoder can be perfect and the output still narrow.

    The existing [PriorDiag] cannot see this. It compares Var(mu_q) + E[var_q]
    against the prior's variance over the WHOLE validation set, so Var(mu_q) is
    dominated by between-geometry spread. What makes sd_ratio is the
    within-geometry spread -- realizations of one part -- and an aggregate
    ratio near 1 is compatible with a badly mismatched conditional.

THREE ENSEMBLES, ONE GEOMETRY  (the eval design: each `_compare_` file holds
125 realizations of the one part in the matching `_infer_` file)

    truth            the 125 realizations                        -> S_true
    posterior-mean   encode each realization, decode mu           -> S_post_mu
    posterior-sample encode each realization, decode a z ~ q      -> S_post_z
    prior            draw z ~ p(z | g) for the part, decode        -> S_prior

    S_post ~ S_true  and  S_prior << S_post   ->  the PRIOR is the bottleneck:
                                                 retrain / recalibrate it with
                                                 the encoder and decoder frozen
    S_post << S_true                          ->  encode/decode loses it: the
                                                 prior cannot put it back;
                                                 decoder-side work (pv loss,
                                                 capacity, spatial latents)
    S_post ~ S_prior << S_true                ->  lost at encoding

LATENT SPACE
    The same comparison directly on z. Per slot and per dimension, the prior's
    standard deviation against the posterior means' standard deviation for this
    part; and the fraction of the prior's variance that lies in the top
    principal directions of the posterior cloud. If the posterior's variation
    for one part lives in a few directions and the prior spreads its variance
    elsewhere, isotropic inflation cannot fix it -- and that is exactly what the
    inflation sweep measured: 2.5x on z gave 1.35x on the field.

Everything is read from the checkpoint and the two HDF5 files named in the
inference config. Nothing is trained. Both the printed table and the JSON dump
carry every number, so the verdict line is a reading of them, not a claim.
"""
import argparse
import json
import os
import sys

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import h5py
import numpy as np
import torch
from torch_geometric.data import Batch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from general_modules.load_config import load_config            # noqa: E402
from general_modules.data_loader import load_data              # noqa: E402
from general_modules.mesh_dataset import MeshGraphDataset      # noqa: E402
from inference_profiles.rollout import (                       # noqa: E402
    _load_model_from_checkpoint, _load_conditional_prior,
    _spread_max_minus_min, Z_DISP_CHANNEL,
)

Z_OUT = Z_DISP_CHANNEL - 3      # z_disp is the 3rd output channel


# ─────────────────────────────────────────────────────────────────────────────
# pure-numpy analysis (unit-tested separately; no torch, no files)
# ─────────────────────────────────────────────────────────────────────────────

def spread_stats(spreads):
    s = np.asarray(spreads, dtype=np.float64)
    return {'n': int(s.size), 'mean': float(s.mean()), 'sd': float(s.std(ddof=0))}


def ensemble_table(truth, post_mu, post_z, prior):
    """sd_ratio of each ensemble against the truth, plus location shift in truth SDs."""
    t = spread_stats(truth)
    out = {'truth': t}
    for name, arr in (('posterior_mean', post_mu), ('posterior_sample', post_z),
                      ('prior', prior)):
        st = spread_stats(arr)
        st['sd_ratio'] = st['sd'] / t['sd'] if t['sd'] > 0 else float('nan')
        st['dmean_over_sd'] = (st['mean'] - t['mean']) / t['sd'] if t['sd'] > 0 else float('nan')
        out[name] = st
    return out


def latent_table(post_mu, prior_z, top_k=4):
    """Compare the posterior means of one part against the prior's draws for it.

    post_mu : [R, S, D]  posterior means, one per realization
    prior_z : [N, S, D]  prior draws for the same geometry
    """
    post_mu = np.asarray(post_mu, dtype=np.float64)
    prior_z = np.asarray(prior_z, dtype=np.float64)
    R, S, D = post_mu.shape
    P = post_mu.reshape(R, S * D)
    Q = prior_z.reshape(prior_z.shape[0], S * D)

    sd_post = P.std(axis=0, ddof=0)
    sd_prior = Q.std(axis=0, ddof=0)
    shift = (Q.mean(axis=0) - P.mean(axis=0)) / np.where(sd_post > 0, sd_post, np.nan)

    per_slot = []
    for s in range(S):
        sl = slice(s * D, (s + 1) * D)
        per_slot.append({
            'slot': s,
            'sd_post_rms': float(np.sqrt(np.mean(sd_post[sl] ** 2))),
            'sd_prior_rms': float(np.sqrt(np.mean(sd_prior[sl] ** 2))),
            'shift_rms_in_post_sd': float(np.sqrt(np.nanmean(shift[sl] ** 2))),
        })
        per_slot[-1]['ratio'] = (per_slot[-1]['sd_prior_rms'] / per_slot[-1]['sd_post_rms']
                                 if per_slot[-1]['sd_post_rms'] > 0 else float('nan'))

    # PCA of the posterior cloud. If the realizations of one part vary along a
    # few directions, the prior's variance should sit there too.
    Pc = P - P.mean(axis=0)
    k = int(min(top_k, R - 1, S * D))
    pca = {'top_k': k}
    if k >= 1:
        _, sv, vt = np.linalg.svd(Pc, full_matrices=False)
        var_post = sv ** 2 / max(R, 1)
        total_post = float(np.sum(Pc ** 2) / max(R, 1))
        basis = vt[:k]                                    # [k, S*D]
        Qc = Q - Q.mean(axis=0)
        proj_prior = Qc @ basis.T                         # [N, k]
        total_prior = float(np.sum(Qc ** 2) / max(Qc.shape[0], 1))
        var_prior_in_top = float(np.sum(proj_prior ** 2) / max(Qc.shape[0], 1))
        var_post_in_top = float(np.sum(var_post[:k]))
        pca.update({
            'post_var_frac_in_top_k': var_post_in_top / total_post if total_post > 0 else float('nan'),
            'prior_var_frac_in_top_k': var_prior_in_top / total_prior if total_prior > 0 else float('nan'),
            # along the posterior's own principal directions, how wide is the
            # prior relative to the posterior?
            'prior_over_post_sd_along_top_k': [
                float(np.sqrt((proj_prior[:, i] ** 2).mean() / var_post[i]))
                if var_post[i] > 0 else float('nan') for i in range(k)],
        })
    return {'per_slot': per_slot, 'pca': pca,
            'overall_sd_ratio_rms': float(np.sqrt(np.mean(sd_prior ** 2)) /
                                          np.sqrt(np.mean(sd_post ** 2)))
            if np.mean(sd_post ** 2) > 0 else float('nan')}


def verdict(ens, lat):
    """A reading of the numbers, stated with its thresholds."""
    st, sp = ens['truth']['sd'], ens['posterior_mean']['sd_ratio']
    sz, sr = ens['posterior_sample']['sd_ratio'], ens['prior']['sd_ratio']
    lines = []
    if not np.isfinite(sp) or not np.isfinite(sr):
        return ['truth spread has zero variance -- nothing to compare against']
    post = max(sp, sz)
    if post >= 0.8 and sr <= 0.7 * post:
        lines.append(
            f"PRIOR MISMATCH: the encoder+decoder reproduce {post:.2f} of the truth's "
            f"spread but the prior path reaches only {sr:.2f}. The decoder is not the "
            f"bottleneck -- the prior's conditional for this part is too narrow (or "
            f"mis-shaped, see PCA). Fix on the prior side: retrain it against a frozen "
            f"encoder/decoder, or recalibrate its covariance per geometry.")
    elif post < 0.8 and sr <= post + 0.1:
        lines.append(
            f"ENCODE/DECODE LOSS: even decoding the posterior gives only {post:.2f} of the "
            f"truth's spread; the prior ({sr:.2f}) merely inherits it. The prior cannot "
            f"put back what encoding removed -- decoder-side work (peak-to-valley loss, "
            f"capacity, spatial latents) is the lever.")
    elif sr > post + 0.1:
        lines.append(
            f"prior path ({sr:.2f}) is WIDER than the posterior path ({post:.2f}); the "
            f"prior adds spread the encoder never carried -- check bias and the PCA "
            f"table before reading this as a good thing.")
    else:
        lines.append(
            f"MIXED: posterior path {post:.2f}, prior path {sr:.2f} of the truth's spread. "
            f"Both stages lose some; read the latent table for where.")
    p = lat.get('pca', {})
    if 'prior_var_frac_in_top_k' in p and np.isfinite(p['prior_var_frac_in_top_k']):
        lines.append(
            f"latent: the posterior cloud puts {p['post_var_frac_in_top_k']:.0%} of its "
            f"variance in its top {p['top_k']} directions; the prior puts "
            f"{p['prior_var_frac_in_top_k']:.0%} of ITS variance there. Along those "
            f"directions the prior is "
            + ', '.join(f"{v:.2f}x" for v in p['prior_over_post_sd_along_top_k'])
            + " as wide as the posterior. Well below 1 with a low prior fraction = the "
              "prior's variance sits where the decoder does not look, which is why "
              "isotropic inflation could not widen the field.")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# the run
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg):
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


def per_graph_spreads(pred, batch_obj, m, dst, dm):
    """Physical z_disp peak-to-valley of each graph in a collated prediction."""
    z = (pred[:, Z_OUT].float() * dst[Z_OUT] + dm[Z_OUT]).detach().cpu().numpy()
    ptr = batch_obj.ptr.tolist()
    return [_spread_max_minus_min(z[ptr[i]:ptr[i + 1]]) for i in range(m)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True,
                    help='an INFERENCE config: modelpath + infer_dataset + eval_dataset')
    ap.add_argument('--n-prior', type=int, default=500)
    ap.add_argument('--chunk', type=int, default=16,
                    help='graphs decoded per forward; bounds memory only')
    ap.add_argument('--out', default=None, help='directory for the JSON dump')
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg['num_timesteps'] = 1
    dev = resolve_device(cfg)
    torch.manual_seed(1234)

    # ---- normalizers from the TRAINING split, exactly as the trainer fit them
    ds = load_data(cfg)
    tr, va, te = ds.split(0.8, 0.1, 0.1, seed=int(cfg['split_seed']))
    for s in (tr, va, te):
        s.augment_geometry = False
    dm = torch.tensor(tr.delta_mean, device=dev)
    dst = torch.tensor(tr.delta_std, device=dev)

    # ---- checkpoint, model, prior
    ck = torch.load(cfg['modelpath'], map_location=dev, weights_only=False)
    c2 = dict(cfg)
    for k, v in ck['model_config'].items():
        c2[k] = v
    c2['num_timesteps'] = 1
    model = _load_model_from_checkpoint(c2, ck, dev)
    prior = _load_conditional_prior(c2, ck, model, dev)
    if prior is None:
        raise SystemExit('this checkpoint has no conditional prior; the comparison needs one')

    # ---- the one geometry (infer file) with the training normalizers
    ev = MeshGraphDataset(cfg['infer_dataset'], c2)
    ev.inherit_preprocessing_from(tr)
    ev.augment_geometry = False
    if len(ev) != 1:
        print(f"  [note] infer_dataset holds {len(ev)} samples; using the first")
    g0 = ev[0]
    g0.y = None
    N = int(g0.x.shape[0])

    # ---- the 125 realizations of that geometry (compare file)
    with h5py.File(cfg['eval_dataset'], 'r') as f:
        sids = list(f['data'].keys())
        ys, truth = [], []
        ref_xyz = None
        with h5py.File(cfg['infer_dataset'], 'r') as fi:
            i0 = list(fi['data'].keys())[0]
            ref_xyz = fi[f'data/{i0}/nodal_data'][0:3, -1, :]
        for sid in sids:
            nd = f[f'data/{sid}/nodal_data']
            if nd.shape[2] != N:
                raise SystemExit(f"realization {sid} has {nd.shape[2]} nodes, geometry has {N}: "
                                 f"the compare file is not this part")
            xyz = nd[0:3, -1, :]
            if not np.allclose(xyz, ref_xyz, atol=1e-6):
                raise SystemExit(f"realization {sid}: node coordinates differ from the infer "
                                 f"geometry -- node ordering does not match, encoder input "
                                 f"would be scrambled")
            ys.append(nd[3:3 + int(c2['output_var']), -1, :].T.astype(np.float32))
            truth.append(_spread_max_minus_min(nd[Z_DISP_CHANNEL, -1, :]))
    R = len(ys)
    print(f"  geometry: {N} nodes;  realizations: {R};  prior draws: {a.n_prior}")

    dm_np, dst_np = tr.delta_mean, tr.delta_std

    # ---- posterior path: encode each realization, decode mean and a sample
    post_mu, post_logvar, sp_mu, sp_z = [], [], [], []
    for start in range(0, R, a.chunk):
        idx = list(range(start, min(start + a.chunk, R)))
        graphs = []
        for i in idx:
            gi = g0.clone()
            gi.y = torch.from_numpy((ys[i] - dm_np) / dst_np)
            graphs.append(gi)
        b = Batch.from_data_list(graphs).to(dev)
        m = len(idx)
        with torch.no_grad():
            pred_z, _, vl, _, _ = model(b, add_noise=False, use_posterior=True)
            mu, lv = vl['mu'], vl['logvar']
            pred_mu, *_ = model(b, add_noise=False, use_posterior=False, fixed_z=mu)
        post_mu.append(mu.float().cpu().numpy())
        post_logvar.append(lv.float().cpu().numpy())
        sp_z += per_graph_spreads(pred_z, b, m, dst, dm)
        sp_mu += per_graph_spreads(pred_mu, b, m, dst, dm)
    post_mu = np.concatenate(post_mu)          # [R, S, D]
    post_logvar = np.concatenate(post_logvar)

    # ---- prior path: draw for the geometry once, decode in chunks
    g1 = Batch.from_data_list([g0.clone()]).to(dev)
    with torch.no_grad():
        zp = prior.sample_n(g1, a.n_prior, temperature=1.0)[0]   # [n, S, D]
    sp_prior = []
    for start in range(0, a.n_prior, a.chunk):
        stop = min(start + a.chunk, a.n_prior)
        m = stop - start
        b = Batch.from_data_list([g0.clone() for _ in range(m)]).to(dev)
        with torch.no_grad():
            pred, *_ = model(b, add_noise=False, use_posterior=False,
                             fixed_z=zp[start:stop])
        sp_prior += per_graph_spreads(pred, b, m, dst, dm)
    prior_np = zp.float().cpu().numpy()

    # ---- tables
    ens = ensemble_table(truth, sp_mu, sp_z, sp_prior)
    lat = latent_table(post_mu, prior_np)
    lat['posterior_sigma_rms'] = float(np.sqrt(np.mean(np.exp(post_logvar))))
    lines = verdict(ens, lat)

    tag = os.path.splitext(os.path.basename(a.config))[0]
    print()
    print(f"POSTERIOR vs PRIOR -- {tag}")
    print(f"{'ensemble':<18}{'n':>6}{'mean spread':>14}{'sd':>10}{'sd_ratio (1)':>14}{'dmean/sd (0)':>14}")
    print('-' * 76)
    for name in ('truth', 'posterior_mean', 'posterior_sample', 'prior'):
        e = ens[name]
        print(f"{name:<18}{e['n']:>6}{e['mean']:>14.4g}{e['sd']:>10.4g}"
              f"{e.get('sd_ratio', 1.0):>14.3f}{e.get('dmean_over_sd', 0.0):>14.3f}")
    print()
    print(f"{'latent slot':<12}{'sd post':>10}{'sd prior':>10}{'ratio':>8}{'shift/sd':>10}")
    print('-' * 50)
    for r in lat['per_slot']:
        print(f"{r['slot']:<12}{r['sd_post_rms']:>10.4f}{r['sd_prior_rms']:>10.4f}"
              f"{r['ratio']:>8.2f}{r['shift_rms_in_post_sd']:>10.2f}")
    print(f"{'all':<12}{'':>10}{'':>10}{lat['overall_sd_ratio_rms']:>8.2f}")
    print(f"posterior sigma (rms of exp(logvar/2)): {lat['posterior_sigma_rms']:.4f}")
    p = lat['pca']
    if 'prior_var_frac_in_top_k' in p:
        print(f"PCA top-{p['top_k']} of the posterior cloud: posterior var frac "
              f"{p['post_var_frac_in_top_k']:.2f}, prior var frac {p['prior_var_frac_in_top_k']:.2f}; "
              f"prior/post sd along them: "
              + ', '.join(f"{v:.2f}" for v in p['prior_over_post_sd_along_top_k']))
    print()
    for ln in lines:
        print('  ' + ln)

    out_dir = a.out or os.path.join(os.path.dirname(cfg.get('inference_output_dir', '.')), 'diag')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'posterior_vs_prior_{tag}.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump({'config': a.config, 'geometry_nodes': N, 'realizations': R,
                   'n_prior': a.n_prior, 'ensembles': ens, 'latent': lat,
                   'verdict': lines,
                   'spreads': {'truth': list(map(float, truth)),
                               'posterior_mean': list(map(float, sp_mu)),
                               'posterior_sample': list(map(float, sp_z)),
                               'prior': list(map(float, sp_prior))}},
                  fh, indent=2)
    print(f"\n  wrote {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
