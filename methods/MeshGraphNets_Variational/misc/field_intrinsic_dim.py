"""How many degrees of freedom does the target field actually have?

Answers the one question that decides whether a global latent z is a bottleneck
or a well-matched inductive bias — BEFORE spending a week building a field-space
generative model:

    (A) SMOOTHNESS. What fraction of each sample's field is captured by a
        low-order polynomial in the reference coordinates? A warpage field that
        is 99% a quadratic bowl/twist has ~10 spatial DOF, and no amount of
        per-node generative capacity can beat a 48-number global code at
        representing it.

    (B) SAMPLE-TO-SAMPLE RANK. Across samples that share a mesh, how many PCA
        components of the field carry 95% of the variance? This is the intrinsic
        dimension of the distribution the generative model has to reproduce —
        directly comparable to `vae_latent_dim`.

Reading the result:
    (A) high (>0.95 at degree 2-3) AND (B) small (<= ~50)
        -> the conditional distribution is genuinely low-dimensional. A global z
           is the right model; per-node field generation buys little and costs
           ~25x at inference. Spend the effort on the latent/prior instead.
    (B) comparable to or above vae_latent_dim
        -> the latent is saturated; raising vae_latent_dim is the cheap first
           test, and a spatially-resolved latent is the structural fix.
    (A) low (fine structure survives every polynomial degree)
        -> the field has local structure a global code cannot express; that is
           the regime where field-space flow matching wins outright.

Usage:
    python misc/field_intrinsic_dim.py ../../dataset/SAOI/saoi_train_bot.h5 \
        --feature 2 --input-var 3
"""
import argparse
import os

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import h5py
import numpy as np


def poly_design(coords, degree):
    """Monomials in (x, y, z) up to `degree`, as columns. coords: [N, 3]."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cols = [np.ones_like(x)]
    for d in range(1, degree + 1):
        for i in range(d + 1):
            for j in range(d + 1 - i):
                k = d - i - j
                cols.append((x ** i) * (y ** j) * (z ** k))
    return np.stack(cols, axis=1)


def explained_by_polynomial(coords, field, degree):
    """R^2 of a least-squares polynomial fit of `field` on `coords`."""
    A = poly_design(coords, degree)
    coef, *_ = np.linalg.lstsq(A, field, rcond=None)
    resid = field - A @ coef
    denom = float(np.var(field))
    if denom < 1e-30:
        return float('nan')
    return 1.0 - float(np.var(resid)) / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('h5', help='mesh HDF5 in the shared dataset contract')
    ap.add_argument('--feature', type=int, default=2,
                    help='state channel to analyse (0-based within the state '
                         'block; 2 = z-displacement for SAOI)')
    ap.add_argument('--input-var', type=int, default=3,
                    help='width of the state block (rows 3 : 3+input_var)')
    ap.add_argument('--max-samples', type=int, default=400)
    ap.add_argument('--max-nodes', type=int, default=20000,
                    help='node subsample cap for the polynomial fit')
    ap.add_argument('--degrees', type=int, nargs='+', default=[1, 2, 3, 4])
    ap.add_argument('--modes', type=int, default=8,
                    help='how many leading PCA modes to classify in (C)')
    args = ap.parse_args()

    row = 3 + args.feature
    assert 0 <= args.feature < args.input_var, "--feature must index the state block"

    with h5py.File(args.h5, 'r') as f:
        sids = sorted(f['data'].keys())[:args.max_samples]
        print(f"{len(sids)} samples from {args.h5}, analysing state row {row}\n")

        # ── (A) smoothness ────────────────────────────────────────────────
        r2 = {d: [] for d in args.degrees}
        by_nodes, subs, fields = {}, {}, {}
        rng = np.random.default_rng(0)
        for sid in sids:
            nd = f['data'][sid]['nodal_data']
            n_nodes = nd.shape[2]
            coords = np.asarray(nd[0:3, 0, :], dtype=np.float64).T   # [N, 3]
            field = np.asarray(nd[row, 0, :], dtype=np.float64)      # [N]
            by_nodes.setdefault(n_nodes, []).append(sid)

            # One shared node subsample per mesh size, so (B) can stack samples
            # row-wise and still stay small enough to SVD with its spatial modes.
            if n_nodes > args.max_nodes:
                if n_nodes not in subs:
                    subs[n_nodes] = rng.choice(n_nodes, args.max_nodes, replace=False)
                idx = subs[n_nodes]
            else:
                subs.setdefault(n_nodes, np.arange(n_nodes))
                idx = subs[n_nodes]
            c_s, y_s = coords[idx], field[idx]
            fields[sid] = (c_s, y_s.astype(np.float32))
            for d in args.degrees:
                r2[d].append(explained_by_polynomial(c_s, y_s, d))

        print("(A) fraction of each sample's field explained by a polynomial "
              "in (x, y, z)")
        print(f"    {'degree':>6}  {'median R^2':>11}  {'10th pct':>9}  {'terms':>6}")
        for d in args.degrees:
            v = np.array(r2[d], dtype=np.float64)
            v = v[np.isfinite(v)]
            terms = poly_design(np.zeros((1, 3)), d).shape[1]
            print(f"    {d:>6}  {np.median(v):>11.4f}  "
                  f"{np.percentile(v, 10):>9.4f}  {terms:>6}")

        # ── (B) sample-to-sample rank on a shared mesh ────────────────────
        n_nodes, group = max(by_nodes.items(), key=lambda kv: len(kv[1]))
        print(f"\n(B) PCA across the {len(group)} samples that share "
              f"{n_nodes} nodes")
        if len(group) < 4:
            print("    not enough samples on any single mesh size; "
                  "(B) needs >= 4. Skipped.")
            return
        coords_g = fields[group[0]][0]
        Y = np.stack([fields[s][1] for s in group]).astype(np.float64)   # [S, n_sub]
        # Subtract the per-node MEAN field: what is left is exactly the epsilon
        # a generative model has to produce -- mu(g) is deterministic and is the
        # deterministic model's job, not the sampler's.
        Y -= Y.mean(axis=0, keepdims=True)
        U, sv, Vh = np.linalg.svd(Y, full_matrices=False)
        energy = sv ** 2
        cum = np.cumsum(energy) / energy.sum()
        print(f"    {'variance kept':>13}  {'components':>10}")
        for thr in (0.90, 0.95, 0.99, 0.999):
            print(f"    {thr:>13.3f}  {int(np.searchsorted(cum, thr) + 1):>10}")
        print(f"    rank ceiling (S-1) = {len(group) - 1}   "
              f"-- a count at the ceiling means MORE samples are needed to "
              f"resolve the true dimension")

        # ── (C) is the VARIATION global or local? ─────────────────────────
        # This is what decides the sampler's architecture. A leading mode that a
        # low-order polynomial explains is a GLOBAL mode: the whole part bows one
        # way or the other, and the sampler must commit to that early (small t),
        # which needs a large receptive field -> keep the multiscale V-cycle.
        # Leading modes that no polynomial explains are LOCAL: the sampler's job
        # is short-range and a shallow fine-mesh GNN suffices, with mu(g) handled
        # once, outside the integration loop, by the full HI-MGN.
        k = min(args.modes, Vh.shape[0])
        deg = max(args.degrees)
        print(f"\n(C) are the leading variation modes global or local? "
              f"(polynomial degree {deg})")
        print(f"    {'mode':>5}  {'% of variance':>13}  {'poly R^2':>9}  verdict")
        share = energy / energy.sum()
        for i in range(k):
            r = explained_by_polynomial(coords_g, Vh[i], deg)
            verdict = ("GLOBAL (smooth)" if r > 0.8 else
                       "mixed" if r > 0.4 else "LOCAL (fine structure)")
            print(f"    {i:>5}  {100 * share[i]:>12.2f}%  {r:>9.4f}  {verdict}")
        glob = sum(share[i] for i in range(k)
                   if explained_by_polynomial(coords_g, Vh[i], deg) > 0.8)
        print(f"\n    variance in GLOBAL modes (top {k}): {100 * glob:.1f}%")
        print("    >50%  -> the stochastic part itself has a global mode; the "
              "sampler needs\n"
              "             multiscale reach at small t. Keep the V-cycle in "
              "the loop.\n"
              "    <20%  -> the stochastic part is local. Factor the model as\n"
              "             y = mu_HI-MGN(g) [1 forward] + eps_sampler(local GNN) "
              "[N forwards]\n"
              "             and only the small local net pays the per-step cost.")


if __name__ == '__main__':
    main()
