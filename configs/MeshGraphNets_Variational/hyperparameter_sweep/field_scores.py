"""Field-level sample realism and distribution match of a sweep job, from whole fields.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/field_scores.py r13 [job ...]

plots.py scores one scalar per field (spread_stat). This script scores the
fields themselves, so it needs them: the job's inference must have run with
`save_rollouts True` (the sweeps set it False; run a subset of scenes to keep
the file count down) and the files are read from `inference_output_dir`, or
from --rollouts. Per held-out geometry, with the FE results and the generated
fields of its scenes on the same mesh and node order:

  distance   d(a, b) = sqrt(mean over nodes of |a_i - b_i|^2), over every
             output channel of the last step. Reported in units of the
             geometry's own FE scatter s = RMS distance of an FE result to the
             FE mean field, so 1.0 is "as far as an FE result is from the mean".

  realism    (a) Is every generated field close to an FE result?
             Each geometry's n FE results are cut into a reference half A and a
             held-out half B (random, --splits times, the permutations of plots.py).
             d_FE  = distance of each B result to its nearest result in A.
             d_gen = distance of each generated field to its nearest result in A.
             The score is median(d_gen) / median(d_FE), and the same ratio of
             the 90th percentiles (the worst tenth). 1 is "as close to the
             FE results as an FE result is to the others". Above 1: the draws
             lie outside the FE cloud. Below 1: they sit nearer the FE results
             than FE results do to each other, which is what a mean-field
             collapse or a copy of the reference does; W2 and the spread ratio
             of plots.py decide which. So the score is read two-sided.

  field W2   (b) Does the set of generated fields match the set of FE results?
             W2 = (min over one-to-one pairings of mean d^2)^(1/2), exact
             (Hungarian), k generated fields against k held-out FE results.
             `floor` pairs k other FE results with the same held-out ones.

Writes infer/field_scores.json and infer/field_realism.png (one split per
geometry, the one with the median ratio) beside the job's other outputs.

Needs numpy, scipy, h5py, matplotlib: run it with a method interpreter, from anywhere.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plots   # noqa: E402  (the sweep's config, job and path helpers)

MIN_FE = 4      # fewer FE results than this in a geometry: not scored
MIN_GEN = 8     # fewer generated fields than this: not scored
CHUNK = 1 << 15


# ------------------------------------------------------------------ loading
def _rows(output_var):
    """nodal_data rows that are predictions: after the three coordinate rows."""
    return slice(3, 3 + output_var)


def load_fe(h5_path, scene_ids, output_var):
    """[n, N * output_var] last-step output fields of the given scenes, and their reference coordinates."""
    import h5py
    fields, coords = [], None
    with h5py.File(h5_path, "r") as f:
        for sid in scene_ids:
            nodal = f[f"data/{sid}/nodal_data"]
            fields.append(np.asarray(nodal[_rows(output_var), -1, :], dtype=np.float32).T.reshape(-1))
            if coords is None:
                coords = np.asarray(nodal[0:3, 0, :], dtype=np.float32)
    return np.stack(fields), coords


def rollout_files(rollouts):
    return sorted(Path(rollouts).glob("rollout_sample*.h5"))


def load_gen(files, groups, output_var):
    """{geometry value: ([m, N * output_var] fields, reference coordinates)} from rollout files."""
    import h5py
    fields, coords = {}, {}
    for path in files:
        with h5py.File(path, "r") as f:
            sid = next(iter(f["data"]))
            value = groups.get(str(sid))
            if value is None:
                continue   # a scene outside the eval file
            nodal = f[f"data/{sid}/nodal_data"]
            fields.setdefault(value, []).append(
                np.asarray(nodal[_rows(output_var), -1, :], dtype=np.float32).T.reshape(-1))
            coords.setdefault(value, np.asarray(nodal[0:3, 0, :], dtype=np.float32))
    return {v: (np.stack(x), coords[v]) for v, x in fields.items()}


# ------------------------------------------------------------------ scoring
def sq_dists(x, y, n_nodes):
    """[len(x), len(y)] squared d: mean over nodes of the squared vector difference.

    Gram form with float64 accumulation over feature chunks. x and y are
    centered on the FE mean field by the caller, so the norms are the size of
    the scatter and the cancellation stays harmless.
    """
    cross = np.zeros((x.shape[0], y.shape[0]))
    xx = np.zeros(x.shape[0])
    yy = np.zeros(y.shape[0])
    for s in range(0, x.shape[1], CHUNK):
        a = x[:, s:s + CHUNK].astype(np.float64)
        b = y[:, s:s + CHUNK].astype(np.float64)
        cross += a @ b.T
        xx += np.einsum("ij,ij->i", a, a)
        yy += np.einsum("ij,ij->i", b, b)
    return np.maximum(xx[:, None] + yy[None, :] - 2.0 * cross, 0.0) / n_nodes


def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    return {"p50": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def score_geometry(fe, gen, n_nodes, perms, seed):
    """Realism ratios and field W2 of one geometry, over the given FE permutations.

    fe [n, F], gen [m, F] (F = nodes * channels), perms: one permutation of range(n) per split.
    Returns (per-split arrays, one split's distances for the figure).
    """
    from scipy.optimize import linear_sum_assignment
    mu = fe.mean(axis=0, dtype=np.float64).astype(np.float32)
    fe, gen = fe - mu, gen - mu
    n, m = fe.shape[0], gen.shape[0]
    scale = math.sqrt(float((fe.astype(np.float64) ** 2).sum()) / (n * n_nodes))
    if not scale > 0.0:
        return None
    dff = sq_dists(fe, fe, n_nodes) / scale ** 2     # all distances in units of the FE scatter
    dgf = sq_dists(gen, fe, n_nodes) / scale ** 2
    ka = n // 2
    k = min(ka, m, n - ka)
    rng = np.random.default_rng(seed + 1)
    out = {key: [] for key in ("ratio", "ratio_p90", "w2_floor", "w2_model")}
    dists = []
    for perm in perms:
        a, b = perm[:ka], perm[ka:]
        d_fe = np.sqrt(dff[np.ix_(b, a)].min(axis=1))
        d_gen = np.sqrt(dgf[:, a].min(axis=1))
        out["ratio"].append(float(np.median(d_gen) / np.median(d_fe)))
        out["ratio_p90"].append(float(np.percentile(d_gen, 90) / np.percentile(d_fe, 90)))
        bk = b[:k]
        cost = dff[np.ix_(a[:k], bk)]
        r, c = linear_sum_assignment(cost)
        out["w2_floor"].append(math.sqrt(float(cost[r, c].mean())))
        sub = rng.choice(m, k, replace=False)
        cost = dgf[np.ix_(sub, bk)]
        r, c = linear_sum_assignment(cost)
        out["w2_model"].append(math.sqrt(float(cost[r, c].mean())))
        dists.append((d_fe, d_gen))
    out = {key: np.asarray(v) for key, v in out.items()}
    pick = int(np.argsort(out["ratio"])[len(perms) // 2])
    return out, dists[pick], {"n_fe": n, "n_gen": m, "k": k, "n_ref": ka, "fe_scatter": scale}


def summarize(per_geometry):
    """Mean ratio and rms W2 over geometries, once per split; then the spread over splits."""
    keys = ("ratio", "ratio_p90")
    out = {}
    for key in keys:
        out[key] = _stats(np.mean([g[key] for g in per_geometry.values()], axis=0))
    for key in ("w2_floor", "w2_model"):
        out[key] = _stats(np.sqrt(np.mean([g[key] ** 2 for g in per_geometry.values()], axis=0)))
    return out


# ------------------------------------------------------------------- figure
def figure(dists, rows, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = list(dists)
    ncol = min(4, len(values))
    nrow = math.ceil(len(values) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.9 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, value in zip(axes.ravel(), values):
        ax.axis("on")
        d_fe, d_gen = dists[value]
        top = float(max(np.percentile(d_fe, 99.5), np.percentile(d_gen, 99.5)))
        bins = np.linspace(0.0, top, 30)
        ax.hist(d_fe, bins=bins, density=True, alpha=0.55, color="steelblue", label="FE to the other FE")
        ax.hist(d_gen, bins=bins, density=True, alpha=0.55, color="darkorange", label="generated to FE")
        r = rows[value]
        ax.set_title(f"{value}   ratio {r['ratio']['p50']:.2f}", fontsize=9)
        ax.tick_params(labelsize=7)
        if ax is axes.ravel()[0]:
            ax.legend(fontsize=7)
    fig.supxlabel("distance to the nearest FE result of the reference half, in units of the FE scatter", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------- jobs
def job_scores(job, spread, rollouts, splits):
    cfg = plots.cfg_values(job["infer"])
    attr = cfg.get("eval_group_attr") or cfg.get("split_group_attr")
    eval_h5 = cfg.get("eval_dataset") or cfg.get("infer_dataset")
    out_dir = cfg.get("inference_output_dir")
    if not (attr and eval_h5 and out_dir):
        raise ValueError("config needs split_group_attr, eval_dataset (or infer_dataset) and inference_output_dir")
    output_var = int(cfg.get("output_var", 0))
    if output_var < 1:
        raise ValueError("config needs output_var")
    out_dir = plots.method_path(job["method"], out_dir)
    eval_h5 = plots.method_path(job["method"], eval_h5)
    files = rollout_files(rollouts or out_dir)
    if not files:
        raise ValueError(f"no rollout_sample*.h5 in {rollouts or out_dir}; "
                         "run the inference with `save_rollouts True`")
    groups = spread._eval_dataset_groups(str(eval_h5), attr)
    if not groups:
        raise ValueError(f"no `{attr}` attribute in {eval_h5}")
    order = list(spread._group_order(groups.values()))
    gens = load_gen(files, groups, output_var)
    fake = {"gt_group": np.asarray([groups[s] for s in groups]), "order": order}
    perms = plots.split_perms(fake, splits)

    rows, skipped, dists = {}, {}, {}
    per_geometry = {}
    for value in order:
        ids = sorted(s for s, v in groups.items() if v == value)
        if len(ids) < MIN_FE or value not in gens or len(gens[value][0]) < MIN_GEN:
            skipped[value] = f"{len(ids)} FE results, {len(gens[value][0]) if value in gens else 0} generated"
            continue
        fe, fe_coords = load_fe(eval_h5, ids, output_var)
        gen, gen_coords = gens[value]
        if fe.shape[1] != gen.shape[1] or not np.allclose(fe_coords, gen_coords, atol=1e-4 * float(np.ptp(fe_coords) or 1.0)):
            raise ValueError(f"{attr} {value}: the generated fields are not on the FE mesh "
                             f"({gen.shape[1] // output_var} vs {fe.shape[1] // output_var} nodes, or another node order); "
                             "the nearest-result distance needs one mesh per geometry")
        scored = score_geometry(fe, gen, fe.shape[1] // output_var, perms[value], plots.SPLIT_SEED)
        if scored is None:
            skipped[value] = "all FE results identical"
            continue
        per, dist, info = scored
        per_geometry[value] = per
        dists[value] = dist
        rows[value] = {key: _stats(per[key]) for key in per} | info
    if not rows:
        raise ValueError("no geometry could be scored: " + "; ".join(f"{k}: {v}" for k, v in skipped.items()))
    result = {"attr": attr, "output_var": output_var, "splits": splits, "seed": plots.SPLIT_SEED,
              "rollouts": str(rollouts or out_dir), "n_files": len(files),
              "summary": summarize(per_geometry), "groups": rows, "skipped": skipped}
    (out_dir / "field_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    png = out_dir / "field_realism.png"
    figure(dists, rows, png, f"{job['job']}: realism, generated vs FE ({splits} splits, the median one shown)")
    return result, png


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("round", help="round folder beside this script (r8 ... r14) or its path")
    ap.add_argument("jobs", nargs="*", help="only these jobs")
    ap.add_argument("--rollouts", help="directory of rollout_sample*.h5 (default: the job's inference_output_dir)")
    ap.add_argument("--splits", type=int, default=plots.SPLITS)
    args = ap.parse_args()

    sweep_dir = Path(args.round)
    if not sweep_dir.is_dir():
        sweep_dir = plots.SWEEP_ROOT / args.round
    spread = plots.load_spread_module()
    print("Field level (distances in units of each geometry's FE scatter)")
    print("  ratio: median distance of a generated field to its nearest FE result / the same for held-out FE results (1 = FE level)")
    for job in plots.read_jobs(sweep_dir):
        if args.jobs and job["job"] not in args.jobs:
            continue
        try:
            res, png = job_scores(job, spread, args.rollouts, args.splits)
        except Exception as exc:   # one broken run never stops the rest
            print(f"  {job['job']:<18} failed: {exc}")
            continue
        s = res["summary"]
        print(f"  {job['job']:<18} ratio {s['ratio']['p50']:.2f} ({s['ratio']['p5']:.2f}-{s['ratio']['p95']:.2f})  "
              f"worst tenth {s['ratio_p90']['p50']:.2f}  field W2 {s['w2_model']['p50']:.2f} "
              f"vs FE alone {s['w2_floor']['p50']:.2f}   {plots.shown(png)}")
        for value, r in res["groups"].items():
            print(f"  {'':<18} {res['attr']} {value:>6}: ratio {r['ratio']['p50']:.2f}  "
                  f"W2 {r['w2_model']['p50']:.2f} vs {r['w2_floor']['p50']:.2f}  ({r['n_gen']} draws, {r['n_fe']} FE)")
        for value, why in res["skipped"].items():
            print(f"  {'':<18} {res['attr']} {value:>6}: skipped ({why})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
