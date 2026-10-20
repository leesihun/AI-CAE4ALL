"""Spread statistic, ensemble scores and the histogram figure.

Copied verbatim from MeshGraphNets_Variational/inference_profiles/rollout.py
(the repos share no imports), so a SAR inference run writes the same
spread_values.npz / spread_metrics.json / histogram_compare.png, scored by
the same code, as an MGN-V one. Keep the two copies identical.
"""
import json
import os

import h5py
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Spread histogram + ensemble scores (generated vs ground-truth eval dataset)
#
# Spread metric (one scalar per realization):
#     spread = max(field_at_nodes) - min(field_at_nodes)   at the final timestep.
#
# WHICH field is `spread_channel`, an index into the OUTPUT block -- the same
# index in both places it is read:
#     ground truth : nodal_data row 3 + spread_channel   (rows 0:3 are coords)
#     rollout      : output channel spread_channel       ([x, y, z, <out>, part])
# It used to be the hardcoded row 5, which is z_disp only on a displacement
# dataset laid out [x, y, z, ux, uy, uz, ...]. On the probabilistic slots row 5
# is velocity_x (ex1) or uy (ex2), so the plot was labelled for a quantity it
# was not measuring. The default 2 reproduces the old behaviour exactly.
# ---------------------------------------------------------------------------
SPREAD_CHANNEL_DEFAULT = 2

# HOW that field is reduced to one number is `spread_stat`, applied identically
# on both sides. `range` (max - min) is the original and the default, but it is
# only informative when the field's extremes differ between realizations. On a
# damage field that saturates at 1, or a displacement field driven to a
# prescribed value, max - min is the boundary condition, not the solution: on
# probabilistic ex2 it measures cv=0.000 across the 250 eval scenes, so every
# calibration score computed from it would be noise. `mean` and `std` summarise
# the field itself; on a 0/1 damage field `mean` is the damaged-node fraction
# (cv=0.047 there), which is what the stochastic crack path actually moves.
#
# `mag_max` is the one VECTOR statistic: the largest Euclidean norm over nodes
# of several output channels, listed in spread_channel (`spread_channel 0, 1, 2`
# on [ux, uy, uz] is max |u|, the total displacement). One component's range
# mixes the amplitude with the orientation: on ex3 shell buckling
# max(ux) - min(ux) moves with where around the circumference the dimple forms
# (it spans 1-2x max |u| across draws of one shell), while max |u| is the
# dataset's own max_deform attribute. The scalar stats take exactly one channel.
SPREAD_STATS = ('range', 'mean', 'std', 'mag_max')
VECTOR_SPREAD_STATS = ('mag_max',)
SPREAD_STAT_DEFAULT = 'range'


def _format_bytes(num_bytes):
    """Compact binary-size formatter for CUDA memory logs."""
    num_bytes = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024.0 or unit == "TiB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0


def _is_cuda_oom(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _spread_stat(field, stat=SPREAD_STAT_DEFAULT):
    """Reduce a node field to the one scalar summarising a realization.

    `field` is [nodes] for one channel, or [nodes, k] for a vector stat.
    See SPREAD_STATS for why the choice is a config key and not just max - min.
    """
    v = np.asarray(field, dtype=np.float64)
    if v.size == 0:
        return float('nan')
    if stat == 'mag_max':
        return float(np.linalg.norm(v.reshape(v.shape[0], -1), axis=1).max())
    if v.ndim != 1:
        raise ValueError(f"spread_stat {stat!r} reduces one channel, got a {v.shape} field")
    if stat == 'range':
        return float(v.max() - v.min())
    if stat == 'mean':
        return float(v.mean())
    if stat == 'std':
        return float(v.std())
    raise ValueError(f"spread_stat must be one of {SPREAD_STATS}, got {stat!r}")


def _spread_stat_expr(label, stat):
    """Axis text for the statistic, e.g. 'max(damage) - min(damage)'."""
    if stat == 'range':
        return f"max({label}) - min({label})"
    if stat == 'mag_max':
        return f"max |{label}|"
    return f"{stat}({label})"


def _spread_index(config):
    """(channel index, stat) from spread_channel / spread_stat, validated.

    The index is an int for a scalar stat and a list for a vector stat: the
    config parser returns a bare scalar for `spread_channel 0` and a list for
    `spread_channel 0, 1, 2`. Either form indexes the output axis of a numpy
    state array directly ([nodes] or [nodes, k]).
    """
    raw = config.get('spread_channel', SPREAD_CHANNEL_DEFAULT)
    stat = str(config.get('spread_stat', SPREAD_STAT_DEFAULT)).lower()
    if stat not in SPREAD_STATS:
        raise ValueError(f"spread_stat must be one of {SPREAD_STATS}, got {stat!r}")
    if isinstance(raw, (list, tuple)):
        idx = [int(c) for c in raw]
        if stat not in VECTOR_SPREAD_STATS:
            raise ValueError(f"spread_stat {stat!r} reduces one channel, but spread_channel "
                             f"lists {idx}; only {VECTOR_SPREAD_STATS} take several")
        return idx, stat
    return int(raw), stat


def _eval_dataset_spreads(h5_path, out_channel=SPREAD_CHANNEL_DEFAULT,
                          stat=SPREAD_STAT_DEFAULT):
    """(spreads, scene_ids): one ground-truth spread per sample in the eval HDF5.

    `out_channel` indexes the OUTPUT block, so the ground-truth row is
    3 + out_channel -- the same channel the rollout side reads. A list of
    channels (vector stat) reads those rows as one [nodes, k] field.

    The ids are returned so the scorer can pair each truth with the ensemble
    generated for that same scene. Without them the generated side is a pooled
    marginal, and a model that ignores geometry entirely scores as well as one
    that conditions correctly.
    """
    rows = 3 + np.asarray(out_channel, dtype=np.int64)
    spreads, ids = [], []
    with h5py.File(h5_path, 'r') as f:
        if 'data' not in f:
            raise RuntimeError(f"No /data group in {h5_path}")
        for sample_id in f['data'].keys():
            nodal = f[f'data/{sample_id}/nodal_data']
            if int(rows.max()) >= nodal.shape[0]:
                raise RuntimeError(
                    f"spread_channel {out_channel} needs nodal_data row {int(rows.max())}, but "
                    f"{h5_path} sample {sample_id} has only {nodal.shape[0]} rows")
            if rows.ndim == 0:
                field = nodal[int(rows), -1, :]
            else:
                field = nodal[:, -1, :][rows].T
            spreads.append(_spread_stat(field, stat))
            ids.append(str(sample_id))
    return np.asarray(spreads, dtype=np.float64), np.asarray(ids, dtype=object)


def _crps_ensemble(ensemble, truth):
    """CRPS of a finite ensemble against one scalar truth (fair estimator).

    CRPS = mean|x_i - y| - sum_ij|x_i - x_j| / (2 m (m-1)). The (m-1) denominator
    is the fair form; with the biased 1/(2 m^2) version a K-member ensemble is
    rewarded for being too narrow, which is the exact defect being scored here.
    """
    x = np.asarray(ensemble, dtype=np.float64)
    m = x.size
    if m == 0:
        return float('nan')
    term1 = float(np.abs(x - float(truth)).mean())
    if m == 1:
        return term1
    term2 = float(np.abs(x[:, None] - x[None, :]).sum()) / (2.0 * m * (m - 1))
    return term1 - term2


def _spread_scores(gt, gt_scene, gen, gen_scene):
    """Ensemble scores for the spread statistic, paired scene by scene.

    Each held-out scene carries exactly ONE truth (the simulation ran once), so
    the calibration question is where that truth falls among the draws for the
    SAME scene. Pooling the draws instead measures the marginal, in which a model
    that ignores the geometry entirely can still look perfect.

    Keys and their targets:
        crps_norm      mean per-scene CRPS / sd(gt)                       0
        spread_skill   mean within-scene sd / RMSE of the ensemble mean   1
        pit_ks         KS distance of the truth ranks from uniform        0
        pit_tails      share of truths in the outer 2%                    0.04
        w1_norm        pooled 1-Wasserstein(gt, gen) / sd(gt)             0
        dmean_norm     pooled (mean gen - mean gt) / sd(gt)               0
        sd_ratio       pooled sd(gen) / sd(gt)                            1
    The last three describe the marginal -- the two curves the histogram draws.
    """
    gt = np.asarray(gt, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    gt_scene = np.asarray([str(s) for s in gt_scene])
    gen_scene = np.asarray([str(s) for s in gen_scene])
    gsd = float(gt.std())
    denom = gsd if gsd > 0 else 1.0

    out = {'n_gt': int(gt.size), 'n_gen': int(gen.size),
           'gt_mean': float(gt.mean()), 'gt_sd': gsd,
           'gen_mean': float(gen.mean()), 'gen_sd': float(gen.std())}

    q = np.linspace(0.0, 1.0, 1001)
    out['w1_norm'] = float(np.abs(np.quantile(gen, q) - np.quantile(gt, q)).mean()) / denom
    out['dmean_norm'] = (float(gen.mean()) - float(gt.mean())) / denom
    out['sd_ratio'] = float(gen.std()) / denom

    truth_by_scene = {s: v for s, v in zip(gt_scene, gt)}
    crps, ranks, spreads, errors = [], [], [], []
    rng = np.random.default_rng(0)
    for scene in np.unique(gen_scene):
        if scene not in truth_by_scene:
            continue
        ens = gen[gen_scene == scene]
        y = float(truth_by_scene[scene])
        crps.append(_crps_ensemble(ens, y))
        # Rank of the truth among the draws, ties broken at random so a
        # degenerate ensemble does not pile every truth into one bin.
        below = int((ens < y).sum())
        tied = int((ens == y).sum())
        ranks.append((below + (int(rng.integers(0, tied + 1)) if tied else 0)) / ens.size)
        spreads.append(float(ens.std(ddof=1)) if ens.size > 1 else 0.0)
        errors.append(float(ens.mean()) - y)
    out['n_scenes'] = len(crps)
    if not crps:
        return out

    out['crps_norm'] = float(np.mean(crps)) / denom
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    out['ens_mean_rmse_norm'] = rmse / denom
    out['spread_skill'] = (float(np.mean(spreads)) / rmse) if rmse > 0 else float('nan')

    r = np.sort(np.asarray(ranks, dtype=np.float64))
    n = r.size
    out['pit_ks'] = float(np.max(np.abs(np.arange(1, n + 1) / n - r)))
    out['pit_tails'] = float(((r < 0.02) | (r > 0.98)).mean())
    out['pit_ranks'] = [float(v) for v in r]
    return out


def _open_in_viewer(path):
    """Best-effort: open a saved image in the OS default viewer (for display)."""
    import sys
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            import subprocess
            subprocess.Popen(['open', path])
        else:
            import subprocess
            subprocess.Popen(['xdg-open', path])
        return True
    except Exception as exc:  # headless box / no viewer — never fatal
        print(f"  (could not open image viewer: {exc})")
        return False


def _plot_spread_histogram(gt, gen, out_path, eval_path=None, rollout_dir=None,
                           bins=60, clip_quantile=0.0, show=False, dpi=150,
                           label='z_disp', scores=None,
                           stat=SPREAD_STAT_DEFAULT, title=None):
    """Overlay GT vs generated spread histograms; save PNG, optionally display.

    Renders headlessly (Agg) so it works on a display-less box; when show=True
    the saved PNG is additionally opened in the OS default viewer. `title`, when
    given, becomes the first line of the axes title (the group a per-geometry
    figure shows).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    gt = np.asarray(gt, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)

    if clip_quantile > 0:
        q = clip_quantile
        lo = float(min(np.quantile(gt, q), np.quantile(gen, q)))
        hi = float(max(np.quantile(gt, 1 - q), np.quantile(gen, 1 - q)))
    else:
        lo = float(min(gt.min(), gen.min()))
        hi = float(max(gt.max(), gen.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1e-8
    bin_edges = np.linspace(lo, hi, bins + 1)

    def _stats(col):
        return (float(col.mean()), float(col.std()), float(col.min()), float(col.max()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gt, bins=bin_edges, density=True, alpha=0.55,
            label=f"eval dataset (n={gt.size:,})", color="steelblue")
    ax.hist(gen, bins=bin_edges, density=True, alpha=0.55,
            label=f"generated (n={gen.size:,})", color="darkorange")
    g_mu, g_sd, g_lo, g_hi = _stats(gt)
    p_mu, p_sd, p_lo, p_hi = _stats(gen)
    score_line = ""
    if scores:
        bits = []
        for key, target in (('crps_norm', '0'), ('spread_skill', '1'),
                            ('pit_ks', '0'), ('sd_ratio', '1')):
            value = scores.get(key)
            if value is not None and np.isfinite(value):
                bits.append(f"{key}={value:.3f} (->{target})")
        if bits:
            # Two per line: four on one line overflow the 8-inch figure.
            score_line = "".join(
                "\n" + "   ".join(bits[i:i + 2]) for i in range(0, len(bits), 2))
    expr = _spread_stat_expr(label, stat)
    head = f"{title}\n" if title else ""
    ax.set_title(
        f"{head}{expr} per realization, final timestep\n"
        f"GT  mu={g_mu:.3e}  sigma={g_sd:.3e}  [{g_lo:.3e}, {g_hi:.3e}]\n"
        f"Gen mu={p_mu:.3e}  sigma={p_sd:.3e}  [{p_lo:.3e}, {p_hi:.3e}]"
        f"{score_line}"
    )
    ax.set_xlabel(expr)
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(alpha=0.3)

    if eval_path is not None or rollout_dir is not None:
        fig.suptitle(f"eval:     {eval_path}\nrollouts: {rollout_dir}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
    else:
        fig.tight_layout()

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved histogram figure: {os.path.abspath(out_path)}")
    if show:
        _open_in_viewer(os.path.abspath(out_path))
    return out_path


def _eval_dataset_groups(h5_path, attr):
    """{scene id: group value} from `data/{id}/metadata.attrs[attr]`.

    Empty when any scene lacks the attribute, so a file without groups falls
    back to the pooled figure alone instead of a partial grouping.
    """
    groups = {}
    with h5py.File(h5_path, 'r') as f:
        for sample_id in f['data'].keys():
            meta = f[f'data/{sample_id}'].get('metadata')
            if meta is None or attr not in meta.attrs:
                return {}
            value = meta.attrs[attr]
            if isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, np.ndarray) and value.size == 1:
                value = value.item()
            groups[str(sample_id)] = str(value)
    return groups


def _group_order(values):
    """Group values sorted numerically when they all parse, else as text."""
    values = sorted({str(v) for v in values})
    try:
        return sorted(values, key=float)
    except ValueError:
        return values


def _spread_by_group(gt, gt_scene, gen, gen_scene, groups, attr, out_dir,
                     eval_path=None, bins=60, clip_quantile=0.0, dpi=150,
                     label='z_disp', stat=SPREAD_STAT_DEFAULT):
    """Score and draw each group (held-out geometry) on its own.

    The pooled histogram adds the between-geometry variance to the
    within-geometry one, so an ensemble that is too narrow for every geometry
    can still draw a pooled curve of the right width. Here each group gets its
    own truths, only the draws generated for its scenes, and scores normalized
    by that group's own sd(gt). Writes histogram_compare_<attr>_<value>.png per
    group and spread_metrics_by_group.json; returns {value: scores}.
    """
    gt = np.asarray(gt, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    gt_scene = np.asarray([str(s) for s in gt_scene])
    gen_scene = np.asarray([str(s) for s in gen_scene])
    gt_group = np.asarray([groups.get(s, '') for s in gt_scene])
    gen_group = np.asarray([groups.get(s, '') for s in gen_scene])

    by_group = {}
    for value in _group_order(v for v in gt_group if v):
        g_mask = gt_group == value
        p_mask = gen_group == value
        if not p_mask.any():
            continue
        scores = _spread_scores(gt[g_mask], gt_scene[g_mask],
                                gen[p_mask], gen_scene[p_mask])
        scores.pop('pit_ranks', None)
        by_group[value] = scores
        safe = ''.join(c if c.isalnum() or c in '-.' else '_' for c in value)
        _plot_spread_histogram(
            gt[g_mask], gen[p_mask],
            os.path.join(out_dir, f'histogram_compare_{attr}_{safe}.png'),
            eval_path=eval_path, rollout_dir=out_dir, bins=bins,
            clip_quantile=clip_quantile, dpi=dpi, label=label, scores=scores,
            stat=stat,
            title=f"{attr} {value}: {int(g_mask.sum())} scenes (scores vs this group's sd)")
        print(f"  [GROUP] {attr}={value} n_gt={int(g_mask.sum())} "
              f"n_gen={int(p_mask.sum())} crps_norm={scores.get('crps_norm', float('nan')):.4f} "
              f"spread_skill={scores.get('spread_skill', float('nan')):.4f} "
              f"dmean_norm={scores['dmean_norm']:.4f} sd_ratio={scores['sd_ratio']:.4f}")
    path = os.path.join(out_dir, 'spread_metrics_by_group.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump({'attr': attr, 'groups': by_group}, fh, indent=2)
    print(f"  [GROUP] metrics -> {path}")
    return by_group


def _rollout_feature_names(dataset_dir, output_dim):
    """Channel names written into each rollout file.

    Prefer the inference dataset's own declared names (metadata/feature_names)
    for the coordinate and state rows, so an evaluator can pair prediction and
    truth channels by name. The legacy NASA-CRM list below used to be written
    for *every* dataset, so ex9's ux/uy came out labelled x_disp(mm)/y_disp(mm),
    matched nothing but x/y/z_coord in the truth file, and the Studio scored the
    coordinates as a perfect model. The trailing row is the node-type/part id.
    """
    legacy = [b'x_coord', b'y_coord', b'z_coord',
              b'x_disp(mm)', b'y_disp(mm)', b'z_disp(mm)',
              b'stress(MPa)', b'Part No.']
    keep = 3 + int(output_dim)
    names = None
    try:
        with h5py.File(dataset_dir, 'r') as f:
            if 'metadata/feature_names' in f:
                raw = f['metadata/feature_names'][()]
                names = [n if isinstance(n, bytes) else str(n).encode('utf-8') for n in raw]
    except Exception:
        names = None
    if names and len(names) >= keep:
        return np.array(names[:keep] + [b'Part No.'])
    return np.array(legacy[:keep] + [b'Part No.'])
