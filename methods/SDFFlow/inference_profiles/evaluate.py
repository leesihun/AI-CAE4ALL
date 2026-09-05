"""
`evaluate` mode: held-out reconstruction metrics for a trained SDF-VAE.

For every shape of one dataset split the encoder mean is decoded through the
same Marching Cubes path the other inference modes use, and the result is
scored against the shape's stored ground truth:

    surface_mean / surface_p95 / surface_max
        exact unsigned distance from every stored GT surface point of the shape
        to the reconstructed mesh (all of them, not a subsample)
    pred_to_gt_mean / pred_to_gt_p95 / chamfer_mean
        the other direction: 8192 points sampled on the reconstructed mesh to
        their nearest stored GT surface point, and the average of the two means.
        Without it a noisy space-filling reconstruction scores well, because
        every GT point still finds some nearby predicted surface.
    sign_accuracy / sign_balanced_accuracy / positive_fraction
        sign(pred) == sign(target) over the stored SDF query points with
        |target| > 0.001, predicted with vae.decode_flat in chunks; the balanced
        form averages the inside and outside rates (trivial baseline 0.5 rather
        than the ~0.64 majority-class floor that positive_fraction records)
    sdf_l1
        truncated-L1 SDF error on the same points (the training objective)
    body_count_raw / watertight / valid
        Marching-Cubes component count before keep_largest, watertightness of
        the kept body, and whether a zero crossing existed at all

With `latent_refine_steps > 0` a second, decoder-frozen latent refinement pass
(inference_profiles/latent_refine.py) is scored too; its fields carry the `ref_`
prefix next to the encoder's `enc_`. The shape's stored query points are then
split in half by a seeded coin flip: the refinement fits one half and the `ref_`
SDF metrics are scored on the OTHER one, so they are not the in-sample fit of
the very points that were optimized against (which improves monotonically with
`latent_refine_steps` and says nothing about accuracy). `enc_*` keep their
meaning -- all stored query points, exactly as without refinement -- and the
same metrics restricted to the held-out half are ADDED as

    enc_sdf_l1_heldout / enc_sign_accuracy_heldout / enc_sign_balanced_accuracy_heldout

which is the apples-to-apples partner of `ref_sdf_l1` / `ref_sign_accuracy`.
The `ref_` rows repeat theirs under those same `_heldout` names (identical
values; `ref_` is already held out) so the aggregate table can put the two
columns side by side, and the fit half is kept explicitly as
`ref_sdf_l1_insample` / `ref_sign_accuracy_insample`.

`eval_num_shapes N > 0` scores a RANDOM subset of the split (seeded by
`eval_seed`), not its first N shapes: under `split_by_parent` the split index
array is parent-grouped, so a head slice would be one or two bracket families.

Architecture and the encoder point budget come from the checkpoint config; the
dataset path, split keys, and eval keys come from the run config (split keys
fall back to the checkpoint's values so an evaluate config that omits them
reproduces the training split). Encoder subsampling is deterministic
(dataset.deterministic = True, seed = eval_seed) so repeated runs and different
checkpoints see identical point clouds per shape.

Outputs: <output_dir>/eval_<split>.json (settings, aggregate, per-shape rows)
and <output_dir>/eval_<split>.csv, plus an aggregate table on stdout.

`eval_task` selects what is evaluated (default `reconstruction`, the above):

  * `descriptor_calibration` -- fit the soft volume/area proxy
    (general_modules/descriptor_proxy.py) against the export-path Marching
    Cubes measurement on `calibration_num_shapes` x `calibration_samples_per_shape`
    conditional samples whose conditions are the TRUE stored conditions of
    shapes drawn (seeded) from `eval_split`, and save the
    `DescriptorCalibration` to `descriptor_calibration_path`. This is the
    artifact `mode sample`'s guidance (C2) and Newton correction (E2) need.
  * `conditional` -- the paired conditional-generation benchmark of
    GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md section 3: for `eval_num_shapes`
    shapes of `eval_split` the target is that shape's TRUE stored conditions
    (restricted to the FM checkpoint's cond_names) and every method in
    `eval_methods` (plain | rejection | c2 | e2 | c2e2) starts from the SAME
    base noise per shape (seeded by eval_seed and the shape index). Reported per
    method and condition name: relative error |actual - target| / |target| in
    RAW units (from_stored for log-stored FEA names) median / p95, valid and
    watertight rates, latent RMS drift from the plain sample, velocity-net
    calls (NFE) and wall time. FEA-named conditions are scored only when
    `condition_audit` is fea / surrogate (else 'not measurable geometrically').
    Writes eval_conditional.json / eval_conditional.csv.

Both non-default tasks need `fm_modelpath` (the FM checkpoint defines the
conditions and records the VAE it was trained against).
"""

import csv
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import trimesh

from general_modules import condition_names as CN
from general_modules.mesh_extraction import decode_sdf_grid, mesh_report, sdf_grid_to_mesh
from general_modules.sdf_dataset import build_dataset_splits
from inference_profiles.latent_refine import refine_latent
from inference_profiles.sample import (
    DEFAULT_GUIDANCE_TARGETS,
    VelocityCallCounter,
    _as_name_list,
    _audit_report,
    apply_newton,
    attach_fea_audit,
    descriptor_targets,
    fea_condition_audit,
    import_descriptor_tools,
    load_calibration,
    load_fm_stack,
    load_vae,
    make_guidance,
    newton_measure_resolution,
    raw_from_stored_targets,
    resolve_condition_audit,
    soft_proxy_names,
)
from model.velocity_net import sample_latents
from training_profiles.setup import resolve_device

EVAL_SPLITS = ('train', 'val', 'test')
EVAL_TASKS = ('reconstruction', 'descriptor_calibration', 'conditional')
EVAL_METHODS = ('plain', 'rejection', 'c2', 'e2', 'c2e2')
DEFAULT_EVAL_METHODS = ('plain', 'rejection', 'e2')
# Methods that need the descriptor tools + a calibration artifact.
CALIBRATED_METHODS = ('c2', 'e2', 'c2e2')
DEFAULT_CALIBRATION_NUM_SHAPES = 64
DEFAULT_CALIBRATION_SAMPLES_PER_SHAPE = 4
SIGN_EPS = 0.001
# Keys that define the dataset split. Precedence: run config > checkpoint config
# > native default, so an evaluate config that omits them scores the model on
# the split it was trained with.
SPLIT_KEYS = ('split_seed', 'split_by_parent', 'overfit_all_shapes', 'overfit_num_shapes')
METRIC_FIELDS = ('surface_mean', 'surface_p95', 'surface_max',
                 'pred_to_gt_mean', 'pred_to_gt_p95', 'chamfer_mean',
                 'sign_accuracy', 'sign_balanced_accuracy', 'sdf_l1')
# Added only when latent refinement is on: the SDF metrics restricted to the half
# of the stored query points the refinement never saw. `enc_` gets them next to
# its all-points values, `ref_` repeats its (already held-out) values under the
# same names, so the aggregate table compares like with like.
HELDOUT_FIELDS = ('sdf_l1_heldout', 'sign_accuracy_heldout',
                  'sign_balanced_accuracy_heldout')
# Points sampled on the reconstructed mesh for the reverse (pred -> GT) distance.
PRED_SURFACE_SAMPLES = 8192

_DISTANCE_BACKEND = None
_OPEN3D_FALLBACK_WARNED = False


def sign_accuracy(pred, target, eps=SIGN_EPS):
    """Fraction of |target| > eps points whose predicted sign matches (NaN if none).

    NOTE this metric has a majority-class floor: DeepJEB query sets are ~64%
    outside, so a decoder emitting a positive constant already scores ~0.64.
    `sign_accuracy_stats` returns the balanced version and the class balance
    itself, which is what the per-shape rows record.
    """
    pred = torch.as_tensor(pred).detach().float().flatten()
    target = torch.as_tensor(target).detach().float().flatten().to(pred.device)
    mask = target.abs() > float(eps)
    if int(mask.sum()) == 0:
        return float('nan')
    return float((torch.sign(pred[mask]) == torch.sign(target[mask])).float().mean().item())


def sign_accuracy_stats(pred, target, eps=SIGN_EPS):
    """(accuracy, balanced accuracy, positive fraction) over |target| > eps points.

    The balanced accuracy is the mean of the inside (target < 0) and outside
    (target > 0) per-class rates, so its trivial baseline is 0.5 whatever the
    class balance; `positive_fraction` records the majority-class floor of the
    plain accuracy for that shape.
    """
    pred = torch.as_tensor(pred).detach().float().flatten()
    target = torch.as_tensor(target).detach().float().flatten().to(pred.device)
    mask = target.abs() > float(eps)
    if int(mask.sum()) == 0:
        return float('nan'), float('nan'), float('nan')
    p, t = pred[mask], target[mask]
    correct = (torch.sign(p) == torch.sign(t)).float()
    outside = t > 0
    inside = ~outside
    rates = [float(correct[m].mean().item()) for m in (outside, inside) if int(m.sum()) > 0]
    balanced = float(np.mean(rates)) if rates else float('nan')
    return (float(correct.mean().item()), balanced, float(outside.float().mean().item()))


def _distance_backend():
    """'open3d' when its RaycastingScene imports (embree, ~30x faster than
    trimesh and agreeing to fp32 rounding on the same mesh), else 'trimesh'.
    Resolved and announced once per process."""
    global _DISTANCE_BACKEND
    if _DISTANCE_BACKEND is None:
        _DISTANCE_BACKEND = 'trimesh'
        try:
            import open3d  # noqa: F401
            # Reach the attribute defensively: a stripped or lazily-imported
            # open3d can have `t` without a populated `t.geometry`, and an
            # AttributeError here must fall through to trimesh, not abort.
            tensor_geometry = getattr(getattr(open3d, 't', None), 'geometry', None)
            if tensor_geometry is not None and hasattr(tensor_geometry, 'RaycastingScene'):
                _DISTANCE_BACKEND = 'open3d'
        except Exception:
            _DISTANCE_BACKEND = 'trimesh'
        print(f'Surface distance backend: {_DISTANCE_BACKEND}'
              + (' (RaycastingScene.compute_distance)' if _DISTANCE_BACKEND == 'open3d'
                 else ' (trimesh.proximity.closest_point)'))
    return _DISTANCE_BACKEND


def _open3d_distances(mesh, pts):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(mesh.vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(mesh.faces, dtype=np.uint32)))
    out = scene.compute_distance(o3d.core.Tensor(pts.astype(np.float32))).numpy()
    return np.asarray(out, dtype=np.float64).reshape(-1)


def surface_distances(mesh, points):
    """Exact unsigned distance from each point to the mesh surface, float64 [N].

    A probe that says open3d imports is not a promise that every mesh survives
    its scene builder (a zero-face or otherwise degenerate Marching-Cubes body
    can raise inside embree), so a failure here falls back to trimesh for that
    call -- the two agree to fp32 rounding -- rather than aborting the whole
    evaluation run. The fallback announces itself once per process.
    """
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if _distance_backend() == 'open3d':
        try:
            return _open3d_distances(mesh, pts)
        except Exception as exc:
            global _OPEN3D_FALLBACK_WARNED
            if not _OPEN3D_FALLBACK_WARNED:
                _OPEN3D_FALLBACK_WARNED = True
                print(f'    WARNING: the open3d distance backend failed on a mesh ({exc}); '
                      'falling back to trimesh.proximity for it (and for any later mesh '
                      'that also fails)')
    _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
    return np.asarray(dist, dtype=np.float64).reshape(-1)


@torch.no_grad()
def predict_sdf(vae, z_flat, points, device, chunk=65536):
    """vae.decode_flat over [N, 3] points in chunks -> float32 [N] on `device`."""
    pts = torch.as_tensor(np.asarray(points, dtype=np.float32), device=device)
    out = torch.empty(pts.shape[0], device=device)
    for i in range(0, pts.shape[0], chunk):
        out[i:i + chunk] = vae.decode_flat(z_flat, pts[i:i + chunk].unsqueeze(0)).squeeze(0).float()
    return out


def nearest_distances(query, reference):
    """Distance from each `query` point to the nearest point of `reference`."""
    q = np.ascontiguousarray(query, dtype=np.float64)
    r = np.ascontiguousarray(reference, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree
        return np.asarray(cKDTree(r).query(q)[0], dtype=np.float64).reshape(-1)
    except Exception:  # no scipy: chunked brute force (8192 x 8192 is small)
        out = np.empty(q.shape[0], dtype=np.float64)
        for i in range(0, q.shape[0], 1024):
            block = q[i:i + 1024]
            out[i:i + 1024] = np.sqrt(
                ((block[:, None, :] - r[None, :, :]) ** 2).sum(-1)).min(axis=1)
        return out


def evaluate_latent(vae, z_flat, gt_surface, sdf_points, sdf_values, resolution, device,
                    clamp_dist=0.1, surface_sample_seed=0):
    """Score one flat latent against a shape's stored ground truth.

    Returns (metrics dict, mesh-or-None). Surface metrics are None when the
    decode had no zero crossing.

    The surface comparison runs in BOTH directions. `surface_*` is the exact
    distance from every stored GT surface point to the reconstructed mesh, which
    alone cannot see geometry the reconstruction invents where the GT has none:
    a noisy, space-filling field scores well on it because every GT point finds
    some nearby predicted surface. `pred_to_gt_*` closes that hole by sampling
    `PRED_SURFACE_SAMPLES` points on the reconstructed mesh and measuring their
    distance to the stored GT surface cloud (a point cloud, so this direction is
    the standard point-cloud Chamfer half, not an exact surface distance).
    `chamfer_mean` is the average of the two means.
    """
    t0 = time.time()
    volume = decode_sdf_grid(vae, z_flat, resolution=resolution, device=device)
    mesh = sdf_grid_to_mesh(volume)
    report = mesh_report(mesh)
    pred = predict_sdf(vae, z_flat, sdf_points, device).cpu()
    target = torch.as_tensor(np.asarray(sdf_values, dtype=np.float32))
    acc, balanced, positive_fraction = sign_accuracy_stats(pred, target, SIGN_EPS)
    metrics = {
        'valid': bool(report['valid']),
        'watertight': bool(report.get('watertight', False)),
        'body_count_raw': report.get('body_count_raw'),
        'faces': report.get('faces'),
        'volume': report.get('volume'),
        'surface_mean': None,
        'surface_p95': None,
        'surface_max': None,
        'pred_to_gt_mean': None,
        'pred_to_gt_p95': None,
        'chamfer_mean': None,
        'sign_accuracy': acc,
        'sign_balanced_accuracy': balanced,
        'positive_fraction': positive_fraction,
        'sdf_l1': float((pred - target.clamp(-clamp_dist, clamp_dist)).abs().mean().item()),
    }
    if report['valid']:
        dist = surface_distances(mesh, gt_surface)
        metrics['surface_mean'] = float(dist.mean())
        metrics['surface_p95'] = float(np.percentile(dist, 95))
        metrics['surface_max'] = float(dist.max())
        try:
            pred_points, _ = trimesh.sample.sample_surface(
                mesh, PRED_SURFACE_SAMPLES, seed=int(surface_sample_seed))
            rev = nearest_distances(pred_points, gt_surface)
            metrics['pred_to_gt_mean'] = float(rev.mean())
            metrics['pred_to_gt_p95'] = float(np.percentile(rev, 95))
            metrics['chamfer_mean'] = 0.5 * (metrics['surface_mean'] + metrics['pred_to_gt_mean'])
        except Exception as exc:  # a degenerate mesh can defeat area sampling
            print(f'    WARNING: reverse surface sampling failed ({exc}); '
                  'pred_to_gt_* and chamfer_mean are unavailable for this shape')
    metrics['seconds'] = float(time.time() - t0)
    return metrics, mesh


def _finite(values):
    return [float(v) for v in values
            if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _mean_median(values):
    vals = _finite(values)
    if not vals:
        return {'mean': None, 'median': None, 'count': 0}
    return {'mean': float(np.mean(vals)), 'median': float(np.median(vals)), 'count': len(vals)}


def aggregate_rows(rows, prefix, fields=METRIC_FIELDS):
    """Mean/median of every metric plus valid / watertight / body-count rates."""
    get = lambda key: [row.get(f'{prefix}_{key}') for row in rows]
    n = len(rows)
    valid = [bool(v) for v in get('valid')]
    watertight = [bool(v) for v in get('watertight')]
    bodies = _finite(get('body_count_raw'))
    agg = {
        'num_shapes': n,
        'valid_rate': float(np.mean(valid)) if n else None,
        'watertight_rate': float(np.mean(watertight)) if n else None,
        'body_count_raw_mean': float(np.mean(bodies)) if bodies else None,
        'single_body_rate': float(np.mean([b == 1 for b in bodies])) if bodies else None,
    }
    for key in fields:
        agg[key] = _mean_median(get(key))
    return agg


def _fmt(value, spec='.5f'):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'n/a'
    return format(value, spec)


def print_aggregate_table(aggregate, prefixes, fields=METRIC_FIELDS):
    """One row per metric, one mean/median column pair per prefix.

    `fields` is METRIC_FIELDS alone without refinement and METRIC_FIELDS +
    HELDOUT_FIELDS with it, so the table stays short when there is no held-out
    half to compare against.
    """
    rate_fields = ('valid_rate', 'watertight_rate', 'single_body_rate', 'body_count_raw_mean')
    width = max(20, max(len(k) for k in tuple(fields) + rate_fields) + 2)
    header = f'{"metric":<{width}}' + ''.join(
        f'{p + " mean":>14}{p + " median":>14}' for p in prefixes)
    print('\n' + header)
    print('-' * len(header))
    for key in fields:
        line = f'{key:<{width}}'
        for p in prefixes:
            stat = aggregate[p][key]
            line += f'{_fmt(stat["mean"]):>14}{_fmt(stat["median"]):>14}'
        print(line)
    for key in rate_fields:
        line = f'{key:<{width}}'
        for p in prefixes:
            line += f'{_fmt(aggregate[p][key], ".4f"):>14}{"":>14}'
        print(line)
    print('-' * len(header))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return value


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ('' if v is None or (isinstance(v, float) and math.isnan(v)) else v)
                             for k, v in row.items()})


def heldout_split(num_points, eval_seed, shape_idx):
    """Seeded fit / held-out halves of one shape's stored query points.

    A coin flip per point from `default_rng([eval_seed, shape_idx])`: the two
    index arrays are disjoint, together cover every point, and are stable for a
    given (eval_seed, shape_idx) whatever else is evaluated, in whatever order.
    Returns (fit_idx, holdout_idx); either can be empty for a pathologically
    small query set, which the caller reads as "do not split this shape".
    """
    mask = np.random.default_rng([int(eval_seed), int(shape_idx)]).random(int(num_points)) < 0.5
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def sdf_metrics(vae, z_flat, points, values, device, clamp_dist, suffix=''):
    """{sdf_l1, sign_accuracy, sign_balanced_accuracy} on one set of query points."""
    pred = predict_sdf(vae, z_flat, points, device).cpu()
    target = torch.as_tensor(np.asarray(values, dtype=np.float32))
    acc, balanced, _ = sign_accuracy_stats(pred, target, SIGN_EPS)
    return {
        f'sdf_l1{suffix}': float(
            (pred - target.clamp(-clamp_dist, clamp_dist)).abs().mean().item()),
        f'sign_accuracy{suffix}': acc,
        f'sign_balanced_accuracy{suffix}': balanced,
    }


def _shape_line(prefix, metrics):
    return (f'{prefix} surf_mean={_fmt(metrics["surface_mean"])} '
            f'chamfer={_fmt(metrics["chamfer_mean"])} '
            f'sign={_fmt(metrics["sign_accuracy"], ".4f")} '
            f'bal={_fmt(metrics["sign_balanced_accuracy"], ".4f")} '
            f'bodies={metrics["body_count_raw"]} wt={int(bool(metrics["watertight"]))}')


def run_reconstruction_eval(config, config_filename='config.txt'):
    """`eval_task reconstruction`: held-out VAE reconstruction metrics (module doc)."""
    device = resolve_device(config)

    vae_path = config.get('vae_modelpath')
    if not vae_path:
        raise ValueError("evaluate mode requires 'vae_modelpath' in the config")
    dataset_dir = config.get('dataset_dir')
    if not dataset_dir:
        raise ValueError("evaluate mode requires 'dataset_dir' in the config")
    out_dir = config.get('output_dir')
    if not out_dir:
        raise ValueError("evaluate mode requires 'output_dir' in the config")

    split = str(config.get('eval_split', 'val')).lower()
    if split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got '{split}'")
    eval_num_shapes = int(config.get('eval_num_shapes', 0))
    eval_seed = int(config.get('eval_seed', 0))
    resolution = int(config.get('mc_resolution', 128))
    refine_steps = int(config.get('latent_refine_steps', 0))
    refine_lr = float(config.get('latent_refine_lr', 0.01))
    refine_prior_weight = float(config.get('latent_refine_prior_weight', 0.0))
    if eval_num_shapes < 0:
        raise ValueError('eval_num_shapes must be a nonnegative integer (0 = whole split)')
    if refine_steps < 0:
        raise ValueError('latent_refine_steps must be a nonnegative integer')

    print(f'Loading VAE checkpoint from {vae_path}')
    vae, vae_ckpt = load_vae(vae_path, device)
    for p in vae.parameters():
        p.requires_grad_(False)
    ckpt_config = dict(vae_ckpt.get('config') or {})
    num_enc = int(ckpt_config.get('num_encoder_points', 4096))
    clamp_dist = float(ckpt_config.get('clamp_dist', 0.1))

    # Architecture / point budget from the checkpoint; dataset path and split
    # keys from the run config (checkpoint values as fallback).
    ds_config = dict(ckpt_config)
    ds_config['dataset_dir'] = dataset_dir
    for key in SPLIT_KEYS:
        if key in config:
            ds_config[key] = config[key]
    split_seed = int(ds_config.get('split_seed', 42))
    datasets = dict(zip(EVAL_SPLITS, build_dataset_splits(ds_config, split_seed)))
    dataset = datasets[split]
    if not hasattr(dataset, 'deterministic'):
        print('WARNING: SDFShapeDataset has no deterministic mode; encoder subsampling '
              'will differ between runs')
    dataset.deterministic = True
    dataset.seed = eval_seed

    total = len(dataset)
    num_shapes = total if eval_num_shapes == 0 else min(eval_num_shapes, total)
    # A SUBSET is drawn at random (seeded by eval_seed), never head-sliced: with
    # split_by_parent the split index array is emitted parent by parent, so the
    # first N shapes would be one or two whole bracket families rather than a
    # cross-section of the split.
    if num_shapes == total:
        selection = list(range(total))
    else:
        selection = sorted(
            int(i) for i in np.random.default_rng(eval_seed).permutation(total)[:num_shapes])
    os.makedirs(out_dir, exist_ok=True)
    prefixes = ['enc'] + (['ref'] if refine_steps > 0 else [])
    print(f'\nEvaluating {num_shapes}/{total} {split} shapes '
          + ('(random subset, seeded by eval_seed) ' if num_shapes != total else '')
          + f'(split_seed={split_seed}, split_by_parent={ds_config.get("split_by_parent", False)}, '
          f'eval_seed={eval_seed}, num_encoder_points={num_enc}, mc_resolution={resolution}'
          + (f', latent_refine_steps={refine_steps} lr={refine_lr:g} '
             f'prior_weight={refine_prior_weight:g}' if refine_steps > 0 else '') + ')')
    if refine_steps > 0:
        print('  Each shape\'s stored query points are halved (seeded by eval_seed and the '
              'shape index):\n  the refinement fits one half, ref_* score the other, and '
              'enc_*_heldout repeat\n  the encoder on that same held-out half '
              '(enc_* themselves stay on all points).')

    rows = []
    t_start = time.time()
    try:
        with h5py.File(dataset_dir, 'r') as h5:
            shapes = h5['shapes']
            for order, i in enumerate(selection):
                item = dataset[i]
                shape_idx = int(item['shape_idx'])
                grp = shapes[f'{shape_idx:05d}']
                gt_surface = grp['surface_points'][:]
                sdf_points = grp['sdf_points'][:]
                sdf_values = grp['sdf_values'][:]
                source = grp.attrs.get('source')
                if isinstance(source, bytes):
                    source = source.decode('utf-8', 'replace')

                surface_points = item['surface_points'].unsqueeze(0).to(device)
                surface_normals = item['surface_normals'].unsqueeze(0).to(device)
                with torch.no_grad():
                    mu, _ = vae.encode(surface_points, surface_normals)
                    z_enc = mu.flatten(1)

                # With refinement on, the stored query set is halved by a
                # seeded coin flip: the refinement fits one half and every ref_
                # SDF metric is scored on the OTHER. Scoring a refined latent on
                # the very points it was optimized against measures fit, not
                # accuracy, and improves monotonically with latent_refine_steps.
                # enc_ stays on ALL points (its meaning must not depend on
                # whether refinement is on) and gains *_heldout copies on the
                # same held-out half, which is what ref_ is comparable to.
                fit_sel = hold_sel = None
                if refine_steps > 0:
                    fit_sel, hold_sel = heldout_split(len(sdf_points), eval_seed, shape_idx)
                    if len(fit_sel) == 0 or len(hold_sel) == 0:
                        print(f'    WARNING: shape {shape_idx:05d} has too few query points '
                              f'({len(sdf_points)}) to split; refining and scoring on all of '
                              'them (ref_ is in-sample for this shape)')
                        fit_sel = hold_sel = np.arange(len(sdf_points))

                row = {'index': order, 'shape_idx': shape_idx,
                       'source': str(source) if source is not None else None}
                enc, _ = evaluate_latent(
                    vae, z_enc, gt_surface, sdf_points, sdf_values, resolution, device,
                    clamp_dist=clamp_dist, surface_sample_seed=eval_seed)
                if refine_steps > 0:
                    enc.update(sdf_metrics(vae, z_enc, sdf_points[hold_sel],
                                           sdf_values[hold_sel], device, clamp_dist,
                                           suffix='_heldout'))
                row.update({f'enc_{k}': v for k, v in enc.items()})
                line = (f'  [{order + 1}/{num_shapes}] shape {shape_idx:05d}: '
                        + _shape_line('enc', enc))

                if refine_steps > 0:
                    row['refine_fit_points'] = int(len(fit_sel))
                    row['refine_heldout_points'] = int(len(hold_sel))
                    fit_points = sdf_points[fit_sel]
                    fit_values = sdf_values[fit_sel]
                    query_points = torch.from_numpy(np.asarray(fit_points, dtype=np.float32)
                                                    ).unsqueeze(0).to(device)
                    query_sdf = torch.from_numpy(np.asarray(fit_values, dtype=np.float32)
                                                 ).unsqueeze(0).to(device)
                    history = []
                    t0 = time.time()
                    z_ref = refine_latent(
                        vae, z_enc, surface_points, surface_normals, query_points, query_sdf,
                        refine_steps, refine_lr, refine_prior_weight, clamp_dist=clamp_dist,
                        history=history)
                    refine_seconds = time.time() - t0
                    ref, _ = evaluate_latent(
                        vae, z_ref, gt_surface, sdf_points[hold_sel], sdf_values[hold_sel],
                        resolution, device, clamp_dist=clamp_dist, surface_sample_seed=eval_seed)
                    # ref_ is already scored on the held-out half; repeat it under
                    # the _heldout names so the table's held-out rows carry both
                    # columns instead of an n/a next to enc_.
                    for key in ('sdf_l1', 'sign_accuracy', 'sign_balanced_accuracy'):
                        ref[f'{key}_heldout'] = ref[key]
                    ref['refine_seconds'] = float(refine_seconds)
                    ref['refine_loss_first'] = history[0][1]
                    ref['refine_loss_last'] = history[-1][1]
                    ref['latent_shift_l2'] = float((z_ref - z_enc).norm().item())
                    # The same two numbers on the half the refinement actually
                    # saw, explicitly labelled so fit is never read as accuracy.
                    fit_pred = predict_sdf(vae, z_ref, fit_points, device).cpu()
                    fit_target = torch.as_tensor(np.asarray(fit_values, dtype=np.float32))
                    ref['sdf_l1_insample'] = float(
                        (fit_pred - fit_target.clamp(-clamp_dist, clamp_dist)).abs().mean().item())
                    ref['sign_accuracy_insample'] = sign_accuracy(fit_pred, fit_target, SIGN_EPS)
                    row.update({f'ref_{k}': v for k, v in ref.items()})
                    line += ' | ' + _shape_line('ref', ref)
                rows.append(row)
                print(line)
    finally:
        for ds in datasets.values():
            ds.close()

    table_fields = METRIC_FIELDS + (HELDOUT_FIELDS if refine_steps > 0 else ())
    aggregate = {p: aggregate_rows(rows, p, table_fields) for p in prefixes}
    print_aggregate_table(aggregate, prefixes, table_fields)
    elapsed = time.time() - t_start
    print(f'Evaluated {len(rows)} shapes in {elapsed:.1f}s '
          f'({elapsed / max(len(rows), 1):.2f}s/shape)')

    summary = {
        'vae_modelpath': vae_path,
        'checkpoint_epoch': vae_ckpt.get('epoch'),
        'dataset_dir': dataset_dir,
        'split': split,
        'split_seed': split_seed,
        'split_by_parent': ds_config.get('split_by_parent', False),
        'split_sizes': {name: len(ds) for name, ds in datasets.items()},
        'eval_seed': eval_seed,
        'eval_num_shapes': eval_num_shapes,
        'num_shapes_evaluated': len(rows),
        'num_encoder_points': num_enc,
        'mc_resolution': resolution,
        'clamp_dist': clamp_dist,
        'sign_eps': SIGN_EPS,
        'surface_distance_backend': _distance_backend(),
        'pred_surface_samples': PRED_SURFACE_SAMPLES,
        'shape_selection': ('all' if num_shapes == total
                            else 'random subset seeded by eval_seed'),
        # With refinement on, the stored query points were halved by a seeded
        # coin flip: the fit half fed the refinement, the held-out half scored
        # every ref_ SDF metric and the enc_*_heldout copies. enc_ itself is
        # still all points; ref_*_insample are the fit-half numbers, which are
        # fit and not accuracy.
        'refine_query_split': ('fit/holdout halves seeded by (eval_seed, shape_idx); '
                               'ref_* and enc_*_heldout are the holdout half, '
                               'enc_* are all points, ref_*_insample the fit half'
                               if refine_steps > 0 else None),
        'latent_refine': ({'steps': refine_steps, 'lr': refine_lr,
                           'prior_weight': refine_prior_weight} if refine_steps > 0 else None),
        'seconds': elapsed,
        'aggregate': aggregate,
        'shapes': rows,
    }
    json_path = os.path.join(out_dir, f'eval_{split}.json')
    with open(json_path, 'w') as f:
        json.dump(_jsonable(summary), f, indent=2)
    csv_path = os.path.join(out_dir, f'eval_{split}.csv')
    _write_csv(csv_path, rows)
    print(f'JSON: {json_path}')
    print(f'CSV:  {csv_path}')
    return summary


# ---------------------------------------------------------------------------
# Shared helpers for the FM-based tasks (descriptor_calibration, conditional)
# ---------------------------------------------------------------------------

def select_shapes(total, num_shapes, eval_seed, allowed=None):
    """Indices into a split: all of them when `num_shapes` is 0 or covers the
    split, else a RANDOM subset of that size seeded by `eval_seed` (sorted) --
    never a head slice, which under split_by_parent would be one or two whole
    bracket families.

    `allowed`, when given, is the list of split positions still in play after
    `eval_exclude_shapes`; the subset is drawn from it, so an excluded shape
    can never enter the sample and `eval_num_shapes` still means what it says.
    """
    total = int(total)
    num_shapes = int(num_shapes)
    if num_shapes < 0:
        raise ValueError('the shape count must be a nonnegative integer (0 = whole split)')
    pool = list(range(total)) if allowed is None else [int(i) for i in allowed]
    if num_shapes == 0 or num_shapes >= len(pool):
        return sorted(pool)
    picked = np.random.default_rng(int(eval_seed)).permutation(len(pool))[:num_shapes]
    return sorted(int(pool[i]) for i in picked)


def excluded_shape_ids(config):
    """`eval_exclude_shapes` as a set of HDF5 shape indices (empty when absent).

    DeepJEB h5 index 2099 (`131_561`) is a partial STL carrying full-bracket
    labels: no generator can hit its conditions, and at n = 32 one such shape
    dominates the p95 column of every method indistinguishably from a genuine
    tail failure. That exclusion is methodology, so it belongs in the config and
    in the printed banner, not in a prose comment nobody can execute.
    """
    raw = config.get('eval_exclude_shapes')
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return set()
    items = raw if isinstance(raw, (list, tuple)) else str(raw).split(',')
    out = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        try:
            out.add(int(text))
        except ValueError as exc:
            raise ValueError('eval_exclude_shapes must be a comma-separated list of HDF5 '
                             f'shape indices; got {item!r}') from exc
    return out


def allowed_positions(dataset, excluded, split):
    """Split positions whose HDF5 shape index is not excluded, plus the dropped ids."""
    allowed, dropped = [], []
    for i in range(len(dataset)):
        (dropped if int(dataset.indices[i]) in excluded else allowed).append(i)
    if not allowed:
        raise ValueError(f'eval_exclude_shapes removed every shape of the {split} split')
    return allowed, [int(dataset.indices[i]) for i in dropped]


def _require(config, key, task):
    value = config.get(key)
    if value is None or str(value).strip() == '':
        raise ValueError(f"eval_task {task} requires '{key}' in the config")
    return value


def _eval_split(config):
    split = str(config.get('eval_split', 'val')).lower()
    if split not in EVAL_SPLITS:
        raise ValueError(f"eval_split must be one of {EVAL_SPLITS}, got '{split}'")
    return split


def _split_for_eval(config, ckpt_config, dataset_dir):
    """Build the three split datasets with the reconstruction task's precedence:
    SPLIT_KEYS present in the run config win, else the checkpoint's values.
    Returns (datasets dict, ds_config, split_seed)."""
    ds_config = dict(ckpt_config or {})
    ds_config['dataset_dir'] = dataset_dir
    for key in SPLIT_KEYS:
        if key in config:
            ds_config[key] = config[key]
    split_seed = int(ds_config.get('split_seed', 42))
    datasets = dict(zip(EVAL_SPLITS, build_dataset_splits(ds_config, split_seed)))
    return datasets, ds_config, split_seed


def true_conditions(dataset, shape_idx, cond_names):
    """The STORED condition vector of `shape_idx` restricted to `cond_names`
    (FM checkpoint order), float64. Raises when the dataset lacks a name --
    an FEA-conditioned FM against a dataset without the `cond_extra` sidecar."""
    stored = np.asarray(dataset.get_cond(int(shape_idx)), dtype=np.float64)
    available = [str(n) for n in dataset.cond_names]
    missing = [n for n in cond_names if n not in available]
    if missing:
        raise ValueError(
            f'the FM checkpoint conditions on {missing}, which this dataset does not carry '
            f'(it has {available}). For FEA names run add_fea_conditions.py on the dataset first.')
    return np.asarray([stored[available.index(n)] for n in cond_names], dtype=np.float64)


def normalize_true_conditions(raw, fm_ckpt, config=None, cond_names=(), label='target'):
    """`(raw - cond_mean) / cond_std` clamped to +/- `cond_clip`, exactly as
    train_fm normalizes the training conditions. Returns (float32 tensor
    [cond_dim], list of bool: which entries the clamp touched, list of the
    condition names that exceeded `max_condition_z`).

    With `config`, the same `max_condition_z` / `condition_ood_policy` guard
    `sample.normalize_condition_request` applies runs FIRST, so those two keys
    are not decorative here. Without it a target beyond the checkpoint's
    `cond_clip` is silently clamped for SAMPLING while the relative error is
    scored against the UNCLAMPED value: a permanent, method-independent error
    floor with no visible signal. The default policy for a bulk scoring task is
    `warn` (`sample` defaults to `error`) -- one unreachable shape should be
    shouted about, not abort a 32-shape benchmark; write `condition_ood_policy
    error` to make it abort.
    """
    mean = fm_ckpt['cond_mean'].squeeze(0).cpu().double()
    std = fm_ckpt['cond_std'].squeeze(0).cpu().double()
    clip = float(fm_ckpt.get('cond_clip') or 5.0)
    cond_n = (torch.as_tensor(np.asarray(raw, dtype=np.float64)) - mean) / std
    exceeded = []
    if config is not None:
        max_z = float(config.get('max_condition_z', clip))
        policy = str(config.get('condition_ood_policy', 'warn')).lower()
        if policy not in ('error', 'warn', 'clamp'):
            raise ValueError('condition_ood_policy must be error, warn, or clamp')
        names = list(cond_names) or [str(i) for i in range(len(cond_n))]
        over = (cond_n.abs() > max_z).tolist()
        exceeded = [names[i] for i, flag in enumerate(over) if flag]
        if exceeded:
            details = ', '.join(f'{names[i]}={float(cond_n[i]):.2f} sigma'
                                for i, flag in enumerate(over) if flag)
            message = (f'{label}: true conditions exceed max_condition_z={max_z:g}: {details}. '
                       'The FM can only be asked for the clamped value while the score is taken '
                       'against the unclamped one, so this shape carries a fixed error floor for '
                       'every method.')
            if policy == 'error':
                raise ValueError(message + ' Set condition_ood_policy warn|clamp, raise '
                                 'max_condition_z, or drop the shape with eval_exclude_shapes.')
            print(f'WARNING: {message}')
            if policy == 'clamp':
                cond_n = cond_n.clamp(-max_z, max_z)
    clipped = (cond_n.abs() > clip).tolist()
    return cond_n.clamp(-clip, clip).float(), clipped, exceeded


def shape_generator(eval_seed, shape_idx, device):
    """The per-shape `torch.Generator` every method draws its base noise from,
    so the methods are PAIRED on z0 (seeded by eval_seed and the shape index)."""
    seed = (int(eval_seed) * 1_000_003 + int(shape_idx)) % (2 ** 62)
    return torch.Generator(device=device).manual_seed(seed)


# ---------------------------------------------------------------------------
# eval_task descriptor_calibration
# ---------------------------------------------------------------------------

def run_descriptor_calibration(config, config_filename='config.txt'):
    """Fit and save the soft-descriptor calibration (module doc)."""
    task = 'descriptor_calibration'
    device = resolve_device(config)
    dataset_dir = _require(config, 'dataset_dir', task)
    out_dir = _require(config, 'output_dir', task)
    calibration_path = _require(config, 'descriptor_calibration_path', task)
    split = _eval_split(config)
    eval_seed = int(config.get('eval_seed', 0))
    num_shapes_requested = int(config.get('calibration_num_shapes', DEFAULT_CALIBRATION_NUM_SHAPES))
    per_shape = int(config.get('calibration_samples_per_shape',
                               DEFAULT_CALIBRATION_SAMPLES_PER_SHAPE))
    if num_shapes_requested < 1:
        raise ValueError('calibration_num_shapes must be a positive integer')
    if per_shape < 1:
        raise ValueError('calibration_samples_per_shape must be a positive integer')
    soft_resolution = int(config.get('soft_descriptor_resolution', 48))
    soft_tau = float(config.get('soft_descriptor_tau', 0.032))
    if soft_tau <= 0:
        raise ValueError('soft_descriptor_tau must be > 0')
    # The "true" measurement is the export path at mc_resolution -- the same
    # grid `mode sample`'s geometric audit scores on, so the calibrated slope
    # maps the proxy onto the number the user is actually shown. E2 re-measures
    # in its own loop at newton_measure_resolution; that is a different (and
    # usually coarser) grid, and Marching Cubes volume moves slightly with
    # resolution, so a mismatch is worth naming rather than leaving implicit.
    # `check_compatible` cannot catch it: it pins `resolution` and `tau`, not
    # the measurement grid.
    measure_resolution = int(config.get('mc_resolution', 128))
    newton_measure = newton_measure_resolution(config)
    min_r2 = float(config.get('calibration_min_r2', 0.5))
    ode_steps = int(config.get('ode_steps', 50))
    cfg_scale = float(config.get('cfg_scale', 1.0))

    tools = import_descriptor_tools()
    stack = load_fm_stack(_require(config, 'fm_modelpath', task), config.get('vae_modelpath'), device)
    datasets, ds_config, split_seed = _split_for_eval(config, stack.fm_ckpt.get('config'), dataset_dir)
    dataset = datasets[split]
    excluded = excluded_shape_ids(config)
    try:
        allowed, dropped = allowed_positions(dataset, excluded, split)
        selection = select_shapes(len(dataset), num_shapes_requested, eval_seed, allowed=allowed)
        shape_indices = [int(dataset.indices[i]) for i in selection]
        generator = torch.Generator(device=device).manual_seed(eval_seed)
        cond_batches, noise_batches = [], []
        clipped_entries = 0
        ood_shapes = []
        for shape_idx in shape_indices:
            noise_batches.append(torch.randn(per_shape, stack.latent_flat_dim, device=device,
                                             generator=generator))
            if stack.cond_dim > 0:
                raw = true_conditions(dataset, shape_idx, stack.cond_names)
                cond_n, clipped, exceeded = normalize_true_conditions(
                    raw, stack.fm_ckpt, config, stack.cond_names,
                    label=f'shape {shape_idx:05d}')
                clipped_entries += int(sum(clipped))
                if exceeded:
                    ood_shapes.append({'shape_idx': shape_idx, 'conditions': exceeded})
                cond_batches.append(cond_n.unsqueeze(0).repeat(per_shape, 1).to(device))
            else:
                cond_batches.append(None)
    finally:
        for ds in datasets.values():
            ds.close()

    names = tuple(tools.proxy.SUPPORTED_SOFT_NAMES)
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nDescriptor calibration on {len(shape_indices)} {split} shape(s) x {per_shape} '
          f'sample(s) = {len(shape_indices) * per_shape} rows (split_seed={split_seed}, '
          f'split_by_parent={ds_config.get("split_by_parent", False)}, eval_seed={eval_seed}, '
          f'soft resolution={soft_resolution}, tau={soft_tau:g}, measure at mc_resolution='
          f'{measure_resolution}, ode_steps={ode_steps}, cfg_scale={cfg_scale:g}'
          + (f'; {clipped_entries} condition entr(y/ies) clipped to +/-cond_clip'
             if clipped_entries else '') + f', min_r2={min_r2:g})')
    if dropped:
        print(f'  eval_exclude_shapes dropped h5 shape(s) {dropped} from the {split} pool')
    if ood_shapes:
        print(f'  WARNING: {len(ood_shapes)} shape(s) exceed max_condition_z: {ood_shapes}')
    if stack.cond_dim == 0:
        print('NOTE: the FM is unconditional; the calibration set is unconditional samples')
    if newton_measure != measure_resolution:
        print(f'NOTE: this calibration is fitted against the export path at mc_resolution '
              f'{measure_resolution}, while E2 would re-measure at newton_measure_resolution '
              f'{newton_measure}. `check_compatible` now PINS the measurement grid, so a sample '
              'or conditional run at that setting will refuse this artifact; leave '
              'newton_measure_resolution unset so it follows mc_resolution.')
    t0 = time.time()
    calibration = tools.calibration.calibrate(
        stack.vae, stack.model, stack.latent_mean, stack.latent_std,
        cond_batches=cond_batches, noise_batches=noise_batches, names=names,
        resolution=soft_resolution, tau=soft_tau, measure_resolution=measure_resolution,
        ode_steps=ode_steps, cfg_scale=cfg_scale, device=device,
        vae_path=stack.vae_path, fm_path=stack.fm_path, cond_names=stack.cond_names,
        split=split, num_shapes=len(shape_indices), samples_per_shape=per_shape,
        min_r2=min_r2,
        extra={'eval_task': task, 'eval_seed': eval_seed, 'dataset_dir': dataset_dir,
               'excluded_shape_indices': sorted(excluded), 'dropped_from_pool': dropped,
               'ood_shapes': ood_shapes, 'condition_ood_policy':
               str(config.get('condition_ood_policy', 'warn')).lower(),
               'split_seed': split_seed,
               'split_by_parent': bool(ds_config.get('split_by_parent', False)),
               'shape_indices': shape_indices, 'clipped_condition_entries': clipped_entries,
               'measure_resolution_key': 'mc_resolution', 'config_filename': config_filename})
    calibration.save(calibration_path)
    elapsed = time.time() - t0
    valid_rows = sum(1 for r in calibration.rows if r.get('valid'))
    summary = {
        'eval_task': task,
        'descriptor_calibration_path': calibration_path,
        'fm_modelpath': stack.fm_path,
        'vae_modelpath': stack.vae_path,
        'fm_sha256': calibration.fm_sha256,
        'vae_sha256': calibration.vae_sha256,
        'dataset_dir': dataset_dir,
        'split': split,
        'split_seed': split_seed,
        'split_by_parent': bool(ds_config.get('split_by_parent', False)),
        'eval_seed': eval_seed,
        'cond_names': stack.cond_names,
        'calibration_num_shapes': len(shape_indices),
        'calibration_samples_per_shape': per_shape,
        'shape_indices': shape_indices,
        'rows_total': len(calibration.rows),
        'rows_valid': valid_rows,
        'soft_descriptor_resolution': soft_resolution,
        'soft_descriptor_tau': soft_tau,
        'measure_resolution': measure_resolution,
        'newton_measure_resolution': newton_measure,
        'calibration_min_r2': min_r2,
        'watertight_rate': calibration.extra.get('watertight_rate'),
        'excluded_shape_indices': sorted(excluded),
        'ood_shapes': ood_shapes,
        'ode_steps': ode_steps,
        'cfg_scale': cfg_scale,
        'coefficients': calibration.coefficients,
        'seconds': elapsed,
    }
    json_path = os.path.join(out_dir, 'eval_descriptor_calibration.json')
    with open(json_path, 'w') as f:
        json.dump(_jsonable(summary), f, indent=2)
    print(f'Calibration: {calibration_path} ({valid_rows}/{len(calibration.rows)} valid rows, '
          f'{elapsed:.1f}s)')
    print(f'JSON: {json_path}')
    return summary


# ---------------------------------------------------------------------------
# eval_task conditional
# ---------------------------------------------------------------------------

def parse_eval_methods(value):
    """Config `eval_methods` (None / scalar / list) -> ordered unique method list."""
    if value is None:
        methods = list(DEFAULT_EVAL_METHODS)
    elif isinstance(value, (list, tuple)):
        methods = [str(v).strip().lower() for v in value if str(v).strip()]
    else:
        methods = [p.strip().lower() for p in str(value).split(',') if p.strip()]
    unknown = [m for m in methods if m not in EVAL_METHODS]
    if unknown:
        raise ValueError(f'eval_methods contains unknown method(s) {unknown}; allowed: '
                         f'{list(EVAL_METHODS)}')
    if not methods:
        raise ValueError('eval_methods must name at least one method')
    ordered = []
    for m in methods:
        if m not in ordered:
            ordered.append(m)
    return ordered


def _decode_and_audit(vae, z_vae_row, resolution, device, cond_names, target_stored, cond_std):
    """Decode one VAE-space latent [1, D] -> (mesh-or-None, audited report)."""
    volume = decode_sdf_grid(vae, z_vae_row, resolution=resolution, device=device)
    mesh = sdf_grid_to_mesh(volume)
    report = mesh_report(mesh)
    report.setdefault('body_count_raw', None)
    _audit_report(report, cond_names, target_stored, cond_std, None)
    return mesh, report


def _rms_drift(z_n, z_ref_n):
    return float((z_n.float() - z_ref_n.float()).pow(2).mean().sqrt().item())


def _pick_best_candidate(reports):
    """Index of the candidate with the lowest condition_score (None last; ties
    and an all-None field keep candidate 0, i.e. the plain sample)."""
    best, best_score = 0, None
    for i, report in enumerate(reports):
        score = report.get('condition_score')
        if score is None:
            continue
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def _percentile(values, q):
    vals = _finite(values)
    return float(np.percentile(vals, q)) if vals else None


def paired_shape_ids(rows, methods, name):
    """Shape ids where EVERY reported method produced a finite `rel_error_<name>`.

    Per-method medians are otherwise taken over method-dependent subsets: a
    method that breaks the meshes it cannot correct drops those rows from its
    own percentile and looks better for it. The paired-z0 protocol makes the
    common subset the natural statistic.
    """
    per_method = []
    for method in methods:
        ok = {r['shape_idx'] for r in rows
              if r['method'] == method and r.get(f'rel_error_{name}') is not None
              and np.isfinite(float(r[f'rel_error_{name}']))}
        per_method.append(ok)
    return set.intersection(*per_method) if per_method else set()


def aggregate_conditional(rows, methods, cond_names):
    """Per method: valid / watertight rates, drift / NFE / time statistics and,
    per condition name, median / p95 / mean relative error over the rows where
    that name could be measured -- plus `n` (how many rows that was) and
    `median_rel_error_paired`, the median over the shapes where every reported
    method could be measured. Read the paired column when comparing methods:
    the unpaired one is a method-selected subsample."""
    paired = {name: paired_shape_ids(rows, methods, name) for name in cond_names}
    out = {}
    for method in methods:
        mrows = [r for r in rows if r['method'] == method]
        n = len(mrows)
        bodies = _finite([r.get('body_count_raw') for r in mrows])
        agg = {
            'num_shapes': n,
            'valid_rate': float(np.mean([bool(r['valid']) for r in mrows])) if n else None,
            'watertight_rate': float(np.mean([bool(r['watertight']) for r in mrows])) if n else None,
            'single_body_rate': float(np.mean([b == 1 for b in bodies])) if bodies else None,
            'body_count_raw_mean': float(np.mean(bodies)) if bodies else None,
            'latent_rms_drift': _mean_median([r.get('latent_rms_drift') for r in mrows]),
            'nfe_per_output': _mean_median([r.get('nfe_per_output') for r in mrows]),
            'velocity_net_calls': _mean_median([r.get('velocity_net_calls') for r in mrows]),
            'seconds': dict(_mean_median([r.get('seconds') for r in mrows]),
                            total=float(sum(_finite([r.get('seconds') for r in mrows])))),
            'per_condition': {},
        }
        for name in cond_names:
            errors = [r.get(f'rel_error_{name}') for r in mrows]
            common = paired[name]
            paired_errors = [r.get(f'rel_error_{name}') for r in mrows
                             if r['shape_idx'] in common]
            agg['per_condition'][name] = {
                'median_rel_error': _mean_median(errors)['median'],
                'p95_rel_error': _percentile(errors, 95),
                'mean_rel_error': _mean_median(errors)['mean'],
                'n': len(_finite(errors)),
                'coverage': (len(_finite(errors)) / n) if n else None,
                'median_rel_error_paired': _mean_median(paired_errors)['median'],
                'p95_rel_error_paired': _percentile(paired_errors, 95),
                'paired_n': len(common),
                'measured_by': next((r.get(f'measured_by_{name}') for r in mrows
                                     if r.get(f'measured_by_{name}')), None),
            }
        out[method] = agg
    return out


def _pct(value):
    return 'n/a' if value is None else f'{100.0 * value:.2f}%'


def print_conditional_table(aggregate, methods, cond_names):
    """One row per method; per condition name a median / p95 relative-error
    column pair, then valid / watertight rates, drift, NFE and seconds."""
    # The header cell is `name + " med"` / `name + " p95"`, four characters
    # longer than the name itself; size on that (plus a two-space gutter) or a
    # long FEA name such as log_max_ver_stress_mpa overflows its column and the
    # header runs together while the data rows stay aligned to the shorter width.
    name_w = max(12, max((len(n) for n in cond_names), default=0) + 6)
    header = f'{"method":<10}'
    for name in cond_names:
        header += f'{name + " med":>{name_w}}{name + " p95":>{name_w}}{"n":>5}'
    header += f'{"valid":>8}{"wt":>8}{"drift":>9}{"NFE/out":>9}{"s/shape":>9}'
    print('\n' + header)
    print('-' * len(header))
    for method in methods:
        agg = aggregate[method]
        line = f'{method:<10}'
        for name in cond_names:
            pc = agg['per_condition'][name]
            line += (f'{_pct(pc["median_rel_error"]):>{name_w}}'
                     f'{_pct(pc["p95_rel_error"]):>{name_w}}{pc["n"]:>5}')
        line += (f'{_fmt(agg["valid_rate"], ".3f"):>8}{_fmt(agg["watertight_rate"], ".3f"):>8}'
                 f'{_fmt(agg["latent_rms_drift"]["median"], ".4f"):>9}'
                 f'{_fmt(agg["nfe_per_output"]["mean"], ".1f"):>9}'
                 f'{_fmt(agg["seconds"]["mean"], ".2f"):>9}')
        print(line)
    print('-' * len(header))
    print('`n` is how many shapes that name could be measured on FOR THAT METHOD -- a median '
          'over a method-selected subset. The paired block below uses only the shapes every '
          'method could be measured on, which is the comparable statistic:')
    paired_names = [n for n in cond_names
                    if any(aggregate[m]['per_condition'][n]['paired_n'] for m in methods)]
    if not paired_names:
        print('  (no condition name is measurable for every method)')
        return
    pheader = f'{"method":<10}' + ''.join(
        f'{n + " med(p)":>{name_w}}{n + " p95(p)":>{name_w}}{"n":>5}' for n in paired_names)
    print(pheader)
    print('-' * len(pheader))
    for method in methods:
        agg = aggregate[method]
        line = f'{method:<10}'
        for name in paired_names:
            pc = agg['per_condition'][name]
            line += (f'{_pct(pc["median_rel_error_paired"]):>{name_w}}'
                     f'{_pct(pc["p95_rel_error_paired"]):>{name_w}}{pc["paired_n"]:>5}')
        print(line)
    print('-' * len(pheader))


def run_conditional_eval(config, config_filename='config.txt'):
    """`eval_task conditional`: paired conditional-generation benchmark (module doc)."""
    task = 'conditional'
    device = resolve_device(config)
    dataset_dir = _require(config, 'dataset_dir', task)
    out_dir = _require(config, 'output_dir', task)
    split = _eval_split(config)
    eval_num_shapes = int(config.get('eval_num_shapes', 0))
    eval_seed = int(config.get('eval_seed', 0))
    resolution = int(config.get('mc_resolution', 128))
    ode_steps = int(config.get('ode_steps', 50))
    cfg_scale = float(config.get('cfg_scale', 1.0))
    candidate_multiplier = int(config.get('candidate_multiplier', 4))
    latent_clip = float(config.get('latent_clip', 0.0))
    soft_resolution = int(config.get('soft_descriptor_resolution', 48))
    soft_tau = float(config.get('soft_descriptor_tau', 0.032))
    methods = parse_eval_methods(config.get('eval_methods'))
    if eval_num_shapes < 0:
        raise ValueError('eval_num_shapes must be a nonnegative integer (0 = whole split)')
    if candidate_multiplier < 1:
        raise ValueError('candidate_multiplier must be at least 1')
    if cfg_scale < 0:
        raise ValueError('cfg_scale must be a nonnegative number')
    if 'plain' not in methods:
        # Every other method is reported RELATIVE to the plain sample (drift,
        # paired z0), so it is always computed; report it too.
        print("NOTE: 'plain' is the pairing reference for every other method and is "
              'always evaluated; adding it to the reported methods.')
        methods = ['plain'] + methods
    needs_tools = any(m in CALIBRATED_METHODS for m in methods)
    newton_config = dict(config)
    if any(m in ('e2', 'c2e2') for m in methods):
        newton_config.setdefault('newton_rounds', 3)
        if int(newton_config['newton_rounds']) <= 0:
            raise ValueError('eval_methods include e2/c2e2 but newton_rounds is 0; set it to a '
                             'positive round count (the pilot used 3)')

    stack = load_fm_stack(_require(config, 'fm_modelpath', task), config.get('vae_modelpath'), device)
    if stack.cond_dim == 0:
        raise ValueError('eval_task conditional needs a CONDITIONAL FM checkpoint (cond_dim 0 here)')
    vae, model, fm_ckpt = stack.vae, stack.model, stack.fm_ckpt
    cond_names = stack.cond_names
    cond_std_np = fm_ckpt['cond_std'].squeeze(0).cpu().numpy().astype(np.float64)
    fea_names = [n for n in cond_names if CN.is_fea(n)]
    geometric_names = [n for n in cond_names if CN.is_geometric(n)]
    # What C2/E2 optimise, resolved once from names alone (no shape needed).
    requested_targets = _as_name_list(config.get('guidance_targets'), DEFAULT_GUIDANCE_TARGETS)
    proxy_target_names = [n for n in requested_targets
                          if n in cond_names and CN.is_geometric(n) and n in soft_proxy_names()]

    tools = calibration = None
    guidance_info = newton_info = None
    if needs_tools:
        tools = import_descriptor_tools()
        calibration = load_calibration(tools, config.get('descriptor_calibration_path'),
                                       stack.vae_path, stack.fm_path, soft_resolution, soft_tau,
                                       measure_resolution=(newton_measure_resolution(config)
                                                           if any(m in ('e2', 'c2e2')
                                                                  for m in methods)
                                                           else resolution))
        # The guard has to be the set C2/E2 can actually ACT on, not merely the
        # geometric names: bbox extents are geometric and have no soft proxy, so
        # a checkpoint conditioned on bbox_x + FEA names passes a
        # `geometric_names` test, then `apply_newton` short-circuits on an empty
        # target dict and the `e2` row silently records the PLAIN latent, mesh
        # and report as an E2 measurement (`c2` crashes instead).
        if not proxy_target_names:
            raise ValueError(
                f'eval_methods {methods} need a condition the soft SDF proxy can measure '
                f'({sorted(soft_proxy_names())}) among the checkpoint cond_names {cond_names} '
                f'and guidance_targets {requested_targets}; C2/E2 act through that proxy, which '
                'cannot measure an FEA quantity and has no bbox form. Drop c2/e2/c2e2 from '
                'eval_methods, or point guidance_targets at volume/area.')
        if 'c2' in methods or 'c2e2' in methods:
            guidance_info = {
                'eta': float(config.get('guidance_eta', 0.1)),
                't_start': float(config.get('guidance_t_start', 0.3)),
                'step_mode': str(config.get('guidance_step_mode', 'velocity_dt')).lower(),
                'soft_descriptor_resolution': soft_resolution,
                'soft_descriptor_tau': soft_tau,
            }
        if 'e2' in methods or 'c2e2' in methods:
            newton_info = {
                'rounds': int(newton_config['newton_rounds']),
                'step_cap_rms': float(config.get('newton_step_cap_rms', 0.12)),
                'line_search_tries': int(config.get('newton_line_search_tries', 3)),
                'measure_resolution': newton_measure_resolution(config),
                'latent_clip': latent_clip,
            }

    # ---- Split, shape selection, and every shape's TRUE conditions ----
    datasets, ds_config, split_seed = _split_for_eval(config, fm_ckpt.get('config'), dataset_dir)
    dataset = datasets[split]
    excluded = excluded_shape_ids(config)
    ood_shapes = []
    try:
        total = len(dataset)
        allowed, dropped = allowed_positions(dataset, excluded, split)
        selection = select_shapes(total, eval_num_shapes, eval_seed, allowed=allowed)
        shapes = []
        with h5py.File(dataset_dir, 'r') as h5:
            for order, i in enumerate(selection):
                shape_idx = int(dataset.indices[i])
                source = h5['shapes'][f'{shape_idx:05d}'].attrs.get('source')
                if isinstance(source, bytes):
                    source = source.decode('utf-8', 'replace')
                target_stored = true_conditions(dataset, shape_idx, cond_names)
                cond_n, clipped, exceeded = normalize_true_conditions(
                    target_stored, fm_ckpt, config, cond_names,
                    label=f'shape {shape_idx:05d}')
                if exceeded:
                    ood_shapes.append({'shape_idx': shape_idx, 'conditions': exceeded})
                shapes.append({'order': order, 'shape_idx': shape_idx,
                               'source': str(source) if source is not None else None,
                               'target_stored': target_stored, 'cond_n': cond_n,
                               'clipped': clipped})
        split_sizes = {name: len(ds) for name, ds in datasets.items()}
    finally:
        for ds in datasets.values():
            ds.close()

    audit_used, audit_reason = resolve_condition_audit(config, cond_names)
    os.makedirs(out_dir, exist_ok=True)
    print(f'\nConditional evaluation: {len(shapes)}/{total} {split} shapes '
          + ('(random subset, seeded by eval_seed) ' if len(shapes) != total else '')
          + f'(split_seed={split_seed}, split_by_parent={ds_config.get("split_by_parent", False)}, '
          f'eval_seed={eval_seed}, methods={methods}, cond_names={cond_names}, '
          f'cond_dropout_mode={str((fm_ckpt.get("config") or {}).get("cond_dropout_mode", "all"))}, '
          f'candidate_multiplier={candidate_multiplier}, cfg_scale={cfg_scale:g}, '
          f'ode_steps={ode_steps}, mc_resolution={resolution}, condition_audit={audit_used})')
    clipped_total = int(sum(sum(shape['clipped']) for shape in shapes))
    if dropped:
        print(f'  eval_exclude_shapes dropped h5 shape(s) {dropped} from the {split} pool '
              '(their labels do not describe their geometry, so no generator can hit them).')
    if clipped_total:
        print(f'  WARNING: {clipped_total} target entr(y/ies) sit beyond the checkpoint '
              "cond_clip and are CLAMPED for sampling while the score uses the unclamped "
              'value; those shapes carry a fixed error floor for every method '
              '(see clipped_condition_entries / ood_shapes in the JSON).')
    if fea_names and audit_used == 'geometric':
        print(f'  FEA-named conditions {fea_names} are not measurable geometrically; set '
              'condition_audit fea|surrogate to score them.')
    scored_names = [n for n in cond_names
                    if CN.is_geometric(n) or audit_used != 'geometric']
    if needs_tools:
        untouched = [n for n in scored_names if n not in proxy_target_names]
        if untouched:
            print(f'  SCOPE: c2/e2/c2e2 optimise {proxy_target_names} only, but {scored_names} '
                  f'are scored -- {untouched} are scored and NOT optimised. A latent step '
                  'chosen by the volume/area proxy displaces those too -- on DeepJEB '
                  'corr(volume, log_max_ver_stress) = -0.76 and corr(volume, log_first_mode) = '
                  '+0.82 -- so a win on the optimised names is not a win on the objective.')
        else:
            print(f'  SCOPE: c2/e2/c2e2 optimise {proxy_target_names}, which is every name '
                  f'scored here ({scored_names}); no scored name is left unoptimised.')
    if 'rejection' in methods and candidate_multiplier == 1:
        print('  NOTE: candidate_multiplier 1 makes rejection identical to plain.')
    print('  Every method starts from the same base noise per shape (seeded by eval_seed and '
          'the shape index); rejection candidate 0 IS that noise row.')

    rows, details = [], []
    meshes_by_method = {m: [] for m in methods}
    reports_by_method = {m: [] for m in methods}
    t_start = time.time()
    for shape in shapes:
        shape_idx, order = shape['shape_idx'], shape['order']
        target_stored = shape['target_stored']
        target_raw = raw_from_stored_targets(cond_names, target_stored)
        cond_row = shape['cond_n'].unsqueeze(0).to(device)
        generator = shape_generator(eval_seed, shape_idx, device)
        noise_all = torch.randn(candidate_multiplier, stack.latent_flat_dim, device=device,
                                generator=generator)
        z0 = noise_all[0:1]
        targets = descriptor_targets(cond_names, target_stored, None, config.get('guidance_targets'),
                                     quiet=(order > 0)) if needs_tools else {}
        results = {}

        def clip(z_n):
            return z_n.clamp(-latent_clip, latent_clip) if latent_clip > 0 else z_n

        def record(method, z_n, mesh, report, seconds, calls, rows_processed, outputs=1,
                   extra=None, detail=None):
            row = {
                'method': method, 'index': order, 'shape_idx': shape_idx, 'source': shape['source'],
                'valid': bool(report['valid']), 'watertight': bool(report.get('watertight', False)),
                'body_count_raw': report.get('body_count_raw'), 'faces': report.get('faces'),
                'condition_score': report.get('condition_score'),
                'latent_rms_drift': _rms_drift(z_n, results['plain'][0]),
                'latent_rms': float(z_n.float().pow(2).mean().sqrt().item()),
                'velocity_net_calls': int(calls),
                'nfe_per_output': float(rows_processed) / max(int(outputs), 1),
                'seconds': float(seconds),
            }
            rel = report.get('condition_rel_error') or {}
            actual = report.get('actual_conditions') or {}
            for name in cond_names:
                row[f'target_{name}'] = target_raw[name]
                row[f'actual_{name}'] = actual.get(name)
                row[f'rel_error_{name}'] = rel.get(name)
                row[f'measured_by_{name}'] = ('geometric' if name in rel else
                                             (None if CN.is_geometric(name) else
                                              'not measurable geometrically'))
            if extra:
                row.update(extra)
            rows.append(row)
            if detail:
                details.append(dict(detail, method=method, shape_idx=shape_idx))
            meshes_by_method[method].append(mesh)
            reports_by_method[method].append((row, report))

        # ---- plain ----
        counter = VelocityCallCounter(model)
        t0 = time.time()
        try:
            z_plain = clip(sample_latents(model, 1, stack.latent_flat_dim, device, cond=cond_row,
                                          cfg_scale=cfg_scale, ode_steps=ode_steps, noise=z0)).detach()
        finally:
            counter.remove()
        mesh, report = _decode_and_audit(vae, z_plain * stack.latent_std + stack.latent_mean,
                                         resolution, device, cond_names, target_stored, cond_std_np)
        results['plain'] = (z_plain, mesh, report)
        plain_calls, plain_rows = counter.calls, counter.rows
        record('plain', z_plain, mesh, report, time.time() - t0, plain_calls, plain_rows)

        # ---- rejection: best of candidate_multiplier by the geometric audit ----
        if 'rejection' in methods:
            counter = VelocityCallCounter(model)
            t0 = time.time()
            try:
                cond_batch = cond_row.repeat(candidate_multiplier, 1)
                z_cands = clip(sample_latents(model, candidate_multiplier, stack.latent_flat_dim,
                                              device, cond=cond_batch, cfg_scale=cfg_scale,
                                              ode_steps=ode_steps, noise=noise_all)).detach()
            finally:
                counter.remove()
            cand_reports, cand_meshes = [], []
            for c in range(candidate_multiplier):
                m_c, r_c = _decode_and_audit(
                    vae, z_cands[c:c + 1] * stack.latent_std + stack.latent_mean, resolution,
                    device, cond_names, target_stored, cond_std_np)
                cand_meshes.append(m_c)
                cand_reports.append(r_c)
            best = _pick_best_candidate(cand_reports)
            record('rejection', z_cands[best:best + 1], cand_meshes[best], cand_reports[best],
                   time.time() - t0, counter.calls, counter.rows, outputs=1,
                   extra={'candidate_multiplier': candidate_multiplier, 'selected_candidate': best,
                          'candidate_scores': json.dumps(_jsonable(
                              [r.get('condition_score') for r in cand_reports]))})

        # ---- c2: calibrated endpoint-prediction guidance ----
        z_c2 = None
        c2_calls = c2_rows = None
        if 'c2' in methods or 'c2e2' in methods:
            guidance_fn = make_guidance(tools, stack, cond_row, None, targets, calibration, config)
            counter = VelocityCallCounter(model)
            t0 = time.time()
            try:
                z_c2 = clip(sample_latents(model, 1, stack.latent_flat_dim, device, cond=cond_row,
                                           cfg_scale=cfg_scale, ode_steps=ode_steps, noise=z0,
                                           guidance_fn=guidance_fn)).detach()
            finally:
                counter.remove()
            c2_seconds = time.time() - t0
            c2_calls, c2_rows = counter.calls, counter.rows
            c2_stats = dict(guidance_fn.stats)
            if 'c2' in methods:
                mesh, report = _decode_and_audit(vae, z_c2 * stack.latent_std + stack.latent_mean,
                                                 resolution, device, cond_names, target_stored,
                                                 cond_std_np)
                record('c2', z_c2, mesh, report, c2_seconds, c2_calls, c2_rows,
                       extra={'guidance_fm_evaluations': c2_stats.get('fm_evaluations'),
                              'guidance_decoder_grids': c2_stats.get('decoder_grids')},
                       detail={'guidance_stats': _jsonable(c2_stats)})

        # ---- e2 / c2e2: Newton correction of the plain / guided latent ----
        for method, z_start, calls, fm_rows in (('e2', z_plain, plain_calls, plain_rows),
                                                ('c2e2', z_c2, c2_calls, c2_rows)):
            if method not in methods:
                continue
            t0 = time.time()
            z_new, info = apply_newton(tools, stack, z_start, targets, calibration, newton_config)
            z_new = z_new.detach()
            mesh, report = _decode_and_audit(vae, z_new * stack.latent_std + stack.latent_mean,
                                             resolution, device, cond_names, target_stored,
                                             cond_std_np)
            # `newton_correct` returns a FLAT history list (one entry per round,
            # tagged with its `row`); `summarize_history` takes that whole list,
            # not one entry of it. One latent per call here, so no `row=` needed.
            history = (info or {}).get('history') or []
            summary = tools.refinement.summarize_history(history) if history else {}
            # NFE: Newton adds decoder/Marching-Cubes measurements, never
            # velocity-net calls, so the FM count is the starting sample's.
            record(method, z_new, mesh, report, time.time() - t0, calls, fm_rows,
                   extra={'newton_accepted_steps': (info or {}).get('accepted_steps'),
                          'newton_mc_measurements': summary.get('measurements'),
                          'newton_residual_initial': summary.get('residual_initial'),
                          'newton_residual_final': summary.get('residual_final'),
                          'newton_seconds': (info or {}).get('seconds')},
                   detail={'newton': _jsonable(info)})

        # ---- progress line ----
        parts = []
        for method in methods:
            row = next(r for r in rows if r['method'] == method and r['shape_idx'] == shape_idx)
            errs = [f'{name[:6]}={_pct(row.get(f"rel_error_{name}"))}' for name in geometric_names]
            parts.append(f'{method}: ' + ' '.join(errs) + f' wt={int(row["watertight"])}')
        print(f'  [{order + 1}/{len(shapes)}] shape {shape_idx:05d}: ' + ' | '.join(parts))

    # ---- Optional structural audit of FEA-named conditions, per method ----
    fea_audit_meta = {}
    if audit_used != 'geometric' and fea_names:
        for method in methods:
            meshes = meshes_by_method[method]
            entries, meta = fea_condition_audit(meshes, cond_names, audit_used, config,
                                                workdir=os.path.join(out_dir, f'fea_audit_{method}'))
            fea_audit_meta[method] = meta
            for (row, report), entry in zip(reports_by_method[method], entries):
                attach_fea_audit(report, entry, cond_names, shapes[row['index']]['target_stored'])
                rel = report.get('fea_condition_rel_error_raw') or {}
                values = (entry or {}).get('values_raw') or {}
                for name in fea_names:
                    if name in rel:
                        row[f'rel_error_{name}'] = rel[name]
                        row[f'actual_{name}'] = values.get(name)
                        row[f'measured_by_{name}'] = f'{audit_used}:{meta["label"]}'
                    elif entry is not None and entry.get('error'):
                        row[f'measured_by_{name}'] = f'{audit_used} failed: {entry["error"]}'
                row['fea_audit_error'] = (entry or {}).get('error')

    aggregate = aggregate_conditional(rows, methods, cond_names)
    print_conditional_table(aggregate, methods, cond_names)
    elapsed = time.time() - t_start
    print(f'Evaluated {len(shapes)} shapes x {len(methods)} methods in {elapsed:.1f}s')

    summary = {
        'eval_task': task,
        'fm_modelpath': stack.fm_path,
        'vae_modelpath': stack.vae_path,
        'fm_epoch': fm_ckpt.get('epoch'),
        'dataset_dir': dataset_dir,
        'split': split,
        'split_seed': split_seed,
        'split_by_parent': bool(ds_config.get('split_by_parent', False)),
        'split_sizes': split_sizes,
        'eval_seed': eval_seed,
        'eval_num_shapes': eval_num_shapes,
        'num_shapes_evaluated': len(shapes),
        'shape_selection': ('all' if len(shapes) == total else 'random subset seeded by eval_seed'),
        'cond_names': cond_names,
        'geometric_names': geometric_names,
        'fea_names': fea_names,
        'not_measurable_geometrically': fea_names if audit_used == 'geometric' else [],
        'cond_dropout_mode': str((fm_ckpt.get('config') or {}).get('cond_dropout_mode', 'all')),
        'methods': methods,
        'candidate_multiplier': candidate_multiplier,
        'cfg_scale': cfg_scale,
        'ode_steps': ode_steps,
        'mc_resolution': resolution,
        'latent_clip': latent_clip,
        'guidance': guidance_info,
        'newton': newton_info,
        'descriptor_calibration_path': (config.get('descriptor_calibration_path')
                                        if needs_tools else None),
        'proxy_target_names': proxy_target_names if needs_tools else [],
        'scored_but_not_optimised': ([n for n in cond_names
                                      if (CN.is_geometric(n) or audit_used != 'geometric')
                                      and n not in proxy_target_names]
                                     if needs_tools else []),
        'excluded_shape_indices': sorted(excluded),
        'dropped_from_pool': dropped,
        'ood_shapes': ood_shapes,
        'condition_ood_policy': str(config.get('condition_ood_policy', 'warn')).lower(),
        'max_condition_z': float(config.get('max_condition_z',
                                            fm_ckpt.get('cond_clip') or 5.0)),
        'condition_audit_backend': {
            'requested': str(config.get('condition_audit', 'geometric')).lower(),
            'used': audit_used, 'reason': audit_reason},
        'fea_audit': fea_audit_meta or None,
        'pairing': ('all methods share the base noise z0 = randn(shape_generator(eval_seed, '
                    'shape_idx)); rejection candidate 0 is that row; e2 corrects the plain '
                    'latent, c2e2 the c2 latent'),
        'relative_error_units': ('raw physical units (from_stored for log-stored FEA names); '
                                 'geometric names are in the normalized-mesh frame'),
        'clipped_condition_entries': int(sum(sum(s['clipped']) for s in shapes)),
        'seconds': elapsed,
        'aggregate': aggregate,
        'rows': rows,
        'details': details,
    }
    json_path = os.path.join(out_dir, 'eval_conditional.json')
    with open(json_path, 'w') as f:
        json.dump(_jsonable(summary), f, indent=2)
    csv_path = os.path.join(out_dir, 'eval_conditional.csv')
    _write_csv(csv_path, rows)
    print(f'JSON: {json_path}')
    print(f'CSV:  {csv_path}')
    return summary


# ---------------------------------------------------------------------------
# mode evaluate: dispatch on eval_task
# ---------------------------------------------------------------------------

def run_evaluate(config, config_filename='config.txt'):
    task = str(config.get('eval_task', 'reconstruction')).lower()
    if task not in EVAL_TASKS:
        raise ValueError(f"eval_task must be one of {EVAL_TASKS}, got '{task}'")
    if task == 'reconstruction':
        return run_reconstruction_eval(config, config_filename)
    if task == 'descriptor_calibration':
        return run_descriptor_calibration(config, config_filename)
    return run_conditional_eval(config, config_filename)
