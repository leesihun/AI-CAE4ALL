"""`mode inference`: N SAR samples per held-out system, in MGN-V's format.

For every scene of `infer_dataset` the run produces `num_samples` draws of
the final-timestep field, writes them (optionally) as rollout HDF5 files with
MeshGraphNets_Variational's layout and names, and -- when `eval_dataset` is
set -- the same spread_values.npz / spread_metrics.json / histogram_compare.png
tail, computed by the same (copied) code, so the ex3 scoring reads a SAR run
exactly like an MGN-V one.

`sample_scope`:
    geometry (default)  one ensemble per system (scenes sharing the
                        `split_group_attr` value AND identical condition rows);
                        every scene of that system is credited with it. SAR
                        conditions only on geometry + conditions, so the
                        scenes' conditional distributions are identical and
                        sampling them separately would only cost time.
    scene               a fresh ensemble per scene (MGN-V's loop, n_scenes x
                        the sampling cost).
"""
import hashlib
import json
import os
import time

import h5py
import numpy as np
import torch

from general_modules import mesh_dataset as md
from sar.nn import SAR
from sar.nn.modules.transolver import set_runtime_options
from training_profiles.common import (as_list, check_normalization_matches, load_checkpoint,
                                      resolve_device)

from .spread import (_eval_dataset_groups, _eval_dataset_spreads, _plot_spread_histogram,
                     _rollout_feature_names, _spread_by_group, _spread_index, _spread_scores,
                     _spread_stat, _spread_stat_expr)


DEFAULT_STEPS = [20, 11, 2]     # upstream examples/Wing sampling notebook


def spread_of(field, idx, stat):
    """Spread statistic of one [N, out] physical field ([b, -1] indexing as MGN-V)."""
    return _spread_stat(np.asarray(field)[:, idx], stat)


def sampling_steps(config, num_iters):
    steps = as_list(config.get('sample_steps', DEFAULT_STEPS if num_iters == 3 else 10))
    if len(steps) == 1:
        steps = steps * num_iters
    if len(steps) != num_iters:
        raise ValueError(f'sample_steps lists {len(steps)} values but the model has '
                         f'{num_iters} scales (sar_num_iters)')
    return steps


@torch.no_grad()
def sample_graph(model, graph, n, config):
    """[n, N, out] normalized samples of one system graph."""
    steps = sampling_steps(config, model.num_iters)
    batch = max(1, int(config.get('sample_batch_size', 8)))
    out = model.sample_n(n, graph, batch_size=min(batch, n), steps=steps)   # [N, n, F]
    return out.permute(1, 0, 2).float().cpu().numpy()


def sample_system(model, static, h5_path, sid, stats, config, n, device, seed=0):
    """[n, N, out] physical samples for the system whose static graph is
    `static`, conditioned on scene `sid`'s condition rows."""
    with h5py.File(h5_path, 'r') as f:
        _, glob = md.read_fields(f, sid, stats, config)
    graph = md.make_graph(static, glob=glob)
    torch.manual_seed(int(seed))
    samples = sample_graph(model, graph, n, config)
    return md.denormalize_target(samples, stats).astype(np.float32)


def load_sar(config, device):
    """The SAR model of `modelpath`, its frozen VAE taken from `vae_modelpath`."""
    modelpath = config['modelpath']
    vae_path = config['vae_modelpath']
    ckpt = load_checkpoint(modelpath)
    if ckpt.get('stage') != 'sar':
        raise ValueError(f'{modelpath} is not a SAR-route SAR checkpoint (stage {ckpt.get("stage")!r})')
    vae_ckpt = load_checkpoint(vae_path)
    if vae_ckpt.get('normalization') != ckpt.get('normalization'):
        raise ValueError(f'{vae_path} is not the VAE {modelpath} was trained over '
                         '(their train-split normalizations differ)')
    stats = ckpt['normalization']
    check_normalization_matches(stats, config, modelpath)
    model = SAR.load_from_checkpoint(modelpath, vae_ckpt=vae_path).to(device)
    model.eval()
    return model, ckpt


def _systems(config, infer_path, sample_ids, scope):
    """Ordered [(key, [scene ids])] of what gets one ensemble."""
    if scope == 'scene':
        return [(str(s), [s]) for s in sample_ids]
    attr = md.group_attr(config)
    groups = md.group_ids(infer_path, sample_ids, attr)
    systems = []
    with h5py.File(infer_path, 'r') as f:
        for key, sids in groups.items():
            # Split a geometry further if its scenes' condition rows differ.
            by_cond = {}
            for sid in sids:
                cond = md.read_conditions_raw(f, sid, config)
                digest = '' if cond is None else hashlib.sha1(
                    np.ascontiguousarray(cond, dtype=np.float64).tobytes()).hexdigest()[:12]
                by_cond.setdefault(digest, []).append(sid)
            if len(by_cond) == 1:
                systems.append((key, sids))
            else:
                for j, members in enumerate(by_cond.values()):
                    systems.append((f'{key}/c{j}', members))
    return systems


def _save_rollout_h5(output_path, sample_id, field, ref_pos, mesh_edge, part_ids, output_dim,
                     model_path, config_filename, rollout_time_s, vae_sample_idx,
                     feature_names, stats):
    """MeshGraphNets_Variational's `_save_rollout_h5` layout for one draw.

    nodal_data [3 + out + 1, 2, N]: t=0 is the zero state a static MGN-V rollout
    starts from, t=1 the sample. normalization_params holds SAR's own
    statistics (min/max scaling), not MGN-V's mean/std set.
    """
    num_nodes = ref_pos.shape[0]
    steps = 1
    num_save_features = 3 + output_dim + 1
    nodal_data = np.zeros((num_save_features, steps + 1, num_nodes), dtype=np.float32)
    for c in range(3):
        nodal_data[c, :, :] = ref_pos[:, c]
    for ch in range(output_dim):
        nodal_data[3 + ch, 1, :] = field[:, ch]
    if part_ids is not None:
        nodal_data[3 + output_dim, :, :] = part_ids[np.newaxis, :]

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_samples'] = 1
        f.attrs['num_features'] = num_save_features
        f.attrs['num_timesteps'] = steps + 1
        f.attrs['output_var'] = int(output_dim)

        sample_grp = f.create_group('data').create_group(str(sample_id))
        sample_grp.create_dataset('nodal_data', data=nodal_data, compression='gzip', compression_opts=4)
        sample_grp.create_dataset('mesh_edge', data=mesh_edge)

        meta_grp = sample_grp.create_group('metadata')
        meta_grp.attrs['sample_id'] = sample_id
        meta_grp.attrs['num_nodes'] = num_nodes
        meta_grp.attrs['num_edges'] = mesh_edge.shape[1]
        meta_grp.attrs['num_timesteps'] = steps + 1
        meta_grp.attrs['model_path'] = model_path
        meta_grp.attrs['config_file'] = config_filename
        meta_grp.attrs['total_rollout_time_s'] = rollout_time_s
        meta_grp.attrs['vae_sample_idx'] = vae_sample_idx
        meta_grp.attrs['hierarchy_seed'] = -1
        meta_grp.create_dataset('feature_min', data=nodal_data.min(axis=(1, 2)).astype(np.float32))
        meta_grp.create_dataset('feature_max', data=nodal_data.max(axis=(1, 2)).astype(np.float32))
        meta_grp.create_dataset('feature_mean', data=nodal_data.mean(axis=(1, 2)).astype(np.float32))
        meta_grp.create_dataset('feature_std', data=nodal_data.std(axis=(1, 2)).astype(np.float32))

        global_meta = f.create_group('metadata')
        global_meta.create_dataset('feature_names', data=feature_names)
        norm_grp = global_meta.create_group('normalization_params')
        norm_grp.attrs['scheme'] = 'sar_minmax'
        for key in ('target_min', 'target_max', 'cond_min', 'cond_max', 'pos_center'):
            norm_grp.create_dataset(key, data=np.asarray(stats[key], dtype=np.float64))
        norm_grp.create_dataset('pos_scale', data=np.float64(stats['pos_scale']))
        norm_grp.create_dataset('edge_r', data=np.float64(stats['edge_r']))


def run_inference(config, config_filename='config.txt'):
    print('\n' + '=' * 60 + '\nSAR SAMPLING INFERENCE\n' + '=' * 60)
    device = resolve_device(config)
    set_runtime_options(dense_attention=bool(config.get('dense_attention', True)),
                        use_checkpointing=False)
    model, ckpt = load_sar(config, device)
    stats = ckpt['normalization']
    num_scales = int(ckpt['num_scales'])
    output_dim = int(stats['output_var'])
    print(f'Model: {config["modelpath"]} (epoch {ckpt.get("epoch")}, {model.num_iters} scales, '
          f'steps {sampling_steps(config, model.num_iters)})')

    infer_path = config['infer_dataset']
    output_dir = config.get('inference_output_dir', '../../output/sar/infer')
    os.makedirs(output_dir, exist_ok=True)
    num_samples = int(config.get('num_samples', 64))
    seed = int(config.get('sample_seed', 0))
    scope = str(config.get('sample_scope', 'geometry')).lower()
    if scope not in ('geometry', 'scene'):
        raise ValueError(f'sample_scope must be geometry or scene, got {scope!r}')
    save_rollouts = bool(config.get('save_rollouts', False))
    eval_dataset = config.get('eval_dataset')
    make_histogram = bool(config.get('make_histogram', eval_dataset is not None))
    z_gen_idx, spread_stat = _spread_index(config)
    spread_label = str(config.get('spread_label', f'channel {z_gen_idx}'))
    feature_names = _rollout_feature_names(infer_path, output_dim)

    sample_ids = md.list_sample_ids(infer_path)
    systems = _systems(config, infer_path, sample_ids, scope)
    print(f'  {len(sample_ids)} scene(s) in {infer_path}; {len(systems)} ensemble(s) of '
          f'{num_samples} (sample_scope {scope})')

    generated_spreads, generated_scene_ids, generated_lams = [], [], []
    t_all = time.time()
    with h5py.File(infer_path, 'r') as f:
        for i, (key, sids) in enumerate(systems):
            t0 = time.time()
            static = md.build_static(f, sids[0], stats, num_scales, config)
            _, glob = md.read_fields(f, sids[0], stats, config)
            graph = md.make_graph(static, glob=glob)
            torch.manual_seed(seed + i)
            fields = md.denormalize_target(sample_graph(model, graph, num_samples, config), stats)
            elapsed = time.time() - t0
            spreads = [spread_of(fields[k], z_gen_idx, spread_stat) for k in range(num_samples)] \
                if output_dim > np.max(z_gen_idx) else []
            for sid in sids:
                generated_spreads.extend(spreads)
                generated_scene_ids.extend([str(sid)] * len(spreads))
                generated_lams.extend([1.0] * len(spreads))
            print(f'  [{i + 1}/{len(systems)}] system {key}: scenes {sids[:6]}{"..." if len(sids) > 6 else ""}, '
                  f'{static["_num_nodes"]} nodes, {num_samples} samples in {elapsed:.1f}s'
                  + (f', spread mean {np.mean(spreads):.4e} sd {np.std(spreads):.4e}' if spreads else ''))
            if save_rollouts:
                for sid in sids:
                    nodal = f[f'data/{sid}/nodal_data']
                    ref_pos = nodal[0:3, 0, :].T
                    part_ids = (np.asarray(nodal[-1, 0, :]).astype(np.int32)
                                if bool(config.get('use_node_types', True)) else None)
                    mesh_edge = np.asarray(f[f'data/{sid}/mesh_edge'])
                    for k in range(num_samples):
                        name = f'rollout_sample{sid}_vaesample{k}_steps1.h5'
                        _save_rollout_h5(os.path.join(output_dir, name), sid, fields[k], ref_pos,
                                         mesh_edge, part_ids, output_dim, config['modelpath'],
                                         config_filename, elapsed, k, feature_names, stats)
            del graph, static, fields
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f'\nSampling complete: {len(systems)} ensemble(s) x {num_samples} sample(s) for '
          f'{len(sample_ids)} scene(s) in {time.time() - t_all:.1f}s')

    if not make_histogram:
        return
    print('\n' + '=' * 60)
    print(f'HISTOGRAM COMPARE ({_spread_stat_expr(spread_label, spread_stat)}, final timestep)')
    print('=' * 60)
    if not eval_dataset:
        print('  Skipped: no `eval_dataset` set in config (needed for the ground-truth comparison).')
        return
    if not os.path.exists(str(eval_dataset)):
        print(f'  Skipped: eval_dataset not found: {eval_dataset}')
        return
    if not generated_spreads:
        print(f'  Skipped: no generated spread values (output_var={output_dim} has no channel {z_gen_idx}).')
        return
    try:
        gt, gt_scene = _eval_dataset_spreads(str(eval_dataset), z_gen_idx, spread_stat)
        gen = np.asarray(generated_spreads, dtype=np.float64)
        print(f'  GT spread values  (1 per eval sample): {gt.size:,}')
        print(f'  Gen spread values (1 per sample):      {gen.size:,}')
        for tag, col in (('GT', gt), ('GEN', gen)):
            print(f'  [SPREAD] {tag} mean={col.mean():.6e} std={col.std():.6e} '
                  f'min={col.min():.6e} max={col.max():.6e} n={col.size}')
        npz_path = os.path.join(output_dir, 'spread_values.npz')
        np.savez_compressed(
            npz_path, gt=gt, gen=gen,
            gt_scene=np.asarray([str(s) for s in gt_scene]),
            gen_scene=np.asarray([str(s) for s in generated_scene_ids]),
            gen_lam=np.asarray(generated_lams, dtype=np.float64))
        print(f'  [SPREAD] values -> {npz_path}')
        scores = _spread_scores(gt, gt_scene, gen, generated_scene_ids)
        scores['channel'] = z_gen_idx
        scores['label'] = spread_label
        scores['stat'] = spread_stat
        scores['eval_dataset'] = str(eval_dataset)
        scores['sample_scope'] = scope
        for key, target in (('crps_norm', 0.0), ('spread_skill', 1.0), ('pit_ks', 0.0),
                            ('pit_tails', 0.04), ('w1_norm', 0.0), ('dmean_norm', 0.0),
                            ('sd_ratio', 1.0)):
            if key in scores:
                print(f'  [SCORE] {key}={scores[key]:.4f} (target {target:g})')
        scores_path = os.path.join(output_dir, 'spread_metrics.json')
        with open(scores_path, 'w', encoding='utf-8') as fh:
            json.dump(scores, fh, indent=2)
        print(f'  [SCORE] metrics -> {scores_path}')
        _plot_spread_histogram(
            gt, gen, os.path.join(output_dir, 'histogram_compare.png'),
            eval_path=str(eval_dataset), rollout_dir=output_dir,
            bins=int(config.get('histogram_bins', 60)),
            clip_quantile=float(config.get('histogram_clip_quantile', 0.0)),
            show=bool(config.get('show_histogram', False)), label=spread_label,
            scores=scores, stat=spread_stat)
    except Exception as exc:  # never let plotting break a finished run
        print(f'  Histogram generation failed: {exc}')
    # One figure per held-out geometry, beside the pooled one: the pooled curve
    # hides per-geometry width errors.
    attr = md.group_attr(config)
    try:
        groups = _eval_dataset_groups(str(eval_dataset), attr)
        if groups:
            _spread_by_group(
                gt, gt_scene, gen, generated_scene_ids, groups, attr, output_dir,
                eval_path=str(eval_dataset), bins=int(config.get('histogram_bins', 60)),
                clip_quantile=float(config.get('histogram_clip_quantile', 0.0)),
                label=spread_label, stat=spread_stat)
        else:
            print(f'  Per-group histograms skipped: no `{attr}` metadata attribute in {eval_dataset}')
    except Exception as exc:
        print(f'  Per-group histogram generation failed: {exc}')
