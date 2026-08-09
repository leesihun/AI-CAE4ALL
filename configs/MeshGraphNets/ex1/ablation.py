#!/usr/bin/env python3
"""HI-MGN ex1 ablation driver: gen -> cost -> train -> infer -> eval -> report.

Five axes, all derived mechanically from `config_train1.txt` so the arms cannot
drift apart:

    voronoi   1-stage / 2-stage (baseline) / 3-stage hierarchy
    mp        mp_per_level shape at a fixed total of 28 blocks
    coarsen   voronoi_seedmean (baseline) vs voronoi_inherit
    totalmp   total message-passing budget 14 / 28 / 38
    interp    learned_interpolation True (baseline) vs False (broadcast)

The baseline sits in all five axes, so the 13 table cells collapse to 9 distinct
configurations. The run list carries 11 entries: the 9 configurations plus two
extra runs of the baseline. Those two are not redundant -- nothing in the
training path calls `torch.manual_seed` (`split_seed` only fixes the data
split), so weight init and geometry augmentation are a fresh random draw every
run. The three baseline runs are therefore the study's only measurement of
run-to-run noise, and without them an R2 gap of a few 1e-3 between arms cannot
be told from scatter.

Usage (from anywhere):

    python configs/MeshGraphNets/ex1/ablation.py gen      # write 22 configs
    python configs/MeshGraphNets/ex1/ablation.py cost     # params + FLOPs, no GPU
    python configs/MeshGraphNets/ex1/ablation.py train    # launch 11 runs
    python configs/MeshGraphNets/ex1/ablation.py infer    # rollout on ex1_infer.h5
    python configs/MeshGraphNets/ex1/ablation.py eval     # R2 / RMSE / peak
    python configs/MeshGraphNets/ex1/ablation.py report   # final table
    python configs/MeshGraphNets/ex1/ablation.py all      # everything in order

`--gpus N` (default 8) sets how many GPUs the 11 runs are packed onto; the
packing is longest-processing-time-first over the measured FLOP costs, and the
chosen GPU is written into each config's `gpu_ids` so the runner and the config
cannot disagree.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MGN_ROOT = REPO_ROOT / 'MeshGraphNets'
BASE_CFG = HERE / 'config_train1.txt'
RESULT_DIR = REPO_ROOT / 'output' / 'meshgraphnets' / 'ex1' / 'ablation'
TRAIN_DATASET = REPO_ROOT / 'dataset' / 'ex1.h5'
OOD_DATASET = REPO_ROOT / 'dataset' / 'ex1_infer.h5'

SPLIT_SEED = 42          # pinned into every generated config by `gen`
PRIMARY_SPLIT = 'infer'  # scores.json key; every arm is scored on ex1_infer.h5
LATENT_DIM = 128
EDGE_IN = 8
FLOP_PROBE_SAMPLES = 5   # samples spanning ex1's 6x size range


# ---------------------------------------------------------------------------
# Arm table -- the single place to add or remove a run
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    name: str
    axis: str
    label: str
    overrides: dict = field(default_factory=dict)
    gpu: int = 0
    cost: float = 1.0
    params: int = 0
    gflops: float = 0.0


ARMS = [
    Arm('base', 'baseline', 'baseline (as shipped)', {}),
    Arm('base_rep1', 'baseline', 'baseline replicate 1', {}),
    Arm('base_rep2', 'baseline', 'baseline replicate 2', {}),

    Arm('vc_1stage', 'voronoi', '1 stage: N -> 100', {
        'multiscale_levels': 1,
        'voronoi_clusters': [100],
        'mp_per_level': [8, 12, 8],
    }),
    Arm('vc_3stage', 'voronoi', '3 stage: N -> 10000 -> 1000 -> 100', {
        'multiscale_levels': 3,
        'voronoi_clusters': [10000, 1000, 100],
        'mp_per_level': [3, 4, 4, 6, 4, 4, 3],
    }),

    Arm('mp_flat', 'mp', 'mp shape flat 5,6,6,6,5', {
        'mp_per_level': [5, 6, 6, 6, 5],
    }),
    Arm('mp_fine', 'mp', 'mp shape fine-heavy 7,5,4,5,7', {
        'mp_per_level': [7, 5, 4, 5, 7],
    }),

    Arm('ct_inherit', 'coarsen', 'coarsening voronoi_inherit', {
        'coarsening_type': 'voronoi_inherit',
    }),

    Arm('tmp_14', 'totalmp', 'total MP 14 (2,3,4,3,2)', {
        'mp_per_level': [2, 3, 4, 3, 2],
    }),
    Arm('tmp_38', 'totalmp', 'total MP 38 (6,8,10,8,6)', {
        'mp_per_level': [6, 8, 10, 8, 6],
    }),

    Arm('interp_bcast', 'interp', 'broadcast prolongation', {
        'learned_interpolation': False,
    }),
]

# Which arm each axis compares against, for the report's delta column.
AXIS_BASELINE = {axis: 'base' for axis in ('voronoi', 'mp', 'coarsen', 'totalmp', 'interp')}


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def _fmt(value) -> str:
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, (list, tuple)):
        return ', '.join(str(v) for v in value)
    return str(value)


def _rewrite(lines, overrides, appends_after=None):
    """Return config lines with `overrides` applied in place.

    The native parser treats a duplicate key as a hard error, so an override
    must replace the existing line rather than be appended. Keys absent from
    the base file are appended after the marker line instead, keeping them
    inside the section they belong to.
    """
    remaining = dict(overrides)
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('%') or stripped == "'":
            out.append(line)
            continue
        key = re.split(r'[\s\t]+', stripped, maxsplit=1)[0]
        if key in remaining:
            comment = ''
            if '#' in line:
                comment = '  # ' + line.split('#', 1)[1].strip()
            out.append(f'{key}\t{_fmt(remaining.pop(key))}{comment}')
        else:
            out.append(line)

    if remaining:
        anchor = appends_after
        idx = len(out)
        if anchor is not None:
            for i, line in enumerate(out):
                if line.strip().startswith(anchor):
                    idx = i + 1
        for key, value in remaining.items():
            out.insert(idx, f'{key}\t{_fmt(value)}')
            idx += 1
    return out


def _read_base():
    text = BASE_CFG.read_text(encoding='utf-8')
    if text.startswith('﻿'):
        raise SystemExit(f'{BASE_CFG} starts with a UTF-8 BOM; the native parser rejects that.')
    return text.splitlines()


def _swap_basename(value: str, new_stem: str) -> str:
    """Replace only the filename, preserving the config's relative path base.

    The shipped paths use inconsistent bases (`../dataset/...` vs
    `../../output/...`), each resolved by whichever process reads it. Deriving
    a new base here would break one of them, so only the leaf changes.
    """
    head, _, tail = value.rpartition('/')
    ext = Path(tail).suffix
    return f'{head}/{new_stem}{ext}' if head else f'{new_stem}{ext}'


def cmd_gen(args):
    base = _read_base()
    base_map = {}
    for line in base:
        s = line.strip()
        if s and not s.startswith('%') and s != "'":
            k, _, v = s.partition('\t')
            base_map[k.strip()] = v.split('#')[0].strip()

    assign_gpus(args.gpus)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for arm in ARMS:
        # --- train config ---
        ov = dict(arm.overrides)
        ov['gpu_ids'] = arm.gpu
        # config_train1.txt omits split_seed and relies on the native default.
        # Pin it: `eval` reproduces the split arithmetic itself to pick the
        # held-out ids, and a changed default would silently score the wrong
        # samples rather than fail.
        ov['split_seed'] = SPLIT_SEED
        ov['log_file_dir'] = _swap_basename(base_map['log_file_dir'], f'abl_{arm.name}_train')
        ov['modelpath'] = _swap_basename(base_map['modelpath'], f'abl_{arm.name}')
        lines = _rewrite(base, ov, appends_after='mp_per_level')
        header = [
            f'%   ABLATION ARM {arm.name} -- axis "{arm.axis}": {arm.label}',
            '%',
            '%   GENERATED by configs/MeshGraphNets/ex1/ablation.py -- do not edit.',
            '%   Edit config_train1.txt or the ARMS table and re-run `ablation.py gen`,',
            '%   so the arms cannot drift apart. Every key not listed as an arm',
            '%   difference is config_train1.txt verbatim.',
            "'",
        ]
        train_path = HERE / f'config_train_abl_{arm.name}.txt'
        train_path.write_text('\n'.join(header + lines) + '\n', encoding='utf-8')
        written.append(train_path)

        # --- inference config ---
        # Every arm is scored on ex1_infer.h5, the established ex1 evaluation
        # mesh (config_infer1.txt uses it). config_train1.txt ships
        # `infer_dataset hex_dataset.h5`, whose state rows are all zero and so
        # cannot serve as ground truth -- that is what this repoints.
        infer_ov = dict(ov)
        infer_ov.update({
            'mode': 'inference',
            'infer_dataset': '../dataset/ex1_infer.h5',
            'infer_timesteps': 1,
            'inference_output_dir': f'../output/meshgraphnets/ex1/ablation/infer_{arm.name}',
            'log_file_dir': _swap_basename(base_map['log_file_dir'], f'abl_{arm.name}_infer'),
            'augment_geometry': False,
            'std_noise': 0.0,
            'Batch_size': 1,
            'Training_epochs': 1,
            'val_interval': 1,
            'test_batch_idx': 0,
            'use_checkpointing': False,
        })
        lines = _rewrite(base, infer_ov, appends_after='mp_per_level')
        p = HERE / f'config_infer_abl_{arm.name}.txt'
        p.write_text(
            '\n'.join([f'%   ABLATION ARM {arm.name} -- inference '
                       f'(GENERATED by ablation.py, do not edit)', "'"] + lines) + '\n',
            encoding='utf-8')
        written.append(p)

    print(f'Wrote {len(written)} configs to {HERE}')
    for arm in ARMS:
        print(f'  {arm.name:<14} gpu={arm.gpu}  {arm.label}')
    return 0


# ---------------------------------------------------------------------------
# Cost model: params (exact, from the real model) + FLOPs (analytic)
# ---------------------------------------------------------------------------

def _mlp_macs(i, h, o):
    return i * h + h * h + h * o


def _gnblock_macs(n, e, d=LATENT_DIM):
    return e * _mlp_macs(3 * d, d, d) + n * _mlp_macs(2 * d, d, d)


def _hierarchy_sizes(arm, sample_ids):
    """Per-level (N, E, E_up) for one arm, averaged over `sample_ids`.

    Uses the repo's own `build_multiscale_hierarchy`, so the geometry here is
    exactly what training will build -- not a reimplementation of it.
    """
    import h5py
    sys.path.insert(0, str(MGN_ROOT))
    from general_modules.multiscale_helpers import build_multiscale_hierarchy

    levels = int(arm.overrides.get('multiscale_levels', 2))
    ct = arm.overrides.get('coarsening_type', 'voronoi_seedmean')
    clusters = arm.overrides.get('voronoi_clusters', [5000, 100])
    if not isinstance(clusters, (list, tuple)):
        clusters = [clusters]
    types = [str(ct)] * levels

    acc = None
    with h5py.File(TRAIN_DATASET, 'r') as f:
        for sid in sample_ids:
            me = f[f'data/{sid}/mesh_edge'][:]
            ei = np.concatenate([me, me[[1, 0], :]], axis=1)
            nodal = f[f'data/{sid}/nodal_data']
            n = nodal.shape[2]
            ref = np.ascontiguousarray(nodal[:3, 0, :].T).astype(np.float32)
            hier = build_multiscale_hierarchy(ei, n, ref, levels, types, list(clusters))
            rows = [[n, ei.shape[1], 0]]
            for entry in hier:
                n_c = int(entry['n_c'])
                e_up = int(entry['up_ei'].shape[1]) if 'up_ei' in entry else 0
                rows.append([n_c, int(entry['c_ei'].shape[1]), e_up])
            arr = np.array(rows, dtype=np.float64)
            acc = arr if acc is None else acc + arr
    return acc / len(sample_ids)


def _forward_gflops(arm, lv):
    """FLOPs of one forward pass (2 per MAC), processor only."""
    d = LATENT_DIM
    mp = arm.overrides.get('mp_per_level', [4, 6, 8, 6, 4])
    L = len(lv) - 1
    if len(mp) != 2 * L + 1:
        # Hierarchy saturated below the configured depth; the model silently
        # runs the shallower V-cycle (see COARSENING_ABLATION_DESIGN.md 3.1).
        print(f'  WARNING [{arm.name}]: hierarchy reached {L} level(s) but '
              f'mp_per_level has {len(mp)} entries for {(len(mp) - 1) // 2}. '
              f'Blocks past the truncation are allocated and never run.')
        L = (len(mp) - 1) // 2
        while len(lv) < L + 1:
            lv = np.vstack([lv, lv[-1]])

    macs = 0
    for i in range(L):
        macs += (mp[i] + mp[2 * L - i]) * _gnblock_macs(lv[i][0], lv[i][1])
    macs += mp[L] * _gnblock_macs(lv[L][0], lv[L][1])

    learned = arm.overrides.get('learned_interpolation', True)
    for i in range(L):
        macs += lv[i + 1][1] * _mlp_macs(EDGE_IN, d, d)          # coarse edge encoder
        macs += lv[i][0] * (2 * d * d)                            # skip_proj
        if learned:
            e_up = lv[i + 1][2]
            macs += e_up * _mlp_macs(2 * d + 3, d, d)             # unpool edge MLP
            macs += lv[i][0] * _mlp_macs(2 * d, d, d)             # unpool node MLP
    return 2 * macs / 1e9


def _count_params(arm):
    """Exact parameter count by constructing the real model."""
    import h5py
    import torch  # noqa: F401  (imported for side effects / availability check)
    sys.path.insert(0, str(MGN_ROOT))
    from model.MeshGraphNets import MeshGraphNets

    cfg = _parsed_config(HERE / f'config_train_abl_{arm.name}.txt')
    with h5py.File(TRAIN_DATASET, 'r') as f:
        cfg['num_timesteps'] = int(f.attrs['num_timesteps'])
        if cfg.get('use_node_types'):
            # Node types are the LAST nodal_data row. Scan every sample, not
            # just the first: a sample-local set would undercount and shrink
            # the encoder's input width, biasing every arm's parameter count
            # by the same wrong amount.
            seen = set()
            for sid in sorted(int(s) for s in f['data'].keys()):
                seen.update(np.unique(f[f'data/{sid}/nodal_data'][-1, 0, :]).tolist())
            cfg['num_node_types'] = len(seen)
    model = MeshGraphNets(cfg, device='cpu')
    return sum(p.numel() for p in model.parameters())


def _parsed_config(path):
    sys.path.insert(0, str(REPO_ROOT))
    from cae_suite.config_parser import parse_config
    parsed = parse_config(Path(path))
    return dict(parsed.values)


def _probe_sample_ids():
    import h5py
    with h5py.File(TRAIN_DATASET, 'r') as f:
        sids = sorted(int(s) for s in f['data'].keys())
        ns = np.array([f[f'data/{s}/nodal_data'].shape[2] for s in sids])
    order = np.argsort(ns)
    picks = np.linspace(0, len(sids) - 1, FLOP_PROBE_SAMPLES).astype(int)
    return [sids[order[i]] for i in picks]


def assign_gpus(num_gpus: int):
    """Longest-processing-time-first packing of the arms onto `num_gpus`."""
    cached = RESULT_DIR / 'cost.json'
    if cached.exists():
        costs = json.loads(cached.read_text())
        for arm in ARMS:
            arm.cost = costs.get(arm.name, {}).get('gflops', 1.0)
    # LPT plus hill climbing, over many seeded restarts, best kept. At the
    # default 11 arms on 8 GPUs this reaches the true optimum (1.90 baseline-
    # forward units: 11 arms on 8 GPUs forces exactly three doubled-up GPUs,
    # and the cheapest possible pairing of the six smallest arms is
    # 1.00+0.90 / 1.00+0.89 / 1.00+0.56). The search is kept general so
    # changing the arm list or --gpus still packs well. Seeded, because
    # `train` and `infer` must derive the same assignment independently.
    costs = [a.cost for a in ARMS]
    rng = np.random.default_rng(0)

    def _pack(order):
        loads = [0.0] * num_gpus
        placement = [0] * len(ARMS)
        for idx in order:
            g = int(np.argmin(loads))
            placement[idx] = g
            loads[g] += costs[idx]
        # hill-climb: move any arm off the busiest GPU when that strictly helps
        for _ in range(100):
            busiest = int(np.argmax(loads))
            moved = False
            for idx in [i for i in range(len(ARMS)) if placement[i] == busiest]:
                for g in range(num_gpus):
                    if g == busiest:
                        continue
                    if max(loads[busiest] - costs[idx],
                           loads[g] + costs[idx]) < loads[busiest] - 1e-12:
                        loads[busiest] -= costs[idx]
                        loads[g] += costs[idx]
                        placement[idx] = g
                        moved = True
                        break
                if moved:
                    break
            if not moved:
                break
        return max(loads), loads, placement

    best = _pack(sorted(range(len(ARMS)), key=lambda i: -costs[i]))
    for _ in range(2000):
        cand = _pack(rng.permutation(len(ARMS)).tolist())
        if cand[0] < best[0] - 1e-12:
            best = cand
    _, loads, placement = best
    for arm, g in zip(ARMS, placement):
        arm.gpu = g
    return loads


def cmd_cost(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sids = _probe_sample_ids()
    print(f'Probing hierarchies on {len(sids)} samples spanning ex1\'s size range: {sids}\n')

    out = {}
    for arm in ARMS:
        if arm.name.startswith('base_rep'):
            src = out['base']
            arm.params, arm.gflops = src['params'], src['gflops']
        else:
            lv = _hierarchy_sizes(arm, sids)
            arm.gflops = _forward_gflops(arm, lv)
            try:
                arm.params = _count_params(arm)
            except Exception as exc:                       # pragma: no cover
                print(f'  [{arm.name}] param count unavailable ({exc}); run `gen` first.')
                arm.params = 0
        arm.cost = arm.gflops
        out[arm.name] = {'params': arm.params, 'gflops': arm.gflops, 'axis': arm.axis,
                         'label': arm.label}

    (RESULT_DIR / 'cost.json').write_text(json.dumps(out, indent=1))
    loads = assign_gpus(args.gpus)

    base = out['base']['gflops']
    print(f'{"arm":<14} {"axis":<9} {"params":>11} {"GFLOP/fwd":>10} {"vs base":>8} {"gpu":>4}')
    for arm in ARMS:
        print(f'{arm.name:<14} {arm.axis:<9} {arm.params:>11,} {arm.gflops:>10.1f} '
              f'{arm.gflops / base:>7.2f}x {arm.gpu:>4}')
    print(f'\nGPU load (relative): ' + '  '.join(f'{i}:{v/base:.2f}' for i, v in enumerate(loads)))
    print(f'makespan {max(loads)/base:.2f}x baseline-forward units')
    print(f'\nWrote {RESULT_DIR / "cost.json"}')
    print('NOTE: total block count fixed at 28 does NOT fix FLOPs -- a level-0 '
          'block costs ~450x a coarsest-level one. Report GFLOP beside every R2.')
    return 0


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _launch(configs, tag, jobs_log):
    procs = []
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for arm, cfg in configs:
        log = RESULT_DIR / f'{tag}_{arm.name}.stdout'
        cmd = [sys.executable, 'AI_CAE4ALL_main.py', '--config', str(cfg)]
        fh = open(log, 'w', encoding='utf-8')
        p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT)
        procs.append((arm, p, fh, time.time()))
        print(f'  [{tag}] {arm.name:<14} gpu={arm.gpu}  pid={p.pid}  -> {log.name}')

    results = {}
    for arm, p, fh, t0 in procs:
        rc = p.wait()
        fh.close()
        elapsed = time.time() - t0
        results[arm.name] = {'rc': rc, 'seconds': round(elapsed, 1), 'gpu': arm.gpu}
        status = 'OK' if rc == 0 else f'FAILED rc={rc}'
        print(f'  [{tag}] {arm.name:<14} {status}  {elapsed/3600:.2f}h')
    jobs_log.write_text(json.dumps(results, indent=1))
    failed = [k for k, v in results.items() if v['rc'] != 0]
    if failed:
        print(f'\n{len(failed)} arm(s) failed: {", ".join(failed)} '
              f'-- see {RESULT_DIR}/{tag}_*.stdout')
    return results


def cmd_train(args):
    assign_gpus(args.gpus)
    configs = [(a, HERE / f'config_train_abl_{a.name}.txt') for a in ARMS]
    missing = [c for _, c in configs if not c.exists()]
    if missing:
        raise SystemExit(f'Missing configs; run `gen` first. First missing: {missing[0]}')
    print(f'Launching {len(configs)} training runs on {args.gpus} GPU(s)\n')
    _launch(configs, 'train', RESULT_DIR / 'train_jobs.json')
    return 0


def cmd_infer(args):
    assign_gpus(args.gpus)
    configs = [(a, HERE / f'config_infer_abl_{a.name}.txt') for a in ARMS]
    missing = [c for _, c in configs if not c.exists()]
    if missing:
        raise SystemExit(f'Missing configs; run `gen` first. First: {missing[0]}')
    print(f'Launching {len(configs)} inference runs on {OOD_DATASET.name}\n')
    _launch(configs, 'infer', RESULT_DIR / 'infer_jobs.json')
    return 0


# ---------------------------------------------------------------------------
# Evaluation: R2 against ground truth
# ---------------------------------------------------------------------------

def _r2(pred, true):
    """Coefficient of determination, NOT Pearson r^2.

    The only r^2 in the repo is `mesh_utils_fast._pearson_r2`, used for figure
    titles; it is scale- and bias-blind, so a prediction at 2x the truth scores
    1.0. That is the wrong statistic for comparing arms.
    """
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _score_arm(arm, eval_sets, output_var):
    """Score one arm over each (name -> (rollout_dir, truth_h5, ids)) entry."""
    import h5py
    per_split = {}
    for split_name, (roll_dir, truth_h5, ids) in eval_sets.items():
        if not roll_dir.is_dir():
            continue
        preds, trues = [], []
        with h5py.File(truth_h5, 'r') as gt:
            for sid in ids:
                hit = list(roll_dir.glob(f'rollout_sample{sid}_steps*.h5'))
                if not hit:
                    continue
                with h5py.File(hit[0], 'r') as f:
                    p = f[f'data/{sid}/nodal_data'][3:3 + output_var, -1, :]
                t = gt[f'data/{sid}/nodal_data'][3:3 + output_var, -1, :]
                preds.append(np.asarray(p, np.float64))
                trues.append(np.asarray(t, np.float64))
        if not preds:
            continue
        pred = np.concatenate(preds, axis=1)
        true = np.concatenate(trues, axis=1)
        per_feat = [_r2(pred[c], true[c]) for c in range(output_var)]
        entry = {
            'n_samples': len(preds),
            # Headline R2 is the MEAN of per-channel R2, not R2 over the
            # stacked array. Pooling first would take SS_tot about a single
            # mean spanning displacements (mm) and stress (MPa), so the
            # largest-variance channel would silently become the whole metric.
            'r2': float(np.mean(per_feat)),
            'r2_pooled': _r2(pred, true),
            'rmse': float(np.sqrt(np.mean((pred - true) ** 2))),
            'r2_per_feature': per_feat,
            'rmse_per_feature': [float(np.sqrt(np.mean((pred[c] - true[c]) ** 2)))
                                 for c in range(output_var)],
            # Stress is the last output channel and the FEA quantity of
            # interest; pooling operators trade RMSE against peak retention
            # (voronoi_inherit is the clear case), so a single R2 hides the
            # difference the coarsening axis is actually making.
            'peak_rel_err': float(
                (np.abs(pred[-1]).max() - np.abs(true[-1]).max()) / np.abs(true[-1]).max()),
        }
        per_split[split_name] = entry
    return per_split


def cmd_eval(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    import h5py
    cfg = _parsed_config(BASE_CFG)
    output_var = int(cfg['output_var'])
    abl = REPO_ROOT / 'output' / 'meshgraphnets' / 'ex1' / 'ablation'

    with h5py.File(OOD_DATASET, 'r') as f:
        ids = sorted(int(s) for s in f['data'].keys())
    print(f'Scoring every arm on {OOD_DATASET.name}: {len(ids)} sample(s), ids {ids}\n')

    scores = {}
    for arm in ARMS:
        sets = {PRIMARY_SPLIT: (abl / f'infer_{arm.name}', OOD_DATASET, ids)}
        s = _score_arm(arm, sets, output_var)
        if not s:
            print(f'  {arm.name:<14} no rollout output -- run `infer` first')
            continue
        scores[arm.name] = s
        e = s[PRIMARY_SPLIT]
        print(f'  {arm.name:<14} R2={e["r2"]:.4f}  RMSE={e["rmse"]:.4g}  '
              f'peak_err={e["peak_rel_err"]:+.1%}  '
              f'per-feature R2={[round(v, 3) for v in e["r2_per_feature"]]}')

    (RESULT_DIR / 'scores.json').write_text(json.dumps(scores, indent=1))
    print(f'\nWrote {RESULT_DIR / "scores.json"}')
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def cmd_report(args):
    cost = json.loads((RESULT_DIR / 'cost.json').read_text()) if (RESULT_DIR / 'cost.json').exists() else {}
    scores = json.loads((RESULT_DIR / 'scores.json').read_text()) if (RESULT_DIR / 'scores.json').exists() else {}
    jobs = json.loads((RESULT_DIR / 'train_jobs.json').read_text()) if (RESULT_DIR / 'train_jobs.json').exists() else {}

    reps = [a.name for a in ARMS if a.axis == 'baseline' and a.name in scores]
    noise = None
    if len(reps) > 1:
        vals = [scores[n][PRIMARY_SPLIT]['r2'] for n in reps if PRIMARY_SPLIT in scores[n]]
        if len(vals) > 1:
            noise = (float(np.mean(vals)), float(np.std(vals)), len(vals))

    rows = []
    base_r2 = scores.get('base', {}).get(PRIMARY_SPLIT, {}).get('r2')
    for arm in ARMS:
        c = cost.get(arm.name, {})
        t = scores.get(arm.name, {}).get(PRIMARY_SPLIT, {})
        d = (t.get('r2') - base_r2) if (base_r2 is not None and t.get('r2') is not None) else None
        pf = t.get('r2_per_feature') or [None] * 4
        rows.append({
            'arm': arm.name, 'axis': arm.axis, 'label': arm.label,
            'params': c.get('params'), 'gflops': c.get('gflops'),
            'r2': t.get('r2'), 'delta_r2': d, 'rmse': t.get('rmse'),
            'peak_err': t.get('peak_rel_err'),
            'r2_dispx': pf[0], 'r2_dispy': pf[1], 'r2_dispz': pf[2], 'r2_stress': pf[3],
            'hours': (jobs.get(arm.name, {}).get('seconds') or 0) / 3600.0,
        })

    def f(x, spec='.4f'):
        return 'n/a' if x is None else format(x, spec)

    md = ['# HI-MGN ex1 ablation', '',
          f'All arms scored on `{OOD_DATASET.name}`. R2 is the mean of per-channel',
          '1 - SS_res/SS_tot on the denormalized field.', '']
    if noise:
        md += [f'Baseline replicates (n={noise[2]}): R2 = {noise[0]:.4f} +/- {noise[1]:.4f}. '
               f'**Treat any |dR2| below ~{2 * noise[1]:.4f} as noise.**', '']
    md += ['| arm | axis | what | params | GFLOP/fwd | R2 | dR2 | RMSE | peak err | R2 stress | hours |',
           '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(
            f'| `{r["arm"]}` | {r["axis"]} | {r["label"]} | '
            f'{f(r["params"], ",") if r["params"] else "n/a"} | {f(r["gflops"], ".1f")} | '
            f'{f(r["r2"])} | {f(r["delta_r2"], "+.4f")} | {f(r["rmse"], ".4g")} | '
            f'{f(r["peak_err"], "+.1%")} | {f(r["r2_stress"])} | {f(r["hours"], ".2f")} |')
    md += ['', '## Reading this table', '',
           '- **FLOPs are not matched across arms.** A fixed 28-block budget still spans a ~3x',
           '  compute range, because a level-0 block costs ~450x a coarsest-level one. An arm',
           '  that wins *and* is cheaper is a much stronger result than one that wins at 1.6x cost.',
           '- **`dR2` against the replicate spread above is the only honest significance test**',
           '  here. Nothing in the training path seeds the RNG, so two identical configs differ.',
           '- **`peak err`** is signed relative error of peak |stress|. Restriction operators trade',
           '  it against RMSE (`ct_inherit` is the designed case), so an arm losing on R2 while',
           '  gaining on peak is not simply worse -- `R2 stress` is broken out for the same reason.',
           '- R2 is NOT the Pearson r2 in the plotting code, which is scale- and bias-blind.']

    out_md = RESULT_DIR / 'report.md'
    out_md.write_text('\n'.join(md) + '\n', encoding='utf-8')

    import csv
    with open(RESULT_DIR / 'report.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print('\n'.join(md))
    print(f'\nWrote {out_md} and {RESULT_DIR / "report.csv"}')
    return 0


def cmd_all(args):
    # `gen` runs twice on purpose: the first pass has no measured costs yet, so
    # GPUs are packed round-robin; `cost` writes cost.json, and the second pass
    # re-packs by real FLOPs and bakes the balanced gpu_ids into the configs.
    for step in (cmd_gen, cmd_cost, cmd_gen, cmd_train, cmd_infer, cmd_eval, cmd_report):
        rc = step(args)
        if rc:
            return rc
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['gen', 'cost', 'train', 'infer', 'eval', 'report', 'all'])
    ap.add_argument('--gpus', type=int, default=8,
                    help='GPUs to pack the 11 runs onto (default 8)')
    args = ap.parse_args()
    return {'gen': cmd_gen, 'cost': cmd_cost, 'train': cmd_train, 'infer': cmd_infer,
            'eval': cmd_eval, 'report': cmd_report, 'all': cmd_all}[args.command](args)


if __name__ == '__main__':
    raise SystemExit(main())
