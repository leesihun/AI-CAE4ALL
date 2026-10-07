#!/usr/bin/env python3
"""Score every deterministic arm on its held-out set and rank the methods.

    python configs/campaigns/dataset_matrix/score_rank.py
    python configs/campaigns/dataset_matrix/score_rank.py --examples deterministic/ex1 deterministic/ex6
    python configs/campaigns/dataset_matrix/score_rank.py --out output/dataset_matrix/_campaign/135
    python configs/campaigns/dataset_matrix/score_rank.py --no-provenance   # outputs run_matrix.py did not make

For each deterministic arm in manifest.json this reads the infer config, the
ground truth it names (`infer_dataset`, or `dataset_dir` for LSH-VAE) and the
predictions it wrote (`rollout_sample*_steps*.h5` under `inference_output_dir`,
or `reconstructions.h5` under `output_dir`), and scores the physical output
rows only -- nodal_data rows 3:3+output_var. Coordinates (rows 0:3) and the
node-type row are copied through by every writer and are never scored; the
row count is read from the rollout's own `output_var` attribute and must agree
with the config.

Provenance: an arm is scored only if run_matrix.py recorded both of its stages
for the configs as they are now -- a train and an infer marker under
output/dataset_matrix/_campaign/<machine>/done/ whose sha256 matches the config
file byte for byte, with the inference started no earlier than the training
finished -- and every prediction file it scores was written after that
inference started. Anything else is `stale` (an older config, checkpoint or
prediction) or `unverified` (no marker at all: outputs made by hand, which
--no-provenance scores anyway). These are the rules run_matrix.py skips a
stage by, so one table never mixes a current arm with a leftover.

Scoring frames follow the README table ("never score a frame the model saw"):

    T == 1        the single frame (rollout: its last frame; frame 0 of the
                  MGN/operator writers is the zero input state)
    T  > 1        t = start .. T-1, start = 1 (frame 0 is the given initial
                  condition). One exception, per example rather than per arm:
                  deterministic/ex6, where LSH-VAE's conditioner observes
                  frames 0 and 1, so EVERY method there is scored on t=2..400.

Metrics, per sample and per output channel c over the scored frames x nodes:

    rel_l2[c] = ||pred - gt|| / ||gt||        (undefined when ||gt|| = 0)
    nrmse[c]  = rmse / std(gt)                (undefined when std(gt) = 0)
    r2[c]     = 1 - sse / sst

The per-sample value is the mean over the defined channels (a macro average:
pooling channels would let their different magnitudes count as skill, which is
the README's "never rank on the sum of MSEs"). An arm's value is the mean over
its samples; the median is printed beside it. Pooled variants (sums over every
sample before the ratio) are in the CSV as *_pooled. An example with fewer than
SMALL_HELD_OUT held-out samples says so in its header (ex1 has 1, ex2 has 5).

On FREE_NODES examples (ex4, ex5, ex6) rel_l2 and nrmse are also computed over
the free nodes only (rel_l2_free, nrmse_free): the node types MeshGraphNets
trains on, read per sample from the GT's node-type row at frame 0. Boundary
nodes that every writer copies from the given state would otherwise dilute
the error. They are printed beside the rank key, never ranked on.

Ranking is within one example, on RANK_KEY, lower is better, with average
ranks for exact ties. Every arm that is not `ok` -- missing / incomplete /
non-finite / mismatch / stale / unverified / error -- shares last place,
(k + 1 + N) / 2 for k scored arms of N, so a broken route costs its method
instead of vanishing from the table; an example with no scored arm is not
ranked. The same ranks are computed on nrmse (rank_nrmse). A rank is
normalised to (rank - 1) / (N - 1), 0 best and 1 last, because LSH-VAE makes
four examples seven-way races and the others are six-way.

The overall table is each method's mean normalised rank over example groups.
ex3_full and ex3_mid are one CRM problem at two mesh resolutions, so they
count once, as "ex3 (CRM)", by the mean of the two. Beside it: the same mean
on nrmse, the groups ranked out of the groups the method runs in, and wins
(the lowest group value on rel_l2; ties are shared).

The `updates` column is the optimizer-update budget generate.py records in
manifest.json (README "학습 예산 기록"): equal budgets make the comparison fair,
they do not show that an arm converged.

Convergence (README "수렴 확인") is read from the training logs: log_file_dir,
or vae_/lc_/fm_log_file_dir for the two-stage pipelines, one line per epoch,
last run only. The planned epochs are cut into BINS windows of 5% and each
window's median training and validation loss is taken. `val 60-80%` is the
change of the validation median from the 55-60% window to the 75-80% one; a
drop larger than FALLING (flag F) says the loss was still falling well into
the schedule, so the budget may be short. Part of any late drop under the
cosine schedule is the learning rate annealing, so F is a screen and a doubled
budget is the test. A window holds only the validated epochs (every
val_interval): one each for ex2 MGN/HI-MGN, so read `train 60-80%`, which
uses every epoch, beside it there. `tail` compares the last window with the best one: on the
deterministic routes, which keep the last epoch, a last window more than
RISING above the best (R) means the kept weights are not the best ones. The
other flags: I the log stops before the planned last epoch, X it has more
epochs than planned, N a loss is NaN/Inf, O the log was last written before
the recorded training started (a reused stage or a leftover), ? no log. The
rank tables show the arm's stage with the largest drop and the flags of all
its stages; the convergence section and convergence.csv list every stage,
geometry arms included. None of it ranks anything or changes the exit status.

Probabilistic arms are ranked the same way on crps_norm from the
spread_metrics.json their inference writes (score_spread.py prints the full
calibration table). Geometry arms have one method per example and are not
ranked; they appear only in the convergence section. Exit status 1 if any
ranked arm is not `ok`.

Needs numpy and h5py: run_matrix.py starts it under the MeshGraphNets
interpreter from ai_cae4all.local.toml.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]
RANK_KEY = 'rel_l2'
# Frames an arm's own input already contains, when that is more than the AR
# default of one (frame 0). Only LSH-VAE differs, and only where its
# conditioner reads field frames (README "채점 프레임 규칙"; prepare.py::make_conditions).
LSH_OBSERVED = {'ex3_full': 0, 'ex3_mid': 0, 'ex6': 2, 'ex9': 1}
METRICS = ('rel_l2', 'nrmse', 'r2')
FREE_METRICS = ('rel_l2', 'nrmse')
# MeshGraphNets node-type codes whose nodes the model is trained on: NORMAL (0)
# and, on cylinder_flow, OUTFLOW (5). Inflow/wall/obstacle/handle nodes are
# prescribed. The node-type row is the last nodal_data row.
FREE_NODES = {'ex4': (0, 5), 'ex5': (0,), 'ex6': (0,)}
# Examples that are one problem at two resolutions and count once overall.
GROUPS = {'ex3_full': 'ex3 (CRM)', 'ex3_mid': 'ex3 (CRM)'}
# Below this many held-out samples the header says a ranking gap may be noise
# (ex1 holds out 1 sample, ex2 holds out 5).
SMALL_HELD_OUT = 10
# run_matrix.py's markers: where they live and how their times are written.
CAMPAIGN = Path('output') / 'dataset_matrix' / '_campaign'
ISO_FORMAT = '%Y-%m-%dT%H:%M:%S%z'
# A prediction file may carry an mtime this much before its inference's start
# time (filesystem timestamp granularity), and still count as written by it.
MTIME_SLACK = 2.0
# Convergence readout: windows per planned run, and the flag thresholds (a
# relative change of the window medians). Provisional until the first runs.
BINS = 20
FALLING = -0.10
RISING = 0.05
FLAG_ORDER = 'FRINXO?'
# One line per epoch in every trainer's log, optionally tagged ([AE], [Prior]).
EPOCH_LINE = re.compile(r'^(?:\[(?P<tag>[^\]]+)\] )?Elapsed:? [0-9.]+s Epoch (?P<epoch>\d+) (?P<rest>.*)$')
# A training continued from its resume state (resume_state.py in each method)
# appends this to the same log before redoing the epochs after that state.
RESUME_LINE = re.compile(r'^==== Resume(?: \[(?P<tag>[^\]]+)\])? at epoch \d+')
NUMBER = r'(nan|-?inf|[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)'
# The first loss of each kind on the line: TrainOpt (MeshGraphNets, operators,
# Transolver), Train / Train recon= / Train fm= (MGN-V, HI_MGNFlow), Recon and
# LC train (SimulGenVAE), TrainSDF and TrainFM (SDFFlow); "Valid skipped" and
# "Val skipped" carry no number.
TRAIN_FIELD = re.compile(r'(?:TrainOpt |Train (?:recon=|fm=)?|Recon |TrainSDF |TrainFM |LC train )' + NUMBER)
VAL_FIELD = re.compile(r'(?:Valid (?:recon=|fm=)?|ValRecon |\bval )' + NUMBER)


def parse_config(root: Path, rel: str) -> dict:
    # The launcher's parser, so a value means here what it meant to the run.
    if str(DEFAULT_ROOT) not in sys.path:
        sys.path.insert(0, str(DEFAULT_ROOT))
    from cae_suite.config_parser import parse_config as _parse
    parsed = _parse(str(root / rel))
    if not parsed.values:
        raise ValueError(f'cannot read {rel}')
    return parsed.values


def method_dir(root: Path, config_rel: str) -> Path:
    # configs/<Name>/... mirrors methods/<Name>/ (layout invariant 1), and the
    # native process runs with that directory as its cwd.
    return root / 'methods' / Path(config_rel).parts[1]


def resolve(base: Path, value) -> Path:
    return Path(os.path.normpath(base / str(value)))


def sample_ids(h5) -> list[str]:
    return sorted(h5['data'].keys(), key=lambda s: (len(s), s))


# ------------------------------------------------------------------ provenance

def marker_time(record: dict, field: str):
    """Seconds since the epoch for a marker's "started"/"finished", or None
    (run_matrix.py::marker_time)."""
    value = record.get(f'{field}_epoch')
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    text = record.get(field)
    if isinstance(text, str):
        try:
            return datetime.strptime(text, ISO_FORMAT).timestamp()
        except ValueError:
            pass
    return None


def run_record(root: Path, pair: dict) -> dict:
    """{'status', 'detail', 'machine', 'since', 'train_started'}: 'ok' when
    some machine's markers record both stages for the configs as they are now,
    with the inference started after the training finished; 'since' is then
    that inference's start time and 'train_started' its training's. Otherwise
    'stale' (markers, none current) or 'unverified' (no marker for this arm
    anywhere)."""
    key = f"{pair['category']}__{pair['example']}__{pair['method']}"
    current = {}
    for stage in ('train', 'infer'):
        try:
            current[stage] = hashlib.sha256((root / pair[stage]).read_bytes()).hexdigest()
        except OSError as exc:
            return dict(status='error', detail=f'cannot read {pair[stage]} ({exc})', machine='', since=None,
                        train_started=None)
    good, stale = [], []
    for done in sorted((root / CAMPAIGN).glob('*/done')):
        markers = {stage: done / f'{key}__{stage}' for stage in ('train', 'infer')}
        if not any(m.exists() for m in markers.values()):
            continue
        records, problems = {}, []
        for stage, marker in markers.items():
            if not marker.exists():
                problems.append(f'no {stage} marker')
                continue
            try:
                record = json.loads(marker.read_text(encoding='utf-8'))
            except (OSError, ValueError):  # the old bash runner's markers are empty
                record = None
            if not isinstance(record, dict):
                problems.append(f'{stage} marker is not a JSON record')
            elif record.get('sha256') != current[stage]:
                problems.append(f'{stage} ran with another version of its config')
            else:
                records[stage] = record
        if not problems:
            trained = marker_time(records['train'], 'finished')
            ran = marker_time(records['infer'], 'started')
            if ran is None:
                ran = marker_time(records['infer'], 'finished')
            if trained is None or ran is None:
                problems.append('marker times cannot be read')
            elif ran + 1.0 < trained:
                problems.append('inference started before the current training finished')
            else:
                good.append((ran, done.parent.name, marker_time(records['train'], 'started')))
                continue
        stale.append(f"{done.parent.name}: {', '.join(problems)}")
    if good:
        since, machine, started = max(good, key=lambda g: g[:2])
        return dict(status='ok', detail='', machine=machine, since=since, train_started=started)
    if stale:
        return dict(status='stale', detail='; '.join(stale), machine='', since=None, train_started=None)
    return dict(status='unverified', detail=f'no run_matrix.py marker under {CAMPAIGN.as_posix()}/*/done '
                '(--no-provenance scores it anyway)', machine='', since=None, train_started=None)


def older_than(paths, since) -> list:
    """The files among `paths` last written before `since` (None: no check)."""
    if since is None:
        return []
    return sorted({p for p in paths if p.stat().st_mtime < since - MTIME_SLACK})


# --------------------------------------------------------------------- scoring

class Acc:
    """Running sums for one channel of one sample."""
    __slots__ = ('sse', 'ssg', 'sg', 'n', 'finite')

    def __init__(self):
        self.sse = self.ssg = self.sg = 0.0
        self.n = 0
        self.finite = True

    def add(self, np, pred, gt):
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        if not np.isfinite(pred).all():
            self.finite = False
            return
        d = pred - gt
        self.sse += float(np.dot(d.ravel(), d.ravel()))
        self.ssg += float(np.dot(gt.ravel(), gt.ravel()))
        self.sg += float(gt.sum())
        self.n += gt.size

    def values(self):
        sst = self.ssg - self.sg * self.sg / self.n if self.n else 0.0
        return {
            'rel_l2': math.sqrt(self.sse / self.ssg) if self.ssg > 0 else None,
            'nrmse': math.sqrt(self.sse / sst) if sst > 0 else None,
            'r2': 1.0 - self.sse / sst if sst > 0 else None,
        }


def macro(values):
    defined = [v for v in values if v is not None]
    return sum(defined) / len(defined) if defined else None


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def median(values):
    if not values:
        return None
    v = sorted(values)
    mid = len(v) // 2
    return v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2


def window(T: int, start: int):
    """GT frame range scored for a file with T frames."""
    return (0, 1) if T == 1 else (start, T)


def score_arm(np, h5py, root: Path, pair: dict, start: int, since=None, free=None) -> dict:
    """Score one arm. `since`: the inference start time every prediction file
    must postdate (None: not checked). `free`: (node-type row, node-type codes)
    for the free-node columns, or None."""
    cfg = parse_config(root, pair['infer'])
    base = method_dir(root, pair['infer'])
    lsh = pair['method'] == 'lsh_vae'
    out = {'example': pair['example'], 'method': pair['method'], 'status': 'ok', 'detail': ''}
    if lsh:
        gt_path = resolve(base, cfg['dataset_dir'])
        pred_files = [resolve(base, cfg['output_dir']) / 'reconstructions.h5']
        n_out = int(cfg['num_var'])
        gt_row0 = int(cfg.get('field_start_row', 3))
    else:
        gt_path = resolve(base, cfg['infer_dataset'])
        pred_dir = resolve(base, cfg['inference_output_dir'])
        pred_files = sorted(Path(p) for p in glob.glob(str(pred_dir / 'rollout_sample*_steps*.h5')))
        n_out = int(cfg['output_var'])
        gt_row0 = 3
    out['prediction'] = str(pred_files[0].parent) if pred_files else ''

    with h5py.File(gt_path, 'r') as gt_h5:
        ids = sample_ids(gt_h5)
        T = int(gt_h5['data'][ids[0]]['nodal_data'].shape[1])
        t0, t1 = window(T, start)
        out.update(samples=len(ids), frames=f'{t0}..{t1 - 1}' if T > 1 else 'static')

        # sample id -> (h5 path, dataset path inside it, first row, frame offset)
        preds = {}
        if lsh:
            if not pred_files[0].is_file():
                out.update(status='missing', detail=f'no {pred_files[0].name}')
                return out
            with h5py.File(pred_files[0], 'r') as f:
                for sid in f['data'].keys():
                    preds[sid] = (pred_files[0], f'data/{sid}/nodal_field', 0)
        else:
            seen = {}
            for path in pred_files:
                m = re.match(r'rollout_sample(.+)_steps(\d+)\.h5$', path.name)
                if not m or '_vaesample' in path.name:
                    continue
                seen.setdefault(m.group(1), []).append((int(m.group(2)), path))
            want = 0 if T == 1 else T - 1
            for sid, files in seen.items():
                # A static rollout file is steps1 (MGN, operators) or steps0
                # (Transolver); an AR one must cover every frame.
                ok = [p for s, p in files if (s <= 1 if T == 1 else s == want)]
                if len(files) > 1:
                    out['detail'] += f'sample {sid} has {len(files)} rollout files; '
                if ok:
                    preds[sid] = (max(ok, key=lambda p: p.stat().st_mtime), f'data/{sid}/nodal_data', 3)
            if not preds and not seen:
                out.update(status='missing', detail='no rollout files')
                return out

        old = older_than([preds[s][0] for s in ids if s in preds], since)
        if old:
            out.update(status='stale', detail=out['detail'] + f'{len(old)} prediction file(s) predate the '
                       f'recorded inference, so an earlier run wrote them (e.g. {old[0].name})')
            return out
        absent = [s for s in ids if s not in preds]
        if absent:
            out.update(status='incomplete', detail=out['detail'] + f'{len(absent)}/{len(ids)} samples '
                       f'missing or short (e.g. {absent[:3]})')
            return out

        per_sample = {k: [] for k in METRICS}
        per_free = {k: [] for k in FREE_METRICS}
        chan = [dict(sse=0.0, ssg=0.0, sg=0.0, n=0) for _ in range(n_out)]
        per_chan = [{k: [] for k in METRICS} for _ in range(n_out)]
        nonfinite = 0
        for sid in ids:
            path, key, row0 = preds[sid]
            gt_ds = gt_h5['data'][sid]['nodal_data']
            mask = None
            if free is not None:
                row, codes = free
                if gt_ds.shape[0] != row + 1:
                    out['detail'] += (f'node-type row {row} is not the last of {gt_ds.shape[0]} rows, '
                                      'free-node columns skipped; ')
                    free = None
                else:
                    types = np.rint(np.asarray(gt_ds[row, 0, :])).astype(np.int64)
                    mask = np.isin(types, codes)
                    if not mask.any():
                        out['detail'] += f'sample {sid} has no free nodes, free-node columns skipped; '
                        free, mask = None, None
            with h5py.File(path, 'r') as f:
                ds = f[key]
                if not lsh:
                    written = int(f.attrs.get('output_var', -1))
                    if written != n_out:
                        out.update(status='mismatch', detail=f'{path.name}: output_var attr {written}, config {n_out}')
                        return out
                P, Tp, Np = ds.shape
                if Np != gt_ds.shape[2] or P < row0 + n_out:
                    out.update(status='mismatch', detail=f'{path.name}: shape {ds.shape} vs GT {gt_ds.shape}')
                    return out
                if T == 1:
                    p_frames = slice(Tp - 1, Tp)
                elif Tp != T:
                    out.update(status='mismatch', detail=f'{path.name}: {Tp} frames, GT has {T}')
                    return out
                else:
                    p_frames = slice(t0, t1)
                accs, free_accs = [], []
                for c in range(n_out):
                    p = ds[row0 + c, p_frames, :]
                    g = gt_ds[gt_row0 + c, t0:t1, :]
                    a = Acc()
                    a.add(np, p, g)
                    accs.append(a)
                    if mask is not None:
                        af = Acc()
                        af.add(np, p[:, mask], g[:, mask])
                        free_accs.append(af)
            if not all(a.finite for a in accs):
                nonfinite += 1
                continue
            vals = [a.values() for a in accs]
            for k in METRICS:
                per_sample[k].append(macro([v[k] for v in vals]))
            if free_accs:
                free_vals = [a.values() for a in free_accs]
                for k in FREE_METRICS:
                    per_free[k].append(macro([v[k] for v in free_vals]))
            for c, (a, v) in enumerate(zip(accs, vals)):
                for k in METRICS:
                    per_chan[c][k].append(v[k])
                for s in ('sse', 'ssg', 'sg', 'n'):
                    chan[c][s] += getattr(a, s)

    if nonfinite:
        out.update(status='non-finite', detail=f'{nonfinite}/{len(ids)} samples have NaN/Inf predictions')
        return out
    for k in METRICS:
        vals = [v for v in per_sample[k] if v is not None]
        out[k] = mean(vals)
        out[k + '_median'] = median(vals)
    if free is not None:
        for k in FREE_METRICS:
            out[k + '_free'] = mean(per_free[k])
    pooled = []
    for c, s in enumerate(chan):
        a = Acc()
        a.sse, a.ssg, a.sg, a.n = s['sse'], s['ssg'], s['sg'], s['n']
        pooled.append(a.values())
        for k in METRICS:
            out[f'{k}[{c}]'] = mean(per_chan[c][k])
    for k in METRICS:
        out[k + '_pooled'] = macro([v[k] for v in pooled])
    return out


# --------------------------------------------------------------------- ranking

def assign_ranks(rows, key, name, higher_better=False) -> int:
    """Rank one example's arms on `key` into r[name] and r[name + '_norm'].
    Exact ties share their average rank; every arm that is not ok or has no
    value shares last place, (k + 1 + N) / 2. Returns k, the arms with a
    value; with k == 0 nothing is ranked."""
    n = len(rows)
    scored = [r for r in rows if r['status'] == 'ok' and r.get(key) is not None]
    k = len(scored)
    if not k:
        return 0
    scored.sort(key=lambda r: -r[key] if higher_better else r[key])
    i = 0
    while i < k:
        j = i
        while j + 1 < k and scored[j + 1][key] == scored[i][key]:
            j += 1
        for r in scored[i:j + 1]:
            r[name] = (i + j + 2) / 2
        i = j + 1
    placed = {id(r) for r in scored}
    for r in rows:
        if id(r) not in placed:
            r[name] = (k + 1 + n) / 2
        r[name + '_norm'] = (r[name] - 1) / (n - 1) if n > 1 else 0.0
    return k


def overall(rows, columns, title) -> list[str]:
    """Mean normalised rank per method over example groups (GROUPS). `columns`
    is ((row key, heading), ...); the first one orders the table and decides
    wins."""
    norm_keys = [k for k, _ in columns]
    groups = {}
    for r in rows:
        if norm_keys[0] in r:
            groups.setdefault(GROUPS.get(r['example'], r['example']), {}).setdefault(r['method'], []).append(r)
    methods = sorted({r['method'] for r in rows})
    runs_in = {m: len({GROUPS.get(r['example'], r['example']) for r in rows if r['method'] == m})
               for m in methods}
    per = {m: {k: [] for k in norm_keys} for m in methods}
    wins = dict.fromkeys(methods, 0)
    for mine in groups.values():
        first = {}
        for m, rs in mine.items():
            for k in norm_keys:
                v = mean([r.get(k) for r in rs])
                if v is not None:
                    per[m][k].append(v)
            first[m] = mean([r[norm_keys[0]] for r in rs])
        best = min(first.values())
        for m, v in first.items():
            if v <= best + 1e-9:
                wins[m] += 1
    summary = sorted((mean(per[m][norm_keys[0]]) if per[m][norm_keys[0]] else math.inf, m) for m in methods)
    head = ''.join(f'{h:>14}' for _, h in columns)
    lines = ['', title, f"  {'method':<16}{head}{'groups':>9}{'wins':>6}"]
    for _, m in summary:
        vals = ''.join(fmt(mean(per[m][k]), 14) for k in norm_keys)
        lines.append(f"  {m:<16}{vals}{len(per[m][norm_keys[0]]):>5}/{runs_in[m]:<3}{wins[m]:>6}")
    return lines


def fmt(v, width=10):
    if v is None:
        return '-'.rjust(width)
    return f'{v:.4g}'.rjust(width)


def fmt_rank(r, name='rank'):
    return f"{r[name]:g}" if name in r else '-'


def fmt_updates(n):
    if not isinstance(n, int):
        return '-'
    for div, unit in ((1e6, 'M'), (1e3, 'k')):
        if n >= div:
            return f'{n / div:.4g}{unit}'
    return str(n)


def ranked_header(k, n):
    if k == 0:
        return f'none of {n} scored, not ranked'
    if k == n:
        return f'all {n} scored'
    return f'{k} of {n} scored, the rest share last place'


# ----------------------------------------------------------------- convergence

def training_stages(root: Path, pair: dict) -> list[dict]:
    """The training stages of an arm, from its train config: stage name, the
    tag its log lines carry, log path, planned epochs, and `val_every` where
    the log repeats the last validation loss between validations (Transolver's
    multi-GPU trainers; elsewhere a skipped validation writes no number)."""
    cfg = parse_config(root, pair['train'])
    base = method_dir(root, pair['train'])
    mode = str(cfg.get('mode', 'train'))

    def stage(name, tag, prefix='', epochs_key='training_epochs', val_every=None):
        # The two-stage pipelines take <stage>_<key> over <key> (train_pipeline.py::build_stage_config).
        log = cfg.get(prefix + 'log_file_dir', cfg.get('log_file_dir'))
        epochs = cfg.get(prefix + epochs_key, cfg.get(epochs_key))
        return dict(stage=name, tag=tag, log=resolve(base, log) if log else None,
                    planned=int(epochs) if epochs is not None else None, val_every=val_every)

    method = pair['method']
    if method in ('lsh_vae', 'sdfflow'):
        second = 'lc' if method == 'lsh_vae' else 'fm'
        names = {'train': ('vae', second), 'train_vae': ('vae',), f'train_{second}': (second,)}.get(mode, ())
        return [stage(n, None, prefix=f'{n}_') for n in names]
    if method == 'chi_mgnflow':
        if mode == 'train':  # one log: [AE] for ae_epochs, then [Prior] for training_epochs
            return [stage('ae', 'AE', epochs_key='ae_epochs'), stage('prior', 'Prior')]
        return [stage('ae', 'AE')] if mode == 'train_ae' else [stage('prior', 'Prior')]
    val_every = int(cfg.get('val_interval', 1)) if method == 'transolver3' else None
    return [stage('train', None, val_every=val_every)]


def read_log(path: Path) -> dict:
    """{tag: [(epoch, train loss, val loss), ...]} for the last run in a log.
    A `==== Run` header (the logs SimulGenVAE and SDFFlow append to) or an
    epoch that does not increase starts a new run; a missing number is None.
    After a `==== Resume` line the run goes on: the first epoch logged after
    it drops that tag's entries from its own number on (logged after the
    state was saved, they are redone), so a training that was cut off and
    continued reads as the one run it is."""
    def number(regex, text):
        m = regex.search(text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    runs, resumed = {}, set()
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('==== Run'):
                runs, resumed = {}, set()
                continue
            m = RESUME_LINE.match(line)
            if m:
                resumed.add(m.group('tag'))
                continue
            m = EPOCH_LINE.match(line.rstrip('\r\n'))
            if not m:
                continue
            epoch, rest, tag = int(m.group('epoch')), m.group('rest'), m.group('tag')
            run = runs.setdefault(tag, [])
            if tag in resumed:
                resumed.discard(tag)
                run[:] = [e for e in run if e[0] < epoch]
            elif run and epoch <= run[-1][0]:
                run.clear()
            run.append((epoch, number(TRAIN_FIELD, rest), number(VAL_FIELD, rest)))
    return runs


def change(a, b):
    """b / a - 1, or None."""
    return b / a - 1.0 if a is not None and b is not None and a > 0 else None


def curve(points, planned: int, val_every=None) -> dict:
    """Window medians and flags for one stage's (epoch, train, val) points."""
    medians = {}
    nonfinite = False
    for name, col in (('train', 1), ('val', 2)):
        bins = [[] for _ in range(BINS)]
        for p in points:
            e, v = p[0], p[col]
            if v is None or (col == 2 and val_every and e % val_every and e != planned - 1):
                continue
            if not math.isfinite(v):
                nonfinite = True
            elif 0 <= e < planned:
                bins[e * BINS // planned].append(v)
        medians[name] = [median(b) for b in bins]
    val = medians['val']
    # 55-60% -> 75-80% of the planned epochs.
    lo, hi = int(0.6 * BINS) - 1, int(0.8 * BINS) - 1
    out = dict(val_60_80=change(val[lo], val[hi]), train_60_80=change(medians['train'][lo], medians['train'][hi]),
               tail=None, best_window='', final_val=None, nonfinite=nonfinite)
    have = [(v, i) for i, v in enumerate(val) if v is not None]
    if have:
        best, i = min(have)
        out.update(tail=change(best, val[-1]), best_window=f'{i * 100 // BINS}-{(i + 1) * 100 // BINS}%')
    finals = [p[2] for p in points if p[2] is not None and math.isfinite(p[2])
              and not (val_every and p[0] % val_every and p[0] != planned - 1)]
    out['final_val'] = finals[-1] if finals else None
    return out


def arm_convergence(root: Path, pair: dict, rec: dict) -> list[dict]:
    """One row per training stage of an arm; never raises."""
    head = {'category': pair['category'], 'example': pair['example'], 'method': pair['method'],
            'provenance': rec['status']}
    try:
        stages = training_stages(root, pair)
    except Exception as exc:
        return [dict(head, stage='?', flags='?', detail=f'{type(exc).__name__}: {exc}')]
    rows = []
    for st in stages:
        row = dict(head, stage=st['stage'], epochs_planned=st['planned'], flags='', detail='')
        rows.append(row)
        try:
            log = st['log']
            if log is not None:
                try:
                    row['log'] = log.relative_to(root).as_posix()
                except ValueError:
                    row['log'] = str(log)
            if log is None or not log.is_file():
                row.update(flags='?', detail='no log_file_dir' if log is None else 'no training log')
                continue
            points = read_log(log).get(st['tag'], [])
            if not points:
                row.update(flags='?', detail='the log has no epoch lines' + (f" [{st['tag']}]" if st['tag'] else ''))
                continue
            planned = st['planned']
            if not planned or planned <= 0:
                row.update(flags='?', detail='no planned epoch count in the train config')
                continue
            c = curve(points, planned, st['val_every'])
            last = points[-1][0]
            row.update(epochs_logged=last + 1, **{k: v for k, v in c.items() if k != 'nonfinite'})
            flags = set()
            if c['val_60_80'] is not None and c['val_60_80'] < FALLING:
                flags.add('F')
            if pair['category'] == 'deterministic' and c['tail'] is not None and c['tail'] > RISING:
                flags.add('R')
            if last < planned - 1:
                flags.add('I')
            if c['nonfinite']:
                flags.add('N')
            if last > planned - 1:
                flags.add('X')
            started = rec.get('train_started')
            if started is not None and log.stat().st_mtime < started - MTIME_SLACK:
                flags.add('O')
            row['flags'] = ''.join(f for f in FLAG_ORDER if f in flags)
        except Exception as exc:  # one unreadable log must not hide the rest
            row.update(flags='?', detail=f'{type(exc).__name__}: {exc}')
    return rows


def pct(v, width):
    return (f'{v * 100:+.1f}%' if v is not None else '-').rjust(width)


def conv_summary(stages) -> dict:
    """The rank-table cell: the stage whose validation loss fell most from 60%
    to 80%, and the flags of every stage."""
    have = [s for s in stages if s.get('val_60_80') is not None]
    pick = min(have, key=lambda s: s['val_60_80']) if have else (stages[0] if stages else {})
    flags = ''.join(f for f in FLAG_ORDER if any(f in s.get('flags', '') for s in stages))
    cell = pct(pick.get('val_60_80'), 0)
    if len(stages) > 1 and pick.get('val_60_80') is not None:
        cell += f" {pick['stage']}"
    if flags:
        cell += f' {flags}'
    return {'conv_val_60_80': pick.get('val_60_80'), 'conv_stage': pick.get('stage', ''),
            'conv_flags': flags, 'conv_cell': cell}


def convergence_lines(conv_rows) -> list[str]:
    lines = ['', 'convergence (training logs, last run; informational, nothing is ranked on it)',
             f'  windows of {100 // BINS}% of the planned epochs, median loss per window; '
             'val/train 60-80% = change from the 55-60% to the 75-80% window,',
             '  tail = last window vs the best one. Under cosine part of a late drop is the LR annealing: '
             'F is a screen, a doubled budget is the test.',
             f'  F val 60-80% below {FALLING * 100:+.0f}%   R deterministic (last epoch kept) and tail above '
             f'{RISING * 100:+.0f}%   I log ends before the planned last epoch',
             '  N NaN/Inf loss   X more epochs than planned   O log older than the recorded training '
             '(reused stage or leftover)   ? no log',
             f"  {'example':<26}{'method':<16}{'stage':<7}{'epochs':>11}{'val 60-80%':>12}{'train 60-80%':>14}"
             f"{'tail':>9}{'best':>10}{'final val':>12}  flags"]
    for r in conv_rows:
        planned = r.get('epochs_planned')
        epochs = f"{r.get('epochs_logged', '-')}/{planned if planned is not None else '-'}"
        prov = f"  [{r['provenance']}]" if r.get('provenance') not in (None, 'ok') else ''
        lines.append(f"  {r['category'] + '/' + r['example']:<26}{r['method']:<16}{r['stage']:<7}{epochs:>11}"
                     f"{pct(r.get('val_60_80'), 12)}{pct(r.get('train_60_80'), 14)}{pct(r.get('tail'), 9)}"
                     f"{r.get('best_window') or '-':>10}{fmt(r.get('final_val'), 12)}  {r['flags']}{prov}")
        if r.get('detail'):
            lines.append(f"  {'':<26}{'':<16}note: {r['detail']}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--manifest', type=Path, default=HERE / 'manifest.json')
    ap.add_argument('--examples', nargs='*', help='category/example to score (default: all)')
    ap.add_argument('--out', type=Path, help='directory for ranking.txt / ranking.csv')
    ap.add_argument('--no-provenance', action='store_true',
                    help='score outputs without run_matrix.py markers (by-hand runs); no staleness check')
    args = ap.parse_args(argv)
    import h5py
    import numpy as np

    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    cases = {(c['category'], c['key']): c for c in manifest['cases']}
    wanted = set(args.examples or [])
    pairs = [p for p in manifest['pairs'] if not wanted or f"{p['category']}/{p['example']}" in wanted]

    def provenance(p):
        if args.no_provenance:
            return dict(status='ok', detail='', machine='', since=None, train_started=None)
        return run_record(args.root, p)

    def unscored(p, rec):
        return {'example': p['example'], 'method': p['method'], 'status': rec['status'], 'detail': rec['detail']}

    lines, rows, conv_rows = [], [], []
    det = [p for p in pairs if p['category'] == 'deterministic']
    examples = sorted({p['example'] for p in det}, key=lambda e: (len(e.split('_')[0]), e))
    for ex in examples:
        arms = [p for p in det if p['example'] == ex]
        start = max([1] + [LSH_OBSERVED.get(ex, 1) for p in arms if p['method'] == 'lsh_vae'])
        free = None
        if ex in FREE_NODES:
            case = cases[('deterministic', ex)]
            free = (3 + len(case['output']) + len(case['conditioner']), FREE_NODES[ex])
        ex_rows = []
        for p in arms:
            rec = provenance(p)
            try:
                r = (score_arm(np, h5py, args.root, p, start, since=rec['since'], free=free)
                     if rec['status'] == 'ok' else unscored(p, rec))
            except Exception as exc:  # one broken arm must not hide the rest
                r = {'example': ex, 'method': p['method'], 'status': 'error',
                     'detail': f'{type(exc).__name__}: {exc}'}
            r.update(category='deterministic', machine=rec['machine'], updates=p.get('updates'),
                     updates_note=p.get('updates_note', ''))
            stages = arm_convergence(args.root, p, rec)
            r.update(conv_summary(stages))
            conv_rows += stages
            ex_rows.append(r)
        k = assign_ranks(ex_rows, RANK_KEY, 'rank')
        assign_ranks(ex_rows, 'nrmse', 'rank_nrmse')
        frames = next((r['frames'] for r in ex_rows if r.get('frames')), '?')
        n = next((r['samples'] for r in ex_rows if r.get('samples')), None)
        note = f'; only {n} held-out sample(s), a close gap is not a difference' if n and n < SMALL_HELD_OUT else ''
        free_head = f"{'free rel_l2':>12}{'free nrmse':>11}" if free else ''
        lines += ['', f'deterministic/{ex}  (n={n if n else "?"}, scored frames {frames}; '
                      f'{ranked_header(k, len(ex_rows))}; ranked on {RANK_KEY}, lower is better{note})',
                  f"  {'rank':<6}{'method':<16}{'rel_l2':>10}{'median':>10}{'nrmse':>10}{'r2':>10}{free_head}"
                  f"{'updates':>9}{'val 60-80%':>18}  per-channel rel_l2"]
        if free:
            lines.append(f"  free = node types {', '.join(map(str, free[1]))} (GT row {free[0]}, frame 0); "
                         'shown, not ranked on')
        order = sorted(ex_rows, key=lambda r: (r['status'] != 'ok' or r.get(RANK_KEY) is None,
                                               r.get('rank', math.inf), r['method']))
        for r in order:
            if r['status'] == 'ok' and r.get(RANK_KEY) is not None:
                chans = ' '.join(fmt(r.get(f'rel_l2[{c}]'), 0) for c in range(16) if f'rel_l2[{c}]' in r)
                free_vals = f"{fmt(r.get('rel_l2_free'), 12)}{fmt(r.get('nrmse_free'), 11)}" if free else ''
                lines.append(f"  {fmt_rank(r):<6}{r['method']:<16}{fmt(r['rel_l2'])}{fmt(r['rel_l2_median'])}"
                             f"{fmt(r['nrmse'])}{fmt(r['r2'])}{free_vals}{fmt_updates(r['updates']):>9}"
                             f"{r['conv_cell']:>18}  {chans}")
                if r['detail']:
                    lines.append(f"  {'':<6}{'':<16}note: {r['detail'].rstrip('; ')}")
            else:
                lines.append(f"  {fmt_rank(r):<6}{r['method']:<16}{r['status']}: {r['detail'].rstrip('; ')}")
        notes = sorted({f"{r['method']}: {r['updates_note']}" for r in ex_rows if r.get('updates_note')})
        lines += [f'  updates, {n_}' for n_ in notes]
        rows += ex_rows

    prob = [p for p in pairs if p['category'] == 'probabilistic']
    for ex in sorted({p['example'] for p in prob}):
        ex_rows = []
        for p in prob:
            if p['example'] != ex:
                continue
            rec = provenance(p)
            r = {'category': 'probabilistic', 'example': ex, 'method': p['method'], 'machine': rec['machine'],
                 'updates': p.get('updates'), 'updates_note': p.get('updates_note', '')}
            try:
                cfg = parse_config(args.root, p['infer'])
                d = resolve(method_dir(args.root, p['infer']), cfg['inference_output_dir'])
                r['prediction'] = str(d)
                f = d / 'spread_metrics.json'
                if rec['status'] != 'ok':
                    r.update(status=rec['status'], detail=rec['detail'])
                elif not f.is_file():
                    r.update(status='missing', detail='no spread_metrics.json')
                elif older_than([f], rec['since']):
                    r.update(status='stale', detail='spread_metrics.json predates the recorded inference, '
                             'so an earlier run wrote it')
                else:
                    m = json.loads(f.read_text(encoding='utf-8'))
                    r.update(status='ok', detail='', crps_norm=m.get('crps_norm'), sd_ratio=m.get('sd_ratio'),
                             spread_skill=m.get('spread_skill'))
            except Exception as exc:  # one broken arm must not hide the rest
                r.update(status='error', detail=f'{type(exc).__name__}: {exc}')
            stages = arm_convergence(args.root, p, rec)
            r.update(conv_summary(stages))
            conv_rows += stages
            ex_rows.append(r)
        k = assign_ranks(ex_rows, 'crps_norm', 'rank')
        lines += ['', f'probabilistic/{ex}  ({ranked_header(k, len(ex_rows))}; ranked on crps_norm, lower is '
                      'better; full table: score_spread.py)',
                  f"  {'rank':<6}{'method':<16}{'crps_norm':>10}{'sd_ratio':>10}{'skill':>10}{'updates':>9}"
                  f"{'val 60-80%':>18}"]
        order = sorted(ex_rows, key=lambda r: (r['status'] != 'ok' or r.get('crps_norm') is None,
                                               r.get('rank', math.inf), r['method']))
        for r in order:
            if r['status'] == 'ok' and r.get('crps_norm') is not None:
                lines.append(f"  {fmt_rank(r):<6}{r['method']:<16}{fmt(r['crps_norm'])}{fmt(r['sd_ratio'])}"
                             f"{fmt(r['spread_skill'])}{fmt_updates(r['updates']):>9}{r['conv_cell']:>18}")
            else:
                lines.append(f"  {fmt_rank(r):<6}{r['method']:<16}{r['status']}: {r['detail']}")
        notes = sorted({f"{r['method']}: {r['updates_note']}" for r in ex_rows if r.get('updates_note')})
        lines += [f'  updates, {n_}' for n_ in notes]
        rows += ex_rows

    det_rows = [r for r in rows if r['category'] == 'deterministic']
    if det_rows:
        lines += overall(det_rows, (('rank_norm', 'nrank rel_l2'), ('rank_nrmse_norm', 'nrank nrmse')),
                         'overall (deterministic): mean normalised rank, 0 = best and 1 = last in each group; '
                         'ex3_full + ex3_mid count once as "ex3 (CRM)"')
        lines += ['  an arm that is not ok counts as tied last in its example; groups = ranked / run in',
                  '  a method with fewer groups is compared on fewer, not harder, cases']
    prob_rows = [r for r in rows if r['category'] == 'probabilistic']
    if prob_rows:
        lines += overall(prob_rows, (('rank_norm', 'nrank crps'),),
                         'overall (probabilistic): mean normalised rank on crps_norm')

    for p in pairs:
        if p['category'] not in ('deterministic', 'probabilistic'):
            conv_rows += arm_convergence(args.root, p, provenance(p))
    if conv_rows:
        lines += convergence_lines(conv_rows)

    text = '\n'.join(lines).lstrip('\n') + '\n'
    print(text, end='')
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'ranking.txt').write_text(text, encoding='utf-8')
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys and k != 'conv_cell']
        with open(args.out / 'ranking.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        conv_keys = ['category', 'example', 'method', 'stage', 'log', 'epochs_logged', 'epochs_planned',
                     'val_60_80', 'train_60_80', 'tail', 'best_window', 'final_val', 'flags', 'provenance', 'detail']
        with open(args.out / 'convergence.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=conv_keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(conv_rows)
        print(f'wrote {args.out / "ranking.txt"}, ranking.csv and convergence.csv')
    bad = [r for r in rows if r['status'] != 'ok']  # convergence never changes it
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
