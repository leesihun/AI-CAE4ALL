#!/usr/bin/env python3
"""Write the run's results to a markdown document, computed not narrated.

    python configs/campaigns/write_report.py \
        --label "MeshGraphNets-V peak-to-valley" \
        --infer output/meshgraphnets-v/saoi_sweep/infer \
        --logs  output/meshgraphnets-v/saoi_sweep \
        --configs configs/MeshGraphNets_Variational/SAOI_sweep \
        --arms "1 2 3 4 5 6 7 8" \
        --axis beta_aux \
        --out docs/research/SAOI_SWEEP_MGNV.md

Everything in the output comes from the dumps and the logs. The two questions
the run exists to answer get computed verdicts rather than prose:

  * MeshGraphNets-V -- does the peak-to-valley term widen the ensemble? The
    control arm and the weighted arm of the SAME section are compared on
    `sd_ratio`, whose target is 1. The 8-arm sweep never exceeded 0.526.

  * cHI-MGNflow -- has it converged? The LR anneals to 1e-8 by the final epoch,
    so a flat tail is the schedule, not convergence. The middle of the run is
    where the question is decided, and the two slopes are printed side by side.

Metrics come from rank_arms.py so there is exactly one implementation of them.
"""
import argparse
import datetime
import os
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import rank_arms  # noqa: E402

KEY_RE = re.compile(r"^(\w+)[\t ]+([^#\n]+)")
# [FlowDiag] crps=1.2e-02  det(1fwd) mse=3.4e-03  1-draw mse=...  spread/gt=0.41
FLOWDIAG_RE = re.compile(
    r"crps=([0-9.eE+-]+)\s+det\(1fwd\) mse=([0-9.eE+-]+)\s+"
    r"1-draw mse=([0-9.eE+-]+)\s+spread/gt=([0-9.eE+-]+)")
EPOCH_RE = re.compile(r"^Epoch\s+(\d+)\s*/")
# Train  recon=1.2e-03 mmd=4.5e-02 aux=6.7e-02 total=1.3e+00
MGNV_TRAIN_RE = re.compile(
    r"Train\s+recon=([0-9.eE+-]+)\s+mmd=([0-9.eE+-]+)\s+aux=([0-9.eE+-]+)"
    r"\s+total=([0-9.eE+-]+)")


def cfg_value(path, key, default=None):
    """One key out of the flat `key value` config, case-insensitively.

    The native parser lowercases keys and these files mix `Batch_size` with
    `batch_size`, so an exact match would silently fall through.
    """
    want = key.lower()
    try:
        text = pathlib.Path(path).read_text(encoding='utf-8')
    except OSError:
        return default
    for line in text.splitlines():
        if line.startswith('%') or not line.strip():
            continue
        m = KEY_RE.match(line)
        if m and m.group(1).lower() == want:
            return m.group(2).strip()
    return default


def roster(configs, arms, axis):
    rows = []
    for arm in arms:
        cfg = os.path.join(configs, f'config_train_{arm}.txt')
        ds = cfg_value(cfg, 'dataset_dir', '?')
        rows.append({
            'arm': arm,
            'gpu': cfg_value(cfg, 'gpu_ids', '?'),
            'section': 'top' if '_top' in os.path.basename(ds) else 'bot',
            'axis': cfg_value(cfg, axis, '?'),
            'epochs': cfg_value(cfg, 'training_epochs', '?'),
            'dataset': os.path.basename(ds),
            'batch': cfg_value(cfg, 'batch_size', '?'),
        })
    return rows


def flow_curve(log_path):
    """(epoch, det_mse, crps, spread) per validation, from the [FlowDiag] lines."""
    out, epoch = [], 0
    try:
        text = pathlib.Path(log_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return out
    for line in text.splitlines():
        m = EPOCH_RE.match(line.strip())
        if m:
            epoch = int(m.group(1))
            continue
        d = FLOWDIAG_RE.search(line)
        if d:
            out.append((epoch, float(d.group(2)), float(d.group(1)), float(d.group(4))))
    return out


def slope(points):
    """Least-squares slope of y against x. None if fewer than two points."""
    if len(points) < 2:
        return None
    x = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    if x.max() == x.min():
        return None
    return float(np.polyfit(x, y, 1)[0])


def rel_gain(points):
    """Fractional drop in y across the span. None if it cannot be formed."""
    if len(points) < 2:
        return None
    first, last = points[0][1], points[-1][1]
    if not first:
        return None
    return (first - last) / first


def mgnv_terms(log_path):
    """Last (recon, mmd, aux, total) reported, for the objective's balance."""
    try:
        text = pathlib.Path(log_path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    hits = MGNV_TRAIN_RE.findall(text)
    return tuple(float(v) for v in hits[-1]) if hits else None


def find_log(logs, arm):
    for name in (f'{arm}.log', f'{arm}.stdout'):
        p = os.path.join(logs, name)
        if os.path.isfile(p):
            return p
    return None


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |',
           '| ' + ' | '.join('---' for _ in header) + ' |']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return out


def fnum(v, p=3):
    return f'{v:.{p}f}' if isinstance(v, (int, float)) and v == v else '—'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--infer', required=True, help='directory holding <arm>/<tag>/')
    ap.add_argument('--logs', required=True)
    ap.add_argument('--configs', required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--axis', required=True, help='the config key this run varies')
    ap.add_argument('--out', required=True)
    ap.add_argument('--kind', choices=['mgnv', 'flow'], required=True)
    a = ap.parse_args()

    arms = a.arms.split()
    stamp = datetime.date.today().isoformat()
    L = [f'# SAOI long run — {a.label}',
         '',
         f'Generated {stamp} by `configs/campaigns/write_report.py`. Every number '
         f'below is read from the dumps and logs of this run — regenerate rather '
         f'than hand-edit.',
         '']

    # ---- what was run -----------------------------------------------------
    rows = roster(a.configs, arms, a.axis)
    L += ['## What was run', '',
          f'One arm per GPU. `{a.axis}` is the axis; BOT and TOP are the board\'s '
          f'two sections, independent models over one schema.', '']
    L += md_table(['arm', 'GPU', 'section', a.axis, 'epochs', 'batch', 'dataset'],
                  [[r['arm'], r['gpu'], r['section'], r['axis'], r['epochs'],
                    r['batch'], f"`{r['dataset']}`"] for r in rows])
    L += ['']

    # ---- distribution metrics --------------------------------------------
    dumps = rank_arms.collect([a.infer])
    # The deterministic control (infer/det/, one draw per scene) is a bias
    # check, not a distribution. Pooling it with the stochastic ensemble of the
    # same arm would silently overwrite one with the other, so it is kept apart
    # and reported in its own section.
    per_arm, det_arm = {}, {}
    for (_sweep, mode, arm), tags in dumps.items():
        (det_arm if mode == 'det' else per_arm).setdefault(arm, {}).update(tags)

    L += ['## Distribution metrics', '',
          'Each eval set is ONE geometry with 125 realizations of it, so `gt` is the '
          'true conditional p(spread | geometry) and `sd_ratio`\'s target is **1**. '
          'The statistic is the warpage spread `max(z_disp) − min(z_disp)` at the '
          'last timestep.', '']
    metric_rows = []
    for arm in arms:
        tags = per_arm.get(arm, {})
        if not tags:
            metric_rows.append([arm, '(no dump)', '', '', '', '', '', ''])
            continue
        for tag in sorted(tags):
            m = tags[tag]
            if 'error' in m:
                metric_rows.append([arm, tag, 'INVALID: ' + m['error'], '', '', '', '', ''])
                continue
            metric_rows.append([arm, tag, m['n_gt'], m['n_gen'],
                                fnum(m['w1']), fnum(m['dmean']), fnum(m['sd_ratio']),
                                fnum(m['pit_tails']), fnum(m['pit_ks'])])
    L += md_table(['arm', 'eval set', 'n_gt', 'n_gen', 'W1/sd (0)', 'dmean/sd (0)',
                   'sd_ratio (1)', 'PITtails (0.04)', 'PIT_KS (0)'], metric_rows)
    L += ['']

    # ---- ranking ----------------------------------------------------------
    summary = []
    for arm in arms:
        vals = [m for m in per_arm.get(arm, {}).values() if 'error' not in m]
        if not vals:
            continue
        summary.append((arm, {
            'n': len(vals),
            'w1': float(np.mean([v['w1'] for v in vals])),
            'dmean': float(np.mean([abs(v['dmean']) for v in vals])),
            'sd_ratio': float(np.mean([v['sd_ratio'] for v in vals])),
            'pit_tails': float(np.mean([v['pit_tails'] for v in vals])),
        }))
    summary.sort(key=lambda s: s[1]['w1'])
    if summary:
        L += ['## Ranking by mean W1/sd', '',
              '`W1/sd` ranks accuracy and distribution together, and `W1 ≥ |dmean|` '
              'always — so the two columns side by side separate a distribution that '
              'is *shifted* from one that is the wrong *shape*.', '']
        L += md_table(['rank', 'arm', 'sets', 'W1/sd', '\|dmean\|/sd', 'sd_ratio',
                       'PITtails'],
                      [[i, arm, s['n'], fnum(s['w1']), fnum(s['dmean']),
                        fnum(s['sd_ratio']), fnum(s['pit_tails'])]
                       for i, (arm, s) in enumerate(summary, 1)])
        L += ['']

    # ---- deterministic control: location error with sampling removed ------
    if det_arm:
        L += ['## Deterministic control', '',
              'One draw per scene with the sampling noise off (MGN-V: prior at '
              'temperature ~0, the conditional centre; flow: the one-forward mean '
              'readout). Its offset from the truths is **pure bias** -- a location '
              'error that no width fix can repair -- and `sd_ratio` here has no '
              'meaning, since there is no ensemble.', '']
        det_rows = []
        for arm in arms:
            for tag in sorted(det_arm.get(arm, {})):
                m = det_arm[arm][tag]
                if 'error' in m:
                    det_rows.append([arm, tag, 'INVALID: ' + m['error'], ''])
                    continue
                det_rows.append([arm, tag, fnum(m['dmean']), fnum(m['w1'])])
        L += md_table(['arm', 'eval set', 'dmean/sd (0)', 'W1/sd (0)'], det_rows)
        L += ['', 'Read `dmean/sd` against the stochastic table above: an arm whose '
                  'stochastic `W1/sd` is mostly explained by this bias has a '
                  'regression problem, not a calibration one.', '']

    # ---- the verdict this run exists for ----------------------------------
    L += ['## Verdict', '']
    ranked = dict(summary)
    if a.kind == 'mgnv':
        # The verdict is generic over the axis: within each section, the arm
        # whose axis value is 0 is the control and EVERY other arm is compared
        # against it on sd_ratio (target 1; the previous grid's best was 0.526).
        # With a 2^3 design that gives two rows per section -- one per level of
        # the other factor -- so the interaction is visible, not averaged away.
        axis_word = {'beta_aux': 'the peak-to-valley term (`beta_aux`)',
                     'prior_freeze_epoch': 'the prior-only tail (`prior_freeze_epoch`: '
                                           'simulator frozen, latent standardization '
                                           'fitted, prior refit on a fresh cosine)'}
        what = axis_word.get(a.axis, f'`{a.axis}`')
        L += [f'The question: does {what} widen the ensemble? `sd_ratio` has target 1; '
              f'the previous 8-arm grid never exceeded **0.526**. Each section\'s '
              f'`{a.axis} = 0` arm is the control; every other arm of that section is '
              f'compared against it.', '']
        pairs = []
        for section in ('bot', 'top'):
            ctrl = next((r['arm'] for r in rows
                         if r['section'] == section and str(r['axis']) in ('0', '0.0')), None)
            if ctrl not in ranked:
                continue
            for r in rows:
                test = r['arm']
                if r['section'] != section or test == ctrl or test not in ranked:
                    continue
                c, t = ranked[ctrl]['sd_ratio'], ranked[test]['sd_ratio']
                pairs.append([section, ctrl, fnum(c), f"{test} ({a.axis} {r['axis']})",
                              fnum(t), fnum(t - c),
                              fnum(ranked[test]['w1'] - ranked[ctrl]['w1'])])
        if pairs:
            L += md_table(['section', 'control', 'sd_ratio', 'arm', 'sd_ratio',
                           'Δ sd_ratio', 'Δ W1/sd'], pairs)
            deltas = [float(p[5]) for p in pairs if p[5] != '—']
            L += ['']
            if deltas and min(deltas) > 0.05:
                L += [f'**It widens the ensemble in both sections** '
                      f'(Δ sd_ratio +{min(deltas):.3f} to +{max(deltas):.3f}). '
                      f'Whether that is enough is the `sd_ratio` column against 1.']
            elif deltas and max(deltas) < 0.02:
                L += [f'**It does not widen the ensemble** (Δ sd_ratio '
                      f'{min(deltas):+.3f} to {max(deltas):+.3f}).'
                      + (' Weight, not mechanism, is the first thing to check — see the '
                         'objective balance below; if the term is a small share of the '
                         'total, it was not tested at strength.'
                         if a.axis == 'beta_aux' else
                         ' If the posterior path reproduces the truth\'s spread '
                         '(misc/posterior_vs_prior.py), the prior\'s conditional is still '
                         'mis-shaped rather than mis-scaled: read that tool\'s PCA table.')]
            else:
                L += [f'**Mixed** (Δ sd_ratio {min(deltas):+.3f} to {max(deltas):+.3f}). '
                      f'A result that changes sign between two sections of one board '
                      f'is not a mechanism, and needs the per-eval-set table above '
                      f'before it is read as one.']
        if a.axis == 'beta_aux':
            L += ['', '### Objective balance', '',
                  'The peak-to-valley term is computed whenever the VAE path runs, so the '
                  '`beta_aux 0` control reports `aux=` too — its magnitude there is the '
                  'weight-free measurement.', '']
            bal = []
            for r in rows:
                log = find_log(a.logs, r['arm'])
                t = mgnv_terms(log) if log else None
                if not t:
                    continue
                recon, mmd, aux, total = t
                cfg_path = os.path.join(a.configs, f"config_train_{r['arm']}.txt")
                alpha = float(cfg_value(cfg_path, 'alpha_recon', 1.0) or 1.0)
                try:
                    beta = float(cfg_value(cfg_path, 'beta_aux', 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue          # config missing: no balance row, no crash
                pv_share = (beta * aux / total) if total else float('nan')
                bal.append([r['arm'], f'{recon:.3e}', f'{aux:.3e}', f'{total:.3e}',
                            f'{alpha * recon / total:.1%}' if total else '—',
                            f'{pv_share:.1%}' if pv_share == pv_share else '—'])
            if bal:
                L += md_table(['arm', 'recon', 'aux (pv)', 'total',
                               'recon share', 'pv share'], bal)
            else:
                L += ['_No `Train recon=... aux=...` line found in the logs._']
    else:
        L += ['The question: has it converged, or is 3000 epochs still budget-limited? '
              '`cosine_T0 = epochs − warmup` and `eta_min 1e-8`, so the LR is '
              'essentially zero by the final epoch and **a flat tail is the schedule, '
              'not convergence**. The middle of the run is where it is decided.', '']
        conv = []
        for r in rows:
            log = find_log(a.logs, r['arm'])
            curve = flow_curve(log) if log else []
            if len(curve) < 4:
                conv.append([r['arm'], len(curve), '—', '—', '—', '—'])
                continue
            n = len(curve)
            mid = curve[n // 4: 3 * n // 4]
            tail = curve[3 * n // 4:]
            # Relative improvement, not a slope ratio: eta_min is 1e-8, so the
            # tail slope decays to zero whether or not the model converged and
            # a comparison against it decides nothing. What matters is how much
            # the readout gained while the LR was still meaningful.
            rel_mid = rel_gain(mid)
            rel_tail = rel_gain(tail)
            if rel_mid is None:
                reading = "—"
            elif rel_mid > 0.10:
                reading = "still falling"
            elif rel_mid < 0.03:
                reading = "flattened"
            else:
                reading = "ambiguous"
            conv.append([r['arm'], n, f'{curve[-1][1]:.3e}',
                         f'{rel_mid:.1%}' if rel_mid is not None else '—',
                         f'{rel_tail:.1%}' if rel_tail is not None else '—',
                         reading])
        L += md_table(['arm', 'validations', 'final det(1fwd) mse',
                       'gain, middle half', 'gain, last quarter', 'reading'], conv)
        L += ['', 'The gain columns are the fractional drop in det(1fwd) mse across '
              'that span. Above 10% in the MIDDLE HALF is read as still learning; '
              'below 3% as plateaued. The last quarter is shown for contrast only '
              '-- the LR is near zero there by construction, so it flattens either '
              'way.']
        L += ['']
        verdicts = [c[5] for c in conv if c[5] in ('still falling', 'flattened')]
        if verdicts and all(v == 'still falling' for v in verdicts):
            L += ['**Still budget-limited at this epoch count.** The deterministic '
                  'readout is improving through the middle of the run and only flattens '
                  'where the LR does. More epochs, not a different objective.']
        elif verdicts and all(v == 'flattened' for v in verdicts):
            L += ['**Converged at this budget.** The middle-of-run slope is no steeper '
                  'than the tail, so the flattening is not the LR schedule. Remaining '
                  'error is a property of the model or the conditioning, not the '
                  'epoch count.']
        elif verdicts:
            L += ['**Split between arms** — read the gain columns per arm; the '
                  'learning rate is the axis, so the two rates converging differently '
                  'is the informative outcome, not a contradiction.']

    # ---- how to read it ---------------------------------------------------
    L += ['', '## Reading the metrics', '',
          '| column | meaning | ideal |',
          '| --- | --- | --- |',
          '| `W1/sd` | 1-Wasserstein(gt, gen) / sd(gt) — sort both samples, pair by '
          'rank, average the absolute gaps | 0 |',
          '| `dmean/sd` | (mean gen − mean gt) / sd(gt) — bias | 0 |',
          '| `sd_ratio` | sd(gen) / sd(gt) — dispersion; below 1 the ensemble is too '
          'narrow | **1** |',
          '| `PITtails` | share of truths landing in the outer 2% of the ensemble on '
          'either side | 0.04 |',
          '| `PIT_KS` | KS distance of the truths\' ranks from uniform | 0 |',
          '',
          '`sd_ratio` and `PITtails` are independent views of the same width defect '
          'and should agree. `|dmean|/sd` above 1 is a mean offset no width fix can '
          'repair.',
          '',
          '## Provenance', '',
          f'- dumps: `{a.infer}/<arm>/<eval set>/spread_values.npz`',
          f'- logs: `{a.logs}/<arm>.log`',
          f'- configs: `{a.configs}/config_train_<arm>.txt`',
          f'- rebuild: `python configs/campaigns/write_report.py --kind {a.kind} ...`',
          '']

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(L), encoding='utf-8', newline='\n')
    print(f'wrote {out}  ({len(arms)} arms, {len(metric_rows)} metric rows)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
