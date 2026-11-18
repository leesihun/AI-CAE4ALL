"""The verdict of prior_diagnosis.py: its JSON turned into plain English
sentences, into a checklist of every candidate cause, and into the short block
that is typed back. Standard library only, so the sweep's brief (any Python
3.8+) and the diagnosis (the method's interpreter) print the same thing.

    summarize(doc)        one diagnosis JSON -> the numbers the verdict reads
    merge([summary, ..])  the seeds of one setting -> their mean
    causes(parts)         every candidate cause: YES / no / ? and the number behind it
    verdict(parts)        {'ho': summary, 'tr': ..., 'ts': ...} -> sentences
    block(parts)          the lines to type back (whole numbers, check digit)
    read_history(path)    a training epoch log -> validation CRPS and prior loss per epoch

Every score is W2 over the FE sd (rms over the geometries, median over the
half-vs-half splits); the sentences and the block show it x100. A share in %
is a difference of two such scores over the gap `prior - posterior`. Those
shares are by construction: each ladder (where / what / when / onto what)
telescopes to the gap on its own, the order of its steps is a choice, and the
ladders are different cuts of the same gap, not parts that add up.

THE CHECKLIST  `CAUSES` below is the fixed list of everything that can make a
prior's draws bad. Each entry gets YES (found), no (ruled out by a number) or
? (this run cannot tell, with the reason). "Noticeable" is the floor's own
scatter over the splits (p95 - p50) or a quarter of the gap, whichever is
larger; the few fixed thresholds are spelled out in the evidence.
"""
import math
import re

SCORES = ('floor', 'P', 'O', 'Q', 'Qf', 'S', 'R5', 'Qc', 'Qcs', 'Qm')
EXTRA = ('Pr', 'Qr', 'Qg', 'Ql', 'Px', 'Qx', 'c10', 'c25', 'c50', 'c75', 'c90')
COMMIT = (('c10', 0.10), ('c25', 0.25), ('c50', 0.50), ('c75', 0.75), ('c90', 0.90))
EARLY = 0.5          # t <= EARLY is the early half of the flow (t = 0 is noise)
PARTS = ('ho', 'tr', 'ts')
# an optional fourth part: the held-out geometries with <stem>.joint.pth, the
# best checkpoint of the epochs before the prior freeze
JOINT = 'jt'

# The candidate causes, numbered from 1 in this order. The numbers are typed
# back in the `cs` line, so new entries go at the END.
CAUSES = (
    ('gap', 'the prior scores worse than the posterior'),
    ('encdec', 'the encoder/decoder loses on its own'),
    ('fit', 'missed already on the training geometries'),
    ('unseen', 'missed on new geometries inside the training range'),
    ('extra', 'missed on geometries outside the training range'),
    ('mean', 'the centre of the latent cloud is wrong'),
    ('width', 'the width of the latent cloud is wrong'),
    ('shape', 'the per-coordinate distributions are wrong'),
    ('joint', 'the combination of coordinates is wrong'),
    ('early', 'the error is made early in the flow (t<0.5)'),
    ('late', 'the error is made late in the flow (t>0.5)'),
    ('off', 'draws land off the posterior and decode to wrong outputs'),
    ('place', 'draws land on the posterior, wrong share per component'),
    ('zl', 'the z_l draws alone bring the gap'),
    ('zg', 'the z_g draws alone bring the gap'),
    ('steps', 'too few sampler steps'),
    ('sharp', 'target too sharp for the sampler, even with the exact field'),
    ('blind', 'the field is no better than one that ignores the geometry'),
    ('moments', "the field is no better than the geometry's mean and width alone"),
    ('memo', 'the field fits the training geometries, not new ones'),
    ('ood', 'the input embedding is outside the training distribution'),
    ('follow', 'the prior cloud does not follow the geometry'),
    ('scale', 'the standardized target is not of unit size'),
    ('points', 'the z_l target is isolated points'),
    ('deadg', 'z_g does not tell the draws apart'),
    ('modes', 'wrong share between the two modes'),
    ('band', "FE results fall outside the prior's 5-95% band"),
    ('ema', 'the EMA weights give a larger gap than the trained ones'),
    ('encema', 'the prior was trained on another encoder than the one used here'),
    ('short', 'training stopped while validation was still improving'),
    ('overfit', 'validation got worse during training'),
    ('one', 'one geometry carries over half the total'),
    ('smallsd', 'the worst geometry has a small FE spread (inflated score)'),
    ('seeds', 'the seeds disagree'),
    ('sampling', 'temperature or inflation is not 1'),
    ('frozen', 'the cost of training the prior on a frozen posterior'),
    ('pair', 'z_g and z_l belong together and the prior does not pair them'),
    ('tail', 'the prior-only epochs after the freeze did not shrink the gap'),
)
NUMBER = {key: i + 1 for i, (key, _) in enumerate(CAUSES)}


# ─────────────────────────────────────────────────────────────────────────────
# small statistics
# ─────────────────────────────────────────────────────────────────────────────

def fin(v):
    return isinstance(v, (int, float)) and v == v and abs(v) != float('inf')


def mean(vals):
    vals = [v for v in vals if fin(v)]
    return sum(vals) / len(vals) if vals else None


def median(vals):
    vals = sorted(v for v in vals if fin(v))
    if not vals:
        return None
    h = len(vals) // 2
    return vals[h] if len(vals) % 2 else 0.5 * (vals[h - 1] + vals[h])


def _ranks(x):
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if fin(a) and fin(b)]
    if len(pairs) < 4:
        return None
    rx, ry = _ranks([p[0] for p in pairs]), _ranks([p[1] for p in pairs])
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# the training log: which epoch was kept, and what validation did around it
# ─────────────────────────────────────────────────────────────────────────────

_EPOCH = re.compile(r'^Elapsed: [\d.]+s Epoch (\d+) ')
_CRPS = re.compile(r'\| CRPS ([0-9.eE+-]+)')
_PRIOR = re.compile(r' prior=([0-9.eE+-]+)')


def read_history(path):
    """A trainer's epoch log -> {'val': [[epoch, validation CRPS]], 'loss':
    [[epoch, training prior loss]]}, or None when it cannot be read. A resumed
    run repeats no epoch, but if one appears twice the later line wins."""
    val, loss = {}, {}
    try:
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                m = _EPOCH.match(line)
                if not m:
                    continue
                epoch = int(m.group(1))
                for found, store in ((_CRPS.search(line), val), (_PRIOR.search(line), loss)):
                    if found:
                        try:
                            store[epoch] = float(found.group(1))
                        except ValueError:
                            pass
    except OSError:
        return None
    if not val and not loss:
        return None
    return {'val': [[e, val[e]] for e in sorted(val)], 'loss': [[e, loss[e]] for e in sorted(loss)]}


def history_stats(tr):
    """doc['training'] -> the numbers of the `train` line. The window is the
    epochs the kept checkpoint was chosen from: from the prior freeze on, the
    whole run without a freeze, and the epochs before the freeze for a
    checkpoint kept from the joint phase (<stem>.joint.pth). The unit of
    "better / worse" is the validation series' own jitter (the median step
    between two validations)."""
    if not tr:
        return {}
    kept, freeze, total = tr.get('kept'), tr.get('freeze') or 0, tr.get('total')
    every = [e for e, _ in (tr.get('val') or []) + (tr.get('loss') or [])]
    end = total - 1 if fin(total) else max(every, default=None)
    start = freeze
    if fin(kept) and kept < freeze:   # kept before the freeze: chosen among the joint epochs
        start, end = 0, freeze - 1 if not fin(end) else min(freeze - 1, end)
    val = [(e, c) for e, c in (tr.get('val') or []) if fin(c) and e >= start and (not fin(end) or e <= end)]
    loss = [(e, x) for e, x in (tr.get('loss') or []) if fin(x) and e >= start and (not fin(end) or e <= end)]
    out = {'ep_kept': kept, 'ep_freeze': freeze, 'ep_total': total, 'ep_start': start, 'ep_end': end}
    if fin(kept) and fin(end) and end > start:
        out['ep_pos'] = min(max((kept - start) / (end - start), 0.0), 1.0)
    if len(val) >= 4:
        c = [x for _, x in val]
        jit = median([abs(b - a) for a, b in zip(c, c[1:])]) or 0.0
        q = max(2, len(c) // 4)
        best = min(c)
        i_best = c.index(best)
        out.update({
            'crps_first': c[0], 'crps_best': best, 'crps_best_ep': val[i_best][0], 'crps_last': c[-1],
            'crps_kept': next((x for e, x in val if e == kept), None), 'crps_jit': jit,
            'n_val': len(c),
            # the last quarter of the validations beats everything before it
            'falling': 1.0 if min(c[-q:]) < min(c[:-q]) - jit else 0.0,
            # the best is before the last quarter and the end is clearly above it
            'worse': 1.0 if i_best < len(c) - q and c[-1] > best + 2.0 * jit else 0.0,
        })
    if loss:
        out.update({'loss_first': loss[0][1], 'loss_last': loss[-1][1],
                    'loss_kept': next((x for e, x in loss if e == kept), None)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# one diagnosis JSON -> the numbers the verdict reads
# ─────────────────────────────────────────────────────────────────────────────

def summarize(doc):
    scores = doc.get('scores') or {}
    groups = doc.get('groups') or {}
    per = doc.get('per_group') or {}
    order = [v for v in groups if v in per]
    sc = {k: (scores.get(k) or {}).get('p50') for k in scores}
    fl = scores.get('floor') or {}
    noise = max((fl.get('p95') or 0.0) - (fl.get('p50') or 0.0), 0.02)

    def med(*path):
        vals = []
        for v in order:
            x = groups[v]
            for key in path:
                x = x.get(key) if isinstance(x, dict) else None
            vals.append(x)
        return median(vals)

    ve = {}
    for key in ('model', 'model_g', 'model_l', 'bias', 'g0', 'g1'):
        early, late = [], []
        for v in order:
            g = groups[v].get('ve') or {}
            for t, x in zip(g.get('t') or [], g.get(key) or []):
                (early if t <= EARLY else late).append(x)
        ve[key] = [mean(early), mean(late)]

    sq = {v: per[v].get('Q') ** 2 for v in order if fin(per[v].get('Q'))}
    worst = None
    if len(sq) > 1 and sum(sq.values()) > 0:
        v = max(sq, key=sq.get)
        tot = sum(sq.values())
        sd_mid = median([groups[x].get('fe_sd') for x in order])
        sd_v = groups[v].get('fe_sd')
        worst = {'id': v, 'share': sq[v] / tot, 'score': math.sqrt(sq[v]),
                 'without': math.sqrt((tot - sq[v]) / (len(sq) - 1)),
                 'sd_rel': sd_v / sd_mid if fin(sd_v) and fin(sd_mid) and sd_mid > 0 else None}

    kinds = {'mean': 0, 'width': 0, 'shape': 0}
    for v in order:
        p = (doc.get('parts') or {}).get(v)
        if not p:
            continue
        c = {'mean': p['bias'] ** 2, 'width': (p['sd_ratio'] - 1.0) ** 2, 'shape': p['shape'] ** 2}
        kinds[max(c, key=c.get)] += 1

    bim = [groups[v] for v in order if groups[v].get('fe_bimodal')]
    ema = doc.get('ema')
    out = {
        'n_groups': len(order), 'n': median([groups[v].get('n') for v in order]),
        'k': doc.get('k'), 'epoch': doc.get('epoch'), 'steps': doc.get('steps'),
        'fine': doc.get('fine'), 'latent_type': doc.get('latent_type'), 'solver': doc.get('solver'),
        'temperature': doc.get('prior_temperature'), 'inflation': doc.get('latent_inflation'),
        'sc': sc, 'noise': noise, 've': ve,
        'off_P': med('off', 'P'), 'off_Q': med('off', 'Q'), 'off_O': med('off', 'O'),
        'cov': med('cov', 'Q'), 'cov_O': med('cov', 'O'), 'rd': med('rd'),
        'far': med('far_emb'), 'far_raw': med('far_raw'),
        'far_max': max([groups[v].get('far_emb') for v in order
                        if fin(groups[v].get('far_emb'))], default=None),
        'raw_out': max([groups[v].get('raw_out') for v in order
                        if fin(groups[v].get('raw_out'))], default=None),
        'track_g': (doc.get('track') or {}).get('g'), 'track_l': (doc.get('track') or {}).get('l'),
        'worst': worst, 'kinds': kinds,
        'pit_lo': mean([groups[v].get('pit_lo') for v in order]),
        'pit_hi': mean([groups[v].get('pit_hi') for v in order]),
        'oor': mean([groups[v].get('oor') for v in order]),
        'n_bimodal': len(bim),
        'share_err': mean([abs(g['share']['Q'] - g['share']['truth']) for g in bim]),
        'share_err_P': mean([abs(g['share']['P'] - g['share']['truth']) for g in bim]),
        'ov_l': med('ov_l'), 'ov_g': med('ov_g'),
        'u_rms_l': med('u_rms_l'), 'u_rms_g': med('u_rms_g'),
        'u_between_l': med('u_between_l'), 'u_within_l': med('u_within_l'),
        'cen_l': med('cen_l'), 'cen_g': med('cen_g'), 'sdr_l': med('sdr_l'), 'sdr_g': med('sdr_g'),
        'geo': [{'id': v, 'gap': (per[v].get('Q') or float('nan')) - (per[v].get('P') or float('nan')),
                 'far': groups[v].get('far_emb'), 'far_raw': groups[v].get('far_raw')} for v in order],
        # EMA: None = an older JSON that did not look
        'ema_has': None if ema is None else (1.0 if ema.get('has') else 0.0),
        'ema_raw': None if ema is None else (1.0 if ema.get('raw') else 0.0),
        'ema_prior_diff': (ema or {}).get('prior_diff'), 'ema_rest_diff': (ema or {}).get('rest_diff'),
        'ema_buffers': None if ema is None or ema.get('buffers_equal') is None
        else (1.0 if ema['buffers_equal'] else 0.0),
        'enc_shift': med('enc_shift'),
        'best_by': (doc.get('training') or {}).get('best_by'),
        'val_interval': (doc.get('training') or {}).get('val_interval'),
    }
    out.update(history_stats(doc.get('training')))
    return out


def what_top(s):
    """The largest step of the `what` ladder of one summary (its main error kind)."""
    items = ladder((s or {}).get('sc') or {}, WHAT)
    return max(items, key=lambda it: it[1])[0] if items else None


def merge(items):
    """The mean of several summaries (the seeds of one setting)."""
    items = [s for s in items if s]
    if not items:
        return None
    if len(items) == 1:
        return items[0]

    def go(vals):
        first = next((v for v in vals if v is not None), None)
        if isinstance(first, dict):
            keys = []
            for v in vals:
                for key in (v or {}):
                    if key not in keys:
                        keys.append(key)
            return {key: go([(v or {}).get(key) for v in vals]) for key in keys}
        if isinstance(first, list):
            if first and isinstance(first[0], dict):
                return [x for v in vals for x in (v or [])]
            size = min(len(v) for v in vals if v is not None)
            return [go([v[i] for v in vals if v is not None]) for i in range(size)]
        if isinstance(first, bool) or isinstance(first, str):
            return first
        if isinstance(first, (int, float)):
            return mean(vals)
        return first
    out = go(items)
    out['seeds'] = len(items)
    out['seed_Q'] = [s['sc'].get('Q') for s in items]
    out['seed_what'] = [what_top(s) for s in items]
    worst = [s.get('worst') for s in items if s.get('worst')]
    out['worst'] = max(worst, key=lambda w: w['share']) if worst else None
    out['worst_ids'] = [w['id'] for w in worst]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# formatting
# ─────────────────────────────────────────────────────────────────────────────

def whole(v):
    """x100 as a whole number, None when it is missing."""
    return int(round(100 * v)) if fin(v) else None


def checked(label, values):
    """One line: the label, whole numbers ('-' = missing), then the check digit
    (the last digit of the sum of the line's other numbers)."""
    return (f'{label} ' + ' '.join('-' if v is None else str(v) for v in values)
            + f' {sum(v for v in values if v is not None) % 10}')


def _w(v):
    x = whole(v)
    return '-' if x is None else str(x)


def _f(v, digits=1):
    return '-' if not fin(v) else f'{v:.{digits}f}'


def _e(v):
    return '-' if not fin(v) else f'{v:.3g}'


def _n(v):
    """A count (a seed mean of counts is a float): whole when it is whole."""
    if not fin(v):
        return '-'
    return str(int(round(v))) if abs(v - round(v)) < 0.05 else f'{v:.1f}'


def _pct(x, total):
    if not fin(x) or not fin(total) or total <= 0:
        return '-'
    return f'{int(round(100 * x / total))}%'


def ladder(sc, keys):
    """Differences along a chain of scores: [(name, top - bottom)] or None."""
    out = []
    for name, top, bottom in keys:
        a, b = sc.get(top), sc.get(bottom)
        if not fin(a) or not fin(b):
            return None
        out.append((name, a - b))
    return out


WHAT = (('mean', 'Q', 'Qc'), ('width', 'Qc', 'Qcs'), ('shape', 'Qcs', 'Qm'), ('joint', 'Qm', 'P'))
WHEN = (('early', 'Q', 'R5'), ('late', 'R5', 'P'))
ONTO = (('off', 'Q', 'S'), ('place', 'S', 'P'))

NAMES = {'fit': 'training geometries', 'unseen': 'unseen, in range', 'extra': 'out of range',
         'mean': 'mean position', 'width': 'width', 'shape': 'per-coordinate shape',
         'joint': 'joint structure', 'early': 'early (t<0.5)', 'late': 'late (t>0.5)',
         'off': 'off the posterior', 'place': 'share between posterior components',
         'ho': 'out-of-range geometries', 'tr': 'training geometries', 'ts': 'unseen in-range geometries'}
MEANS = {
    'fit': 'Already wrong on the geometries it trained on: training, capacity or objective, not generalization.',
    'unseen': 'Does not carry over to new geometries inside the training range: generalization of geometry -> latent distribution.',
    'extra': 'Breaks only outside the training range: extrapolation.',
    'mean': 'The centre of the latent cloud is wrong per geometry: the mean the prior predicts from the condition.',
    'width': 'The width of the latent cloud is wrong.',
    'shape': 'The per-coordinate distributions (mode shares, tails) are wrong.',
    'joint': 'Each coordinate is right on its own, the combination is not: z the posterior never produces.',
    'early': 'The draw heads for the wrong component early in the flow.',
    'late': 'The early direction is right; the draw does not land on the posterior at the end.',
    'off': 'Draws land where the posterior has no mass and the decoder turns them into wrong outputs.',
    'place': 'Draws land near the posterior, with the wrong share per component.',
    'steps': 'Too few sampler steps.',
}


# ─────────────────────────────────────────────────────────────────────────────
# the checklist
# ─────────────────────────────────────────────────────────────────────────────

def _gap(s):
    sc = (s or {}).get('sc') or {}
    return sc['Q'] - sc['P'] if fin(sc.get('Q')) and fin(sc.get('P')) else None


def far_corr(parts):
    """Rank correlation, over the unseen geometries, between how far a geometry's
    input embedding lies from the training ones and its prior gap."""
    geo = [g for tag in ('ts', 'ho') for g in ((parts.get(tag) or {}).get('geo') or [])]
    return spearman([g.get('far') for g in geo], [g.get('gap') for g in geo])


def causes(parts):
    """Every candidate cause in the order of CAUSES:
    [(number, key, name, 'YES' | 'no' | '?', the number behind it)]."""
    main = next((t for t in PARTS if parts.get(t)), None)
    if main is None:
        return []
    s = parts[main]
    sc = s['sc']
    F, P, Q = sc.get('floor'), sc.get('P'), sc.get('Q')
    if not (fin(F) and fin(P) and fin(Q)):
        return []
    G, noise = Q - P, s['noise']
    big = max(noise, 0.25 * G)
    live = G > noise
    rows = {}

    def put(key, status, why):
        rows[key] = (status if isinstance(status, str) else ('YES' if status else 'no'), why)

    put('gap', live, f'prior {_w(Q)}, posterior {_w(P)}: gap {_w(G)}, noise {_w(noise)}')
    put('encdec', P - F > noise, f'posterior {_w(P)}, floor {_w(F)}: {_w(P - F)} above, noise {_w(noise)}')

    # where: the gap of each part ------------------------------------------
    gap = {t: _gap(parts.get(t)) for t in PARTS}
    known = [t for t in PARTS if gap[t] is not None]
    thr = max([parts[t]['noise'] for t in known] + [0.25 * max(gap[t] for t in known)])
    for key, top, bottom in (('fit', 'tr', None), ('unseen', 'ts', 'tr'), ('extra', 'ho', 'ts')):
        need = [t for t in (top, bottom) if t and gap[t] is None]
        if need:
            put(key, '?', 'needs the ' + ' and '.join(need) + ' part')
        elif bottom is None:
            put(key, gap[top] >= thr, f'gap on {top} {_w(gap[top])}, threshold {_w(thr)}')
        else:
            d = gap[top] - gap[bottom]
            put(key, d >= thr, f'gap on {top} {_w(gap[top])} - on {bottom} {_w(gap[bottom])} = {_w(d)}, '
                               f'threshold {_w(thr)}')

    # the ladders -----------------------------------------------------------
    def step(key, top, bottom, usable=True, why_not=''):
        a, b = sc.get(top), sc.get(bottom)
        if not live:
            put(key, 'no', 'no gap to explain')
        elif not usable:
            put(key, '?', why_not)
        elif not (fin(a) and fin(b)):
            put(key, '?', f'{top if not fin(a) else bottom} was not computed')
        else:
            put(key, a - b >= big, f'{top} {_w(a)} - {bottom} {_w(b)} = {_w(a - b)} ({_pct(a - b, G)} of the gap)')

    for keys in (WHAT, WHEN):
        for key, top, bottom in keys:
            step(key, top, bottom)
    s_ok = not (fin(s.get('rd')) and s['rd'] < 0.7)
    for key, top, bottom in ONTO:
        step(key, top, bottom, s_ok, f'S is unreliable here ({_f(s.get("rd"), 2)} < 0.70)')
    Px, Qx = sc.get('Px'), sc.get('Qx')
    if s.get('latent_type') == 'local':
        put('zl', live, 'z_l is the only latent of this model')
        put('zg', 'no', 'this model has no z_g')
        put('pair', 'no', 'this model has no z_g')
    else:
        # Ql / Qg pair two unrelated draws, so they are read against Px (two
        # unrelated posterior samples), not against P
        base = 'Px' if fin(Px) else 'P'
        step('zl', 'Ql', base)
        step('zg', 'Qg', base)
        if not live:
            put('pair', 'no', 'no gap to explain')
        elif not (fin(Px) and fin(Qx)):
            put('pair', '?', 'Px / Qx were not computed')
        else:
            put('pair', Px - P >= big and Qx - Q <= noise,
                f'unpairing z_g and z_l costs the posterior {_w(Px - P)} (Px {_w(Px)}), the prior '
                f'{_w(Qx - Q)} (Qx {_w(Qx)}); YES when the first is at least {_w(big)} and the second within '
                f'the noise {_w(noise)}')
    step('steps', 'Q', 'Qf')
    O = sc.get('O')
    if fin(O):
        put('sharp', O - P > noise, f'exact field {_w(O)}, posterior {_w(P)}: {_w(O - P)}, noise {_w(noise)}')
    else:
        put('sharp', '?', 'O was not computed')

    # the field -------------------------------------------------------------
    ve = s.get('ve') or {}
    m, g0, g1 = mean(ve.get('model') or []), mean(ve.get('g0') or []), mean(ve.get('g1') or [])
    if fin(m) and fin(g0) and fin(g1):
        put('blind', m >= g0, f'velocity error {_w(m)}% of the signal; geometry-blind field {_w(g0)}%')
        put('moments', m >= g1, f'velocity error {_w(m)}%; per-geometry Gaussian field {_w(g1)}%')
    else:
        put('blind', '?', 'no velocity error')
        put('moments', '?', 'no velocity error')
    tr = parts.get('tr') if main != 'tr' else None
    m_tr = mean(((tr or {}).get('ve') or {}).get('model') or [])
    if fin(m) and fin(m_tr) and m_tr > 0:
        put('memo', m >= 2.0 * m_tr, f'velocity error {_w(m_tr)}% on training, {_w(m)}% on {main} '
                                     f'({_f(m / m_tr)}x; YES from 2x)')
    else:
        put('memo', '?', 'needs the tr part next to an unseen one')

    # input, output, target -------------------------------------------------
    if fin(s.get('far')):
        ts = parts.get('ts') if main != 'ts' else None
        base = ts['far'] if ts and fin(ts.get('far')) else 1.0
        put('ood', s['far'] > 1.5 * base, f'embedding distance {_f(s["far"], 2)} (1 = among training geometries; '
                                          f'YES above {_f(1.5 * base, 2)})')
    else:
        put('ood', '?', 'no training reference (--train-ref 0)')
    tl, tg = s.get('track_l'), s.get('track_g')
    if fin(tl) or fin(tg):
        put('follow', any(fin(x) and x < 0.5 for x in (tl, tg)),
            f'R2 of the prior cloud against the posterior cloud: z_l {_f(tl, 2)}, z_g {_f(tg, 2)} (YES below 0.50)')
    else:
        put('follow', '?', 'fewer than 3 geometries')
    ul, ug = s.get('u_rms_l'), s.get('u_rms_g')
    if fin(ul):
        kept = s.get('ep_kept') if fin(s.get('ep_kept')) else s.get('epoch')
        unfit = fin(kept) and fin(s.get('ep_freeze')) and 0 < kept < s['ep_freeze']
        put('scale', any(fin(x) and not 0.5 <= x <= 2.0 for x in (ul, ug)),
            f'size after standardization: z_l {_f(ul, 2)}, z_g {_f(ug, 2)} (1 is right; YES outside 0.5-2)'
            + ('; the standardization is fitted at the freeze and this checkpoint is from before it' if unfit else ''))
    else:
        put('scale', '?', 'not recorded')
    if fin(s.get('ov_l')):
        put('points', s['ov_l'] < 0.3, f"one posterior's width / spacing of the draws {_f(s['ov_l'], 2)} (YES below 0.30)")
    else:
        put('points', '?', 'not recorded')
    if s.get('latent_type') == 'local':
        put('deadg', 'no', 'this model has no z_g')
    elif fin(s.get('ov_g')):
        put('deadg', s['ov_g'] > 1.0, f"one posterior's width / spacing of the draws {_f(s['ov_g'], 2)} (YES above 1)")
    else:
        put('deadg', '?', 'not recorded')

    # the output statistic ----------------------------------------------------
    if s.get('n_bimodal') and fin(s.get('share_err')) and fin(s.get('share_err_P')) and fin(s.get('n')):
        tol = 1.0 / math.sqrt(s['n'])
        put('modes', s['share_err'] - s['share_err_P'] > tol,
            f'upper-mode share off by {_pct(s["share_err"], 1.0)} (posterior {_pct(s["share_err_P"], 1.0)}) on '
            f'{_n(s["n_bimodal"])} two-mode geometries; YES above +{_pct(tol, 1.0)}')
    else:
        put('modes', 'no', 'no two-mode geometry among these')
    if fin(s.get('pit_lo')) and fin(s.get('pit_hi')):
        out_band = s['pit_lo'] + s['pit_hi']
        put('band', out_band > 0.3, f'{_pct(out_band, 1.0)} of the FE results outside (10% is right; YES above 30%)')
    else:
        put('band', '?', 'not recorded')

    # training: EMA, checkpoint ---------------------------------------------
    if s.get('ema_has') is None:
        put('ema', '?', 'this JSON predates the EMA test')
        put('encema', '?', 'this JSON predates the EMA test')
    elif s['ema_has'] < 0.5:
        put('ema', 'no', 'the checkpoint has no EMA weights: the trained weights are the ones used')
        put('encema', 'no', 'the checkpoint has no EMA weights')
    else:
        Pr, Qr = sc.get('Pr'), sc.get('Qr')
        if (s.get('ema_raw') or 0) < 0.5 or not (fin(Pr) and fin(Qr)):
            put('ema', '?', 'the trained (non-EMA) weights were not run (--no-raw, or they did not load)')
        else:   # the gap of each set of weights against its own posterior
            d = G - (Qr - Pr)
            thr = max(noise, 0.25 * max(G, 0.0))
            put('ema', d >= thr,
                f'gap {_w(G)} with the EMA weights ({_w(Q)} - {_w(P)}), {_w(Qr - Pr)} with the trained ones '
                f'({_w(Qr)} - {_w(Pr)}): the EMA costs {_w(d)} (YES from {_w(thr)})')
        if fin(s.get('enc_shift')):
            put('encema', s['enc_shift'] > 0.5, f'posterior means of the trained and the EMA encoder differ by '
                                                f'{_f(s["enc_shift"], 2)} posterior sigma (YES above 0.5)')
        else:
            put('encema', '?', 'the trained (non-EMA) weights did not load')
    if fin(s.get('falling')) and fin(s.get('worse')):
        put('short', s['falling'] >= 0.5,
            f'validation CRPS best {_e(s.get("crps_best"))} at epoch {_n(s.get("crps_best_ep"))}, last '
            f'{_e(s.get("crps_last"))}, step between validations {_e(s.get("crps_jit"))}')
        put('overfit', s['worse'] >= 0.5,
            f'validation CRPS best {_e(s.get("crps_best"))} at epoch {_n(s.get("crps_best_ep"))} -> last '
            f'{_e(s.get("crps_last"))}; training prior loss {_e(s.get("loss_first"))} -> {_e(s.get("loss_last"))}')
    else:
        why = 'no validation CRPS in the training log (or no log found)'
        put('short', '?', why)
        put('overfit', '?', why)

    # the evaluation itself ---------------------------------------------------
    w = s.get('worst')
    if w:
        put('one', w['share'] > 0.5, f'geometry {w["id"]} carries {_pct(w["share"], 1.0)} of the squared total')
        if fin(w.get('sd_rel')):
            put('smallsd', w['sd_rel'] < 0.5, f'FE sd of geometry {w["id"]} is {_f(w["sd_rel"], 2)} of the median '
                                              f'geometry\'s (YES below 0.50)')
        else:
            put('smallsd', '?', 'not recorded')
    else:
        put('one', 'no', 'a single geometry')
        put('smallsd', 'no', 'a single geometry')
    if s.get('seeds', 1) > 1:
        qs = [q for q in s.get('seed_Q') or [] if fin(q)]
        tops = [t for t in s.get('seed_what') or [] if t]
        spread = max(qs) - min(qs) if qs else 0.0
        put('seeds', len(set(tops)) > 1 or spread >= big,
            f'prior per seed {" ".join(_w(q) for q in qs)} (range {_w(spread)}, threshold {_w(big)}); '
            f'main error kind per seed: {" / ".join(NAMES.get(t, t) for t in tops) or "-"}')
    else:
        put('seeds', '?', 'one seed')
    temp, infl = s.get('temperature'), s.get('inflation')
    if fin(temp) and fin(infl):
        put('sampling', abs(temp - 1.0) > 1e-6 or abs(infl - 1.0) > 1e-6, f'temperature {_f(temp, 2)}, inflation {_f(infl, 2)}')
    else:
        put('sampling', '?', 'not recorded')
    put('frozen', '?', 'needs a run that trains the prior and the encoder together; what this run shows is '
                       'whether the frozen target is hard to fit (17, 23, 24) and what the tail bought (38)')
    jt = parts.get(JOINT)
    gj = _gap(jt)
    if fin(s.get('ep_freeze')) and s['ep_freeze'] == 0:
        put('tail', 'no', 'this run has no prior-only phase')
    elif gj is None or main != 'ho':
        put('tail', '?', 'needs the jt part: <stem>.joint.pth on the held-out geometries')
    elif G <= noise and gj <= max(noise, jt['noise']):
        put('tail', 'no', f'no gap at either checkpoint: {_w(gj)} at the joint-phase one (epoch {_n(jt.get("epoch"))}), '
                          f'{_w(G)} at the kept one')
    else:
        thr = max(noise, jt['noise'])
        put('tail', G > gj - thr,
            f'gap {_w(gj)} at the joint-phase checkpoint (epoch {_n(jt.get("epoch"))}), {_w(G)} at the kept one: '
            f'{_w(gj - G)} smaller (YES below {_w(thr)})')

    return [(NUMBER[key], key, name) + rows[key] for key, name in CAUSES]


def cause_line(rows):
    """`cs <numbers found> u <numbers that cannot be told> <check digit>`."""
    yes = [n for n, _, _, status, _ in rows if status == 'YES']
    unk = [n for n, _, _, status, _ in rows if status == '?']
    return ' '.join(['cs'] + [str(n) for n in yes] + ['u'] + [str(n) for n in unk] + [str(sum(yes + unk) % 10)])


# ─────────────────────────────────────────────────────────────────────────────
# the typed block
# ─────────────────────────────────────────────────────────────────────────────

def block(parts):
    """The lines to type back: one score line per part, then ve, in, ex, cs."""
    lines = []
    for tag in PARTS:
        s = parts.get(tag)
        if s:
            lines.append(checked(tag, [whole(s['sc'].get(k)) for k in SCORES]))
    ve = []
    for tag in ('tr', 'ho'):
        s = parts.get(tag) or (parts.get('ts') if tag == 'ho' else None)
        ve += [whole(x) for x in ((s or {}).get('ve') or {}).get('model', [None, None])]
    ref = parts.get('ho') or parts.get('ts') or parts.get('tr') or {}
    for key in ('g0', 'g1'):
        ve += [whole(x) for x in (ref.get('ve') or {}).get(key, [None, None])]
    lines.append(checked('ve', ve))
    ts = parts.get('ts') or {}
    lines.append(checked('in', [whole(ts.get('far')), whole(ref.get('far')), whole(far_corr(parts)),
                                whole(ref.get('rd')), whole(ref.get('off_Q')), whole(ref.get('off_P')),
                                whole(ref.get('cov')), whole(ref.get('track_l')), whole(ref.get('track_g'))]))
    main = next((parts[t] for t in PARTS if parts.get(t)), {})
    kept, last = main.get('crps_kept'), main.get('crps_last')
    jt = (parts.get(JOINT) or {}).get('sc') or {}
    lines.append(checked('ex', [whole((main.get('sc') or {}).get(k)) for k in EXTRA]
                         + [whole(main.get('ep_pos')),
                            whole(last / kept) if fin(kept) and fin(last) and kept > 0 else None,
                            whole(jt.get('P')), whole(jt.get('Q'))]))
    rows = causes(parts)
    if rows:
        lines.append(cause_line(rows))
    return lines


BLOCK_LEGEND = """\
COLUMNS (do not type)   whole numbers = x100. The LAST number of every line is a check digit.
  ho tr ts   held-out / 16 training / test geometries, ten scores each (W2 over FE sd):
     floor   FE half against FE half: nothing can score lower
     P       posterior sample decoded (encoder + decoder alone)
     O       the EXACT flow of this geometry's posterior, through the model's sampler (should = P)
     Q       the prior (what inference does)
     Qf      the prior with 4x the sampler steps
     S       the prior's draws read off at the nearest posterior component (no decoder error)
     R5      exact flow for t < 0.5, the model's flow after
     Qc      the prior's draws moved to the posterior's mean, per coordinate
     Qcs     ... and rescaled to its width
     Qm      ... and given its exact per-coordinate distribution (only the joint structure is the prior's)
  ve   the model's velocity error over the exact velocity's size: training early(t<=0.5), late,
       held-out early, late; then a field that ignores the geometry (early, late) and the best
       per-geometry Gaussian field (early, late), both on the held-out geometries
  in   distance of the input embedding from the training geometries (1 = as far as they are from
       each other): test, held-out; its rank correlation with the gap; reliability of S (below 70:
       do not trust S); distance of the prior's draws from the posterior (posterior sample itself
       next to it, ~100); share of posterior components the draws reach (100 = as an exact sampler);
       how well the prior's cloud follows the geometry, R2 for z_l and z_g (100 = fully, <=0 = not)
  ex   on the first part: Pr Qr (P and Q with the trained, non-EMA weights), Qg (only z_g from the
       prior), Ql (only z_l from the prior), Px (z_g and z_l of two different posterior samples),
       Qx (of two different prior draws); the model's field until t = .10 .25 .50 .75 .90 and the
       exact one after; where the kept epoch lies in the epochs it was chosen from (0 = first,
       100 = last); validation CRPS at the last epoch over the kept epoch's; P and Q of the
       joint-phase checkpoint (<stem>.joint.pth) on the held-out geometries
  cs  the numbers of the CAUSES found, then `u`, then the numbers that cannot be told here
"""


# ─────────────────────────────────────────────────────────────────────────────
# the sentences
# ─────────────────────────────────────────────────────────────────────────────

def verdict(parts, label='', short=False):
    """Plain sentences from {'ho': summary, 'tr': ..., 'ts': ...} (any subset).
    short: the premise, the conclusion and the numbers of the causes found."""
    main = next((t for t in PARTS if parts.get(t)), None)
    if main is None:
        return ['no diagnosis yet.']
    s = parts[main]
    sc = s['sc']
    F, P, Q = sc.get('floor'), sc.get('P'), sc.get('Q')
    if not (fin(F) and fin(P) and fin(Q)):
        return ['scores are missing.']
    G = Q - P
    noise = s['noise']
    big = max(noise, 0.25 * G)
    rows = causes(parts)
    status = {key: st for _, key, _, st, _ in rows}
    head = [f'==== VERDICT {label}: {_n(s["n_groups"])} {NAMES[main]} (scores x100, lower is better) ====']
    body = []

    # premise -----------------------------------------------------------------
    line = f'premise floor {_w(F)}, posterior {_w(P)}, prior {_w(Q)}. Gap {_w(G)} (what the rest explains).'
    if P - F <= noise:
        line += ' Encoder/decoder are fine.'
    else:
        line += f' The posterior is {_w(P - F)} above the floor: the encoder/decoder loses too.'
    head.append(line)
    if s.get('seeds', 1) > 1:
        kinds = ' / '.join(NAMES.get(t, '-') for t in s.get('seed_what') or [])
        head.append(f'        prior per seed: {" ".join(_w(x) for x in s["seed_Q"])} (the rest is the seed mean); '
                    f'main error kind per seed: {kinds}')
    if G <= noise:
        head.append(f"        The gap is inside the floor's scatter ({_w(noise)}): the prior is not the problem here.")

    # is the method valid -----------------------------------------------------
    O = sc.get('O')
    if fin(O):
        if O - P <= noise:
            line = (f'check   the exact flow through the same sampler gives {_w(O)} = the posterior. '
                    f'The numbers below hold.')
        else:
            line = (f'check   even the exact flow gives {_w(O)} through this sampler: {_w(O - P)} ({_pct(O - P, G)}) '
                    f'of the gap stays with a perfect field (the target is too sharp for the steps).')
        if fin(s.get('rd')) and s['rd'] < 0.7:
            line += f' Warning: S reliability {_f(s["rd"], 2)} < 0.70, ignore the "onto" line.'
        body.append(line)

    found = []   # (share of the gap, key, points)
    s_ok = not (fin(s.get('rd')) and s['rd'] < 0.7)

    def row(title, items, total, rank=True):
        body.append(title + ' | '.join(f'{NAMES[k]} {_w(x)} ({_pct(x, total)})' for k, x in items))
        k, x = max(items, key=lambda it: it[1])
        if rank and x >= max(noise, 0.25 * total):
            found.append((x / total if total > 0 else 0.0, k, x))

    # where -------------------------------------------------------------------
    gap = {t: _gap(parts.get(t)) for t in PARTS if _gap(parts.get(t)) is not None}
    if len(gap) == 3 and gap['ho'] > noise:
        row('where   ', [('fit', gap['tr']), ('unseen', gap['ts'] - gap['tr']), ('extra', gap['ho'] - gap['ts'])],
            gap['ho'])
        body.append(f'        (gap per part: training {_w(gap["tr"])}, in range {_w(gap["ts"])}, out of range '
                    f'{_w(gap["ho"])}; the floors differ, so this is a rough comparison)')
    elif len(gap) > 1:
        body.append('where   ' + ' | '.join(f'{NAMES[t]} {_w(gap[t])}' for t in PARTS if t in gap))

    if G > noise:
        for title, keys in (('what    ', WHAT), ('when    ', WHEN), ('onto    ', ONTO)):
            items = ladder(sc, keys)
            if items:
                row(title, items, G, rank=keys is not ONTO or s_ok)
            if keys is WHEN:
                chain = [('exact all the way', P)] + [(f't={t:.2f}', sc.get(k)) for k, t in COMMIT] \
                    + [('model all the way', sc.get('S'))]
                if s_ok and all(fin(v) for _, v in chain):
                    jumps = [(b[1] - a[1], a[0], b[0]) for a, b in zip(chain, chain[1:])]
                    d, a, b = max(jumps)
                    body.append("        the model's field until t, the exact one after: "
                                + ', '.join(f'{name} {_w(v)}' for name, v in chain)
                                + f'. Largest step: {a} -> {b} (+{_w(d)})')
        body.append(f'        distance from the posterior {_f(s.get("off_Q"))} posterior sigma (a posterior sample '
                    f'{_f(s.get("off_P"))}, the exact flow {_f(s.get("off_O"))}), components reached '
                    f'{_pct(s.get("cov"), 1.0)}')
        Qg, Ql, Px, Qx = (sc.get(k) for k in ('Qg', 'Ql', 'Px', 'Qx'))
        if fin(Qg) and fin(Ql):
            base = Px if fin(Px) else P
            body.append(f'latent  only z_g from the prior {_w(Qg)} ({_w(Qg - base)} above the reference) | only z_l '
                        f'from the prior {_w(Ql)} ({_w(Ql - base)} above) | both {_w(Q)}. Reference '
                        f'{_w(base)} = z_g and z_l from two different posterior samples'
                        + (f' (the same sample: {_w(P)})' if fin(Px) else ''))
            if fin(Px) and fin(Qx):
                line = (f'        pairing: mixing z_g and z_l of two draws costs the posterior {_w(Px - P)}, the '
                        f'prior {_w(Qx - Q)}.')
                if Px - P <= noise:
                    line += ' z_g and z_l need not match: each can be judged on its own.'
                elif Qx - Q <= noise:
                    line += " They belong together, and the prior's own pairing carries nothing."
                else:
                    line += ' They belong together, and the prior pairs them too.'
                body.append(line)

    # sampler -----------------------------------------------------------------
    Qf = sc.get('Qf')
    if fin(Qf):
        d = Q - Qf
        line = (f'sampler {s.get("solver") or "-"} {_n(s.get("steps"))} steps; {_n(s.get("fine") or 4)}x the steps '
                f'-> {_w(Qf)} (difference {_w(d)}, {_pct(d, G) if G > noise else "-"}). ')
        line += 'The step count is part of the cause.' if d >= big else 'The step count is not the cause.'
        if fin(s.get('temperature')) and fin(s.get('inflation')):
            line += f' Temperature {_f(s["temperature"], 2)}, inflation {_f(s["inflation"], 2)}.'
        body.append(line)
        if d >= big and G > noise:
            found.append((d / G if G > 0 else 0.0, 'steps', d))

    # field -------------------------------------------------------------------
    ve = s.get('ve') or {}
    if fin((ve.get('model') or [None])[0]):
        def pair(v):
            return f'{_w(v[0])}/{_w(v[1])}' if v else '-'
        line = f'field   velocity error over signal, % (early/late): {NAMES[main]} {pair(ve.get("model"))}'
        tr = parts.get('tr')
        if tr and main != 'tr' and fin(((tr.get('ve') or {}).get('model') or [None])[0]):
            line += f', {NAMES["tr"]} {pair(tr["ve"]["model"])}'
        line += (f' | a field that ignores the geometry {pair(ve.get("g0"))} | per-geometry Gaussian field '
                 f'{pair(ve.get("g1"))} | constant part of the error {_pct(mean(ve.get("bias") or []), 1.0)}')
        if fin((ve.get('model_g') or [None])[0]):
            line += f' | by latent: z_l {pair(ve.get("model_l"))}, z_g {pair(ve.get("model_g"))}'
        body.append(line)
        if status.get('blind') == 'YES':
            body.append('        The field is no better than ignoring the geometry: the condition is unused or misused.')
        elif status.get('moments') == 'YES':
            body.append("        The field is worse than a Gaussian that knows the geometry's mean and width.")
        elif status.get('moments') == 'no':
            body.append('        The field beats the per-geometry Gaussian: mean and width are caught, '
                        'what is left is finer structure.')

    # input -------------------------------------------------------------------
    if fin(s.get('far')):
        ts = parts.get('ts')
        line = (f'input   distance from the training geometries (1 = among themselves): {NAMES[main]} '
                f'{_f(s["far"], 2)} (max {_f(s.get("far_max"), 2)})')
        if ts and main != 'ts' and fin(ts.get('far')):
            line += f', {NAMES["ts"]} {_f(ts["far"], 2)}'
        rc = far_corr(parts)
        if fin(rc):
            line += f'. Rank correlation with the gap {_f(rc, 2)}'
        if fin(s.get('raw_out')):
            line += f'. Largest input statistic outside the training range {_f(s["raw_out"])} sd'
        body.append(line)
        if status.get('ood') == 'YES':
            body.append("        The input embedding is outside the training distribution: the prior's trunk "
                        "has not seen such an input.")
        elif G > noise:
            body.append('        The input embedding is as near as a training geometry: the input is not '
                        'unfamiliar, the output is wrong.')

    # output, target ----------------------------------------------------------
    if fin(s.get('track_l')) or fin(s.get('track_g')):
        body.append(f'output  how the prior cloud follows the geometry, R2: z_l {_f(s.get("track_l"), 2)}, z_g '
                    f'{_f(s.get("track_g"), 2)} (1 = moves with the posterior cloud, <=0 = worse than a constant). '
                    f'Centre offset z_l {_f(s.get("cen_l"), 2)} z_g {_f(s.get("cen_g"), 2)} sd, width ratio z_l '
                    f'{_f(s.get("sdr_l"), 2)} z_g {_f(s.get("sdr_g"), 2)}')
    if fin(s.get('ov_l')):
        body.append(f"target  one posterior's width / spacing of the draws: z_l {_f(s.get('ov_l'), 2)}, z_g "
                    f"{_f(s.get('ov_g'), 2)} (below 0.3 = isolated points, above 1 = the draws are not told apart). "
                    f"Size after standardization z_l {_f(s.get('u_rms_l'), 2)} z_g {_f(s.get('u_rms_g'), 2)} "
                    f"(1 is right)")

    # calibration, concentration ----------------------------------------------
    if fin(s.get('pit_lo')):
        line = (f"calib   FE results below the prior's 5% point {_pct(s['pit_lo'], 1.0)}, above its 95% point "
                f"{_pct(s['pit_hi'], 1.0)} (5%/5% is right). Prior draws outside the FE range "
                f"{_pct(s.get('oor'), 1.0)}")
        if s.get('n_bimodal'):
            line += (f'. On the {_n(s["n_bimodal"])} two-mode geometries the upper share is off by '
                     f'{_pct(s.get("share_err"), 1.0)} (posterior {_pct(s.get("share_err_P"), 1.0)})')
        body.append(line)
    w = s.get('worst')
    if w:
        line = (f'weight  geometry {w["id"]} alone carries {_pct(w["share"], 1.0)} of the squared total '
                f'(score {_w(w["score"])}); without it {_w(w["without"])}')
        if fin(w.get('sd_rel')):
            line += f"; its FE sd is {_f(w['sd_rel'], 2)} of the median geometry's"
        if s.get('seeds', 1) > 1:
            line += f'. Worst geometry per seed: {" ".join(str(x) for x in s["worst_ids"])}'
        k = s.get('kinds') or {}
        line += (f'. Dominant error per geometry: mean {_n(k.get("mean", 0))}, width {_n(k.get("width", 0))}, '
                 f'shape {_n(k.get("shape", 0))}')
        body.append(line)

    # training ----------------------------------------------------------------
    if fin(s.get('ep_kept')) or fin(s.get('epoch')):
        kept = s.get('ep_kept') if fin(s.get('ep_kept')) else s.get('epoch')
        line = f'train   kept epoch {_n(kept)}'
        if fin(s.get('ep_total')):
            line += f' of 0-{_n(s["ep_total"] - 1)}'
            if s.get('ep_freeze'):
                line += f' (prior alone from {_n(s["ep_freeze"])})'
                if fin(kept) and kept < s['ep_freeze']:
                    line += ', a checkpoint of the joint phase'
        if s.get('best_by'):
            line += f', chosen by validation {s["best_by"]}'
            if fin(s.get('val_interval')):
                line += f' every {_n(s["val_interval"])} epochs'
        if fin(s.get('crps_best')):
            line += (f'. Validation CRPS over those epochs: first {_e(s["crps_first"])}, best {_e(s["crps_best"])} '
                     f'at epoch {_n(s["crps_best_ep"])}, last {_e(s["crps_last"])}')
        if fin(s.get('loss_first')):
            line += (f'. Training prior loss {_e(s["loss_first"])} -> {_e(s.get("loss_kept"))} (kept) -> '
                     f'{_e(s["loss_last"])} (last)')
        if not fin(s.get('crps_best')) and not fin(s.get('loss_first')):
            line += '. No training log was found: the epochs around it cannot be read'
        body.append(line)
    if s.get('ema_has') is not None:
        if s['ema_has'] < 0.5:
            body.append('        EMA: the checkpoint has none; the trained weights are the ones used.')
        else:
            line = (f'        EMA: with the trained (non-EMA) weights posterior {_w(sc.get("Pr"))}, prior '
                    f'{_w(sc.get("Qr"))}; with the EMA weights (inference) posterior {_w(P)}, prior {_w(Q)}')
            if fin(s.get('ema_prior_diff')):
                line += (f'. Weights differ by {_pct(s["ema_prior_diff"], 1.0)} (prior), '
                         f'{_pct(s.get("ema_rest_diff"), 1.0)} (encoder/decoder)')
            if fin(s.get('enc_shift')):
                line += f'; posterior means differ by {_f(s["enc_shift"], 2)} posterior sigma'
            if fin(s.get('ema_buffers')) and s['ema_buffers'] < 0.5:
                line += '. WARNING: the standardization buffers of the two differ'
            body.append(line)
    jt = parts.get(JOINT)
    if jt and main == 'ho' and _gap(jt) is not None:
        gj = _gap(jt)
        line = (f'tail    joint-phase checkpoint (epoch {_n(jt.get("epoch"))}): posterior {_w(jt["sc"]["P"])}, prior '
                f'{_w(jt["sc"]["Q"])}, gap {_w(gj)} -> kept checkpoint: posterior {_w(P)}, prior {_w(Q)}, gap {_w(G)}. ')
        line += ('The prior-only epochs did not shrink the gap.' if status.get('tail') == 'YES'
                 else 'There is no gap at either checkpoint.' if G <= noise
                 else f'The prior-only epochs shrank the gap by {_w(gj - G)}.')
        body.append(line)
        body.append('        (the joint-phase checkpoint is that phase\'s best epoch, not the state at the freeze: its '
                    'encoder/decoder is its own and its latent is not standardized yet)')

    # conclusion --------------------------------------------------------------
    so = []
    if G > noise:
        found.sort(key=lambda f: -f[0])
        if found:
            for i, (share, key, pts) in enumerate(found):
                lead = 'so      ' if i == 0 else '        '
                text = MEANS[key] if key == 'steps' else f'{NAMES[key]}: {MEANS[key]}'
                so.append(f'{lead}{i + 1}) {text} [{_w(pts)}, {int(round(100 * share))}%]')
        else:
            so.append('so      No single item exceeds 25% of the gap: several causes, evenly mixed.')
    yes = [n for n, _, _, st, _ in rows if st == 'YES']
    unk = [n for n, _, _, st, _ in rows if st == '?']
    if short:
        return head + so + [f'causes  found: {" ".join(map(str, yes)) or "-"} | cannot be told here: '
                            f'{" ".join(map(str, unk)) or "-"} (numbers of the CAUSES table)']

    table = [f'CAUSES  every candidate, fixed order: YES = found, no = ruled out, ? = cannot be told here '
             f'({len(yes)} found, {len(unk)} open)']
    width = max(len(name) for _, name in CAUSES)
    for n, _, name, st, why in rows:
        table.append(f'  {n:>2} {st:<3} {name:<{width}}  {why}')
    note = ['note    The % are by construction (the order of the steps is a choice) and each line cuts the same gap '
            'another way: do not add lines. A ? is not a "no". Not answerable from one checkpoint: how another '
            'epoch would score on these geometries (only the validation curve is known), whether a miss on the '
            'training geometries (3) is a network too small or too little training beyond what 30 says, and what '
            'training the prior and the encoder together would give (36).']
    return head + body + so + table + note
