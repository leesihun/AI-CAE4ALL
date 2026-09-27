"""FEA-corrected surrogate search: the surrogate proposes, the solver decides.

`opt_analysis surrogate` ranks every candidate with a HI-MGN forward pass, and on
DeepJEB that ranking is useful (Spearman 0.67 on vertical deflection, 0.79 on
peak stress over 125 generated shapes) while its absolute numbers are not: it
reads u_z about 19% low overall and 27% low below 0.8 kg, exactly the light
tail a mass search walks into. The search's winner therefore sits on the
surrogate's constraint boundary and past the solver's -- the v2 run's pick read
u_z 0.34 mm under tet4 against a 0.2 mm limit.

`opt_fea_rounds` wraps the surrogate search in a few rounds of real solves:

1. the baseline population is solved with tet4 at the verification resolution,
   and the allowables are calibrated on *those* numbers, not the surrogate's;
2. each round fits log(FEA / surrogate) for peak stress and deflection on every
   solved pair -- an intercept, log mass (the under-prediction grows as the part
   thins) and, once both kinds are present, an offset for searched designs (a
   search selects the shapes the surrogate is most optimistic about, so its
   picks are more biased than a random draw) -- and runs CMA-ES on the
   surrogate multiplied by that correction plus `margin` residual deviations;
3. the `topk` best distinct candidates under the corrected objective are
   generated at the verification resolution and solved, in parallel child
   processes, and join the fit;
4. the delivered design is the best *solved* design under the feasibility-first
   rule. Its numbers are the solver's, and the same STL is what
   `verify_with_fea.py` re-solves, so the verdict is reproducible.

Every solve runs `verify_with_fea.solve_surface` on an exported STL in its own
process with a hard timeout: gmsh runs in-process and can hang on a sliver
face, which would otherwise wedge the whole run, and going through the STL
file means an in-run solve and a later re-solve see byte-identical input.

  python design_loop/fea_correction.py --solve-job JOB.json   # one solve (internal)
"""

import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_loop.loop import calibrate, select_best      # noqa: E402
from design_loop.problem import MassObjective             # noqa: E402

# The responses the correction rescales. Mass is geometry, not response: the
# surrogate's mass is the marching-cubes surface volume and differs from the
# tet mesh's by a small constant factor, corrected without a margin.
CORRECTED_QUANTITIES = ('vertical_displacement', 'peak_von_mises', 'max_displacement')
# The per-case keys that carry the same three quantities.
_CASE_KEYS = {'max_vertical_displacement': 'vertical_displacement',
              'peak_von_mises': 'peak_von_mises',
              'max_displacement': 'max_displacement'}
MIN_PAIRS = 3                  # below this the correction is the identity
CANDIDATE_DIR = 'fea_candidates'
FEA_THREADS_PER_WORKER = 2
SOLVE_TIMEOUT_S = 1800


def _positive(value):
    try:
        return value is not None and float(value) > 0 and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _key(x):
    return tuple(np.round(np.asarray(x, dtype=float), 12).tolist())


# --------------------------------------------------------------------------- #
# The correction
# --------------------------------------------------------------------------- #

class Correction:
    """A fitted log(FEA / surrogate) model per response quantity.

    `terms[q]` holds the fitted columns (a subset of 'intercept', 'log_mass',
    'searched'), their coefficients and the residual standard deviation; a
    quantity with no columns is left as the surrogate predicted it. The linear
    predictor is clipped to +-`clip` so a fit on a handful of pairs cannot
    rescale a response by more than e^clip, and `margin` residual deviations
    are added on top, which makes a predicted response larger -- a
    conservative correction for a stress or deflection limit.
    """

    def __init__(self, terms, mass_log_factor, mass_ref, margin=0.5, clip=1.5,
                 ridge=1.0, notes=(), pairs=0, searched_pairs=0):
        self.terms = terms
        self.mass_log_factor = float(mass_log_factor)
        self.mass_ref = float(mass_ref)
        self.margin = float(margin)
        self.clip = float(clip)
        self.ridge = float(ridge)
        self.notes = list(notes)
        self.pairs = int(pairs)
        self.searched_pairs = int(searched_pairs)

    def log_factor(self, quantity, mass, searched=True, with_margin=True):
        term = self.terms.get(quantity) or {}
        columns = term.get('columns') or []
        if not columns:
            return 0.0
        row = []
        for column in columns:
            if column == 'intercept':
                row.append(1.0)
            elif column == 'log_mass':
                row.append(math.log(max(float(mass), 1e-12) / self.mass_ref))
            elif column == 'searched':
                row.append(1.0 if searched else 0.0)
            else:
                raise ValueError(f'unknown correction column {column!r}')
        predictor = float(np.clip(np.dot(term['coef'], row), -self.clip, self.clip))
        return predictor + (self.margin * float(term.get('sigma', 0.0)) if with_margin else 0.0)

    def factor(self, quantity, mass, searched=True, with_margin=True):
        return math.exp(self.log_factor(quantity, mass, searched, with_margin))

    def apply(self, fea, searched=True, with_margin=True):
        """A corrected copy of a surrogate result dict (SI, `_result_record` shape).

        The factors are evaluated at the surrogate's own mass (known before any
        solve). Top-level and per-case values of the corrected quantities are
        rescaled; everything else is copied unchanged.
        """
        mass = float(fea['mass'])
        factors = {q: self.factor(q, mass, searched, with_margin) for q in CORRECTED_QUANTITIES}
        out = dict(fea)
        out['mass'] = mass * math.exp(self.mass_log_factor)
        for quantity, f in factors.items():
            if fea.get(quantity) is not None:
                out[quantity] = float(fea[quantity]) * f
        cases = {}
        for name, values in (fea.get('cases') or {}).items():
            case = dict(values)
            for key, quantity in _CASE_KEYS.items():
                if case.get(key) is not None:
                    case[key] = float(case[key]) * factors[quantity]
            cases[name] = case
        if 'cases' in fea:
            out['cases'] = cases
        out['corrected'] = True
        return out

    def to_dict(self):
        terms = {}
        for quantity, term in self.terms.items():
            entry = {k: term[k] for k in ('columns', 'coef', 'sigma', 'n') if k in term}
            if term.get('columns'):
                entry['factor_at_mass_ref'] = {
                    'baseline': self.factor(quantity, self.mass_ref, False, False),
                    'searched': self.factor(quantity, self.mass_ref, True, False)}
            terms[quantity] = entry
        return {'form': 'log(FEA/surrogate) = intercept + b*log(m_sur/mass_ref) '
                        '+ c*searched, clipped, + margin*sigma',
                'mass_ref_kg': self.mass_ref, 'margin_k': self.margin, 'clip': self.clip,
                'ridge': self.ridge, 'pairs': self.pairs, 'searched_pairs': self.searched_pairs,
                'mass_factor': math.exp(self.mass_log_factor), 'terms': terms,
                'notes': self.notes}


def fit_correction(pairs, mass_ref, margin=0.5, ridge=1.0, min_pairs_for_slope=6, clip=1.5):
    """Fit a `Correction` on (surrogate_fea, fea_fea, searched) pairs.

    Per quantity, y = log(FEA / surrogate) is regressed on an intercept; with at
    least `min_pairs_for_slope` pairs also on log(m_surrogate / mass_ref), and on
    a searched-design indicator when both baseline and searched pairs are
    present. `ridge` shrinks every coefficient except the intercept, so a slope
    fitted on a dozen points stays modest. Fewer than `MIN_PAIRS` usable pairs
    leave the quantity uncorrected, and the returned notes say so.
    """
    pairs = [(s, f, bool(searched)) for s, f, searched in pairs if s and f]
    notes, terms = [], {}
    for quantity in CORRECTED_QUANTITIES:
        rows = [(s, f, searched) for s, f, searched in pairs
                if _positive(s.get(quantity)) and _positive(f.get(quantity))
                and _positive(s.get('mass'))]
        n = len(rows)
        if n < MIN_PAIRS:
            terms[quantity] = {'columns': [], 'coef': [], 'sigma': 0.0, 'n': n}
            if any(s.get(quantity) is not None for s, _, _ in pairs):
                notes.append(f'{quantity}: {n} usable pair(s); left uncorrected '
                             f'(needs >= {MIN_PAIRS})')
            continue
        y = np.log([float(f[quantity]) / float(s[quantity]) for s, f, _ in rows])
        columns, cols = ['intercept'], [np.ones(n)]
        if n >= min_pairs_for_slope:
            columns.append('log_mass')
            cols.append(np.log([float(s['mass']) / float(mass_ref) for s, _, _ in rows]))
            flags = np.array([searched for _, _, searched in rows], dtype=float)
            if 0 < flags.sum() < n:
                columns.append('searched')
                cols.append(flags)
        else:
            notes.append(f'{quantity}: {n} pairs; constant factor only '
                         f'(slope needs >= {min_pairs_for_slope})')
        X = np.column_stack(cols)
        penalty = float(ridge) * np.eye(len(columns))
        penalty[0, 0] = 0.0
        coef = np.linalg.lstsq(X.T @ X + penalty, X.T @ y, rcond=None)[0]
        resid = y - X @ coef
        sigma = float(np.sqrt(np.sum(resid ** 2) / max(n - len(columns), 1)))
        terms[quantity] = {'columns': columns, 'coef': [float(c) for c in coef],
                           'sigma': sigma, 'n': n}
    mass_ratios = [math.log(float(f['mass']) / float(s['mass'])) for s, f, _ in pairs
                   if _positive(s.get('mass')) and _positive(f.get('mass'))]
    mass_log_factor = float(np.mean(mass_ratios)) if mass_ratios else 0.0
    return Correction(terms, mass_log_factor, mass_ref, margin=margin, clip=clip, ridge=ridge,
                      notes=notes, pairs=len(pairs),
                      searched_pairs=sum(1 for _, _, searched in pairs if searched))


class CorrectedObjective:
    """`objective` evaluated on the corrected surrogate result.

    Drop-in for `surrogate_search`, which only calls it and reads
    `failure_score`; the penalty dict it returns is the corrected view.
    """

    def __init__(self, objective, correction, searched=True):
        self.objective = objective
        self.correction = correction
        self.searched = bool(searched)
        self.failure_score = objective.failure_score

    def __call__(self, result):
        return self.objective(self.correction.apply(result, searched=self.searched))


# --------------------------------------------------------------------------- #
# Solver records
# --------------------------------------------------------------------------- #

def truth_fea(rec):
    """A `verify_with_fea.solve_surface` record (mm / MPa) as an SI result dict.

    The measures are the ones `verify_with_fea.limit_verdicts` judges a
    surrogate run with: peak von Mises and max |u| on the label surface (the
    statistic the surrogate predicts), vertical deflection as the solver's max
    |u_z| over every node. Returns None when the solve failed or the
    label-surface statistics could not be formed.
    """
    if not rec or 'error' in rec:
        return None
    surface = rec.get('label_surface')
    if not surface or not surface.get('cases'):
        return None
    cases = {}
    for name, values in surface['cases'].items():
        cases[name] = {'peak_von_mises': float(values['peak_von_mises_MPa']) * 1e6,
                       'max_displacement': float(values['max_displacement_mm']) * 1e-3,
                       'max_vertical_displacement':
                           float(values['max_vertical_displacement_mm']) * 1e-3}
    uz = rec.get('vertical_displacement_mm')
    if uz is not None and 'vertical' in cases:
        cases['vertical']['max_vertical_displacement'] = float(uz) * 1e-3
    fea = {
        'mass': float(rec['mass_kg']),
        'cases': cases,
        'worst_case': max(cases, key=lambda n: cases[n]['peak_von_mises']),
        'peak_von_mises': float(surface['peak_von_mises_MPa']) * 1e6,
        'max_displacement': float(surface['max_displacement_mm']) * 1e-3,
        'vertical_displacement': float(uz) * 1e-3 if uz is not None else None,
        'num_tets': rec.get('tets'),
        'num_nodes': rec.get('nodes'),
    }
    if rec.get('max_von_mises_MPa') is not None:
        fea['max_von_mises'] = float(rec['max_von_mises_MPa']) * 1e6
    return fea


def brief(fea):
    """mm / MPa summary of an SI result dict (surrogate, corrected or solved)."""
    if not fea:
        return None
    out = {'mass_kg': float(fea['mass']),
           'peak_von_mises_MPa': float(fea['peak_von_mises']) / 1e6,
           'max_displacement_mm': float(fea['max_displacement']) * 1e3}
    if fea.get('vertical_displacement') is not None:
        out['vertical_displacement_mm'] = float(fea['vertical_displacement']) * 1e3
    return out


def problem_spec(bracket, resolution):
    """Everything a child solve needs to rebuild the run's `Bracket`."""
    m = bracket.material
    return {'material': {'name': m.name, 'E': m.E, 'nu': m.nu, 'rho': m.rho,
                         'yield_stress': m.yield_stress},
            'length_scale': bracket.length_scale,
            'load_cases': list(bracket.load_cases),
            'stress_percentile': bracket.stress_percentile,
            'resolution': {k: resolution[k]
                           for k in ('target_faces', 'mesh_size_max', 'target_nodes')}}


def _write_json(path, payload):
    with open(path + '.tmp', 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1, default=float)
    os.replace(path + '.tmp', path)


def _solve_job(job_path):
    """Child-process entry: solve one STL, write its record next to the job."""
    with open(job_path, encoding='utf-8') as fh:
        job = json.load(fh)
    started = time.time()
    try:
        from design_loop import fea
        from design_loop.problem import Bracket
        from design_loop.verify_with_fea import load_stl, solve_surface
        bracket = Bracket(material=fea.Material(**job['material']),
                          length_scale=float(job['length_scale']),
                          load_cases=tuple(job['load_cases']),
                          stress_percentile=float(job['stress_percentile']))
        rec = solve_surface(load_stl(job['stl']), bracket, job['resolution'])
    except Exception as exc:
        rec = {'error': f'{type(exc).__name__}: {exc}'[:2000]}
    rec['seconds'] = time.time() - started
    _write_json(job['out'], rec)
    print(describe(rec), flush=True)
    return 0


def describe(rec):
    if 'error' in rec:
        return f"FAILED {rec['error'][:200]}"
    uz = rec.get('vertical_displacement_mm')
    surface = rec.get('label_surface') or {}
    return (f"mass {rec['mass_kg']:.4f} kg"
            + (f" | u_z {uz:.4f} mm" if uz is not None else '')
            + (f" | label-surface peak vM {surface['peak_von_mises_MPa']:.1f} MPa"
               if surface.get('peak_von_mises_MPa') is not None else '')
            + f" | {rec.get('tets', 0):,} tets ({rec.get('seconds', 0):.0f} s)")


def solve_stls(jobs, problem, workers=4, timeout=SOLVE_TIMEOUT_S,
               threads=FEA_THREADS_PER_WORKER, verbose=True):
    """Solve [{'name', 'stl', 'out'}, ...] in parallel child processes.

    One process per shape with a hard `timeout`, `workers` at a time, each held
    to `threads` BLAS/OpenMP threads. Returns {name: record}; a timeout or a
    crashed child is an error record, never an exception.
    """
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads))
    script = os.path.abspath(__file__)

    def one(job):
        job_path = os.path.splitext(job['out'])[0] + '.job.json'
        if os.path.exists(job['out']):
            os.remove(job['out'])
        _write_json(job_path, dict(problem, stl=job['stl'], out=job['out']))
        started = time.time()
        try:
            proc = subprocess.run([sys.executable, '-u', script, '--solve-job', job_path],
                                  env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return job['name'], {'error': f'TimeoutError: tet4 solve exceeded {timeout:.0f} s',
                                 'seconds': time.time() - started}
        if proc.returncode == 0 and os.path.exists(job['out']):
            with open(job['out'], encoding='utf-8') as fh:
                return job['name'], json.load(fh)
        tail = (proc.stderr or proc.stdout or '').strip()[-600:]
        return job['name'], {'error': f'solver process exited {proc.returncode}: {tail}',
                             'seconds': time.time() - started}

    results = {}
    if not jobs:
        return results
    with ThreadPoolExecutor(max(1, min(int(workers), len(jobs)))) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            name, rec = future.result()
            results[name] = rec
            if verbose:
                print(f'  [tet4 {i}/{len(jobs)}] {name}: {describe(rec)}', flush=True)
    return results


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #

def pick_candidates(records, correction, objective, exclude_keys, k):
    """The `k` best distinct unsolved search records under the corrected objective.

    Ranked feasible-first, then by corrected score (the rule the delivered
    design is chosen by), ties broken by evaluation order. Records whose design
    vector is in `exclude_keys` (already solved) or repeats an earlier pick are
    skipped. Returns [(record, score, penalty), ...].
    """
    scored = []
    for rec in records:
        if not rec.get('ok'):
            continue
        score, penalty = objective(correction.apply(rec['fea'], searched=True))
        scored.append((not penalty['feasible'], score, rec.get('index', 0), rec, penalty))
    scored.sort(key=lambda t: t[:3])
    seen, picks = set(exclude_keys), []
    for _, score, _, rec, penalty in scored:
        key = _key(rec['x'])
        if key in seen:
            continue
        seen.add(key)
        picks.append((rec, score, penalty))
        if len(picks) >= k:
            break
    return picks


def _log_error(truth, predicted, quantity):
    if _positive(truth.get(quantity)) and _positive(predicted.get(quantity)):
        return math.log(float(predicted[quantity]) / float(truth[quantity]))
    return None


def prediction_check(solved_picks):
    """Out-of-sample error of the raw and corrected surrogate on the solved picks.

    Each pick carries the central corrected prediction made *before* it was
    solved, so this is the correction's error on designs it had not seen.
    """
    out = {}
    for quantity in ('vertical_displacement', 'peak_von_mises'):
        raw, corrected = [], []
        for rec in solved_picks:
            a = _log_error(rec['fea'], rec['surrogate_fea'], quantity)
            b = _log_error(rec['fea'], rec.get('predicted_central') or {}, quantity)
            if a is not None and b is not None:
                raw.append(a)
                corrected.append(b)
        if raw:
            out[quantity] = {
                'n': len(raw),
                'raw_mean_abs_log_error': float(np.mean(np.abs(raw))),
                'corrected_mean_abs_log_error': float(np.mean(np.abs(corrected))),
                'raw_bias': float(math.exp(np.mean(raw)) - 1.0),
                'corrected_bias': float(math.exp(np.mean(corrected)) - 1.0),
            }
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _penalty_brief(penalty):
    return {k: penalty[k] for k in ('feasible', 'stress_violation', 'disp_violation')
            if k in penalty}


def _truth_summary(rec):
    out = {'name': rec['name'], 'tag': rec['tag'], 'round': rec['round'],
           'index': rec['index'], 'stl': rec.get('stl')}
    if rec.get('ok'):
        out.update(fea=brief(rec['fea']), score=rec.get('score'),
                   penalty=_penalty_brief(rec.get('penalty') or {}),
                   surrogate=brief(rec['surrogate_fea']))
        if rec.get('predicted') is not None:
            out['predicted'] = brief(rec['predicted'])
        if rec.get('tets') is not None:
            out['tets'] = rec['tets']
    else:
        out['error'] = rec.get('error')
    if rec.get('seconds') is not None:
        out['seconds'] = rec['seconds']
    return out


def run_fea_corrected(generator, surrogate, bracket, baseline, out_dir, *, rounds,
                      resolution, topk=4, workers=4, margin=0.5, budget=120, popsize=8,
                      sigma0=1.0, seed=0, stress_margin=1.0, disp_margin=1.0,
                      vertical_disp_allow=None, stress_weight=6.0, disp_weight=3.0,
                      solve_fn=None, verbose=True):
    """Run the FEA-corrected search (module docstring). Returns a result dict.

    `baseline` is the surrogate-scored baseline population (with 'x' and
    'fea'); `resolution` holds mc_resolution / target_faces / mesh_size_max /
    target_nodes for every solve. `solve_fn(jobs) -> {name: record}` replaces
    the parallel tet4 solver (tests).
    """
    from design_loop.surrogate import surrogate_search

    started = time.time()
    cand_dir = os.path.join(out_dir, CANDIDATE_DIR)
    if os.path.isdir(cand_dir):
        shutil.rmtree(cand_dir)
    os.makedirs(cand_dir)
    if solve_fn is None:
        problem = problem_spec(bracket, resolution)

        def solve_fn(jobs):
            return solve_stls(jobs, problem, workers=workers, verbose=verbose)

    stats = {'solves': 0, 'solve_failures': 0, 'generation_failures': 0,
             'solve_wall_time_s': 0.0}

    def solve(entries):
        """Generate each entry at the verification resolution, export, solve."""
        records, jobs = [], []
        for entry in entries:
            rec = dict(entry, stl=f"{CANDIDATE_DIR}/{entry['name']}.stl")
            try:
                mesh, _ = generator.generate(np.asarray(entry['x'], dtype=float),
                                             mc_resolution=resolution['mc_resolution'])
                if mesh is None:
                    raise RuntimeError('no zero crossing in the decoded SDF')
                path = os.path.join(cand_dir, f"{entry['name']}.stl")
                mesh.export(path)
            except Exception as exc:
                rec.update(ok=False, error=f'{type(exc).__name__}: {exc}', stl=None)
                stats['generation_failures'] += 1
                records.append(rec)
                continue
            jobs.append({'name': entry['name'], 'stl': os.path.abspath(path),
                         'out': os.path.abspath(os.path.join(cand_dir, f"{entry['name']}.json"))})
            records.append(rec)
        t = time.time()
        solved = solve_fn(jobs) if jobs else {}
        stats['solve_wall_time_s'] += time.time() - t
        for rec in records:
            if rec.get('ok') is False:
                continue
            raw = solved.get(rec['name']) or {'error': 'no solver record returned'}
            stats['solves'] += 1
            rec['seconds'] = raw.get('seconds')
            rec['tets'] = raw.get('tets')
            fea = truth_fea(raw)
            if fea is None:
                rec.update(ok=False, error=raw.get('error') or raw.get('label_surface_error')
                           or 'no label-surface statistics')
                stats['solve_failures'] += 1
            else:
                rec.update(ok=True, fea=fea)
        return records

    def score(records, objective):
        for rec in records:
            if rec.get('ok'):
                rec['score'], rec['penalty'] = objective(rec['fea'])
            else:
                rec['score'] = objective.failure_score
                rec['penalty'] = {'feasible': False, 'reason': rec.get('error')}

    # ---- Baseline population, solved -------------------------------------- #
    base_entries = [{'name': f"baseline_{int(r['index']):03d}", 'tag': 'baseline', 'round': 0,
                     'index': int(r['index']), 'x': list(r['x']), 'surrogate_fea': r['fea']}
                    for r in baseline if r.get('ok')]
    if verbose:
        print(f"  solving the {len(base_entries)} baseline designs with tet4 "
              f"(MC{resolution['mc_resolution']}, {resolution['target_faces']} faces, "
              f"mesh_size_max {resolution['mesh_size_max']}, {workers} worker(s))", flush=True)
    truth = solve(base_entries)
    truth_base = [t for t in truth if t.get('ok')]
    try:
        limits = calibrate(truth_base, stress_margin=stress_margin, disp_margin=disp_margin)
    except RuntimeError as exc:
        raise RuntimeError(f'opt_fea_rounds: {exc} under tet4 at the verification '
                           'resolution; cannot calibrate FEA allowables') from exc
    limits['vertical_disp_allow'] = vertical_disp_allow
    uz = [t['fea']['vertical_displacement'] for t in truth_base
          if t['fea'].get('vertical_displacement') is not None]
    if uz:
        limits['vertical_disp_range'] = [float(min(uz)), float(max(uz))]
        limits['vertical_disp_median'] = float(np.median(uz))
    limits['calibrated_on'] = 'fea'
    limits['resolution'] = dict(resolution)
    objective = MassObjective(mass_ref=limits['mass_ref'], stress_allow=limits['stress_allow'],
                              disp_allow=limits['disp_allow'], stress_weight=stress_weight,
                              disp_weight=disp_weight, vertical_disp_allow=vertical_disp_allow)
    score(truth, objective)
    pairs = [(t['surrogate_fea'], t['fea'], False) for t in truth_base]
    if verbose:
        feasible = sum(1 for t in truth_base if t['penalty']['feasible'])
        print(f"  FEA allowables: stress {limits['stress_allow'] / 1e6:.1f} MPa, mass_ref "
              f"{limits['mass_ref']:.4f} kg; {feasible}/{len(truth_base)} baseline designs "
              'FEA-feasible', flush=True)

    # ---- Rounds ------------------------------------------------------------ #
    per_round = max(int(popsize), int(budget) // int(rounds))
    search_history, log, rounds_log, solved_picks = [], [], [], []
    solved_keys = {_key(t['x']) for t in truth}
    generations = 0
    for r in range(1, int(rounds) + 1):
        correction = fit_correction(pairs, limits['mass_ref'], margin=margin)
        incumbent = select_best([t for t in truth if t.get('ok')])
        if verbose:
            factors = ', '.join(
                f"{q} x{correction.factor(q, limits['mass_ref'], True, False):.3f}"
                for q in ('vertical_displacement', 'peak_von_mises')
                if (correction.terms.get(q) or {}).get('columns'))
            print(f"\n--- FEA-corrected round {r}/{rounds}: {len(pairs)} solved pair(s)"
                  f"{'; ' + factors if factors else ''}; starting from {incumbent['name']} "
                  f"({incumbent['fea']['mass']:.4f} kg, "
                  f"{'feasible' if incumbent['penalty']['feasible'] else 'infeasible'}) ---",
                  flush=True)
        _, _, round_log, history = surrogate_search(
            generator, surrogate, CorrectedObjective(objective, correction),
            x0=np.asarray(incumbent['x'], dtype=float), sigma0=sigma0, budget=per_round,
            popsize=popsize, seed=int(seed) + r - 1, verbose=verbose)
        offset = len(search_history)
        for rec in history:
            rec['index'] = int(rec['index']) + offset
            rec['round'] = r
        for entry in round_log:
            log.append(dict(entry, round=r, evaluations=entry['evaluations'] + offset,
                            generation=entry['generation'] + generations))
        generations += len(round_log)
        search_history += history

        picks = pick_candidates(search_history, correction, objective, solved_keys, topk)
        entries = []
        for rec, pick_score, pick_penalty in picks:
            solved_keys.add(_key(rec['x']))
            entries.append({'name': f"search_{int(rec['index']):03d}", 'tag': 'search',
                            'round': int(rec['round']), 'picked_in_round': r,
                            'index': int(rec['index']), 'x': list(rec['x']),
                            'surrogate_fea': rec['fea'],
                            'predicted': correction.apply(rec['fea'], searched=True),
                            'predicted_central': correction.apply(rec['fea'], searched=True,
                                                                  with_margin=False),
                            'predicted_score': pick_score,
                            'predicted_feasible': bool(pick_penalty['feasible'])})
        if verbose:
            print(f'  solving the {len(entries)} best candidate(s) under the corrected '
                  'surrogate with tet4', flush=True)
        new = solve(entries)
        score(new, objective)
        for t in new:
            if t.get('ok'):
                pairs.append((t['surrogate_fea'], t['fea'], True))
                solved_picks.append(t)
        truth += new
        best = select_best([t for t in truth if t.get('ok')])
        corrected_feasible = sum(
            1 for rec in history if rec.get('ok') and rec['penalty'].get('feasible'))
        rounds_log.append({
            'round': r, 'pairs_fitted': correction.pairs,
            'searched_pairs_fitted': correction.searched_pairs,
            'correction': correction.to_dict(),
            'search_evaluations': len(history),
            'search_feasible_corrected': corrected_feasible,
            'picks': [dict(_truth_summary(t), predicted_feasible=t['predicted_feasible'],
                           predicted_score=t['predicted_score']) for t in new],
            'picks_feasible_fea': sum(1 for t in new
                                      if t.get('ok') and t['penalty']['feasible']),
            'incumbent': {'name': best['name'], 'mass_kg': float(best['fea']['mass']),
                          'score': float(best['score']),
                          'feasible': bool(best['penalty']['feasible'])},
        })
        if verbose:
            print(f"  round {r}: {rounds_log[-1]['picks_feasible_fea']}/{len(new)} pick(s) "
                  f"FEA-feasible; incumbent {best['name']} {best['fea']['mass']:.4f} kg "
                  f"({'feasible' if best['penalty']['feasible'] else 'infeasible'})", flush=True)

    final = fit_correction(pairs, limits['mass_ref'], margin=margin)
    solved = [t for t in truth if t.get('ok')]
    solved_base = [t for t in solved if t['tag'] == 'baseline']
    delivered = select_best(solved)
    reference = select_best(solved_base)
    typical = sorted(solved_base, key=lambda t: t['fea']['mass'])[len(solved_base) // 2]

    def design(rec, source):
        return {'name': rec['name'], 'source': source, 'stl': rec['stl'],
                'fea': brief(rec['fea']), 'surrogate': brief(rec['surrogate_fea']),
                'score': float(rec['score']), 'penalty': _penalty_brief(rec['penalty'])}

    summary = {
        'rounds': int(rounds), 'topk': int(topk), 'margin_k': float(margin),
        'workers': int(workers), 'search_budget_per_round': per_round,
        'resolution': dict(resolution),
        'solves': stats['solves'], 'solve_failures': stats['solve_failures'],
        'generation_failures': stats['generation_failures'],
        'solve_wall_time_s': stats['solve_wall_time_s'],
        'wall_time_s': time.time() - started,
        'baseline': [_truth_summary(t) for t in truth if t['tag'] == 'baseline'],
        'rounds_log': rounds_log,
        'final_correction': final.to_dict(),
        'designs': {
            'optimized': design(delivered, (f"search round {delivered['round']}"
                                            if delivered['tag'] == 'search'
                                            else 'baseline member')),
            'baseline': design(reference, 'best baseline member (FEA)'),
            'typical': design(typical, 'median-mass baseline member (FEA)'),
        },
        'feasible_solved': sum(1 for t in solved if t['penalty']['feasible']),
        'improved_on_baseline': delivered is not reference,
        'prediction_check': prediction_check(solved_picks),
    }
    return {'limits': limits, 'objective': objective, 'truth': truth,
            'search_history': search_history, 'log': log, 'delivered': delivered,
            'reference': reference, 'typical': typical, 'correction': final,
            'summary': summary}


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--solve-job':
        sys.exit(_solve_job(sys.argv[2]))
    sys.exit('usage: fea_correction.py --solve-job JOB.json  (run by mode optimize)')
