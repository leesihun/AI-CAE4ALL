"""The closed loop: generate -> mesh -> analyze -> score -> search.

Search runs with CMA-ES over the generator's design vector. Every evaluation is
a full forward chain -- flow-matching ODE, marching cubes, gmsh tetrahedra, a
linear-static solve per load case -- so failures are expected and are absorbed
as a finite penalty rather than an exception, which keeps the population moving
instead of collapsing.
"""

import json
import os
import time
import traceback

import numpy as np

from .mesher import tet_mesh_from_surface
from .problem import Bracket, MassObjective


class Evaluator:
    """One design vector in, one score out, with the full chain cached per call."""

    def __init__(self, generator, bracket, objective=None, mesh_size_max=0.05,
                 target_faces=12000, artifact_dir=None):
        self.generator = generator
        self.bracket = bracket
        self.objective = objective
        self.mesh_size_max = mesh_size_max
        self.target_faces = target_faces
        self.artifact_dir = artifact_dir
        self.history = []
        self.failures = {}

    def analyze(self, x, mc_resolution=None, target_faces=None, mesh_size_max=None,
                return_fields=False):
        """Run the chain. Returns a record dict; 'ok' False carries the reason."""
        record = {'x': np.asarray(x, dtype=float).tolist()}
        timings = {}
        t = time.time()
        try:
            mesh, gen_info = self.generator.generate(x, mc_resolution=mc_resolution)
            timings['generate'] = time.time() - t
            if mesh is None:
                raise RuntimeError('no zero crossing in the decoded SDF')
            record['generation'] = gen_info

            t = time.time()
            nodes, tets, mesh_info = tet_mesh_from_surface(
                mesh,
                mesh_size_max=mesh_size_max or self.mesh_size_max,
                target_faces=target_faces or self.target_faces)
            timings['mesh'] = time.time() - t
            record['mesh'] = mesh_info

            t = time.time()
            record['fea'] = self.bracket.analyze(nodes, tets, return_fields=return_fields)
            if return_fields:
                record['fields'] = record['fea'].pop('fields')
            timings['fea'] = time.time() - t
            record['ok'] = True
            record['mesh_object'] = mesh
        except Exception as exc:
            record['ok'] = False
            record['error'] = f'{type(exc).__name__}: {exc}'
            record['traceback'] = traceback.format_exc(limit=3)
            self.failures[type(exc).__name__] = self.failures.get(type(exc).__name__, 0) + 1
        record['timings'] = timings
        return record

    def __call__(self, x):
        """CMA-ES entry point: score only, with failures penalized not raised."""
        record = self.analyze(x)
        if not record['ok']:
            record['score'] = self.objective.failure_score
            record['penalty'] = {'feasible': False, 'reason': record['error']}
        else:
            score, penalty = self.objective(record['fea'])
            record['score'] = score
            record['penalty'] = penalty
        record.pop('mesh_object', None)
        record['index'] = len(self.history)
        self.history.append(record)
        return record['score']


def baseline_population(evaluator, size=12, seed=0, spread=1.0):
    """Sample the design space to calibrate the objective against typical DeepJEB parts.

    Allowables anchored to this population make the constraints active by
    construction: the target is a bracket lighter than a typical sample without
    being weaker or more compliant than one.
    """
    rng = np.random.default_rng(seed)
    lo, hi = evaluator.generator.bounds()
    records = []
    for i in range(size):
        x = rng.uniform(lo, hi) * spread
        record = evaluator.analyze(x)
        record['index'] = i
        status = 'ok' if record['ok'] else record['error']
        if record['ok']:
            f = record['fea']
            status = (f"mass={f['mass']:.3f}kg  peak_vm={f['peak_von_mises'] / 1e6:.1f}MPa  "
                      f"disp={f['max_displacement'] * 1e3:.3f}mm  tets={f['num_tets']}")
        print(f'  baseline {i:02d}/{size}: {status}', flush=True)
        record.pop('mesh_object', None)
        records.append(record)
    return records


def calibrate(records, stress_margin=1.0, disp_margin=1.0):
    """Median-of-population allowables. Raises if too few designs survived."""
    ok = [r for r in records if r['ok']]
    if len(ok) < 3:
        raise RuntimeError(f'only {len(ok)} baseline designs analyzed successfully')
    mass = np.array([r['fea']['mass'] for r in ok])
    stress = np.array([r['fea']['peak_von_mises'] for r in ok])
    disp = np.array([r['fea']['max_displacement'] for r in ok])
    return {
        'population': len(ok),
        'mass_ref': float(np.median(mass)),
        'mass_range': [float(mass.min()), float(mass.max())],
        'stress_allow': float(np.median(stress) * stress_margin),
        'stress_range': [float(stress.min()), float(stress.max())],
        'disp_allow': float(np.median(disp) * disp_margin),
        'disp_range': [float(disp.min()), float(disp.max())],
    }


def run(evaluator, x0=None, sigma0=1.0, budget=120, popsize=8, seed=0, verbose=True):
    """CMA-ES over the bounded design space. Returns (best_x, best_score, log)."""
    import cma

    lo, hi = evaluator.generator.bounds()
    if x0 is None:
        x0 = np.zeros(evaluator.generator.n_design)
    options = {
        'bounds': [lo.tolist(), hi.tolist()],
        'popsize': popsize,
        'maxfevals': budget,
        'seed': seed + 1,          # cma treats seed 0 as "pick randomly"
        'verbose': -9,
    }
    es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float), sigma0, options)

    log = []
    generation = 0
    while not es.stop() and len(evaluator.history) < budget:
        solutions = es.ask()
        scores = []
        for x in solutions:
            scores.append(evaluator(x))
        es.tell(solutions, scores)
        generation += 1
        best = min(evaluator.history, key=lambda r: r['score'])
        feasible = [r for r in evaluator.history
                    if r['ok'] and r['penalty'].get('feasible')]
        entry = {
            'generation': generation,
            'evaluations': len(evaluator.history),
            'generation_best': float(min(scores)),
            'generation_median': float(np.median(scores)),
            'best_score': float(best['score']),
            'best_mass': float(best['fea']['mass']) if best['ok'] else None,
            'feasible_count': len(feasible),
        }
        log.append(entry)
        if verbose:
            print(f"  gen {generation:02d} | evals {entry['evaluations']:3d} | "
                  f"gen-best {entry['generation_best']:.4f} | "
                  f"gen-median {entry['generation_median']:.4f} | "
                  f"best {entry['best_score']:.4f} | feasible {len(feasible)}", flush=True)

    best = min(evaluator.history, key=lambda r: r['score'])
    return np.asarray(best['x']), best['score'], log


def save_history(path, evaluator, extra=None):
    payload = {
        'evaluations': [{k: v for k, v in r.items()
                         if k not in ('mesh_object', 'traceback', 'fields')}
                        for r in evaluator.history],
        'failures': evaluator.failures,
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, default=float)
    return path
