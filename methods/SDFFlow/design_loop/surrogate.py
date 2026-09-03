"""Surrogate analysis: HI-MGN stands in for the structural solve.

The FEA path in `problem.py` costs about 13 s per candidate -- gmsh plus a
linear solve per load case. This path replaces it with one forward pass of a
trained DeepJEB surrogate, and does it **a whole generation at a time**: every
candidate in a CMA-ES population is bridged into a single HDF5 and scored in one
native inference call, so the process start-up that dominates a single-shape
prediction is amortised across the batch.

Running the surrogate as a subprocess rather than importing it is deliberate.
It is how every other stage of the suite is invoked, it keeps the method repos
isolated (HI-MGN gets its own interpreter via the launcher's settings), and the
checkpoint's own `model_config` stays authoritative for architecture.

Mass does not come from the surrogate. It is a geometric property of the
generated mesh, so it is computed exactly and for free -- only the fields that
would need a solver are predicted.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

from design_loop.deepjeb_bridge import (
    DEEPJEB_CENTRE, DEEPJEB_MAX_SIDE, LOAD_CASES, SDF_TARGET_EXTENT,
    mesh_to_records, write_inference_contract,
)

# <suite>/methods/SDFFlow/design_loop/surrogate.py -> <suite>
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUITE = os.path.dirname(os.path.dirname(_REPO))

STRESS_PERCENTILE = 99.5


class SurrogateError(RuntimeError):
    pass


class HIMGNSurrogate:
    """Batch structural prediction for generated brackets."""

    def __init__(self, config_path, checkpoint, python=None, load_cases=('ver', 'dia'),
                 target_nodes=5000, density=4430.0, workdir=None,
                 stress_percentile=STRESS_PERCENTILE):
        self.config_path = os.path.abspath(config_path)
        self.checkpoint = os.path.abspath(checkpoint)
        self.python = python or sys.executable
        self.load_cases = tuple(load_cases)
        unknown = set(self.load_cases) - set(LOAD_CASES)
        if unknown:
            raise SurrogateError(f'unknown load case(s) {sorted(unknown)}; '
                                 f'available {LOAD_CASES}')
        self.target_nodes = int(target_nodes)
        self.density = float(density)
        self.stress_percentile = float(stress_percentile)
        # Absolute, deliberately: the native subprocess this class shells out to
        # runs with cwd=_SUITE, not whatever directory the caller (SDFFlow's own
        # process, cwd=methods/SDFFlow) is in. A relative workdir resolved
        # correctly for this process's own file I/O but silently pointed the
        # child process's --config and infer_dataset at the wrong base directory.
        self.workdir = os.path.abspath(workdir or tempfile.mkdtemp(prefix='himgn_screen_'))
        os.makedirs(self.workdir, exist_ok=True)
        self.calls = 0
        self.predicted = 0

    # ------------------------------------------------------------------ #

    def mass_of(self, mesh):
        """Exact mass in kg from the generated geometry, in the DeepJEB frame."""
        scale = DEEPJEB_MAX_SIDE / SDF_TARGET_EXTENT       # normalized -> mm
        volume_mm3 = abs(float(mesh.volume)) * scale ** 3
        return volume_mm3 * 1e-9 * self.density            # mm^3 -> m^3 -> kg

    def analyze_batch(self, meshes, names=None):
        """Predict fields for a list of generated meshes.

        Returns one result dict per mesh, in the order given; an entry is None
        where that candidate could not be bridged.
        """
        names = names or [f'cand{i:03d}' for i in range(len(meshes))]
        records, owner = [], []
        for index, (mesh, name) in enumerate(zip(meshes, names)):
            try:
                recs = mesh_to_records(mesh, load_cases=self.load_cases,
                                       target_nodes=self.target_nodes, name=name)
            except Exception:
                continue
            records += recs
            owner += [index] * len(recs)
        if not records:
            return [None] * len(meshes)

        call = self.calls
        self.calls += 1
        infer_path = os.path.join(self.workdir, f'batch{call:04d}.h5')
        write_inference_contract(infer_path, records)
        rollout_dir = os.path.join(self.workdir, f'batch{call:04d}_rollout')
        predictions = self._run_native(infer_path, rollout_dir, len(records))
        self.predicted += len(records)

        results = [None] * len(meshes)
        for sample_id, rec in enumerate(records, start=1):
            pred = predictions.get(sample_id)
            if pred is None:
                continue
            index = owner[sample_id - 1]
            entry = results[index] or {'cases': {}, 'mass': self.mass_of(meshes[index])}
            # Each generated candidate is sampled independently and may land a
            # few nodes either side of target_nodes. Preserve this record's
            # actual graph size; using records[0] mislabeled every later design
            # with the first candidate's cardinality.
            entry['num_nodes'] = int(rec['nodal'].shape[2])
            # DeepJEB's own units are MPa and mm; every downstream consumer
            # (MassObjective, calibrate(), the printed and JSON reports) is
            # written against the FEA path's SI convention (Pa, metres) --
            # measured directly: an uncorrected run printed "peak_vm=0.0MPa"
            # and "disp=213.9mm" for real per-file values of 89 MPa / 0.21 mm,
            # off by the /1e6 and *1e3 the FEA-side print statements apply.
            # Convert once here so nothing downstream has to special-case units.
            stress_mpa, disp_mm = pred[3, :], pred[4, :]
            entry['cases'][rec['case']] = {
                'peak_von_mises': float(np.percentile(np.abs(stress_mpa),
                                                      self.stress_percentile) * 1e6),
                'max_von_mises': float(np.abs(stress_mpa).max() * 1e6),
                'max_displacement': float(np.abs(disp_mm).max() * 1e-3),
            }
            results[index] = entry

        for index, entry in enumerate(results):
            if entry is None or not entry['cases']:
                results[index] = None
                continue
            cases = entry['cases']
            entry.update(
                peak_von_mises=max(c['peak_von_mises'] for c in cases.values()),
                max_von_mises=max(c['max_von_mises'] for c in cases.values()),
                max_displacement=max(c['max_displacement'] for c in cases.values()),
                volume=entry['mass'] / self.density,
            )
        return results

    # ------------------------------------------------------------------ #

    def _run_native(self, infer_path, rollout_dir, expected):
        """One launcher call over the whole batch; returns {sample_id: [F,N]}."""
        import h5py

        overrides = {
            'infer_dataset': infer_path,
            'modelpath': self.checkpoint,
            'inference_output_dir': rollout_dir,
        }
        config_path = self._materialise_config(overrides, rollout_dir)
        cmd = [self.python, os.path.join(_SUITE, 'AI_CAE4ALL_main.py'),
               '--config', config_path]
        proc = subprocess.run(cmd, cwd=_SUITE, capture_output=True, text=True,
                              encoding='utf-8', errors='replace')
        if proc.returncode != 0:
            tail = (proc.stdout or '')[-1500:] + (proc.stderr or '')[-1500:]
            raise SurrogateError(f'surrogate inference failed (exit {proc.returncode}):\n{tail}')

        predictions = {}
        search = rollout_dir if os.path.isdir(rollout_dir) else \
            os.path.join(_SUITE, 'output', 'chi-mgnflow', 'rollout')
        import glob
        import re
        for path in glob.glob(os.path.join(search, 'rollout_sample*_steps*.h5')):
            m = re.search(r'rollout_sample(\d+)_', os.path.basename(path))
            if not m:
                continue
            with h5py.File(path, 'r') as f:
                key = next(iter(f['data']))
                predictions[int(m.group(1))] = f['data'][key]['nodal_data'][:, -1, :]
        if not predictions:
            raise SurrogateError(f'no rollout files under {search}')
        return predictions

    def _materialise_config(self, overrides, rollout_dir):
        """Copy the surrogate config with this batch's paths substituted in."""
        with open(self.config_path, encoding='utf-8') as fh:
            lines = fh.read().splitlines()
        seen = set()
        out = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('%'):
                key = stripped.split('#')[0].split()[0].lower() if stripped.split() else ''
                if key in overrides:
                    out.append(f'{key}\t{overrides[key]}')
                    seen.add(key)
                    continue
            out.append(line)
        for key, value in overrides.items():
            if key not in seen:
                out.append(f'{key}\t{value}')

        os.makedirs(rollout_dir, exist_ok=True)
        path = os.path.join(rollout_dir, 'config_surrogate.txt')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(out) + '\n')
        return path

    def stats(self):
        return {'native_calls': self.calls, 'graphs_predicted': self.predicted,
                'load_cases': list(self.load_cases), 'workdir': self.workdir}


class SurrogateEvaluator:
    """Same call surface as `loop.Evaluator`, backed by the HI-MGN surrogate.

    `run_optimize` swaps this in for the FEA `Evaluator` when `opt_analysis
    surrogate` is set, so the same baseline-calibration / CMA-ES / verification
    driver in `loop.py` runs unchanged over either backend. What is NOT
    available here: a volume mesh (no tet count, no mesh_sensitivity), per-load
    interface node sets (no stress-field render), and compliance (needs nodal
    displacement work done through a solved system, which this path never
    forms). Those fields are absent from surrogate records rather than filled
    with zero or null placeholders that could be mistaken for measurements.
    """

    def __init__(self, generator, surrogate, objective=None):
        self.generator = generator
        self.surrogate = surrogate
        self.objective = objective
        self.history = []
        self.failures = {}
        self.supports_fields = False

    def analyze(self, x, mc_resolution=None, target_faces=None, mesh_size_max=None,
               return_fields=False):
        record = {'x': np.asarray(x, dtype=float).tolist(), 'timings': {}}
        import time
        t = time.time()
        try:
            mesh, gen_info = self.generator.generate(x, mc_resolution=mc_resolution)
            record['timings']['generate'] = time.time() - t
            if mesh is None:
                raise RuntimeError('no zero crossing in the decoded SDF')
            record['generation'] = gen_info

            t = time.time()
            result = self.surrogate.analyze_batch([mesh])[0]
            record['timings']['surrogate'] = time.time() - t
            if result is None:
                raise SurrogateError('surrogate could not bridge or predict this shape')

            record['fea'] = {
                'mass': result['mass'], 'volume': result['volume'],
                'num_nodes': result['num_nodes'],
                'cases': result['cases'], 'worst_case': max(result['cases'],
                    key=lambda name: result['cases'][name]['peak_von_mises']),
                'peak_von_mises': result['peak_von_mises'],
                'max_von_mises': result['max_von_mises'],
                'max_displacement': result['max_displacement'],
            }
            record['mesh'] = {'num_nodes': result['num_nodes'], 'surrogate': True}
            record['ok'] = True
            record['mesh_object'] = mesh
        except Exception as exc:
            record['ok'] = False
            record['error'] = f'{type(exc).__name__}: {exc}'
            self.failures[type(exc).__name__] = self.failures.get(type(exc).__name__, 0) + 1
        return record

    def __call__(self, x):
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


def _result_record(x, mesh, gen_info, result):
    """Build the FEA-shaped record dict for one candidate's surrogate result."""
    record = {'x': np.asarray(x, dtype=float).tolist(), 'generation': gen_info}
    if mesh is None:
        record['ok'] = False
        record['error'] = 'RuntimeError: no zero crossing in the decoded SDF'
        return record
    if result is None:
        record['ok'] = False
        record['error'] = 'SurrogateError: surrogate could not bridge or predict this shape'
        return record
    record['ok'] = True
    record['mesh_object'] = mesh
    record['fea'] = {
        'mass': result['mass'], 'volume': result['volume'],
        'num_nodes': result['num_nodes'],
        'cases': result['cases'],
        'worst_case': max(result['cases'],
                          key=lambda name: result['cases'][name]['peak_von_mises']),
        'peak_von_mises': result['peak_von_mises'],
        'max_von_mises': result['max_von_mises'],
        'max_displacement': result['max_displacement'],
    }
    record['mesh'] = {'num_nodes': result['num_nodes'], 'surrogate': True}
    return record


def surrogate_baseline_population(generator, surrogate, size=12, seed=0, verbose=True):
    """Batch equivalent of `loop.baseline_population`: one native call for the
    whole population instead of `size` separate ones.

    Mesh generation stays per-candidate (a few seconds each on GPU, cheap); only
    the surrogate's subprocess call -- which pays for loading the model and
    building its coarsening hierarchy regardless of batch size, ~1-2 minutes
    fixed cost -- is amortised across the population. Serial calls here would
    turn a `size`-candidate baseline into `size` full model reloads.
    """
    rng = np.random.default_rng(seed)
    lo, hi = generator.bounds()
    designs = [rng.uniform(lo, hi) for _ in range(size)]
    meshes, gen_infos = [], []
    for x in designs:
        mesh, info = generator.generate(x)
        meshes.append(mesh)
        gen_infos.append(info)

    valid_idx = [i for i, m in enumerate(meshes) if m is not None]
    results = [None] * size
    if valid_idx:
        batch_results = surrogate.analyze_batch([meshes[i] for i in valid_idx])
        for local_i, global_i in enumerate(valid_idx):
            results[global_i] = batch_results[local_i]

    records = []
    for i, (x, mesh, info, result) in enumerate(zip(designs, meshes, gen_infos, results)):
        record = _result_record(x, mesh, info, result)
        record['index'] = i
        if verbose:
            status = 'ok' if record['ok'] else record['error']
            if record['ok']:
                f = record['fea']
                status = (f"mass={f['mass']:.3f}kg  peak_vm={f['peak_von_mises'] / 1e6:.1f}MPa  "
                          f"disp={f['max_displacement'] * 1e3:.3f}mm  nodes={f['num_nodes']}")
            print(f'  baseline {i:02d}/{size}: {status}', flush=True)
        record.pop('mesh_object', None)
        records.append(record)
    return records


def surrogate_search(generator, surrogate, objective, x0, sigma0=1.0, budget=24,
                     popsize=6, seed=0, verbose=True):
    """CMA-ES over the surrogate, batching every generation into one native call.

    Same return shape as `loop.run` (best_x, best_score, log) plus the full
    per-candidate history, so `run_optimize`'s reporting code is unchanged
    whichever backend produced it.
    """
    import cma

    lo, hi = generator.bounds()
    es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=float), sigma0, {
        'bounds': [lo.tolist(), hi.tolist()], 'popsize': popsize,
        'maxfevals': budget, 'seed': seed + 1, 'verbose': -9,
    })

    history, log, generation = [], [], 0
    while not es.stop() and len(history) < budget:
        solutions = es.ask()
        meshes, gen_infos = [], []
        for x in solutions:
            mesh, info = generator.generate(x)
            meshes.append(mesh)
            gen_infos.append(info)

        valid_idx = [i for i, m in enumerate(meshes) if m is not None]
        results = [None] * len(solutions)
        if valid_idx:
            batch_results = surrogate.analyze_batch([meshes[i] for i in valid_idx])
            for local_i, global_i in enumerate(valid_idx):
                results[global_i] = batch_results[local_i]

        scores = []
        for x, mesh, info, result in zip(solutions, meshes, gen_infos, results):
            record = _result_record(x, mesh, info, result)
            record.pop('mesh_object', None)
            if record['ok']:
                score, penalty = objective(record['fea'])
            else:
                score, penalty = objective.failure_score, {'feasible': False,
                                                            'reason': record['error']}
            record['score'], record['penalty'] = score, penalty
            record['index'] = len(history)
            history.append(record)
            scores.append(score)
        es.tell(solutions, scores)
        generation += 1

        best = min(history, key=lambda r: r['score'])
        feasible = [r for r in history if r['ok'] and r['penalty'].get('feasible')]
        entry = {
            'generation': generation, 'evaluations': len(history),
            'generation_best': float(min(scores)), 'generation_median': float(np.median(scores)),
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

    best = min(history, key=lambda r: r['score'])
    return np.asarray(best['x']), best['score'], log, history
