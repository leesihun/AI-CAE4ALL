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
    DEEPJEB_CENTRE, DEEPJEB_MAX_SIDE, LABEL_SURFACE_FACES, LOAD_CASES, SDF_TARGET_EXTENT,
    VER_COND_VAR, VER_FEATURE_NAMES, VER_INPUT_VAR, VER_OUTPUT_VAR,
    mesh_to_records, mesh_to_ver_record, serving_surface, write_inference_contract,
)
from design_loop.loop import rank_failures_last, select_best
from design_loop.problem import find_interfaces, vertical_displacement

# <suite>/methods/SDFFlow/design_loop/surrogate.py -> <suite>
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUITE = os.path.dirname(os.path.dirname(_REPO))

STRESS_PERCENTILE = 99.5
NO_PREDICTION = 'SurrogateError: surrogate could not bridge or predict this shape'


class SurrogateError(RuntimeError):
    pass


# Two DeepJEB surrogate layouts exist. 'ex10' (build_deepjeb_mgn): stress,
# |u|, u_z for four load cases told apart by one-hot condition rows, BCs from
# `problem.find_interfaces`. 'ver' (ex13, OpenRadioss vertical load): u_x,
# u_y, u_z, von Mises for the vertical case only, BCs as a node-type row from
# the bore/lug rule in `design_loop.interfaces`.
LAYOUTS = ('ex10', 'ver')


def layout_from_config(config_path):
    """The surrogate layout its inference config declares: 'ver' for the ex13
    contract (output_var 4, cond_var 0), else 'ex10'."""
    values = {}
    with open(config_path, encoding='utf-8') as fh:
        for line in fh:
            parts = line.split('%')[0].split('#')[0].split()
            if len(parts) >= 2:
                values[parts[0].lower()] = parts[1]
    try:
        out_var, cond_var = int(values.get('output_var', 0)), int(values.get('cond_var', 0))
    except ValueError:
        return 'ex10'
    return 'ver' if (out_var, cond_var) == (VER_OUTPUT_VAR, VER_COND_VAR) else 'ex10'


def gpu_id_from_config(config):
    """The GPU the calling SDFFlow run was given, for the nested surrogate call.

    First entry of `gpu_ids` (the one `resolve_device` uses); None for a CPU
    run or no setting, which leaves the surrogate config's own value in force.
    """
    value = config.get('gpu_ids')
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None or str(value).strip().lower() == 'cpu':
        return None
    return int(value)


class HIMGNSurrogate:
    """Batch structural prediction for generated brackets."""

    def __init__(self, config_path, checkpoint, python=None, load_cases=('ver', 'dia'),
                 target_nodes=5000, density=4430.0, workdir=None,
                 stress_percentile=STRESS_PERCENTILE, length_scale=None, gpu_id=None,
                 surface_faces=None, layout=None):
        self.config_path = os.path.abspath(config_path)
        if layout is None:
            layout = (layout_from_config(self.config_path)
                      if os.path.isfile(self.config_path) else 'ex10')
        if layout not in LAYOUTS:
            raise SurrogateError(f'unknown surrogate layout {layout!r}; available {LAYOUTS}')
        self.layout = layout
        # Face budget of the surface a candidate is bridged from; the labels'
        # own (see deepjeb_bridge.serving_surface). 0 bridges the raw MC mesh.
        # The 'ver' labels were clustered from a fine tet4 boundary, which the
        # raw mc_resolution-128 surface matches (graph edge p95 4.88 vs 4.89 mm)
        # better than a 12000-face re-surfacing does (5.95 mm).
        if surface_faces is None:
            surface_faces = LABEL_SURFACE_FACES if layout == 'ex10' else 0
        self.surface_faces = int(surface_faces or 0)
        # Metres per normalized SDF unit, for mass only. The FEA backend's
        # Bracket uses opt_length_scale (0.19/1.8 by default); without it the
        # mass fell back to the DeepJEB frame (184.18 mm across), 8.9% below the
        # FEA mass of the very same shape. None keeps that frame for callers
        # that compare against DeepJEB's own labels. The predicted fields are
        # not rescaled -- they are whatever scale the checkpoint was labelled at.
        self.length_scale = float(length_scale) if length_scale else None
        # The nested launcher call otherwise takes gpu_ids from the surrogate
        # config file, a second hidden device setting that need not exist on
        # this host (the checked-in ex10 config says 5; aarl has 0-3).
        self.gpu_id = None if gpu_id is None else int(gpu_id)
        self.checkpoint = os.path.abspath(checkpoint)
        self.python = python or sys.executable
        self.load_cases = tuple(load_cases)
        unknown = set(self.load_cases) - set(LOAD_CASES)
        if unknown:
            raise SurrogateError(f'unknown load case(s) {sorted(unknown)}; '
                                 f'available {LOAD_CASES}')
        if layout == 'ver' and self.load_cases != ('ver',):
            raise SurrogateError('this surrogate was trained on the vertical load case only; '
                                 f'set opt_load_cases vertical (got {list(self.load_cases)})')
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
        # Why each None of the last analyze_batch call is None, by batch index.
        self.last_errors = {}
        # 'ver' layout: every registration scale found, and how many shapes
        # could not be registered (the summary's frame record).
        self.frame_scales = []
        self.frame_failures = 0

    # ------------------------------------------------------------------ #

    def mass_of(self, mesh, frame_scale=1.0):
        """Exact mass in kg from the generated geometry.

        At `length_scale` when one was given (the FEA backend's frame), else in
        the DeepJEB frame. `frame_scale` is the shape's own registration scale
        (`deepjeb_bridge.registered_millimetres`; 'ver' layout): the frame its
        predicted fields are in, so mass and fields describe one part. Without
        it, ex1 samples' masses were off by up to ~10% (scale^3, 0.968-1.005).
        """
        if self.length_scale is not None:
            unit = self.length_scale * frame_scale            # metres per unit
            return abs(float(mesh.volume)) * unit ** 3 * self.density
        scale = DEEPJEB_MAX_SIDE / SDF_TARGET_EXTENT * frame_scale   # normalized -> mm
        volume_mm3 = abs(float(mesh.volume)) * scale ** 3
        return volume_mm3 * 1e-9 * self.density            # mm^3 -> m^3 -> kg

    def analyze_batch(self, meshes, names=None):
        """Predict fields for a list of generated meshes.

        Returns one result dict per mesh, in the order given; an entry is None
        where that candidate could not be bridged or predicted, and
        `self.last_errors[index]` then says why.

        A candidate is refused before prediction on the rule the FEA backend
        refuses it on (`problem.find_interfaces`, applied to the same surface
        gmsh would be handed): a shape with no loaded lug or only one mounting
        pad is not a bracket, and a surrogate will still predict a field for it.
        """
        names = names or [f'cand{i:03d}' for i in range(len(meshes))]
        self.last_errors = {}
        records, owner = [], []
        for index, (mesh, name) in enumerate(zip(meshes, names)):
            try:
                surface = serving_surface(mesh, self.surface_faces)
                if self.layout == 'ver':
                    # The bore/lug gates inside node_types are this layout's refusal rule.
                    recs = [mesh_to_ver_record(surface, target_nodes=self.target_nodes,
                                               name=name)]
                else:
                    find_interfaces(np.asarray(surface.vertices), np.asarray(surface.faces))
                    recs = mesh_to_records(surface, load_cases=self.load_cases,
                                           target_nodes=self.target_nodes, name=name)
            except Exception as exc:
                self.last_errors[index] = f'{type(exc).__name__}: {exc}'
                self.frame_failures += 'frame registration failed' in str(exc)
                continue
            if recs[0].get('frame'):
                self.frame_scales.append(recs[0]['frame']['scale'])
            records += recs
            owner += [index] * len(recs)
        if not records:
            return [None] * len(meshes)

        call = self.calls
        self.calls += 1
        infer_path = os.path.join(self.workdir, f'batch{call:04d}.h5')
        if self.layout == 'ver':
            write_inference_contract(infer_path, records, feature_names=VER_FEATURE_NAMES,
                                     input_var=VER_INPUT_VAR, output_var=VER_OUTPUT_VAR,
                                     cond_var=VER_COND_VAR)
        else:
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
            if results[index] is None:
                frame = rec.get('frame')
                results[index] = {
                    'cases': {},
                    'mass': self.mass_of(meshes[index],
                                         frame['scale'] if frame else 1.0),
                    'frame': frame,
                }
            entry = results[index]
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
            if self.layout == 'ver':
                # rows u_x, u_y, u_z (mm), von Mises (MPa); |u| is formed here.
                stress_mpa = pred[6, :]
                disp_mm = np.linalg.norm(pred[3:6, :], axis=0)
            else:
                stress_mpa, disp_mm = pred[3, :], pred[4, :]
            case = {
                'peak_von_mises': float(np.percentile(np.abs(stress_mpa),
                                                      self.stress_percentile) * 1e6),
                'max_von_mises': float(np.abs(stress_mpa).max() * 1e6),
                'max_displacement': float(np.abs(disp_mm).max() * 1e-3),
            }
            # Row 5 is u_z in both layouts (FEATURE_NAMES in build_deepjeb_mgn,
            # VER_FEATURE_NAMES); a checkpoint trained without it simply has no
            # vertical-deflection prediction.
            if pred.shape[0] > 5:
                case['max_vertical_displacement'] = float(np.abs(pred[5, :]).max() * 1e-3)
            entry['cases'][rec['case']] = case
            results[index] = entry

        for index, entry in enumerate(results):
            if entry is None or not entry['cases']:
                results[index] = None
                self.last_errors.setdefault(index, 'SurrogateError: no prediction returned '
                                                   'for this candidate')
                continue
            cases = entry['cases']
            entry.update(
                peak_von_mises=max(c['peak_von_mises'] for c in cases.values()),
                max_von_mises=max(c['max_von_mises'] for c in cases.values()),
                max_displacement=max(c['max_displacement'] for c in cases.values()),
                vertical_displacement=vertical_displacement(cases),
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
        if self.gpu_id is not None:
            overrides['gpu_ids'] = str(self.gpu_id)
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
                'load_cases': list(self.load_cases), 'workdir': self.workdir,
                'layout': self.layout,
                'mass_length_scale': self.length_scale, 'gpu_id': self.gpu_id,
                'surface_faces': self.surface_faces, 'frame': self.frame_stats()}

    def frame_stats(self):
        """Per-shape registration over every candidate bridged ('ver' layout only)."""
        if self.layout != 'ver':
            return None
        scales = np.asarray(self.frame_scales, dtype=float)
        return {'registered': int(scales.size), 'failed': int(self.frame_failures),
                'scale_min': float(scales.min()) if scales.size else None,
                'scale_max': float(scales.max()) if scales.size else None,
                'scale_median': float(np.median(scales)) if scales.size else None}


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
                # The bridge's own reason (e.g. the interface rule), recorded
                # as the FEA path records it, not wrapped in a generic one.
                reason = (getattr(self.surrogate, 'last_errors', None) or {}).get(0)
                record['ok'] = False
                record['error'] = reason or NO_PREDICTION
                kind = record['error'].split(':', 1)[0]
                self.failures[kind] = self.failures.get(kind, 0) + 1
                return record

            record['fea'] = {
                'mass': result['mass'], 'volume': result['volume'],
                'num_nodes': result['num_nodes'],
                'cases': result['cases'], 'worst_case': max(result['cases'],
                    key=lambda name: result['cases'][name]['peak_von_mises']),
                'peak_von_mises': result['peak_von_mises'],
                'max_von_mises': result['max_von_mises'],
                'max_displacement': result['max_displacement'],
                'vertical_displacement': result.get('vertical_displacement'),
                'frame': result.get('frame'),
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


def _result_record(x, mesh, gen_info, result, reason=None):
    """Build the FEA-shaped record dict for one candidate's surrogate result."""
    record = {'x': np.asarray(x, dtype=float).tolist(), 'generation': gen_info}
    if mesh is None:
        record['ok'] = False
        record['error'] = 'RuntimeError: no zero crossing in the decoded SDF'
        return record
    if result is None:
        record['ok'] = False
        record['error'] = reason or NO_PREDICTION
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
        'vertical_displacement': result.get('vertical_displacement'),
        'frame': result.get('frame'),
    }
    record['mesh'] = {'num_nodes': result['num_nodes'], 'surrogate': True}
    return record


def _analyze_valid(surrogate, meshes):
    """One batch call over the meshes that exist: (results, {index: reason})."""
    valid_idx = [i for i, m in enumerate(meshes) if m is not None]
    results, reasons = [None] * len(meshes), {}
    if valid_idx:
        batch_results = surrogate.analyze_batch([meshes[i] for i in valid_idx])
        errors = getattr(surrogate, 'last_errors', None) or {}
        for local_i, global_i in enumerate(valid_idx):
            results[global_i] = batch_results[local_i]
            if local_i in errors:
                reasons[global_i] = errors[local_i]
    return results, reasons


def surrogate_baseline_population(generator, surrogate, size=12, seed=0, verbose=True,
                                  batch_size=None, on_chunk=None):
    """Batch equivalent of `loop.baseline_population`: one native call per chunk
    of `batch_size` designs instead of `size` separate ones.

    Mesh generation stays per-candidate (a few seconds each on GPU, cheap); only
    the surrogate's subprocess call -- which pays for loading the model and
    building its coarsening hierarchy regardless of batch size, ~1-2 minutes
    fixed cost -- is amortised across the chunk. Serial calls here would turn a
    `size`-candidate baseline into `size` full model reloads.

    Streaming screening: with `batch_size` below `size` the population is
    generated and predicted chunk by chunk, so a thousand-design screen holds
    one chunk of meshes at a time and reports results as it goes instead of
    only at the end. `on_chunk(records_so_far, size)` runs after every chunk.
    `batch_size` None or 0 keeps the whole population in one call. The designs
    are drawn up front, so the population does not depend on the chunking.
    """
    rng = np.random.default_rng(seed)
    lo, hi = generator.bounds()
    designs = [rng.uniform(lo, hi) for _ in range(size)]
    chunk = int(batch_size) if batch_size and int(batch_size) > 0 else max(size, 1)
    width = max(2, len(str(max(size - 1, 0))))

    records = []
    for start in range(0, size, chunk):
        chunk_designs = designs[start:start + chunk]
        meshes, gen_infos, gen_errors = [], [], {}
        for local_i, x in enumerate(chunk_designs):
            # One design that crashes the decoder must not end a long screen.
            try:
                mesh, info = generator.generate(x)
            except Exception as exc:
                mesh, info = None, {}
                gen_errors[local_i] = f'{type(exc).__name__}: {exc}'
            meshes.append(mesh)
            gen_infos.append(info)

        try:
            results, reasons = _analyze_valid(surrogate, meshes)
        except Exception as exc:
            # Before any chunk has predicted, a failed call means the surrogate
            # itself is broken: stop. Later, it costs this chunk only.
            if not any(r['ok'] for r in records):
                raise
            error = f'{type(exc).__name__}: {exc}'
            print(f'  chunk {start}-{start + len(meshes) - 1} failed: '
                  f'{error.splitlines()[0]}', flush=True)
            results, reasons = [None] * len(meshes), dict.fromkeys(range(len(meshes)), error)

        for local_i, (x, mesh, info, result) in enumerate(
                zip(chunk_designs, meshes, gen_infos, results)):
            i = start + local_i
            if local_i in gen_errors:
                record = {'x': np.asarray(x, dtype=float).tolist(), 'generation': info,
                          'ok': False, 'error': gen_errors[local_i]}
            else:
                record = _result_record(x, mesh, info, result, reasons.get(local_i))
            record['index'] = i
            if verbose:
                status = 'ok' if record['ok'] else record['error']
                if record['ok']:
                    f = record['fea']
                    uz = f.get('vertical_displacement')
                    status = (f"mass={f['mass']:.3f}kg  peak_vm={f['peak_von_mises'] / 1e6:.1f}MPa  "
                              f"disp={f['max_displacement'] * 1e3:.3f}mm  "
                              + (f"u_z={uz * 1e3:.4f}mm  " if uz is not None else '')
                              + f"nodes={f['num_nodes']}")
                print(f'  baseline {i:0{width}d}/{size}: {status}', flush=True)
            record.pop('mesh_object', None)
            records.append(record)
        if on_chunk is not None:
            on_chunk(records, size)
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

        results, reasons = _analyze_valid(surrogate, meshes)

        generation_records = []
        for i, (x, mesh, info, result) in enumerate(zip(solutions, meshes, gen_infos, results)):
            record = _result_record(x, mesh, info, result, reasons.get(i))
            record.pop('mesh_object', None)
            if record['ok']:
                score, penalty = objective(record['fea'])
            else:
                score, penalty = objective.failure_score, {'feasible': False,
                                                            'reason': record['error']}
            record['score'], record['penalty'] = score, penalty
            record['index'] = len(history)
            history.append(record)
            generation_records.append(record)
        scores = rank_failures_last(generation_records, objective.failure_score)
        es.tell(solutions, scores)
        generation += 1

        best = select_best(history)
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

    best = select_best(history)
    return np.asarray(best['x']), best['score'], log, history
