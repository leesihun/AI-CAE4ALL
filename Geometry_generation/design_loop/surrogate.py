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

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SUITE = os.path.dirname(_REPO)
if os.path.join(_SUITE, 'dataset') not in sys.path:
    sys.path.insert(0, os.path.join(_SUITE, 'dataset'))

from deepjeb_bridge import (                                        # noqa: E402
    DEEPJEB_CENTRE, DEEPJEB_MAX_SIDE, LOAD_CASES, SDF_TARGET_EXTENT,
    mesh_to_records, write_inference_contract,
)

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
        self.workdir = workdir or tempfile.mkdtemp(prefix='himgn_screen_')
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
            stress, disp = pred[3, :], pred[4, :]
            entry['cases'][rec['case']] = {
                'peak_von_mises': float(np.percentile(np.abs(stress),
                                                      self.stress_percentile)),
                'max_von_mises': float(np.abs(stress).max()),
                'max_displacement': float(np.abs(disp).max()),
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
                num_nodes=int(records[0]['nodal'].shape[2]),
                num_tets=0,                    # no volume mesh on this path
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
            os.path.join(_REPO, '..', 'cHI-MGNflow', 'outputs', 'rollout')
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
