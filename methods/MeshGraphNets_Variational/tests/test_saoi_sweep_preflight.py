"""The sweep must reject an eval file from a different physical condition."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / 'configs/MeshGraphNets_Variational/SAOI_run/check_eval_inputs.py'
spec = spec_from_file_location('saoi_check_eval_inputs', CHECKER)
checker = module_from_spec(spec)
spec.loader.exec_module(checker)


def _write(path, thicknesses):
    nodes = 5
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    xyz = np.arange(nodes * 3, dtype=np.float32).reshape(3, nodes)
    with h5py.File(path, 'w') as handle:
        for sid, thickness in enumerate(thicknesses):
            nd = np.zeros((8, 1, nodes), dtype=np.float32)
            nd[:3, 0] = xyz
            nd[6, 0] = thickness       # input_var=3 -> first condition row
            nd[7, 0] = 4.0
            group = handle.create_group(f'data/{sid}')
            group.create_dataset('nodal_data', data=nd)
            group.create_dataset('mesh_edge', data=edges)


def test_same_condition_check_accepts_matching_realizations(tmp_path):
    infer = tmp_path / 'infer.h5'
    compare = tmp_path / 'compare.h5'
    _write(infer, [0.8])
    _write(compare, [0.8, 0.8, 0.8])
    cfg = {'input_var': '3', 'cond_var': '2', 'use_node_types': 'false'}
    assert checker.same_condition_problems(infer, compare, cfg) == []


def test_same_condition_check_rejects_thickness_mismatch(tmp_path):
    infer = tmp_path / 'infer.h5'
    compare = tmp_path / 'compare.h5'
    _write(infer, [0.8])
    _write(compare, [0.8, 1.1])
    cfg = {'input_var': '3', 'cond_var': '2', 'use_node_types': 'false'}
    problems = checker.same_condition_problems(infer, compare, cfg)
    assert any('conditioning rows' in problem for problem in problems)
