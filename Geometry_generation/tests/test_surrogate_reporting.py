"""Truthfulness regressions for the HI-MGN optimization reporting path."""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_loop import surrogate as surrogate_module  # noqa: E402
from inference_profiles.optimize import _brief, _write_report  # noqa: E402


def test_surrogate_batch_preserves_each_candidates_actual_node_count(monkeypatch, tmp_path):
    node_counts = {"first": 7, "second": 11}

    def fake_records(_mesh, *, load_cases, target_nodes, name):
        del load_cases, target_nodes
        return [{
            "case": "ver",
            "nodal": np.zeros((5, 1, node_counts[name]), dtype=np.float32),
        }]

    monkeypatch.setattr(surrogate_module, "mesh_to_records", fake_records)
    monkeypatch.setattr(surrogate_module, "write_inference_contract", lambda *_: None)
    model = surrogate_module.HIMGNSurrogate(
        "infer.txt", "model.pth", load_cases=("ver",), workdir=tmp_path,
    )
    model._run_native = lambda *_: {
        1: np.zeros((5, 7), dtype=np.float32),
        2: np.zeros((5, 11), dtype=np.float32),
    }

    results = model.analyze_batch(
        [SimpleNamespace(volume=1.0), SimpleNamespace(volume=2.0)],
        names=["first", "second"],
    )

    assert [result["num_nodes"] for result in results] == [7, 11]


def _field_result():
    return {
        "mass": 1.0,
        "peak_von_mises": 100e6,
        "max_von_mises": 110e6,
        "max_displacement": 0.001,
        "max_compliance": 0.0,
        "num_nodes": 5003,
        "num_tets": 0,
        "cases": {
            "vertical": {
                "peak_von_mises": 100e6,
                "max_displacement": 0.001,
                "compliance": None,
            },
        },
    }


def test_brief_uses_backend_specific_mesh_cardinality():
    surrogate = _brief(_field_result(), "surrogate")
    fea = _brief(_field_result(), "fea")
    assert surrogate["num_nodes"] == 5003
    assert "num_tets" not in surrogate
    assert fea["num_tets"] == 0
    assert "num_nodes" not in fea


def test_surrogate_report_never_claims_fea_verification_or_mesh_sensitivity(tmp_path):
    baseline = _brief(_field_result(), "surrogate")
    optimized = {**baseline, "mass_kg": 0.9, "num_nodes": 4997}
    summary = {
        "analysis_backend": "surrogate",
        "wall_time_s": 60.0,
        "total_evaluations": 3,
        "search_success_rate": 1.0,
        "failures": {},
        "limits": {
            "population": 2,
            "mass_ref": 1.0,
            "stress_allow": 120e6,
            "disp_allow": 0.002,
        },
        "verified": {
            "baseline": baseline,
            "optimized": optimized,
            "mass_change_pct": -10.0,
            "stress_change_pct": 0.0,
            "disp_change_pct": 0.0,
        },
        # A legacy summary may still contain this; the report must not present
        # it as convergence evidence for a surface-graph surrogate.
        "mesh_sensitivity": {
            "search_tets": 0,
            "verify_tets": 0,
            "mass_change_pct": 0.0,
            "peak_stress_change_pct": 0.0,
            "disp_change_pct": 0.0,
        },
    }

    _write_report(tmp_path, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Surrogate re-evaluation (not FEA verified)" in report
    assert "surface graph nodes" in report
    assert "tetrahedra" not in report
    assert "Discretization sensitivity" not in report

