"""Truthfulness regressions for the HI-MGN optimization reporting path."""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_loop import surrogate as surrogate_module  # noqa: E402
from design_loop.surrogate import SurrogateEvaluator, _result_record  # noqa: E402
from design_loop.problem import MassObjective  # noqa: E402
from inference_profiles import optimize as optimize_module  # noqa: E402
from inference_profiles.optimize import (  # noqa: E402
    FEA_VERIFIED_NAME,
    _brief,
    _compact_fea_verification,
    _fea_verification_lines,
    _fea_verify,
    _flag,
    _summarize,
    _verification_settings,
    _write_report,
)


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
    monkeypatch.setattr(surrogate_module, "serving_surface", lambda mesh, _faces: mesh)
    monkeypatch.setattr(surrogate_module, "find_interfaces", lambda *_: None)
    model = surrogate_module.HIMGNSurrogate(
        "infer.txt", "model.pth", load_cases=("ver",), workdir=tmp_path,
    )
    model._run_native = lambda *_: {
        1: np.zeros((5, 7), dtype=np.float32),
        2: np.zeros((5, 11), dtype=np.float32),
    }

    results = model.analyze_batch(
        [SimpleNamespace(volume=1.0, vertices=np.zeros((3, 3)), faces=np.zeros((1, 3))),
         SimpleNamespace(volume=2.0, vertices=np.zeros((3, 3)), faces=np.zeros((1, 3)))],
        names=["first", "second"],
    )

    assert [result["num_nodes"] for result in results] == [7, 11]
    assert all("num_tets" not in result for result in results)
    assert all("compliance" not in case
               for result in results for case in result["cases"].values())
    assert model.last_errors == {}


def test_surrogate_refuses_a_shape_the_fea_path_would_refuse(monkeypatch, tmp_path):
    """A shape with no loaded lug gets no prediction, and says why."""
    import trimesh

    predicted = []
    monkeypatch.setattr(surrogate_module, "write_inference_contract", lambda *_: None)
    model = surrogate_module.HIMGNSurrogate(
        "infer.txt", "model.pth", load_cases=("ver",), workdir=tmp_path, surface_faces=0,
    )
    model._run_native = lambda *_: predicted.append(True) or {}

    # A flat slab: both mounting-pad ends exist, but its top is 0.2 above the
    # mount, far below any real bracket's lug crown (min 0.62 over 418 labels).
    # Subdivided so the pad and lug bands both hold surface vertices; the crown
    # rule, not an empty band, is what must refuse it.
    slab = trimesh.creation.box(extents=(1.0, 1.8, 0.2)).subdivide().subdivide().subdivide()
    results = model.analyze_batch([slab], names=["slab"])

    assert results == [None]
    assert not predicted
    assert "lug crown only" in model.last_errors[0]

    generator = SimpleNamespace(generate=lambda *_args, **_kwargs: (slab, {}))
    record = SurrogateEvaluator(generator, model).analyze([0.0])
    assert record["ok"] is False
    assert record["error"].startswith("ValueError: lug crown only")


def test_batched_records_carry_the_bridge_reason():
    surrogate = SimpleNamespace(
        analyze_batch=lambda meshes: [None for _ in meshes],
        last_errors={0: "ValueError: only one mounting-pad end found"},
    )
    results, reasons = surrogate_module._analyze_valid(surrogate, [None, SimpleNamespace()])
    assert results == [None, None]
    assert reasons == {1: "ValueError: only one mounting-pad end found"}
    record = _result_record([0.0], SimpleNamespace(), {}, None, reasons.get(1))
    assert record["error"] == "ValueError: only one mounting-pad end found"
    assert _result_record([0.0], SimpleNamespace(), {}, None)["error"] == surrogate_module.NO_PREDICTION


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


def _surrogate_result():
    return {
        "mass": 1.0,
        "volume": 0.25,
        "num_nodes": 5003,
        "peak_von_mises": 100e6,
        "max_von_mises": 110e6,
        "max_displacement": 0.001,
        "cases": {
            "vertical": {
                "peak_von_mises": 100e6,
                "max_von_mises": 110e6,
                "max_displacement": 0.001,
            },
        },
    }


def _assert_no_fea_placeholders(record):
    assert "num_tets" not in record["fea"]
    assert "max_compliance" not in record["fea"]
    assert "num_tets" not in record["mesh"]
    assert all("compliance" not in case for case in record["fea"]["cases"].values())


def test_surrogate_internal_records_do_not_publish_fea_placeholders():
    result = _surrogate_result()
    batched = _result_record([0.0], SimpleNamespace(), {}, result)
    _assert_no_fea_placeholders(batched)

    generator = SimpleNamespace(generate=lambda *_args, **_kwargs: (SimpleNamespace(), {}))
    surrogate = SimpleNamespace(analyze_batch=lambda _meshes: [result])
    verified = SurrogateEvaluator(generator, surrogate).analyze([0.0])
    _assert_no_fea_placeholders(verified)


def test_brief_uses_backend_specific_mesh_cardinality():
    surrogate = _brief(_field_result(), "surrogate")
    fea = _brief(_field_result(), "fea")
    assert surrogate["num_nodes"] == 5003
    assert "num_tets" not in surrogate
    assert "max_compliance_J" not in surrogate
    assert all("compliance_J" not in case for case in surrogate["cases"].values())
    assert fea["num_tets"] == 0
    assert "num_nodes" not in fea
    assert fea["max_compliance_J"] == 0.0
    assert fea["cases"]["vertical"]["compliance_J"] is None


def test_verification_settings_are_backend_specific():
    surrogate = _verification_settings("surrogate", 160, 30000, 0.035, 5003)
    fea = _verification_settings("fea", 160, 30000, 0.035)
    assert surrogate == {"mc_resolution": 160, "target_nodes": 5003}
    assert fea == {
        "mc_resolution": 160,
        "target_faces": 30000,
        "mesh_size_max": 0.035,
    }


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


def _summary_for(verified, reference_verified, typical_verified, **kwargs):
    limits = {"population": 3, "mass_ref": 1.0, "stress_allow": 120e6,
              "disp_allow": 0.002, "vertical_disp_allow": None}
    reference = {"ok": True, "fea": _field_result(), "score": 1.0,
                 "penalty": {"feasible": True}, "x": [0.0]}
    evaluator = SimpleNamespace(failures={}, objective=MassObjective(1.0, 120e6, 0.002))
    material = SimpleNamespace(name="Ti", E=1.0, nu=0.3, rho=1.0, yield_stress=1.0)
    return _summarize(limits, reference, reference_verified, verified, [0.0], 0.9,
                      [], [], [reference], evaluator, material, ("vertical",),
                      SimpleNamespace(length_scale=0.1, stress_percentile=99.5), 1.0, reference,
                      typical_verified, "surrogate", **kwargs)


def test_a_failed_baseline_reanalysis_keeps_the_optimized_numbers(tmp_path):
    ok = {"ok": True, "fea": {**_field_result(), "mass": 0.8}}
    failed = {"ok": False, "error": "SurrogateError: could not bridge"}
    summary = _summary_for(ok, failed, None, verify_evaluations=3)
    v = summary["verified"]
    assert v["optimized"]["mass_kg"] == 0.8
    assert "baseline" not in v and "mass_change_pct" not in v
    assert v["failed"] == {"baseline": failed["error"]}
    assert v["optimized_penalty"]["feasible"] is True
    assert summary["total_evaluations"] == 1 + 0 + 3

    summary.update(wall_time_s=60.0, search_success_rate=1.0, best_search_score=0.9)
    _write_report(tmp_path, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "no baseline comparison" in report
    assert "The baseline design failed verification" in report


def test_report_says_so_when_the_search_did_not_beat_its_start(tmp_path):
    same = {"ok": True, "fea": _field_result()}
    summary = _summary_for(same, same, same, verify_evaluations=2)
    summary.update(wall_time_s=60.0, search_success_rate=1.0, best_search_score=1.2,
                   search_improved_on_baseline=False)
    assert summary["verified"]["mass_change_pct"] == 0.0
    _write_report(tmp_path, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "did not improve on its starting design" in report


def test_verification_limits_follow_the_baseline_resolution_ratio(tmp_path):
    """A refined re-analysis is judged on refined-resolution allowables.

    The search-resolution stress allowable is 120 MPa. The baseline designs
    read 1.3x their search peak at the verification resolution, so a delivered
    design at 150 MPa there (115 MPa at the search resolution) is within the
    transported 156 MPa -- and judged against the raw 120 MPa it would have
    been reported infeasible from resolution drift alone.
    """
    from inference_profiles.optimize import _verification_objective

    def rec(x, peak, disp=0.001, mass=1.0, uz=None):
        fea = {**_field_result(), "peak_von_mises": peak, "max_displacement": disp,
               "mass": mass, "vertical_displacement": uz}
        return {"ok": True, "x": [x], "index": int(x), "fea": fea}

    objective = MassObjective(1.0, 120e6, 0.002, vertical_disp_allow=0.0002)
    reference, reference_v = rec(0, 100e6, uz=1e-4), rec(0, 130e6, disp=0.0012, uz=1.2e-4)
    typical, typical_v = rec(1, 90e6, uz=1e-4), rec(1, 117e6, disp=0.0012, uz=1.2e-4)
    failed = {"ok": False, "x": [2.0], "error": "ValueError: x"}
    transported, info = _verification_objective(
        objective, [(reference, reference_v), (typical, typical_v), (reference, reference_v),
                    (failed, typical_v)])

    assert info["reference_designs"] == 2 and info["baseline_indices"] == [0, 1]
    assert np.isclose(info["stress_factor"], 1.3)
    assert np.isclose(transported.stress_allow, 156e6)
    assert np.isclose(transported.disp_allow, 0.0024)
    assert transported.vertical_disp_allow == objective.vertical_disp_allow

    delivered_v = rec(3, 150e6, uz=1.5e-4)["fea"]
    assert objective(delivered_v)[1]["feasible"] is False
    assert transported(delivered_v)[1]["feasible"] is True
    # The vertical limit is absolute: a refined 0.25 mm drop still violates 0.2 mm.
    assert transported(rec(3, 150e6, uz=2.5e-4)["fea"])[1]["feasible"] is False

    # No usable pair: nothing is rescaled.
    same, info = _verification_objective(objective, [(failed, reference_v)])
    assert info["reference_designs"] == 0 and same.stress_allow == objective.stress_allow

    # _summarize scores the delivered design with the transported objective.
    verified = {"ok": True, "fea": delivered_v}
    summary = _summary_for(verified, reference_v, typical_v, verify_evaluations=3,
                           verify_objective=transported, verification_limits=info)
    assert summary["verification_limits"] is info
    assert summary["verified"]["optimized_penalty"]["feasible"] is True
    assert summary["verified"]["optimized_search_penalty"] == {"feasible": True}
    summary.update(wall_time_s=60.0, search_success_rate=1.0, best_search_score=0.9)
    _write_report(tmp_path, summary)
    assert "Limits at the verification resolution" in (tmp_path / "report.md").read_text(
        encoding="utf-8")


def _fea_report(optimized_uz=0.26, optimized_met=False):
    """A verify_with_fea.py json: the surrogate said feasible, the solver may not."""
    def design(mass, peak, uz, met, surface_peak, surface_uz):
        return {
            "mass_kg": mass, "peak_von_mises_MPa": peak, "max_displacement_mm": 0.5,
            "vertical_displacement_mm": uz, "tets": 40000, "nodes": 9000,
            "label_surface": {"peak_von_mises_MPa": surface_peak,
                              "vertical_displacement_mm": surface_uz, "nodes": 5000},
            "surrogate": {"mass_kg": mass},
            "limits": [
                {"limit": "vertical max |u_z|", "unit": "mm", "value": uz, "allow": 0.2,
                 "met": met, "statistic": "solver"},
                {"limit": "peak von Mises (calibrated)", "unit": "MPa", "value": surface_peak,
                 "allow": 120.0, "met": True, "statistic": "label_surface"},
            ],
        }
    return {
        "solver": "tet4",
        "resolution": {"source": "surrogate.label_resolution", "target_faces": 12000,
                       "mesh_size_max": 0.05, "target_nodes": 5000},
        "designs": {
            "optimized": design(0.81, 140.0, optimized_uz, optimized_met, 95.0, 0.25),
            "baseline": design(0.95, 120.0, 0.18, True, 90.0, 0.17),
            "typical": {"error": "RuntimeError: gmsh failed"},
        },
    }


def test_flag_reads_the_launchers_spellings():
    assert _flag({"k": "True"}, "k") and _flag({"k": "yes"}, "k") and _flag({"k": True}, "k")
    assert not _flag({"k": "false"}, "k") and not _flag({"k": "0"}, "k")
    assert not _flag({"k": " off "}, "k") and not _flag({"k": ""}, "k") and not _flag({}, "k")


def test_compact_fea_verification_keeps_solver_numbers_verdicts_and_errors():
    compact = _compact_fea_verification(_fea_report())
    assert compact["solver"] == "tet4"
    assert compact["resolution"]["target_faces"] == 12000
    best = compact["designs"]["optimized"]
    assert best["mass_kg"] == 0.81 and best["vertical_displacement_mm"] == 0.26
    assert best["tets"] == 40000
    assert best["label_surface_peak_von_mises_MPa"] == 95.0
    assert best["label_surface_vertical_displacement_mm"] == 0.25
    assert best["limits"][0]["met"] is False
    # Per-node duplicates of the surrogate record stay in the json file only.
    assert "surrogate" not in best and "nodes" not in best
    assert compact["designs"]["typical"] == {"error": "RuntimeError: gmsh failed"}


def _surrogate_summary_with_fea(fea_verification):
    fast = {"ok": True, "fea": {**_field_result(), "mass": 0.8, "vertical_displacement": 1.5e-4}}
    base = {"ok": True, "fea": {**_field_result(), "vertical_displacement": 1.7e-4}}
    summary = _summary_for(fast, base, base, verify_evaluations=3)
    summary.update(wall_time_s=60.0, search_success_rate=1.0, best_search_score=0.9,
                   stress_percentile=99.5, fea_verification=fea_verification)
    return summary


def test_fea_verification_section_reports_a_violated_limit(tmp_path):
    summary = _surrogate_summary_with_fea(dict(_compact_fea_verification(_fea_report()),
                                               file=FEA_VERIFIED_NAME))
    text = "\n".join(_fea_verification_lines(summary))
    assert "## FEA verification (`opt_fea_verify`)" in text
    assert "12000 surface faces, mesh_size_max 0.05" in text
    # surrogate / solver on the label-surface measure, side by side
    assert "| optimized | 0.8100 | 100.0 / 95.0 | 0.1500 / 0.2500 |" in text
    assert "u_z **violated**" in text and "stress (calibrated) met" in text
    assert "| typical | re-solve failed: `RuntimeError: gmsh failed` |" in text
    assert "**violates** the vertical limit: max |u_z| 0.2600 mm > 0.2000 mm" in text
    assert "do not use it as is" in text
    assert "0.9500 -> 0.8100 kg (-14.7%)" in text
    # A table cell never carries a raw pipe from a limit's name.
    rows = [line for line in text.splitlines() if line.startswith("| optimized")]
    assert rows and rows[0].count("|") - rows[0].count("\\|") == 6

    _write_report(tmp_path, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert report.index("## FEA verification") < report.index("## Solver")
    assert "the **FEA verification** section below re-solves the result" in report


def test_fea_verification_section_flags_an_unsolved_delivered_design():
    fea = _compact_fea_verification(_fea_report())
    # The retry ladder's error joins its attempts with ' | '.
    fea["designs"]["optimized"] = {
        "error": "MeshingError: meshing failed at every face budget; 12000: overlapping "
                 "facets | 15000: overlapping facets"}
    text = "\n".join(_fea_verification_lines(_surrogate_summary_with_fea(dict(
        fea, file=FEA_VERIFIED_NAME))))
    rows = [line for line in text.splitlines() if line.startswith("| optimized")]
    # Same five cells as the header: the error's pipes stay inside its cell.
    assert rows and rows[0].count("|") - rows[0].count("\\|") == 6
    assert "12000: overlapping facets \\| 15000" in rows[0]
    assert "verdict is **unverified**; do not use it as is." in text
    assert "Solver mass, delivered vs best baseline" not in text


def test_fea_verification_section_reports_a_met_limit():
    summary = _surrogate_summary_with_fea(_compact_fea_verification(
        _fea_report(optimized_uz=0.19, optimized_met=True)))
    text = "\n".join(_fea_verification_lines(summary))
    assert "**meets** the vertical limit: max |u_z| 0.1900 mm <= 0.2000 mm" in text
    assert "do not use it as is" not in text


def test_fea_verification_section_says_when_the_resolve_did_not_complete(tmp_path):
    summary = _surrogate_summary_with_fea({"error": "ModuleNotFoundError: No module named 'gmsh'"})
    text = "\n".join(_fea_verification_lines(summary))
    assert "did not complete" in text and "No module named 'gmsh'" in text
    assert "| design |" not in text


def test_no_fea_verification_keeps_the_old_accuracy_advice(tmp_path):
    summary = _surrogate_summary_with_fea(None)
    del summary["fea_verification"]
    _write_report(tmp_path, summary)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## FEA verification" not in report
    assert "`opt_fea_verify true`" in report


def test_fea_verify_runs_verify_with_fea_in_process(monkeypatch, tmp_path):
    from design_loop import verify_with_fea
    calls = []

    def fake_main(argv):
        calls.append(argv)
        out = argv[argv.index("--json-out") + 1]
        with open(out, "w", encoding="utf-8") as fh:
            import json
            json.dump(_fea_report(), fh)
        return 0

    monkeypatch.setattr(verify_with_fea, "main", fake_main)
    result = _fea_verify(str(tmp_path))
    assert calls == [["--run-dir", str(tmp_path), "--json-out",
                      os.path.join(str(tmp_path), FEA_VERIFIED_NAME)]]
    assert result["file"] == FEA_VERIFIED_NAME and result["wall_time_s"] >= 0
    assert result["designs"]["optimized"]["mass_kg"] == 0.81
    assert FEA_VERIFIED_NAME in optimize_module._OWNED_OUTPUTS


def test_fea_verify_records_a_failure_instead_of_raising(monkeypatch, tmp_path):
    from design_loop import verify_with_fea

    def broken(_argv):
        raise SystemExit(2)

    monkeypatch.setattr(verify_with_fea, "main", broken)
    result = _fea_verify(str(tmp_path))
    assert result["error"] == "SystemExit: 2"
    assert "designs" not in result


def _bracket_interface_points(drop=None):
    """Points on the four bolt bore walls and both lug ear walls (DeepJEB mm frame),
    plus free points away from every interface."""
    from design_loop import interfaces as itf

    pts, kinds = [], []
    ang = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
    for name, (cx, cy) in itf.BOLTS.items():
        if name == drop:
            continue
        for z in np.linspace(*itf.BOLT_Z, 5):
            pts += [(cx + itf.R_BOLT * np.cos(a), cy + itf.R_BOLT * np.sin(a), z) for a in ang]
            kinds += [itf.NODE_BOLT] * len(ang)
    for name, (y0, y1) in itf.EARS.items():
        if name == drop:
            continue
        for y in np.linspace(y0, y1, 5):
            pts += [(itf.LUG_XZ[0] + itf.R_LUG * np.cos(a), y,
                     itf.LUG_XZ[1] + itf.R_LUG * np.sin(a)) for a in ang]
            kinds += [itf.NODE_LUG] * len(ang)
    pts += [(20.0, -70.0, 30.0), (60.0, -100.0, 5.0)]
    kinds += [itf.NODE_FREE] * 2
    return np.asarray(pts), np.asarray(kinds)


def test_node_types_marks_the_bores_and_ears_and_nothing_else():
    from design_loop.interfaces import node_types

    points, kinds = _bracket_interface_points()
    np.testing.assert_array_equal(node_types(points), kinds)


def test_node_types_refuses_a_shape_missing_a_bore():
    import pytest
    from design_loop.interfaces import InterfaceError, node_types

    points, _ = _bracket_interface_points(drop="bolt3")
    with pytest.raises(InterfaceError, match="bolt3"):
        node_types(points)


def test_layout_follows_the_inference_config(tmp_path):
    import pytest

    ver = tmp_path / "ver.txt"
    ver.write_text("model meshgraphnets\ninput_var 4\noutput_var 4\n% cond_var 2\n")
    old = tmp_path / "ex10.txt"
    old.write_text("model meshgraphnets\ninput_var 3\noutput_var 3\ncond_var 4\n")
    assert surrogate_module.layout_from_config(str(ver)) == "ver"
    assert surrogate_module.layout_from_config(str(old)) == "ex10"

    model = surrogate_module.HIMGNSurrogate(str(ver), "model.pth", load_cases=("ver",),
                                            workdir=tmp_path)
    assert (model.layout, model.surface_faces) == ("ver", 0)
    with pytest.raises(surrogate_module.SurrogateError, match="vertical load case only"):
        surrogate_module.HIMGNSurrogate(str(ver), "model.pth", load_cases=("ver", "dia"),
                                        workdir=tmp_path)


def test_ver_layout_refuses_a_shape_without_the_interfaces(monkeypatch, tmp_path):
    import trimesh

    predicted = []
    monkeypatch.setattr(surrogate_module, "write_inference_contract", lambda *_, **__: None)
    model = surrogate_module.HIMGNSurrogate("infer.txt", "model.pth", load_cases=("ver",),
                                            workdir=tmp_path, layout="ver")
    model._run_native = lambda *_: predicted.append(True) or {}
    slab = trimesh.creation.box(extents=(1.0, 1.8, 0.2)).subdivide().subdivide()

    assert model.analyze_batch([slab], names=["slab"]) == [None]
    assert not predicted
    assert model.last_errors[0].startswith("InterfaceError: interface gate failed")
