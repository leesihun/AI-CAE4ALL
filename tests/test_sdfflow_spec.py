from pathlib import Path

from cae_suite.config_parser import ParsedConfig
from cae_suite.diagnostics import DiagnosticReport, Severity
from cae_suite.preflight import PreflightOptions, run_preflight
from cae_suite.registry import MethodRegistry
from cae_suite.settings import LocalSettings
from cae_suite.specs.base import SpecValidationContext
from cae_suite.specs.sdfflow import validate_sdfflow


def _diagnostics(**overrides):
    values = {
        "model": "sdfflow",
        "mode": "optimize",
        "gpu_ids": [0],
        "opt_analysis": "fea",
        "opt_load_cases": ["vertical"],
        "opt_condition_dims": ["volume"],
        **overrides,
    }
    parsed = ParsedConfig(source_path=Path("test-config.txt"), values=values)
    report = DiagnosticReport()
    validate_sdfflow(SpecValidationContext(
        parsed=parsed,
        mode="optimize",
        model_id="sdfflow",
        repository_root=Path.cwd(),
        report=report,
    ))
    return report.diagnostics


def _diagnose(mode, values):
    """Spec-layer diagnostics for one mode and one set of written config values."""
    parsed = ParsedConfig(
        source_path=Path("test-config.txt"),
        values={"model": "sdfflow", "mode": mode, "gpu_ids": [0], **values},
    )
    report = DiagnosticReport()
    validate_sdfflow(SpecValidationContext(
        parsed=parsed,
        mode=mode,
        model_id="sdfflow",
        repository_root=Path.cwd(),
        report=report,
    ))
    return report.diagnostics


def _codes(mode, values):
    return {item.code for item in _diagnose(mode, values)}


def _errors(mode, values):
    return {item.code for item in _diagnose(mode, values) if item.severity is Severity.ERROR}


def test_fea_does_not_require_surrogate_files():
    assert not [item for item in _diagnostics() if item.code == "SDF-OPT-SURROGATE-001"]


def test_fea_mesh_controls_are_conditionally_recommended():
    diagnostics = _diagnostics(opt_analysis="fea")
    assert {item.field for item in diagnostics if item.code == "SDF-OPT-FEA-REC-001"} == {
        "opt_target_faces",
        "opt_mesh_size_max",
    }


def test_surrogate_does_not_recommend_unused_fea_mesh_controls():
    diagnostics = _diagnostics(opt_analysis="surrogate")
    assert not [item for item in diagnostics if item.code == "SDF-OPT-FEA-REC-001"]


def test_surrogate_requires_checkpoint_and_config():
    diagnostics = _diagnostics(opt_analysis="surrogate")
    assert {item.field for item in diagnostics if item.code == "SDF-OPT-SURROGATE-001"} == {
        "opt_surrogate_checkpoint",
        "opt_surrogate_config",
    }


def test_surrogate_files_satisfy_conditional_requirement():
    diagnostics = _diagnostics(
        opt_analysis="surrogate",
        opt_surrogate_checkpoint="model.pth",
        opt_surrogate_config="infer.txt",
    )
    assert not [item for item in diagnostics if item.code == "SDF-OPT-SURROGATE-001"]


def test_invalid_analysis_backend_and_target_node_count_are_rejected():
    diagnostics = _diagnostics(opt_analysis="invented", opt_surrogate_target_nodes=0)
    assert any(item.code == "SDF-OPT-ANALYSIS-001" for item in diagnostics)
    assert any(
        item.code == "SDF-OPT-POSITIVE-001" and item.field == "opt_surrogate_target_nodes"
        for item in diagnostics
    )


def test_published_optimize_defaults_satisfy_required_fields(tmp_path):
    suite_root = Path.cwd()
    config = tmp_path / "optimize-defaults.txt"
    config.write_text("\n".join((
        "model sdfflow",
        "mode optimize",
        "gpu_ids 0",
        "vae_modelpath vae.pth",
        "fm_modelpath fm.pth",
        "output_dir output",
        "seed 0",
        "opt_load_cases vertical,diagonal",
    )), encoding="utf-8")
    result = run_preflight(
        config,
        suite_root=suite_root,
        registry=MethodRegistry(suite_root),
        settings=LocalSettings(base_dir=suite_root),
        options=PreflightOptions(
            skip_filesystem=True,
            skip_native=True,
            skip_environment=True,
            skip_dataset=True,
        ),
    )
    missing_fields = {
        item.field for item in result.report.diagnostics if item.code == "CFG-REQ-001"
    }
    assert not missing_fields


# ---------------------------------------------------------------------------
# Conditional-generation validators (CONDITIONAL_GENERATION_DESIGN_2026-09).
# One negative case per ERROR code plus a positive case per shipped config, so
# a flipped severity or a dropped PathRule fails here rather than at run time.
# ---------------------------------------------------------------------------

def test_cond_dropout_mode_enum():
    assert "SDF-CDROP-001" in _errors("train_fm", {"cond_dropout_mode": "sometimes"})
    assert "SDF-CDROP-001" not in _codes("train_fm", {"cond_dropout_mode": "per_dim"})


def test_cond_dropout_all_prob_range_and_starved_cfg_branch():
    assert "SDF-CDROP-002" in _errors("train_fm", {"cond_dropout_all_prob": 1.0})
    assert "SDF-CDROP-003" in _codes(
        "train_fm", {"cond_dropout_mode": "per_dim", "cond_dropout_all_prob": 0.0})
    assert "SDF-CDROP-003" not in _codes(
        "train_fm", {"cond_dropout_mode": "per_dim", "cond_dropout_all_prob": 0.1})


def test_guidance_step_mode_and_audit_and_eval_task_enums():
    assert "SDF-GUIDE-001" in _errors("sample", {"guidance_step_mode": "jump"})
    assert "SDF-AUDIT-001" in _errors("sample", {"condition_audit": "ansys"})
    assert "SDF-EVAL-002" in _errors("evaluate", {"eval_task": "generative"})
    assert "SDF-EVAL-003" in _errors(
        "evaluate", {"eval_task": "conditional", "eval_methods": ["plain", "e3"]})


def test_guidance_enabled_accepts_a_truthy_non_bool():
    """`guidance_enabled 1` parses to int 1 and turns guidance ON natively."""
    for written in (True, 1, "yes"):
        assert "SDF-CALIB-001" in _errors(
            "sample", {"guidance_enabled": written, "cond_values": [0.3, 4.8]})
    assert "SDF-CALIB-001" not in _codes(
        "sample", {"guidance_enabled": False, "cond_values": [0.3, 4.8]})


def test_newton_without_calibration_is_an_error():
    assert "SDF-CALIB-001" in _errors(
        "sample", {"newton_rounds": 3, "cond_values": [0.3, 4.8]})
    assert "SDF-NEWTON-001" in _errors("sample", {"newton_rounds": -1})


def test_eval_methods_default_still_needs_the_calibration():
    """The native default (plain, rejection, e2) contains e2, which reads it."""
    assert "SDF-EVAL-007" in _errors(
        "evaluate", {"eval_task": "conditional", "fm_modelpath": "fm.pth"})
    assert "SDF-EVAL-007" not in _codes(
        "evaluate", {"eval_task": "conditional", "fm_modelpath": "fm.pth",
                     "eval_methods": ["plain", "rejection"]})


def test_non_reconstruction_eval_task_requires_fm_modelpath():
    assert "SDF-EVAL-004" in _errors("evaluate", {"eval_task": "descriptor_calibration",
                                                  "descriptor_calibration_path": "c.pth"})
    assert "SDF-EVAL-004" not in _codes("evaluate", {"eval_task": "descriptor_calibration",
                                                     "descriptor_calibration_path": "c.pth",
                                                     "fm_modelpath": "fm.pth"})


def test_calibration_task_requires_its_output_path():
    assert "SDF-EVAL-007" in _errors(
        "evaluate", {"eval_task": "descriptor_calibration", "fm_modelpath": "fm.pth"})


def test_eval_exclude_shapes_and_min_r2_values():
    assert "SDF-EVAL-009" in _errors("evaluate", {"eval_exclude_shapes": ["2099", "abc"]})
    assert "SDF-EVAL-009" not in _codes("evaluate", {"eval_exclude_shapes": [2099]})
    assert "SDF-CALIB-005" in _errors("evaluate", {"calibration_min_r2": 1.5})


def test_cond_sweep_requires_both_endpoints_of_equal_length_and_two_steps():
    base = {"interpolation_space": "cond_sweep"}
    assert "SDF-SWEEP-003" in _errors("interpolate", base)
    assert "SDF-SWEEP-004" in _errors(
        "interpolate", dict(base, cond_values_a=[0.2, 4.0], cond_values_b=[0.3, 4.8, 6.8]))
    # Native precondition: interpolate.py raises for sweep_steps < 2.
    assert "SDF-SWEEP-001" in _errors(
        "interpolate", dict(base, cond_values_a=[0.2, 4.0], cond_values_b=[0.3, 4.8],
                            sweep_steps=1))
    assert "SDF-SWEEP-001" not in _codes(
        "interpolate", dict(base, cond_values_a=[0.2, 4.0], cond_values_b=[0.3, 4.8],
                            sweep_steps=5))
    # sweep_steps is inert outside cond_sweep and must not raise there.
    assert "SDF-SWEEP-001" not in _codes(
        "interpolate", {"interpolation_space": "slerp_noise", "sample_index_b": 1,
                        "sweep_steps": 1})


def test_condition_entries_accept_nan_but_not_words():
    assert "SDF-COND-003" in _errors("sample", {"cond_values": [0.3, "big"]})
    assert "SDF-COND-003" not in _codes("sample", {"cond_values": [0.3, "nan"]})
    assert "SDF-COND-PARTIAL-001" in _codes("sample", {"cond_values": [0.3, "nan"]})


def test_structural_audit_settings_are_validated_outside_optimize():
    """condition_audit fea|surrogate makes the opt_* block a sample/evaluate key set."""
    codes = _codes("sample", {"condition_audit": "fea", "opt_material_nu": 0.9,
                              "opt_length_scale": -1.0, "opt_stress_percentile": 150.0})
    assert {"SDF-OPT-NU-001", "SDF-OPT-POSITIVE-001", "SDF-OPT-PERCENTILE-001"} <= codes
    assert "SDF-OPT-SURROGATE-001" in _errors("sample", {"condition_audit": "surrogate"})
    assert "SDF-OPT-SURROGATE-001" not in _codes(
        "sample", {"condition_audit": "surrogate", "opt_surrogate_config": "infer.txt",
                   "opt_surrogate_checkpoint": "model.pth"})


CONDITIONAL_CONFIGS = (
    "config_train_v3_fea.txt",
    "config_calibrate_descriptors.txt",
    "config_sample_conditional.txt",
    "config_cond_sweep.txt",
    "config_evaluate_conditional.txt",
)


def test_checked_in_conditional_configs_have_no_spec_layer_findings():
    """The five shipped ex5 configs must be clean at the spec layer (their only
    expected preflight findings are PATH-INPUT-001 on the untrained checkpoints
    and, for the FEA training config, the dataset-layer SDF-COND-FEA-003)."""
    from cae_suite.config_parser import parse_config

    suite_root = Path(__file__).resolve().parents[1]
    repository_root = suite_root / "methods" / "SDFFlow"
    for name in CONDITIONAL_CONFIGS:
        path = suite_root / "configs" / "SDFFlow" / name
        parsed = parse_config(path)
        mode = str(parsed.values["mode"]).lower()
        report = DiagnosticReport()
        validate_sdfflow(SpecValidationContext(
            parsed=parsed,
            mode=mode,
            model_id="sdfflow",
            repository_root=repository_root,
            report=report,
        ))
        offending = [f"{d.code} {d.message}" for d in report.diagnostics
                     if d.severity in (Severity.ERROR, Severity.WARNING)]
        assert not offending, f"{name}: {offending}"


# ---------------------------------------------------------------------------
# Dataset layer: the FEA condition sidecar (SDF-COND-FEA-003)
# ---------------------------------------------------------------------------

def _sdf_dataset_diagnostics(metadata, mode="train", **values):
    """Run the dataset-layer SDFFlow cross-check against probe metadata."""
    from cae_suite.preflight import PreflightResult, _validate_dataset_against_config

    parsed = ParsedConfig(
        source_path=Path("test-config.txt"),
        values={"model": "sdfflow", "mode": mode, "gpu_ids": [0], **values},
    )
    result = PreflightResult(parsed=parsed, report=DiagnosticReport(), mode=mode)
    result.dataset_metadata = dict(metadata)
    _validate_dataset_against_config(result, "dataset_dir")
    return {item.code for item in result.report.diagnostics if item.severity is Severity.ERROR}


def _probe_sdf_metadata(path, with_sidecar):
    """Build a minimal sdf_hdf5 file and return dataset_probe's metadata for it."""
    import h5py
    import numpy as np

    from cae_suite.dataset_probe import _sdf_report

    with h5py.File(path, "w") as handle:
        handle.attrs["cond_names"] = ["bbox_x", "bbox_y", "bbox_z", "volume", "area"]
        if with_sidecar:
            handle.attrs["cond_extra_names"] = ["log_max_ver_stress_mpa"]
            handle.create_dataset("cond_extra", data=np.zeros((1, 1), dtype="float32"))
        shape = handle.create_group("shapes/00000")
        shape.create_dataset("surface_points", data=np.zeros((4, 3), dtype="float32"))
        shape.create_dataset("surface_normals", data=np.zeros((4, 3), dtype="float32"))
        shape.create_dataset("sdf_points", data=np.zeros((4, 3), dtype="float32"))
        shape.create_dataset("sdf_values", data=np.zeros((4,), dtype="float32"))
        shape.create_dataset("cond", data=np.zeros((5,), dtype="float32"))
    with h5py.File(path, "r") as handle:
        return _sdf_report(handle)["metadata"]


def test_fea_condition_names_without_the_sidecar_are_a_dataset_layer_error(tmp_path):
    """The whole point: fail in preflight, not after the VAE stage has trained."""
    fea = ["volume", "area", "log_max_ver_stress_mpa"]
    bare = _probe_sdf_metadata(tmp_path / "bare.h5", with_sidecar=False)
    assert bare["cond_names"] == ["bbox_x", "bbox_y", "bbox_z", "volume", "area"]
    assert bare["cond_extra_names"] == []
    assert "SDF-COND-FEA-003" in _sdf_dataset_diagnostics(
        bare, use_conditions=True, condition_names=fea)

    withcar = _probe_sdf_metadata(tmp_path / "fea.h5", with_sidecar=True)
    assert withcar["cond_extra_names"] == ["log_max_ver_stress_mpa"]
    assert "SDF-COND-FEA-003" not in _sdf_dataset_diagnostics(
        withcar, use_conditions=True, condition_names=fea)


def test_sidecar_check_is_silent_when_the_probe_did_not_run():
    """--skip-dataset-check leaves no vocabulary; the check must not guess."""
    assert "SDF-COND-FEA-003" not in _sdf_dataset_diagnostics(
        {"shape_count": 1}, use_conditions=True,
        condition_names=["log_max_ver_stress_mpa"])


def test_use_conditions_written_as_one_is_still_validated(tmp_path):
    """`use_conditions 1` parses to int 1; the native reader is bool()."""
    bare = _probe_sdf_metadata(tmp_path / "bare.h5", with_sidecar=False)
    assert "SDF-COND-FEA-003" in _sdf_dataset_diagnostics(
        bare, use_conditions=1, condition_names=["log_max_ver_stress_mpa"])
    assert "SDF-COND-001" in _errors("train_fm", {"use_conditions": 1})
