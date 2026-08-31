from pathlib import Path

from cae_suite.config_parser import ParsedConfig
from cae_suite.diagnostics import DiagnosticReport
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


def test_fea_does_not_require_surrogate_files():
    assert not [item for item in _diagnostics() if item.code == "SDF-OPT-SURROGATE-001"]


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
