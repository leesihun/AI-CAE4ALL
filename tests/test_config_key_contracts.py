from pathlib import Path

from cae_suite.config_parser import ParsedConfig
from cae_suite.diagnostics import DiagnosticReport, Severity
from cae_suite.specs.base import SpecValidationContext
from cae_suite.specs.chi_mgnflow import validate_chi_mgnflow
from cae_suite.specs.neural_operator import build_neural_operator_spec, validate_neural_operator
from cae_suite.specs.simulgenvae import build_simulgenvae_spec, validate_simulgenvae
from cae_suite.specs.transolver import build_transolver_spec, validate_transolver


ROOT = Path(__file__).resolve().parents[1]


def _diagnostics(validator, model, **values):
    parsed = ParsedConfig(
        source_path=Path("contract-test.txt"),
        values={"model": model, "mode": "train", **values},
    )
    report = DiagnosticReport()
    validator(SpecValidationContext(
        parsed=parsed,
        mode="train",
        model_id=model,
        repository_root=ROOT,
        report=report,
    ))
    return report.diagnostics


def test_neural_operator_runtime_controls_are_registered_with_real_defaults():
    spec = build_neural_operator_spec()
    assert {"use_parallel_stats", "train_eval_subset_size", "gino_domain_padding"} <= spec.known_keys
    assert spec.defaults["use_parallel_stats"] is True
    assert spec.defaults["train_eval_subset_size"] == 128


def test_chi_mgnflow_rejects_the_unimplemented_copied_pipeline_surface():
    split = _diagnostics(
        validate_chi_mgnflow,
        "chi-mgnflow",
        parallel_mode="model_split",
        gpu_ids=[0, 1],
    )
    assert any(item.code == "FLOW-PARALLEL" for item in split)

    stale = _diagnostics(
        validate_chi_mgnflow,
        "chi-mgnflow",
        pipeline_microbatches=4,
        std_noise=0.1,
    )
    assert sum(item.code == "FLOW-RUNTIME-REMOVED" for item in stale) == 2

    legacy_zero = _diagnostics(
        validate_chi_mgnflow,
        "chi-mgnflow",
        std_noise=0,
        message_passing_num=1,
    )
    assert any(item.code == "FLOW-LEGACY-NOISE" for item in legacy_zero)
    assert not [item for item in legacy_zero if item.severity == Severity.ERROR]


def test_neural_operator_closed_choices_match_executable_native_paths():
    invalid_common = _diagnostics(
        validate_neural_operator,
        "deeponet",
        coordinate_normalization="per_axis",
        out_of_bounds_policy="zero",
        sdf_source="invented",
        integration_weight_source="area",
    )
    assert {item.code for item in invalid_common} >= {
        "NOVAR-COORD-001", "NOVAR-OOB-001", "NOVAR-SDF-SOURCE",
        "NOVAR-INTEGRATION-WEIGHTS",
    }

    invalid_point = _diagnostics(
        validate_neural_operator,
        "point_deeponet",
        point_variant="paper",
        point_sampling="fps",
        pointnet_activation="gelu",
        pointnet_norm="layer",
        point_branch_merge="concat",
        point_output_activation="sigmoid",
    )
    assert sum(item.code == "NOVAR-POINT-CHOICE" for item in invalid_point) == 6

    all_points = _diagnostics(
        validate_neural_operator, "point_deeponet", point_sensor_count=0
    )
    assert not [item for item in all_points if item.field == "point_sensor_count"]


def test_neural_operator_fourier_mode_limits_match_each_native_kernel():
    fno = _diagnostics(
        validate_neural_operator,
        "fno",
        operator_dim=2,
        fno_grid_resolution=[8, 8],
        fno_modes=[5, 4],
    )
    assert any(item.code == "NOVAR-FNO-001" for item in fno)

    paper_ok = _diagnostics(
        validate_neural_operator,
        "gino",
        gino_variant="paper_decoder",
        gino_grid_resolution=[8, 8, 8],
        gino_fno_modes=[8, 8, 8],
    )
    assert not [item for item in paper_ok if item.code == "NOVAR-GINO-PAPER-GRID"]

    paper_bad = _diagnostics(
        validate_neural_operator,
        "gino",
        gino_variant="paper_decoder",
        gino_grid_resolution=[8, 8, 8],
        gino_fno_modes=[9, 8, 8],
    )
    assert any(item.code == "NOVAR-GINO-PAPER-GRID" for item in paper_bad)


def test_gino_variant_and_domain_padding_match_the_native_paper_decoder_contract():
    invalid = _diagnostics(
        validate_neural_operator,
        "gino",
        gino_variant="invented",
        gino_domain_padding=1.0,
    )
    assert {item.code for item in invalid} >= {"NOVAR-GINO-VARIANT", "NOVAR-GINO-PADDING"}

    inactive = _diagnostics(
        validate_neural_operator,
        "gino",
        gino_variant="mesh_state",
        gino_domain_padding=0.25,
    )
    assert any(item.code == "NOVAR-GINO-PADDING-INACTIVE" for item in inactive)

    split = _diagnostics(
        validate_neural_operator,
        "gino",
        gino_variant="paper_decoder",
        parallel_mode="model_split",
        gpu_ids=[0, 1],
    )
    assert any(item.code == "NOVAR-GINO-PARALLEL" for item in split)


def test_transolver_test_batch_idx_accepts_scalar_or_list_and_rejects_negative_values():
    spec = build_transolver_spec()
    assert "test_batch_idx" in spec.known_keys
    assert not [
        item for item in _diagnostics(validate_transolver, "transolver", test_batch_idx=0)
        if item.code == "TRANS-TEST-INDEX"
    ]
    assert not [
        item for item in _diagnostics(validate_transolver, "transolver", test_batch_idx=[0, 2])
        if item.code == "TRANS-TEST-INDEX"
    ]
    assert any(
        item.code == "TRANS-TEST-INDEX" and item.severity == Severity.ERROR
        for item in _diagnostics(validate_transolver, "transolver", test_batch_idx=[0, -1])
    )


def test_simulgen_fsdp_wrap_threshold_is_registered_and_positive():
    spec = build_simulgenvae_spec()
    assert "fsdp_min_params" in spec.known_keys
    assert spec.defaults["fsdp_min_params"] == 1_000_000
    assert any(
        item.code == "SGV-POSITIVE-001" and item.field == "fsdp_min_params"
        for item in _diagnostics(validate_simulgenvae, "simulgenvae", fsdp_min_params=0)
    )
    for key in ("load_all", "plot_mode", "recon_iter"):
        assert any(
            item.code == "SGV-REMOVED-NOOP" and item.field == key
            for item in _diagnostics(validate_simulgenvae, "simulgenvae", **{key: 1})
        )


def test_studio_catalog_extensions_match_suite_controls():
    source = (ROOT / "studio" / "src" / "constants.js").read_text(encoding="utf-8")
    assert 'KEY_CATALOGS.operator, "gino_domain_padding"' in source
    assert 'KEY_CATALOGS.simulgenvae, "fsdp_min_params"' in source
    assert 'SIMULGEN_REMOVED_NOOPS' in source
    assert 'gino_variant: ["mesh_state", "paper_decoder"]' in source
