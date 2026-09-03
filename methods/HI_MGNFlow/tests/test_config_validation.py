import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "general_modules"
    / "config_validation.py"
)
SPEC = importlib.util.spec_from_file_location("chi_config_validation", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
validate_runtime_config = MODULE.validate_runtime_config


def test_ddp_is_the_only_executable_parallel_mode():
    assert validate_runtime_config({}) == "ddp"
    assert validate_runtime_config({"parallel_mode": " DDP "}) == "ddp"
    with pytest.raises(ValueError, match="not implemented for cHI-MGNflow"):
        validate_runtime_config({"parallel_mode": "model_split"})
    with pytest.raises(ValueError, match="must be 'ddp'"):
        validate_runtime_config({"parallel_mode": "single"})


def test_copied_pipeline_and_noise_controls_cannot_silently_do_nothing():
    assert validate_runtime_config({"std_noise": 0}) == "ddp"
    with pytest.raises(ValueError, match="pipeline_microbatches"):
        validate_runtime_config({"pipeline_microbatches": 2})
    with pytest.raises(ValueError, match="not implemented"):
        validate_runtime_config({"std_noise": 0.1})
    with pytest.raises(ValueError, match="noise_gamma"):
        validate_runtime_config({"noise_gamma": 0.1})
