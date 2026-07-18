"""Fail-fast guards in validate_common_config for accepted-but-unimplemented
features: this repository never writes normalization statistics back into a
source HDF5, and the dataset loader does not yet attach `global_conditions`
to graphs, so both must be rejected at config load rather than crash with an
AttributeError mid-run (regressions found 2026-07-17).
"""

import pytest

from general_modules.config_validation import validate_common_config


def _minimal_cfg(**overrides):
    cfg = {'model': 'deeponet', 'mode': 'train'}
    cfg.update(overrides)
    return cfg


def test_write_preprocessing_true_rejected():
    with pytest.raises(ValueError, match="write_preprocessing"):
        validate_common_config(_minimal_cfg(write_preprocessing=True))


def test_write_preprocessing_false_accepted():
    validate_common_config(_minimal_cfg(write_preprocessing=False))


def test_global_condition_features_rejected_until_loader_attaches_them():
    # Single name (parses to a bare lowercase string) and list-of-names forms.
    with pytest.raises(ValueError, match="global_condition_features"):
        validate_common_config(_minimal_cfg(global_condition_features='force'))
    with pytest.raises(ValueError, match="global_condition_features"):
        validate_common_config(_minimal_cfg(global_condition_features=['force', 'mass']))


def test_global_condition_features_none_accepted():
    validate_common_config(_minimal_cfg(global_condition_features='none'))
