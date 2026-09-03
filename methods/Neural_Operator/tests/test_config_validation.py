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


def test_parallel_stats_and_train_eval_subset_are_authorable():
    validate_common_config(_minimal_cfg(
        use_parallel_stats=False,
        train_eval_subset_size=32,
    ))


def test_paper_decoder_domain_padding_is_authorable():
    validate_common_config(_minimal_cfg(
        model='gino',
        gino_variant='paper_decoder',
        gino_domain_padding=0.25,
    ))


def test_point_deeponet_closed_choices_fail_before_model_construction():
    for key, value in (
        ('point_variant', 'paper'),
        ('point_sampling', 'fps'),
        ('pointnet_activation', 'gelu'),
        ('pointnet_norm', 'layer'),
        ('point_branch_merge', 'concat'),
        ('point_output_activation', 'sigmoid'),
    ):
        with pytest.raises(ValueError, match=key):
            validate_common_config(_minimal_cfg(model='point_deeponet', **{key: value}))


def test_point_sensor_count_zero_keeps_native_all_points_contract():
    validate_common_config(_minimal_cfg(model='point_deeponet', point_sensor_count=0))


def test_common_adapter_choices_and_ranges_fail_fast():
    for key, value in (
        ('coordinate_normalization', 'per_axis'),
        ('dimension_tolerance', 0.0),
        ('grid_padding', -0.1),
        ('out_of_bounds_policy', 'zero'),
        ('sdf_source', 'invented'),
        ('integration_weight_source', 'area'),
    ):
        with pytest.raises(ValueError, match=key):
            validate_common_config(_minimal_cfg(**{key: value}))


def test_sdf_dataset_source_is_authorable():
    validate_common_config(_minimal_cfg(sdf_source='dataset'))


def test_fno_mode_corners_cannot_overlap():
    with pytest.raises(ValueError, match=r'fno_modes\[0\]'):
        validate_common_config(_minimal_cfg(
            model='fno', operator_dim=2,
            fno_grid_resolution=[8, 8], fno_modes=[5, 4],
        ))


def test_paper_decoder_uses_centered_total_mode_convention():
    validate_common_config(_minimal_cfg(
        model='gino', gino_variant='paper_decoder',
        gino_grid_resolution=[8, 8, 8], gino_fno_modes=[8, 8, 8],
    ))
    with pytest.raises(ValueError, match=r'gino_fno_modes\[0\]'):
        validate_common_config(_minimal_cfg(
            model='gino', gino_variant='paper_decoder',
            gino_grid_resolution=[8, 8, 8], gino_fno_modes=[9, 8, 8],
        ))


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_train_eval_subset_size_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match='train_eval_subset_size'):
        validate_common_config(_minimal_cfg(train_eval_subset_size=value))
