import numpy as np
import pytest
from general_modules import fom_dataset as F


def test_holdout_extremes_do_not_fit_scaler(monkeypatch):
    data = np.arange(10, dtype=np.float32).reshape(10, 1, 1)
    train_idx, val_idx, test_idx = F._split_indices(10, 42)
    data[val_idx] = 1000
    data[test_idx] = -1000
    monkeypatch.setattr(F, 'load_fom_from_hdf5', lambda _: (data, list(range(10))))
    train, val, test = F.build_dataset_splits({}, 42)
    np.testing.assert_array_equal(train.normalization['field_max'], data[train_idx].max(axis=(0, 1)))
    assert set(train.indices).isdisjoint(val.indices)
    assert set(train.indices).isdisjoint(test.indices)
    assert set(val.indices).isdisjoint(test.indices)
    assert val.data[val.indices].max() > 0.7
    lc_train, lc_val, lc_test = F.build_dataset_splits(
        {}, 42, normalization=train.normalization, split_manifest=train.split_manifest)
    assert lc_train.indices == train.indices and lc_val.indices == val.indices
    assert lc_test.indices == test.indices
    np.testing.assert_array_equal(lc_train.data, train.data)
    with pytest.raises(ValueError, match='split_seed'):
        F.build_dataset_splits({}, 7, split_manifest=train.split_manifest)


@pytest.mark.parametrize('n', [3, 4, 6, 10, 11])
def test_small_splits_are_disjoint_and_complete(n):
    parts = F._split_indices(n, 42)
    assert all(len(p) > 0 for p in parts)
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(n))


@pytest.mark.parametrize('n', [0, 1, 2])
def test_insufficient_samples_fail_clearly(n):
    with pytest.raises(ValueError, match='three samples'):
        F._split_indices(n, 42)
