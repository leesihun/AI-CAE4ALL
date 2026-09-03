import torch

from model.adapters.point_sampling import PointSampler, stable_hash


def test_without_replacement_when_enough_nodes():
    sampler = PointSampler(sensor_count=10, base_seed=0)
    idx = sampler.sample_indices(num_nodes=100, sample_id=1, time_idx=None, training=True)
    assert idx.shape[0] == 10
    assert len(set(idx.tolist())) == 10  # no duplicates


def test_with_replacement_when_too_few_nodes():
    sampler = PointSampler(sensor_count=20, base_seed=0)
    idx = sampler.sample_indices(num_nodes=5, sample_id=1, time_idx=None, training=True)
    assert idx.shape[0] == 20
    assert idx.max().item() < 5


def test_eval_sampling_is_deterministic_across_calls():
    sampler = PointSampler(sensor_count=10, base_seed=0)
    sampler.set_epoch(5)  # must not matter for eval
    idx1 = sampler.sample_indices(num_nodes=50, sample_id=3, time_idx=2, training=False)
    idx2 = sampler.sample_indices(num_nodes=50, sample_id=3, time_idx=2, training=False)
    assert torch.equal(idx1, idx2)


def test_training_resamples_across_epochs():
    sampler = PointSampler(sensor_count=10, base_seed=0, resample_each_epoch=True)
    sampler.set_epoch(0)
    idx0 = sampler.sample_indices(num_nodes=100, sample_id=1, time_idx=None, training=True)
    sampler.set_epoch(1)
    idx1 = sampler.sample_indices(num_nodes=100, sample_id=1, time_idx=None, training=True)
    assert not torch.equal(idx0, idx1)


def test_training_pinned_epoch_when_resample_disabled():
    sampler = PointSampler(sensor_count=10, base_seed=0, resample_each_epoch=False)
    sampler.set_epoch(0)
    idx0 = sampler.sample_indices(num_nodes=100, sample_id=1, time_idx=None, training=True)
    sampler.set_epoch(7)
    idx1 = sampler.sample_indices(num_nodes=100, sample_id=1, time_idx=None, training=True)
    assert torch.equal(idx0, idx1)


def test_different_samples_get_different_indices():
    sampler = PointSampler(sensor_count=10, base_seed=0)
    idx_a = sampler.sample_indices(num_nodes=100, sample_id=1, time_idx=None, training=False)
    idx_b = sampler.sample_indices(num_nodes=100, sample_id=2, time_idx=None, training=False)
    assert not torch.equal(idx_a, idx_b)


def test_stable_hash_deterministic():
    assert stable_hash(1, 2, 3) == stable_hash(1, 2, 3)
    assert stable_hash(1, 2, 3) != stable_hash(1, 2, 4)


def test_rejects_zero_sensor_count():
    import pytest
    with pytest.raises(ValueError):
        PointSampler(sensor_count=0)
