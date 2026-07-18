import torch

from training_profiles.training_loop import _build_loss_weights, _loss_from_errors, _per_node_loss, _accum_window_size


def test_loss_weights_normalize_to_sum_one():
    config = {'feature_loss_weights': [1.0, 1.0, 2.0, 4.0]}
    w = _build_loss_weights(config, torch.device('cpu'))
    assert torch.allclose(w.sum(), torch.tensor(1.0))
    assert torch.allclose(w, torch.tensor([1.0, 1.0, 2.0, 4.0]) / 8.0)


def test_equal_weights_reduce_to_feature_mean():
    config = {'feature_loss_weights': None}
    w = _build_loss_weights(config, torch.device('cpu'))
    errors = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    per_node = _per_node_loss(errors, w)
    assert torch.allclose(per_node, torch.tensor([2.0, 3.0]))  # mean over features


def test_weighted_per_node_matches_manual_computation():
    w = torch.tensor([0.25, 0.75])
    errors = torch.tensor([[1.0, 3.0], [2.0, 4.0]])
    per_node = _per_node_loss(errors, w)
    expected = torch.tensor([1 * 0.25 + 3 * 0.75, 2 * 0.25 + 4 * 0.75])
    assert torch.allclose(per_node, expected)


def test_loss_from_errors_sum_and_count():
    errors = torch.ones(5, 3) * 2.0  # per_node = mean(2,2,2)=2 for all 5 nodes
    loss, loss_sum, loss_count = _loss_from_errors(errors, None)
    assert loss_count == 5
    assert torch.allclose(loss_sum, torch.tensor(10.0))
    assert torch.allclose(loss, torch.tensor(2.0))


def test_accumulation_sums_are_node_weighted_not_batch_averaged():
    """Two microbatches of different sizes: the epoch-level mean must be the
    node-weighted mean (sum/count), not the mean of the two per-batch means."""
    errors_a = torch.ones(3, 2) * 1.0   # per_node=1.0, 3 nodes -> sum=3
    errors_b = torch.ones(7, 2) * 5.0   # per_node=5.0, 7 nodes -> sum=35
    _, sum_a, count_a = _loss_from_errors(errors_a, None)
    _, sum_b, count_b = _loss_from_errors(errors_b, None)
    total_sum = (sum_a + sum_b).item()
    total_count = count_a + count_b
    node_weighted_mean = total_sum / total_count
    batch_averaged_mean = (1.0 + 5.0) / 2  # what it would be if WRONGLY averaged per-batch
    assert abs(node_weighted_mean - 3.8) < 1e-6
    assert abs(node_weighted_mean - batch_averaged_mean) > 1e-6


def test_accum_window_size_handles_uneven_last_window():
    # 10 batches, accumulate every 3 -> windows of size 3,3,3,1
    assert _accum_window_size(0, 10, 3) == 3
    assert _accum_window_size(2, 10, 3) == 3
    assert _accum_window_size(9, 10, 3) == 1
