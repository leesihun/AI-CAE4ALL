import pytest
import torch

from training_profiles.setup import build_optimizer_scheduler


def _optimizer(weight_decay=None):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    config = {"learningr": 0.001, "warmup_epochs": 1}
    if weight_decay is not None:
        config["weight_decay"] = weight_decay
    optimizer, _, _, _ = build_optimizer_scheduler(config, [parameter], total_epochs=3)
    return optimizer


def test_weight_decay_reaches_every_route_through_shared_optimizer_builder():
    optimizer = _optimizer(0.025)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.025)


def test_omitted_weight_decay_preserves_historical_zero_default():
    assert _optimizer().param_groups[0]["weight_decay"] == 0.0
