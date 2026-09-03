import math

import torch

from model.siren import SineLayer, Siren


def test_first_layer_init_bounds():
    layer = SineLayer(10, 32, omega0=30.0, is_first=True)
    bound = 1.0 / 10
    assert layer.linear.weight.min().item() >= -bound - 1e-6
    assert layer.linear.weight.max().item() <= bound + 1e-6
    assert torch.all(layer.linear.bias == 0)


def test_hidden_layer_init_bounds():
    omega0 = 30.0
    layer = SineLayer(32, 32, omega0=omega0, is_first=False)
    bound = math.sqrt(6.0 / 32) / omega0
    assert layer.linear.weight.min().item() >= -bound - 1e-6
    assert layer.linear.weight.max().item() <= bound + 1e-6


def test_forward_is_finite_and_bounded():
    siren = Siren(in_features=5, hidden_features=64, depth=3, omega0=30.0)
    x = torch.randn(20, 5)
    out = siren(x)
    assert out.shape == (20, 64)
    assert torch.isfinite(out).all()
    assert torch.all(out >= -1.0 - 1e-5) and torch.all(out <= 1.0 + 1e-5)  # sine output


def test_depth_controls_layer_count():
    siren = Siren(in_features=4, hidden_features=16, depth=5, omega0=30.0)
    assert len(siren.layers) == 5
