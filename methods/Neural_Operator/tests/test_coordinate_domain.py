import pytest
import torch

from model.adapters.coordinate_domain import CoordinateDomain


def make_domain(policy='error'):
    return CoordinateDomain(
        active_axes=(0, 1),
        grid_bound_min=torch.tensor([-1.0, -2.0]),
        grid_bound_max=torch.tensor([1.0, 2.0]),
        out_of_bounds_policy=policy,
    )


def test_to_unit_box_maps_bounds_to_0_1():
    domain = make_domain()
    pos = torch.tensor([[-1.0, -2.0, 99.0], [1.0, 2.0, 99.0], [0.0, 0.0, 99.0]])
    c01, oob = domain.to_unit_box(pos)
    assert oob == 0
    assert torch.allclose(c01, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]), atol=1e-6)


def test_out_of_bounds_error_policy_raises():
    domain = make_domain('error')
    pos = torch.tensor([[-5.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        domain.to_unit_box(pos)


def test_out_of_bounds_clamp_policy_clips_and_counts():
    domain = make_domain('clamp')
    pos = torch.tensor([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    c01, oob = domain.to_unit_box(pos)
    assert oob == 1
    assert c01[0, 0].item() == 0.0  # clamped to lower bound


def test_roundtrip_dict():
    domain = make_domain()
    d = domain.to_dict()
    restored = CoordinateDomain.from_dict(d)
    assert restored.active_axes == domain.active_axes
    assert torch.allclose(restored.grid_bound_min, domain.grid_bound_min)
    assert torch.allclose(restored.grid_bound_max, domain.grid_bound_max)


def test_select_active_picks_correct_columns_3d():
    domain = CoordinateDomain(
        active_axes=(0, 2),
        grid_bound_min=torch.tensor([0.0, 0.0]),
        grid_bound_max=torch.tensor([1.0, 1.0]),
    )
    pos = torch.tensor([[1.0, 999.0, 3.0]])
    selected = domain.select_active(pos)
    assert torch.allclose(selected, torch.tensor([[1.0, 3.0]]))
