"""Minimum-image edges agree between loader and differentiable rollout."""
import numpy as np
import pytest
import torch
from general_modules.edge_features import compute_edge_attr, deformed_edge_attr_torch, minimum_image


def test_periodic_seam_numpy_torch_and_gradient():
    pos = np.array([[-.5, 0, 0], [.5, 0, 0]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 0]])
    box = [128 / 127, 0, 0]
    expected = compute_edge_attr(pos, pos, edges, periodic_box=box)
    np.testing.assert_allclose(expected[:, 3], 1 / 127, atol=1e-7)
    np.testing.assert_allclose(expected[:, 4:], expected[:, :4])
    t = torch.tensor(pos, requires_grad=True)
    actual = deformed_edge_attr_torch(t, torch.tensor(edges), periodic_box=box)
    np.testing.assert_allclose(actual.detach().numpy(), expected[:, :4], atol=1e-7)
    actual.sum().backward()
    assert torch.isfinite(t.grad).all()


@pytest.mark.parametrize('box', [[1, 0], [-1, 0, 0], [float('nan'), 0, 0]])
def test_invalid_periodic_box(box):
    with pytest.raises(ValueError, match='periodic_box'):
        minimum_image(np.zeros((1, 3)), box)
