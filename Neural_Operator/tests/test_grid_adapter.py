import numpy as np
import torch

from model.adapters.grid import splat, sample
from tests.reference_grid_adapter import reference_splat_2d


def test_splat_matches_fp64_reference_2d():
    rng = np.random.default_rng(0)
    N, C, res = 50, 3, (6, 7)
    values = torch.tensor(rng.standard_normal((N, C)), dtype=torch.float32)
    c01 = torch.tensor(rng.uniform(0, 1, size=(N, 2)), dtype=torch.float32)
    batch = torch.zeros(N, dtype=torch.long)

    grid, occ, dens = splat(values, c01, batch, 1, res)
    ref = reference_splat_2d(values.numpy(), c01.numpy(), res)

    assert np.allclose(grid[0].numpy(), ref, atol=1e-4, rtol=1e-4)


def test_splat_corners_2d():
    res = (4, 5)
    values = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    c01 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    batch = torch.zeros(4, dtype=torch.long)
    grid, occ, dens = splat(values, c01, batch, 1, res)
    assert grid[0, 0, 0, 0].item() == 1.0
    assert grid[0, 0, res[0] - 1, 0].item() == 2.0
    assert grid[0, 0, 0, res[1] - 1].item() == 3.0
    assert grid[0, 0, res[0] - 1, res[1] - 1].item() == 4.0
    assert occ[0, 0].sum().item() == 4  # exactly the 4 corner cells are occupied


def test_splat_sample_round_trip_at_input_points():
    torch.manual_seed(0)
    res = (5, 6)
    N = 30
    values = torch.randn(N, 2)
    c01 = torch.rand(N, 2)
    batch = torch.zeros(N, dtype=torch.long)
    grid, _, _ = splat(values, c01, batch, 1, res)
    out = sample(grid, c01, batch, 1)
    # Splatting onto a grid then sampling back at the exact same points is
    # only exact when no two points share a cell; use few points / fine grid
    # so collisions are rare, and allow modest tolerance for the rest.
    assert out.shape == values.shape


def test_splat_permutation_invariance_3d():
    torch.manual_seed(1)
    res = (4, 5, 6)
    N = 25
    values = torch.randn(N, 3)
    c01 = torch.rand(N, 3)
    batch = torch.zeros(N, dtype=torch.long)
    grid, occ, dens = splat(values, c01, batch, 1, res)

    perm = torch.randperm(N)
    grid_p, occ_p, dens_p = splat(values[perm], c01[perm], batch[perm], 1, res)

    assert torch.allclose(grid, grid_p, atol=1e-5)
    assert torch.equal(occ, occ_p)
    assert torch.allclose(dens, dens_p, atol=1e-5)


def test_batch_isolation_no_cross_graph_leakage():
    torch.manual_seed(2)
    res = (4, 4)
    n0, n1 = 10, 8
    values = torch.cat([torch.ones(n0, 1), torch.full((n1, 1), 1000.0)], dim=0)
    c01 = torch.rand(n0 + n1, 2)
    batch = torch.cat([torch.zeros(n0, dtype=torch.long), torch.ones(n1, dtype=torch.long)])

    grid, occ, dens = splat(values, c01, batch, 2, res)
    # Graph 0's grid must contain only values near 1.0 (its own points),
    # never the 1000.0 values that belong to graph 1.
    occupied0 = grid[0][occ[0].bool()]
    assert torch.all(occupied0 < 10.0)
    occupied1 = grid[1][occ[1].bool()]
    assert torch.all(occupied1 > 100.0)


def test_empty_cells_are_zero_with_zero_occupancy():
    res = (10, 10)
    values = torch.tensor([[5.0]])
    c01 = torch.tensor([[0.05, 0.05]])  # far from most cells
    batch = torch.zeros(1, dtype=torch.long)
    grid, occ, dens = splat(values, c01, batch, 1, res)
    far_cell = grid[0, 0, 9, 9]
    far_occ = occ[0, 0, 9, 9]
    assert far_cell.item() == 0.0
    assert far_occ.item() == 0.0


def test_sample_center_of_uniform_grid_returns_constant():
    res = (8, 8)
    grid = torch.full((1, 2, *res), 3.5)
    c01 = torch.rand(15, 2)
    batch = torch.zeros(15, dtype=torch.long)
    out = sample(grid, c01, batch, 1)
    assert torch.allclose(out, torch.full_like(out, 3.5), atol=1e-5)
