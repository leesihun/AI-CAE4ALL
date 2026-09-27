import numpy as np

from model.adapters.radius_neighbors import min_reachable_radius


def test_min_reachable_radius_matches_half_cell_diagonal():
    r = min_reachable_radius((3, 3), dim=2)
    # cell size = 1/(3-1) = 0.5 per axis; diagonal = sqrt(0.5^2+0.5^2); half of that
    expected = np.sqrt(0.5 ** 2 + 0.5 ** 2) / 2.0
    assert abs(r - expected) < 1e-9
