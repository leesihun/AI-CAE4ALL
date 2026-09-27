"""Grid-coverage radius arithmetic (IMPLEMENTATION_PLAN.md section 8.4).

No operator core builds radius graphs; `min_reachable_radius` feeds the
grid-coverage report in misc/audit_input_identifiability.py.
"""

import numpy as np


def min_reachable_radius(resolution, dim: int) -> float:
    """Half the diagonal of one grid cell in [0,1]^d -- the smallest radius
    that provably reaches every grid point from its nearest neighbors
    (section 8.4's coverage arithmetic)."""
    cell_sizes = [1.0 / max(r - 1, 1) for r in resolution]
    diag = float(np.sqrt(sum(c ** 2 for c in cell_sizes[:dim])))
    return diag / 2.0
