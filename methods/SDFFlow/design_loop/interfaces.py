"""DeepJEB interface geometry and the geometric boundary-condition rule (vertical case).

The ex13 surrogate (`dataset/deterministic/ex13_deepjeb_ver.h5`) was labelled
with this rule: every DeepJEB bracket carries the same boolean-merged interface
features, measured on all 421 FieldMesh brackets (mm, DeepJEB frame):

  bolt bores 1-4 (axis z): r = 5.080, wall z = 0.300 .. 9.967, centre offset <= 0.005 mm
  lug bores (axis y) at (x, z) = (-19.88, 46.72): r = 9.53,
                           ear walls y = [-90.6, -84.1] and [-62.0, -55.5]

Vertical load case (arXiv 2406.09047): 8000 lbf = 35 586 N along +z at the lug
pin centre; the bolt bore walls are clamped (RBE2 to a fixed master), the lug
ear bore walls carry the load (RBE3). The labels' node_type row is 0 free,
1 bolt bore wall, 2 lug ear bore wall.

The labelling run applied the rule at tol 0.05 mm on the exact CAD surfaces
(D:/CAE_datasets_raw/deepjeb_ver_fea/src/interfaces.py, the source of these
constants). A generated surface reaches the rule through marching cubes, which
places a bore wall 0.06-0.4 mm off its radius (measured on mc_resolution 128
samples of the ex1 SDFFlow model), so serving uses `SERVE_TOL` instead.

The gates keep the labelling run's meaning -- a shape whose bores do not close
(coverage) or do not run the wall height (span) is not a bracket the labels
describe, and is refused rather than predicted -- but are applied to the
5000-node graph, whose ~3.5 mm edges cut a 6.5 mm lug ear wall to as little as
74 % of its span on real brackets. `SERVE_GATES` are set so all 424 held-out
ex13 graphs pass (their minima: 17 nodes, 301 deg, span 0.74); the labelling
run's own gates (12 nodes, 300 deg, 0.8) refuse 5 of them.
"""

import numpy as np

BOLTS = {'bolt1': (53.88, 4.49), 'bolt2': (1.30, 3.04),
         'bolt3': (1.26, -147.54), 'bolt4': (40.36, -146.70)}
R_BOLT = 5.08
BOLT_Z = (0.300, 9.967)

LUG_XZ = (-19.88, 46.72)
R_LUG = 9.53
EARS = {'ear_a': (-90.6, -84.1), 'ear_b': (-62.0, -55.5)}
LUG_REF = (-19.88, -73.05, 46.72)

SERVE_TOL = 0.6          # mm, generated surfaces; see the module docstring
SERVE_GATES = dict(min_nodes=12, min_cov=290.0, min_span=0.6)
NODE_FREE, NODE_BOLT, NODE_LUG = 0, 1, 2


class InterfaceError(RuntimeError):
    pass


def _coverage(u, v):
    """Angular coverage in degrees: 360 minus the largest gap."""
    if len(u) < 3:
        return 0.0
    ang = np.sort(np.degrees(np.arctan2(v, u)) % 360)
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    return float(360 - gaps.max())


def select_interface_nodes(X, candidates, tol, *, min_nodes=12, min_cov=300.0,
                           min_span=0.8, raise_on_fail=True):
    """Return ({name: node ids}, report). X: (N, 3) mm; candidates: boundary node ids."""
    X = np.asarray(X, float)
    cand = np.asarray(candidates)
    P = X[cand]
    sel, rep, bad = {}, {}, []
    for name, (cx, cy) in BOLTS.items():
        du, dv = P[:, 0] - cx, P[:, 1] - cy
        rho = np.hypot(du, dv)
        m = (np.abs(rho - R_BOLT) < tol) & (P[:, 2] > BOLT_Z[0] - tol) & (P[:, 2] < BOLT_Z[1] + tol)
        span = float(np.ptp(P[m, 2])) if m.any() else 0.0
        cov = _coverage(du[m], dv[m])
        rep[name] = dict(n=int(m.sum()), cov=cov, span=span, span_frac=span / (BOLT_Z[1] - BOLT_Z[0]))
        if m.sum() < min_nodes or cov < min_cov or span < min_span * (BOLT_Z[1] - BOLT_Z[0]):
            bad.append(name)
        sel[name] = cand[m]
    for name, (y0, y1) in EARS.items():
        du, dv = P[:, 0] - LUG_XZ[0], P[:, 2] - LUG_XZ[1]
        rho = np.hypot(du, dv)
        m = (np.abs(rho - R_LUG) < tol) & (P[:, 1] > y0 - tol) & (P[:, 1] < y1 + tol)
        span = float(np.ptp(P[m, 1])) if m.any() else 0.0
        cov = _coverage(du[m], dv[m])
        rep[name] = dict(n=int(m.sum()), cov=cov, span=span, span_frac=span / (y1 - y0))
        if m.sum() < min_nodes or cov < min_cov or span < min_span * (y1 - y0):
            bad.append(name)
        sel[name] = cand[m]
    rep['failed'] = bad
    if bad and raise_on_fail:
        raise InterfaceError(f'interface gate failed: {bad} {[rep[b] for b in bad]}')
    return sel, rep


def node_types(X, tol=SERVE_TOL, gates=None):
    """node_type row for a surface graph in the DeepJEB mm frame (raises InterfaceError)."""
    sel, _ = select_interface_nodes(X, np.arange(len(X)), tol, **(gates or SERVE_GATES))
    types = np.full(len(X), NODE_FREE, dtype=np.int8)
    for name, ids in sel.items():
        types[ids] = NODE_BOLT if name in BOLTS else NODE_LUG
    return types
