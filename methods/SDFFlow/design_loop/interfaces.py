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
constants). A generated surface reaches the rule through marching cubes and a
5000-node clustering, so serving uses `SERVE_TOL` instead -- and only after
`register_to_interfaces` has put the shape in its own frame: the fixed
population map alone left ~1/3 of ex1 samples scaled 1-3% and shifted ~1 mm,
which the gates refused as missing bores.

The gates keep the labelling run's meaning -- a shape whose bores do not close
(coverage) or do not run the wall height (span) is not a bracket the labels
describe, and is refused rather than predicted. `SERVE_GATES` are set so all
424 held-out ex13 graphs pass (their minima: 17 nodes, 301 deg, span 0.74); the
labelling run's own gates (12 nodes, 300 deg, 0.8) refuse 5 of them. Serving
(`deepjeb_bridge.mesh_to_ver_record`) judges them on the full marching-cubes
surface, not the 5000-node graph: the graph's ~3.5 mm edges can leave one ring
of nodes inside a 6.5 mm ear band, which refused 4 of 64 ex1 samples whose
full surface has the whole bore (64/64 pass there: span >= 0.74, coverage
>= 334 deg). The node types are the same either way -- `decimate` keeps
original vertices -- so only the refusal moved.
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


# Frame registration of a generated surface (see `register_to_interfaces`).
REGISTER_MIN_POINTS = 12        # wall points a bore fit needs
REGISTER_MIN_BOLTS = 3          # bores that must fit for the similarity to be determined
REGISTER_MAX_RESIDUAL = 0.5     # mm, bore centre left over after the similarity
REGISTER_MAX_RADIUS_ERR = 1.0   # mm, fitted bore radius vs the canonical one
REGISTER_MAX_LUG_DX = 1.0       # mm, lug axis x after registration (not fitted, checked)
# mm past the bore radius the first pass searches. Inside the boss (outer R
# >= 13.8 bolt, ~18.7 lug) the only inward-facing wall is the bore, so this
# admits a ~6 mm frame offset (ex1 samples: <= ~3 mm) without other surfaces.
REGISTER_WIDE = 6.0


def vertex_normals(vertices, faces):
    """Area-weighted unit vertex normals of a triangle surface."""
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _circle_fit(u, v):
    """Algebraic least-squares circle: (centre u, centre v, radius)."""
    A = np.c_[2 * u, 2 * v, np.ones_like(u)]
    (a, b, c), *_ = np.linalg.lstsq(A, u ** 2 + v ** 2, rcond=None)
    return float(a), float(b), float(np.sqrt(max(c + a * a + b * b, 0.0)))


def _bore_wall(du, dv, nu, nv, rho, mask):
    """Points of `mask` whose normal points at the bore axis (the wall faces inward)."""
    return mask & ((nu * du + nv * dv) / np.maximum(rho, 1e-9) < -0.5)


def register_to_interfaces(X, normals):
    """Similarity (isotropic scale + shift) carrying a surface onto the canonical interfaces.

    `X` is a generated surface already put in the DeepJEB frame by the fixed
    population map (`deepjeb_bridge.normalized_to_millimetres`). That map is
    right on average only: `normalize_mesh` scales each bracket's own longest
    side to the same length, and the brackets' true lengths differ, so a
    generated shape comes out scaled 0.97-1.01 and shifted up to ~3 mm. The
    interfaces are the one part every DeepJEB bracket shares exactly (boolean-
    merged, centre spread <= 0.005 mm), so they fix the frame: the four bolt
    bores (axis z) give the scale and the x-y shift, the lug bore (axis y) gives
    the z shift. Measured on 16 ex1 SDFFlow samples: the four fitted bore
    centres match the canonical ones to <= 0.03 mm after the fit, and the
    serving gate then passes all 16 (it refused 5 in the population frame).

    Two passes: a wide search window around each canonical axis, then a tight
    one around the bore radius in the registered frame.

    Returns (X_registered, frame) with frame = {scale, shift, ok, reason,
    bolt_residual, bolt_radius, lug_dx, lug_radius, bolts_fitted};
    X_registered = scale * X + shift. On failure X is returned unchanged with
    `ok` False and the reason -- the gates downstream then decide.
    """
    X = np.asarray(X, float)
    N = np.asarray(normals, float)
    frame = {'scale': 1.0, 'shift': [0.0, 0.0, 0.0], 'ok': False, 'reason': None}
    scale, shift = 1.0, np.zeros(3)
    for window in ('wide', 'tight'):
        W = scale * X + shift
        src, dst, radii = [], [], []
        for cx, cy in BOLTS.values():
            du, dv = W[:, 0] - cx, W[:, 1] - cy
            rho = np.hypot(du, dv)
            near = (rho < R_BOLT + REGISTER_WIDE) if window == 'wide' else (np.abs(rho - R_BOLT) < 1.0)
            m = near & (W[:, 2] > BOLT_Z[0] + 1.0) & (W[:, 2] < BOLT_Z[1] - 1.0) \
                & (np.abs(N[:, 2]) < 0.5)
            m = _bore_wall(du, dv, N[:, 0], N[:, 1], rho, m)
            if m.sum() < REGISTER_MIN_POINTS:
                continue
            a, b, r = _circle_fit(W[m, 0], W[m, 1])
            # Coverage about the fitted axis: seen from the canonical one, a
            # bore a few mm off subtends far less than its true arc.
            if _coverage(W[m, 0] - a, W[m, 1] - b) < 180.0:
                continue
            # Fit in X coordinates: W = scale * X + shift, so map the centre back.
            src.append(((a - shift[0]) / scale, (b - shift[1]) / scale))
            dst.append((cx, cy))
            radii.append(r)
        frame['bolts_fitted'] = len(src)
        if len(src) < REGISTER_MIN_BOLTS:
            frame['reason'] = (f'only {len(src)} bolt bore(s) found ({window} pass); '
                               f'{REGISTER_MIN_BOLTS} fix the frame')
            return X, frame
        S, D = np.asarray(src), np.asarray(dst)
        sc, dc = S.mean(0), D.mean(0)
        scale = float(np.sqrt(((D - dc) ** 2).sum() / ((S - sc) ** 2).sum()))
        shift = np.array([dc[0] - scale * sc[0], dc[1] - scale * sc[1], shift[2]])

        # z shift from the lug bore: circle in x-z over the ear walls.
        W = scale * X + shift
        du, dv = W[:, 0] - LUG_XZ[0], W[:, 2] - LUG_XZ[1]
        rho = np.hypot(du, dv)
        ears = np.zeros(len(W), bool)
        for y0, y1 in EARS.values():
            ears |= (W[:, 1] > y0 + 0.5) & (W[:, 1] < y1 - 0.5)
        near = (rho < R_LUG + REGISTER_WIDE) if window == 'wide' else (np.abs(rho - R_LUG) < 1.0)
        m = _bore_wall(du, dv, N[:, 0], N[:, 2], rho, near & ears & (np.abs(N[:, 1]) < 0.5))
        lx = lz = lr = None
        if m.sum() >= REGISTER_MIN_POINTS:
            lx, lz, lr = _circle_fit(W[m, 0], W[m, 2])
        if lx is None or _coverage(W[m, 0] - lx, W[m, 2] - lz) < 180.0:
            frame['reason'] = f'lug bore not found ({window} pass)'
            return X, frame
        shift[2] += LUG_XZ[1] - lz
        lug_dx = lx - LUG_XZ[0]

    residual = float(np.abs(scale * (S - sc) + dc - D).max())
    frame.update(scale=scale, shift=shift.tolist(), bolt_residual=residual,
                 bolt_radius=[float(r) for r in radii], lug_dx=float(lug_dx),
                 lug_radius=float(lr))
    worst_r = max(abs(r - R_BOLT) for r in radii)
    if residual > REGISTER_MAX_RESIDUAL:
        frame['reason'] = f'bolt bores off the canonical pattern by {residual:.2f} mm'
    elif worst_r > REGISTER_MAX_RADIUS_ERR:
        frame['reason'] = f'bolt bore radius off by {worst_r:.2f} mm'
    elif abs(lug_dx) > REGISTER_MAX_LUG_DX:
        frame['reason'] = f'lug bore axis off by {lug_dx:+.2f} mm in x'
    if frame['reason']:
        return X, frame
    frame['ok'] = True
    return scale * X + shift, frame


def node_types(X, tol=SERVE_TOL, gates=None):
    """node_type row for a surface graph in the DeepJEB mm frame (raises InterfaceError)."""
    sel, _ = select_interface_nodes(X, np.arange(len(X)), tol, **(gates or SERVE_GATES))
    types = np.full(len(X), NODE_FREE, dtype=np.int8)
    for name, ids in sel.items():
        types[ids] = NODE_BOLT if name in BOLTS else NODE_LUG
    return types
