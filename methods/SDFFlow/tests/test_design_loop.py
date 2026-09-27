"""Verification tests for the `optimize` mode's meshing and FEA.

The solver in `design_loop/fea.py` is written from scratch, so it is checked
against things with known answers rather than against plausibility: the
constant-strain patch test (exact to machine precision for a correct element),
rigid-body motion producing zero strain energy, and a cantilever converging
toward beam theory from the stiff side as tet4 must. The load table is pinned
to the GE challenge values in SI, because DeepJEB's labels were produced with
them and the solver works in N, m, Pa.

Run from `methods/SDFFlow`:  python -m pytest -q tests/test_design_loop.py
"""

import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_loop import fea                                          # noqa: E402
from design_loop.mesher import tet_mesh_from_surface                 # noqa: E402
from design_loop.problem import LBF_TO_N, LBIN_TO_NM, Bracket, MassObjective  # noqa: E402


def box_mesh(extents, divisions):
    """Structured tetrahedralization of a box via Kuhn's 6-tet decomposition.

    Deliberately gmsh-free: these tests verify the solver, and a hand-built
    conforming mesh keeps a mesher failure from masquerading as an FEA failure.
    (gmsh is exercised separately in `test_mesher_fills_a_dense_surface`, on the
    kind of dense closed surface the production path actually feeds it -- a
    subdivided box trips `PLC Error: A segment and a facet intersect`.)
    """
    nx, ny, nz = divisions
    ex, ey, ez = extents
    xs, ys, zs = (np.linspace(0, ex, nx + 1), np.linspace(0, ey, ny + 1),
                  np.linspace(0, ez, nz + 1))
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1)
    nodes = grid.reshape(-1, 3)

    def node_id(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    # Kuhn: one tet per permutation of the axes, sharing the cube's main diagonal.
    corners = [((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
               ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)),
               ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    tets = []
    for first, second in corners:
        tets.append(np.stack([
            node_id(i, j, k),
            node_id(i + first[0], j + first[1], k + first[2]),
            node_id(i + second[0], j + second[1], k + second[2]),
            node_id(i + 1, j + 1, k + 1)], axis=1))
    return nodes, np.concatenate(tets, axis=0)


@pytest.fixture(scope='module')
def patch_box():
    return box_mesh((1.0, 1.0, 1.0), (4, 4, 4))


def test_patch_test_reproduces_a_linear_displacement_field(patch_box):
    """Prescribing u = Ax on the boundary must give exactly u = Ax inside."""
    nodes, tets = patch_box
    material = fea.Material(E=200e9, nu=0.3)
    K, B, vol, tets = fea.assemble(nodes, tets, material)

    A = np.array([[1.0e-4, 2.0e-5, -3.0e-5],
                  [5.0e-6, -2.0e-4, 1.0e-5],
                  [-4.0e-5, 7.0e-6, 3.0e-4]])
    offset = np.array([1e-3, -2e-3, 5e-4])
    exact = nodes @ A.T + offset

    boundary = np.unique(fea.boundary_faces(tets))
    prescribed = (boundary[:, None] * 3 + np.arange(3)[None, :]).ravel()
    ndof = nodes.shape[0] * 3
    free = np.setdiff1d(np.arange(ndof), prescribed)
    assert free.size > 0, 'mesh has no interior nodes; refine it'

    u = np.zeros(ndof)
    u[prescribed] = exact.ravel()[prescribed]
    rhs = -(K[free][:, prescribed] @ u[prescribed])
    u[free] = np.linalg.solve(K[free][:, free].toarray(), rhs)

    error = np.abs(u - exact.ravel()).max() / np.abs(exact).max()
    assert error < 1e-9, f'patch test displacement error {error:.2e}'

    strain = 0.5 * (A + A.T)
    expected = material.constitutive() @ np.array([
        strain[0, 0], strain[1, 1], strain[2, 2],
        2 * strain[0, 1], 2 * strain[1, 2], 2 * strain[0, 2]])
    stress, _ = fea.element_stress(B, material.constitutive(), u.reshape(-1, 3), tets)
    stress_error = np.abs(stress - expected[None, :]).max() / np.abs(expected).max()
    assert stress_error < 1e-8, f'patch test stress error {stress_error:.2e}'


def test_rigid_body_motion_stores_no_energy(patch_box):
    """Every rigid-body mode must map to zero strain energy."""
    nodes, tets = patch_box
    material = fea.Material()
    K, _, _, _ = fea.assemble(nodes, tets, material)
    modes = fea.rigid_body_modes(nodes)
    scale = float(np.abs(K.diagonal()).max())
    for i in range(modes.shape[1]):
        mode = modes[:, i]
        energy = float(mode @ (K @ mode)) / (scale * float(mode @ mode))
        assert abs(energy) < 1e-12, f'rigid body mode {i} stores energy {energy:.2e}'


def test_cantilever_converges_toward_beam_theory_from_the_stiff_side():
    """Tet4 under-predicts deflection and must improve as the mesh refines."""
    L, b, h = 0.20, 0.02, 0.02
    E, nu, P = 200e9, 0.3, 100.0
    inertia = b * h ** 3 / 12.0
    shear_modulus = E / (2.0 * (1.0 + nu))
    analytic = P * L ** 3 / (3.0 * E * inertia) \
        + P * L / ((5.0 / 6.0) * shear_modulus * b * h)              # Timoshenko

    material = fea.Material(E=E, nu=nu)
    ratios = []
    for divisions in ((2, 12, 2), (3, 24, 3), (5, 40, 5)):
        nodes, tets = box_mesh((b, L, h), divisions)
        K, _, _, tets = fea.assemble(nodes, tets, material)
        faces_tri = fea.boundary_faces(tets)

        root = np.flatnonzero(nodes[:, 1] <= nodes[:, 1].min() + 1e-9)
        tip = np.flatnonzero(nodes[:, 1] >= nodes[:, 1].max() - 1e-9)
        assert len(root) >= 3 and len(tip) >= 3

        ndof = nodes.shape[0] * 3
        force = np.zeros(ndof)
        weights = fea.face_area_weights(nodes, faces_tri, tip)
        force[tip * 3 + 2] = weights[tip] * P

        fixed = (root[:, None] * 3 + np.arange(3)[None, :]).ravel()
        u, _ = fea.LinearSolver(K, ndof, fixed, nodes).solve(force)
        ratios.append(float(np.abs(u[tip, 2]).mean() / analytic))

    assert all(0.0 < r < 1.05 for r in ratios), f'tet4 should not exceed beam theory: {ratios}'
    assert ratios[-1] > ratios[0], f'refinement must reduce the tet4 stiffness bias: {ratios}'
    assert ratios[-1] > 0.55, f'finest mesh still far too stiff: {ratios}'


def test_interface_detection_rejects_geometry_without_both_pads():
    """A shape with no mounting pads must fail loudly, not analyze as a cantilever."""
    nodes, tets = box_mesh((0.4, 0.4, 0.4), (3, 3, 3))
    nodes = nodes - nodes.mean(axis=0)                               # centred, |y| < 0.6
    faces = fea.boundary_faces(tets)
    with pytest.raises(ValueError, match='mount|lug'):
        Bracket().interfaces(nodes, faces)


def test_load_cases_are_the_ge_challenge_loads_in_si():
    """Magnitudes must be the challenge's imperial loads converted to N / N*m.

    DeepJEB's FEA labels were produced with the SI values of the GE bracket
    challenge (arXiv 2406.09047: 35.6 / 37.8 / 42.3 kN and 565 N*m). The FEA
    path works in N, m, Pa, so the bare imperial numbers this table used to
    hold (8000 / 8500 / 9500 read as N, 5000 read as N*m) understated the
    forces 4.448x and overstated the torsion 8.85x relative to one another.
    """
    assert LBF_TO_N == 4.4482216152605
    assert LBIN_TO_NM == 0.1129848290276167
    assert np.isclose(LBIN_TO_NM, LBF_TO_N * 0.0254, rtol=0.0, atol=1e-15)

    cases = Bracket.LOAD_CASES
    magnitude = {name: float(np.linalg.norm(cases[name]['vector']))
                 for name in ('vertical', 'horizontal', 'diagonal')}
    assert np.isclose(magnitude['vertical'], 8000.0 * LBF_TO_N, rtol=1e-12)
    assert np.isclose(magnitude['horizontal'], 8500.0 * LBF_TO_N, rtol=1e-12)
    assert np.isclose(magnitude['diagonal'], 9500.0 * LBF_TO_N, rtol=1e-12)
    assert np.isclose(cases['torsion']['magnitude'], 5000.0 * LBIN_TO_NM, rtol=1e-12)

    # The paper's figures are these conversions rounded to three significant digits.
    assert round(magnitude['vertical'] / 1e3, 1) == 35.6
    assert round(magnitude['horizontal'] / 1e3, 1) == 37.8
    assert round(magnitude['diagonal'] / 1e3, 1) == 42.3
    assert round(cases['torsion']['magnitude']) == 565

    # A bare imperial number would sit an order of magnitude off on either side.
    assert all(m > 3.0e4 for m in magnitude.values()), magnitude
    assert 5.0e2 < cases['torsion']['magnitude'] < 6.0e2

    # Axes are this repo's occupancy-derived frame and must not move: vertical
    # +z, horizontal +y, torsion about y, and the diagonal in the y-z plane with
    # its historical (cos 42 along y, sin 42 along z) decomposition.
    unit = {name: np.asarray(cases[name]['vector']) / magnitude[name] for name in magnitude}
    assert np.allclose(unit['vertical'], (0.0, 0.0, 1.0), atol=1e-12)
    assert np.allclose(unit['horizontal'], (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(unit['diagonal'],
                       (0.0, np.cos(np.deg2rad(42.0)), np.sin(np.deg2rad(42.0))), atol=1e-12)
    assert np.allclose(cases['torsion']['axis'], (0.0, 1.0, 0.0), atol=1e-12)


def test_si_conversion_is_a_pure_rescaling_of_the_legacy_loads():
    """Linear statics: the SI fix must scale the vertical case by exactly LBF_TO_N.

    Guards that the unit conversion is the *only* change to load assembly --
    same interface nodes, same area weights, same direction -- so the ratio
    between stresses recorded before and after the change is the conversion
    factor and nothing else (compliance f.u scales by its square).
    """
    class LegacyBracket(Bracket):
        LOAD_CASES = dict(Bracket.LOAD_CASES,
                          vertical=dict(kind='force', vector=(0.0, 0.0, 8000.0)))

    nodes, tets = box_mesh((1.0, 1.8, 0.6), (6, 12, 4))
    nodes = nodes - nodes.mean(axis=0)
    nodes[:, 1] *= 1.8 / (nodes[:, 1].max() - nodes[:, 1].min())

    new = Bracket(load_cases=('vertical',)).analyze(nodes, tets)['cases']['vertical']
    old = LegacyBracket(load_cases=('vertical',)).analyze(nodes, tets)['cases']['vertical']
    for key in ('max_von_mises', 'peak_von_mises', 'max_displacement'):
        ratio = new[key] / old[key]
        assert np.isclose(ratio, LBF_TO_N, rtol=1e-8), f'{key}: ratio {ratio} != {LBF_TO_N}'
    ratio = new['compliance'] / old['compliance']
    assert np.isclose(ratio, LBF_TO_N ** 2, rtol=1e-8), f'compliance: ratio {ratio}'


def test_load_cases_apply_the_requested_resultant():
    """Force load cases must sum to the specified SI vector on the lug."""
    nodes, tets = box_mesh((1.0, 1.8, 0.6), (6, 12, 4))
    nodes = nodes - nodes.mean(axis=0)
    nodes[:, 1] *= 1.8 / (nodes[:, 1].max() - nodes[:, 1].min())
    faces = fea.boundary_faces(tets)
    bracket = Bracket()
    mount, lug = bracket.interfaces(nodes, faces)
    scaled = nodes * bracket.length_scale

    for name in ('vertical', 'horizontal', 'diagonal'):
        f = bracket._load_vector(name, scaled, faces, lug, nodes.shape[0] * 3)
        resultant = f.reshape(-1, 3).sum(axis=0)
        expected = np.asarray(Bracket.LOAD_CASES[name]['vector'])
        assert np.allclose(resultant, expected, rtol=1e-9, atol=1e-6), \
            f'{name}: resultant {resultant} != {expected}'

    f = bracket._load_vector('torsion', scaled, faces, lug, nodes.shape[0] * 3)
    forces = f.reshape(-1, 3)
    moment = np.cross(scaled, forces).sum(axis=0)
    assert abs(moment[1] - Bracket.LOAD_CASES['torsion']['magnitude']) \
        / Bracket.LOAD_CASES['torsion']['magnitude'] < 1e-6, f'torsion moment {moment}'


def test_vertical_displacement_is_uz_of_the_vertical_case():
    """The vertical requirement reads |u_z| under the +z load, not the worst |u|.

    A diagonal load deflects more in total than the vertical one; if the
    top-level value were a max over cases (like max_displacement) the vertical
    limit would silently constrain the wrong load.
    """
    nodes, tets = box_mesh((1.0, 1.8, 0.6), (6, 12, 4))
    nodes = nodes - nodes.mean(axis=0)
    nodes[:, 1] *= 1.8 / (nodes[:, 1].max() - nodes[:, 1].min())
    both = Bracket(load_cases=('vertical', 'diagonal'))
    result = both.analyze(nodes, tets, return_fields=True)
    u = result['fields']['cases']['vertical']['displacement']
    assert np.isclose(result['vertical_displacement'], np.abs(u[:, 2]).max(), rtol=1e-12)
    vert = result['cases']['vertical']
    assert vert['max_vertical_displacement'] <= vert['max_displacement'] * (1 + 1e-12)
    assert result['vertical_displacement'] == vert['max_vertical_displacement']
    no_vertical = Bracket(load_cases=('diagonal',)).analyze(nodes, tets)
    assert no_vertical['vertical_displacement'] is None


def test_vertical_limit_replaces_the_calibrated_deflection_allowable():
    result = {'mass': 1.0, 'peak_von_mises': 1.0, 'max_displacement': 9.0,
              'vertical_displacement': 0.5e-3}
    calibrated = MassObjective(1.0, 2.0, 1.0)
    assert not calibrated(result)[1]['feasible']        # |u| 9 > allow 1
    limited = MassObjective(1.0, 2.0, 1.0, vertical_disp_allow=1e-3)
    score, penalty = limited(result)
    assert penalty['feasible'] and score == 1.0         # 0.5 mm <= 1 mm; |u| ignored
    tight = MassObjective(1.0, 2.0, 1.0, vertical_disp_allow=0.25e-3)
    score, penalty = tight(result)
    assert np.isclose(penalty['disp_violation'], 1.0) and np.isclose(score, 1.0 + 3.0)
    with pytest.raises(KeyError):
        limited({**result, 'vertical_displacement': None})
    # 0 / None mean "off", matching opt_vertical_disp_max 0.
    assert MassObjective(1.0, 2.0, 1.0, vertical_disp_allow=0).vertical_disp_allow is None


def test_structured_mesh_is_conforming_and_fills_the_box():
    """Guard the test fixture itself: volumes must sum to the box exactly."""
    extents = (1.0, 1.8, 0.6)
    nodes, tets = box_mesh(extents, (4, 6, 3))
    _, vol = fea.element_gradients(nodes, tets)
    assert abs(np.abs(vol).sum() - np.prod(extents)) / np.prod(extents) < 1e-12
    faces = fea.boundary_faces(tets)
    assert len(faces) == 2 * 2 * (4 * 6 + 6 * 3 + 4 * 3)   # two triangles per quad facet


def test_mesher_fills_a_dense_surface():
    """gmsh path: a dense closed surface must tetrahedralize with no inverted elements."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    nodes, tets, info = tet_mesh_from_surface(sphere, mesh_size_max=0.08, target_faces=4000)
    assert info['negative_jacobians'] == 0
    assert info['num_tets'] > 1000
    _, vol = fea.element_gradients(nodes, tets)
    sphere_volume = 4.0 / 3.0 * np.pi * 0.5 ** 3
    assert abs(np.abs(vol).sum() - sphere_volume) / sphere_volume < 0.02


def test_gentler_decimation_runs_only_after_the_labels_ladder_fails(monkeypatch):
    """The aggression fallback must never change a shape the labels could mesh.

    Regression: a surrogate-optimized bracket failed FEA verification at every
    face budget because default decimation pinched a thin member; the fallback
    walks the ladder again gently, but only once the default ladder is spent.
    """
    from design_loop import mesher

    calls = []

    def fake(mesh, mesh_size_max=0.05, target_faces=12000, aggression=None, **kwargs):
        calls.append((target_faces, aggression))
        if aggression is None:
            raise RuntimeError('Invalid boundary mesh (overlapping facets)')
        return 'nodes', 'tets', {}

    monkeypatch.setattr(mesher, 'tet_mesh_from_surface', fake)
    _, _, info = mesher.tet_mesh_with_retries(None, target_faces=1000)
    ladder = [int(1000 * f) for f in mesher.FACE_BUDGET_LADDER]
    assert calls == [(b, None) for b in ladder] + [(ladder[0], 3)]
    assert info['decimation_aggression'] == 3 and info['face_budget'] == ladder[0]
    assert info['face_budget_attempts'] == len(ladder) + 1

    calls.clear()
    monkeypatch.setattr(mesher, 'tet_mesh_from_surface',
                        lambda mesh, aggression=None, **kw: calls.append(aggression) or (0, 0, {}))
    _, _, info = mesher.tet_mesh_with_retries(None)
    assert calls == [None] and info['decimation_aggression'] is None

    monkeypatch.setattr(mesher, 'tet_mesh_from_surface',
                        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('nope')))
    with pytest.raises(mesher.MeshingError, match=r'\(aggression 3\): nope'):
        mesher.tet_mesh_with_retries(None)


_SUITE = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
VAE_PATH = os.path.join(_SUITE, 'output', 'geometry_generation', 'ex1', 'sdfflow_vae.pth')
FM_PATH = os.path.join(_SUITE, 'output', 'geometry_generation', 'ex1', 'sdfflow_fm.pth')
requires_checkpoints = pytest.mark.skipif(
    not (os.path.exists(VAE_PATH) and os.path.exists(FM_PATH)),
    reason='trained SDFFlow checkpoints not present')


@pytest.fixture(scope='module')
def generator():
    from design_loop.generator import SDFFlowGenerator
    return SDFFlowGenerator(VAE_PATH, FM_PATH, device='cpu', subspace_dim=12)


@requires_checkpoints
def test_bounds_are_stable_and_inscribe_the_shell(generator):
    """`bounds()` is called from several places; it must not drift between calls.

    Regression: it used to take a `latent_range` argument and cache it on the
    instance, so a later no-argument call from the CMA-ES setup silently reset
    the configured range back to the default.
    """
    lo, hi = generator.bounds()
    again_lo, again_hi = generator.bounds()
    assert np.array_equal(lo, again_lo) and np.array_equal(hi, again_hi)

    d = generator.subspace_dim
    assert np.allclose(hi[:d], generator.shell_scale)
    corner = generator.latent_range * np.sqrt(d)
    assert np.isclose(corner, generator.shell_scale * np.sqrt(d))


@requires_checkpoints
def test_latent_range_override_is_honoured():
    from design_loop.generator import SDFFlowGenerator
    wide = SDFFlowGenerator(VAE_PATH, FM_PATH, device='cpu', subspace_dim=12,
                            latent_range=3.0)
    assert wide.latent_range == 3.0
    assert np.allclose(wide.bounds()[1][:wide.subspace_dim], 3.0)


# ---- Noise chart (checkpoint-free) ---------------------------------------- #
# The FM start noise is a pure function of the design vector, so the chart is
# tested directly at the DeepJEB latent size rather than behind a checkpoint.

DEEPJEB_LATENT_DIM = 512 * 32


def _chart(kind, dim=DEEPJEB_LATENT_DIM, k=12, **kwargs):
    from design_loop.generator import NoiseChart
    return NoiseChart(dim, k, subspace_seed=0, base_seed=0, kind=kind, **kwargs)


def test_gnomonic_noise_stays_standard_normal_at_any_design():
    """|z| stays ~sqrt(D) for every |x|: each design is a draw the FM prior could make."""
    import torch
    chart = _chart('gnomonic_v2')
    d = chart.latent_flat_dim
    rng = np.random.default_rng(0)
    for scale in (0.0, 0.1, 1.0, 4.33, 100.0):
        x = rng.standard_normal(chart.subspace_dim) * scale
        z = chart(x).squeeze(0)
        assert z.shape == (d,)
        # |z|^2 ~ chi^2_D: mean D, sd sqrt(2D) -> relative 6-sigma band ~0.066 at D=16384.
        assert abs(float(torch.linalg.norm(z)) / np.sqrt(d) - 1.0) < 0.04
        assert abs(float(z.std()) - 1.0) < 0.04


def test_gnomonic_origin_is_the_base_draw_and_nothing_is_clipped():
    import torch
    chart = _chart('gnomonic_v2')
    k = chart.subspace_dim
    base = torch.randn(chart.latent_flat_dim, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(chart(np.zeros(k)).squeeze(0), base)
    # v1 clipped everything past its shell onto one point per ray; here every
    # magnitude along a ray is a distinct noise, turning monotonically away from eps_0.
    direction = np.eye(k)[0]
    cosines = []
    for t in (0.5, 1.0, 4.0, 16.0):
        z = chart(direction * t).squeeze(0)
        cosines.append(float(torch.nn.functional.cosine_similarity(z, base, dim=0)))
    assert all(a > b for a, b in zip(cosines, cosines[1:]))
    # |x| = 1 is a 45-degree turn: cos = 1/sqrt(1 + |x|^2), up to the O(1/sqrt(D))
    # overlap of two independent Gaussian draws.
    assert abs(cosines[1] - 1 / np.sqrt(2)) < 0.03


def test_gnomonic_design_actually_moves_the_noise():
    """The v1 failure: at D = 16384 a full-range design moved ~3% of the norm."""
    import torch
    corner = np.full(12, 1.25)                       # the default box corner

    def moved(chart):
        step = chart(corner) - chart(np.zeros(12))
        return float(torch.linalg.norm(step)) / np.sqrt(chart.latent_flat_dim)

    # gnomonic: |z(x) - eps_0|^2 / D -> 2 - 2 / sqrt(1 + |x|^2) = 1.55 at |x| = 4.33.
    assert abs(moved(_chart('gnomonic_v2')) - np.sqrt(1.55)) < 0.05
    assert moved(_chart('subspace_v1')) < 0.05       # 4.33 / 128 = 0.034


def test_legacy_subspace_chart_still_clips_to_its_shell():
    """subspace_v1 is kept so old summaries replay with the chart they were searched in."""
    import torch
    chart = _chart('subspace_v1', dim=512, shell_scale=1.25)
    radius = chart.shell_scale * np.sqrt(chart.subspace_dim)
    rng = np.random.default_rng(0)
    for scale in (0.1, 1.0, 10.0):
        x = rng.standard_normal(chart.subspace_dim) * scale
        in_subspace = chart(x).squeeze(0) @ chart.basis.T
        assert float(torch.linalg.norm(in_subspace)) <= radius * (1 + 1e-5)
        if np.linalg.norm(x) <= radius:
            assert np.allclose(in_subspace.numpy(), x, atol=1e-4)


def test_unknown_noise_param_is_rejected():
    with pytest.raises(ValueError, match='noise_param'):
        _chart('gnomonic', dim=64)


# ---- Selection and bookkeeping --------------------------------------------- #

def _record(score, feasible, ok=True):
    return {'ok': ok, 'score': score, 'x': [score],
            'penalty': {'feasible': feasible}, 'fea': {'mass': score}}


def test_select_best_prefers_feasible_over_a_lower_infeasible_score():
    """The quadratic exterior penalty puts its optimum just past the limit."""
    from design_loop.loop import select_best
    slightly_over = _record(0.80, feasible=False)
    within = _record(0.85, feasible=True)
    worse_within = _record(0.95, feasible=True)
    assert select_best([slightly_over, worse_within, within]) is within
    # No feasible record: fall back to the solved minimum, then to anything.
    failed = {'ok': False, 'score': 10.0, 'penalty': {'feasible': False}}
    assert select_best([failed, slightly_over]) is slightly_over
    assert select_best([failed]) is failed
    # The exterior penalty is unbounded: a badly violating solve can score
    # above failure_score, and a shape that could not be analysed must still
    # never be the delivered design while any solve exists.
    far_over = _record(40.0, feasible=False)
    cheap_failure = {'ok': False, 'score': 1.0, 'penalty': {'feasible': False}}
    assert select_best([cheap_failure, far_over]) is far_over


def test_failures_rank_behind_every_solved_candidate():
    """What CMA-ES is told: a failure is worse than the worst solve, always."""
    from design_loop.loop import rank_failures_last
    records = [_record(0.9, True), {'ok': False}, _record(40.0, False), {'ok': False}]
    scores = rank_failures_last(records, failure_score=10.0)
    assert scores[0] == 0.9 and scores[2] == 40.0
    assert scores[1] == scores[3] == 41.0
    assert [r['score'] for r in records] == scores
    # With every solve under failure_score, failures keep failure_score.
    assert rank_failures_last([_record(0.9, True), {'ok': False}], 10.0) == [0.9, 10.0]
    assert rank_failures_last([{'ok': False}], 10.0) == [10.0]


def test_interface_detection_rejects_a_shape_with_no_lug_crown():
    """Both pads present, but no part reaches a lug's height above them.

    Over the 418 DeepJEB label brackets the crown sits 0.623..0.645 above the
    mount (normalized frame) and a pad's own top at most 0.304 above it, so a
    flat plate must be refused as having no loaded interface -- not loaded on
    its own top face as if that were the lug.
    """
    from design_loop.problem import LUG_MIN_HEIGHT, find_interfaces
    nodes, tets = box_mesh((1.0, 1.8, 0.2), (6, 12, 2))
    nodes = nodes - nodes.mean(axis=0)
    faces = fea.boundary_faces(tets)
    with pytest.raises(ValueError, match='lug crown only 0.200'):
        find_interfaces(nodes, faces)
    assert 0.304 < LUG_MIN_HEIGHT < 0.623

    tall, tall_tets = box_mesh((1.0, 1.8, 0.6), (6, 12, 4))
    tall = tall - tall.mean(axis=0)
    mount, lug = find_interfaces(tall, fea.boundary_faces(tall_tets))
    assert len(mount) >= 12 and len(lug) >= 6


def test_gpu_id_from_config():
    from design_loop.surrogate import gpu_id_from_config
    assert gpu_id_from_config({'gpu_ids': 3}) == 3
    assert gpu_id_from_config({'gpu_ids': '5'}) == 5
    assert gpu_id_from_config({'gpu_ids': [7, 1]}) == 7
    assert gpu_id_from_config({'gpu_ids': 'cpu'}) is None
    assert gpu_id_from_config({'gpu_ids': []}) is None
    assert gpu_id_from_config({}) is None


def test_surrogate_mass_matches_the_fea_frame_when_given_a_length_scale(tmp_path):
    """Surrogate and FEA runs must weigh the same shape the same."""
    from design_loop.surrogate import HIMGNSurrogate
    box = trimesh.creation.box(extents=(1.0, 1.8, 0.6))
    scale = 0.19 / 1.8
    sur = HIMGNSurrogate('infer.txt', 'model.pth', load_cases=('ver',),
                         workdir=str(tmp_path), density=4430.0, length_scale=scale)
    # Bracket.analyze's mass: tet volumes of the length_scale-scaled nodes x rho
    # (the solve itself needs bolt-hole interfaces a plain box does not have).
    nodes, tets = box_mesh((1.0, 1.8, 0.6), (2, 3, 1))
    _, vol = fea.element_gradients(nodes * Bracket(length_scale=scale).length_scale, tets)
    fea_mass = float(np.abs(vol).sum()) * fea.Material().rho
    assert abs(sur.mass_of(box) - fea_mass) / fea_mass < 1e-9
    legacy = HIMGNSurrogate('infer.txt', 'model.pth', load_cases=('ver',),
                            workdir=str(tmp_path), density=4430.0)
    assert legacy.mass_of(box) != pytest.approx(fea_mass, rel=1e-3)


def test_rerun_clears_the_previous_runs_outputs(tmp_path):
    from inference_profiles.optimize import _OWNED_OUTPUTS, _clear_owned_outputs
    for name in _OWNED_OUTPUTS:
        (tmp_path / name).write_text('stale')
    (tmp_path / 'notes.txt').write_text('keep')
    (tmp_path / 'surrogate_batches').mkdir()
    _clear_owned_outputs(str(tmp_path))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['notes.txt', 'surrogate_batches']


def test_fea_verdicts_check_the_absolute_vertical_limit():
    from design_loop.verify_with_fea import limit_verdicts
    limits = {'vertical_disp_allow': 2e-4, 'stress_allow': 300e6, 'disp_allow': 1e-3}
    results = {'optimized': {'vertical_displacement_mm': 0.25, 'peak_von_mises_MPa': 200.0,
                             'max_displacement_mm': 0.3},
               'baseline': {'vertical_displacement_mm': 0.15, 'peak_von_mises_MPa': 320.0,
                            'max_displacement_mm': 0.2}}
    v = limit_verdicts(limits, results, 'surrogate')
    by = {name: {c['limit']: c for c in checks} for name, checks in v.items()}
    assert by['optimized']['vertical max |u_z|']['met'] is False
    assert by['baseline']['vertical max |u_z|']['met'] is True
    # The vertical limit replaces the calibrated |u| allowable, as in the search.
    assert not any('max |u|' in k for k in by['optimized'])
    assert by['baseline']['peak von Mises (calibrated)']['met'] is False
    assert 'surrogate' in by['baseline']['peak von Mises (calibrated)']['note']


def test_verification_resolution_follows_what_the_run_recorded():
    """A surrogate run re-solves at its labels' resolution, an FEA run at its own."""
    from design_loop.verify_with_fea import solve_resolution
    surrogate = {'analysis_backend': 'surrogate',
                 'surrogate': {'label_resolution': {'target_faces': 9000,
                                                    'mesh_size_max': 0.06,
                                                    'target_nodes': 4000}},
                 'verification_settings': {'mc_resolution': 160, 'target_nodes': 5000}}
    res = solve_resolution(surrogate)
    assert (res['target_faces'], res['mesh_size_max'], res['target_nodes']) == (9000, 0.06, 4000)
    assert res['source'] == 'surrogate.label_resolution'
    # A summary written before label_resolution existed falls back to the label constants.
    old = solve_resolution({'analysis_backend': 'surrogate', 'surrogate': {}})
    assert (old['target_faces'], old['mesh_size_max']) == (12000, 0.05)
    assert 'predates' in old['source']
    fea_run = {'analysis_backend': 'fea',
               'verification_settings': {'mc_resolution': 160, 'target_faces': 30000,
                                         'mesh_size_max': 0.035}}
    res = solve_resolution(fea_run)
    assert (res['target_faces'], res['mesh_size_max']) == (30000, 0.035)
    assert res['source'] == 'verification_settings'


def test_fea_verdicts_judge_a_surrogate_allowable_on_the_label_statistic():
    """The calibrated stress allowable is read in the measure it was calibrated in,
    at the verification resolution; the vertical limit stays on the solver's u_z."""
    from design_loop.verify_with_fea import limit_verdicts
    limits = {'vertical_disp_allow': 2e-4, 'stress_allow': 300e6, 'disp_allow': 1e-3}
    transported = {'stress_allow_MPa': 330.0, 'disp_allow_mm': 1.1}
    rec = {'vertical_displacement_mm': 0.19, 'peak_von_mises_MPa': 360.0,
           'max_displacement_mm': 0.3,
           'label_surface': {'peak_von_mises_MPa': 310.0, 'max_displacement_mm': 0.29,
                             'vertical_displacement_mm': 0.18}}
    by = {c['limit']: c for c in limit_verdicts(limits, {'d': rec}, 'surrogate',
                                                transported)['d']}
    stress = by['peak von Mises (calibrated)']
    assert stress['value'] == 310.0 and stress['allow'] == 330.0 and stress['met'] is True
    assert stress['statistic'] == 'label_surface'
    vertical = by['vertical max |u_z|']
    assert vertical['value'] == 0.19 and vertical['allow'] == 0.2
    assert vertical['statistic'] == 'solver'
    # An FEA run's allowable was calibrated on the solver's own statistic.
    by = {c['limit']: c for c in limit_verdicts(limits, {'d': rec}, 'fea', transported)['d']}
    assert by['peak von Mises (calibrated)']['value'] == 360.0
    assert by['peak von Mises (calibrated)']['met'] is False


def test_surface_statistics_reduce_the_solve_as_the_labels_do():
    """Boundary nodes only, per case, then max over cases -- the surrogate's measure."""
    from design_loop.verify_with_fea import surface_statistics
    nodes, tets = box_mesh((1.0, 1.8, 0.6), (6, 12, 4))
    nodes = nodes - nodes.mean(axis=0)
    nodes[:, 1] *= 1.8 / (nodes[:, 1].max() - nodes[:, 1].min())
    result = Bracket(load_cases=('vertical', 'diagonal')).analyze(nodes, tets,
                                                                 return_fields=True)
    stats = surface_statistics(nodes, result, target_nodes=10 ** 6, percentile=99.5)
    used = np.unique(result['fields']['faces'])
    assert stats['nodes'] == len(used)                   # nothing decimated at this target
    assert set(stats['cases']) == {'vertical', 'diagonal'}
    expect = {}
    for name, case in result['fields']['cases'].items():
        expect[name] = np.percentile(np.abs(case['von_mises_nodal'][used]) / 1e6, 99.5)
    assert np.isclose(stats['peak_von_mises_MPa'], max(expect.values()), rtol=1e-9)
    u = result['fields']['cases']['vertical']['displacement']
    assert np.isclose(stats['vertical_displacement_mm'], np.abs(u[used, 2]).max() * 1e3,
                      rtol=1e-9)
