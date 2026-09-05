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
from design_loop.problem import LBF_TO_N, LBIN_TO_NM, Bracket           # noqa: E402


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


@requires_checkpoints
def test_noise_never_leaves_the_gaussian_shell(generator):
    """Any design vector, in or out of bounds, must map inside the shell."""
    import torch
    radius = generator.shell_scale * np.sqrt(generator.subspace_dim)
    rng = np.random.default_rng(0)
    for scale in (0.1, 1.0, 10.0):
        x = rng.standard_normal(generator.subspace_dim) * scale
        composed = generator._noise(x)
        in_subspace = (composed.squeeze(0) @ generator.basis.T)
        assert float(torch.linalg.norm(in_subspace)) <= radius * (1 + 1e-5)
