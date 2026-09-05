"""Deterministic CPU tests for the descriptor proxy / calibration / E2 / C2 tools.

No checkpoint is needed: the decoder is `MockDecoder` (latent[0] = sphere
radius) or an analytic sphere/box SDF, and the velocity net is a zero or
linear map. Everything the tools do -- soft quadrature, affine calibration,
damped Newton with backtracking on the real Marching Cubes measurement,
endpoint-prediction guidance -- is checked against closed-form values.

Tolerances stated in each test come from `scratchpad/probe_proxy.py`
(2026-09-05): on an exact sphere SDF the cell-centre soft volume matches the
analytically smeared value V + 8 pi r tau^2 pi^2/6 to < 0.01% and the soft
area matches A + 4 pi tau^2 pi^2/3 to ~0.3%; Marching Cubes at res 96 measures
a sphere's volume to 0.2%.

Run from `methods/SDFFlow`:  python -m pytest -q tests/test_conditional_tools.py
"""

import inspect
import json
import math
import os
import sys

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('numpy')
pytest.importorskip('trimesh')
pytest.importorskip('skimage')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from general_modules import descriptor_calibration as dcal          # noqa: E402
from general_modules.descriptor_calibration import (                 # noqa: E402
    DescriptorCalibration, calibrate, fit_affine, true_descriptors)
from general_modules.descriptor_guidance import (                    # noqa: E402
    call_velocity, make_c2_guidance, total_guidance_strength)
from general_modules.descriptor_proxy import (                       # noqa: E402
    MockDecoder, box_area, box_sdf, box_volume, soft_descriptors, sphere_area, sphere_sdf,
    sphere_volume, supported_soft_names)
from general_modules.descriptor_refinement import (                  # noqa: E402
    cap_step, newton_correct, relative_residual, split_history_by_row, summarize_history)

torch.manual_seed(0)
torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

D = 8
LATENT_MEAN = torch.zeros(1, D)
LATENT_STD = torch.ones(1, D)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class AnalyticSphere:
    def __init__(self, radius):
        self.radius = radius

    def decode_flat(self, z_flat, points):
        return sphere_sdf(points, self.radius)


class AnalyticBox:
    def __init__(self, half):
        self.half = half

    def decode_flat(self, z_flat, points):
        return box_sdf(points, self.half)


class ZeroVelocity(torch.nn.Module):
    """v = 0: the FM state is already the endpoint, so x1_hat == z_next."""

    def forward(self, z, t, cond=None, cond_mask=None):
        return torch.zeros_like(z)


class ShrinkVelocity(torch.nn.Module):
    """v = -z: the endpoint prediction x1_hat = t * z depends on z through the net."""

    def forward(self, z, t, cond=None, cond_mask=None):
        return -z


class LegacyVelocity(torch.nn.Module):
    """A velocity net that predates the cond_mask keyword."""

    def forward(self, z, t, cond=None):
        return torch.zeros_like(z)


def radius_latent(radius, batch=1):
    z = torch.zeros(batch, D)
    z[:, 0] = torch.as_tensor(radius, dtype=torch.float32)
    return z


def smeared_sphere_volume(r, tau):
    return sphere_volume(r) + 8.0 * math.pi * r * tau ** 2 * math.pi ** 2 / 6.0


def smeared_sphere_area(r, tau):
    return sphere_area(r) + 4.0 * math.pi * tau ** 2 * math.pi ** 2 / 3.0


@pytest.fixture(scope='module')
def mock():
    return MockDecoder(D)


@pytest.fixture(scope='module')
def mock_calibration(mock):
    """Affine fit of the soft proxy on the mock, exactly as evaluate.py would build it."""
    radii = torch.linspace(0.3, 0.7, 6)
    cal = calibrate(mock, None, LATENT_MEAN, LATENT_STD,
                    z_batches=[radius_latent(radii, batch=len(radii))],
                    resolution=48, tau=0.032, measure_resolution=96, verbose=False,
                    split='mock', num_shapes=6, samples_per_shape=1)
    return cal


# ---------------------------------------------------------------------------
# (1) soft descriptors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('radius', [0.5, 0.7])
def test_soft_volume_and_area_match_analytic_sphere(radius):
    """Pilot settings (res 48, tau 0.032): volume within 0.2% of the smeared
    analytic value (5% of the raw 4/3 pi r^3), area within 0.6% of the smeared
    value (2% of the raw 4 pi r^2)."""
    z = torch.zeros(1, D)
    soft = soft_descriptors(AnalyticSphere(radius), z, resolution=48, tau=0.032)
    v = soft['volume'].item()
    a = soft['area'].item()
    assert abs(v / smeared_sphere_volume(radius, 0.032) - 1) < 2e-3
    assert abs(v / sphere_volume(radius) - 1) < 0.05
    assert abs(a / smeared_sphere_area(radius, 0.032) - 1) < 6e-3
    assert abs(a / sphere_area(radius) - 1) < 0.02


def test_soft_descriptors_converge_with_sharper_tau():
    """res 64, tau 0.01: volume within 0.5% and area within 1% of the raw sphere values."""
    z = torch.zeros(1, D)
    soft = soft_descriptors(AnalyticSphere(0.5), z, resolution=64, tau=0.01)
    assert abs(soft['volume'].item() / sphere_volume(0.5) - 1) < 5e-3
    assert abs(soft['area'].item() / sphere_area(0.5) - 1) < 1e-2


@pytest.mark.parametrize('half, res, tau, vol_tol, area_tol', [
    ((0.6, 0.6, 0.6), 48, 0.032, 0.03, 0.03),
    ((0.4, 0.3, 0.5), 64, 0.02, 0.03, 0.04),
])
def test_soft_volume_and_area_match_analytic_box(half, res, tau, vol_tol, area_tol):
    """Boxes: volume 8 hx hy hz and area 8 (hx hy + hy hz + hz hx). The finite-
    difference area under-resolves the edges, so the area tolerance is looser."""
    z = torch.zeros(1, D)
    soft = soft_descriptors(AnalyticBox(half), z, resolution=res, tau=tau)
    assert abs(soft['volume'].item() / box_volume(half) - 1) < vol_tol
    assert abs(soft['area'].item() / box_area(half) - 1) < area_tol


def test_soft_descriptors_monotone_in_radius(mock):
    radii = torch.linspace(0.2, 0.8, 7)
    soft = soft_descriptors(mock, radius_latent(radii, batch=len(radii)), resolution=48, tau=0.032)
    for name in ('volume', 'area'):
        values = soft[name].tolist()
        assert all(b > a for a, b in zip(values, values[1:])), (name, values)


def test_soft_volume_gradient_wrt_radius_is_positive_and_exact_elsewhere(mock):
    """d(soft volume)/d radius must be positive and within 3% of 4 pi r^2; the
    mock ignores latent[1:], so those Jacobian entries are exactly zero."""
    z = radius_latent([0.4, 0.5], batch=2).requires_grad_(True)
    soft = soft_descriptors(mock, z, resolution=48, tau=0.032)
    grad, = torch.autograd.grad(soft['volume'].sum(), z)
    for row, r in zip(grad, (0.4, 0.5)):
        assert row[0].item() > 0
        assert abs(row[0].item() / (4 * math.pi * r * r) - 1) < 0.03
        assert torch.equal(row[1:], torch.zeros(D - 1))
    grad_area, = torch.autograd.grad(
        soft_descriptors(mock, z, resolution=48, tau=0.032)['area'].sum(), z)
    assert (grad_area[:, 0] > 0).all()


def test_soft_descriptors_reject_unknown_names(mock):
    assert supported_soft_names(('bbox_x', 'volume', 'mass_kg', 'area')) == ('volume', 'area')
    with pytest.raises(ValueError):
        soft_descriptors(mock, radius_latent(0.5), names=('bbox_x',))


# ---------------------------------------------------------------------------
# (2) calibration
# ---------------------------------------------------------------------------

def test_fit_affine_recovers_known_coefficients():
    import numpy as np
    rng = np.random.default_rng(0)
    true = np.linspace(0.2, 1.4, 40)
    proxy = 0.86 * true + 0.40 + rng.normal(scale=1e-4, size=true.shape)
    fit = fit_affine(proxy, true)
    assert abs(fit['a'] - 0.86) < 1e-3
    assert abs(fit['b'] - 0.40) < 1e-3
    assert fit['r2'] > 0.9999
    assert fit['n'] == 40
    # non-finite pairs are dropped, not propagated
    proxy[3] = np.nan
    assert fit_affine(proxy, true)['n'] == 39
    with pytest.raises(ValueError):
        fit_affine([1.0, 2.0], [0.5, 0.5])


def test_calibration_save_load_roundtrip_and_check_compatible(tmp_path):
    vae_file = tmp_path / 'vae.pth'
    fm_file = tmp_path / 'fm.pth'
    vae_file.write_bytes(b'vae-bytes')
    fm_file.write_bytes(b'fm-bytes')
    cal = DescriptorCalibration(
        coefficients={'volume': {'a': 0.86, 'b': 0.40, 'r2': 0.98, 'n': 24},
                      'area': {'a': 0.357, 'b': 1.653, 'r2': 0.598, 'n': 24}},
        resolution=48, tau=0.032, measure_resolution=96,
        vae_sha256=dcal.file_sha256(str(vae_file)), fm_sha256=dcal.file_sha256(str(fm_file)),
        cond_names=['volume', 'area'], split='val', num_shapes=6, samples_per_shape=4)
    path = str(tmp_path / 'sub' / 'descriptor_calibration.pth')
    cal.save(path)
    loaded = DescriptorCalibration.load(path)
    assert loaded.to_dict() == cal.to_dict()
    assert loaded.proxy_target('volume', 1.0) == pytest.approx(0.86 + 0.40)
    assert loaded.names == ('volume', 'area')

    assert loaded.check_compatible(str(vae_file), str(fm_file), 48, 0.032, names=('volume',))
    with pytest.raises(ValueError, match='tau'):
        loaded.check_compatible(str(vae_file), str(fm_file), 48, 0.02)
    with pytest.raises(ValueError, match='resolution'):
        loaded.check_compatible(str(vae_file), str(fm_file), 64, 0.032)
    with pytest.raises(ValueError, match='coefficients'):
        loaded.check_compatible(str(vae_file), str(fm_file), 48, 0.032, names=('bbox_x',))
    fm_file.write_bytes(b'fm-bytes-retrained')
    with pytest.raises(ValueError, match='sha256'):
        loaded.check_compatible(str(vae_file), str(fm_file), 48, 0.032)
    with pytest.raises(ValueError, match='does not exist'):
        loaded.check_compatible(str(tmp_path / 'missing.pth'), None, 48, 0.032)
    # A calibration built without checkpoint files skips the hash check.
    bare = DescriptorCalibration(loaded.coefficients, 48, 0.032)
    assert bare.check_compatible(str(vae_file), str(fm_file), 48, 0.032)
    with pytest.raises(ValueError):
        DescriptorCalibration.from_dict({'format': 'something_else'})


def test_calibrate_helper_fits_the_mock_proxy(mock, mock_calibration):
    """On the mock the proxy is the smeared sphere volume, linear in the true
    volume with slope ~1 (R^2 > 0.99); every row decodes to a valid mesh."""
    cal = mock_calibration
    assert cal.names == ('volume', 'area')
    assert cal['volume']['r2'] > 0.99
    assert 0.9 < cal['volume']['a'] < 1.2
    assert cal['area']['r2'] > 0.99
    assert len(cal.rows) == 6 and all(r['valid'] for r in cal.rows)
    assert cal.extra['valid_rows'] == 6
    assert cal.vae_sha256 == '' and cal.fm_sha256 == ''
    assert cal.split == 'mock' and cal.num_shapes == 6


def test_calibrate_with_cond_batches_runs_the_sampler_and_keeps_invalid_rows(mock):
    """cond_batches path: ZeroVelocity leaves noise unchanged, so the fixed
    noise rows ARE the latents; a negative radius decodes to no zero crossing
    and must be recorded as invalid and excluded from the fit."""
    noise = radius_latent([0.35, 0.5, 0.65, -0.2], batch=4)
    cond = torch.zeros(4, 2)
    cal = calibrate(mock, ZeroVelocity(), LATENT_MEAN, LATENT_STD, cond_batches=[cond],
                    noise_batches=[noise], resolution=32, tau=0.032, measure_resolution=64,
                    ode_steps=4, verbose=False)
    assert [r['valid'] for r in cal.rows] == [True, True, True, False]
    assert cal['volume']['n'] == 3
    assert cal.rows[0]['cond_normalized'] == [0.0, 0.0]
    assert math.isnan(cal.rows[3]['true_volume'])


# ---------------------------------------------------------------------------
# (3) E2 Newton correction
# ---------------------------------------------------------------------------

def test_newton_correct_drives_sphere_volume_to_target(mock, mock_calibration):
    z0 = radius_latent(0.4)
    target = {'volume': sphere_volume(0.55)}
    z1, history = newton_correct(mock, z0, target, mock_calibration, LATENT_MEAN, LATENT_STD,
                                 rounds=3, step_cap_rms=0.12, line_search_tries=3,
                                 measure_resolution=96, resolution=48, tau=0.032)
    assert z1.shape == z0.shape
    assert torch.equal(z1[:, 1:], z0[:, 1:])  # the mock's dead coordinates never move
    truth = true_descriptors(mock, z1, measure_resolution=96)
    assert truth['valid']
    assert abs(truth['volume'] / target['volume'] - 1) < 0.01
    # Flat per-round history (contract): first entry is the initial measurement.
    assert isinstance(history, list) and all(isinstance(h, dict) for h in history)
    assert history[0]['round'] == -1 and history[0]['step_accepted'] is None
    assert all(h['row'] == 0 for h in history)
    rounds = [h for h in history if h['round'] >= 0]
    assert 1 <= len(rounds) <= 3
    assert history[0]['residual_after'] > rounds[-1]['residual_after']
    for h in rounds:
        if h['step_accepted']:
            assert h['residual_after'] < h['residual_before']
        else:
            assert h['residual_after'] == h['residual_before']
        assert h['step_rms'] <= 0.12 + 1e-6
        assert h['true_after']['valid'] and math.isfinite(h['true_after']['volume'])
    # The consumer pattern sample.py uses on the returned history.
    assert sum(1 for h in history if isinstance(h, dict) and h.get('step_accepted')) >= 1
    # Plain types only: the history goes straight into sample_<seed>_meta.json.
    json.dumps(history)
    summary = summarize_history(history)
    assert summary['accepted'] >= 1 and summary['residual_final'] < 0.01
    assert summary['measurements'] == 1 + sum(h['tries'] for h in rounds)


def test_newton_correct_joint_volume_area_targets(mock, mock_calibration):
    """Two consistent targets (the sphere of radius 0.6) through the 2 x D solve."""
    z0 = radius_latent(0.45)
    target = {'volume': sphere_volume(0.6), 'area': sphere_area(0.6), 'bbox_x': 1.2}
    z1, history = newton_correct(mock, z0, target, mock_calibration, LATENT_MEAN, LATENT_STD,
                                 rounds=3, measure_resolution=96)
    truth = true_descriptors(mock, z1, measure_resolution=96)
    assert abs(truth['volume'] / target['volume'] - 1) < 0.01
    assert abs(truth['area'] / target['area'] - 1) < 0.01
    assert history[0]['ignored_targets'] == ['bbox_x']
    assert set(history[1]['proxy_before']) == {'volume', 'area'}


def test_newton_correct_respects_the_rms_cap(mock, mock_calibration):
    """cap 0.02 x sqrt(D): the uncapped step (~0.2 in radius) must be scaled down."""
    z0 = radius_latent(0.4)
    target = {'volume': sphere_volume(0.6)}
    z1, history = newton_correct(mock, z0, target, mock_calibration, LATENT_MEAN, LATENT_STD,
                                 rounds=1, step_cap_rms=0.02, measure_resolution=64)
    first = history[1]
    assert first['capped'] is True
    assert first['step_rms'] <= 0.02 + 1e-7
    assert first['step_accepted'] is True
    assert abs(float(z1[0, 0] - z0[0, 0])) <= 0.02 * math.sqrt(D) + 1e-6
    dz, raw, capped = cap_step(torch.ones(4) * 0.5, 0.1)
    assert capped and abs(float(dz.norm()) - 0.1 * 2.0) < 1e-6 and raw == pytest.approx(1.0)


def test_newton_correct_never_accepts_a_worse_step(mock):
    """A calibration with the wrong slope sign proposes steps in the wrong
    direction; every backtracking try must be rejected and z left untouched."""
    wrong = DescriptorCalibration({'volume': {'a': -1.0, 'b': 0.0, 'r2': 1.0, 'n': 2}}, 48, 0.032)
    z0 = radius_latent(0.4)
    z1, history = newton_correct(mock, z0, {'volume': sphere_volume(0.55)}, wrong,
                                 LATENT_MEAN, LATENT_STD, rounds=3, measure_resolution=64)
    assert torch.equal(z1, z0)
    rounds = [h for h in history if h['round'] >= 0]
    assert len(rounds) == 1 and rounds[0]['step_accepted'] is False and rounds[0]['tries'] == 3
    assert rounds[0]['residual_after'] == rounds[0]['residual_before']
    assert summarize_history(history)['accepted'] == 0


def test_newton_correct_skips_an_invalid_start(mock, mock_calibration):
    z0 = radius_latent(-0.3)  # no zero crossing anywhere
    z1, history = newton_correct(mock, z0, {'volume': 0.5}, mock_calibration,
                                 LATENT_MEAN, LATENT_STD, rounds=2, measure_resolution=48)
    assert torch.equal(z1, z0)
    assert len(history) == 1 and 'skipped' in history[0]
    assert summarize_history(history)['skipped'] and summarize_history(history)['rounds'] == 0


def test_newton_correct_accepts_vae_space_latents(mock, mock_calibration):
    """normalized=False: same correction expressed through a non-trivial
    latent_mean / latent_std, returned in VAE space."""
    mean = torch.full((1, D), 0.1)
    std = torch.full((1, D), 2.0)
    z_vae = radius_latent(0.4)
    z_out, _ = newton_correct(mock, z_vae, {'volume': sphere_volume(0.55)}, mock_calibration,
                              mean, std, rounds=3, measure_resolution=96, normalized=False)
    truth = true_descriptors(mock, z_out, measure_resolution=96)
    assert abs(truth['volume'] / sphere_volume(0.55) - 1) < 0.01


def test_newton_correct_batched_rows_share_one_flat_history(mock, mock_calibration):
    """Two latents in one call: the flat history is ordered row 0 then row 1,
    each row starts with its own round -1 entry, and the summaries per row
    match a per-row call (same seedless, deterministic arithmetic)."""
    z0 = radius_latent([0.4, 0.7], batch=2)
    target = {'volume': sphere_volume(0.55)}
    z_batch, history = newton_correct(mock, z0, target, mock_calibration, LATENT_MEAN, LATENT_STD,
                                      rounds=2, measure_resolution=64)
    rows = split_history_by_row(history)
    assert len(rows) == 2 and [r[0]['row'] for r in rows] == [0, 1]
    assert all(r[0]['round'] == -1 for r in rows)
    assert [h['row'] for h in history] == sorted(h['row'] for h in history)
    with pytest.raises(ValueError, match='row'):
        summarize_history(history)
    for b in range(2):
        z_single, hist_single = newton_correct(mock, z0[b:b + 1], target, mock_calibration,
                                               LATENT_MEAN, LATENT_STD, rounds=2,
                                               measure_resolution=64)
        assert torch.equal(z_single, z_batch[b:b + 1])
        assert summarize_history(history, row=b) == summarize_history(hist_single)
    # Both rows end closer to the target than they started.
    for b in range(2):
        s = summarize_history(history, row=b)
        assert s['residual_final'] < s['residual_initial']


def test_relative_residual_handles_missing_values():
    assert relative_residual({'volume': 1.1}, {'volume': 1.0}, ['volume']) == pytest.approx(0.1)
    assert relative_residual({'volume': float('nan')}, {'volume': 1.0}, ['volume']) == math.inf
    assert relative_residual({}, {'volume': 1.0}, ['volume']) == math.inf


# ---------------------------------------------------------------------------
# (4) C2 guidance
# ---------------------------------------------------------------------------

def test_c2_guidance_zero_outside_window_and_signed_inside(mock, mock_calibration):
    target = {'volume': sphere_volume(0.6), 'mass_kg': 1.0}
    guide = make_c2_guidance(mock, ZeroVelocity(), None, None, target, mock_calibration,
                             eta=0.1, t_start=0.3, step_mode='per_step_jump', resolution=48,
                             tau=0.032, latent_mean=LATENT_MEAN, latent_std=LATENT_STD,
                             verbose=False)
    assert guide.names == ['volume'] and guide.stats['ignored_targets'] == ['mass_kg']
    z = radius_latent([0.4, 0.8], batch=2)
    for t_out in (0.0, 0.28, 1.0, torch.tensor(0.1), torch.full((2,), 1.0)):
        delta = guide(z, t_out, 0.02)
        assert torch.equal(delta, torch.zeros_like(z))
    assert guide.stats['fm_evaluations'] == 0

    delta = guide(z, 0.5, 0.02)
    assert delta.shape == z.shape and delta.dtype == z.dtype
    # Row 0 is too small -> grow the radius; row 1 is too big -> shrink it.
    assert delta[0, 0].item() > 0 and delta[1, 0].item() < 0
    # The mock's dead coordinates get exactly zero gradient.
    assert torch.equal(delta[:, 1:], torch.zeros(2, D - 1))
    # RMS normalization: per-row RMS of delta == eta * (1 - t).
    rms = delta.pow(2).mean(dim=1).sqrt()
    assert torch.allclose(rms, torch.full((2,), 0.1 * 0.5), rtol=1e-4)
    assert guide.stats['fm_evaluations'] == 1 and guide.stats['active_calls'] == 1
    # The gradient must flow through the velocity net too.
    guide_shrink = make_c2_guidance(mock, ShrinkVelocity(), None, None, target, mock_calibration,
                                    eta=0.1, latent_mean=LATENT_MEAN, latent_std=LATENT_STD,
                                    verbose=False)
    d2 = guide_shrink(z, 0.5, 0.02)
    assert d2[0, 0].item() > 0 and torch.isfinite(d2).all()


def test_c2_guidance_step_modes_agree_at_reference_and_velocity_dt_scales_with_dt(
        mock, mock_calibration):
    target = {'volume': sphere_volume(0.6), 'area': sphere_area(0.6)}
    common = dict(eta=0.1, t_start=0.3, resolution=48, tau=0.032, ode_steps_ref=50, verbose=False)
    jump = make_c2_guidance(mock, ZeroVelocity(), None, None, target, mock_calibration,
                            step_mode='per_step_jump', latent_mean=LATENT_MEAN,
                            latent_std=LATENT_STD, **common)
    vel = make_c2_guidance(mock, ZeroVelocity(), None, None, target, mock_calibration,
                           step_mode='velocity_dt', latent_mean=LATENT_MEAN,
                           latent_std=LATENT_STD, **common)
    z = radius_latent(0.4)
    d_jump_50 = jump(z, 0.5, 1.0 / 50)
    d_vel_50 = vel(z, 0.5, 1.0 / 50)
    d_vel_25 = vel(z, 0.5, 1.0 / 25)
    d_vel_100 = vel(z, 0.5, 1.0 / 100)
    d_jump_25 = jump(z, 0.5, 1.0 / 25)
    assert torch.allclose(d_jump_50, d_vel_50, rtol=1e-6, atol=1e-9)
    assert torch.allclose(d_vel_25, 2.0 * d_vel_50, rtol=1e-6, atol=1e-9)
    assert torch.allclose(d_vel_100, 0.5 * d_vel_50, rtol=1e-6, atol=1e-9)
    assert torch.equal(d_jump_25, d_jump_50)  # the pilot mode ignores dt
    # Totals over a whole trajectory: velocity_dt is NFE-invariant, the pilot mode is not.
    ref = total_guidance_strength(50, step_mode='velocity_dt')
    assert total_guidance_strength(25, step_mode='velocity_dt') == pytest.approx(ref, rel=0.05)
    assert total_guidance_strength(100, step_mode='velocity_dt') == pytest.approx(ref, rel=0.05)
    assert total_guidance_strength(50, step_mode='per_step_jump') == pytest.approx(ref)
    assert total_guidance_strength(100, step_mode='per_step_jump') > 1.9 * ref


def test_total_guidance_strength_matches_its_documented_numbers():
    """`velocity_dt` is a Riemann sum, so it is only APPROXIMATELY NFE-invariant.

    The docstring quotes the totals and the analytic limit; pin both, because
    the +11% at 10 steps is the reason a cheap preview is not comparable with
    the 50-step run it is read against.
    """
    measured = {n: total_guidance_strength(n) for n in (10, 25, 50, 100, 200)}
    assert measured == pytest.approx(
        {10: 1.4000, 25: 1.2240, 50: 1.2600, 100: 1.2425, 200: 1.2338}, abs=1e-4)
    limit = 0.1 * 50 * (1.0 - 0.3) ** 2 / 2       # eta * ode_steps_ref * (1 - t_start)^2 / 2
    assert limit == pytest.approx(1.2250)
    assert measured[10] / limit > 1.11, 'the 10-step over-guidance the docstring warns about'
    for n in (25, 50, 100, 200):
        assert measured[n] == pytest.approx(limit, rel=0.03)


def test_c2_guidance_validates_its_arguments(mock, mock_calibration):
    target = {'volume': sphere_volume(0.6)}
    with pytest.raises(ValueError):
        make_c2_guidance(mock, ZeroVelocity(), None, None, target, mock_calibration,
                         step_mode='nope', verbose=False)
    with pytest.raises(ValueError):
        make_c2_guidance(mock, ZeroVelocity(), None, None, target, mock_calibration,
                         eta=0.0, verbose=False)
    with pytest.raises(ValueError):
        make_c2_guidance(mock, ZeroVelocity(), None, None, {'bbox_x': 1.8}, mock_calibration,
                         verbose=False)


def test_public_signatures_match_the_shared_contract():
    """Positional order is part of the cross-agent contract (section C)."""
    from general_modules.descriptor_proxy import soft_descriptors as _soft
    assert list(inspect.signature(_soft).parameters) == [
        'vae', 'z_flat', 'names', 'resolution', 'tau', 'bound', 'chunk']
    assert list(inspect.signature(make_c2_guidance).parameters)[:12] == [
        'vae', 'fm_model', 'cond', 'cond_mask', 'targets', 'calibration', 'eta', 't_start',
        'step_mode', 'resolution', 'tau', 'ode_steps_ref']
    assert list(inspect.signature(newton_correct).parameters)[:12] == [
        'vae', 'z_flat', 'targets', 'calibration', 'latent_mean', 'latent_std', 'rounds',
        'step_cap_rms', 'line_search_tries', 'measure_resolution', 'resolution', 'tau']
    defaults = {k: v.default for k, v in inspect.signature(newton_correct).parameters.items()}
    assert (defaults['rounds'], defaults['step_cap_rms'], defaults['line_search_tries'],
            defaults['measure_resolution'], defaults['resolution'], defaults['tau']) == (
        3, 0.12, 3, 96, 48, 0.032)
    gdef = {k: v.default for k, v in inspect.signature(make_c2_guidance).parameters.items()}
    assert (gdef['eta'], gdef['t_start'], gdef['step_mode'], gdef['resolution'], gdef['tau'],
            gdef['ode_steps_ref']) == (0.1, 0.3, 'velocity_dt', 48, 0.032, 50)
    assert list(inspect.signature(DescriptorCalibration.check_compatible).parameters)[1:5] == [
        'vae_path', 'fm_path', 'resolution', 'tau']
    assert list(inspect.signature(fit_affine).parameters) == ['proxy', 'true']


def test_call_velocity_falls_back_for_a_legacy_model():
    z = torch.zeros(3, D)
    t = torch.zeros(3)
    assert torch.equal(call_velocity(LegacyVelocity(), z, t, cond=None, cond_mask=None), z)
    ones = torch.ones(3, 2, dtype=torch.bool)
    assert torch.equal(call_velocity(LegacyVelocity(), z, t, cond=torch.zeros(3, 2), cond_mask=ones), z)
    partial = ones.clone()
    partial[0, 1] = False
    with pytest.raises(TypeError, match='cond_mask'):
        call_velocity(LegacyVelocity(), z, t, cond=torch.zeros(3, 2), cond_mask=partial)
    # A model that takes cond_mask receives it untouched (no swallowed errors).
    assert torch.equal(call_velocity(ZeroVelocity(), z, t, cond=None, cond_mask=partial), z)


# ---------------------------------------------------------------------------
# (5) disabled paths are bitwise no-ops
# ---------------------------------------------------------------------------

def test_newton_rounds_zero_is_a_bitwise_noop(mock, mock_calibration):
    z0 = torch.randn(3, D)
    z1, history = newton_correct(mock, z0, {'volume': 0.5}, mock_calibration,
                                 LATENT_MEAN, LATENT_STD, rounds=0)
    assert z1 is z0  # the very same tensor, not a copy
    assert history == []
    assert summarize_history(history)['rounds'] == 0


def _euler(model, noise, ode_steps, guidance_fn=None):
    z = noise.clone()
    dt = 1.0 / ode_steps
    for i in range(ode_steps):
        t = torch.full((z.shape[0],), i * dt)
        z = z + model(z, t, cond=None, cond_mask=None) * dt
        if guidance_fn is not None:
            z = z + guidance_fn(z, (i + 1) * dt, dt)
    return z


def test_guidance_outside_its_window_leaves_the_trajectory_bitwise_unchanged(mock, mock_calibration):
    """A window that never opens (t_start just below 1 with 10 steps) must add
    exact zeros, so the guided and unguided Euler trajectories are identical."""
    model = ShrinkVelocity()
    guide = make_c2_guidance(mock, model, None, None, {'volume': sphere_volume(0.6)},
                             mock_calibration, t_start=0.95, latent_mean=LATENT_MEAN,
                             latent_std=LATENT_STD, verbose=False)
    noise = radius_latent([0.4, 0.7], batch=2) + 0.01 * torch.randn(2, D)
    plain = _euler(model, noise, 10)
    guided = _euler(model, noise, 10, guidance_fn=guide)
    assert torch.equal(plain, guided)
    assert guide.stats['calls'] == 10 and guide.stats['active_calls'] == 0


def test_sample_latents_guidance_hook_is_a_noop_when_it_returns_zeros(mock, mock_calibration):
    """Once `sample_latents` accepts `guidance_fn`, a zero-returning callback
    must reproduce the unguided integration bit for bit (skipped until then)."""
    from model.velocity_net import sample_latents
    if 'guidance_fn' not in inspect.signature(sample_latents).parameters:
        pytest.skip('sample_latents has no guidance_fn parameter yet')
    model = ShrinkVelocity()
    noise = torch.randn(2, D)
    plain = sample_latents(model, 2, D, 'cpu', ode_steps=8, noise=noise)
    guided = sample_latents(model, 2, D, 'cpu', ode_steps=8, noise=noise,
                            guidance_fn=lambda z, t, dt: torch.zeros_like(z))
    assert torch.equal(plain, guided)


# ---------------------------------------------------------------------------
# (6) the export-path truth never substitutes a convex hull
# ---------------------------------------------------------------------------

class TornSphere:
    """A sphere the grid cannot close: radius 1.2 against the bound-1.0 domain.

    Marching Cubes returns an open surface whose `convex_hull.volume` (6.383)
    is far above the solid it actually bounds -- the substitution
    `sdf_sampling.mesh_descriptors` makes and the one every consumer here used
    to read as the export-path truth.
    """

    def decode_flat(self, z_flat, points):
        return sphere_sdf(points, 1.2)


def test_true_descriptors_reports_nan_volume_for_a_non_watertight_mesh():
    truth = true_descriptors(TornSphere(), torch.zeros(1, D), measure_resolution=96)
    assert truth['valid'] and not truth['watertight']
    assert math.isnan(truth['volume']), 'a torn mesh must not report its convex hull as volume'
    assert truth['area'] > 0, 'area is well defined on an open surface and is kept'
    # This is what mesh_extraction.mesh_report says about the same mesh.
    from general_modules.mesh_extraction import decode_sdf_grid, mesh_report, sdf_grid_to_mesh
    grid = decode_sdf_grid(TornSphere(), torch.zeros(1, D), resolution=96)
    assert mesh_report(sdf_grid_to_mesh(grid))['volume'] is None


def test_calibrate_drops_the_volume_pair_of_a_torn_row_only(mock):
    """A non-watertight row keeps its area pair and loses its volume pair."""
    radii = torch.tensor([0.3, 0.4, 0.5, 0.6, 1.2])
    cal = calibrate(mock, None, LATENT_MEAN, LATENT_STD,
                    z_batches=[radius_latent(radii, batch=len(radii))],
                    resolution=48, tau=0.032, measure_resolution=96, verbose=False,
                    split='mock', num_shapes=5, samples_per_shape=1)
    assert cal['area']['n'] == 5
    assert cal['volume']['n'] == 4, 'the r=1.2 row has no measurable volume'
    assert cal.extra['watertight_rows'] == 4
    assert cal.extra['watertight_rate'] == pytest.approx(0.8)


def test_calibrate_refuses_a_fit_below_min_r2(mock):
    with pytest.raises(ValueError, match='too weak to use'):
        calibrate(mock, None, LATENT_MEAN, LATENT_STD,
                  z_batches=[radius_latent(torch.linspace(0.3, 0.7, 6), batch=6)],
                  resolution=48, tau=0.032, measure_resolution=96, verbose=False,
                  min_r2=1.0)


def test_check_compatible_pins_the_measurement_grid(tmp_path):
    cal = DescriptorCalibration({'volume': {'a': 1.0, 'b': 0.0, 'r2': 0.99, 'n': 8}},
                                resolution=48, tau=0.032, measure_resolution=128)
    assert cal.check_compatible(None, None, 48, 0.032, measure_resolution=128)
    assert cal.check_compatible(None, None, 48, 0.032)          # unspecified = unchecked
    with pytest.raises(ValueError, match='Marching Cubes grid'):
        cal.check_compatible(None, None, 48, 0.032, measure_resolution=96)


def test_newton_rejects_a_step_that_tears_the_mesh_open(mock, mock_calibration):
    """require_watertight defaults to True whenever `volume` is a target."""
    z0 = radius_latent(0.5)
    z_out, history = newton_correct(mock, z0, {'volume': 20.0}, mock_calibration,
                                    LATENT_MEAN, LATENT_STD, rounds=1, step_cap_rms=5.0,
                                    measure_resolution=48, resolution=32)
    # Volume 20 is unreachable inside the bound-1.0 domain, so every
    # backtracking try decodes an OPEN sphere. Its convex hull would look like
    # a huge improvement (residual 0.87 -> ~0.5 in the run that found this);
    # measured honestly its volume is NaN, the residual is inf, and no step is
    # taken.
    rounds = [h for h in history if h['round'] >= 0]
    assert rounds and not any(h['step_accepted'] for h in rounds)
    assert torch.equal(z_out, z0)


def test_newton_keeps_the_corrected_latent_inside_latent_clip(mock, mock_calibration):
    z0 = radius_latent(0.35)
    z_out, history = newton_correct(mock, z0, {'volume': 1.2}, mock_calibration,
                                    LATENT_MEAN, LATENT_STD, rounds=3, step_cap_rms=0.5,
                                    measure_resolution=48, resolution=32, latent_clip=0.5)
    assert float(z_out.abs().max()) <= 0.5 + 1e-6
    unclipped, _ = newton_correct(mock, z0, {'volume': 1.2}, mock_calibration,
                                  LATENT_MEAN, LATENT_STD, rounds=3, step_cap_rms=0.5,
                                  measure_resolution=48, resolution=32)
    assert float(unclipped.abs().max()) > 0.5, 'the clip must actually be binding here'
