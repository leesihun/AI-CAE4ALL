"""CPU tests for the conditional-inference wiring (contract section D):

  * partial `cond_values` ('nan' = unspecified) -> cond_mask, and the clear
    error on a legacy `cond_dropout_mode all` checkpoint (sample.py);
  * `interpolation_space cond_sweep` (interpolate.py): fixed noise, swept
    conditions, endpoints reproduce plain conditional samples, K distinct
    latents, strip PNG + metadata fields;
  * `eval_task conditional` (evaluate.py) with plain + rejection (and, when the
    descriptor modules import, c2 / e2 / c2e2) writing eval_conditional.json /
    .csv with the contract fields;
  * `eval_task descriptor_calibration` writing a loadable, compatible
    `DescriptorCalibration`;
  * `mode sample` with guidance + Newton on that calibration.

No trained checkpoint exists and none is trained here. The VAE and the
velocity net are RANDOMLY INITIALISED from configs/SDFFlow/config_train.txt
(via general_modules.load_config, shrunk to CPU-test sizes) and saved in the
real checkpoint payload format; a 12-shape synthetic HDF5 (with a one-column
FEA `cond_extra` sidecar) is built under pytest's tmp dir. Where a test needs
a decoder whose geometry is KNOWN (calibration slope, Newton actually reducing
the error) `SDFVAE` is monkeypatched with a subclass whose `decode_flat` is an
analytic sphere of latent-dependent radius; parameters and checkpoint format
are unchanged, so the loaders and the whole inference path run as in
production.

Run from `methods/SDFFlow`:  python -m pytest -q tests/test_conditional_inference.py
"""

import csv
import json
import math
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
h5py = pytest.importorskip('h5py')
trimesh = pytest.importorskip('trimesh')
pytest.importorskip('skimage')
pytest.importorskip('matplotlib')

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = os.path.dirname(os.path.dirname(REPO))
sys.path.insert(0, REPO)

from general_modules import condition_names as CN                       # noqa: E402
from general_modules.load_config import load_config, parse_value          # noqa: E402
from general_modules.sdf_dataset import build_dataset_splits                # noqa: E402
from general_modules.sdf_sampling import COND_NAMES, synthetic_sample     # noqa: E402
from inference_profiles import evaluate as evaluate_mod                   # noqa: E402
from inference_profiles import interpolate as interpolate_mod             # noqa: E402
from inference_profiles import sample as sample_mod                       # noqa: E402
from model.sdf_vae import SDFVAE                                          # noqa: E402
from model.velocity_net import VelocityNet, sample_latents                # noqa: E402

try:
    from general_modules import descriptor_calibration as _dcal           # noqa: E402
    from general_modules import descriptor_guidance as _dg                # noqa: F401,E402
    from general_modules import descriptor_proxy as _dp                   # noqa: F401,E402
    from general_modules import descriptor_refinement as _dr              # noqa: F401,E402
    HAVE_DESCRIPTOR_TOOLS = True
    DESCRIPTOR_SKIP_REASON = ''
except ImportError as exc:  # pragma: no cover - depends on the concurrent modules
    HAVE_DESCRIPTOR_TOOLS = False
    DESCRIPTOR_SKIP_REASON = f'descriptor modules not importable: {exc}'

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

CONFIG_TRAIN = os.path.join(SUITE, 'configs', 'SDFFlow', 'config_train.txt')
NUM_SHAPES = 12
FEA_NAME = 'log_max_ver_stress_mpa'           # one FEA-named sidecar column
GEOM_NAMES = ('volume', 'area')
MC_RES = 24
ODE_STEPS = 6
SOFT_RES = 16
SOFT_TAU = 0.032

needs_descriptor_tools = pytest.mark.skipif(not HAVE_DESCRIPTOR_TOOLS, reason=DESCRIPTOR_SKIP_REASON)


# ---------------------------------------------------------------------------
# fixtures: synthetic dataset, tiny random VAE / FM checkpoints
# ---------------------------------------------------------------------------

def _tiny_config(dataset_path):
    """configs/SDFFlow/config_train.txt through the native parser, shrunk to
    CPU-test sizes (architecture keys only; the recipe's semantics are kept)."""
    config = load_config(CONFIG_TRAIN)
    config.update({
        'dataset_dir': dataset_path, 'split_seed': 42,
        'latent_tokens': 1, 'latent_dim': 16, 'decoder_type': 'mlp', 'decoder_hidden': 32,
        'decoder_layers': 2, 'encoder_dim': 32, 'encoder_heads': 2, 'encoder_blocks': 1,
        'fourier_bands': 2, 'num_encoder_points': 256, 'num_query_points': 256,
        'fm_hidden': 32, 'fm_blocks': 2, 'fm_cond_hidden': 16, 'fm_arch': 'mlp',
    })
    return config


def _randomize_zero_init(model, std=0.05, generator=None):
    """AdaLN-Zero nets start as v = 0 with the condition disconnected (zero
    modulation + zero out_proj); give those layers random weights so the test
    velocity actually depends on z, t and cond."""
    for name, param in model.named_parameters():
        if name.endswith(('modulation.weight', 'out_proj.weight')):
            with torch.no_grad():
                param.copy_(torch.randn(param.shape, generator=generator) * std)


def _write_synthetic_dataset(path, num_shapes=NUM_SHAPES, seed=0):
    rng = np.random.default_rng(seed)
    conds = []
    with h5py.File(path, 'w') as h5:
        shapes = h5.create_group('shapes')
        for i in range(num_shapes):
            sample, cond = synthetic_sample(rng, 512, 512, 256, mc_resolution=32)
            grp = shapes.create_group(f'{i:05d}')
            for key, arr in sample.items():
                grp.create_dataset(key, data=arr)
            grp.create_dataset('cond', data=cond)
            grp.attrs['source'] = f'synthetic_{i}'
            conds.append(cond)
        h5.attrs['num_shapes'] = num_shapes
        h5.attrs['cond_names'] = COND_NAMES
        # A one-column FEA sidecar (contract A): a fake log-stress that varies
        # with the volume, so it has non-zero std and a known raw value.
        stress_mpa = 400.0 + 900.0 * np.asarray([c[3] for c in conds], dtype=np.float64)
        extra = np.asarray([CN.to_stored(FEA_NAME, v) for v in stress_mpa], dtype=np.float32)[:, None]
        h5.create_dataset(CN.SIDECAR_DATASET, data=extra)
        h5.attrs[CN.SIDECAR_NAMES_ATTR] = [FEA_NAME]
        h5.attrs[CN.SIDECAR_SOURCE_ATTR] = 'synthetic (test)'
        h5.attrs[CN.SIDECAR_TRANSFORMS_ATTR] = json.dumps({FEA_NAME: 'log'})
        h5.attrs[CN.SIDECAR_CREATED_ATTR] = '2026-09-05'
    return path


def _vae_payload(vae, config):
    return {'schema_version': 'sdfflow_infer_v1', 'stage': 'vae', 'epoch': 0,
            'model_state': vae.state_dict(), 'ema_state': None, 'config': dict(config),
            'cond_mean': None, 'cond_std': None, 'cond_names': None}


def _fm_payload(fm, config, vae_path, latent_mean, latent_std, cond_stats, cond_names):
    """Mirror of training_profiles/train_fm.py::checkpoint_payload."""
    cond_mean, cond_std, cond_min, cond_max = cond_stats
    return {
        'schema_version': 'sdfflow_infer_v1', 'stage': 'fm', 'epoch': 0,
        'model_state': fm.state_dict(), 'ema_state': None, 'config': dict(config),
        'vae_modelpath': vae_path,
        'latent_flat_dim': int(latent_mean.shape[1]),
        'latent_mean': latent_mean.cpu(), 'latent_std': latent_std.cpu(),
        'cond_dim': len(cond_names),
        'cond_mean': cond_mean, 'cond_std': cond_std, 'cond_min': cond_min, 'cond_max': cond_max,
        'cond_clip': 5.0, 'cond_names': list(cond_names),
    }


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    """Synthetic dataset + random VAE + two random FMs ('all' on volume,area;
    'per_dim' on volume,area,<FEA>) saved as real checkpoints."""
    root = tmp_path_factory.mktemp('sdfflow_cond')
    dataset_path = _write_synthetic_dataset(str(root / 'synthetic12.h5'))
    config = _tiny_config(dataset_path)
    torch.manual_seed(0)
    vae = SDFVAE(config)
    # Widen the scalar head so the random SDF field has zero crossings that
    # Marching Cubes turns into a few blobs instead of +-1e-5 noise.
    with torch.no_grad():
        torch.nn.init.normal_(vae.decoder.out.weight, std=0.05)
    vae.eval()
    vae_path = str(root / 'sdfflow_vae.pth')
    torch.save(_vae_payload(vae, config), vae_path)

    train_ds, val_ds, test_ds = build_dataset_splits(dict(config), 42)
    try:
        train_ds.deterministic = True
        latents, conds = [], []
        with torch.no_grad():
            for i in range(len(train_ds)):
                item = train_ds[i]
                mu, _ = vae.encode(item['surface_points'][None], item['surface_normals'][None])
                latents.append(mu.flatten(1))
                conds.append(item['cond'])
        z = torch.cat(latents)
        latent_mean = z.mean(dim=0, keepdim=True)
        latent_std = z.std(dim=0, keepdim=True).clamp_min(1e-6)
        dataset_cond_names = list(train_ds.cond_names)
        cond_all = torch.stack(conds).double()
    finally:
        for ds in (train_ds, val_ds, test_ds):
            ds.close()

    def cond_stats(names):
        idx = [dataset_cond_names.index(n) for n in names]
        c = cond_all[:, idx].float()
        return (c.mean(0, keepdim=True), c.std(0, keepdim=True).clamp_min(1e-6),
                c.amin(0, keepdim=True), c.amax(0, keepdim=True))

    fms = {}
    for tag, mode, names in (('all', 'all', list(GEOM_NAMES)),
                             ('per_dim', 'per_dim', list(GEOM_NAMES) + [FEA_NAME])):
        fm_config = dict(config, cond_dropout_mode=mode, use_conditions=True,
                         condition_names=names)
        gen = torch.Generator().manual_seed(1 if tag == 'all' else 2)
        torch.manual_seed(3)
        fm = VelocityNet(fm_config, int(z.shape[1]), cond_dim=len(names))
        _randomize_zero_init(fm, std=0.05, generator=gen)
        fm.eval()
        path = str(root / f'sdfflow_fm_{tag}.pth')
        torch.save(_fm_payload(fm, fm_config, vae_path, latent_mean, latent_std,
                               cond_stats(names), names), path)
        fms[tag] = path
    return {'root': root, 'dataset': dataset_path, 'config': config, 'vae_path': vae_path,
            'fm_all': fms['all'], 'fm_per_dim': fms['per_dim'],
            'latent_flat_dim': int(z.shape[1]), 'dataset_cond_names': dataset_cond_names}


class SphereVAE(SDFVAE):
    """Same parameters / checkpoint format as SDFVAE, but `decode_flat` is an
    analytic sphere whose radius is a smooth function of the first latent
    coordinate: radius = 0.2 + 0.5 * sigmoid(z[:, 0]) in (0.2, 0.7). Marching
    Cubes then yields watertight meshes with known volume / area, so the
    calibration slope and the Newton correction can be checked for real."""

    def decode_flat(self, z_flat, query_points):
        z_flat = torch.as_tensor(z_flat)
        if z_flat.dim() == 1:
            z_flat = z_flat.unsqueeze(0)
        radius = 0.2 + 0.5 * torch.sigmoid(z_flat[:, 0].float())
        pts = query_points if query_points.dim() == 3 else query_points.unsqueeze(0)
        if pts.shape[0] == 1 and z_flat.shape[0] > 1:
            pts = pts.expand(z_flat.shape[0], -1, -1)
        return pts.norm(dim=-1) - radius[:, None].to(pts.dtype)


@pytest.fixture
def sphere_decoder(monkeypatch):
    """Route every loader (`load_vae` in sample.py; interpolate/evaluate import
    it from there) through SphereVAE for the duration of one test."""
    monkeypatch.setattr(sample_mod, 'SDFVAE', SphereVAE)
    return SphereVAE


def _load_fm(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Keep every test on the CPU, on a CUDA box too.

    `training_profiles/setup.py::resolve_device` has no "use the CPU" key: it
    takes the CUDA branch whenever `torch.cuda.is_available()`, and the models
    here are tiny enough that a GPU round trip only adds context-init latency
    (and would make the 8 GB dev card contend with a real run). Patching the
    availability probe rather than `resolve_device` itself keeps the production
    resolver in the call path.
    """
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


# ---------------------------------------------------------------------------
# (1) partial conditions
# ---------------------------------------------------------------------------

def test_native_parser_keeps_nan_entries_as_strings():
    parsed = parse_value('0.25,nan,6.0')
    assert parsed == ['0.25', 'nan', '6.0']
    names = ['volume', 'area', FEA_NAME]
    raw = sample_mod.parse_condition_values(parsed, names)
    assert raw[0] == 0.25 and math.isnan(raw[1]) and raw[2] == 6.0
    with pytest.raises(ValueError, match='entries'):
        sample_mod.parse_condition_values(['0.25', 'nan'], names)
    with pytest.raises(ValueError, match='not a number'):
        sample_mod.parse_condition_values(['0.25', 'abc', '6.0'], names)


def test_partial_condition_mask_and_legacy_checkpoint_error(world):
    per_dim = _load_fm(world['fm_per_dim'])
    legacy = _load_fm(world['fm_all'])
    names = per_dim['cond_names']
    raw = sample_mod.parse_condition_values(['0.25', 'nan', 'nan'], names)
    mask = sample_mod.condition_mask_from_values(raw, per_dim)
    assert mask.dtype == torch.bool and mask.tolist() == [True, False, False]
    cond_n, mask2, request = sample_mod.normalize_condition_request(raw, per_dim, {})
    assert torch.equal(mask, mask2)
    assert request['partial'] is True and request['cond_mask'] == [True, False, False]
    assert request['specified'] == {'volume': True, 'area': False, FEA_NAME: False}
    assert request['raw']['area'] is None and cond_n[1].item() == 0.0 and cond_n[2].item() == 0.0
    assert request['cond_dropout_mode'] == 'per_dim'

    raw_legacy = sample_mod.parse_condition_values(['0.25', 'nan'], legacy['cond_names'])
    with pytest.raises(ValueError, match="cond_dropout_mode 'all'"):
        sample_mod.condition_mask_from_values(raw_legacy, legacy)
    with pytest.raises(ValueError, match='unconditional'):
        sample_mod.condition_mask_from_values(np.array([np.nan, np.nan, np.nan]), per_dim)
    # A fully specified request needs no per_dim network.
    full = sample_mod.condition_mask_from_values(np.array([0.25, 6.0]), legacy)
    assert full.tolist() == [True, True]


def test_run_sample_partial_request_end_to_end(world, tmp_path):
    out_dir = tmp_path / 'samples'
    config = {'fm_modelpath': world['fm_per_dim'], 'output_dir': str(out_dir), 'num_samples': 1,
              'seed': 0, 'ode_steps': ODE_STEPS, 'mc_resolution': MC_RES, 'cfg_scale': 1.0,
              'candidate_multiplier': 2, 'gpu_ids': 0,
              'cond_values': ['0.25', 'nan', 'nan'], 'condition_ood_policy': 'clamp',
              'condition_audit': 'fea'}   # fea backend: gmsh may be present, but
    #                                       the request leaves the FEA name unspecified
    #                                       and the meshes are random blobs; the audit
    #                                       must either run or fall back cleanly.
    meta = sample_mod.run_sample(config)
    assert meta['condition_request']['partial'] is True
    assert meta['condition_request']['cond_mask'] == [True, False, False]
    assert meta['num_candidates'] == 2 and meta['velocity_net_calls'] == ODE_STEPS
    assert meta['nfe_per_candidate'] == pytest.approx(ODE_STEPS)
    assert meta['condition_audit_backend']['requested'] == 'fea'
    assert meta['condition_audit_backend']['used'] in ('fea', 'geometric')
    assert meta['condition_audit']['not_measurable_geometrically'] == [FEA_NAME]
    assert len(meta['results']) == 1
    meta_path = out_dir / 'sample_0_meta.json'
    assert meta_path.exists()
    on_disk = json.loads(meta_path.read_text())
    assert on_disk['cond_names'] == ['volume', 'area', FEA_NAME]

    # The same request against the legacy 'all' checkpoint is refused clearly.
    config_legacy = dict(config, fm_modelpath=world['fm_all'], cond_values=['0.25', 'nan'],
                         condition_audit='geometric')
    with pytest.raises(ValueError, match="cond_dropout_mode 'all'"):
        sample_mod.run_sample(config_legacy)


def test_resolve_condition_audit_falls_back_with_one_message(capsys):
    used, reason = sample_mod.resolve_condition_audit({'condition_audit': 'fea'}, ['volume', 'area'])
    assert used == 'geometric' and 'no FEA-named' in reason
    used, reason = sample_mod.resolve_condition_audit(
        {'condition_audit': 'surrogate'}, ['volume', FEA_NAME])
    assert used == 'geometric' and 'unavailable' in reason
    out = capsys.readouterr().out
    assert 'Falling back to the geometric audit' in out
    with pytest.raises(ValueError):
        sample_mod.resolve_condition_audit({'condition_audit': 'nope'}, ['volume'])


def test_load_cases_used_reads_bracket_table_at_runtime():
    pytest.importorskip('scipy')
    from design_loop.problem import Bracket
    used = sample_mod.load_cases_used()
    assert set(used) == set(Bracket.LOAD_CASES)
    for name, spec in Bracket.LOAD_CASES.items():
        entry = used[name]
        assert entry['kind'] == spec['kind']
        if spec['kind'] == 'force':
            assert entry['magnitude_n'] == pytest.approx(float(np.linalg.norm(spec['vector'])))
        else:
            assert entry['magnitude_nm'] == pytest.approx(float(spec['magnitude']))
    assert sample_mod.FEA_AUDIT_LABEL == 'tet4_linear_relative_only'


def test_convert_structural_result_units():
    result = {'cases': {'vertical': {'max_von_mises': 250e6, 'max_displacement': 0.0021}}}
    physical = {'mass_kg': 1.5, 'volume_mm3': 340000.0, 'surface_area_mm2': 51000.0}
    values = sample_mod.convert_structural_result(
        result, [FEA_NAME, 'log_max_ver_magdisp_mm', 'mass_kg', 'log_max_hor_stress_mpa'], physical)
    assert values[FEA_NAME] == pytest.approx(250.0)                # Pa -> MPa
    assert values['log_max_ver_magdisp_mm'] == pytest.approx(2.1)  # m -> mm
    assert values['mass_kg'] == 1.5
    assert values['log_max_hor_stress_mpa'] is None                # case not analysed
    stored = sample_mod._stored_values(values)
    assert stored[FEA_NAME] == pytest.approx(math.log(250.0))
    assert stored['log_max_hor_stress_mpa'] is None


def test_convert_structural_result_picks_the_percentile_stress_measure():
    """A stress condition is scored on `peak_von_mises` -- the measure
    `opt_stress_percentile` configures -- not the element maximum, which on a
    tet4 mesh is a corner singularity that keeps climbing with refinement. Both
    backends report both, and a result carrying only one falls back to it."""
    both = {'cases': {'vertical': {'peak_von_mises': 180e6, 'max_von_mises': 420e6,
                                   'max_displacement': 0.001}}}
    assert sample_mod.DEFAULT_STRESS_MEASURE == 'peak_von_mises'
    scored = sample_mod.convert_structural_result(both, [FEA_NAME], {})
    assert scored[FEA_NAME] == pytest.approx(180.0)
    alt = sample_mod.convert_structural_result(both, [FEA_NAME], {},
                                               stress_measure='max_von_mises')
    assert alt[FEA_NAME] == pytest.approx(420.0)
    only_max = {'cases': {'vertical': {'max_von_mises': 420e6}}}
    assert sample_mod.convert_structural_result(only_max, [FEA_NAME], {})[FEA_NAME] \
        == pytest.approx(420.0)


def test_fea_condition_audit_scores_a_mass_name_without_touching_the_solver():
    """The audit's conversion layer, exercised without gmsh: a `mass` name is
    pure mesh arithmetic, so `solver_needed` is False and no meshing happens.
    Also pins `attach_fea_audit`'s relative error, which is computed in RAW
    units against a stored-space request."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    cond_names = ['volume', 'mass_kg']
    scale, rho = 0.19 / 1.8, 4430.0
    entries, meta = sample_mod.fea_condition_audit(
        [sphere, None], cond_names, 'fea',
        {'opt_length_scale': scale, 'opt_material_rho': rho})
    assert meta['backend'] == 'fea' and meta['label'] == sample_mod.FEA_AUDIT_LABEL
    assert meta['load_cases_needed'] == [] and meta['load_cases_used'] is None
    assert meta['measured_names'] == ['mass_kg'] and meta['not_measurable'] == []
    assert meta['stress_measure'] == 'peak_von_mises'
    assert entries[1] is None                      # an invalid decode stays None

    entry = entries[0]
    expected_kg = abs(sphere.volume) * scale ** 3 * rho
    assert entry['values_raw']['mass_kg'] == pytest.approx(expected_kg, rel=1e-9)
    assert entry['values_stored']['mass_kg'] == pytest.approx(expected_kg)  # identity transform
    assert entry['physical']['volume_mm3'] == pytest.approx(
        abs(sphere.volume) * (scale * 1e3) ** 3)
    assert entry['error'] is None
    assert entry['values_raw_max_von_mises'] == {}  # no stress name was requested

    report = {'valid': True}
    target_stored = np.array([np.nan, expected_kg * 1.1])
    sample_mod.attach_fea_audit(report, entry, cond_names, target_stored)
    assert report['fea_condition_rel_error_raw']['mass_kg'] == pytest.approx(0.1 / 1.1, rel=1e-6)
    assert report['fea_condition_abs_error_stored']['mass_kg'] == pytest.approx(
        abs(expected_kg * 0.1), rel=1e-6)
    assert 'volume' not in report['fea_condition_rel_error_raw']   # geometric, not FEA


def test_condition_summary_reports_the_request_even_when_nothing_is_auditable():
    """Every decode invalid (or volume unavailable on a non-watertight mesh)
    must still leave the reader the request shape: which names were asked for
    and which ones a geometric audit could never have scored."""
    cond_names = ['volume', 'area', FEA_NAME]
    target = np.array([0.25, 6.0, 5.9])
    mask = np.array([True, False, True])
    summary = sample_mod._condition_summary([{'valid': False}], cond_names, target, mask)
    assert summary['audited_meshes'] == 0
    assert summary['not_measurable_geometrically'] == [FEA_NAME]
    assert summary['specified_names'] == ['volume', FEA_NAME]
    assert summary['audited_names'] == [] and summary['median_rel_error'] == {}
    assert summary['best_condition_score'] is None


def test_descriptor_targets_drops_names_without_a_soft_proxy(capsys):
    """`guidance_targets bbox_x` is geometric but has no differentiable proxy;
    it must be dropped with a note (leaving guidance inactive) rather than
    exploding inside make_c2_guidance."""
    cond_names = ['bbox_x', 'volume', FEA_NAME]
    raw = np.array([1.8, 0.25, 250.0])
    assert 'volume' in sample_mod.soft_proxy_names()
    assert 'bbox_x' not in sample_mod.soft_proxy_names()
    targets = sample_mod.descriptor_targets(cond_names, raw, requested='bbox_x,volume,' + FEA_NAME)
    assert targets == {'volume': pytest.approx(0.25)}
    out = capsys.readouterr().out
    assert 'no soft proxy' in out and 'FEA-named targets' in out
    # Nothing proxyable left -> an empty dict, which run_sample reads as "inactive".
    assert sample_mod.descriptor_targets(cond_names, raw, requested='bbox_x') == {}
    # An unspecified (nan) entry is dropped too, whatever else is requested.
    partial = np.array([1.8, np.nan, 250.0])
    assert sample_mod.descriptor_targets(cond_names, partial,
                                         mask=np.array([True, False, True])) == {}


# ---------------------------------------------------------------------------
# (2) cond_sweep
# ---------------------------------------------------------------------------

def test_integrate_cond_sweep_endpoints_match_plain_samples(world):
    ckpt = _load_fm(world['fm_per_dim'])
    D = world['latent_flat_dim']
    model = VelocityNet(ckpt['config'], D, cond_dim=ckpt['cond_dim'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    eps = torch.randn(3, D, generator=torch.Generator().manual_seed(7))[1]
    cond_a = torch.tensor([-1.0, 0.5, 0.0])
    cond_b = torch.tensor([1.5, -0.5, 0.0])
    mask = torch.tensor([True, True, False])
    alphas = interpolate_mod.sweep_alphas(4)
    stack = interpolate_mod.sweep_conditions(cond_a, cond_b, alphas)
    assert torch.equal(stack[0], cond_a) and torch.equal(stack[-1], cond_b)
    z = interpolate_mod.integrate_cond_sweep(model, eps, stack, mask, 'cpu', ode_steps=ODE_STEPS)
    assert z.shape == (4, D)
    for k, cond in ((0, cond_a), (3, cond_b)):
        plain = sample_latents(model, 1, D, 'cpu', cond=cond[None], ode_steps=ODE_STEPS,
                               noise=eps[None], cond_mask=mask[None])
        assert torch.allclose(z[k:k + 1], plain, rtol=1e-4, atol=1e-5)
    # The condition moves the latent: every step is distinct.
    for i in range(4):
        for j in range(i + 1, 4):
            assert not torch.allclose(z[i], z[j], atol=1e-6)
    with pytest.raises(ValueError):
        interpolate_mod.sweep_alphas(1)


def test_run_interpolate_cond_sweep_writes_stls_strip_and_metadata(world, tmp_path):
    out_dir = tmp_path / 'sweep'
    ckpt = _load_fm(world['fm_per_dim'])
    mean = ckpt['cond_mean'].squeeze(0)
    std = ckpt['cond_std'].squeeze(0)
    a = (mean - 0.5 * std).tolist()
    b = (mean + 0.5 * std).tolist()
    config = {'fm_modelpath': world['fm_per_dim'], 'output_dir': str(out_dir), 'seed': 3,
              'source_num_samples': 2, 'sample_index_a': 1, 'sample_index_b': 0,
              'interpolation_space': 'cond_sweep', 'sweep_steps': 3, 'ode_steps': ODE_STEPS,
              'mc_resolution': MC_RES, 'plot_dpi': 40, 'gpu_ids': 0,
              'cond_values_a': [f'{a[0]:.6f}', f'{a[1]:.6f}', 'nan'],
              'cond_values_b': [f'{b[0]:.6f}', f'{b[1]:.6f}', 'nan']}
    meta = interpolate_mod.run_interpolate(config)
    assert meta['interpolation_space'] == 'cond_sweep'
    assert meta['sweep_steps'] == 3 and meta['alphas'] == [0.0, 0.5, 1.0]
    assert meta['partial'] is True and meta['cond_mask'] == [True, True, False]
    assert meta['cond_dropout_mode'] == 'per_dim'
    assert len(meta['steps']) == 3 and len(meta['body_count_raw']) == 3
    for k, step in enumerate(meta['steps']):
        assert step['step'] == k
        assert set(step['requested_raw']) == {'volume', 'area', FEA_NAME}
        assert step['requested_raw'][FEA_NAME] is None
        assert 'not_measurable_geometrically' in step and step['not_measurable_geometrically'] == [FEA_NAME]
        if step['valid']:
            assert os.path.exists(step['path'])
            assert os.path.basename(step['path']) == f'sample_3_sweep_{k}.stl'
    assert meta['steps'][0]['requested_raw']['volume'] == pytest.approx(a[0], rel=1e-5)
    assert meta['steps'][-1]['requested_raw']['volume'] == pytest.approx(b[0], rel=1e-5)
    assert meta['steps'][1]['requested_raw']['volume'] == pytest.approx(0.5 * (a[0] + b[0]), rel=1e-5)
    # `steps` is the compact per-panel view and `results` the full mesh
    # reports: two different shapes, not the same dicts serialized twice.
    assert len(meta['results']) == 3
    assert all('extents' not in s for s in meta['steps'])
    assert any('extents' in r for r in meta['results'] if r['valid'])
    assert os.path.exists(meta['plot_path'])
    assert os.path.exists(out_dir / 'cond_sweep_3_001_meta.json')
    assert meta['velocity_net_calls'] == ODE_STEPS
    assert all(d > 0 for d in meta['latent_distances']['consecutive_l2'])
    assert 'condition_request_a' in meta and 'condition_request_b' in meta

    # Endpoint 0 reproduces a plain conditional sample from the same noise row:
    # decode that sample and compare with the exported STL.
    D = world['latent_flat_dim']
    model = VelocityNet(ckpt['config'], D, cond_dim=ckpt['cond_dim'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    eps = torch.randn(2, D, generator=torch.Generator().manual_seed(3))[1]
    raw_a = sample_mod.parse_condition_values(config['cond_values_a'], ckpt['cond_names'])
    cond_a_n, mask, _ = sample_mod.normalize_condition_request(raw_a, ckpt, {})
    z_plain = sample_latents(model, 1, D, 'cpu', cond=cond_a_n[None], ode_steps=ODE_STEPS,
                             noise=eps[None], cond_mask=mask[None])
    vae, _ = sample_mod.load_vae(world['vae_path'], 'cpu')
    from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh
    grid = decode_sdf_grid(vae, z_plain * ckpt['latent_std'] + ckpt['latent_mean'], resolution=MC_RES)
    mesh = sdf_grid_to_mesh(grid)
    if meta['steps'][0]['valid'] and mesh is not None:
        exported = trimesh.load(meta['steps'][0]['path'], force='mesh')
        assert exported.volume == pytest.approx(mesh.volume, abs=1e-4)
        assert exported.area == pytest.approx(mesh.area, rel=1e-4)


def test_cond_sweep_rejects_mismatched_masks_and_legacy_partial(world, tmp_path):
    base = {'fm_modelpath': world['fm_per_dim'], 'output_dir': str(tmp_path / 'x'), 'seed': 0,
            'source_num_samples': 1, 'sample_index_a': 0, 'interpolation_space': 'cond_sweep',
            'sweep_steps': 2, 'ode_steps': 2, 'mc_resolution': 8, 'gpu_ids': 0}
    with pytest.raises(ValueError, match='SAME entries'):
        interpolate_mod.run_interpolate(dict(base, cond_values_a=['0.2', '5.0', 'nan'],
                                             cond_values_b=['0.3', 'nan', 'nan']))
    with pytest.raises(ValueError, match='cond_values_a and cond_values_b'):
        interpolate_mod.run_interpolate(dict(base))
    with pytest.raises(ValueError, match='not cond_values'):
        interpolate_mod.run_interpolate(dict(base, cond_values=['0.2', '5.0', '6.0'],
                                             cond_values_a=['0.2', '5.0', '6.0'],
                                             cond_values_b=['0.3', '5.0', '6.0']))
    with pytest.raises(ValueError, match="cond_dropout_mode 'all'"):
        interpolate_mod.run_interpolate(dict(base, fm_modelpath=world['fm_all'],
                                             cond_values_a=['0.2', 'nan'], cond_values_b=['0.3', 'nan']))
    with pytest.raises(ValueError, match='sweep_steps'):
        interpolate_mod.run_interpolate(dict(base, sweep_steps=1, cond_values_a=['0.2', '5.0', '6.0'],
                                             cond_values_b=['0.3', '5.0', '6.0']))


def test_legacy_interpolation_spaces_still_reject_conditions(world, tmp_path):
    base = {'fm_modelpath': world['fm_all'], 'output_dir': str(tmp_path / 'legacy'), 'seed': 0,
            'source_num_samples': 2, 'sample_index_a': 0, 'sample_index_b': 1,
            'ode_steps': 2, 'mc_resolution': 8, 'gpu_ids': 0, 'cond_values': ['0.2', '5.0']}
    for space in ('slerp_noise', 'lerp_latent'):
        with pytest.raises(ValueError, match='unconditional samples only'):
            interpolate_mod.run_interpolate(dict(base, interpolation_space=space))


# ---------------------------------------------------------------------------
# (3) eval_task conditional
# ---------------------------------------------------------------------------

def _conditional_config(world, out_dir, methods, **overrides):
    config = {'fm_modelpath': world['fm_per_dim'], 'dataset_dir': world['dataset'],
              'output_dir': str(out_dir), 'eval_task': 'conditional', 'eval_split': 'train',
              'eval_num_shapes': 2, 'eval_seed': 1, 'mc_resolution': MC_RES, 'ode_steps': ODE_STEPS,
              'candidate_multiplier': 3, 'eval_methods': methods, 'gpu_ids': 0,
              'soft_descriptor_resolution': SOFT_RES, 'soft_descriptor_tau': SOFT_TAU,
              'newton_measure_resolution': MC_RES, 'newton_rounds': 2}
    config.update(overrides)
    return config


CONTRACT_AGG_KEYS = {'num_shapes', 'valid_rate', 'watertight_rate', 'latent_rms_drift',
                     'nfe_per_output', 'velocity_net_calls', 'seconds', 'per_condition'}


def test_evaluate_conditional_plain_rejection_writes_json_and_csv(world, tmp_path):
    out_dir = tmp_path / 'eval_cond'
    summary = evaluate_mod.run_evaluate(_conditional_config(world, out_dir, ['plain', 'rejection']))
    assert summary['eval_task'] == 'conditional'
    assert summary['methods'] == ['plain', 'rejection']
    assert summary['num_shapes_evaluated'] == 2
    assert summary['cond_names'] == ['volume', 'area', FEA_NAME]
    assert summary['not_measurable_geometrically'] == [FEA_NAME]
    assert summary['condition_audit_backend']['used'] == 'geometric'
    for method in ('plain', 'rejection'):
        agg = summary['aggregate'][method]
        assert CONTRACT_AGG_KEYS <= set(agg)
        assert agg['num_shapes'] == 2
        for name in ('volume', 'area', FEA_NAME):
            pc = agg['per_condition'][name]
            assert {'median_rel_error', 'p95_rel_error', 'n'} <= set(pc)
        assert agg['per_condition'][FEA_NAME]['n'] == 0
        assert agg['per_condition'][FEA_NAME]['measured_by'] == 'not measurable geometrically'
        # area is always measurable on a valid mesh
        assert agg['per_condition']['area']['n'] == agg['valid_rate'] * 2
    plain_rows = [r for r in summary['rows'] if r['method'] == 'plain']
    rej_rows = [r for r in summary['rows'] if r['method'] == 'rejection']
    assert len(plain_rows) == 2 and len(rej_rows) == 2
    for row in plain_rows:
        assert row['latent_rms_drift'] == 0.0
        assert row['velocity_net_calls'] == ODE_STEPS and row['nfe_per_output'] == ODE_STEPS
        assert row[f'target_{FEA_NAME}'] > 300.0       # raw MPa, not the stored log
    for row in rej_rows:
        assert row['candidate_multiplier'] == 3
        assert row['nfe_per_output'] == 3 * ODE_STEPS  # trajectory-equivalent NFE per retained output
        assert 0 <= row['selected_candidate'] < 3
        if row['selected_candidate'] == 0:
            assert row['latent_rms_drift'] == pytest.approx(0.0, abs=1e-6)
        else:
            assert row['latent_rms_drift'] > 0
    # Both methods paired on the same shapes.
    assert [r['shape_idx'] for r in plain_rows] == [r['shape_idx'] for r in rej_rows]

    json_path = out_dir / 'eval_conditional.json'
    csv_path = out_dir / 'eval_conditional.csv'
    assert json_path.exists() and csv_path.exists()
    on_disk = json.loads(json_path.read_text())
    assert on_disk['aggregate']['plain']['per_condition']['volume'].keys() >= {'median_rel_error', 'p95_rel_error'}
    with open(csv_path, newline='') as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 4
    assert {'method', 'shape_idx', 'valid', 'watertight', 'latent_rms_drift', 'nfe_per_output',
            'seconds', 'rel_error_volume', 'rel_error_area', f'rel_error_{FEA_NAME}'} <= set(csv_rows[0])


def test_evaluate_conditional_validates_inputs(world, tmp_path):
    with pytest.raises(ValueError, match='unknown method'):
        evaluate_mod.parse_eval_methods('plain,bogus')
    assert evaluate_mod.parse_eval_methods(None) == ['plain', 'rejection', 'e2']
    assert evaluate_mod.parse_eval_methods(['e2', 'plain', 'e2']) == ['e2', 'plain']
    assert evaluate_mod.parse_eval_methods('c2e2') == ['c2e2']
    cfg = _conditional_config(world, tmp_path / 'v', ['plain'])
    del cfg['fm_modelpath']
    with pytest.raises(ValueError, match='fm_modelpath'):
        evaluate_mod.run_evaluate(cfg)
    with pytest.raises(ValueError, match='eval_task'):
        evaluate_mod.run_evaluate(dict(cfg, eval_task='nope'))
    assert evaluate_mod.select_shapes(10, 0, 0) == list(range(10))
    assert evaluate_mod.select_shapes(10, 20, 0) == list(range(10))
    sub = evaluate_mod.select_shapes(10, 3, 5)
    assert len(sub) == 3 and sub == sorted(sub) and sub == evaluate_mod.select_shapes(10, 3, 5)
    g1 = evaluate_mod.shape_generator(1, 7, 'cpu')
    g2 = evaluate_mod.shape_generator(1, 7, 'cpu')
    assert torch.equal(torch.randn(4, generator=g1), torch.randn(4, generator=g2))


# ---------------------------------------------------------------------------
# (4) eval_task descriptor_calibration, then e2 / c2 / c2e2 on it
# ---------------------------------------------------------------------------

@pytest.fixture
def calibration_path(world, tmp_path, sphere_decoder):
    out_dir = tmp_path / 'calib'
    path = str(out_dir / 'descriptor_calibration.pth')
    config = {'fm_modelpath': world['fm_per_dim'], 'dataset_dir': world['dataset'],
              'output_dir': str(out_dir), 'eval_task': 'descriptor_calibration',
              'eval_split': 'train', 'eval_seed': 0, 'calibration_num_shapes': 3,
              'calibration_samples_per_shape': 2, 'mc_resolution': MC_RES, 'ode_steps': ODE_STEPS,
              'soft_descriptor_resolution': SOFT_RES, 'soft_descriptor_tau': SOFT_TAU,
              'descriptor_calibration_path': path, 'gpu_ids': 0}
    summary = evaluate_mod.run_evaluate(config)
    assert summary['eval_task'] == 'descriptor_calibration'
    assert summary['rows_total'] == 6 and summary['rows_valid'] == 6
    assert os.path.exists(path)
    assert (out_dir / 'eval_descriptor_calibration.json').exists()
    return path


@needs_descriptor_tools
def test_descriptor_calibration_task_writes_a_loadable_compatible_calibration(world, calibration_path):
    cal = _dcal.DescriptorCalibration.load(calibration_path)
    assert cal.names == ('volume', 'area')
    assert cal.resolution == SOFT_RES and cal.tau == pytest.approx(SOFT_TAU)
    assert cal.measure_resolution == MC_RES
    assert cal.cond_names == ['volume', 'area', FEA_NAME]
    assert cal.split == 'train' and cal.num_shapes == 3 and cal.samples_per_shape == 2
    assert cal.vae_sha256 and cal.fm_sha256
    # On analytic spheres the soft proxy is ~linear in the truth with slope ~1.
    assert cal['volume']['r2'] > 0.95 and 0.7 < cal['volume']['a'] < 1.3
    assert cal['volume']['n'] == 6
    assert cal.check_compatible(world['vae_path'], world['fm_per_dim'], SOFT_RES, SOFT_TAU)
    with pytest.raises(ValueError, match='sha256'):
        cal.check_compatible(world['vae_path'], world['fm_all'], SOFT_RES, SOFT_TAU)
    with pytest.raises(ValueError, match='resolution'):
        cal.check_compatible(world['vae_path'], world['fm_per_dim'], SOFT_RES + 8, SOFT_TAU)
    assert cal.extra['eval_task'] == 'descriptor_calibration'
    assert len(cal.extra['shape_indices']) == 3


@needs_descriptor_tools
def test_evaluate_conditional_with_e2_c2_c2e2(world, tmp_path, sphere_decoder, calibration_path):
    out_dir = tmp_path / 'eval_cond_e2'
    config = _conditional_config(world, out_dir, ['plain', 'e2', 'c2', 'c2e2'],
                                 descriptor_calibration_path=calibration_path,
                                 guidance_eta=0.2, guidance_t_start=0.3)
    summary = evaluate_mod.run_evaluate(config)
    assert summary['methods'] == ['plain', 'e2', 'c2', 'c2e2']
    assert summary['newton']['rounds'] == 2 and summary['guidance']['eta'] == 0.2
    assert summary['proxy_target_names'] == ['volume', 'area']
    agg = summary['aggregate']
    for method in ('plain', 'e2', 'c2', 'c2e2'):
        assert agg[method]['valid_rate'] == 1.0 and agg[method]['watertight_rate'] == 1.0
    e2_rows = [r for r in summary['rows'] if r['method'] == 'e2']
    plain_rows = [r for r in summary['rows'] if r['method'] == 'plain']
    for e2, plain in zip(e2_rows, plain_rows):
        assert e2['shape_idx'] == plain['shape_idx']
        assert e2['newton_residual_final'] <= e2['newton_residual_initial'] + 1e-12
        assert e2['newton_mc_measurements'] >= 1
        assert e2['velocity_net_calls'] == plain['velocity_net_calls']  # Newton adds no FM calls
        if e2['newton_accepted_steps']:
            assert e2['latent_rms_drift'] > 0
            assert e2['rel_error_volume'] < plain['rel_error_volume']
    c2_rows = [r for r in summary['rows'] if r['method'] == 'c2']
    for c2 in c2_rows:
        assert c2['guidance_fm_evaluations'] >= 1
        assert c2['velocity_net_calls'] == ODE_STEPS + c2['guidance_fm_evaluations']
        assert c2['nfe_per_output'] > ODE_STEPS
        assert c2['latent_rms_drift'] > 0
    assert any(d.get('newton') for d in summary['details'])
    assert (out_dir / 'eval_conditional.csv').exists()
    with open(out_dir / 'eval_conditional.csv', newline='') as f:
        assert len(list(csv.DictReader(f))) == 8


@needs_descriptor_tools
def test_run_sample_with_guidance_and_newton(world, tmp_path, sphere_decoder, calibration_path):
    out_dir = tmp_path / 'guided'
    from general_modules.descriptor_proxy import sphere_volume
    target_volume = sphere_volume(0.45)
    config = {'fm_modelpath': world['fm_per_dim'], 'output_dir': str(out_dir), 'num_samples': 1,
              'seed': 0, 'ode_steps': ODE_STEPS, 'mc_resolution': MC_RES, 'cfg_scale': 1.0,
              'candidate_multiplier': 1, 'gpu_ids': 0,
              'cond_values': [f'{target_volume:.6f}', 'nan', 'nan'],
              'condition_ood_policy': 'clamp', 'guidance_enabled': True, 'guidance_eta': 0.2,
              'guidance_targets': 'volume,area', 'newton_rounds': 2, 'newton_measure_resolution': MC_RES,
              'soft_descriptor_resolution': SOFT_RES, 'soft_descriptor_tau': SOFT_TAU,
              'descriptor_calibration_path': calibration_path}
    meta = sample_mod.run_sample(config)
    # The target came back through a 6-decimal config string, as a real config
    # would; compare against the request, not the full-precision constant.
    assert meta['guidance']['targets'] == {'volume': pytest.approx(target_volume, abs=1e-6)}
    assert meta['guidance']['step_mode'] == 'velocity_dt'
    # Live callback counters: the window really opened and cost what it cost.
    stats = meta['guidance']['stats']
    assert stats['calls'] == ODE_STEPS and 0 < stats['active_calls'] <= ODE_STEPS
    assert stats['fm_evaluations'] == stats['active_calls'] == stats['decoder_grids']
    assert stats['names'] == ['volume'] and stats['step_mode'] == 'velocity_dt'
    assert meta['newton']['rounds'] == 2
    assert meta['descriptor_calibration_path'] == calibration_path
    assert meta['velocity_net_calls'] > ODE_STEPS       # lookahead calls are counted
    result = meta['results'][0]
    assert result['valid'] and result['watertight']
    assert 'newton' in result and 'pre_newton' in result
    assert result['newton']['rounds'] == 2
    if result['newton']['accepted_steps']:
        assert result['condition_rel_error']['volume'] < result['pre_newton']['condition_rel_error']['volume']
    # A wrong calibration pairing is refused before anything is sampled. The
    # request is FULLY specified here on purpose: a partial one against the
    # 'all' checkpoint is refused earlier still, by the cond_mask check, which
    # would hide the calibration guard this asserts.
    with pytest.raises(ValueError, match='not compatible'):
        sample_mod.run_sample(dict(config, fm_modelpath=world['fm_all'],
                                   cond_values=[f'{target_volume:.6f}', '5.0'],
                                   output_dir=str(tmp_path / 'guided_bad')))
    with pytest.raises(FileNotFoundError):
        sample_mod.run_sample(dict(config, descriptor_calibration_path=str(tmp_path / 'missing.pth')))


def test_guidance_and_newton_are_inert_without_conditions(world, tmp_path, capsys):
    out_dir = tmp_path / 'inert'
    config = {'fm_modelpath': world['fm_all'], 'output_dir': str(out_dir), 'num_samples': 1,
              'seed': 5, 'ode_steps': 2, 'mc_resolution': 8, 'gpu_ids': 0,
              'guidance_enabled': True, 'newton_rounds': 3}
    meta = sample_mod.run_sample(dict(config, candidate_multiplier=4))
    assert meta['guidance'] is None and meta['newton'] is None
    assert meta['descriptor_calibration_path'] is None
    # An unconditional run cannot rank candidates, so the multiplier is a no-op
    # -- and says so rather than silently drawing num_samples.
    assert meta['num_candidates'] == 1
    out = capsys.readouterr().out
    assert 'inactive' in out and 'candidate_multiplier 4 needs cond_values' in out


# ---------------------------------------------------------------------------
# (5) the default task still works: eval_task reconstruction
# ---------------------------------------------------------------------------

def test_eval_task_reconstruction_is_the_default_and_still_runs(world, tmp_path):
    """`eval_task` dispatch must not have disturbed the default path. This is
    the only coverage `run_reconstruction_eval` has, so it also pins the
    refinement half-split arrangement the aggregate table compares across.
    """
    out_dir = tmp_path / 'eval_recon'
    config = {'vae_modelpath': world['vae_path'], 'dataset_dir': world['dataset'],
              'output_dir': str(out_dir), 'eval_split': 'train', 'eval_num_shapes': 2,
              'eval_seed': 0, 'mc_resolution': MC_RES, 'gpu_ids': 0,
              'latent_refine_steps': 3, 'latent_refine_lr': 0.01,
              'latent_refine_prior_weight': 0.0}
    summary = evaluate_mod.run_evaluate(config)          # eval_task omitted
    assert summary['num_shapes_evaluated'] == 2
    assert summary['refine_query_split']
    for prefix in ('enc', 'ref'):
        agg = summary['aggregate'][prefix]
        assert agg['num_shapes'] == 2
        for field in ('sdf_l1', 'sign_balanced_accuracy', 'chamfer_mean'):
            assert 'median' in agg[field]
        # The held-out halves are what enc and ref may be quoted against.
        assert 'sdf_l1_heldout' in agg
    row = summary['shapes'][0]
    assert row['enc_sdf_l1'] is not None and row['ref_latent_shift_l2'] is not None
    assert (out_dir / 'eval_train.json').exists() and (out_dir / 'eval_train.csv').exists()
    # An explicit reconstruction task takes the same path; without refinement
    # there is no ref_ column pair and no half-split at all.
    again = evaluate_mod.run_evaluate(dict(config, eval_task='reconstruction',
                                           output_dir=str(tmp_path / 'eval_recon2'),
                                           latent_refine_steps=0))
    assert again['num_shapes_evaluated'] == 2
    assert set(again['aggregate']) == {'enc'} and again['refine_query_split'] is None


def test_condition_score_penalises_an_unmeasurable_dimension():
    """A mean over fewer terms is not comparable; a torn decode must not win.

    Candidate A is watertight with a large volume error; candidate B is torn,
    so `volume` is unmeasurable and only `area` could be scored. Under the old
    plain mean over the scorable dimensions B scored 0.500 against A's 0.721
    and the ranking preferred the broken mesh.
    """
    from inference_profiles.sample import _audit_report

    cond_names = ['volume', 'area']
    target = np.array([0.30, 4.40])
    cond_std = np.array([0.10, 0.50])
    watertight = {'valid': True, 'watertight': True, 'volume': 0.40, 'area': 4.50,
                  'extents': [1.0, 1.8, 1.0]}
    torn = {'valid': True, 'watertight': False, 'volume': None, 'area': 4.65,
            'extents': [1.0, 1.8, 1.0]}
    _audit_report(watertight, cond_names, target, cond_std)
    _audit_report(torn, cond_names, target, cond_std)
    assert watertight['condition_score_support'] == [2, 2]
    assert torn['condition_score_support'] == [1, 2]
    assert torn['condition_score_unmeasurable'] == ['volume']
    assert torn['condition_score'] > watertight['condition_score']
    # With nothing missing the value is the plain RMS, unchanged.
    assert watertight['condition_score'] == pytest.approx(
        float(np.sqrt(np.mean(np.array([1.0, 0.2]) ** 2))))


# ---------------------------------------------------------------------------
# (6) review regressions: audit unit constants, E2 grid / latent-clip wiring,
#     and the evaluate-side exclusion / OOD / paired statistics
# ---------------------------------------------------------------------------

def test_fea_audit_defaults_to_the_deepjeb_label_constants():
    """The audit is scored against `bracket_labels.csv`, so its unit constants
    are the LABELS' own (183.8 mm mean longest extent, 4470 kg/m3), not the
    design-loop optimisation defaults (190 mm, 4430) -- those put a fixed
    +9.7% volume / +8.7% mass bias into every absolute-unit error on a perfect
    decode, and `mass_kg` needs no solver, so that bias would be the whole
    reported "error". `opt_length_scale` / `opt_material_rho` still override,
    and the metadata records which was used."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    entries, meta = sample_mod.fea_condition_audit([sphere], ['mass_kg'], 'fea', {})
    assert meta['length_scale_m_per_unit'] == pytest.approx(sample_mod.DEEPJEB_AUDIT_LENGTH_SCALE)
    assert meta['density_kg_m3'] == pytest.approx(sample_mod.DEEPJEB_LABEL_DENSITY)
    assert meta['length_scale_source'].startswith('deepjeb_label_calibrated')
    assert meta['density_source'].startswith('deepjeb_label_calibrated')
    assert 'estimator floor' in meta['scale_bias_note']
    calibrated = entries[0]['values_raw']['mass_kg']
    assert calibrated == pytest.approx(
        abs(sphere.volume) * sample_mod.DEEPJEB_AUDIT_LENGTH_SCALE ** 3
        * sample_mod.DEEPJEB_LABEL_DENSITY, rel=1e-9)

    # The design-loop constants are still reachable -- and still biased.
    legacy_entries, legacy_meta = sample_mod.fea_condition_audit(
        [sphere], ['mass_kg'], 'fea',
        {'opt_length_scale': sample_mod.DEFAULT_LENGTH_SCALE, 'opt_material_rho': 4430.0})
    assert legacy_meta['length_scale_source'] == 'opt_length_scale'
    assert legacy_meta['density_source'] == 'opt_material_rho'
    ratio = legacy_entries[0]['values_raw']['mass_kg'] / calibrated
    assert ratio == pytest.approx((0.19 / 0.1838) ** 3 * (4430.0 / 4470.0), rel=1e-9)
    assert 1.09 < ratio < 1.10      # the +9.7% median bias measured on the real labels


def test_newton_measure_resolution_follows_mc_resolution():
    """E2 must measure and accept on the grid it is SCORED on: the audit, the
    exported mesh and the calibration fit all use `mc_resolution`. An
    independent default of 96 made the loop drive one operator's residual to
    zero while the table reported another's (0.02-0.10% apart on a sphere,
    against a quoted E2 accuracy of 0.077-0.28%)."""
    assert sample_mod.newton_measure_resolution({'mc_resolution': 128}) == 128
    assert sample_mod.newton_measure_resolution({'mc_resolution': 96}) == 96
    assert sample_mod.newton_measure_resolution({}) == 128      # same default as mc_resolution
    # An explicit value still wins; the calibration then pins the mismatch.
    assert sample_mod.newton_measure_resolution(
        {'mc_resolution': 128, 'newton_measure_resolution': 96}) == 96


def test_apply_newton_forwards_the_latent_clip_and_the_measure_grid():
    """`latent_clip` bounds every arm of the benchmark; the Newton output used
    to be written back and decoded UNclipped, so E2 was credited with accuracy
    obtained in a latent region plain/rejection were forbidden to reach. The
    clip now goes into the line search (a candidate is clamped BEFORE it is
    measured) and is recorded next to the measure grid in the metadata."""
    from types import SimpleNamespace

    seen = {}

    def fake_newton_correct(vae, z_flat, targets, calibration, latent_mean, latent_std, **kw):
        seen.update(kw)
        seen['targets'] = dict(targets)
        return z_flat.clamp(-kw['latent_clip'], kw['latent_clip']), [
            {'row': 0, 'round': 0, 'step_accepted': True}]

    tools = SimpleNamespace(refinement=SimpleNamespace(newton_correct=fake_newton_correct))
    stack = SimpleNamespace(vae=None, latent_mean=torch.zeros(1, 4), latent_std=torch.ones(1, 4))
    z = torch.tensor([[6.0, -7.0, 0.5, 0.0]])
    config = {'newton_rounds': 2, 'latent_clip': 5.0, 'mc_resolution': 128,
              'newton_step_cap_rms': 0.12, 'newton_line_search_tries': 3}
    z_new, info = sample_mod.apply_newton(tools, stack, z, {'volume': 0.3}, object(), config)
    assert seen['latent_clip'] == 5.0
    assert seen['measure_resolution'] == 128             # follows mc_resolution
    assert seen['rounds'] == 2 and seen['targets'] == {'volume': 0.3}
    assert float(z_new.abs().max()) <= 5.0
    assert info['latent_clip'] == 5.0 and info['measure_resolution'] == 128
    assert info['accepted_steps'] == 1 and info['latent_rms_drift'] > 0

    # Inert without a proxy-backed target, or without rounds.
    z_out, info_out = sample_mod.apply_newton(tools, stack, z, {}, object(), config)
    assert info_out is None and z_out is z
    z_out, info_out = sample_mod.apply_newton(tools, stack, z, {'volume': 0.3}, object(),
                                              dict(config, newton_rounds=0))
    assert info_out is None and z_out is z


class _FakeSplit:
    """Minimal stand-in for an `SDFShapeDataset` split: positions -> h5 ids."""

    def __init__(self, indices):
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)


def test_eval_exclude_shapes_drops_before_the_seeded_subset_is_drawn():
    """DeepJEB h5 2099 is a partial STL carrying full-bracket labels: at n = 32
    one such shape owns the p95 column of every method. The exclusion has to
    act on the POOL, before the subset is drawn, or `eval_seed` decides whether
    the methodology applies at all (seeds 3, 8 and 10 pick 2099 up)."""
    assert evaluate_mod.excluded_shape_ids({}) == set()
    assert evaluate_mod.excluded_shape_ids({'eval_exclude_shapes': ''}) == set()
    assert evaluate_mod.excluded_shape_ids({'eval_exclude_shapes': 2099}) == {2099}
    assert evaluate_mod.excluded_shape_ids({'eval_exclude_shapes': '2099, 7'}) == {2099, 7}
    assert evaluate_mod.excluded_shape_ids({'eval_exclude_shapes': [2099, '7']}) == {2099, 7}
    with pytest.raises(ValueError, match='eval_exclude_shapes'):
        evaluate_mod.excluded_shape_ids({'eval_exclude_shapes': 'x'})

    split = _FakeSplit([10, 2099, 11, 12, 13])
    allowed, dropped = evaluate_mod.allowed_positions(split, {2099}, 'test')
    assert allowed == [0, 2, 3, 4] and dropped == [2099]
    # Whatever the seed, the excluded position never enters the sample, and
    # eval_num_shapes still means what it says.
    for seed in range(12):
        picked = evaluate_mod.select_shapes(len(split), 3, seed, allowed=allowed)
        assert len(picked) == 3 and 1 not in picked and picked == sorted(picked)
    assert evaluate_mod.select_shapes(len(split), 0, 0, allowed=allowed) == allowed
    with pytest.raises(ValueError, match='removed every shape'):
        evaluate_mod.allowed_positions(_FakeSplit([2099]), {2099}, 'test')


def _fake_fm_ckpt(mean, std, clip=5.0):
    return {'cond_mean': torch.tensor([mean]), 'cond_std': torch.tensor([std]), 'cond_clip': clip}


def test_normalize_true_conditions_honours_max_condition_z_and_the_ood_policy(capsys):
    """`max_condition_z` / `condition_ood_policy` used to be read by `mode
    sample` only, so the shipped evaluate config set both while a target beyond
    the checkpoint's `cond_clip` was silently clamped for SAMPLING and scored
    against the UNCLAMPED value -- a fixed, method-independent error floor with
    no visible signal."""
    ckpt = _fake_fm_ckpt([0.0, 0.0], [1.0, 1.0], clip=5.0)
    names = ['volume', 'area']
    raw = np.array([0.5, 6.0])          # area sits at z = 6.0, beyond cond_clip

    # No config: the legacy clamp, reported through `clipped`.
    cond_n, clipped, exceeded = evaluate_mod.normalize_true_conditions(raw, ckpt)
    assert clipped == [False, True] and exceeded == []
    assert float(cond_n[1]) == pytest.approx(5.0)

    # warn (the bulk-benchmark default): shouted about, never silent.
    cond_n, clipped, exceeded = evaluate_mod.normalize_true_conditions(
        raw, ckpt, {'max_condition_z': 4.0}, names, label='shape 00007')
    assert exceeded == ['area']
    out = capsys.readouterr().out
    assert 'WARNING' in out and 'shape 00007' in out
    assert float(cond_n[1]) == pytest.approx(5.0)       # warn does not move the target

    # clamp: the target actually asked for is the clamped one.
    cond_n, _, exceeded = evaluate_mod.normalize_true_conditions(
        raw, ckpt, {'max_condition_z': 4.0, 'condition_ood_policy': 'clamp'}, names)
    assert exceeded == ['area'] and float(cond_n[1]) == pytest.approx(4.0)

    # error: the run stops, naming the escape hatches.
    with pytest.raises(ValueError, match='max_condition_z'):
        evaluate_mod.normalize_true_conditions(
            raw, ckpt, {'max_condition_z': 4.0, 'condition_ood_policy': 'error'}, names)
    with pytest.raises(ValueError, match='condition_ood_policy'):
        evaluate_mod.normalize_true_conditions(
            raw, ckpt, {'condition_ood_policy': 'nope'}, names)

    # In-envelope targets are untouched and report nothing.
    cond_n, clipped, exceeded = evaluate_mod.normalize_true_conditions(
        np.array([0.5, 1.0]), ckpt, {'max_condition_z': 4.0}, names)
    assert exceeded == [] and clipped == [False, False]
    assert cond_n.tolist() == pytest.approx([0.5, 1.0])


def test_aggregate_conditional_reports_n_and_a_paired_median():
    """Per-method medians are taken over method-dependent subsets: a method
    that breaks the meshes it cannot correct drops those rows from its own
    percentile and looks better for it. The aggregate therefore carries the
    per-condition `n` / coverage and a paired median over the shapes EVERY
    reported method could be measured on."""
    def row(method, shape_idx, err, watertight=True):
        return {'method': method, 'shape_idx': shape_idx, 'valid': True,
                'watertight': watertight, 'rel_error_volume': err,
                'latent_rms_drift': 0.0, 'nfe_per_output': 50, 'velocity_net_calls': 50,
                'seconds': 1.0, 'body_count_raw': 1}

    rows = [row('plain', i, 0.06) for i in range(4)]
    # e2 is measurable on two shapes only (it tore the other two open) and looks
    # 6x better on the subset it kept.
    rows += [row('e2', 0, 0.01), row('e2', 1, 0.01),
             row('e2', 2, None, watertight=False), row('e2', 3, None, watertight=False)]
    agg = evaluate_mod.aggregate_conditional(rows, ['plain', 'e2'], ['volume'])
    plain, e2 = agg['plain']['per_condition']['volume'], agg['e2']['per_condition']['volume']
    assert plain['n'] == 4 and plain['coverage'] == pytest.approx(1.0)
    assert e2['n'] == 2 and e2['coverage'] == pytest.approx(0.5)
    assert e2['median_rel_error'] == pytest.approx(0.01)      # the flattering subset
    assert plain['paired_n'] == e2['paired_n'] == 2           # shapes 0 and 1 only
    assert e2['median_rel_error_paired'] == pytest.approx(0.01)
    assert plain['median_rel_error_paired'] == pytest.approx(0.06)
    assert agg['e2']['watertight_rate'] == pytest.approx(0.5)
    assert evaluate_mod.paired_shape_ids(rows, ['plain', 'e2'], 'volume') == {0, 1}


@needs_descriptor_tools
def test_conditional_eval_refuses_calibrated_methods_without_a_proxy_target(
        world, tmp_path, sphere_decoder, calibration_path):
    """C2/E2 act only through the soft proxy (`volume` / `area`). A checkpoint
    whose only geometric name has no proxy passed the old `geometric_names`
    guard, and then `apply_newton` short-circuited so the `e2` row recorded the
    PLAIN latent, mesh and report as an E2 measurement (while `c2` crashed
    inside make_c2_guidance -- two opposite failures from one cause)."""
    config = _conditional_config(world, tmp_path / 'no_proxy', ['plain', 'e2'],
                                 descriptor_calibration_path=calibration_path,
                                 guidance_targets='bbox_x')
    with pytest.raises(ValueError, match='soft SDF proxy'):
        evaluate_mod.run_evaluate(config)
    # The uncalibrated arms are unaffected by the same request.
    summary = evaluate_mod.run_evaluate(
        _conditional_config(world, tmp_path / 'plain_only', ['plain', 'rejection'],
                            guidance_targets='bbox_x'))
    assert summary['methods'] == ['plain', 'rejection']
