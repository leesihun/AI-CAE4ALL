"""
Inference: generate geometries from the trained SDF-VAE + FM stack, or
round-trip reconstruct an input mesh through the VAE.

Modes (config `mode`):
    sample      noise -> FM ODE (optional conditions + CFG) -> SDF -> STL
    reconstruct input mesh -> encoder mu -> SDF -> STL, plus an optional
                test-time latent refinement pass (`latent_refine_steps > 0`,
                see inference_profiles/latent_refine.py) that writes a second
                `<base>_recon_refined.stl` next to the encoder result.

Conditional sampling (`cond_values`) accepts PARTIAL requests: an entry written
as the literal `nan` means "unspecified" and is masked out of the condition
embedding. That needs an FM checkpoint trained with `cond_dropout_mode
per_dim`; a legacy (`all`) checkpoint has no per-dimension null embedding and
the request is rejected with a clear error.

Two opt-in descriptor-accuracy tools act on the GEOMETRIC targets of a
conditional request (volume, area, bbox_*; FEA-named conditions are outside the
reach of a differentiable SDF proxy and are only *audited*, see below):

  * `guidance_enabled` -- calibrated endpoint-prediction guidance (pilot "C2",
    general_modules/descriptor_guidance.py) applied inside the FM ODE through
    `sample_latents(..., guidance_fn=...)`.
  * `newton_rounds > 0` -- proxy-Jacobian Newton correction of the retained
    latents (pilot "E2", general_modules/descriptor_refinement.py), a
    post-process after candidate ranking.

Both need `descriptor_calibration_path` (written by `mode evaluate` with
`eval_task descriptor_calibration`) and are bit-for-bit inert when off.

`condition_audit geometric|fea|surrogate` chooses how the decoded meshes are
measured against the request. `geometric` (default) is the existing
Marching-Cubes descriptor audit. `fea` and `surrogate` additionally re-measure
FEA-named conditions (mass, per-load-case peak stress / displacement) on the
exported meshes through `design_loop` -- gmsh + the tet4 solver, or the HI-MGN
surrogate -- and report both the stored (log-transformed) and raw values. Both
are best-effort: when the backend is unavailable one message is printed and
the run falls back to the geometric audit; the metadata records which backend
actually ran. FEA values from the tet4 loop are labelled
`tet4_linear_relative_only` (see methods/SDFFlow/CLAUDE.md, "tet4 is stiff").
"""

import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from general_modules import condition_names as CN
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE
from model.velocity_net import VelocityNet, sample_latents
from training_profiles.setup import load_checkpoint, resolve_device

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIT_BACKENDS = ('geometric', 'fea', 'surrogate')
GUIDANCE_STEP_MODES = ('velocity_dt', 'per_step_jump')
# Labels recorded next to every FEA-derived value so nobody quotes a tet4
# linear-static number as an absolute DeepJEB-comparable stress.
FEA_AUDIT_LABEL = 'tet4_linear_relative_only'
SURROGATE_AUDIT_LABEL = 'himgn_surrogate_relative_only'
# design_loop/problem.py uses the GE-challenge case names; DeepJEB's own files
# (and the surrogate's training contract) use the short codes.
_SURROGATE_CASE_NAMES = {'vertical': 'ver', 'horizontal': 'hor',
                         'diagonal': 'dia', 'torsion': 'tor'}
# metres per normalized unit: the DeepJEB bracket is ~0.19 m long and the
# normalized mesh has its longest side at 1.8 (design_loop/problem.py default).
DEFAULT_LENGTH_SCALE = 0.19 / 1.8
# ...but the CONDITION AUDIT is scored against bracket_labels.csv, and the
# labels' own implied constants are a 183.8 mm mean longest extent (0.79% CV)
# and 4470 kg/m3 (mean mass(kg) / mean volume(mm3) over all 2138 rows, CV
# 0.0002%). 0.19 m and 4430 kg/m3 are the design-loop OPTIMISATION defaults,
# where only relative comparisons matter; used for the audit they put a fixed,
# model-independent bias into every absolute-unit error -- measured over the
# 2137 usable shapes: mass +8.7%, volume +9.7%, area +6.4% median against the
# labels, on a perfect decode. mass_kg needs no solver at all, so that bias
# would be the whole reported "error". The audit therefore defaults to the
# label-calibrated constants; `opt_length_scale` / `opt_material_rho` still
# override, and `fea_audit` metadata records which was used.
DEEPJEB_AUDIT_LENGTH_SCALE = 0.1838 / 1.8
DEEPJEB_LABEL_DENSITY = 4470.0
# Residual accuracy floor of the audit's mass/volume/area path once the scale
# is calibrated: the per-shape longest extent still varies 0.79% (CV), which is
# 3 x that on a volume and 2 x on an area.
AUDIT_SCALE_CV = 0.0079
DEFAULT_GUIDANCE_TARGETS = ('volume', 'area')
# Both structural backends report two von Mises numbers per load case. The
# percentile one is what `opt_stress_percentile` configures and what the
# optimize loop scores on; the element maximum (DeepJEB's own label) is a tet4
# singularity that grows with refinement, so it is reported beside the scored
# value, never in place of it.
STRESS_MEASURES = ('peak_von_mises', 'max_von_mises')
# Normalized-error units charged to a geometric condition the decoded mesh
# could not provide (see `_audit_report`). Three train standard deviations is
# well outside anything a working decode produces, so an unmeasurable
# dimension always loses to a measured one, while two equally broken meshes
# still rank by what they could be measured on.
UNMEASURABLE_SCORE_PENALTY = 3.0
DEFAULT_STRESS_MEASURE = 'peak_von_mises'
# Kinds the structural backends can re-measure on a decoded mesh. Frequencies
# need an eigen-solve, inertia/CG a frame the normalized mesh does not carry.
MEASURABLE_FEA_KINDS = ('mass', 'volume', 'area', 'stress', 'displacement')


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return value


def _as_name_list(value, default=()):
    """Config list/scalar/None -> list of lowercase str names."""
    if value is None:
        return [str(v).strip().lower() for v in default]
    if isinstance(value, (list, tuple)):
        return [str(v).strip().lower() for v in value]
    return [part.strip().lower() for part in str(value).split(',') if part.strip()]


class VelocityCallCounter:
    """Counts velocity-net forward calls (and the rows they processed) through a
    forward hook, so NFE can be reported for any sampler -- including calls a
    guidance callback makes on its own."""

    def __init__(self, model):
        self.calls = 0
        self.rows = 0
        self._handle = model.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.calls += 1
        if inputs and torch.is_tensor(inputs[0]):
            self.rows += int(inputs[0].shape[0])

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _model_state(ckpt):
    """Prefer EMA weights; strip AveragedModel 'module.' prefix."""
    state = ckpt.get('ema_state') or ckpt['model_state']
    if ckpt.get('ema_state') is not None:
        state = {k.replace('module.', '', 1): v for k, v in state.items() if k != 'n_averaged'}
    return state


def load_vae(path, device):
    ckpt = load_checkpoint(path, device)
    vae = SDFVAE(ckpt['config']).to(device)
    vae.load_state_dict(_model_state(ckpt))
    vae.eval()
    return vae, ckpt


def load_fm_stack(fm_path, vae_path=None, device='cpu'):
    """Load the FM checkpoint, the VAE it names (or `vae_path`), and rebuild the
    velocity net. Returns a namespace shared by sample / interpolate / evaluate:
    vae, vae_ckpt, vae_path, fm_ckpt, fm_path, model, latent_flat_dim, cond_dim,
    cond_names, latent_mean, latent_std (both on `device`), device.

    The VAE parameters are frozen (`requires_grad_(False)`): the descriptor
    tools differentiate through `vae.decode_flat` with respect to the LATENT
    only, and a trainable decoder would silently double the autograd graph.
    """
    print(f'Loading FM checkpoint from {fm_path}')
    fm_ckpt = load_checkpoint(fm_path, device)
    vae_path = vae_path or fm_ckpt['vae_modelpath']
    print(f'Loading VAE checkpoint from {vae_path}')
    vae, vae_ckpt = load_vae(vae_path, device)
    for p in vae.parameters():
        p.requires_grad_(False)

    latent_flat_dim = int(fm_ckpt['latent_flat_dim'])
    cond_dim = int(fm_ckpt['cond_dim'])
    model = VelocityNet(fm_ckpt['config'], latent_flat_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(_model_state(fm_ckpt))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return SimpleNamespace(
        vae=vae, vae_ckpt=vae_ckpt, vae_path=vae_path,
        fm_ckpt=fm_ckpt, fm_path=fm_path, model=model,
        latent_flat_dim=latent_flat_dim, cond_dim=cond_dim,
        cond_names=[str(n) for n in fm_ckpt.get('cond_names', [])],
        latent_mean=fm_ckpt['latent_mean'].to(device),
        latent_std=fm_ckpt['latent_std'].to(device),
        device=device,
    )


def cond_dropout_mode_of(fm_ckpt):
    """`cond_dropout_mode` the FM was trained with ('all' for legacy checkpoints)."""
    ckpt_config = fm_ckpt.get('config') or {}
    return str(ckpt_config.get('cond_dropout_mode', 'all') or 'all').lower()


# ---------------------------------------------------------------------------
# Condition requests (full or partial)
# ---------------------------------------------------------------------------

def parse_condition_values(values, cond_names, label='cond_values'):
    """Config `cond_values` (scalar, list, or None) -> float64 array in checkpoint
    condition order, NaN for every entry written as the literal `nan`
    ("unspecified"). Returns None when `values` is None.

    The native parser turns `0.3,nan,0.5` into a list of strings (`'nan'` fails
    its int() fast path, which demotes the whole list), so entries arrive as
    str, int, or float; `float()` accepts all three spellings.
    """
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        values = [values]
    cond_names = list(cond_names)
    if len(values) != len(cond_names):
        raise ValueError(f'{label} must have {len(cond_names)} entries ({cond_names}), '
                         f'got {len(values)}')
    out = np.empty(len(values), dtype=np.float64)
    for i, value in enumerate(values):
        try:
            out[i] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} entry {i} ({cond_names[i]}) is not a number or 'nan': "
                             f'{value!r}') from exc
    return out


def condition_mask_from_values(raw, fm_ckpt, label='cond_values'):
    """Bool mask (True = specified) for a parsed request.

    Any NaN entry needs an FM trained with `cond_dropout_mode per_dim`: a
    legacy `all` checkpoint has a single null embedding for the whole vector and
    cannot represent "volume given, area free". An all-NaN request is refused
    too -- that is unconditional generation, spelled by omitting `cond_values`.
    """
    raw = np.asarray(raw, dtype=np.float64)
    mask = ~np.isnan(raw)
    if not mask.all():
        mode = cond_dropout_mode_of(fm_ckpt)
        names = [str(n) for n in fm_ckpt.get('cond_names', [])]
        unspecified = [names[i] if i < len(names) else str(i)
                       for i in np.flatnonzero(~mask).tolist()]
        if mode != 'per_dim':
            raise ValueError(
                f"{label} leaves {unspecified} unspecified ('nan'), but this FM checkpoint "
                f"was trained with cond_dropout_mode '{mode}': it has one null embedding "
                'for the whole condition vector and cannot mask individual dimensions. '
                'Either give every condition a value, or sample from an FM trained with '
                'cond_dropout_mode per_dim.')
        if not mask.any():
            raise ValueError(f"every {label} entry is 'nan'; that is unconditional generation "
                             f'-- omit {label} instead.')
    return torch.from_numpy(mask)


def normalize_condition_request(raw, fm_ckpt, config, label='cond_values'):
    """Normalize a (possibly partial) raw request with the checkpoint statistics.

    Returns (cond_n float32 [cond_dim] with 0 in unspecified dims, mask bool
    [cond_dim], request dict). Applies the `max_condition_z` /
    `condition_ood_policy` guard and the train-range extrapolation report to the
    SPECIFIED dimensions only.
    """
    cond_names = [str(n) for n in fm_ckpt['cond_names']]
    cond_dim = len(cond_names)
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (cond_dim,):
        raise ValueError(f'{label} must have {cond_dim} entries ({cond_names}), got {raw.shape}')
    mask = condition_mask_from_values(raw, fm_ckpt, label=label)
    mask_np = mask.numpy()

    cond_mean = fm_ckpt['cond_mean'].squeeze(0).cpu().double()
    cond_std = fm_ckpt['cond_std'].squeeze(0).cpu().double()
    raw_t = torch.from_numpy(np.nan_to_num(raw, nan=0.0))
    cond_n = (raw_t - cond_mean) / cond_std
    cond_n = torch.where(mask, cond_n, torch.zeros_like(cond_n))

    max_condition_z = float(config.get('max_condition_z', fm_ckpt.get('cond_clip') or 5.0))
    ood_policy = str(config.get('condition_ood_policy', 'error')).lower()
    if ood_policy not in ('error', 'warn', 'clamp'):
        raise ValueError('condition_ood_policy must be error, warn, or clamp')
    excessive = (cond_n.abs() > max_condition_z) & mask
    if excessive.any():
        details = ', '.join(f'{cond_names[i]}={float(cond_n[i]):.2f} sigma'
                            for i in torch.where(excessive)[0].tolist())
        message = (f'Condition request ({label}) exceeds max_condition_z={max_condition_z:g}: '
                   f'{details}')
        if ood_policy == 'error':
            raise ValueError(message)
        print(f'WARNING: {message}')
        if ood_policy == 'clamp':
            cond_n = cond_n.clamp(-max_condition_z, max_condition_z)
            print('Condition input was clamped; requested raw values remain in metadata.')

    extrapolated = []
    cond_min = fm_ckpt.get('cond_min')
    cond_max = fm_ckpt.get('cond_max')
    if cond_min is not None and cond_max is not None:
        lo = cond_min.squeeze(0).cpu().double().numpy()
        hi = cond_max.squeeze(0).cpu().double().numpy()
        extrapolated = [cond_names[i] for i in range(cond_dim)
                        if mask_np[i] and (raw[i] < lo[i] or raw[i] > hi[i])]
        if extrapolated:
            print(f'Extrapolated dimensions ({label}): {extrapolated}')

    request = {
        'normalized': {name: (float(cond_n[i]) if mask_np[i] else None)
                       for i, name in enumerate(cond_names)},
        'raw': {name: (float(raw[i]) if mask_np[i] else None)
                for i, name in enumerate(cond_names)},
        'specified': {name: bool(mask_np[i]) for i, name in enumerate(cond_names)},
        'cond_mask': mask_np.tolist(),
        'partial': bool(not mask_np.all()),
        'cond_dropout_mode': cond_dropout_mode_of(fm_ckpt),
        'max_condition_z': max_condition_z,
        'ood_policy': ood_policy,
        'extrapolated_dimensions': extrapolated,
    }
    return cond_n.float(), mask, request


def raw_from_stored_targets(cond_names, target_stored):
    """Stored-space targets -> raw physical units per name (exp for log-stored
    FEA names, identity for geometric names). Unknown names pass through
    unchanged; an unspecified (NaN) entry maps to None.

    This is the direction every REPORT wants: `cond_values`, the dataset rows
    and the FM checkpoint all live in stored space, but a relative error is
    only meaningful in MPa / mm / kg. `_stored_values` is the inverse, used on
    what the structural backends measure. `evaluate.run_conditional_eval`
    imports this rather than keeping its own copy.
    """
    out = {}
    for name, value in zip(cond_names, np.asarray(target_stored, dtype=np.float64)):
        if np.isnan(value):
            out[name] = None
        elif CN.is_known(name):
            out[name] = float(CN.from_stored(name, float(value)))
        else:
            out[name] = float(value)
    return out


# ---------------------------------------------------------------------------
# Geometric audit of decoded meshes
# ---------------------------------------------------------------------------

def _actual_conditions(report, cond_names):
    """Decoded-mesh geometric descriptors in checkpoint condition order.

    Returns a float64 array with NaN for every condition the mesh cannot
    provide: FEA-named conditions (not geometric), `volume` on a non-watertight
    mesh, or everything when the decode had no zero crossing.
    """
    cond_names = list(cond_names)
    actual = np.full(len(cond_names), np.nan, dtype=np.float64)
    if not report.get('valid'):
        return actual
    extents = report.get('extents')
    values = {
        'bbox_x': extents[0] if extents else None,
        'bbox_y': extents[1] if extents else None,
        'bbox_z': extents[2] if extents else None,
        'volume': report.get('volume'),
        'area': report.get('area'),
    }
    for i, name in enumerate(cond_names):
        value = values.get(name)
        if value is not None:
            actual[i] = float(value)
    return actual


def _audit_report(report, cond_names, target, cond_std, mask=None):
    """Attach the geometric condition audit of one decoded mesh to its report.

    Scores the dimensions that are (a) specified in the request (`mask`) and
    (b) geometric, i.e. measurable on a mesh IN PRINCIPLE. `condition_score` is
    an RMS normalized error over that fixed support, with a fixed penalty
    (`UNMEASURABLE_SCORE_PENALTY`, in normalized-error units) charged for every
    such dimension this particular mesh could not provide -- `volume` on a
    non-watertight decode. A plain mean over only what could be measured is not
    comparable between candidates: dropping a term can only lower the score, so
    the candidate ranking (`mode sample`) and `_pick_best_candidate` (the
    benchmark's `rejection` arm) would systematically prefer the torn meshes.
    On the ex5 condition set only 2 of 6 names are geometric and `volume` is
    the one that disappears, so half the scorable support went missing exactly
    when the mesh broke. With nothing missing the value is unchanged from the
    plain RMS. None when the request specifies no geometric dimension at all.
    FEA-named conditions are listed under `not_measurable_geometrically`.
    """
    cond_names = list(cond_names)
    target = np.asarray(target, dtype=np.float64)
    actual = _actual_conditions(report, cond_names)
    specified = np.ones(len(cond_names), dtype=bool) if mask is None \
        else np.asarray(mask, dtype=bool)
    geometric = np.asarray([CN.is_geometric(n) for n in cond_names], dtype=bool)
    support = specified & geometric & ~np.isnan(target)
    scorable = support & ~np.isnan(actual)
    report['not_measurable_geometrically'] = [
        name for name in cond_names if not CN.is_geometric(name)]
    report['audited_names'] = [cond_names[i] for i in np.flatnonzero(scorable)]
    report['condition_score_support'] = [int(scorable.sum()), int(support.sum())]
    report['condition_score_unmeasurable'] = [
        cond_names[i] for i in np.flatnonzero(support & ~scorable)]
    if not support.any():
        report['condition_score'] = None
        return
    idx = np.flatnonzero(scorable)
    missing = int(support.sum()) - int(scorable.sum())
    squares = float(missing) * UNMEASURABLE_SCORE_PENALTY ** 2
    if idx.size:
        error = actual[idx] - target[idx]
        normalized_error = error / np.asarray(cond_std, dtype=np.float64)[idx]
        names = [cond_names[i] for i in idx]
        report['actual_conditions'] = dict(zip(names, actual[idx].tolist()))
        report['condition_abs_error'] = dict(zip(names, np.abs(error).tolist()))
        report['condition_rel_error'] = dict(zip(
            names, (np.abs(error) / np.maximum(np.abs(target[idx]), 1e-8)).tolist()))
        squares += float((normalized_error ** 2).sum())
    report['condition_score'] = float(np.sqrt(squares / float(support.sum())))


def _condition_summary(results, cond_names, target, mask=None):
    cond_names = list(cond_names)
    target = np.asarray(target, dtype=np.float64)
    audited = [r for r in results if r.get('actual_conditions')]
    # The request-shape fields are reported even when NOTHING could be audited
    # (every decode invalid, or volume unavailable on non-watertight meshes):
    # a reader still needs to see which names were asked for and which ones a
    # geometric audit could never have scored.
    base = {
        'audited_meshes': len(audited),
        'specified_names': [name for i, name in enumerate(cond_names)
                            if mask is None or bool(mask[i])],
        'not_measurable_geometrically': [n for n in cond_names if not CN.is_geometric(n)],
    }
    if not audited:
        return dict(base, audited_names=[], target={}, median_actual={},
                    median_abs_error={}, median_rel_error={},
                    best_condition_score=None, median_condition_score=None)
    names = sorted({name for r in audited for name in r['actual_conditions']},
                   key=cond_names.index)
    actual = np.asarray([[r['actual_conditions'].get(name, np.nan) for name in names]
                         for r in audited], dtype=np.float64)
    tgt = np.asarray([target[cond_names.index(name)] for name in names], dtype=np.float64)
    error = np.abs(actual - tgt[None, :])
    scores = [r['condition_score'] for r in audited if r.get('condition_score') is not None]
    return {
        **base,
        'audited_names': names,
        'target': dict(zip(names, tgt.tolist())),
        'median_actual': dict(zip(names, np.nanmedian(actual, axis=0).tolist())),
        'median_abs_error': dict(zip(names, np.nanmedian(error, axis=0).tolist())),
        'median_rel_error': dict(zip(
            names, np.nanmedian(error / np.maximum(np.abs(tgt[None, :]), 1e-8), axis=0).tolist())),
        'best_condition_score': float(min(scores)) if scores else None,
        'median_condition_score': float(np.median(scores)) if scores else None,
    }


# ---------------------------------------------------------------------------
# Descriptor tools (guidance / Newton / calibration) -- optional modules
# ---------------------------------------------------------------------------

def import_descriptor_tools():
    """Import general_modules.descriptor_{proxy,calibration,refinement,guidance}
    or raise an ImportError that names what is missing and which config keys
    asked for it."""
    try:
        from general_modules import descriptor_calibration, descriptor_guidance
        from general_modules import descriptor_proxy, descriptor_refinement
    except ImportError as exc:
        raise ImportError(
            'guidance_enabled, newton_rounds > 0, eval_methods c2/e2/c2e2 and eval_task '
            'descriptor_calibration need general_modules/descriptor_proxy.py, '
            'descriptor_calibration.py, descriptor_refinement.py and descriptor_guidance.py '
            f'importable from methods/SDFFlow ({exc}).') from exc
    return SimpleNamespace(proxy=descriptor_proxy, calibration=descriptor_calibration,
                           refinement=descriptor_refinement, guidance=descriptor_guidance)


def load_calibration(tools, path, vae_path, fm_path, resolution, tau,
                     measure_resolution=None):
    """Load a DescriptorCalibration and refuse to use it against the wrong
    checkpoints / proxy settings (silent reuse of stale coefficients is exactly
    what turned pilot round 1's guidance into a 23% volume error)."""
    if not path:
        raise ValueError(
            'descriptor_calibration_path is required when guidance_enabled is True, '
            'newton_rounds > 0, or eval_methods include c2/e2/c2e2. Create it with mode '
            'evaluate + eval_task descriptor_calibration on the same VAE/FM pair.')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'descriptor_calibration_path {path} does not exist. Create it with mode evaluate '
            '+ eval_task descriptor_calibration (configs/SDFFlow/config_calibrate_descriptors.txt) '
            'on the same VAE/FM pair and the same soft_descriptor_resolution / '
            'soft_descriptor_tau.')
    calibration = tools.calibration.DescriptorCalibration.load(path)
    calibration.check_compatible(vae_path, fm_path, resolution, tau,
                                 measure_resolution=measure_resolution)
    print(f'Descriptor calibration: {path} (resolution={resolution}, tau={tau:g}, '
          f'fitted against the export path at mc_resolution {calibration.measure_resolution})')
    return calibration


def soft_proxy_names():
    """Descriptor names the differentiable proxy can actually compute.

    Read from `general_modules/descriptor_proxy.py` at call time rather than
    mirrored here, so adding a proxy there needs no edit in this file. Falls
    back to the guidance default when the module is absent -- the tools import
    that follows raises the real, named error.
    """
    try:
        from general_modules.descriptor_proxy import SUPPORTED_SOFT_NAMES
    except ImportError:
        return tuple(DEFAULT_GUIDANCE_TARGETS)
    return tuple(SUPPORTED_SOFT_NAMES)


def descriptor_targets(cond_names, raw_target, mask=None, requested=None, quiet=False):
    """Geometric target dict {name: raw value} for the proxy tools.

    Restricted to `requested` names (default volume, area) that are checkpoint
    conditions, specified in the request, geometric, AND backed by a soft
    proxy. FEA-named requests are ignored with a printed note (a soft SDF proxy
    cannot measure a stress), and so are geometric names with no proxy: the
    bbox extents are near-constant in DeepJEB and have no clean occupancy form,
    so `guidance_targets bbox_x` leaves guidance/Newton inactive with a message
    rather than failing inside `make_c2_guidance`.
    """
    cond_names = list(cond_names)
    raw_target = np.asarray(raw_target, dtype=np.float64)
    requested = _as_name_list(requested, DEFAULT_GUIDANCE_TARGETS)
    proxied = soft_proxy_names()
    targets, ignored_fea, unspecified, unknown, no_proxy = {}, [], [], [], []
    for name in requested:
        if name not in cond_names:
            unknown.append(name)
            continue
        i = cond_names.index(name)
        if mask is not None and not bool(mask[i]):
            unspecified.append(name)
            continue
        if np.isnan(raw_target[i]):
            unspecified.append(name)
            continue
        if not CN.is_geometric(name):
            ignored_fea.append(name)
        elif name not in proxied:
            no_proxy.append(name)
        else:
            targets[name] = float(raw_target[i])
    if not quiet:
        if ignored_fea:
            print(f'NOTE: descriptor proxies cannot measure FEA-named targets {ignored_fea}; '
                  'guidance/Newton act on the geometric targets only '
                  f'({sorted(targets) or "none"}). Use condition_audit fea|surrogate to '
                  'measure them after decoding.')
        if no_proxy:
            print(f'NOTE: guidance_targets {no_proxy} are geometric but have no soft proxy '
                  f'(supported: {list(proxied)}); ignored.')
        if unknown:
            print(f'NOTE: guidance_targets {unknown} are not conditions of this checkpoint '
                  f'({cond_names}); ignored.')
        if unspecified:
            print(f'NOTE: guidance_targets {unspecified} are unspecified (nan) in the request; '
                  'ignored.')
    return targets


def make_guidance(tools, stack, cond, cond_mask, targets, calibration, config):
    """Build the C2 guidance callback for `sample_latents(..., guidance_fn=...)`."""
    eta = float(config.get('guidance_eta', 0.1))
    t_start = float(config.get('guidance_t_start', 0.3))
    step_mode = str(config.get('guidance_step_mode', 'velocity_dt')).lower()
    if step_mode not in GUIDANCE_STEP_MODES:
        raise ValueError(f'guidance_step_mode must be one of {GUIDANCE_STEP_MODES}, got {step_mode!r}')
    if eta <= 0:
        raise ValueError('guidance_eta must be > 0')
    resolution = int(config.get('soft_descriptor_resolution', 48))
    tau = float(config.get('soft_descriptor_tau', 0.032))
    if tau <= 0:
        raise ValueError('soft_descriptor_tau must be > 0')
    return tools.guidance.make_c2_guidance(
        stack.vae, stack.model, cond, cond_mask, targets, calibration,
        eta=eta, t_start=t_start, step_mode=step_mode, resolution=resolution, tau=tau,
        ode_steps_ref=50, latent_mean=stack.latent_mean, latent_std=stack.latent_std)


def newton_measure_resolution(config):
    """The Marching Cubes grid E2 measures and accepts on.

    Defaults to `mc_resolution` -- the grid the audit, the exported mesh and the
    descriptor calibration all use -- rather than an independent 96, so the
    Newton loop drives the residual of the same operator it is scored on. MC
    volume/area move with resolution (an analytic sphere shifts +0.09% / +0.02%
    in volume from res 96 to 128 at r = 0.35 / 0.70), which is the same order as
    the 0.077-0.28% accuracies E2 is quoted at, so a mismatch is a systematic
    offset the output cannot distinguish from model error.
    `DescriptorCalibration.check_compatible` refuses a calibration fitted on a
    different grid.
    """
    return int(config.get('newton_measure_resolution', config.get('mc_resolution', 128)))


def apply_newton(tools, stack, z_n, targets, calibration, config):
    """E2 correction of one NORMALIZED flat latent [1, D]. Returns (z_n', info)."""
    rounds = int(config.get('newton_rounds', 0))
    if rounds <= 0 or not targets:
        return z_n, None
    t0 = time.time()
    z_new, history = tools.refinement.newton_correct(
        stack.vae, z_n, targets, calibration, stack.latent_mean, stack.latent_std,
        rounds=rounds,
        step_cap_rms=float(config.get('newton_step_cap_rms', 0.12)),
        line_search_tries=int(config.get('newton_line_search_tries', 3)),
        measure_resolution=newton_measure_resolution(config),
        resolution=int(config.get('soft_descriptor_resolution', 48)),
        tau=float(config.get('soft_descriptor_tau', 0.032)),
        latent_clip=float(config.get('latent_clip', 0.0) or 0.0))
    z_new = z_new.detach()
    drift = float((z_new - z_n).pow(2).mean().sqrt().item())
    info = {
        'rounds': rounds,
        'targets': dict(targets),
        'latent_clip': float(config.get('latent_clip', 0.0) or 0.0),
        'measure_resolution': newton_measure_resolution(config),
        'history': _jsonable(history),
        'accepted_steps': int(sum(1 for h in history if isinstance(h, dict)
                                  and h.get('step_accepted'))),
        'latent_rms_drift': drift,
        'seconds': float(time.time() - t0),
    }
    return z_new, info


# ---------------------------------------------------------------------------
# FEA / surrogate condition audit
# ---------------------------------------------------------------------------

def fea_condition_names(cond_names):
    return [str(n) for n in cond_names if CN.is_fea(n)]


def resolve_condition_audit(config, cond_names):
    """Decide which audit backend actually runs. Returns (backend, reason).

    Falls back to 'geometric' with ONE printed message when the requested
    structural backend cannot run here (no FEA-named conditions to measure, a
    missing gmsh/pyamg, or a missing surrogate config/checkpoint).
    """
    requested = str(config.get('condition_audit', 'geometric')).lower()
    if requested not in AUDIT_BACKENDS:
        raise ValueError(f'condition_audit must be one of {AUDIT_BACKENDS}, got {requested!r}')
    if requested == 'geometric':
        return 'geometric', 'requested'
    fea_names = fea_condition_names(cond_names)
    if not fea_names:
        print(f'condition_audit {requested}: this checkpoint has no FEA-named conditions '
              f'({list(cond_names)}); running the geometric audit only.')
        return 'geometric', 'no FEA-named conditions in the checkpoint'

    problems = []
    if requested == 'fea':
        for module in ('gmsh', 'pyamg'):
            try:
                __import__(module)
            except Exception as exc:  # ImportError or a broken native lib
                problems.append(f'{module} ({type(exc).__name__}: {exc})')
        try:
            from design_loop.mesher import tet_mesh_from_surface  # noqa: F401
            from design_loop.problem import Bracket  # noqa: F401
        except Exception as exc:
            problems.append(f'design_loop ({type(exc).__name__}: {exc})')
    else:
        surrogate_config = config.get('opt_surrogate_config')
        surrogate_checkpoint = config.get('opt_surrogate_checkpoint')
        if not surrogate_config or not os.path.exists(str(surrogate_config)):
            problems.append(f'opt_surrogate_config missing or not found ({surrogate_config!r})')
        if not surrogate_checkpoint or not os.path.exists(str(surrogate_checkpoint)):
            problems.append('opt_surrogate_checkpoint missing or not found '
                            f'({surrogate_checkpoint!r})')
        try:
            from design_loop.surrogate import HIMGNSurrogate  # noqa: F401
        except Exception as exc:
            problems.append(f'design_loop.surrogate ({type(exc).__name__}: {exc})')
    if problems:
        print(f'condition_audit {requested} is unavailable here: ' + '; '.join(problems)
              + '. Falling back to the geometric audit; FEA-named conditions '
              f'{fea_names} are reported as not measurable.')
        return 'geometric', 'unavailable: ' + '; '.join(problems)
    return requested, 'ok'


def load_cases_used():
    """The load magnitudes design_loop/problem.py applies RIGHT NOW, read from
    `Bracket.LOAD_CASES` at runtime (never duplicated here), for the metadata."""
    from design_loop.problem import Bracket
    out = {}
    for name, spec in Bracket.LOAD_CASES.items():
        entry = {'kind': spec.get('kind')}
        if spec.get('kind') == 'force':
            vector = [float(v) for v in spec.get('vector', ())]
            entry['vector_n'] = vector
            entry['magnitude_n'] = float(np.linalg.norm(vector)) if vector else None
        else:
            entry['axis'] = [float(v) for v in spec.get('axis', ())]
            entry['magnitude_nm'] = float(spec.get('magnitude', float('nan')))
        out[name] = entry
    return out


def mesh_physical_quantities(mesh, length_scale, density):
    """Mass/volume/area of a normalized mesh in the registry's units (kg, mm3, mm2).

    `volume` (and therefore mass) is None for a non-watertight mesh rather than
    a convex-hull guess, so an audit never scores a hole-ridden decode as a
    plausible bracket mass.
    """
    scale_mm = float(length_scale) * 1e3
    volume_norm = float(abs(mesh.volume)) if mesh.is_watertight else None
    return {
        'volume_mm3': (volume_norm * scale_mm ** 3) if volume_norm is not None else None,
        'surface_area_mm2': float(mesh.area) * scale_mm ** 2,
        'mass_kg': (volume_norm * float(length_scale) ** 3 * float(density))
        if volume_norm is not None else None,
    }


def convert_structural_result(result, fea_names, physical, stress_measure=DEFAULT_STRESS_MEASURE):
    """SI result dict (Pa, m; `cases` keyed by GE load-case names, as returned
    by `Bracket.analyze` or remapped from `HIMGNSurrogate.analyze_batch`) ->
    {name: raw value in the registry unit} for the FEA-named conditions.
    Names the backend cannot provide map to None.

    `stress_measure` picks which von Mises number a `kind == 'stress'` name
    reads. Both backends report two: `peak_von_mises`, the
    `opt_stress_percentile`-th percentile of the nodal field (the default, and
    what the optimize loop scores designs on), and `max_von_mises`, the element
    maximum. DeepJEB's own label is the maximum, but on a tet4 mesh that is a
    re-entrant-corner singularity that keeps climbing with refinement, so the
    percentile is the comparable number and the maximum is reported next to it
    rather than instead of it. A backend missing the requested measure falls
    back to the other one.
    """
    cases = (result or {}).get('cases', {}) or {}
    values = {}
    for name in fea_names:
        entry = CN.describe(name)
        if entry is None:
            values[name] = None
            continue
        kind, load_case = entry['kind'], entry['load_case']
        case = cases.get(load_case) if load_case else None
        value = None
        if kind == 'stress' and case is not None:
            other = next(m for m in STRESS_MEASURES if m != stress_measure)
            pascals = case.get(stress_measure)
            if pascals is None:
                pascals = case.get(other)
            if pascals is not None:
                value = float(pascals) / 1e6                    # Pa -> MPa
        elif kind == 'displacement' and case is not None \
                and case.get('max_displacement') is not None:
            value = float(case['max_displacement']) * 1e3       # m -> mm
        elif kind == 'mass':
            value = physical.get('mass_kg')
        elif kind == 'volume':
            value = physical.get('volume_mm3')
        elif kind == 'area':
            value = physical.get('surface_area_mm2')
        values[name] = None if value is None else float(value)
    return values


def _stored_values(values_raw):
    out = {}
    for name, value in values_raw.items():
        if value is None:
            out[name] = None
        else:
            stored = CN.to_stored(name, value)
            out[name] = None if (stored is None or math.isnan(stored)) else float(stored)
    return out


def fea_condition_audit(meshes, cond_names, backend, config, workdir=None):
    """Re-measure the FEA-named conditions of decoded meshes.

    `meshes`: list of trimesh (None entries are skipped). Returns (entries,
    meta): one dict per mesh with `values_raw` (MPa / mm / kg / mm3 / mm2),
    `values_stored` (the checkpoint's stored space, e.g. ln MPa),
    `values_raw_max_von_mises` (the element-maximum stress beside the scored
    percentile), `error`, `label`, and the meshing/solver info; `meta` records
    the backend, the length scale, the density, the stress measure, the load
    cases needed and -- for `fea` -- the load magnitudes read from
    `Bracket.LOAD_CASES` at runtime.

    The solver runs only when a requested name needs it: a `mass_kg` /
    `volume_mm3` / `surface_area_mm2` audit is pure mesh arithmetic, so it
    neither imports gmsh nor meshes anything.
    """
    if backend not in ('fea', 'surrogate'):
        raise ValueError(f'fea_condition_audit needs backend fea or surrogate, got {backend!r}')
    fea_names = fea_condition_names(cond_names)
    length_scale_configured = str(config.get('opt_length_scale', '')).strip() != ''
    density_configured = str(config.get('opt_material_rho', '')).strip() != ''
    length_scale = float(config.get('opt_length_scale', DEEPJEB_AUDIT_LENGTH_SCALE))
    density = float(config.get('opt_material_rho', DEEPJEB_LABEL_DENSITY))
    described = {name: CN.describe(name) for name in fea_names}
    needed_cases = sorted({d['load_case'] for d in described.values() if d and d['load_case']})
    solver_needed = any(d and d['kind'] in ('stress', 'displacement') for d in described.values())
    label = FEA_AUDIT_LABEL if backend == 'fea' else SURROGATE_AUDIT_LABEL
    stress_percentile = float(config.get('opt_stress_percentile', 99.5))
    meta = {
        'backend': backend,
        'label': label,
        'length_scale_m_per_unit': length_scale,
        'length_scale_source': 'opt_length_scale' if length_scale_configured
        else 'deepjeb_label_calibrated (0.1838 m / 1.8 units)',
        'density_kg_m3': density,
        'density_source': 'opt_material_rho' if density_configured
        else 'deepjeb_label_calibrated (mean mass / mean volume = 4470 kg/m3)',
        'scale_bias_note': (
            'mass_kg / volume_mm3 / surface_area_mm2 are pure mesh arithmetic at this length '
            f'scale and density. The DeepJEB longest extent still varies {100 * AUDIT_SCALE_CV:.2f}% '
            f'(CV) shape to shape, so read {300 * AUDIT_SCALE_CV:.1f}% (volume, mass) / '
            f'{200 * AUDIT_SCALE_CV:.1f}% (area) as the estimator floor of these three names, '
            'not as generator error. The design-loop defaults (0.19 m, 4430 kg/m3) would add a '
            'further +8.7% / +9.7% / +6.4% median bias against the labels.'),
        'stress_measure': DEFAULT_STRESS_MEASURE,
        'stress_percentile': stress_percentile,
        'stress_measure_note': (
            f'stress conditions are scored on {DEFAULT_STRESS_MEASURE} (the {stress_percentile:g}th '
            'percentile of the nodal von Mises field); the element maximum -- DeepJEB\'s own label, '
            'and a tet4 singularity that keeps climbing with refinement -- is reported beside it '
            'under values_raw_max_von_mises'),
        'load_cases_needed': needed_cases,
        'load_cases_used': None,
        'measured_names': [n for n in fea_names
                           if described[n] and described[n]['kind'] in MEASURABLE_FEA_KINDS],
        'not_measurable': [n for n in fea_names
                           if not described[n] or described[n]['kind'] not in MEASURABLE_FEA_KINDS],
        'units': {n: (described[n]['unit'] if described[n] else None) for n in fea_names},
    }
    entries = [None] * len(meshes)
    valid_idx = [i for i, m in enumerate(meshes) if m is not None]
    physical = {i: mesh_physical_quantities(meshes[i], length_scale, density) for i in valid_idx}

    results = {i: None for i in valid_idx}
    errors = {i: None for i in valid_idx}
    infos = {i: None for i in valid_idx}
    t0 = time.time()
    if solver_needed and valid_idx:
        if backend == 'fea':
            from design_loop import fea
            from design_loop.mesher import tet_mesh_from_surface
            from design_loop.problem import Bracket
            meta['load_cases_used'] = load_cases_used()
            material = fea.Material(
                E=float(config.get('opt_material_e', 113.8e9)),
                nu=float(config.get('opt_material_nu', 0.342)),
                rho=density,
                yield_stress=float(config.get('opt_yield_stress', 903e6)))
            bracket = Bracket(material=material, length_scale=length_scale,
                              load_cases=tuple(needed_cases),
                              stress_percentile=stress_percentile)
            mesh_size_max = float(config.get('opt_mesh_size_max', 0.05))
            target_faces = int(config.get('opt_target_faces', 12000))
            print(f'FEA audit ({label}): {len(valid_idx)} mesh(es), load cases {needed_cases}, '
                  f'loads {meta["load_cases_used"]}')
            for i in valid_idx:
                try:
                    nodes, tets, info = tet_mesh_from_surface(
                        meshes[i], mesh_size_max=mesh_size_max, target_faces=target_faces)
                    infos[i] = info
                    results[i] = bracket.analyze(nodes, tets)
                except Exception as exc:
                    errors[i] = f'{type(exc).__name__}: {exc}'
        else:
            from design_loop.surrogate import HIMGNSurrogate
            surrogate = HIMGNSurrogate(
                config_path=config['opt_surrogate_config'],
                checkpoint=config['opt_surrogate_checkpoint'],
                load_cases=tuple(_SURROGATE_CASE_NAMES[c] for c in needed_cases),
                target_nodes=int(config.get('opt_surrogate_target_nodes', 5000)),
                density=density,
                stress_percentile=stress_percentile,
                workdir=workdir)
            print(f'Surrogate audit ({label}): {len(valid_idx)} mesh(es), load cases {needed_cases}')
            try:
                batch = surrogate.analyze_batch([meshes[i] for i in valid_idx])
            except Exception as exc:
                batch = [None] * len(valid_idx)
                for i in valid_idx:
                    errors[i] = f'{type(exc).__name__}: {exc}'
            long_names = {v: k for k, v in _SURROGATE_CASE_NAMES.items()}
            for i, res in zip(valid_idx, batch):
                if res is None:
                    errors[i] = errors[i] or 'SurrogateError: surrogate could not bridge or predict this shape'
                    continue
                remapped = dict(res)
                remapped['cases'] = {long_names.get(k, k): v for k, v in (res.get('cases') or {}).items()}
                results[i] = remapped
                infos[i] = {'num_nodes': res.get('num_nodes'), 'surrogate': True}
            meta['surrogate'] = surrogate.stats()
    meta['seconds'] = float(time.time() - t0)

    for i in valid_idx:
        values_raw = convert_structural_result(results[i], fea_names, physical[i])
        # The element-maximum von Mises next to the scored percentile, so a
        # reader can see how far apart the two measures are on this mesh
        # instead of having to trust that the gap is small (on tet4 it is not).
        alt_raw = convert_structural_result(results[i], fea_names, physical[i],
                                            stress_measure='max_von_mises')
        entries[i] = {
            'backend': backend,
            'label': label,
            'stress_measure': DEFAULT_STRESS_MEASURE,
            'values_raw': values_raw,
            'values_stored': _stored_values(values_raw),
            'values_raw_max_von_mises': {n: alt_raw[n] for n in fea_names
                                         if (described[n] or {}).get('kind') == 'stress'},
            'physical': physical[i],
            'error': errors[i],
            'mesh_info': _jsonable(infos[i]) if infos[i] is not None else None,
        }
    return entries, meta


def attach_fea_audit(report, entry, cond_names, target_stored):
    """Store one FEA audit entry on a mesh report with per-name errors against
    the stored-space request (abs) and the raw-unit request (relative)."""
    report['fea_audit'] = entry
    if entry is None:
        return
    target_stored = np.asarray(target_stored, dtype=np.float64)
    rel, abs_stored = {}, {}
    for name, raw_value in entry['values_raw'].items():
        i = list(cond_names).index(name)
        if raw_value is None or np.isnan(target_stored[i]):
            continue
        target_raw = float(CN.from_stored(name, float(target_stored[i])))
        rel[name] = abs(raw_value - target_raw) / max(abs(target_raw), 1e-12)
        stored = entry['values_stored'].get(name)
        if stored is not None:
            abs_stored[name] = abs(stored - float(target_stored[i]))
    report['fea_condition_rel_error_raw'] = rel
    report['fea_condition_abs_error_stored'] = abs_stored


# ---------------------------------------------------------------------------
# mode sample
# ---------------------------------------------------------------------------

def run_sample(config, config_filename='config.txt'):
    device = resolve_device(config)

    fm_path = config.get('fm_modelpath', '../../output/geometry_generation/sdfflow_fm.pth')
    stack = load_fm_stack(fm_path, config.get('vae_modelpath'), device)
    vae, model, fm_ckpt, vae_path = stack.vae, stack.model, stack.fm_ckpt, stack.vae_path
    latent_flat_dim, cond_dim = stack.latent_flat_dim, stack.cond_dim
    cond_names = stack.cond_names

    num_samples = int(config.get('num_samples', 8))
    ode_steps = int(config.get('ode_steps', 50))
    cfg_scale = float(config.get('cfg_scale', 1.0))
    resolution = int(config.get('mc_resolution', 128))
    seed = int(config.get('seed', 0))
    candidate_multiplier = int(config.get('candidate_multiplier', 1))
    if candidate_multiplier < 1:
        raise ValueError('candidate_multiplier must be at least 1')
    out_dir = config.get('output_dir', '../../output/geometry_generation/samples')
    os.makedirs(out_dir, exist_ok=True)

    guidance_enabled = bool(config.get('guidance_enabled', False))
    newton_rounds = int(config.get('newton_rounds', 0))
    if newton_rounds < 0:
        raise ValueError('newton_rounds must be a nonnegative integer')
    soft_resolution = int(config.get('soft_descriptor_resolution', 48))
    soft_tau = float(config.get('soft_descriptor_tau', 0.032))
    calibration_path = config.get('descriptor_calibration_path')

    # ---- Conditions ----
    cond = cond_mask_batch = None
    cond_values = config.get('cond_values', None)
    condition_request = None
    target = cond_std_np = mask = None
    total_candidates = num_samples
    if cond_values is not None and cond_dim > 0:
        raw = parse_condition_values(cond_values, cond_names)
        cond_n, mask, condition_request = normalize_condition_request(raw, fm_ckpt, config)
        total_candidates = num_samples * candidate_multiplier
        cond = cond_n.unsqueeze(0).repeat(total_candidates, 1).to(device)
        if condition_request['partial']:
            cond_mask_batch = mask.unsqueeze(0).repeat(total_candidates, 1).to(device)
        target = raw
        cond_std_np = fm_ckpt['cond_std'].squeeze(0).cpu().numpy().astype(np.float64)
        shown = {name: ('unspecified' if not condition_request['specified'][name]
                        else condition_request['raw'][name]) for name in cond_names}
        print(f'Conditional generation: {shown} (cfg_scale={cfg_scale}, '
              f'candidates={total_candidates}'
              + (f", cond_dropout_mode={condition_request['cond_dropout_mode']}, partial request"
                 if condition_request['partial'] else '') + ')')
    else:
        if cond_values is not None and cond_dim == 0:
            print('NOTE: cond_values given but the FM checkpoint is unconditional; ignored.')
        if candidate_multiplier > 1:
            # Ranking needs a request to rank against, so an unconditional run
            # draws num_samples and nothing else. Say so: silently ignoring a
            # 4x compute setting reads as a bug from the log.
            print(f'NOTE: candidate_multiplier {candidate_multiplier} needs cond_values to rank '
                  f'candidates against; an unconditional run draws num_samples and no more.')
        print('Unconditional generation')

    # ---- Descriptor tools (guidance / Newton) ----
    tools = calibration = None
    targets = {}
    guidance_fn = None
    guidance_info = None
    if (guidance_enabled or newton_rounds > 0):
        if target is None:
            print('NOTE: guidance_enabled / newton_rounds act on a conditional request; '
                  'no cond_values given, so both are inactive for this run.')
            guidance_enabled, newton_rounds = False, 0
        else:
            targets = descriptor_targets(cond_names, target, mask, config.get('guidance_targets'))
            if not targets:
                print('NOTE: no geometric guidance target is specified in this request; '
                      'guidance/Newton are inactive for this run.')
                guidance_enabled, newton_rounds = False, 0
    if guidance_enabled or newton_rounds > 0:
        tools = import_descriptor_tools()
        # The grid the calibration must have been fitted on: what E2 measures
        # and accepts on when it runs, else the grid the audit reports on.
        effective_measure = (newton_measure_resolution(config) if newton_rounds > 0
                             else resolution)
        calibration = load_calibration(tools, calibration_path, vae_path, fm_path,
                                       soft_resolution, soft_tau,
                                       measure_resolution=effective_measure)
        if guidance_enabled:
            guidance_fn = make_guidance(tools, stack, cond, cond_mask_batch, targets,
                                        calibration, config)
            guidance_info = {
                'eta': float(config.get('guidance_eta', 0.1)),
                't_start': float(config.get('guidance_t_start', 0.3)),
                'step_mode': str(config.get('guidance_step_mode', 'velocity_dt')).lower(),
                'targets': dict(targets),
                'soft_descriptor_resolution': soft_resolution,
                'soft_descriptor_tau': soft_tau,
            }
            print(f'C2 guidance: targets={targets} eta={guidance_info["eta"]:g} '
                  f't_start={guidance_info["t_start"]:g} step_mode={guidance_info["step_mode"]}')

    # ---- Sample latents ----
    # Extra sampler arguments are passed ONLY when set, so an off configuration
    # reaches sample_latents with exactly the legacy call signature.
    sample_kwargs = {}
    if cond_mask_batch is not None:
        sample_kwargs['cond_mask'] = cond_mask_batch
    if guidance_fn is not None:
        sample_kwargs['guidance_fn'] = guidance_fn
    counter = VelocityCallCounter(model)
    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    try:
        z_n = sample_latents(model, total_candidates, latent_flat_dim, device,
                             cond=cond, cfg_scale=cfg_scale, ode_steps=ode_steps,
                             generator=generator, **sample_kwargs)
    finally:
        counter.remove()
    sample_seconds = time.time() - t0
    if guidance_info is not None and guidance_fn is not None:
        # Live counters from the callback: how many of the ODE steps fell inside
        # the [t_start, 1) window, and what each cost. Without them a reader
        # cannot tell an active guidance run from one whose window never opened.
        guidance_info['stats'] = _jsonable(dict(guidance_fn.stats))
    latent_clip = float(config.get('latent_clip', 0.0))
    clipped_latent_fraction = 0.0
    if latent_clip > 0:
        clipped_latent_fraction = float((z_n.abs() > latent_clip).float().mean().item())
        z_n = z_n.clamp(-latent_clip, latent_clip)
    z_n = z_n.detach()
    z = z_n * stack.latent_std + stack.latent_mean
    nfe_per_output = counter.rows / max(total_candidates, 1)
    print(f'Sampled {total_candidates} latents in {sample_seconds:.2f}s '
          f'({ode_steps} ODE steps, {counter.calls} velocity-net calls, '
          f'{nfe_per_output:.1f} NFE per candidate)')

    # ---- Decode and export ----
    candidates = []
    for candidate_index in range(total_candidates):
        volume = decode_sdf_grid(
            vae, z[candidate_index:candidate_index + 1], resolution=resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        report['candidate_index'] = candidate_index
        if target is not None:
            _audit_report(report, cond_names, target, cond_std_np, mask)
        candidates.append((mesh, report, candidate_index))
        if total_candidates > num_samples and ((candidate_index + 1) % 16 == 0
                                               or candidate_index + 1 == total_candidates):
            print(f'  decoded candidates: {candidate_index + 1}/{total_candidates}')

    if target is not None and candidate_multiplier > 1:
        candidates.sort(key=lambda item: (
            item[1].get('condition_score') is None,
            item[1].get('condition_score') if item[1].get('condition_score') is not None
            else float('inf')))
    selected = candidates[:num_samples]

    # ---- Optional E2 Newton correction of the retained latents ----
    newton_summary = None
    if newton_rounds > 0:
        print(f'E2 Newton correction: {newton_rounds} round(s) on {len(selected)} retained '
              f'latent(s), targets={targets}')
        corrected = []
        drifts = []
        for mesh, report, candidate_index in selected:
            pre = {k: report.get(k) for k in ('actual_conditions', 'condition_rel_error',
                                              'condition_score', 'valid', 'watertight')}
            z_row = z_n[candidate_index:candidate_index + 1]
            z_new, info = apply_newton(tools, stack, z_row, targets, calibration, config)
            z_n[candidate_index:candidate_index + 1] = z_new
            z_dec = z_new * stack.latent_std + stack.latent_mean
            volume = decode_sdf_grid(vae, z_dec, resolution=resolution, device=device)
            mesh = sdf_grid_to_mesh(volume)
            report = mesh_report(mesh)
            report['candidate_index'] = candidate_index
            _audit_report(report, cond_names, target, cond_std_np, mask)
            report['pre_newton'] = pre
            report['newton'] = info
            if info is not None:
                drifts.append(info['latent_rms_drift'])
            corrected.append((mesh, report, candidate_index))
        selected = corrected
        newton_summary = {
            'rounds': newton_rounds,
            'targets': dict(targets),
            'step_cap_rms': float(config.get('newton_step_cap_rms', 0.12)),
            'line_search_tries': int(config.get('newton_line_search_tries', 3)),
            'measure_resolution': newton_measure_resolution(config),
            'median_latent_rms_drift': float(np.median(drifts)) if drifts else None,
            'latent_clip': latent_clip,
            # `clipped_latent_fraction` in the metadata is measured on the
            # SAMPLER output; the Newton step is taken after it, so this is the
            # fraction of the exported latents' coordinates sitting on the box.
            'post_newton_clipped_fraction': (
                float((z_n[[c for _, _, c in selected]].abs() >= latent_clip).float().mean().item())
                if latent_clip > 0 else 0.0),
        }

    results = []
    selected_meshes = []
    for i, (mesh, report, _) in enumerate(selected):
        report['index'] = i
        if report['valid']:
            path = os.path.join(out_dir, f'sample_{seed}_{i:03d}.stl')
            mesh.export(path)
            report['path'] = path
            score_text = (f' score={report["condition_score"]:.3f}'
                          if report.get('condition_score') is not None else '')
            print(f'  sample {i:03d}: watertight={report["watertight"]} '
                  f'faces={report["faces"]} extents={np.round(report["extents"], 3).tolist()}'
                  f'{score_text}'
                  f' -> {path}')
        else:
            print(f'  sample {i:03d}: NO ZERO CROSSING (rejected)')
        results.append(report)
        selected_meshes.append(mesh if report['valid'] else None)

    condition_audit = (_condition_summary(results, cond_names, target, mask)
                       if target is not None else None)

    # ---- Optional structural audit of FEA-named conditions ----
    audit_backend = {'requested': str(config.get('condition_audit', 'geometric')).lower(),
                     'used': 'geometric', 'reason': 'unconditional run' if target is None else None}
    fea_audit_meta = None
    if target is not None:
        used, reason = resolve_condition_audit(config, cond_names)
        audit_backend.update(used=used, reason=reason)
        if used != 'geometric':
            entries, fea_audit_meta = fea_condition_audit(
                selected_meshes, cond_names, used, config,
                workdir=os.path.join(out_dir, 'surrogate_audit'))
            for report, entry in zip(results, entries):
                attach_fea_audit(report, entry, cond_names, target)
            measured = [r['fea_audit']['values_raw'] for r in results
                        if r.get('fea_audit') and not r['fea_audit'].get('error')]
            failed = sum(1 for r in results if r.get('fea_audit') and r['fea_audit'].get('error'))
            print(f'{used} audit ({fea_audit_meta["label"]}): {len(measured)} measured, '
                  f'{failed} failed, not measurable: {fea_audit_meta["not_measurable"]}')
            if measured:
                for name in fea_audit_meta['measured_names']:
                    vals = [m[name] for m in measured if m.get(name) is not None]
                    if vals:
                        i = cond_names.index(name)
                        tgt = None if np.isnan(target[i]) else float(CN.from_stored(name, float(target[i])))
                        print(f'  {name}: median actual={np.median(vals):.4g} '
                              f'{fea_audit_meta["units"][name]} '
                              + (f'(requested {tgt:.4g})' if tgt is not None else '(unspecified)'))

    meta = {
        'fm_modelpath': fm_path,
        'vae_modelpath': vae_path,
        'seed': seed,
        'cfg_scale': cfg_scale,
        'ode_steps': ode_steps,
        'mc_resolution': resolution,
        'cond_names': cond_names,
        'cond_values': cond_values,
        'condition_request': condition_request,
        'candidate_multiplier': candidate_multiplier,
        'num_candidates': total_candidates,
        'latent_clip': latent_clip,
        'clipped_latent_fraction': clipped_latent_fraction,
        'velocity_net_calls': counter.calls,
        'nfe_per_candidate': nfe_per_output,
        'sample_seconds': sample_seconds,
        'guidance': guidance_info,
        'newton': newton_summary,
        'descriptor_calibration_path': calibration_path if (guidance_info or newton_summary) else None,
        'condition_audit_backend': audit_backend,
        'fea_audit': fea_audit_meta,
        'condition_audit': condition_audit,
        'results': results,
    }
    meta_path = os.path.join(out_dir, f'sample_{seed}_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(_jsonable(meta), f, indent=2)
    valid = sum(1 for r in results if r['valid'])
    watertight = sum(1 for r in results if r.get('watertight'))
    print(f'\nDone: {valid}/{num_samples} valid, {watertight}/{num_samples} watertight. '
          f'Metadata: {meta_path}')
    if condition_audit:
        print(f'Condition audit: median actual={condition_audit.get("median_actual")}')
        print(f'Condition audit: median relative error={condition_audit.get("median_rel_error")}')
    return meta


# ---------------------------------------------------------------------------
# mode reconstruct
# ---------------------------------------------------------------------------

def _export_recon(mesh, report, path, label):
    """Export one reconstruction (if valid) and print its one-line report."""
    report = dict(report)
    # None on older builds that predate mesh_report's body_count_raw field.
    report.setdefault('body_count_raw', None)
    if report['valid']:
        mesh.export(path)
        report['path'] = path
        print(f'{label}: watertight={report["watertight"]} faces={report["faces"]} '
              f'body_count_raw={report["body_count_raw"]} -> {path}')
    else:
        report['path'] = None
        print(f'{label}: FAILED, NO ZERO CROSSING')
    return report


@torch.no_grad()
def _sign_accuracy(pred, target, eps=0.001):
    """Fraction of |target| > eps points whose predicted sign matches (NaN if none).

    Raw accuracy, so it carries a majority-class floor: the sampler's query sets
    are ~64% outside, and a decoder emitting a positive constant already scores
    ~0.64. `evaluate.sign_accuracy_stats` is the balanced form (mean of the
    inside and outside rates, trivial baseline 0.5) with the class balance
    beside it; compare `reconstruct`'s number against 0.64, not against 0.5.
    """
    mask = target.abs() > eps
    if int(mask.sum()) == 0:
        return float('nan')
    return float((torch.sign(pred[mask]) == torch.sign(target[mask])).float().mean().item())


def run_reconstruct(config, config_filename='config.txt'):
    import trimesh
    from general_modules.sdf_sampling import normalize_mesh, sample_mesh_sdf
    from inference_profiles.latent_refine import refine_latent

    device = resolve_device(config)
    vae_path = config.get('vae_modelpath', '../../output/geometry_generation/sdfflow_vae.pth')
    vae, vae_ckpt = load_vae(vae_path, device)

    input_path = config.get('input_mesh')
    if not input_path:
        raise ValueError("reconstruct mode requires 'input_mesh' in the config")
    resolution = int(config.get('mc_resolution', 128))
    num_enc = int(vae_ckpt['config'].get('num_encoder_points', 4096))
    clamp_dist = float(vae_ckpt['config'].get('clamp_dist', 0.1))
    out_dir = config.get('output_dir', '../../output/geometry_generation/recon')
    seed = int(config.get('seed', 0))
    refine_steps = int(config.get('latent_refine_steps', 0))
    refine_lr = float(config.get('latent_refine_lr', 0.01))
    refine_prior_weight = float(config.get('latent_refine_prior_weight', 0.0))
    if refine_steps < 0:
        raise ValueError('latent_refine_steps must be a nonnegative integer')
    os.makedirs(out_dir, exist_ok=True)

    mesh = trimesh.load(input_path, force='mesh')
    mesh, _, _ = normalize_mesh(mesh)
    labels = None
    if refine_steps > 0:
        # Refinement needs SDF labels for the input mesh: the same sampler the
        # dataset builder uses, seeded, on the normalized mesh. Its surface
        # samples double as the encoder point cloud so both reconstructions see
        # the same surface.
        t0 = time.time()
        labels = sample_mesh_sdf(
            mesh, num_surface=num_enc, num_near=4096, num_uniform=1024,
            rng=np.random.default_rng(seed))
        points = labels['surface_points']
        normals = labels['surface_normals']
        print(f'Sampled {num_enc} surface points and {len(labels["sdf_points"])} SDF labels '
              f'for latent refinement in {time.time() - t0:.2f}s')
    else:
        points, face_idx = trimesh.sample.sample_surface(mesh, num_enc, seed=seed)
        normals = mesh.face_normals[face_idx]

    surface_points = torch.from_numpy(np.asarray(points, dtype=np.float32)).unsqueeze(0).to(device)
    surface_normals = torch.from_numpy(np.asarray(normals, dtype=np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        mu, _ = vae.encode(surface_points, surface_normals)
        z_enc = mu.flatten(1)
        volume = decode_sdf_grid(vae, z_enc, resolution=resolution, device=device)

    base = os.path.splitext(os.path.basename(input_path))[0]
    recon = sdf_grid_to_mesh(volume)
    results = {
        'encoder': _export_recon(
            recon, mesh_report(recon), os.path.join(out_dir, f'{base}_recon.stl'),
            'Reconstruction (encoder mu)'),
    }
    refine_info = None
    if refine_steps > 0:
        query_points = torch.from_numpy(labels['sdf_points']).unsqueeze(0).to(device)
        query_sdf = torch.from_numpy(labels['sdf_values']).unsqueeze(0).to(device)
        history = []
        t0 = time.time()
        z_ref = refine_latent(
            vae, z_enc, surface_points, surface_normals, query_points, query_sdf,
            refine_steps, refine_lr, refine_prior_weight, clamp_dist=clamp_dist,
            history=history)
        refine_seconds = time.time() - t0
        with torch.no_grad():
            sign_enc = _sign_accuracy(vae.decode_flat(z_enc, query_points).squeeze(0).float(),
                                      query_sdf.squeeze(0))
            sign_ref = _sign_accuracy(vae.decode_flat(z_ref, query_points).squeeze(0).float(),
                                      query_sdf.squeeze(0))
            volume = decode_sdf_grid(vae, z_ref, resolution=resolution, device=device)
        latent_shift = float((z_ref - z_enc).norm())
        print(f'Latent refinement: {refine_steps} Adam steps (lr={refine_lr:g}, '
              f'prior_weight={refine_prior_weight:g}) in {refine_seconds:.2f}s; '
              f'loss {history[0][1]:.5f} -> {history[-1][1]:.5f}, '
              f'sign accuracy {sign_enc:.4f} -> {sign_ref:.4f}, '
              f'|z_ref - z_enc| = {latent_shift:.4f}')
        recon_ref = sdf_grid_to_mesh(volume)
        results['refined'] = _export_recon(
            recon_ref, mesh_report(recon_ref),
            os.path.join(out_dir, f'{base}_recon_refined.stl'),
            'Reconstruction (refined latent)')
        refine_info = {
            'latent_refine_steps': refine_steps,
            'latent_refine_lr': refine_lr,
            'latent_refine_prior_weight': refine_prior_weight,
            'clamp_dist': clamp_dist,
            'num_surface': num_enc, 'num_near': 4096, 'num_uniform': 1024, 'label_seed': seed,
            'seconds': refine_seconds,
            'loss_first': history[0][1], 'loss_last': history[-1][1],
            'sign_accuracy_encoder': sign_enc, 'sign_accuracy_refined': sign_ref,
            'latent_shift_l2': latent_shift,
        }

    meta = {
        'vae_modelpath': vae_path,
        'input_mesh': input_path,
        'mc_resolution': resolution,
        'num_encoder_points': num_enc,
        'seed': seed,
        'refinement': refine_info,
        'results': results,
    }
    meta_path = os.path.join(out_dir, f'{base}_recon_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata: {meta_path}')
