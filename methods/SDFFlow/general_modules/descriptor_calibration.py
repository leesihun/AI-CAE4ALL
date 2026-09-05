"""
Affine calibration of the soft descriptor proxy against the export-path truth.

The soft volume/area in `descriptor_proxy.py` are biased but, on the pilot
checkpoint, almost perfectly LINEAR in the Marching Cubes measurement
(volume R^2 = 0.98, `soft ~= 0.86 * true + 0.40`; area R^2 = 0.60). C2 and E2
therefore work in PROXY units: the requested true target is mapped forward,

    proxy_target = a * true_target + b            (`DescriptorCalibration.proxy_target`)

and the residual `a * (target - true)` is expressed in the proxy's own scale
(`descriptor_refinement.py`). This is the pilot's forward map, not the inverse
`(soft - b) / a`.

A calibration is only valid for the exact VAE + FM pair, soft resolution, and
tau it was fitted on, so the artifact records the checkpoint SHA-256s and the
proxy settings, and `check_compatible` refuses a mismatch with a ValueError
instead of silently reusing stale coefficients (GUIDANCE_MECHANISMS 3.1 item 5).
The artifact is a versioned sidecar (`torch.save` of a plain dict, loadable
with `weights_only=True`); nothing is written into a checkpoint.

`calibrate` is the one-shot helper that produces it: generate (or accept)
latents, decode each once for the differentiable proxy and once through the
real export path (`decode_sdf_grid` -> `sdf_grid_to_mesh` -> `true_descriptors`
at `measure_resolution`), and fit per descriptor. `true_descriptors` measures
the mesh the way `mesh_extraction.mesh_report` does -- volume only when the
mesh is watertight -- so a torn decode contributes no volume row to the fit
instead of a convex hull.
"""

import datetime as _dt
import hashlib
import inspect
import os
import time

import numpy as np
import torch

from general_modules.descriptor_proxy import SUPPORTED_SOFT_NAMES, soft_descriptors
from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh
from general_modules.sdf_sampling import COND_NAMES

CALIBRATION_FORMAT = 'sdfflow_descriptor_calibration_v1'


def file_sha256(path, chunk=1 << 20):
    """SHA-256 hex digest of a file's bytes (chunked). Raises FileNotFoundError."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fit_affine(proxy, true):
    """Least-squares fit proxy ~= a * true + b.

    Non-finite pairs are dropped. Returns dict(a, b, r2, n) with plain floats;
    `r2` is the coefficient of determination of the fit on the proxy values.
    Raises ValueError with fewer than two usable pairs or a constant `true`.
    """
    proxy = np.asarray(proxy, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if proxy.shape != true.shape:
        raise ValueError(f'proxy {proxy.shape} and true {true.shape} differ in length')
    keep = np.isfinite(proxy) & np.isfinite(true)
    proxy, true = proxy[keep], true[keep]
    n = int(proxy.size)
    if n < 2:
        raise ValueError(f'fit_affine needs at least 2 finite pairs, got {n}')
    true_mean = true.mean()
    proxy_mean = proxy.mean()
    var_true = float(((true - true_mean) ** 2).sum())
    if var_true <= 0:
        raise ValueError('fit_affine: the true values are constant; the slope is undefined')
    a = float(((true - true_mean) * (proxy - proxy_mean)).sum() / var_true)
    b = float(proxy_mean - a * true_mean)
    residual = proxy - (a * true + b)
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((proxy - proxy_mean) ** 2).sum())
    r2 = 1.0 if ss_tot <= 0 and ss_res <= 0 else (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return {'a': a, 'b': b, 'r2': float(r2), 'n': n}


def true_descriptors(vae, z_flat, measure_resolution=96, bound=1.0, device=None):
    """Export-path measurement of one VAE-space latent [1, D] or [D].

    Returns dict with the five `COND_NAMES` (bbox_x, bbox_y, bbox_z, volume,
    area) as floats, plus `valid`, `watertight`, `body_count_raw`. `valid`
    False means no zero crossing; the descriptor entries are then NaN.

    `volume` is NaN on a NON-WATERTIGHT mesh, exactly as
    `mesh_extraction.mesh_report` reports None for one. It deliberately does
    NOT go through `sdf_sampling.mesh_descriptors`, which substitutes
    `mesh.convex_hull.volume` there -- 2-4x the solid volume on a bracket with
    lightening holes. Every consumer of this function treats the number as the
    export-path truth (the calibration fit, E2's accept/reject line search),
    while the geometric audit that REPORTS the result uses `mesh_report`, so a
    hull substitution makes the two halves of the system disagree about the
    volume of one mesh. NaN is dropped by `fit_affine` and mapped to `inf` by
    `descriptor_refinement.relative_residual`, i.e. such a candidate is
    rejected rather than scored against a hull. `area` is well defined on an
    open surface and is kept.
    """
    z = torch.as_tensor(z_flat)
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.shape[0] != 1:
        raise ValueError(f'true_descriptors measures one latent at a time, got {tuple(z.shape)}')
    device = device if device is not None else z.device
    grid = decode_sdf_grid(vae, z.to(device), resolution=int(measure_resolution),
                           bound=bound, device=device)
    mesh = sdf_grid_to_mesh(grid, bound=bound)
    out = {name: float('nan') for name in COND_NAMES}
    out.update({'valid': False, 'watertight': False, 'body_count_raw': None})
    if mesh is None:
        return out
    watertight = bool(mesh.is_watertight)
    extents = [float(e) for e in mesh.extents]
    out.update({'bbox_x': extents[0], 'bbox_y': extents[1], 'bbox_z': extents[2],
                'volume': float(abs(mesh.volume)) if watertight else float('nan'),
                'area': float(mesh.area)})
    out['valid'] = True
    out['watertight'] = watertight
    out['body_count_raw'] = mesh.metadata.get('body_count_raw')
    return out


class DescriptorCalibration:
    """Per-descriptor affine map proxy ~= a * true + b with its provenance.

    Attributes (all plain Python types so the artifact round-trips through
    `torch.save` / `torch.load(weights_only=True)`):
        coefficients: dict name -> {'a', 'b', 'r2', 'n'}
        resolution, tau: soft-descriptor grid and temperature the fit used
        measure_resolution: Marching Cubes grid of the true measurements
        vae_sha256, fm_sha256: SHA-256 of the checkpoint files ('' = unknown)
        cond_names: the FM checkpoint's condition names (context only)
        split, num_shapes, samples_per_shape: how the calibration set was drawn
        created: ISO-8601 timestamp
        rows: optional per-sample records (proxy/true/valid) for residual audits
    """

    def __init__(self, coefficients, resolution, tau, measure_resolution=96,
                 vae_sha256='', fm_sha256='', cond_names=(), split='', num_shapes=0,
                 samples_per_shape=0, created=None, rows=None, extra=None):
        self.coefficients = {str(k): {kk: (int(vv) if kk == 'n' else float(vv))
                                      for kk, vv in dict(v).items()}
                             for k, v in dict(coefficients).items()}
        self.resolution = int(resolution)
        self.tau = float(tau)
        self.measure_resolution = int(measure_resolution)
        self.vae_sha256 = str(vae_sha256 or '')
        self.fm_sha256 = str(fm_sha256 or '')
        self.cond_names = [str(n) for n in cond_names]
        self.split = str(split or '')
        self.num_shapes = int(num_shapes)
        self.samples_per_shape = int(samples_per_shape)
        self.created = str(created or _dt.datetime.now().isoformat(timespec='seconds'))
        self.rows = list(rows or [])
        self.extra = dict(extra or {})

    # ---- accessors -------------------------------------------------------
    @property
    def names(self):
        return tuple(self.coefficients.keys())

    def __contains__(self, name):
        return name in self.coefficients

    def __getitem__(self, name):
        return self.coefficients[name]

    def slope(self, name):
        return float(self.coefficients[name]['a'])

    def proxy_target(self, name, true_target):
        """Forward map of a TRUE target into proxy units: a * true + b."""
        coef = self.coefficients[name]
        return float(coef['a']) * float(true_target) + float(coef['b'])

    def describe(self):
        lines = [f'DescriptorCalibration(resolution={self.resolution}, tau={self.tau:g}, '
                 f'measure_resolution={self.measure_resolution}, split={self.split!r}, '
                 f'shapes={self.num_shapes} x {self.samples_per_shape}, created={self.created})']
        for name, coef in self.coefficients.items():
            lines.append(f'  {name:>8s}: proxy = {coef["a"]:.4f} * true + {coef["b"]:+.4f}'
                         f'   r2={coef["r2"]:.4f}  n={coef["n"]}')
        return '\n'.join(lines)

    # ---- (de)serialisation ------------------------------------------------
    def to_dict(self):
        return {
            'format': CALIBRATION_FORMAT,
            'coefficients': {k: dict(v) for k, v in self.coefficients.items()},
            'resolution': self.resolution,
            'tau': self.tau,
            'measure_resolution': self.measure_resolution,
            'vae_sha256': self.vae_sha256,
            'fm_sha256': self.fm_sha256,
            'cond_names': list(self.cond_names),
            'split': self.split,
            'num_shapes': self.num_shapes,
            'samples_per_shape': self.samples_per_shape,
            'created': self.created,
            'rows': [dict(r) for r in self.rows],
            'extra': dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload):
        fmt = payload.get('format')
        if fmt != CALIBRATION_FORMAT:
            raise ValueError(f'not a descriptor calibration artifact (format={fmt!r}, '
                             f'expected {CALIBRATION_FORMAT!r})')
        return cls(
            coefficients=payload['coefficients'], resolution=payload['resolution'],
            tau=payload['tau'], measure_resolution=payload.get('measure_resolution', 96),
            vae_sha256=payload.get('vae_sha256', ''), fm_sha256=payload.get('fm_sha256', ''),
            cond_names=payload.get('cond_names', ()), split=payload.get('split', ''),
            num_shapes=payload.get('num_shapes', 0),
            samples_per_shape=payload.get('samples_per_shape', 0),
            created=payload.get('created'), rows=payload.get('rows'), extra=payload.get('extra'))

    def save(self, path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        torch.save(self.to_dict(), path)
        return path

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f'descriptor calibration not found: {path}')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        return cls.from_dict(payload)

    # ---- guard ------------------------------------------------------------
    def check_compatible(self, vae_path, fm_path, resolution, tau, measure_resolution=None,
                         names=None, tau_tol=1e-9):
        """Raise ValueError unless this calibration was fitted for exactly this
        VAE file, FM file, soft resolution, and tau (and covers `names`).

        A path of None skips that checkpoint's hash check; a recorded hash of
        '' (calibration built without files) also skips it.

        `measure_resolution`, when given, also pins the Marching Cubes grid the
        slope was fitted against: `a` and `b` map the proxy onto the export
        path AT THAT GRID, and MC volume/area move with resolution (measured on
        an analytic sphere: +0.09% / +0.02% in volume from res 96 to 128 at
        r = 0.35 / 0.70), which is the same order as the accuracy C2/E2 are
        quoted at. Pass the grid the run will actually measure and report on
        (`newton_measure_resolution`, else `mc_resolution`).
        """
        problems = []
        for label, path, recorded in (('VAE', vae_path, self.vae_sha256),
                                      ('FM', fm_path, self.fm_sha256)):
            if path is None or not recorded:
                continue
            if not os.path.exists(path):
                problems.append(f'{label} checkpoint {path} does not exist')
                continue
            actual = file_sha256(path)
            if actual != recorded:
                problems.append(f'{label} checkpoint {path} sha256 {actual[:12]}... differs from '
                                f'the calibrated {recorded[:12]}...')
        if int(resolution) != self.resolution:
            problems.append(f'soft_descriptor_resolution {int(resolution)} != calibrated '
                            f'{self.resolution}')
        if abs(float(tau) - self.tau) > tau_tol:
            problems.append(f'soft_descriptor_tau {float(tau):g} != calibrated {self.tau:g}')
        if measure_resolution is not None and int(measure_resolution) != self.measure_resolution:
            problems.append(
                f'the run measures the true descriptors on a Marching Cubes grid of '
                f'{int(measure_resolution)} but this calibration was fitted against '
                f'{self.measure_resolution}; set newton_measure_resolution / mc_resolution to '
                f'{self.measure_resolution}, or re-fit the calibration at '
                f'{int(measure_resolution)}')
        if names is not None:
            missing = [n for n in names if n not in self.coefficients]
            if missing:
                problems.append(f'calibration has no coefficients for {missing} '
                                f'(has {list(self.coefficients)})')
        if problems:
            raise ValueError('descriptor calibration is not compatible with this run: '
                             + '; '.join(problems)
                             + '. Re-run the descriptor_calibration eval task for this '
                             'checkpoint pair and proxy settings.')
        return True


def _sample_latents_compat(fm_model, num, latent_flat_dim, device, cond, cfg_scale, ode_steps,
                           generator, noise, cond_mask):
    """`sample_latents` with `cond_mask`, tolerating a sampler that predates it."""
    from model.velocity_net import sample_latents
    kwargs = dict(cond=cond, cfg_scale=cfg_scale, ode_steps=ode_steps, generator=generator,
                  noise=noise)
    params = inspect.signature(sample_latents).parameters
    if 'cond_mask' in params:
        kwargs['cond_mask'] = cond_mask
    elif cond_mask is not None:
        raise TypeError('this sample_latents has no cond_mask parameter but a cond_mask was given')
    return sample_latents(fm_model, num, latent_flat_dim, device, **kwargs)


def calibrate(vae, fm_model, latent_mean, latent_std, cond_batches=None, z_batches=None, *,
              names=SUPPORTED_SOFT_NAMES, resolution=48, tau=0.032, measure_resolution=96,
              bound=1.0, chunk=32768, ode_steps=50, cfg_scale=1.0, cond_mask_batches=None,
              noise_batches=None, generator=None, device=None, vae_path=None, fm_path=None,
              cond_names=(), split='', num_shapes=0, samples_per_shape=0, verbose=True,
              extra=None, min_r2=0.0):
    """Generate/decode a calibration set and fit the proxy -> true affine maps.

    Exactly one of `cond_batches` / `z_batches` must be given:
        cond_batches: iterable of NORMALIZED condition tensors [b, cond_dim]
            (or None entries for unconditional batches); each is integrated
            with `sample_latents(fm_model, ...)` to a normalized latent.
        z_batches: iterable of NORMALIZED flat latents [b, D] (skips the FM).
    `latent_mean` / `latent_std` ([1, D] or [D]) de-normalize to VAE space.
    `cond_mask_batches` / `noise_batches` are optional parallel iterables
    forwarded to `sample_latents` (per-dim partial conditions, fixed noise).

    Per row the proxy is `soft_descriptors(vae, z, names, resolution, tau)`
    under no_grad and the truth `true_descriptors(..., measure_resolution)`;
    rows with an invalid mesh are recorded but excluded from the fit, and a
    non-watertight row contributes no `volume` pair (its true volume is NaN,
    which `fit_affine` drops) while its `area` pair is kept -- area is well
    defined on an open surface, volume is not. The watertight rate of the set is
    printed and recorded in `extra`, because a low one means the volume fit
    rests on fewer rows than the header count suggests.
    `vae_path` / `fm_path`, when given, are hashed into the artifact.

    `min_r2` (0 = off) refuses to return a calibration whose per-name fit is
    weaker than that: `a` and `b` ARE the mechanism C2/E2 work through, and the
    pilot's uncalibrated round 1 made volume error worse (7.6% -> 23%), so a
    slope fitted at r2 0.2 is worse than no tool at all.

    Returns a `DescriptorCalibration` (its `.rows` hold every sample).
    """
    names = tuple(names)
    unsupported = [n for n in names if n not in SUPPORTED_SOFT_NAMES]
    if unsupported:
        raise ValueError(f'no soft proxy for {unsupported}; supported: {SUPPORTED_SOFT_NAMES}')
    if (cond_batches is None) == (z_batches is None):
        raise ValueError('calibrate needs exactly one of cond_batches or z_batches')
    latent_mean = torch.as_tensor(latent_mean).float().reshape(1, -1)
    latent_std = torch.as_tensor(latent_std).float().reshape(1, -1)
    latent_flat_dim = latent_mean.shape[1]
    if device is None:
        device = latent_mean.device
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)

    rows = []
    t_start = time.time()
    if z_batches is not None:
        z_iter = ((torch.as_tensor(z).float().to(device), None) for z in z_batches)
    else:
        if fm_model is None:
            raise ValueError('calibrate with cond_batches requires fm_model')
        cond_batches = list(cond_batches)
        masks = list(cond_mask_batches) if cond_mask_batches is not None else [None] * len(cond_batches)
        noises = list(noise_batches) if noise_batches is not None else [None] * len(cond_batches)
        if len(masks) != len(cond_batches) or len(noises) != len(cond_batches):
            raise ValueError('cond_mask_batches / noise_batches must parallel cond_batches')

        def _generate():
            for cond, mask, noise in zip(cond_batches, masks, noises):
                if cond is not None:
                    cond = torch.as_tensor(cond).float().to(device)
                    b = cond.shape[0]
                elif noise is not None:
                    b = int(torch.as_tensor(noise).shape[0])
                else:
                    raise ValueError('an unconditional calibration batch needs `noise` to fix its size')
                if mask is not None:
                    mask = torch.as_tensor(mask).to(device=device, dtype=torch.bool)
                if noise is not None:
                    noise = torch.as_tensor(noise).float().to(device)
                z_n = _sample_latents_compat(fm_model, b, latent_flat_dim, device, cond, cfg_scale,
                                             ode_steps, generator, noise, mask)
                yield z_n, cond
        z_iter = _generate()

    index = 0
    for z_n, cond in z_iter:
        z = z_n * latent_std + latent_mean
        with torch.no_grad():
            proxy = soft_descriptors(vae, z, names=names, resolution=resolution, tau=tau,
                                     bound=bound, chunk=chunk)
        for i in range(z.shape[0]):
            truth = true_descriptors(vae, z[i:i + 1], measure_resolution=measure_resolution,
                                     bound=bound, device=device)
            row = {'index': index, 'valid': bool(truth['valid']),
                   'watertight': bool(truth['watertight']),
                   'body_count_raw': truth['body_count_raw'],
                   'latent_rms_normalized': float(z_n[i].float().pow(2).mean().sqrt().item())}
            for name in names:
                row[f'proxy_{name}'] = float(proxy[name][i].item())
                row[f'true_{name}'] = float(truth[name])
            if cond is not None:
                row['cond_normalized'] = [float(v) for v in cond[i].tolist()]
            rows.append(row)
            index += 1
        if verbose:
            print(f'  calibration: {index} samples measured ({time.time() - t_start:.1f}s)', flush=True)

    coefficients = {}
    valid_rows = [r for r in rows if r['valid']]
    watertight_rows = sum(1 for r in valid_rows if r.get('watertight'))
    watertight_rate = (watertight_rows / len(valid_rows)) if valid_rows else None
    for name in names:
        coefficients[name] = fit_affine([r[f'proxy_{name}'] for r in valid_rows],
                                        [r[f'true_{name}'] for r in valid_rows])
    weak = {n: coefficients[n]['r2'] for n in names if coefficients[n]['r2'] < float(min_r2)}
    if weak:
        raise ValueError(
            f'descriptor calibration fit is too weak to use: {weak} below min_r2={float(min_r2):g} '
            f'(rows: {len(valid_rows)} valid of {len(rows)}, watertight rate '
            f'{"n/a" if watertight_rate is None else f"{watertight_rate:.3f}"}). C2/E2 apply '
            'proxy_target = a * true + b, so a poorly determined slope makes the correction '
            'worse, not weaker. Fit on more shapes, or lower calibration_min_r2 deliberately.')
    calibration = DescriptorCalibration(
        coefficients=coefficients, resolution=resolution, tau=tau,
        measure_resolution=measure_resolution,
        vae_sha256=file_sha256(vae_path) if vae_path else '',
        fm_sha256=file_sha256(fm_path) if fm_path else '',
        cond_names=cond_names, split=split, num_shapes=num_shapes,
        samples_per_shape=samples_per_shape, rows=rows,
        extra=dict(extra or {}, valid_rows=len(valid_rows), total_rows=len(rows),
                   watertight_rows=watertight_rows, watertight_rate=watertight_rate,
                   min_r2=float(min_r2),
                   volume_fit_rows=coefficients.get('volume', {}).get('n'),
                   ode_steps=int(ode_steps), cfg_scale=float(cfg_scale), bound=float(bound),
                   seconds=float(time.time() - t_start)))
    if verbose:
        print(calibration.describe(), flush=True)
        print(f'  rows: {len(valid_rows)}/{len(rows)} valid, watertight rate '
              + ('n/a' if watertight_rate is None else f'{watertight_rate:.3f}')
              + ' (a non-watertight row has no true volume, so it is dropped from the volume '
                'fit only -- read `n` per name above)', flush=True)
    return calibration
