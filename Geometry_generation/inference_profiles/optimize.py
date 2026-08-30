"""Mode `optimize`: closed-loop geometry optimization over the trained generator.

Chains the four stages the suite already owns separately -- SDFFlow generation,
gmsh meshing, a linear-static structural solve, and a population search -- into
one run driven by the same flat config format as every other mode.
"""

import json
import os
import time

import numpy as np

from design_loop import fea
from design_loop.generator import SDFFlowGenerator
from design_loop.loop import Evaluator, baseline_population, calibrate, run, save_history
from design_loop.problem import Bracket, MassObjective


def _as_list(value):
    if value is None:
        return None
    return [v.strip() for v in str(value).split(',')] if isinstance(value, str) else list(value)


def _flt(config, key, default):
    value = config.get(key, default)
    return float(value) if value is not None else None


def _int(config, key, default):
    value = config.get(key, default)
    return int(value) if value is not None else None


def run_optimize(config, config_filename='config.txt'):
    out_dir = config.get('output_dir', './outputs/optimization')
    os.makedirs(out_dir, exist_ok=True)
    started = time.time()

    # ---- Generator: the shape parameterization -------------------------- #
    subspace_dim = _int(config, 'opt_subspace_dim', 12)
    generator = SDFFlowGenerator(
        config.get('vae_modelpath'),
        config.get('fm_modelpath', './outputs/sdfflow_fm.pth'),
        device='cuda' if str(config.get('gpu_ids', 0)) != 'cpu' else 'cpu',
        subspace_dim=subspace_dim,
        subspace_seed=_int(config, 'opt_subspace_seed', 0),
        base_seed=_int(config, 'seed', 0),
        ode_steps=_int(config, 'ode_steps', 50),
        mc_resolution=_int(config, 'mc_resolution', 128),
        cond_dims=tuple(_as_list(config.get('opt_condition_dims', 'volume,area')) or ()),
        shell_scale=_flt(config, 'opt_shell_scale', 1.25),
        latent_range=_flt(config, 'opt_latent_range', None),
    )
    print(f'Design space: {subspace_dim} latent directions + '
          f'{len(generator.cond_dims)} conditions {generator.cond_dims} '
          f'= {generator.n_design} variables', flush=True)

    # ---- Structural problem --------------------------------------------- #
    material = fea.Material(
        E=_flt(config, 'opt_material_e', 113.8e9),
        nu=_flt(config, 'opt_material_nu', 0.342),
        rho=_flt(config, 'opt_material_rho', 4430.0),
        yield_stress=_flt(config, 'opt_yield_stress', 903e6),
    )
    load_cases = tuple(_as_list(config.get('opt_load_cases', 'vertical,diagonal')))
    unknown = set(load_cases) - set(Bracket.LOAD_CASES)
    if unknown:
        raise ValueError(f'unknown load case(s) {sorted(unknown)}; '
                         f'available: {sorted(Bracket.LOAD_CASES)}')
    bracket = Bracket(material=material,
                      length_scale=_flt(config, 'opt_length_scale', 0.19 / 1.8),
                      load_cases=load_cases,
                      stress_percentile=_flt(config, 'opt_stress_percentile', 99.5))
    print(f'Structural problem: {material.name}, load cases {load_cases}, '
          f'part length {bracket.length_scale * 1.8 * 1e3:.0f} mm', flush=True)

    evaluator = Evaluator(generator, bracket,
                          mesh_size_max=_flt(config, 'opt_mesh_size_max', 0.05),
                          target_faces=_int(config, 'opt_target_faces', 12000))

    # ---- Stage 1: baseline population and calibration -------------------- #
    baseline_size = _int(config, 'opt_baseline_size', 12)
    print(f'\n=== Baseline population ({baseline_size} designs) ===', flush=True)
    baseline = baseline_population(evaluator, size=baseline_size,
                                   seed=_int(config, 'opt_seed', 0))
    limits = calibrate(baseline,
                       stress_margin=_flt(config, 'opt_stress_margin', 1.0),
                       disp_margin=_flt(config, 'opt_disp_margin', 1.0))
    yield_ratio = limits['stress_allow'] / material.yield_stress
    print(f"\nCalibrated from {limits['population']} analyzed designs:"
          f"\n  mass_ref     {limits['mass_ref']:.4f} kg"
          f"   (range {limits['mass_range'][0]:.4f}-{limits['mass_range'][1]:.4f})"
          f"\n  stress_allow {limits['stress_allow'] / 1e6:.1f} MPa"
          f"   ({yield_ratio * 100:.1f}% of yield)"
          f"\n  disp_allow   {limits['disp_allow'] * 1e3:.4f} mm", flush=True)

    evaluator.objective = MassObjective(
        mass_ref=limits['mass_ref'],
        stress_allow=limits['stress_allow'],
        disp_allow=limits['disp_allow'],
        stress_weight=_flt(config, 'opt_stress_weight', 6.0),
        disp_weight=_flt(config, 'opt_disp_weight', 3.0),
    )
    baseline_scored = []
    for record in baseline:
        if record['ok']:
            score, penalty = evaluator.objective(record['fea'])
            record['score'], record['penalty'] = score, penalty
            baseline_scored.append(record)
    reference = min(baseline_scored, key=lambda r: r['score'])
    print(f"  best baseline score {reference['score']:.4f} "
          f"(mass {reference['fea']['mass']:.4f} kg)", flush=True)
    # Two honest references: the median member is what "a typical DeepJEB
    # bracket" means, the best member is the harder target the search starts from.
    typical = sorted(baseline_scored, key=lambda r: r['fea']['mass'])[len(baseline_scored) // 2]
    print(f"  typical (median-mass) baseline: mass {typical['fea']['mass']:.4f} kg", flush=True)

    # ---- Stage 2: CMA-ES over the design space --------------------------- #
    budget = _int(config, 'opt_budget', 120)
    popsize = _int(config, 'opt_popsize', 8)
    print(f'\n=== CMA-ES search (budget {budget} evaluations, popsize {popsize}) ===',
          flush=True)
    evaluator.history = []          # the search log is separate from the baseline
    best_x, best_score, log = run(
        evaluator,
        x0=np.asarray(reference['x']),
        sigma0=_flt(config, 'opt_sigma0', 1.0),
        budget=budget,
        popsize=popsize,
        seed=_int(config, 'opt_seed', 0),
    )
    search_history = evaluator.history

    # ---- Stage 3: refined verification of the winner --------------------- #
    print('\n=== Verification of the best design at refined resolution ===', flush=True)
    verify_res = _int(config, 'opt_verify_resolution', 160)
    verify_faces = _int(config, 'opt_verify_target_faces', 30000)
    verify_size = _flt(config, 'opt_verify_mesh_size_max', 0.035)
    search_best = min((r for r in search_history if r['ok']), key=lambda r: r['score'],
                      default=None)
    verified = evaluator.analyze(best_x, mc_resolution=verify_res,
                                 target_faces=verify_faces, mesh_size_max=verify_size,
                                 return_fields=True)
    reference_verified = evaluator.analyze(np.asarray(reference['x']),
                                           mc_resolution=verify_res,
                                           target_faces=verify_faces,
                                           mesh_size_max=verify_size,
                                           return_fields=True)
    typical_verified = evaluator.analyze(np.asarray(typical['x']),
                                         mc_resolution=verify_res,
                                         target_faces=verify_faces,
                                         mesh_size_max=verify_size)

    for tag, record in (('optimized', verified), ('baseline', reference_verified),
                        ('typical', typical_verified)):
        if record['ok']:
            mesh = record.pop('mesh_object', None)
            if mesh is None:
                print(f"  {tag:9s}: (mesh already released)", flush=True)
            path = os.path.join(out_dir, f'{tag}.stl')
            mesh.export(path)
            record['stl'] = path
            f = record['fea']
            print(f"  {tag:9s}: mass {f['mass']:.4f} kg | "
                  f"peak vM {f['peak_von_mises'] / 1e6:.1f} MPa | "
                  f"max disp {f['max_displacement'] * 1e3:.4f} mm | "
                  f"{f['num_tets']} tets", flush=True)
        else:
            print(f'  {tag:9s}: FAILED {record["error"]}', flush=True)

    # ---- Reporting -------------------------------------------------------- #
    summary = _summarize(limits, reference, reference_verified, verified,
                         best_x, best_score, log, search_history, baseline,
                         evaluator, material, load_cases, bracket, time.time() - started,
                         search_best, typical_verified)
    summary['design_space'] = {
        'subspace_dim': generator.subspace_dim,
        'condition_dims': list(generator.cond_dims),
        'n_design': generator.n_design,
        'latent_range': generator.latent_range,
        'shell_scale': generator.shell_scale,
        'subspace_seed': _int(config, 'opt_subspace_seed', 0),
        'base_seed': _int(config, 'seed', 0),
        'ode_steps': generator.ode_steps,
    }
    summary['verification_settings'] = {
        'mc_resolution': verify_res, 'target_faces': verify_faces,
        'mesh_size_max': verify_size,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=float)
    _strip = ('mesh_object', 'traceback', 'fields')
    save_history(os.path.join(out_dir, 'history.json'), evaluator,
                 extra={'baseline': [{k: v for k, v in r.items() if k not in _strip}
                                     for r in baseline],
                        'search': [{k: v for k, v in r.items() if k not in _strip}
                                   for r in search_history],
                        'limits': limits})
    _plot(out_dir, baseline, search_history, log, limits)
    if verified['ok'] and reference_verified['ok']:
        try:
            from design_loop.visualize import render_comparison
            path = render_comparison(
                os.path.join(out_dir, 'stress_comparison.png'),
                reference_verified['fields'], reference_verified['fea'],
                verified['fields'], verified['fea'],
                verified['fea']['worst_case'])
            print(f'  wrote {path}', flush=True)
        except Exception as exc:
            print(f'  (skipping stress render: {type(exc).__name__}: {exc})', flush=True)
    _write_report(out_dir, summary)

    print(f'\nWrote results to {out_dir}', flush=True)
    print(f"Total wall time {summary['wall_time_s'] / 60:.1f} min "
          f"over {summary['total_evaluations']} analyses", flush=True)
    return summary


def _summarize(limits, reference, reference_verified, verified, best_x, best_score,
               log, search_history, baseline, evaluator, material, load_cases,
               bracket, wall_time, search_best=None, typical_verified=None):
    ok = [r for r in search_history if r['ok']]
    feasible = [r for r in ok if r['penalty'].get('feasible')]
    summary = {
        'wall_time_s': wall_time,
        'total_evaluations': len(baseline) + len(search_history) + 2,
        'search_evaluations': len(search_history),
        'search_success_rate': len(ok) / max(len(search_history), 1),
        'feasible_designs': len(feasible),
        'failures': evaluator.failures,
        'limits': limits,
        'material': {'name': material.name, 'E': material.E, 'nu': material.nu,
                     'rho': material.rho, 'yield_stress': material.yield_stress},
        'load_cases': list(load_cases),
        'length_scale': bracket.length_scale,
        'best_x': np.asarray(best_x).tolist(),
        'best_search_score': float(best_score),
        'convergence': log,
    }
    ok_baseline = [r for r in baseline if r['ok']]
    if ok_baseline:
        summary['baseline_population'] = {
            'median_mass_kg': float(np.median([r['fea']['mass'] for r in ok_baseline])),
            'best_of_population_mass_kg': float(reference['fea']['mass']),
        }
    # Same design, search mesh vs verification mesh: the discretization sensitivity.
    if search_best is not None and search_best['ok'] and verified['ok']:
        a, b = search_best['fea'], verified['fea']
        summary['mesh_sensitivity'] = {
            'search_tets': a['num_tets'], 'verify_tets': b['num_tets'],
            'mass_change_pct': 100.0 * (b['mass'] - a['mass']) / a['mass'],
            'peak_stress_change_pct': 100.0 * (b['peak_von_mises'] - a['peak_von_mises'])
            / a['peak_von_mises'],
            'disp_change_pct': 100.0 * (b['max_displacement'] - a['max_displacement'])
            / a['max_displacement'],
        }
    if verified['ok'] and reference_verified['ok']:
        a, b = reference_verified['fea'], verified['fea']
        summary['verified'] = {
            'baseline': _brief(a), 'optimized': _brief(b),
            'mass_change_pct': 100.0 * (b['mass'] - a['mass']) / a['mass'],
            'stress_change_pct': 100.0 * (b['peak_von_mises'] - a['peak_von_mises'])
            / a['peak_von_mises'],
            'disp_change_pct': 100.0 * (b['max_displacement'] - a['max_displacement'])
            / a['max_displacement'],
            'stiffness_to_mass_gain_pct': 100.0 * (
                (a['max_compliance'] * a['mass']) / (b['max_compliance'] * b['mass']) - 1.0),
        }
        if typical_verified is not None and typical_verified['ok']:
            t = typical_verified['fea']
            summary['verified']['typical'] = _brief(t)
            summary['verified']['vs_typical'] = {
                'mass_change_pct': 100.0 * (b['mass'] - t['mass']) / t['mass'],
                'stress_change_pct': 100.0 * (b['peak_von_mises'] - t['peak_von_mises'])
                / t['peak_von_mises'],
                'disp_change_pct': 100.0 * (b['max_displacement'] - t['max_displacement'])
                / t['max_displacement'],
            }
    return summary


def _brief(f):
    return {
        'mass_kg': f['mass'],
        'peak_von_mises_MPa': f['peak_von_mises'] / 1e6,
        'max_von_mises_MPa': f['max_von_mises'] / 1e6,
        'max_displacement_mm': f['max_displacement'] * 1e3,
        'max_compliance_J': f['max_compliance'],
        'num_tets': f['num_tets'],
        'cases': {k: {'peak_von_mises_MPa': v['peak_von_mises'] / 1e6,
                      'max_displacement_mm': v['max_displacement'] * 1e3,
                      'compliance_J': v['compliance']} for k, v in f['cases'].items()},
    }


def _plot(out_dir, baseline, search_history, log, limits):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'  (skipping convergence plot: {exc})', flush=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    evals = [e['evaluations'] for e in log]
    axes[0].plot(evals, [e['best_score'] for e in log], 'o-', label='best so far')
    axes[0].plot(evals, [e['generation_median'] for e in log], 's--', alpha=0.6,
                 label='generation median')
    axes[0].set_xlabel('FEA evaluations')
    axes[0].set_ylabel('objective  (mass ratio + penalties)')
    axes[0].set_title('CMA-ES convergence')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    base_ok = [r for r in baseline if r['ok']]
    srch_ok = [r for r in search_history if r['ok']]
    axes[1].scatter([r['fea']['mass'] for r in base_ok],
                    [r['fea']['peak_von_mises'] / 1e6 for r in base_ok],
                    marker='s', s=45, label='baseline population')
    axes[1].scatter([r['fea']['mass'] for r in srch_ok],
                    [r['fea']['peak_von_mises'] / 1e6 for r in srch_ok],
                    c=[r['index'] for r in srch_ok], cmap='viridis', s=22,
                    label='search', alpha=0.85)
    axes[1].axhline(limits['stress_allow'] / 1e6, color='crimson', ls='--',
                    label='stress allowable')
    axes[1].axvline(limits['mass_ref'], color='gray', ls=':', label='reference mass')
    axes[1].set_xlabel('mass (kg)')
    axes[1].set_ylabel('peak von Mises (MPa)')
    axes[1].set_title('Design space explored')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    axes[2].scatter([r['fea']['mass'] for r in base_ok],
                    [r['fea']['max_displacement'] * 1e3 for r in base_ok],
                    marker='s', s=45, label='baseline population')
    axes[2].scatter([r['fea']['mass'] for r in srch_ok],
                    [r['fea']['max_displacement'] * 1e3 for r in srch_ok],
                    c=[r['index'] for r in srch_ok], cmap='viridis', s=22,
                    label='search', alpha=0.85)
    axes[2].axhline(limits['disp_allow'] * 1e3, color='crimson', ls='--',
                    label='deflection allowable')
    axes[2].set_xlabel('mass (kg)')
    axes[2].set_ylabel('max displacement (mm)')
    axes[2].set_title('Mass vs deflection')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, 'convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}', flush=True)


def _write_report(out_dir, summary):
    lines = ['# DeepJEB closed-loop geometry optimization', '']
    lines.append(f"Wall time {summary['wall_time_s'] / 60:.1f} min over "
                 f"{summary['total_evaluations']} generate-mesh-solve evaluations "
                 f"({summary['search_success_rate'] * 100:.0f}% of search evaluations "
                 f"completed the chain).")
    lines.append('')
    limits = summary['limits']
    lines += ['## Calibration', '',
              f"Allowables anchored to the median of a {limits['population']}-design "
              'random population from the same generator:', '',
              f"- reference mass **{limits['mass_ref']:.4f} kg**",
              f"- stress allowable **{limits['stress_allow'] / 1e6:.1f} MPa**",
              f"- deflection allowable **{limits['disp_allow'] * 1e3:.4f} mm**", '']
    if 'baseline_population' in summary:
        bp = summary['baseline_population']
        lines += [f"Population median mass {bp['median_mass_kg']:.4f} kg; the best-scoring "
                  f"member ({bp['best_of_population_mass_kg']:.4f} kg) is the comparison "
                  'baseline below, which is the harder reference of the two.', '']
    if 'verified' in summary:
        v = summary['verified']
        a, b = v['baseline'], v['optimized']
        lines += ['## Verified result (refined mesh)', '',
                  '| quantity | baseline | optimized | change |',
                  '| --- | ---: | ---: | ---: |',
                  f"| mass (kg) | {a['mass_kg']:.4f} | {b['mass_kg']:.4f} | "
                  f"{v['mass_change_pct']:+.1f}% |",
                  f"| peak von Mises (MPa) | {a['peak_von_mises_MPa']:.1f} | "
                  f"{b['peak_von_mises_MPa']:.1f} | {v['stress_change_pct']:+.1f}% |",
                  f"| max displacement (mm) | {a['max_displacement_mm']:.4f} | "
                  f"{b['max_displacement_mm']:.4f} | {v['disp_change_pct']:+.1f}% |",
                  f"| max compliance (J) | {a['max_compliance_J']:.4f} | "
                  f"{b['max_compliance_J']:.4f} | |",
                  f"| tetrahedra | {a['num_tets']} | {b['num_tets']} | |", '',
                  f"Stiffness-per-unit-mass gain: **{v['stiffness_to_mass_gain_pct']:+.1f}%**",
                  '']
        if 'vs_typical' in v:
            t, vt = v['typical'], v['vs_typical']
            lines += ['### Against a typical population member', '',
                      'The table above compares against the *best* of the random '
                      'population, which is the hardest reference available. Against '
                      'the median-mass member -- what "a typical DeepJEB bracket" '
                      'means -- the same optimized design is:', '',
                      f"- mass {t['mass_kg']:.4f} -> {b['mass_kg']:.4f} kg "
                      f"(**{vt['mass_change_pct']:+.1f}%**)",
                      f"- peak von Mises {t['peak_von_mises_MPa']:.1f} -> "
                      f"{b['peak_von_mises_MPa']:.1f} MPa ({vt['stress_change_pct']:+.1f}%)",
                      f"- max displacement {t['max_displacement_mm']:.4f} -> "
                      f"{b['max_displacement_mm']:.4f} mm ({vt['disp_change_pct']:+.1f}%)", '']
    lines += ['## Solver', '',
              'Linear-static 4-node tetrahedra, AMG-preconditioned CG, nodal von Mises '
              'volume-averaged from the constant per-element stress. The element passes '
              'the constant-strain patch test to 1e-9 but shear-locks: on a slender '
              'cantilever it recovers 0.40/0.65/0.83/0.90 of the Timoshenko tip '
              'deflection at 288/1.3k/6k/16.5k tets. Absolute deflection and stress here '
              'are therefore optimistic; the baseline-vs-optimized comparison at equal '
              'discretization is the meaningful quantity.', '']

    if 'mesh_sensitivity' in summary:
        m = summary['mesh_sensitivity']
        lines += ['## Discretization sensitivity', '',
                  f"The same winning design re-analyzed on the verification mesh "
                  f"({m['search_tets']} -> {m['verify_tets']} tets) moves by "
                  f"{m['mass_change_pct']:+.1f}% in mass, "
                  f"{m['peak_stress_change_pct']:+.1f}% in peak von Mises and "
                  f"{m['disp_change_pct']:+.1f}% in deflection. The search mesh is therefore "
                  'a ranking device, not a converged absolute; baseline and optimized are '
                  'compared above on the same refined mesh so the comparison is unaffected.',
                  '']

    if summary['failures']:
        lines += ['## Failure modes', '']
        lines += [f'- `{k}`: {n}' for k, n in sorted(summary['failures'].items())]
        lines.append('')
    path = os.path.join(out_dir, 'report.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'  wrote {path}', flush=True)
