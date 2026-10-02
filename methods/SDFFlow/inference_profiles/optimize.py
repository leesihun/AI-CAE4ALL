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
from design_loop.deepjeb_bridge import LABEL_MESH_SIZE_MAX, LABEL_SURFACE_FACES, apply_frame
from design_loop.generator import NOISE_PARAM_DEFAULT, SDFFlowGenerator
from design_loop.loop import (
    Evaluator, baseline_population, calibrate, run, save_history, select_best,
)
from design_loop.problem import Bracket, MassObjective
from design_loop.surrogate import (
    HIMGNSurrogate, SurrogateEvaluator, gpu_id_from_config, surrogate_baseline_population,
    surrogate_search,
)
from training_profiles.setup import resolve_device

# Bracket.LOAD_CASES uses the GE-challenge names this suite's FEA path was
# written against; DeepJEB's own files (and therefore the surrogate's training
# data) use its short codes. One config vocabulary (the GE names) covers both
# backends so `opt_load_cases` never has to change when `opt_analysis` does.
_SURROGATE_CASE_NAMES = {'vertical': 'ver', 'horizontal': 'hor',
                        'diagonal': 'dia', 'torsion': 'tor'}


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


def _flag(config, key, default=False):
    # The launcher's `_flag` reads the same spellings (cae_suite/specs/sdfflow.py).
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {'', 'false', '0', 'no', 'off'}
    return bool(value)


# Every file this mode writes into output_dir. A re-run into the same folder
# must not leave a previous run's artifact beside the new ones: a crash after
# the search, or a backend without a stress render, would otherwise present the
# old summary/picture as this run's result. fea_verified.json is here because
# it describes the STLs it was solved on, which a re-run replaces.
_OWNED_OUTPUTS = ('summary.json', 'history.json', 'convergence.png',
                  'stress_comparison.png', 'report.md', 'screening.csv',
                  'optimized.stl', 'baseline.stl', 'typical.stl', 'fea_verified.json',
                  'optimized_mm.stl', 'baseline_mm.stl', 'typical_mm.stl')

SCREENING_TABLE_NAME = 'screening.csv'


def _clear_owned_outputs(out_dir):
    removed = []
    for name in _OWNED_OUTPUTS:
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(name)
    if removed:
        print(f"Cleared previous run's outputs in {out_dir}: {', '.join(removed)}", flush=True)


# The constants the DeepJEB surrogate labels were solved with
# (build_deepjeb_fea.py builds `Bracket()` with every default).
# The launcher's SDF-OPT-SURROGATE-002 mirrors this table and tolerance
# (cae_suite/specs/sdfflow.py::SURROGATE_LABEL_CONSTANTS).
_SURROGATE_LABEL_CONSTANTS = {'E': 113.8e9, 'nu': 0.342, 'length_scale': 0.19 / 1.8}
# The ex13 ('ver') labels were solved in DeepJEB's own frame (longest side
# 184.181 mm), with the paper's Ti-6Al-4V (D:/CAE_datasets_raw/deepjeb_ver_fea).
_SURROGATE_LABEL_CONSTANTS_VER = {'E': 113.8e9, 'nu': 0.342, 'length_scale': 0.184181 / 1.8}


def _label_constants(layout):
    return _SURROGATE_LABEL_CONSTANTS_VER if layout == 'ver' else _SURROGATE_LABEL_CONSTANTS
# Relative: the checked-in 0.10556 is 0.19/1.8 written to config precision.
_LABEL_CONSTANT_RTOL = 1e-3


def _warn_surrogate_constants(material, bracket, layout='ex10'):
    used = {'E': material.E, 'nu': material.nu, 'length_scale': bracket.length_scale}
    moved = [f'{k} {used[k]:g} (labels: {v:g})' for k, v in _label_constants(layout).items()
             if abs(used[k] - v) > _LABEL_CONSTANT_RTOL * abs(v)]
    if moved:
        print('  WARNING: ' + ', '.join(moved) + ' -- the surrogate cannot honour these; '
              'its stress and displacement stay at the label constants.', flush=True)


def _device(config):
    """The configured GPU, not whatever CUDA's current device happens to be.

    `'cuda'` alone resolves to the current device, cuda:0, so `gpu_ids 3`
    silently ran the whole loop on GPU 0. `resolve_device` is what every other
    SDFFlow mode uses; `gpu_ids cpu` stays an explicit CPU run.
    """
    if str(config.get('gpu_ids', 0)).strip().lower() == 'cpu':
        return 'cpu'
    return str(resolve_device(config))


def run_optimize(config, config_filename='config.txt'):
    out_dir = config.get('output_dir', '../../output/geometry_generation/optimization')
    os.makedirs(out_dir, exist_ok=True)
    _clear_owned_outputs(out_dir)
    started = time.time()

    # ---- Generator: the shape parameterization -------------------------- #
    subspace_dim = _int(config, 'opt_subspace_dim', 12)
    generator = SDFFlowGenerator(
        config.get('vae_modelpath'),
        config.get('fm_modelpath', '../../output/geometry_generation/sdfflow_fm.pth'),
        device=_device(config),
        subspace_dim=subspace_dim,
        subspace_seed=_int(config, 'opt_subspace_seed', 0),
        base_seed=_int(config, 'seed', 0),
        ode_steps=_int(config, 'ode_steps', 50),
        mc_resolution=_int(config, 'mc_resolution', 128),
        cond_dims=tuple(_as_list(config.get('opt_condition_dims', 'volume,area')) or ()),
        shell_scale=_flt(config, 'opt_shell_scale', 1.25),
        latent_range=_flt(config, 'opt_latent_range', None),
        noise_param=NOISE_PARAM_DEFAULT,
    )
    print(f'Design space: {subspace_dim} latent directions + '
          f'{len(generator.cond_dims)} conditions {generator.cond_dims} '
          f'= {generator.n_design} variables over a {generator.latent_flat_dim}-d '
          f'flow-matching noise space ({generator.noise_param})', flush=True)

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
    # A designer's deflection requirement: max |u_z| under the vertical load,
    # in mm. 0 (the default) keeps the population-calibrated |u| allowable.
    vertical_limit_mm = _flt(config, 'opt_vertical_disp_max', 0.0) or 0.0
    if vertical_limit_mm < 0:
        raise ValueError(f'opt_vertical_disp_max must be >= 0 mm, got {vertical_limit_mm}')
    vertical_disp_allow = vertical_limit_mm * 1e-3 if vertical_limit_mm > 0 else None
    if vertical_disp_allow is not None and 'vertical' not in load_cases:
        raise ValueError("opt_vertical_disp_max constrains the vertical load case; "
                         f"add 'vertical' to opt_load_cases (got {list(load_cases)})")
    bracket = Bracket(material=material,
                      length_scale=_flt(config, 'opt_length_scale', 0.19 / 1.8),
                      load_cases=load_cases,
                      stress_percentile=_flt(config, 'opt_stress_percentile', 99.5))
    print(f'Structural problem: {material.name}, load cases {load_cases}, '
          f'part length {bracket.length_scale * 1.8 * 1e3:.0f} mm', flush=True)

    analysis_backend = str(config.get('opt_analysis', 'fea')).lower()
    if analysis_backend not in ('fea', 'surrogate'):
        raise ValueError(f"opt_analysis must be 'fea' or 'surrogate', got {analysis_backend!r}")
    # Re-solve the surrogate's result with the real solver once the search is
    # done. An FEA run already delivers solver numbers at a refined resolution.
    fea_verify = _flag(config, 'opt_fea_verify')
    if fea_verify and analysis_backend == 'fea':
        print('opt_fea_verify: this run is solved by FEA throughout and its winner is '
              're-solved at the refined resolution anyway; nothing extra to do.', flush=True)
        fea_verify = False

    if analysis_backend == 'surrogate':
        surrogate = HIMGNSurrogate(
            config_path=config['opt_surrogate_config'],
            checkpoint=config['opt_surrogate_checkpoint'],
            python=config.get('opt_surrogate_python') or None,
            load_cases=tuple(_SURROGATE_CASE_NAMES[c] for c in load_cases),
            target_nodes=_int(config, 'opt_surrogate_target_nodes', 5000),
            density=material.rho,
            stress_percentile=_flt(config, 'opt_stress_percentile', 99.5),
            workdir=os.path.join(out_dir, 'surrogate_batches'),
            length_scale=bracket.length_scale,
            gpu_id=gpu_id_from_config(config),
        )
        evaluator = SurrogateEvaluator(generator, surrogate)
        print(f'Analysis backend: HI-MGN surrogate ({surrogate.checkpoint})', flush=True)
        print('  The surrogate predicts the structural response it was trained on: its '
              'labels fix E, nu, the part scale and the load cases, and changing '
              'opt_material_e/nu or opt_length_scale here moves only the mass. Its '
              'accuracy is its held-out score, not this run -- '
              + ('opt_fea_verify re-solves the result with the real solver at the end.'
                 if fea_verify else
                 'confirm the winner with design_loop/verify_with_fea.py (or set '
                 'opt_fea_verify) before acting on it.'), flush=True)
        _warn_surrogate_constants(material, bracket, surrogate.layout)
        if surrogate.layout == 'ver' and fea_verify:
            # fea.py's Bracket clamps the pad bottoms and loads the lug crown
            # (problem.find_interfaces); the ver labels clamp the bolt bores and
            # load the lug bores through RBE3 in OpenRadioss. Re-solving with the
            # former would grade the surrogate against a different problem.
            raise ValueError('opt_fea_verify re-solves with fea.py and its pad/lug-crown '
                             'boundary conditions, not the bore/lug rule this surrogate '
                             'was labelled with; set opt_fea_verify False and check the '
                             'winner with the deepjeb_ver_fea OpenRadioss pipeline')
    else:
        evaluator = Evaluator(generator, bracket,
                              mesh_size_max=_flt(config, 'opt_mesh_size_max', 0.05),
                              target_faces=_int(config, 'opt_target_faces', 12000))
        print(f'Analysis backend: FEA ({bracket.load_cases})', flush=True)

    # ---- Stage 1: baseline population and calibration -------------------- #
    baseline_size = _int(config, 'opt_baseline_size', 12)
    budget = _int(config, 'opt_budget', 120)
    if budget is None or budget < 0:
        raise ValueError(f'opt_budget must be >= 0 (0 = screening only, no search), '
                         f'got {budget}')
    # opt_budget 0 runs no search: the population is the result. Every design
    # is generated and analysed once, and the lightest one that meets the
    # limits is delivered, instead of seeding CMA-ES with it.
    screening_only = budget == 0
    screen_batch = _int(config, 'opt_screen_batch', 64)
    if screen_batch is None or screen_batch < 0:
        raise ValueError(f'opt_screen_batch must be >= 0 (0 = one call), got {screen_batch}')
    stress_margin = _flt(config, 'opt_stress_margin', 1.0)
    if screening_only:
        via = (f'HI-MGN in chunks of {screen_batch or baseline_size}'
               if analysis_backend == 'surrogate' else 'FEA one by one')
        print(f'\n=== Screening ({baseline_size} generated designs, {via}) ===', flush=True)
    else:
        print(f'\n=== Baseline population ({baseline_size} designs) ===', flush=True)
    screen_started = time.time()
    if analysis_backend == 'surrogate':
        # One native call per `opt_screen_batch` designs instead of one per
        # design -- see `surrogate_baseline_population`'s docstring.
        baseline = surrogate_baseline_population(
            generator, surrogate, size=baseline_size, seed=_int(config, 'opt_seed', 0),
            batch_size=screen_batch,
            on_chunk=_screen_progress(vertical_disp_allow, screen_started))
    else:
        baseline = baseline_population(evaluator, size=baseline_size,
                                       seed=_int(config, 'opt_seed', 0))
    screen_wall_time = time.time() - screen_started
    limits = calibrate(baseline, stress_margin=stress_margin,
                       disp_margin=_flt(config, 'opt_disp_margin', 1.0))
    limits['vertical_disp_allow'] = vertical_disp_allow
    vertical_ok = [r['fea']['vertical_displacement'] for r in baseline
                   if r['ok'] and r['fea'].get('vertical_displacement') is not None]
    if vertical_ok:
        limits['vertical_disp_range'] = [float(min(vertical_ok)), float(max(vertical_ok))]
        limits['vertical_disp_median'] = float(np.median(vertical_ok))
    if limits['stress_allow'] is not None:
        stress_line = (f"\n  stress_allow {limits['stress_allow'] / 1e6:.1f} MPa"
                       f"   ({limits['stress_allow'] / material.yield_stress * 100:.1f}% "
                       'of yield)')
    else:
        lo, hi = limits['stress_range']
        stress_line = (f"\n  stress_allow off (opt_stress_margin 0; population peak vM "
                       f"{lo / 1e6:.1f}-{hi / 1e6:.1f} MPa)")
    print(f"\nCalibrated from {limits['population']} analyzed designs:"
          f"\n  mass_ref     {limits['mass_ref']:.4f} kg"
          f"   (range {limits['mass_range'][0]:.4f}-{limits['mass_range'][1]:.4f})"
          + stress_line +
          f"\n  disp_allow   {limits['disp_allow'] * 1e3:.4f} mm"
          + (' (not applied: the vertical limit replaces it)' if vertical_disp_allow else ''),
          flush=True)
    if vertical_disp_allow is not None:
        spread = ''
        if 'vertical_disp_range' in limits:
            lo, hi = limits['vertical_disp_range']
            spread = (f"   (population {lo * 1e3:.4f}-{hi * 1e3:.4f} mm, "
                      f"median {limits['vertical_disp_median'] * 1e3:.4f})")
        print(f"  vertical |u_z| limit {vertical_disp_allow * 1e3:.4f} mm "
              f"(opt_vertical_disp_max){spread}", flush=True)

    evaluator.objective = MassObjective(
        mass_ref=limits['mass_ref'],
        stress_allow=limits['stress_allow'],
        disp_allow=limits['disp_allow'],
        stress_weight=_flt(config, 'opt_stress_weight', 6.0),
        disp_weight=_flt(config, 'opt_disp_weight', 3.0),
        vertical_disp_allow=vertical_disp_allow,
    )
    baseline_scored = []
    for record in baseline:
        if record['ok']:
            score, penalty = evaluator.objective(record['fea'])
            record['score'], record['penalty'] = score, penalty
            baseline_scored.append(record)
    # The same feasibility-first rule the search reports with, so the search
    # starts from -- and is compared against -- the member it would have picked.
    reference = select_best(baseline_scored)
    feasible_screened = [r for r in baseline_scored if r['penalty'].get('feasible')]
    if screening_only:
        # A feasible record scores mass / mass_ref exactly, so the feasible
        # minimum is the lightest design meeting every active limit.
        uz = reference['fea'].get('vertical_displacement')
        print(f"  {len(feasible_screened)} of {len(baseline_scored)} solved designs meet the "
              f"limits; lightest: #{reference['index']} mass {reference['fea']['mass']:.4f} kg"
              + (f" | u_z {uz * 1e3:.4f} mm" if uz is not None else '')
              + ('' if feasible_screened else
                 ' (none feasible -- delivering the best-scoring solved design)'),
              flush=True)
    else:
        print(f"  best baseline score {reference['score']:.4f} "
              f"(mass {reference['fea']['mass']:.4f} kg)", flush=True)
    # Two honest references: the median member is what "a typical DeepJEB
    # bracket" means, the best member is the harder target the search starts from.
    typical = sorted(baseline_scored, key=lambda r: r['fea']['mass'])[len(baseline_scored) // 2]
    print(f"  typical (median-mass) baseline: mass {typical['fea']['mass']:.4f} kg", flush=True)

    # ---- Stage 2: CMA-ES over the design space --------------------------- #
    popsize = _int(config, 'opt_popsize', 8)
    if screening_only:
        print('\n=== No search (opt_budget 0): the screen above is the result ===',
              flush=True)
        best_x, best_score, log, search_history = np.asarray(reference['x']), \
            float(reference['score']), [], []
        evaluator.history = []
    elif analysis_backend == 'surrogate':
        print(f'\n=== CMA-ES search (budget {budget} evaluations, popsize {popsize}) ===',
              flush=True)
        # Batched per-generation, not per-candidate -- see `surrogate_search`.
        best_x, best_score, log, search_history = surrogate_search(
            generator, surrogate, evaluator.objective,
            x0=np.asarray(reference['x']),
            sigma0=_flt(config, 'opt_sigma0', 1.0),
            budget=budget, popsize=popsize, seed=_int(config, 'opt_seed', 0),
        )
        evaluator.history = search_history       # so save_history sees it below
    else:
        print(f'\n=== CMA-ES search (budget {budget} evaluations, popsize {popsize}) ===',
              flush=True)
        evaluator.history = []      # the search log is separate from the baseline
        best_x, best_score, log = run(
            evaluator,
            x0=np.asarray(reference['x']),
            sigma0=_flt(config, 'opt_sigma0', 1.0),
            budget=budget,
            popsize=popsize,
            seed=_int(config, 'opt_seed', 0),
        )
        search_history = evaluator.history
    if analysis_backend == 'surrogate':
        # surrogate_baseline_population/surrogate_search build their own record
        # lists rather than calling through evaluator, so its failure tally
        # never saw them -- recompute it here or every surrogate run reports
        # zero failures no matter how many candidates actually failed.
        for record in baseline + search_history:
            if not record['ok']:
                kind = record['error'].split(':', 1)[0]
                evaluator.failures[kind] = evaluator.failures.get(kind, 0) + 1

    # ---- Stage 3: refined verification of the winner --------------------- #
    verification_label = ('Surrogate re-evaluation of the best design '
                          '(not FEA verified)' if analysis_backend == 'surrogate'
                          else 'Verification of the best design at refined resolution')
    print(f'\n=== {verification_label} ===', flush=True)
    verify_res = _int(config, 'opt_verify_resolution', 160)
    verify_faces = _int(config, 'opt_verify_target_faces', 30000)
    verify_size = _flt(config, 'opt_verify_mesh_size_max', 0.035)
    search_best = select_best(search_history) if search_history else None
    # CMA-ES never scores x0 itself, so the search's best can lose to the
    # baseline member it started from. Deliver whichever of the two wins under
    # the same feasibility-first rule the search used (search listed first, so a
    # tie keeps the searched design): calling a design worse than its own
    # starting point "optimized" would report the comparison backwards.
    delivered = select_best([r for r in (search_best, reference) if r is not None])
    search_improved = delivered is not reference
    if screening_only:
        print(f"  Delivering screened design #{reference['index']} (the lightest meeting the "
              'limits); no baseline column -- it is the screen\'s own pick.', flush=True)
    elif not search_improved:
        best_x = np.asarray(reference['x'])
        print(f"  The search did not beat its starting design (search best "
              f"{best_score:.4f} vs best baseline {reference['score']:.4f}); "
              'delivering that baseline member as the result.', flush=True)
    verify_kwargs = dict(mc_resolution=verify_res, target_faces=verify_faces,
                         mesh_size_max=verify_size)
    verified = evaluator.analyze(best_x, return_fields=True, **verify_kwargs)
    # When the delivered design *is* the baseline, one analysis serves both
    # columns instead of re-solving the identical shape.
    reference_verified = (evaluator.analyze(np.asarray(reference['x']), return_fields=True,
                                            **verify_kwargs)
                          if search_improved else verified)
    typical_verified = evaluator.analyze(np.asarray(typical['x']), **verify_kwargs)
    verify_evaluations = 3 if search_improved else 2

    # A screen has no "best baseline" distinct from its result: exporting it
    # would make verify_with_fea re-solve the delivered shape a second time.
    exported = ((('optimized', verified), ('typical', typical_verified)) if screening_only
                else (('optimized', verified), ('baseline', reference_verified),
                      ('typical', typical_verified)))
    for tag, record in exported:
        if record['ok']:
            mesh = record.get('mesh_object')
            f = record['fea']
            if mesh is not None:
                mesh.export(os.path.join(out_dir, f'{tag}.stl'))
                if f.get('frame'):
                    # The 'ver' surrogate's frame (bores registered, DeepJEB mm):
                    # the part its numbers describe, and the input the
                    # OpenRadioss check (deepjeb_ver_fea, mm STLs) solves.
                    import trimesh
                    trimesh.Trimesh(apply_frame(mesh.vertices, f['frame']), mesh.faces,
                                    process=False).export(
                        os.path.join(out_dir, f'{tag}_mm.stl'))
            mesh_note = (f"{f['num_tets']} tets" if analysis_backend == 'fea'
                         else f"{f['num_nodes']} surface nodes (surrogate)")
            uz = f.get('vertical_displacement')
            uz_note = f" | vertical u_z {uz * 1e3:.4f} mm" if uz is not None else ''
            print(f"  {tag:9s}: mass {f['mass']:.4f} kg | "
                  f"peak vM {f['peak_von_mises'] / 1e6:.1f} MPa | "
                  f"max disp {f['max_displacement'] * 1e3:.4f} mm{uz_note} | "
                  f"{mesh_note}", flush=True)
        else:
            print(f'  {tag:9s}: FAILED {record["error"]}', flush=True)
    for record in (verified, reference_verified, typical_verified):
        record.pop('mesh_object', None)

    # The allowables were calibrated at the search resolution; carry them to
    # the verification resolution before judging the re-analysed design.
    verify_objective, verification_limits = _verification_objective(
        evaluator.objective, [(reference, reference_verified), (typical, typical_verified)])
    stress_note = (f"stress_allow {verify_objective.stress_allow / 1e6:.1f} MPa "
                   f"(x{verification_limits['stress_factor']:.3f} from "
                   f"{verification_limits['reference_designs']} baseline design(s) analysed "
                   'at both resolutions)' if verify_objective.stress_allow is not None
                   else 'stress constraint off')
    print(f'  verification limits: {stress_note}; the vertical limit is absolute and is '
          'not rescaled', flush=True)

    # ---- Reporting -------------------------------------------------------- #
    summary = _summarize(limits, reference,
                         None if screening_only else reference_verified, verified,
                         best_x, best_score, log, search_history, baseline,
                         evaluator, material, load_cases, bracket, time.time() - started,
                         delivered, typical_verified, analysis_backend,
                         verify_evaluations=verify_evaluations,
                         verify_objective=verify_objective,
                         verification_limits=verification_limits)
    # A screen has no search to have improved on anything.
    summary['search_improved_on_baseline'] = None if screening_only else bool(search_improved)
    if screening_only:
        summary['screening'] = _screening_summary(
            baseline, reference, typical, limits,
            screen_batch if analysis_backend == 'surrogate' else None, screen_wall_time)
        _write_screening_table(out_dir, baseline, reference, typical, limits, analysis_backend)
    if analysis_backend == 'surrogate':
        summary['surrogate'] = dict(surrogate.stats(), checkpoint=surrogate.checkpoint,
                                    label_constants=dict(_label_constants(surrogate.layout)),
                                    # How the training labels were produced, which is
                                    # the resolution verify_with_fea.py must solve at
                                    # for the surrogate's numbers to be comparable.
                                    label_resolution=({
                                        'target_faces': LABEL_SURFACE_FACES,
                                        'mesh_size_max': LABEL_MESH_SIZE_MAX,
                                        'target_nodes': 5000,
                                    } if surrogate.layout == 'ex10' else {
                                        # Not reproducible with fea.py: see the
                                        # opt_fea_verify refusal above.
                                        'solver': 'OpenRadioss implicit, tet4 L0.7 mm',
                                        'target_nodes': 5000,
                                    }))
    summary['design_space'] = {
        'noise_param': generator.noise_param,
        'latent_flat_dim': generator.latent_flat_dim,
        'subspace_dim': generator.subspace_dim,
        'condition_dims': list(generator.cond_dims),
        'n_design': generator.n_design,
        'latent_range': generator.latent_range,
        'shell_scale': generator.shell_scale,
        'subspace_seed': _int(config, 'opt_subspace_seed', 0),
        'base_seed': _int(config, 'seed', 0),
        'ode_steps': generator.ode_steps,
    }
    summary['verification_settings'] = _verification_settings(
        analysis_backend, verify_res, verify_faces, verify_size,
        surrogate.target_nodes if analysis_backend == 'surrogate' else None,
    )
    # The resolution every search candidate was analysed at.
    if analysis_backend == 'surrogate':
        summary['search_settings'] = {'mc_resolution': generator.mc_resolution,
                                      'target_nodes': surrogate.target_nodes,
                                      'surface_faces': surrogate.surface_faces}
    else:
        summary['search_settings'] = {'mc_resolution': generator.mc_resolution,
                                      'target_faces': evaluator.target_faces,
                                      'mesh_size_max': evaluator.mesh_size_max}
    _write_summary(out_dir, summary)
    _strip = ('mesh_object', 'traceback', 'fields')
    save_history(os.path.join(out_dir, 'history.json'), evaluator,
                 extra={'baseline': [{k: v for k, v in r.items() if k not in _strip}
                                     for r in baseline],
                        'search': [{k: v for k, v in r.items() if k not in _strip}
                                   for r in search_history],
                        'limits': limits, 'analysis_backend': analysis_backend})
    _plot(out_dir, baseline, search_history, log, limits, analysis_backend,
          delivered_index=reference['index'] if screening_only else None)
    if screening_only:
        print('  (skipping stress render: a screen delivers one design, with no baseline '
              'to compare it against)', flush=True)
    elif not search_improved:
        print('  (skipping stress render: baseline and optimized are the same design)',
              flush=True)
    elif (analysis_backend == 'fea' and verified['ok'] and reference_verified['ok']
            and 'fields' in verified and 'fields' in reference_verified):
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
    elif analysis_backend == 'surrogate':
        print('  (skipping stress render: the surrogate path has no interface node '
              'sets or per-node field mesh to draw -- see the accuracy notice above)',
              flush=True)
    _write_report(out_dir, summary)
    if fea_verify:
        # verify_with_fea reads the summary.json and STLs just written, so it
        # runs after them; the summary and report are then rewritten with its
        # verdict beside the surrogate's numbers.
        summary['fea_verification'] = _fea_verify(out_dir)
        summary['wall_time_s'] = time.time() - started
        _write_summary(out_dir, summary)
        _write_report(out_dir, summary)

    print(f'\nWrote results to {out_dir}', flush=True)
    print(f"Total wall time {summary['wall_time_s'] / 60:.1f} min "
          f"over {summary['total_evaluations']} analyses", flush=True)
    return summary


def _pct(new, old):
    return 100.0 * (new - old) / old


def _write_summary(out_dir, summary):
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2, default=float)


FEA_VERIFIED_NAME = 'fea_verified.json'


def _fea_verify(out_dir):
    """`opt_fea_verify`: re-solve the surrogate run's three STLs with tet4 FEA.

    This runs `design_loop/verify_with_fea.py` in-process, so it uses the same
    solver, resolution (the surrogate labels' own), statistics and limit
    verdicts as the command-line tool. It writes the same `fea_verified.json`.
    The summary keeps a compact copy for the report and the Studio table.

    The surrogate's result stays in place. A missing gmsh or pyamg, or a solver
    crash, is recorded here as an error; it does not fail the run.
    """
    print('\n=== FEA verification of the surrogate result (opt_fea_verify) ===', flush=True)
    path = os.path.join(out_dir, FEA_VERIFIED_NAME)
    started = time.time()
    try:
        from design_loop import verify_with_fea
        verify_with_fea.main(['--run-dir', out_dir, '--json-out', path])
        with open(path, encoding='utf-8') as fh:
            report = json.load(fh)
    except (Exception, SystemExit) as exc:
        error = f'{type(exc).__name__}: {exc}'
        print(f'  FEA verification FAILED: {error}', flush=True)
        return {'error': error, 'wall_time_s': time.time() - started}
    return dict(_compact_fea_verification(report), file=FEA_VERIFIED_NAME,
                wall_time_s=time.time() - started)


def _compact_fea_verification(report):
    """The per-design solver numbers and verdicts from a fea_verified.json."""
    designs = {}
    for name, rec in (report.get('designs') or {}).items():
        if 'error' in rec:
            designs[name] = {'error': rec['error']}
            continue
        brief = {key: rec[key] for key in ('mass_kg', 'peak_von_mises_MPa',
                                           'max_displacement_mm', 'vertical_displacement_mm',
                                           'tets')
                 if rec.get(key) is not None}
        surface = rec.get('label_surface') or {}
        for key in ('peak_von_mises_MPa', 'vertical_displacement_mm'):
            if surface.get(key) is not None:
                brief[f'label_surface_{key}'] = surface[key]
        if rec.get('limits'):
            brief['limits'] = rec['limits']
        designs[name] = brief
    return {'solver': report.get('solver'), 'resolution': report.get('resolution'),
            'designs': designs}


def _verification_objective(objective, pairs):
    """The search's allowables carried to the verification resolution.

    `stress_allow` and `disp_allow` are medians of the baseline population
    *at the search resolution*. A finer tet mesh (or marching-cubes grid)
    reads a different peak on the same shape -- tet4 stiffness and stress
    concentrations both move with it -- so judging the refined re-analysis
    against the raw search-resolution allowable would report resolution
    drift as a design verdict. Each allowable is scaled by the median
    verify/search ratio over the baseline designs analysed at both
    resolutions (the best and the typical member), `mass_ref` likewise so the
    verified score stays on the search's scale. The vertical limit is the
    designer's absolute requirement and is not rescaled.

    `pairs` is [(search_record, verified_record), ...]; failed or repeated
    designs are skipped, and with no usable pair every factor is 1.
    Returns (MassObjective, dict describing the transport).
    """
    ratios = {'peak_von_mises': [], 'max_displacement': [], 'mass': []}
    seen, indices = set(), []
    for search, verify in pairs:
        if not (search and verify and search.get('ok') and verify.get('ok')):
            continue
        key = tuple(np.round(np.asarray(search['x'], dtype=float), 12).tolist())
        if key in seen:
            continue
        seen.add(key)
        indices.append(search.get('index'))
        for name, values in ratios.items():
            before, after = search['fea'][name], verify['fea'][name]
            if before and before > 0 and after is not None:
                values.append(float(after) / float(before))
    factor = {name: float(np.median(values)) if values else 1.0
              for name, values in ratios.items()}
    transported = MassObjective(
        mass_ref=objective.mass_ref * factor['mass'],
        stress_allow=(objective.stress_allow * factor['peak_von_mises']
                      if objective.stress_allow is not None else None),
        disp_allow=objective.disp_allow * factor['max_displacement'],
        stress_weight=objective.stress_weight,
        disp_weight=objective.disp_weight,
        failure_score=objective.failure_score,
        vertical_disp_allow=objective.vertical_disp_allow,
    )
    return transported, {
        'reference_designs': len(indices),
        'baseline_indices': indices,
        'stress_factor': factor['peak_von_mises'],
        'disp_factor': factor['max_displacement'],
        'mass_factor': factor['mass'],
        # None: no stress constraint (opt_stress_margin 0).
        'stress_allow_MPa': (transported.stress_allow / 1e6
                             if transported.stress_allow is not None else None),
        'search_stress_allow_MPa': (objective.stress_allow / 1e6
                                    if objective.stress_allow is not None else None),
        'disp_allow_mm': transported.disp_allow * 1e3,
        'mass_ref_kg': transported.mass_ref,
        'vertical_disp_allow_mm': (objective.vertical_disp_allow * 1e3
                                   if objective.vertical_disp_allow else None),
        'rule': ('allowable x median(verify/search) over baseline designs analysed at '
                 'both resolutions; vertical limit absolute'),
    }


def _summarize(limits, reference, reference_verified, verified, best_x, best_score,
               log, search_history, baseline, evaluator, material, load_cases,
               bracket, wall_time, delivered=None, typical_verified=None,
               analysis_backend='fea', verify_evaluations=3, verify_objective=None,
               verification_limits=None):
    ok = [r for r in search_history if r['ok']]
    feasible = [r for r in ok if r['penalty'].get('feasible')]
    summary = {
        'analysis_backend': analysis_backend,
        'wall_time_s': wall_time,
        'total_evaluations': len(baseline) + len(search_history) + int(verify_evaluations),
        'search_evaluations': len(search_history),
        'search_success_rate': len(ok) / max(len(search_history), 1),
        'feasible_designs': len(feasible),
        'failures': evaluator.failures,
        'limits': limits,
        'material': {'name': material.name, 'E': material.E, 'nu': material.nu,
                     'rho': material.rho, 'yield_stress': material.yield_stress},
        'load_cases': list(load_cases),
        'length_scale': bracket.length_scale,
        # verify_with_fea.py re-solves at this percentile so its peak stress is
        # the same measure the run was scored on.
        'stress_percentile': bracket.stress_percentile,
        # The delivered design -- the search's best, or the baseline member it
        # started from when the search did not beat it.
        'best_x': np.asarray(best_x).tolist(),
        # The search's own best score at the search resolution, kept raw even
        # when the delivered design is the baseline member.
        'best_search_score': float(best_score),
        'convergence': log,
    }
    if verification_limits is not None:
        summary['verification_limits'] = verification_limits
    ok_baseline = [r for r in baseline if r['ok']]
    if ok_baseline:
        summary['baseline_population'] = {
            'median_mass_kg': float(np.median([r['fea']['mass'] for r in ok_baseline])),
            'best_of_population_mass_kg': float(reference['fea']['mass']),
        }
    # Same design, search mesh vs verification mesh: a meaningful
    # discretization sensitivity only on the tetrahedral FEA path. The
    # surrogate always resamples a surface graph to opt_surrogate_target_nodes;
    # calling that a refined-mesh convergence check would invent evidence.
    if (analysis_backend == 'fea' and delivered is not None
            and delivered['ok'] and verified['ok']):
        a, b = delivered['fea'], verified['fea']
        summary['mesh_sensitivity'] = {
            'search_tets': a['num_tets'], 'verify_tets': b['num_tets'],
            'mass_change_pct': _pct(b['mass'], a['mass']),
            'peak_stress_change_pct': _pct(b['peak_von_mises'], a['peak_von_mises']),
            'disp_change_pct': _pct(b['max_displacement'], a['max_displacement']),
        }

    # One brief per design that survived verification; a comparison only where
    # both of its sides did. A failed baseline re-analysis used to drop the
    # optimized design's own numbers from the summary along with it.
    records = {'optimized': verified, 'baseline': reference_verified,
               'typical': typical_verified}
    ver = {tag: _brief(r['fea'], analysis_backend)
           for tag, r in records.items() if r is not None and r['ok']}
    failed = {tag: r['error'] for tag, r in records.items()
              if r is not None and not r['ok']}
    if failed:
        ver['failed'] = failed
    if verified['ok']:
        b = verified['fea']
        # The delivered design scored at the verification resolution, against
        # the allowables carried there (`_verification_objective`): whether it
        # still meets the limits there is the result, not its search-mesh score.
        ver['optimized_score'], ver['optimized_penalty'] = (
            verify_objective or evaluator.objective)(b)
        if delivered is not None and delivered.get('ok') and 'penalty' in delivered:
            ver['optimized_search_penalty'] = delivered['penalty']
        if limits.get('vertical_disp_allow'):
            ver['vertical_limit_met'] = bool(
                b.get('vertical_displacement') is not None
                and b['vertical_displacement'] <= limits['vertical_disp_allow'])
    # No baseline re-analysis in a screen (opt_budget 0): nothing to compare.
    if verified['ok'] and reference_verified is not None and reference_verified['ok']:
        a, b = reference_verified['fea'], verified['fea']
        ver.update(
            mass_change_pct=_pct(b['mass'], a['mass']),
            stress_change_pct=_pct(b['peak_von_mises'], a['peak_von_mises']),
            disp_change_pct=_pct(b['max_displacement'], a['max_displacement']),
        )
        if a.get('vertical_displacement') and b.get('vertical_displacement') is not None:
            ver['vertical_disp_change_pct'] = _pct(b['vertical_displacement'],
                                                   a['vertical_displacement'])
        # Compliance is only ever formed by solving the FEA system. It is absent
        # from surrogate records and summaries rather than represented by a
        # misleading zero or null placeholder.
        if analysis_backend == 'fea':
            ver['stiffness_to_mass_gain_pct'] = 100.0 * (
                (a['max_compliance'] * a['mass']) / (b['max_compliance'] * b['mass']) - 1.0)
    if verified['ok'] and typical_verified is not None and typical_verified['ok']:
        b, t = verified['fea'], typical_verified['fea']
        ver['vs_typical'] = {
            'mass_change_pct': _pct(b['mass'], t['mass']),
            'stress_change_pct': _pct(b['peak_von_mises'], t['peak_von_mises']),
            'disp_change_pct': _pct(b['max_displacement'], t['max_displacement']),
        }
    if ver:
        summary['verified'] = ver
    return summary


def _brief(f, analysis_backend='fea'):
    cases = {
        name: {
            'peak_von_mises_MPa': values['peak_von_mises'] / 1e6,
            'max_displacement_mm': values['max_displacement'] * 1e3,
            **({'max_vertical_displacement_mm': values['max_vertical_displacement'] * 1e3}
               if 'max_vertical_displacement' in values else {}),
        }
        for name, values in f['cases'].items()
    }
    result = {
        'mass_kg': f['mass'],
        'peak_von_mises_MPa': f['peak_von_mises'] / 1e6,
        'max_von_mises_MPa': f['max_von_mises'] / 1e6,
        'max_displacement_mm': f['max_displacement'] * 1e3,
        'cases': cases,
    }
    if f.get('vertical_displacement') is not None:
        result['vertical_displacement_mm'] = f['vertical_displacement'] * 1e3
    if analysis_backend == 'fea':
        result['max_compliance_J'] = f['max_compliance']
        for name, values in f['cases'].items():
            cases[name]['compliance_J'] = values['compliance']
    cardinality_key = 'num_nodes' if analysis_backend == 'surrogate' else 'num_tets'
    result[cardinality_key] = f[cardinality_key]
    return result


def _verification_settings(analysis_backend, resolution, target_faces,
                           mesh_size_max, target_nodes=None):
    settings = {'mc_resolution': resolution}
    if analysis_backend == 'surrogate':
        settings['target_nodes'] = target_nodes
    else:
        settings.update(target_faces=target_faces, mesh_size_max=mesh_size_max)
    return settings


# ---- Screening (opt_budget 0) ---------------------------------------------- #

def _meets_vertical(record, vertical_disp_allow):
    uz = record['fea'].get('vertical_displacement')
    return uz is not None and uz <= vertical_disp_allow


def _screen_progress(vertical_disp_allow, started):
    """`on_chunk` for a surrogate population: one progress line per native call.

    The stress and calibrated-deflection limits need the whole population, so
    the running "lightest" here applies the vertical limit only (or none); the
    delivered design is chosen after calibration.
    """
    def report(records, total):
        solved = [r for r in records if r['ok']]
        pool = ([r for r in solved if _meets_vertical(r, vertical_disp_allow)]
                if vertical_disp_allow is not None else solved)
        line = f'  screened {len(records)}/{total} | solved {len(solved)}'
        if vertical_disp_allow is not None:
            line += f' | u_z <= {vertical_disp_allow * 1e3:.3f} mm: {len(pool)}'
        if pool:
            best = min(pool, key=lambda r: r['fea']['mass'])
            uz = best['fea'].get('vertical_displacement')
            line += (f" | lightest {'meeting it' if vertical_disp_allow is not None else 'solved'}"
                     f" #{best['index']} {best['fea']['mass']:.4f} kg"
                     + (f' (u_z {uz * 1e3:.4f} mm)' if uz is not None else ''))
        elapsed = time.time() - started
        line += f' | {elapsed / 60:.1f} min, {elapsed / max(len(records), 1):.1f} s/design'
        print(line, flush=True)
    return report


def _screen_id(index):
    return f'screen_{int(index):04d}'


def _screening_summary(baseline, delivered, typical, limits, batch_size, wall_time):
    solved = [r for r in baseline if r['ok']]
    summary = {
        'designs': len(baseline),
        'solved': len(solved),
        'feasible': sum(1 for r in solved if (r.get('penalty') or {}).get('feasible')),
        'delivered_index': delivered['index'],
        'delivered_id': _screen_id(delivered['index']),
        'delivered_feasible': bool((delivered.get('penalty') or {}).get('feasible')),
        'typical_index': typical['index'],
        'typical_id': _screen_id(typical['index']),
        'table': SCREENING_TABLE_NAME,
        'wall_time_s': wall_time,
        'seconds_per_design': wall_time / max(len(baseline), 1),
    }
    vertical = limits.get('vertical_disp_allow')
    if vertical:
        summary['meeting_vertical_limit'] = sum(1 for r in solved
                                                if _meets_vertical(r, vertical))
    if batch_size is not None:
        summary['batch_size'] = batch_size
    return summary


def _write_screening_table(out_dir, baseline, delivered, typical, limits, analysis_backend):
    """screening.csv: one row per screened design at the screening resolution.

    A value the analysis did not produce is an empty cell, never 0. `role`
    and `path` mark the delivered and typical designs, whose STLs (and
    verification numbers in summary.json) are the refined re-analysis. Every
    other screened design keeps its numbers only: no STL is written for it.
    """
    import csv

    vertical = limits.get('vertical_disp_allow')
    size_key = 'num_nodes' if analysis_backend == 'surrogate' else 'num_tets'
    roles = {}
    for role, record, stl in (('delivered', delivered, 'optimized.stl'),
                              ('typical', typical, 'typical.stl')):
        entry = roles.setdefault(record['index'], ([], []))
        entry[0].append(role)
        entry[1].append(stl)

    def value(f, key, scale, digits):
        v = f.get(key)
        return '' if v is None else round(float(v) * scale, digits)

    columns = ['id', 'index', 'mass_kg', 'peak_von_mises_mpa', 'max_displacement_mm',
               'vertical_displacement_mm', 'vertical_limit_met', 'feasible', 'score',
               size_key, 'role', 'path', 'error']
    path = os.path.join(out_dir, SCREENING_TABLE_NAME)
    with open(path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for r in baseline:
            f = r.get('fea') if r['ok'] else {}
            uz = f.get('vertical_displacement')
            role, stl = roles.get(r['index'], ([], []))
            error = '' if r['ok'] else (str(r.get('error', '')).splitlines() or [''])[0][:300]
            writer.writerow({
                'id': _screen_id(r['index']), 'index': r['index'],
                'mass_kg': value(f, 'mass', 1.0, 6),
                'peak_von_mises_mpa': value(f, 'peak_von_mises', 1e-6, 4),
                'max_displacement_mm': value(f, 'max_displacement', 1e3, 6),
                'vertical_displacement_mm': value(f, 'vertical_displacement', 1e3, 6),
                'vertical_limit_met': ('' if not vertical or uz is None
                                       else str(uz <= vertical).lower()),
                'feasible': (str(bool(r['penalty']['feasible'])).lower()
                             if r['ok'] and 'penalty' in r else ''),
                'score': round(float(r['score']), 6) if r['ok'] and 'score' in r else '',
                size_key: f.get(size_key, ''),
                # One file per cell, so the path always resolves; a design that is
                # both delivered and typical is written to optimized.stl first.
                'role': ';'.join(role), 'path': stl[0] if stl else '', 'error': error,
            })
    print(f'  wrote {path} ({len(baseline)} designs)', flush=True)
    return path


def _plot(out_dir, baseline, search_history, log, limits, analysis_backend='fea',
          delivered_index=None):
    """convergence.png. `delivered_index` set = a screen (opt_budget 0): panel 0
    is the running lightest feasible mass over the screened designs, and the
    delivered design is starred in the other two."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f'  (skipping convergence plot: {exc})', flush=True)
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    base_ok = [r for r in baseline if r['ok']]
    srch_ok = [r for r in search_history if r['ok']]
    base_label = 'screened designs' if delivered_index is not None else 'baseline population'
    base_style = dict(marker='s', s=45) if len(base_ok) <= 60 else dict(marker='.', s=10,
                                                                        alpha=0.6)
    delivered = next((r for r in base_ok if r['index'] == delivered_index), None)

    if delivered_index is not None:
        order = sorted(base_ok, key=lambda r: r['index'])
        feas = [r for r in order if (r.get('penalty') or {}).get('feasible')]
        axes[0].scatter([r['index'] + 1 for r in order], [r['fea']['mass'] for r in order],
                        s=8, color='0.75', label='screened (solved)')
        axes[0].scatter([r['index'] + 1 for r in feas], [r['fea']['mass'] for r in feas],
                        s=10, color='tab:blue', label='meets the limits')
        xs, ys, running = [], [], None
        for r in feas:
            running = r['fea']['mass'] if running is None else min(running, r['fea']['mass'])
            xs.append(r['index'] + 1)
            ys.append(running)
        if xs:
            xs.append(len(baseline))
            ys.append(ys[-1])
            axes[0].step(xs, ys, where='post', color='crimson', lw=2,
                         label='lightest meeting the limits')
        axes[0].set_xlabel('designs screened')
        axes[0].set_ylabel('mass (kg)')
        axes[0].set_title('Screening: lightest feasible design so far')
    else:
        evals = [e['evaluations'] for e in log]
        axes[0].plot(evals, [e['best_score'] for e in log], 'o-', label='best so far')
        axes[0].plot(evals, [e['generation_median'] for e in log], 's--', alpha=0.6,
                     label='generation median')
        axes[0].set_xlabel('surrogate evaluations' if analysis_backend == 'surrogate'
                           else 'FEA evaluations')
        axes[0].set_ylabel('objective  (mass ratio + penalties)')
        axes[0].set_title('CMA-ES convergence')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].scatter([r['fea']['mass'] for r in base_ok],
                    [r['fea']['peak_von_mises'] / 1e6 for r in base_ok],
                    label=base_label, **base_style)
    if srch_ok:
        axes[1].scatter([r['fea']['mass'] for r in srch_ok],
                        [r['fea']['peak_von_mises'] / 1e6 for r in srch_ok],
                        c=[r['index'] for r in srch_ok], cmap='viridis', s=22,
                        label='search', alpha=0.85)
    if delivered is not None:
        axes[1].scatter([delivered['fea']['mass']], [delivered['fea']['peak_von_mises'] / 1e6],
                        marker='*', s=220, color='crimson', edgecolor='k', zorder=5,
                        label='delivered')
    if limits.get('stress_allow') is not None:
        axes[1].axhline(limits['stress_allow'] / 1e6, color='crimson', ls='--',
                        label='stress allowable')
    axes[1].axvline(limits['mass_ref'], color='gray', ls=':', label='reference mass')
    axes[1].set_xlabel('mass (kg)')
    axes[1].set_ylabel('peak von Mises (MPa)')
    axes[1].set_title('Design space explored')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Plot the deflection the objective actually constrains.
    vertical = bool(limits.get('vertical_disp_allow'))
    key, allow = (('vertical_displacement', limits['vertical_disp_allow']) if vertical
                  else ('max_displacement', limits['disp_allow']))
    base_d = [r for r in base_ok if r['fea'].get(key) is not None]
    srch_d = [r for r in srch_ok if r['fea'].get(key) is not None]
    axes[2].scatter([r['fea']['mass'] for r in base_d],
                    [r['fea'][key] * 1e3 for r in base_d],
                    label=base_label, **base_style)
    if srch_d:
        axes[2].scatter([r['fea']['mass'] for r in srch_d],
                        [r['fea'][key] * 1e3 for r in srch_d],
                        c=[r['index'] for r in srch_d], cmap='viridis', s=22,
                        label='search', alpha=0.85)
    if delivered is not None and delivered['fea'].get(key) is not None:
        axes[2].scatter([delivered['fea']['mass']], [delivered['fea'][key] * 1e3],
                        marker='*', s=220, color='crimson', edgecolor='k', zorder=5,
                        label='delivered')
    axes[2].axhline(allow * 1e3, color='crimson', ls='--',
                    label='vertical limit' if vertical else 'deflection allowable')
    axes[2].set_xlabel('mass (kg)')
    axes[2].set_ylabel('vertical-case max |u_z| (mm)' if vertical else 'max displacement (mm)')
    axes[2].set_title('Mass vs vertical deflection' if vertical else 'Mass vs deflection')
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, 'convergence.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}', flush=True)


def _write_report(out_dir, summary):
    backend = summary.get('analysis_backend', 'fea')
    lines = ['# DeepJEB closed-loop geometry optimization', '']
    screen = summary.get('screening')
    lines.append(f"Analysis backend: **{'FEA (gmsh + linear-static solve)' if backend == 'fea' else 'HI-MGN AI surrogate'}**.")
    completed = (f"{screen['solved']} of {screen['designs']} screened designs completed the "
                 'chain' if screen else
                 f"{summary['search_success_rate'] * 100:.0f}% of search evaluations "
                 'completed the chain')
    lines.append(f"Wall time {summary['wall_time_s'] / 60:.1f} min over "
                 f"{summary['total_evaluations']} generate-{'mesh-solve' if backend == 'fea' else 'analyze'} "
                 f"evaluations ({completed}).")
    if backend == 'surrogate':
        sur = summary.get('surrogate') or {}
        lines += ['', '> **Accuracy notice.** Every number below is a HI-MGN prediction, not a '
                  'solve. The surrogate predicts the response it was trained on -- its labels '
                  'fix E, nu, the part scale and the load cases -- so changing the material '
                  'stiffness or `opt_length_scale` moves only the mass. How far to trust it '
                  "is the checkpoint's held-out score, not this run: "
                  + ('the **FEA verification** section below re-solves the result with the '
                     'real solver.' if summary.get('fea_verification') else
                     'confirm the winner with `design_loop/verify_with_fea.py` (or '
                     '`opt_fea_verify true`) before acting on it.')
                  + (f" Checkpoint: `{sur['checkpoint']}`." if sur.get('checkpoint') else '')]
    lines.append('')
    limits = summary['limits']
    if limits.get('stress_allow') is not None:
        stress_item = f"- stress allowable **{limits['stress_allow'] / 1e6:.1f} MPa**"
    else:
        lo, hi = limits.get('stress_range') or (None, None)
        stress_item = ('- stress constraint **off** (`opt_stress_margin 0`)'
                       + (f"; the population's peak von Mises spans {lo / 1e6:.1f}-"
                          f"{hi / 1e6:.1f} MPa" if lo is not None else ''))
    lines += ['## Calibration', '',
              f"Allowables anchored to the median of a {limits['population']}-design "
              'random population from the same generator:', '',
              f"- reference mass **{limits['mass_ref']:.4f} kg**",
              stress_item,
              f"- deflection allowable **{limits['disp_allow'] * 1e3:.4f} mm**"
              + (' (not applied -- replaced by the vertical limit below)'
                 if limits.get('vertical_disp_allow') else ''), '']
    if limits.get('vertical_disp_allow'):
        lines += [f"Deflection requirement (user-set, `opt_vertical_disp_max`): max |u_z| "
                  f"under the vertical load case **<= {limits['vertical_disp_allow'] * 1e3:.4f} mm**."]
        if 'vertical_disp_range' in limits:
            lo, hi = limits['vertical_disp_range']
            lines.append(f"The random population spans {lo * 1e3:.4f}-{hi * 1e3:.4f} mm "
                         f"(median {limits['vertical_disp_median'] * 1e3:.4f} mm).")
        lines.append('')
    if screen:
        sur = backend == 'surrogate'
        lines += ['## Screening', '',
                  f"No search was run (`opt_budget 0`). {screen['designs']} designs were "
                  'drawn uniformly from the design space, generated by SDFFlow and analysed '
                  + (f"by HI-MGN in chunks of {screen['batch_size']} per native call"
                     if sur and screen.get('batch_size') else
                     'by HI-MGN' if sur else 'by FEA one at a time')
                  + f" ({screen['seconds_per_design']:.1f} s per design, "
                  f"{screen['wall_time_s'] / 60:.1f} min in total).", '',
                  f"- solved: **{screen['solved']}** of {screen['designs']}"]
        if 'meeting_vertical_limit' in screen:
            lines.append(f"- meeting the vertical limit: **{screen['meeting_vertical_limit']}**")
        lines += [f"- meeting every active limit: **{screen['feasible']}**",
                  f"- delivered: `{screen['delivered_id']}`, "
                  + ('the lightest of those' if screen['delivered_feasible'] else
                     '**no screened design met the limits**; this is the best-scoring '
                     'solved design'),
                  f"- typical (median mass): `{screen['typical_id']}`", '',
                  f"Every screened design is a row of `{screen['table']}` (screening "
                  'resolution); the delivered and typical designs are re-generated at the '
                  'verification resolution below.', '']
    if 'baseline_population' in summary and not screen:
        bp = summary['baseline_population']
        lines += [f"Population median mass {bp['median_mass_kg']:.4f} kg; the best-scoring "
                  f"member ({bp['best_of_population_mass_kg']:.4f} kg) is the comparison "
                  'baseline below, which is the harder reference of the two.', '']
    if summary.get('search_improved_on_baseline') is False:
        lines += ['> **The search did not improve on its starting design.** No searched '
                  'design beat the best baseline member under the feasibility-first rule '
                  f"(search best score {summary['best_search_score']:.4f}), so the delivered "
                  '"optimized" design below *is* that baseline member and the two columns '
                  'are the same shape.', '']
    v = summary.get('verified') or {}
    for tag, error in sorted((v.get('failed') or {}).items()):
        lines += [f'- The {tag} design failed verification: `{error}`']
    if v.get('failed'):
        lines.append('')
    if 'optimized' in v and 'baseline' not in v:
        b = v['optimized']
        if screen:
            how = ('FEA at the refined resolution' if backend == 'fea' else
                   'HI-MGN (surrogate, not FEA verified)')
            lines += ['## Delivered design (lightest screened design meeting the limits)', '',
                      f"`{screen['delivered_id']}` re-generated at the verification "
                      f"resolution and re-analysed by {how}:", '']
        else:
            lines += ['## Optimized design (no baseline comparison)', '']
        lines += [f"- mass {b['mass_kg']:.4f} kg",
                  f"- peak von Mises {b['peak_von_mises_MPa']:.1f} MPa",
                  f"- max displacement {b['max_displacement_mm']:.4f} mm"]
        if 'vertical_displacement_mm' in b:
            met = v.get('vertical_limit_met')
            lines.append(f"- vertical-case max |u_z| {b['vertical_displacement_mm']:.4f} mm"
                         + ('' if met is None else (' (limit met)' if met
                                                    else ' (**limit violated**)')))
        lines.append('')
    if 'optimized' in v and 'baseline' in v:
        a, b = v['baseline'], v['optimized']
        result_heading = ('Verified result (refined mesh)' if backend == 'fea'
                          else 'Surrogate re-evaluation (not FEA verified)')
        table = [f'## {result_heading}', '',
                '| quantity | baseline | optimized | change |',
                '| --- | ---: | ---: | ---: |',
                f"| mass (kg) | {a['mass_kg']:.4f} | {b['mass_kg']:.4f} | "
                f"{v['mass_change_pct']:+.1f}% |",
                f"| peak von Mises (MPa) | {a['peak_von_mises_MPa']:.1f} | "
                f"{b['peak_von_mises_MPa']:.1f} | {v['stress_change_pct']:+.1f}% |",
                f"| max displacement (mm) | {a['max_displacement_mm']:.4f} | "
                f"{b['max_displacement_mm']:.4f} | {v['disp_change_pct']:+.1f}% |"]
        if 'vertical_disp_change_pct' in v:
            met = v.get('vertical_limit_met')
            table.append(f"| vertical-case max \\|u_z\\| (mm) | {a['vertical_displacement_mm']:.4f} | "
                         f"{b['vertical_displacement_mm']:.4f} | {v['vertical_disp_change_pct']:+.1f}%"
                         + ('' if met is None else (' (limit met)' if met else ' (**limit violated**)'))
                         + ' |')
        if 'stiffness_to_mass_gain_pct' in v:
            table.append(f"| max compliance (J) | {a['max_compliance_J']:.4f} | "
                        f"{b['max_compliance_J']:.4f} | |")
        cardinality_key = 'num_tets' if backend == 'fea' else 'num_nodes'
        cardinality_label = 'tetrahedra' if backend == 'fea' else 'surface graph nodes'
        if cardinality_key in a and cardinality_key in b:
            table.append(f"| {cardinality_label} | {a[cardinality_key]} | "
                         f"{b[cardinality_key]} | |")
        lines += table + ['']
        if 'stiffness_to_mass_gain_pct' in v:
            lines += [f"Stiffness-per-unit-mass gain: "
                     f"**{v['stiffness_to_mass_gain_pct']:+.1f}%**", '']
        else:
            lines += ['Compliance is not formed on the surrogate path (no solved system to '
                     'take it from); mass, stress, and displacement above are the '
                     'comparison.', '']
    if 'optimized' in v and 'vs_typical' in v and 'typical' in v:
        b, t, vt = v['optimized'], v['typical'], v['vs_typical']
        intro = ('The table above compares against the *best* of the random '
                 'population, which is the hardest reference available. Against '
                 'the median-mass member -- what "a typical DeepJEB bracket" '
                 'means -- the same optimized design is:' if 'baseline' in v else
                 'Against the median-mass member of the same population -- what "a '
                 'typical DeepJEB bracket" means -- analysed at the same resolution, '
                 'the delivered design is:')
        lines += ['### Against a typical population member', '', intro, '',
                  f"- mass {t['mass_kg']:.4f} -> {b['mass_kg']:.4f} kg "
                  f"(**{vt['mass_change_pct']:+.1f}%**)",
                  f"- peak von Mises {t['peak_von_mises_MPa']:.1f} -> "
                  f"{b['peak_von_mises_MPa']:.1f} MPa ({vt['stress_change_pct']:+.1f}%)",
                  f"- max displacement {t['max_displacement_mm']:.4f} -> "
                  f"{b['max_displacement_mm']:.4f} mm ({vt['disp_change_pct']:+.1f}%)"]
        if t.get('vertical_displacement_mm') is not None and \
                b.get('vertical_displacement_mm') is not None:
            lines.append(f"- vertical-case max |u_z| {t['vertical_displacement_mm']:.4f} -> "
                         f"{b['vertical_displacement_mm']:.4f} mm")
        lines.append('')
    vl = summary.get('verification_limits')
    if vl and 'optimized_penalty' in v:
        pen = v['optimized_penalty']
        search_pen = v.get('optimized_search_penalty')
        if vl.get('stress_allow_MPa') is not None:
            stress_text = (f"The stress allowable was calibrated at the search resolution "
                           f"({vl['search_stress_allow_MPa']:.1f} MPa). The same "
                           f"{vl['reference_designs']} baseline design(s) re-analysed at the "
                           f"verification resolution read x{vl['stress_factor']:.3f} that "
                           f"peak, so the delivered design is judged there against "
                           f"**{vl['stress_allow_MPa']:.1f} MPa**")
        else:
            stress_text = 'No stress constraint is applied (`opt_stress_margin 0`)'
        lines += ['### Limits at the verification resolution', '',
                  stress_text
                  + ('; the vertical limit is the absolute requirement and is not rescaled'
                     if vl.get('vertical_disp_allow_mm') else '') + '.', '',
                  f"- at the verification resolution: "
                  f"**{'feasible' if pen.get('feasible') else 'infeasible'}** "
                  f"(stress violation {pen['stress_violation']:.3f}, deflection violation "
                  f"{pen['disp_violation']:.3f})"]
        if search_pen is not None:
            detail = (f" (stress violation {search_pen['stress_violation']:.3f}, "
                      f"deflection violation {search_pen['disp_violation']:.3f})"
                      if 'stress_violation' in search_pen and 'disp_violation' in search_pen
                      else '')
            lines.append(f"- at the search resolution: "
                         f"{'feasible' if search_pen.get('feasible') else 'infeasible'}{detail}")
        lines.append('')
    if summary.get('fea_verification'):
        lines += _fea_verification_lines(summary)
    if backend == 'fea':
        lines += ['## Solver', '',
                  'Linear-static 4-node tetrahedra, AMG-preconditioned CG, nodal von Mises '
                  'volume-averaged from the constant per-element stress. The element passes '
                  'the constant-strain patch test to 1e-9 but shear-locks: on a slender '
                  'cantilever it recovers 0.40/0.65/0.83/0.90 of the Timoshenko tip '
                  'deflection at 288/1.3k/6k/16.5k tets. Absolute deflection and stress here '
                  'are therefore optimistic; the baseline-vs-optimized comparison at equal '
                  'discretization is the meaningful quantity.', '']
    else:
        lines += ['## Solver', '',
                  'HI-MGN (hierarchical multiscale graph network), one forward pass per '
                  'candidate in place of a mesh-and-solve. See the accuracy notice above '
                  'for what this checkpoint can and cannot be trusted for.', '']

    if backend == 'fea' and 'mesh_sensitivity' in summary:
        m = summary['mesh_sensitivity']
        lines += ['## Discretization sensitivity', '',
                  f"The same winning design re-analyzed at the verification resolution "
                  f"({m['search_tets']} -> {m['verify_tets']} tets) moves by "
                  f"{m['mass_change_pct']:+.1f}% in mass, "
                  f"{m['peak_stress_change_pct']:+.1f}% in peak von Mises and "
                  f"{m['disp_change_pct']:+.1f}% in deflection. The search resolution is "
                  'therefore a ranking device, not a converged absolute; baseline and '
                  'optimized are compared above at the same refined resolution so the '
                  'comparison is unaffected.', '']

    if summary['failures']:
        lines += ['## Failure modes', '']
        lines += [f'- `{k}`: {n}' for k, n in sorted(summary['failures'].items())]
        lines.append('')
    path = os.path.join(out_dir, 'report.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'  wrote {path}', flush=True)


_LIMIT_SHORT = {'vertical max |u_z|': 'u_z', 'max |u| (calibrated)': '|u| (calibrated)',
                'peak von Mises (calibrated)': 'stress (calibrated)'}


def _fea_verification_lines(summary):
    """report.md's section for `opt_fea_verify`: surrogate vs solver, same shapes."""
    fv = summary['fea_verification']
    lines = ['## FEA verification (`opt_fea_verify`)', '']
    if fv.get('error'):
        return lines + [f"The FEA re-solve did not complete: `{fv['error']}`. Every number "
                        "above is still the surrogate's.", '']
    res = fv.get('resolution') or {}
    pct = summary.get('stress_percentile', 99.5)
    at = (f" ({res['target_faces']} surface faces, mesh_size_max {res['mesh_size_max']})"
          if 'target_faces' in res and 'mesh_size_max' in res else '')
    count = len(fv.get('designs') or {})
    count_word = {1: 'The STL', 2: 'The two STLs', 3: 'The three STLs'}.get(
        count, f'The {count} STLs')
    lines += [f'{count_word} above re-solved with the real solver (gmsh tetrahedra, '
              "linear-static tet4) at the resolution the surrogate's training labels were "
              f'solved at{at}. Surrogate and solver are compared on the labels\' own measure '
              f'(p{pct:g} of von Mises and the max |u_z| over the decimated boundary nodes); '
              "each limit verdict is the one verify_with_fea.py gives. Full detail: "
              f"`{fv.get('file', FEA_VERIFIED_NAME)}`.", '',
              '| design | FEA mass (kg) | peak von Mises (MPa), surrogate / FEA | '
              'vertical max \\|u_z\\| (mm), surrogate / FEA | limits under FEA |',
              '| --- | ---: | ---: | ---: | --- |']
    v = summary.get('verified') or {}
    designs = fv.get('designs') or {}

    def pair(surrogate, solver, fmt):
        left = format(surrogate, fmt) if surrogate is not None else '--'
        right = format(solver, fmt) if solver is not None else '--'
        return f'{left} / {right}'

    for tag in ('optimized', 'baseline', 'typical'):
        rec = designs.get(tag)
        if not rec:
            continue
        if 'error' in rec:
            # A retry ladder joins its attempts with ' | ', which split this row
            # into a dozen columns; `\|` stays inside the cell.
            error = str(rec['error']).replace('|', '\\|')
            lines.append(f"| {tag} | re-solve failed: `{error}` | | | |")
            continue
        sur = v.get(tag) or {}
        verdicts = '; '.join(
            f"{_LIMIT_SHORT.get(c['limit'], c['limit'])} "
            f"{'met' if c['met'] else '**violated**'}".replace('|', '\\|')
            for c in rec.get('limits') or []) or '--'
        mass = rec.get('mass_kg')
        lines.append(
            f"| {tag} | {'--' if mass is None else format(mass, '.4f')} | "
            f"{pair(sur.get('peak_von_mises_MPa'), rec.get('label_surface_peak_von_mises_MPa'), '.1f')} | "
            f"{pair(sur.get('vertical_displacement_mm'), rec.get('label_surface_vertical_displacement_mm'), '.4f')} | "
            f"{verdicts} |")
    lines.append('')
    best = designs.get('optimized') or {}
    vertical = next((c for c in best.get('limits') or [] if c['limit'] == 'vertical max |u_z|'),
                    None)
    if 'error' in best:
        # Silence here read as a pass: the delivered design is the one shape
        # this section exists to check.
        lines.append("The delivered design could not be re-solved, so its surrogate feasibility "
                     "verdict is **unverified**; do not use it as is.")
    if vertical is not None:
        if vertical['met']:
            lines.append(f"Under the real solver the delivered design **meets** the vertical "
                         f"limit: max |u_z| {vertical['value']:.4f} mm <= "
                         f"{vertical['allow']:.4f} mm.")
        else:
            lines.append(f"Under the real solver the delivered design **violates** the "
                         f"vertical limit: max |u_z| {vertical['value']:.4f} mm > "
                         f"{vertical['allow']:.4f} mm. The surrogate's feasibility verdict "
                         'does not hold for this shape; do not use it as is.')
    base = designs.get('baseline') or {}
    if best.get('mass_kg') and base.get('mass_kg') and summary.get('search_improved_on_baseline') is not False:
        lines.append(f"Solver mass, delivered vs best baseline: {base['mass_kg']:.4f} -> "
                     f"{best['mass_kg']:.4f} kg ({_pct(best['mass_kg'], base['mass_kg']):+.1f}%).")
    if vertical is not None or best.get('mass_kg') or 'error' in best:
        lines.append('')
    return lines
