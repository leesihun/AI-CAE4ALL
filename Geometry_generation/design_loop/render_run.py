"""Regenerate a finished run's stress comparison without redoing the search.

`optimize` already writes `stress_comparison.png`, but a run made before that
existed -- or one whose figure needs different framing -- can be re-rendered
from `summary.json` alone: it stores the winning design vector and the
generator settings, so the two verification analyses are reproducible in a
couple of minutes instead of hours.

    python -m design_loop.render_run ../output/geometry_generation/ex1/optimization
"""

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from design_loop import fea                                          # noqa: E402
from design_loop.generator import SDFFlowGenerator                   # noqa: E402
from design_loop.loop import Evaluator                               # noqa: E402
from design_loop.problem import Bracket                              # noqa: E402
from design_loop.visualize import render_comparison                  # noqa: E402


def baseline_vector(history):
    """The best-scoring baseline design -- the comparison reference `optimize` uses."""
    scored = [r for r in history.get('baseline', [])
              if r.get('ok') and r.get('score') is not None]
    if not scored:
        raise SystemExit('history.json has no scored baseline designs')
    return np.asarray(min(scored, key=lambda r: r['score'])['x'], dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', help='directory holding summary.json and history.json')
    parser.add_argument('--vae', default='../output/geometry_generation/ex1/sdfflow_vae.pth')
    parser.add_argument('--fm', default='../output/geometry_generation/ex1/sdfflow_fm.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mc-resolution', type=int, default=160)
    parser.add_argument('--target-faces', type=int, default=30000)
    parser.add_argument('--mesh-size-max', type=float, default=0.035)
    parser.add_argument('--from-summary', action='store_true',
                        help="use the run's own verification settings instead of the flags")
    parser.add_argument('--output', default=None)
    args = parser.parse_args(argv)

    with open(os.path.join(args.run_dir, 'summary.json'), encoding='utf-8') as fh:
        summary = json.load(fh)
    with open(os.path.join(args.run_dir, 'history.json'), encoding='utf-8') as fh:
        history = json.load(fh)

    if args.from_summary and 'verification_settings' in summary:
        v = summary['verification_settings']
        args.mc_resolution = int(v['mc_resolution'])
        args.target_faces = int(v['target_faces'])
        args.mesh_size_max = float(v['mesh_size_max'])

    best_x = np.asarray(summary['best_x'], dtype=float)
    base_x = baseline_vector(history)
    # `design_space` pins the exact parameterization; older runs predate it, so
    # fall back to the shipped defaults and reconstruct the split from len(best_x).
    space = summary.get('design_space')
    if space is None:
        cond_dims = ('volume', 'area')
        space = {'subspace_dim': len(best_x) - len(cond_dims),
                 'condition_dims': list(cond_dims), 'subspace_seed': 0,
                 'base_seed': 0, 'ode_steps': 50}
        print('summary.json predates design_space; assuming '
              f"subspace_dim={space['subspace_dim']}, conditions={cond_dims}")
    generator = SDFFlowGenerator(args.vae, args.fm, device=args.device,
                                 subspace_dim=space['subspace_dim'],
                                 subspace_seed=space['subspace_seed'],
                                 base_seed=space['base_seed'],
                                 ode_steps=space['ode_steps'],
                                 cond_dims=tuple(space['condition_dims']),
                                 mc_resolution=args.mc_resolution)
    if generator.n_design != len(best_x):
        raise SystemExit(f'reconstructed design space has {generator.n_design} variables '
                         f'but summary.json stores a {len(best_x)}-vector')
    material_spec = summary['material']
    bracket = Bracket(
        material=fea.Material(E=material_spec['E'], nu=material_spec['nu'],
                              rho=material_spec['rho'],
                              yield_stress=material_spec['yield_stress']),
        length_scale=summary['length_scale'],
        load_cases=tuple(summary['load_cases']))
    evaluator = Evaluator(generator, bracket)

    results = {}
    for tag, x in (('baseline', base_x), ('optimized', best_x)):
        record = evaluator.analyze(x, mc_resolution=args.mc_resolution,
                                   target_faces=args.target_faces,
                                   mesh_size_max=args.mesh_size_max,
                                   return_fields=True)
        if not record['ok']:
            raise SystemExit(f'{tag} design failed to analyze: {record["error"]}')
        f = record['fea']
        print(f"  {tag:9s}: mass {f['mass']:.4f} kg | "
              f"peak vM {f['peak_von_mises'] / 1e6:.1f} MPa | "
              f"max disp {f['max_displacement'] * 1e3:.4f} mm | "
              f"{f['num_tets']} tets", flush=True)
        results[tag] = record

    path = args.output or os.path.join(args.run_dir, 'stress_comparison.png')
    render_comparison(path,
                      results['baseline']['fields'], results['baseline']['fea'],
                      results['optimized']['fields'], results['optimized']['fea'],
                      results['optimized']['fea']['worst_case'])
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
