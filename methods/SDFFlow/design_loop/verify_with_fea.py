"""Re-analyze finished design STLs with the real tet4 solver.

`mode optimize --opt_analysis surrogate` scores every candidate with the HI-MGN
forward pass, so its whole reported table -- including the winner -- is the
surrogate's opinion. That is fine for *ranking* candidates but it cannot tell
you whether the winner is feasible, and on the ex4 checkpoint the surrogate is
measured at field R^2 ~ 0 with peak stress about 8x low. A design search whose
constraint is evaluated by a model that systematically under-predicts stress
will happily thin the part past the real limit.

This script closes that loop: it meshes the exported STLs with gmsh and runs the
same `Bracket.analyze` the FEA backend uses, with the same material, load cases,
length scale and stress percentile, so surrogate and solver numbers are directly
comparable on identical geometry.

Still relative-only in absolute terms -- tet4 is stiff and the measure is the
99.5th percentile rather than a true max -- but it is the same solver on every
design here, so the comparison between them is sound.

  python methods/SDFFlow/design_loop/verify_with_fea.py \
      --run-dir output/geometry_generation/ex4/optimization_surrogate \
      --json-out output/geometry_generation/ex4/optimization_surrogate/fea_verified.json
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from design_loop import fea                                    # noqa: E402
from design_loop.mesher import tet_mesh_from_surface           # noqa: E402
from design_loop.problem import Bracket                        # noqa: E402

DESIGNS = ('optimized', 'baseline', 'typical')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--summary', default=None,
                    help='defaults to <run-dir>/summary.json')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--target-faces', type=int, default=25000,
                    help='surface budget handed to gmsh; 25k gives ~61k tets')
    ap.add_argument('--mesh-size-max', type=float, default=0.05)
    args = ap.parse_args(argv)

    summary_path = args.summary or os.path.join(args.run_dir, 'summary.json')
    with open(summary_path, encoding='utf-8') as fh:
        summary = json.load(fh)

    mat_cfg = summary.get('material') or {}
    material = fea.Material(
        **{k: mat_cfg[k] for k in ('name', 'E', 'nu', 'rho') if k in mat_cfg}
    ) if mat_cfg else fea.Material()
    load_cases = tuple(summary.get('load_cases') or ('vertical', 'diagonal'))
    length_scale = float(summary.get('length_scale') or (0.19 / 1.8))
    pct = float((summary.get('verification_settings') or {}).get('stress_percentile', 99.5))

    bracket = Bracket(material=material, length_scale=length_scale,
                      load_cases=load_cases, stress_percentile=pct)
    print(f'solver: tet4, {material.name}, cases {load_cases}, '
          f'part length {length_scale * 1.8 * 1e3:.1f} mm, p{pct} von Mises\n')

    verified = summary.get('verified') or {}
    results = {}
    for name in DESIGNS:
        stl = os.path.join(args.run_dir, f'{name}.stl')
        if not os.path.exists(stl):
            print(f'{name:10s} SKIP (no {name}.stl)')
            continue
        # STL stores every triangle with its own copy of each vertex, so loading
        # with process=False leaves a 3N-vertex triangle soup that is non-manifold
        # by construction -- gmsh's classify step then dies with "single/multiply
        # ended GEdge", and any tets it does build come out as slivers. Let trimesh
        # merge coincident vertices so the surface is a real manifold first.
        # trimesh's default processing merges coincident vertices and drops
        # degenerate/duplicate faces; the explicit removers were dropped in
        # trimesh 4.x. merge_vertices stays as a belt-and-braces no-op.
        mesh = trimesh.load(stl)
        mesh.merge_vertices()
        print(f'{name:10s} surface {len(mesh.vertices):,} verts / {len(mesh.faces):,} faces'
              f'  watertight={mesh.is_watertight}')
        try:
            nodes, tets, info = tet_mesh_from_surface(
                mesh, mesh_size_max=args.mesh_size_max, target_faces=args.target_faces)
        except Exception as exc:
            print(f'{name:10s} MESH FAILED: {type(exc).__name__}: {exc}')
            results[name] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        out = bracket.analyze(nodes, tets)
        rec = {
            'mass_kg': float(out['mass']),
            'peak_von_mises_MPa': float(out['peak_von_mises']) / 1e6,
            'max_displacement_mm': float(out['max_displacement']) * 1e3,
            'tets': int(len(tets)), 'nodes': int(len(nodes)),
        }
        sur = verified.get(name) or {}
        rec['surrogate'] = {
            'mass_kg': sur.get('mass_kg'),
            'peak_von_mises_MPa': sur.get('peak_von_mises_MPa'),
            'max_displacement_mm': sur.get('max_displacement_mm'),
        }
        results[name] = rec
        print(f'{name:10s} mass {rec["mass_kg"]:.4f} kg   '
              f'peak {rec["peak_von_mises_MPa"]:8.1f} MPa   '
              f'disp {rec["max_displacement_mm"]:.4f} mm   ({rec["tets"]:,} tets)')

    ok = {k: v for k, v in results.items() if 'error' not in v}

    if 'optimized' in ok and 'baseline' in ok:
        o, b = ok['optimized'], ok['baseline']
        print('\n--- solver verdict: optimized vs baseline (best of population) ---')
        for key, unit in (('mass_kg', 'kg'), ('peak_von_mises_MPa', 'MPa'),
                          ('max_displacement_mm', 'mm')):
            d = 100.0 * (o[key] - b[key]) / b[key] if b[key] else float('nan')
            print(f'  {key:22s} {b[key]:10.4f} -> {o[key]:10.4f} {unit:3s}  {d:+7.1f}%')

    print('\n--- surrogate vs solver on the SAME geometry (is the surrogate trustworthy here?) ---')
    for name, rec in ok.items():
        s = rec.get('surrogate') or {}
        for key, unit in (('peak_von_mises_MPa', 'MPa'), ('max_displacement_mm', 'mm')):
            sv, fv = s.get(key), rec[key]
            if sv is None or not fv:
                continue
            print(f'  {name:10s} {key:22s} surrogate {sv:9.2f} vs solver {fv:9.2f} {unit:3s}'
                  f'   ratio {sv / fv:5.2f}x')

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or '.', exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump({'solver': 'tet4', 'load_cases': list(load_cases),
                       'length_scale': length_scale, 'stress_percentile': pct,
                       'designs': results}, fh, indent=2, default=float)
        print(f'\nwrote {args.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
