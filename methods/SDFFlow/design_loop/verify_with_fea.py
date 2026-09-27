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

"Comparable" needs three things beyond the solver itself, all read from the
run's summary.json rather than guessed:

- the **resolution**: a surrogate run is re-solved at the resolution its
  training labels were produced at (`surrogate.label_resolution`), an FEA run
  at its own verification resolution (`verification_settings`), each through
  the labels' face-budget retry ladder (`mesher.tet_mesh_with_retries`);
- the **statistic**: the solver's own peak is a percentile over every tet node,
  but the surrogate's labels are a percentile over the boundary nodes decimated
  to 5000 (`build_deepjeb_fea.label_surface`). Both are reported, and a
  surrogate run's calibrated allowable is judged on the label measure it was
  calibrated in;
- the **allowable**: the one carried to the verification resolution
  (`verification_limits`), since the STLs are the verification-resolution
  shapes. The vertical limit is absolute and is judged on the solver's max |u_z|.

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
from design_loop.build_deepjeb_fea import CASE_KEYS, label_surface  # noqa: E402
from design_loop.deepjeb_bridge import (                       # noqa: E402
    LABEL_MESH_SIZE_MAX, LABEL_SURFACE_FACES,
)
from design_loop.mesher import tet_mesh_with_retries           # noqa: E402
from design_loop.problem import Bracket                        # noqa: E402

DESIGNS = ('optimized', 'baseline', 'typical')
LABEL_TARGET_NODES = 5000


def solve_resolution(summary):
    """Where to re-solve a run's STLs, and why (see the module docstring)."""
    corrected = (summary.get('fea_correction') or {}).get('resolution') or {}
    if 'target_faces' in corrected and 'mesh_size_max' in corrected:
        # An `opt_fea_rounds` run solved its candidates -- and calibrated its
        # allowables -- at this resolution, so a re-solve must match it.
        return {'source': 'fea_correction.resolution',
                'target_faces': int(corrected['target_faces']),
                'mesh_size_max': float(corrected['mesh_size_max']),
                'target_nodes': int(corrected.get('target_nodes', LABEL_TARGET_NODES))}
    if summary.get('analysis_backend', 'fea') == 'surrogate':
        label = (summary.get('surrogate') or {}).get('label_resolution') or {}
        return {'source': ('surrogate.label_resolution' if label
                           else 'label-pipeline defaults (summary predates label_resolution)'),
                'target_faces': int(label.get('target_faces', LABEL_SURFACE_FACES)),
                'mesh_size_max': float(label.get('mesh_size_max', LABEL_MESH_SIZE_MAX)),
                'target_nodes': int(label.get('target_nodes', LABEL_TARGET_NODES))}
    verify = summary.get('verification_settings') or {}
    have = 'target_faces' in verify and 'mesh_size_max' in verify
    return {'source': ('verification_settings' if have
                       else 'optimize verification defaults (not in summary)'),
            'target_faces': int(verify.get('target_faces', 30000)),
            'mesh_size_max': float(verify.get('mesh_size_max', 0.035)),
            'target_nodes': LABEL_TARGET_NODES}


def surface_statistics(nodes, result, target_nodes, percentile):
    """The solved fields reduced exactly as the surrogate's labels were.

    `result` is `Bracket.analyze(..., return_fields=True)`. Per load case:
    the `percentile` of |von Mises| and the max |u| and |u_z| over the
    boundary nodes decimated to `target_nodes` -- the numbers `HIMGNSurrogate`
    computes from its prediction, so the two are one measure.
    """
    _, small = label_surface(nodes, result, target_nodes)
    cases = {}
    for short, rows in small['fields'].items():
        cases[CASE_KEYS[short]] = {
            'peak_von_mises_MPa': float(np.percentile(np.abs(rows[0]), percentile)),
            'max_displacement_mm': float(np.abs(rows[1]).max()),
            'max_vertical_displacement_mm': float(np.abs(rows[2]).max()),
        }
    out = {'nodes': int(len(small['vertices'])), 'cases': cases,
           'peak_von_mises_MPa': max(c['peak_von_mises_MPa'] for c in cases.values()),
           'max_displacement_mm': max(c['max_displacement_mm'] for c in cases.values())}
    if 'vertical' in cases:
        out['vertical_displacement_mm'] = cases['vertical']['max_vertical_displacement_mm']
    return out


def load_stl(path):
    """An exported design STL as a manifold surface gmsh can classify.

    STL stores every triangle with its own copy of each vertex, so loading with
    process=False leaves a 3N-vertex triangle soup that is non-manifold by
    construction -- gmsh's classify step then dies with "single/multiply ended
    GEdge", and any tets it does build come out as slivers. trimesh's default
    processing merges coincident vertices and drops degenerate/duplicate faces
    (the explicit removers were dropped in trimesh 4.x); merge_vertices stays
    as a belt-and-braces no-op.
    """
    mesh = trimesh.load(path)
    mesh.merge_vertices()
    return mesh


def solve_surface(mesh, bracket, resolution):
    """Mesh one design surface and solve it: the record this tool reports.

    Meshes through the labels' face-budget retry ladder at `resolution`
    (see `solve_resolution`), solves every load case, and adds the
    label-surface statistics. A meshing or solver failure raises; the caller
    records it. `opt_fea_rounds` solves its candidates through this same
    function, so an in-run solve and a later re-solve of the same STL agree.
    """
    nodes, tets, info = tet_mesh_with_retries(
        mesh, mesh_size_max=resolution['mesh_size_max'],
        target_faces=resolution['target_faces'])
    out = bracket.analyze(nodes, tets, return_fields=True)
    rec = {
        'mass_kg': float(out['mass']),
        'peak_von_mises_MPa': float(out['peak_von_mises']) / 1e6,
        'max_von_mises_MPa': float(out['max_von_mises']) / 1e6,
        'max_displacement_mm': float(out['max_displacement']) * 1e3,
        'tets': int(len(tets)), 'nodes': int(len(nodes)),
        'face_budget': info.get('face_budget'),
        'decimation_aggression': info.get('decimation_aggression'),
    }
    if out.get('vertical_displacement') is not None:
        rec['vertical_displacement_mm'] = float(out['vertical_displacement']) * 1e3
    try:
        rec['label_surface'] = surface_statistics(nodes, out, resolution['target_nodes'],
                                                  bracket.stress_percentile)
    except Exception as exc:
        rec['label_surface_error'] = f'{type(exc).__name__}: {exc}'
    return rec


def limit_verdicts(limits, results, analysis_backend='fea', verification_limits=None):
    """Each design's solver result against the limits its run searched under.

    `vertical_disp_allow` (`opt_vertical_disp_max`) is an absolute requirement,
    so the solver's max |u_z| answers it directly. The stress allowable -- and
    the |u| allowable when no vertical limit is set -- are *calibrated* from the
    run's own baseline population. They are taken at the verification
    resolution when the run recorded one (`verification_limits`), since the
    STLs are verification-resolution shapes. On a surrogate run the population
    was scored by the surrogate, so its allowable is on the surrogate's scale:
    the check reads the label-surface statistic (the measure the allowable was
    calibrated in) and stays flagged rather than presented as a plain pass/fail.
    """
    surrogate = analysis_backend == 'surrogate'
    calibrated_note = ('allowable calibrated on surrogate predictions; judged on the '
                       'label-surface statistic' if surrogate else '')
    vl = verification_limits or {}
    verdicts = {}
    for name, rec in results.items():
        measured = rec.get('label_surface') if surrogate and rec.get('label_surface') else rec
        statistic = 'label_surface' if measured is not rec else 'solver'
        checks = []
        vertical = limits.get('vertical_disp_allow')
        if vertical and rec.get('vertical_displacement_mm') is not None:
            allow = vertical * 1e3
            checks.append({'limit': 'vertical max |u_z|', 'unit': 'mm',
                           'value': rec['vertical_displacement_mm'], 'allow': allow,
                           'met': bool(rec['vertical_displacement_mm'] <= allow),
                           'statistic': 'solver'})
        elif not vertical and limits.get('disp_allow'):
            allow = vl.get('disp_allow_mm') or limits['disp_allow'] * 1e3
            checks.append({'limit': 'max |u| (calibrated)', 'unit': 'mm',
                           'value': measured['max_displacement_mm'], 'allow': allow,
                           'met': bool(measured['max_displacement_mm'] <= allow),
                           'statistic': statistic, 'note': calibrated_note})
        if limits.get('stress_allow'):
            allow = vl.get('stress_allow_MPa') or limits['stress_allow'] / 1e6
            checks.append({'limit': 'peak von Mises (calibrated)', 'unit': 'MPa',
                           'value': measured['peak_von_mises_MPa'], 'allow': allow,
                           'met': bool(measured['peak_von_mises_MPa'] <= allow),
                           'statistic': statistic, 'note': calibrated_note})
        if checks:
            verdicts[name] = checks
    return verdicts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--summary', default=None,
                    help='defaults to <run-dir>/summary.json')
    ap.add_argument('--json-out', default=None)
    ap.add_argument('--target-faces', type=int, default=None,
                    help='surface budget handed to gmsh (default: from the summary, '
                         'see solve_resolution)')
    ap.add_argument('--mesh-size-max', type=float, default=None,
                    help='interior tet size (default: from the summary)')
    ap.add_argument('--target-nodes', type=int, default=None,
                    help='label-surface decimation target (default: from the summary)')
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
    pct = float(summary.get('stress_percentile')
                or (summary.get('verification_settings') or {}).get('stress_percentile', 99.5))

    bracket = Bracket(material=material, length_scale=length_scale,
                      load_cases=load_cases, stress_percentile=pct)
    backend = summary.get('analysis_backend', 'fea')
    resolution = solve_resolution(summary)
    for key in ('target_faces', 'mesh_size_max', 'target_nodes'):
        if getattr(args, key) is not None:
            resolution[key] = getattr(args, key)
            resolution['source'] = 'command line (overrides the summary)'
    verification_limits = summary.get('verification_limits')
    print(f'solver: tet4, {material.name}, cases {load_cases}, '
          f'part length {length_scale * 1.8 * 1e3:.1f} mm, p{pct} von Mises')
    print(f"resolution: {resolution['target_faces']} surface faces, mesh_size_max "
          f"{resolution['mesh_size_max']}, label surface {resolution['target_nodes']} nodes "
          f"({resolution['source']})\n")

    verified = summary.get('verified') or {}
    results = {}
    for name in DESIGNS:
        stl = os.path.join(args.run_dir, f'{name}.stl')
        if not os.path.exists(stl):
            print(f'{name:10s} SKIP (no {name}.stl)')
            continue
        mesh = load_stl(stl)
        print(f'{name:10s} surface {len(mesh.vertices):,} verts / {len(mesh.faces):,} faces'
              f'  watertight={mesh.is_watertight}')
        try:
            rec = solve_surface(mesh, bracket, resolution)
        except Exception as exc:
            print(f'{name:10s} FAILED: {type(exc).__name__}: {exc}')
            results[name] = {'error': f'{type(exc).__name__}: {exc}'}
            continue
        sur = verified.get(name) or {}
        rec['surrogate'] = {
            'mass_kg': sur.get('mass_kg'),
            'peak_von_mises_MPa': sur.get('peak_von_mises_MPa'),
            'max_displacement_mm': sur.get('max_displacement_mm'),
            'vertical_displacement_mm': sur.get('vertical_displacement_mm'),
        }
        results[name] = rec
        uz = rec.get('vertical_displacement_mm')
        ls = rec.get('label_surface') or {}
        print(f'{name:10s} mass {rec["mass_kg"]:.4f} kg   '
              f'peak {rec["peak_von_mises_MPa"]:8.1f} MPa   '
              f'disp {rec["max_displacement_mm"]:.4f} mm   '
              + (f'u_z {uz:.4f} mm   ' if uz is not None else '')
              + f'({rec["tets"]:,} tets)'
              + (f'   label-surface peak {ls["peak_von_mises_MPa"]:.1f} MPa' if ls else ''))

    ok = {k: v for k, v in results.items() if 'error' not in v}

    if 'optimized' in ok and 'baseline' in ok:
        o, b = ok['optimized'], ok['baseline']
        print('\n--- solver verdict: optimized vs baseline (best of population) ---')
        for key, unit in (('mass_kg', 'kg'), ('peak_von_mises_MPa', 'MPa'),
                          ('max_displacement_mm', 'mm'), ('vertical_displacement_mm', 'mm')):
            if key not in o or key not in b:
                continue
            d = 100.0 * (o[key] - b[key]) / b[key] if b[key] else float('nan')
            print(f'  {key:24s} {b[key]:10.4f} -> {o[key]:10.4f} {unit:3s}  {d:+7.1f}%')

    verdicts = limit_verdicts(summary.get('limits') or {}, ok, backend,
                              verification_limits)
    if verdicts:
        print("\n--- solver verdict against the run's limits ---")
        for name, checks in verdicts.items():
            for check in checks:
                print(f"  {name:10s} {check['limit']:28s} {check['value']:10.4f} "
                      f"{'<=' if check['met'] else '> '} {check['allow']:10.4f} "
                      f"{check['unit']:3s}  {'met' if check['met'] else 'VIOLATED'}"
                      + (f"  ({check['note']})" if check.get('note') else ''))
            results[name]['limits'] = checks

    print('\n--- surrogate vs solver on the SAME geometry, SAME statistic '
          '(label surface) ---')
    for name, rec in ok.items():
        s, ls = rec.get('surrogate') or {}, rec.get('label_surface') or {}
        for key, unit in (('peak_von_mises_MPa', 'MPa'), ('max_displacement_mm', 'mm'),
                          ('vertical_displacement_mm', 'mm')):
            sv, fv = s.get(key), ls.get(key)
            if sv is None or not fv:
                continue
            print(f'  {name:10s} {key:24s} surrogate {sv:9.4f} vs solver {fv:9.4f} {unit:3s}'
                  f'   ratio {sv / fv:5.2f}x')

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or '.', exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump({'solver': 'tet4', 'load_cases': list(load_cases),
                       'length_scale': length_scale, 'stress_percentile': pct,
                       'analysis_backend': backend, 'resolution': resolution,
                       'verification_limits': verification_limits,
                       'definitions': {
                           'peak_von_mises_MPa': f'p{pct} of the volume-averaged nodal '
                                                 'von Mises over every tet node, worst case',
                           'label_surface.peak_von_mises_MPa':
                               f'p{pct} of |von Mises| over the boundary nodes decimated '
                               f"to {resolution['target_nodes']} (the surrogate labels' "
                               'measure), worst case',
                           'vertical_displacement_mm': 'max |u_z| over every node, '
                                                       'vertical load case',
                       },
                       'designs': results}, fh, indent=2, default=float)
        print(f'\nwrote {args.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
