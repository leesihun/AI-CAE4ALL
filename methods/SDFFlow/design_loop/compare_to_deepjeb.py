"""Place an optimized design against the real DeepJEB population.

`mode optimize` already reports the design against its own baseline population --
the random draws from the generator that the allowables were calibrated on. That
answers "did the search improve on where it started". It does not answer "how does
this compare to the brackets a human actually designed", which needs DeepJEB's own
per-design labels (`Scalar/bracket_labels.csv`, 2138 rows).

What is and is not comparable across those two sources:

  mass        DIRECTLY comparable. It is volume x density with no solver in the
              path, so as long as `opt_length_scale` is calibrated (0.1838 m /
              1.8 units, the extent implied by the label file) the number means
              the same thing on both sides.
  stress      RELATIVE ONLY. The loop reports the 99.5th-percentile von Mises
              from a tet4 solve; DeepJEB reports a true max from its own FEA.
              tet4 is stiff and a percentile is not a max, so treat the ratio as
              an ordering signal, never as an error.
  displacement  RELATIVE ONLY, same reason.

So the honest headline is the mass percentile; stress and displacement are
reported alongside it as context, explicitly labelled.

  python methods/SDFFlow/design_loop/compare_to_deepjeb.py \
      --summary output/geometry_generation/ex4/optimization_surrogate/summary.json \
      --labels D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv \
      --json-out output/geometry_generation/ex4/optimization_surrogate/vs_deepjeb.json
"""

import argparse
import csv
import json
import os
import sys

# Load cases the optimize config searched over, mapped to the label columns.
CASE_COLUMNS = {
    'vertical':   ('max_ver_stress(MPa)', 'abs_max_ver_magdisp(mm)'),
    'horizontal': ('max_hor_stress(MPa)', 'abs_max_hor_magdisp(mm)'),
    'diagonal':   ('max_dia_stress(MPa)', 'abs_max_dia_magdisp(mm)'),
    'torsional':  ('max_tor_stress(MPa)', 'abs_max_tor_magdisp(mm)'),
}


def load_labels(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        for row in csv.DictReader(fh):
            try:
                rec = {'item': row['item_name'], 'mass': float(row['mass(kg)'])}
                for case, (scol, dcol) in CASE_COLUMNS.items():
                    rec[case + '_stress'] = float(row[scol])
                    rec[case + '_disp'] = float(row[dcol])
                rows.append(rec)
            except (KeyError, ValueError):
                continue
    return rows


def percentile_of(values, x):
    """Fraction of the population at or below x, in percent."""
    if not values:
        return float('nan')
    return 100.0 * sum(1 for v in values if v <= x) / len(values)


def quantiles(values):
    s = sorted(values)
    n = len(s)
    def q(p):
        return s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return {'min': s[0], 'p05': q(0.05), 'p25': q(0.25), 'median': q(0.50),
            'p75': q(0.75), 'p95': q(0.95), 'max': s[-1]}


def pick(summary, *names):
    """First present key among `names`, walking one level of nesting."""
    for n in names:
        if isinstance(summary, dict) and n in summary:
            return summary[n]
    return None


def brief_of(summary, which):
    """The optimized / baseline / typical design record, whichever shape it is in."""
    ver = summary.get('verified') or {}
    if which in ver:
        return ver[which]
    return summary.get(which)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--labels', default='D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv')
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args(argv)

    with open(args.summary, encoding='utf-8') as fh:
        summary = json.load(fh)
    labels = load_labels(args.labels)
    if not labels:
        print(f'no usable rows in {args.labels}', file=sys.stderr)
        return 1

    cases = summary.get('load_cases') or summary.get('opt_load_cases') or ['vertical', 'diagonal']
    if isinstance(cases, str):
        cases = [c.strip() for c in cases.split(',') if c.strip()]

    opt = brief_of(summary, 'optimized')
    base = brief_of(summary, 'baseline')
    typical = brief_of(summary, 'typical')
    if not opt:
        print('summary.json has no optimized record; keys: '
              + ', '.join(sorted(summary)), file=sys.stderr)
        return 1

    def val(rec, *names):
        if not isinstance(rec, dict):
            return None
        for n in names:
            if n in rec and isinstance(rec[n], (int, float)):
                return float(rec[n])
        fea = rec.get('fea')
        if isinstance(fea, dict):
            for n in names:
                if n in fea and isinstance(fea[n], (int, float)):
                    return float(fea[n])
        return None

    out = {'source_summary': os.path.abspath(args.summary),
           'population_size': len(labels), 'load_cases': cases, 'axes': {}}

    print(f'\nOptimized design vs the real DeepJEB population ({len(labels)} brackets)')
    print('=' * 76)

    # --- mass: the one directly comparable axis ---------------------------- #
    m_opt = val(opt, 'mass')
    masses = [r['mass'] for r in labels]
    qs = quantiles(masses)
    if m_opt is not None:
        pct = percentile_of(masses, m_opt)
        out['axes']['mass_kg'] = {'optimized': m_opt, 'percentile': pct,
                                  'population': qs, 'comparable': True}
        print('\nMASS (kg) -- directly comparable, no solver in the path')
        print(f'  optimized            {m_opt:.4f}')
        print(f'  DeepJEB median       {qs["median"]:.4f}   '
              f'(p05 {qs["p05"]:.4f} / p95 {qs["p95"]:.4f}, range {qs["min"]:.4f}-{qs["max"]:.4f})')
        print(f'  percentile           {pct:.1f}%  '
              f'({"lighter" if pct < 50 else "heavier"} than {pct:.0f}% of real brackets)')
        rel = 100.0 * (m_opt - qs['median']) / qs['median']
        print(f'  vs median            {rel:+.1f}%')
        for name, rec in (('baseline (best of population)', base), ('typical (median-mass)', typical)):
            v = val(rec, 'mass')
            if v is not None:
                print(f'  loop {name:<30s} {v:.4f}   '
                      f'(optimized is {100.0 * (m_opt - v) / v:+.1f}% vs this)')
                out['axes']['mass_kg'].setdefault('loop_reference', {})[name] = v

    # --- stress / displacement: ordering only ------------------------------ #
    s_opt = val(opt, 'peak_von_mises', 'peak_stress')
    if s_opt is not None and s_opt > 1e4:      # loop stores Pa; labels are MPa
        s_opt = s_opt / 1e6
    d_opt = val(opt, 'max_displacement')
    if d_opt is not None and d_opt < 0.05:     # loop stores m; labels are mm
        d_opt = d_opt * 1000.0

    for case in cases:
        if case not in CASE_COLUMNS:
            continue
        svals = [r[case + '_stress'] for r in labels]
        dvals = [r[case + '_disp'] for r in labels]
        sq, dq = quantiles(svals), quantiles(dvals)
        entry = {'population_stress_mpa': sq, 'population_disp_mm': dq, 'comparable': False}
        print(f'\n{case.upper()} -- RELATIVE ONLY '
              f'(loop: 99.5th-percentile von Mises on tet4; labels: true max from DeepJEB FEA)')
        print(f'  DeepJEB stress  median {sq["median"]:8.1f} MPa   '
              f'(p05 {sq["p05"]:.1f} / p95 {sq["p95"]:.1f})')
        if s_opt is not None:
            entry['optimized_stress_mpa'] = s_opt
            entry['stress_percentile'] = percentile_of(svals, s_opt)
            print(f'  optimized       {s_opt:8.1f} MPa   '
                  f'-> sits at the {entry["stress_percentile"]:.0f}th percentile of the label set')
        print(f'  DeepJEB disp    median {dq["median"]:8.4f} mm    '
              f'(p05 {dq["p05"]:.4f} / p95 {dq["p95"]:.4f})')
        if d_opt is not None:
            entry['optimized_disp_mm'] = d_opt
            entry['disp_percentile'] = percentile_of(dvals, d_opt)
            print(f'  optimized       {d_opt:8.4f} mm    '
                  f'-> {entry["disp_percentile"]:.0f}th percentile')
        out['axes'][case] = entry

    print('\nRead the mass line as a result. Read the stress and displacement lines as'
          '\nordering context only -- the two sides do not measure the same quantity.')

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or '.', exist_ok=True)
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f'\nwrote {args.json_out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
