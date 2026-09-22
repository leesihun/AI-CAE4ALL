#!/usr/bin/env python3
"""
Append DrivAerML geometry + aerodynamic labels to an SDFFlow HDF5 as the
``cond_extra`` sidecar.

    python add_drivaerml_conditions.py \
        --h5 ../../dataset/geometry_generation/ex2_drivaerml.h5 \
        --geo_csv D:/CAE_datasets_raw/drivaerml/geo_parameters_all.csv \
        --force_csv D:/CAE_datasets_raw/drivaerml/force_mom_all.csv --dry_run
    python add_drivaerml_conditions.py --list_names

The join key is the per-shape ``source`` attr's run id (trailing digits of the
file stem, e.g. ``drivaer_123.stl`` -> ``123``) against each CSV's ``run``
column (also accepted with a leading/trailing space, as DrivAerML ships it in
``geo_parameters_all.csv``). Mirrors ``add_fea_conditions.py``'s refuse-first
CLI contract, but against ``general_modules.drivaerml_conditions`` (its own
registry -- see that module's docstring for why it is not folded into
``condition_names.FEA_CONDITIONS``).

What is written (root of the HDF5, append-only, nothing else is touched):

    cond_extra                 float32 [num_shapes, k]   row i = shape i
    cond_extra_names           attr: array of str, length k
    cond_extra_source          attr: "<geo_csv> + <force_csv>"
    cond_extra_transforms      attr: JSON {name: 'identity'}
    cond_extra_created         attr: ISO-8601 timestamp
    cond_extra_csv_columns     attr: JSON {name: csv column}
    cond_extra_missing_count   attr: int, rows written as NaN
    cond_extra_missing_rows    attr: int32 indices of those rows
"""

import argparse
import csv
import datetime
import json
import os
import sys

import h5py
import numpy as np

from general_modules.drivaerml_conditions import (
    DRIVAERML_CONDITIONS, SIDECAR_CREATED_ATTR, SIDECAR_CSV_COLUMNS_ATTR,
    SIDECAR_DATASET, SIDECAR_MISSING_ATTR, SIDECAR_MISSING_ROWS_ATTR,
    SIDECAR_NAMES_ATTR, SIDECAR_SOURCE_ATTR, SIDECAR_TRANSFORMS_ATTR,
    all_names, describe, run_id_from_source, to_stored)

EXIT_OK = 0
EXIT_REFUSED = 2

RUN_ID_COLUMN = 'run'
MISSING_ROWS_ATTR_CAP = 8192


class Refusal(Exception):
    """A precondition the user must resolve (unmatched shapes, existing sidecar, ...)."""


def parse_names(arg):
    if arg is None or not str(arg).strip():
        return all_names()
    names = [p.strip().lower() for p in str(arg).split(',') if p.strip()]
    if not names:
        raise Refusal('--names parsed to an empty list')
    unknown = [n for n in names if n not in DRIVAERML_CONDITIONS]
    if unknown:
        raise Refusal(f'Unknown DrivAerML condition name(s) {unknown}. Registered: {all_names()}')
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise Refusal(f'--names contains duplicates: {dupes}')
    return names


def load_csv(csv_path):
    """CSV -> {run_id: {column: str}}, keyed by the stripped run-id column.

    DrivAerML's own two CSVs disagree on that column's case
    (``geo_parameters_all.csv`` -> ``Run``, ``force_mom_all.csv`` -> ``run``),
    so the match is case-insensitive.
    """
    if not os.path.isfile(csv_path):
        raise Refusal(f'CSV not found: {csv_path}')
    rows = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = [c.strip() for c in (reader.fieldnames or [])]
        run_col = next((c for c in columns if c.lower() == RUN_ID_COLUMN), None)
        if run_col is None:
            raise Refusal(f'CSV {csv_path} has no {RUN_ID_COLUMN!r} column; columns: {columns}')
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items()}
            key = str(row.get(run_col, '')).strip()
            if key:
                rows[key] = row
    if not rows:
        raise Refusal(f'CSV {csv_path} has no data rows')
    return rows, columns


def check_csv_columns(names, geo_columns, force_columns):
    missing = {}
    for name in names:
        col = DRIVAERML_CONDITIONS[name]['csv_column']
        if col not in geo_columns and col not in force_columns:
            missing[name] = col
    if missing:
        raise Refusal('Neither CSV has the column(s) required by the requested names: '
                       + ', '.join(f'{n} <- {c!r}' for n, c in missing.items()))


def read_h5_shapes(h5_path):
    if not os.path.isfile(h5_path):
        raise Refusal(f'HDF5 not found: {h5_path}')
    with h5py.File(h5_path, 'r') as h5:
        if 'shapes' not in h5:
            raise Refusal(f'{h5_path} has no "shapes" group; is it an SDFFlow dataset?')
        shapes = h5['shapes']
        num_shapes = int(h5.attrs.get('num_shapes', len(shapes)))
        base_names = [str(n) for n in h5.attrs.get('cond_names', [])]
        entries = []
        for idx in range(num_shapes):
            grp = shapes.get(f'{idx:05d}')
            if grp is None:
                raise Refusal(f'shape group {idx:05d} is missing although num_shapes={num_shapes}')
            source = grp.attrs.get('source')
            entries.append((idx, source, run_id_from_source(source)))
        exists = SIDECAR_DATASET in h5
    return num_shapes, base_names, entries, exists


def _parse_cell(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return float('nan')


def build_matrix(names, entries, geo_rows, force_rows):
    k = len(names)
    matrix = np.full((len(entries), k), np.nan, dtype=np.float64)
    unmatched = []
    used_geo, used_force = set(), set()
    for idx, source, run_id in entries:
        geo_row = geo_rows.get(run_id) if run_id is not None else None
        force_row = force_rows.get(run_id) if run_id is not None else None
        if geo_row is None and force_row is None:
            unmatched.append((idx, source, run_id))
            continue
        if geo_row is not None:
            used_geo.add(run_id)
        if force_row is not None:
            used_force.add(run_id)
        for j, name in enumerate(names):
            col = DRIVAERML_CONDITIONS[name]['csv_column']
            row = geo_row if (geo_row is not None and col in geo_row) else force_row
            if row is not None and col in row:
                matrix[idx, j] = to_stored(name, _parse_cell(row.get(col)))
    matched_mask = np.ones(len(entries), dtype=bool)
    for idx, _, _ in unmatched:
        matched_mask[idx] = False
    nonfinite = {}
    for j, name in enumerate(names):
        count = int((~np.isfinite(matrix[matched_mask, j])).sum())
        if count:
            nonfinite[name] = count
    unused = (len(geo_rows) - len(used_geo)) + (len(force_rows) - len(used_force))
    return matrix.astype(np.float32), unmatched, nonfinite, unused


def summarize(names, matrix):
    out = []
    for j, name in enumerate(names):
        col = matrix[:, j].astype(np.float64)
        finite = col[np.isfinite(col)]
        entry = DRIVAERML_CONDITIONS[name]
        if finite.size == 0:
            stats = dict(n=0, mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
        else:
            stats = dict(n=int(finite.size), mean=float(finite.mean()), std=float(finite.std()),
                         min=float(finite.min()), max=float(finite.max()))
        out.append(dict(name=name, csv_column=entry['csv_column'], **stats))
    return out


def print_table(stats_rows, num_shapes):
    name_w = max(len('name'), *(len(r['name']) for r in stats_rows))
    col_w = max(len('csv_column'), *(len(r['csv_column']) for r in stats_rows))
    header = (f'{"name":<{name_w}}  {"csv_column":<{col_w}}  {"n":>5}  '
              f'{"mean":>11}  {"std":>11}  {"min":>11}  {"max":>11}')
    print(f'Value statistics over {num_shapes} shapes:')
    print(header)
    print('-' * len(header))
    for r in stats_rows:
        print(f'{r["name"]:<{name_w}}  {r["csv_column"]:<{col_w}}  {r["n"]:>5d}  '
              f'{r["mean"]:>11.5g}  {r["std"]:>11.5g}  {r["min"]:>11.5g}  {r["max"]:>11.5g}')


def write_sidecar(h5_path, names, matrix, geo_csv, force_csv, overwrite, missing_count):
    transforms = {name: 'identity' for name in names}
    csv_columns = {name: DRIVAERML_CONDITIONS[name]['csv_column'] for name in names}
    with h5py.File(h5_path, 'r+') as h5:
        if SIDECAR_DATASET in h5:
            if not overwrite:
                raise Refusal(f'{h5_path} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')
            del h5[SIDECAR_DATASET]
        h5.create_dataset(SIDECAR_DATASET, data=matrix.astype(np.float32))
        h5.attrs[SIDECAR_NAMES_ATTR] = np.array(names, dtype=h5py.string_dtype(encoding='utf-8'))
        h5.attrs[SIDECAR_SOURCE_ATTR] = f'{os.path.abspath(geo_csv)} + {os.path.abspath(force_csv)}'
        h5.attrs[SIDECAR_TRANSFORMS_ATTR] = json.dumps(transforms)
        h5.attrs[SIDECAR_CREATED_ATTR] = datetime.datetime.now().isoformat(timespec='seconds')
        h5.attrs[SIDECAR_CSV_COLUMNS_ATTR] = json.dumps(csv_columns)
        h5.attrs[SIDECAR_MISSING_ATTR] = int(missing_count)
        missing_rows = np.flatnonzero((~np.isfinite(matrix)).any(axis=1))
        h5.attrs[SIDECAR_MISSING_ROWS_ATTR] = missing_rows[:MISSING_ROWS_ATTR_CAP].astype(np.int32)


def verify(h5_path, names, matrix):
    from general_modules.sdf_dataset import SDFShapeDataset, read_cond_extra

    with h5py.File(h5_path, 'r') as h5:
        stored_names, stored = read_cond_extra(h5)
    if stored_names != list(names):
        raise RuntimeError(f'verify: names round-trip mismatch {stored_names} != {list(names)}')
    if stored.shape != matrix.shape or not np.array_equal(
            np.isfinite(stored), np.isfinite(matrix)) or not np.allclose(
            np.nan_to_num(stored), np.nan_to_num(matrix), rtol=0, atol=0):
        raise RuntimeError('verify: sidecar values do not round-trip')
    ds = SDFShapeDataset(h5_path, [0], num_encoder_points=1, num_query_points=1, deterministic=True)
    try:
        cond = ds.get_cond(0)
        if ds.cond_names != ds.base_cond_names + list(names) or cond.shape[0] != ds.cond_dim:
            raise RuntimeError(f'verify: dataset merge failed: cond_names={ds.cond_names}')
        print(f'Verified: SDFShapeDataset reports cond_dim={ds.cond_dim} '
              f'({len(ds.base_cond_names)} geometric + {len(names)} DrivAerML)')
    finally:
        ds.close()


def _print_registry():
    print('Registered DrivAerML condition names (name <- csv column [unit, kind]):')
    for name in all_names():
        e = describe(name)
        print(f'  {name:<24} <- {e["csv_column"]:<22} [{e["unit"]}, {e["kind"]}]')


def build_parser():
    p = argparse.ArgumentParser(
        description='Append DrivAerML geometry + aero labels to an SDFFlow HDF5 as cond_extra.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h5', help='SDFFlow HDF5 dataset to extend in place')
    p.add_argument('--geo_csv', help='geo_parameters_all.csv')
    p.add_argument('--force_csv', help='force_mom_all.csv')
    p.add_argument('--names', default=None,
                   help='Comma-separated condition names to store (default: every registered name)')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--allow_missing', action='store_true')
    p.add_argument('--list_names', action='store_true')
    return p


def run(args):
    if args.list_names:
        _print_registry()
        return EXIT_OK
    if not args.h5 or not args.geo_csv or not args.force_csv:
        raise Refusal('--h5, --geo_csv and --force_csv are required (or use --list_names)')

    names = parse_names(args.names)
    print(f'Conditions ({len(names)}): {names}')

    num_shapes, base_names, entries, exists = read_h5_shapes(args.h5)
    print(f'HDF5: {args.h5} -> {num_shapes} shapes, geometric cond_names={base_names}, '
          f'{SIDECAR_DATASET} {"present" if exists else "absent"}')
    if exists and not args.overwrite:
        if args.dry_run:
            print(f'NOTE: {SIDECAR_DATASET!r} already exists; a real run needs --overwrite')
        else:
            raise Refusal(f'{args.h5} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')

    geo_rows, geo_columns = load_csv(args.geo_csv)
    force_rows, force_columns = load_csv(args.force_csv)
    print(f'geo_csv: {len(geo_rows)} rows; force_csv: {len(force_rows)} rows')
    check_csv_columns(names, geo_columns, force_columns)

    matrix, unmatched, nonfinite, unused = build_matrix(names, entries, geo_rows, force_rows)
    matched = num_shapes - len(unmatched)
    print(f'Join on run id: {matched}/{num_shapes} shapes matched; {unused} CSV row(s) unused')

    if unmatched:
        print(f'Unmatched shapes ({len(unmatched)}):')
        for idx, source, run_id in unmatched:
            print(f'  shape {idx:05d}: source={source!r} -> run={run_id!r}')
        if not args.allow_missing:
            raise Refusal(f'{len(unmatched)} shape(s) have no CSV row; fix the CSVs or pass '
                          '--allow_missing to write NaN rows for them')
        print(f'WARNING: --allow_missing: {len(unmatched)} shape(s) written as NaN rows')
    if nonfinite:
        print('Non-finite stored values among matched shapes:')
        for name, count in nonfinite.items():
            print(f'  {name}: {count}')
        if not args.allow_missing:
            raise Refusal('non-finite stored values; drop those names with --names or pass --allow_missing')
        print('WARNING: --allow_missing: non-finite values written as NaN')

    print_table(summarize(names, matrix), num_shapes)
    missing_count = int((~np.isfinite(matrix)).any(axis=1).sum())

    if args.dry_run:
        print(f'DRY RUN: nothing written. Would create {SIDECAR_DATASET} float32 '
              f'{tuple(matrix.shape)} in {args.h5}' + (' (replacing the existing sidecar)'
                                                       if exists else ''))
        return EXIT_OK

    write_sidecar(args.h5, names, matrix, args.geo_csv, args.force_csv, args.overwrite, missing_count)
    print(f'Wrote {SIDECAR_DATASET} float32 {tuple(matrix.shape)} to {args.h5}')
    verify(args.h5, names, matrix)
    return EXIT_OK


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except Refusal as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return EXIT_REFUSED


if __name__ == '__main__':
    sys.exit(main())
