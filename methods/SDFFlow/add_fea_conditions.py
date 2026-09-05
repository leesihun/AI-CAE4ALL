#!/usr/bin/env python3
"""
Append DeepJEB FEA labels to an SDFFlow HDF5 as the ``cond_extra`` sidecar.

    python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 \
        --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv --dry_run
    python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 \
        --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv
    python add_fea_conditions.py --list_names

The join key is the per-shape ``source`` attr's basename without extension
(``DeepJEB/SurfaceMesh/101_428.stl`` -> ``101_428``) against the CSV's
``item_name`` column. The run REFUSES (exit 2) when any shape is unmatched or
any stored value is non-finite (e.g. ``log`` of a non-positive label) unless
``--allow_missing`` is given, in which case those rows are written as NaN with
a WARNING; and when ``cond_extra`` already exists unless ``--overwrite``.

What is written (root of the HDF5, append-only, nothing else is touched):

    cond_extra                 float32 [num_shapes, k]   row i = shape i, STORED space
    cond_extra_names           attr: array of str, length k
    cond_extra_source          attr: absolute csv path
    cond_extra_transforms      attr: JSON {name: 'identity'|'log'}
    cond_extra_created         attr: ISO-8601 timestamp
    cond_extra_csv_columns     attr: JSON {name: csv column}      (provenance)
    cond_extra_missing_count   attr: int, rows written as NaN     (provenance)
    cond_extra_missing_rows    attr: int32 indices of those rows (provenance; a
                               future validator can exclude them without re-reading
                               the whole matrix. Capped at 8192 entries so the attr
                               stays inside HDF5's 64 KB compact limit)

``general_modules.sdf_dataset.SDFShapeDataset`` then reports
``cond_names = base names + cond_extra_names`` and returns the concatenation
from ``__getitem__['cond']`` / ``get_cond``; ``condition_names`` in an FM
config selects from the merged list. A file without the sidecar behaves
exactly as before.
"""

import argparse
import csv
import datetime
import json
import os
import sys

import h5py
import numpy as np

from general_modules.condition_names import (
    FEA_CONDITIONS, GEOMETRIC_NAMES, SIDECAR_CREATED_ATTR, SIDECAR_CSV_COLUMNS_ATTR,
    SIDECAR_DATASET, SIDECAR_MISSING_ATTR, SIDECAR_MISSING_ROWS_ATTR,
    SIDECAR_NAMES_ATTR, SIDECAR_SOURCE_ATTR,
    SIDECAR_TRANSFORMS_ATTR, all_fea_names, describe, item_name_from_source,
    normalize_name, to_stored)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2

ITEM_NAME_COLUMN = 'item_name'


class Refusal(Exception):
    """A precondition the user must resolve (unmatched shapes, existing sidecar, ...)."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def parse_names(arg):
    """``--names a,b,c`` -> validated list in the order given (default: registry order)."""
    if arg is None or not str(arg).strip():
        return all_fea_names()
    names = [normalize_name(part) for part in str(arg).split(',') if part.strip()]
    if not names:
        raise Refusal('--names parsed to an empty list')
    unknown = [n for n in names if n not in FEA_CONDITIONS]
    if unknown:
        geometric = [n for n in unknown if n in GEOMETRIC_NAMES]
        hint = (f' ({geometric} are geometric descriptors already in every cond row)'
                if geometric else '')
        raise Refusal(f'Unknown FEA condition name(s) {unknown}{hint}. '
                      f'Registered: {all_fea_names()}')
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise Refusal(f'--names contains duplicates: {dupes}')
    return names


def load_csv(csv_path):
    """CSV -> (rows: {item_name: {column: str}}, columns: list[str])."""
    if not os.path.isfile(csv_path):
        raise Refusal(f'CSV not found: {csv_path}')
    rows = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        if ITEM_NAME_COLUMN not in columns:
            raise Refusal(f'CSV {csv_path} has no {ITEM_NAME_COLUMN!r} column; columns: {columns}')
        duplicates = []
        for row in reader:
            key = str(row.get(ITEM_NAME_COLUMN, '')).strip()
            if not key:
                continue
            if key in rows:
                duplicates.append(key)
            rows[key] = row
    if duplicates:
        raise Refusal(f'CSV has duplicate {ITEM_NAME_COLUMN} values: '
                      f'{sorted(set(duplicates))[:20]}{" ..." if len(set(duplicates)) > 20 else ""}')
    if not rows:
        raise Refusal(f'CSV {csv_path} has no data rows')
    return rows, columns


def check_csv_columns(names, columns):
    missing = {}
    for name in names:
        col = FEA_CONDITIONS[name]['csv_column']
        if col not in columns:
            missing[name] = col
    if missing:
        raise Refusal('CSV lacks the column(s) required by the requested names: '
                      + ', '.join(f'{n} <- {c!r}' for n, c in missing.items()))


def read_h5_shapes(h5_path):
    """(num_shapes, base cond names, [(shape_idx, source, item_name), ...], sidecar_exists)."""
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
            entries.append((idx, source, item_name_from_source(source)))
        exists = SIDECAR_DATASET in h5
    return num_shapes, base_names, entries, exists


# ---------------------------------------------------------------------------
# Join and transform
# ---------------------------------------------------------------------------

def _parse_cell(text):
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return float('nan')


def build_matrix(names, entries, rows):
    """Stored-space matrix [num_shapes, k] plus the bookkeeping the CLI reports.

    Returns (matrix float32, unmatched [(idx, source, item_name)],
    nonfinite {name: count over matched shapes}, unused_csv_rows int).
    """
    k = len(names)
    matrix = np.full((len(entries), k), np.nan, dtype=np.float64)
    unmatched = []
    used = set()
    for idx, source, item in entries:
        row = rows.get(item) if item is not None else None
        if row is None:
            unmatched.append((idx, source, item))
            continue
        used.add(item)
        for j, name in enumerate(names):
            raw = _parse_cell(row.get(FEA_CONDITIONS[name]['csv_column']))
            matrix[idx, j] = to_stored(name, raw)
    matched_mask = np.ones(len(entries), dtype=bool)
    for idx, _, _ in unmatched:
        matched_mask[idx] = False
    nonfinite = {}
    for j, name in enumerate(names):
        count = int((~np.isfinite(matrix[matched_mask, j])).sum())
        if count:
            nonfinite[name] = count
    unused = len(rows) - len(used)
    return matrix.astype(np.float32), unmatched, nonfinite, unused


def summarize(names, matrix):
    """Per-name stats of the STORED values (NaN-aware). Returns list of dicts."""
    out = []
    for j, name in enumerate(names):
        col = matrix[:, j].astype(np.float64)
        finite = col[np.isfinite(col)]
        entry = FEA_CONDITIONS[name]
        if finite.size == 0:
            stats = dict(n=0, mean=np.nan, std=np.nan, cv=np.nan, min=np.nan, max=np.nan)
        else:
            mean = float(finite.mean())
            std = float(finite.std())
            stats = dict(n=int(finite.size), mean=mean, std=std,
                         cv=(std / abs(mean) if abs(mean) > 0 else np.nan),
                         min=float(finite.min()), max=float(finite.max()))
        out.append(dict(name=name, csv_column=entry['csv_column'],
                        transform=entry['transform'], unit=entry['unit'], **stats))
    return out


def print_table(stats_rows, num_shapes):
    name_w = max(len('name'), *(len(r['name']) for r in stats_rows))
    col_w = max(len('csv_column'), *(len(r['csv_column']) for r in stats_rows))
    header = (f'{"name":<{name_w}}  {"csv_column":<{col_w}}  {"tf":<8}  {"n":>5}  '
              f'{"mean":>11}  {"std":>11}  {"cv":>7}  {"min":>11}  {"max":>11}')
    print(f'Stored (transformed) value statistics over {num_shapes} shapes '
          '(cv = std/|mean| for identity names; for log names the std of ln(x) itself '
          'approximates the raw coefficient of variation):')
    print(header)
    print('-' * len(header))
    for r in stats_rows:
        cv = f'{r["cv"]:>7.3f}' if r['transform'] == 'identity' else f'{"-":>7}'
        print(f'{r["name"]:<{name_w}}  {r["csv_column"]:<{col_w}}  {r["transform"]:<8}  '
              f'{r["n"]:>5d}  {r["mean"]:>11.5g}  {r["std"]:>11.5g}  {cv}  '
              f'{r["min"]:>11.5g}  {r["max"]:>11.5g}')


# ---------------------------------------------------------------------------
# Write / verify
# ---------------------------------------------------------------------------

MISSING_ROWS_ATTR_CAP = 8192  # int32 entries; keeps the attr inside HDF5's 64 KB limit


def write_sidecar(h5_path, names, matrix, csv_path, overwrite, missing_count):
    transforms = {name: FEA_CONDITIONS[name]['transform'] for name in names}
    csv_columns = {name: FEA_CONDITIONS[name]['csv_column'] for name in names}
    with h5py.File(h5_path, 'r+') as h5:
        if SIDECAR_DATASET in h5:
            if not overwrite:
                raise Refusal(f'{h5_path} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')
            del h5[SIDECAR_DATASET]
        h5.create_dataset(SIDECAR_DATASET, data=matrix.astype(np.float32))
        h5.attrs[SIDECAR_NAMES_ATTR] = np.array(names, dtype=h5py.string_dtype(encoding='utf-8'))
        h5.attrs[SIDECAR_SOURCE_ATTR] = os.path.abspath(csv_path)
        h5.attrs[SIDECAR_TRANSFORMS_ATTR] = json.dumps(transforms)
        h5.attrs[SIDECAR_CREATED_ATTR] = datetime.datetime.now().isoformat(timespec='seconds')
        h5.attrs[SIDECAR_CSV_COLUMNS_ATTR] = json.dumps(csv_columns)
        h5.attrs[SIDECAR_MISSING_ATTR] = int(missing_count)
        missing_rows = np.flatnonzero((~np.isfinite(matrix)).any(axis=1))
        h5.attrs[SIDECAR_MISSING_ROWS_ATTR] = missing_rows[:MISSING_ROWS_ATTR_CAP].astype(np.int32)


def verify(h5_path, names, matrix):
    """Re-open through the dataset class and check the merge end to end."""
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
            raise RuntimeError(f'verify: dataset merge failed: cond_names={ds.cond_names}, '
                               f'cond shape {cond.shape}')
        print(f'Verified: SDFShapeDataset reports cond_dim={ds.cond_dim} '
              f'({len(ds.base_cond_names)} geometric + {len(names)} FEA); shape 0 cond = '
              f'{np.array2string(cond, precision=4, max_line_width=200)}')
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_registry():
    print('Registered FEA condition names (name <- csv column [transform, unit, kind, load_case]):')
    for name in all_fea_names():
        e = describe(name)
        print(f'  {name:<28} <- {e["csv_column"]:<26} [{e["transform"]}, {e["unit"]}, '
              f'{e["kind"]}, {e["load_case"]}]')
    print(f'Geometric names already in every cond row: {list(GEOMETRIC_NAMES)}')


def build_parser():
    p = argparse.ArgumentParser(
        description='Append DeepJEB FEA labels to an SDFFlow HDF5 as the cond_extra sidecar.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h5', help='SDFFlow HDF5 dataset to extend in place')
    p.add_argument('--csv', help='bracket_labels.csv (item_name column + label columns)')
    p.add_argument('--names', default=None,
                   help='Comma-separated FEA condition names to store (default: every registered name)')
    p.add_argument('--dry_run', action='store_true',
                   help='Join, transform, and print statistics without writing anything')
    p.add_argument('--overwrite', action='store_true',
                   help='Replace an existing cond_extra sidecar')
    p.add_argument('--allow_missing', action='store_true',
                   help='Write NaN rows for unmatched shapes / non-finite labels instead of refusing')
    p.add_argument('--list_names', action='store_true',
                   help='Print the registry and exit')
    return p


def run(args):
    if args.list_names:
        _print_registry()
        return EXIT_OK
    if not args.h5 or not args.csv:
        raise Refusal('--h5 and --csv are required (or use --list_names)')

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

    rows, columns = load_csv(args.csv)
    print(f'CSV: {args.csv} -> {len(rows)} rows, {len(columns)} columns')
    check_csv_columns(names, columns)

    matrix, unmatched, nonfinite, unused = build_matrix(names, entries, rows)
    matched = num_shapes - len(unmatched)
    print(f'Join on {ITEM_NAME_COLUMN}: {matched}/{num_shapes} shapes matched; '
          f'{unused} CSV row(s) unused')

    if unmatched:
        print(f'Unmatched shapes ({len(unmatched)}):')
        for idx, source, item in unmatched:
            print(f'  shape {idx:05d}: source={source!r} -> {ITEM_NAME_COLUMN}={item!r}')
        if not args.allow_missing:
            raise Refusal(f'{len(unmatched)} shape(s) have no CSV row; fix the CSV or pass '
                          '--allow_missing to write NaN rows for them')
        print(f'WARNING: --allow_missing: {len(unmatched)} shape(s) written as NaN rows; '
              'compute_cond_stats will propagate NaN -- exclude those shapes or names before FM training')
    if nonfinite:
        print('Non-finite stored values among matched shapes (empty cell, or log of a '
              'non-positive label):')
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

    write_sidecar(args.h5, names, matrix, args.csv, args.overwrite, missing_count)
    print(f'Wrote {SIDECAR_DATASET} float32 {tuple(matrix.shape)} + attrs '
          f'[{SIDECAR_NAMES_ATTR}, {SIDECAR_SOURCE_ATTR}, {SIDECAR_TRANSFORMS_ATTR}, '
          f'{SIDECAR_CREATED_ATTR}, {SIDECAR_CSV_COLUMNS_ATTR}, '
          f'{SIDECAR_MISSING_ATTR}={missing_count}, {SIDECAR_MISSING_ROWS_ATTR}] '
          f'to {args.h5}')
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
