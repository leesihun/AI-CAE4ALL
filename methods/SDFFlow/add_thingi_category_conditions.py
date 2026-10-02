#!/usr/bin/env python3
"""
Append a one-hot Thingi10K category label to an SDFFlow HDF5 as the
``cond_extra`` sidecar.

    python add_thingi_category_conditions.py \
        --h5 ../../dataset/geometry_generation/ex4_thingi10k.h5 \
        --categories /data/Lee/CAE_datasets_raw/thingi10k/categories.csv --dry_run
    python add_thingi_category_conditions.py --list_categories

The category is the uploader's Thingiverse category from the Thingi10K
release's ``contextual_data.csv``, joined to files through
``input_summary.csv`` (file -> Thing). ``D:/CAE_datasets_raw/thingi10k/
make_thingi_lists.py`` writes the per-file ``categories.csv``
(``file_id,category``) this script reads; the join key is the per-shape
``source`` basename without extension (``.../raw_meshes/32770.stl`` ->
``32770``). Uncategorised Things ("None" in the release) are their own
``none`` class, so every shape gets exactly one 1 and the vocabulary is
closed at the 11 classes the release uses. A category outside that list is
refused rather than dropped. Mirrors ``add_mcb_class_conditions.py``'s
refuse-first CLI and writes the same ``cond_extra`` layout, so
``SDFShapeDataset`` merges it identically.

What is written (root of the HDF5, append-only):

    cond_extra                 float32 [num_shapes, 11]  one-hot row per shape
    cond_extra_names           attr: ('cat_3d_printing', ..., 'cat_none')
    cond_extra_source          attr: "Thingi10K contextual_data.csv Category"
    cond_extra_transforms      attr: JSON {name: 'identity'}
    cond_extra_created         attr: ISO-8601 timestamp
    cond_extra_csv_columns     attr: JSON {name: '<category string>'}
    cond_extra_missing_count   attr: int, rows written as NaN (unmatched file)
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

from general_modules.condition_names import (
    SIDECAR_CREATED_ATTR, SIDECAR_CSV_COLUMNS_ATTR, SIDECAR_DATASET,
    SIDECAR_MISSING_ATTR, SIDECAR_MISSING_ROWS_ATTR, SIDECAR_NAMES_ATTR,
    SIDECAR_SOURCE_ATTR, SIDECAR_TRANSFORMS_ATTR)

EXIT_OK = 0
EXIT_REFUSED = 2
MISSING_ROWS_ATTR_CAP = 8192

# The Thingi10K release's category vocabulary (category string -> stored name).
THINGI_CATEGORIES = {
    '3d-printing': 'cat_3d_printing',
    'art': 'cat_art',
    'fashion': 'cat_fashion',
    'gadgets': 'cat_gadgets',
    'hobby': 'cat_hobby',
    'household': 'cat_household',
    'learning': 'cat_learning',
    'models': 'cat_models',
    'tools': 'cat_tools',
    'toys-and-games': 'cat_toys_and_games',
    'none': 'cat_none',
}
STORED_NAMES = list(THINGI_CATEGORIES.values())


class Refusal(Exception):
    """A precondition the user must resolve (unmatched shapes, existing sidecar, ...)."""


def stem_from_source(source):
    """Builder ``source`` attr -> file stem (``'.../raw_meshes/32770.stl' -> '32770'``)."""
    if source is None:
        return None
    if isinstance(source, bytes):
        source = source.decode('utf-8', 'replace')
    base = str(source).replace('\\', '/').rstrip('/').split('/')[-1]
    return os.path.splitext(base)[0] or None


def read_categories(csv_path):
    if not os.path.isfile(csv_path):
        raise Refusal(f'categories CSV not found: {csv_path}')
    table = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not {'file_id', 'category'} <= set(reader.fieldnames or ()):
            raise Refusal(f'{csv_path} needs columns file_id,category; has {reader.fieldnames}')
        for row in reader:
            key = str(row['file_id']).strip()
            cat = (str(row['category']).strip().lower() or 'none')
            table[key] = 'none' if cat in ('nan', 'null') else cat
    return table


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
            entries.append((idx, source, stem_from_source(source)))
        exists = SIDECAR_DATASET in h5
    return num_shapes, base_names, entries, exists


def build_matrix(entries, table):
    k = len(STORED_NAMES)
    matrix = np.full((len(entries), k), np.nan, dtype=np.float32)
    unmatched, unknown = [], []
    counts = {cat: 0 for cat in THINGI_CATEGORIES}
    for idx, source, stem in entries:
        cat = table.get(stem) if stem is not None else None
        if cat is None:
            unmatched.append((idx, source, stem))
            continue
        if cat not in THINGI_CATEGORIES:
            unknown.append((idx, source, cat))
            continue
        row = np.zeros(k, dtype=np.float32)
        row[STORED_NAMES.index(THINGI_CATEGORIES[cat])] = 1.0
        matrix[idx] = row
        counts[cat] += 1
    return matrix, unmatched, unknown, counts


def print_table(counts, num_shapes):
    name_w = max(len('category'), *(len(c) for c in counts))
    print(f'Category distribution over {num_shapes} shapes:')
    print(f'{"category":<{name_w}}  {"stored_name":<20}  {"count":>6}')
    for cat, stored in THINGI_CATEGORIES.items():
        print(f'{cat:<{name_w}}  {stored:<20}  {counts[cat]:>6d}')


def write_sidecar(h5_path, matrix, overwrite, missing_count):
    transforms = {name: 'identity' for name in STORED_NAMES}
    csv_columns = dict(zip(STORED_NAMES, THINGI_CATEGORIES.keys()))
    with h5py.File(h5_path, 'r+') as h5:
        if SIDECAR_DATASET in h5:
            if not overwrite:
                raise Refusal(f'{h5_path} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')
            del h5[SIDECAR_DATASET]
        h5.create_dataset(SIDECAR_DATASET, data=matrix.astype(np.float32))
        h5.attrs[SIDECAR_NAMES_ATTR] = np.array(STORED_NAMES, dtype=h5py.string_dtype(encoding='utf-8'))
        h5.attrs[SIDECAR_SOURCE_ATTR] = 'Thingi10K contextual_data.csv Category'
        h5.attrs[SIDECAR_TRANSFORMS_ATTR] = json.dumps(transforms)
        h5.attrs[SIDECAR_CREATED_ATTR] = datetime.datetime.now().isoformat(timespec='seconds')
        h5.attrs[SIDECAR_CSV_COLUMNS_ATTR] = json.dumps(csv_columns)
        h5.attrs[SIDECAR_MISSING_ATTR] = int(missing_count)
        missing_rows = np.flatnonzero((~np.isfinite(matrix)).any(axis=1))
        h5.attrs[SIDECAR_MISSING_ROWS_ATTR] = missing_rows[:MISSING_ROWS_ATTR_CAP].astype(np.int32)


def verify(h5_path, matrix):
    from general_modules.sdf_dataset import SDFShapeDataset, read_cond_extra

    with h5py.File(h5_path, 'r') as h5:
        stored_names, stored = read_cond_extra(h5)
    if stored_names != STORED_NAMES:
        raise RuntimeError(f'verify: names round-trip mismatch {stored_names} != {STORED_NAMES}')
    if stored.shape != matrix.shape or not np.array_equal(
            np.isfinite(stored), np.isfinite(matrix)) or not np.allclose(
            np.nan_to_num(stored), np.nan_to_num(matrix), rtol=0, atol=0):
        raise RuntimeError('verify: sidecar values do not round-trip')
    ds = SDFShapeDataset(h5_path, [0], num_encoder_points=1, num_query_points=1, deterministic=True)
    try:
        cond = ds.get_cond(0)
        if ds.cond_names != ds.base_cond_names + STORED_NAMES or cond.shape[0] != ds.cond_dim:
            raise RuntimeError(f'verify: dataset merge failed: cond_names={ds.cond_names}')
        print(f'Verified: SDFShapeDataset reports cond_dim={ds.cond_dim} '
              f'({len(ds.base_cond_names)} geometric + {len(STORED_NAMES)} Thingi10K category)')
    finally:
        ds.close()


def build_parser():
    p = argparse.ArgumentParser(
        description='Append a one-hot Thingi10K category label to an SDFFlow HDF5 as cond_extra.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h5', help='SDFFlow HDF5 dataset to extend in place')
    p.add_argument('--categories', help='file_id,category CSV (make_thingi_lists.py)')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--allow_missing', action='store_true')
    p.add_argument('--list_categories', action='store_true')
    return p


def run(args):
    if args.list_categories:
        print('Registered Thingi10K categories (category -> stored one-hot name):')
        for cat, stored in THINGI_CATEGORIES.items():
            print(f'  {cat!r:<18} -> {stored}')
        return EXIT_OK
    if not args.h5 or not args.categories:
        raise Refusal('--h5 and --categories are required (or use --list_categories)')

    table = read_categories(args.categories)
    num_shapes, base_names, entries, exists = read_h5_shapes(args.h5)
    print(f'HDF5: {args.h5} -> {num_shapes} shapes, geometric cond_names={base_names}, '
          f'{SIDECAR_DATASET} {"present" if exists else "absent"}; CSV rows={len(table)}')
    if exists and not args.overwrite:
        if args.dry_run:
            print(f'NOTE: {SIDECAR_DATASET!r} already exists; a real run needs --overwrite')
        else:
            raise Refusal(f'{args.h5} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')

    matrix, unmatched, unknown, counts = build_matrix(entries, table)

    if unknown:
        print(f'Shape(s) whose category is outside the registered vocabulary ({len(unknown)}):')
        for idx, source, cat in unknown[:20]:
            print(f'  shape {idx:05d}: source={source!r} -> category={cat!r}')
        raise Refusal(f'{len(unknown)} shape(s) have an unregistered category; '
                      'update THINGI_CATEGORIES, do not silently drop them')

    if unmatched:
        print(f'Shape(s) with no row in {args.categories} ({len(unmatched)}):')
        for idx, source, _ in unmatched[:20]:
            print(f'  shape {idx:05d}: source={source!r}')
        if not args.allow_missing:
            raise Refusal(f'{len(unmatched)} shape(s) have no category; pass '
                          '--allow_missing to write NaN rows for them')
        print(f'WARNING: --allow_missing: {len(unmatched)} shape(s) written as NaN rows')

    print_table(counts, num_shapes)
    missing_count = int((~np.isfinite(matrix)).any(axis=1).sum())

    if args.dry_run:
        print(f'DRY RUN: nothing written. Would create {SIDECAR_DATASET} float32 '
              f'{tuple(matrix.shape)} in {args.h5}' + (' (replacing the existing sidecar)'
                                                       if exists else ''))
        return EXIT_OK

    write_sidecar(args.h5, matrix, args.overwrite, missing_count)
    print(f'Wrote {SIDECAR_DATASET} float32 {tuple(matrix.shape)} to {args.h5}')
    verify(args.h5, matrix)
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
