#!/usr/bin/env python3
"""
Append a one-hot MCB nut-family class label to an SDFFlow HDF5 as the
``cond_extra`` sidecar.

    python add_mcb_class_conditions.py \
        --h5 ../../dataset/geometry_generation/ex3_mcb_nuts.h5 --dry_run
    python add_mcb_class_conditions.py --list_classes

Unlike ``add_fea_conditions.py``/``add_drivaerml_conditions.py`` (CSV-join by
an id), MCB carries its only label as the immediate parent directory name of
each mesh file (``dataset_org_norm/{train,test}/<ClassName>/<id>.obj``), so the
join key here is that directory name, read straight from each shape's
``source`` attr -- no CSV involved. Fixed to the 5-class fastener subset the
build was scoped to (see ``SDFFlow/CLAUDE.md`` "Conditioning datasets beyond
DeepJEB" / MCB section); a shape whose source resolves to a 6th class is
refused rather than silently dropped, since it means the subset changed
without this script being updated. Mirrors ``add_fea_conditions.py``'s
refuse-first CLI contract; writes the exact same ``cond_extra`` sidecar
layout as the DeepJEB and DrivAerML scripts, so ``SDFShapeDataset`` merges any
of the three identically.

What is written (root of the HDF5, append-only):

    cond_extra                 float32 [num_shapes, 5]   one-hot row per shape
    cond_extra_names           attr: ('class_hexagonal_nuts', ..., 'class_locknuts')
    cond_extra_source          attr: "MCB dataset_org_norm parent-directory name"
    cond_extra_transforms      attr: JSON {name: 'identity'}
    cond_extra_created         attr: ISO-8601 timestamp
    cond_extra_csv_columns     attr: JSON {name: '<original class folder name>'}
    cond_extra_missing_count   attr: int, rows written as NaN (unresolved class)
    cond_extra_missing_rows    attr: int32 indices of those rows
"""

import argparse
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

# The 5-class subset this build was scoped to (folder name -> stored name).
MCB_CLASSES = {
    'Hexagonal nuts': 'class_hexagonal_nuts',
    'Slotted nuts': 'class_slotted_nuts',
    'Square nuts': 'class_square_nuts',
    'Castle nuts': 'class_castle_nuts',
    'Locknuts': 'class_locknuts',
}
STORED_NAMES = list(MCB_CLASSES.values())


class Refusal(Exception):
    """A precondition the user must resolve (unmatched shapes, existing sidecar, ...)."""


def class_from_source(source):
    """Builder ``source`` attr -> MCB class folder name, or None.

    ``'.../dataset_org_norm/train/Hexagonal nuts/00012345.obj' -> 'Hexagonal nuts'``:
    the immediate parent directory name, whatever comes above it (train/test).
    """
    if source is None:
        return None
    if isinstance(source, bytes):
        source = source.decode('utf-8', 'replace')
    parts = str(source).replace('\\', '/').rstrip('/').split('/')
    return parts[-2] if len(parts) >= 2 else None


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
            entries.append((idx, source, class_from_source(source)))
        exists = SIDECAR_DATASET in h5
    return num_shapes, base_names, entries, exists


def build_matrix(entries):
    k = len(STORED_NAMES)
    matrix = np.full((len(entries), k), np.nan, dtype=np.float32)
    unmatched, unknown_class = [], []
    counts = {folder: 0 for folder in MCB_CLASSES}
    for idx, source, folder in entries:
        if folder is None:
            unmatched.append((idx, source, folder))
            continue
        if folder not in MCB_CLASSES:
            unknown_class.append((idx, source, folder))
            continue
        row = np.zeros(k, dtype=np.float32)
        row[STORED_NAMES.index(MCB_CLASSES[folder])] = 1.0
        matrix[idx] = row
        counts[folder] += 1
    return matrix, unmatched, unknown_class, counts


def print_table(counts, num_shapes):
    name_w = max(len('class'), *(len(c) for c in counts))
    print(f'Class distribution over {num_shapes} shapes:')
    print(f'{"class":<{name_w}}  {"stored_name":<24}  {"count":>6}')
    for folder, stored in MCB_CLASSES.items():
        print(f'{folder:<{name_w}}  {stored:<24}  {counts[folder]:>6d}')


def write_sidecar(h5_path, matrix, overwrite, missing_count):
    transforms = {name: 'identity' for name in STORED_NAMES}
    csv_columns = dict(zip(STORED_NAMES, MCB_CLASSES.keys()))
    with h5py.File(h5_path, 'r+') as h5:
        if SIDECAR_DATASET in h5:
            if not overwrite:
                raise Refusal(f'{h5_path} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')
            del h5[SIDECAR_DATASET]
        h5.create_dataset(SIDECAR_DATASET, data=matrix.astype(np.float32))
        h5.attrs[SIDECAR_NAMES_ATTR] = np.array(STORED_NAMES, dtype=h5py.string_dtype(encoding='utf-8'))
        h5.attrs[SIDECAR_SOURCE_ATTR] = 'MCB dataset_org_norm parent-directory name'
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
              f'({len(ds.base_cond_names)} geometric + {len(STORED_NAMES)} MCB class)')
    finally:
        ds.close()


def build_parser():
    p = argparse.ArgumentParser(
        description='Append a one-hot MCB nut-family class label to an SDFFlow HDF5 as cond_extra.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--h5', help='SDFFlow HDF5 dataset to extend in place')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--allow_missing', action='store_true')
    p.add_argument('--list_classes', action='store_true')
    return p


def run(args):
    if args.list_classes:
        print('Registered MCB classes (folder name -> stored one-hot name):')
        for folder, stored in MCB_CLASSES.items():
            print(f'  {folder!r:<20} -> {stored}')
        return EXIT_OK
    if not args.h5:
        raise Refusal('--h5 is required (or use --list_classes)')

    num_shapes, base_names, entries, exists = read_h5_shapes(args.h5)
    print(f'HDF5: {args.h5} -> {num_shapes} shapes, geometric cond_names={base_names}, '
          f'{SIDECAR_DATASET} {"present" if exists else "absent"}')
    if exists and not args.overwrite:
        if args.dry_run:
            print(f'NOTE: {SIDECAR_DATASET!r} already exists; a real run needs --overwrite')
        else:
            raise Refusal(f'{args.h5} already has {SIDECAR_DATASET!r}; pass --overwrite to replace it')

    matrix, unmatched, unknown_class, counts = build_matrix(entries)

    if unknown_class:
        print(f'Shape(s) whose source resolves to a class outside the 5-class subset ({len(unknown_class)}):')
        for idx, source, folder in unknown_class:
            print(f'  shape {idx:05d}: source={source!r} -> class={folder!r}')
        raise Refusal(f'{len(unknown_class)} shape(s) belong to an unregistered class; '
                       'the build scope changed -- update MCB_CLASSES, do not silently drop them')

    if unmatched:
        print(f'Shape(s) with no resolvable class from source ({len(unmatched)}):')
        for idx, source, _ in unmatched:
            print(f'  shape {idx:05d}: source={source!r}')
        if not args.allow_missing:
            raise Refusal(f'{len(unmatched)} shape(s) have no resolvable class; pass '
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
