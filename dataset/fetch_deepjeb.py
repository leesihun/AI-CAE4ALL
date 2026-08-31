"""Fetch a subset of the DeepJEB FieldMesh HDF5 files from the public Drive folder.

DeepJEB ships one ~50 MB HDF5 per bracket (270k nodes, tet10 cells, four load
cases plus two mode shapes), so the full 2,138-bracket release is ~107 GB. That
is far more than a graph surrogate on this hardware needs, and Google Drive
throttles anonymous bulk pulls, so this fetches a *stratified subset* drawn from
the official split files and is safe to re-run: existing non-empty files are
skipped, so an interrupted pull resumes.

  python dataset/fetch_deepjeb.py --train 300 --test 120

Dataset: Hong, Kwon, Shin, Park & Kang, "DeepJEB: 3D Deep Learning-Based
Synthetic Jet Engine Bracket Dataset", ASME JMD 147(4) 041703 (2025).
Licensed ODC-By v1.0 -- attribution required, redistribution permitted.
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_ROOT = os.environ.get('DEEPJEB_RAW', 'D:/CAE_datasets_raw/deepjeb')
FIELDMESH_DIR = os.path.join(RAW_ROOT, 'FieldMesh')
INDEX = os.path.join(RAW_ROOT, 'fieldmesh_index.json')
MIN_BYTES = 1 << 20          # anything smaller is a Drive error page, not an HDF5

_print_lock = threading.Lock()


def log(message):
    with _print_lock:
        print(message, flush=True)


def load_index():
    with open(INDEX, encoding='utf-8') as fh:
        return {e['path'].rsplit('/', 1)[-1][:-3]: e['id'] for e in json.load(fh)}


def load_split(name):
    path = os.path.join(RAW_ROOT, 'Metadata', f'{name}_split_random.json')
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def pick(ids, count, seed):
    ids = sorted(ids)
    if count <= 0 or count >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, count))


def fetch_one(item, file_id, retries=3):
    """Download one bracket. Returns (item, bytes, status)."""
    import gdown

    out = os.path.join(FIELDMESH_DIR, f'{item}.h5')
    if os.path.exists(out) and os.path.getsize(out) >= MIN_BYTES:
        return item, os.path.getsize(out), 'have'
    for attempt in range(retries):
        try:
            gdown.download(id=file_id, output=out, quiet=True)
            size = os.path.getsize(out) if os.path.exists(out) else 0
            if size >= MIN_BYTES:
                return item, size, 'ok'
            # A quota or permission page lands as a few KB of HTML.
            if os.path.exists(out):
                os.remove(out)
        except Exception as exc:                      # network / quota / parse
            if attempt == retries - 1:
                return item, 0, f'fail: {type(exc).__name__}: {exc}'
        time.sleep(3 * (attempt + 1))
    return item, 0, 'fail: too small (quota page?)'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=300, help='brackets from the train split')
    parser.add_argument('--test', type=int, default=120, help='brackets from the test split')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--manifest', default=os.path.join(RAW_ROOT, 'subset_manifest.json'))
    args = parser.parse_args(argv)

    os.makedirs(FIELDMESH_DIR, exist_ok=True)
    index = load_index()
    train = [i for i in pick(load_split('train'), args.train, args.seed) if i in index]
    test = [i for i in pick(load_split('test'), args.test, args.seed) if i in index]
    wanted = [(i, 'train') for i in train] + [(i, 'test') for i in test]
    log(f'requesting {len(train)} train + {len(test)} test = {len(wanted)} brackets')

    with open(args.manifest, 'w', encoding='utf-8') as fh:
        json.dump({'train': train, 'test': test, 'seed': args.seed}, fh, indent=1)

    done = failed = have = 0
    total_bytes = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, item, index[item]): item for item, _ in wanted}
        for n, future in enumerate(as_completed(futures), 1):
            item, size, status = future.result()
            total_bytes += size
            if status == 'ok':
                done += 1
            elif status == 'have':
                have += 1
            else:
                failed += 1
                log(f'  !! {item}: {status}')
            if n % 10 == 0 or n == len(futures):
                rate = total_bytes / max(time.time() - started, 1e-9) / 1e6
                log(f'  {n}/{len(futures)}  new={done} cached={have} failed={failed}  '
                    f'{total_bytes / 1e9:.1f} GB  {rate:.1f} MB/s')

    log(f'\ndownloaded {done}, already had {have}, failed {failed}')
    log(f'total {total_bytes / 1e9:.2f} GB in {FIELDMESH_DIR}')
    return 1 if failed and not (done + have) else 0


if __name__ == '__main__':
    sys.exit(main())
