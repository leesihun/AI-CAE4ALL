"""Lifecycle of the on-disk multiscale hierarchy cache.

The cache is per-run: built at dataset construction and deleted once training
finishes, because its signature pins the source HDF5's size+mtime and training
rewrites that file with the train-derived normalization stats. These tests pin
the delete/prune rules — especially the guards that stop one job from removing
a cache a concurrent job is still streaming from.
"""
import os
import time
from pathlib import Path
import sys

import h5py
import pytest


MGN_ROOT = Path(__file__).resolve().parents[1]
if str(MGN_ROOT) not in sys.path:
    sys.path.insert(0, str(MGN_ROOT))

from general_modules import multiscale_cache as mc  # noqa: E402


def _make_cache(path: Path) -> Path:
    with h5py.File(path, 'w') as f:
        f.attrs['signature'] = '{}'
        f.attrs['format_version'] = mc.FORMAT_VERSION
        f.create_dataset('sample_ids', data=[0, 1, 2])
    return path


def _backdate(path: Path, seconds: float = 3600.0) -> None:
    """Age a file past _prune_siblings' grace window."""
    t = time.time() - seconds
    os.utime(path, (t, t))


def test_prune_removes_leftovers_but_spares_the_keeper(tmp_path):
    keep = _make_cache(tmp_path / 'ds.mscache.keepkeep.h5')
    leftover = _make_cache(tmp_path / 'ds.mscache.aaaaaaaa.h5')
    _backdate(keep)
    _backdate(leftover)

    mc._prune_siblings(str(keep))

    assert not leftover.exists()
    assert keep.exists()


def test_prune_spares_a_recent_build_and_other_datasets(tmp_path):
    """A cache just renamed into place has no reader open yet, so only the grace
    window protects it; another dataset's cache must never match the glob."""
    keep = _make_cache(tmp_path / 'ds.mscache.keepkeep.h5')
    fresh = _make_cache(tmp_path / 'ds.mscache.ffffffff.h5')       # left at "now"
    other = _make_cache(tmp_path / 'otherds.mscache.cccccccc.h5')
    in_progress = tmp_path / 'ds.mscache.dddddddd.h5.tmp.999'
    in_progress.write_bytes(b'')
    _backdate(keep)
    _backdate(other)
    _backdate(in_progress)

    mc._prune_siblings(str(keep))

    assert fresh.exists(), 'recent build must survive the grace window'
    assert other.exists(), "another dataset's cache must not be globbed"
    assert in_progress.exists(), 'a .tmp build must not be globbed'


def test_release_skips_a_cache_that_is_still_open(tmp_path):
    """Windows refuses the delete outright; POSIX is covered by the shared flock
    HierarchyCacheReader takes. Either way the file must survive."""
    cache = _make_cache(tmp_path / 'ds.mscache.keepkeep.h5')
    reader = mc.HierarchyCacheReader(str(cache))
    reader.has(0)  # forces the handle (and the POSIX lock) open
    try:
        mc.release_cache(str(cache), {})
        assert cache.exists()
    finally:
        reader.close()

    mc.release_cache(str(cache), {})
    assert not cache.exists()


@pytest.mark.parametrize('keep_flag, still_there', [(True, True), (False, False)])
def test_hierarchy_cache_keep_opt_out(tmp_path, keep_flag, still_there):
    cache = _make_cache(tmp_path / 'ds.mscache.eeeeeeee.h5')
    mc.release_cache(str(cache), {'hierarchy_cache_keep': keep_flag})
    assert cache.exists() is still_there


def test_release_is_a_noop_on_missing_paths(tmp_path):
    mc.release_cache('', {})
    mc.release_cache(str(tmp_path / 'does.mscache.notexist.h5'), {})
    assert mc._try_delete(str(tmp_path / 'does.mscache.notexist.h5')) is True
