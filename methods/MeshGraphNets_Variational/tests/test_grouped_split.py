"""Repeated load cases from one geometry must never cross a split."""
import h5py
import pytest
from general_modules.grouped_split import grouped_split_ids


def test_group_split_is_complete_disjoint_and_deterministic(tmp_path):
    path = tmp_path / 'loads.h5'
    ids = [str(i) for i in range(40)]
    with h5py.File(path, 'w') as f:
        for i, sid in enumerate(ids):
            f.create_group(f'data/{sid}/metadata').attrs['bracket'] = str(i // 4)
    a = grouped_split_ids(path, ids, 'bracket', .8, .1, 42)
    assert a == grouped_split_ids(path, ids, 'bracket', .8, .1, 42)
    assert sorted(sum(a, []), key=int) == ids
    groups = [{int(s) // 4 for s in part} for part in a]
    assert all(groups)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    with pytest.raises(ValueError, match='lacks metadata'):
        grouped_split_ids(path, ids, 'missing', .8, .1, 42)
