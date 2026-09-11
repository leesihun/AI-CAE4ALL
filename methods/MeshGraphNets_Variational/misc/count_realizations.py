"""How many realizations of each geometry does a training set hold?

    python misc/count_realizations.py ../../dataset/SAOI/saoi_train_bot.h5

WHY THIS DECIDES THE FIX
    A conditional prior p(z | g) can only learn the within-geometry spread it
    is shown. If the training file holds ONE realization per part, every
    condition has a single target and the prior learns a near point mass per
    part -- widened only by the posterior floor (posterior_min_std) and the
    aggregate MMD, which is exactly a spread that is too narrow by a roughly
    constant factor on every arm. No amount of prior retraining recovers
    variation the data never contained; the fix would be data-side.

    If parts carry many realizations, a frozen-target prior retrain is the
    right first move.

    "Part No." is identically 0 in this data, so geometry identity comes from
    the mesh itself: two samples are the same part when their reference
    coordinate block (rows 0:3) is identical.
"""
import hashlib
import sys
from collections import Counter

import h5py
import numpy as np


def main(path):
    groups = Counter()
    with h5py.File(path, 'r') as f:
        ids = list(f['data'].keys())
        for sid in ids:
            xyz = f[f'data/{sid}/nodal_data'][0:3, 0, :]
            # Hash the reference geometry after quantizing to 1e-4 length units.
            # Casting to float32 was NOT enough: a ~1e-9 export jitter sits
            # below float32 resolution but still flips the rounding of some
            # value in a 150-number block almost every time, so two exports of
            # the same mesh hashed apart. 1e-4 is far below any distance that
            # separates two different parts and far above any export noise.
            q = np.round(np.asarray(xyz, dtype=np.float64), 4)
            key = hashlib.sha1(np.ascontiguousarray(q).tobytes()).hexdigest()
            groups[key] += 1
    sizes = sorted(groups.values(), reverse=True)
    n_parts = len(sizes)
    print(f"{path}")
    print(f"  samples          : {len(ids)}")
    print(f"  distinct parts   : {n_parts}   (by reference-coordinate hash)")
    print(f"  realizations/part: min {min(sizes)}  median {int(np.median(sizes))}  max {max(sizes)}")
    hist = Counter(sizes)
    print(f"  parts with N realizations:")
    for n in sorted(hist):
        print(f"    {n:>5} realizations : {hist[n]} part(s)")
    print()
    single = sum(1 for s in sizes if s == 1)
    if single == n_parts:
        print("  -> ONE realization per part. The prior has never seen within-part "
              "variation; retraining it cannot create spread the data lacks.")
    elif single > n_parts // 2:
        print(f"  -> {single}/{n_parts} parts have a single realization. Within-part "
              f"spread is learnable only from the rest.")
    else:
        print("  -> Parts carry repeated realizations. A frozen-target prior retrain "
              "has the data it needs.")
    return 0


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
