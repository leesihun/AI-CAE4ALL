"""Generate a tiny sample tabular dataset for the MLP surrogate.

Writes the X[S,N] / Y[S,M] HDF5 contract (dataset/DATASET_FORMAT.md) that the
shipped configs/MLP/ex1 templates point at, so `--check` and a real run work
out of the box. Regenerate with:  python dataset/mlp/make_sample.py
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    here = Path(__file__).resolve().parent
    rng = np.random.default_rng(42)
    n_in, m_out, n_samples = 3, 2, 512
    weight = rng.normal(size=(n_in, m_out)).astype(np.float32)

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, n_in)).astype(np.float32)
        y = (x @ weight + 0.05 * rng.normal(size=(n, m_out))).astype(np.float32)
        return x, y

    input_names = np.array(["param_0", "param_1", "param_2"], dtype="S16")
    output_names = np.array(["qoi_0", "qoi_1"], dtype="S16")
    for stem, count in (("train", n_samples), ("infer", 64)):
        x, y = make(count)
        with h5py.File(here / f"{stem}.h5", "w") as handle:
            handle.create_dataset("X", data=x)
            handle.create_dataset("Y", data=y)
            handle.create_dataset("input_names", data=input_names)
            handle.create_dataset("output_names", data=output_names)
        print(f"wrote {here / f'{stem}.h5'}  X{ x.shape }  Y{ y.shape }")


if __name__ == "__main__":
    main()
