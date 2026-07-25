"""Tabular dataset IO and feature normalization for the MLP surrogate.

The canonical dataset is a tabular HDF5 with two float datasets:
    X  [num_samples, N_in]   input parameters
    Y  [num_samples, M_out]  output quantities of interest
Optional string datasets ``input_names`` / ``output_names`` label the columns.
There are no meshes, edges, or timesteps here (see dataset/DATASET_FORMAT.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def load_xy(path: str, *, require_y: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Read X (and optionally Y) from a tabular HDF5 as float32 arrays."""
    import h5py

    with h5py.File(path, "r") as handle:
        if "X" not in handle:
            raise SystemExit(f"{path}: missing root dataset 'X' (tabular X/Y contract).")
        x = np.asarray(handle["X"][...], dtype=np.float32)
        if x.ndim != 2:
            raise SystemExit(f"{path}: X must be rank 2 [S,N]; got shape {x.shape}.")
        y = None
        if "Y" in handle:
            y = np.asarray(handle["Y"][...], dtype=np.float32)
            if y.ndim != 2 or y.shape[0] != x.shape[0]:
                raise SystemExit(f"{path}: Y must be rank 2 [S,M] sharing X's sample axis; got {y.shape}.")
        elif require_y:
            raise SystemExit(f"{path}: missing root dataset 'Y' (required for training).")
    return x, y


@dataclass
class Normalizer:
    """Per-column standardization. mode: standard | minmax | none."""

    mode: str
    a: np.ndarray  # standard: mean;   minmax: min;   none: zeros
    b: np.ndarray  # standard: std;    minmax: range; none: ones

    @classmethod
    def fit(cls, array: np.ndarray, mode: str) -> "Normalizer":
        mode = mode.lower()
        cols = array.shape[1]
        if mode == "standard":
            a = array.mean(axis=0)
            b = array.std(axis=0)
        elif mode == "minmax":
            a = array.min(axis=0)
            b = array.max(axis=0) - a
        elif mode == "none":
            a = np.zeros(cols, dtype=np.float64)
            b = np.ones(cols, dtype=np.float64)
        else:
            raise SystemExit(f"Unknown normalization mode {mode!r}.")
        b = np.where(np.abs(b) < 1e-8, 1.0, b)  # guard constant columns
        return cls(mode=mode, a=a.astype(np.float64), b=b.astype(np.float64))

    def transform(self, array: np.ndarray) -> np.ndarray:
        return ((array - self.a) / self.b).astype(np.float32)

    def inverse(self, array: np.ndarray) -> np.ndarray:
        return (array * self.b + self.a).astype(np.float32)

    def to_dict(self) -> dict:
        # Plain lists keep the checkpoint loadable with torch weights_only=True.
        return {"mode": self.mode, "a": self.a.tolist(), "b": self.b.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Normalizer":
        return cls(mode=data["mode"], a=np.asarray(data["a"], dtype=np.float64), b=np.asarray(data["b"], dtype=np.float64))


def split_indices(num_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic 80/10/10 train/val/test split. Tiny sets fall back to all-train."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_samples)
    if num_samples < 10:
        return order, order[:0], order[:0]
    n_val = max(1, int(round(0.1 * num_samples)))
    n_test = max(1, int(round(0.1 * num_samples)))
    n_train = num_samples - n_val - n_test
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
