"""Periodic training visualization for the MLP surrogate.

The other methods in the suite render a field on its mesh; this one is tabular
(N scalar inputs -> M scalar outputs), so the equivalent picture is a parity
plot: predicted against true, one panel per output column, on the held-out
split and in physical units. A surrogate that has learned the mapping puts its
points on the diagonal; a surrogate that has learned the output mean puts them
on a horizontal band, which is the failure this plot is here to make obvious.

matplotlib is imported lazily and a missing install degrades to a printed note
instead of killing a training run.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_MISSING_BACKEND_WARNED = False
_MAX_PANELS = 9


def _load_backend():
    """Import matplotlib with the Agg backend, or return None once, quietly."""
    global _MISSING_BACKEND_WARNED
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover - environment dependent
        if not _MISSING_BACKEND_WARNED:
            _MISSING_BACKEND_WARNED = True
            print(f"  [viz] matplotlib unavailable ({exc}); skipping parity plots. "
                  f"pip install matplotlib to enable them.")
        return None


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination; nan for a constant target (undefined)."""
    denominator = float(((true - true.mean()) ** 2).sum())
    if denominator < 1e-30:
        return float("nan")
    return 1.0 - float(((true - pred) ** 2).sum()) / denominator


def parity_plot(y_true, y_pred, path, *, epoch=None, split="val", labels=None,
                dpi=140, max_points=5000, seed=0):
    """Write a parity panel per output column. Returns the path, or None.

    y_true / y_pred: [S, M] arrays in physical (denormalized) units.
    """
    plt = _load_backend()
    if plt is None:
        return None

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim != 2 or y_true.shape != y_pred.shape or y_true.size == 0:
        print(f"  [viz] skipping parity plot: shapes {y_true.shape} vs {y_pred.shape}")
        return None

    num_samples, num_outputs = y_true.shape
    if max_points > 0 and num_samples > max_points:
        picked = np.random.default_rng(seed).choice(num_samples, max_points, replace=False)
        y_true, y_pred = y_true[picked], y_pred[picked]

    shown = min(num_outputs, _MAX_PANELS)
    cols = min(3, shown)
    rows = math.ceil(shown / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.0 * rows),
                             dpi=dpi, squeeze=False, facecolor="white")

    for index in range(rows * cols):
        ax = axes[index // cols][index % cols]
        if index >= shown:
            ax.set_axis_off()
            continue
        true_col, pred_col = y_true[:, index], y_pred[:, index]
        ax.scatter(true_col, pred_col, s=7, alpha=0.45, linewidths=0, color="#3B82C4")

        # One shared span for both axes: a parity plot with independently
        # scaled axes can make a badly biased fit look like a clean diagonal.
        low = float(min(true_col.min(), pred_col.min()))
        high = float(max(true_col.max(), pred_col.max()))
        pad = 0.04 * (high - low) if high > low else 1.0
        low, high = low - pad, high + pad
        ax.plot([low, high], [low, high], color="#111111", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")

        rmse = float(np.sqrt(((pred_col - true_col) ** 2).mean()))
        name = labels[index] if labels is not None and index < len(labels) else f"output {index}"
        ax.set_title(f"{name}\nR2={_r2(true_col, pred_col):.4f}  RMSE={rmse:.4g}", fontsize=10)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.grid(True, alpha=0.25)

    header = f"MLP parity -- {split} split, {num_samples} samples"
    if epoch is not None:
        header += f", epoch {epoch}"
    if num_outputs > shown:
        header += f"  (first {shown} of {num_outputs} outputs)"
    fig.suptitle(header, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)
