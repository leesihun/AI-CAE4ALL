"""End-to-end smoke test: synthetic X/Y -> train -> checkpoint -> infer -> [S,M]."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

# Make the MLP repo root importable (mlp package + MLP_main entrypoint).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import MLP_main  # noqa: E402

N_IN, M_OUT, N_SAMPLES = 3, 2, 200


def _write_xy(path: Path, x: np.ndarray, y: np.ndarray | None) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x.astype(np.float32))
        if y is not None:
            handle.create_dataset("Y", data=y.astype(np.float32))


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(N_SAMPLES, N_IN)).astype(np.float32)
    weight = rng.normal(size=(N_IN, M_OUT)).astype(np.float32)
    y = x @ weight + 0.05 * rng.normal(size=(N_SAMPLES, M_OUT)).astype(np.float32)
    return x, y


def _config(tmp: Path, mode: str, ckpt: Path, data: Path, out_dir: Path) -> Path:
    lines = [
        "model mlp",
        f"mode {mode}",
        "gpu_ids -1",
        f"modelpath {ckpt.as_posix()}",
        "input_var 3",
        "output_var 2",
        "hidden_layers 32, 32",
        "activation gelu",
        "input_normalization standard",
        "output_normalization standard",
    ]
    if mode == "train":
        lines += [
            f"dataset_dir {data.as_posix()}",
            "training_epochs 40",
            "batch_size 32",
            "learningr 0.005",
            "val_interval 5",
            "use_ema True",
        ]
    else:
        lines += [
            f"infer_dataset {data.as_posix()}",
            f"inference_output_dir {out_dir.as_posix()}",
        ]
    path = tmp / f"config_{mode}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_train_then_infer(tmp_path: Path) -> None:
    x, y = _synthetic()
    train_h5 = tmp_path / "train.h5"
    infer_h5 = tmp_path / "infer.h5"
    _write_xy(train_h5, x, y)
    _write_xy(infer_h5, x, y)  # include Y so metrics are exercised

    ckpt = tmp_path / "mlp.pth"
    out_dir = tmp_path / "predictions"

    assert MLP_main.main(["--config", str(_config(tmp_path, "train", ckpt, train_h5, out_dir))]) == 0
    assert ckpt.is_file(), "training did not write a checkpoint"

    assert MLP_main.main(["--config", str(_config(tmp_path, "inference", ckpt, infer_h5, out_dir))]) == 0
    pred_path = out_dir / "predictions.h5"
    assert pred_path.is_file(), "inference did not write predictions"

    with h5py.File(pred_path, "r") as handle:
        y_pred = np.asarray(handle["Y_pred"][...])
    assert y_pred.shape == (N_SAMPLES, M_OUT)

    # A linear target learned for 40 epochs should beat the mean predictor.
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))
    assert rmse < y.std(), f"model no better than the mean predictor (rmse={rmse:.3f}, std={y.std():.3f})"
