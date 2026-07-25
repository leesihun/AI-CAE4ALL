"""Inference for the MLP surrogate: predict M outputs from N inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import Params
from .data import Normalizer, load_xy
from .model import build_model
from .train import _device


def _load_checkpoint(path: str):
    # We authored this file, so weights_only=False is safe here (the launcher's
    # checkpoint_probe uses weights_only=True separately for untrusted inspection).
    return torch.load(path, map_location="cpu", weights_only=False)


def infer(params: Params, config_path: str) -> int:
    device = _device(params.gpu_ids)
    checkpoint = _load_checkpoint(params.modelpath)
    arch = checkpoint["config"]
    model = build_model(arch).to(device)
    state = checkpoint.get("ema_state") or checkpoint["model_state"]
    model.load_state_dict(state)
    model.eval()

    x_norm = Normalizer.from_dict(checkpoint["normalization"]["input"])
    y_norm = Normalizer.from_dict(checkpoint["normalization"]["output"])

    x, y = load_xy(params.infer_dataset, require_y=False)
    if x.shape[1] != int(arch["input_var"]):
        raise SystemExit(f"infer_dataset X has {x.shape[1]} columns but the checkpoint expects {arch['input_var']}.")

    with torch.no_grad():
        xb = torch.from_numpy(x_norm.transform(x)).to(device)
        y_pred = y_norm.inverse(model(xb).cpu().numpy())

    out_dir = Path(params.inference_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.h5"
    import h5py

    with h5py.File(out_path, "w") as handle:
        handle.create_dataset("X", data=x.astype(np.float32))
        handle.create_dataset("Y_pred", data=y_pred.astype(np.float32))
        if y is not None:
            handle.create_dataset("Y_true", data=y.astype(np.float32))

    print(f"MLP Surrogate inference | {x.shape[0]} samples | {arch['input_var']} -> {arch['output_var']}")
    if y is not None:
        mae = np.mean(np.abs(y_pred - y), axis=0)
        rmse = np.sqrt(np.mean((y_pred - y) ** 2, axis=0))
        print(f"per-output MAE : {np.array2string(mae, precision=4)}")
        print(f"per-output RMSE: {np.array2string(rmse, precision=4)}")
    print(f"Predictions: {out_path}")
    return 0
