"""Training loop for the MLP surrogate."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import Params
from .data import Normalizer, load_xy, split_indices
from .model import MLP

_LOSSES = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": nn.HuberLoss}


def _device(gpu_ids: list[int]) -> torch.device:
    first = gpu_ids[0] if gpu_ids else -1
    if first is not None and first >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{first}")
    return torch.device("cpu")


def _make_loader(x: np.ndarray, y: np.ndarray, params: Params, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    kwargs: dict = {"batch_size": params.batch_size, "shuffle": shuffle, "num_workers": params.num_workers}
    if params.num_workers > 0:
        kwargs["prefetch_factor"] = params.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _scheduler(optimizer: torch.optim.Optimizer, warmup: int, total: int):
    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    if loader is None or len(loader.dataset) == 0:
        return float("nan")
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            total += loss_fn(model(xb), yb).item() * xb.shape[0]
            count += xb.shape[0]
    return total / max(1, count)


def train(params: Params, config_path: str) -> int:
    torch.manual_seed(params.split_seed)
    device = _device(params.gpu_ids)
    print(f"MLP Surrogate | device={device} | epochs={params.training_epochs} | batch={params.batch_size}")

    x, y = load_xy(params.dataset_dir, require_y=True)
    if x.shape[1] != params.input_var:
        raise SystemExit(f"X has {x.shape[1]} columns but input_var={params.input_var}.")
    if y.shape[1] != params.output_var:
        raise SystemExit(f"Y has {y.shape[1]} columns but output_var={params.output_var}.")

    train_idx, val_idx, _test_idx = split_indices(x.shape[0], params.split_seed)
    x_norm = Normalizer.fit(x[train_idx], params.input_normalization)
    y_norm = Normalizer.fit(y[train_idx], params.output_normalization)
    xt, yt = x_norm.transform(x[train_idx]), y_norm.transform(y[train_idx])
    train_loader = _make_loader(xt, yt, params, shuffle=True)
    val_loader = None
    if len(val_idx) > 0:
        val_loader = _make_loader(x_norm.transform(x[val_idx]), y_norm.transform(y[val_idx]), params, shuffle=False)
    print(f"Samples: train={len(train_idx)} val={len(val_idx)} | N={params.input_var} -> M={params.output_var}")

    model = MLP(
        params.input_var, params.output_var, params.hidden_layers,
        activation=params.activation, dropout=params.dropout,
        norm=params.norm, output_activation=params.output_activation,
    ).to(device)
    if params.use_compile:
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.learningr, weight_decay=params.weight_decay)
    scheduler = _scheduler(optimizer, params.warmup_epochs, params.training_epochs)
    loss_fn = _LOSSES[params.loss]().to(device)
    use_amp = params.use_amp and device.type == "cuda"

    ema: dict[str, torch.Tensor] | None = None
    if params.use_ema:
        ema = {name: param.detach().clone() for name, param in model.state_dict().items()}

    best_val = float("inf")
    global_step = 0
    arch = {
        "model": "mlp", "input_var": params.input_var, "output_var": params.output_var,
        "hidden_layers": list(params.hidden_layers), "activation": params.activation,
        "dropout": params.dropout, "norm": params.norm, "output_activation": params.output_activation,
    }

    def save(tag: str) -> None:
        checkpoint = {
            "model_state": model.state_dict(),
            "selected_model": "mlp",
            "config": arch,
            "normalization": {"input": x_norm.to_dict(), "output": y_norm.to_dict()},
        }
        if ema is not None:
            checkpoint["ema_state"] = ema
        Path(params.modelpath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, params.modelpath)

    for epoch in range(params.training_epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(xb), yb)
            loss.backward()
            if params.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
            optimizer.step()
            running += loss.item() * xb.shape[0]
            if ema is not None:
                global_step += 1
                # Decay warmup (timm/TF ModelEma style): a low effective decay early
                # lets EMA track fast, so it stays useful on short training runs
                # instead of retaining a chunk of the random initialization.
                decay = min(params.ema_decay, (1 + global_step) / (10 + global_step))
                for name, param in model.state_dict().items():
                    if ema[name].is_floating_point():
                        ema[name].mul_(decay).add_(param.detach(), alpha=1 - decay)
                    else:
                        ema[name].copy_(param)
        scheduler.step()
        train_loss = running / max(1, len(train_idx))

        if val_loader is not None and (epoch + 1) % params.val_interval == 0:
            val_loss = _evaluate(model, val_loader, loss_fn, device)
            print(f"epoch {epoch + 1:4d} | train {train_loss:.6f} | val {val_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")
            if val_loss < best_val:
                best_val = val_loss
                save("best")
        elif (epoch + 1) % max(1, params.val_interval) == 0:
            print(f"epoch {epoch + 1:4d} | train {train_loss:.6f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if params.checkpoint_interval > 0 and (epoch + 1) % params.checkpoint_interval == 0:
            save("periodic")

    # Always leave a usable checkpoint (covers no-val and val-never-improved runs).
    if best_val == float("inf"):
        save("final")
    print(f"Done. Checkpoint: {params.modelpath}")
    return 0
