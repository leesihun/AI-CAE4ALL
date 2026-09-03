"""Flat ``key value`` config parsing for the MLP surrogate entrypoint.

Faithful to the repository's native config style (shared by the ML methods and
mirrored in ``cae_suite/config_parser.py``): ``%`` starts a full-line comment,
``#`` starts an inline comment, keys are lowercased, and a value is the rest of
the line. Self-contained so the entrypoint has no dependency on ``cae_suite``
(it runs in the method environment).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def load_config(path: str) -> dict[str, str]:
    """Parse a flat config file into ``{lowercased_key: raw_value_string}``."""
    values: dict[str, str] = {}
    # utf-8-sig tolerates (strips) a BOM; the launcher treats a BOM as a hard error,
    # so a config that passes preflight will not have one here anyway.
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            body = stripped.split("#", 1)[0].strip()
            if not body:
                continue
            parts = body.split(None, 1)
            if len(parts) != 2 or not parts[1].strip():
                continue
            values[parts[0].strip().lower()] = parts[1].strip()
    return values


def _as_tokens(value: str) -> list[str]:
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value.split()


def get_str(cfg: dict[str, str], key: str, default: str) -> str:
    return str(cfg.get(key, default)).strip().lower()


def get_int(cfg: dict[str, str], key: str, default: int) -> int:
    return int(cfg[key]) if key in cfg else default


def get_float(cfg: dict[str, str], key: str, default: float) -> float:
    return float(cfg[key]) if key in cfg else default


def get_bool(cfg: dict[str, str], key: str, default: bool) -> bool:
    if key not in cfg:
        return default
    return str(cfg[key]).strip().lower() in {"true", "1", "yes"}


def get_int_list(cfg: dict[str, str], key: str, default: list[int]) -> list[int]:
    if key not in cfg:
        return list(default)
    return [int(token) for token in _as_tokens(cfg[key])]


@dataclass
class Params:
    """Typed view of an MLP config, applying the same defaults as the spec."""

    mode: str = "train"
    gpu_ids: list[int] = field(default_factory=lambda: [-1])
    # IO
    modelpath: str = ""
    dataset_dir: str = ""
    infer_dataset: str = ""
    inference_output_dir: str = "../../output/mlp/predictions"
    log_file_dir: str = ""
    split_seed: int = 42
    # Problem shape
    input_var: int = 0
    output_var: int = 0
    # Architecture
    hidden_layers: list[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "gelu"
    dropout: float = 0.0
    norm: str = "none"
    input_normalization: str = "standard"
    output_normalization: str = "standard"
    output_activation: str = "none"
    # Optimization
    loss: str = "mse"
    training_epochs: int = 200
    batch_size: int = 32
    learningr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    num_workers: int = 0
    prefetch_factor: int = 4
    max_grad_norm: float = 3.0
    # Runtime / EMA / eval
    use_amp: bool = False
    use_ema: bool = False
    ema_decay: float = 0.999
    use_compile: bool = False
    val_interval: int = 5
    checkpoint_interval: int = 0


def params_from_config(cfg: dict[str, str]) -> Params:
    return Params(
        mode=get_str(cfg, "mode", "train"),
        gpu_ids=get_int_list(cfg, "gpu_ids", [-1]),
        modelpath=cfg.get("modelpath", ""),
        dataset_dir=cfg.get("dataset_dir", ""),
        infer_dataset=cfg.get("infer_dataset", ""),
        inference_output_dir=cfg.get("inference_output_dir", "../../output/mlp/predictions"),
        log_file_dir=cfg.get("log_file_dir", ""),
        split_seed=get_int(cfg, "split_seed", 42),
        input_var=get_int(cfg, "input_var", 0),
        output_var=get_int(cfg, "output_var", 0),
        hidden_layers=get_int_list(cfg, "hidden_layers", [256, 256, 128]),
        activation=get_str(cfg, "activation", "gelu"),
        dropout=get_float(cfg, "dropout", 0.0),
        norm=get_str(cfg, "norm", "none"),
        input_normalization=get_str(cfg, "input_normalization", "standard"),
        output_normalization=get_str(cfg, "output_normalization", "standard"),
        output_activation=get_str(cfg, "output_activation", "none"),
        loss=get_str(cfg, "loss", "mse"),
        training_epochs=get_int(cfg, "training_epochs", 200),
        batch_size=get_int(cfg, "batch_size", 32),
        learningr=get_float(cfg, "learningr", 1e-3),
        weight_decay=get_float(cfg, "weight_decay", 1e-4),
        warmup_epochs=get_int(cfg, "warmup_epochs", 3),
        num_workers=get_int(cfg, "num_workers", 0),
        prefetch_factor=get_int(cfg, "prefetch_factor", 4),
        max_grad_norm=get_float(cfg, "max_grad_norm", 3.0),
        use_amp=get_bool(cfg, "use_amp", False),
        use_ema=get_bool(cfg, "use_ema", False),
        ema_decay=get_float(cfg, "ema_decay", 0.999),
        use_compile=get_bool(cfg, "use_compile", False),
        val_interval=get_int(cfg, "val_interval", 5),
        checkpoint_interval=get_int(cfg, "checkpoint_interval", 0),
    )


# Closed choice sets, mirrored from cae_suite/specs/mlp.py. Without these an
# unknown value reaches model construction as a bare KeyError on the activation
# or normalization lookup table; keep the two lists in sync.
_CHOICES = {
    "activation": {"relu", "gelu", "silu", "tanh"},
    "norm": {"none", "batch", "layer"},
    "input_normalization": {"standard", "minmax", "none"},
    "output_normalization": {"standard", "minmax", "none"},
    "output_activation": {"none", "relu", "sigmoid", "tanh", "softplus"},
    "loss": {"mse", "mae", "huber"},
}


def validate(params: Params) -> None:
    """Cheap native self-check; the launcher preflight is the primary gate."""
    if params.input_var < 1 or params.output_var < 1:
        raise SystemExit("input_var and output_var must be positive integers.")
    if not params.hidden_layers or any(width < 1 for width in params.hidden_layers):
        raise SystemExit("hidden_layers must be one or more positive integers.")
    if not (0.0 <= params.dropout < 1.0):
        raise SystemExit("dropout must be in [0, 1).")
    for name, allowed in _CHOICES.items():
        value = getattr(params, name)
        if value not in allowed:
            raise SystemExit(
                f"{name} must be one of {sorted(allowed)}; got {value!r}."
            )
