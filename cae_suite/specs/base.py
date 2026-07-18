from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from ..diagnostics import DiagnosticReport, Severity

if TYPE_CHECKING:
    from ..config_parser import ParsedConfig


class PathKind(str, Enum):
    INPUT_FILE = "input_file"
    INPUT_DIR = "input_dir"
    OUTPUT_FILE = "output_file"
    OUTPUT_DIR = "output_dir"


@dataclass(frozen=True)
class PathRule:
    field: str
    kind: PathKind
    modes: frozenset[str] = frozenset()

    def active(self, mode: str | None) -> bool:
        return not self.modes or mode in self.modes


@dataclass
class SpecValidationContext:
    parsed: "ParsedConfig"
    mode: str | None
    model_id: str
    repository_root: Path
    report: DiagnosticReport

    @property
    def values(self) -> dict[str, Any]:
        return self.parsed.values

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        field_name: str | None = None,
        hint: str | None = None,
        promote_in_strict: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.report.add(
            code,
            severity,
            message,
            field_name=field_name,
            location=self.parsed.location(field_name) if field_name else None,
            hint=hint,
            promote_in_strict=promote_in_strict,
            details=details,
        )


Validator = Callable[[SpecValidationContext], None]


@dataclass(frozen=True)
class MethodSpec:
    spec_id: str
    display_name: str
    model_ids: tuple[str, ...]
    repository: str
    entrypoint: str
    valid_modes: tuple[str, ...]
    known_keys: frozenset[str]
    required_common: frozenset[str] = frozenset({"model", "mode", "gpu_ids"})
    required_by_mode: dict[str, frozenset[str]] = field(default_factory=dict)
    required_by_model: dict[str, frozenset[str]] = field(default_factory=dict)
    recommended_by_mode: dict[str, frozenset[str]] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    defaults_by_mode: dict[str, dict[str, Any]] = field(default_factory=dict)
    path_rules: tuple[PathRule, ...] = ()
    validators: tuple[Validator, ...] = ()
    import_modules: tuple[str, ...] = ()
    dataset_kind: str | None = None
    native_probe: bool = True

    def required_fields(self, model_id: str, mode: str | None) -> frozenset[str]:
        fields = set(self.required_common)
        if mode is not None:
            fields.update(self.required_by_mode.get(mode, ()))
        fields.update(self.required_by_model.get(model_id, ()))
        return frozenset(fields)


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def validate_common_values(ctx: SpecValidationContext) -> None:
    values = ctx.values
    gpu_ids = values.get("gpu_ids")
    if gpu_ids is not None:
        ids = as_list(gpu_ids)
        if not ids or any(integer(v) is None for v in ids):
            ctx.add(
                "CFG-GPU-001",
                Severity.ERROR,
                "gpu_ids must be an integer or a comma-separated list of integers.",
                field_name="gpu_ids",
            )
        else:
            parsed_ids = [integer(v) for v in ids]
            assert all(v is not None for v in parsed_ids)
            int_ids = [int(v) for v in parsed_ids]
            if any(v < -1 for v in int_ids):
                ctx.add(
                    "CFG-GPU-002",
                    Severity.ERROR,
                    "gpu_ids may contain nonnegative CUDA IDs or the CPU sentinel -1.",
                    field_name="gpu_ids",
                )
            if -1 in int_ids and len(int_ids) > 1:
                ctx.add(
                    "CFG-GPU-003",
                    Severity.ERROR,
                    "The CPU sentinel -1 cannot be combined with CUDA device IDs.",
                    field_name="gpu_ids",
                )
            if len(set(int_ids)) != len(int_ids):
                ctx.add(
                    "CFG-GPU-004",
                    Severity.ERROR,
                    "gpu_ids contains duplicate device IDs.",
                    field_name="gpu_ids",
                )

    positive_ints = (
        "input_var",
        "output_var",
        "latent_dim",
        "training_epochs",
        "batch_size",
        "num_workers",
        "grad_accum_steps",
        "infer_timesteps",
    )
    allow_zero = {"num_workers"}
    for name in positive_ints:
        if name not in values:
            continue
        parsed = integer(values[name])
        minimum = 0 if name in allow_zero else 1
        if parsed is None or parsed < minimum:
            ctx.add(
                "CFG-TYPE-INT",
                Severity.ERROR,
                f"{name} must be an integer >= {minimum}; got {values[name]!r}.",
                field_name=name,
            )

    positive_numbers = ("learningr", "weight_decay", "ema_decay")
    for name in positive_numbers:
        if name not in values:
            continue
        parsed = numeric(values[name])
        minimum = 0.0 if name == "weight_decay" else 0.0
        if parsed is None or parsed < minimum or (name != "weight_decay" and parsed == 0):
            ctx.add(
                "CFG-TYPE-NUMBER",
                Severity.ERROR,
                f"{name} must be a valid positive number; got {values[name]!r}.",
                field_name=name,
            )

    output_var = integer(values.get("output_var"))
    weights = values.get("feature_loss_weights")
    if output_var is not None and weights is not None:
        weight_list = as_list(weights)
        if len(weight_list) != output_var:
            ctx.add(
                "CFG-WEIGHTS-001",
                Severity.ERROR,
                f"feature_loss_weights has {len(weight_list)} entries; output_var is {output_var}.",
                field_name="feature_loss_weights",
            )
        elif any(numeric(v) is None or numeric(v) < 0 for v in weight_list):
            ctx.add(
                "CFG-WEIGHTS-002",
                Severity.ERROR,
                "feature_loss_weights must contain nonnegative numbers.",
                field_name="feature_loss_weights",
            )


def validate_positive_fields(ctx: SpecValidationContext, names: tuple[str, ...], code: str) -> None:
    for name in names:
        if name not in ctx.values:
            continue
        value = numeric(ctx.values[name])
        if value is None or value <= 0:
            ctx.add(code, Severity.ERROR, f"{name} must be > 0; got {ctx.values[name]!r}.", field_name=name)


def validate_nonnegative_int_fields(ctx: SpecValidationContext, names: tuple[str, ...], code: str) -> None:
    for name in names:
        if name not in ctx.values:
            continue
        value = integer(ctx.values[name])
        if value is None or value < 0:
            ctx.add(code, Severity.ERROR, f"{name} must be a nonnegative integer; got {ctx.values[name]!r}.", field_name=name)
