from __future__ import annotations

from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path

from .diagnostics import DiagnosticReport, Severity
from .specs import (
    MethodSpec,
    build_meshgraphnets_spec,
    build_neural_operator_spec,
    build_sdfflow_spec,
    build_transolver_spec,
    build_variational_spec,
)


@dataclass(frozen=True)
class ResolvedMethod:
    spec: MethodSpec
    model_id: str
    repository_root: Path
    entrypoint: Path


class MethodRegistry:
    def __init__(self, suite_root: Path, specs: tuple[MethodSpec, ...] | None = None) -> None:
        self.suite_root = suite_root.resolve()
        self.specs = specs or (
            build_meshgraphnets_spec(),
            build_variational_spec(),
            build_neural_operator_spec(),
            build_transolver_spec(),
            build_sdfflow_spec(),
        )
        self._by_model: dict[str, MethodSpec] = {}
        for spec in self.specs:
            for model_id in spec.model_ids:
                key = model_id.lower()
                if key in self._by_model:
                    raise ValueError(f"Duplicate registered model ID: {model_id}")
                self._by_model[key] = spec

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_model))

    def resolve(self, model_value: object, report: DiagnosticReport) -> ResolvedMethod | None:
        if model_value is None:
            report.add(
                "CFG-COMMON-001",
                Severity.ERROR,
                "Missing required config field 'model'.",
                field_name="model",
                hint=f"Set model to one of: {', '.join(self.model_ids)}",
            )
            return None
        if not isinstance(model_value, str):
            report.add(
                "ROUTE-MODEL-001",
                Severity.ERROR,
                f"model must be a single name, got {model_value!r}.",
                field_name="model",
            )
            return None
        model_id = model_value.strip().lower()
        spec = self._by_model.get(model_id)
        if spec is None:
            matches = get_close_matches(model_id, self.model_ids, n=1, cutoff=0.6)
            hint = f"Did you mean '{matches[0]}'? " if matches else ""
            hint += f"Supported models: {', '.join(self.model_ids)}"
            report.add(
                "ROUTE-MODEL-002",
                Severity.ERROR,
                f"Unknown model {model_id!r}.",
                field_name="model",
                hint=hint,
            )
            return None

        repository_root = (self.suite_root / spec.repository).resolve()
        entrypoint = (repository_root / spec.entrypoint).resolve()
        if not repository_root.is_dir():
            report.add(
                "ROUTE-INSTALL-001",
                Severity.ERROR,
                f"Registered repository is missing: {repository_root}",
            )
        if not entrypoint.is_file():
            report.add(
                "ROUTE-INSTALL-002",
                Severity.ERROR,
                f"Registered entrypoint is missing: {entrypoint}",
            )
        return ResolvedMethod(spec=spec, model_id=model_id, repository_root=repository_root, entrypoint=entrypoint)

    def validate_installations(self) -> DiagnosticReport:
        report = DiagnosticReport()
        for model_id in self.model_ids:
            self.resolve(model_id, report)
        return report
