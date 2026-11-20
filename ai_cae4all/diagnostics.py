from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Iterable


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTICE = "notice"


@dataclass(frozen=True)
class SourceLocation:
    path: Path
    line: int
    column: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "line": self.line, "column": self.column}


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Severity
    message: str
    field: str | None = None
    location: SourceLocation | None = None
    hint: str | None = None
    promote_in_strict: bool = False
    details: dict[str, Any] = dc_field(default_factory=dict)

    def effective_severity(self, strict: bool) -> Severity:
        if strict and self.promote_in_strict and self.severity is Severity.WARNING:
            return Severity.ERROR
        return self.severity

    def to_dict(self, strict: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "severity": self.effective_severity(strict).value,
            "original_severity": self.severity.value,
            "message": self.message,
        }
        if self.field is not None:
            result["field"] = self.field
        if self.location is not None:
            result["location"] = self.location.to_dict()
        if self.hint is not None:
            result["hint"] = self.hint
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class DiagnosticReport:
    diagnostics: list[Diagnostic] = dc_field(default_factory=list)

    def add(
        self,
        code: str,
        severity: Severity,
        message: str,
        *,
        field_name: str | None = None,
        location: SourceLocation | None = None,
        hint: str | None = None,
        promote_in_strict: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                code=code,
                severity=severity,
                message=message,
                field=field_name,
                location=location,
                hint=hint,
                promote_in_strict=promote_in_strict,
                details=details or {},
            )
        )

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics.extend(diagnostics)

    def count(self, severity: Severity, *, strict: bool = False) -> int:
        return sum(d.effective_severity(strict) is severity for d in self.diagnostics)

    def has_errors(self, *, strict: bool = False) -> bool:
        return self.count(Severity.ERROR, strict=strict) > 0

    def to_dict(self, *, strict: bool = False) -> dict[str, Any]:
        return {
            "summary": {
                "errors": self.count(Severity.ERROR, strict=strict),
                "warnings": self.count(Severity.WARNING, strict=strict),
                "notices": self.count(Severity.NOTICE, strict=strict),
            },
            "diagnostics": [d.to_dict(strict=strict) for d in self.sorted(strict=strict)],
        }

    def write_json(self, path: Path, *, strict: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(strict=strict), indent=2), encoding="utf-8")

    def sorted(self, *, strict: bool = False) -> list[Diagnostic]:
        rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.NOTICE: 2}
        return sorted(
            self.diagnostics,
            key=lambda d: (
                rank[d.effective_severity(strict)],
                str(d.location.path) if d.location else "",
                d.location.line if d.location else 0,
                d.code,
            ),
        )


_ANSI_RESET = "\x1b[0m"
_ANSI_BY_SEVERITY = {
    Severity.ERROR: "\x1b[31m",
    Severity.WARNING: "\x1b[33m",
    Severity.NOTICE: "\x1b[2m",
}


def render_report(
    report: DiagnosticReport,
    *,
    strict: bool = False,
    show_notices: bool = False,
    color: bool = False,
) -> str:
    def paint(text: str, ansi: str) -> str:
        return f"{ansi}{text}{_ANSI_RESET}" if color else text

    errors = report.count(Severity.ERROR, strict=strict)
    warnings = report.count(Severity.WARNING, strict=strict)
    notices = report.count(Severity.NOTICE, strict=strict)
    verdict = paint("FAILED", "\x1b[31m") if errors else paint("PASSED", "\x1b[32m")
    lines = [f"Preflight: {verdict} ({errors} errors, {warnings} warnings, {notices} notices)"]

    groups = (
        (Severity.ERROR, "ERRORS"),
        (Severity.WARNING, "WARNINGS"),
        (Severity.NOTICE, "NOTICES"),
    )
    sorted_diags = report.sorted(strict=strict)
    for severity, title in groups:
        selected = [d for d in sorted_diags if d.effective_severity(strict) is severity]
        if not selected:
            continue
        if severity is Severity.NOTICE and not show_notices:
            lines.append(f"{title}: {len(selected)} (use --show-defaults or --explain-config to expand)")
            continue
        lines.append("")
        lines.append(paint(f"{title} ({len(selected)})", _ANSI_BY_SEVERITY[severity]))
        for diag in selected:
            where = ""
            if diag.location is not None:
                where = f" ({diag.location.path}:{diag.location.line})"
            field_text = f" [{diag.field}]" if diag.field else ""
            code_text = paint(f"[{diag.code}]", _ANSI_BY_SEVERITY[severity])
            lines.append(f"  {code_text}{field_text}{where} {diag.message}")
            if diag.hint:
                lines.append(f"    Hint: {diag.hint}")
    return "\n".join(lines)
