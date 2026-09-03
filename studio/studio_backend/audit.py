"""Repository-wide config audit and single-config explanation.

These mirror `cae_suite/cli.py --audit-configs` and `--explain-config`
exactly (same diagnostic-code buckets, same skip-layer choices for the bulk
audit) so the Studio "Config audit" and "Explain this config" views report
the same verdict a human would get running the CLI directly. Nothing here
re-implements validation; both functions call `cae_suite.preflight.run_preflight`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cae_suite.config_discovery import checked_in_config_paths
from studio_backend.paths import SUITE_ROOT, json_safe, relative
from studio_backend.state import STATE, studio_preflight_diagnostics
from studio_backend.suite_bridge import PreflightOptions, Severity, run_preflight

# Mirrors cae_suite/cli.py's _INACTIVE_CODES / _CHECKPOINT_CODES exactly so the
# Studio's explanation buckets never drift from the CLI's.
_INACTIVE_CODES = {
    "NOVAR-INACTIVE", "NOVAR-REMOVED", "MGN-REMOVED-VAR",
    "MGN-VARIATIONAL-IGNORED", "MGNV-REMOVED", "TRANS-MGN-KEY",
}
_CHECKPOINT_CODES = {
    "MGNV-CKPT-OVERRIDE", "TRANS-CKPT-001", "NOVAR-CKPT-001",
    "CHECKPOINT-MODEL-001", "CHECKPOINT-STAGE-001", "CHECKPOINT-NORM-001",
}


def audit_configs(*, strict: bool = False) -> dict[str, Any]:
    """Structurally audit the same checked-in config*.txt set as the CLI."""
    STATE.require_suite()
    registry = STATE.registry
    settings = STATE.settings

    paths = checked_in_config_paths(SUITE_ROOT)

    files: list[dict[str, Any]] = []
    total_errors = 0
    total_warnings = 0
    for path in sorted(paths, key=str):
        result = run_preflight(
            path,
            suite_root=SUITE_ROOT,
            registry=registry,
            settings=settings,
            options=PreflightOptions(
                strict=strict,
                skip_filesystem=True,
                skip_native=True,
                skip_environment=True,
                skip_dataset=True,
            ),
        )
        errors = result.report.count(Severity.ERROR, strict=strict)
        warnings = result.report.count(Severity.WARNING, strict=strict)
        total_errors += errors
        total_warnings += warnings
        model = result.resolved.model_id if result.resolved is not None else None
        files.append(
            {
                "path": relative(path),
                "model": model,
                "mode": result.mode,
                "status": "PASS" if errors == 0 else "FAIL",
                "errors": errors,
                "warnings": warnings,
                "report": result.report.to_dict(strict=strict),
            }
        )
    return {
        "strict": strict,
        "summary": {"files": len(paths), "errors": total_errors, "warnings": total_warnings},
        "files": files,
    }


def explain_config(
    text: str,
    *,
    label: str = "config",
    strict: bool = False,
    skip_filesystem: bool = False,
    skip_native: bool = False,
    skip_environment: bool = False,
) -> dict[str, Any]:
    """Bucket a config's keys exactly like `--explain-config` does, as structured JSON."""
    STATE.require_suite()
    config_path = STATE.save_config(text, label, "explain")
    result = run_preflight(
        config_path,
        suite_root=SUITE_ROOT,
        registry=STATE.registry,
        settings=STATE.settings,
        options=PreflightOptions(
            strict=strict,
            skip_filesystem=skip_filesystem,
            skip_native=skip_native,
            skip_environment=skip_environment,
        ),
    )
    studio_diagnostics = studio_preflight_diagnostics(result)
    diagnostics = list(result.report.diagnostics)

    fields_by_code: dict[str, set[str]] = {}
    for diagnostic in diagnostics:
        if diagnostic.field:
            fields_by_code.setdefault(diagnostic.code, set()).add(diagnostic.field)
    for diagnostic in studio_diagnostics:
        if diagnostic.get("field"):
            fields_by_code.setdefault(diagnostic["code"], set()).add(diagnostic["field"])

    def collect(codes: set[str]) -> list[str]:
        names: set[str] = set()
        for code in codes:
            names.update(fields_by_code.get(code, ()))
        return sorted(names)

    parsed = result.parsed
    configured = {
        key: {
            "value": parsed.values[key],
            "line": parsed.location(key).line if parsed.location(key) else None,
        }
        for key in sorted(parsed.values)
    }

    required_present: list[str] = []
    required_missing: list[str] = []
    if result.resolved is not None:
        spec = result.resolved.spec
        required = spec.required_fields(result.resolved.model_id, result.mode)
        active_defaults = dict(spec.defaults)
        if result.mode is not None:
            active_defaults.update(spec.defaults_by_mode.get(result.mode, {}))
        required_present = sorted(
            name for name in required
            if name in parsed.values and parsed.values[name] not in (None, "", [])
        )
        required_missing = sorted(
            name for name in required
            if (name not in parsed.values or parsed.values[name] in (None, "", []))
            and name not in active_defaults
        )

    checkpoint_fields = sorted(set(collect(_CHECKPOINT_CODES)) | set(result.checkpoint_metadata))
    malformed = [
        {"code": d.code, "line": d.location.line if d.location else None, "message": d.message}
        for d in diagnostics
        if d.code.startswith("CFG-SYNTAX-") and d.severity is Severity.ERROR
    ]

    route = None
    if result.resolved is not None:
        route = {
            "model": result.resolved.model_id,
            "method": result.resolved.spec.display_name,
            "mode": result.mode,
        }

    report = result.report.to_dict(strict=strict)
    report["diagnostics"] = json_safe(studio_diagnostics) + report["diagnostics"]

    return {
        "config_path": relative(config_path),
        "route": route,
        "configured": configured,
        "required_present": required_present,
        "required_missing": required_missing,
        "recommended_missing": collect({"CFG-REC-001"}),
        "optional_defaulted": collect({"CFG-DEFAULT-001"}),
        "inactive_or_removed": collect(_INACTIVE_CODES),
        "checkpoint_owned": checkpoint_fields,
        "unknown_keys": collect({"CFG-UNKNOWN-001"}),
        "malformed_lines": malformed,
        "report": report,
    }
