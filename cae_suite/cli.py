from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

from .diagnostics import DiagnosticReport, Severity, render_report
from .launcher import launch_and_wait
from .preflight import PreflightOptions, PreflightResult, run_preflight
from .registry import MethodRegistry
from .settings import LocalSettings


def _suite_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatically validate and launch the AI-CAE method selected by config 'model'."
    )
    parser.add_argument("--config", type=str, help="Path to a native flat text config.")
    parser.add_argument("--check", action="store_true", help="Validate only; do not launch.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show the exact native command without launching.")
    parser.add_argument("--strict", action="store_true", help="Promote selected warnings to errors.")
    parser.add_argument("--explain-config", action="store_true", help="Show configured/defaulted/inactive diagnostics without launching.")
    parser.add_argument("--show-defaults", action="store_true", help="Expand default and notice diagnostics.")
    parser.add_argument("--list-models", action="store_true", help="List registered model IDs and installation health.")
    parser.add_argument("--describe", metavar="MODEL", help="Describe routing and required fields for a model.")
    parser.add_argument("--audit-configs", action="store_true", help="Structurally audit checked-in config*.txt files.")
    parser.add_argument("--skip-native-check", action="store_true", help="Skip the selected method's native config probe.")
    parser.add_argument("--skip-filesystem-check", action="store_true", help="Skip dataset/checkpoint/output path checks.")
    parser.add_argument("--skip-environment-check", action="store_true", help="Skip interpreter dependency and CUDA visibility checks.")
    parser.add_argument("--python", dest="python_override", help="Override the selected method's Python interpreter.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors in the report (colors are used only on a terminal).")
    parser.add_argument("--json-report", type=str, help="Write the validation report to this JSON path.")
    return parser


def _print_models(registry: MethodRegistry) -> int:
    print("AI-CAE4ALL registered models")
    print()
    errors = False
    seen_specs: set[str] = set()
    for model_id in registry.model_ids:
        report = DiagnosticReport()
        resolved = registry.resolve(model_id, report)
        if resolved is None:
            continue
        installed = resolved.repository_root.is_dir() and resolved.entrypoint.is_file()
        errors = errors or not installed
        print(f"{model_id:22} {'OK' if installed else 'MISSING':8} {resolved.spec.display_name}")
        print(f"{'':22} modes={','.join(resolved.spec.valid_modes)}")
        if resolved.spec.spec_id not in seen_specs:
            print(f"{'':22} repo={resolved.repository_root}")
            print(f"{'':22} entrypoint={resolved.entrypoint.name}")
            seen_specs.add(resolved.spec.spec_id)
    return 3 if errors else 0


def _describe_model(registry: MethodRegistry, model_id: str) -> int:
    report = DiagnosticReport()
    resolved = registry.resolve(model_id.lower(), report)
    if resolved is None or report.has_errors():
        print(render_report(report, show_notices=True))
        return 3
    spec = resolved.spec
    print(f"Model       : {resolved.model_id}")
    print(f"Method      : {spec.display_name}")
    print(f"Repository  : {resolved.repository_root}")
    print(f"Entrypoint  : {resolved.entrypoint}")
    print(f"Modes       : {', '.join(spec.valid_modes)}")
    print(f"Dataset     : {spec.dataset_kind or 'method-defined'}")
    for mode in spec.valid_modes:
        fields = sorted(spec.required_fields(resolved.model_id, mode))
        print(f"Required ({mode}): {', '.join(fields)}")
    return 0


def _print_route(result: PreflightResult) -> None:
    print("AI-CAE4ALL")
    print(f"Config      : {result.parsed.source_path}")
    if result.resolved is not None:
        print(f"Model       : {result.resolved.model_id}")
        print(f"Mode        : {result.mode or '<missing>'}")
        print(f"Repository  : {result.resolved.repository_root}")
        print(f"Entrypoint  : {result.resolved.entrypoint.name}")
    else:
        print(f"Model       : {result.parsed.values.get('model', '<missing>')}")
        print(f"Mode        : {result.mode or result.parsed.values.get('mode', '<missing>')}")
    if result.python_executable is not None:
        print(f"Python      : {result.python_executable}")
    if result.command:
        print(f"Command     : {subprocess_display(result.command)}")
    if result.dataset_metadata:
        print(f"Dataset     : {result.dataset_metadata}")
    if result.checkpoint_metadata:
        for field_name, metadata in result.checkpoint_metadata.items():
            summary = {
                key: metadata.get(key)
                for key in ("stage", "selected_model", "model_config_model", "schema_version", "checkpoint_version", "has_normalization", "has_ema")
                if metadata.get(key) is not None
            }
            print(f"Checkpoint  : {field_name} {summary}")
    print()


_INACTIVE_CODES = {
    "NOVAR-INACTIVE", "NOVAR-REMOVED", "MGN-REMOVED-VAR",
    "MGN-VARIATIONAL-IGNORED", "MGNV-REMOVED", "TRANS-MGN-KEY",
}
_CHECKPOINT_CODES = {
    "MGNV-CKPT-OVERRIDE", "TRANS-CKPT-001", "NOVAR-CKPT-001",
    "CHECKPOINT-MODEL-001", "CHECKPOINT-STAGE-001", "CHECKPOINT-NORM-001",
}


def _print_config_explanation(result: PreflightResult) -> None:
    parsed = result.parsed
    fields_by_code: dict[str, set[str]] = {}
    for diagnostic in result.report.diagnostics:
        if diagnostic.field:
            fields_by_code.setdefault(diagnostic.code, set()).add(diagnostic.field)

    def collect(codes: set[str]) -> list[str]:
        names: set[str] = set()
        for code in codes:
            names.update(fields_by_code.get(code, ()))
        return sorted(names)

    def show(title: str, names: list[str]) -> None:
        print(f"{title} ({len(names)}): {', '.join(names) if names else '<none>'}")

    print("CONFIG EXPLANATION")
    print(f"Explicitly configured ({len(parsed.values)}):")
    for key in sorted(parsed.values):
        location = parsed.location(key)
        line = f" line {location.line}" if location else ""
        print(f"  {key} = {parsed.values[key]!r}{line}")
    if result.resolved is not None:
        required = result.resolved.spec.required_fields(result.resolved.model_id, result.mode)
        show("Required and present", sorted(name for name in required if name in parsed.values))
        show("Required and missing", sorted(name for name in required if name not in parsed.values))
    show("Recommended but missing", collect({"CFG-REC-001"}))
    show("Optional; native default applies", collect({"CFG-DEFAULT-001"}))
    show("Inactive/ignored/removed for this model", collect(_INACTIVE_CODES))
    checkpoint_fields = set(collect(_CHECKPOINT_CODES)) | set(result.checkpoint_metadata)
    show("Checkpoint-owned or checkpoint-validated", sorted(checkpoint_fields))
    show("Unknown keys", collect({"CFG-UNKNOWN-001"}))
    malformed = [d for d in result.report.diagnostics if d.code.startswith("CFG-SYNTAX-") and d.severity is Severity.ERROR]
    lines = ", ".join(f"line {d.location.line}" for d in malformed if d.location)
    print(f"Malformed lines ({len(malformed)}): {lines if lines else '<none>'}")
    print()


def subprocess_display(command: list[str]) -> str:
    if sys.platform == "win32":
        return " ".join(f'"{part}"' if " " in part else part for part in command)
    return shlex.join(command)


def _preflight_exit_code(result: PreflightResult, *, strict: bool) -> int:
    error_codes = {
        diagnostic.code
        for diagnostic in result.report.diagnostics
        if diagnostic.effective_severity(strict) is Severity.ERROR
    }
    if any(code.startswith("ROUTE-") for code in error_codes):
        return 3
    if any(code.startswith("ENV-") for code in error_codes):
        return 4
    if any(code.startswith("NATIVE-CHECK-") for code in error_codes):
        return 5
    return 2


def _audit_configs(
    registry: MethodRegistry,
    settings: LocalSettings,
    *,
    suite_root: Path,
    strict: bool,
    json_report: str | None,
) -> int:
    paths: set[Path] = set()
    for spec in registry.specs:
        repo = suite_root / spec.repository
        if not repo.is_dir():
            continue
        for path in repo.rglob("config*.txt"):
            lowered = {part.lower() for part in path.parts}
            if "outputs" in lowered or ".git" in lowered or "__pycache__" in lowered:
                continue
            paths.add(path.resolve())

    print(f"Auditing {len(paths)} checked-in configs (structural checks only)")
    total_errors = 0
    total_warnings = 0
    combined: dict[str, object] = {"configs": []}
    for path in sorted(paths, key=str):
        result = run_preflight(
            path,
            suite_root=suite_root,
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
        relative = path.relative_to(suite_root)
        status = "PASS" if errors == 0 else "FAIL"
        print(f"{status:4}  errors={errors:<2} warnings={warnings:<2}  {relative}")
        combined["configs"].append({"path": str(relative), **result.report.to_dict(strict=strict)})

    combined["summary"] = {"files": len(paths), "errors": total_errors, "warnings": total_warnings}
    print()
    print(f"Audit complete: files={len(paths)}, errors={total_errors}, warnings={total_warnings}")
    if json_report:
        import json
        output = Path(json_report).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"JSON report: {output}")
    return 2 if total_errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    suite_root = _suite_root()
    registry = MethodRegistry(suite_root)

    if args.list_models:
        return _print_models(registry)
    if args.describe:
        return _describe_model(registry, args.describe)
    try:
        settings = LocalSettings.load(suite_root)
    except Exception as exc:
        print(f"[ENV-SETTINGS-001] Could not load AI-CAE4ALL local settings: {type(exc).__name__}: {exc}")
        return 4
    if args.audit_configs:
        return _audit_configs(
            registry,
            settings,
            suite_root=suite_root,
            strict=args.strict,
            json_report=args.json_report,
        )
    if not args.config:
        parser.error("--config is required unless --list-models, --describe, or --audit-configs is used")

    options = PreflightOptions(
        strict=args.strict,
        skip_filesystem=args.skip_filesystem_check,
        skip_native=args.skip_native_check,
        skip_environment=args.skip_environment_check,
        python_override=args.python_override,
    )
    result = run_preflight(
        args.config,
        suite_root=suite_root,
        registry=registry,
        settings=settings,
        options=options,
    )
    _print_route(result)
    if args.explain_config:
        _print_config_explanation(result)
    show_notices = args.show_defaults or args.explain_config
    use_color = not args.no_color and sys.stdout.isatty()
    print(render_report(result.report, strict=args.strict, show_notices=show_notices, color=use_color))

    if args.json_report:
        output = Path(args.json_report).expanduser().resolve()
        result.report.write_json(output, strict=args.strict)
        print(f"JSON report: {output}")

    if result.report.has_errors(strict=args.strict):
        print("\nNo model process was started.")
        return _preflight_exit_code(result, strict=args.strict)
    if args.check or args.dry_run or args.explain_config:
        print("\nValidation completed; no model process was started by request.")
        return 0
    if result.resolved is None or not result.command:
        print("\nNo model process was started because routing did not produce a command.")
        return 3

    print(f"\nStarting {result.resolved.spec.display_name}...", flush=True)
    return launch_and_wait(result.command, cwd=result.resolved.repository_root)


if __name__ == "__main__":
    raise SystemExit(main())
