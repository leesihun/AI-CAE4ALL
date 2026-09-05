from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from .diagnostics import DiagnosticReport, Severity, SourceLocation


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Keys whose value is a filesystem path. The native parsers lowercase every
# string value, which silently rewrites `dataset/NASA_CRM.h5` into a path that
# only resolves on a case-insensitive filesystem; these keys are exempt in the
# native parsers and must be exempt here too or the mirror would disagree.
# Union over every method spec — this parser runs before `model` routing.
PATH_KEYS = frozenset({
    "dataset_dir",
    "infer_dataset",
    "eval_dataset",
    "modelpath",
    "init_modelpath",
    "inference_output_dir",
    "hierarchy_cache_dir",
    "sdf_sidecar",
    "log_dir",
    "log_file_dir",
    "vae_log_file_dir",
    "fm_log_file_dir",
    "lc_log_file_dir",
    "pipeline_log_file",
    "output_dir",
    "opt_surrogate_checkpoint",
    "opt_surrogate_config",
    "param_dir",
    "init_vae_modelpath",
    "vae_modelpath",
    "vae_best_modelpath",
    "fm_modelpath",
    "descriptor_calibration_path",
    "lc_modelpath",
    "input_mesh",
    "input_geometry",
    "output_dataset",
})


@dataclass(frozen=True)
class ConfigEntry:
    key: str
    raw_key: str
    raw_value: str
    value: Any
    location: SourceLocation


@dataclass
class ParsedConfig:
    source_path: Path
    values: dict[str, Any] = field(default_factory=dict)
    raw_values: dict[str, str] = field(default_factory=dict)
    locations: dict[str, SourceLocation] = field(default_factory=dict)
    entries: list[ConfigEntry] = field(default_factory=list)
    duplicates: dict[str, list[ConfigEntry]] = field(default_factory=dict)
    diagnostics: DiagnosticReport = field(default_factory=DiagnosticReport)

    def location(self, key: str) -> SourceLocation | None:
        return self.locations.get(key.lower())


def parse_value(value_str: str, preserve_case: bool = False) -> Any:
    """Mirror the current native config parser conversion behavior.

    ``preserve_case`` (set for `PATH_KEYS`) skips only the string-lowercasing;
    the bool/int/float/list typing is deliberately identical either way.
    """
    value_str = value_str.strip()

    def _text(part: str) -> str:
        return part if preserve_case else part.lower()

    if "," in value_str:
        parts = [part.strip() for part in value_str.split(",")]
        try:
            return [int(part) if "." not in part else float(part) for part in parts]
        except ValueError:
            return [_text(part) for part in parts]

    if " " in value_str:
        parts = value_str.split()
        if len(parts) > 1:
            try:
                return [int(part) if "." not in part else float(part) for part in parts]
            except ValueError:
                return [_text(part) for part in parts]

    if value_str.lower() in ("true", "false"):
        return value_str.lower() == "true"

    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        return _text(value_str)


def parse_config(path: str | Path) -> ParsedConfig:
    source = Path(path).expanduser().resolve()
    parsed = ParsedConfig(source_path=source)

    if not source.exists():
        parsed.diagnostics.add(
            "CFG-FILE-001",
            Severity.ERROR,
            f"Config file does not exist: {source}",
            hint="Pass an existing flat text config with --config.",
        )
        return parsed
    if not source.is_file():
        parsed.diagnostics.add(
            "CFG-FILE-002",
            Severity.ERROR,
            f"Config path is not a regular file: {source}",
        )
        return parsed

    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        parsed.diagnostics.add(
            "CFG-FILE-003",
            Severity.ERROR,
            f"Config is not valid UTF-8: {exc}",
        )
        return parsed
    except OSError as exc:
        parsed.diagnostics.add(
            "CFG-FILE-004",
            Severity.ERROR,
            f"Could not read config: {exc}",
        )
        return parsed

    if text.startswith(chr(0xFEFF)):
        parsed.diagnostics.add(
            "CFG-FILE-005",
            Severity.ERROR,
            "Config starts with a UTF-8 byte-order mark; the native parsers do not strip it and would misread the first key.",
            location=SourceLocation(source, 1),
            hint="Save the file as UTF-8 without BOM (PowerShell 5.1 'Out-File -Encoding utf8' and Notepad add one).",
        )
        text = text.lstrip(chr(0xFEFF))
    lines = text.splitlines()

    seen: dict[str, ConfigEntry] = {}
    for line_number, original in enumerate(lines, start=1):
        stripped = original.strip()
        if not stripped or stripped.startswith("%"):
            continue
        if stripped in {"'", '"'}:
            parsed.diagnostics.add(
                "CFG-SYNTAX-LEGACY",
                Severity.NOTICE,
                "Ignored a legacy quote-only section separator.",
                location=SourceLocation(source, line_number),
            )
            continue

        body = stripped.split("#", 1)[0].strip()
        if not body:
            continue

        parts = body.split(None, 1)
        if len(parts) != 2 or not parts[1].strip():
            parsed.diagnostics.add(
                "CFG-SYNTAX-001",
                Severity.ERROR,
                "Expected a key followed by a value.",
                location=SourceLocation(source, line_number),
                hint="Use: key value",
            )
            continue

        raw_key, raw_value = parts[0].strip(), parts[1].strip()
        if not _KEY_RE.fullmatch(raw_key):
            parsed.diagnostics.add(
                "CFG-SYNTAX-002",
                Severity.ERROR,
                f"Invalid config key {raw_key!r}.",
                location=SourceLocation(source, line_number),
            )
            continue

        key = raw_key.lower()
        if key == "reserved":
            continue
        value = parse_value(raw_value, preserve_case=key in PATH_KEYS)
        entry = ConfigEntry(
            key=key,
            raw_key=raw_key,
            raw_value=raw_value,
            value=value,
            location=SourceLocation(source, line_number),
        )
        parsed.entries.append(entry)

        if key in seen:
            parsed.duplicates.setdefault(key, [seen[key]]).append(entry)
            parsed.diagnostics.add(
                "CFG-SYNTAX-003",
                Severity.ERROR,
                f"Duplicate key {key!r}; the native parser would silently use the last value.",
                field_name=key,
                location=entry.location,
                hint=f"Remove one definition; the first is on line {seen[key].location.line}.",
            )
        else:
            seen[key] = entry

        parsed.values[key] = value
        parsed.raw_values[key] = raw_value
        parsed.locations[key] = entry.location

        if raw_value.startswith(('"', "'")) or raw_value.endswith(('"', "'")):
            parsed.diagnostics.add(
                "CFG-SYNTAX-004",
                Severity.WARNING,
                "Quoted values are not supported by the native config parser; quote characters become part of the value.",
                field_name=key,
                location=entry.location,
                promote_in_strict=True,
            )

    return parsed
