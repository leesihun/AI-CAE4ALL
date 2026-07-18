from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
from typing import Any

from .config_parser import ParsedConfig, parse_config
from .diagnostics import DiagnosticReport, Severity
from .path_checks import validate_paths
from .registry import MethodRegistry, ResolvedMethod
from .settings import LocalSettings
from .specs import SpecValidationContext


@dataclass(frozen=True)
class PreflightOptions:
    strict: bool = False
    skip_filesystem: bool = False
    skip_native: bool = False
    skip_environment: bool = False
    skip_dataset: bool = False
    python_override: str | None = None


@dataclass
class PreflightResult:
    parsed: ParsedConfig
    report: DiagnosticReport
    resolved: ResolvedMethod | None = None
    mode: str | None = None
    python_executable: Path | None = None
    command: list[str] = field(default_factory=list)
    resolved_paths: dict[str, Path] = field(default_factory=dict)
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    checkpoint_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


def _missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def _validate_spec(result: PreflightResult) -> None:
    assert result.resolved is not None
    spec = result.resolved.spec
    parsed = result.parsed
    report = result.report
    model_id = result.resolved.model_id
    mode = result.mode

    for name in sorted(spec.required_fields(model_id, mode)):
        if name not in parsed.values or _missing(parsed.values.get(name)):
            report.add(
                "CFG-REQ-001",
                Severity.ERROR,
                f"{name} is required for {model_id} {mode or 'configuration'}.",
                field_name=name,
                hint=f"Add a '{name} <value>' line to the config.",
            )

    for name in sorted(set(parsed.values) - set(spec.known_keys)):
        report.add(
            "CFG-UNKNOWN-001",
            Severity.WARNING,
            f"Unknown config key for {model_id}: {name}",
            field_name=name,
            location=parsed.location(name),
            hint="Check for a typo or update the method specification if this is a new supported key.",
            promote_in_strict=True,
        )

    if mode is not None:
        for name in sorted(spec.recommended_by_mode.get(mode, ())):
            if name not in parsed.values:
                report.add(
                    "CFG-REC-001",
                    Severity.WARNING,
                    f"Recommended field {name} is absent; verify the native default is intended.",
                    field_name=name,
                    promote_in_strict=True,
                )

    active_defaults = dict(spec.defaults)
    if mode is not None:
        active_defaults.update(spec.defaults_by_mode.get(mode, {}))
    for name, default in sorted(active_defaults.items()):
        if name not in parsed.values:
            report.add(
                "CFG-DEFAULT-001",
                Severity.NOTICE,
                f"{name} is absent; native/default behavior is {default!r}. The config was not modified.",
                field_name=name,
            )

    context = SpecValidationContext(
        parsed=parsed,
        mode=mode,
        model_id=model_id,
        repository_root=result.resolved.repository_root,
        report=report,
    )
    for validator in spec.validators:
        validator(context)


def _probe_environment(
    python_executable: Path,
    modules: tuple[str, ...],
    repository_root: Path,
    gpu_ids: Any,
    report: DiagnosticReport,
) -> None:
    if not python_executable.is_file():
        report.add(
            "ENV-PYTHON-001",
            Severity.ERROR,
            f"Python interpreter does not exist: {python_executable}",
            hint="Set --python or configure cae_suite.local.toml.",
        )
        return
    code = (
        "import importlib, json, sys; "
        "mods=json.loads(sys.argv[1]); failed={}; "
        "\nfor m in mods:\n"
        " try: importlib.import_module(m)\n"
        " except Exception as e: failed[m]=f'{type(e).__name__}: {e}'\n"
        "cuda_count=None\n"
        "if 'torch' in mods and 'torch' not in failed:\n"
        " import torch; cuda_count=torch.cuda.device_count()\n"
        "print(json.dumps({'version':sys.version.split()[0],'failed':failed,'cuda_count':cuda_count}))"
    )
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", code, json.dumps(list(modules))],
            cwd=repository_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.add("ENV-PYTHON-002", Severity.ERROR, f"Could not probe Python environment: {exc}")
        return
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        report.add(
            "ENV-PYTHON-003",
            Severity.ERROR,
            f"Python environment probe returned invalid output (exit {completed.returncode}).",
            hint=completed.stderr.strip()[-500:] or None,
        )
        return
    failed = payload.get("failed", {})
    if failed:
        for module, error in failed.items():
            report.add(
                "ENV-IMPORT-001",
                Severity.ERROR,
                f"Required module {module!r} is unavailable in {python_executable}: {error}",
                hint="Use the method's Python environment or install its requirements.",
            )
    cuda_count = payload.get("cuda_count")
    requested = gpu_ids if isinstance(gpu_ids, list) else [gpu_ids]
    requested_cuda = [value for value in requested if isinstance(value, int) and value >= 0]
    if requested_cuda and isinstance(cuda_count, int):
        if cuda_count == 0:
            report.add(
                "ENV-CUDA-001",
                Severity.WARNING,
                f"The config requests CUDA GPU IDs {requested_cuda}, but the selected Python environment reports no visible CUDA devices; native code may fall back to CPU.",
                field_name="gpu_ids",
            )
        else:
            unavailable = [value for value in requested_cuda if value >= cuda_count]
            if unavailable:
                report.add(
                    "ENV-CUDA-002",
                    Severity.ERROR,
                    f"Requested GPU IDs {unavailable} are not visible; the environment reports {cuda_count} CUDA device(s).",
                    field_name="gpu_ids",
                )


def _probe_dataset(
    result: PreflightResult,
    suite_root: Path,
) -> None:
    assert result.resolved is not None and result.python_executable is not None
    kind = result.resolved.spec.dataset_kind
    if kind is None:
        return
    field_name = "dataset_dir" if result.mode in {"train", "train_vae", "train_fm"} else "infer_dataset"
    path = result.resolved_paths.get(field_name)
    if path is None or not path.is_file():
        return
    script = suite_root / "cae_suite" / "dataset_probe.py"
    try:
        completed = subprocess.run(
            [str(result.python_executable), str(script), kind, str(path)],
            cwd=result.resolved.repository_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.report.add("DATASET-PROBE-001", Severity.WARNING, f"Dataset schema probe could not run: {exc}")
        return
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result.report.add(
            "DATASET-PROBE-002",
            Severity.WARNING,
            "Dataset schema probe returned invalid output.",
            hint=completed.stderr.strip()[-500:] or None,
        )
        return
    for message in payload.get("errors", []):
        result.report.add(
            "DATASET-SCHEMA-001",
            Severity.ERROR,
            message,
            field_name=field_name,
            location=result.parsed.location(field_name),
        )
    for message in payload.get("warnings", []):
        result.report.add(
            "DATASET-SCHEMA-002",
            Severity.WARNING,
            message,
            field_name=field_name,
            location=result.parsed.location(field_name),
        )
    result.dataset_metadata = payload.get("metadata", {})
    _validate_dataset_against_config(result, field_name)


def _validate_dataset_against_config(result: PreflightResult, field_name: str) -> None:
    metadata = result.dataset_metadata
    nodal_shape = metadata.get("nodal_shape")
    if not nodal_shape or len(nodal_shape) != 3:
        return
    feature_count, timesteps, _nodes = nodal_shape
    values = result.parsed.values
    input_var = values.get("input_var")
    output_var = values.get("output_var")
    try:
        required_features = 3 + max(int(input_var), int(output_var))
    except (TypeError, ValueError):
        return
    if feature_count < required_features:
        result.report.add(
            "DATASET-FEATURES-001",
            Severity.ERROR,
            f"Dataset has {feature_count} feature rows but the config needs at least {required_features}.",
            field_name=field_name,
            location=result.parsed.location(field_name),
        )
    if values.get("use_node_types", False) is True and feature_count <= 7:
        result.report.add(
            "DATASET-NODETYPE-001",
            Severity.ERROR,
            f"use_node_types=True requires a node-type row, but the dataset has only {feature_count} feature rows.",
            field_name="use_node_types",
            location=result.parsed.location("use_node_types"),
        )
    if timesteps > 1 and input_var != output_var:
        result.report.add(
            "DATASET-TEMPORAL-001",
            Severity.ERROR,
            f"Temporal data (T={timesteps}) requires input_var == output_var; got {input_var} and {output_var}.",
            field_name="input_var",
            location=result.parsed.location("input_var"),
        )


def _probe_native(result: PreflightResult, suite_root: Path) -> None:
    assert result.resolved is not None and result.python_executable is not None
    script = suite_root / "cae_suite" / "native_probe.py"
    try:
        completed = subprocess.run(
            [str(result.python_executable), str(script), str(result.parsed.source_path)],
            cwd=result.resolved.repository_root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        result.report.add("NATIVE-CHECK-001", Severity.ERROR, f"Native config validation probe could not run: {exc}")
        return
    marker = "__CAE_SUITE_NATIVE_RESULT__"
    payload = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(marker):
            try:
                payload = json.loads(line[len(marker):])
            except json.JSONDecodeError:
                payload = None
            break
    if payload is None:
        result.report.add(
            "NATIVE-CHECK-002",
            Severity.ERROR,
            f"Native config validation returned no structured result (exit {completed.returncode}).",
            hint=completed.stderr.strip()[-1000:] or completed.stdout.strip()[-1000:] or None,
        )
    elif not payload.get("ok", False):
        result.report.add(
            "NATIVE-CHECK-003",
            Severity.ERROR,
            f"Native config validation failed: {payload.get('error', 'unknown error')}",
            hint=completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else None,
        )


def _probe_checkpoints(result: PreflightResult, suite_root: Path) -> None:
    assert result.resolved is not None and result.python_executable is not None
    script = suite_root / "cae_suite" / "checkpoint_probe.py"
    checkpoint_fields = [
        (field_name, path)
        for field_name, path in result.resolved_paths.items()
        if field_name.endswith("modelpath") and path.is_file()
    ]
    for field_name, path in checkpoint_fields:
        # Training output paths are not inputs. A present old output is reported by
        # path validation, but it should not define the new run's architecture.
        if result.mode == "train" and field_name == "modelpath":
            continue
        if result.mode == "train_vae" and field_name == "vae_modelpath":
            continue
        if result.mode == "train_fm" and field_name == "fm_modelpath":
            continue
        try:
            completed = subprocess.run(
                [str(result.python_executable), str(script), str(path)],
                cwd=result.resolved.repository_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            result.report.add("CHECKPOINT-PROBE-001", Severity.WARNING, f"Checkpoint metadata probe could not run for {field_name}: {exc}", field_name=field_name)
            continue
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {"ok": False, "error": "invalid probe output"}
        if not payload.get("ok", False):
            result.report.add(
                "CHECKPOINT-PROBE-002",
                Severity.WARNING,
                f"Safe metadata inspection was unavailable for {field_name}: {payload.get('error', 'unknown error')}",
                field_name=field_name,
                location=result.parsed.location(field_name),
                hint="The native runtime remains the authoritative checkpoint loader; weights_only=False was not used by the suite.",
            )
            continue
        result.checkpoint_metadata[field_name] = payload
        selected = payload.get("selected_model") or payload.get("model_config_model")
        if isinstance(selected, str):
            selected_normalized = selected.lower().strip()
            if selected_normalized != result.resolved.model_id:
                result.report.add(
                    "CHECKPOINT-MODEL-001",
                    Severity.ERROR,
                    f"Checkpoint model {selected_normalized!r} does not match config model {result.resolved.model_id!r}.",
                    field_name=field_name,
                    location=result.parsed.location(field_name),
                )
        expected_stage = (
            "vae" if field_name in {"vae_modelpath", "init_vae_modelpath"}
            else "fm" if field_name == "fm_modelpath"
            else None
        )
        actual_stage = payload.get("stage")
        if expected_stage and actual_stage and actual_stage != expected_stage:
            result.report.add(
                "CHECKPOINT-STAGE-001",
                Severity.ERROR,
                f"{field_name} expects an {expected_stage!r} checkpoint but the file records stage {actual_stage!r}.",
                field_name=field_name,
                location=result.parsed.location(field_name),
            )
        if field_name == "modelpath" and not payload.get("has_normalization", False):
            result.report.add(
                "CHECKPOINT-NORM-001",
                Severity.WARNING,
                "Checkpoint metadata does not show normalization statistics; solver inference may reject it.",
                field_name=field_name,
                location=result.parsed.location(field_name),
            )


def run_preflight(
    config_path: str | Path,
    *,
    suite_root: Path,
    registry: MethodRegistry,
    settings: LocalSettings,
    options: PreflightOptions,
) -> PreflightResult:
    parsed = parse_config(config_path)
    result = PreflightResult(parsed=parsed, report=parsed.diagnostics)
    resolved = registry.resolve(parsed.values.get("model"), result.report)
    result.resolved = resolved

    mode_value = parsed.values.get("mode")
    if mode_value is None:
        result.report.add(
            "CFG-COMMON-002",
            Severity.ERROR,
            "Missing required config field 'mode'.",
            field_name="mode",
            hint="Set a mode supported by the selected model.",
        )
    elif not isinstance(mode_value, str):
        result.report.add(
            "ROUTE-MODE-001",
            Severity.ERROR,
            f"mode must be one name, got {mode_value!r}.",
            field_name="mode",
            location=parsed.location("mode"),
        )
    else:
        result.mode = mode_value.lower().strip()

    if resolved is None:
        return result
    if result.mode is not None and result.mode not in resolved.spec.valid_modes:
        result.report.add(
            "ROUTE-MODE-002",
            Severity.ERROR,
            f"Mode {result.mode!r} is unsupported for {resolved.model_id}.",
            field_name="mode",
            location=parsed.location("mode"),
            hint=f"Supported modes: {', '.join(resolved.spec.valid_modes)}",
        )

    _validate_spec(result)
    result.python_executable = settings.resolve_python(resolved.model_id, resolved.spec.spec_id, options.python_override)

    if not options.skip_filesystem:
        result.resolved_paths = validate_paths(parsed, resolved.spec, result.mode, resolved.repository_root, result.report)
    if not options.skip_environment and not result.report.has_errors(strict=options.strict):
        _probe_environment(
            result.python_executable,
            resolved.spec.import_modules,
            resolved.repository_root,
            parsed.values.get("gpu_ids"),
            result.report,
        )
    if (
        not options.skip_filesystem
        and not options.skip_dataset
        and not result.report.has_errors(strict=options.strict)
    ):
        _probe_dataset(result, suite_root)
    if not options.skip_filesystem and not result.report.has_errors(strict=options.strict):
        _probe_checkpoints(result, suite_root)
    if (
        resolved.spec.native_probe
        and not options.skip_native
        and not result.report.has_errors(strict=options.strict)
    ):
        _probe_native(result, suite_root)

    if result.python_executable is not None:
        result.command = [str(result.python_executable), str(resolved.entrypoint), "--config", str(parsed.source_path)]
    return result
