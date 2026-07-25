"""Durable Studio state: suite registry/settings, preflight, and the job engine.

A StudioState is a process-lifetime singleton (STATE, below). It owns the one
registry/settings pair loaded from cae_suite, runs preflight through the real
launcher pipeline, and supervises subprocess jobs (pipeline runs and one-off
command jobs such as portable inference or a PyInstaller build). Job records
are persisted as JSON next to their log file so a Studio restart can recover
and mark interrupted jobs rather than losing history.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from studio_backend.paths import (
    JOB_RUNTIME,
    MAX_BODY,
    SUITE_ROOT,
    json_safe,
    relative,
    slug,
    utc_now,
)
from studio_backend.suite_bridge import (
    SUITE_IMPORT_ERROR,
    LocalSettings,
    MethodRegistry,
    PreflightOptions,
    Severity,
    run_preflight,
)


def studio_preflight_diagnostics(result: Any) -> list[dict[str, Any]]:
    if result.resolved is None or result.resolved.model_id != "simulgenvae":
        return []
    diagnostics: list[dict[str, Any]] = []
    values = result.parsed.values
    repository = result.resolved.repository_root

    def error(code: str, field: str, message: str, hint: str) -> None:
        diagnostics.append(
            {
                "code": code,
                "severity": "error",
                "original_severity": "error",
                "message": message,
                "field": field,
                "hint": hint,
            }
        )

    param_value = values.get("param_dir")
    param_path = (repository / str(param_value)).resolve() if param_value else None
    if param_path is not None and not param_path.exists():
        error(
            "STUDIO-SGV-PARAM-001",
            "param_dir",
            f"SimulGen condition source does not exist: {param_path}",
            "Select a real headerless CSV or image directory. The Studio smoke fixture can create a runnable local example.",
        )

    dataset_value = values.get("dataset_dir")
    dataset_path = (repository / str(dataset_value)).resolve() if dataset_value else None
    sample_count: int | None = None
    if dataset_path is not None and dataset_path.is_file():
        try:
            import h5py

            with h5py.File(dataset_path, "r") as handle:
                if "data" not in handle:
                    error(
                        "STUDIO-SGV-DATA-001",
                        "dataset_dir",
                        "SimulGen dataset is missing the root data group.",
                        "Use the shared mesh HDF5 contract: data/{id}/nodal_data[F,T,N].",
                    )
                else:
                    sample_ids = sorted(handle["data"].keys(), key=str)
                    sample_count = len(sample_ids)
                    shapes: dict[tuple[int, int], list[str]] = {}
                    for sample_id in sample_ids:
                        group = handle["data"][sample_id]
                        if "nodal_data" not in group or len(group["nodal_data"].shape) != 3:
                            error(
                                "STUDIO-SGV-DATA-002",
                                "dataset_dir",
                                f"Sample {sample_id} has no rank-3 nodal_data[F,T,N].",
                                "Rebuild or select a compatible fixed-geometry mesh HDF5.",
                            )
                            continue
                        _, timesteps, nodes = group["nodal_data"].shape
                        shapes.setdefault((int(timesteps), int(nodes)), []).append(str(sample_id))
                    if len(shapes) > 1:
                        detail = ", ".join(f"(T={shape[0]},N={shape[1]}):{len(ids)}" for shape, ids in list(shapes.items())[:8])
                        error(
                            "STUDIO-SGV-FIXED-001",
                            "dataset_dir",
                            f"SimulGen requires one fixed (T,N) shape, but the HDF5 contains {len(shapes)} shapes: {detail}",
                            "Use a fixed-geometry dataset or separate incompatible meshes before training.",
                        )
        except OSError as exc:
            error(
                "STUDIO-SGV-DATA-003",
                "dataset_dir",
                f"Could not inspect the full SimulGen HDF5: {exc}",
                "Check that the file is a readable HDF5 dataset.",
            )
    if (
        sample_count is not None
        and param_path is not None
        and param_path.is_file()
        and str(values.get("lc_data_type", "csv")).lower() == "csv"
    ):
        try:
            with param_path.open("r", encoding="utf-8-sig", newline="") as handle:
                condition_rows = sum(1 for row in csv.reader(handle) if row)
            if condition_rows != sample_count:
                error(
                    "STUDIO-SGV-PARAM-002",
                    "param_dir",
                    f"Condition CSV has {condition_rows} rows, but the HDF5 has {sample_count} samples.",
                    "Provide exactly one condition row per sorted SimulGen sample ID.",
                )
        except OSError as exc:
            error(
                "STUDIO-SGV-PARAM-003",
                "param_dir",
                f"Could not read SimulGen condition CSV: {exc}",
                "Select a readable headerless CSV.",
            )
    return diagnostics


class PreflightFailure(Exception):
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        super().__init__("One or more pipeline steps failed preflight.")
        self.failures = failures


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        pid = int(process.pid)
        script = (
            f"$rootId={pid};"
            "$all=Get-CimInstance Win32_Process;"
            "function Stop-Desc([int]$id){"
            "$children=@($all|Where-Object {$_.ParentProcessId -eq $id});"
            "foreach($child in $children){Stop-Desc ([int]$child.ProcessId)};"
            "Stop-Process -Id $id -Force -ErrorAction SilentlyContinue"
            "};Stop-Desc $rootId"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
    else:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return


class StudioState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.registry = None
        self.settings = None
        self.start_error: str | None = None
        try:
            if SUITE_IMPORT_ERROR is not None:
                raise SUITE_IMPORT_ERROR
            self.registry = MethodRegistry(SUITE_ROOT)
            self.settings = LocalSettings.load(SUITE_ROOT)
        except Exception as exc:
            self.start_error = f"{type(exc).__name__}: {exc}"
        self._recover_jobs()

    def _recover_jobs(self) -> None:
        if not JOB_RUNTIME.is_dir():
            return
        for metadata_path in JOB_RUNTIME.glob("*/job.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                job_id = str(payload["id"])
                status = str(payload.get("status", "unknown"))
                if status in {"queued", "running"}:
                    status = "interrupted"
                    payload["status"] = status
                    payload["finished_at"] = utc_now()
                    payload["pid"] = None
                job = {
                    **payload,
                    "status": status,
                    "log_path": SUITE_ROOT / str(payload["log_path"]),
                    "metadata_path": metadata_path,
                }
                self.jobs[job_id] = job
                if payload.get("status") == "interrupted":
                    self._persist_job(job)
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    def require_suite(self) -> None:
        if self.start_error or self.registry is None or self.settings is None:
            raise RuntimeError(self.start_error or "AI-CAE4ALL suite modules are unavailable.")

    def models(self) -> list[dict[str, Any]]:
        self.require_suite()
        installation_report = self.registry.validate_installations()
        broken_repositories = {
            str(diag.message).split(": ", 1)[-1]
            for diag in installation_report.diagnostics
            if diag.severity is Severity.ERROR
        }
        payload: list[dict[str, Any]] = []
        for model_id in self.registry.model_ids:
            report = type(installation_report)()
            resolved = self.registry.resolve(model_id, report)
            if resolved is None:
                continue
            spec = resolved.spec
            required = {
                mode: sorted(spec.required_fields(model_id, mode))
                for mode in spec.valid_modes
            }
            healthy = (
                str(resolved.repository_root) not in broken_repositories
                and resolved.repository_root.is_dir()
                and resolved.entrypoint.is_file()
            )
            payload.append(
                {
                    "model": model_id,
                    "method": spec.display_name,
                    "spec_id": spec.spec_id,
                    "modes": list(spec.valid_modes),
                    "dataset_kind": spec.dataset_kind,
                    "repository": relative(resolved.repository_root),
                    "entrypoint": relative(resolved.entrypoint),
                    "known_keys": sorted(spec.known_keys),
                    "required": required,
                    "defaults": json_safe(spec.defaults),
                    "defaults_by_mode": json_safe(spec.defaults_by_mode),
                    "path_rules": [
                        {"field": rule.field, "kind": rule.kind.value, "modes": sorted(rule.modes)}
                        for rule in spec.path_rules
                    ],
                    "healthy": healthy,
                    "native_probe": spec.native_probe,
                }
            )
        return payload

    def health(self) -> dict[str, Any]:
        model_payload: list[dict[str, Any]] = []
        if not self.start_error:
            model_payload = self.models()
        return {
            "ok": self.start_error is None,
            "error": self.start_error,
            "suite_root": str(SUITE_ROOT),
            "frontend_root": str(Path(__file__).resolve().parent.parent),
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "models": len(model_payload),
            "healthy_models": sum(1 for model in model_payload if model["healthy"]),
            "time": utc_now(),
        }

    def save_config(self, text: str, label: str, purpose: str) -> Path:
        from studio_backend.paths import CONFIG_RUNTIME

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Configuration text is empty.")
        if len(text.encode("utf-8")) > MAX_BODY:
            raise ValueError("Configuration exceeds the 2 MiB Studio limit.")
        CONFIG_RUNTIME.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = CONFIG_RUNTIME / f"{stamp}-{slug(label)}-{slug(purpose)}-{uuid.uuid4().hex[:8]}.txt"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return path

    def preflight(
        self,
        text: str,
        *,
        label: str = "config",
        strict: bool = False,
        skip_filesystem: bool = False,
        skip_native: bool = False,
        skip_environment: bool = False,
        skip_dataset: bool = False,
    ) -> tuple[dict[str, Any], Path, Any]:
        self.require_suite()
        config_path = self.save_config(text, label, "preflight")
        result = run_preflight(
            config_path,
            suite_root=SUITE_ROOT,
            registry=self.registry,
            settings=self.settings,
            options=PreflightOptions(
                strict=strict,
                skip_filesystem=skip_filesystem,
                skip_native=skip_native,
                skip_environment=skip_environment,
                skip_dataset=skip_dataset,
            ),
        )
        report = result.report.to_dict(strict=strict)
        studio_diagnostics = studio_preflight_diagnostics(result)
        if studio_diagnostics:
            report["diagnostics"] = studio_diagnostics + report["diagnostics"]
            report["summary"]["errors"] += sum(1 for item in studio_diagnostics if item["severity"] == "error")
            report["summary"]["warnings"] += sum(1 for item in studio_diagnostics if item["severity"] == "warning")
            report["summary"]["notices"] += sum(1 for item in studio_diagnostics if item["severity"] == "notice")
        route = None
        if result.resolved is not None:
            route = {
                "model": result.resolved.model_id,
                "method": result.resolved.spec.display_name,
                "mode": result.mode,
                "repository": relative(result.resolved.repository_root),
                "entrypoint": relative(result.resolved.entrypoint),
                "python": str(result.python_executable) if result.python_executable else None,
                "command": result.command,
            }
        payload = {
            "ok": report["summary"]["errors"] == 0,
            "strict": strict,
            "config_path": relative(config_path),
            "report": report,
            "route": route,
            "resolved_paths": {key: relative(path) for key, path in result.resolved_paths.items()},
            "dataset_metadata": json_safe(result.dataset_metadata),
            "checkpoint_metadata": json_safe(result.checkpoint_metadata),
        }
        return payload, config_path, result

    def create_pipeline_job(
        self,
        steps: list[dict[str, Any]],
        *,
        label: str,
        strict: bool,
    ) -> dict[str, Any]:
        if not steps:
            raise ValueError("The pipeline has no executable model steps.")
        prepared: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for index, step in enumerate(steps):
            text = str(step.get("config", ""))
            step_label = str(step.get("label") or f"step-{index + 1}")
            payload, path, result = self.preflight(
                text,
                label=step_label,
                strict=strict,
                skip_filesystem=index > 0,
                skip_native=index > 0,
            )
            payload["deferred_dependency_checks"] = index > 0
            prepared.append({"label": step_label, "path": path, "preflight": payload, "result": result})
            if not payload["ok"]:
                failures.append({"step": index, "label": step_label, "preflight": payload})
        if failures:
            raise PreflightFailure(failures)

        JOB_RUNTIME.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOB_RUNTIME / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_id,
            "label": label or "AI-CAE4ALL pipeline",
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "pid": None,
            "current_step": 0,
            "total_steps": len(prepared),
            "step_label": None,
            "cancel_requested": False,
            "log_path": job_dir / "run.log",
            "metadata_path": job_dir / "job.json",
            "steps": [
                {
                    "label": item["label"],
                    "config_path": relative(item["path"]),
                    "route": item["preflight"]["route"],
                    "summary": item["preflight"]["report"]["summary"],
                }
                for item in prepared
            ],
        }
        with self.lock:
            self.jobs[job_id] = job
            self._persist_job(job)
        thread = threading.Thread(target=self._run_pipeline, args=(job_id, prepared, strict), daemon=True)
        thread.start()
        return self.public_job(job, include_log=True)

    def create_command_job(
        self,
        *,
        label: str,
        step_label: str,
        command: list[str],
        cwd: Path,
    ) -> dict[str, Any]:
        JOB_RUNTIME.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOB_RUNTIME / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_id,
            "label": label,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "pid": None,
            "current_step": 0,
            "total_steps": 1,
            "step_label": step_label,
            "cancel_requested": False,
            "log_path": job_dir / "run.log",
            "metadata_path": job_dir / "job.json",
            "steps": [{"label": step_label, "command": command, "cwd": relative(cwd)}],
        }
        with self.lock:
            self.jobs[job_id] = job
            self._persist_job(job)
        thread = threading.Thread(
            target=self._run_command_job,
            args=(job_id, command, cwd),
            daemon=True,
        )
        thread.start()
        return self.public_job(job, include_log=True)

    def _run_command_job(self, job_id: str, command: list[str], cwd: Path) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.update(status="running", started_at=utc_now(), current_step=1)
            self._persist_job(job)
        self._append_log(job, f"[studio] Command: {subprocess.list2cmdline(command)}\n\n")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        returncode = 127
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            with self.lock:
                self.processes[job_id] = process
                job["pid"] = process.pid
                self._persist_job(job)
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job, line)
            returncode = process.wait()
        except OSError as exc:
            self._append_log(job, f"[studio] Failed to start process: {type(exc).__name__}: {exc}\n")
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                job["pid"] = None
        with self.lock:
            if job["cancel_requested"]:
                status = "cancelled"
            elif returncode == 0:
                status = "completed"
            else:
                status = "failed"
            job.update(status=status, returncode=returncode, finished_at=utc_now())
            self._persist_job(job)
        self._append_log(job, f"\n[studio] Command job {status}.\n")

    def _persist_job(self, job: dict[str, Any]) -> None:
        payload = {
            key: json_safe(value)
            for key, value in job.items()
            if key not in {"log_path", "metadata_path"}
        }
        payload["log_path"] = relative(job["log_path"])
        job["metadata_path"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _append_log(self, job: dict[str, Any], text: str) -> None:
        with job["log_path"].open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)

    def _run_pipeline(self, job_id: str, prepared: list[dict[str, Any]], strict: bool) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job.update(status="running", started_at=utc_now())
            self._persist_job(job)
        self._append_log(job, f"[studio] Starting {job['label']} with {len(prepared)} executable step(s).\n")
        final_code = 0
        for index, item in enumerate(prepared, start=1):
            with self.lock:
                if job["cancel_requested"]:
                    final_code = -1
                    break
                job.update(current_step=index, step_label=item["label"])
                self._persist_job(job)
            command = [
                sys.executable,
                str(SUITE_ROOT / "AI_CAE4ALL_main.py"),
                "--config",
                str(item["path"]),
                "--no-color",
            ]
            if strict:
                command.append("--strict")
            self._append_log(job, f"\n[studio] Step {index}/{len(prepared)}: {item['label']}\n")
            self._append_log(job, f"[studio] Command: {subprocess.list2cmdline(command)}\n\n")
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            try:
                process = subprocess.Popen(
                    command,
                    cwd=SUITE_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
            except OSError as exc:
                self._append_log(job, f"[studio] Failed to start process: {type(exc).__name__}: {exc}\n")
                final_code = 127
                break
            with self.lock:
                self.processes[job_id] = process
                job["pid"] = process.pid
                self._persist_job(job)
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job, line)
            final_code = process.wait()
            with self.lock:
                self.processes.pop(job_id, None)
                job["pid"] = None
                self._persist_job(job)
            if final_code != 0:
                self._append_log(job, f"\n[studio] Step failed with exit code {final_code}.\n")
                break
        with self.lock:
            if job["cancel_requested"]:
                status = "cancelled"
            elif final_code == 0:
                status = "completed"
            else:
                status = "failed"
            job.update(status=status, returncode=final_code, finished_at=utc_now(), pid=None)
            self._persist_job(job)
        self._append_log(job, f"[studio] Pipeline {status}.\n")

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job["cancel_requested"] = True
            process = self.processes.get(job_id)
            self._persist_job(job)
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
        return self.public_job(job, include_log=True)

    def public_job(self, job: dict[str, Any], *, include_log: bool = False) -> dict[str, Any]:
        payload = {
            key: json_safe(value)
            for key, value in job.items()
            if key not in {"log_path", "metadata_path"}
        }
        payload["log_path"] = relative(job["log_path"])
        if include_log:
            try:
                text = job["log_path"].read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            payload["log"] = text[-120_000:]
        return payload

    def list_jobs(self) -> list[dict[str, Any]]:
        with self.lock:
            jobs = [self.public_job(job) for job in self.jobs.values()]
        jobs.sort(key=lambda item: item["created_at"], reverse=True)
        return jobs

    def get_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return self.public_job(job, include_log=True)


STATE = StudioState()
