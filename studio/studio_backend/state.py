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
import io
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from studio_backend.analysis import (
    export_artifact,
    run_field_evaluation,
    run_model_comparison,
    run_optimization,
    write_candidate_table,
    write_optimize_summary_table,
    write_screening_table,
)
from studio_backend.prediction_preview import (
    invalidate_prediction_runs,
    outputs_since,
    scan_output_dir,
)
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
    resolve_native_path,
    run_preflight,
)


# Mirrors _LC_MODES in cae_suite/specs/simulgenvae.py: the modes that run the
# latent conditioner and therefore read its condition source.
SGV_LC_MODES = frozenset({"train", "train_lc", "reconstruct"})

# What a tqdm repaint looks like: `45%|####5     | 18/40` or a rate such as
# `7.1it/s`, `?it/s`, `2.30s/it` (an unsized bar shows only the rate).
PROGRESS_FRAME_RE = re.compile(r"\d+%\||[0-9?.]+\s*(?:it|s)/(?:s|it)\b")


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

    # param_dir is read only by the file-backed conditioner sources (csv/image)
    # in the modes that run the LC, exactly as SGV-LC-002 in
    # cae_suite/specs/simulgenvae.py scopes it. `lc_data_type hdf5` takes the
    # conditions from dataset_dir, so a stale param_dir left in such a config
    # is dead and must not block the launch.
    mode = str(values.get("mode", "")).lower()
    lc_data_type = str(values.get("lc_data_type", "")).lower()
    param_used = mode in SGV_LC_MODES and lc_data_type in {"csv", "image"}
    param_value = values.get("param_dir") if param_used else None
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
        and lc_data_type == "csv"
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


# The job drawer gets only a recent tail; read_job_log returns the full log.
LOG_TAIL_CHARS = 120_000


class PreflightFailure(Exception):
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        super().__init__("One or more pipeline steps failed preflight.")
        self.failures = failures


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    terminate_pid_tree(int(process.pid))


def terminate_pid_tree(pid: int) -> None:
    """Kill a process and its descendants by pid alone.

    Split out of terminate_process_tree so a job adopted after a Studio restart
    -- whose Popen object died with the previous server -- can still be
    cancelled. Both launch paths start the child as its own process group
    (CREATE_NEW_PROCESS_GROUP / start_new_session), so on POSIX the group id is
    the pid and killpg reaches the whole tree without a parent-child link.
    """
    if os.name == "nt":
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
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return


_CANVAS_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _canvas_id(pipeline: Any) -> str:
    # fullmatch, not match with "$": "$" also matches before a trailing
    # newline, which let "abc\n" through. A non-string is never an id -- str()
    # would have turned 0 into "0".
    value = pipeline.get("canvas_id") if isinstance(pipeline, dict) else None
    return value if isinstance(value, str) and _CANVAS_ID.fullmatch(value) else ""


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

    @staticmethod
    def _process_alive(pid: Any) -> bool:
        """Is the recorded launcher process still running?

        Recovery used to declare every job it found in a running state
        "interrupted" on sight. That is right after a crash and badly wrong
        otherwise, because `STATE = StudioState()` runs at *module import*: any
        second process that so much as imports studio_backend.state -- a test, a
        one-off script, a second Studio instance -- silently rewrote the live
        server's job records. Observed exactly that: a 60-epoch training run was
        stamped interrupted (pid None, finished_at now) while it went on
        training, because an unrelated import check ran in another shell.
        """
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        if os.name == "nt":
            try:
                completed = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
                    capture_output=True, text=True, timeout=10, check=False,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            return f'"{pid}"' in (completed.stdout or "")
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, ValueError):
            return False
        except PermissionError:
            return True
        return True

    def _recover_jobs(self) -> None:
        if not JOB_RUNTIME.is_dir():
            return
        for metadata_path in JOB_RUNTIME.glob("*/job.json"):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                job_id = str(payload["id"])
                status = str(payload.get("status", "unknown"))
                if status in {"queued", "running"} and not self._process_alive(payload.get("pid")):
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
                elif status in {"queued", "running"}:
                    self._adopt_running_job(job_id, int(payload["pid"]))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    ADOPT_POLL_SECONDS = 5.0

    def _adopt_running_job(self, job_id: str, pid: int) -> None:
        """Watch a job whose native process outlived the previous Studio server.

        Recovery used to leave such a job "running" forever: nothing was left
        to wait on its process, so it never got a finished_at, and cancel could
        not reach it because its Popen died with the old server. The watcher
        polls the recorded pid and finalises the record once it is gone.
        """
        thread = threading.Thread(
            target=self._watch_adopted_job, args=(job_id, pid), daemon=True
        )
        thread.start()

    def _watch_adopted_job(self, job_id: str, pid: int) -> None:
        while self._process_alive(pid):
            time.sleep(self.ADOPT_POLL_SECONDS)
        # Give a server that still owns this job (StudioState is built at
        # import in any process) time to write its own final record first.
        time.sleep(1.0)
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.get("status") not in {"queued", "running"}:
                return
            try:
                on_disk = json.loads(job["metadata_path"].read_text(encoding="utf-8"))
            except (OSError, ValueError):
                on_disk = {}
            if on_disk and (
                on_disk.get("status") not in {"queued", "running"} or on_disk.get("pid") != pid
            ):
                # Another live process owns this job and has moved it on;
                # take its record rather than overwrite it with a guess.
                job.update({key: value for key, value in on_disk.items() if key != "log_path"})
                return
            cancelled = bool(job.get("cancel_requested"))
            job.update(
                status="cancelled" if cancelled else "interrupted",
                finished_at=utc_now(),
                returncode=None,
                pid=None,
                note=(
                    "The Studio server restarted while this job ran. Its process "
                    "(pid {0}) was adopted and has since exited, but its exit code "
                    "cannot be recovered and any later pipeline steps were not run."
                ).format(pid),
            )
            try:
                self._persist_job(job)
            except OSError:
                pass
        try:
            self._append_log(job, "\n[studio] Adopted process {0} exited; exit code unknown.\n".format(pid))
        except OSError:
            pass

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
            "studio_root": str(Path(__file__).resolve().parent.parent),
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
        config_path = self.save_config(text, label, "preflight")
        payload, result = self._preflight_config_path(
            config_path,
            strict=strict,
            skip_filesystem=skip_filesystem,
            skip_native=skip_native,
            skip_environment=skip_environment,
            skip_dataset=skip_dataset,
        )
        return payload, config_path, result

    def _preflight_config_path(
        self,
        config_path: Path,
        *,
        strict: bool = False,
        skip_filesystem: bool = False,
        skip_native: bool = False,
        skip_environment: bool = False,
        skip_dataset: bool = False,
    ) -> tuple[dict[str, Any], Any]:
        """Validate the exact config file that a native process will receive.

        ``preflight`` is the text-entry API and therefore serializes first. The
        pipeline launch gate calls this path-based form so it re-reads the saved
        snapshot immediately before launch instead of validating a second copy
        or stale browser text.
        """
        self.require_suite()
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
        return payload, result

    # Blocks that do analysis rather than launching a native process. They were
    # drawn on the canvas, typed, and edge-connected, but "Run pipeline" only
    # ever emitted launcher steps -- so the graph depicted a flow the Run button
    # did not run and the user had to re-enter every path in a side workspace.
    ANALYSIS_ACTIONS = {
        "evaluation": run_field_evaluation,
        "export": export_artifact,
        "comparison": run_model_comparison,
        # Without this the Optimization block was dead on the canvas: the graph
        # drew it, the inspector configured it, the backend implemented it, and
        # the pipeline ran to "completed" with the step silently dropped.
        "optimization": run_optimization,
    }

    @staticmethod
    def _resolve_step_reference(
        value: Any, produced: dict[str, str], produced_dirs: dict[str, str] | None = None
    ) -> Any:
        """Substitute `@results:<node_id>` with what that step actually wrote.

        An analysis step's input cannot be known when the pipeline is submitted:
        inference writes to an epoch-numbered directory that only exists once it
        has run. So the frontend sends a reference to the producing block and the
        substitution happens here, at execution time, against the results the
        earlier step recorded.

        `@artifacts:<node_id>` names the step's whole output directory instead
        (a CAD Generator's STLs, report and figures). A step that recorded no
        directory falls back to its results file, so the reference never
        resolves to less than `@results:` would.
        """
        if not isinstance(value, str):
            return value
        if value.startswith("@results:"):
            return produced.get(value[len("@results:"):], "")
        if value.startswith("@artifacts:"):
            node_id = value[len("@artifacts:"):]
            return (produced_dirs or {}).get(node_id) or produced.get(node_id, "")
        return value

    def _run_analysis_step(
        self,
        job: dict[str, Any],
        index: int,
        item: dict[str, Any],
        produced: dict[str, str],
        produced_dirs: dict[str, str] | None = None,
    ) -> int:
        action = item.get("action", "")
        handler = self.ANALYSIS_ACTIONS.get(action)
        if handler is None:
            self._append_log(job, "[studio] Unknown analysis action: {0}.\n".format(action))
            return 2
        payload = {
            key: self._resolve_step_reference(value, produced, produced_dirs)
            for key, value in (item.get("payload") or {}).items()
        }
        missing = [
            key for key, value in payload.items()
            if (key.endswith("_path") or key == "path") and not value
        ]
        if missing:
            self._append_log(
                job,
                "[studio] {0} skipped: {1} is empty because the block it reads from produced no output.\n".format(
                    item["label"], ", ".join(missing)
                ),
            )
            return 2
        self._append_log(job, "[studio] Running {0} ({1}).\n".format(item["label"], action))
        try:
            result = handler(payload)
        except (ValueError, OSError, KeyError) as exc:
            self._append_log(job, "[studio] {0} failed: {1}\n".format(item["label"], exc))
            return 2
        except Exception as exc:
            # Anything else (a TypeError from a malformed field, say) used to
            # escape and kill the runner thread, leaving the job "running"
            # forever. It is a failed step, with the traceback in the log.
            self._append_log(
                job,
                "[studio] {0} failed: {1}: {2}\n{3}".format(
                    item["label"], type(exc).__name__, exc, traceback.format_exc()
                ),
            )
            return 2
        written = result.get("report_path") or result.get("path") or ""
        with self.lock:
            steps = job.get("steps") or []
            if index - 1 < len(steps):
                steps[index - 1]["results"] = written
                steps[index - 1]["analysis"] = {
                    key: value
                    for key, value in result.items()
                    if key in {"evaluated_samples", "aggregate", "per_sample_csv", "report_path", "path", "size"}
                }
            self._persist_job(job)
        if item.get("node_id"):
            produced[item["node_id"]] = written
        self._append_log(
            job,
            "[studio] {0} wrote {1}.\n".format(item["label"], written or "no artifact"),
        )
        return 0

    def create_pipeline_job(
        self,
        steps: list[dict[str, Any]],
        *,
        label: str,
        strict: bool,
        target_node_id: str = "",
        pipeline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not steps:
            raise ValueError("The pipeline has no executable model steps.")
        prepared: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        launcher_seen = 0
        for index, step in enumerate(steps):
            step_label = str(step.get("label") or f"step-{index + 1}")
            node_id = str(step.get("node_id", "")).strip()[:200]
            node_type = str(step.get("node_type", "")).strip()[:200]
            if str(step.get("kind", "launcher")) == "analysis":
                # No config text and nothing for the launcher's preflight to
                # check; these run in-process against the Studio APIs. Their
                # shape is still checked here, so a bad step is a 400 now
                # rather than a runner crash minutes into the pipeline.
                action = str(step.get("action", ""))
                if action not in self.ANALYSIS_ACTIONS:
                    raise ValueError(
                        "Analysis step {0!r} has unknown action {1!r}; expected one of {2}.".format(
                            step_label, action, ", ".join(sorted(self.ANALYSIS_ACTIONS))
                        )
                    )
                step_payload = step.get("payload")
                if step_payload is None:
                    step_payload = {}
                if not isinstance(step_payload, dict):
                    raise ValueError(
                        "Analysis step {0!r} payload must be a JSON object, not {1}.".format(
                            step_label, type(step_payload).__name__
                        )
                    )
                prepared.append(
                    {
                        "kind": "analysis",
                        "label": step_label,
                        "action": action,
                        "payload": step_payload,
                        "node_id": node_id,
                        "node_type": node_type,
                    }
                )
                continue
            text = str(step.get("config", ""))
            # Dependency checks are skipped for every launcher step after the
            # first, so this must count launcher steps -- not the mixed index,
            # which would wrongly re-run full checks after an analysis step.
            deferred = launcher_seen > 0
            launcher_seen += 1
            payload, path, result = self.preflight(
                text,
                label=step_label,
                strict=strict,
                skip_filesystem=deferred,
                skip_native=deferred,
            )
            payload["deferred_dependency_checks"] = deferred
            prepared.append(
                {
                    "kind": "launcher",
                    "label": step_label,
                    "path": path,
                    "preflight": payload,
                    "result": result,
                    "node_id": node_id,
                    "node_type": node_type,
                }
            )
            if not payload["ok"]:
                failures.append({"step": index, "label": step_label, "preflight": payload})
        if failures:
            raise PreflightFailure(failures)

        JOB_RUNTIME.mkdir(parents=True, exist_ok=True)
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOB_RUNTIME / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # The graph that produced the run is stored beside it so the exact
        # pipeline can be reloaded later. It lives in its own file rather than
        # in job.json so listing every job stays cheap.
        has_pipeline = False
        if isinstance(pipeline, dict) and pipeline.get("nodes"):
            try:
                (job_dir / "pipeline.json").write_text(
                    json.dumps(pipeline, indent=1), encoding="utf-8"
                )
                has_pipeline = True
            except (OSError, TypeError, ValueError):
                has_pipeline = False
        job = {
            "id": job_id,
            "has_pipeline": has_pipeline,
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
            "target_node_id": str(target_node_id or "").strip()[:200],
            # Which canvas launched the run. Node ids alone cannot say: every
            # canvas made from one template shares them, so a fresh copy of a
            # template used to claim another pipeline's live job.
            "canvas_id": _canvas_id(pipeline),
            "log_path": job_dir / "run.log",
            "metadata_path": job_dir / "job.json",
            "steps": [
                {
                    "label": item["label"],
                    "kind": item.get("kind", "launcher"),
                    "node_id": item["node_id"],
                    "node_type": item["node_type"],
                    "config_path": relative(item["path"]) if item.get("path") else "",
                    "route": (item.get("preflight") or {}).get("route"),
                    "summary": (item.get("preflight") or {}).get("report", {}).get("summary"),
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
        returncode = 127
        crashed = False
        try:
            returncode = self._execute_command_job(job, job_id, command, cwd)
        except Exception:
            # Without this an unexpected error killed the runner thread and
            # left the job "running" with no finished_at, uncancellable.
            crashed = True
            returncode = 1
            self._log_runner_crash(job)
        finally:
            status = self._finalise_job(job_id, job, returncode, crashed=crashed)
        self._append_log_quietly(job, f"\n[studio] Command job {status}.\n")

    def _execute_command_job(self, job: dict[str, Any], job_id: str, command: list[str], cwd: Path) -> int:
        with self.lock:
            job.update(status="running", started_at=utc_now(), current_step=1)
            self._persist_job(job)
        self._append_log(job, f"[studio] Command: {subprocess.list2cmdline(command)}\n\n")
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        returncode = 127
        with self.lock:
            if job["cancel_requested"]:
                self._append_log(job, "[studio] Cancelled before the process started.\n")
                return -1
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # Binary pipe on purpose: text=True turns on universal newlines,
                # which rewrites tqdm's carriage returns into real newlines and
                # makes an in-place repaint indistinguishable from a new line.
                # _collapse_progress needs to see the '\r' to fold them away.
                bufsize=0,
                env=env,
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
            self._register_process(job_id, job, process)
            assert process.stdout is not None
            stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
            for line in self._collapse_progress(stream):
                self._append_log(job, line)
            returncode = process.wait()
        except OSError as exc:
            self._append_log(job, f"[studio] Failed to start process: {type(exc).__name__}: {exc}\n")
        finally:
            with self.lock:
                self.processes.pop(job_id, None)
                job["pid"] = None
        return returncode

    def _register_process(self, job_id: str, job: dict[str, Any], process: subprocess.Popen[Any]) -> None:
        """Publish a just-started process to cancel_job, honouring a late cancel.

        The cancel flag is checked just before Popen, but a cancel can still land
        between that check and this registration, when cancel_job finds no
        process to kill. Re-checking under the same lock closes that window.
        """
        with self.lock:
            self.processes[job_id] = process
            job["pid"] = process.pid
            self._persist_job(job)
            cancelled = bool(job["cancel_requested"])
        if cancelled:
            terminate_process_tree(process)

    def _log_runner_crash(self, job: dict[str, Any]) -> None:
        self._append_log_quietly(
            job,
            "\n[studio] The Studio job runner failed unexpectedly:\n" + traceback.format_exc(),
        )

    def _append_log_quietly(self, job: dict[str, Any], text: str) -> None:
        try:
            self._append_log(job, text)
        except OSError:
            pass

    def _finalise_job(self, job_id: str, job: dict[str, Any], code: int, *, crashed: bool) -> str:
        """Record the terminal status; runs from a `finally`, so it never raises.

        A cancel only counts when it interrupted something: a pipeline whose
        last step had already succeeded when the cancel arrived is 'completed'.
        """
        process = None
        with self.lock:
            process = self.processes.pop(job_id, None)
        if process is not None:
            # Only reachable when the runner crashed while a child was running.
            try:
                terminate_process_tree(process)
            except Exception:
                pass
        with self.lock:
            if crashed:
                status = "failed"
            elif code == 0:
                status = "completed"
            elif job.get("cancel_requested"):
                status = "cancelled"
            else:
                status = "failed"
            job.update(status=status, returncode=code, finished_at=utc_now(), pid=None)
            try:
                self._persist_job(job)
            except Exception:
                pass
        return status

    def _record_step_results(
        self,
        job: dict[str, Any],
        index: int,
        item: dict[str, Any],
        step_started: float,
        produced: dict[str, str] | None = None,
        produced_dirs: dict[str, str] | None = None,
    ) -> None:
        """Pin a finished step to the prediction directory it just wrote.

        For a *training* step the output directory genuinely cannot be guessed:
        the periodic prediction dumps land under `log_file_dir`'s run directory
        with the epoch in the path, and the epoch is only known to the training
        loop. Asking the filesystem what appeared while the step was running --
        under this step's repository for the old layout, and under the suite's
        `output/` root where every native now writes -- answers it exactly, for
        one directory scan (see prediction_preview.outputs_since).

        An *inference* step is not like that. `inference_output_dir` is a path
        rule the user sets and preflight resolves, so the repository scan is not
        only unnecessary there, it is wrong: point the key anywhere outside the
        method repo -- which the config sheet invites -- and the scan finds
        nothing, every downstream analysis block is skipped for "no output", and
        the pipeline fails after a full training run. Prefer what preflight
        resolved, and fall back to the scan only when the key is absent.
        """
        preflight = item.get("preflight") or {}
        repository = (preflight.get("route") or {}).get("repository")
        resolved = preflight.get("resolved_paths") or {}

        def configured_dir(key: str) -> Path | None:
            """Where this step was told to write, resolved to a real directory.

            `resolved_paths` alone is not enough: every launcher step after the
            first is preflighted with `skip_filesystem=True` (dependency checks
            are deferred), and path resolution belongs to that layer -- so for
            step 2 onwards the dict is empty. Reading the key back out of the
            config text the step actually ran keeps this working for the
            inference and generation steps, which are never step 1.

            Native processes run with cwd set to their method repository, which
            is what the `../output/...` form in these configs is relative to.
            """
            value = str(resolved.get(key) or "").strip()
            if value:
                return SUITE_ROOT / value
            config_path = item.get("path")
            if not config_path or not repository:
                return None
            try:
                text = Path(config_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            for line in text.splitlines():
                parts = line.split("%", 1)[0].split()
                if len(parts) >= 2 and parts[0] == key:
                    # Resolve exactly as launcher preflight and the native
                    # config loaders do: relative to the method repository the
                    # step runs in, so ``../../output/...`` reaches the single
                    # suite-wide output/ directory.
                    return resolve_native_path(parts[1], SUITE_ROOT / repository)
            return None
        # A CAD generation step writes STLs and a metadata JSON, not prediction
        # HDF5s, so the prediction scan finds nothing for it and the block's
        # output edge stays empty. Publish the candidate table instead -- that is
        # what the Optimization block downstream actually reads.
        generated_dir = configured_dir("output_dir") if item.get("node_type") == "run.cad_generator" else None
        if generated_dir is not None:
            # `mode optimize` writes summary.json, not sample_*_meta.json --
            # try its tables first so a closed-loop run's winner/baseline/typical
            # comparison shows up instead of silently falling through empty. A
            # screen (opt_budget 0) publishes every screened design instead.
            table = (write_screening_table(generated_dir)
                     or write_optimize_summary_table(generated_dir)
                     or write_candidate_table(generated_dir))
            # The table is what the Optimization block reads, but it is not what
            # the run *is*: the geometry, report and figures sit beside it. The
            # directory is recorded separately so "Open results" can show the
            # STLs (the shared viewer rejects a CSV) and Export can package the
            # whole run rather than the one table.
            results_dir = relative(generated_dir) if generated_dir.is_dir() else ""
            with self.lock:
                steps = job.get("steps") or []
                if index - 1 < len(steps):
                    if table:
                        steps[index - 1]["results"] = table["path"]
                        steps[index - 1]["results_samples"] = table["rows"]
                    if results_dir:
                        steps[index - 1]["results_dir"] = results_dir
                self._persist_job(job)
            if item.get("node_id"):
                if produced is not None and table:
                    produced[item["node_id"]] = table["path"]
                if produced_dirs is not None and results_dir:
                    produced_dirs[item["node_id"]] = results_dir
            if table:
                self._append_log(
                    job,
                    "[studio] Step {0} tabulated {1} design(s) to {2}.\n".format(
                        index, table["rows"], table["path"]
                    ),
                )
            # Terminal either way: a generator's outputs are exactly its
            # output_dir. The repository-wide scan below would pin whatever
            # unrelated file happened to change during the step.
            return
        configured = configured_dir("inference_output_dir")
        found: list[dict[str, Any]] = []
        if configured is not None:
            try:
                found = scan_output_dir(configured)
            except OSError:
                found = []
        if not found:
            if not repository:
                return
            try:
                found = outputs_since(SUITE_ROOT / repository, step_started)
            except OSError:
                return
            # "Written since this step started" is only exact while nothing else
            # runs. A 21 h SDFFlow train step was pinned to the 480 HI-MGN
            # rollouts another job wrote into output/ meanwhile, and its output
            # edge then carried them. Keep what lies under a directory this
            # step's own config writes to; with no such path, nothing narrows it.
            roots = self._own_output_roots(item, repository, resolved)
            if roots and found:
                mine = [
                    entry for entry in found
                    if any(root == (SUITE_ROOT / entry["path"]).resolve()
                           or root in (SUITE_ROOT / entry["path"]).resolve().parents
                           for root in roots)
                ]
                if not mine:
                    self._append_log(
                        job,
                        "[studio] Step {0}: {1} prediction folder(s) changed during this step "
                        "outside its own output paths (newest: {2}); not attributed to it.\n".format(
                            index, len(found), found[0]["path"]
                        ),
                    )
                found = mine
        if not found:
            return
        with self.lock:
            steps = job.get("steps") or []
            if index - 1 < len(steps):
                steps[index - 1]["results"] = found[0]["path"]
                steps[index - 1]["results_samples"] = found[0]["samples"]
            self._persist_job(job)
        if produced is not None and item.get("node_id"):
            produced[item["node_id"]] = found[0]["path"]
        # New results exist; the run catalog must not keep serving the old list.
        invalidate_prediction_runs()
        self._append_log(
            job,
            "[studio] Step {0} wrote {1} result file(s) to {2}.\n".format(
                index, found[0]["samples"], found[0]["path"]
            ),
        )

    @staticmethod
    def _own_output_roots(item: dict[str, Any], repository: str, resolved: dict[str, Any]) -> list[Path]:
        """The run directories a step's own config names under an output root.

        Every path value is taken -- `log_file_dir`, `modelpath`, `output_dir`,
        the stage-prefixed SDFFlow/SimulGen-VAE keys -- as its directory, or its
        parent when it names a file (train.log, model.pth). Inputs under
        dataset/ never qualify. Paths come from what preflight resolved and from
        the config text the step ran, since steps after the first skip the
        filesystem layer and arrive with no resolved paths.
        """
        values = [SUITE_ROOT / str(value) for value in resolved.values() if str(value or "").strip()]
        config_path = item.get("path")
        if config_path:
            try:
                text = Path(config_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            for line in text.splitlines():
                parts = line.split("%", 1)[0].split()
                if len(parts) >= 2 and ("/" in parts[1] or "\\" in parts[1]):
                    values.append(resolve_native_path(parts[1].rstrip(","), SUITE_ROOT / repository))
        output_roots = [SUITE_ROOT / "output", SUITE_ROOT / "outputs",
                        SUITE_ROOT / repository / "output", SUITE_ROOT / repository / "outputs"]
        roots: list[Path] = []
        for value in values:
            path = value.resolve()
            directory = path.parent if path.suffix else path
            if any(directory == root or root in directory.parents for root in output_roots):
                if directory not in roots:
                    roots.append(directory)
        return roots

    def _persist_job(self, job: dict[str, Any]) -> None:
        payload = {
            key: json_safe(value)
            for key, value in job.items()
            if key not in {"log_path", "metadata_path"}
        }
        payload["log_path"] = relative(job["log_path"])
        job["metadata_path"].write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _collapse_progress(stream: Any, keep_every: int = 40) -> Any:
        """Yield the child's output with tqdm's in-place repaints collapsed.

        Python's text-mode iteration splits on universal newlines, and that
        includes the bare `\\r` tqdm uses to repaint a bar in place. Every frame
        therefore arrived as its own line: a 3-epoch ex9 run wrote 5 546 lines of
        which 200 were real content (3.6%), and the drawer showed 17 useful lines
        out of 1 549. A terminal collapses those frames; so does this.

        A progress line is still emitted every `keep_every` frames, so a long
        epoch does not look frozen and the log keeps a coarse trace of it.

        Only a blank or bar-shaped piece is a repaint. Under DDP another rank's
        `\\r`+frame can land between rank 0's epoch text and its newline, and
        that text, now ending in `\\r`, is real content that must survive.
        """
        pending = 0
        for line in stream:
            if line.endswith("\r") or (not line.endswith("\n") and "\r" in line):
                text = line.replace("\r", "")
                if text.strip() and not PROGRESS_FRAME_RE.search(text):
                    yield text + "\n"
                    continue
                pending += 1
                if pending % keep_every:
                    continue
                yield line.replace("\r", "") + "\n"
                continue
            if pending:
                pending = 0
            yield line

    def _append_log(self, job: dict[str, Any], text: str) -> None:
        with job["log_path"].open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(text)

    def _run_pipeline(self, job_id: str, prepared: list[dict[str, Any]], strict: bool) -> None:
        with self.lock:
            job = self.jobs[job_id]
        final_code = 1
        crashed = False
        try:
            final_code = self._execute_pipeline(job_id, job, prepared, strict)
        except Exception:
            # Without this an analysis/bookkeeping error killed the runner
            # thread and the job stayed "running" forever, uncancellable.
            crashed = True
            final_code = 1
            self._log_runner_crash(job)
        finally:
            status = self._finalise_job(job_id, job, final_code, crashed=crashed)
        self._append_log_quietly(job, f"[studio] Pipeline {status}.\n")

    def _execute_pipeline(
        self, job_id: str, job: dict[str, Any], prepared: list[dict[str, Any]], strict: bool
    ) -> int:
        with self.lock:
            job.update(status="running", started_at=utc_now())
            self._persist_job(job)
        self._append_log(job, f"[studio] Starting {job['label']} with {len(prepared)} executable step(s).\n")
        final_code = 0
        # node_id -> whatever that step wrote, so a later analysis step can name
        # its input by block instead of by a path nobody can know up front.
        produced: dict[str, str] = {}
        # node_id -> the directory that step wrote, for `@artifacts:` references.
        produced_dirs: dict[str, str] = {}
        for index, item in enumerate(prepared, start=1):
            with self.lock:
                if job["cancel_requested"]:
                    final_code = -1
                    break
                job.update(current_step=index, step_label=item["label"])
                self._persist_job(job)
            if item.get("kind") == "analysis":
                final_code = self._run_analysis_step(job, index, item, produced, produced_dirs)
                if final_code != 0:
                    break
                continue
            self._append_log(job, f"\n[studio] Step {index}/{len(prepared)}: {item['label']}\n")
            self._append_log(
                job,
                "[studio] Launch preflight (exact saved config): {0}\n".format(item["path"]),
            )
            checked_at = utc_now()
            try:
                # Submission-time checks deliberately defer downstream paths
                # that an upstream step has not produced yet. At this boundary
                # those dependencies must exist: validate every layer against
                # the exact serialized file, then launch the command returned by
                # that same authoritative result.
                payload, result = self._preflight_config_path(
                    Path(item["path"]),
                    strict=strict,
                    skip_filesystem=False,
                    skip_native=False,
                    skip_environment=False,
                    skip_dataset=False,
                )
            except Exception as exc:
                diagnostic = {
                    "code": "STUDIO-LAUNCH-PREFLIGHT-001",
                    "severity": "error",
                    "original_severity": "error",
                    "message": f"Launch preflight could not run: {type(exc).__name__}: {exc}",
                    "field": None,
                    "hint": "The native process was not started.",
                }
                payload = {
                    "ok": False,
                    "strict": strict,
                    "config_path": relative(Path(item["path"])),
                    "report": {
                        "summary": {"errors": 1, "warnings": 0, "notices": 0},
                        "diagnostics": [diagnostic],
                    },
                    "route": None,
                    "resolved_paths": {},
                    "dataset_metadata": {},
                    "checkpoint_metadata": {},
                }
                result = None

            if payload["ok"] and (
                result is None or result.resolved is None or not result.command
            ):
                diagnostic = {
                    "code": "STUDIO-LAUNCH-PREFLIGHT-002",
                    "severity": "error",
                    "original_severity": "error",
                    "message": "Launch preflight passed without producing a native command.",
                    "field": None,
                    "hint": "Check the selected model route and method installation.",
                }
                payload["ok"] = False
                payload["report"]["diagnostics"].append(diagnostic)
                payload["report"]["summary"]["errors"] += 1

            # Replace the deferred submission result so downstream artifact
            # discovery sees the fully resolved output paths from this gate.
            item["preflight"] = payload
            item["result"] = result
            summary = payload["report"]["summary"]
            self._append_log(
                job,
                "[studio] Launch preflight {0}: {1} error(s), {2} warning(s), {3} notice(s).\n".format(
                    "passed" if payload["ok"] else "failed",
                    summary.get("errors", 0),
                    summary.get("warnings", 0),
                    summary.get("notices", 0),
                ),
            )
            for diagnostic in payload["report"].get("diagnostics", []):
                if diagnostic.get("severity") not in {"error", "warning"}:
                    continue
                self._append_log(
                    job,
                    "[studio] [{0}] {1}\n".format(
                        diagnostic.get("code", "PREFLIGHT"), diagnostic.get("message", "")
                    ),
                )
            with self.lock:
                steps = job.get("steps") or []
                if index - 1 < len(steps):
                    steps[index - 1]["route"] = payload.get("route")
                    steps[index - 1]["summary"] = summary
                    steps[index - 1]["launch_preflight"] = {
                        **payload,
                        "checked_at": checked_at,
                    }
                if not payload["ok"]:
                    job["diagnostics"] = [
                        {
                            **diagnostic,
                            "nodeId": item.get("node_id", ""),
                            "stepLabel": item["label"],
                        }
                        for diagnostic in payload["report"].get("diagnostics", [])
                        if diagnostic.get("severity") == "error"
                    ]
                self._persist_job(job)
            if not payload["ok"]:
                self._append_log(job, "[studio] Native process was not started.\n")
                final_code = 2
                break

            assert result is not None and result.resolved is not None
            command = list(result.command)
            command_cwd = result.resolved.repository_root
            self._append_log(job, f"[studio] Command: {subprocess.list2cmdline(command)}\n\n")
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Filesystem mtimes have coarse resolution, so start the window a
            # second early rather than miss a fast step's own output.
            step_started = time.time() - 1
            with self.lock:
                # The launch preflight can take a while (it probes the method
                # venv); a cancel that arrived during it must stop the launch
                # here, not be discovered after the step has run to completion.
                if job["cancel_requested"]:
                    self._append_log(job, "[studio] Cancelled before the native process started.\n")
                    final_code = -1
                    break
            try:
                process = subprocess.Popen(
                    command,
                    cwd=command_cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    # Binary, so _collapse_progress can still see tqdm's '\r'.
                    bufsize=0,
                    env=env,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
            except OSError as exc:
                self._append_log(job, f"[studio] Failed to start process: {type(exc).__name__}: {exc}\n")
                final_code = 127
                break
            self._register_process(job_id, job, process)
            assert process.stdout is not None
            stream = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
            for line in self._collapse_progress(stream):
                self._append_log(job, line)
            final_code = process.wait()
            with self.lock:
                self.processes.pop(job_id, None)
                job["pid"] = None
                self._persist_job(job)
            if final_code == 0:
                self._record_step_results(job, index, item, step_started, produced, produced_dirs)
            if final_code != 0:
                self._append_log(job, f"\n[studio] Step failed with exit code {final_code}.\n")
                break
        return final_code

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            job["cancel_requested"] = True
            process = self.processes.get(job_id)
            orphan_pid = job.get("pid") if process is None and job.get("status") in {"queued", "running"} else None
            self._persist_job(job)
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
        elif orphan_pid and self._process_alive(orphan_pid):
            # A job adopted after a restart has a pid but no Popen; its
            # watcher finalises the record once the tree is gone.
            terminate_pid_tree(int(orphan_pid))
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
            payload["log_truncated"] = len(text) > LOG_TAIL_CHARS
            if payload["log_truncated"]:
                # Say so instead of silently starting mid-line: the drawer
                # otherwise reads as if the run began at this point.
                tail = text[-LOG_TAIL_CHARS:]
                newline = tail.find("\n")
                if 0 <= newline < 2_000:
                    tail = tail[newline + 1:]
                payload["log"] = (
                    "[studio] Log truncated: showing the last {0:,} of {1:,} characters.\n".format(
                        len(tail), len(text)
                    )
                    + tail
                )
            else:
                payload["log"] = text
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
            payload = self.public_job(job, include_log=True)
        # Only the detail view carries the saved graph, so listing jobs stays cheap.
        if job.get("has_pipeline"):
            pipeline_path = JOB_RUNTIME / job_id / "pipeline.json"
            try:
                payload["pipeline"] = json.loads(pipeline_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload["pipeline"] = None
        return payload

    def read_job_log(self, job_id: str) -> str:
        """Read the complete persisted log for metrics/history extraction.

        The interactive log drawer intentionally returns only a recent tail,
        but metric history must not silently lose early epochs on long runs.
        """

        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            path = job["log_path"]
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""


STATE = StudioState()
