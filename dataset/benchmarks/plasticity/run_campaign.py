#!/usr/bin/env python3
"""Run the seven-model Plasticity campaign without changing model runtimes.

The orchestrator is intentionally conservative.  It schedules independently
owned launcher trees through ``AI_CAE4ALL_main.py`` only when a strict
resource record allows their placement, records every phase, and only reuses a
completed model after the strict rollout evaluator succeeds again.  Partial or
otherwise ambiguous model output directories are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import h5py
import numpy as np
import torch

from compare_results import _validate_result as _strict_validate_result


CAMPAIGN_MODELS = (
    "meshgraphnets",
    "hi_meshgraphnets",
    "point_deeponet",
    "deeponet",
    "fno",
    "gino",
    "transolver",
)
SCHEMA_VERSION = "plasticity_campaign_v2"
EVALUATION_SCHEMA_VERSION = "plasticity_rollout_evaluation_v1"
GATE_SCHEMA_VERSION = "paper_validation_completion_gate_v1"
RESOURCE_INDEX_SCHEMA_VERSION = "plasticity_resource_probe_index_v1"
RESOURCE_PROBE_SCHEMA_VERSION = "plasticity_resource_probe_v1"
GATE_WAIVER_STATUS = "waived_by_user"
REQUIRED_PAPER_VALIDATIONS = (
    "fno",
    "transolver",
    "deeponet",
    "point_deeponet",
    "gino",
)
GPU_PEAK_LIMIT_MIB = 6656
MIN_PAIR_THROUGHPUT_GAIN = 0.10
CPU_THREAD_LIMIT = 8
MGN_MODELS = frozenset({"meshgraphnets", "hi_meshgraphnets"})
TRAIN_PROFILES: dict[str, tuple[tuple[str, int, int], ...]] = {
    "meshgraphnets": (("canonical", 4, 1), ("fallback_2x2", 2, 2), ("fallback_1x4", 1, 4)),
    "hi_meshgraphnets": (("canonical", 4, 1), ("fallback_2x2", 2, 2), ("fallback_1x4", 1, 4)),
    "point_deeponet": (("canonical", 2, 2),),
    "deeponet": (("canonical", 4, 1), ("fallback_2x2", 2, 2), ("fallback_1x4", 1, 4)),
    "fno": (("canonical", 4, 1), ("fallback_2x2", 2, 2), ("fallback_1x4", 1, 4)),
    "gino": (("canonical", 1, 4),),
    "transolver": (("canonical", 1, 4),),
}
ROLLOUT_RE = re.compile(r"^rollout_sample(?P<sample>\d+)_steps19\.h5$")
EXPECTED_ROLLOUTS = 100
PINNED_TRUTH_SHA256 = "5970cdcd362e94f5a54e0f7d18893b11c51f5e1ab345712bddfbbe8d130ad8be"
PINNED_CONFIG_SHA256 = {
    "meshgraphnets": {
        "train": "febb60900121b673f918924aa8f6cd547e700e65f8d4098595f506c213895846",
        "infer": "0839d0775e6c0101bffd01e039f33a0339c71b568ee874ecc58ad60cb91cb15f",
    },
    "hi_meshgraphnets": {
        "train": "36b3f1077914f616d267cba34d49518d061786581c6b50cd6c981b8efca037f8",
        "infer": "e7ef90f949340fc44763e9eca6e7ec70f57efbee41f45c7bcd8ea0af85b10c46",
    },
    "point_deeponet": {
        "train": "63f8479c6edf4506d4edd80d730d25d659dc3e74199bae2fdad2ed76b5fda9c9",
        "infer": "5fa4711fc1ea6a28299879df65e43bc22bf5984d9c2ab57a2b5c167fa8e007e3",
    },
    "deeponet": {
        "train": "86a094efd82f0ec5ef8583c8ee8c7bab8bf8e86cb91ccbac1e7b5ff48dc0fa7e",
        "infer": "f4e389c9ca79e4f21fd8907a40ff6639fe98089c39d3f3a88c91f6b8d8cae7c2",
    },
    "fno": {
        "train": "3a5b44d63fac574db6bbfa3240e9c34c092b1111f6d7810d3a0271f48283c3f0",
        "infer": "2066ff0a8fc095ad93690a6fcd6b1bad53f20ab16c7e75e46805d0d6e1fd5664",
    },
    "gino": {
        "train": "d759e33fc9452b34c6dbef8604faaead5571c6ab497be5be71437c85bbd9b551",
        "infer": "700a5340bc9bd04313070a01be6c7c8c5a5486db0e936fd166947aa6b40b2e3b",
    },
    "transolver": {
        "train": "829827780001e860cdb754ec109bd30a1da5e042c9f06950952b555e9066736a",
        "infer": "760f9f6beb2bc62c161247c992e1371148ce4d9144ae6e823ec6c5e83fc142fe",
    },
}


class CampaignError(RuntimeError):
    """A fail-fast campaign error with a user-actionable message."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class Runner(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path | None,
        stderr_path: Path | None,
        job_id: str | None = None,
        env: Mapping[str, str] | None = None,
        on_start: Callable[[int], None] | None = None,
    ) -> CommandResult: ...


class CampaignLock:
    """Kernel-owned exclusive lock; the JSON PID is diagnostic only.

    The lock file deliberately persists.  A crashed process can leave stale
    metadata, but kernel byte-range/flock ownership is released by the OS and
    is the only authority used for exclusion.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None
        self.metadata: dict[str, object] = {}

    def acquire(self) -> "CampaignLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            try:
                owner = self.path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                owner = "<unreadable>"
            raise CampaignError(
                f"Another Plasticity campaign owns {self.path}; lock metadata={owner!r}"
            ) from exc

        self._handle = handle
        self.metadata = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": _utc_now(),
            "argv": sys.argv,
        }
        encoded = (json.dumps(self.metadata, sort_keys=True) + "\n").encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        return self

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self._handle = None

    def __enter__(self) -> "CampaignLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()


class _WindowsJob:
    """Minimal kill-on-close Windows Job Object wrapper."""

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9

    def __init__(self, process: subprocess.Popen[object]) -> None:
        from ctypes import wintypes

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        self._kernel32 = kernel32
        self._handle = handle
        try:
            limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            limits.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                handle,
                self.JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
            process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
            if not kernel32.AssignProcessToJobObject(handle, process_handle):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        except BaseException:
            kernel32.CloseHandle(handle)
            self._handle = None
            raise

    def terminate(self, exit_code: int = 1) -> None:
        if self._handle is not None:
            self._kernel32.TerminateJobObject(self._handle, exit_code)

    def close(self) -> None:
        if self._handle is not None:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class _OwnedProcess:
    def __init__(self, process: subprocess.Popen[object], windows_job: _WindowsJob | None) -> None:
        self.process = process
        self.windows_job = windows_job
        self._tree_lock = threading.Lock()
        self._tree_closed = False

    def terminate_tree(self, *, graceful_seconds: float = 5.0, kill_seconds: float = 5.0) -> None:
        with self._tree_lock:
            if self._tree_closed:
                return
            if os.name == "nt":
                if self.process.poll() is None:
                    try:
                        self.process.send_signal(signal.CTRL_BREAK_EVENT)
                    except (OSError, ValueError):
                        pass
                    try:
                        self.process.wait(timeout=graceful_seconds)
                    except subprocess.TimeoutExpired:
                        pass
                if self.windows_job is not None:
                    self.windows_job.terminate(1)
                elif self.process.poll() is None:
                    self.process.kill()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.process.wait(timeout=graceful_seconds)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
            try:
                self.process.wait(timeout=kill_seconds)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=kill_seconds)

    def close_tree(self) -> None:
        """Normal parent exit must not leave a separately spawned trainer alive."""
        with self._tree_lock:
            if self._tree_closed:
                return
            if os.name == "nt":
                if self.windows_job is not None:
                    self.windows_job.close()  # KILL_ON_JOB_CLOSE owns all descendants.
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            self._tree_closed = True


class SubprocessRunner:
    """Synchronous call interface with independently owned concurrent trees."""

    def __init__(self) -> None:
        self._active_lock = threading.Lock()
        self._active: dict[str, _OwnedProcess] = {}

    def cancel_active(self) -> None:
        with self._active_lock:
            active = list(self._active.values())
        for owned in active:
            owned.terminate_tree()

    def cancel_job(self, job_id: str) -> None:
        with self._active_lock:
            active = self._active.get(job_id)
        if active is not None:
            active.terminate_tree()

    @staticmethod
    def _emergency_windows_tree_kill(process: subprocess.Popen[object]) -> None:
        try:
            killer = subprocess.Popen(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            killer.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        stdout_path: Path | None,
        stderr_path: Path | None,
        job_id: str | None = None,
        env: Mapping[str, str] | None = None,
        on_start: Callable[[int], None] | None = None,
    ) -> CommandResult:
        stdout_handle = None
        stderr_handle = None
        owned: _OwnedProcess | None = None
        try:
            capture = stdout_path is None or stderr_path is None
            if capture:
                stdout_target = subprocess.PIPE
                stderr_target = subprocess.PIPE
            else:
                assert stdout_path is not None and stderr_path is not None
                stdout_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                stdout_handle = stdout_path.open("wb")
                stderr_handle = stderr_path.open("wb")
                stdout_target = stdout_handle
                stderr_target = stderr_handle

            popen_kwargs: dict[str, object] = {
                "cwd": cwd,
                "stdout": stdout_target,
                "stderr": stderr_target,
            }
            if env is not None:
                popen_kwargs["env"] = dict(env)
            if capture:
                popen_kwargs.update({"text": True, "errors": "replace"})
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True

            process = subprocess.Popen(list(command), **popen_kwargs)
            windows_job = None
            if os.name == "nt":
                try:
                    windows_job = _WindowsJob(process)
                except BaseException:
                    self._emergency_windows_tree_kill(process)
                    raise
            owned = _OwnedProcess(process, windows_job)
            active_key = job_id or f"anonymous-{threading.get_ident()}"
            with self._active_lock:
                if active_key in self._active:
                    owned.terminate_tree()
                    raise RuntimeError(f"SubprocessRunner already owns job {active_key!r}")
                self._active[active_key] = owned
            if on_start is not None:
                on_start(int(process.pid))

            if capture:
                stdout_text, stderr_text = process.communicate()
                return CommandResult(
                    int(process.returncode), str(stdout_text or ""), str(stderr_text or "")
                )
            return CommandResult(int(process.wait()))
        except KeyboardInterrupt:
            if owned is not None:
                owned.terminate_tree()
            raise
        except OSError as exc:
            message = f"{type(exc).__name__}: {exc}\n"
            if stderr_path is not None:
                stderr_path.parent.mkdir(parents=True, exist_ok=True)
                stderr_path.write_text(message, encoding="utf-8")
            return CommandResult(127, stderr=message)
        except BaseException:
            if owned is not None:
                owned.terminate_tree()
            raise
        finally:
            with self._active_lock:
                if owned is not None:
                    for key, candidate in list(self._active.items()):
                        if candidate is owned:
                            self._active.pop(key, None)
            if owned is not None:
                owned.close_tree()
            if stdout_handle is not None:
                stdout_handle.close()
            if stderr_handle is not None:
                stderr_handle.close()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_head(path: Path, size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        digest.update(handle.read(size))
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _update_hdf5_digest(
    digest: "hashlib._Hash", group: h5py.Group, path: str = ""
) -> None:
    for key in sorted(group.keys()):
        child_path = f"{path}/{key}"
        if child_path == "/metadata/normalization_params":
            continue
        child = group[key]
        digest.update(child_path.encode("utf-8"))
        attrs = {
            name: _jsonable(child.attrs[name])
            for name in sorted(child.attrs.keys())
        }
        digest.update(json.dumps(attrs, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        if isinstance(child, h5py.Group):
            digest.update(b"G")
            _update_hdf5_digest(digest, child, child_path)
        else:
            digest.update(b"D")
            digest.update(str(child.shape).encode("ascii"))
            digest.update(str(child.dtype).encode("ascii"))
            if child.shape == ():
                digest.update(np.asarray(child[()]).tobytes())
            elif child.chunks:
                for chunk in child.iter_chunks():
                    digest.update(np.asarray(child[chunk]).tobytes(order="C"))
            else:
                digest.update(np.asarray(child[...]).tobytes(order="C"))


def _hdf5_digest_excluding_normalization(path: Path) -> str:
    digest = hashlib.sha256()
    with h5py.File(path, "r") as handle:
        root_attrs = {
            name: _jsonable(handle.attrs[name]) for name in sorted(handle.attrs.keys())
        }
        digest.update(
            json.dumps(root_attrs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        _update_hdf5_digest(digest, handle)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _flat_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("%") or stripped == "'":
            continue
        fields = stripped.split(None, 1)
        if len(fields) == 2:
            values[fields[0]] = fields[1].strip()
    return values


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignError(
            f"Cannot read JSON object {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"JSON root must be an object: {path}")
    return payload


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _repo_artifact(
    suite_root: Path,
    reference: object,
    *,
    label: str,
) -> tuple[Path, str]:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise CampaignError(f"{label} must contain exactly path and sha256")
    raw_path = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip() or Path(raw_path).is_absolute():
        raise CampaignError(f"{label}.path must be a nonempty repo-relative path")
    if not _is_sha256(expected_hash):
        raise CampaignError(f"{label}.sha256 is malformed")
    path = (suite_root / raw_path).resolve()
    try:
        path.relative_to(suite_root.resolve())
    except ValueError as exc:
        raise CampaignError(f"{label}.path escapes the suite root: {raw_path}") from exc
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"{label} artifact is missing: {path}")
    actual_hash = _sha256(path)
    if actual_hash != expected_hash:
        raise CampaignError(
            f"{label} hash mismatch: expected={expected_hash}, actual={actual_hash}"
        )
    return path, actual_hash


def validate_paper_completion_gate(path: Path, suite_root: Path) -> dict[str, object]:
    """Fail closed unless every own-paper result and the report are hash-bound."""
    path = path.resolve()
    if not path.is_file():
        raise CampaignError(
            "Real Plasticity execution is locked until the paper-validation "
            f"completion gate exists: {path}"
        )
    payload = _read_json_object(path)
    expected_root = {
        "schema_version",
        "complete",
        "status",
        "completed_at",
        "report",
        "validations",
    }
    if set(payload) != expected_root:
        raise CampaignError("Paper-validation gate root fields are not exact")
    if payload.get("schema_version") != GATE_SCHEMA_VERSION:
        raise CampaignError(f"Paper-validation gate schema must be {GATE_SCHEMA_VERSION}")
    if payload.get("complete") is not True or payload.get("status") != "passed":
        raise CampaignError("Paper-validation gate is not explicitly complete and passed")
    if not isinstance(payload.get("completed_at"), str) or not payload["completed_at"]:
        raise CampaignError("Paper-validation gate completed_at is missing")
    try:
        datetime.fromisoformat(str(payload["completed_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CampaignError("Paper-validation gate completed_at is not ISO-8601") from exc
    report_path, report_hash = _repo_artifact(
        suite_root, payload.get("report"), label="paper gate report"
    )
    validations = payload.get("validations")
    if not isinstance(validations, dict) or set(validations) != set(REQUIRED_PAPER_VALIDATIONS):
        raise CampaignError("Paper-validation gate must contain exactly the five validations")
    primary_paths: set[Path] = set()
    validated: dict[str, object] = {}
    required_entry = {
        "complete",
        "status",
        "benchmark",
        "metric",
        "paper_value",
        "measured_value",
        "primary_artifact",
    }
    for model in REQUIRED_PAPER_VALIDATIONS:
        entry = validations[model]
        if not isinstance(entry, dict):
            raise CampaignError(f"Paper-validation entry {model} is not an object")
        extra = set(entry) - (required_entry | {"supporting_artifacts"})
        if not required_entry.issubset(entry) or extra:
            raise CampaignError(f"Paper-validation entry {model} fields are not exact")
        if entry.get("complete") is not True or entry.get("status") != "passed":
            raise CampaignError(f"Paper-validation entry {model} is not complete and passed")
        for key in ("benchmark", "metric"):
            if not isinstance(entry.get(key), str) or not str(entry[key]).strip():
                raise CampaignError(f"Paper-validation entry {model}.{key} is empty")
        for key in ("paper_value", "measured_value"):
            try:
                value = float(entry[key])
            except (TypeError, ValueError) as exc:
                raise CampaignError(f"Paper-validation entry {model}.{key} is not numeric") from exc
            if not math.isfinite(value):
                raise CampaignError(f"Paper-validation entry {model}.{key} is non-finite")
        primary_path, primary_hash = _repo_artifact(
            suite_root,
            entry.get("primary_artifact"),
            label=f"paper validation {model} primary_artifact",
        )
        if primary_path == report_path or primary_path in primary_paths:
            raise CampaignError("Paper-validation primary artifacts alias each other or the report")
        primary_paths.add(primary_path)
        supporting = entry.get("supporting_artifacts", [])
        if model != "transolver" and "supporting_artifacts" in entry:
            raise CampaignError(
                f"Paper-validation entry {model} may not have supporting_artifacts"
            )
        if not isinstance(supporting, list):
            raise CampaignError(f"Paper-validation entry {model}.supporting_artifacts is not a list")
        supporting_records = []
        for index, reference in enumerate(supporting):
            support_path, support_hash = _repo_artifact(
                suite_root,
                reference,
                label=f"paper validation {model} supporting_artifacts[{index}]",
            )
            if support_path == report_path or support_path in primary_paths:
                raise CampaignError("Paper-validation artifacts contain an alias")
            supporting_records.append({"path": str(support_path), "sha256": support_hash})
        validated[model] = {
            "benchmark": entry["benchmark"],
            "metric": entry["metric"],
            "paper_value": float(entry["paper_value"]),
            "measured_value": float(entry["measured_value"]),
            "primary_artifact": {"path": str(primary_path), "sha256": primary_hash},
            "supporting_artifacts": supporting_records,
        }
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": GATE_SCHEMA_VERSION,
        "status": "passed",
        "complete": True,
        "report": {"path": str(report_path), "sha256": report_hash},
        "validations": validated,
    }


def paper_completion_gate_or_user_waiver(
    path: Path,
    suite_root: Path,
    *,
    allow_incomplete: bool,
) -> dict[str, object]:
    """Return the strict gate, or an explicit non-passing user waiver."""
    try:
        return validate_paper_completion_gate(path, suite_root)
    except CampaignError as exc:
        if not allow_incomplete:
            raise
        resolved = path.resolve()
        return {
            "path": str(resolved),
            "sha256": _sha256(resolved) if resolved.is_file() else None,
            "schema_version": GATE_SCHEMA_VERSION,
            "status": GATE_WAIVER_STATUS,
            "complete": False,
            "validations": {},
            "authorization": {
                "type": "explicit_cli_override",
                "scope": "plasticity_campaign",
                "reason": "user redirected priority to the Plasticity campaign",
                "strict_gate_error": str(exc),
            },
        }


def _render_runtime_config(source: Path, updates: Mapping[str, str]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        fields = stripped.split(None, 1)
        if stripped and not stripped.startswith(("%", "#")) and stripped != "'" and len(fields) == 2:
            key = fields[0]
            if key in updates:
                lines.append(f"{key}\t{updates[key]}")
                seen.add(key)
                continue
        lines.append(raw_line)
    for key, value in updates.items():
        if key not in seen:
            lines.append(f"{key}\t{value}")
    return "\n".join(lines) + "\n"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _display(command: Sequence[str]) -> str:
    return subprocess.list2cmdline(list(command))


class PlasticityCampaign:
    def __init__(
        self,
        suite_root: Path,
        *,
        python_executable: str | Path = sys.executable,
        runner: Runner | None = None,
        pinned_config_sha256: dict[str, dict[str, str]] | None = None,
        pinned_truth_sha256: str = PINNED_TRUTH_SHA256,
        result_validator: Callable[[Path, str], tuple[dict[str, object], dict[str, object]]] = _strict_validate_result,
        allow_incomplete_paper_validation: bool = False,
        epoch_budget: int = 500,
    ) -> None:
        self.suite_root = suite_root.resolve()
        self.python = str(python_executable)
        self.runner = runner or SubprocessRunner()
        self.pinned_config_sha256 = pinned_config_sha256 or PINNED_CONFIG_SHA256
        self.pinned_truth_sha256 = pinned_truth_sha256
        self.result_validator = result_validator
        self.allow_incomplete_paper_validation = bool(
            allow_incomplete_paper_validation
        )
        self.epoch_budget = int(epoch_budget)
        if self.epoch_budget <= 0:
            raise CampaignError("epoch_budget must be a positive integer")
        self.benchmark_dir = self.suite_root / "dataset" / "benchmarks" / "plasticity"
        self.config_dir = self.suite_root / "configs" / "benchmarks" / "plasticity"
        self.output_root = self.suite_root / "output" / "benchmarks" / "plasticity"
        self.log_root = self.output_root / "campaign_logs"
        self.status_path = self.output_root / "campaign_status.json"
        self.lock_path = self.output_root / ".campaign.lock"
        self.paper_gate_path = self.suite_root / "output" / "benchmarks" / "paper_validation_completion_gate.json"
        self.resource_index_path = self.output_root / "resource_probe" / "index.json"
        self.runtime_config_root = self.output_root / "campaign_runtime_configs"
        self.suite_entrypoint = self.suite_root / "AI_CAE4ALL_main.py"
        self.evaluator = self.benchmark_dir / "evaluate_rollouts.py"
        self.comparator = self.benchmark_dir / "compare_results.py"
        self.truth = self.benchmark_dir / "plasticity_seed42_test.h5"
        self.source = self.benchmark_dir / "plasticity.h5"
        self.mgn_working_copy = self.benchmark_dir / "plasticity_meshgraphnets_runtime.h5"
        self.hi_mgn_working_copy = self.benchmark_dir / "plasticity_hi_meshgraphnets_runtime.h5"
        self.transolver_working_copy = self.benchmark_dir / "plasticity_transolver_runtime.h5"
        self.provenance = self.benchmark_dir / "plasticity.provenance.json"
        self.status: dict[str, object] = {}
        self._status_lock = threading.RLock()
        self.effective_train_configs = {
            model: self.train_config(model) for model in CAMPAIGN_MODELS
        }
        self.effective_infer_configs = {
            model: self.infer_config(model) for model in CAMPAIGN_MODELS
        }
        self.execution_plan: dict[str, object] = {}

    def train_config(self, model: str) -> Path:
        return self.config_dir / f"config_train_{model}.txt"

    def infer_config(self, model: str) -> Path:
        return self.config_dir / f"config_infer_{model}.txt"

    def model_root(self, model: str) -> Path:
        return self.output_root / model

    def checkpoint(self, model: str) -> Path:
        return self.model_root(model) / "model.pth"

    def inference_dir(self, model: str) -> Path:
        return self.model_root(model) / "inference"

    def metrics_json(self, model: str) -> Path:
        return self.inference_dir(model) / "plasticity_metrics.json"

    def metrics_csv(self, model: str) -> Path:
        return self.inference_dir(model) / "plasticity_per_case_time.csv"

    def _suite_command(self, config: Path, *, check: bool = False) -> list[str]:
        command = [
            self.python,
            str(self.suite_entrypoint),
            "--config",
            str(config),
            "--no-color",
        ]
        if check:
            command.append("--check")
        return command

    def _evaluator_command(
        self, model: str, *, output_json: Path | None = None, output_csv: Path | None = None
    ) -> list[str]:
        command = [
            self.python,
            str(self.evaluator),
            "--model",
            model,
            "--ground-truth",
            str(self.truth),
            "--predictions",
            str(self.inference_dir(model)),
        ]
        if output_json is not None:
            command.extend(["--output-json", str(output_json)])
        if output_csv is not None:
            command.extend(["--output-csv", str(output_csv)])
        return command

    def _comparison_command(self) -> list[str]:
        return [
            self.python,
            str(self.comparator),
            "--results-root",
            str(self.output_root),
            "--output-dir",
            str(self.output_root),
        ]

    def _verify_static_contract(self) -> dict[str, object]:
        required = [
            self.suite_entrypoint,
            self.evaluator,
            self.comparator,
            self.truth,
            self.source,
            self.mgn_working_copy,
            self.hi_mgn_working_copy,
            self.provenance,
        ]
        required.extend(self.train_config(model) for model in CAMPAIGN_MODELS)
        required.extend(self.infer_config(model) for model in CAMPAIGN_MODELS)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise CampaignError(f"Required campaign files are missing: {missing}")

        provenance = json.loads(self.provenance.read_text(encoding="utf-8"))
        recorded_hash = str(provenance.get("converted_hdf5_sha256", "")).lower()
        actual_hash = _sha256(self.source)
        if not recorded_hash or actual_hash != recorded_hash:
            raise CampaignError(
                "Original plasticity.h5 no longer matches plasticity.provenance.json: "
                f"recorded={recorded_hash or '<missing>'}, actual={actual_hash}"
            )
        truth_hash = _sha256(self.truth)
        if truth_hash != self.pinned_truth_sha256:
            raise CampaignError(
                "plasticity_seed42_test.h5 is not the pinned campaign truth: "
                f"expected={self.pinned_truth_sha256}, actual={truth_hash}"
            )

        if set(self.pinned_config_sha256) != set(CAMPAIGN_MODELS):
            raise CampaignError("Pinned config hash map does not cover exactly the seven campaign models")

        config_hashes: dict[str, dict[str, str]] = {}
        for model in CAMPAIGN_MODELS:
            train_path = self.train_config(model)
            infer_path = self.infer_config(model)
            train = _flat_config(train_path)
            infer = _flat_config(infer_path)
            expected_model = "meshgraphnets" if model in MGN_MODELS else model
            if train.get("model", "").lower() != expected_model or train.get("mode") != "train":
                raise CampaignError(f"Unexpected model/mode in {train_path}")
            if infer.get("model", "").lower() != expected_model or infer.get("mode") != "inference":
                raise CampaignError(f"Unexpected model/mode in {infer_path}")

            expected_dataset = {
                "meshgraphnets": "../dataset/benchmarks/plasticity/plasticity_meshgraphnets_runtime.h5",
                "hi_meshgraphnets": "../dataset/benchmarks/plasticity/plasticity_hi_meshgraphnets_runtime.h5",
            }.get(model, "../dataset/benchmarks/plasticity/plasticity.h5")
            expected_modelpath = f"../output/benchmarks/plasticity/{model}/model.pth"
            expected_inference = f"../output/benchmarks/plasticity/{model}/inference"
            for phase, path, parsed in (
                ("train", train_path, train),
                ("infer", infer_path, infer),
            ):
                actual_config_hash = _sha256(path)
                expected_config_hash = self.pinned_config_sha256.get(model, {}).get(phase)
                if actual_config_hash != expected_config_hash:
                    raise CampaignError(
                        f"Pinned config drift: {path} expected={expected_config_hash}, "
                        f"actual={actual_config_hash}"
                    )
                if parsed.get("dataset_dir") != expected_dataset:
                    raise CampaignError(f"Unexpected dataset_dir in {path}")
                if parsed.get("infer_dataset") != "../dataset/benchmarks/plasticity/plasticity_seed42_test.h5":
                    raise CampaignError(f"Unexpected infer_dataset in {path}")
                if parsed.get("modelpath") != expected_modelpath:
                    raise CampaignError(f"Unexpected modelpath in {path}")
                if parsed.get("inference_output_dir") != expected_inference:
                    raise CampaignError(f"Unexpected inference_output_dir in {path}")
                semantic_expected = {
                    "gpu_ids": "0",
                    "parallel_mode": "ddp",
                    "split_seed": "42",
                    "infer_timesteps": "19",
                    "input_var": "4",
                    "output_var": "4",
                }
                for key, expected in semantic_expected.items():
                    if parsed.get(key) != expected:
                        raise CampaignError(
                            f"Pinned semantic drift in {path}: {key}={parsed.get(key)!r}, "
                            f"expected {expected!r}"
                        )
                expected_epochs = "500" if phase == "train" else "1"
                if parsed.get("training_epochs") != expected_epochs:
                    raise CampaignError(
                        f"Pinned semantic drift in {path}: training_epochs must be {expected_epochs}"
                    )
            config_hashes[model] = {
                "train": _sha256(train_path),
                "infer": _sha256(infer_path),
            }

        return {
            "original_hdf5": str(self.source),
            "original_hdf5_sha256": actual_hash,
            "provenance": str(self.provenance),
            "mgn_working_copy": str(self.mgn_working_copy),
            "mgn_working_copy_sha256_at_start": _sha256(self.mgn_working_copy),
            "hi_mgn_working_copy": str(self.hi_mgn_working_copy),
            "hi_mgn_working_copy_sha256_at_start": _sha256(self.hi_mgn_working_copy),
            "transolver_working_copy": str(self.transolver_working_copy),
            "transolver_working_copy_sha256_at_start": (
                _sha256(self.transolver_working_copy)
                if self.transolver_working_copy.is_file()
                else None
            ),
            "ground_truth": str(self.truth),
            "ground_truth_sha256": truth_hash,
            "config_sha256": config_hashes,
        }

    @staticmethod
    def _probe_identity_key(identity: Mapping[str, object]) -> str:
        encoded = json.dumps(
            dict(identity), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _validate_iso_timestamp(value: object, *, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise CampaignError(f"{label} is missing")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise CampaignError(f"{label} is not an ISO timestamp") from exc
        return value

    def _validate_probe_reference(
        self,
        map_name: str,
        map_key: str,
        reference: object,
    ) -> dict[str, object]:
        expected_ref = {
            "identity_key",
            "identity",
            "result_path",
            "result_sha256",
            "completed_at",
            "run_id",
        }
        if not isinstance(reference, dict) or set(reference) != expected_ref:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] fields are not exact")
        if reference.get("identity_key") != map_key or not _is_sha256(map_key):
            raise CampaignError(f"resource index {map_name}[{map_key!r}] identity key is invalid")
        identity = reference.get("identity")
        identity_fields = {
            "mode",
            "device",
            "models",
            "models_key",
            "profile_key",
            "source_sha256",
            "selected_ids_sha256",
            "cases",
            "seed",
        }
        if not isinstance(identity, dict) or set(identity) != identity_fields:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] identity is malformed")
        if self._probe_identity_key(identity) != map_key:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] identity hash is invalid")
        if identity.get("source_sha256") != _sha256(self.source):
            raise CampaignError(f"resource index {map_name}[{map_key!r}] source hash is stale")
        if not _is_sha256(identity.get("selected_ids_sha256")):
            raise CampaignError(f"resource index {map_name}[{map_key!r}] selected IDs hash is invalid")
        if identity.get("mode") not in {"single", "pair"}:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] mode is invalid")
        if identity.get("device") not in {"cpu", "gpu"}:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] device is invalid")
        if not isinstance(identity.get("cases"), int) or int(identity["cases"]) <= 0:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] cases is invalid")
        if not isinstance(identity.get("seed"), int):
            raise CampaignError(f"resource index {map_name}[{map_key!r}] seed is invalid")
        models = identity.get("models")
        if not isinstance(models, list) or len(models) not in {1, 2}:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] models are invalid")
        parsed_models: dict[str, dict[str, object]] = {}
        expected_model_fields = {"model", "profile", "batch_size", "grad_accum_steps"}
        for item in models:
            if not isinstance(item, dict) or set(item) != expected_model_fields:
                raise CampaignError(f"resource index {map_name}[{map_key!r}] model identity is malformed")
            model = item.get("model")
            profile = item.get("profile")
            batch_size = item.get("batch_size")
            accumulation = item.get("grad_accum_steps")
            if model not in CAMPAIGN_MODELS or model in parsed_models:
                raise CampaignError(f"resource index {map_name}[{map_key!r}] model name is invalid")
            matches = {
                name: (batch, accum)
                for name, batch, accum in TRAIN_PROFILES[str(model)]
            }
            if (
                profile not in matches
                or not isinstance(batch_size, int)
                or not isinstance(accumulation, int)
                or matches[str(profile)] != (batch_size, accumulation)
                or batch_size * accumulation != 4
            ):
                raise CampaignError(f"resource index {map_name}[{map_key!r}] profile is invalid")
            parsed_models[str(model)] = dict(item)
        ordered_names = sorted(parsed_models)
        if [str(item["model"]) for item in models] != ordered_names:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] models are not sorted")
        models_key = ",".join(ordered_names)
        profile_key = ",".join(
            f"{model}={parsed_models[model]['profile']}" for model in ordered_names
        )
        if identity.get("models_key") != models_key or identity.get("profile_key") != profile_key:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] identity keys are inconsistent")
        expected_count = 1 if identity["mode"] == "single" else 2
        if len(parsed_models) != expected_count:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] mode/model count differs")

        result_path_raw = reference.get("result_path")
        if not isinstance(result_path_raw, str) or not Path(result_path_raw).is_absolute():
            raise CampaignError(f"resource index {map_name}[{map_key!r}] result_path is not absolute")
        result_path = Path(result_path_raw).resolve()
        runs_root = (self.output_root / "resource_probe" / "runs").resolve()
        try:
            result_path.relative_to(runs_root)
        except ValueError as exc:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] result escapes runs root") from exc
        if (
            result_path.name != "resource_probe_result.json"
            or not result_path.is_file()
            or result_path.is_symlink()
        ):
            raise CampaignError(f"resource index {map_name}[{map_key!r}] result is missing")
        expected_hash = reference.get("result_sha256")
        if not _is_sha256(expected_hash) or _sha256(result_path) != expected_hash:
            raise CampaignError(f"resource index {map_name}[{map_key!r}] result hash is invalid")
        completed_at = self._validate_iso_timestamp(
            reference.get("completed_at"), label=f"resource index {map_name} completed_at"
        )
        if not isinstance(reference.get("run_id"), str) or not str(reference["run_id"]).strip():
            raise CampaignError(f"resource index {map_name}[{map_key!r}] run_id is invalid")

        record = _read_json_object(result_path)
        required_root = {
            "schema_version",
            "mode",
            "state",
            "complete",
            "completed_at",
            "run_id",
            "device",
            "gate",
            "dataset",
            "models",
            "baselines",
            "identity",
            "identity_key",
        }
        if not required_root.issubset(record):
            raise CampaignError(f"resource result {result_path} omits required root fields")
        if (
            record.get("schema_version") != RESOURCE_PROBE_SCHEMA_VERSION
            or record.get("state") != "complete"
            or record.get("complete") is not True
            or record.get("mode") != identity["mode"]
            or record.get("device") != identity["device"]
            or record.get("run_id") != reference["run_id"]
            or record.get("identity") != identity
            or record.get("identity_key") != map_key
        ):
            raise CampaignError(f"resource result {result_path} root identity is inconsistent")
        if record.get("completed_at") != completed_at:
            raise CampaignError(f"resource result {result_path} completion time is inconsistent")
        gate = record.get("gate")
        strict_gate = bool(
            isinstance(gate, dict)
            and gate.get("schema_version") == GATE_SCHEMA_VERSION
            and gate.get("complete") is True
            and gate.get("status") == "passed"
        )
        waiver = gate.get("authorization") if isinstance(gate, dict) else None
        waived_gate = bool(
            isinstance(gate, dict)
            and gate.get("schema_version") == GATE_SCHEMA_VERSION
            and gate.get("complete") is False
            and gate.get("status") == GATE_WAIVER_STATUS
            and isinstance(waiver, dict)
            and waiver.get("type") == "explicit_cli_override"
            and waiver.get("scope") == "plasticity_resource_probe"
        )
        if not (strict_gate or waived_gate):
            raise CampaignError(f"resource result {result_path} gate is invalid")
        dataset = record.get("dataset")
        if (
            not isinstance(dataset, dict)
            or dataset.get("source_sha256") != identity["source_sha256"]
            or dataset.get("selected_ids_sha256") != identity["selected_ids_sha256"]
            or dataset.get("cases") != identity["cases"]
            or dataset.get("seed") != identity["seed"]
        ):
            raise CampaignError(f"resource result {result_path} dataset identity is inconsistent")
        model_records = record.get("models")
        if not isinstance(model_records, dict) or set(model_records) != set(parsed_models):
            raise CampaignError(f"resource result {result_path} model records are inconsistent")
        for model, expected in parsed_models.items():
            model_record = model_records[model]
            if (
                not isinstance(model_record, dict)
                or model_record.get("complete") is not True
                or model_record.get("profile") != expected["profile"]
                or model_record.get("batch_size") != expected["batch_size"]
                or model_record.get("grad_accum_steps") != expected["grad_accum_steps"]
            ):
                raise CampaignError(f"resource result {result_path} model record {model} is invalid")
        return {
            "map": map_name,
            "identity_key": map_key,
            "identity": identity,
            "models": parsed_models,
            "path": str(result_path),
            "sha256": expected_hash,
            "completed_at": completed_at,
            "record": record,
        }

    def _load_resource_evidence(self) -> tuple[list[dict[str, object]], list[str]]:
        if not self.resource_index_path.is_file():
            return [], ["resource probe index is absent; conservative scheduling selected"]
        diagnostics: list[str] = []
        try:
            index = _read_json_object(self.resource_index_path)
            expected_root = {
                "schema_version",
                "updated_at",
                "latest_completed_single",
                "latest_cpu_eligible",
                "latest_certified_pair",
            }
            if set(index) != expected_root or index.get("schema_version") != RESOURCE_INDEX_SCHEMA_VERSION:
                raise CampaignError("resource probe index root/schema is not exact")
            self._validate_iso_timestamp(index.get("updated_at"), label="resource index updated_at")
        except CampaignError as exc:
            return [], [f"resource probe index rejected: {exc}"]
        validated: list[dict[str, object]] = []
        for map_name in (
            "latest_completed_single",
            "latest_cpu_eligible",
            "latest_certified_pair",
        ):
            mapping = index.get(map_name)
            if not isinstance(mapping, dict):
                return [], [f"resource probe index {map_name} is not an object"]
            for map_key, reference in sorted(mapping.items()):
                try:
                    validated.append(
                        self._validate_probe_reference(map_name, str(map_key), reference)
                    )
                except CampaignError as exc:
                    diagnostics.append(f"{map_name}[{map_key}] rejected: {exc}")
        return validated, diagnostics

    @staticmethod
    def _fallback_profile(model: str) -> tuple[str, int, int]:
        if model in {"point_deeponet", "gino", "transolver"}:
            return TRAIN_PROFILES[model][0]
        return TRAIN_PROFILES[model][-1]

    @staticmethod
    def _gpu_single_is_safe(evidence: Mapping[str, object]) -> bool:
        record = evidence["record"]
        assert isinstance(record, dict)
        gpu = record.get("gpu")
        fit = record.get("resource_fit")
        try:
            return bool(
                isinstance(gpu, dict)
                and gpu.get("complete") is True
                and int(gpu.get("peak_total_used_mib", GPU_PEAK_LIMIT_MIB + 1)) <= GPU_PEAK_LIMIT_MIB
                and isinstance(fit, dict)
                and fit.get("peak_pass") is True
                and int(fit.get("peak_limit_mib", -1)) == GPU_PEAK_LIMIT_MIB
                and int(fit.get("observed_peak_total_used_mib", GPU_PEAK_LIMIT_MIB + 1))
                <= GPU_PEAK_LIMIT_MIB
            )
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _cpu_single_is_safe(evidence: Mapping[str, object]) -> bool:
        record = evidence["record"]
        assert isinstance(record, dict)
        placement = record.get("placement")
        projection = record.get("projection")
        try:
            return bool(
                isinstance(placement, dict)
                and placement.get("cpu_eligible") is True
                and isinstance(projection, dict)
                and projection.get("conservative") is True
                and math.isfinite(float(projection.get("projected_wall_seconds_500_epochs", math.inf)))
                and float(projection.get("projected_wall_seconds_500_epochs", 0.0)) > 0.0
            )
        except (TypeError, ValueError, OverflowError):
            return False

    @staticmethod
    def _gpu_pair_is_safe(evidence: Mapping[str, object]) -> bool:
        record = evidence["record"]
        assert isinstance(record, dict)
        certification = record.get("certification")
        gpu = record.get("gpu")
        baselines = record.get("baselines")
        if not isinstance(certification, dict):
            return False
        try:
            improvement = float(certification.get("throughput_improvement_fraction", -math.inf))
            observed_peak = int(certification.get("observed_peak_total_used_mib", GPU_PEAK_LIMIT_MIB + 1))
        except (TypeError, ValueError, OverflowError):
            return False
        try:
            return bool(
                certification.get("certified") is True
                and certification.get("peak_pass") is True
                and certification.get("throughput_pass") is True
                and int(certification.get("peak_limit_mib", -1)) == GPU_PEAK_LIMIT_MIB
                and float(certification.get("required_throughput_improvement", -1.0))
                == MIN_PAIR_THROUGHPUT_GAIN
                and math.isfinite(improvement)
                and improvement >= MIN_PAIR_THROUGHPUT_GAIN
                and observed_peak <= GPU_PEAK_LIMIT_MIB
                and isinstance(gpu, dict)
                and gpu.get("complete") is True
                and isinstance(baselines, list)
                and len(baselines) == 2
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def _build_execution_plan(self, fresh_models: Sequence[str]) -> dict[str, object]:
        fresh = tuple(model for model in CAMPAIGN_MODELS if model in set(fresh_models))
        evidence, diagnostics = self._load_resource_evidence()
        models_plan: dict[str, dict[str, object]] = {}
        for model in fresh:
            profile, batch, accumulation = self._fallback_profile(model)
            models_plan[model] = {
                "placement": "gpu",
                "profile": profile,
                "batch_size": batch,
                "grad_accum_steps": accumulation,
                "effective_batch_size": batch * accumulation,
                "evidence": None,
                "selection_reason": "conservative fallback",
            }

        completed_singles = [
            item
            for item in evidence
            if item["map"] == "latest_completed_single"
            and item["identity"]["mode"] == "single"  # type: ignore[index]
            and item["identity"]["device"] == "gpu"  # type: ignore[index]
            and self._gpu_single_is_safe(item)
        ]
        for model in fresh:
            candidates = [item for item in completed_singles if set(item["models"]) == {model}]
            if not candidates:
                continue
            rank = {name: index for index, (name, _, _) in enumerate(TRAIN_PROFILES[model])}
            best_rank = min(
                rank[str(item["models"][model]["profile"])]  # type: ignore[index]
                for item in candidates
            )
            chosen = max(
                (
                    item
                    for item in candidates
                    if rank[str(item["models"][model]["profile"])] == best_rank  # type: ignore[index]
                ),
                key=lambda item: str(item["completed_at"]),
            )
            identity_model = chosen["models"][model]  # type: ignore[index]
            assert isinstance(identity_model, dict)
            models_plan[model].update(
                {
                    "profile": identity_model["profile"],
                    "batch_size": identity_model["batch_size"],
                    "grad_accum_steps": identity_model["grad_accum_steps"],
                    "evidence": {
                        "identity_key": chosen["identity_key"],
                        "path": chosen["path"],
                        "sha256": chosen["sha256"],
                    },
                    "selection_reason": "strict completed single-GPU probe",
                }
            )

        cpu_candidates = [
            item
            for item in evidence
            if item["map"] == "latest_cpu_eligible"
            and item["identity"]["mode"] == "single"  # type: ignore[index]
            and item["identity"]["device"] == "cpu"  # type: ignore[index]
            and set(item["models"]) == {"deeponet"}
            and self._cpu_single_is_safe(item)
        ]
        cpu_model: str | None = None
        if "deeponet" in models_plan and cpu_candidates:
            cpu_candidates.sort(key=lambda item: str(item["completed_at"]), reverse=True)
            chosen = cpu_candidates[0]
            identity_model = chosen["models"]["deeponet"]  # type: ignore[index]
            assert isinstance(identity_model, dict)
            models_plan["deeponet"].update(
                {
                    "placement": "cpu",
                    "profile": identity_model["profile"],
                    "batch_size": identity_model["batch_size"],
                    "grad_accum_steps": identity_model["grad_accum_steps"],
                    "evidence": {
                        "identity_key": chosen["identity_key"],
                        "path": chosen["path"],
                        "sha256": chosen["sha256"],
                    },
                    "selection_reason": "strict CPU-eligible DeepONet probe",
                }
            )
            cpu_model = "deeponet"

        pair_candidates: list[dict[str, object]] = []
        for item in evidence:
            pair_models = tuple(sorted(item["models"]))
            if (
                item["map"] != "latest_certified_pair"
                or item["identity"]["mode"] != "pair"  # type: ignore[index]
                or item["identity"]["device"] != "gpu"  # type: ignore[index]
                or not set(pair_models).issubset(models_plan)
                or cpu_model in pair_models
                or not self._gpu_pair_is_safe(item)
            ):
                continue
            pair_candidates.append(item)
        pair_candidates.sort(key=lambda item: tuple(CAMPAIGN_MODELS.index(m) for m in sorted(item["models"])))

        best: list[dict[str, object]] = []
        def choose_pairs(index: int, used: frozenset[str], chosen: list[dict[str, object]]) -> None:
            nonlocal best
            if index == len(pair_candidates):
                if len(chosen) > len(best):
                    best = list(chosen)
                return
            choose_pairs(index + 1, used, chosen)
            candidate = pair_candidates[index]
            names = frozenset(str(name) for name in candidate["models"])
            if not names & used:
                choose_pairs(index + 1, used | names, [*chosen, candidate])
        choose_pairs(0, frozenset(), [])

        paired: set[str] = set()
        gpu_groups: list[list[str]] = []
        for item in sorted(best, key=lambda value: min(CAMPAIGN_MODELS.index(m) for m in value["models"])):
            group = [model for model in CAMPAIGN_MODELS if model in item["models"]]
            for model in group:
                identity_model = item["models"][model]  # type: ignore[index]
                models_plan[model].update(
                    {
                        "placement": "gpu",
                        "profile": identity_model["profile"],
                        "batch_size": identity_model["batch_size"],
                        "grad_accum_steps": identity_model["grad_accum_steps"],
                        "evidence": {
                            "identity_key": item["identity_key"],
                            "path": item["path"],
                            "sha256": item["sha256"],
                        },
                        "selection_reason": "strict certified GPU pair",
                    }
                )
            paired.update(group)
            gpu_groups.append(group)
        gpu_groups.extend(
            [model]
            for model in fresh
            if model != cpu_model and model not in paired
        )
        for model, plan in models_plan.items():
            plan["effective_batch_size"] = int(plan["batch_size"]) * int(plan["grad_accum_steps"])
            if plan["effective_batch_size"] != 4:
                raise CampaignError(f"internal scheduler error: {model} effective batch is not four")
            if model == "point_deeponet" and (
                plan["batch_size"] != 2 or plan["grad_accum_steps"] != 2
            ):
                raise CampaignError("Point-DeepONet may only use its BatchNorm-safe 2x2 profile")
        return {
            "models": models_plan,
            "cpu_model": cpu_model,
            "gpu_groups": gpu_groups,
            "resource_index": str(self.resource_index_path),
            "resource_index_sha256": (
                _sha256(self.resource_index_path) if self.resource_index_path.is_file() else None
            ),
            "diagnostics": diagnostics,
        }

    @staticmethod
    def _require_array(
        container: dict[str, object], name: str, shape: tuple[int, ...]
    ) -> None:
        if name not in container:
            raise CampaignError(f"Checkpoint normalization is missing {name}")
        value = container[name]
        if torch.is_tensor(value):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.shape != shape or not np.all(np.isfinite(array)):
            raise CampaignError(
                f"Checkpoint normalization {name} has shape {array.shape}, expected {shape}, "
                "or contains non-finite values"
            )

    def _validate_checkpoint(self, model: str) -> dict[str, object]:
        path = self.checkpoint(model)
        if not path.is_file() or path.stat().st_size <= 0:
            raise CampaignError(f"{model}: checkpoint is missing or empty: {path}")
        stat_before = path.stat()
        digest = _sha256(path)
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise CampaignError(
                f"{model}: checkpoint cannot be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        stat_after = path.stat()
        if (
            stat_before.st_size != stat_after.st_size
            or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        ):
            raise CampaignError(f"{model}: checkpoint changed while being validated")
        if not isinstance(checkpoint, dict):
            raise CampaignError(f"{model}: checkpoint root is not a dictionary")
        expected_final_epoch = self.epoch_budget - 1
        if int(checkpoint.get("epoch", -1)) != expected_final_epoch:
            raise CampaignError(
                f"{model}: checkpoint epoch={checkpoint.get('epoch')!r}, "
                f"expected final epoch {expected_final_epoch} for "
                f"epoch_budget={self.epoch_budget}"
            )
        for key in ("model_state_dict", "ema_state_dict", "normalization", "model_config"):
            value = checkpoint.get(key)
            if not isinstance(value, dict) or not value:
                raise CampaignError(f"{model}: checkpoint {key} is missing or empty")

        normalization = checkpoint["normalization"]
        model_config = checkpoint["model_config"]
        assert isinstance(normalization, dict)
        assert isinstance(model_config, dict)
        self._require_array(normalization, "node_mean", (8,))
        self._require_array(normalization, "node_std", (8,))
        self._require_array(normalization, "delta_mean", (4,))
        self._require_array(normalization, "delta_std", (4,))

        identity: dict[str, object] = {
            "path": str(path),
            "sha256": digest,
            "bytes": stat_after.st_size,
            "epoch": expected_final_epoch,
            "epoch_budget": self.epoch_budget,
            "config_sha256": _sha256(self.effective_train_configs[model]),
        }
        if model in MGN_MODELS:
            self._require_array(normalization, "edge_mean", (8,))
            self._require_array(normalization, "edge_std", (8,))
            expected = {
                "input_var": 4,
                "output_var": 4,
                "edge_var": 8,
                "latent_dim": 128,
                "message_passing_num": 15,
                "positional_features": 4,
                "use_node_types": False,
                "use_world_edges": False,
                "use_multiscale": model == "hi_meshgraphnets",
            }
            if model == "hi_meshgraphnets":
                expected.update(
                    {
                        "multiscale_levels": 2,
                        "mp_per_level": [4, 6, 8, 6, 4],
                        "coarsening_type": "voronoi_seedmean",
                        "voronoi_clusters": [500, 100],
                    }
                )
            for key, value in expected.items():
                if model_config.get(key) != value:
                    raise CampaignError(
                        f"{model}: checkpoint model_config {key}={model_config.get(key)!r}, "
                        f"expected {value!r}"
                    )
            if model == "hi_meshgraphnets":
                for name in ("coarse_edge_means", "coarse_edge_stds"):
                    values = normalization.get(name)
                    if not isinstance(values, (list, tuple)) or len(values) != 2:
                        raise CampaignError(
                            f"hi_meshgraphnets: checkpoint normalization {name} must have two levels"
                        )
                    for level, value in enumerate(values):
                        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                        if array.shape != (8,) or not np.all(np.isfinite(array)):
                            raise CampaignError(
                                f"hi_meshgraphnets: {name}[{level}] must be finite shape (8,)"
                            )
            identity["checkpoint_schema"] = (
                "hi_meshgraphnets_final_v1"
                if model == "hi_meshgraphnets"
                else "meshgraphnets_final_v1"
            )
        elif model in {"point_deeponet", "deeponet", "fno", "gino"}:
            if checkpoint.get("schema_version") != "deeponet_repo_v1":
                raise CampaignError(f"{model}: wrong Neural_Operator checkpoint schema")
            if checkpoint.get("selected_model") != model:
                raise CampaignError(
                    f"{model}: checkpoint selected_model={checkpoint.get('selected_model')!r}"
                )
            if model_config.get("model_name") != model:
                raise CampaignError(f"{model}: checkpoint model_config identifies another model")
            expected_architectures: dict[str, dict[str, object]] = {
                "point_deeponet": {
                    "point_variant": "mesh_state",
                    "point_sensor_count": 2048,
                    "point_hidden_channels": 128,
                    "point_feature_dim": 128,
                    "pointnet_depth": 3,
                    "point_condition_depth": 2,
                    "point_trunk_depth": 3,
                    "point_refiner_depth": 2,
                    "point_siren_omega0": 30.0,
                    "point_output_activation": "identity",
                    "query_dim": 6,
                },
                "deeponet": {
                    "deeponet_branch_source": "fixed_sensors",
                    "deeponet_sensor_resolution": [32, 16],
                    "deeponet_hidden_channels": 256,
                    "deeponet_branch_depth": 3,
                    "deeponet_trunk_depth": 3,
                    "deeponet_basis_dim": 128,
                    "deeponet_activation": "silu",
                    "deeponet_multi_output": "split_both",
                    "branch_in_dim": 5120,
                    "query_dim": 6,
                },
                "fno": {
                    "fno_grid_resolution": [64, 32],
                    "fno_modes": [16, 12],
                    "fno_hidden_channels": 64,
                    "fno_layers": 4,
                    "fno_use_channel_mlp": True,
                    "fno_norm": "none",
                    "in_channels": 12,
                },
                "gino": {
                    "gino_variant": "mesh_state",
                    "gino_grid_resolution": [64, 32],
                    "gino_fno_modes": [16, 12],
                    "gino_fno_hidden_channels": 64,
                    "gino_fno_layers": 4,
                    "gino_in_radius": 0.08,
                    "gino_out_radius": 0.08,
                    "gino_kernel_hidden": 64,
                    "gino_use_torch_cluster": False,
                    "source_feat_dim": 8,
                },
            }
            for key, value in expected_architectures[model].items():
                if model_config.get(key) != value:
                    raise CampaignError(
                        f"{model}: checkpoint model_config {key}={model_config.get(key)!r}, "
                        f"expected {value!r}"
                    )
            data_config = checkpoint.get("data_config")
            if not isinstance(data_config, dict):
                raise CampaignError(f"{model}: checkpoint data_config is missing")
            expected_data = {
                "input_var": 4,
                "output_var": 4,
                "positional_dim": 4,
                "node_type_dim": 0,
                "global_condition_dim": 0,
                "operator_dim": 2,
                "active_axes": [0, 1],
                "has_sdf": False,
                "num_timesteps": 20,
            }
            for key, value in expected_data.items():
                if data_config.get(key) != value:
                    raise CampaignError(
                        f"{model}: checkpoint data_config {key}={data_config.get(key)!r}, "
                        f"expected {value!r}"
                    )
            source_reference = checkpoint.get("source_reference")
            if not isinstance(source_reference, dict):
                raise CampaignError(f"{model}: checkpoint source_reference is missing")
            config_reference = Path(str(source_reference.get("config_file", ""))).resolve()
            allowed_config_references = {
                self.train_config(model).resolve(),
                (
                    self.runtime_config_root
                    / model
                    / f"config_train_{model}.txt"
                ).resolve(),
            }
            if config_reference not in allowed_config_references or not config_reference.is_file():
                raise CampaignError(f"{model}: checkpoint source config is not an allowed campaign config")
            identity["config_sha256"] = _sha256(config_reference)
            dataset_reference = source_reference.get("dataset")
            if not isinstance(dataset_reference, dict):
                raise CampaignError(f"{model}: checkpoint source dataset fingerprint is missing")
            dataset_path = Path(str(dataset_reference.get("path", ""))).resolve()
            if dataset_path != self.source.resolve():
                raise CampaignError(f"{model}: checkpoint source dataset is not plasticity.h5")
            if int(dataset_reference.get("size", -1)) != self.source.stat().st_size:
                raise CampaignError(f"{model}: checkpoint source dataset size is inconsistent")
            if dataset_reference.get("head_sha1") != _sha1_head(self.source):
                raise CampaignError(f"{model}: checkpoint source dataset fingerprint is inconsistent")
            identity["checkpoint_schema"] = "deeponet_repo_v1"
            identity["selected_model"] = model
        else:
            if int(checkpoint.get("checkpoint_version", -1)) != 1:
                raise CampaignError("transolver: wrong checkpoint_version")
            expected_model = {
                "model": "transolver",
                "input_var": 4,
                "output_var": 4,
                "positional_features": 4,
                "use_node_types": False,
                "latent_dim": 128,
                "num_layers": 8,
                "num_heads": 8,
                "slice_num": 64,
                "attention_kernel": "slice_space",
                "mlp_ratio": 1,
                "dropout": 0.0,
                "temperature_init": 0.5,
                "temperature_min": 0.1,
                "temperature_max": 5.0,
                "small_output_init": True,
                "use_checkpointing": True,
                "num_timesteps": 20,
            }
            for key, value in expected_model.items():
                if model_config.get(key) != value:
                    raise CampaignError(
                        f"transolver: checkpoint model_config {key}={model_config.get(key)!r}, "
                        f"expected {value!r}"
                    )
            data_config = checkpoint.get("data_config")
            if not isinstance(data_config, dict):
                raise CampaignError("transolver: checkpoint data_config is missing")
            expected_trans_data = {
                "split_seed": 42,
                "coordinate_normalization": "centered_isotropic",
                "num_timesteps": 20,
                "chunk_size": 1024,
                "infer_mode": "direct",
                "infer_chunk_size": 0,
                "feature_loss_weights": [1.0, 1.0, 1.0, 1.0],
                "std_noise": 0.0,
                "noise_gamma": 1,
            }
            for key, value in expected_trans_data.items():
                if data_config.get(key) != value:
                    raise CampaignError(
                        f"transolver: checkpoint data_config {key}={data_config.get(key)!r}, "
                        f"expected {value!r}"
                    )
            identity["checkpoint_schema"] = "transolver_checkpoint_v1"
            identity["selected_model"] = "transolver"
        return identity

    def _checkpoint_hash_unchanged(self, model: str, expected: str) -> None:
        path = self.checkpoint(model)
        if not path.is_file() or _sha256(path) != expected:
            raise CampaignError(
                f"{model}: checkpoint changed between validation, inference, and evaluation"
            )

    def _verify_mgn_normalization_only_drift(self, model: str) -> dict[str, object]:
        if model not in MGN_MODELS:
            raise CampaignError(f"Normalization-copy validation is not defined for {model}")
        working_path = (
            self.mgn_working_copy
            if model == "meshgraphnets"
            else self.hi_mgn_working_copy
        )
        source_hash = _sha256(self.source)
        working_hash = _sha256(working_path)
        if working_hash == source_hash:
            return {"mode": "bit_identical", "working_sha256": working_hash}
        source_digest = _hdf5_digest_excluding_normalization(self.source)
        working_digest = _hdf5_digest_excluding_normalization(working_path)
        if source_digest != working_digest:
            raise CampaignError(
                f"{model} working copy differs outside metadata/normalization_params"
            )
        allowed_extra_datasets = {
            "node_mean": (8,),
            "node_std": (8,),
            "edge_mean": (8,),
            "edge_std": (8,),
            "delta_mean": (4,),
            "delta_std": (4,),
        }
        allowed_extra_attrs = {
            "edge_feature_layout",
            "edge_var",
            "normalization_source",
            "split_seed",
        }
        with h5py.File(self.source, "r") as source_file, h5py.File(
            working_path, "r"
        ) as working_file:
            source_group = source_file["metadata/normalization_params"]
            working_group = working_file["metadata/normalization_params"]
            source_keys = set(source_group.keys())
            working_keys = set(working_group.keys())
            if working_keys - source_keys != set(allowed_extra_datasets):
                raise CampaignError("MeshGraphNets working-copy normalization dataset drift is not allowed")
            if set(working_group.attrs) - set(source_group.attrs) != allowed_extra_attrs:
                raise CampaignError("MeshGraphNets working-copy normalization attribute drift is not allowed")
            for key in source_keys:
                left = np.asarray(source_group[key][...])
                right = np.asarray(working_group[key][...])
                if left.shape != right.shape or not np.array_equal(left, right):
                    raise CampaignError(f"MeshGraphNets working copy changed source normalization {key}")
            for key, shape in allowed_extra_datasets.items():
                array = np.asarray(working_group[key][...])
                if array.shape != shape or not np.all(np.isfinite(array)):
                    raise CampaignError(f"MeshGraphNets working-copy normalization {key} is invalid")
            if int(working_group.attrs.get("edge_var", -1)) != 8:
                raise CampaignError("MeshGraphNets working-copy edge_var is invalid")
            if int(working_group.attrs.get("split_seed", -1)) != 42:
                raise CampaignError("MeshGraphNets working-copy split_seed is invalid")
            if str(working_group.attrs.get("normalization_source", "")) != "train_split":
                raise CampaignError("MeshGraphNets working-copy normalization_source is invalid")
        return {
            "mode": "normalization_only_drift",
            "working_sha256": working_hash,
            "logical_sha256_excluding_normalization": working_digest,
        }

    def _verify_transolver_normalization_only_drift(self) -> dict[str, object]:
        if not self.transolver_working_copy.is_file():
            raise CampaignError("Transolver runtime HDF5 is missing")
        source_hash = _sha256(self.source)
        working_hash = _sha256(self.transolver_working_copy)
        if working_hash == source_hash:
            return {"mode": "bit_identical", "working_sha256": working_hash}
        source_digest = _hdf5_digest_excluding_normalization(self.source)
        working_digest = _hdf5_digest_excluding_normalization(
            self.transolver_working_copy
        )
        if source_digest != working_digest:
            raise CampaignError(
                "Transolver runtime HDF5 differs outside metadata/normalization_params"
            )
        with h5py.File(self.source, "r") as source_file, h5py.File(
            self.transolver_working_copy, "r"
        ) as working_file:
            source_root = source_file["metadata/normalization_params"]
            working_root = working_file["metadata/normalization_params"]
            if set(working_root.keys()) - set(source_root.keys()) != {"transolver"}:
                raise CampaignError("Transolver normalization namespace drift is invalid")
            if set(working_root.attrs) != set(source_root.attrs):
                raise CampaignError("Transolver changed root normalization attributes")
            group = working_root["transolver"]
            expected_shapes = {
                "node_mean": (8,),
                "node_std": (8,),
                "delta_mean": (4,),
                "delta_std": (4,),
            }
            if set(group.keys()) != set(expected_shapes):
                raise CampaignError("Transolver normalization arrays are incomplete")
            for name, shape in expected_shapes.items():
                value = np.asarray(group[name][...])
                if value.shape != shape or not np.all(np.isfinite(value)):
                    raise CampaignError(f"Transolver normalization {name} is invalid")
            if set(group.attrs) != {
                "position_scale",
                "coordinate_normalization",
                "normalization_source",
                "split_seed",
            }:
                raise CampaignError("Transolver normalization attributes are invalid")
            if not math.isfinite(float(group.attrs["position_scale"])) or float(
                group.attrs["position_scale"]
            ) <= 0:
                raise CampaignError("Transolver normalization position_scale is invalid")
            if str(group.attrs["coordinate_normalization"]) != "centered_isotropic":
                raise CampaignError("Transolver coordinate normalization is invalid")
            if str(group.attrs["normalization_source"]) != "train_split":
                raise CampaignError("Transolver normalization source is invalid")
            if int(group.attrs["split_seed"]) != 42:
                raise CampaignError("Transolver normalization split seed is invalid")
        return {
            "mode": "normalization_only_drift",
            "working_sha256": working_hash,
            "logical_sha256_excluding_normalization": working_digest,
        }

    def _model_files(self, model: str) -> list[Path]:
        root = self.model_root(model)
        if not root.exists():
            return []
        return sorted((path for path in root.rglob("*") if path.is_file()), key=str)

    def _validate_rollout_filenames(self, model: str) -> tuple[bool, str]:
        inference = self.inference_dir(model)
        if not inference.is_dir():
            return False, "inference directory is missing"
        h5_files = sorted(inference.glob("*.h5"))
        malformed = [path.name for path in h5_files if ROLLOUT_RE.fullmatch(path.name) is None]
        if malformed:
            return False, f"unexpected rollout HDF5 names: {malformed[:5]}"
        if len(h5_files) != EXPECTED_ROLLOUTS:
            return False, f"found {len(h5_files)} rollout HDF5 files, expected 100"
        return True, ""

    def _preexisting_metrics_shape(self, model: str) -> tuple[bool, str]:
        summary = self.metrics_json(model)
        per_time = self.metrics_csv(model)
        if not summary.is_file() or not per_time.is_file():
            return False, "metrics JSON and CSV are not both present"
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
            if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
                return False, "metrics schema is not the strict Plasticity schema"
            if payload.get("model") != model or payload.get("complete") is not True:
                return False, "metrics JSON is not complete for this model"
            validation = payload.get("validation", {})
            if not isinstance(validation, dict) or int(validation.get("evaluated_cases", -1)) != 100:
                return False, "metrics JSON does not contain 100 evaluated cases"
            with per_time.open(newline="", encoding="utf-8") as handle:
                row_count = sum(1 for _ in csv.reader(handle)) - 1
            if row_count != 1900:
                return False, f"metrics CSV has {row_count} rows, expected 1900"
        except Exception as exc:
            return False, f"metrics cannot be parsed: {type(exc).__name__}: {exc}"
        return True, ""

    def _parse_valid_metrics(self, model: str, path: Path | None = None) -> dict[str, object]:
        summary = path or self.metrics_json(model)
        try:
            row, identity = self.result_validator(summary, model)
        except Exception as exc:
            raise CampaignError(
                f"{model}: strict result/artifact reconstruction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        payload = json.loads(summary.read_text(encoding="utf-8"))
        predictions = payload["predictions"]
        assert isinstance(predictions, dict)
        return {
            "primary_metric_name": "mean_per_case_full_trajectory_relative_l2",
            "primary_metric_value": row["mean_per_case_full_trajectory_relative_l2"],
            "prediction_manifest_sha256": predictions.get("manifest_sha256"),
            "metrics_json": str(summary),
            "strict_identity": identity,
        }

    def _materialize_transolver_copy(self) -> dict[str, object]:
        source_stat = self.source.stat()
        source_hash = _sha256(self.source)
        if self.transolver_working_copy.exists():
            if not self.transolver_working_copy.is_file():
                raise CampaignError("Transolver runtime HDF5 path is not a regular file")
            working_hash = _sha256(self.transolver_working_copy)
            if working_hash != source_hash:
                raise CampaignError(
                    "Fresh Transolver campaign requires an absent or bit-identical pristine "
                    "plasticity_transolver_runtime.h5"
                )
            return {
                "mode": "existing_bit_identical",
                "path": str(self.transolver_working_copy),
                "sha256": working_hash,
            }
        temporary = self.transolver_working_copy.with_name(
            f".{self.transolver_working_copy.name}.tmp.{os.getpid()}"
        )
        if temporary.exists():
            raise CampaignError(f"Refusing stale Transolver copy temporary file: {temporary}")
        try:
            shutil.copy2(self.source, temporary)
            copied_hash = _sha256(temporary)
            source_after = self.source.stat()
            if (
                copied_hash != source_hash
                or source_after.st_size != source_stat.st_size
                or source_after.st_mtime_ns != source_stat.st_mtime_ns
                or _sha256(self.source) != source_hash
            ):
                raise CampaignError("plasticity.h5 changed while creating the Transolver copy")
            try:
                os.link(temporary, self.transolver_working_copy)
            except FileExistsError as exc:
                raise CampaignError("Transolver runtime HDF5 appeared during atomic creation") from exc
            if _sha256(self.transolver_working_copy) != source_hash:
                raise CampaignError("Atomically created Transolver runtime HDF5 failed its hash gate")
        finally:
            if temporary.exists():
                temporary.unlink()
        return {
            "mode": "atomically_created_bit_identical",
            "path": str(self.transolver_working_copy),
            "sha256": source_hash,
        }

    def _materialize_runtime_configs(self, fresh_models: Sequence[str]) -> dict[str, object]:
        plan_models = self.execution_plan.get("models")
        if not isinstance(plan_models, dict):
            raise CampaignError("Internal scheduler plan is missing model placements")
        records: dict[str, object] = {}
        transolver_copy: dict[str, object] | None = None
        if "transolver" in fresh_models:
            transolver_copy = self._materialize_transolver_copy()
        for model in fresh_models:
            placement = plan_models.get(model)
            if not isinstance(placement, dict):
                raise CampaignError(f"Internal scheduler placement is missing for {model}")
            updates: dict[str, str] = {
                "batch_size": str(placement["batch_size"]),
                "grad_accum_steps": str(placement["grad_accum_steps"]),
                "training_epochs": str(self.epoch_budget),
            }
            if self.epoch_budget < 500:
                # Keep a short preliminary run from spending most of its budget
                # in the canonical three-epoch warmup, and emit validation each
                # epoch so convergence is auditable.
                updates["warmup_epochs"] = str(min(3, max(1, self.epoch_budget // 5)))
                updates["val_interval"] = "1"
            infer_updates: dict[str, str] = {}
            if placement.get("placement") == "cpu":
                cpu_updates = {
                    "gpu_ids": "-1",
                    "use_amp": "False",
                    "num_workers": "0",
                }
                updates.update(cpu_updates)
                infer_updates.update(cpu_updates)
            if model == "transolver":
                runtime_dataset = (
                    "../dataset/benchmarks/plasticity/"
                    "plasticity_transolver_runtime.h5"
                )
                updates.update(
                    {
                        "dataset_dir": runtime_dataset,
                        "write_preprocessing": "True",
                    }
                )
                infer_updates.update(
                    {
                        "dataset_dir": runtime_dataset,
                        "write_preprocessing": "False",
                    }
                )
            runtime_dir = self.runtime_config_root / model
            train_path = runtime_dir / f"config_train_{model}.txt"
            infer_path = runtime_dir / f"config_infer_{model}.txt"
            _atomic_text(train_path, _render_runtime_config(self.train_config(model), updates))
            _atomic_text(
                infer_path,
                _render_runtime_config(self.infer_config(model), infer_updates),
            )
            self.effective_train_configs[model] = train_path
            self.effective_infer_configs[model] = infer_path
            records[model] = {
                "canonical_train": str(self.train_config(model)),
                "canonical_train_sha256": _sha256(self.train_config(model)),
                "effective_train": str(train_path),
                "effective_train_sha256": _sha256(train_path),
                "canonical_infer": str(self.infer_config(model)),
                "canonical_infer_sha256": _sha256(self.infer_config(model)),
                "effective_infer": str(infer_path),
                "effective_infer_sha256": _sha256(infer_path),
                "train_updates": updates,
                "infer_updates": infer_updates,
            }
        return {"configs": records, "transolver_runtime_hdf5": transolver_copy}

    def _job_environment(self, model: str) -> dict[str, str] | None:
        models = self.execution_plan.get("models")
        if not isinstance(models, dict) or not isinstance(models.get(model), dict):
            return None
        if models[model].get("placement") != "cpu":
            return None
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "OMP_NUM_THREADS": str(CPU_THREAD_LIMIT),
                "MKL_NUM_THREADS": str(CPU_THREAD_LIMIT),
                "OPENBLAS_NUM_THREADS": str(CPU_THREAD_LIMIT),
                "NUMEXPR_NUM_THREADS": str(CPU_THREAD_LIMIT),
            }
        )
        return environment

    def _allowed_complete_files(self, model: str) -> tuple[bool, str]:
        root = self.model_root(model)
        allowed_root = {"model.pth", "train.log", "infer.log"}
        for child in root.iterdir():
            if child.is_dir():
                if child.name != "inference":
                    return False, f"unexpected directory {child.name!r}"
            elif child.name not in allowed_root:
                return False, f"unexpected file {child.name!r}"
        allowed_inference = {"plasticity_metrics.json", "plasticity_per_case_time.csv"}
        for child in self.inference_dir(model).iterdir():
            if child.is_dir():
                return False, f"unexpected inference subdirectory {child.name!r}"
            if child.suffix.lower() == ".h5":
                continue
            if child.name not in allowed_inference:
                return False, f"unexpected inference file {child.name!r}"
        return True, ""

    def _run_logged(
        self,
        command: Sequence[str],
        *,
        model: str | None,
        phase: str,
        dry_run: bool,
        job_id: str | None = None,
        env: Mapping[str, str] | None = None,
        on_start: Callable[[int], None] | None = None,
    ) -> tuple[CommandResult, dict[str, object]]:
        if dry_run:
            result = self.runner.run(
                command,
                cwd=self.suite_root,
                stdout_path=None,
                stderr_path=None,
                job_id=job_id,
                env=env,
                on_start=on_start,
            )
            record = {
                "command": list(command),
                "returncode": result.returncode,
                "job_id": job_id,
            }
            return result, record

        directory = self.log_root / (model if model is not None else "campaign")
        stdout_path = directory / f"{phase}.attempt01.stdout.log"
        stderr_path = directory / f"{phase}.attempt01.stderr.log"
        result = self.runner.run(
            command,
            cwd=self.suite_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            job_id=job_id,
            env=env,
            on_start=on_start,
        )
        return result, {
            "command": list(command),
            "returncode": result.returncode,
            "job_id": job_id,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }

    def _assess_existing(self, model: str, *, dry_run: bool) -> dict[str, object]:
        files = self._model_files(model)
        if not files:
            return {"state": "fresh", "reason": "no preexisting model artifacts"}

        checkpoint = self.checkpoint(model)
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            return {"state": "ambiguous", "reason": "nonempty output lacks a usable checkpoint"}
        try:
            checkpoint_identity = self._validate_checkpoint(model)
        except Exception as exc:
            return {"state": "ambiguous", "reason": str(exc)}
        checkpoint_hash = str(checkpoint_identity["sha256"])
        names_ok, reason = self._validate_rollout_filenames(model)
        if not names_ok:
            return {"state": "ambiguous", "reason": reason}
        metrics_ok, reason = self._preexisting_metrics_shape(model)
        if not metrics_ok:
            return {"state": "ambiguous", "reason": reason}
        allowed, reason = self._allowed_complete_files(model)
        if not allowed:
            return {"state": "ambiguous", "reason": reason}

        if dry_run:
            with tempfile.TemporaryDirectory(prefix="plasticity-campaign-validate-") as temporary:
                temp_root = Path(temporary)
                output_json = temp_root / f"{model}.json"
                output_csv = temp_root / f"{model}.csv"
                command = self._evaluator_command(
                    model, output_json=output_json, output_csv=output_csv
                )
                result, command_record = self._run_logged(
                    command, model=model, phase="validate_existing", dry_run=True
                )
                if result.returncode != 0:
                    return {
                        "state": "ambiguous",
                        "reason": f"strict evaluator returned {result.returncode}",
                        "validation": command_record,
                    }
                try:
                    metrics = self._parse_valid_metrics(model, output_json)
                except Exception as exc:
                    return {"state": "ambiguous", "reason": str(exc)}
        else:
            command = self._evaluator_command(model)
            result, command_record = self._run_logged(
                command, model=model, phase="validate_existing", dry_run=False
            )
            if result.returncode != 0:
                return {
                    "state": "ambiguous",
                    "reason": f"strict evaluator returned {result.returncode}",
                    "validation": command_record,
                }
            try:
                metrics = self._parse_valid_metrics(model)
            except Exception as exc:
                return {"state": "ambiguous", "reason": str(exc)}

        try:
            self._checkpoint_hash_unchanged(model, checkpoint_hash)
            allowed, reason = self._allowed_complete_files(model)
            if not allowed:
                raise CampaignError(
                    f"{model}: post-recovery-evaluation allowed-file audit failed: {reason}"
                )
            runtime_dataset_validation = (
                self._verify_mgn_normalization_only_drift(model)
                if model in MGN_MODELS
                else self._verify_transolver_normalization_only_drift()
                if model == "transolver"
                else None
            )
        except Exception as exc:
            return {"state": "ambiguous", "reason": str(exc)}

        return {
            "state": "complete",
            "reason": "checkpoint, 100 rollouts, and metrics passed strict re-evaluation",
            "recovery_validated": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_identity": checkpoint_identity,
            "runtime_dataset_validation": runtime_dataset_validation,
            "mgn_working_copy_validation": (
                runtime_dataset_validation if model in MGN_MODELS else None
            ),
            "metrics": metrics,
            "validation": command_record,
        }

    def _initial_status(
        self,
        *,
        selected: tuple[str, ...],
        static_contract: dict[str, object],
        assessments: dict[str, dict[str, object]],
        paper_gate: dict[str, object],
        execution_plan: dict[str, object],
    ) -> dict[str, object]:
        now = _utc_now()
        models: dict[str, object] = {}
        for model in CAMPAIGN_MODELS:
            assessment = assessments[model]
            models[model] = {
                "selected": model in selected,
                "state": assessment["state"],
                "assessment": assessment,
                "placement": (
                    execution_plan.get("models", {}).get(model)  # type: ignore[union-attr]
                    if isinstance(execution_plan.get("models"), dict)
                    else None
                ),
                "preflight": {"state": "pending"},
                "train": {"state": "skipped" if assessment["state"] == "complete" else "pending"},
                "infer": {"state": "skipped" if assessment["state"] == "complete" else "pending"},
                "evaluate": {
                    "state": "validated_existing" if assessment["state"] == "complete" else "pending"
                },
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
            "state": "initialized",
            "complete": False,
            "model_order": list(CAMPAIGN_MODELS),
            "selected_models": list(selected),
            "epoch_budget": self.epoch_budget,
            "expected_final_epoch": self.epoch_budget - 1,
            "static_contract": static_contract,
            "paper_validation_gate": paper_gate,
            "execution_plan": execution_plan,
            "models": models,
            "comparison": {"state": "not_run"},
            "last_error": None,
        }

    def _write_status(self) -> None:
        with self._status_lock:
            self.status["updated_at"] = _utc_now()
            self.status["complete"] = bool(
                self.status.get("state") == "complete"
                and isinstance(self.status.get("comparison"), dict)
                and self.status["comparison"].get("state") == "complete"  # type: ignore[index]
            )
            snapshot = json.loads(json.dumps(self.status))
            _atomic_json(self.status_path, snapshot)

    def _model_status(self, model: str) -> dict[str, object]:
        with self._status_lock:
            models = self.status["models"]
            assert isinstance(models, dict)
            value = models[model]
            assert isinstance(value, dict)
            return value

    def _preflight_all_train_configs(self, *, dry_run: bool) -> None:
        if not dry_run:
            self.status["state"] = "preflight"
            self._write_status()
        failures: list[str] = []
        for model in CAMPAIGN_MODELS:
            config = self.train_config(model) if dry_run else self.effective_train_configs[model]
            command = self._suite_command(config, check=True)
            result, record = self._run_logged(
                command,
                model=model,
                phase="preflight_train",
                dry_run=dry_run,
                job_id=f"{model}.preflight.attempt01",
                env=(None if dry_run else self._job_environment(model)),
            )
            record["state"] = "passed" if result.returncode == 0 else "failed"
            if dry_run:
                print(f"{'PASS' if result.returncode == 0 else 'FAIL'} preflight {model}")
            else:
                self._model_status(model)["preflight"] = record
                self._write_status()
            if result.returncode != 0:
                failures.append(f"{model}={result.returncode}")
        if failures:
            raise CampaignError(
                "Train-config preflight failed before any model launch: " + ", ".join(failures)
            )

    def _run_model_phase(self, model: str, phase: str, command: Sequence[str]) -> None:
        job_id = f"{model}.{phase}.attempt01"
        environment = self._job_environment(model)
        plan_models = self.execution_plan.get("models")
        assert isinstance(plan_models, dict) and isinstance(plan_models[model], dict)
        resource_record = dict(plan_models[model])
        environment_overrides = (
            {
                key: environment[key]
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            }
            if environment is not None
            else {}
        )
        with self._status_lock:
            model_status = self._model_status(model)
            model_status["state"] = "running"
            model_status[phase] = {
                "state": "running",
                "command": list(command),
                "job_id": job_id,
                "attempt": 1,
                "resources": resource_record,
                "environment_overrides": environment_overrides,
                "started_at": _utc_now(),
            }
            self.status["state"] = "running"
            self._write_status()

        def record_pid(pid: int) -> None:
            with self._status_lock:
                phase_status = self._model_status(model).get(phase)
                if isinstance(phase_status, dict):
                    phase_status["pid"] = pid
                    phase_status["pid_recorded_at"] = _utc_now()
                    self._write_status()

        result, record = self._run_logged(
            command,
            model=model,
            phase=phase,
            dry_run=False,
            job_id=job_id,
            env=environment,
            on_start=record_pid,
        )
        record.update(
            {
                "state": "passed" if result.returncode == 0 else "failed",
                "finished_at": _utc_now(),
                "attempt": 1,
                "resources": resource_record,
                "environment_overrides": environment_overrides,
            }
        )
        with self._status_lock:
            self._model_status(model)[phase] = record
            self._write_status()
        if result.returncode != 0:
            raise CampaignError(
                f"{model} {phase} failed with exit code {result.returncode}; "
                f"see {record['stdout']} and {record['stderr']}"
            )

    def _run_fresh_model(self, model: str) -> None:
        print(f"[{model}] training", flush=True)
        self._run_model_phase(
            model, "train", self._suite_command(self.effective_train_configs[model])
        )
        checkpoint = self.checkpoint(model)
        checkpoint_identity = self._validate_checkpoint(model)
        checkpoint_hash = str(checkpoint_identity["sha256"])

        print(f"[{model}] inference", flush=True)
        self._run_model_phase(
            model, "infer", self._suite_command(self.effective_infer_configs[model])
        )
        self._checkpoint_hash_unchanged(model, checkpoint_hash)
        names_ok, reason = self._validate_rollout_filenames(model)
        if not names_ok:
            raise CampaignError(f"{model} inference output is incomplete: {reason}")

        print(f"[{model}] strict rollout evaluation", flush=True)
        self._run_model_phase(model, "evaluate", self._evaluator_command(model))
        metrics = self._parse_valid_metrics(model)
        self._checkpoint_hash_unchanged(model, checkpoint_hash)
        allowed, reason = self._allowed_complete_files(model)
        if not allowed:
            raise CampaignError(f"{model}: post-evaluation allowed-file audit failed: {reason}")
        runtime_dataset_validation = (
            self._verify_mgn_normalization_only_drift(model)
            if model in MGN_MODELS
            else self._verify_transolver_normalization_only_drift()
            if model == "transolver"
            else None
        )
        with self._status_lock:
            model_status = self._model_status(model)
            model_status.update(
                {
                    "state": "complete",
                    "completed_at": _utc_now(),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_identity": checkpoint_identity,
                    "metrics": metrics,
                    "runtime_dataset_validation": runtime_dataset_validation,
                    "mgn_working_copy_validation": (
                        runtime_dataset_validation if model in MGN_MODELS else None
                    ),
                    "recovery_validated": False,
                }
            )
            self._write_status()

    def _comparison_artifacts_exist(self) -> bool:
        return any(
            (self.output_root / name).is_file()
            for name in ("comparison.json", "comparison.csv", "comparison.md")
        )

    def _all_models_complete(self) -> bool:
        return all(self._model_status(model).get("state") == "complete" for model in CAMPAIGN_MODELS)

    def _cancel_active_jobs(self) -> None:
        cancel = getattr(self.runner, "cancel_active", None)
        if callable(cancel):
            cancel()

    def _run_gpu_groups(self, groups: Sequence[Sequence[str]]) -> None:
        for group in groups:
            if len(group) == 1:
                self._run_fresh_model(str(group[0]))
                continue
            if len(group) != 2:
                raise CampaignError(f"Internal scheduler produced invalid GPU group: {group}")
            print(f"[campaign] certified concurrent GPU pair: {', '.join(group)}", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {pool.submit(self._run_fresh_model, model): model for model in group}
                try:
                    for future in concurrent.futures.as_completed(futures):
                        future.result()
                except BaseException:
                    self._cancel_active_jobs()
                    for future in futures:
                        future.cancel()
                    raise

    def _run_scheduled_models(self) -> None:
        cpu_model = self.execution_plan.get("cpu_model")
        gpu_groups = self.execution_plan.get("gpu_groups")
        if cpu_model is not None and cpu_model != "deeponet":
            raise CampaignError("Only DeepONet may occupy the CPU campaign lane")
        if not isinstance(gpu_groups, list):
            raise CampaignError("Internal scheduler omitted GPU groups")
        if cpu_model is None:
            self._run_gpu_groups(gpu_groups)
            return
        print("[campaign] CPU DeepONet lane enabled by strict resource evidence", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            cpu_future = pool.submit(self._run_fresh_model, str(cpu_model))
            gpu_future = pool.submit(self._run_gpu_groups, gpu_groups)
            try:
                for future in concurrent.futures.as_completed((cpu_future, gpu_future)):
                    future.result()
            except BaseException:
                self._cancel_active_jobs()
                cpu_future.cancel()
                gpu_future.cancel()
                raise

    def _run_comparison(self) -> None:
        if not self._all_models_complete():
            raise CampaignError("Internal guard refused comparison before all seven models completed")
        command = self._comparison_command()
        self.status["comparison"] = {
            "state": "running",
            "command": command,
            "started_at": _utc_now(),
        }
        self._write_status()
        result, record = self._run_logged(
            command,
            model=None,
            phase="compare_results",
            dry_run=False,
            job_id="campaign.compare_results.attempt01",
        )
        record.update(
            {
                "state": "passed" if result.returncode == 0 else "failed",
                "finished_at": _utc_now(),
            }
        )
        if result.returncode != 0:
            self.status["comparison"] = record
            raise CampaignError(
                f"compare_results.py failed with exit code {result.returncode}; "
                f"see {record['stdout']} and {record['stderr']}"
            )
        comparison_path = self.output_root / "comparison.json"
        if not comparison_path.is_file():
            raise CampaignError("Comparison returned success without comparison.json")
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison.get("complete") is not True:
            raise CampaignError("Comparison did not mark the seven-model result complete")
        present = comparison.get("models_present_and_valid")
        if not isinstance(present, list) or set(present) != set(CAMPAIGN_MODELS):
            raise CampaignError("Comparison is missing one or more campaign models")
        record["comparison_json"] = str(comparison_path)
        self.status["comparison"] = {**record, "state": "complete"}
        self.status["state"] = "complete"
        self._write_status()

    def _dry_run_plan(
        self,
        *,
        selected: tuple[str, ...],
        assessments: dict[str, dict[str, object]],
    ) -> None:
        print("\nDRY RUN: no status, log, model, inference, or metric files will be written.")
        diagnostics = self.execution_plan.get("diagnostics", [])
        if diagnostics:
            print("  resource evidence: conservative fallback")
            for diagnostic in diagnostics:
                print(f"    - {diagnostic}")
        for model in selected:
            assessment = assessments[model]
            if assessment["state"] == "complete":
                print(f"SKIP {model}: strict recovery validation passed")
                continue
            placement = self.execution_plan.get("models", {}).get(model, {})  # type: ignore[union-attr]
            print(
                f"RUN  {model}: {placement.get('placement')} profile={placement.get('profile')} "
                f"batch={placement.get('batch_size')}x{placement.get('grad_accum_steps')}"
            )
            print(f"  train:    {_display(self._suite_command(self.train_config(model)))}")
            print(f"  inference:{_display(self._suite_command(self.infer_config(model)))}")
            print(f"  evaluate: {_display(self._evaluator_command(model))}")
        would_complete = all(
            assessments[model]["state"] == "complete" or model in selected
            for model in CAMPAIGN_MODELS
        )
        if would_complete:
            print(f"  compare:  {_display(self._comparison_command())}")
        else:
            print("  compare:  deferred until all seven models are strictly complete")

    def run(self, *, models: Sequence[str] = CAMPAIGN_MODELS, dry_run: bool = False) -> int:
        if dry_run:
            return self._run_locked(models=models, dry_run=True)
        with CampaignLock(self.lock_path):
            return self._run_locked(models=models, dry_run=False)

    def _run_locked(
        self, *, models: Sequence[str] = CAMPAIGN_MODELS, dry_run: bool = False
    ) -> int:
        selected_set = set(models)
        unknown = sorted(selected_set - set(CAMPAIGN_MODELS))
        if unknown:
            raise CampaignError(f"Unknown models: {unknown}")
        if not selected_set:
            raise CampaignError("At least one model must be selected")
        selected = tuple(model for model in CAMPAIGN_MODELS if model in selected_set)

        static_contract = self._verify_static_contract()
        paper_gate: dict[str, object] = {}
        if not dry_run:
            paper_gate = paper_completion_gate_or_user_waiver(
                self.paper_gate_path,
                self.suite_root,
                allow_incomplete=self.allow_incomplete_paper_validation,
            )
        assessments = {
            model: self._assess_existing(model, dry_run=dry_run)
            for model in CAMPAIGN_MODELS
        }
        selected_ambiguous = {
            model: assessments[model]["reason"]
            for model in selected
            if assessments[model]["state"] == "ambiguous"
        }
        all_preexisting_complete = all(
            assessments[model]["state"] == "complete" for model in CAMPAIGN_MODELS
        )
        stale_comparison = self._comparison_artifacts_exist() and not all_preexisting_complete
        fresh_selected = tuple(
            model for model in selected if assessments[model]["state"] == "fresh"
        )
        self.execution_plan = self._build_execution_plan(fresh_selected)
        for model, hash_key, filename in (
            (
                "meshgraphnets",
                "mgn_working_copy_sha256_at_start",
                "plasticity_meshgraphnets_runtime.h5",
            ),
            (
                "hi_meshgraphnets",
                "hi_mgn_working_copy_sha256_at_start",
                "plasticity_hi_meshgraphnets_runtime.h5",
            ),
        ):
            if (
                model in fresh_selected
                and static_contract[hash_key] != static_contract["original_hdf5_sha256"]
            ):
                raise CampaignError(
                    f"Fresh {model} campaign requires a bit-identical working copy; "
                    f"recreate {filename} from plasticity.h5"
                )
        if (
            "transolver" in fresh_selected
            and static_contract["transolver_working_copy_sha256_at_start"] is not None
            and static_contract["transolver_working_copy_sha256_at_start"]
            != static_contract["original_hdf5_sha256"]
        ):
            raise CampaignError(
                "Fresh Transolver campaign requires an absent or bit-identical pristine "
                "plasticity_transolver_runtime.h5"
            )

        if dry_run:
            if selected_ambiguous:
                raise CampaignError(
                    "Refusing ambiguous preexisting selected-model outputs: "
                    + json.dumps(selected_ambiguous, sort_keys=True)
                )
            if stale_comparison:
                raise CampaignError(
                    "Comparison artifacts exist although all seven model outputs are not valid"
                )
            self._preflight_all_train_configs(dry_run=True)
            self._dry_run_plan(selected=selected, assessments=assessments)
            return 0

        self.status = self._initial_status(
            selected=selected,
            static_contract=static_contract,
            assessments=assessments,
            paper_gate=paper_gate,
            execution_plan=self.execution_plan,
        )
        self._write_status()
        try:
            if selected_ambiguous:
                raise CampaignError(
                    "Refusing ambiguous preexisting selected-model outputs. Move or remove only "
                    "the named model output before retrying: "
                    + json.dumps(selected_ambiguous, sort_keys=True)
                )
            if stale_comparison:
                raise CampaignError(
                    "Refusing stale comparison artifacts because all seven model outputs are not valid"
                )

            materialized = self._materialize_runtime_configs(fresh_selected)
            with self._status_lock:
                self.status["runtime_materialization"] = materialized
                self.status["execution_plan"] = self.execution_plan
                plan_models = self.execution_plan.get("models")
                if isinstance(plan_models, dict):
                    for model, placement in plan_models.items():
                        if isinstance(placement, dict):
                            self._model_status(model)["placement"] = placement
                self._write_status()
            self._preflight_all_train_configs(dry_run=False)
            for model in selected:
                model_status = self._model_status(model)
                if model_status.get("state") == "complete":
                    print(f"[{model}] skipped: strict recovery validation passed", flush=True)
            self._run_scheduled_models()

            if self._all_models_complete():
                print("[campaign] comparing all seven validated results", flush=True)
                self._run_comparison()
            else:
                self.status["state"] = "incomplete"
                self.status["comparison"] = {
                    "state": "not_run",
                    "reason": "one or more of the seven models is not complete",
                }
                self._write_status()
            return 0
        except BaseException as exc:
            self._cancel_active_jobs()
            with self._status_lock:
                self.status["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                self.status["complete"] = False
                self.status["last_error"] = f"{type(exc).__name__}: {exc}"
                models_status = self.status.get("models")
                if isinstance(models_status, dict):
                    for value in models_status.values():
                        if isinstance(value, dict) and value.get("state") == "running":
                            value["state"] = self.status["state"]
                            value["error"] = self.status["last_error"]
                            for phase in ("train", "infer", "evaluate"):
                                phase_status = value.get(phase)
                                if isinstance(phase_status, dict) and phase_status.get("state") == "running":
                                    phase_status["state"] = self.status["state"]
                                    phase_status["finished_at"] = _utc_now()
                                    phase_status["error"] = self.status["last_error"]
                self._write_status()
            raise


def _parse_models(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return CAMPAIGN_MODELS
    requested: list[str] = []
    for value in values:
        requested.extend(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(requested) - set(CAMPAIGN_MODELS))
    if unknown:
        raise CampaignError(f"Unknown --models values: {unknown}")
    return tuple(model for model in CAMPAIGN_MODELS if model in set(requested))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Canonical subset for recovery/tests; space- or comma-separated",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run read-only gates/preflights and print the exact plan without launching models",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python used for the suite launcher and benchmark tools",
    )
    parser.add_argument(
        "--allow-incomplete-paper-validation",
        action="store_true",
        help=(
            "Explicitly record a user-directed Plasticity priority waiver when the "
            "strict paper-validation gate is absent or incomplete"
        ),
    )
    parser.add_argument(
        "--epoch-budget",
        type=int,
        default=500,
        help=(
            "Training epochs for this campaign; default 500 preserves the canonical "
            "budget, while a smaller explicit value creates an auditable preliminary run"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    suite_root = Path(__file__).resolve().parents[3]
    try:
        models = _parse_models(args.models)
        return PlasticityCampaign(
            suite_root,
            python_executable=args.python,
            allow_incomplete_paper_validation=args.allow_incomplete_paper_validation,
            epoch_budget=args.epoch_budget,
        ).run(models=models, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("Plasticity campaign interrupted.", file=sys.stderr)
        return 130
    except CampaignError as exc:
        print(f"Plasticity campaign refused/failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
