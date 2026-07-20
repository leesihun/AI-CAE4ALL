#!/usr/bin/env python3
"""Isolated, fail-closed resource probes for the Plasticity benchmark.

This utility is deliberately separate from the production campaign.  It builds
a deterministic small HDF5 from the provenance-pinned source, writes runtime
configs and outputs below an isolated probe run root, and can run either one
model or an explicitly requested two-model GPU pair.  It never edits the
checked-in configs, benchmark source HDF5, or canonical campaign outputs.

Real execution normally requires both ``--execute`` and a strict
machine-readable paper-validation completion gate.  An explicit
``--allow-incomplete-paper-validation`` flag records a user-directed Plasticity
priority waiver without mislabeling unfinished paper validation as passed.
``--dry-run`` is read-only and never creates the probe dataset, runtime configs,
output directories, or result records.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
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
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA_VERSION = "plasticity_resource_probe_v1"
GATE_SCHEMA_VERSION = "paper_validation_completion_gate_v1"
GATE_STATUS = "passed"
GATE_WAIVER_STATUS = "waived_by_user"
REQUIRED_PAPER_VALIDATIONS = (
    "fno",
    "transolver",
    "deeponet",
    "point_deeponet",
    "gino",
)

MODELS = (
    "meshgraphnets",
    "hi_meshgraphnets",
    "deeponet",
    "point_deeponet",
    "fno",
    "gino",
    "transolver",
)
MGN_MODELS = frozenset({"meshgraphnets", "hi_meshgraphnets"})
WRITABLE_DATASET_MODELS = frozenset({"meshgraphnets", "hi_meshgraphnets", "transolver"})

DEFAULT_CASES = 32
DEFAULT_SEED = 42
DEFAULT_GPU_PEAK_LIMIT_MIB = 6656
DEFAULT_PAIR_IMPROVEMENT = 0.10
DEFAULT_MAX_BASELINE_USED_MIB = 1536
DEFAULT_POLL_SECONDS = 0.5

PROFILES: dict[str, dict[str, tuple[int, int]]] = {
    "meshgraphnets": {
        "canonical": (4, 1),
        "fallback_2x2": (2, 2),
        "fallback_1x4": (1, 4),
    },
    "hi_meshgraphnets": {
        "canonical": (4, 1),
        "fallback_2x2": (2, 2),
        "fallback_1x4": (1, 4),
    },
    "deeponet": {
        "canonical": (4, 1),
        "fallback_2x2": (2, 2),
        "fallback_1x4": (1, 4),
    },
    "point_deeponet": {
        "canonical": (2, 2),
    },
    "fno": {
        "canonical": (4, 1),
        "fallback_2x2": (2, 2),
        "fallback_1x4": (1, 4),
    },
    "gino": {"canonical": (1, 4)},
    "transolver": {"canonical": (1, 4)},
}
DEFAULT_PROFILE = {model: "canonical" for model in MODELS}
DEFAULT_PROFILE["hi_meshgraphnets"] = "fallback_1x4"

PROFILE_ALIASES = {
    "2x2": "fallback_2x2",
    "1x4": "fallback_1x4",
}

PROGRESS_RE = re.compile(r"(?P<done>\d+)\s*/\s*(?P<total>\d+)")
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


class ProbeError(RuntimeError):
    """A fail-closed probe refusal or execution failure."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
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


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    # Reject NaN/Infinity so every record is strict JSON consumable outside
    # Python (in particular by the campaign scheduler).
    _atomic_write_text(
        path,
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"Cannot read JSON object {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProbeError(f"JSON root must be an object: {path}")
    return payload


def validate_completion_gate(path: Path) -> dict[str, object]:
    """Validate the explicit all-paper-validations completion gate."""
    path = path.resolve()
    if not path.is_file():
        raise ProbeError(
            "Real resource probes are locked until the paper-validation completion "
            f"gate exists: {path}"
        )
    payload = _read_json_object(path)
    if payload.get("schema_version") != GATE_SCHEMA_VERSION:
        raise ProbeError(
            f"Paper-validation gate schema must be {GATE_SCHEMA_VERSION!r}: {path}"
        )
    if payload.get("complete") is not True or payload.get("status") != GATE_STATUS:
        raise ProbeError("Paper-validation gate is not explicitly complete and passed")
    validations = payload.get("validations")
    if not isinstance(validations, dict):
        raise ProbeError("Paper-validation gate is missing its validations object")
    failures: list[str] = []
    for name in REQUIRED_PAPER_VALIDATIONS:
        value = validations.get(name)
        if not isinstance(value, dict):
            failures.append(f"{name}=missing")
            continue
        if value.get("complete") is not True or value.get("status") != GATE_STATUS:
            failures.append(
                f"{name}=complete:{value.get('complete')!r},status:{value.get('status')!r}"
            )
    if failures:
        raise ProbeError(
            "Paper-validation gate has unfinished or failed validations: " + ", ".join(failures)
        )
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema_version": payload["schema_version"],
        "status": payload["status"],
        "complete": True,
        "validations": list(REQUIRED_PAPER_VALIDATIONS),
    }


def completion_gate_or_user_waiver(
    path: Path, *, allow_incomplete: bool
) -> dict[str, object]:
    """Return the strict gate, or an explicit non-passing user waiver."""
    try:
        return validate_completion_gate(path)
    except ProbeError as exc:
        if not allow_incomplete:
            raise
        resolved = path.resolve()
        return {
            "path": str(resolved),
            "sha256": _sha256_file(resolved) if resolved.is_file() else None,
            "schema_version": GATE_SCHEMA_VERSION,
            "status": GATE_WAIVER_STATUS,
            "complete": False,
            "validations": [],
            "authorization": {
                "type": "explicit_cli_override",
                "scope": "plasticity_resource_probe",
                "reason": "user redirected priority to the Plasticity campaign",
                "strict_gate_error": str(exc),
            },
        }


def _suite_split(sample_ids: Sequence[int], seed: int) -> tuple[np.ndarray, ...]:
    shuffled = np.asarray(sample_ids, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * 0.8)
    n_val = int(len(shuffled) * 0.1)
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


def deterministic_probe_ids(sample_ids: Sequence[int], cases: int, seed: int) -> tuple[int, ...]:
    """Choose probe cases only from the suite's deterministic training partition."""
    if cases <= 0:
        raise ProbeError("Probe case count must be positive")
    unique = sorted(set(int(value) for value in sample_ids))
    if len(unique) != len(sample_ids):
        raise ProbeError("Source sample IDs are not unique")
    train_ids, _, _ = _suite_split(unique, seed)
    if cases > len(train_ids):
        raise ProbeError(
            f"Requested {cases} probe cases but only {len(train_ids)} suite-training cases exist"
        )
    return tuple(sorted(int(value) for value in train_ids[:cases]))


def _filtered_split(source: h5py.File, name: str, selected: set[int]) -> np.ndarray:
    values = np.asarray(source[f"splits/{name}"][:], dtype=np.int64)
    return np.asarray([int(value) for value in values if int(value) in selected], dtype=np.int64)


def _copy_probe_hdf5(source: Path, temporary: Path, selected_ids: Sequence[int], seed: int) -> None:
    selected = set(int(value) for value in selected_ids)
    with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
        if "data" not in src or "metadata" not in src or "splits" not in src or "topology" not in src:
            raise ProbeError("Plasticity source HDF5 is missing a required root group")
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["num_samples"] = len(selected_ids)
        dst.attrs["resource_probe"] = True
        dst.attrs["resource_probe_cases"] = len(selected_ids)
        dst.attrs["resource_probe_seed"] = seed
        dst.attrs["resource_probe_source_split"] = "suite_seed42_train"
        dst.attrs["resource_probe_selected_ids_sha256"] = _sha256_text(
            ",".join(str(value) for value in selected_ids)
        )

        src.copy("topology", dst)
        metadata = dst.create_group("metadata")
        for name in src["metadata"]:
            if name != "splits":
                src.copy(src[f"metadata/{name}"], metadata, name=name)

        splits = dst.create_group("splits")
        for name in ("train", "val", "test", "unused"):
            splits.create_dataset(name, data=_filtered_split(src, name, selected))
        metadata["splits"] = splits

        data = dst.create_group("data")
        topology_edge = dst.get("topology/mesh_edge_structured_quad")
        for sample_id in selected_ids:
            source_group = src[f"data/{sample_id}"]
            destination_group = data.create_group(str(sample_id))
            for name in source_group:
                if name == "mesh_edge" and topology_edge is not None:
                    destination_group[name] = topology_edge
                else:
                    src.copy(source_group[name], destination_group, name=name)


def audit_probe_hdf5(path: Path, selected_ids: Sequence[int], seed: int) -> dict[str, object]:
    expected = tuple(sorted(int(value) for value in selected_ids))
    with h5py.File(path, "r") as handle:
        actual = tuple(sorted(int(key) for key in handle["data"].keys()))
        if actual != expected:
            raise ProbeError("Probe HDF5 sample IDs differ from the deterministic selection")
        if int(handle.attrs.get("num_samples", -1)) != len(expected):
            raise ProbeError("Probe HDF5 num_samples attribute is inconsistent")
        if bool(handle.attrs.get("resource_probe", False)) is not True:
            raise ProbeError("Probe HDF5 is missing its resource_probe marker")
        if int(handle.attrs.get("resource_probe_seed", -1)) != seed:
            raise ProbeError("Probe HDF5 seed marker is inconsistent")
        if set(handle.keys()) != {"data", "metadata", "splits", "topology"}:
            raise ProbeError("Probe HDF5 root schema differs from the Plasticity schema")
        first = handle[f"data/{expected[0]}"]
        required_sample = {"nodal_data", "mesh_edge", "die_profile", "metadata"}
        if set(first.keys()) != required_sample:
            raise ProbeError("Probe HDF5 sample schema is incomplete")
        num_timesteps = int(handle.attrs.get("num_timesteps", first["nodal_data"].shape[1]))
        return {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "cases": len(expected),
            "seed": seed,
            "selected_ids": list(expected),
            "selected_ids_sha256": _sha256_text(",".join(str(value) for value in expected)),
            "num_timesteps": num_timesteps,
        }


def prepare_probe_hdf5(
    source: Path,
    output: Path,
    *,
    cases: int = DEFAULT_CASES,
    seed: int = DEFAULT_SEED,
    expected_source_sha256: str | None = None,
) -> dict[str, object]:
    """Atomically create and validate a deterministic probe HDF5."""
    source = source.resolve()
    output = output.resolve()
    if not source.is_file():
        raise ProbeError(f"Plasticity source HDF5 is missing: {source}")
    if output.exists():
        raise ProbeError(f"Refusing to overwrite an existing probe HDF5: {output}")
    source_stat_before = source.stat()
    source_hash_before = _sha256_file(source)
    if expected_source_sha256 and source_hash_before != expected_source_sha256.lower():
        raise ProbeError(
            "Plasticity source HDF5 does not match provenance: "
            f"actual={source_hash_before}, expected={expected_source_sha256.lower()}"
        )
    with h5py.File(source, "r") as handle:
        sample_ids = sorted(int(key) for key in handle["data"].keys())
    selected_ids = deterministic_probe_ids(sample_ids, cases, seed)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        _copy_probe_hdf5(source, temporary, selected_ids, seed)
        result = audit_probe_hdf5(temporary, selected_ids, seed)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    source_stat_after = source.stat()
    source_hash_after = _sha256_file(source)
    if (
        source_stat_before.st_size != source_stat_after.st_size
        or source_stat_before.st_mtime_ns != source_stat_after.st_mtime_ns
        or source_hash_before != source_hash_after
    ):
        raise ProbeError("Authoritative Plasticity source changed while preparing the probe")
    result.update(
        {
            "path": str(output),
            "source": str(source),
            "source_sha256": source_hash_after,
        }
    )
    return result


def _flat_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("%") or stripped.startswith("#") or stripped == "'":
            continue
        fields = stripped.split(None, 1)
        if len(fields) == 2:
            values[fields[0]] = fields[1].strip()
    return values


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


def resolve_profile(model: str, profile: str | None) -> tuple[str, int, int]:
    if model not in MODELS:
        raise ProbeError(f"Unknown Plasticity probe model: {model}")
    name = PROFILE_ALIASES.get(profile or DEFAULT_PROFILE[model], profile or DEFAULT_PROFILE[model])
    if name not in PROFILES[model]:
        raise ProbeError(
            f"Profile {name!r} is not available for {model}; expected {sorted(PROFILES[model])}"
        )
    batch_size, grad_accum_steps = PROFILES[model][name]
    return name, batch_size, grad_accum_steps


def materialize_runtime_config(
    canonical: Path,
    output: Path,
    *,
    model: str,
    profile: str,
    batch_size: int,
    grad_accum_steps: int,
    device: str,
    gpu_id: int,
    dataset: Path,
    model_root: Path,
) -> dict[str, object]:
    if not canonical.is_file():
        raise ProbeError(f"Canonical Plasticity config is missing: {canonical}")
    canonical_values = _flat_config(canonical)
    expected_native = "meshgraphnets" if model in MGN_MODELS else model
    if canonical_values.get("model", "").lower() != expected_native:
        raise ProbeError(f"Canonical config model identity is wrong for {model}: {canonical}")
    if canonical_values.get("mode") != "train":
        raise ProbeError(f"Canonical resource probe config must be a train config: {canonical}")

    suite_root = canonical.resolve().parents[3]
    backend_name = (
        "MeshGraphNets"
        if model in MGN_MODELS
        else "transolver"
        if model == "transolver"
        else "Neural_Operator"
    )
    log_base = suite_root / backend_name / "outputs"
    log_target = (model_root / "train.log").resolve()
    # All three backend log initializers prepend the literal ``outputs/``.
    # Supplying a Windows absolute path would therefore produce ``outputs/C:``.
    log_config_value = os.path.relpath(log_target, log_base.resolve())

    updates = {
        "gpu_ids": str(gpu_id if device == "gpu" else -1),
        "modelpath": str((model_root / "model.pth").resolve()),
        "dataset_dir": str(dataset.resolve()),
        "infer_dataset": str(dataset.resolve()),
        "inference_output_dir": str((model_root / "inference").resolve()),
        "log_file_dir": log_config_value,
        "training_epochs": "1",
        "batch_size": str(batch_size),
        "grad_accum_steps": str(grad_accum_steps),
    }
    if "checkpoint_interval" in canonical_values:
        updates["checkpoint_interval"] = "0"
    if device == "cpu":
        updates["use_amp"] = "False"
        updates["num_workers"] = "0"

    rendered = _render_runtime_config(canonical, updates)
    _atomic_write_text(output, rendered)
    parsed = _flat_config(output)
    for key, expected in updates.items():
        if parsed.get(key) != expected:
            raise ProbeError(f"Runtime config materialization failed for {key}: {output}")
    for path_key in ("modelpath", "dataset_dir", "infer_dataset", "inference_output_dir", "log_file_dir"):
        value = (
            (log_base / parsed[path_key]).resolve()
            if path_key == "log_file_dir"
            else Path(parsed[path_key]).resolve()
        )
        if path_key != "dataset_dir" and path_key != "infer_dataset":
            try:
                value.relative_to(model_root.resolve())
            except ValueError as exc:
                raise ProbeError(f"Runtime output path escaped isolated model root: {value}") from exc
    return {
        "canonical": str(canonical.resolve()),
        "canonical_sha256": _sha256_file(canonical),
        "runtime": str(output.resolve()),
        "runtime_sha256": _sha256_file(output),
        "profile": profile,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "device": device,
        "gpu_ids": parsed["gpu_ids"],
        "updates": updates,
    }


def parse_completed_batches(text: str, expected_batches: int) -> list[int]:
    """Return completed train-batch totals parsed from tqdm output."""
    completed: list[int] = []
    for match in PROGRESS_RE.finditer(text.replace("\r", "\n")):
        done = int(match.group("done"))
        total = int(match.group("total"))
        if total == expected_batches and done == total:
            completed.append(done)
    return completed


def expected_training_work(dataset_record: Mapping[str, object], batch_size: int) -> dict[str, int]:
    cases = int(dataset_record["cases"])
    timesteps = int(dataset_record["num_timesteps"])
    train_cases = int(cases * 0.8)
    train_pairs = train_cases * max(timesteps - 1, 1)
    return {
        "train_cases": train_cases,
        "train_pairs": train_pairs,
        "expected_batches_per_epoch": math.ceil(train_pairs / batch_size),
        "epochs": 1,
    }


class ProbeLock:
    """Kernel-owned lock shared with the canonical Plasticity campaign."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> "ProbeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise ProbeError(
                f"Another Plasticity campaign/resource probe owns the lock: {self.path}"
            ) from exc
        self._handle = handle
        metadata = {
            "action": "plasticity_resource_probe",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": _utc_now(),
            "argv": sys.argv,
        }
        encoded = (json.dumps(metadata, sort_keys=True) + "\n").encode("utf-8")
        handle.seek(0)
        handle.truncate()
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
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


class _WindowsJob:
    """Kill-on-close Job Object for a launcher and every descendant process."""

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    EXTENDED_LIMIT_INFORMATION_CLASS = 9

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
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

        class IoCounters(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimit),
                ("IoInfo", IoCounters),
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
            limits = ExtendedLimit()
            limits.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                handle,
                self.EXTENDED_LIMIT_INFORMATION_CLASS,
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


class ManagedChild:
    """One owned launcher process tree with isolated stdout/stderr files."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        stdout_handle,
        stderr_handle,
        windows_job: _WindowsJob | None,
    ) -> None:
        self.process = process
        self.stdout_handle = stdout_handle
        self.stderr_handle = stderr_handle
        self.windows_job = windows_job

    @classmethod
    def start(
        cls,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
    ) -> "ManagedChild":
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("wb")
        stderr_handle = stderr_path.open("wb")
        kwargs: dict[str, object] = {
            "cwd": cwd,
            "env": dict(env),
            "stdout": stdout_handle,
            "stderr": stderr_handle,
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(list(command), **kwargs)  # type: ignore[arg-type]
            windows_job = _WindowsJob(process) if os.name == "nt" else None
            return cls(process, stdout_handle, stderr_handle, windows_job)
        except BaseException:
            # A process can exist even when Windows job-object construction
            # fails.  Do not leave that newly owned launcher running.
            if process is not None and process.poll() is None:
                try:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            stdout_handle.close()
            stderr_handle.close()
            raise

    def terminate_tree(self) -> None:
        if self.process.poll() is None:
            if os.name == "nt":
                try:
                    self.process.send_signal(signal.CTRL_BREAK_EVENT)
                except (OSError, ValueError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if self.windows_job is not None:
                        self.windows_job.terminate(1)
                    else:
                        self.process.kill()
            else:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def close(self) -> None:
        self.stdout_handle.close()
        self.stderr_handle.close()
        if self.windows_job is not None:
            self.windows_job.close()


@dataclass(frozen=True)
class GpuSample:
    timestamp: float
    total_mib: int
    used_mib: int
    utilization_percent: int


def query_nvidia_smi(gpu_id: int) -> GpuSample:
    command = [
        "nvidia-smi",
        f"--id={gpu_id}",
        "--query-gpu=memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    kwargs: dict[str, object] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": 10,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        completed = subprocess.run(command, **kwargs)  # type: ignore[arg-type]
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeError(f"nvidia-smi query failed: {type(exc).__name__}: {exc}") from exc
    if completed.returncode != 0:
        raise ProbeError(
            f"nvidia-smi query returned {completed.returncode}: {(completed.stderr or '').strip()}"
        )
    row = next(csv.reader([completed.stdout.strip()]), None)
    if row is None or len(row) != 3:
        raise ProbeError(f"Unexpected nvidia-smi output: {completed.stdout!r}")
    try:
        total, used, utilization = (int(value.strip()) for value in row)
    except ValueError as exc:
        raise ProbeError(f"Non-integer nvidia-smi output: {row!r}") from exc
    return GpuSample(time.time(), total, used, utilization)


class GpuMonitor:
    def __init__(
        self,
        gpu_id: int,
        poll_seconds: float,
        sample_provider: Callable[[int], GpuSample] = query_nvidia_smi,
    ) -> None:
        self.gpu_id = gpu_id
        self.poll_seconds = poll_seconds
        self.sample_provider = sample_provider
        self.samples: list[GpuSample] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, baseline: GpuSample) -> None:
        self.samples.append(baseline)

        def poll() -> None:
            while not self._stop.wait(self.poll_seconds):
                try:
                    self.samples.append(self.sample_provider(self.gpu_id))
                except Exception as exc:  # monitoring failure must be recorded, not hidden
                    self.errors.append(f"{type(exc).__name__}: {exc}")

        self._thread = threading.Thread(target=poll, name="plasticity-gpu-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, object]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.poll_seconds * 3))
        if not self.samples:
            raise ProbeError("GPU monitor did not collect any samples")
        used = [sample.used_mib for sample in self.samples]
        total_values = {sample.total_mib for sample in self.samples}
        if len(total_values) != 1:
            raise ProbeError("GPU total memory changed during the probe")
        return {
            "gpu_id": self.gpu_id,
            "total_mib": next(iter(total_values)),
            "baseline_used_mib": self.samples[0].used_mib,
            "baseline_utilization_percent": self.samples[0].utilization_percent,
            "peak_total_used_mib": max(used),
            "minimum_total_used_mib": min(used),
            "samples": len(self.samples),
            "poll_seconds": self.poll_seconds,
            "monitor_errors": list(self.errors),
            "complete": not self.errors,
        }


def child_environment(device: str, cpu_threads: int) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    recorded: dict[str, str] = {}
    if device == "cpu":
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = str(cpu_threads)
            recorded[name] = str(cpu_threads)
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        recorded["CUDA_VISIBLE_DEVICES"] = "-1"
    return env, recorded


def _read_logs(stdout_path: Path, stderr_path: Path) -> str:
    return stdout_path.read_text(encoding="utf-8", errors="replace") + "\n" + stderr_path.read_text(
        encoding="utf-8", errors="replace"
    )


def summarize_child(
    *,
    model: str,
    profile: str,
    batch_size: int,
    grad_accum_steps: int,
    returncode: int,
    wall_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
    checkpoint: Path,
    work: Mapping[str, int],
) -> dict[str, object]:
    expected_batches = int(work["expected_batches_per_epoch"])
    completed = parse_completed_batches(_read_logs(stdout_path, stderr_path), expected_batches)
    checkpoint_ok = checkpoint.is_file() and checkpoint.stat().st_size > 0
    complete = returncode == 0 and bool(completed) and checkpoint_ok
    train_pairs = int(work["train_pairs"]) if complete else 0
    return {
        "model": model,
        "profile": profile,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": batch_size * grad_accum_steps,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "checkpoint": str(checkpoint),
        "checkpoint_present": checkpoint_ok,
        "expected_batches_per_epoch": expected_batches,
        "completed_batches_per_epoch": completed,
        "train_pairs_processed": train_pairs,
        "throughput_pairs_per_second": train_pairs / wall_seconds if train_pairs and wall_seconds > 0 else 0.0,
        "complete": complete,
    }


@dataclass(frozen=True)
class ModelPlan:
    model: str
    profile: str
    batch_size: int
    grad_accum_steps: int
    canonical_config: Path
    runtime_config: Path
    dataset: Path
    model_root: Path
    command: tuple[str, ...]
    environment_overrides: dict[str, str]
    work: dict[str, int]


def canonical_config_path(suite_root: Path, model: str) -> Path:
    return suite_root / "configs" / "benchmarks" / "plasticity" / f"config_train_{model}.txt"


def _parse_profile_arguments(models: Sequence[str], values: Sequence[str] | None) -> dict[str, str]:
    resolved = {model: DEFAULT_PROFILE[model] for model in models}
    if not values:
        return resolved
    for raw in values:
        if "=" in raw:
            model, value = raw.split("=", 1)
            model = model.strip().lower()
            if model not in models:
                raise ProbeError(f"Profile assignment targets an unselected model: {raw}")
            resolved[model] = value.strip().lower()
        elif len(models) == 1:
            resolved[models[0]] = raw.strip().lower()
        else:
            raise ProbeError("Pair probes require keyed profiles such as fno=fallback_1x4")
    return resolved


def _provenance_source_hash(suite_root: Path) -> str:
    path = suite_root / "dataset" / "benchmarks" / "plasticity" / "plasticity.provenance.json"
    payload = _read_json_object(path)
    value = str(payload.get("converted_hdf5_sha256", "")).lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ProbeError(f"Plasticity provenance has no valid converted HDF5 SHA256: {path}")
    return value


def _atomic_copy(source: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        raise ProbeError(f"Refusing to overwrite isolated probe copy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        source_hash = _sha256_file(source)
        copy_hash = _sha256_file(temporary)
        if copy_hash != source_hash:
            raise ProbeError(f"Probe HDF5 copy hash mismatch: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": copy_hash,
        "copied_from": str(source.resolve()),
        "writable_runtime_copy": True,
    }


def _runtime_dataset_path(run_root: Path, model: str, common_dataset: Path) -> Path:
    if model in WRITABLE_DATASET_MODELS:
        return run_root / "datasets" / f"plasticity_probe_{model}_runtime.h5"
    return common_dataset


def build_dry_run_plan(
    *,
    suite_root: Path,
    output_root: Path,
    run_id: str,
    models: Sequence[str],
    profiles: Mapping[str, str],
    device: str,
    gpu_id: int,
    cpu_threads: int,
    cases: int,
    seed: int,
    python_executable: str,
    gate_path: Path,
) -> dict[str, object]:
    source = suite_root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
    if not source.is_file():
        raise ProbeError(f"Plasticity source HDF5 is missing: {source}")
    run_root = output_root / "runs" / run_id
    common_dataset = run_root / "datasets" / f"plasticity_probe_{cases}_seed{seed}.h5"
    env, recorded_env = child_environment(device, cpu_threads)
    del env
    planned_models: dict[str, object] = {}
    for model in models:
        canonical = canonical_config_path(suite_root, model)
        if not canonical.is_file():
            raise ProbeError(f"Canonical Plasticity config is missing: {canonical}")
        profile, batch_size, grad_accum_steps = resolve_profile(model, profiles.get(model))
        runtime_config = run_root / "configs" / f"config_train_{model}_{profile}.txt"
        model_root = run_root / "models" / model
        dataset = _runtime_dataset_path(run_root, model, common_dataset)
        command = (
            python_executable,
            str((suite_root / "AI_CAE4ALL_main.py").resolve()),
            "--config",
            str(runtime_config.resolve()),
            "--no-color",
        )
        planned_models[model] = {
            "profile": profile,
            "batch_size": batch_size,
            "grad_accum_steps": grad_accum_steps,
            "effective_batch_size": batch_size * grad_accum_steps,
            "canonical_config": str(canonical.resolve()),
            "runtime_config": str(runtime_config.resolve()),
            "dataset": str(dataset.resolve()),
            "model_root": str(model_root.resolve()),
            "command": list(command),
            "environment_overrides": recorded_env,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "dry_run": True,
        "real_training_launched": False,
        "run_id": run_id,
        "mode": "pair" if len(models) == 2 else "single",
        "device": device,
        "models": planned_models,
        "dataset": {
            "source": str(source.resolve()),
            "would_create": str(common_dataset.resolve()),
            "cases": cases,
            "seed": seed,
            "writable_runtime_copies_are_separate": sorted(WRITABLE_DATASET_MODELS),
        },
        "paper_validation_gate": {
            "required_for_execute": True,
            "path": str(gate_path.resolve()),
            "validated_in_dry_run": False,
        },
        "output_root": str(run_root.resolve()),
    }


def prepare_model_plans(
    *,
    suite_root: Path,
    run_root: Path,
    models: Sequence[str],
    profiles: Mapping[str, str],
    device: str,
    gpu_id: int,
    cpu_threads: int,
    python_executable: str,
    common_dataset: Path,
    dataset_record: Mapping[str, object],
) -> tuple[dict[str, ModelPlan], dict[str, object]]:
    _, recorded_env = child_environment(device, cpu_threads)
    plans: dict[str, ModelPlan] = {}
    dataset_copies: dict[str, object] = {}
    for model in models:
        profile, batch_size, grad_accum_steps = resolve_profile(model, profiles.get(model))
        dataset = _runtime_dataset_path(run_root, model, common_dataset)
        if model in WRITABLE_DATASET_MODELS:
            dataset_copies[model] = _atomic_copy(common_dataset, dataset)
            if dataset.resolve() in {common_dataset.resolve(), (suite_root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5").resolve()}:
                raise ProbeError(f"Writable probe dataset aliases a protected source: {dataset}")
        canonical = canonical_config_path(suite_root, model)
        runtime_config = run_root / "configs" / f"config_train_{model}_{profile}.txt"
        model_root = run_root / "models" / model
        config_record = materialize_runtime_config(
            canonical,
            runtime_config,
            model=model,
            profile=profile,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            device=device,
            gpu_id=gpu_id,
            dataset=dataset,
            model_root=model_root,
        )
        work = expected_training_work(dataset_record, batch_size)
        command = (
            python_executable,
            str((suite_root / "AI_CAE4ALL_main.py").resolve()),
            "--config",
            str(runtime_config.resolve()),
            "--no-color",
        )
        plans[model] = ModelPlan(
            model=model,
            profile=profile,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            canonical_config=canonical,
            runtime_config=runtime_config,
            dataset=dataset,
            model_root=model_root,
            command=command,
            environment_overrides=dict(recorded_env),
            work=work,
        )
        config_record["work"] = work
        config_record["command"] = list(command)
        dataset_copies[f"{model}_config"] = config_record
    return plans, dataset_copies


def _single_projection(model_record: Mapping[str, object]) -> dict[str, object]:
    throughput = float(model_record.get("throughput_pairs_per_second", 0.0))
    full_pairs = 789 * 19 * 500
    projected_seconds = full_pairs / throughput if throughput > 0 else None
    return {
        "probe_train_pairs": int(model_record.get("train_pairs_processed", 0)),
        "probe_wall_seconds": float(model_record.get("wall_seconds", 0.0)),
        "projected_full_train_pairs": full_pairs,
        "projected_wall_seconds_500_epochs": projected_seconds,
        "projected_wall_hours_500_epochs": (
            projected_seconds / 3600.0 if projected_seconds is not None else None
        ),
        "basis": "32-case one-epoch end-to-end throughput",
        "conservative": True,
        "includes_probe_startup_validation_test_and_checkpoint_overhead_per_projected_epoch": True,
    }


def load_pair_baselines(
    paths: Sequence[Path],
    *,
    models: Sequence[str],
    profiles: Mapping[str, str],
    dataset_record: Mapping[str, object],
) -> list[dict[str, object]]:
    if len(paths) != 2:
        raise ProbeError("A pair probe requires exactly two --baseline-record paths")
    expected_ids_hash = dataset_record.get("selected_ids_sha256")
    expected_source_hash = dataset_record.get("source_sha256")
    by_model: dict[str, dict[str, object]] = {}
    for path in paths:
        path = path.resolve()
        payload = _read_json_object(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ProbeError(f"Baseline has the wrong resource-probe schema: {path}")
        if payload.get("mode") != "single" or payload.get("device") != "gpu":
            raise ProbeError(f"Baseline must be a completed single GPU probe: {path}")
        if payload.get("state") != "complete" or payload.get("complete") is not True:
            raise ProbeError(f"Baseline probe is not complete: {path}")
        dataset = payload.get("dataset")
        records = payload.get("models")
        if not isinstance(dataset, dict) or not isinstance(records, dict) or len(records) != 1:
            raise ProbeError(f"Baseline probe has invalid dataset/model records: {path}")
        if (
            dataset.get("selected_ids_sha256") != expected_ids_hash
            or dataset.get("source_sha256") != expected_source_hash
        ):
            raise ProbeError(f"Baseline probe dataset identity differs from the pair probe: {path}")
        model = next(iter(records))
        record = records[model]
        if model not in models or not isinstance(record, dict):
            raise ProbeError(f"Baseline model is not one of the requested pair: {path}")
        expected_profile, _, _ = resolve_profile(model, profiles.get(model))
        if record.get("profile") != expected_profile or record.get("complete") is not True:
            raise ProbeError(f"Baseline profile/completion does not match pair model {model}: {path}")
        try:
            wall_seconds = float(record["wall_seconds"])
            train_pairs = int(record["train_pairs_processed"])
            throughput = float(record["throughput_pairs_per_second"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProbeError(f"Baseline has malformed timing/work fields: {path}") from exc
        if (
            wall_seconds <= 0
            or train_pairs <= 0
            or throughput <= 0
            or not math.isfinite(wall_seconds)
            or not math.isfinite(throughput)
        ):
            raise ProbeError(f"Baseline timing/work fields must be finite and positive: {path}")
        if model in by_model:
            raise ProbeError(f"Duplicate baseline for pair model {model}")
        by_model[model] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "model": model,
            "profile": record["profile"],
            "wall_seconds": wall_seconds,
            "train_pairs_processed": train_pairs,
            "throughput_pairs_per_second": throughput,
        }
    if set(by_model) != set(models):
        raise ProbeError("Baseline records do not cover both requested pair models")
    return [by_model[model] for model in models]


def certify_pair(
    *,
    model_records: Mapping[str, Mapping[str, object]],
    baselines: Sequence[Mapping[str, object]],
    pair_wall_seconds: float,
    gpu_record: Mapping[str, object],
    peak_limit_mib: int = DEFAULT_GPU_PEAK_LIMIT_MIB,
    required_improvement: float = DEFAULT_PAIR_IMPROVEMENT,
) -> dict[str, object]:
    reasons: list[str] = []
    models_complete = all(record.get("complete") is True for record in model_records.values())
    if not models_complete:
        reasons.append("one or both pair children did not complete")
    baseline_by_model = {str(value["model"]): value for value in baselines}
    pair_pairs = sum(int(record.get("train_pairs_processed", 0)) for record in model_records.values())
    baseline_pairs = sum(int(value.get("train_pairs_processed", 0)) for value in baselines)
    if pair_pairs <= 0 or baseline_pairs != pair_pairs:
        reasons.append("pair and individual baseline work counts do not match")
    sequential_seconds = sum(float(value.get("wall_seconds", 0.0)) for value in baselines)
    sequential_rate = baseline_pairs / sequential_seconds if baseline_pairs > 0 and sequential_seconds > 0 else 0.0
    pair_rate = pair_pairs / pair_wall_seconds if pair_pairs > 0 and pair_wall_seconds > 0 else 0.0
    improvement = pair_rate / sequential_rate - 1.0 if sequential_rate > 0 else None
    throughput_pass = improvement is not None and improvement >= required_improvement
    if not throughput_pass:
        observed = "unavailable" if improvement is None else f"{improvement:.6f}"
        reasons.append(f"aggregate throughput improvement {observed} is below {required_improvement:.6f}")
    monitor_complete = gpu_record.get("complete") is True
    peak = int(gpu_record.get("peak_total_used_mib", 2**31 - 1))
    peak_pass = monitor_complete and peak <= peak_limit_mib
    if not peak_pass:
        reasons.append(f"GPU peak {peak} MiB exceeds {peak_limit_mib} MiB or monitoring was incomplete")
    certified = models_complete and pair_pairs == baseline_pairs and throughput_pass and peak_pass
    return {
        "certified": certified,
        "peak_limit_mib": peak_limit_mib,
        "observed_peak_total_used_mib": peak,
        "peak_pass": peak_pass,
        "required_throughput_improvement": required_improvement,
        "sequential_baseline_wall_seconds": sequential_seconds,
        "sequential_baseline_pairs_per_second": sequential_rate,
        "pair_wall_seconds": pair_wall_seconds,
        "pair_aggregate_pairs_per_second": pair_rate,
        "throughput_improvement_fraction": improvement,
        "throughput_pass": throughput_pass,
        "baseline_models": sorted(baseline_by_model),
        "reasons": reasons,
    }


def launch_model_plans(
    plans: Mapping[str, ModelPlan],
    *,
    suite_root: Path,
    device: str,
    cpu_threads: int,
) -> tuple[dict[str, dict[str, object]], float]:
    """Launch one plan or an explicit pair and wait for all owned trees."""
    children: dict[str, ManagedChild] = {}
    paths: dict[str, tuple[Path, Path]] = {}
    started: dict[str, float] = {}
    finished: dict[str, float] = {}
    pair_started = time.perf_counter()
    try:
        for model, plan in plans.items():
            stdout_path = plan.model_root / "probe.stdout.log"
            stderr_path = plan.model_root / "probe.stderr.log"
            env, _ = child_environment(device, cpu_threads)
            env["PYTHONUNBUFFERED"] = "1"
            started[model] = time.perf_counter()
            children[model] = ManagedChild.start(
                plan.command,
                cwd=suite_root,
                env=env,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            paths[model] = (stdout_path, stderr_path)

        while len(finished) < len(children):
            failed = False
            for model, child in children.items():
                if model in finished:
                    continue
                returncode = child.process.poll()
                if returncode is not None:
                    finished[model] = time.perf_counter()
                    failed = failed or returncode != 0
            if failed:
                for model, child in children.items():
                    if model not in finished:
                        child.terminate_tree()
                        finished[model] = time.perf_counter()
                break
            if len(finished) < len(children):
                time.sleep(0.1)
    except BaseException:
        for child in children.values():
            child.terminate_tree()
        raise
    finally:
        for model, child in children.items():
            if model not in finished:
                finished[model] = time.perf_counter()
            child.close()

    pair_wall = time.perf_counter() - pair_started
    records: dict[str, dict[str, object]] = {}
    for model, plan in plans.items():
        stdout_path, stderr_path = paths[model]
        records[model] = summarize_child(
            model=model,
            profile=plan.profile,
            batch_size=plan.batch_size,
            grad_accum_steps=plan.grad_accum_steps,
            returncode=int(children[model].process.returncode),
            wall_seconds=finished[model] - started[model],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            checkpoint=plan.model_root / "model.pth",
            work=plan.work,
        )
    return records, pair_wall


def _index_identity(result: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    dataset = result.get("dataset")
    models = result.get("models")
    if not isinstance(dataset, dict) or not isinstance(models, dict):
        raise ProbeError("Cannot index a result without dataset and model records")
    model_identity: list[dict[str, object]] = []
    for model in sorted(models):
        record = models[model]
        if not isinstance(record, dict):
            raise ProbeError("Cannot index a malformed model record")
        model_identity.append(
            {
                "model": model,
                "profile": record.get("profile"),
                "batch_size": record.get("batch_size"),
                "grad_accum_steps": record.get("grad_accum_steps"),
            }
        )
    models_key = ",".join(str(record["model"]) for record in model_identity)
    profile_key = ",".join(
        f'{record["model"]}={record["profile"]}' for record in model_identity
    )
    identity = {
        "mode": result.get("mode"),
        "device": result.get("device"),
        "models": model_identity,
        "models_key": models_key,
        "profile_key": profile_key,
        "source_sha256": dataset.get("source_sha256"),
        "selected_ids_sha256": dataset.get("selected_ids_sha256"),
        "cases": dataset.get("cases"),
        "seed": dataset.get("seed"),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical), identity


def update_probe_index(index_path: Path, result_path: Path, result: Mapping[str, object]) -> None:
    identity_key, identity = _index_identity(result)
    if result.get("identity") != identity or result.get("identity_key") != identity_key:
        raise ProbeError(
            "Completed resource-probe result does not embed its exact canonical identity"
        )
    if index_path.exists():
        index = _read_json_object(index_path)
        if index.get("schema_version") != "plasticity_resource_probe_index_v1":
            raise ProbeError(f"Resource-probe index has an unexpected schema: {index_path}")
    else:
        index = {
            "schema_version": "plasticity_resource_probe_index_v1",
            "latest_completed_single": {},
            "latest_cpu_eligible": {},
            "latest_certified_pair": {},
        }
    for key in ("latest_completed_single", "latest_cpu_eligible", "latest_certified_pair"):
        if not isinstance(index.get(key), dict):
            raise ProbeError(f"Resource-probe index field {key} is malformed")
    entry = {
        "identity_key": identity_key,
        "identity": identity,
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256_file(result_path),
        "completed_at": result.get("completed_at"),
        "run_id": result.get("run_id"),
    }
    mode = result.get("mode")
    if mode == "single" and result.get("complete") is True:
        index["latest_completed_single"][identity_key] = entry  # type: ignore[index]
        placement = result.get("placement")
        if isinstance(placement, dict) and placement.get("cpu_eligible") is True:
            index["latest_cpu_eligible"][identity_key] = entry  # type: ignore[index]
    elif mode == "pair":
        certification = result.get("certification")
        if isinstance(certification, dict) and certification.get("certified") is True:
            index["latest_certified_pair"][identity_key] = entry  # type: ignore[index]
    index["updated_at"] = _utc_now()
    _atomic_write_json(index_path, index)


def _new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"


def _validate_isolated_output_root(suite_root: Path, output_root: Path) -> None:
    """Keep probe runs out of every canonical config/data/model output tree."""
    campaign_output = (suite_root / "output" / "benchmarks" / "plasticity").resolve()
    default_probe_output = (campaign_output / "resource_probe").resolve()
    protected = (
        (suite_root / "configs" / "benchmarks" / "plasticity").resolve(),
        (suite_root / "dataset" / "benchmarks" / "plasticity").resolve(),
    )
    for path in protected:
        if output_root == path or output_root.is_relative_to(path) or path.is_relative_to(output_root):
            raise ProbeError(f"Probe output root overlaps a protected canonical tree: {path}")
    if output_root == campaign_output or (
        output_root.is_relative_to(campaign_output)
        and not output_root.is_relative_to(default_probe_output)
    ):
        raise ProbeError(
            "Probe output below the canonical Plasticity output tree must use its "
            f"dedicated resource_probe subtree: {default_probe_output}"
        )


def execute_probe(
    *,
    suite_root: Path,
    output_root: Path,
    run_id: str,
    models: Sequence[str],
    profiles: Mapping[str, str],
    device: str,
    gpu_id: int,
    cpu_threads: int,
    cases: int,
    seed: int,
    python_executable: str,
    gate_path: Path,
    baseline_paths: Sequence[Path],
    poll_seconds: float,
    peak_limit_mib: int,
    required_improvement: float,
    max_baseline_used_mib: int,
    max_baseline_utilization: int,
    allow_incomplete_paper_validation: bool = False,
) -> tuple[int, Path]:
    gate_record = completion_gate_or_user_waiver(
        gate_path, allow_incomplete=allow_incomplete_paper_validation
    )
    _validate_isolated_output_root(suite_root, output_root)
    source = suite_root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
    expected_source_hash = _provenance_source_hash(suite_root)
    run_root = (output_root / "runs" / run_id).resolve()
    result_path = run_root / "resource_probe_result.json"
    if run_root.exists():
        raise ProbeError(f"Refusing to reuse an existing probe run root: {run_root}")

    lock_path = suite_root / "output" / "benchmarks" / "plasticity" / ".campaign.lock"
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mode": "pair" if len(models) == 2 else "single",
        "device": device,
        "state": "initializing",
        "complete": False,
        "created_at": _utc_now(),
        "completed_at": None,
        "gate": gate_record,
        "models": {},
    }
    with ProbeLock(lock_path):
        run_root.mkdir(parents=True, exist_ok=False)
        _atomic_write_json(result_path, result)
        monitor: GpuMonitor | None = None
        gpu_record: dict[str, object] | None = None
        try:
            common_dataset = run_root / "datasets" / f"plasticity_probe_{cases}_seed{seed}.h5"
            dataset_record = prepare_probe_hdf5(
                source,
                common_dataset,
                cases=cases,
                seed=seed,
                expected_source_sha256=expected_source_hash,
            )
            plans, materialized = prepare_model_plans(
                suite_root=suite_root,
                run_root=run_root,
                models=models,
                profiles=profiles,
                device=device,
                gpu_id=gpu_id,
                cpu_threads=cpu_threads,
                python_executable=python_executable,
                common_dataset=common_dataset,
                dataset_record=dataset_record,
            )
            baselines: list[dict[str, object]] = []
            if len(models) == 2:
                baselines = load_pair_baselines(
                    baseline_paths,
                    models=models,
                    profiles=profiles,
                    dataset_record=dataset_record,
                )

            result.update(
                {
                    "state": "prepared",
                    "dataset": dataset_record,
                    "materialized": materialized,
                    "commands": {model: list(plan.command) for model, plan in plans.items()},
                    "environment_overrides": {
                        model: plan.environment_overrides for model, plan in plans.items()
                    },
                    "baselines": baselines,
                }
            )
            _atomic_write_json(result_path, result)

            if device == "gpu":
                baseline = query_nvidia_smi(gpu_id)
                if baseline.used_mib > max_baseline_used_mib:
                    raise ProbeError(
                        f"GPU {gpu_id} baseline usage {baseline.used_mib} MiB exceeds "
                        f"the safe launch ceiling {max_baseline_used_mib} MiB"
                    )
                if baseline.utilization_percent > max_baseline_utilization:
                    raise ProbeError(
                        f"GPU {gpu_id} baseline utilization {baseline.utilization_percent}% exceeds "
                        f"the safe launch ceiling {max_baseline_utilization}%"
                    )
                monitor = GpuMonitor(gpu_id, poll_seconds)
                monitor.start(baseline)

            result["state"] = "running"
            result["started_at"] = _utc_now()
            _atomic_write_json(result_path, result)
            model_records, pair_wall = launch_model_plans(
                plans,
                suite_root=suite_root,
                device=device,
                cpu_threads=cpu_threads,
            )
            if monitor is not None:
                gpu_record = monitor.stop()
                monitor = None
                result["gpu"] = gpu_record
            result["models"] = model_records
            result["wall_seconds"] = pair_wall
            children_complete = all(value.get("complete") is True for value in model_records.values())

            if len(models) == 2:
                assert gpu_record is not None
                certification = certify_pair(
                    model_records=model_records,
                    baselines=baselines,
                    pair_wall_seconds=pair_wall,
                    gpu_record=gpu_record,
                    peak_limit_mib=peak_limit_mib,
                    required_improvement=required_improvement,
                )
                result["certification"] = certification
                exit_code = 0 if certification["certified"] is True else 3
            else:
                model = models[0]
                model_record = model_records[model]
                projection = _single_projection(model_record)
                result["projection"] = projection
                projected_seconds = projection["projected_wall_seconds_500_epochs"]
                cpu_eligible = (
                    device == "cpu"
                    and model == "deeponet"
                    and children_complete
                    and isinstance(projected_seconds, (int, float))
                    and math.isfinite(projected_seconds)
                )
                result["placement"] = {
                    "cpu_eligible": cpu_eligible,
                    "model": model,
                    "device": device,
                    "profile": model_record["profile"],
                    "reason": (
                        "complete isolated DeepONet CPU probe with a finite conservative projection"
                        if cpu_eligible
                        else "only a complete matching DeepONet CPU record authorizes CPU placement"
                    ),
                }
                if gpu_record is not None:
                    result["resource_fit"] = {
                        "peak_limit_mib": peak_limit_mib,
                        "observed_peak_total_used_mib": gpu_record["peak_total_used_mib"],
                        "peak_pass": (
                            gpu_record.get("complete") is True
                            and int(gpu_record["peak_total_used_mib"]) <= peak_limit_mib
                        ),
                    }
                exit_code = 0 if children_complete else 2

            result["state"] = "complete" if children_complete else "failed"
            result["complete"] = children_complete
            result["completed_at"] = _utc_now()
            if children_complete:
                identity_key, identity = _index_identity(result)
                result["identity_key"] = identity_key
                result["identity"] = identity
            _atomic_write_json(result_path, result)
            if children_complete:
                update_probe_index(output_root / "index.json", result_path, result)
            return exit_code, result_path
        except BaseException as exc:
            if monitor is not None:
                try:
                    result["gpu"] = monitor.stop()
                except Exception as monitor_exc:
                    result["gpu_monitor_stop_error"] = (
                        f"{type(monitor_exc).__name__}: {monitor_exc}"
                    )
            result["state"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            result["complete"] = False
            result["completed_at"] = _utc_now()
            result["error"] = f"{type(exc).__name__}: {exc}"
            _atomic_write_json(result_path, result)
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    suite_root = here.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument("--dry-run", action="store_true", help="Print a read-only plan")
    execution.add_argument(
        "--execute",
        action="store_true",
        help="Run the real probe; requires a complete passing paper-validation gate",
    )
    parser.add_argument(
        "--allow-incomplete-paper-validation",
        action="store_true",
        help=(
            "Explicitly record a user-directed Plasticity priority waiver when the "
            "strict paper-validation gate is absent or incomplete"
        ),
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--model", choices=MODELS)
    selection.add_argument("--pair", nargs=2, metavar=("MODEL_A", "MODEL_B"))
    parser.add_argument(
        "--profile",
        action="append",
        help="Single profile name or pair assignment MODEL=PROFILE; repeat as needed",
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run-id")
    parser.add_argument("--suite-root", type=Path, default=suite_root)
    parser.add_argument("--source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--validation-gate", type=Path)
    parser.add_argument("--baseline-record", type=Path, action="append", default=[])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--gpu-peak-limit-mib", type=int, default=DEFAULT_GPU_PEAK_LIMIT_MIB)
    parser.add_argument(
        "--required-throughput-improvement",
        type=float,
        default=DEFAULT_PAIR_IMPROVEMENT,
    )
    parser.add_argument(
        "--max-baseline-used-mib", type=int, default=DEFAULT_MAX_BASELINE_USED_MIB
    )
    parser.add_argument("--max-baseline-utilization", type=int, default=10)
    return parser.parse_args(argv)


def _validated_cli(args: argparse.Namespace) -> tuple[Path, Path, Path, str, tuple[str, ...], dict[str, str]]:
    suite_root = args.suite_root.resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else suite_root / "output" / "benchmarks" / "plasticity" / "resource_probe"
    )
    _validate_isolated_output_root(suite_root, output_root)
    gate_path = (
        args.validation_gate.resolve()
        if args.validation_gate
        else suite_root / "output" / "benchmarks" / "paper_validation_completion_gate.json"
    )
    run_id = args.run_id or ("DRY_RUN" if args.dry_run else _new_run_id())
    if SAFE_RUN_ID_RE.fullmatch(run_id) is None:
        raise ProbeError("--run-id must contain only letters, digits, dot, underscore, and hyphen")
    models = (args.model,) if args.model else tuple(str(value).lower() for value in args.pair)
    if len(set(models)) != len(models):
        raise ProbeError("A pair probe requires two distinct models")
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ProbeError(f"Unknown pair model(s): {unknown}")
    if len(models) == 2 and args.device != "gpu":
        raise ProbeError("Explicit pair probes support GPU only")
    if args.execute and len(models) == 2 and len(args.baseline_record) != 2:
        raise ProbeError("A real pair probe requires two --baseline-record paths")
    if args.execute and len(models) == 1 and args.baseline_record:
        raise ProbeError("--baseline-record is valid only for pair probes")
    if not 8 <= args.cases <= 64:
        raise ProbeError("--cases must be between 8 and 64; the production default is 32")
    if args.gpu_id < 0:
        raise ProbeError("--gpu-id must be nonnegative")
    if args.cpu_threads <= 0:
        raise ProbeError("--cpu-threads must be positive")
    if args.poll_seconds < 0.1:
        raise ProbeError("--poll-seconds must be at least 0.1")
    if not 0.0 <= args.required_throughput_improvement <= 1.0:
        raise ProbeError("--required-throughput-improvement must be in [0, 1]")
    profiles = _parse_profile_arguments(models, args.profile)
    for model in models:
        resolve_profile(model, profiles[model])
    return suite_root, output_root, gate_path, run_id, models, profiles


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        suite_root, output_root, gate_path, run_id, models, profiles = _validated_cli(args)
        if args.source is not None:
            expected = suite_root / "dataset" / "benchmarks" / "plasticity" / "plasticity.h5"
            if args.source.resolve() != expected.resolve():
                raise ProbeError("The production resource probe accepts only the authoritative source HDF5")
        if args.dry_run:
            plan = build_dry_run_plan(
                suite_root=suite_root,
                output_root=output_root,
                run_id=run_id,
                models=models,
                profiles=profiles,
                device=args.device,
                gpu_id=args.gpu_id,
                cpu_threads=args.cpu_threads,
                cases=args.cases,
                seed=args.seed,
                python_executable=args.python,
                gate_path=gate_path,
            )
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0

        exit_code, result_path = execute_probe(
            suite_root=suite_root,
            output_root=output_root,
            run_id=run_id,
            models=models,
            profiles=profiles,
            device=args.device,
            gpu_id=args.gpu_id,
            cpu_threads=args.cpu_threads,
            cases=args.cases,
            seed=args.seed,
            python_executable=args.python,
            gate_path=gate_path,
            baseline_paths=args.baseline_record,
            poll_seconds=args.poll_seconds,
            peak_limit_mib=args.gpu_peak_limit_mib,
            required_improvement=args.required_throughput_improvement,
            max_baseline_used_mib=args.max_baseline_used_mib,
            max_baseline_utilization=args.max_baseline_utilization,
            allow_incomplete_paper_validation=args.allow_incomplete_paper_validation,
        )
        print(f"Resource-probe record: {result_path}")
        return exit_code
    except KeyboardInterrupt:
        print("Plasticity resource probe interrupted", file=sys.stderr)
        return 130
    except ProbeError as exc:
        print(f"Plasticity resource probe refused/failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
