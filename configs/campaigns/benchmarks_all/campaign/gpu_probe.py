"""Live GPU state via `nvidia-smi` subprocess calls.

Mirrors the existing subprocess.run(..., timeout=..., check=False) + graceful
empty-result-on-failure pattern already used in
studio/studio_backend/system_info.py::gpu_inventory -- no pynvml/GPUtil
dependency added, consistent with this repo's established idiom for
GPU-inventory probing (and with the root launcher's near-zero dependency
policy in pyproject.toml).
"""

from __future__ import annotations

import subprocess
from typing import Protocol


class GpuProbe(Protocol):
    def vram_mb(self, gpu_ids: list[str]) -> dict[str, tuple[int, int] | None]:
        """(free_mb, total_mb) for each requested GPU id, or None if unknown
        (nvidia-smi missing/erroring, or that index not reported)."""
        ...

    def compute_app_pids(self, gpu_id: str) -> list[tuple[int, int | None]]:
        """(pid, used_memory_mb) for every process nvidia-smi currently
        attributes to this GPU, regardless of which job (if any) owns it --
        ownership is resolved by the caller via SubprocessLauncher.pids_owned_by.
        used_memory_mb is None when nvidia-smi can't report it (observed as
        literal "[N/A]" on some consumer/WDDM driver configs, e.g. GeForce
        cards) -- the pid is still returned, since pid is what ownership
        resolution actually needs."""
        ...


class NvidiaSmiProbe:
    def __init__(self, timeout: float = 8.0) -> None:
        self._timeout = timeout
        self._uuid_by_index: dict[str, str] | None = None

    def vram_mb(self, gpu_ids: list[str]) -> dict[str, tuple[int, int] | None]:
        result: dict[str, tuple[int, int] | None] = {g: None for g in gpu_ids}
        for row in self._run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
                               "--format=csv,noheader,nounits"]):
            parts = [p.strip() for p in row.split(",")]
            if len(parts) != 3 or parts[0] not in result:
                continue
            try:
                result[parts[0]] = (int(float(parts[1])), int(float(parts[2])))
            except ValueError:
                continue
        return result

    def compute_app_pids(self, gpu_id: str) -> list[tuple[int, int | None]]:
        target_uuid = self._index_to_uuid().get(gpu_id)
        if target_uuid is None:
            return []
        owned: list[tuple[int, int | None]] = []
        for row in self._run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                               "--format=csv,noheader,nounits"]):
            parts = [p.strip() for p in row.split(",")]
            if len(parts) != 3 or parts[0] != target_uuid:
                continue
            try:
                pid = int(parts[1])
            except ValueError:
                continue
            try:
                used_mb: int | None = int(float(parts[2]))
            except ValueError:
                used_mb = None  # e.g. "[N/A]" -- keep the pid, drop only the memory figure
            owned.append((pid, used_mb))
        return owned

    def _index_to_uuid(self) -> dict[str, str]:
        if self._uuid_by_index is None:
            self._uuid_by_index = {}
            for row in self._run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]):
                parts = [p.strip() for p in row.split(",")]
                if len(parts) == 2:
                    self._uuid_by_index[parts[0]] = parts[1]
        return self._uuid_by_index

    def _run(self, command: list[str]) -> list[str]:
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                        timeout=self._timeout, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return []
        if completed.returncode != 0:
            return []
        return [line for line in completed.stdout.splitlines() if line.strip()]


class FakeGpuProbe:
    """Test double: scripted responses, no subprocess/hardware dependency."""

    def __init__(self) -> None:
        self.vram_responses: list[dict[str, tuple[int, int] | None]] = []
        self.compute_app_responses: dict[str, list[tuple[int, int | None]]] = {}

    def vram_mb(self, gpu_ids: list[str]) -> dict[str, tuple[int, int] | None]:
        if self.vram_responses:
            return self.vram_responses.pop(0)
        return {g: None for g in gpu_ids}

    def compute_app_pids(self, gpu_id: str) -> list[tuple[int, int | None]]:
        return self.compute_app_responses.get(gpu_id, [])
