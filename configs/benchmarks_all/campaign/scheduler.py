"""Dynamic shared-queue scheduler: a single pending queue serving every
selected GPU, live-VRAM-aware admission with a per-GPU warm-up cooldown, and
log-mtime-staleness stall detection with a forceful process-tree kill.

Deliberately polling, never blocking: no code path here calls Popen.wait().
A hung job (e.g. one rank of a distributed job deadlocked in an NCCL/gloo
collective after a sibling rank died) therefore cannot block admission of
other jobs onto the same GPU -- the old bash `lane_worker` blocked on exactly
this, since it awaited each job's process synchronously with no timeout, so
every later job queued on that same lane never even started.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .gpu_probe import GpuProbe
from .manifest import ManifestWriter
from .process import Job, SubprocessLauncher
from .runtime_config import RuntimeConfigError, write_runtime_config


@dataclass
class GpuState:
    id: str
    running: list[Job] = field(default_factory=list)
    last_admission_time: float = float("-inf")


class Scheduler:
    def __init__(
        self,
        jobs: list[Job],
        gpu_list: list[str],
        python_bin: str,
        cwd: Path,
        runtime_config_root: Path,
        probe: GpuProbe,
        launcher: SubprocessLauncher,
        manifest: ManifestWriter,
        *,
        vram_target_util_percent: float,
        max_concurrency_per_gpu: int,
        admit_warmup_sec: float,
        stall_timeout_sec: float,
        poll_interval_sec: float,
        blocked_abort_sec: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        log_mtime_fn: Callable[[Path], float] | None = None,
        on_event: Callable[[str], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.python_bin = python_bin
        self.cwd = cwd
        self.runtime_config_root = runtime_config_root
        self.probe = probe
        self.launcher = launcher
        self.manifest = manifest
        self.vram_target_util_percent = vram_target_util_percent
        self.max_concurrency_per_gpu = max_concurrency_per_gpu
        self.admit_warmup_sec = admit_warmup_sec
        self.stall_timeout_sec = stall_timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.blocked_abort_sec = blocked_abort_sec
        self.clock = clock
        self.wall_clock = wall_clock
        self.log_mtime_fn = log_mtime_fn or self._default_log_mtime
        self.on_event = on_event or print
        self.sleep = sleep
        self._blocked_since: float | None = None
        self._last_blocked_warning: float | None = None
        self._last_vram: dict[str, tuple[int, int] | None] = {}
        self.aborted_reason: str | None = None

        self.gpus = {g: GpuState(id=g) for g in gpu_list}
        self.all_jobs = list(jobs)
        # Heavy (light=False) jobs first, stable sort: they claim a GPU's
        # concurrency slot before light jobs get a chance to fill it, so a
        # later heavy job doesn't get starved of a slot on a GPU that's
        # actually got plenty of free VRAM -- see campaign_runner.py's module
        # docstring / the design write-up for the fragmentation scenario this
        # avoids (concurrency-slot exhaustion, not VRAM exhaustion, since
        # light jobs use ~0.1-0.2GB each).
        self.pending: deque[Job] = deque(sorted(self.all_jobs, key=lambda j: j.light))
        self.manifest.register(self.all_jobs)

    def _default_log_mtime(self, path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return self.wall_clock()

    # -- lifecycle --

    def run(self) -> int:
        try:
            while self.pending or any(g.running for g in self.gpus.values()):
                self.tick()
                if self.aborted_reason:
                    break
        except KeyboardInterrupt:
            # Jobs are launched into their own session/process group precisely
            # so a stall-kill can take out their whole tree -- the flip side is
            # that Ctrl-C here does NOT reach them, so without this they'd
            # survive as orphans still holding VRAM.
            self.on_event("\nInterrupted -- killing all running jobs...")
            self._kill_all_running("INTERRUPTED")
            return 130
        if self.aborted_reason:
            return 1
        return 0 if all(j.outcome == "SUCCESS" for j in self.all_jobs) else 1

    def tick(self) -> None:
        self.reap_tick()
        self.stall_tick()
        self.admit_tick()
        self.blocked_tick()
        if not self.aborted_reason and (self.pending or any(g.running for g in self.gpus.values())):
            self.sleep(self.poll_interval_sec)

    def blocked_tick(self) -> None:
        """Guard against the campaign silently doing nothing forever.

        If work is queued but nothing is running and nothing is admissible --
        e.g. every GPU sits above VRAM_TARGET_UTIL_PERCENT because of memory
        the campaign doesn't own (another tenant, a leaked process, or simply
        a baseline desktop load above the target) -- the admission loop would
        otherwise spin indefinitely, printing nothing. That is the same
        "campaign appears to be running but no job is" failure this scheduler
        exists to eliminate, so surface it loudly and, if it persists with
        zero jobs running (nothing in flight to lose), stop with a real
        diagnostic instead of hanging.
        """
        idle = bool(self.pending) and not any(g.running for g in self.gpus.values())
        if not idle:
            self._blocked_since = None
            self._last_blocked_warning = None
            return

        now = self.clock()
        if self._blocked_since is None:
            self._blocked_since = now
            self._last_blocked_warning = None
        blocked_for = now - self._blocked_since

        if self._last_blocked_warning is None or now - self._last_blocked_warning >= 300:
            self._last_blocked_warning = now
            self.on_event(
                f"WARNING: {len(self.pending)} job(s) queued but none admissible "
                f"for {blocked_for / 60:.1f}min -- {self._gpu_util_summary()} "
                f"(target < {self.vram_target_util_percent}% used)"
            )

        if self.blocked_abort_sec > 0 and blocked_for >= self.blocked_abort_sec:
            self.aborted_reason = (
                f"no job could be admitted for {blocked_for / 60:.1f}min with "
                f"{len(self.pending)} still queued and nothing running; "
                f"{self._gpu_util_summary()}. Raise VRAM_TARGET_UTIL_PERCENT, free the "
                f"GPUs, or set BLOCKED_ABORT_MIN=0 to wait indefinitely."
            )
            self.on_event(f"ABORT: {self.aborted_reason}")

    def _gpu_util_summary(self) -> str:
        """Reports the reading admit_tick actually decided on this tick, rather
        than issuing a second nvidia-smi call -- otherwise the diagnostic could
        quote numbers from a different instant than the decision it explains."""
        parts = []
        for gpu_id, info in self._last_vram.items():
            if info is None:
                parts.append(f"gpu{gpu_id}=unknown")
            else:
                free_mb, total_mb = info
                pct = 100.0 * (total_mb - free_mb) / total_mb if total_mb > 0 else 100.0
                parts.append(f"gpu{gpu_id}={pct:.0f}% used")
        return "measured: " + ", ".join(parts)

    def _kill_all_running(self, outcome: str) -> None:
        for gpu in self.gpus.values():
            for job in list(gpu.running):
                self.launcher.kill_process_tree(job)
                self._finalize(job, gpu, exit_code=None, outcome=outcome)

    # -- individual ticks (exposed separately so tests can drive them without
    # real sleeping or hardware) --

    def reap_tick(self) -> None:
        for gpu in self.gpus.values():
            for job in list(gpu.running):
                rc = self.launcher.poll(job)
                if rc is not None:
                    self._finalize(job, gpu, exit_code=rc)

    def stall_tick(self) -> None:
        now = self.wall_clock()
        for gpu in self.gpus.values():
            for job in list(gpu.running):
                idle = now - self.log_mtime_fn(job.log_path)
                if idle >= self.stall_timeout_sec:
                    self.manifest.event("STALL_DETECTED", job, idle_s=round(idle, 1))
                    self.on_event(
                        f"[{job.label}] STALL DETECTED (no log activity for "
                        f"{idle / 60:.1f}min) -- killing gpu={job.gpu}"
                    )
                    self.launcher.kill_process_tree(job)
                    self._finalize(job, gpu, exit_code=None, outcome="STALL")

    def admit_tick(self) -> None:
        if not self.pending:
            return
        vram = self.probe.vram_mb(list(self.gpus.keys()))
        self._last_vram = vram
        now = self.clock()
        for gpu_id, gpu in self.gpus.items():
            if not self.pending:
                return
            if now - gpu.last_admission_time < self.admit_warmup_sec:
                continue
            if len(gpu.running) >= self.max_concurrency_per_gpu:
                continue
            info = vram.get(gpu_id)
            if info is None:
                if gpu.running:
                    continue  # probe unavailable and a job's already there: don't stack blindly
            else:
                free_mb, total_mb = info
                used_pct = 100.0 * (total_mb - free_mb) / total_mb if total_mb > 0 else 100.0
                if used_pct >= self.vram_target_util_percent:
                    continue
            job = self.pending.popleft()
            self._admit(job, gpu)

    # -- internals --

    def _admit(self, job: Job, gpu: GpuState) -> None:
        job.gpu = gpu.id
        job.admission_runtime_config = self.runtime_config_root / f"{job.mode}_{job.label}.txt"
        try:
            write_runtime_config(job.mode, job.canonical_config, gpu.id, job.admission_runtime_config)
        except RuntimeConfigError as exc:
            job.finished_at = self.wall_clock()
            job.outcome = "FAILED"
            self.on_event(f"[{job.label}] FAILED to write runtime config: {exc}")
            self.manifest.event("FAILED", job, error=str(exc))
            self.manifest.job_finalized(job)
            return
        self.launcher.start(job, self.python_bin, self.cwd)
        job.admitted_at = self.wall_clock()
        gpu.running.append(job)
        gpu.last_admission_time = self.clock()
        self.manifest.event("ADMITTED", job)
        self.on_event(f"[{job.label}] {job.mode.upper()} START gpu={gpu.id} -> {job.log_path}")

    def _finalize(self, job: Job, gpu: GpuState, *, exit_code: int | None, outcome: str | None = None) -> None:
        job.finished_at = self.wall_clock()
        job.exit_code = exit_code
        job.outcome = outcome or ("SUCCESS" if exit_code == 0 else "FAILED")
        gpu.running.remove(job)
        self._orphan_sweep(job)
        self.manifest.event(job.outcome, job, exit_code=exit_code)
        self.manifest.job_finalized(job)
        if job.outcome == "SUCCESS":
            self.on_event(f"[{job.label}] {job.mode.upper()} DONE")
        else:
            self.on_event(f"[{job.label}] {job.mode.upper()} {job.outcome} -- see {job.log_path}")

    def _orphan_sweep(self, job: Job) -> None:
        """Belt-and-suspenders: after a job's own process tree is gone (either
        it exited cleanly or we just killed it), verify nvidia-smi agrees
        nothing owned by this job is still running on its GPU. Needed because
        several jobs can now share one GPU -- a plain "any compute PID on
        this GPU" check would false-positive against a sibling job still
        legitimately running there, so ownership is resolved via
        SubprocessLauncher.pids_owned_by (pgid on POSIX, a process-tree walk
        on Windows)."""
        if job.gpu is None:
            job.orphan_check = "SKIPPED"
            return
        leftover = self.probe.compute_app_pids(job.gpu)
        owned = self.launcher.pids_owned_by(job, [pid for pid, _ in leftover])
        if not owned:
            job.orphan_check = "CLEAN"
            return
        self.launcher.kill_process_tree(job)
        self.sleep(2)  # short, bounded grace -- not indefinite
        leftover2 = self.probe.compute_app_pids(job.gpu)
        still = self.launcher.pids_owned_by(job, [pid for pid, _ in leftover2])
        if still:
            job.orphan_check = "WARN"
            self.manifest.event("ORPHAN_WARNING", job, pids=sorted(still))
            self.on_event(f"[{job.label}] WARNING: {len(still)} process(es) survived teardown on gpu={job.gpu}")
        else:
            job.orphan_check = "CLEAN"
