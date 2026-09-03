"""Non-blocking subprocess lifecycle for campaign jobs: start, poll, and a
forceful whole-process-tree kill for stall recovery.

Deliberately does not reuse cae_suite.launcher.launch_and_wait: that helper
blocks on Popen.wait() and only sends a *soft* SIGINT/CTRL_BREAK_EVENT on
KeyboardInterrupt. The scheduler needs the opposite shape -- never block, and
kill *hard* when a job stalls -- so this module owns its own primitives,
reusing only the same session/process-group Popen kwargs shape.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Literal, Optional

Outcome = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED", "STALL", "INTERRUPTED"]


def new_session_kwargs() -> dict:
    """Popen kwargs that put a child in its own session/process group, so a
    POSIX killpg(pid, ...) by the child's own pid reaches the whole tree it
    spawns (DataLoader workers, mp.spawn ranks) -- same shape as
    cae_suite/launcher.py::launch_and_wait."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


@dataclass(eq=False)
class Job:
    label: str
    ex_slot: str
    light: bool
    mode: str
    canonical_config: Path
    log_path: Path
    preflight_runtime_config: Optional[Path] = None
    admission_runtime_config: Optional[Path] = None
    gpu: Optional[str] = None
    popen: Optional[subprocess.Popen] = None
    pgid: Optional[int] = None
    admitted_at: Optional[float] = None
    finished_at: Optional[float] = None
    outcome: Outcome = "PENDING"
    exit_code: Optional[int] = None
    orphan_check: Optional[str] = None
    log_fh: Optional[IO[bytes]] = field(default=None, repr=False)


class SubprocessLauncher:
    """Real launcher: actual subprocesses, actual nvidia-smi/PowerShell calls."""

    def start(self, job: Job, python_bin: str, cwd: Path) -> None:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(job.log_path, "wb")
        env = os.environ.copy()
        # The stall detector's only signal is log-file mtime -- buffered stdout
        # previously looked identical to a genuine hang (see prior incident
        # notes), so every launched job must flush eagerly.
        env["PYTHONUNBUFFERED"] = "1"
        popen = subprocess.Popen(
            [python_bin, "AI_CAE4ALL_main.py", "--config", str(job.admission_runtime_config)],
            cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT, env=env,
            **new_session_kwargs(),
        )
        job.popen = popen
        job.log_fh = log_fh
        job.pgid = popen.pid if os.name != "nt" else None
        job.outcome = "RUNNING"

    def poll(self, job: Job) -> Optional[int]:
        if job.popen is None:
            return None
        rc = job.popen.poll()
        if rc is not None:
            self._close_log(job)
        return rc

    def kill_process_tree(self, job: Job) -> None:
        if job.popen is None:
            return
        if os.name == "nt":
            self._kill_tree_windows(job.popen.pid)
        else:
            try:
                os.killpg(os.getpgid(job.popen.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            job.popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._close_log(job)

    def pids_owned_by(self, job: Job, candidate_pids: list[int]) -> set[int]:
        """Which of these nvidia-smi-reported PIDs belong to this job's tree.
        Needed so the post-exit orphan sweep on a GPU shared by several
        concurrent jobs doesn't kill a sibling job that's still legitimately
        running there."""
        if job.popen is None or not candidate_pids:
            return set()
        if os.name == "nt":
            tree = self._tree_pids_windows(job.popen.pid)
            return set(candidate_pids) & tree
        owned: set[int] = set()
        for pid in candidate_pids:
            try:
                if os.getpgid(pid) == job.pgid:
                    owned.add(pid)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        return owned

    def _close_log(self, job: Job) -> None:
        if job.log_fh is not None:
            try:
                job.log_fh.close()
            except OSError:
                pass
            job.log_fh = None

    # -- Windows: no killpg/process-group descendant tracking exists, so a
    # point-in-time Win32_Process parent-id walk is the only way to catch
    # orphaned `multiprocessing.spawn` workers. Best-effort: a grandchild
    # spawned in the gap between the walk and the kill can survive. Real
    # campaigns run on the Linux 8-GPU box (see repo notes), so POSIX
    # correctness is the priority here; this is a documented platform gap,
    # not assumed parity. --
    @staticmethod
    def _tree_pids_windows(root_pid: int) -> set[int]:
        script = (
            "$ErrorActionPreference='SilentlyContinue';"
            f"$ids=[System.Collections.Generic.List[int]]::new(); $ids.Add({root_pid});"
            "$i=0;"
            "while($i -lt $ids.Count){"
            # $pid is PowerShell's read-only automatic variable for the CURRENT
            # process -- assigning to it throws "Cannot overwrite variable PID"
            # and silently truncates the walk to just the root. Use $procId.
            "  $procId=$ids[$i]; $i++;"
            "  Get-CimInstance Win32_Process -Filter \"ParentProcessId=$procId\" |"
            "    ForEach-Object { if(-not $ids.Contains([int]$_.ProcessId)){ $ids.Add([int]$_.ProcessId) } }"
            "}"
            "$ids -join ','"
        )
        try:
            completed = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {root_pid}
        if completed.returncode != 0 or not completed.stdout.strip():
            return {root_pid}
        ids = {int(tok) for tok in completed.stdout.strip().split(",") if tok.strip().isdigit()}
        return ids or {root_pid}

    @classmethod
    def _kill_tree_windows(cls, root_pid: int) -> None:
        ids = cls._tree_pids_windows(root_pid)
        id_list = ",".join(str(i) for i in ids)
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                 f"Stop-Process -Id {id_list} -Force -ErrorAction SilentlyContinue"],
                timeout=15, check=False, capture_output=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
