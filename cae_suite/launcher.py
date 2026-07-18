from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


def launch_and_wait(command: list[str], *, cwd: Path) -> int:
    kwargs: dict[str, object] = {"cwd": cwd, "env": os.environ.copy()}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **kwargs)
    try:
        return process.wait()
    except KeyboardInterrupt:
        _interrupt_process_group(process)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return 130


def _interrupt_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(process.pid, signal.SIGINT)
    except (OSError, ValueError):
        process.terminate()
    time.sleep(0.05)
