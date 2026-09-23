#!/usr/bin/env python3
"""Eight-lane campaign runner for the dataset/method baseline matrix.

    bash configs/run_all_135.sh        machine 135
    bash configs/run_all_136.sh        machine 136

Both wrappers exec this file with --machine. Each machine owns a disjoint half
of the examples in manifest.json (MACHINES below) and runs every arm of that
half -- config_train, then config_infer -- through AI_CAE4ALL_main.py, exactly
as a hand launch would.

Lanes. The `gpu_ids` digit in each config is its home lane (LANE_GPU in
generate.py):

    lane 0  deeponet + fno        lane 4  meshgraphnets
    lane 1  point_deeponet        lane 5  himgn
    lane 2  (no home arms)        lane 6  himgn_v + lsh_vae
    lane 3  transolver3           lane 7  chi_mgnflow + sdfflow

Behaviour
  - Start gate: checks at once, then every GATE_INTERVAL seconds (3600), and
    starts once EXPECTED_GPUS (8) GPUs are visible, all at 0% utilisation with
    no compute process on them, so whatever already runs finishes first. One
    check is GATE_SAMPLES readings GATE_SAMPLE_GAP seconds apart;
    GATE_ALLOW_APPS=1 drops the compute-process condition.
  - One worker per GPU. Worker i runs lane i in manifest (example) order, then
    takes the next arm of whichever lane is furthest behind. A worker with
    nothing left to take stays until every running arm has finished, because
    an arm whose card is lost comes back to the queue.
  - Every stage runs on exactly one card. The worker writes a launch copy of
    the config -- its bytes unchanged except gpu_ids set to 0, under a header
    naming the source and its sha256 -- and starts the launcher on that copy
    with CUDA_VISIBLE_DEVICES=<the card's UUID>. The checked-in configs are
    never modified, and a taken-over arm runs unchanged on any card.
  - A failed training skips that arm's inference. A failed stage is that
    arm's own problem: it is reported and the worker moves on to the next arm.
  - A lost GPU stops only its own worker. nvidia-smi is polled every
    WATCH_INTERVAL seconds (300) while a stage runs, and after any stage that
    does not succeed the card is also opened through the CUDA driver. A card
    confirmed gone has its stage killed and the arm goes back to the front of
    its lane for another card (once per arm). The other workers carry on and
    take over the dead card's lane.
  - Resume: a finished stage leaves a marker under
    output/dataset_matrix/_campaign/<machine>/done/ and is skipped next time.
    Retraining an arm always re-runs its inference.
  - Stop: Ctrl-C or SIGTERM (and SIGHUP, unless started under nohup) sends
    SIGTERM to every running stage and SIGKILL after KILL_GRACE seconds (at
    once on a second signal); nothing is recorded for those stages. SIGINT is
    never forwarded: the trainers turn it into a normal-looking exit.

Modes (environment variable or flag):
  DRY_RUN=1        --dry-run        print the plan; writes nothing
  CHECK=1          --check          launcher --check on every stage still to run
  SKIP_GPU_GATE=1  --skip-gpu-gate  start without waiting for an idle box
Tuning: GATE_INTERVAL GATE_SAMPLES GATE_SAMPLE_GAP GATE_ALLOW_APPS
        EXPECTED_GPUS WATCH_INTERVAL KILL_GRACE DEAD_CONFIRM_GAP
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MANIFEST = HERE / "manifest.json"
LAUNCHER = ROOT / "AI_CAE4ALL_main.py"
SCORE_SPREAD = HERE / "score_spread.py"
LOCAL_TOMLS = (ROOT / "ai_cae4all.local.toml", ROOT / "cae_suite.local.toml")
LAUNCHER_PYTHON = (3, 10)  # pyproject.toml: requires-python >= 3.10

# Each machine owns a disjoint set of examples. The heavy slots (ex2 at 200k
# nodes x 50 steps, ex3_full, ex4 at 599 steps, ex6 at 400, ex7 at 1000
# samples) are split across the two boxes rather than stacked on one.
MACHINES = {
    "135": (
        "deterministic/ex1", "deterministic/ex2", "deterministic/ex3_full",
        "deterministic/ex5", "deterministic/ex8", "deterministic/ex10",
        "probabilistic/ex1",
    ),
    "136": (
        "deterministic/ex3_mid", "deterministic/ex4", "deterministic/ex6",
        "deterministic/ex7", "deterministic/ex9", "probabilistic/ex2",
        "geometry_generation/ex1",
    ),
}

STAGES = ("train", "infer")
NUM_LANES = 8
MAX_DEAD_REQUEUES = 1    # an arm that has lost two cards is not handed to a third
NVSMI_TIMEOUT = 60
CUDA_PROBE_TIMEOUT = 120
IDLE_POLL = 10           # a worker with nothing to take looks again this often
CHECK_JOBS = 4
CHECK_TIMEOUT = 3600
PRELUDE_BYTES = 4 << 20  # the launcher's own lines always come before the model's

# Stage outcomes, as the report prints them.
OK = "ok"
FAILED = "failed"
NOT_LAUNCHED = "not launched"
INTERRUPTED = "interrupted"
GPU_LOST = "gpu lost"
STOPPED = "stopped"
SKIPPED = "skipped"

GPU_IDS_LINE = re.compile(rb"gpu_ids([ \t]+)([0-7])([ \t]*)")
# cae_suite/cli.py prints "Starting <display name>..." immediately before it
# spawns the model process, and "No model process was started" whenever
# preflight or routing stopped it.
STARTED_LINE = re.compile(rb"(?m)^Starting .+\.\.\.\r?$")
NOT_STARTED = b"No model process was started"
# The mesh-family trainers catch KeyboardInterrupt, print a line containing
# this and return normally, so an interrupted run can exit 0. It is never a
# finished one.
INTERRUPT_MARK = b"interrupted by user"
TRUTHY = {"1", "true", "yes", "on"}
# NVML states nvidia-smi prints in place of a value for a card that is still
# enumerated but dead.
FAULT_MARKS = ("gpu is lost", "requires reset")

# Run by the runner's own interpreter with CUDA_VISIBLE_DEVICES set to one
# card: opens a context and allocates 1 MiB through the driver API, so the
# answer is about the card itself, independent of any method venv. Out of
# memory (2) and device unavailable (46) mean someone else holds the card,
# not that it is broken.
CUDA_PROBE = r"""
import ctypes, sys
try:
    cuda = ctypes.CDLL("libcuda.so.1")
except OSError as exc:
    print("unknown cannot load libcuda.so.1: %s" % exc)
    sys.exit(0)
def call(name, *args):
    fn = getattr(cuda, name, None)
    if fn is None:
        print("unknown libcuda.so.1 has no %s" % name)
        sys.exit(0)
    rc = fn(*args)
    if rc:
        print("%s %s returned CUDA error %d" % ("unknown" if rc in (2, 46) else "unusable", name, rc))
        sys.exit(0)
call("cuInit", 0)
count = ctypes.c_int(0)
call("cuDeviceGetCount", ctypes.byref(count))
if count.value < 1:
    print("unusable the CUDA driver sees no device")
    sys.exit(0)
dev = ctypes.c_int(0)
call("cuDeviceGet", ctypes.byref(dev), 0)
ctx = ctypes.c_void_p()
call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), dev)
call("cuCtxSetCurrent", ctx)
ptr = ctypes.c_uint64(0)
call("cuMemAlloc_v2", ctypes.byref(ptr), ctypes.c_size_t(1 << 20))
call("cuMemFree_v2", ptr)
print("usable")
"""

# --------------------------------------------------------------------- helpers

def rel(path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def fmt_secs(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")

class UsageError(Exception):
    """A bad mode or tuning value: main() prints it and starts nothing (exit 2)."""

def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY

def env_number(name: str, default, cast=float):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        raise UsageError(f"{name}={raw!r} is not {'a whole number' if cast is int else 'a number'}")
    if not math.isfinite(value):  # nan would make every wait on it endless
        raise UsageError(f"{name}={raw!r} is not a finite number")
    if value < 0:
        raise UsageError(f"{name}={raw!r} must not be negative")
    return value

def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)

def config_key(line: bytes):
    """The key the native parsers see on this line, or None for a comment."""
    text = line.decode("utf-8", "replace").strip()
    if not text or text.startswith("%"):
        return None
    text = text.split("#")[0].strip()
    return text.split()[0].lower() if text else None

def parse_lane(data: bytes):
    """(lane, None) from the single `gpu_ids <digit>` line, or (None, problem)."""
    if data.startswith(b"\xef\xbb\xbf"):
        return None, "starts with a UTF-8 BOM"
    hits = [ln for ln in data.split(b"\n") if config_key(ln) == "gpu_ids"]
    if len(hits) != 1:
        return None, f"has {len(hits)} gpu_ids lines; the runner needs exactly one"
    match = GPU_IDS_LINE.fullmatch(hits[0].rstrip(b"\r"))
    if not match:
        return None, (f"gpu_ids line {hits[0].decode('utf-8', 'replace').strip()!r} "
                      "is not a single GPU index 0-7")
    return int(match.group(2)), None

def launch_bytes(source_rel: str, data: bytes, lane: int, gpu) -> bytes:
    """The config's own bytes with gpu_ids set to 0, under a provenance header."""
    lines, rewritten = data.split(b"\n"), 0
    for i, line in enumerate(lines):
        if config_key(line) != "gpu_ids":
            continue
        body = line.rstrip(b"\r")
        match = GPU_IDS_LINE.fullmatch(body)
        if not match:
            raise ValueError(f"unexpected gpu_ids line in {source_rel}")
        lines[i] = b"gpu_ids" + match.group(1) + b"0" + match.group(3) + line[len(body):]
        rewritten += 1
    if rewritten != 1:
        raise ValueError(f"{source_rel} has {rewritten} gpu_ids lines")
    header = (
        "% dataset_matrix launch copy written by run_matrix.py -- do not edit.\n"
        f"% source: {source_rel}\n"
        f"% source_sha256: {sha256(data)}\n"
        f"% binding: gpu_ids {lane} -> 0, run on GPU {gpu.index} ({gpu.uuid}) "
        "through CUDA_VISIBLE_DEVICES\n"
    )
    return header.encode("utf-8") + b"\n".join(lines)

# -------------------------------------------------------------------- the plan

class Arm:
    """One (example, method) pair: a train stage, then an infer stage."""

    def __init__(self, category, example, method, lane, rank, rels):
        self.category, self.example, self.method = category, example, method
        self.lane, self.rank, self.rels = lane, rank, rels
        self.key = f"{category}__{example}__{method}"
        self.label = f"{category}/{example}/{method}"
        self.dead_requeues = 0

    def path(self, stage: str) -> Path:
        return ROOT / self.rels[stage]

def toml_problem():
    """The launcher reads ai_cae4all.local.toml with tomllib, or tomli on 3.10."""
    present = [p for p in LOCAL_TOMLS if p.exists()]
    if not present:
        return None
    for module in ("tomllib", "tomli"):
        try:
            __import__(module)
            return None
        except ImportError:
            pass
    return (f"{present[0].name} exists but {sys.executable} (Python "
            f"{platform.python_version()}) has neither tomllib nor tomli, so the "
            "launcher cannot read it: use Python 3.11+ (PYTHON=...) or pip install tomli")

def launcher_problem():
    """None if the launcher imports under this interpreter, else why not."""
    if sys.version_info < LAUNCHER_PYTHON:
        return (f"the launcher needs Python >= 3.10 (pyproject.toml) but this is "
                f"{sys.executable} (Python {platform.python_version()}); set PYTHON=...")
    try:
        cp = subprocess.run([sys.executable, "-B", "-c", "import cae_suite.cli"],
                            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"could not test-import the launcher: {exc}"
    if cp.returncode:
        tail = cp.stdout.decode("utf-8", "replace").strip().splitlines()[-3:]
        return f"the launcher does not import under {sys.executable}: " + " | ".join(tail)
    return None

def load_plan(machine: str):
    """(arms, problems, warnings) for one machine; any problem means do not start."""
    problems, warnings = [], []
    try:
        pairs = json.loads(MANIFEST.read_text(encoding="utf-8"))["pairs"]
        entries = [(p["category"], p["example"], p["method"], {s: p[s] for s in STAGES})
                   for p in pairs]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [], [f"cannot read the pairs in {rel(MANIFEST)}: {exc!r}"], []

    ranks = {}
    for category, example, _, _ in entries:
        ranks.setdefault(f"{category}/{example}", len(ranks))
    owners = {}
    for name, examples in MACHINES.items():
        for example in examples:
            owners.setdefault(example, []).append(name)
    for example, names in sorted(owners.items()):
        if len(names) > 1:
            problems.append(f"{example} is listed {len(names)} times in MACHINES "
                            f"(machines {', '.join(names)})")
        if example not in ranks:
            problems.append(f"MACHINES lists {example}, which manifest.json does not contain")
    for example in ranks:
        if example not in owners:
            problems.append(f"{example} is in manifest.json but MACHINES gives it to no machine")

    arms, keys, mine = [], set(), set(MACHINES[machine])
    for i, (category, example, method, rels) in enumerate(entries):
        name = f"{category}/{example}"
        if name not in mine:
            continue
        lanes = {}
        for stage in STAGES:
            try:
                lane, err = parse_lane((ROOT / rels[stage]).read_bytes())
            except OSError as exc:
                problems.append(f"{rels[stage]}: {exc.strerror or exc}")
                continue
            if err:
                problems.append(f"{rels[stage]}: {err}")
            else:
                lanes[stage] = lane
        if len(lanes) == 2 and lanes["train"] != lanes["infer"]:
            problems.append(f"{name}/{method}: config_train says gpu_ids {lanes['train']} "
                            f"but config_infer says {lanes['infer']}")
        arm = Arm(category, example, method, lanes.get("train", lanes.get("infer", -1)),
                  (ranks[name], i), rels)
        if arm.key in keys:
            problems.append(f"{arm.label} appears twice in manifest.json")
        keys.add(arm.key)
        arms.append(arm)

    by_method = {}
    for arm in arms:
        if arm.lane >= 0:  # -1: neither config parsed, already a problem above
            by_method.setdefault(arm.method, set()).add(arm.lane)
    for method, lanes in sorted(by_method.items()):
        if len(lanes) > 1:
            warnings.append(f"{method} arms sit on lanes {sorted(lanes)}; generate.py's "
                            "LANE_GPU keeps each method on one lane")
    if not LAUNCHER.is_file():
        problems.append(f"launcher not found: {LAUNCHER}")
    problem = toml_problem()
    if problem:
        problems.append(problem)
    arms.sort(key=lambda a: a.rank)
    return arms, problems, warnings

class StatePaths:
    def __init__(self, machine: str):
        self.state = ROOT / "output" / "dataset_matrix" / "_campaign" / machine
        self.logs = self.state / "logs"
        self.stage_logs = self.logs / "stages"
        self.done = self.state / "done"
        self.launch = self.state / "launch"
        self.lock = self.state / "runner.lock"

    def marker(self, arm: Arm, stage: str) -> Path:
        return self.done / f"{arm.key}__{stage}"

def marker_state(paths: StatePaths, arm: Arm, stage: str):
    """(done, note). A marker whose config has changed since still counts as done."""
    marker = paths.marker(arm, stage)
    if not marker.exists():
        return False, ""
    try:
        raw = marker.read_bytes()
    except OSError as exc:
        return True, f"marker unreadable ({exc}); counted as done"
    if not raw.strip():
        return True, "empty marker from the old bash runner, config hash unknown; counted as done"
    try:
        recorded = json.loads(raw.decode("utf-8")).get("sha256")
    except (ValueError, AttributeError):
        return True, "marker is not JSON; counted as done"
    try:
        current = sha256(arm.path(stage).read_bytes())
    except OSError:
        current = None
    if recorded != current:
        return True, (f"config changed since it ran (marker {str(recorded)[:12]}, now "
                      f"{str(current)[:12]}); delete {rel(marker)} to run it again")
    return True, ""

# ---------------------------------------------------------- nvidia-smi and CUDA

class Gpu:
    def __init__(self, index: int, uuid: str):
        self.index, self.uuid = index, uuid

_STUCK = {}
_STUCK_LOCK = threading.Lock()

def run_bounded(name: str, argv, timeout: float, env=None):
    """(returncode, output), or (None, reason) if it could not run or hung.

    Never blocks much past `timeout`. While an earlier `name` is still stuck
    after a kill (a wedged driver can make that permanent) it is not started
    again, so a hung GPU does not pile up processes.
    """
    with _STUCK_LOCK:
        stuck = _STUCK.get(name)
        if stuck is not None and stuck.poll() is None:
            return None, f"an earlier {name} (pid {stuck.pid}) is still stuck"
        _STUCK.pop(name, None)
    try:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        return None, f"{name} could not run: {exc}"
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            with _STUCK_LOCK:
                _STUCK[name] = proc
        return None, f"{name} did not answer within {timeout:g}s"
    return proc.returncode, out.decode("utf-8", "replace")

def query_gpus():
    """([{index, uuid, util, raw, fault}], "") or (None, reason).

    A card that has fallen off the bus is simply absent while the others are
    still listed, even though nvidia-smi then exits non-zero.
    """
    rc, out = run_bounded("nvidia-smi", ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu",
                                         "--format=csv,noheader,nounits"], NVSMI_TIMEOUT)
    if rc is None:
        return None, out
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and parts[0].isdigit() and parts[1].startswith(("GPU-", "MIG-")):
            gpus.append({"index": int(parts[0]), "uuid": parts[1], "raw": parts[2],
                         "util": int(parts[2]) if parts[2].isdigit() else None,
                         "fault": any(m in parts[2].lower() for m in FAULT_MARKS)})
    if not gpus:
        return None, f"nvidia-smi listed no GPUs (exit {rc}): {out.strip()[:300]!r}"
    return gpus, ""

def query_apps():
    rc, out = run_bounded("nvidia-smi", ["nvidia-smi",
                                         "--query-compute-apps=pid,gpu_uuid,process_name,used_memory",
                                         "--format=csv,noheader"], NVSMI_TIMEOUT)
    if rc is None:
        return None, out
    if rc != 0:
        return None, f"nvidia-smi --query-compute-apps exited {rc}: {out.strip()[:300]!r}"
    return [ln.strip() for ln in out.splitlines() if ln.count(",") >= 3], ""

def cuda_usable(uuid: str):
    """(True, "") / (False, why) / (None, why) for one card, through the CUDA driver."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER="PCI_BUS_ID")
    rc, out = run_bounded(f"CUDA probe of {uuid}", [sys.executable, "-c", CUDA_PROBE],
                          CUDA_PROBE_TIMEOUT, env=env)
    if rc is None:
        return None, out
    lines = out.strip().splitlines()
    words = lines[-1].split(None, 1) if lines else []
    verdict = words[0] if words else ""
    detail = words[1] if len(words) > 1 else ""
    if rc == 0 and verdict == "usable":
        return True, ""
    if rc == 0 and verdict == "unusable":
        return False, detail
    if rc == 0 and verdict == "unknown" and detail:
        return None, detail
    return None, f"CUDA probe exited {rc}: {out.strip()[-300:]!r}"

class GpuProbe:
    """The nvidia-smi listing, shared by the workers so they do not all poll it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._at = float("-inf")
        self._uuids = None

    def alive(self, uuid: str, max_age: float = 60.0):
        """True, False (not listed, or listed as lost), or None when nvidia-smi
        gave no listing."""
        with self._lock:
            if time.monotonic() - self._at > max_age:
                gpus, _ = query_gpus()
                self._uuids = (None if gpus is None
                               else {g["uuid"] for g in gpus if not g["fault"]})
                self._at = time.monotonic()
            uuids = self._uuids
        return None if uuids is None else uuid in uuids

# ------------------------------------------------------------ process control

def tagged(prefix: bytes):
    """{pid: tag} for every process whose environment carries `prefix`."""
    found = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return found
    me = os.getpid()
    for name in names:
        if not name.isdigit() or int(name) == me:
            continue
        try:
            with open(f"/proc/{name}/environ", "rb") as fh:
                blob = fh.read()
        except OSError:
            continue
        for item in blob.split(b"\0"):
            if item.startswith(prefix):
                found[int(name)] = item.decode("utf-8", "replace")
                break
    return found

def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
    except OSError:
        return "?"
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()[:200]

def send(pid: int, sig, group: bool = False) -> None:
    try:
        (os.killpg if group else os.kill)(pid, sig)
    except OSError:
        pass

def terminate(ctx, prefix: bytes, proc=None, say=None) -> bool:
    """SIGTERM every process tagged `prefix` (and proc's own group); SIGKILL
    after the grace period, or at once after a second stop signal. SIGINT is
    never used. True once nothing is left."""
    say = say or ctx.log
    sig = signal.SIGKILL if ctx.force_kill else signal.SIGTERM
    deadline = time.monotonic() + ctx.s.kill_grace
    signalled = set()
    while True:
        if proc is not None and proc.poll() is None and proc.pid not in signalled:
            send(proc.pid, sig, group=True)  # the launcher leads its own session
            signalled.add(proc.pid)
        for pid in set(tagged(prefix)) - signalled:
            send(pid, sig)
            signalled.add(pid)
        time.sleep(0.5)
        left = sorted(tagged(prefix))
        if (proc is None or proc.poll() is not None) and not left:
            return True
        if sig != signal.SIGKILL and (ctx.force_kill or time.monotonic() >= deadline):
            say(f"pids {left or [proc.pid]} still alive after SIGTERM; sending SIGKILL")
            sig, signalled, deadline = signal.SIGKILL, set(), time.monotonic() + 15
        elif sig == signal.SIGKILL and time.monotonic() >= deadline:
            say(f"pids {left or [proc.pid]} survived SIGKILL (stuck in the driver?); leaving them")
            return False

def scan_attempt(log_path: Path, offset: int):
    """(started, not_started, interrupted) for the attempt that begins at offset."""
    with open(log_path, "rb") as fh:
        fh.seek(offset)
        head = fh.read(PRELUDE_BYTES)
        started = bool(STARTED_LINE.search(head))
        not_started = NOT_STARTED in head
        interrupted, carry, chunk = False, b"", head
        while chunk:
            window = carry + chunk
            if INTERRUPT_MARK in window:
                interrupted = True
                break
            carry = window[-(len(INTERRUPT_MARK) - 1):]
            chunk = fh.read(PRELUDE_BYTES)
    return started, not_started, interrupted

# ------------------------------------------------------------------ run state

class Settings:
    def __init__(self, args):
        self.skip_gate = args.skip_gpu_gate or env_flag("SKIP_GPU_GATE")
        # at least 1 s: at 0 a closed gate would re-poll nvidia-smi and log without pause
        self.gate_interval = max(1.0, env_number("GATE_INTERVAL", 3600.0))
        self.gate_samples = max(1, env_number("GATE_SAMPLES", 3, int))
        self.gate_gap = env_number("GATE_SAMPLE_GAP", 20.0)
        self.gate_allow_apps = env_flag("GATE_ALLOW_APPS")
        self.expected_gpus = env_number("EXPECTED_GPUS", 8, int)
        if self.expected_gpus < 1:
            raise UsageError("EXPECTED_GPUS must be at least 1")
        self.watch_interval = max(1.0, env_number("WATCH_INTERVAL", 300.0))
        self.kill_grace = env_number("KILL_GRACE", 30.0)
        self.confirm_gap = env_number("DEAD_CONFIRM_GAP", 15.0)

class Log:
    def __init__(self, path=None):
        self._lock = threading.RLock()
        self._fh = open(path, "a", encoding="utf-8") if path else None

    def __call__(self, msg: str, tag: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"[{stamp}]{' [' + tag + ']' if tag else ''} {part}"
                 for part in str(msg).splitlines() or [""]]
        with self._lock:
            for line in lines:
                if self._fh is not None:
                    try:
                        self._fh.write(line + "\n")
                        self._fh.flush()
                    except (OSError, ValueError):
                        pass
                try:
                    print(line, flush=True)
                except (OSError, ValueError):
                    pass

class ArmQueue:
    """Per-lane FIFOs. A worker takes its own lane first; once that is empty it
    takes the head of the lane furthest behind in example order (ties: the
    longest lane, then the lowest). Counts the arms in flight, so a worker
    with nothing to take can tell "all done" from "one may still come back"."""

    def __init__(self, arms):
        self._lock = threading.Lock()
        self._lanes = {lane: deque() for lane in range(NUM_LANES)}
        self._in_flight = 0
        for arm in sorted(arms, key=lambda a: a.rank):
            self._lanes[arm.lane].append(arm)

    def claim(self, lane: int):
        with self._lock:
            own = self._lanes.get(lane)
            if own:
                self._in_flight += 1
                return own.popleft(), False
            waiting = [(q[0].rank[0], -len(q), n) for n, q in self._lanes.items() if q]
            if not waiting:
                return None, False
            self._in_flight += 1
            return self._lanes[min(waiting)[2]].popleft(), True

    def finish(self, arm: Arm, requeue: bool) -> None:
        """One claimed arm is off its worker; requeue puts it back at the front
        of its lane in the same step, so no idle worker sees a gap."""
        with self._lock:
            self._in_flight -= 1
            if requeue:
                self._lanes[arm.lane].appendleft(arm)

    def drained(self) -> bool:
        """Nothing queued and nothing in flight: no arm can come back."""
        with self._lock:
            return self._in_flight == 0 and not any(self._lanes.values())

    def remaining(self):
        with self._lock:
            return [arm for lane in sorted(self._lanes) for arm in self._lanes[lane]]

class Ctx:
    def __init__(self, machine: str, settings: Settings):
        self.machine, self.s = machine, settings
        self.paths = StatePaths(machine)
        self.python = sys.executable
        self.run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
        self.run_tag = f"dataset_matrix:{machine}:{self.run_id}"
        self.stop_requested = False
        self.force_kill = False
        self.stop_signal = None
        self.probe = GpuProbe()
        self.queue = None
        self.log = Log()
        self.workers = {}
        self._results = {}
        self._results_lock = threading.Lock()

    def stopping(self) -> bool:
        return self.stop_requested

    def sleep(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while not self.stop_requested:
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(1.0, left))

    def record(self, arm: Arm, stage: str, outcome: str, **info) -> None:
        with self._results_lock:
            self._results[(arm.key, stage)] = dict(outcome=outcome, **info)

    def result(self, arm: Arm, stage: str):
        with self._results_lock:
            return self._results.get((arm.key, stage))

# -------------------------------------------------------------------- stages

GONE_FROM_NVSMI = "no longer listed by nvidia-smi, or listed as lost"

def card_state(ctx: Ctx, gpu: Gpu):
    """(False, why) if the card is gone -- nvidia-smi lists other cards but not
    this one (or lists it as lost), or the CUDA driver cannot open it --
    (True, "") if it answers, (None, why) if neither check could tell."""
    listed = ctx.probe.alive(gpu.uuid, max_age=0)
    if listed is False:
        return False, GONE_FROM_NVSMI
    usable, why = cuda_usable(gpu.uuid)
    if usable is False:
        return False, f"the CUDA driver cannot use it ({why})"
    if usable is None and listed is None:
        return None, f"nvidia-smi gave no listing and the CUDA probe could not tell ({why})"
    return True, ""

def gpu_confirmed_lost(ctx: Ctx, gpu: Gpu):
    """(True, why) only if two checks, confirm_gap apart, both find the card gone."""
    why = ""
    for attempt in range(2):
        if attempt:
            ctx.sleep(ctx.s.confirm_gap)
        state, why = card_state(ctx, gpu)
        if state is not False:
            return False, why
    return True, why

def run_stage(ctx: Ctx, gpu: Gpu, arm: Arm, stage: str) -> str:
    tag = f"gpu{gpu.index}"
    say = lambda msg: ctx.log(f"  {stage}: {msg}", tag)  # noqa: E731
    if ctx.stopping():
        return STOPPED
    try:
        data = arm.path(stage).read_bytes()
        launch = ctx.paths.launch / f"{arm.key}__{stage}.txt"
        body = launch_bytes(arm.rels[stage], data, arm.lane, gpu)
        write_atomic(launch, body)
        if launch.read_bytes() != body or parse_lane(body) != (0, None):
            raise ValueError(f"{rel(launch)} did not read back as written")
    except (OSError, ValueError) as exc:
        say(f"cannot write the launch copy: {exc}")
        ctx.record(arm, stage, FAILED, detail=f"launch copy: {exc}", gpu=gpu.index)
        return FAILED

    stage_tag = f"{ctx.run_id}:{gpu.index}:{arm.key}:{stage}"
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu.uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONUNBUFFERED": "1",
        "CAE4ALL_RUN_TAG": ctx.run_tag,
        "CAE4ALL_STAGE_TAG": stage_tag,
    })
    command = [ctx.python, str(LAUNCHER), "--config", str(launch)]
    log_path = ctx.paths.stage_logs / f"{arm.key}__{stage}.log"
    header = (
        f"\n===== {now_iso()}  run {ctx.run_id}  machine {ctx.machine}  "
        f"GPU {gpu.index} {gpu.uuid}  {arm.label} {stage} =====\n"
        f"config   {arm.rels[stage]}  sha256 {sha256(data)}\n"
        f"launch   {rel(launch)}  (gpu_ids {arm.lane} -> 0)\n"
        f"command  CUDA_VISIBLE_DEVICES={gpu.uuid} {' '.join(command)}\n\n"
    )
    say(f"running, log {rel(log_path)}")
    started = time.monotonic()
    try:
        with open(log_path, "ab") as fh:
            fh.write(header.encode("utf-8"))
            fh.flush()
            offset = os.fstat(fh.fileno()).st_size
            proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                    stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as exc:
        say(f"could not start the launcher: {exc}")
        ctx.record(arm, stage, FAILED, detail=f"could not start: {exc}", gpu=gpu.index)
        return FAILED

    prefix = f"CAE4ALL_STAGE_TAG={stage_tag}".encode()
    stopped = lost = settled = False
    misses, next_watch = 0, started + ctx.s.watch_interval
    try:
        while True:
            try:
                proc.wait(timeout=2.0)
                break
            except subprocess.TimeoutExpired:
                pass
            if ctx.stopping():
                stopped = True
                say("stop requested; terminating the stage")
                terminate(ctx, prefix, proc, say)
                break
            if time.monotonic() >= next_watch:
                listed = ctx.probe.alive(gpu.uuid, max_age=0)
                if listed is False:
                    misses += 1
                    say(f"GPU missing from nvidia-smi or listed as lost ({misses}/2)")
                elif listed is True:
                    misses = 0
                if misses >= 2:
                    lost = True
                    say("GPU lost while running; terminating the stage")
                    terminate(ctx, prefix, proc, say)
                    break
                next_watch = time.monotonic() + (ctx.s.confirm_gap if misses
                                                 else ctx.s.watch_interval)
        settled = True
    finally:
        if not settled:
            terminate(ctx, prefix, proc, say)
    rc = proc.poll()
    elapsed = time.monotonic() - started
    if not (stopped or lost) and tagged(prefix):
        say("processes of this stage outlived its launcher; terminating them")
        terminate(ctx, prefix, None, say)

    started_ok, not_started, interrupted = scan_attempt(log_path, offset)
    launched = started_ok and not not_started
    why = ""
    if stopped or (ctx.stopping() and rc != 0):
        outcome = STOPPED
    elif lost:
        outcome, why = GPU_LOST, GONE_FROM_NVSMI
    elif launched and interrupted:
        outcome = INTERRUPTED
    elif launched and rc == 0:
        outcome = OK
    else:
        gone, why = gpu_confirmed_lost(ctx, gpu)
        outcome = GPU_LOST if gone else (FAILED if launched else NOT_LAUNCHED)

    with open(log_path, "ab") as fh:
        fh.write(f"\n===== {now_iso()}  exit {rc}  after {fmt_secs(elapsed)}  -> "
                 f"{outcome} =====\n".encode("utf-8"))
    if outcome == OK:
        marker = {
            "config": arm.rels[stage], "sha256": sha256(data), "finished": now_iso(),
            "machine": ctx.machine, "run_id": ctx.run_id, "gpu_index": gpu.index,
            "gpu_uuid": gpu.uuid, "elapsed_s": round(elapsed, 1),
            "launch_copy": rel(launch), "log": rel(log_path),
        }
        write_atomic(ctx.paths.marker(arm, stage),
                     (json.dumps(marker, indent=1) + "\n").encode("utf-8"))
    detail = {
        OK: "",
        STOPPED: "stopped by the operator",
        GPU_LOST: why,
        NOT_LAUNCHED: "preflight or routing did not start the model; see the log",
        INTERRUPTED: "the model printed 'interrupted by user'; not counted as finished",
        FAILED: f"exit {rc}",
    }[outcome]
    ctx.record(arm, stage, outcome, detail=detail, rc=rc, gpu=gpu.index, uuid=gpu.uuid,
               log=rel(log_path), elapsed_s=round(elapsed, 1))
    say(f"{outcome} (exit {rc}, {fmt_secs(elapsed)})"
        + (f" - {detail}" if detail and outcome != FAILED else ""))
    return outcome

def run_arm(ctx: Ctx, gpu: Gpu, arm: Arm, stolen: bool) -> str:
    tag = f"gpu{gpu.index}"
    ctx.log(arm.label + (f"  (taken over from lane {arm.lane})" if stolen else ""), tag)
    done, note = marker_state(ctx.paths, arm, "train")
    if done:
        ctx.log("  train: already done" + (f" - {note}" if note else ""), tag)
    else:
        outcome = run_stage(ctx, gpu, arm, "train")
        if outcome != OK:
            if outcome in (FAILED, NOT_LAUNCHED, INTERRUPTED):
                ctx.record(arm, "infer", SKIPPED, detail="its training did not finish")
                ctx.log("  infer: skipped - training did not finish", tag)
            return outcome
        stale = ctx.paths.marker(arm, "infer")
        if stale.exists():
            stale.unlink()  # it scored the checkpoint that was just replaced
            ctx.log("  infer: the earlier result belongs to the previous checkpoint; "
                    "running it again", tag)
    done, note = marker_state(ctx.paths, arm, "infer")
    if done:
        ctx.log("  infer: already done" + (f" - {note}" if note else ""), tag)
        return OK
    return run_stage(ctx, gpu, arm, "infer")

def worker_loop(ctx: Ctx, gpu: Gpu, tag: str) -> str:
    standing_by = False
    while not ctx.stopping():
        if ctx.probe.alive(gpu.uuid) is False:
            gone, why = gpu_confirmed_lost(ctx, gpu)
            if gone:
                ctx.log(f"GPU lost ({why}); this worker stops and the other cards take over", tag)
                return "GPU lost"
        arm, stolen = ctx.queue.claim(gpu.index)
        if arm is None:
            if ctx.queue.drained():
                return "queue empty"
            if not standing_by:
                ctx.log("nothing left to take; standing by until the running arms finish "
                        "(an arm from a lost card comes back here)", tag)
                standing_by = True
            ctx.sleep(IDLE_POLL)
            continue
        standing_by = False
        requeue = False
        try:
            try:
                outcome = run_arm(ctx, gpu, arm, stolen)
            except Exception:  # a runner bug on one arm must not take the lane down
                ctx.log(f"runner error on {arm.label}:\n" + traceback.format_exc(), tag)
                stage = "infer" if marker_state(ctx.paths, arm, "train")[0] else "train"
                ctx.record(arm, stage, FAILED, detail="runner error; see runner.log",
                           gpu=gpu.index)
                outcome = FAILED
            if outcome == GPU_LOST:
                arm.dead_requeues += 1
                requeue = arm.dead_requeues <= MAX_DEAD_REQUEUES
                if requeue:
                    ctx.log(f"{arm.label} goes back to the front of lane {arm.lane} "
                            "for another card", tag)
                else:
                    ctx.log(f"{arm.label} has now been on two cards that were lost; "
                            "it is not handed to a third", tag)
        finally:
            ctx.queue.finish(arm, requeue)
        if outcome == STOPPED:
            return "stopped"
        if outcome == GPU_LOST:
            return "GPU lost"
    return "stopped"

def worker(ctx: Ctx, gpu: Gpu) -> None:
    tag = f"gpu{gpu.index}"
    ctx.workers[gpu.index] = "running"
    ctx.log(f"worker started on {gpu.uuid}", tag)
    try:
        reason = worker_loop(ctx, gpu, tag)
    except Exception:
        reason = "runner error"
        ctx.log("runner error:\n" + traceback.format_exc(), tag)
    ctx.workers[gpu.index] = reason
    ctx.log(f"worker finished: {reason}", tag)

# ---------------------------------------------------------------------- gate

def gate_reading(ctx: Ctx):
    gpus, err = query_gpus()
    if gpus is None:
        return None, err
    if len(gpus) != ctx.s.expected_gpus:
        return None, (f"{len(gpus)} GPU(s) visible, expected {ctx.s.expected_gpus} "
                      "(EXPECTED_GPUS=<n> runs on fewer)")
    busy = [f"GPU{g['index']}={g['util']}%" if g["util"] is not None else
            f"GPU{g['index']}={g['raw']}" for g in gpus if g["util"] != 0]
    if busy:
        return None, "utilisation not 0% on " + ", ".join(busy)
    if not ctx.s.gate_allow_apps:
        apps, err = query_apps()
        if apps is None:
            return None, err
        if apps:
            more = f" (+{len(apps) - 4} more)" if len(apps) > 4 else ""
            return None, (f"{len(apps)} compute process(es) on the GPUs: "
                          + "; ".join(apps[:4]) + more)
    return gpus, ""

def gate_condition(s: Settings) -> str:
    return (f"{s.expected_gpus} GPUs at 0% utilisation"
            + ("" if s.gate_allow_apps else " with no compute process"))

def wait_for_gate(ctx: Ctx):
    """The GPU list once the box is idle (at once with SKIP_GPU_GATE), None if not."""
    if ctx.s.skip_gate:
        gpus, err = query_gpus()
        if gpus is None:
            ctx.log(f"SKIP_GPU_GATE=1 but nvidia-smi gave no GPU list: {err}")
            return None
        ctx.log(f"SKIP_GPU_GATE=1 - starting on {len(gpus)} GPU(s) without waiting")
        if len(gpus) != ctx.s.expected_gpus:
            ctx.log(f"note: {len(gpus)} GPU(s) visible, not the expected {ctx.s.expected_gpus}")
        return gpus
    while not ctx.stopping():
        gpus, reason = None, ""
        for sample in range(ctx.s.gate_samples):
            if sample:
                ctx.sleep(ctx.s.gate_gap)
                if ctx.stopping():
                    return None
            gpus, reason = gate_reading(ctx)
            if gpus is None:
                break
        if gpus is not None:
            ctx.log(f"gate open: {gate_condition(ctx.s)} over {ctx.s.gate_samples} reading(s)")
            return gpus
        at = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + ctx.s.gate_interval))
        ctx.log(f"gate closed: {reason}. Waiting for {gate_condition(ctx.s)}; "
                f"next check at {at}")
        ctx.sleep(ctx.s.gate_interval)
    return None

# -------------------------------------------------------------------- report

def status_word(done: bool, res) -> str:
    if done:
        return "done*" if res and res["outcome"] == OK else "done"
    return res["outcome"] if res else "pending"

def write_report(ctx: Ctx, arms, gpus, started_at: str, stopped: bool) -> int:
    rows = []
    for arm in arms:
        for stage in STAGES:
            done, note = marker_state(ctx.paths, arm, stage)
            rows.append({"arm": arm.label, "lane": arm.lane, "stage": stage,
                         "config": arm.rels[stage], "done": done, "marker_note": note,
                         "this_run": ctx.result(arm, stage)})
    n_done = sum(r["done"] for r in rows)
    counts = {}
    for r in rows:
        if r["this_run"]:
            counts[r["this_run"]["outcome"]] = counts.get(r["this_run"]["outcome"], 0) + 1
    code = 130 if stopped else (0 if n_done == len(rows) else 1)
    lines = [
        f"dataset_matrix machine {ctx.machine}  run {ctx.run_id}",
        f"started {started_at}  finished {now_iso()}  exit {code}",
        f"stages done: {n_done}/{len(rows)}"
        + ("   this run: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
           if counts else ""),
        "workers: " + (", ".join(f"GPU{i} {w}" for i, w in sorted(ctx.workers.items()))
                       or "none started"),
        "",
        f"{'lane':<5}{'arm':<48}{'train':<14}{'infer':<14}",
    ]
    cells = {(r["arm"], r["stage"]): r for r in rows}
    for arm in sorted(arms, key=lambda a: (a.lane, a.rank)):
        t, i = cells[(arm.label, "train")], cells[(arm.label, "infer")]
        lines.append(f"{arm.lane:<5}{arm.label:<48}{status_word(t['done'], t['this_run']):<14}"
                     f"{status_word(i['done'], i['this_run']):<14}")
    lines += ["", "done* = finished in this run"]
    open_rows = [r for r in rows if not r["done"]]
    if open_rows:
        lines += ["", "not done:"]
        for r in open_rows:
            res = r["this_run"]
            if res:
                where = f" on GPU{res['gpu']}" if res.get("gpu") is not None else ""
                why = f" ({res['detail']})" if res.get("detail") else ""
                log = f"  log {res['log']}" if res.get("log") else ""
                lines.append(f"  {r['arm']} {r['stage']}: {res['outcome']}{where}{why}{log}")
            else:
                lines.append(f"  {r['arm']} {r['stage']}: not attempted in this run")
        lines.append("Run the same command again to pick these up; finished stages are skipped.")
    notes = [r for r in rows if r["marker_note"]]
    if notes:
        lines += ["", "markers to look at:"]
        lines += [f"  {r['arm']} {r['stage']}: {r['marker_note']}" for r in notes]
    text = "\n".join(lines) + "\n"
    report = {
        "machine": ctx.machine, "run_id": ctx.run_id, "started": started_at,
        "finished": now_iso(), "exit_code": code, "stopped": stopped,
        "stop_signal": ctx.stop_signal, "stages_done": n_done, "stages_total": len(rows),
        "gpus": [{"index": g.index, "uuid": g.uuid} for g in gpus],
        "workers": {str(k): v for k, v in sorted(ctx.workers.items())},
        "stages": rows,
    }
    try:
        write_atomic(ctx.paths.state / "report.json",
                     (json.dumps(report, indent=1) + "\n").encode("utf-8"))
        write_atomic(ctx.paths.state / "report.txt", text.encode("utf-8"))
    except OSError as exc:
        ctx.log(f"could not write the report files: {exc}")
    ctx.log(text.rstrip("\n"))
    ctx.log(f"report: {rel(ctx.paths.state / 'report.txt')} (and report.json)")
    return code

def spread_scores(ctx: Ctx, arms) -> None:
    """Informational only. Each machine owns one probabilistic example, so the
    other machine's arms always show up as missing here."""
    if not SCORE_SPREAD.is_file() or not any(a.category == "probabilistic" for a in arms):
        return
    ctx.log("probabilistic spread scores (informational):")
    try:
        cp = subprocess.run([ctx.python, str(SCORE_SPREAD), "--csv",
                             str(ctx.paths.state / "spread_scores.csv")],
                            cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=1800)
        ctx.log(cp.stdout.decode("utf-8", "replace").rstrip() or "(no output)")
    except (OSError, subprocess.TimeoutExpired) as exc:
        ctx.log(f"score_spread.py did not run: {exc}")

# ---------------------------------------------------------------------- modes

def lock_state(paths: StatePaths):
    """True if a runner holds the lock, False if not, None if this cannot tell."""
    if not paths.lock.exists():
        return False
    try:
        import fcntl
    except ImportError:
        return None
    try:
        fd = os.open(str(paths.lock), os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    except OSError:
        return None
    finally:
        os.close(fd)

def dry_run(machine: str, arms, s: Settings) -> int:
    paths = StatePaths(machine)
    held = lock_state(paths)
    print(f"dataset_matrix runner - machine {machine} - DRY RUN (nothing is written)")
    print(f"  root      {ROOT}")
    print(f"  python    {sys.executable} (Python {platform.python_version()})")
    problem = launcher_problem()
    print(f"  launcher  {rel(LAUNCHER)}: " + (f"PROBLEM - {problem}" if problem else "imports OK"))
    print(f"  examples  {', '.join(MACHINES[machine])}")
    print(f"  state     {rel(paths.state)}"
          + ("   <- a runner is active on it right now" if held else ""))
    gpus, _ = query_gpus()
    uuid_of = {g["index"]: g["uuid"] for g in gpus} if gpus else {}
    total = done_total = 0
    first = None
    for lane in range(NUM_LANES):
        lane_arms = [a for a in arms if a.lane == lane]
        print()
        if not lane_arms:
            print(f"lane {lane}: no home arms - its worker takes over other lanes' arms from the start")
            continue
        print(f"lane {lane}: {len(lane_arms)} arm(s)")
        for arm in lane_arms:
            cells, notes = [], []
            train_done = marker_state(paths, arm, "train")[0]
            for stage in STAGES:
                done, note = marker_state(paths, arm, stage)
                if stage == "infer" and done and not train_done:  # see run_arm
                    done, note = False, "a successful retrain removes this marker and runs it again"
                total += 1
                done_total += done
                cells.append(f"{stage} {'done' if done else 'pending'}{' (!)' if note else ''}")
                if note:
                    notes.append(f"{stage}: {note}")
                if not done and first is None:
                    first = (arm, stage)
            print(f"    {arm.label:<46} {cells[0]:<16} {cells[1]}")
            for note in notes:
                print(f"        (!) {note}")
    print()
    print(f"{len(arms)} arm(s), {total} stage(s): {done_total} done, {total - done_total} to run")
    if first is not None:
        arm, stage = first
        uuid = uuid_of.get(arm.lane, f"<UUID of GPU {arm.lane}>")
        print()
        print(f"first stage, as lane {arm.lane}'s worker would launch it:")
        print(f"  CUDA_VISIBLE_DEVICES={uuid} {sys.executable} {rel(LAUNCHER)} --config "
              f"{rel(paths.launch / (arm.key + '__' + stage + '.txt'))}")
        print(f"  launch copy = {arm.rels[stage]} with gpu_ids {arm.lane} -> 0")
        print(f"  log        -> {rel(paths.stage_logs / (arm.key + '__' + stage + '.log'))}")
    print()
    if s.skip_gate:
        print("gate: SKIP_GPU_GATE=1 - would start at once")
    else:
        visible, reason = gate_reading(Ctx(machine, s))
        print(f"gate: needs {gate_condition(s)}; {s.gate_samples} reading(s) "
              f"{fmt_secs(s.gate_gap)} apart, checked every {fmt_secs(s.gate_interval)}")
        print("gate now: " + ("would open" if visible else f"closed - {reason}"))
    return 0

def kill_group(proc) -> None:
    if proc.poll() is None:
        if os.name == "posix":
            send(proc.pid, signal.SIGKILL, group=True)
        else:
            proc.kill()

def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt

def check_mode(machine: str, arms, s: Settings) -> int:
    """launcher --check on every stage a run would still execute. An infer stage
    whose training has not run yet is deferred: its checkpoint does not exist,
    so its preflight cannot pass before the real run."""
    paths = StatePaths(machine)
    jobs, deferred, finished = [], 0, 0
    for arm in arms:
        if not marker_state(paths, arm, "train")[0]:
            jobs.append((arm, "train"))
            deferred += 1
        elif not marker_state(paths, arm, "infer")[0]:
            jobs.append((arm, "infer"))
        else:
            finished += 1
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"note: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} is set, so the "
              "checks see only those GPUs and higher gpu_ids fail ENV-CUDA-002")
    print(f"dataset_matrix machine {machine} - CHECK: {len(jobs)} config(s) through "
          f"'{rel(LAUNCHER)} --check', {CHECK_JOBS} at a time")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    running, lock, abort = {}, threading.Lock(), threading.Event()

    def one(job):
        arm, stage = job
        started = time.monotonic()
        if abort.is_set():
            return job, None, "cancelled", 0.0
        try:
            proc = subprocess.Popen([sys.executable, str(LAUNCHER), "--config",
                                     str(arm.path(stage)), "--check"],
                                    cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    start_new_session=(os.name == "posix"))
        except OSError as exc:
            return job, None, str(exc), 0.0
        with lock:
            running[proc.pid] = proc
            if abort.is_set():
                kill_group(proc)
        try:
            out, _ = proc.communicate(timeout=CHECK_TIMEOUT)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                out = b""
            out += f"\n[run_matrix] --check timed out after {CHECK_TIMEOUT}s".encode()
        finally:
            with lock:
                running.pop(proc.pid, None)
        return job, proc.returncode, out.decode("utf-8", "replace"), time.monotonic() - started

    if os.name == "posix":
        signal.signal(signal.SIGTERM, _raise_interrupt)
    pool = ThreadPoolExecutor(max_workers=CHECK_JOBS)
    futures, failed, n = [], 0, 0
    try:
        for job in jobs:  # inside the try: an interrupt here must still abort the queue
            futures.append(pool.submit(one, job))
        pending = set(futures)
        while pending:
            done_now, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
            for future in done_now:
                (arm, stage), rc, out, secs = future.result()
                n += 1
                ok = rc == 0
                failed += not ok
                print(f"[{n:>3}/{len(jobs)}] {'PASS' if ok else 'FAIL'}  {arm.rels[stage]}"
                      f"  ({fmt_secs(secs)}{'' if ok else f', exit {rc}'})", flush=True)
                if not ok:
                    lines = out.splitlines()
                    coded = [ln for ln in lines
                             if re.search(r"\b[A-Z]+(?:-[A-Z0-9]+)+-\d{3}\b", ln)]
                    for ln in (coded or lines[-15:])[:25]:
                        print(f"          {ln}")
    except KeyboardInterrupt:
        with lock:
            abort.set()
            procs = list(running.values())
        for future in futures:
            future.cancel()
        for proc in procs:
            kill_group(proc)
        pool.shutdown(wait=True)
        print(f"\nCHECK interrupted after {n} of {len(jobs)} config(s); {failed} failed so far")
        return 130
    pool.shutdown(wait=True)
    print()
    print(f"{len(jobs) - failed} passed, {failed} failed; {deferred} infer config(s) deferred "
          f"until their training has run; {finished} arm(s) already done")
    return 1 if failed else 0

def run(machine: str, arms, s: Settings) -> int:
    if os.name != "posix":
        print("Run mode needs Linux (process groups, /proc). DRY_RUN=1 and CHECK=1 work here.")
        return 2
    import fcntl

    ctx = Ctx(machine, s)
    paths = ctx.paths
    for d in (paths.state, paths.logs, paths.stage_logs, paths.done, paths.launch):
        d.mkdir(parents=True, exist_ok=True)
    lock_fh = open(paths.lock, "a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fh.seek(0)
        holder = lock_fh.read().strip() or "?"
        print(f"Another run_matrix.py for machine {machine} is running (pid and run id: "
              f"{holder}); it holds {rel(paths.lock)}. Nothing was started.")
        return 2
    except OSError as exc:
        print(f"note: {rel(paths.lock)} cannot be locked on this filesystem ({exc}); "
              "make sure only one runner per machine is started")
    else:
        lock_fh.seek(0)
        lock_fh.truncate()
        lock_fh.write(f"{os.getpid()} {ctx.run_id}\n")
        lock_fh.flush()

    leftovers = tagged(f"CAE4ALL_RUN_TAG=dataset_matrix:{machine}:".encode())
    if leftovers:
        print(f"{len(leftovers)} process(es) from an earlier run of machine {machine} "
              "are still alive:")
        for pid, tag in sorted(leftovers.items()):
            print(f"  pid {pid}  {tag}  {cmdline(pid)}")
        print("They would share GPUs with this run. Stop them (kill -TERM <pid> ...) and "
              "start again. Nothing was started.")
        return 2

    def on_signal(signum, _frame):
        if ctx.stop_requested:
            ctx.force_kill = True
        ctx.stop_requested = True
        ctx.stop_signal = ctx.stop_signal or signal.Signals(signum).name

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    hup = "ignored (nohup)"
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, on_signal)
        hup = "stops the run"

    ctx.log = log = Log(paths.logs / "runner.log")
    started_at = now_iso()
    log(f"=== dataset_matrix machine {machine}  run {ctx.run_id}  pid {os.getpid()} ===")
    log(f"python {ctx.python} (Python {platform.python_version()}); SIGHUP {hup}")
    log(f"examples: {', '.join(MACHINES[machine])}")
    log(f"gate: {'skipped (SKIP_GPU_GATE=1)' if s.skip_gate else gate_condition(s)}; "
        f"watch every {fmt_secs(s.watch_interval)}; kill grace {fmt_secs(s.kill_grace)}")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        log(f"note: CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} is ignored; "
            "every stage is bound to one card by UUID")
    if shutil.which("nvidia-smi") is None:
        log("nvidia-smi is not on PATH; the runner needs it to find and watch the GPUs. "
            "Nothing was started.")
        return 2

    pending = []
    for arm in arms:
        states = [marker_state(paths, arm, stage) for stage in STAGES]
        for stage, (_, note) in zip(STAGES, states):
            if note:
                log(f"marker: {arm.label} {stage}: {note}")
        if not all(done for done, _ in states):
            pending.append(arm)
    log(f"{len(pending)} of {len(arms)} arm(s) still to run; per lane: "
        + ", ".join(f"{lane}:{sum(a.lane == lane for a in pending)}" for lane in range(NUM_LANES)))

    gpus = []
    if pending:
        ctx.queue = ArmQueue(pending)
        snapshot = wait_for_gate(ctx)
        if snapshot is None:
            if ctx.stopping():
                return write_report(ctx, arms, [], started_at, stopped=True)
            return 2
        gpus = sorted((Gpu(g["index"], g["uuid"]) for g in snapshot), key=lambda g: g.index)
        if len({g.index for g in gpus}) != len(gpus) or len({g.uuid for g in gpus}) != len(gpus):
            log("nvidia-smi reported duplicate GPU indices or UUIDs; nothing was started")
            return 2
        log("GPUs: " + ", ".join(f"{g.index}={g.uuid}" for g in gpus))
        threads = []
        for gpu in gpus:
            t = threading.Thread(target=worker, args=(ctx, gpu), name=f"gpu{gpu.index}",
                                 daemon=True)
            t.start()
            threads.append(t)

        announced = False
        while True:
            alive = [t for t in threads if t.is_alive()]
            if not alive:
                break
            if ctx.stopping() and not announced:
                log(f"{ctx.stop_signal}: stopping - terminating every running stage "
                    f"(SIGKILL after {fmt_secs(s.kill_grace)}; a second signal kills at once)")
                announced = True
            alive[0].join(timeout=1.0)

        run_prefix = f"CAE4ALL_RUN_TAG={ctx.run_tag}".encode()
        if tagged(run_prefix):
            log("processes of this run are still alive after the workers stopped; "
                "terminating them")
            terminate(ctx, run_prefix)
        left = ctx.queue.remaining()
        if left and not ctx.stopping():
            log(f"every worker stopped with {len(left)} arm(s) still queued: "
                + ", ".join(a.label for a in left))
    code = write_report(ctx, arms, gpus, started_at, stopped=ctx.stopping())
    if not ctx.stopping():
        spread_scores(ctx, arms)
    return code

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, choices=sorted(MACHINES))
    ap.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    ap.add_argument("--check", action="store_true",
                    help="run launcher --check on every stage still to run")
    ap.add_argument("--skip-gpu-gate", action="store_true",
                    help="start without waiting for an idle box")
    args = ap.parse_args(argv)
    dry = args.dry_run or env_flag("DRY_RUN")
    check = args.check or env_flag("CHECK")
    if dry and check:
        print("DRY_RUN and CHECK are separate modes; pick one.")
        return 2
    try:
        settings = Settings(args)
    except UsageError as exc:
        print(f"{exc}; nothing was started.")
        return 2
    arms, problems, warnings = load_plan(args.machine)
    if not dry:
        problem = launcher_problem()
        if problem:
            problems.append(problem)
    for warning in warnings:
        print(f"warning: {warning}")
    if problems:
        print(f"{len(problems)} problem(s) in the plan; nothing was started:")
        for problem in problems:
            print(f"  - {problem}")
        return 2
    if dry:
        return dry_run(args.machine, arms, settings)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if check:
        return check_mode(args.machine, arms, settings)
    return run(args.machine, arms, settings)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
