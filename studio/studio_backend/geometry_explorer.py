"""Asynchronous, latest-wins native geometry and structural preview service."""
from __future__ import annotations

import atexit
from collections import deque
import copy
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

from .geometry_worker import PROFILES
from .paths import RUNTIME_ROOT, SUITE_ROOT, relative, utc_now


TERMINAL = {"completed", "failed", "superseded"}


def _integer(payload: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer between {lo} and {hi}.")
    try:
        number = int(value)
        if float(value) != number or not lo <= number <= hi:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{key} must be an integer between {lo} and {hi}.") from None
    return number


def validate_generation(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Generation request must be a JSON object.")
    profile = str(payload.get("profile", "deepjeb")).lower()
    if profile not in PROFILES:
        raise ValueError("profile must be car, deepjeb, or mcb.")
    try:
        alpha = float(payload.get("alpha", 0.0))
    except (TypeError, ValueError):
        raise ValueError("alpha must be a finite number between 0 and 1.") from None
    if isinstance(payload.get("alpha"), bool) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1.")
    a = _integer(payload, "index_a", 0, 0, 255)
    b = _integer(payload, "index_b", 1, 0, 255)
    if a == b:
        raise ValueError("index_a and index_b must be distinct.")
    count = _integer(payload, "source_num_samples", max(16, a + 1, b + 1), 2, 256)
    if max(a, b) >= count:
        raise ValueError("source_num_samples must exceed both endpoint indices.")
    resolution = _integer(payload, "mc_resolution", 64, 32, 96)
    steps = _integer(payload, "ode_steps", 50, 10, 100)
    seed = _integer(payload, "seed", 42, 0, 2**32 - 1)
    client = str(payload.get("client_id") or "default")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,100}", client):
        raise ValueError("client_id must contain 1–100 letters, digits, dots, underscores, colons, or hyphens.")
    normalized = {"profile": profile, "alpha": alpha, "seed": seed, "index_a": a, "index_b": b,
                  "source_num_samples": count, "mc_resolution": resolution, "ode_steps": steps,
                  "client_id": client}
    if payload.get("request_revision") is not None:
        normalized["request_revision"] = _integer(payload, "request_revision", 0, 0, 2**53 - 1)
    return normalized


class ResidentWorker:
    """Private line-JSON protocol; only the service's scheduler may call it."""
    def __init__(self, root: Path, runtime: Path, timeout_s: float = 180):
        self.root, self.runtime = root, runtime
        self.process = None
        self.log = None
        self.timeout_s = timeout_s
        self.closed = False

    def _start(self):
        if self.closed:
            raise RuntimeError("Geometry worker is shutting down.")
        if self.process is not None and self.process.poll() is None:
            return
        self._dispose()
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.log = (self.runtime / "worker.log").open("a", encoding="utf-8")
        command = [sys.executable, "-u", str(self.root / "studio" / "studio_backend" / "geometry_worker.py")]
        self.process = subprocess.Popen(command, cwd=self.root / "methods" / "SDFFlow",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding="utf-8", bufsize=1)

    def call(self, operation: str, job_id: str, payload: dict, output_dir: Path, progress) -> dict:
        self._start()
        request = {"operation": operation, "id": job_id, "payload": payload, "output_dir": str(output_dir)}
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        process = self.process
        expired = threading.Event()
        def watchdog():
            expired.set()
            if process.poll() is None:
                process.kill()
        timer = threading.Timer(self.timeout_s, watchdog)
        timer.daemon = True
        timer.start()
        try:
            while True:
                line = process.stdout.readline()
                if not line:
                    if expired.is_set():
                        raise RuntimeError(f"Native geometry generation exceeded {self.timeout_s:g}s; the worker was stopped and will restart on the next request.")
                    code = process.poll()
                    raise RuntimeError(f"Native geometry worker stopped (exit={code}); inspect {relative(self.runtime / 'worker.log')}.")
                message = json.loads(line)
                if message.get("id") != job_id:
                    continue
                if message["type"] == "progress":
                    progress(message.pop("phase"), **{k: v for k, v in message.items() if k not in {"id", "type"}})
                elif message["type"] == "failed":
                    raise RuntimeError(message["error"])
                elif message["type"] == "completed":
                    return message["result"]
        finally:
            timer.cancel()

    def close(self):
        self.closed = True
        self._dispose()

    def _dispose(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            for stream in (process.stdin, process.stdout):
                if stream:
                    stream.close()
        if self.log:
            self.log.close()
            self.log = None


class GeometryExplorerService:
    def __init__(self, suite_root: Path = SUITE_ROOT, runtime: Path | None = None,
                 worker=None, analyzer=None):
        self.root = Path(suite_root).resolve()
        self.runtime = Path(runtime or RUNTIME_ROOT / "geometry_explorer").resolve()
        self.worker = worker or ResidentWorker(self.root, self.runtime)
        self.analyzer = analyzer
        self.condition = threading.Condition(threading.RLock())
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue: deque[str] = deque()
        self.latest: dict[str, str] = {}
        self.latest_generation: dict[str, str] = {}
        self.revisions: dict[str, int] = {}
        self.thread = None
        self.stopping = False
        self.closed = False
        self.cancel_event = threading.Event()

    def profiles(self) -> dict:
        items = []
        for key, spec in PROFILES.items():
            missing = [spec[name] for name in ("fm", "vae") if not (self.root / spec[name]).is_file()]
            items.append({"id": key, "title": spec["title"], "available": not missing,
                          "missing_checkpoints": missing, "checkpoint": spec["fm"],
                          "vae_checkpoint": spec["vae"], "conditioning": spec["conditioning"],
                          "default": {"seed": 42, "index_a": 0, "index_b": 1, **spec.get("endpoints", {}),
                                      "source_num_samples": 16, "mc_resolution": 64, "ode_steps": 50}})
        hi_checkpoint = "output/dataset_matrix/deterministic/ex13/himgn/model.pth"
        return {"profiles": items, "analysis": {"profile": "deepjeb", "checkpoint": hi_checkpoint,
                 "available": (self.root / hi_checkpoint).is_file(), "graph_nodes": 5000,
                 "method": "native surface graph + HI-MGN surrogate; vertical load case"},
                "runtime": {"resident": True, "serialized": True, "latest_wins": True,
                            "fresh_arbitrary_alpha": True, "preview_resolution": 64}}

    def _snapshot(self, job: dict) -> dict:
        public = {k: v for k, v in job.items() if not k.startswith("_")}
        public["elapsed_s"] = round((job.get("_finished") or time.perf_counter()) - job["_created"], 3)
        return copy.deepcopy(public)

    def job(self, job_id: str) -> dict:
        with self.condition:
            if job_id not in self.jobs:
                raise ValueError("Unknown geometry explorer job id.")
            return self._snapshot(self.jobs[job_id])

    def _create(self, kind: str, payload: dict) -> dict:
        with self.condition:
            if self.closed:
                raise ValueError("Geometry explorer is shutting down; restart Studio before creating another job.")
            client = payload["client_id"]
            revision = payload.get("request_revision")
            if revision is not None and revision < self.revisions.get(client, -1):
                # HTTP requests can arrive out of order even when the browser
                # debounces correctly. Do not let an older revision cancel the
                # current request after it reaches the server later.
                return {"id": uuid.uuid4().hex, "kind": kind, "status": "superseded",
                        "phase": "superseded", "superseded": True, "elapsed_s": 0,
                        "created_at": utc_now(), **payload}
            if revision is not None:
                self.revisions[client] = revision
            # A new geometry invalidates pending/running geometry and analysis
            # for this browser. Native GPU work already underway finishes, but
            # its stale result is never presented as the newest geometry.
            for old in self.jobs.values():
                if old.get("client_id") == client and old["status"] not in TERMINAL:
                    old["status"], old["phase"] = "superseded", "superseded"
                    old["superseded"] = True
                    old["_finished"] = time.perf_counter()
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "kind": kind, "status": "queued", "phase": "queued",
                   "created_at": utc_now(), "superseded": False, **payload,
                   "_created": time.perf_counter()}
            self.jobs[job_id] = job
            self.latest[client] = job_id
            if kind == "generation":
                self.latest_generation[client] = job_id
            self.queue.append(job_id)
            # Bound in-memory state, preserving the current generated artifact
            # referenced by any queued analysis job.
            protected = set(self.latest_generation.values()) | {
                j.get("generation_id") for j in self.jobs.values() if j["status"] not in TERMINAL}
            for old_id in list(self.jobs):
                if len(self.jobs) <= 128:
                    break
                if old_id != job_id and old_id not in protected and self.jobs[old_id]["status"] in TERMINAL:
                    del self.jobs[old_id]
            if self.thread is None or not self.thread.is_alive():
                self.stopping = False
                self.thread = threading.Thread(target=self._run, name="studio-native-geometry", daemon=True)
                self.thread.start()
            self.condition.notify_all()
            return self._snapshot(job)

    def generate(self, payload: dict) -> dict:
        normalized = validate_generation(payload)
        spec = PROFILES[normalized["profile"]]
        missing = [spec[k] for k in ("fm", "vae") if not (self.root / spec[k]).is_file()]
        if missing:
            raise ValueError("Required geometry checkpoint(s) are missing: " + ", ".join(missing))
        return self._create("generation", normalized)

    def analyze(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Analysis request must be a JSON object.")
        generation_id = str(payload.get("generation_id") or "")
        with self.condition:
            generation = self.jobs.get(generation_id)
            if generation is None or generation.get("kind") != "generation" or generation["status"] != "completed":
                raise ValueError("Analysis requires a completed, current generation_id.")
            if generation["profile"] != "deepjeb":
                raise ValueError("The installed HI-MGN checkpoint is trained for DeepJEB; structural analysis is available for that profile.")
            client = str(payload.get("client_id") or generation["client_id"])
            if client != generation["client_id"]:
                raise ValueError("Analysis client_id must match the geometry generation.")
            if self.latest_generation.get(client) != generation_id:
                raise ValueError("That geometry has been superseded; analyze the newest completed generation.")
            checkpoint = self.root / "output/dataset_matrix/deterministic/ex13/himgn/model.pth"
            if not checkpoint.is_file():
                raise ValueError("Required HI-MGN checkpoint is missing: " + str(checkpoint.relative_to(self.root)))
            normalized = {"generation_id": generation_id, "profile": "deepjeb",
                          "alpha": generation["alpha"], "client_id": client}
            if payload.get("request_revision") is not None:
                normalized["request_revision"] = _integer(payload, "request_revision", 0, 0, 2**53 - 1)
            return self._create("analysis", normalized)

    def _progress(self, job_id: str, phase: str, **data):
        with self.condition:
            job = self.jobs.get(job_id)
            if job is not None and job["status"] != "superseded":
                job["phase"] = phase
                job.update(data)

    def _run(self):
        while True:
            with self.condition:
                while not self.queue and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                job_id = self.queue.popleft()
                job = self.jobs.get(job_id)
                if job is None or job["status"] == "superseded":
                    continue
                job["status"], job["phase"] = "running", "starting"
                payload = {k: v for k, v in job.items() if not k.startswith("_")}
            output = self.runtime / "jobs" / job_id
            output.mkdir(parents=True, exist_ok=True)
            progress = lambda phase, **data: self._progress(job_id, phase, **data)
            try:
                if job["kind"] == "generation":
                    native = self.worker.call("generate", job_id, payload, output, progress)
                    mesh = json.loads(Path(native["mesh_path"]).read_text(encoding="utf-8"))
                    result = {**native, "generation_id": job_id, "mesh": mesh}
                else:
                    generation = self.jobs[payload["generation_id"]]
                    self.worker.call("park", job_id, {}, output, progress)
                    if self.analyzer is None:
                        from .geometry_analysis import analyze_geometry
                        analyzer = analyze_geometry
                    else:
                        analyzer = self.analyzer
                    options = {"graph_nodes": 5000, "progress": progress}
                    if self.analyzer is None:
                        options["cancel_event"] = self.cancel_event
                    result = analyzer(self.root / generation["result"]["stl_path"], output, self.root,
                                      **options)
                    result["generation_id"] = generation["id"]
                # Paths are repository-relative externally; no caller-supplied
                # artifact paths are accepted by either generate or analyze.
                for key in ("stl_path", "mesh_path", "graph_path", "field_path"):
                    if result.get(key):
                        result[key] = relative(self.root / result[key])
                with self.condition:
                    if job["status"] != "superseded":
                        job["status"], job["phase"] = "completed", "completed"
                        job["result"] = result
                        job["_finished"] = time.perf_counter()
                        (output / "job.json").write_text(json.dumps(self._snapshot(job), allow_nan=False), encoding="utf-8")
            except Exception as exc:
                with self.condition:
                    if job["status"] != "superseded":
                        job["status"], job["phase"] = "failed", "failed"
                        job["error"] = str(exc)
                        job["_finished"] = time.perf_counter()
                        (output / "job.json").write_text(json.dumps(self._snapshot(job)), encoding="utf-8")

    def close(self):
        with self.condition:
            self.closed = True
            self.stopping = True
            self.cancel_event.set()
            for item in self.jobs.values():
                if item["status"] not in TERMINAL:
                    item.update(status="superseded", phase="shutdown", superseded=True)
                    item["_finished"] = time.perf_counter()
            self.condition.notify_all()
        self.worker.close()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=8)


SERVICE = GeometryExplorerService()
atexit.register(SERVICE.close)


def profiles() -> dict:
    return SERVICE.profiles()


def generate(payload: dict) -> dict:
    return SERVICE.generate(payload)


def job(job_id: str) -> dict:
    return SERVICE.job(job_id)


def analyze(payload: dict) -> dict:
    return SERVICE.analyze(payload)
