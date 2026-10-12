"""Serving boundary and concurrency tests; native checkpoints are smoke-tested separately."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time

import pytest

from studio_backend.geometry_explorer import GeometryExplorerService, validate_generation
from studio_backend.geometry_worker import PROFILES


def wait_for(service, job_id, status="completed", timeout=4):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = service.job(job_id)
        if result["status"] == status:
            return result
        if result["status"] == "failed":
            raise AssertionError(result["error"])
        time.sleep(.01)
    raise AssertionError(service.job(job_id))


class GatedWorker:
    def __init__(self, block_first=False):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not block_first:
            self.release.set()

    def call(self, operation, job_id, payload, output_dir, progress):
        self.calls.append((operation, payload.get("alpha")))
        if operation == "park":
            return {"active_device": "cpu"}
        if len(self.calls) == 1:
            self.started.set()
            assert self.release.wait(4)
        mesh = {"vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]], "faces": [[0, 1, 2]],
                "alpha": payload["alpha"], "provenance": {"test_fixture": True}}
        mesh_path = output_dir / "mesh.json"
        mesh_path.write_text(json.dumps(mesh), encoding="utf-8")
        stl = output_dir / "geometry.stl"
        stl.write_text("explicit test fixture", encoding="utf-8")
        return {"mesh_path": str(mesh_path), "stl_path": str(stl), "timings": {"total_s": .01}}

    def close(self):
        self.release.set()


@pytest.fixture
def factory(tmp_path):
    services = []
    def make(worker=None, analyzer=None):
        for spec in PROFILES.values():
            for key in ("vae", "fm"):
                p = tmp_path / spec[key]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
        hi = tmp_path / "output/dataset_matrix/deterministic/ex13/himgn/model.pth"
        hi.parent.mkdir(parents=True, exist_ok=True)
        hi.touch()
        service = GeometryExplorerService(tmp_path, tmp_path / "runtime", worker or GatedWorker(), analyzer)
        services.append(service)
        return service
    yield make
    for service in services:
        service.close()


def test_arbitrary_alpha_preserved_and_invalid_numeric_requests_rejected():
    assert validate_generation({"alpha": .41327})["alpha"] == .41327
    for alpha in (float("nan"), float("inf"), -.01, 1.01, True, "bad"):
        with pytest.raises(ValueError, match="alpha"):
            validate_generation({"alpha": alpha})
    for extra in ({"index_a": 1, "index_b": 1}, {"index_b": 7, "source_num_samples": 7},
                  {"seed": 4.2}, {"mc_resolution": 256}, {"client_id": "../bad"}):
        with pytest.raises(ValueError):
            validate_generation({"alpha": .41327, **extra})


def test_latest_wins_drops_pending_and_never_publishes_running_stale_result(factory):
    worker = GatedWorker(block_first=True)
    service = factory(worker)
    first = service.generate({"alpha": .137, "client_id": "browser"})
    assert worker.started.wait(2)
    second = service.generate({"alpha": .281, "client_id": "browser"})
    latest = service.generate({"alpha": .41327, "client_id": "browser"})
    assert service.job(first["id"])["status"] == "superseded"
    assert service.job(second["id"])["status"] == "superseded"
    worker.release.set()
    completed = wait_for(service, latest["id"])
    assert completed["result"]["mesh"]["alpha"] == .41327
    assert worker.calls == [("generate", .137), ("generate", .41327)]
    assert "result" not in service.job(first["id"])


def test_independent_browser_requests_are_not_superseded(factory):
    worker = GatedWorker(block_first=True)
    service = factory(worker)
    first = service.generate({"alpha": .12, "client_id": "a"})
    assert worker.started.wait(2)
    second = service.generate({"alpha": .34, "client_id": "b"})
    worker.release.set()
    assert wait_for(service, first["id"])["result"]["mesh"]["alpha"] == .12
    assert wait_for(service, second["id"])["result"]["mesh"]["alpha"] == .34


def test_late_http_revision_cannot_supersede_newer_desired_geometry(factory):
    worker = GatedWorker(block_first=True)
    service = factory(worker)
    current = service.generate({"alpha": .62, "client_id": "browser", "request_revision": 8})
    assert worker.started.wait(2)
    stale = service.generate({"alpha": .21, "client_id": "browser", "request_revision": 7})
    assert stale["status"] == "superseded"
    assert service.job(current["id"])["status"] == "running"
    worker.release.set()
    assert wait_for(service, current["id"])["result"]["mesh"]["alpha"] == .62
    assert worker.calls == [("generate", .62)]


def test_analysis_parks_gpu_uses_generated_artifact_and_rejects_stale_geometry(factory):
    calls = []
    def analyzer(stl, output, root, graph_nodes, progress):
        assert stl.is_file()
        assert graph_nodes == 5000
        calls.append(stl)
        progress("inference", graph={"vertices": [[0, 0, 0]], "edges": []})
        return {"graph": {"vertices": [[0, 0, 0]]}, "field": {"values": [-.04], "unit": "mm"}}
    worker = GatedWorker()
    service = factory(worker, analyzer)
    generated = service.generate({"alpha": .41327})
    completed = wait_for(service, generated["id"])
    analyzed = service.analyze({"generation_id": generated["id"]})
    result = wait_for(service, analyzed["id"])
    assert worker.calls[-1][0] == "park"
    assert result["result"]["generation_id"] == generated["id"]
    assert calls == [Path(completed["result"]["stl_path"])]
    newer = service.generate({"alpha": .51})
    wait_for(service, newer["id"])
    with pytest.raises(ValueError, match="superseded"):
        service.analyze({"generation_id": generated["id"]})


def test_missing_checkpoints_are_visible_and_fail_before_job_creation(tmp_path):
    service = GeometryExplorerService(tmp_path, tmp_path / "runtime", GatedWorker())
    try:
        assert all(not p["available"] and len(p["missing_checkpoints"]) == 2
                   for p in service.profiles()["profiles"])
        with pytest.raises(ValueError, match="checkpoint.*missing"):
            service.generate({"profile": "car", "alpha": .43})
        assert not service.jobs
    finally:
        service.close()


def test_analysis_rejects_other_profile_and_unfinished_generation(factory):
    worker = GatedWorker(block_first=True)
    service = factory(worker)
    car = service.generate({"profile": "car", "alpha": .57})
    assert worker.started.wait(2)
    with pytest.raises(ValueError, match="completed"):
        service.analyze({"generation_id": car["id"]})
    worker.release.set()
    wait_for(service, car["id"])
    with pytest.raises(ValueError, match="DeepJEB"):
        service.analyze({"generation_id": car["id"]})


def test_csv_catalog_skips_explorer_scratch_and_is_not_crowded_out(monkeypatch, tmp_path):
    from studio_backend import paths, suite_bridge
    runtime, output = tmp_path / "runtime", tmp_path / "output"
    (runtime / "geometry_explorer" / "jobs" / "a").mkdir(parents=True)
    (runtime / "geometry_explorer" / "jobs" / "a" / "draft.csv").write_text("x\n1\n")
    (runtime / "geometry_explorer" / "jobs" / "a" / "mesh.json").write_text("{}")
    (output / "screen").mkdir(parents=True)
    (output / "screen" / "screening.csv").write_text("id,mass\na,1\n")
    for index in range(5):
        (output / f"newer_{index}.png").write_bytes(b"png")
    monkeypatch.setattr(suite_bridge, "RUNTIME_ROOT", runtime)
    monkeypatch.setattr(suite_bridge, "SUITE_ROOT", tmp_path)
    monkeypatch.setattr(suite_bridge, "CONFIG_RUNTIME", runtime / "configs")
    monkeypatch.setattr(suite_bridge, "GEOMETRY_EXPLORER_RUNTIME", runtime / "geometry_explorer")
    monkeypatch.setattr(paths, "FILE_LIMIT", 3)
    csv_names = [Path(item["path"]).name for item in suite_bridge.file_catalog("csv")["items"]]
    assert csv_names == ["screening.csv"]
    artifact_names = {Path(item["path"]).name for item in suite_bridge.file_catalog("artifact")["items"]}
    assert "mesh.json" not in artifact_names and "draft.csv" not in artifact_names
