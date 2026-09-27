"""Regression tests for the back-end hardening sweep.

Each class pins one defect that was reproduced before it was fixed: runner
threads that died and left a job "running" forever, cancels that still
launched, adopted jobs nobody watched, evaluation metrics that pooled fields or
paired unrelated samples, preview routes that answered 500 for a bad
selection, and a server that answered any Host / Origin / body type.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from studio_backend.paths import RUNTIME_ROOT, SUITE_ROOT, relative, safe_repo_path
from studio_backend.state import JOB_RUNTIME, StudioState, studio_preflight_diagnostics


def _ok_preflight() -> dict:
    return {
        "ok": True,
        "strict": False,
        "config_path": "studio/runtime/configs/test.txt",
        "report": {"summary": {"errors": 0, "warnings": 0, "notices": 0}, "diagnostics": []},
        "route": {"model": "test"},
        "resolved_paths": {},
        "dataset_metadata": {},
        "checkpoint_metadata": {},
    }


class FakeProcess:
    def __init__(self, returncode: int = 0, on_wait=None) -> None:
        self.pid = 4321
        self.stdout = io.BytesIO(b"native process ran\n")
        self.returncode = returncode
        self.on_wait = on_wait

    def wait(self) -> int:
        if self.on_wait is not None:
            self.on_wait()
        return self.returncode

    def poll(self):
        return None


class EngineCase(unittest.TestCase):
    """A StudioState without the import-time registry load or job recovery."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.repository = self.root / "method"
        self.repository.mkdir()
        self.engine = object.__new__(StudioState)
        self.engine.lock = threading.RLock()
        self.engine.jobs = {}
        self.engine.processes = {}

    def tearDown(self) -> None:
        self._directory.cleanup()

    def make_job(self, job_id: str = "job-under-test", steps: int = 1, **overrides) -> dict:
        job = {
            "id": job_id,
            "label": "hardening test",
            "status": "queued",
            "created_at": "2026-09-24T00:00:00+00:00",
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "pid": None,
            "current_step": 0,
            "total_steps": steps,
            "step_label": None,
            "cancel_requested": False,
            "log_path": self.root / f"{job_id}.log",
            "metadata_path": self.root / f"{job_id}.json",
            "steps": [{"label": f"step-{index + 1}"} for index in range(steps)],
        }
        job.update(overrides)
        self.engine.jobs[job_id] = job
        return job

    def launcher_step(self, label: str = "train") -> dict:
        config = self.root / f"{label}.txt"
        config.write_text("model test\nmode train\n", encoding="utf-8")
        return {"kind": "launcher", "label": label, "path": config, "node_id": "n1", "node_type": "model.test"}

    def install_gate(self, on_gate=None) -> None:
        repository = self.repository

        def gate(engine, path, **options):
            if on_gate is not None:
                on_gate()
            result = types.SimpleNamespace(
                command=["method-python", "entry.py", "--config", str(path)],
                resolved=types.SimpleNamespace(repository_root=repository),
            )
            return _ok_preflight(), result

        self.engine._preflight_config_path = types.MethodType(gate, self.engine)

    def log(self, job: dict) -> str:
        return job["log_path"].read_text(encoding="utf-8")


class RunnerCrashTests(EngineCase):
    """Item 1: every runner ends in a terminal status, whatever is raised."""

    def test_analysis_step_type_error_is_a_failed_step_not_a_dead_thread(self) -> None:
        job = self.make_job()

        def broken(payload):
            raise TypeError("unsupported operand")

        self.engine.ANALYSIS_ACTIONS = {"evaluation": broken}
        prepared = [{"kind": "analysis", "label": "evaluate", "action": "evaluation", "payload": {}, "node_id": "e1"}]
        self.engine._run_pipeline(job["id"], prepared, strict=False)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["returncode"], 2)
        self.assertIsNotNone(job["finished_at"])
        self.assertIn("TypeError: unsupported operand", self.log(job))

    def test_bookkeeping_crash_after_a_step_still_finalises_the_job(self) -> None:
        job = self.make_job()
        self.install_gate()

        def explode(*args, **kwargs):
            raise RuntimeError("bookkeeping bug")

        self.engine._record_step_results = explode
        with mock.patch("studio_backend.state.subprocess.Popen", return_value=FakeProcess()):
            self.engine._run_pipeline(job["id"], [self.launcher_step()], strict=False)
        self.assertEqual(job["status"], "failed")
        self.assertIsNotNone(job["finished_at"])
        self.assertIsNone(job["pid"])
        self.assertEqual(self.engine.processes, {})
        self.assertIn("failed unexpectedly", self.log(job))
        self.assertEqual(json.loads(job["metadata_path"].read_text(encoding="utf-8"))["status"], "failed")

    def test_command_job_crash_is_finalised(self) -> None:
        job = self.make_job()

        def explode(*args, **kwargs):
            raise RuntimeError("runner bug")

        self.engine._execute_command_job = explode
        self.engine._run_command_job(job["id"], ["python", "-V"], self.repository)
        self.assertEqual(job["status"], "failed")
        self.assertIsNotNone(job["finished_at"])

    def test_bad_analysis_steps_are_rejected_at_submission(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown action"):
            self.engine.create_pipeline_job(
                [{"kind": "analysis", "label": "x", "action": "evaluate-typo"}], label="t", strict=False
            )
        with self.assertRaisesRegex(ValueError, "payload must be a JSON object"):
            self.engine.create_pipeline_job(
                [{"kind": "analysis", "label": "x", "action": "evaluation", "payload": ["a"]}],
                label="t",
                strict=False,
            )


class CancelTests(EngineCase):
    """Items 2 and 3: a cancel means what it says, including for adopted jobs."""

    def test_cancel_during_launch_preflight_never_starts_the_process(self) -> None:
        job = self.make_job()
        self.install_gate(on_gate=lambda: job.update(cancel_requested=True))
        with mock.patch("studio_backend.state.subprocess.Popen") as popen:
            self.engine._run_pipeline(job["id"], [self.launcher_step()], strict=False)
        popen.assert_not_called()
        self.assertEqual(job["status"], "cancelled")
        self.assertEqual(job["returncode"], -1)
        self.assertIn("Cancelled before the native process started", self.log(job))

    def test_cancel_after_the_last_step_succeeded_is_completed(self) -> None:
        job = self.make_job()
        self.install_gate()
        self.engine._record_step_results = lambda *args, **kwargs: None
        process = FakeProcess(0, on_wait=lambda: job.update(cancel_requested=True))
        with mock.patch("studio_backend.state.subprocess.Popen", return_value=process):
            self.engine._run_pipeline(job["id"], [self.launcher_step()], strict=False)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["returncode"], 0)

    def test_cancel_between_popen_and_registration_kills_the_new_process(self) -> None:
        job = self.make_job()
        self.install_gate()

        def popen(*args, **kwargs):
            job["cancel_requested"] = True  # lands after the pre-Popen check
            return FakeProcess(1)

        with mock.patch("studio_backend.state.subprocess.Popen", side_effect=popen), mock.patch(
            "studio_backend.state.terminate_process_tree"
        ) as terminate:
            self.engine._run_pipeline(job["id"], [self.launcher_step()], strict=False)
        terminate.assert_called_once()
        self.assertEqual(job["status"], "cancelled")

    def test_cancel_reaches_an_adopted_job_by_pid(self) -> None:
        job = self.make_job(status="running", pid=98765)
        job["log_path"].write_text("", encoding="utf-8")
        self.engine._process_alive = lambda pid: True
        with mock.patch("studio_backend.state.terminate_pid_tree") as terminate:
            self.engine.cancel_job(job["id"])
        terminate.assert_called_once_with(98765)
        self.assertTrue(job["cancel_requested"])

    def test_watcher_finalises_an_adopted_job_as_interrupted(self) -> None:
        job = self.make_job(status="running", pid=98765, started_at="2026-09-24T00:00:00+00:00")
        self.engine._persist_job(job)
        self.engine._process_alive = lambda pid: False
        with mock.patch("studio_backend.state.time.sleep"):
            self.engine._watch_adopted_job(job["id"], 98765)
        self.assertEqual(job["status"], "interrupted")
        self.assertIsNone(job["pid"])
        self.assertIsNone(job["returncode"])
        self.assertIsNotNone(job["finished_at"])
        self.assertIn("restarted", job["note"])
        self.assertEqual(json.loads(job["metadata_path"].read_text(encoding="utf-8"))["status"], "interrupted")

    def test_watcher_takes_the_record_another_live_owner_wrote(self) -> None:
        job = self.make_job(status="running", pid=98765)
        self.engine._persist_job({**job, "status": "completed", "returncode": 0, "pid": None})
        self.engine._process_alive = lambda pid: False
        with mock.patch("studio_backend.state.time.sleep"):
            self.engine._watch_adopted_job(job["id"], 98765)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["returncode"], 0)

    def test_recovery_adopts_a_job_whose_process_is_still_alive(self) -> None:
        # conftest.py redirects JOB_RUNTIME; never write into the live jobs dir.
        self.assertNotIn(SUITE_ROOT.resolve(), Path(JOB_RUNTIME).resolve().parents)
        job_dir = JOB_RUNTIME / "adopt-test"
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            (job_dir / "job.json").write_text(
                json.dumps({"id": "adopt-test", "status": "running", "pid": 98765, "log_path": "run.log"}),
                encoding="utf-8",
            )
            adopted: list[tuple[str, int]] = []
            self.engine._process_alive = lambda pid: True
            self.engine._adopt_running_job = lambda job_id, pid: adopted.append((job_id, pid))
            self.engine._recover_jobs()
            self.assertEqual(adopted, [("adopt-test", 98765)])
            self.assertEqual(self.engine.jobs["adopt-test"]["status"], "running")
        finally:
            for item in job_dir.iterdir():
                item.unlink()
            job_dir.rmdir()

    def test_truncated_log_is_marked(self) -> None:
        job = self.make_job()
        job["log_path"].write_text("head line\n" + ("x" * 40 + "\n") * 20, encoding="utf-8")
        with mock.patch("studio_backend.state.LOG_TAIL_CHARS", 100):
            payload = self.engine.public_job(job, include_log=True)
        self.assertTrue(payload["log_truncated"])
        self.assertTrue(payload["log"].startswith("[studio] Log truncated"))
        self.assertNotIn("head line", payload["log"])
        with mock.patch("studio_backend.state.LOG_TAIL_CHARS", 100_000):
            self.assertFalse(self.engine.public_job(job, include_log=True)["log_truncated"])


class SimulGenParamScopeTests(unittest.TestCase):
    """Item 12: STUDIO-SGV-PARAM-001 only where param_dir is actually read."""

    def codes(self, **values) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            result = types.SimpleNamespace(
                resolved=types.SimpleNamespace(model_id="simulgenvae", repository_root=Path(directory)),
                parsed=types.SimpleNamespace(values=values),
            )
            return [item["code"] for item in studio_preflight_diagnostics(result)]

    def test_param_dir_checked_only_for_file_backed_lc_modes(self) -> None:
        missing = "no-such-conditions.csv"
        self.assertIn("STUDIO-SGV-PARAM-001", self.codes(mode="train", lc_data_type="csv", param_dir=missing))
        self.assertIn("STUDIO-SGV-PARAM-001", self.codes(mode="reconstruct", lc_data_type="image", param_dir=missing))
        # hdf5 conditions come from dataset_dir; train_vae never runs the LC.
        self.assertEqual(self.codes(mode="train", lc_data_type="hdf5", param_dir=missing), [])
        self.assertEqual(self.codes(mode="train_vae", lc_data_type="csv", param_dir=missing), [])


class ProgressCollapseTests(unittest.TestCase):
    """tqdm repaints fold away; text that merely ends in `\\r` does not."""

    @staticmethod
    def collapse(raw: bytes, keep_every: int = 40) -> str:
        # The same wrapper the runner puts on the child's binary pipe.
        stream = io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", newline="")
        return "".join(StudioState._collapse_progress(stream, keep_every=keep_every))

    def test_bar_repaints_are_folded(self) -> None:
        frames = b"".join(b"\r %2d%%|#   | %d/40 [00:0%d<00:03, 7.1it/s]" % (i, i, i % 10) for i in range(10))
        out = self.collapse(b"Epoch 0/2 TrainOpt: 1.0e-01 LR: 1e-04\n" + frames + b"\nEpoch 1/2 TrainOpt: 9.0e-02\n")
        self.assertEqual(out.count("Epoch"), 2)
        self.assertLessEqual(out.count("it/s"), 1)

    def test_epoch_text_split_by_another_ranks_frame_survives(self) -> None:
        # Rank 0 prints its text, rank 1 repaints (`\r` + frame), rank 0's
        # newline follows. The text used to end in `\r` and be dropped as a frame.
        raw = (
            b"Epoch 1/4 TrainOpt: 1.0e-01 Valid: 2.0e-01 LR: 1e-04\n"
            b"Epoch 2/4 TrainOpt: 9.0e-02 Valid: 1.9e-01 LR: 1e-04"
            b"\r  0%|          | 0/40 [00:00<?, ?it/s]\n"
            b"Epoch 3/4 TrainOpt: 8.0e-02 Valid: 1.8e-01 LR: 1e-04\r\n"
        )
        lines = self.collapse(raw).splitlines()
        self.assertIn("Epoch 2/4 TrainOpt: 9.0e-02 Valid: 1.9e-01 LR: 1e-04", lines)
        self.assertIn("Epoch 3/4 TrainOpt: 8.0e-02 Valid: 1.8e-01 LR: 1e-04", lines)
        self.assertEqual(sum(line.startswith("Epoch") for line in lines), 3)


class EvaluationMetricTests(unittest.TestCase):
    """Items 4-7: evaluation scores the right rows, per field, on the right pairs."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import h5py  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional feature dependency
            raise unittest.SkipTest(str(exc)) from exc
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory(prefix="hardening-eval-", dir=RUNTIME_ROOT)
        self.root = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def mesh(self, path: Path, samples: dict) -> None:
        import h5py

        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            handle.attrs["output_var"] = 2
            for sample_id, values in samples.items():
                handle.create_group(f"data/{sample_id}").create_dataset("nodal_data", data=values)

    def test_embedded_1d_vectors_are_scored_row_by_row(self) -> None:
        import h5py
        import numpy as np

        from studio_backend.analysis import run_field_evaluation

        prediction = self.root / "pred.h5"
        truth = self.root / "truth.h5"
        with h5py.File(prediction, "w") as handle:
            handle["Y_pred"] = np.array([1.0, 2.0, 3.0, 10.0])
            handle["Y_true"] = np.array([1.0, 2.0, 3.0, 4.0])
        with h5py.File(truth, "w") as handle:
            handle["Y"] = np.array([1.0, 2.0, 3.0, 4.0])
        report = run_field_evaluation({"prediction_path": str(prediction), "truth_path": str(truth)})
        with Path(report["per_sample_csv"]).open(encoding="utf-8") as handle:
            maes = [float(row["mae"]) for row in csv.DictReader(handle)]
        # Every row used to carry the whole-vector metric (1.5 four times).
        self.assertEqual(maes, [0.0, 0.0, 0.0, 6.0])

    def test_r2_and_relative_l2_are_per_field_macro_averages(self) -> None:
        import numpy as np

        from studio_backend.analysis import run_field_evaluation

        nodes = 2000
        coordinates = np.random.default_rng(0).random((3, 1, nodes))
        truth = np.zeros((5, 1, nodes))
        truth[0:3] = coordinates
        truth[3, 0] = np.sin(np.linspace(0, 6, nodes))
        truth[4, 0] = 1000 + np.cos(np.linspace(0, 6, nodes))
        prediction = truth.copy()
        prediction[3, 0] = 0.0  # constant predictors: no skill on either field
        prediction[4, 0] = 1000.0
        self.mesh(self.root / "pred" / "p.h5", {"0": prediction})
        self.mesh(self.root / "truth" / "t.h5", {"0": truth})
        report = run_field_evaluation({
            "prediction_path": str(self.root / "pred"),
            "truth_path": str(self.root / "truth"),
            "confirm_mapping": True,
        })
        aggregate = report["aggregate"]
        # Pooled over both fields the 1000 offset made R2 look near-perfect.
        self.assertLess(aggregate["r2"]["mean"], 0.01)
        self.assertGreater(aggregate["r2_pooled"]["mean"], 0.9)
        self.assertAlmostEqual(aggregate["relative_l2"]["mean"], 0.5, places=2)
        self.assertEqual(len(report["aggregate_by_field"]), 2)
        for field in report["aggregate_by_field"].values():
            self.assertLess(field["r2"]["mean"], 0.01)
        with Path(report["per_sample_csv"]).open(encoding="utf-8") as handle:
            columns = next(csv.reader(handle))
        for key in ("relative_l2", "r2", "mae", "rmse", "max_absolute_error", "relative_l2_pooled", "r2_pooled"):
            self.assertIn(key, columns)
        self.assertEqual(sum(column.startswith("r2[") for column in columns), 2)

    def test_single_sample_files_with_different_ids_are_not_paired(self) -> None:
        import numpy as np

        from studio_backend.analysis import evaluation_schema

        values = np.random.default_rng(1).random((5, 1, 50))
        self.mesh(self.root / "pred" / "rollout.h5", {"42": values})
        self.mesh(self.root / "truth" / "held.h5", {"7": values})
        schema = evaluation_schema({"prediction_path": str(self.root / "pred"), "truth_path": str(self.root / "truth")})
        self.assertFalse(schema["compatible"])
        self.assertEqual(schema["sample_matching"]["strategy"], "single-id-mismatch")
        self.assertTrue(any("sample IDs differ" in error for error in schema["errors"]))

    def test_shared_file_stem_does_not_pair_multi_sample_files(self) -> None:
        import numpy as np

        from studio_backend.analysis import evaluation_schema

        values = np.random.default_rng(2).random((5, 1, 50))
        self.mesh(self.root / "pred" / "ex_infer.h5", {"100": values, "101": values})
        self.mesh(self.root / "truth" / "ex_infer.h5", {"0": values, "1": values, "2": values})
        schema = evaluation_schema({"prediction_path": str(self.root / "pred"), "truth_path": str(self.root / "truth")})
        self.assertFalse(schema["compatible"])
        self.assertEqual(schema["sample_matching"]["overlap_count"], 0)


class PreviewInputTests(unittest.TestCase):
    """Item 9 and suspects e/f: bad selections are clear 400-class errors."""

    def test_missing_sample_id_is_a_value_error(self) -> None:
        try:
            import h5py
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional feature dependency
            raise unittest.SkipTest(str(exc)) from exc
        from studio_backend.hdf5_preview import hdf5_sample

        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as directory:
            path = Path(directory) / "mesh.h5"
            with h5py.File(path, "w") as handle:
                handle.create_group("data/0").create_dataset("nodal_data", data=np.zeros((4, 1, 8)))
            with self.assertRaisesRegex(ValueError, "'99' was not found under data/"):
                hdf5_sample(path, "99", 3, 0)

    def test_safe_repo_path_does_not_decode_twice(self) -> None:
        self.assertEqual(safe_repo_path("dataset/run%201.h5"), (SUITE_ROOT / "dataset" / "run%201.h5").resolve())

    def test_directory_geometry_scan_is_capped(self) -> None:
        from studio_backend.geometry_preview import _geometry_paths

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(6):
                (root / f"part{index}.stl").write_text("solid x\nendsolid x\n", encoding="utf-8")
            files, truncated = _geometry_paths(root)
            self.assertEqual((len(files), truncated), (6, False))
            files, truncated = _geometry_paths(root, scan_limit=3)
            self.assertTrue(truncated)


class HttpGuardTests(unittest.TestCase):
    """Items 10 and 11: status codes for bad files and foreign requests."""

    @classmethod
    def setUpClass(cls) -> None:
        from studio_backend.http_handler import create_server

        cls.server = create_server("127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def call(self, path: str, *, method: str = "GET", data: bytes | None = None, headers: dict | None = None):
        request = Request(self.base + path, data=data, method=method, headers=headers or {})
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read() or b"{}")
        except HTTPError as exc:
            body = exc.read()
            try:
                return exc.code, json.loads(body or b"{}")
            except ValueError:
                return exc.code, {}

    def test_non_hdf5_selection_is_400_not_500(self) -> None:
        for route in ("/api/hdf5/samples", "/api/hdf5/sample"):
            status, payload = self.call(f"{route}?path=CLAUDE.md")
            self.assertEqual(status, 400, route)
            self.assertIn(".h5", payload["error"])
        # h5py reports a non-HDF5 file with the right suffix as OSError.
        with mock.patch(
            "studio_backend.http_handler.hdf5_samples",
            side_effect=OSError("Unable to open file (file signature not found)"),
        ):
            status, payload = self.call("/api/hdf5/samples?path=dataset/not-really.h5")
        self.assertEqual(status, 400)
        self.assertIn("file signature not found", payload["error"])

    def test_foreign_host_is_refused(self) -> None:
        status, _ = self.call("/api/no-such-route", headers={"Host": f"evil.example:{self.port}"})
        self.assertEqual(status, 403)
        status, _ = self.call("/api/no-such-route", headers={"Host": f"localhost:{self.port}"})
        self.assertEqual(status, 404)

    def test_foreign_origin_post_is_refused(self) -> None:
        body = b"{}"
        json_type = {"Content-Type": "application/json"}
        for origin in ("http://evil.example", "null"):
            status, _ = self.call(
                "/api/jobs/nope/cancel", method="POST", data=body, headers={**json_type, "Origin": origin}
            )
            self.assertEqual(status, 403, origin)
        status, _ = self.call(
            "/api/jobs/nope/cancel", method="POST", data=body, headers={**json_type, "Origin": self.base}
        )
        self.assertEqual(status, 404)

    def test_json_routes_require_json_but_uploads_do_not(self) -> None:
        status, _ = self.call(
            "/api/jobs/nope/cancel", method="POST", data=b"{}", headers={"Content-Type": "text/plain"}
        )
        self.assertEqual(status, 415)
        status, payload = self.call(
            "/api/upload?kind=not-a-kind",
            method="POST",
            data=b"raw bytes",
            headers={"Content-Type": "application/octet-stream", "X-Filename": "a.stl"},
        )
        self.assertEqual(status, 400, payload)

    def test_result_figures_and_reports_are_readable(self) -> None:
        """An optimize run's convergence.png and report.md open from the GUI."""
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="figures-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            (root / "convergence.png").write_bytes(png)
            (root / "drawing.svg").write_text("<svg onload='alert(1)'/>", encoding="utf-8")
            (root / "report.md").write_text("# Report\n", encoding="utf-8")
            # The test runtime sits outside the suite, so this is absolute there.
            rel = quote(relative(root))

            with urlopen(f"{self.base}/api/image?path={rel}/convergence.png", timeout=10) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "image/png")
                self.assertEqual(response.read(), png)
            status, payload = self.call(f"/api/image?path={rel}/drawing.svg")
            self.assertEqual(status, 400)
            self.assertIn(".png", payload["error"])
            status, _ = self.call(f"/api/image?path={rel}/missing.png")
            self.assertEqual(status, 400)
            # Result roots only: a figure under the source tree is not served.
            status, _ = self.call("/api/image?path=docs/figure.png")
            self.assertEqual(status, 400)

            status, payload = self.call(f"/api/text?path={rel}/report.md")
            self.assertEqual(status, 200)
            self.assertEqual(payload["text"], "# Report\n")


if __name__ == "__main__":
    unittest.main()
