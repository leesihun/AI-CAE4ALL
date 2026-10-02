from __future__ import annotations

import io
import json
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

from studio_backend.paths import RUNTIME_ROOT, relative
from studio_backend.state import StudioState


def preflight_payload(ok: bool, code: str = "") -> dict:
    diagnostics = [] if ok else [{
        "code": code or "TEST-JIT-001",
        "severity": "error",
        "original_severity": "error",
        "message": "exact saved config rejected",
        "field": "dataset_dir",
        "hint": "fix the saved step",
    }]
    return {
        "ok": ok,
        "strict": False,
        "config_path": "studio/runtime/configs/test.txt",
        "report": {
            "summary": {"errors": 0 if ok else 1, "warnings": 0, "notices": 0},
            "diagnostics": diagnostics,
        },
        "route": {"model": "test"} if ok else None,
        "resolved_paths": {},
        "dataset_metadata": {},
        "checkpoint_metadata": {},
    }


class FakeProcess:
    def __init__(self) -> None:
        self.pid = 4321
        self.stdout = io.BytesIO(b"native process ran\n")

    def wait(self) -> int:
        return 0


class PipelineLaunchGateTests(unittest.TestCase):
    def test_each_launcher_rechecks_exact_file_and_failure_prevents_second_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "method"
            repository.mkdir()
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("model test\nmode train\n", encoding="utf-8")
            second.write_text("model test\nmode inference\n", encoding="utf-8")

            engine = object.__new__(StudioState)
            engine.lock = threading.RLock()
            engine.jobs = {}
            engine.processes = {}
            job = {
                "id": "jit-test",
                "label": "JIT gate test",
                "status": "queued",
                "started_at": None,
                "finished_at": None,
                "returncode": None,
                "pid": None,
                "current_step": 0,
                "total_steps": 2,
                "step_label": None,
                "cancel_requested": False,
                "log_path": root / "run.log",
                "metadata_path": root / "job.json",
                "steps": [{"label": "first"}, {"label": "second"}],
            }
            engine.jobs[job["id"]] = job
            seen: list[tuple[Path, str, dict]] = []

            def gate(self, path: Path, **options):
                seen.append((path, path.read_text(encoding="utf-8"), options))
                if path == second:
                    return preflight_payload(False), None
                result = types.SimpleNamespace(
                    command=["method-python", "entry.py", "--config", str(path)],
                    resolved=types.SimpleNamespace(repository_root=repository),
                )
                return preflight_payload(True), result

            engine._preflight_config_path = types.MethodType(gate, engine)
            engine._record_step_results = types.MethodType(lambda *args, **kwargs: None, engine)
            prepared = [
                {"kind": "launcher", "label": "first", "path": first, "node_id": "n1", "node_type": "model.test"},
                {"kind": "launcher", "label": "second", "path": second, "node_id": "n2", "node_type": "run.inference"},
            ]
            # The gate must read what is on disk at launch time, not submission
            # text cached in memory.
            second.write_text("model test\nmode inference\ndataset_dir missing-now.h5\n", encoding="utf-8")

            with mock.patch("studio_backend.state.subprocess.Popen", return_value=FakeProcess()) as popen:
                engine._run_pipeline(job["id"], prepared, strict=False)

            self.assertEqual([item[0] for item in seen], [first, second])
            self.assertIn("missing-now.h5", seen[1][1])
            for _, _, options in seen:
                self.assertEqual(options, {
                    "strict": False,
                    "skip_filesystem": False,
                    "skip_native": False,
                    "skip_environment": False,
                    "skip_dataset": False,
                })
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(popen.call_args.args[0][0], "method-python")
            self.assertEqual(popen.call_args.kwargs["cwd"], repository)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["returncode"], 2)
            self.assertTrue(job["steps"][0]["launch_preflight"]["ok"])
            self.assertFalse(job["steps"][1]["launch_preflight"]["ok"])
            self.assertEqual(job["diagnostics"][0]["nodeId"], "n2")
            log = job["log_path"].read_text(encoding="utf-8")
            self.assertIn("Launch preflight failed", log)
            self.assertIn("Native process was not started", log)


class CanvasOwnershipTests(unittest.TestCase):
    """A job records which canvas launched it; node ids cannot say so.

    Every canvas made from one template shares its node ids, so a freshly
    loaded copy of the SDFFlow template painted another pipeline's live
    training run onto its own blocks.
    """

    def _create(self, pipeline):
        engine = object.__new__(StudioState)
        engine.lock = threading.RLock()
        engine.jobs = {}
        engine.processes = {}
        action = sorted(StudioState.ANALYSIS_ACTIONS)[0]
        steps = [{"kind": "analysis", "label": "a", "action": action, "payload": {},
                  "node_id": "trainer", "node_type": "model.sdfflow"}]
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch("studio_backend.state.JOB_RUNTIME", Path(directory)), \
                mock.patch("studio_backend.state.threading.Thread"):
            return engine.create_pipeline_job(steps, label="t", strict=False, pipeline=pipeline)

    def test_canvas_id_is_recorded(self) -> None:
        job = self._create({"canvas_id": "abc-123_X", "nodes": [{"id": "trainer"}]})
        self.assertEqual(job["canvas_id"], "abc-123_X")

    def test_missing_or_hostile_canvas_id_is_empty(self) -> None:
        self.assertEqual(self._create(None)["canvas_id"], "")
        self.assertEqual(self._create({"canvas_id": "<script>", "nodes": [{"id": "t"}]})["canvas_id"], "")
        # "$" in re.match also matches before a trailing newline; a number or
        # null is not an id even though str() would make it look like one.
        for hostile in ("abc\n", None, 0, 12345, ["abc"], "", "x" * 65):
            with self.subTest(canvas_id=hostile):
                job = self._create({"canvas_id": hostile, "nodes": [{"id": "t"}]})
                self.assertEqual(job["canvas_id"], "")


class StepReferenceTests(unittest.TestCase):
    """`@results:` names a step's table, `@artifacts:` the directory it wrote."""

    def test_artifacts_prefers_directory_and_falls_back_to_results(self) -> None:
        resolve = StudioState._resolve_step_reference
        produced = {"gen": "output/run/optimize_summary.csv", "inf": "output/run/rollouts"}
        dirs = {"gen": "output/run"}
        self.assertEqual(resolve("@artifacts:gen", produced, dirs), "output/run")
        # No directory recorded: never resolve to less than @results: would.
        self.assertEqual(resolve("@artifacts:inf", produced, dirs), "output/run/rollouts")
        self.assertEqual(resolve("@artifacts:inf", produced), "output/run/rollouts")
        self.assertEqual(resolve("@artifacts:missing", produced, dirs), "")
        # @results: is unchanged by the directory map.
        self.assertEqual(resolve("@results:gen", produced, dirs), "output/run/optimize_summary.csv")
        self.assertEqual(resolve("@results:missing", produced, dirs), "")
        for value in (None, 0, 3.5, ["@results:gen"], "plain/path.csv", "@other:gen"):
            self.assertEqual(resolve(value, produced, dirs), value)


class GeneratorStepResultsTests(unittest.TestCase):
    """A CAD Generator step publishes its table AND its whole output folder."""

    def _engine(self, root: Path):
        engine = object.__new__(StudioState)
        engine.lock = threading.RLock()
        engine.jobs = {}
        engine.processes = {}
        job = {
            "id": "gen-test",
            "log_path": root / "run.log",
            "metadata_path": root / "job.json",
            "steps": [{"label": "generate"}],
        }
        return engine, job

    def _item(self, output: Path) -> dict:
        return {
            "kind": "launcher",
            "node_id": "gen",
            "node_type": "run.cad_generator",
            "preflight": {
                "route": {"repository": "methods/SDFFlow"},
                "resolved_paths": {"output_dir": relative(output)},
            },
        }

    def test_optimize_run_records_table_and_directory(self) -> None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gen-step-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            output = root / "optimize"
            output.mkdir()
            (output / "summary.json").write_text(json.dumps({
                "analysis_backend": "fea",
                "verified": {"optimized": {"mass_kg": 0.5, "peak_von_mises_MPa": 200.0,
                                           "max_displacement_mm": 0.1, "num_tets": 9}},
            }), encoding="utf-8")
            engine, job = self._engine(root)
            produced: dict = {}
            produced_dirs: dict = {}
            with mock.patch("studio_backend.state.outputs_since",
                            side_effect=AssertionError("generator step must not scan")):
                engine._record_step_results(job, 1, self._item(output), 0.0, produced, produced_dirs)

            step = job["steps"][0]
            self.assertEqual(step["results"], relative(output / "optimize_summary.csv"))
            self.assertEqual(step["results_samples"], 1)
            self.assertEqual(step["results_dir"], relative(output))
            self.assertEqual(produced, {"gen": step["results"]})
            self.assertEqual(produced_dirs, {"gen": step["results_dir"]})
            self.assertIn("tabulated 1 design(s)", job["log_path"].read_text(encoding="utf-8"))
            self.assertEqual(json.loads(job["metadata_path"].read_text(encoding="utf-8"))
                             ["steps"][0]["results_dir"], step["results_dir"])

    def test_tableless_run_still_publishes_directory_and_never_scans(self) -> None:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="gen-step-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            output = root / "samples"
            output.mkdir()
            (output / "shape.stl").write_text("solid s\nendsolid s\n", encoding="utf-8")
            engine, job = self._engine(root)
            produced: dict = {}
            produced_dirs: dict = {}
            with mock.patch("studio_backend.state.outputs_since",
                            side_effect=AssertionError("generator step must not scan")):
                engine._record_step_results(job, 1, self._item(output), 0.0, produced, produced_dirs)

            step = job["steps"][0]
            self.assertNotIn("results", step)
            self.assertEqual(step["results_dir"], relative(output))
            self.assertEqual(produced, {})
            self.assertEqual(produced_dirs, {"gen": relative(output)})


class TrainingStepAttributionTests(unittest.TestCase):
    """The since-step-start scan must not pin another run's rollouts to a step.

    A 21 h SDFFlow train step was reported as having written the 480 HI-MGN
    rollouts a concurrent job put into output/, and its output edge carried them.
    """

    UNRELATED = {"path": "output/dataset_matrix/deterministic/ex13/himgn/inference",
                 "samples": 480, "modified": 2.0}
    OWN = {"path": "output/geometry_generation/studio/vae_recon/epoch00499",
           "samples": 2, "modified": 1.0}

    def _record(self, found: list, resolved: dict) -> tuple[dict, dict, str]:
        RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="train-step-", dir=RUNTIME_ROOT) as directory:
            root = Path(directory)
            engine = object.__new__(StudioState)
            engine.lock = threading.RLock()
            engine.jobs = {}
            engine.processes = {}
            job = {"id": "train-test", "log_path": root / "run.log",
                   "metadata_path": root / "job.json", "steps": [{"label": "train"}]}
            item = {"kind": "launcher", "node_id": "trainer", "node_type": "run.cad_trainer",
                    "preflight": {"route": {"repository": "methods/SDFFlow"},
                                  "resolved_paths": resolved}}
            produced: dict = {}
            with mock.patch("studio_backend.state.outputs_since", return_value=found), \
                    mock.patch("studio_backend.state.invalidate_prediction_runs"):
                engine._record_step_results(job, 1, item, 0.0, produced, {})
            log = job["log_path"].read_text(encoding="utf-8") if job["log_path"].exists() else ""
            return job["steps"][0], produced, log

    OWN_PATHS = {"vae_log_file_dir": "output/geometry_generation/studio/vae_train.log",
                 "dataset_dir": "dataset/geometry_generation/ex1_deepjeb.h5"}

    def test_foreign_rollouts_are_not_attributed(self) -> None:
        step, produced, log = self._record([self.UNRELATED], self.OWN_PATHS)
        self.assertNotIn("results", step)
        self.assertEqual(produced, {})
        self.assertIn("not attributed", log)
        self.assertNotIn("wrote 480", log)

    def test_own_dump_wins_over_a_newer_foreign_one(self) -> None:
        step, produced, _ = self._record([self.UNRELATED, self.OWN], self.OWN_PATHS)
        self.assertEqual(step["results"], self.OWN["path"])
        self.assertEqual(produced, {"trainer": self.OWN["path"]})

    def test_input_paths_alone_do_not_narrow_the_scan(self) -> None:
        # dataset/ is an input root: with no output path named there is nothing
        # to scope by, and the newest recent folder is still pinned.
        step, _, _ = self._record([self.UNRELATED],
                                  {"dataset_dir": "dataset/geometry_generation/ex1_deepjeb.h5"})
        self.assertEqual(step["results"], self.UNRELATED["path"])


if __name__ == "__main__":
    unittest.main()
