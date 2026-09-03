from __future__ import annotations

import io
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
