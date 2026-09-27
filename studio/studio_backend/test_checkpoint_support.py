from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_backend.suite_bridge import (
    PORTABLE_INFERENCE_MODELS,
    STANDALONE_INFERENCE_MODELS,
    _model_from_architecture,
    _model_from_path,
    checkpoint_metadata,
)
from studio_backend.native_jobs import create_inference_job
from studio_backend.paths import RUNTIME_ROOT, SUITE_ROOT
from studio_backend.system_info import deployment_status


class _Registry:
    model_ids = (
        "meshgraphnets", "meshgraphnets-v", "chi-mgnflow", "transolver",
        "fno", "deeponet", "point_deeponet", "mlp",
        "simulgenvae", "sdfflow",
    )


class CheckpointSupportContractTests(unittest.TestCase):
    def test_chi_flow_keys_win_over_the_shared_mgn_backbone(self) -> None:
        architecture = {
            "message_passing_num": 15,
            "edge_var": 8,
            "flow_time_freqs": 16,
            "flow_solver": "heun",
        }
        self.assertEqual(_model_from_architecture(architecture), "chi-mgnflow")
        self.assertEqual(_model_from_path(Path("output/chi_mgnflow/best.pth")), "chi-mgnflow")

    def test_old_chi_checkpoint_is_native_standalone_but_not_portable(self) -> None:
        probe = {
            "ok": True,
            "model_config": {
                "message_passing_num": 15,
                "edge_var": 8,
                "flow_time_freqs": 16,
                "flow_solver": "heun",
            },
            "data_config": {},
        }
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(probe) + "\n", stderr="")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy-flow.pth"
            checkpoint.touch()
            with patch("subprocess.run", return_value=completed):
                facts = checkpoint_metadata(checkpoint, _Registry(), None)
        self.assertTrue(facts["ok"])
        self.assertEqual(facts["model"], "chi-mgnflow")
        self.assertTrue(facts["standalone_inference"])
        self.assertFalse(facts["portable_inference"])
        self.assertIn("chi-mgnflow", STANDALONE_INFERENCE_MODELS)
        self.assertNotIn("chi-mgnflow", PORTABLE_INFERENCE_MODELS)

    def test_deployment_inventory_distinguishes_models_from_drivers(self) -> None:
        status = deployment_status()
        self.assertEqual(status["models"], list(PORTABLE_INFERENCE_MODELS))
        self.assertEqual(len(status["models"]), 7)
        self.assertEqual(len(status["driver_families"]), 5)

    def test_sdfflow_schema_is_reported_as_the_model_source(self) -> None:
        probe = {
            "ok": True,
            "schema_version": "sdfflow_infer_v1",
            "model_config": {},
            "data_config": {},
        }
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps(probe) + "\n", stderr="")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "merged.pth"
            checkpoint.touch()
            with patch("subprocess.run", return_value=completed):
                facts = checkpoint_metadata(checkpoint, _Registry(), None)
        self.assertEqual(facts["model"], "sdfflow")
        self.assertEqual(facts["model_source"], "checkpoint schema")
        self.assertTrue(facts["portable_inference"])

    def test_portable_job_rejects_chi_before_creating_a_process(self) -> None:
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as directory:
            checkpoint = Path(directory) / "flow.pth"
            checkpoint.touch()
            with patch(
                "studio_backend.suite_bridge.checkpoint_metadata",
                return_value={
                    "ok": True,
                    "model": "chi-mgnflow",
                    "portable_inference": False,
                },
            ):
                with self.assertRaisesRegex(ValueError, "native Inference block"):
                    create_inference_job({"checkpoint": str(checkpoint)})

    def test_non_geometry_portable_jobs_require_input_before_creating_a_process(self) -> None:
        models = (
            "meshgraphnets", "meshgraphnets-v", "transolver", "fno",
            "deeponet", "point_deeponet",
        )
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as directory:
            checkpoint = Path(directory) / "portable.pth"
            checkpoint.touch()
            for model in models:
                with self.subTest(model=model), patch(
                    "studio_backend.suite_bridge.checkpoint_metadata",
                    return_value={"ok": True, "model": model, "portable_inference": True},
                ), patch("studio_backend.native_jobs.STATE.create_command_job") as create_job:
                    with self.assertRaisesRegex(ValueError, "requires an input HDF5"):
                        create_inference_job({"checkpoint": str(checkpoint)})
                    create_job.assert_not_called()

    def test_sdfflow_portable_job_allows_no_input(self) -> None:
        with tempfile.TemporaryDirectory(dir=RUNTIME_ROOT) as directory:
            checkpoint = Path(directory) / "sdfflow.pth"
            checkpoint.touch()
            with patch(
                "studio_backend.suite_bridge.checkpoint_metadata",
                return_value={"ok": True, "model": "sdfflow", "portable_inference": True},
            ), patch(
                "studio_backend.native_jobs.STATE.create_command_job",
                return_value={"id": "sdfflow-job"},
            ) as create_job:
                result = create_inference_job({"checkpoint": str(checkpoint)})
            self.assertEqual(result, {"id": "sdfflow-job"})
            command = create_job.call_args.kwargs["command"]
            self.assertNotIn("--input", command)

    def test_portable_classifier_rejects_shared_signatures_instead_of_misrouting(self) -> None:
        inference_root = str(SUITE_ROOT / "inference")
        if inference_root not in sys.path:
            sys.path.insert(0, inference_root)
        from cae_infer import detect_family

        with patch("cae_infer.torch.load", return_value={
            "stage": "vae",
            "config": {"model": "simulgenvae", "num_filter_enc": "16, 32"},
        }):
            with self.assertRaisesRegex(ValueError, "Could not classify"):
                detect_family("simulgen.pth")
        with patch("cae_infer.torch.load", return_value={
            "model_config": {
                "message_passing_num": 15,
                "flow_time_freqs": 16,
            },
        }):
            with self.assertRaisesRegex(ValueError, "cHI-MGNflow"):
                detect_family("flow.pth")
        with patch("cae_infer.torch.load", return_value={
            "stage": "fm",
            "config": {"model": "sdfflow"},
        }):
            self.assertEqual(detect_family("sdfflow.pth"), "geometry")


def _has_torch_and_numpy() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy"))


class CheckpointProbeTorchVersionTests(unittest.TestCase):
    """The probe must read a real checkpoint on torch 2.4, which lacks `safe_globals`.

    The aarl Studio venv runs torch 2.4.1. Its weights_only unpickler refuses
    `_codecs.encode` and, whatever the allowlist, the BUILD step that rebuilds
    a numpy array, so every checkpoint check there reported CHECKPOINT-PROBE-002
    "safe metadata inspection was unavailable" -- registering the allowlist
    with `add_safe_globals` did not change that.
    """

    @staticmethod
    def _probe():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "checkpoint_probe_under_test", SUITE_ROOT / "cae_suite" / "checkpoint_probe.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _torch(serialization, load=None):
        import types

        calls = []

        def record(path, **kwargs):
            calls.append(kwargs)
            return load(path) if load else {"stage": "fm"}

        torch = types.SimpleNamespace(serialization=serialization, load=record,
                                      __version__="2.4.1")
        return torch, calls

    @staticmethod
    def _save_suite_like_checkpoint(path: Path) -> None:
        """What the trainers write: tensors, numpy normalization, a TorchVersion."""
        import numpy as np
        import torch

        torch.save({
            "stage": "fm",
            "epoch": 499,
            "torch_version": torch.__version__,
            "model_config": {"model": "sdfflow", "latent_dim": 64, "hidden": [256, 256]},
            "model_state_dict": {"w": torch.ones(3, 2), "p": torch.nn.Parameter(torch.zeros(4))},
            "normalization": {
                "mean": np.arange(3, dtype=np.float32),
                "std": np.float64(2.5),
                "count": np.int64(7),
            },
        }, path)

    def test_torch_24_falls_back_to_the_metadata_reader(self) -> None:
        import pickle
        import types

        if not _has_torch_and_numpy():
            self.skipTest("torch and numpy are needed to write a checkpoint")
        import numpy as np

        def refuse(_path):
            raise pickle.UnpicklingError("Weights only load failed ... numpy.dtype")

        torch, calls = self._torch(types.SimpleNamespace(add_safe_globals=lambda _a: None),
                                   load=refuse)
        probe = self._probe()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fm.pth"
            self._save_suite_like_checkpoint(checkpoint)
            loaded = probe._load(torch, checkpoint)
        # torch's own weights_only loader always goes first.
        self.assertEqual(calls, [{"map_location": "cpu", "weights_only": True}])
        self.assertEqual(loaded["stage"], "fm")
        self.assertEqual(loaded["epoch"], 499)
        self.assertEqual(loaded["model_config"]["hidden"], [256, 256])
        np.testing.assert_array_equal(loaded["normalization"]["mean"], [0.0, 1.0, 2.0])
        self.assertEqual(float(loaded["normalization"]["std"]), 2.5)
        self.assertIsInstance(loaded["torch_version"], str)
        # Tensor bytes are never read: every tensor is the same placeholder.
        self.assertIs(loaded["model_state_dict"]["w"], probe._TENSOR)
        self.assertIs(loaded["model_state_dict"]["p"], probe._TENSOR)

    def test_metadata_reader_refuses_a_code_carrying_pickle(self) -> None:
        import os
        import pickle

        if not _has_torch_and_numpy():
            self.skipTest("torch is needed to write a checkpoint")
        import torch

        probe = self._probe()
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pwned"

            class Payload:
                def __reduce__(self):
                    return os.makedirs, (str(marker),)

            checkpoint = Path(directory) / "evil.pth"
            torch.save({"stage": "fm", "normalization": Payload()}, checkpoint)
            with self.assertRaisesRegex(pickle.UnpicklingError, "refuses global"):
                probe._load_metadata_only(checkpoint)
            self.assertFalse(marker.exists())

    def test_the_installed_torch_reads_a_suite_checkpoint(self) -> None:
        """End to end on whatever torch this venv has: 2.4 takes the fallback,
        2.5+ the scoped allowlist; either way the probe reports the metadata."""
        import contextlib
        import io

        if not _has_torch_and_numpy():
            self.skipTest("torch and numpy are needed to write a checkpoint")
        probe = self._probe()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fm.pth"
            self._save_suite_like_checkpoint(checkpoint)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = probe.main([str(checkpoint)])
        report = json.loads(stdout.getvalue())
        self.assertEqual(code, 0, report)
        self.assertTrue(report["ok"])
        self.assertTrue(report["has_normalization"])
        self.assertEqual(report["stage"], "fm")
        self.assertEqual(report["epoch"], 499)
        self.assertEqual(report["model_config"],
                         {"model": "sdfflow", "latent_dim": 64, "hidden": "256, 256"})

    def test_torch_25_uses_the_scoped_context_manager(self) -> None:
        import contextlib
        import types

        scoped = []

        @contextlib.contextmanager
        def safe_globals(allowed):
            scoped.append(list(allowed))
            yield

        def refuse(_allowed):
            raise AssertionError("global allowlist used where a scoped one exists")

        torch, calls = self._torch(types.SimpleNamespace(safe_globals=safe_globals,
                                                         add_safe_globals=refuse))
        probe = self._probe()
        with patch.object(probe, "_inert_safe_globals", return_value=["ndarray"]):
            probe._load(torch, Path("x.pth"))
        self.assertEqual(scoped, [["ndarray"]])
        self.assertEqual(calls, [{"map_location": "cpu", "weights_only": True}])


if __name__ == "__main__":
    unittest.main()
