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
        "fno", "gino", "deeponet", "point_deeponet", "mlp",
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
        self.assertEqual(len(status["models"]), 8)
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
            "gino", "deeponet", "point_deeponet",
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


if __name__ == "__main__":
    unittest.main()
