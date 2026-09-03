from __future__ import annotations

import unittest

from studio_backend.training_metrics import parse_training_log, training_metrics_catalog


class TrainingMetricParserTests(unittest.TestCase):
    def test_colon_separated_vae_metrics(self) -> None:
        result = parse_training_log(
            "Epoch 0/2 Recon: 5.6643e-01 KL: 4.7627e-01 Beta: 1e-04 ValRecon: 5.7278e-01 LR: 1e-03\n"
            "Epoch 1/2 Recon: 4.0e-01 KL: 3.0e-01 Beta: 2e-04 ValRecon: 4.5e-01 LR: 9e-04\n"
        )
        self.assertEqual([item["key"] for item in result["metrics"]], ["recon", "kl", "beta", "valrecon", "lr"])
        self.assertEqual(result["metric_count"], 5)
        self.assertEqual(result["point_count"], 10)
        self.assertAlmostEqual(result["metrics"][0]["last"], 0.4)

    def test_space_separated_and_multistep_metrics(self) -> None:
        result = parse_training_log(
            "[studio] Step 1/2: VAE train\n"
            "Epoch 0/1 Recon 0.5 LR 1e-3\n"
            "[studio] Step 2/2: LC train\n"
            "Epoch 0/1 LC train 5.2 val 5.7 LR 8e-4\n"
        )
        self.assertEqual(
            [item["key"] for item in result["metrics"]],
            ["step_1__recon", "step_1__lr", "step_2__lc_train", "step_2__val", "step_2__lr"],
        )
        self.assertEqual(result["metrics"][2]["label"], "Step 2 · LC train")

    def test_parser_anchors_events_and_does_not_carry_units_into_labels(self) -> None:
        result = parse_training_log(
            "Epoch 0/1 LR: 1.00e-04 | Train fm=6.45e+00 | Valid fm=3.65e+00 | "
            "CRPS 1.55e+00 spread 0.000 | VRAM peak=0.11GB reserved=0.12GB\n"
            "  -> Model saved at epoch 0: new best recon=3.65e+00\n"
            "Training finished. Kept checkpoint: epoch 0 (best by recon), "
            "validation loss 3.65e+00\n"
        )
        keys = {item["key"] for item in result["metrics"]}
        self.assertIn("vram_peak", keys)
        self.assertIn("reserved", keys)
        self.assertNotIn("gb_reserved", keys)
        self.assertNotIn("new_best_recon", keys)
        self.assertNotIn("by_recon_validation_loss", keys)

    def test_inference_only_rollout_is_not_a_training_metrics_job(self) -> None:
        job = {
            "id": "infer-only",
            "status": "completed",
            "log": "[studio] Step 1/1: MeshGraphNets · inference\n"
                   "  Step 1/19 | time: 0.143s | disp range: [0.1, 0.2]\n",
            "steps": [{
                "node_id": "infer",
                "node_type": "run.inference",
                "route": {"model": "meshgraphnets", "mode": "inference"},
            }],
        }

        class FakeState:
            def list_jobs(self):
                return [{"id": job["id"]}]

            def get_job(self, _job_id):
                return job

        self.assertEqual(training_metrics_catalog(FakeState()), {
            "items": [], "count": 0, "source": "Studio job logs",
        })

    def test_catalog_preserves_node_and_route_lineage(self) -> None:
        job = {
            "id": "run-1",
            "label": "MLP training",
            "status": "completed",
            "created_at": "2026-07-27T00:00:00Z",
            "finished_at": "2026-07-27T00:01:00Z",
            "current_step": 1,
            "total_steps": 1,
            "target_node_id": "mlp",
            "log_path": "studio/runtime/jobs/run-1/run.log",
            "log": "Epoch 0/2 loss: 1.0 val_loss: 1.2\nEpoch 1/2 loss: 0.5 val_loss: 0.7\n",
            "steps": [{
                "label": "Simple MLP · train",
                "node_id": "mlp",
                "node_type": "model.mlp",
                "route": {"model": "mlp", "mode": "train"},
            }],
        }

        class FakeState:
            def list_jobs(self):
                return [{"id": "run-1"}]

            def get_job(self, job_id):
                self.last_job_id = job_id
                return job

        result = training_metrics_catalog(FakeState(), node_id="mlp", model_id="mlp")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["node_ids"], ["mlp"])
        self.assertEqual(result["items"][0]["lineage"][0]["node_type"], "model.mlp")
        self.assertEqual(result["items"][0]["target_node_id"], "mlp")

    def test_catalog_status_is_the_multistep_pipeline_job_status(self) -> None:
        job = {
            "id": "run-multistep",
            "label": "Train then infer",
            "status": "failed",
            "current_step": 2,
            "total_steps": 2,
            "log": "[studio] Step 1/2: train\nEpoch 0/1 loss: 0.5\n"
                   "[studio] Step 2/2: inference\n"
                   "Step 1/19 | time: 0.143s | disp range: [0.1, 0.2]\n",
            "steps": [
                {
                    "label": "MLP train",
                    "node_id": "trainer",
                    "node_type": "model.mlp",
                    "route": {"model": "mlp", "mode": "train"},
                },
                {
                    "label": "MLP inference",
                    "node_id": "infer",
                    "node_type": "run.inference",
                    "route": {"model": "mlp", "mode": "inference"},
                },
            ],
        }

        class FakeState:
            def list_jobs(self):
                return [{"id": job["id"]}]

            def get_job(self, _job_id):
                return job

        item = training_metrics_catalog(FakeState())["items"][0]
        self.assertEqual(item["status"], "failed")
        self.assertEqual((item["current_step"], item["total_steps"]), (2, 2))
        self.assertEqual([step["mode"] for step in item["lineage"]], ["train", "inference"])
        self.assertEqual([step["mode"] for step in item["training_lineage"]], ["train"])
        self.assertEqual([metric["key"] for metric in item["metrics"]], ["step_1__loss"])


if __name__ == "__main__":
    unittest.main()
