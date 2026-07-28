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
            "log_path": "frontend/runtime/jobs/run-1/run.log",
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


if __name__ == "__main__":
    unittest.main()
