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

    def test_tagged_stage_lines_are_parsed_and_scoped_per_tag(self) -> None:
        # HI_MGNFlow prints `[AE]`/`[Prior]` before `Epoch`; these lines used
        # to match nothing, so a chi-mgnflow run had no Train Metrics at all.
        result = parse_training_log(
            "[AE] Epoch 0/2 LR: 1.00e-04 | Train recon=1.2e-01 kl=3.0e-04 | "
            "Valid recon=1.3e-01 kl=2.0e-04 | VRAM peak=0.11GB reserved=0.12GB\n"
            "[AE] Epoch 1/2 LR: 5.00e-05 | Train recon=1.0e-01 kl=2.5e-04 | "
            "VRAM peak=0.11GB reserved=0.12GB\n"
            "[Prior] Epoch 0/2 LR: 1.00e-04 | Train fm=6.45e+00 | Valid fm=3.65e+00\n"
            "[Prior] Epoch 1/2 LR: 5.00e-05 | Train fm=5.00e+00\n"
        )
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual([point["x"] for point in by_key["ae__lr"]["points"]], [0, 1])
        self.assertEqual([point["x"] for point in by_key["prior__lr"]["points"]], [0, 1])
        self.assertEqual(by_key["prior__lr"]["label"], "Prior · LR")
        # `kl` follows `Train recon=` and `Valid recon=`: two splits, two series,
        # and the train one keeps its key on epochs that skip validation.
        self.assertEqual([point["y"] for point in by_key["ae__train_kl"]["points"]], [3.0e-04, 2.5e-04])
        self.assertEqual([point["y"] for point in by_key["ae__valid_kl"]["points"]], [2.0e-04])
        self.assertNotIn("ae__kl", by_key)
        self.assertIn("ae__reserved", by_key)

    def test_one_process_two_stage_pipeline_splits_at_the_stage_header(self) -> None:
        # SDFFlow `mode train` runs VAE then FM in one process; both print `LR`.
        result = parse_training_log(
            "\n[Pipeline 1/2] Training VAE (no checkpoint)\n"
            "Epoch 0/2 TrainSDF: 2.5e-03 KL: 2.0e+03 LR: 1e-04\n"
            "Epoch 1/2 TrainSDF: 2.0e-03 KL: 1.9e+03 LR: 5e-05\n"
            "\n[Pipeline 2/2] Training FM (no checkpoint)\n"
            "Epoch 0/2 TrainFM: 1.0e+00 ValidFM: 1.1e+00 LR: 1e-04\n"
            "Epoch 1/2 TrainFM: 9.0e-01 LR: 5e-05\n"
        )
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(
            list(by_key), ["vae__trainsdf", "vae__kl", "vae__lr", "fm__trainfm", "fm__validfm", "fm__lr"]
        )
        self.assertEqual([point["x"] for point in by_key["vae__lr"]["points"]], [0, 1])
        self.assertEqual([point["x"] for point in by_key["fm__lr"]["points"]], [0, 1])
        self.assertEqual(by_key["fm__lr"]["label"], "FM · LR")

    def test_reused_first_stage_names_only_the_stage_that_trains(self) -> None:
        result = parse_training_log(
            "[Pipeline 1/2] Reusing VAE: checkpoint is current\n"
            "[Pipeline 2/2] Training LC (no checkpoint)\n"
            "Epoch 0/1 LC train 5.2e+00 val 5.7e+00 LR 8e-04\n"
        )
        self.assertEqual([item["key"] for item in result["metrics"]], ["lc__lc_train", "lc__val", "lc__lr"])
        # The stage name is not repeated inside the metric label.
        self.assertEqual(result["metrics"][0]["label"], "LC · train")

    def test_nonfinite_values_end_labels_and_are_reported_as_divergence(self) -> None:
        # `f"{nan:.2e}"` prints `nan`. It used to stay in the label text, so the
        # next finite value was filed under a one-point series `TrainOpt: nan Valid`.
        result = parse_training_log(
            "Epoch 0/100 TrainOpt: 1.0e-01 Valid: 2.0e-01 LR: 1.00e-04\n"
            "Epoch 1/100 TrainOpt: 5.0e-02 Valid: 1.0e-01 LR: 1.00e-04\n"
            "Epoch 2/100 TrainOpt: inf Valid: 3.10e+02 LR: 1.00e-04\n"
            "Epoch 3/100 TrainOpt: nan Valid: nan LR: 1.00e-04\n"
            "Epoch 4/100 TrainOpt: nan Valid: nan LR: 1.00e-04\n"
        )
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(list(by_key), ["trainopt", "valid", "lr"])
        trainopt = by_key["trainopt"]
        self.assertEqual([point["x"] for point in trainopt["points"]], [0, 1])
        self.assertEqual(trainopt["nonfinite_count"], 3)
        self.assertEqual(trainopt["nonfinite_ranges"], [[2, 4]])
        self.assertTrue(trainopt["diverged"])
        self.assertEqual(trainopt["diverged_at"], 2)
        self.assertEqual(trainopt["last_nonfinite"], "nan")
        self.assertAlmostEqual(trainopt["last"], 5.0e-02)
        valid = by_key["valid"]
        self.assertEqual([point["x"] for point in valid["points"]], [0, 1, 2])
        self.assertEqual(valid["diverged_at"], 3)
        self.assertFalse(by_key["lr"]["diverged"])
        self.assertEqual(by_key["lr"]["nonfinite_ranges"], [])
        self.assertEqual(result["point_count"], 2 + 3 + 5)

    def test_a_recovered_spike_is_not_divergence(self) -> None:
        # fp16 overflow makes one epoch mean inf; GradScaler skips the step and
        # the next epoch is finite again. (A `train_ae` run: its keys stay plain.)
        result = parse_training_log(
            "[AE] Epoch 1/5 LR: 1.00e-04 | Train recon=4.1e-02 kl=4.40e+00 | VRAM peak=1.28GB reserved=1.63GB\n"
            "[AE] Epoch 2/5 LR: 1.00e-04 | Train recon=inf kl=4.30e+00 | Valid recon=nan kl=4.20e+00 | "
            "VRAM peak=1.28GB reserved=1.63GB\n"
            "[AE] Epoch 3/5 LR: 1.00e-04 | Train recon=3.0e-02 kl=4.20e+00 | VRAM peak=1.28GB reserved=1.63GB\n",
            stage_hints={1: "ae"},
        )
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(
            list(by_key), ["lr", "train_recon", "train_kl", "vram_peak", "reserved", "valid_recon", "valid_kl"]
        )
        self.assertEqual([point["x"] for point in by_key["train_kl"]["points"]], [1, 2, 3])
        self.assertEqual([point["x"] for point in by_key["train_recon"]["points"]], [1, 3])
        self.assertEqual(by_key["train_recon"]["nonfinite_ranges"], [[2, 2]])
        self.assertFalse(by_key["train_recon"]["diverged"])
        self.assertIsNone(by_key["train_recon"]["last_nonfinite"])
        # A series that never had a finite value is still listed, with no min/max/last.
        valid_recon = by_key["valid_recon"]
        self.assertEqual(valid_recon["count"], 0)
        self.assertIsNone(valid_recon["last"])
        self.assertTrue(valid_recon["diverged"])

    def test_nonfinite_tokens_do_not_split_words(self) -> None:
        result = parse_training_log(
            "Epoch 5/10 nan_count: 3 inference_time: 1.5e+00 info_gain: 2.0e-01 loss: -inf\n"
        )
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(list(by_key), ["nan_count", "inference_time", "info_gain", "loss"])
        self.assertEqual(by_key["nan_count"]["last"], 3.0)
        self.assertEqual(by_key["loss"]["last_nonfinite"], "-inf")

    def test_transolver_distributed_order_keeps_valid_after_lr(self) -> None:
        result = parse_training_log("Epoch 5/200 TrainOpt: nan LR: 5.00e-05 Valid: nan\n")
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(list(by_key), ["trainopt", "lr", "valid"])
        self.assertEqual(by_key["lr"]["last"], 5.0e-05)
        self.assertTrue(by_key["valid"]["diverged"])

    def test_epoch_line_behind_a_glued_progress_frame_is_parsed(self) -> None:
        # Under DDP a rank>0 tqdm frame can land in front of rank 0's epoch print.
        result = parse_training_log(
            "Epoch 0/3 TrainOpt: 1.5e-03 Valid: 2.5e-03 LR: 1.0e-04\n"
            "  0%|          | 0/40 [00:00<?, ?it/s]Epoch 1/3 TrainOpt: 1.2e-03 Valid: 2.3e-03 LR: 1.0e-04\n"
        )
        by_key = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(list(by_key), ["trainopt", "valid", "lr"])
        self.assertEqual([point["x"] for point in by_key["valid"]["points"]], [0, 1])
        tagged = parse_training_log(
            " 45%|####5     | 18/40 [00:02<00:03, 7.1it/s, loss=0.12]"
            "[AE] Epoch 2/3 LR: 1.0e-04 | Train recon=1.0e-01\n"
        )
        self.assertEqual([item["key"] for item in tagged["metrics"]], ["ae__lr", "ae__train_recon"])
        # A progress frame with no epoch line behind it is still not a metric.
        self.assertEqual(
            parse_training_log(" 45%|####5     | 18/40 [00:02<00:03, 7.1it/s, loss=0.12]\n")["metrics"], []
        )

    def test_compare_keys_do_not_depend_on_the_job_shape(self) -> None:
        # SDFFlow: `mode train` (one process, two stages) vs `train_vae` / `train_fm` alone.
        pipeline = parse_training_log(
            "[Pipeline 1/2] Training VAE (no checkpoint)\n"
            "Epoch 0/2 TrainSDF: 2.5e-03 KL: 2.0e+03 LR: 1e-04\n"
            "[Pipeline 2/2] Training FM (no checkpoint)\n"
            "Epoch 0/2 TrainFM: 1.0e+00 LR: 1e-04\n",
            training_steps={1},
            declared_total_steps=1,
        )
        vae_alone = parse_training_log(
            "Epoch 0/2 TrainSDF: 2.4e-03 KL: 2.1e+03 LR: 1e-04\n",
            training_steps={1},
            declared_total_steps=1,
            stage_hints={1: "vae"},
        )
        fm_alone = parse_training_log(
            "Epoch 0/2 TrainFM: 9.0e-01 LR: 1e-04\n",
            training_steps={1},
            declared_total_steps=1,
            stage_hints={1: "fm"},
        )
        compare = {item["key"]: item["compare_key"] for item in pipeline["metrics"]}
        self.assertEqual(compare["vae__trainsdf"], "vae__trainsdf")
        self.assertEqual(compare["vae__lr"], "vae__lr")
        self.assertEqual(compare["fm__lr"], "fm__lr")
        self.assertEqual({item["compare_key"] for item in vae_alone["metrics"]}, {"vae__trainsdf", "vae__kl", "vae__lr"})
        self.assertEqual({item["compare_key"] for item in fm_alone["metrics"]}, {"fm__trainfm", "fm__lr"})
        # The plain keys of a single-stage run are unchanged.
        self.assertEqual([item["key"] for item in vae_alone["metrics"]], ["trainsdf", "kl", "lr"])

        # HI_MGNFlow: `[AE]`/`[Prior]` in `mode train` vs `train_ae` alone.
        both = parse_training_log(
            "[AE] Epoch 0/1 LR: 1.0e-04 | Train recon=1.0e-01\n[Prior] Epoch 0/1 LR: 1.0e-04 | Train fm=6.0e+00\n"
        )
        ae_alone = parse_training_log(
            "[AE] Epoch 0/1 LR: 1.0e-04 | Train recon=1.0e-01\n", stage_hints={1: "ae"}
        )
        self.assertEqual(
            {item["compare_key"] for item in both["metrics"] if item["key"].startswith("ae__")},
            {item["compare_key"] for item in ae_alone["metrics"]},
        )

    def test_a_trailing_inference_step_does_not_scope_the_training_keys(self) -> None:
        log = "[studio] Step 1/2: train\nEpoch 0/1 loss: 0.5\n[studio] Step 2/2: inference\n"
        train_then_infer = parse_training_log(log, training_steps={1}, declared_total_steps=2)
        train_only = parse_training_log("[studio] Step 1/1: train\nEpoch 0/1 loss: 0.4\n", training_steps={1})
        self.assertEqual([item["key"] for item in train_then_infer["metrics"]], ["loss"])
        self.assertEqual(
            [item["compare_key"] for item in train_then_infer["metrics"]],
            [item["compare_key"] for item in train_only["metrics"]],
        )

    def test_two_training_steps_keep_distinct_keys(self) -> None:
        result = parse_training_log(
            "[studio] Step 1/2: MLP train\nEpoch 0/1 loss: 0.5\n"
            "[studio] Step 2/2: MLP train\nEpoch 0/1 loss: 0.4\n",
            training_steps={1, 2},
            declared_total_steps=2,
        )
        metrics = result["metrics"]
        self.assertEqual([item["key"] for item in metrics], ["step_1__loss", "step_2__loss"])
        # Two series for one comparable metric: Compare falls back to the full key.
        self.assertEqual([item["compare_key"] for item in metrics], ["step_1__loss", "step_2__loss"])

    def test_epoch_anchored_periodic_test_lines_become_series(self) -> None:
        # The mesh trainers print the test_interval pass as its own
        # `Epoch N Test loss: ... (12.3s)` line after the epoch line.
        result = parse_training_log(
            "Epoch 0/3 TrainOpt: 1.00e-01 Valid: 2.00e-01 LR: 1.00e-03\n"
            "  Epoch 0 Test loss: 3.21e-03 (12.3s)\n"
            "  Epoch 0 Train reconstruction loss: 2.20e-03\n"
            "Epoch 1/3 TrainOpt: 5.00e-02 Valid: 1.00e-01 LR: 9.00e-04\n"
            "Epoch 2/3 TrainOpt: 4.00e-02 Valid: 9.00e-02 LR: 8.00e-04\n"
            "  Epoch 2 Test loss: 2.00e-03 (11.0s)\n"
        )
        metrics = {item["key"]: item for item in result["metrics"]}
        self.assertEqual(
            list(metrics), ["trainopt", "valid", "lr", "test_loss", "train_reconstruction_loss"]
        )
        self.assertEqual([(p["x"], p["y"]) for p in metrics["test_loss"]["points"]], [(0, 3.21e-03), (2, 2.00e-03)])
        # The same-epoch line does not read as a counter restart.
        self.assertEqual([p["x"] for p in metrics["trainopt"]["points"]], [0, 1, 2])

    def test_hi_mgn_stage_tagged_test_lines_and_det_readout(self) -> None:
        result = parse_training_log(
            "[AE] Epoch 0/1 LR: 1.00e-03 | Train recon=1.00e-01 kl=2.00e+01 | Valid recon=2.00e-01 kl=3.00e+01\n"
            "  [AE] Epoch 0 Test loss: 5.36e-01 (108.4s)\n"
            "[Prior] Epoch 0/1 LR: 1.00e-03 | Train fm=1.00e+00 | Valid fm=9.00e-01"
            " | CRPS 2.62e-01 spread 0.355 det mse 7.40e-01\n"
            "  [Prior] Epoch 0 Test loss: 4.00e-01 (50.0s)\n"
        )
        values = {item["key"]: item["last"] for item in result["metrics"]}
        self.assertEqual(values["ae__test_loss"], 5.36e-01)
        self.assertEqual(values["prior__test_loss"], 4.00e-01)
        self.assertEqual(values["prior__det_mse"], 7.40e-01)
        self.assertEqual(values["prior__spread"], 0.355)

    def test_epoch_counter_restart_without_header_starts_a_new_stage(self) -> None:
        result = parse_training_log(
            "Epoch 0/3 loss 1.0\nEpoch 1/3 loss 0.5\nEpoch 2/3 loss 0.4\n"
            "Epoch 0/2 loss 3.0\nEpoch 1/2 loss 2.0\n"
        )
        series = {item["label"]: [point["x"] for point in item["points"]] for item in result["metrics"]}
        self.assertEqual(series, {"Stage 1 · loss": [0, 1, 2], "Stage 2 · loss": [0, 1]})

    def test_a_stage_tag_scopes_its_keys_from_the_first_line(self) -> None:
        # HI-MGN `mode train` prints `[AE]` for the whole AE stage before the
        # first `[Prior]` line; the AE keys must be the same before and after.
        ae_line = "[AE] Epoch 0/1 LR: 1.0e-04 | Train recon=1.0e-01\n"
        ae_only = parse_training_log(ae_line)
        both = parse_training_log(ae_line + "[Prior] Epoch 0/1 LR: 1.0e-04 | Train fm=6.0e+00\n")
        self.assertEqual([item["key"] for item in ae_only["metrics"]], ["ae__lr", "ae__train_recon"])
        self.assertEqual(
            [(item["key"], item["compare_key"], item["label"]) for item in ae_only["metrics"]],
            [
                (item["key"], item["compare_key"], item["label"])
                for item in both["metrics"]
                if item["key"].startswith("ae__")
            ],
        )
        # `train_ae` names the stage already: the tag stays out of the key and
        # enters only the compare key, which matches the `mode train` one.
        ae_alone = parse_training_log(ae_line, stage_hints={1: "ae"})
        self.assertEqual([item["key"] for item in ae_alone["metrics"]], ["lr", "train_recon"])
        self.assertEqual(
            [item["compare_key"] for item in ae_alone["metrics"]],
            [item["compare_key"] for item in ae_only["metrics"]],
        )

    def test_a_single_constant_tag_does_not_rename_the_series(self) -> None:
        result = parse_training_log(
            "[model_split] epoch 0/2 train_loss=1.0e-01 lr=1.00e-04 elapsed=1.2s\n"
            "[model_split] epoch 1/2 train_loss=5.0e-02 lr=5.00e-05 elapsed=2.4s\n"
        )
        self.assertEqual([item["key"] for item in result["metrics"]], ["train_loss", "lr", "elapsed"])

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
        # One training step: the trailing inference step does not scope the key,
        # so this run lines up with a train-only run of the same config.
        self.assertEqual([metric["key"] for metric in item["metrics"]], ["loss"])


if __name__ == "__main__":
    unittest.main()
