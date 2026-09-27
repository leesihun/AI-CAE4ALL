from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_backend.suite_bridge import SUITE_ROOT, benchmark_roster


class BenchmarkRosterTests(unittest.TestCase):
    def test_reads_campaign_manifest_and_keeps_missing_configs_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "configs" / "campaigns" / "dataset_matrix" / "manifest.json"
            existing = root / "configs" / "Method" / "ex4" / "config_train.txt"
            manifest.parent.mkdir(parents=True)
            existing.parent.mkdir(parents=True)
            existing.write_text("model demo\nmode train\n", encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "category": "deterministic",
                                "example": "ex4",
                                "method": "demo",
                                "train": "configs/Method/ex4/config_train.txt",
                                "infer": "configs/Method/ex4/config_infer.txt",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch("studio_backend.suite_bridge.SUITE_ROOT", root):
                result = benchmark_roster()

        self.assertEqual([item["role"] for item in result["items"]], ["train", "infer"])
        self.assertEqual(result["items"][0]["label"], "ex4/demo · train")
        self.assertEqual(result["items"][0]["ex_slot"], "ex4")
        self.assertEqual(result["items"][0]["category"], "deterministic")
        self.assertEqual(result["items"][0]["model"], "demo")
        self.assertEqual(result["items"][0]["mode"], "train")
        self.assertEqual(result["items"][1]["model"], "")
        self.assertTrue(result["items"][0]["exists"])
        self.assertFalse(result["items"][1]["exists"])

    def test_missing_manifest_yields_empty_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch("studio_backend.suite_bridge.SUITE_ROOT", Path(directory)):
                self.assertEqual(benchmark_roster(), {"items": [], "roster": None})

    def test_checked_in_manifest_is_populated_and_complete(self) -> None:
        result = benchmark_roster()
        self.assertTrue(result["items"], "the checked-in campaign manifest produced no rows")
        missing = [item["path"] for item in result["items"] if not item["exists"]]
        self.assertEqual(missing, [], f"manifest names configs that are not on disk: {missing[:5]}")
        self.assertTrue((SUITE_ROOT / result["roster"]).is_file())
        unrouted = [item["path"] for item in result["items"] if not item["model"]]
        self.assertEqual(unrouted, [], f"configs with no model line: {unrouted[:5]}")


if __name__ == "__main__":
    unittest.main()
