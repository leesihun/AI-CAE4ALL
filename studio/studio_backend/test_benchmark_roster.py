from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from studio_backend.suite_bridge import benchmark_roster


class BenchmarkRosterTests(unittest.TestCase):
    def test_reads_campaign_roster_and_keeps_missing_arms_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roster = root / "configs" / "campaigns" / "benchmarks_all" / "roster.tsv"
            existing = root / "configs" / "Method" / "ex4" / "config_train.txt"
            roster.parent.mkdir(parents=True)
            existing.parent.mkdir(parents=True)
            existing.write_text("model demo\nmode train\n", encoding="utf-8")
            roster.write_text(
                "label\ttrain_config\tex_slot\tlight\n"
                "present\tconfigs/Method/ex4/config_train.txt\tex4\t1\n"
                "missing\tconfigs/Method/ex5/config_train.txt\tex5\t0\n",
                encoding="utf-8",
            )

            with patch("studio_backend.suite_bridge.SUITE_ROOT", root):
                result = benchmark_roster()

        self.assertEqual([item["label"] for item in result["items"]], ["present", "missing"])
        self.assertTrue(result["items"][0]["exists"])
        self.assertTrue(result["items"][0]["light"])
        self.assertFalse(result["items"][1]["exists"])


if __name__ == "__main__":
    unittest.main()
