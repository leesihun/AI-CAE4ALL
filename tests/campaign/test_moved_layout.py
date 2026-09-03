from __future__ import annotations

import importlib.util
from pathlib import Path

import campaign_runner
import score_rollouts


ROOT = Path(__file__).resolve().parents[2]


def test_benchmark_campaign_anchors_to_suite_root():
    assert campaign_runner.REPO_ROOT == ROOT
    assert (campaign_runner.REPO_ROOT / "AI_CAE4ALL_main.py").is_file()


def test_benchmark_scorer_resolves_against_the_method_repository():
    """Native paths are relative to methods/<name>/, exactly as the launcher runs them."""
    assert Path(score_rollouts.REPO_ROOT) == ROOT
    assert Path(score_rollouts.resolve("Neural_Operator", "../../dataset/ex1.h5")) == (
        ROOT / "dataset" / "ex1.h5"
    )
    assert Path(
        score_rollouts.resolve("Neural_Operator", "../../output/neural_operator/rollout/ex1")
    ) == (ROOT / "output" / "neural_operator" / "rollout" / "ex1")
    assert Path(score_rollouts.resolve("Neural_Operator", "local/rollout")) == (
        ROOT / "methods" / "Neural_Operator" / "local" / "rollout"
    )


def test_ex3_generator_anchors_to_suite_root():
    path = ROOT / "configs" / "campaigns" / "ex3" / "generate_full_configs.py"
    spec = importlib.util.spec_from_file_location("ex3_generate_full_configs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.ROOT == ROOT
