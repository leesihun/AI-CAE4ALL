import ast
from pathlib import Path

import pytest

from cae_suite.specs import (
    build_chi_mgnflow_spec,
    build_meshgraphnets_spec,
    build_mlp_spec,
    build_neural_operator_spec,
    build_sdfflow_spec,
    build_simulgenvae_spec,
    build_transolver_spec,
    build_variational_spec,
)


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    "tests", "test", "misc", "output", "outputs", ".venv", "venv",
    "docs", "examples", "benchmarks", "__pycache__",
}
CONFIG_NAMES = {"config", "cfg", "stage_config", "model_config", "data_config"}

# These values are constructed after parsing from dataset/checkpoint/runtime
# state, or are fields in a small internal flow-options dict. They are not
# authorable flat-file settings and therefore do not belong in MethodSpec.
RUNTIME_INTERNAL = {
    "methods/MeshGraphNets": {
        "_ddp_port", "_norm_stats", "_pin_memory", "log_dir",
        "num_node_types", "num_timesteps",
    },
    "methods/MeshGraphNets_Variational": {
        "_ddp_port", "_norm_stats", "_pin_memory", "log_dir",
        "num_node_types", "num_timesteps",
    },
    "methods/HI_MGNFlow": {
        "_ddp_port", "_norm_stats", "_pin_memory", "log_dir",
        "num_node_types", "num_timesteps",
        "det_prob", "logit_scale", "t_sampling", "weighting",
    },
    "methods/Neural_Operator": {"_norm_stats", "_paper_target_mean", "_paper_target_std"},
    "methods/Transolver": {
        "_ddp_port", "_norm_stats", "_pin_memory", "log_dir",
        "noise_std_ratio", "num_node_types", "num_timesteps",
    },
    "methods/SDFFlow": {"_ddp_port", "cond_dim"},
    "methods/SimulGenVAE": {"_ddp_port", "num_channels", "num_samples", "num_time"},
    "methods/MLP": set(),
}

CASES = (
    ("methods/MeshGraphNets", build_meshgraphnets_spec),
    ("methods/MeshGraphNets_Variational", build_variational_spec),
    ("methods/HI_MGNFlow", build_chi_mgnflow_spec),
    ("methods/Neural_Operator", build_neural_operator_spec),
    ("methods/Transolver", build_transolver_spec),
    ("methods/SDFFlow", build_sdfflow_spec),
    ("methods/SimulGenVAE", build_simulgenvae_spec),
    ("methods/MLP", build_mlp_spec),
)


def _is_config_object(node):
    """Recognize both local ``config`` dicts and stored ``self.config`` dicts."""
    return (
        isinstance(node, ast.Name) and node.id in CONFIG_NAMES
    ) or (
        isinstance(node, ast.Attribute) and node.attr in CONFIG_NAMES
    )


def _consumed_literals(repository):
    found = {}
    root = ROOT / repository
    for path in root.rglob("*.py"):
        if EXCLUDED_PARTS.intersection(path.relative_to(root).parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            key = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"get", "pop", "setdefault"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _is_config_object(node.func.value)
            ):
                key = node.args[0].value
            elif (
                isinstance(node, ast.Subscript)
                and _is_config_object(node.value)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                key = node.slice.value
            if key:
                found.setdefault(key, f"{path.relative_to(ROOT)}:{node.lineno}")
    return found


@pytest.mark.parametrize("repository,spec_builder", CASES)
def test_native_config_literals_are_registered_or_explicitly_runtime_internal(
    repository, spec_builder
):
    consumed = _consumed_literals(repository)
    unexplained = sorted(
        set(consumed) - set(spec_builder().known_keys) - RUNTIME_INTERNAL[repository]
    )
    assert not unexplained, {key: consumed[key] for key in unexplained}
