import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]

TRAINING_ENTRYPOINTS = (
    "methods/HI_MGNFlow/training_profiles/single_training.py",
    "methods/HI_MGNFlow/training_profiles/distributed_training.py",
    "methods/MeshGraphNets/training_profiles/single_training.py",
    "methods/MeshGraphNets/training_profiles/distributed_training.py",
    "methods/MeshGraphNets_Variational/training_profiles/single_training.py",
    "methods/MeshGraphNets_Variational/training_profiles/distributed_training.py",
    "methods/Neural_Operator/training_profiles/single_training.py",
    "methods/Neural_Operator/training_profiles/distributed_training.py",
    "methods/Transolver/training_profiles/single_training.py",
    "methods/Transolver/training_profiles/distributed_training.py",
)


def _is_num_workers_subscript(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "config"
        and isinstance(node.slice, ast.Constant)
        and node.slice.value == "num_workers"
    )


def _is_num_workers_default(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "config"
        and node.func.attr == "get"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "num_workers"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == 0
    )


def _assigns_num_workers_default(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "num_workers"
                for target in node.targets)
        and any(_is_num_workers_default(child) for child in ast.walk(node.value))
    )


@pytest.mark.parametrize("relative_path", TRAINING_ENTRYPOINTS)
def test_native_training_entrypoint_defaults_num_workers(relative_path):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))

    assert any(_assigns_num_workers_default(node) for node in ast.walk(tree))
    assert not any(_is_num_workers_subscript(node) for node in ast.walk(tree))
