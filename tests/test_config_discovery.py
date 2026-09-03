from pathlib import Path

from cae_suite.config_discovery import checked_in_config_paths


def test_checked_in_config_discovery_uses_only_top_level_configs(tmp_path: Path):
    expected = {
        tmp_path / "configs" / "MeshGraphNets" / "config_train.txt",
        tmp_path / "configs" / "nested" / "config_infer_case.txt",
    }
    ignored = {
        tmp_path / "MeshGraphNets" / "config_legacy.txt",
        tmp_path / "configs" / "nested" / "settings.txt",
        tmp_path / "configs" / "outputs" / "config_generated.txt",
        tmp_path / "configs" / "__pycache__" / "config_cached.txt",
    }
    for path in expected | ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model meshgraphnets\n", encoding="utf-8")

    assert checked_in_config_paths(tmp_path) == {path.resolve() for path in expected}
