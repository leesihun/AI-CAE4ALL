import pytest

from general_modules.load_config import load_config


@pytest.mark.parametrize("key", ["load_all", "plot_mode", "recon_iter"])
def test_retired_standalone_noops_fail_in_native_loader(tmp_path, key):
    config = tmp_path / "config.txt"
    config.write_text(f"model simulgenvae\nmode train\n{key} 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no HDF5-native runtime effect"):
        load_config(config)
