import pytest

from general_modules.mesh_dataset import MeshGraphDataset
from model.factory import build_model, MODEL_REGISTRY
from tests.conftest import base_config_2d


def test_all_four_models_registered():
    assert set(MODEL_REGISTRY.keys()) == {"deeponet", "point_deeponet", "fno", "gino"}


def test_factory_rejects_unknown_model_name(tiny_static_2d_h5):
    cfg = base_config_2d(tiny_static_2d_h5, model='deeponet', deeponet_sensor_resolution=[8, 8])
    ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
    train, _, _ = ds.split(0.8, 0.1, 0.1, seed=42)
    cfg['model'] = 'bogus'
    with pytest.raises(ValueError, match="Unknown model"):
        build_model(cfg, train)


def test_switching_model_requires_no_data_code_change(tiny_static_2d_h5):
    """The same dataset/config skeleton builds all four models by only
    changing `model` and each model's own knobs (section 1's core promise)."""
    for model_name, extra in [
        ('deeponet', {'deeponet_sensor_resolution': [8, 8]}),
        ('point_deeponet', {'point_sensor_count': 16}),
        ('fno', {'fno_grid_resolution': [8, 8], 'fno_modes': [3, 4]}),
        ('gino', {'gino_grid_resolution': [6, 6], 'gino_fno_modes': [2, 3],
                  'gino_in_radius': 0.35, 'gino_out_radius': 0.35}),
    ]:
        cfg = base_config_2d(tiny_static_2d_h5, model=model_name, **extra)
        ds = MeshGraphDataset(tiny_static_2d_h5, cfg)
        train, _, _ = ds.split(0.8, 0.1, 0.1, seed=42)
        model, data_spec, domain = build_model(cfg, train)
        assert model.model_name == model_name
