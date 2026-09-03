"""Model construction: the only path that builds an OperatorWrapper from a
config + train-fit dataset (IMPLEMENTATION_PLAN.md section 6.3).

MODEL_REGISTRY/VALIDATORS are populated as each architecture module is
implemented (deeponet first, then point_deeponet, fno, gino) -- see the
bottom of this file for the registration calls.
"""

from general_modules.config_validation import validate_model_config
from general_modules.data_spec import build_data_spec_from_dataset
from model.adapters.coordinate_domain import CoordinateDomain
from model.operator_wrapper import OperatorWrapper

MODEL_REGISTRY = {}
VALIDATORS = {}


def register_model(name, core_cls, validate_fn):
    MODEL_REGISTRY[name] = core_cls
    VALIDATORS[name] = validate_fn


def _resolve_core_class(model_name, config):
    """Select an opt-in architecture variant without changing registry defaults."""
    if (model_name == "gino" and
            str(config.get("gino_variant", "mesh_state")).lower() == "paper_decoder"):
        from model.gino_carcfd import CarCFDGINODecoder
        return CarCFDGINODecoder
    return MODEL_REGISTRY[model_name]


def build_model(config, train_dataset):
    """Build (OperatorWrapper, DataSpec, CoordinateDomain) for the selected model.

    `train_dataset` must already have `prepare_preprocessing()` applied (i.e.
    the first element returned by `MeshGraphDataset.split(...)`).
    """
    data_spec = build_data_spec_from_dataset(train_dataset, config)

    model_name = str(config.get('model', '')).lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'; expected one of {sorted(MODEL_REGISTRY)}.")

    validate_model_config(config, data_spec)  # dispatches to VALIDATORS[model_name]

    coordinate_domain = CoordinateDomain.from_dataset(
        train_dataset, out_of_bounds_policy=str(config.get('out_of_bounds_policy', 'error')).lower(),
    )

    core_cls = _resolve_core_class(model_name, config)
    core = core_cls(config, data_spec, coordinate_domain)

    core_params = sum(p.numel() for p in core.parameters())
    print(f"[factory] Built '{model_name}' core with {core_params:,} parameters.")

    wrapper = OperatorWrapper(core, config)
    return wrapper, data_spec, coordinate_domain


def build_model_from_checkpoint(config, checkpoint):
    """Construct (OperatorWrapper, DataSpec, CoordinateDomain) purely from a
    checkpoint's saved metadata -- no dataset object is read (section 14).
    `config['model']` must already equal `checkpoint['selected_model']`
    (the caller, inference_profiles/rollout.py, enforces this before calling).
    """
    from general_modules.data_spec import DataSpec

    model_name = checkpoint['selected_model']
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Checkpoint selected_model='{model_name}' is not a registered model.")

    data_spec = DataSpec.from_dict(checkpoint['data_config'])
    coordinate_domain = CoordinateDomain.from_dict(checkpoint['adapter_config'])

    # Overlay every checkpointed architecture key into config so the model's
    # __init__ (which reads from `config.get(...)`) reconstructs identically
    # regardless of what the runtime config file said (section 13's overlay rule).
    model_config = checkpoint.get('model_config', {})
    for k, v in model_config.items():
        if k == 'model_name':
            continue
        old = config.get(k)
        config[k] = v
        if old is not None and old != v:
            print(f"  [checkpoint overlay] {k}: {old} -> {v}")
    config['input_var'] = data_spec.input_var
    config['output_var'] = data_spec.output_var

    core_cls = _resolve_core_class(model_name, config)
    core = core_cls(config, data_spec, coordinate_domain)
    wrapper = OperatorWrapper(core, config)

    if 'ema_state_dict' in checkpoint:
        ema_sd = checkpoint['ema_state_dict']
        model_sd = {k[len('module.'):]: v for k, v in ema_sd.items() if k.startswith('module.')}
        wrapper.load_state_dict(model_sd, strict=True)
        print("  Loaded EMA weights from checkpoint")
    else:
        wrapper.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("  Loaded training weights from checkpoint (no EMA available)")

    return wrapper, data_spec, coordinate_domain


from model.deeponet import DeepONet, validate_config as _validate_deeponet  # noqa: E402
register_model("deeponet", DeepONet, _validate_deeponet)

from model.point_deeponet import PointDeepONet, validate_config as _validate_point_deeponet  # noqa: E402
register_model("point_deeponet", PointDeepONet, _validate_point_deeponet)

from model.fno import MeshFNO, validate_config as _validate_fno  # noqa: E402
register_model("fno", MeshFNO, _validate_fno)

from model.gino import MeshGINO, validate_config as _validate_gino  # noqa: E402


def _validate_gino_variant(config, data_spec):
    if str(config.get("gino_variant", "mesh_state")).lower() == "paper_decoder":
        from model.gino_carcfd import validate_carcfd_config
        return validate_carcfd_config(config, data_spec)
    return _validate_gino(config, data_spec)


register_model("gino", MeshGINO, _validate_gino_variant)
