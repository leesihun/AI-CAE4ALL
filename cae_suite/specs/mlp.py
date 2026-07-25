from __future__ import annotations

from ..diagnostics import Severity
from .base import (
    MethodSpec,
    PathKind,
    PathRule,
    SpecValidationContext,
    as_list,
    integer,
    numeric,
    validate_common_values,
)


# The launcher validates against these; an unlisted key becomes CFG-UNKNOWN-001.
# The parametric MLP is tabular, not mesh: it reuses the common optimizer/runtime
# keys but none of the mesh/operator geometry keys.
MLP_KEYS = frozenset(
    {
        # Routing + IO
        "model", "mode", "gpu_ids", "log_file_dir", "modelpath",
        "dataset_dir", "infer_dataset", "inference_output_dir", "split_seed",
        # Problem shape: input_var = N inputs, output_var = M outputs
        "input_var", "output_var",
        # Architecture
        "hidden_layers", "activation", "dropout", "norm",
        "input_normalization", "output_normalization", "output_activation",
        # Optimization / loss
        "loss", "training_epochs", "batch_size", "learningr", "weight_decay",
        "warmup_epochs", "num_workers", "prefetch_factor", "max_grad_norm",
        # Runtime / EMA / evaluation
        "use_amp", "use_ema", "ema_decay", "use_compile",
        "val_interval", "checkpoint_interval",
    }
)

_ACTIVATIONS = {"relu", "gelu", "silu", "tanh"}
_NORMS = {"none", "batch", "layer"}
_NORMALIZATION = {"standard", "minmax", "none"}
_LOSSES = {"mse", "mae", "huber"}
_OUTPUT_ACTIVATIONS = {"none", "relu", "sigmoid", "tanh", "softplus"}


def _enum(ctx: SpecValidationContext, name: str, allowed: set[str], code: str) -> None:
    if name in ctx.values:
        value = str(ctx.values[name]).lower()
        if value not in allowed:
            ctx.add(
                code,
                Severity.ERROR,
                f"{name} must be one of {sorted(allowed)}; got {ctx.values[name]!r}.",
                field_name=name,
            )


def validate_mlp(ctx: SpecValidationContext) -> None:
    # Covers input_var/output_var/training_epochs/batch_size/learningr/weight_decay/
    # ema_decay/num_workers/gpu_ids range checks shared with every method.
    validate_common_values(ctx)
    values = ctx.values

    if "hidden_layers" in values:
        layers = as_list(values["hidden_layers"])
        if not layers or any(integer(v) is None or integer(v) < 1 for v in layers):
            ctx.add(
                "MLP-LAYERS-001",
                Severity.ERROR,
                "hidden_layers must be one or more positive integers (comma- or space-separated).",
                field_name="hidden_layers",
            )

    _enum(ctx, "activation", _ACTIVATIONS, "MLP-ACT-001")
    _enum(ctx, "norm", _NORMS, "MLP-NORM-001")
    _enum(ctx, "input_normalization", _NORMALIZATION, "MLP-INORM-001")
    _enum(ctx, "output_normalization", _NORMALIZATION, "MLP-ONORM-001")
    _enum(ctx, "loss", _LOSSES, "MLP-LOSS-001")
    _enum(ctx, "output_activation", _OUTPUT_ACTIVATIONS, "MLP-OACT-001")

    if "dropout" in values:
        dropout = numeric(values["dropout"])
        if dropout is None or not (0.0 <= dropout < 1.0):
            ctx.add(
                "MLP-DROPOUT-001",
                Severity.ERROR,
                f"dropout must be a float in [0, 1); got {values['dropout']!r}.",
                field_name="dropout",
            )


def build_mlp_spec() -> MethodSpec:
    return MethodSpec(
        spec_id="mlp",
        display_name="MLP Surrogate",
        model_ids=("mlp",),
        repository="MLP",
        entrypoint="MLP_main.py",
        valid_modes=("train", "inference"),
        known_keys=MLP_KEYS,
        required_by_mode={
            "train": frozenset(
                {"dataset_dir", "modelpath", "input_var", "output_var", "training_epochs", "batch_size", "learningr"}
            ),
            "inference": frozenset({"modelpath", "infer_dataset", "input_var", "output_var"}),
        },
        recommended_by_mode={"train": frozenset({"hidden_layers", "split_seed", "val_interval"})},
        defaults={
            "hidden_layers": "256,256,128",
            "activation": "gelu",
            "dropout": 0.0,
            "norm": "none",
            "input_normalization": "standard",
            "output_normalization": "standard",
            "output_activation": "none",
            "loss": "mse",
        },
        defaults_by_mode={"inference": {"inference_output_dir": "outputs/predictions"}},
        path_rules=(
            PathRule("dataset_dir", PathKind.INPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.OUTPUT_FILE, frozenset({"train"})),
            PathRule("modelpath", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("infer_dataset", PathKind.INPUT_FILE, frozenset({"inference"})),
            PathRule("inference_output_dir", PathKind.OUTPUT_DIR, frozenset({"inference"})),
        ),
        validators=(validate_mlp,),
        import_modules=("torch", "h5py", "numpy"),
        dataset_kind="table_hdf5",   # tabular X/Y HDF5, not the mesh contract
        native_probe=False,          # no general_modules.load_config to probe
    )
