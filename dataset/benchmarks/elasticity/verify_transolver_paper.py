#!/usr/bin/env python3
"""CPU parity/preflight for the isolated Transolver Elasticity reproduction.

This imports the pinned upstream implementation itself, maps corresponding
parameters, and compares initialization, forward values, gradients, one
AdamW/gradient-clipping step, the cosine schedule, data splits, normalization,
and relative-L2. It intentionally does not exercise CUDA.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parents[2]
TRANSOLVER_ROOT = SUITE_ROOT / "transolver"
OFFICIAL_ROOT = HERE / "source" / "transolver_official"
OFFICIAL_BENCHMARK = OFFICIAL_ROOT / "PDE-Solving-StandardBenchmark"
RUNNER_PATH = HERE / "train_transolver_paper.py"
CONFIG_PATH = (
    SUITE_ROOT
    / "configs"
    / "benchmarks"
    / "elasticity"
    / "config_train_transolver_paper_validation.txt"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("paper_elasticity_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_official():
    """Import upstream modules without requiring the otherwise-unused timm."""
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "model" or name.startswith("model.")
    }
    for name in saved_modules:
        del sys.modules[name]

    # Upstream uses only timm's trunc_normal_ in this model. PyTorch's native
    # implementation has the same signature/algorithm for these arguments.
    timm = types.ModuleType("timm")
    timm_models = types.ModuleType("timm.models")
    timm_layers = types.ModuleType("timm.models.layers")
    timm_layers.trunc_normal_ = torch.nn.init.trunc_normal_
    timm.models = timm_models
    timm_models.layers = timm_layers
    sys.modules["timm"] = timm
    sys.modules["timm.models"] = timm_models
    sys.modules["timm.models.layers"] = timm_layers

    added_einops_stub = "einops" not in sys.modules
    if added_einops_stub:
        einops = types.ModuleType("einops")

        def rearrange(tensor, pattern, **_kwargs):
            if pattern.replace(" ", "") == "bhnd->bn(hd)":
                batch, heads, nodes, dim = tensor.shape
                return tensor.permute(0, 2, 1, 3).reshape(
                    batch, nodes, heads * dim
                )
            raise NotImplementedError(f"einops parity stub: {pattern!r}")

        def repeat(*_args, **_kwargs):
            raise NotImplementedError("einops.repeat is unused by this benchmark")

        einops.rearrange = rearrange
        einops.repeat = repeat
        sys.modules["einops"] = einops

    sys.path.insert(0, str(OFFICIAL_BENCHMARK))
    try:
        official_model = importlib.import_module("model.Transolver_Irregular_Mesh")
        official_normalizer = importlib.import_module("utils.normalizer")
        official_loss = importlib.import_module("utils.testloss")
    finally:
        sys.path.pop(0)
        for name in list(sys.modules):
            if (
                name == "model"
                or name.startswith("model.")
                or name == "utils"
                or name.startswith("utils.")
                or name == "timm"
                or name.startswith("timm.")
                or (added_einops_stub and name == "einops")
            ):
                del sys.modules[name]
        sys.modules.update(saved_modules)
    return official_model.Model, official_normalizer.UnitTransformer, official_loss.TestLoss


def _parameter_pairs(local, official):
    pairs = [
        ("placeholder", local.placeholder, official.placeholder),
        ("preprocess.linear_pre", local.preprocess[0], official.preprocess.linear_pre[0]),
        ("preprocess.linear_post", local.preprocess[2], official.preprocess.linear_post),
    ]
    for index, (local_block, official_block) in enumerate(
        zip(local.blocks, official.blocks)
    ):
        prefix = f"blocks.{index}"
        pairs.extend([
            (f"{prefix}.ln_1", local_block.ln_1, official_block.ln_1),
            (f"{prefix}.attn.temperature", local_block.attn.temperature, official_block.Attn.temperature),
            (f"{prefix}.attn.in_project_x", local_block.attn.in_project_x, official_block.Attn.in_project_x),
            (f"{prefix}.attn.in_project_fx", local_block.attn.in_project_fx, official_block.Attn.in_project_fx),
            (f"{prefix}.attn.in_project_slice", local_block.attn.in_project_slice, official_block.Attn.in_project_slice),
            (f"{prefix}.attn.to_q", local_block.attn.to_q, official_block.Attn.to_q),
            (f"{prefix}.attn.to_k", local_block.attn.to_k, official_block.Attn.to_k),
            (f"{prefix}.attn.to_v", local_block.attn.to_v, official_block.Attn.to_v),
            (f"{prefix}.attn.to_out", local_block.attn.to_out, official_block.Attn.to_out[0]),
            (f"{prefix}.ln_2", local_block.ln_2, official_block.ln_2),
            (f"{prefix}.ffn.linear_pre", local_block.ffn.linear_pre, official_block.mlp.linear_pre[0]),
            (f"{prefix}.ffn.linear_post", local_block.ffn.linear_post, official_block.mlp.linear_post),
        ])
        if local_block.last_layer:
            pairs.extend([
                (f"{prefix}.ln_3", local_block.ln_3, official_block.ln_3),
                (f"{prefix}.head", local_block.head, official_block.mlp2),
            ])
    return pairs


def _tensor_pairs(local, official):
    for name, local_value, official_value in _parameter_pairs(local, official):
        if isinstance(local_value, torch.Tensor):
            yield name, local_value, official_value
            continue
        local_state = dict(local_value.named_parameters(recurse=False))
        official_state = dict(official_value.named_parameters(recurse=False))
        if local_state.keys() != official_state.keys():
            raise AssertionError(
                f"{name}: parameter names differ: {local_state.keys()} vs {official_state.keys()}"
            )
        for subname in local_state:
            yield f"{name}.{subname}", local_state[subname], official_state[subname]


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _assert_close(name: str, left: torch.Tensor, right: torch.Tensor, *, atol=1e-12):
    if not torch.allclose(left, right, rtol=1e-12, atol=atol):
        raise AssertionError(f"{name}: max absolute difference {_max_abs(left, right):.3e}")


def main() -> None:
    torch.set_num_threads(1)
    OfficialModel, UnitTransformer, TestLoss = _load_official()

    if str(TRANSOLVER_ROOT) not in sys.path:
        sys.path.insert(0, str(TRANSOLVER_ROOT))
    from model.paper_elasticity import PaperElasticityTransolver

    runner = _load_runner()
    config = runner.load_config(CONFIG_PATH)
    runner.validate_config(config)

    commit = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={OFFICIAL_ROOT.as_posix()}",
            "-C",
            str(OFFICIAL_ROOT),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if commit != runner.OFFICIAL_COMMIT:
        raise AssertionError(f"Official checkout is {commit}, expected {runner.OFFICIAL_COMMIT}")

    train_xy, train_sigma, test_xy, test_sigma = runner.load_paper_data(
        runner.suite_path(config["source_dir"])
    )
    if train_xy.shape != (1000, 972, 2) or test_xy.shape != (200, 972, 2):
        raise AssertionError("Unexpected paper split shape")
    source_xy = np.load(
        runner.suite_path(config["source_dir"]) / "Random_UnitCell_XY_10.npy",
        mmap_mode="r",
    )
    if not np.array_equal(
        train_xy[0].numpy(), np.asarray(source_xy[:, :, 0], dtype=np.float32)
    ):
        raise AssertionError("Training split does not start at source case 0")
    if not np.array_equal(
        train_xy[-1].numpy(), np.asarray(source_xy[:, :, 999], dtype=np.float32)
    ):
        raise AssertionError("Training split does not end at source case 999")
    if not np.array_equal(
        test_xy[0].numpy(), np.asarray(source_xy[:, :, 1800], dtype=np.float32)
    ):
        raise AssertionError("Test split does not start at source case 1800")
    if not np.array_equal(
        test_xy[-1].numpy(), np.asarray(source_xy[:, :, 1999], dtype=np.float32)
    ):
        raise AssertionError("Test split does not end at source case 1999")

    official_normalizer = UnitTransformer(train_sigma)
    local_mean = train_sigma.mean().reshape(1, 1)
    local_std = (train_sigma.std() + 1.0e-8).reshape(1, 1)
    _assert_close("normalizer mean", local_mean, official_normalizer.mean, atol=0.0)
    _assert_close("normalizer std", local_std, official_normalizer.std, atol=0.0)
    local_encoded = (train_sigma - local_mean) / local_std
    official_encoded = official_normalizer.encode(train_sigma)
    _assert_close("encoded training targets", local_encoded, official_encoded, atol=0.0)
    local_round_trip = local_encoded * local_std + local_mean
    official_round_trip = official_normalizer.decode(official_encoded)
    _assert_close(
        "decoded training-target round trip",
        local_round_trip,
        official_round_trip,
        atol=0.0,
    )
    minimum_test_target_norm = float(
        torch.linalg.vector_norm(test_sigma, dim=1).min().item()
    )
    if minimum_test_target_norm <= 0.0:
        raise AssertionError("Official relative-L2 denominator is zero")

    dimensions = dict(
        hidden_dim=16,
        num_layers=2,
        num_heads=4,
        slice_num=5,
        mlp_ratio=1,
        dropout=0.0,
    )
    torch.manual_seed(1729)
    official = OfficialModel(
        space_dim=2,
        n_layers=dimensions["num_layers"],
        n_hidden=dimensions["hidden_dim"],
        dropout=dimensions["dropout"],
        n_head=dimensions["num_heads"],
        Time_Input=False,
        act="gelu",
        mlp_ratio=dimensions["mlp_ratio"],
        fun_dim=0,
        out_dim=1,
        slice_num=dimensions["slice_num"],
        ref=8,
        unified_pos=False,
    ).double()
    torch.manual_seed(1729)
    local = PaperElasticityTransolver(**dimensions).double()

    initialization_max_abs = 0.0
    for name, local_parameter, official_parameter in _tensor_pairs(local, official):
        initialization_max_abs = max(
            initialization_max_abs, _max_abs(local_parameter, official_parameter)
        )
        _assert_close(f"initialization {name}", local_parameter, official_parameter, atol=0.0)

    # A temperature outside the shared implementation's normal clamp range
    # explicitly verifies that this paper subclass remains unclamped.
    with torch.no_grad():
        for local_block, official_block in zip(local.blocks, official.blocks):
            local_block.attn.temperature.fill_(7.0)
            official_block.Attn.temperature.fill_(7.0)

    torch.manual_seed(2024)
    xy_local = torch.randn(1, 23, 2, dtype=torch.float64, requires_grad=True)
    xy_official = xy_local.detach().clone().requires_grad_(True)
    out_local = local(xy_local)
    out_official = official(xy_official, None).squeeze(-1)
    _assert_close("full forward", out_local, out_official)

    target = torch.rand_like(out_local) * 1000.0 + 1.0
    decoded_local = out_local * local_std.double() + local_mean.double()
    decoded_official = out_official * official_normalizer.std.double() + official_normalizer.mean.double()
    loss_local = runner.relative_l2(decoded_local, target).sum()
    loss_official = TestLoss(size_average=False)(decoded_official, target)
    _assert_close("decoded relative L2", loss_local, loss_official)
    loss_local.backward()
    loss_official.backward()
    _assert_close("input gradient", xy_local.grad, xy_official.grad)
    gradient_max_abs = 0.0
    for name, local_parameter, official_parameter in _tensor_pairs(local, official):
        gradient_max_abs = max(
            gradient_max_abs,
            _max_abs(local_parameter.grad, official_parameter.grad),
        )
        _assert_close(
            f"gradient {name}", local_parameter.grad, official_parameter.grad
        )

    torch.nn.utils.clip_grad_norm_(local.parameters(), 0.1)
    torch.nn.utils.clip_grad_norm_(official.parameters(), 0.1)
    optimizer_local = torch.optim.AdamW(local.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    optimizer_official = torch.optim.AdamW(official.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    scheduler_local = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_local, T_max=500)
    scheduler_official = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_official, T_max=500)
    optimizer_local.step()
    optimizer_official.step()
    scheduler_local.step()
    scheduler_official.step()
    optimizer_step_max_abs = 0.0
    for name, local_parameter, official_parameter in _tensor_pairs(local, official):
        optimizer_step_max_abs = max(
            optimizer_step_max_abs, _max_abs(local_parameter, official_parameter)
        )
        _assert_close(f"AdamW step {name}", local_parameter, official_parameter)
    if optimizer_local.param_groups[0]["lr"] != optimizer_official.param_groups[0]["lr"]:
        raise AssertionError("CosineAnnealingLR differs from official construction")

    torch.manual_seed(314159)
    full_model = PaperElasticityTransolver().cpu()
    full_parameter_count = sum(parameter.numel() for parameter in full_model.parameters())
    torch.manual_seed(314159)
    official_full = OfficialModel(
        space_dim=2,
        n_layers=8,
        n_hidden=128,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=1,
        slice_num=64,
        ref=8,
        unified_pos=False,
    ).cpu()
    official_full_count = sum(parameter.numel() for parameter in official_full.parameters())
    if full_parameter_count != official_full_count:
        raise AssertionError(
            f"Full parameter count differs: {full_parameter_count} vs {official_full_count}"
        )
    for name, local_parameter, official_parameter in _tensor_pairs(
        full_model, official_full
    ):
        _assert_close(
            f"full paper initialization {name}",
            local_parameter,
            official_parameter,
            atol=0.0,
        )
    with torch.no_grad():
        full_xy = torch.randn(1, 11, 2)
        full_local_output = full_model(full_xy)
        full_official_output = official_full(full_xy, None).squeeze(-1)
    _assert_close(
        "full paper topology forward",
        full_local_output,
        full_official_output,
        atol=1.0e-7,
    )

    print(json.dumps({
        "status": "all_cpu_parity_checks_passed",
        "official_commit": commit,
        "official_train_source_indices": [0, 999],
        "official_test_source_indices": [1800, 1999],
        "source_shapes": {
            "train_xy": list(train_xy.shape),
            "train_sigma": list(train_sigma.shape),
            "test_xy": list(test_xy.shape),
            "test_sigma": list(test_sigma.shape),
        },
        "full_paper_parameter_count": full_parameter_count,
        "full_paper_forward_max_abs": _max_abs(
            full_local_output, full_official_output
        ),
        "initialization_max_abs": initialization_max_abs,
        "forward_max_abs": _max_abs(out_local, out_official),
        "loss_abs_difference": abs(float(loss_local.item() - loss_official.item())),
        "gradient_max_abs": gradient_max_abs,
        "optimizer_step_max_abs": optimizer_step_max_abs,
        "lr_after_first_cosine_step": optimizer_local.param_groups[0]["lr"],
        "unclamped_temperature_test_value": 7.0,
        "target_mean": float(local_mean.item()),
        "target_std_unbiased_plus_1e-8": float(local_std.item()),
        "minimum_test_target_l2_norm": minimum_test_target_norm,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
