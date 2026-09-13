#!/usr/bin/env python3
"""Generate the AI-CAE4ALL native configs for the iFEM dense48 sweep.

This is the single source of truth for the roster, the GPU packing and candidates.tsv.
Regenerate after any edit:

    python make_configs.py

Everything lives inside the AI-CAE4ALL tree, so every path is RELATIVE and resolves from
the method repository root (the launcher runs each method with cwd = methods/<Name>/,
cae_suite/cli.py:291):

    ../../dataset/iFEM_dense48/train.h5          training data
    ../../output/iFEM_dense48/<ARM>/model.pth    checkpoint
    ../../output/iFEM_dense48/<ARM>/train.log    log; its directory also receives the
                                                 periodic train/test visualisation dumps
    ../../output/iFEM_dense48/<ARM>/infer/       held-out predictions

No absolute paths and nothing outside AI-CAE4ALL, so the bundle is portable to any box
that has the suite checked out.

Per-model key sets (validated against cae_suite/specs/*.py, the launcher's key registry):
  * transolver     : write_test_predictions, max_grad_norm, write_preprocessing; NO
                     display_testset, NO checkpoint_interval, NO edge_var.
  * meshgraphnets  : edge_var, message_passing_num; NO max_grad_norm, NO
                     write_preprocessing, NO coordinate_normalization, NO
                     write_test_predictions.
  * deeponet       : the deeponet_* block, checkpoint_interval; NO write_test_predictions.
Any key outside a model's known_keys is a CFG-UNKNOWN-001 warning (error under --strict).

Floats are written in decimal form on purpose: the native parser types a token with a '.'
as float and one without as int, so '1e-4' would reach AdamW as the STRING '1e-4'.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path


def write_lf(path, text):
    """Write LF-terminated UTF-8.

    The newline= argument of Path.write_text is Python 3.10+, and the box that runs this
    sweep is on 3.9 (conda env AARL), where it raises TypeError before a single
    config is written.
    """
    with io.open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


CAMPAIGN = "iFEM_dense48"
NUM_GPUS = 4

DATASET = f"../../dataset/{CAMPAIGN}"
OUTPUT = f"../../output/{CAMPAIGN}"
TRAIN_H5 = f"{DATASET}/train.h5"
INFER_H5 = f"{DATASET}/infer.h5"
# MeshGraphNets opens dataset_dir with h5py 'r+' and writes metadata/normalization_params
# into it unconditionally (mesh_dataset.py:502-550 via single_training.py:52) -- there is
# no opt-out key. Each MGN arm therefore trains on its own copy, which train_all.sh
# creates just before the arm runs and deletes afterwards, so peak disk stays small.
MGN_PRIVATE = DATASET + "/native_train/{arm}/train.h5"

# Row layout of the dataset. Not tunable: nodal_data is [20, 1, 81] and the loaders
# require 3 + max(input_var, output_var) + cond_var <= 20, an exact fit.
#   rows 0-2   x, y, z                            -> graph.pos
#   rows 3-6   u_eps1_x/y, u_eps2_x/y             -> graph.y, the 4 targets
#   rows 7-19  phi, trace_eps1_x/y, trace_eps2_x/y, traction_x/y, body_x/y,
#              gamma_dir_x/y, gamma_traction_x/y  -> the 13 conditioning columns of graph.x
# num_timesteps == 1, so the leading input_var columns of graph.x are ZEROS: the model
# sees position plus the 13 conditions and nothing else.
CONTRACT = dict(input_var=4, output_var=4, cond_var=13)

# u_eps2 (targets 3 and 4) is nonzero in 18.5% of train cases, 0.40% of nodes, and carries
# 0.02% of the squared target energy, while the per-channel z-score divides it by a small
# std. 0.05 keeps it from dominating the gradient.
LOSS_WEIGHTS = [1.0, 1.0, 0.05, 0.05]

# ---------------------------------------------------------------------------
# The sweep. One BASE per model, then one variation per line changing exactly ONE key,
# so every arm is a controlled comparison against its base.
# ---------------------------------------------------------------------------
BASE = {
    "transolver": dict(
        latent_dim=256, num_layers=4, num_heads=8, slice_num=64,
        batch_size=32, learningr=0.0003, weight_decay=0.00001, epochs=250,
    ),
    "meshgraphnets": dict(
        latent_dim=128, message_passing_num=8,
        batch_size=32, learningr=0.0001, weight_decay=0.00001, epochs=250,
    ),
    "deeponet": dict(
        hidden_channels=256, basis_dim=128, branch_depth=3, trunk_depth=3,
        batch_size=32, learningr=0.0003, weight_decay=0.00001, epochs=250,
    ),
}

VARIANTS = [
    # id,   model,            change (empty = the base arm),                  axis
    ("t01", "transolver", {}, "base"),
    ("t02", "transolver", dict(num_layers=2), "num_layers down"),
    ("t03", "transolver", dict(num_layers=8), "num_layers up"),
    ("t04", "transolver", dict(num_layers=16), "num_layers up"),
    ("t05", "transolver", dict(latent_dim=128), "latent_dim down"),
    ("t06", "transolver", dict(latent_dim=512), "latent_dim up"),
    ("t07", "transolver", dict(num_heads=4), "num_heads down"),
    ("t08", "transolver", dict(num_heads=16), "num_heads up"),
    ("t09", "transolver", dict(slice_num=16), "slice_num down"),
    ("t10", "transolver", dict(slice_num=32), "slice_num down"),
    ("t11", "transolver", dict(slice_num=128), "slice_num up"),
    ("t12", "transolver", dict(batch_size=16), "batch_size down"),
    ("t13", "transolver", dict(batch_size=8), "batch_size down"),
    ("t14", "transolver", dict(epochs=600), "training_epochs up"),
    ("t15", "transolver", dict(epochs=1000), "training_epochs up"),

    ("m01", "meshgraphnets", {}, "base"),
    ("m02", "meshgraphnets", dict(message_passing_num=4), "message_passing_num down"),
    ("m03", "meshgraphnets", dict(message_passing_num=2), "message_passing_num down"),
    ("m04", "meshgraphnets", dict(message_passing_num=16), "message_passing_num up"),
    ("m05", "meshgraphnets", dict(latent_dim=64), "latent_dim down"),
    ("m06", "meshgraphnets", dict(latent_dim=256), "latent_dim up"),
    ("m07", "meshgraphnets", dict(latent_dim=512), "latent_dim up"),
    ("m08", "meshgraphnets", dict(batch_size=16), "batch_size down"),
    ("m09", "meshgraphnets", dict(batch_size=8), "batch_size down"),
    ("m10", "meshgraphnets", dict(epochs=600), "training_epochs up"),

    ("d01", "deeponet", {}, "base"),
    ("d02", "deeponet", dict(hidden_channels=128), "hidden_channels down"),
    ("d03", "deeponet", dict(hidden_channels=512), "hidden_channels up"),
    ("d04", "deeponet", dict(basis_dim=64), "basis_dim down"),
    ("d05", "deeponet", dict(basis_dim=256), "basis_dim up"),
    ("d06", "deeponet", dict(batch_size=16), "batch_size down"),
    ("d07", "deeponet", dict(batch_size=8), "batch_size down"),
    ("d08", "deeponet", dict(epochs=600), "training_epochs up"),
]

REPO = {"transolver": "Transolver", "meshgraphnets": "MeshGraphNets", "deeponet": "Neural_Operator"}


def arm_spec(arm_id, model, change):
    spec = dict(BASE[model])
    spec.update(change)
    spec["id"] = arm_id.upper()
    spec["model"] = model
    return spec


def est_hours(s) -> float:
    """Scheduling estimate only, from per-epoch timings measured on this dataset.

    Transolver is CPU-dispatch bound: PhysicsAttentionIrregular.forward loops over the
    graphs of a batch in Python once per layer (model/physics_attention.py:361-362), so
    epoch cost tracks sample_count x num_layers and is nearly flat in batch_size,
    latent_dim, num_heads and slice_num. 1632 s/epoch was measured at 16 layers.
    """
    if s["model"] == "transolver":
        sec = 102.0 * s["num_layers"]
    elif s["model"] == "meshgraphnets":
        sec = 26.0 + 2.1 * s["message_passing_num"] * (s["latent_dim"] / 128.0)
    else:
        sec = 61.0 * (s["hidden_channels"] / 256.0) ** 0.5
    return s["epochs"] * sec / 3600.0


def pack_gpus(specs):
    """Longest-processing-time bin packing: every GPU stays busy and the makespan is
    near-minimal. Assigns each arm a gpu and returns the per-gpu queues in run order."""
    load = [0.0] * NUM_GPUS
    queues = [[] for _ in range(NUM_GPUS)]
    for s in sorted(specs, key=est_hours, reverse=True):
        gpu = min(range(NUM_GPUS), key=lambda g: load[g])
        s["gpu"] = gpu
        load[gpu] += est_hours(s)
        queues[gpu].append(s["id"])
    return queues, load


def fmt(value) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        text = f"{value:.10f}".rstrip("0")
        return text + "0" if text.endswith(".") else text
    if isinstance(value, (list, tuple)):
        return ", ".join(fmt(v) for v in value)
    return str(value)


def render(header_lines, blocks) -> str:
    lines = list(header_lines)
    for title, items in blocks:
        lines.append("")
        lines.append(f"% {title}")
        for key, value in items:
            lines.append(f"{key}\t{fmt(value)}")
    return "\n".join(lines) + "\n"


def contract_block():
    return [
        ("input_var", CONTRACT["input_var"]),
        ("output_var", CONTRACT["output_var"]),
        ("cond_var", CONTRACT["cond_var"]),
        ("positional_features", 0),
        # Must stay False: the file has exactly 20 rows, so there is no node-type row and
        # the loader would one-hot gamma_traction_y instead.
        ("use_node_types", False),
        ("feature_loss_weights", LOSS_WEIGHTS),
        ("time_integration", "ar_ot"),
        ("use_world_edges", False),
        ("use_multiscale", False),
    ]


def architecture(s):
    if s["model"] == "transolver":
        return [
            ("latent_dim", s["latent_dim"]),
            ("num_layers", s["num_layers"]),
            ("num_heads", s["num_heads"]),
            ("mlp_ratio", 2),
            ("slice_num", s["slice_num"]),
            ("dropout", 0.0),
            # Explicit on purpose: the repo default is slice_space, which is ~40% slower.
            ("attention_kernel", "naive"),
            ("chunk_size", 0),
            ("coordinate_normalization", "centered_isotropic"),
            ("small_output_init", True),
            ("temperature_init", 0.5),
            ("temperature_min", 0.1),
            ("temperature_max", 5.0),
            ("amortized_training", False),
            ("infer_mode", "direct"),
            ("infer_chunk_size", 0),
        ]
    if s["model"] == "meshgraphnets":
        # edge_var is fixed at 8 by edge_features.py; any other value is a hard error.
        return [
            ("latent_dim", s["latent_dim"]),
            ("message_passing_num", s["message_passing_num"]),
            ("edge_var", 8),
        ]
    return [
        ("coordinate_normalization", "centered_isotropic"),
        # z is identically 0 in all 47,929 samples, so the active axes resolve to (0, 1).
        ("operator_dim", 2),
        # Every sample shares the same 9x9 unit grid (verified bit-identical across all
        # samples), so a 9x9 sensor grid is exactly bijective and needs no padding slack.
        ("grid_padding", 0.0),
        ("dimension_tolerance", 0.0001),
        ("out_of_bounds_policy", "error"),
        ("sdf_source", "none"),
        ("sdf_sidecar", "none"),
        ("global_condition_features", "none"),
        ("integration_weight_source", "none"),
        ("deeponet_branch_source", "fixed_sensors"),
        ("deeponet_sensor_resolution", [9, 9]),
        ("deeponet_hidden_channels", s["hidden_channels"]),
        ("deeponet_branch_depth", s["branch_depth"]),
        ("deeponet_trunk_depth", s["trunk_depth"]),
        ("deeponet_basis_dim", s["basis_dim"]),
        ("deeponet_activation", "silu"),
        ("deeponet_multi_output", "split_both"),
        ("deeponet_max_branch_params", 100000000),
        ("infer_query_chunk_size", 0),
    ]


def header(s, axis, mode):
    return [
        f"% AI-CAE4ALL native config -- {CAMPAIGN} sweep, arm {s['id']} ({mode}).",
        f"% Axis: {axis}.  Generated by configs/campaigns/{CAMPAIGN}/make_configs.py -- do not hand-edit.",
        "% Launch from the AI-CAE4ALL suite root:",
        "%   python AI_CAE4ALL_main.py --config <this file>",
        "% Paths are relative to the method repo (methods/<Name>/), which is the launcher's cwd.",
    ]


def train_config(s, axis) -> str:
    arm = s["id"]
    train_h5 = MGN_PRIVATE.format(arm=arm) if s["model"] == "meshgraphnets" else TRAIN_H5
    paths = [
        ("dataset_dir", train_h5),
        ("modelpath", f"{OUTPUT}/{arm}/model.pth"),
        ("log_file_dir", f"{OUTPUT}/{arm}/train.log"),
        ("split_seed", 42),
    ]
    opt = [
        ("training_epochs", s["epochs"]),
        ("batch_size", s["batch_size"]),
        ("learningr", s["learningr"]),
        ("weight_decay", s["weight_decay"]),
        ("warmup_epochs", 10),
        ("grad_accum_steps", 1),
        ("num_workers", 4),
        ("prefetch_factor", 4),
    ]
    if s["model"] != "meshgraphnets":
        opt.append(("max_grad_norm", 1.0))

    runtime = [
        # std_noise only perturbs the leading output_var columns of graph.x, which are the
        # ZERO state block on this static dataset; it regularizes nothing here.
        ("std_noise", 0.0),
        ("noise_gamma", 1),
        # augment_geometry rotates x[:, :3] and y[:, :3] as one xyz vector. Here those are
        # two unrelated 2-D fields and the directional condition channels are never
        # rotated, so the augmentation is physically wrong. It must stay False.
        ("augment_geometry", False),
        # Explicit: the native default is True.
        ("use_amp", False),
        ("use_ema", False),
        # torch.compile breaks on the per-graph .item() in physics attention.
        ("use_compile", False),
        ("use_checkpointing", False),
        ("use_parallel_stats", False),
    ]
    if s["model"] != "meshgraphnets":
        runtime.append(("write_preprocessing", False))

    # Visualisation: periodic train/test reconstruction dumps land under the log directory,
    # i.e. ../../output/iFEM_dense48/<ARM>/.
    val = [
        ("val_interval", 1),
        ("test_interval", 25),
        ("test_max_batches", 32),
        ("test_batch_idx", [0, 1, 2, 3, 4, 5, 6, 7]),
        ("display_trainset", True),
    ]
    if s["model"] != "transolver":
        val.append(("display_testset", True))
        # The PNG colours one channel and defaults to the last one, which here is
        # u_eps2's second component: identically zero in 85% of the samples, so the
        # ground-truth panel would be blank most of the time. Channel 0 is nonzero
        # everywhere (mean per-node std 0.85 vs 0.03).
        val.append(("plot_feature_idx", 0))
    if s["model"] == "transolver":
        # Transolver has no renderer at all -- test_model dumps numeric HDF5 only --
        # so display_testset/plot_feature_idx are not in its spec. This writes the
        # predictions themselves instead.
        val.append(("write_test_predictions", True))
    if s["model"] == "deeponet":
        # The only method with periodic checkpointing; the other two save once, after the
        # final epoch, and print "No checkpoint saved" if interrupted.
        val.append(("checkpoint_interval", 25))

    return render(header(s, axis, "train"), [
        ("Run", [("model", s["model"]), ("gpu_ids", s["gpu"]),
                 ("parallel_mode", "ddp"), ("mode", "train")]),
        ("Training data and checkpoint (inside the AI-CAE4ALL tree)", paths),
        ("Data contract (fixed by the dataset row layout)", contract_block()),
        ("Model", architecture(s)),
        ("Optimization", opt),
        ("Runtime", runtime),
        ("Validation and periodic train/test visualisation", val),
    ])


def infer_config(s, axis) -> str:
    arm = s["id"]
    paths = [
        ("infer_dataset", INFER_H5),
        ("modelpath", f"{OUTPUT}/{arm}/model.pth"),
        ("log_file_dir", f"{OUTPUT}/{arm}/infer.log"),
        ("inference_output_dir", f"{OUTPUT}/{arm}/infer"),
        ("infer_timesteps", 1),
    ]
    return render(header(s, axis, "inference"), [
        ("Run", [("model", s["model"]), ("gpu_ids", s["gpu"]),
                 ("parallel_mode", "ddp"), ("mode", "inference")]),
        ("Held-out inference: 8,482 cases over 177 geometries absent from training", paths),
        ("Data contract (fixed by the dataset row layout)", contract_block()),
        ("Architecture -- replaced by checkpoint['model_config'] at load; mirrored so the "
         "file states the truth", architecture(s)),
        ("Inference runtime", [("batch_size", s["batch_size"]), ("use_amp", False),
                               ("use_compile", False)]),
    ])


def describe(s) -> str:
    if s["model"] == "transolver":
        return (f"latent {s['latent_dim']}, {s['num_layers']} layers, "
                f"{s['num_heads']} heads, slice {s['slice_num']}")
    if s["model"] == "meshgraphnets":
        return f"latent {s['latent_dim']}, mp {s['message_passing_num']}"
    return f"hidden {s['hidden_channels']}, basis {s['basis_dim']}"


def main() -> None:
    here = Path(__file__).resolve().parent
    # here = <bundle>/configs/campaigns/<CAMPAIGN>; configs root is two levels up
    configs_root = here.parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(configs_root),
                    help="configs/ root to write into (default: this bundle's configs/)")
    args = ap.parse_args()
    out_root = Path(args.out)

    specs = [arm_spec(i, m, c) for i, m, c, _ in VARIANTS]
    axes = {i.upper(): a for i, _, _, a in VARIANTS}
    queues, load = pack_gpus(specs)

    for s in specs:
        arm = s["id"]
        d = out_root / REPO[s["model"]] / CAMPAIGN
        d.mkdir(parents=True, exist_ok=True)
        write_lf(d / f"config_train_{arm.lower()}.txt", train_config(s, axes[arm]))
        write_lf(d / f"config_infer_{arm.lower()}.txt", infer_config(s, axes[arm]))

    # candidates.tsv -- the roster of record
    cols = ["candidate", "model", "axis", "gpu", "epochs", "batch_size", "learningr",
            "weight_decay", "architecture", "est_gpu_hours", "train_config", "infer_config"]
    rows = ["\t".join(cols)]
    for s in sorted(specs, key=lambda x: x["id"]):
        arm = s["id"]
        rows.append("\t".join([
            arm, s["model"], axes[arm], str(s["gpu"]), str(s["epochs"]), str(s["batch_size"]),
            fmt(s["learningr"]), fmt(s["weight_decay"]), describe(s), f"{est_hours(s):.0f}",
            f"configs/{REPO[s['model']]}/{CAMPAIGN}/config_train_{arm.lower()}.txt",
            f"configs/{REPO[s['model']]}/{CAMPAIGN}/config_infer_{arm.lower()}.txt",
        ]))
    write_lf(here / "candidates.tsv", "\n".join(rows) + "\n")

    # queues.sh -- consumed by train_all.sh, so the packing can never drift from the configs
    q = ["#!/usr/bin/env bash",
         "# Generated by make_configs.py -- do not hand-edit.",
         "# Longest-processing-time packing over the GPUs; every device stays busy.",
         f"ARMS_ALL=\"{' '.join(sorted(s['id'] for s in specs))}\"",
         f"ARMS_MGN=\"{' '.join(sorted(s['id'] for s in specs if s['model'] == 'meshgraphnets'))}\"",
         f"NUM_GPUS={NUM_GPUS}"]
    for g in range(NUM_GPUS):
        q.append(f"QUEUE_{g}=\"{' '.join(queues[g])}\"   # ~{load[g]:.0f} h")
    for s in specs:
        q.append(f"CONFIG_DIR_{s['id']}=\"configs/{REPO[s['model']]}/{CAMPAIGN}\"")
    write_lf(here / "queues.sh", "\n".join(q) + "\n")

    total = sum(est_hours(s) for s in specs)
    print(f"wrote {2 * len(specs)} configs for {len(specs)} arms")
    print(f"  {total:.0f} GPU-hours total; makespan {max(load):.0f} h = {max(load) / 24:.1f} days "
          f"on {NUM_GPUS} GPUs")
    for g in range(NUM_GPUS):
        print(f"  gpu {g}: {load[g]:6.0f} h  {' '.join(queues[g])}")


if __name__ == "__main__":
    main()
