"""Job factories that shell out to real native entrypoints.

Each function here builds an argv list and hands it to STATE.create_command_job
so it runs, logs, and can be cancelled exactly like every other Studio job.
Nothing here re-implements the native tool; it only assembles its command line.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
import time
from typing import Any

from studio_backend.paths import RUNTIME_ROOT, SUITE_ROOT, relative, safe_repo_path, slug
from studio_backend.state import STATE

UPLOAD_SUFFIXES = {
    "dataset": {".h5", ".hdf5", ".csv", ".json"},
    "geometry": {
        ".stl", ".step", ".stp", ".iges", ".igs", ".brep",
        ".obj", ".ply", ".off", ".msh", ".vtk", ".vtu", ".vtp",
    },
    "checkpoint": {".pth", ".pt", ".ckpt"},
}


def create_simulgen_smoke_fixture() -> dict[str, Any]:
    try:
        import h5py
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("h5py and numpy are required for the SimulGen smoke fixture.") from exc
    root = RUNTIME_ROOT / "simulgen-smoke"
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / "fixed_geometry.h5"
    conditions = root / "conditions.csv"
    if not dataset.exists():
        rng = np.random.default_rng(42)
        sample_count, features, timesteps, nodes = 8, 4, 4, 10
        theta = np.linspace(0.0, 2.0 * np.pi, nodes, endpoint=False, dtype=np.float32)
        coordinates = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        with h5py.File(dataset, "w") as handle:
            handle.attrs["num_samples"] = sample_count
            handle.attrs["studio_fixture"] = True
            data_group = handle.create_group("data")
            for sample_index in range(sample_count):
                group = data_group.create_group(str(sample_index))
                nodal = np.zeros((features, timesteps, nodes), dtype=np.float32)
                nodal[0:3] = coordinates[:, None, :]
                amplitude = 0.5 + 0.1 * sample_index
                for time_index in range(timesteps):
                    nodal[3, time_index] = amplitude * np.sin(theta + time_index * 0.3)
                nodal[3] += rng.normal(0.0, 0.01, size=(timesteps, nodes)).astype(np.float32)
                group.create_dataset("nodal_data", data=nodal)
                edges = np.stack([np.arange(nodes), (np.arange(nodes) + 1) % nodes]).astype(np.int64)
                group.create_dataset("mesh_edge", data=edges)
    if not conditions.exists():
        with conditions.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for sample_index in range(8):
                writer.writerow([0.5 + 0.1 * sample_index, sample_index / 7.0])

    method_dataset = "../frontend/runtime/simulgen-smoke/fixed_geometry.h5"
    method_conditions = "../frontend/runtime/simulgen-smoke/conditions.csv"
    method_vae = "../frontend/runtime/simulgen-smoke/simulgenvae_vae.pth"
    method_lc = "../frontend/runtime/simulgen-smoke/simulgenvae_lc.pth"
    config = "\n".join(
        [
            "model simulgenvae",
            "mode train_vae",
            "gpu_ids -1",
            "parallel_mode single",
            "split_seed 42",
            f"dataset_dir {method_dataset}",
            f"vae_modelpath {method_vae}",
            f"lc_modelpath {method_lc}",
            f"param_dir {method_conditions}",
            "lc_data_type csv",
            "num_var 1",
            "field_start_row 3",
            "node_start 0",
            "node_end 0",
            "timesteps_reduced 0",
            "latent_dim 2",
            "latent_dim_end 4",
            "num_filter_enc 16 8",
            "network_size small",
            "loss_type 1",
            "alpha 1",
            "init_beta_divisor 4",
            "training_epochs 1",
            "batch_size 2",
            "learningr 0.001",
            "weight_decay 0",
            "num_workers 0",
            "val_interval 1",
            "use_amp False",
            "use_ema False",
            "lc_filter 8 8",
            "lc_dropout 0.1",
            "output_dir ../frontend/runtime/simulgen-smoke/reconstruction",
            "log_file_dir ../frontend/runtime/simulgen-smoke/vae.log",
        ]
    ) + "\n"
    return {
        "dataset": relative(dataset),
        "conditions": relative(conditions),
        "vae_checkpoint": relative(root / "simulgenvae_vae.pth"),
        "lc_checkpoint": relative(root / "simulgenvae_lc.pth"),
        "config": config,
        "mode": "train_vae",
        "scientific_use": False,
        "note": "Deterministic tiny fixture for exercising the real SimulGen VAE path; not scientific evidence.",
    }


def create_inference_job(payload: dict[str, Any]) -> dict[str, Any]:
    checkpoint = safe_repo_path(
        str(payload.get("checkpoint", "")),
        (SUITE_ROOT / "output", SUITE_ROOT / "outputs", RUNTIME_ROOT),
    )
    if not checkpoint.is_file() or checkpoint.suffix.lower() not in {".pth", ".pt", ".ckpt"}:
        raise ValueError("Select an existing .pth, .pt, or .ckpt checkpoint.")
    output_root = RUNTIME_ROOT / "inference"
    output_root.mkdir(parents=True, exist_ok=True)
    output_name = slug(str(payload.get("output_name") or f"inference-{int(time.time())}"))
    output = output_root / output_name
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SUITE_ROOT / "inference" / "run_inference.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cpu",
    ]
    input_value = str(payload.get("input", "")).strip()
    if input_value:
        input_path = safe_repo_path(input_value, (SUITE_ROOT / "dataset", RUNTIME_ROOT))
        if not input_path.is_file():
            raise ValueError("Selected inference input does not exist.")
        command.extend(["--input", str(input_path)])
    option_map = {
        "timesteps": "--timesteps",
        "query_chunk_size": "--query-chunk-size",
        "num_samples": "--num-samples",
        "ode_steps": "--ode-steps",
        "cfg_scale": "--cfg-scale",
        "mc_resolution": "--mc-resolution",
        "seed": "--seed",
        "split_seed": "--split-seed",
        "cond_values": "--cond-values",
    }
    for key, flag in option_map.items():
        value = payload.get(key)
        if value is not None and str(value).strip() != "":
            command.extend([flag, str(value).strip()])
    return STATE.create_command_job(
        label=f"Portable inference · {checkpoint.name}",
        step_label="CPU inference bundle",
        command=command,
        cwd=SUITE_ROOT / "inference",
    )


def create_exe_build_job() -> dict[str, Any]:
    if importlib.util.find_spec("PyInstaller") is None:
        raise ValueError("PyInstaller is not installed in the current Studio interpreter.")
    deploy_root = RUNTIME_ROOT / "deploy"
    dist_path = deploy_root / "dist"
    work_path = deploy_root / "build"
    dist_path.mkdir(parents=True, exist_ok=True)
    work_path.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        "pyinstaller.spec",
    ]
    return STATE.create_command_job(
        label="Build portable AI-CAE4ALL inference .exe",
        step_label="PyInstaller bundle",
        command=command,
        cwd=SUITE_ROOT / "inference",
    )
