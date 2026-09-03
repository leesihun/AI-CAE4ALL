"""Host/GPU/deployment inventory: read-only shell probes, no suite imports."""

from __future__ import annotations

import importlib.util
import subprocess
from typing import Any

from studio_backend.paths import RUNTIME_ROOT, SUITE_ROOT, file_record, relative


# Public model IDs accepted by inference/run_inference.py.  Keep this separate
# from the five implementation families: the neural-operator driver serves four
# distinct model IDs, while each of the other drivers serves one.
PORTABLE_INFERENCE_MODELS = (
    "point_deeponet", "deeponet", "fno", "gino", "transolver",
    "meshgraphnets", "meshgraphnets-v", "sdfflow",
)
PORTABLE_DRIVER_FAMILIES = (
    "neural_operator", "transolver", "meshgraphnets", "meshgraphnets_v", "geometry",
)


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "memory_total_mb": parts[2],
                "memory_used_mb": parts[3],
                "utilization_percent": parts[4],
                "temperature_c": parts[5],
                "driver": parts[6],
            }
        )
    return rows


def deployment_status() -> dict[str, Any]:
    exe = SUITE_ROOT / "inference" / "dist" / "run_inference" / "run_inference.exe"
    runtime_exe = RUNTIME_ROOT / "deploy" / "dist" / "run_inference" / "run_inference.exe"
    selected = runtime_exe if runtime_exe.is_file() else exe
    return {
        "pyinstaller_available": importlib.util.find_spec("PyInstaller") is not None,
        "bundle_cli": relative(SUITE_ROOT / "inference" / "run_inference.py"),
        "existing_exe": file_record(selected, "executable") if selected.is_file() else None,
        "api_endpoint": "/api/inference/run",
        # `families` is retained for saved/custom clients from the first Studio
        # release, where this field actually contained model IDs.
        "families": list(PORTABLE_INFERENCE_MODELS),
        "models": list(PORTABLE_INFERENCE_MODELS),
        "driver_families": list(PORTABLE_DRIVER_FAMILIES),
    }
