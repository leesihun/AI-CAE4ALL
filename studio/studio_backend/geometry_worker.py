"""Resident native SDFFlow process for the Studio geometry explorer.

This process owns SDFFlow's native import namespace. Studio and HI-MGN never
import its ``general_modules`` or ``model`` packages into their interpreters.
Each generate request integrates its exact (unquantized) alpha and decodes a
new SDF; this worker does not read prerecorded demo geometry.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback


SUITE_ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    "car": {
        "title": "Car / DrivAerML",
        "vae": "output/geometry_generation/ex2_drivaerml/sdfflow/vae.pth",
        "fm": "output/geometry_generation/ex2_drivaerml/sdfflow/fm_best.pth",
        "conditioning": "unconditional",
        # DrivAerML morphs one body, so most noise pairs differ by a few
        # percent; this pair spans the widest/tallest vs. narrowest body.
        "endpoints": {"seed": 3, "index_a": 3, "index_b": 2},
    },
    "deepjeb": {
        "title": "DeepJEB bracket",
        "vae": "output/geometry_generation/studio/sdfflow_vae.pth",
        "fm": "output/geometry_generation/studio/sdfflow_fm.pth",
        "conditioning": "unconditional",
        # Skeletal bracket -> filled plate (volume 0.14 -> 0.34).
        "endpoints": {"seed": 42, "index_a": 0, "index_b": 13},
    },
    "mcb": {
        "title": "MCB hexagonal nuts",
        "vae": "output/geometry_generation/ex3_mcb/sdfflow/vae.pth",
        "fm": "output/geometry_generation/ex3_mcb/sdfflow/fm_best.pth",
        "conditioning": "partial hexagonal-nut class condition",
        "cond_values": ("nan", "nan", "nan", "nan", "nan", 1, 0, 0, 0, 0, "nan"),
    },
}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeGeometryWorker:
    def __init__(self, suite_root: Path = SUITE_ROOT):
        self.root = Path(suite_root).resolve()
        sys.path.insert(0, str(self.root / "methods" / "SDFFlow"))
        import numpy as np
        import torch
        from general_modules.mesh_extraction import decode_sdf_grid, mesh_report, sdf_grid_to_mesh
        from inference_profiles.interpolate import slerp
        from inference_profiles.sample import load_fm_stack, normalize_condition_request, parse_condition_values
        from model.velocity_net import sample_latents

        self.np, self.torch = np, torch
        self.decode, self.report, self.marching_cubes = decode_sdf_grid, mesh_report, sdf_grid_to_mesh
        self.slerp, self.load_stack, self.sample = slerp, load_fm_stack, sample_latents
        self.normalize_condition, self.parse_condition = normalize_condition_request, parse_condition_values
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.stacks: dict = {}
        self.active_profile: str | None = None
        self.provenance: dict = {}

    def _move(self, stack, device: str) -> None:
        stack.vae.to(device)
        stack.model.to(device)
        stack.latent_mean = stack.latent_mean.to(device)
        stack.latent_std = stack.latent_std.to(device)
        stack.device = device

    def park(self) -> dict:
        """Keep checkpoints resident in host RAM while native HI-MGN uses GPU."""
        for stack in self.stacks.values():
            self._move(stack, "cpu")
        self.active_profile = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        return {"resident_profiles": list(self.stacks), "active_device": "cpu"}

    def _stack(self, profile: str, progress):
        if self.active_profile != profile:
            self.park()
        if profile not in self.stacks:
            spec = PROFILES[profile]
            missing = [str(self.root / spec[k]) for k in ("fm", "vae") if not (self.root / spec[k]).is_file()]
            if missing:
                raise FileNotFoundError("Required geometry checkpoint(s) are missing: " + ", ".join(missing))
            progress("loading_models", profile=profile)
            # CPU loading keeps optimizer and checkpoint tensors off the GPU.
            # Only model modules and latent statistics are moved to CUDA.
            stack = self.load_stack(str(self.root / spec["fm"]), str(self.root / spec["vae"]), device="cpu")
            # Native training checkpoints also carry optimizer/EMA copies.
            # The serving modules already own their weights; retain only the
            # small normalization/condition metadata in resident host memory.
            stack.fm_ckpt = {key: stack.fm_ckpt[key] for key in (
                "config", "cond_names", "cond_mean", "cond_std", "cond_min", "cond_max",
                "cond_clip", "latent_mean", "latent_std", "cond_dim", "latent_flat_dim",
            ) if key in stack.fm_ckpt}
            stack.vae_ckpt = None
            self.stacks[profile] = stack
            self.provenance[profile] = {
                "checkpoint": spec["fm"], "vae_checkpoint": spec["vae"],
                "checkpoint_sha256": sha256(self.root / spec["fm"]),
                "vae_checkpoint_sha256": sha256(self.root / spec["vae"]),
                "native_source_sha256": {
                    path: sha256(self.root / path) for path in (
                        "methods/SDFFlow/inference_profiles/interpolate.py",
                        "methods/SDFFlow/model/velocity_net.py",
                        "methods/SDFFlow/general_modules/mesh_extraction.py",
                    )
                },
            }
        stack = self.stacks[profile]
        self._move(stack, self.device)
        self.active_profile = profile
        return stack

    def generate(self, payload: dict, output_dir: Path, progress=lambda phase, **data: None) -> dict:
        started = time.perf_counter()
        profile = payload["profile"]
        spec = PROFILES[profile]
        stack = self._stack(profile, progress)
        torch, np = self.torch, self.np
        alpha = float(payload["alpha"])
        seed, a, b = int(payload["seed"]), int(payload["index_a"]), int(payload["index_b"])
        batch = int(payload["source_num_samples"])
        resolution, steps = int(payload["mc_resolution"]), int(payload["ode_steps"])
        output_dir.mkdir(parents=True, exist_ok=True)
        timings = {"model_ready_s": time.perf_counter() - started}
        with torch.inference_mode():
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noise = torch.randn(batch, stack.latent_flat_dim, device=self.device, generator=generator)
            row = noise[a] if alpha == 0.0 else noise[b] if alpha == 1.0 else self.slerp(noise[a], noise[b], alpha)
            kwargs = {}
            condition_request = None
            if spec.get("cond_values"):
                raw = self.parse_condition(spec["cond_values"], stack.cond_names)
                condition, mask, condition_request = self.normalize_condition(raw, stack.fm_ckpt, {})
                kwargs["cond"] = condition.reshape(1, -1).to(self.device)
                kwargs["cfg_scale"] = 1.0
                if condition_request["partial"]:
                    kwargs["cond_mask"] = mask.reshape(1, -1).to(self.device)
            progress("flow_integration", alpha=alpha, ode_steps=steps)
            tick = time.perf_counter()
            z_normalized = self.sample(stack.model, 1, stack.latent_flat_dim, self.device,
                                       ode_steps=steps, noise=row.reshape(1, -1), **kwargs)
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            timings["flow_s"] = time.perf_counter() - tick
            z = z_normalized * stack.latent_std + stack.latent_mean
            progress("sdf_decode", mc_resolution=resolution)
            tick = time.perf_counter()
            volume = self.decode(stack.vae, z, resolution=resolution, device=self.device)
            timings["sdf_s"] = time.perf_counter() - tick
        progress("marching_cubes", mc_resolution=resolution)
        tick = time.perf_counter()
        mesh = self.marching_cubes(volume)
        report = self.report(mesh)
        if mesh is None or not report["valid"]:
            raise RuntimeError("The generated SDF has no zero crossing; choose another alpha or seed.")
        if not np.isfinite(mesh.vertices).all():
            raise RuntimeError("Native geometry contains non-finite vertices.")
        timings["marching_cubes_s"] = time.perf_counter() - tick
        stl = output_dir / "geometry.stl"
        mesh.export(stl)
        provenance = {
            **self.provenance[profile],
            "method": "native SDFFlow noise SLERP -> flow ODE -> SDF -> Marching Cubes",
            "fresh_model_computation": True, "interpolation_space": "slerp_noise",
            "profile": profile, "alpha": alpha, "seed": seed,
            "index_a": a, "index_b": b, "source_num_samples": batch,
            "ode_steps": steps, "mc_resolution": resolution, "device": self.device,
            "conditioning": spec["conditioning"], "condition_request": condition_request,
            "geometry_sha256": sha256(stl),
        }
        mesh_data = {
            "name": f"{spec['title']} · alpha {alpha:g}", "profile": profile, "alpha": alpha,
            "vertices": np.asarray(mesh.vertices, dtype=np.float32).round(7).tolist(),
            "faces": np.asarray(mesh.faces, dtype=np.int32).tolist(),
            "watertight": report["watertight"], "volume": report["volume"],
            "coordinate_system": "original xyz; z-up", "provenance": provenance,
        }
        write_json(output_dir / "mesh.json", mesh_data)
        timings["total_s"] = time.perf_counter() - started
        metadata = {"mesh_path": str(output_dir / "mesh.json"), "stl_path": str(stl),
                    "report": report, "provenance": provenance, "timings": timings,
                    "resident_profiles": list(self.stacks)}
        write_json(output_dir / "generation.json", metadata)
        return metadata


def main() -> None:
    protocol = sys.stdout
    worker = None
    def emit(value):
        protocol.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        protocol.flush()
    emit({"type": "ready", "pid": __import__("os").getpid()})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            request_id = request["id"]
            with contextlib.redirect_stdout(sys.stderr):
                if worker is None:
                    worker = NativeGeometryWorker()
                if request.get("operation") == "park":
                    result = worker.park()
                else:
                    progress = lambda phase, **data: emit({"type": "progress", "id": request_id,
                                                            "phase": phase, **data})
                    result = worker.generate(request["payload"], Path(request["output_dir"]), progress)
            emit({"type": "completed", "id": request_id, "result": result})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"type": "failed", "id": locals().get("request_id"), "error": str(exc)})


if __name__ == "__main__":
    main()
