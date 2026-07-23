"""Explicit-args refactor of Geometry_generation/inference_profiles/sample.py's
`run_sample`. Generative, not rollout: noise -> FM ODE (optional condition +
CFG) -> SDF-VAE decode -> Marching Cubes -> STL. `input` is unused (kept in
the signature for a uniform driver contract across families).

Single combined checkpoint (INFERENCE_BUNDLE_PLAN.md section 5.5): the FM
`.pth` embeds its frozen VAE under `ckpt['vae']`. Old two-file checkpoints
(pre-merge) are supported via a fallback to `ckpt['vae_modelpath']` so this
driver also runs against checkpoints trained before the training-side change,
as long as that path is still reachable.
"""

import json
import os
import time

import numpy as np
import torch

from general_modules.mesh_extraction import decode_sdf_grid, sdf_grid_to_mesh, mesh_report
from model.sdf_vae import SDFVAE
from model.velocity_net import VelocityNet, sample_latents


def _model_state(ckpt):
    """Prefer EMA weights; strip AveragedModel 'module.' prefix."""
    state = ckpt.get("ema_state") or ckpt["model_state"]
    if ckpt.get("ema_state") is not None:
        state = {k.replace("module.", "", 1): v for k, v in state.items() if k != "n_averaged"}
    return state


def _load_vae(fm_ckpt, device):
    """Rebuild the SDF-VAE from the embedded `ckpt['vae']` block, falling back
    to `ckpt['vae_modelpath']` for pre-merge (two-file) checkpoints."""
    vae_block = fm_ckpt.get("vae")
    if vae_block is not None:
        vae = SDFVAE(vae_block["config"]).to(device)
        vae.load_state_dict(_model_state(vae_block))
    else:
        vae_path = fm_ckpt.get("vae_modelpath")
        if not vae_path or not os.path.exists(vae_path):
            raise FileNotFoundError(
                "This checkpoint has no embedded 'vae' block (pre-merge FM checkpoint) "
                f"and its recorded vae_modelpath ({vae_path!r}) is not reachable. Run "
                "Geometry_generation/merge_sdfflow_checkpoint.py to produce a single "
                "self-contained .pth, or make the VAE file available at that path."
            )
        vae_ckpt = torch.load(vae_path, map_location="cpu", weights_only=False)
        vae = SDFVAE(vae_ckpt["config"]).to(device)
        vae.load_state_dict(_model_state(vae_ckpt))
    vae.eval()
    return vae


def run(checkpoint: str, input: str, output: str, device: torch.device,
        timesteps: int = None, query_chunk_size: int = 0, num_samples: int = 1,
        ode_steps: int = 50, cfg_scale: float = 1.0, mc_resolution: int = 128,
        seed: int = None, cond_values=None, **_ignored) -> str:
    print(f"Loading SDFFlow checkpoint: {checkpoint}")
    fm_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if fm_ckpt.get("stage") != "fm":
        raise ValueError(f"'{checkpoint}' is not an FM checkpoint (stage={fm_ckpt.get('stage')!r}).")

    vae = _load_vae(fm_ckpt, device)

    latent_flat_dim = int(fm_ckpt["latent_flat_dim"])
    cond_dim = int(fm_ckpt["cond_dim"])
    model = VelocityNet(fm_ckpt["config"], latent_flat_dim, cond_dim=cond_dim).to(device)
    model.load_state_dict(_model_state(fm_ckpt))
    model.eval()

    seed = 0 if seed is None else int(seed)
    os.makedirs(output, exist_ok=True)

    cond = None
    target = cond_std_np = None
    if cond_values is not None and cond_dim > 0:
        if not isinstance(cond_values, list):
            cond_values = [float(v) for v in str(cond_values).split(",")]
        if len(cond_values) != cond_dim:
            raise ValueError(f"--cond-values must have {cond_dim} entries "
                              f"({fm_ckpt['cond_names']}), got {len(cond_values)}")
        raw = torch.tensor([float(v) for v in cond_values], dtype=torch.float32)
        cond_mean = fm_ckpt["cond_mean"].squeeze(0).cpu()
        cond_std = fm_ckpt["cond_std"].squeeze(0).cpu()
        cond_n = (raw - cond_mean) / cond_std
        max_condition_z = float(fm_ckpt.get("cond_clip") or 5.0)
        excessive = cond_n.abs() > max_condition_z
        if excessive.any():
            details = ", ".join(
                f"{fm_ckpt['cond_names'][i]}={float(cond_n[i]):.2f} sigma"
                for i in torch.where(excessive)[0].tolist())
            raise ValueError(f"Condition request exceeds max_condition_z={max_condition_z:g}: {details}")
        cond = cond_n.unsqueeze(0).repeat(num_samples, 1).to(device)
        target = raw.numpy().astype(np.float64)
        cond_std_np = cond_std.numpy().astype(np.float64)
        print(f"Conditional generation: {dict(zip(fm_ckpt['cond_names'], cond_values))} "
              f"(cfg_scale={cfg_scale})")
    else:
        print("Unconditional generation")

    generator = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    z_n = sample_latents(model, num_samples, latent_flat_dim, device,
                          cond=cond, cfg_scale=cfg_scale, ode_steps=ode_steps,
                          generator=generator)
    z = z_n * fm_ckpt["latent_std"].to(device) + fm_ckpt["latent_mean"].to(device)
    print(f"Sampled {num_samples} latent(s) in {time.time() - t0:.2f}s ({ode_steps} ODE steps)")

    results = []
    last_path = None
    for i in range(num_samples):
        volume = decode_sdf_grid(vae, z[i:i + 1], resolution=mc_resolution, device=device)
        mesh = sdf_grid_to_mesh(volume)
        report = mesh_report(mesh)
        report["index"] = i
        if report["valid"]:
            path = os.path.join(output, f"sample_{seed}_{i:03d}.stl")
            mesh.export(path)
            report["path"] = path
            last_path = path
            print(f"  sample {i:03d}: watertight={report['watertight']} "
                  f"faces={report['faces']} -> {path}")
        else:
            print(f"  sample {i:03d}: NO ZERO CROSSING (rejected)")
        results.append(report)

    meta = {
        "checkpoint": checkpoint, "seed": seed, "cfg_scale": cfg_scale,
        "ode_steps": ode_steps, "mc_resolution": mc_resolution,
        "cond_names": fm_ckpt["cond_names"], "cond_values": cond_values,
        "results": results,
    }
    meta_path = os.path.join(output, f"sample_{seed}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    valid = sum(1 for r in results if r["valid"])
    print(f"\nDone: {valid}/{num_samples} valid. Metadata: {meta_path}")
    return last_path or meta_path
