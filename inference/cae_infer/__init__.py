"""Public API for the stand-alone inference bundle.

    from cae_infer import infer
    infer(checkpoint="model.pth", input="scene.h5", output="out/")

The checkpoint alone decides which of the five model families runs --
`detect_family` is deliberately dumb and explicit (see its docstring). Only
one family may be loaded per process (registry.py); this is never a problem
in practice because both entrypoints (the CLI and this function) handle one
checkpoint per invocation.
"""

import torch

from .registry import load_driver

FAMILIES = ("neural_operator", "transolver", "meshgraphnets", "meshgraphnets_v", "geometry")


def detect_family(checkpoint_path: str) -> str:
    """Classify a checkpoint by the keys its own training-side `save_checkpoint`
    writes -- no filename convention, no user-supplied hint required.

    - schema_version == 'deeponet_repo_v1'      -> neural_operator
      (point_deeponet / deeponet / fno / gino; checkpoint['selected_model']
      picks the exact architecture inside the family)
    - schema_version == 'sdfflow_infer_v1', or an SDFFlow-identified
      legacy 'stage' in {'vae', 'fm'}           -> geometry (SDFFlow)
    - 'checkpoint_version' present               -> transolver
    - 'model_config' present with 'use_vae' key  -> meshgraphnets_v
    - 'model_config' present (MGN-shaped, no
      'use_vae')                                 -> meshgraphnets

    cHI-MGNflow and SimulGen-VAE are rejected rather than being routed through
    the superficially similar MGN and stage='vae' signatures respectively.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    schema = ckpt.get("schema_version")
    if schema == "deeponet_repo_v1":
        return "neural_operator"
    config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    config_model = str(config.get("model", "")).lower().strip()
    sdfflow_keys = {"sdf_h5", "surface_points", "query_points", "fm_hidden", "fm_blocks"}
    # SimulGen-VAE also writes stage='vae'.  Stage alone is therefore not an
    # SDFFlow signature: require its explicit identity or geometry-only keys
    # for legacy stage checkpoints that predate schema_version.
    if schema == "sdfflow_infer_v1" or (
        ckpt.get("stage") in ("vae", "fm")
        and (config_model == "sdfflow" or sdfflow_keys.intersection(config))
    ):
        return "geometry"
    if "checkpoint_version" in ckpt:
        return "transolver"

    model_config = ckpt.get("model_config")
    if isinstance(model_config, dict) and "message_passing_num" in model_config:
        model_id = str(model_config.get("model", "")).lower().strip()
        flow_keys = {"flow_time_freqs", "flow_t_sampling", "flow_solver"}
        if model_id == "chi-mgnflow" or flow_keys.intersection(model_config):
            raise ValueError(
                "This is a cHI-MGNflow checkpoint. The portable CPU bundle does "
                "not contain the cHI flow driver; run it through AI-CAE4ALL's "
                "native cHI-MGNflow Inference block instead."
            )
        return "meshgraphnets_v" if "use_vae" in model_config else "meshgraphnets"

    raise ValueError(
        f"Could not classify checkpoint '{checkpoint_path}' as any of {FAMILIES}. "
        "Top-level keys found: " + ", ".join(sorted(ckpt.keys()))
    )


def infer(checkpoint: str, input: str = None, output: str = None, *,
          device: str = "cpu", timesteps: int = None, query_chunk_size: int = 0,
          **family_opts) -> str:
    """Detect family -> load that family's driver -> run it. Returns the
    output path (a file for rollout families, a directory for `geometry`).
    `input` is unused by the `geometry` family (it generates, not rolls out).
    """
    from .common.device import resolve_device

    family = detect_family(checkpoint)
    driver = load_driver(family)
    dev = resolve_device(device)
    return driver.run(
        checkpoint=checkpoint, input=input, output=output, device=dev,
        timesteps=timesteps, query_chunk_size=query_chunk_size, **family_opts,
    )
