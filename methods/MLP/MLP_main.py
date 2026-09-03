"""Launcher entrypoint: run the MLP surrogate from a flat ``--config`` file.

Invoked by ``AI_CAE4ALL_main.py`` after preflight, exactly like every method's
native entrypoint: ``python MLP/MLP_main.py --config <file>``. The launcher runs
this with the working directory set to this repository (``MLP/``), so relative
config paths resolve from here.

Config keys (see configs/MLP/*.txt and CONFIGURATION_REFERENCE.md section 9.10):
    model mlp                  mode train|inference
    dataset_dir <train.h5>     infer_dataset <infer.h5>   modelpath <ckpt.pth>
    input_var N                output_var M
    hidden_layers 256,256,128  activation gelu   dropout 0.0   norm none
    input_normalization standard   output_normalization standard
    loss mse   training_epochs 200   batch_size 32   learningr 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put this repo on sys.path so the local ``mlp`` package imports regardless of
# the launcher's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mlp.config import load_config, params_from_config, validate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MLP parametric surrogate (config-driven launcher entrypoint)")
    ap.add_argument("--config", required=True, help="Path to a flat key/value config file")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    params = params_from_config(cfg)
    validate(params)

    print("MLP Surrogate")
    print(f"Config : {Path(args.config).resolve()}")
    print(f"Mode   : {params.mode}")
    print()

    if params.mode == "train":
        from mlp.train import train
        return train(params, args.config)
    if params.mode == "inference":
        from mlp.infer import infer
        return infer(params, args.config)
    raise SystemExit(f"Unsupported mode {params.mode!r}; expected train or inference.")


if __name__ == "__main__":
    raise SystemExit(main())
