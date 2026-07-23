"""argparse surface -> cae_infer.infer(...). Superset of flags; each family
driver ignores what it doesn't use (see README.md for the per-family subset).
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_inference",
        description="Stand-alone CPU inference for AI-CAE4ALL checkpoints "
                     "(point_deeponet, deeponet, fno, gino, transolver, "
                     "meshgraphnets, meshgraphnets-v, sdfflow). The family is "
                     "auto-detected from the checkpoint -- just point it at a .pth.",
    )
    p.add_argument("--checkpoint", required=True, help="Path to the .pth checkpoint.")
    p.add_argument("--input", default=None,
                    help="Input HDF5 mesh dataset (MGN contract). Not used by sdfflow.")
    p.add_argument("--output", required=True,
                    help="Output .h5 (rollout families) or directory (sdfflow STLs).")
    p.add_argument("--device", default="cpu", choices=["cpu", "auto"],
                    help="This bundle is CPU-only; kept for CLI parity.")
    p.add_argument("--timesteps", type=int, default=None,
                    help="Rollout steps (default: full trajectory from the input file).")
    p.add_argument("--query-chunk-size", type=int, default=0, dest="query_chunk_size",
                    help="Neural_Operator memory control for point/query decode (0 = no chunking).")
    p.add_argument("--num-samples", type=int, default=1, dest="num_samples",
                    help="sdfflow only: number of geometries to sample.")
    p.add_argument("--ode-steps", type=int, default=50, dest="ode_steps",
                    help="sdfflow only: flow-matching ODE steps.")
    p.add_argument("--cfg-scale", type=float, default=1.0, dest="cfg_scale",
                    help="sdfflow only: classifier-free guidance scale.")
    p.add_argument("--mc-resolution", type=int, default=128, dest="mc_resolution",
                    help="sdfflow only: Marching Cubes grid resolution.")
    p.add_argument("--seed", type=int, default=None,
                    help="sdfflow only: sampler seed for reproducibility.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    from cae_infer import infer

    try:
        result = infer(
            checkpoint=args.checkpoint, input=args.input, output=args.output,
            device=args.device, timesteps=args.timesteps,
            query_chunk_size=args.query_chunk_size, num_samples=args.num_samples,
            ode_steps=args.ode_steps, cfg_scale=args.cfg_scale,
            mc_resolution=args.mc_resolution, seed=args.seed,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. Output: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
