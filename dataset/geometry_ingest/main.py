"""Launcher entrypoint: run geometry ingest from a flat ``--config`` file.

Invoked by ``AI_CAE4ALL_main.py`` after preflight, exactly like every method's
native entrypoint: ``python dataset/geometry_ingest/main.py --config <file>``.
The launcher runs this with the working directory set to this repository
(``dataset/geometry_ingest``), so relative config paths resolve from here.

Config keys (see configs/*.txt and CONFIGURATION_REFERENCE.md):
    model geometry_ingest      mode ingest|inspect
    input_geometry <dir>       output_dataset <file.h5>
    reader auto|trimesh|gmsh   mesh_type surface|volume
    emit graph[,pointcloud]    num_points N   resample_method fps|random
    num_fields K               mesh_size_max/min   seed   limit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put the parent (dataset/) on sys.path so this script can import the
# geometry_ingest package regardless of the launcher's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geometry_ingest.config import load_config, params_from_config  # noqa: E402
from geometry_ingest.pipeline import gather_paths, run_ingest        # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Geometry ingest (config-driven launcher entrypoint)")
    ap.add_argument("--config", required=True, help="Path to a flat key/value config file")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    mode = str(cfg.get("mode", "ingest")).lower()
    dry_run = mode == "inspect"

    input_geometry = cfg.get("input_geometry")
    if not input_geometry:
        raise SystemExit("input_geometry is required.")
    output = cfg.get("output_dataset")
    if not dry_run and not output:
        raise SystemExit("output_dataset is required for mode=ingest.")

    params = params_from_config(cfg)
    paths = gather_paths(input_geometry, params.limit)
    if not paths:
        raise SystemExit(f"No geometry files found under {input_geometry}")

    print("geometry_ingest")
    print(f"Config    : {Path(args.config).resolve()}")
    print(f"Mode      : {mode}")
    print(f"Input     : {input_geometry}  ({len(paths)} file(s))")
    print(f"Reader    : {params.reader}   mesh_type={'volume' if params.volume else 'surface'}   emit={','.join(params.emit)}")
    if not dry_run:
        print(f"Output    : {output}")
    print()
    return run_ingest(paths, params, output, dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
