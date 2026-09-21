"""Pack the production campaign's draws into the suite's shared mesh HDF5
contract (docs/reference/DATASET_FORMAT.md), row layout per main.tex Sec 4.

    data/{sample_id}/nodal_data   [8, 1, num_nodes]
    data/{sample_id}/mesh_edge    [2, num_edges]
    latent/{sample_id}/...        dimple list + sigma -- diagnostic only

Rows (num_timesteps=1: one static post-buckling end state per draw):

    0:3   reference coordinates of the NOMINAL (perfect) cylinder   input, never predicted
    3:6   output: final displacement ux,uy,uz (buckled - nominal)   input AND output
    6     condition: nodal thickness (uniform per geometry)         input-only
    7     condition: node type (0 interior, 1 clamped, 2 loaded)    input-only

input_var=3, output_var=3, cond_var=2. The old 11-row layout (deleted at
07043c4, git history only) had a `6:9 Component A signature offset` row from
the earlier Matern-spectral imperfection law; main.tex's unified single
RSA-dimple mechanism has nothing deterministic left to store there, so that
row is gone and thickness/node_type shifted from 9,10 down to 6,7.

Displacement is measured against the ANALYTIC nominal geometry
(geometry.ShellMesh(...).coords * length_scale), never the draw's own t=0
ANIM frame -- that frame is already imperfect (mesh.coords + imperfection
displacement along the normal), so diffing against it would silently cancel
out the imperfection's own contribution to the field. The nominal reference
is cheap to regenerate in closed form and does not depend on the draw.

Only the LAST ANIM frame is converted per draw (t=0 is never read from disk):
run_batch.py's production decks keep exactly 2 frames (`anim_dt=tend`), and
the first one is discarded here anyway since the nominal reference replaces
it.

The dimple list {theta_k, z_k, a_k} + sigma is never persisted by
run_batch.run_one_draw (only 4 scalar diagnostics land in summary.json), but
it is fully reproducible: apply_imperfection() is a pure function of
(mesh, rng), and run_one_draw's RSA retry loop seeds rng as
np.random.default_rng((seed, attempt)) for attempt=0,1,2,... trying the next
attempt only when RSA placement fails outright. Replaying that identical
loop here regenerates a byte-identical draw. Each replay is cross-checked
against summary.json's K/sigma/max_abs_over_t/rms_over_t as a correctness
guard -- a mismatch would mean the replay diverged from the original attempt
index, and the sample is kept (nodal_data does not depend on the replay) but
its latent/ group is skipped with a warning.

Run on aarl only (buckling-campaign-aarl-only): ANIM->VTK conversion needs
the vendor `anim_to_vtk_linux64_gf` binary from the Linux OpenRadioss install.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry
import imperfection
from run_batch import RADIOSS_ROOT

ANIM_TO_VTK = RADIOSS_ROOT / "exec" / "anim_to_vtk_linux64_gf"

FEATURE_NAMES = ["x", "y", "z", "ux", "uy", "uz", "thickness", "node_type"]
ROW_LAYOUT = ("0:3 nominal reference coords | 3:6 final displacement (output) | "
              "6 thickness | 7 node type (0 interior, 1 clamped, 2 loaded)")


def read_vtk_points(path):
    """Minimal ASCII-VTK point reader (mirrors load_verify.py) -- aarl's bare
    python3 has numpy but no pyvista, and only point coordinates are needed."""
    with open(path) as f:
        tok = f.read().split()
    i = tok.index("POINTS")
    n = int(tok[i + 1])
    vals = np.array(tok[i + 3: i + 3 + 3 * n], dtype=np.float64)
    return vals.reshape(n, 3)


def anim_frames(draw_dir, run_name="draw"):
    return sorted(draw_dir.glob(f"{run_name}A[0-9][0-9][0-9]"))


def final_frame_points(draw_dir, run_name, tmp_dir, tag):
    """Convert only the LAST kept ANIM frame (t=tend) and return its node
    coordinates in physical (length_scale-scaled) units, node-index-aligned
    with geometry.ShellMesh.coords -- same alignment frame_radial_profile
    (load_verify.py) already relies on.

    `tag` must be unique per caller: every draw's last frame is literally
    named f"{run_name}A002" (run_name="draw" for all production draws), so
    a tmp path keyed only on the frame name collides across the concurrent
    draws of a shape -- concurrent threads then tear each other's VTK file
    mid-write/mid-read."""
    frames = anim_frames(draw_dir, run_name)
    if not frames:
        return None
    fr = frames[-1]
    vtk_path = tmp_dir / f"{tag}_{fr.name}.vtk"
    with open(vtk_path, "w") as fh:
        subprocess.run([str(ANIM_TO_VTK), fr.name], cwd=fr.parent,
                        stdout=fh, stderr=subprocess.DEVNULL, check=True)
    pts = read_vtk_points(vtk_path)
    vtk_path.unlink(missing_ok=True)
    return pts


def mesh_edges(elements):
    """Unique undirected edges from 4-node quad connectivity, as [2, E]
    int64 -- the loader (MeshGraphDataset) builds the bidirectional graph
    representation itself (DATASET_FORMAT.md), so only one direction is
    stored here."""
    a, b = [], []
    for el in elements:
        for k in range(4):
            u, v = int(el[k]), int(el[(k + 1) % 4])
            if u != v:
                a.append(min(u, v))
                b.append(max(u, v))
    e = np.unique(np.stack([a, b], axis=1), axis=0)
    return np.ascontiguousarray(e.T.astype(np.int64))


def node_type_row(mesh):
    nt = np.zeros(mesh.n_nodes, dtype=np.float32)
    nt[mesh.bottom_nodes] = 1.0
    nt[mesh.top_nodes] = 2.0
    return nt


def replay_imperfection(mesh, seed, max_attempts=50):
    """Reproduce run_batch.run_one_draw's exact RSA retry loop -- same
    (seed, attempt) rng stream on the same mesh regenerates a byte-identical
    draw, since apply_imperfection is a pure function of (mesh, rng)."""
    for attempt in range(max_attempts):
        rng = np.random.default_rng((seed, attempt))
        try:
            return imperfection.apply_imperfection(mesh, rng)
        except RuntimeError:
            continue
    return None, None


def pack_draw(draw_dir, seed, mesh, nominal_coords, thickness_row, ntype_row,
              length_scale, tmp_dir):
    """Returns (rows[8,N], diag_dict, latent_dict_or_None) for one draw, or
    None if the draw has no summary.json (failed or not-yet-run) or its ANIM
    frame is missing/unreadable."""
    summary_path = draw_dir / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())

    pts = final_frame_points(draw_dir, "draw", tmp_dir, tag=f"{draw_dir.parent.name}_d{seed:03d}")
    if pts is None:
        return None
    if pts.shape[0] == mesh.n_nodes + 2:
        # ANIM appends the 2 RBODY master/reference nodes (bottom + top rim
        # rigid bodies, on the centerline at z=0 and z=L) after the shell's
        # own nodes -- confirmed by inspection: last 2 points sit at
        # x=y~0 with z matching the two rim planes. Not real shell nodes.
        pts = pts[:mesh.n_nodes]
    elif pts.shape[0] != mesh.n_nodes:
        return None
    disp = (pts - nominal_coords).astype(np.float32)          # [N, 3], physical units

    rows = np.concatenate([
        nominal_coords.T.astype(np.float32),   # 0:3
        disp.T,                                # 3:6
        thickness_row[None, :],                # 6
        ntype_row[None, :],                    # 7
    ], axis=0)

    # Raw full-vector displacement is dominated by the imposed axial
    # end-shortening (nearly every node translates down by roughly the same
    # amount under the displacement-controlled load), which is essentially
    # constant across draws of one shape -- confirmed on shape 000: 100
    # draws span max_deform_over_t = 107.25..107.98, <1% spread. That makes
    # it a poor "how differently did this buckle" signal. The radial
    # deviation from the (untapered, so z-independent) nominal radius
    # isolates the buckling MODE SHAPE from that uniform axial translation
    # and is what actually varies draw-to-draw with the imperfection.
    mag = np.linalg.norm(disp, axis=1)
    r_nom = np.linalg.norm(nominal_coords[:, :2], axis=1)
    r_now = np.linalg.norm(pts[:, :2], axis=1)
    radial_dev = r_now - r_nom
    t_phys = float(thickness_row[0])
    diag = dict(
        seed=seed, K=summary["K"], sigma=summary["sigma"],
        max_abs_over_t=summary["max_abs_over_t"], rms_over_t=summary["rms_over_t"],
        final_time=summary["final_time"], ke_ie_ratio=summary["ke_ie_ratio"],
        max_deform=float(mag.max()), max_deform_over_t=float(mag.max() / t_phys),
        rms_deform_over_t=float(np.sqrt((mag ** 2).mean()) / t_phys),
        max_radial_over_t=float(np.abs(radial_dev).max() / t_phys),
        rms_radial_over_t=float(np.sqrt((radial_dev ** 2).mean()) / t_phys),
    )

    _, record = replay_imperfection(mesh, seed)
    latent = None
    if record is not None:
        ok = (record["K"] == summary["K"]
              and abs(record["sigma"] - summary["sigma"]) < 1e-9
              and abs(record["max_abs_over_t"] - summary["max_abs_over_t"]) < 1e-6
              and abs(record["rms_over_t"] - summary["rms_over_t"]) < 1e-6)
        if ok:
            latent = record
        else:
            print(f"    WARNING seed={seed}: RSA replay mismatch, skipping latent/", flush=True)
    else:
        print(f"    WARNING seed={seed}: RSA replay never converged, skipping latent/", flush=True)

    return rows, diag, latent


def pack_shape(h5_data, h5_latent, run_root, idx, tier, r_over_t, l_over_r,
               n_draws, length_scale, tmp_dir, max_workers):
    mesh = geometry.ShellMesh(r_over_t=r_over_t, l_over_r=l_over_r)
    nominal_coords = (mesh.coords * length_scale).astype(np.float64)
    thickness_row = np.full(mesh.n_nodes, mesh.t * length_scale, dtype=np.float32)
    ntype_row = node_type_row(mesh)
    edges = mesh_edges(mesh.elements)
    shape_name = f"{idx:03d}_{tier}_rt{r_over_t:.1f}_lr{l_over_r:.3f}"

    n_written, n_missing, diags = 0, 0, []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(pack_draw, run_root / f"draw_{i:03d}", i, mesh,
                      nominal_coords, thickness_row, ntype_row, length_scale,
                      tmp_dir): i
            for i in range(n_draws)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            out = fut.result()
            if out is None:
                n_missing += 1
                continue
            rows, diag, latent = out
            sample_id = f"{shape_name}_d{i:03d}"
            grp = h5_data.create_group(sample_id)
            grp.create_dataset("nodal_data", data=rows[:, None, :],
                               compression="gzip", compression_opts=4)
            grp.create_dataset("mesh_edge", data=edges,
                               compression="gzip", compression_opts=4)
            grp.attrs["shape_idx"] = idx
            grp.attrs["tier"] = tier
            grp.attrs["r_over_t"] = r_over_t
            grp.attrs["l_over_r"] = l_over_r
            grp.attrs["n_nodes"] = mesh.n_nodes
            grp.attrs["n_elements"] = len(mesh.elements)
            grp.attrs["input_var"] = 3
            grp.attrs["output_var"] = 3
            grp.attrs["cond_var"] = 2
            grp.attrs["num_timesteps"] = 1
            for k, v in diag.items():
                grp.attrs[k] = v
            if latent is not None:
                lg = h5_latent.create_group(sample_id)
                lg.create_dataset("centers", data=latent["centers"])
                lg.create_dataset("amps_physical", data=latent["amps_physical"])
                lg.create_dataset("amps_normalized", data=latent["amps_normalized"])
                lg.attrs["sigma"] = latent["sigma"]
                lg.attrs["K"] = latent["K"]
            n_written += 1
            diags.append(dict(sample_id=sample_id, shape_idx=idx, tier=tier,
                               r_over_t=r_over_t, l_over_r=l_over_r, **diag))
    print(f"  {shape_name}: {n_written}/{n_draws} packed ({n_missing} missing summary.json/ANIM)",
          flush=True)
    return n_written, n_missing, diags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root", help="production campaign root (contains manifest.json + shape dirs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0, help="shape index range start")
    ap.add_argument("--end", type=int, default=None, help="shape index range end (exclusive)")
    ap.add_argument("--length-scale", type=float, default=100.0)
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--tmp-dir", default=None)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    manifest = json.loads((runs_root / "manifest.json").read_text())
    if args.end is None:
        args.end = len(manifest)
    rows_to_pack = manifest[args.start:args.end]
    if not rows_to_pack:
        raise SystemExit("nothing to pack in the given --start/--end range")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else runs_root / "_pack_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_total, n_missing_total = 0, 0
    all_diags = []
    with h5py.File(out_path, "w") as h:
        d = h.create_group("data")
        lat = h.create_group("latent")
        for entry in rows_to_pack:
            n_written, n_missing, diags = pack_shape(
                d, lat, Path(entry["run_root"]), entry["idx"], entry["tier"],
                entry["r_over_t"], entry["l_over_r"], entry["draws"],
                args.length_scale, tmp_dir, args.max_workers,
            )
            n_total += n_written
            n_missing_total += n_missing
            all_diags.extend(diags)

        h.attrs["num_samples"] = n_total
        h.attrs["num_features"] = len(FEATURE_NAMES)
        h.attrs["num_timesteps"] = 1
        h.attrs["input_var"] = 3
        h.attrs["output_var"] = 3
        h.attrs["cond_var"] = 2
        h.attrs["row_layout"] = ROW_LAYOUT
        h.attrs["length_scale"] = args.length_scale
        h.attrs["n_shapes_packed"] = len(rows_to_pack)
        h.attrs["n_missing_draws"] = n_missing_total
        h.attrs["source_runs_root"] = str(runs_root.resolve())
        metadata = h.create_group("metadata")
        metadata.create_dataset("feature_names", data=FEATURE_NAMES,
                                dtype=h5py.string_dtype("utf-8"))

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    print(f"\nwrote {out_path}  {out_path.stat().st_size / 1e6:.1f} MB  "
          f"{n_total} samples over {len(rows_to_pack)} shapes  "
          f"({n_missing_total} missing draws)")
    if all_diags:
        radial = np.asarray([r["max_radial_over_t"] for r in all_diags])
        raw = np.asarray([r["max_deform_over_t"] for r in all_diags])
        print(f"max_radial_over_t (buckling-mode amplitude, isolates axial rigid "
              f"translation): min={radial.min():.3f} median={np.median(radial):.3f} "
              f"max={radial.max():.3f}")
        print(f"max_deform_over_t (raw full-vector, dominated by imposed axial "
              f"compression): min={raw.min():.3f} median={np.median(raw):.3f} "
              f"max={raw.max():.3f}")
        summary_path = out_path.with_suffix(".diagnostics.json")
        summary_path.write_text(json.dumps(all_diags))
        print(f"per-draw diagnostics (incl. both deformation metrics) dumped to "
              f"{summary_path} for histogramming")


if __name__ == "__main__":
    main()
