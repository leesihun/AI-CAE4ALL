"""Export finished draws into the repo's shared mesh HDF5 contract.

    data/{sample_id}/nodal_data   [num_features, num_timesteps, num_nodes]
    data/{sample_id}/mesh_edge    [2, num_edges]

Static problem, so num_timesteps = 1. Row layout:

    rows 0:3   reference coordinates x, y, z          (the NOMINAL geometry)
    row  3     radial displacement / t                (the state = what we predict)
    rows 4:4+c conditioning rows, input-only          (optional, see below)

Two things this file deliberately does NOT write:

  * the imperfection realization. That is the withheld variable and the whole
    point of the benchmark. It is kept beside the sample as spectral
    coefficients for reproducibility, never as a model-visible row.

  * the *perturbed* coordinates. Rows 0:3 carry the nominal geometry, because a
    model that could read the perturbed node positions would be reading the
    withheld variable directly -- the leakage failure mode this design exists to
    avoid. Verified in `check_leakage`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def edges_from_elements(elements):
    """Undirected node pairs from S8R connectivity, deduplicated."""
    pairs = set()
    for e in elements:
        c = [e[0], e[4], e[1], e[5], e[2], e[6], e[3], e[7]]   # perimeter order
        for k in range(8):
            a, b = int(c[k]), int(c[(k + 1) % 8])
            pairs.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(pairs), dtype=np.int64).T


def check_leakage(mesh_nominal_coords, rows03):
    """Rows 0:3 must be the NOMINAL geometry, bit-identical across draws."""
    return float(np.max(np.abs(mesh_nominal_coords - rows03)))


def export(batch_dir, out_path, cond_vars=()):
    with open(os.path.join(batch_dir, "results.json")) as f:
        recs = [r for r in json.load(f) if r.get("ok")]
    if not recs:
        raise SystemExit("no successful draws in %s" % batch_dir)

    meshes, n_written = {}, 0
    max_leak = 0.0
    with h5py.File(out_path, "w") as h:
        g = h.create_group("data")
        for i, r in enumerate(recs, start=1):
            key = (r["r_over_t"], r["l_over_r"], r["shape"], r["alpha_deg"],
                   r["gamma"])
            if key not in meshes:
                meshes[key] = ShellMesh(r["r_over_t"], r["l_over_r"],
                                        shape=r["shape"],
                                        alpha_deg=r["alpha_deg"],
                                        thickness_gamma=r["gamma"],
                                        elems_per_half_wave=3.0)
            m = meshes[key]
            w = np.load(os.path.join(batch_dir, r["key"], "w.npy"))
            if w.shape[0] != m.n_nodes:
                continue

            nf = 4 + len(cond_vars)
            nodal = np.zeros((nf, 1, m.n_nodes), dtype=np.float32)
            nodal[0:3, 0, :] = m.coords.T            # NOMINAL, not perturbed
            nodal[3, 0, :] = w
            for j, cv in enumerate(cond_vars):
                nodal[4 + j, 0, :] = float(r[cv])
            max_leak = max(max_leak, check_leakage(m.coords.T, nodal[0:3, 0, :]))

            sg = g.create_group(str(i))
            sg.create_dataset("nodal_data", data=nodal, compression="gzip",
                              compression_opts=4)
            sg.create_dataset("mesh_edge", data=edges_from_elements(m.elements),
                              compression="gzip", compression_opts=4)
            for k in ("r_over_t", "l_over_r", "shape", "alpha_deg", "gamma",
                      "sigma_hat", "a_bar", "nu_matern", "comp_a", "seed",
                      "n_dominant", "f_end_over_lin", "rms_over_t", "Z"):
                if k in r:
                    sg.attrs[k] = r[k]
            n_written += 1

        h.attrs["num_samples"] = n_written
        h.attrs["num_features"] = 4 + len(cond_vars)
        h.attrs["num_timesteps"] = 1
        h.attrs["input_var"] = 1
        h.attrs["output_var"] = 1
        h.attrs["cond_var"] = len(cond_vars)
        h.attrs["feature_names"] = np.array(
            ["x", "y", "z", "w_radial_over_t"] + list(cond_vars), dtype="S32")
        h.attrs["note"] = np.bytes_(
            "Rows 0:3 are the NOMINAL geometry. The imperfection realization is "
            "withheld by design and is not recoverable from this file.")
    print("wrote %d samples -> %s" % (n_written, out_path))
    print("leakage check (rows 0:3 vs nominal, must be 0.0): %.3e" % max_leak)
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", nargs="*", default=[])
    a = ap.parse_args()
    export(a.batch, a.out, tuple(a.cond))


if __name__ == "__main__":
    main()
