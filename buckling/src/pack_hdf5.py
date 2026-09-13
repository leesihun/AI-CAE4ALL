"""Pack production draws into the suite's shared mesh HDF5 contract.

Layout (docs/reference/DATASET_FORMAT.md):

    data/{sample_id}/nodal_data   [num_features, num_timesteps, num_nodes]
    data/{sample_id}/mesh_edge    [2, num_edges]

Rows, with num_timesteps = 1 (this is a static benchmark):

    0:3   reference coordinates of the PERFECT shell (x, y, z)
    3:6   output: buckled displacement (ux, uy, uz)          <- output_var = 3
    6:9   conditions: Component A signature offset (known)   <- input-only
    9     condition: nodal thickness (R_NOM = 1 units)        <- input-only
    10    condition: node type (0 interior, 1 clamped base, 2 loaded top)

So input_var = 3, output_var = 3, cond_var = 5. The withheld Component B is
NOT written into nodal_data anywhere -- that is the whole point. It is stored
separately under `latent/` so a diagnostic can condition on it, but a model
reading the documented rows cannot see it.

The spread across samples that share a sample-family is therefore irreducible
from the model's inputs, which is the one-to-many property being benchmarked.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometry import ShellMesh
from run_production import (EPH, COMP_A, N_PANELS, SIGMA_HAT, A_BAR, NU_MATERN,
                            geom_key, cache_problem)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "work", "production")

GEOMETRY_KEYS = ("shape", "r_over_t", "l_over_r", "alpha_deg", "thickness_gamma")
FEATURE_NAMES = ["x", "y", "z", "ux", "uy", "uz", "component_a_x",
                 "component_a_y", "component_a_z", "thickness", "node_type"]


def valid_draw(z, record, mesh, shorten):
    """Reject incorrect physics/provenance and malformed arrays before export."""
    g = {k: record[k] for k in GEOMETRY_KEYS}
    problem = cache_problem(z, g, record["seed"], shorten)
    if problem:
        raise ValueError(problem)
    shapes = {"disp": (mesh.n_nodes, 3), "w": (mesh.n_nodes,)}
    for key, shape in shapes.items():
        if z[key].shape != shape or not np.isfinite(z[key]).all():
            raise ValueError("invalid %s array" % key)
    for key in ("imp_C_real", "imp_C_imag", "k1", "k2", "spectrum",
                "drift", "n_share", "rms_over_t"):
        if not np.isfinite(z[key]).all():
            raise ValueError("nonfinite %s" % key)
    if z["imp_C_real"].shape != (z["k1"].size, z["k2"].size) or z["imp_C_imag"].shape != z["imp_C_real"].shape:
        raise ValueError("inconsistent spectral coefficient shape")
    if float(z["drift"]) < 0:
        raise ValueError("negative RMS drift")
    # Solver displacement is measured from the imperfect starting surface.
    # Check the stored radial diagnostic and actual prescribed shortening.
    from analyse import radial
    if not np.allclose(radial(mesh, z["disp"]) / mesh.t, z["w"], rtol=2e-5, atol=2e-5):
        raise ValueError("w does not match displacement projected onto reference radial direction")
    target = -shorten * mesh.end_shortening_cr
    if not np.allclose(z["disp"][mesh.top_nodes, 2], target, rtol=2e-5, atol=1e-8):
        raise ValueError("displacement does not match the plan's target shortening")


def load_plan(path):
    with open(path, encoding="utf-8") as f:
        entries = json.load(f)
    plan = {}
    for entry in entries:
        key = geom_key(entry)
        if key in plan or entry.get("tier") not in {"train", "t1", "t2", "t3"}:
            raise ValueError("duplicate geometry or missing/invalid tier in plan: " + key)
        plan[key] = entry
    return plan


def edges_from_elements(elements, n_nodes):
    """Undirected unique edge list from S8R connectivity, as [2, E], 0-based.

    S8R node ordering is 4 corners then 4 mid-side nodes; the perimeter walk
    c0-m0-c1-m1-c2-m2-c3-m3 back to c0 is the element boundary.
    """
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    a, b = [], []
    for el in elements:
        ring = [el[k] for k in order]
        for k in range(len(ring)):
            u, v = ring[k], ring[(k + 1) % len(ring)]
            if u != v:
                a.append(min(u, v))
                b.append(max(u, v))
    e = np.unique(np.stack([np.asarray(a), np.asarray(b)], axis=1), axis=0)
    both = np.concatenate([e, e[:, ::-1]], axis=0)     # both directions
    return np.ascontiguousarray(both.T.astype(np.int64))


def component_a_field(m):
    """The KNOWN process signature, as a nodal offset vector. Model input."""
    from imperfection import process_signature, critical_wavenumber
    wA, info = process_signature(m.theta, m.z, m.L, m.t, N_PANELS,
                                 COMP_A, seed=N_PANELS,
                                 critical_n=critical_wavenumber(m.r_over_t))
    return (wA[:, None] * m.normal).astype(np.float32), info


def node_type_row(m):
    nt = np.zeros(m.n_nodes, dtype=np.float32)
    nt[m.bottom_nodes] = 1.0
    nt[m.top_nodes] = 2.0
    return nt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", required=True,
                    help="explicit comma list of tiers; training loaders do not enforce stored tier labels")
    ap.add_argument("--plan", required=True,
                    help="authoritative plan json (required to prevent accidental OOD/train mixing)")
    ap.add_argument("--work-dir", default=W)
    ap.add_argument("--max-drift", type=float, default=0.15,
                    help="reject draws whose settle residual exceeds this")
    a = ap.parse_args()

    if not np.isfinite(a.max_drift) or a.max_drift < 0:
        raise SystemExit("--max-drift must be finite and nonnegative")
    results_path = os.path.join(a.work_dir, a.results)
    with open(results_path, "rb") as f:
        results_bytes = f.read()
    recs = [r for r in json.loads(results_bytes) if r.get("ok")]
    if len({r["key"] for r in recs}) != len(recs):
        raise SystemExit("duplicate successful sample IDs in results")
    plan = load_plan(a.plan)
    want = set(a.tiers.split(","))
    if not want or not want <= {"train", "t1", "t2", "t3"}:
        raise SystemExit("unknown --tiers value")

    kept, dropped_drift, dropped_tier = [], 0, 0
    rejected = []
    meshes = {}
    for r in recs:
        gk = r["key"].rsplit("_s", 1)[0]
        if gk not in plan:
            raise SystemExit("results geometry missing from plan: " + gk)
        entry = plan[gk]
        tier = entry["tier"]
        if tier not in want:
            dropped_tier += 1
            continue
        if any(r[k] != entry[k] for k in GEOMETRY_KEYS):
            raise SystemExit("results geometry metadata disagrees with plan: " + gk)
        if not entry["seed0"] <= r["seed"] < entry["seed0"] + entry["n_draws"]:
            raise SystemExit("sample seed outside plan: " + r["key"])
        if gk not in meshes:
            meshes[gk] = ShellMesh(**{k: r[k] for k in GEOMETRY_KEYS}, elems_per_half_wave=EPH)
        draw_path = os.path.join(a.work_dir, r["key"], "draw.npz")
        try:
            with np.load(draw_path) as z:
                valid_draw(z, r, meshes[gk], entry.get("shorten", 1.5))
                r["drift"] = float(z["drift"])
                r["_provenance"] = "versioned" if "run_spec_json" in z else "legacy_uniform_unversioned"
        except (ValueError, KeyError, OSError) as exc:
            rejected.append(dict(key=r["key"], reason=str(exc)))
            continue
        if r["drift"] > a.max_drift:
            dropped_drift += 1
            rejected.append(dict(key=r["key"], reason="RMS drift exceeds retention threshold", drift=r["drift"]))
            continue
        r["_tier"] = tier
        kept.append(r)

    print("%d solver-success draws -> %d kept (%d outside tiers, %d rejected, including %d over drift %.3f)"
          % (len(recs), len(kept), dropped_tier, len(rejected), dropped_drift, a.max_drift))
    if not kept:
        raise SystemExit("nothing to pack")

    by_geom = defaultdict(list)
    for r in kept:
        by_geom[r["key"].rsplit("_s", 1)[0]].append(r)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_written = 0
    with h5py.File(a.out, "x") as h:
        d = h.create_group("data")
        lat = h.create_group("latent")
        for gk in sorted(by_geom):
            rs = sorted(by_geom[gk], key=lambda x: x["seed"])
            g0 = rs[0]
            m = ShellMesh(g0["r_over_t"], g0["l_over_r"], shape=g0["shape"],
                          alpha_deg=g0["alpha_deg"],
                          thickness_gamma=g0["thickness_gamma"],
                          elems_per_half_wave=EPH)
            edges = edges_from_elements(m.elements, m.n_nodes)
            aoff, ainfo = component_a_field(m)
            thick = m.thickness_at(m.z).astype(np.float32)
            ntype = node_type_row(m)
            coords = m.coords.T.astype(np.float32)        # [3, N]

            for r in rs:
                draw_path = os.path.join(a.work_dir, r["key"], "draw.npz")
                z = np.load(draw_path)
                disp = z["disp"][:m.n_nodes, :3].T.astype(np.float32)   # [3, N]
                rows = np.concatenate([
                    coords,                                # 0:3 reference coords
                    disp,                                  # 3:6 output
                    aoff.T,                                # 6:9 known signature
                    thick[None, :],                        # 9   actual thickness
                    ntype[None, :],                        # 10  node type
                ], axis=0)
                sid = r["key"]
                grp = d.create_group(sid)
                grp.create_dataset("nodal_data", data=rows[:, None, :],
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("mesh_edge", data=edges,
                                   compression="gzip", compression_opts=4)
                for k, v in (("shape", g0["shape"]), ("tier", r["_tier"]),
                             ("geometry", gk)):
                    grp.attrs[k] = v
                for k in ("r_over_t", "l_over_r", "alpha_deg", "thickness_gamma"):
                    grp.attrs[k] = float(g0[k])
                grp.attrs["seed"] = int(r["seed"])
                grp.attrs["n_dominant"] = int(r["n_dominant"])
                grp.attrs["n_share"] = float(r["n_share"])
                grp.attrs["rms_over_t"] = float(r["rms_over_t"])
                grp.attrs["drift"] = float(r.get("drift", 0.0))
                grp.attrs["Z"] = float(r.get("Z", m.batdorf_Z))
                grp.attrs["input_var"] = 3
                grp.attrs["output_var"] = 3
                grp.attrs["cond_var"] = 5
                grp.attrs["num_features"] = int(rows.shape[0])
                grp.attrs["num_timesteps"] = 1
                grp.attrs["num_nodes"] = m.n_nodes
                grp.attrs["provenance"] = r["_provenance"]
                grp.attrs["shorten"] = float(plan[gk].get("shorten", 1.5))
                with open(draw_path, "rb") as source:
                    grp.attrs["source_npz_sha256"] = hashlib.sha256(source.read()).hexdigest()
                if "run_spec_json" in z:
                    grp.attrs["run_spec_json"] = str(z["run_spec_json"])

                # withheld latent, kept OUT of nodal_data on purpose
                lg = lat.create_group(sid)
                lg.create_dataset("spectrum", data=z["spectrum"])
                lg.create_dataset("imp_C_real", data=z["imp_C_real"],
                                  compression="gzip", compression_opts=4)
                lg.create_dataset("imp_C_imag", data=z["imp_C_imag"],
                                  compression="gzip", compression_opts=4)
                lg.create_dataset("k1", data=z["k1"])
                lg.create_dataset("k2", data=z["k2"])
                z.close()
                n_written += 1

        h.attrs["input_var"] = 3
        h.attrs["output_var"] = 3
        h.attrs["cond_var"] = 5
        h.attrs["num_timesteps"] = 1
        h.attrs["row_layout"] = ("0:3 ref coords | 3:6 displacement (output) | "
                                 "6:9 Component A offset | 9 thickness (R_NOM=1 units) | "
                                 "10 node type")
        h.attrs["sigma_hat"] = SIGMA_HAT
        h.attrs["a_bar"] = A_BAR
        h.attrs["nu_matern"] = NU_MATERN
        h.attrs["comp_a_rms_over_t"] = COMP_A
        h.attrs["n_geometries"] = len(by_geom)
        h.attrs["n_samples"] = n_written
        h.attrs["num_samples"] = n_written
        h.attrs["num_features"] = len(FEATURE_NAMES)
        h.attrs["max_drift"] = a.max_drift
        h.attrs["retained_tiers"] = ",".join(sorted(want))
        h.attrs["source_results_sha256"] = hashlib.sha256(results_bytes).hexdigest()
        h.attrs["source_results"] = os.path.abspath(results_path)
        h.attrs["source_plan"] = os.path.abspath(a.plan)
        h.attrs["displacement_reference"] = "imperfect initial surface; not total offset from perfect mesh"
        h.attrs["tier_labels_enforced_by_loader"] = False
        metadata = h.create_group("metadata")
        metadata.create_dataset("feature_names", data=FEATURE_NAMES, dtype=h5py.string_dtype("utf-8"))
        metadata.create_dataset("rejected_json", data=json.dumps(rejected), dtype=h5py.string_dtype("utf-8"))
        metadata.create_dataset("plan_json", data=json.dumps(list(plan.values())), dtype=h5py.string_dtype("utf-8"))

    print("wrote %s  %.1f MB  %d samples over %d geometries"
          % (a.out, os.path.getsize(a.out) / 1e6, n_written, len(by_geom)))
    for gk in sorted(by_geom):
        ns = [r["n_dominant"] for r in by_geom[gk]]
        print("  %-30s %3d draws  n in [%d,%d]  distinct=%d"
              % (gk, len(by_geom[gk]), min(ns), max(ns), len(set(ns))))


if __name__ == "__main__":
    main()
