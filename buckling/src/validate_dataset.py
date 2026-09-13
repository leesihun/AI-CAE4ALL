"""Audit every sample, model input invariance, BCs and solve provenance.

This validates the documented input rows and graph, not a training
configuration. Loaders must zero static target inputs and exclude diagnostics
and sample IDs. RMS drift is not an equilibrium certificate.
"""
import argparse
from collections import defaultdict
import sys

import h5py
import numpy as np

from geometry import ShellMesh
from pack_hdf5 import EPH, GEOMETRY_KEYS, component_a_field, edges_from_elements, node_type_row
from run_production import cache_problem, geom_key


def audit(path):
    fail, warn = [], []
    by_geom = defaultdict(list)
    reference = {}
    stds = []
    with h5py.File(path, "r") as h:
        if "data" not in h or not len(h["data"]):
            return ["D5 missing or empty data group"], warn
        sids = sorted(h["data"])
        for key, expected in (("input_var", 3), ("output_var", 3), ("cond_var", 5),
                              ("num_timesteps", 1), ("num_features", 11),
                              ("num_samples", len(sids))):
            if h.attrs.get(key) != expected:
                fail.append("D5 root %s must equal %s" % (key, expected))
        if "latent" not in h or set(h["latent"]) != set(sids):
            fail.append("D4 latent sample IDs do not match data sample IDs")
        max_drift = h.attrs.get("max_drift", np.nan)
        if not np.isfinite(max_drift) or max_drift < 0:
            fail.append("D5 missing or invalid retention threshold")
        tiers = set()
        for sid in sids:
            try:
                grp = h["data"][sid]
                g = {key: grp.attrs[key] for key in GEOMETRY_KEYS}
                gk = str(grp.attrs["geometry"])
                tier = str(grp.attrs["tier"])
                tiers.add(tier)
                if gk != geom_key(g) or tier not in {"train", "t1", "t2", "t3"}:
                    raise ValueError("geometry key or tier metadata mismatch")
                if gk not in reference:
                    mesh = ShellMesh(**g, elems_per_half_wave=EPH)
                    offset, _ = component_a_field(mesh)
                    inputs = np.concatenate([mesh.coords.T.astype(np.float32), offset.T,
                                             mesh.thickness_at(mesh.z)[None].astype(np.float32),
                                             node_type_row(mesh)[None]])
                    reference[gk] = (mesh, inputs, edges_from_elements(mesh.elements, mesh.n_nodes), tier, g)
                mesh, expected_inputs, expected_edges, first_tier, first_geometry = reference[gk]
                if g != first_geometry or tier != first_tier:
                    fail.append("D4 geometry has inconsistent parameters/tiers: " + gk)
                nd = grp["nodal_data"][:]
                if nd.shape != (11, 1, mesh.n_nodes) or not np.isfinite(nd).all():
                    raise ValueError("nodal_data shape/nonfinite values")
                for key, expected in (("input_var", 3), ("output_var", 3), ("cond_var", 5),
                                      ("num_features", 11), ("num_timesteps", 1), ("num_nodes", mesh.n_nodes)):
                    if grp.attrs.get(key) != expected:
                        fail.append("D5 %s attribute %s mismatch" % (sid, key))
                actual_inputs = nd[[0, 1, 2, 6, 7, 8, 9, 10], 0]
                # Cover ALL input rows for ALL draws, including nominal
                # coordinates that would otherwise reveal Component B.
                if not np.array_equal(actual_inputs, expected_inputs):
                    fail.append("D4 inputs differ from deterministic geometry/signature: " + sid)
                edges = grp["mesh_edge"][:]
                if edges.dtype.kind not in "iu" or not np.array_equal(edges, expected_edges):
                    fail.append("D5 graph differs from reference S8R topology: " + sid)
                drift = grp.attrs.get("drift", np.nan)
                if not np.isfinite(drift) or drift < 0 or drift > max_drift:
                    fail.append("D5 invalid or rejected RMS drift: " + sid)
                output = nd[3:6, 0]
                shorten = float(grp.attrs["shorten"])
                target = -shorten * mesh.end_shortening_cr
                if not np.allclose(output[2, mesh.top_nodes], target, rtol=2e-5, atol=1e-8):
                    fail.append("D5 prescribed shortening mismatch: " + sid)
                if np.max(np.abs(output[:, mesh.bottom_nodes])) > 1e-8:
                    fail.append("D5 clamped base moved: " + sid)
                if np.max(np.abs(output[:2, mesh.top_nodes])) > 1e-8:
                    fail.append("D5 clamped top moved laterally: " + sid)
                z = {}
                if "run_spec_json" in grp.attrs:
                    z["run_spec_json"] = grp.attrs["run_spec_json"]
                problem = cache_problem(z, g, int(grp.attrs["seed"]), shorten)
                if problem:
                    fail.append("D5 invalid solve provenance %s: %s" % (sid, problem))
                if "source_npz_sha256" not in grp.attrs:
                    fail.append("D5 missing source hash: " + sid)
                if "latent" in h and sid in h["latent"]:
                    latent = h["latent"][sid]
                    for key in ("imp_C_real", "imp_C_imag", "k1", "k2", "spectrum"):
                        if key not in latent or not np.isfinite(latent[key][:]).all():
                            raise ValueError("missing/nonfinite diagnostic " + key)
                    if latent["imp_C_real"].shape != (latent["k1"].size, latent["k2"].size) or latent["imp_C_imag"].shape != latent["imp_C_real"].shape:
                        raise ValueError("spectral coefficient shape mismatch")
                stds.append(output.std(axis=1))
                for out_row in range(3):
                    for in_row in range(actual_inputs.shape[0]):
                        if np.array_equal(output[out_row], actual_inputs[in_row]):
                            fail.append("D2 output exactly copies an input: " + sid)
                by_geom[gk].append(sid)
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                fail.append("D5 %s: %s" % (sid, exc))
        if h.attrs.get("n_geometries") != len(reference):
            fail.append("D5 n_geometries mismatch")
        if ",".join(sorted(tiers)) != h.attrs.get("retained_tiers"):
            fail.append("D5 retained tier metadata mismatch")
        if len(tiers) > 1:
            warn.append("Multiple tiers: training loaders do not enforce tier attributes")
        if stds:
            stds = np.stack(stds)
            for row in range(3):
                count = int(np.count_nonzero(stds[:, row] < 1e-9))
                if count:
                    fail.append("D1 output row %d near-constant in %d samples" % (3 + row, count))
        for gk, ids in sorted(by_geom.items()):
            if len(ids) < 3:
                warn.append("D3 insufficient draws for spread check: " + gk)
                continue
            fields = np.stack([h["data"][sid]["nodal_data"][3:6, 0].ravel() for sid in ids])
            mu = fields.mean(axis=0)
            spread = float(np.mean(np.linalg.norm(fields - mu, axis=1)) / max(np.linalg.norm(mu), 1e-12))
            print("%s: %d draws, relative within-geometry field spread %.4f" % (gk, len(ids), spread))
            if spread < .02:
                fail.append("D3 negligible within-geometry spread: " + gk)
        print("Audited all %d samples across %d geometries" % (len(sids), len(reference)))
    return fail, warn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    fail, warn = audit(args.path)
    for message in fail:
        print("FAIL " + message)
    for message in warn:
        print("WARN " + message)
    print("FAILED" if fail else "PASSED" + (" with warnings" if warn else ""))
    return int(bool(fail))


if __name__ == "__main__":
    sys.exit(main())
