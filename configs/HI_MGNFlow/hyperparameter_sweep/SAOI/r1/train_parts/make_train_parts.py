#!/usr/bin/env python3
"""Cut infer/compare pairs for parts the model TRAINED on out of the SAOI training files.

The sweep scores the prior on three parts it never saw. When those come out
wrong there are two readings: the prior cannot reproduce a part's process
scatter at all, or it can and does not carry it to a new part. A part from the
training file separates them -- there the prior has no generalization excuse,
so its draws must match that part's own realizations.

A part is a set of training samples sharing one condition: node coordinates,
mesh topology and conditioning rows identical, only the displacement rows
differ. That is the definition SAOI_run/check_eval_inputs.py enforces on every
_infer_/_compare_ pair, applied here to group the training file. The largest
groups become

    dataset/SAOI/train_part<k>_infer_<half>.h5     one realization: the part's geometry and conditions
    dataset/SAOI/train_part<k>_compare_<half>.h5   every realization of that part

and the whole grouping is written to

    output/chi-mgnflow/saoi_b300_sweep/train_parts/census_<half>.json

so the part count, realizations per part and the within- versus between-part
scatter of the peak-to-valley statistic are on record.

Validation samples are not excluded: split_seed draws samples, not parts, so a
part's realizations sit in both splits and together they are its distribution.

    python make_train_parts.py                 # both halves, two parts each
    python make_train_parts.py --census-only   # the grouping and census only, no HDF5
"""
import argparse
import hashlib
import json
import pathlib
import sys

import h5py
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[5]          # configs/HI_MGNFlow/hyperparameter_sweep/SAOI/r1/train_parts/ -> repo root
DATA_DIR = REPO_ROOT / "dataset" / "SAOI"
CENSUS_DIR = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_b300_sweep" / "train_parts"

# Row layout the configs declare: 0:3 coordinates, 3:6 displacement (input_var 3),
# 6:9 input-only conditions (cond_var 3).
INPUT_VAR = 3
COND_VAR = 3
COND_START = 3 + INPUT_VAR
# rollout.py scores max - min of output channel 2, i.e. nodal_data row 5.
Z_DISP_ROW = 5


def _canon(a, decimals):
    # +0.0 folds -0.0 into 0.0, which would otherwise hash differently.
    return np.ascontiguousarray(np.round(np.asarray(a, dtype=np.float64), decimals) + 0.0)


def _digest(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _describe_row(row):
    """A condition row in one line: its value if constant over nodes, else its spread."""
    u = np.unique(row)
    if u.size == 1:
        return f"constant {u[0]:.6g}"
    if u.size <= 6:
        return "values " + ", ".join(f"{v:.6g}" for v in u)
    return f"{u.size} distinct, {u.min():.6g}..{u.max():.6g}"


def census(path, decimals):
    """Group the samples of one training file by condition."""
    groups, geom, topo, info, pv = {}, {}, {}, {}, {}
    with h5py.File(path, "r") as f:
        names = None
        if "metadata/feature_names" in f:
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f["metadata/feature_names"][()]]
        ids = sorted(int(k) for k in f["data"].keys())
        n_rows = None
        for sid in ids:
            g = f[f"data/{sid}"]
            nd = g["nodal_data"][:, -1, :]       # [rows, nodes]; static data has one step
            edge = g["mesh_edge"][:]
            n_rows = nd.shape[0]
            if n_rows < COND_START + COND_VAR:
                sys.exit(f"{path}: sample {sid} has {n_rows} rows, the configs need "
                         f"{COND_START + COND_VAR} (coords + {INPUT_VAR} state + {COND_VAR} conditions)")
            xyz = _canon(nd[0:3], decimals)
            key = _digest(xyz, _canon(nd[COND_START:COND_START + COND_VAR], decimals), edge)
            groups.setdefault(key, []).append(sid)
            # Coarser groupings, to say which ingredient splits a part when
            # every group comes out with one realization.
            geom.setdefault(_digest(xyz, edge), []).append(sid)
            topo.setdefault(_digest(edge), []).append(sid)
            pv.setdefault(key, []).append(float(nd[Z_DISP_ROW].max() - nd[Z_DISP_ROW].min()))
            if key not in info:
                info[key] = {"num_nodes": int(nd.shape[1]), "num_edges": int(edge.shape[1]),
                             "conditions": [_describe_row(r) for r in nd[COND_START:COND_START + COND_VAR]]}
    parts = []
    for key, sids in groups.items():
        v = np.asarray(pv[key])
        parts.append({"size": len(sids), "sample_ids": sids,
                      "pv_mean": float(v.mean()), "pv_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
                      **info[key]})
    parts.sort(key=lambda p: (-p["size"], p["sample_ids"][0]))
    return {"file": str(path), "samples": len(ids), "rows": n_rows, "feature_names": names,
            "geometries": len(geom), "largest_geometry": max(len(v) for v in geom.values()),
            "meshes": len(topo), "largest_mesh": max(len(v) for v in topo.values()),
            "parts": parts}


def print_census(c, shown):
    parts = c["parts"]
    sizes = np.asarray([p["size"] for p in parts])
    names = c["feature_names"] or []
    print(f"{c['file']}")
    print(f"  {c['samples']} samples, {c['rows']} rows" + (f" ({', '.join(names)})" if names else ""))
    print(f"  distinct parts (same coordinates + mesh + condition rows): {len(parts)}")
    print(f"  realizations per part: min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}")
    print(f"  ignoring the condition rows: {c['geometries']} geometries (largest {c['largest_geometry']}); "
          f"mesh topology alone: {c['meshes']} (largest {c['largest_mesh']})")
    if c["geometries"] < len(parts):
        print("    <- condition rows split some geometries into several parts")
    multi = [p for p in parts if p["size"] > 1]
    if multi:
        within = float(np.median([p["pv_sd"] for p in multi]))
        between = float(np.std([p["pv_mean"] for p in parts], ddof=1)) if len(parts) > 1 else 0.0
        print(f"  z_disp peak-to-valley: within-part sd (median over parts) {within:.4g}, "
              f"between-part sd of the part means {between:.4g}")
    # Whether a condition is one number per part (a part ID, a thickness) or a
    # field over the nodes is what says how the model can tell parts apart.
    for j in range(COND_VAR):
        row = COND_START + j
        label = names[row] if row < len(names) else f"row {row}"
        per_part = all(p["conditions"][j].startswith("constant") for p in parts)
        values = {p["conditions"][j] for p in parts}
        print(f"  condition row {row} ({label}): "
              + ("one value per part" if per_part else "varies over the nodes")
              + f", {len(values)} distinct across parts; first: "
              + "; ".join(p["conditions"][j] for p in parts[:3]) + (" ..." if len(parts) > 3 else ""))
    print(f"  largest parts:")
    for k, p in enumerate(parts[:max(shown, 5)], 1):
        ids = p["sample_ids"]
        print(f"    #{k:<3} {p['size']:>4} realizations  {p['num_nodes']} nodes  "
              f"PV {p['pv_mean']:.4g} +- {p['pv_sd']:.4g}  ids {ids[:4]}{' ...' if len(ids) > 4 else ''}")


def write_subset(src, dst, sample_ids):
    """Copy the chosen samples (and the file's metadata) into a new HDF5, atomically."""
    tmp = dst.with_name(dst.name + ".tmp")
    with h5py.File(src, "r") as fi, h5py.File(tmp, "w") as fo:
        for k, v in fi.attrs.items():
            fo.attrs[k] = v
        if "metadata" in fi:
            fi.copy("metadata", fo)
        data = fo.create_group("data")
        for sid in sample_ids:
            fi.copy(f"data/{sid}", data, name=str(sid))
    tmp.replace(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halves", nargs="+", default=["bot", "top"])
    ap.add_argument("--parts", type=int, default=2, help="largest parts to cut per half")
    ap.add_argument("--min-size", type=int, default=10,
                    help="refuse a part with fewer realizations: its histogram says nothing")
    ap.add_argument("--decimals", type=int, default=5,
                    help="rounding of coordinates/conditions in the grouping key")
    ap.add_argument("--census-only", action="store_true")
    args = ap.parse_args()

    CENSUS_DIR.mkdir(parents=True, exist_ok=True)
    rc = 0
    for half in args.halves:
        src = DATA_DIR / f"saoi_train_{half}.h5"
        if not src.is_file():
            print(f"MISSING {src}")
            rc = 1
            continue
        c = census(src, args.decimals)
        print_census(c, args.parts)
        out = CENSUS_DIR / f"census_{half}.json"
        out.write_text(json.dumps(c, indent=1))
        print(f"  census -> {out}")
        if args.census_only:
            print()
            continue
        chosen = c["parts"][:args.parts]
        small = [p for p in chosen if p["size"] < args.min_size]
        if len(chosen) < args.parts or small:
            print(f"  FAIL: need {args.parts} parts with >= {args.min_size} realizations; "
                  f"the largest have {[p['size'] for p in chosen]}. If every part has one "
                  f"realization, the coordinates or condition rows differ between realizations "
                  f"(try --decimals 3), and the census above says which.")
            rc = 1
            continue
        for k, p in enumerate(chosen, 1):
            infer = DATA_DIR / f"train_part{k}_infer_{half}.h5"
            compare = DATA_DIR / f"train_part{k}_compare_{half}.h5"
            write_subset(src, infer, p["sample_ids"][:1])
            write_subset(src, compare, p["sample_ids"])
            print(f"  part{k}: {infer.name} (sample {p['sample_ids'][0]}), "
                  f"{compare.name} ({p['size']} realizations)")
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main())
