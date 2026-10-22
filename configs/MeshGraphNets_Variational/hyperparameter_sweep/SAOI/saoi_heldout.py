#!/usr/bin/env python3
"""SAOI geometry labels and held-out parts for the B300 round-2 SAOI node.

run_sweep.sh runs it with --method mgnv before any preflight. (Its
HI-MGNflow twin went with that route on 2026-10-19: it was merged into
meshgraphnets-v as latent_type local, and its old repo and configs are in
junk/legacy_HI_MGNFlow/.) Two runners may still start it at the same moment,
so it holds a lock on output/b300_r2/saoi/heldout/ and leaves every file that
is already right untouched. Safe to rerun.

Per board half this script

  1. groups dataset/SAOI/saoi_train_<half>.h5 by geometry: node coordinates +
     mesh, condition rows left out, so variants of one geometry that differ
     only in thickness or part number stay on one side of the split;
  2. writes dataset/SAOI/saoi_train_<half>_bygeom.h5 = the source file plus
     data/<id>/metadata.attrs['geometry_id']. The *_grp and *_nopart arms
     train on it with split_group_attr geometry_id. An existing copy whose
     labels already match is reused;
  3. computes the split those arms' trainers will compute, with the trainers'
     own grouped_split_ids, their split_seed and the 0.8/0.1 ratios
     hardcoded in mesh_dataset.build_dataset_splits;
  4. cuts the largest TEST-role parts (geometries no grp arm trains or selects
     a checkpoint on) into
         output/b300_r2/saoi/heldout/data/geosplit_test<k>_infer_<half>.h5    one realization
         output/b300_r2/saoi/heldout/data/geosplit_test<k>_compare_<half>.h5  every realization
     A part is the samples with identical coordinates, mesh AND condition
     rows -- the infer/compare contract of every SAOI eval pair;
  5. writes, for every arm of the chosen method(s), one inference config per
     cut part into
         output/b300_r2/saoi/heldout/configs/mgnv/
     derived from that arm's s26fe_main config (part, output dir and draw
     count change; nothing else), and the plan into heldout/split_<half>.json.

The same parts are read by every arm. For v_smp (random sample split)
they are SEEN parts -- about 80% of their realizations were trained on, and
split_<half>.json records the exact share. For the grp and nopart arms they
are UNSEEN geometries. That pairing is the generalization gap at fixed parts.

    python saoi_heldout.py                # both halves, MGN-V configs
    python saoi_heldout.py --method mgnv  # the same (what run_sweep.sh does)
    python saoi_heldout.py --plan-only    # grouping and split only; writes no HDF5
"""
import argparse
import contextlib
import hashlib
import json
import pathlib
import re
import shutil
import sys
import tempfile
import time

import h5py
import numpy as np

sys.dont_write_bytecode = True       # the import below comes from the repo; keep bytecode out of it
HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]          # configs/<method>/hyperparameter_sweep/SAOI/ -> repo root
DATA_DIR = REPO_ROOT / "dataset" / "SAOI"
OUT_DIR = REPO_ROOT / "output" / "b300_r2" / "saoi" / "heldout"
CUT_DIR = OUT_DIR / "data"
CONFIG_OUT = OUT_DIR / "configs"     # one subdirectory per method
CUT_REL = "../../output/b300_r2/saoi/heldout/data"     # as a config spells it (cwd = methods/<Name>/)

SPLIT_SRC = {"mgnv": REPO_ROOT / "methods" / "MeshGraphNets_Variational" / "general_modules" / "grouped_split.py"}
CFG_DIRS = {"mgnv": REPO_ROOT / "configs" / "MeshGraphNets_Variational" / "hyperparameter_sweep" / "SAOI"}
ARMS = {"mgnv": ["v_smp", "v_grp", "v_grp_rep", "v_nopart"]}
SMP_ARMS = {"v_smp"}

sys.path.insert(0, str(SPLIT_SRC["mgnv"].parent))
from grouped_split import grouped_split_ids  # noqa: E402  the function the trainer calls

ATTR = "geometry_id"
TRAIN_RATIO, VAL_RATIO = 0.8, 0.1    # mesh_dataset.build_dataset_splits hardcodes 0.8 / 0.1 / 0.1
# Row layout of saoi_train_<half>.h5: 0:3 coordinates, 3:6 displacement, 6:9
# stress / thickness / Part No. The part key uses all three condition rows even
# for the nopart arms: a part is a physical condition, not what one arm reads.
COND_START, COND_VAR = 6, 3
Z_DISP_ROW = 5                       # rollout.py scores max - min of output channel 2


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


def config_value(path, key):
    """One value of a flat `key value` config, comments stripped, key case-insensitive."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("%", 1)[0].split("#", 1)[0].strip()
        m = re.match(r"(\S+)\s+(\S+)", line)
        if m and m.group(1).lower() == key.lower():
            return m.group(2)
    sys.exit(f"SAOI_HELDOUT FAIL {path.name}: no '{key}' line")


def group(path, decimals):
    """Geometries of one training file, each with its parts and their peak-to-valley values."""
    geoms = {}
    with h5py.File(path, "r") as f:
        ids = sorted(int(k) for k in f["data"].keys())
        for sid in ids:
            nd = f[f"data/{sid}/nodal_data"][:, -1, :]      # [rows, nodes]; static data has one step
            edge = f[f"data/{sid}/mesh_edge"][:]
            if nd.shape[0] < COND_START + COND_VAR:
                sys.exit(f"SAOI_HELDOUT FAIL {path.name}: sample {sid} has {nd.shape[0]} rows, "
                         f"need {COND_START + COND_VAR}")
            xyz = _canon(nd[0:3], decimals)
            gkey = _digest(xyz, edge)
            pkey = _digest(xyz, _canon(nd[COND_START:COND_START + COND_VAR], decimals), edge)
            g = geoms.setdefault(gkey, {"sample_ids": [], "num_nodes": int(nd.shape[1]), "parts": {}})
            g["sample_ids"].append(sid)
            p = g["parts"].setdefault(pkey, {"sample_ids": [], "pv": []})
            p["sample_ids"].append(sid)
            p["pv"].append(float(nd[Z_DISP_ROW].max() - nd[Z_DISP_ROW].min()))
    out = []
    # Ids in order of first appearance, so a relabel of the same file is stable.
    for gid, g in enumerate(sorted(geoms.values(), key=lambda g: g["sample_ids"][0])):
        parts = []
        for p in sorted(g["parts"].values(), key=lambda p: (-len(p["sample_ids"]), p["sample_ids"][0])):
            v = np.asarray(p.pop("pv"))
            parts.append({"size": len(p["sample_ids"]), "pv_mean": float(v.mean()),
                          "pv_sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, **p})
        out.append({"geometry_id": gid, "size": len(g["sample_ids"]), "num_nodes": g["num_nodes"],
                    "sample_ids": g["sample_ids"], "parts": parts})
    return ids, out


def write_labels(path, gid_of, labels_only=False):
    """Put geometry_id on every sample of `path`, or make a file of nothing but the labels."""
    with h5py.File(path, "w" if labels_only else "r+") as f:
        for sid, gid in gid_of.items():
            name = f"data/{sid}"
            g = f.require_group(name) if labels_only else f[name]
            if "metadata" in g and not isinstance(g["metadata"], h5py.Group):
                sys.exit(f"SAOI_HELDOUT FAIL {path.name}: {name}/metadata is not a group")
            g.require_group("metadata").attrs[ATTR] = np.int64(gid)


def labels_match(path, gid_of):
    try:
        with h5py.File(path, "r") as f:
            if sorted(int(k) for k in f["data"].keys()) != sorted(gid_of):
                return False
            for sid, gid in gid_of.items():
                meta = f[f"data/{sid}"].get("metadata")
                if meta is None or int(meta.attrs.get(ATTR, -1)) != gid:
                    return False
    except (OSError, KeyError, ValueError):
        return False
    return True


def grouped_split(ids, gid_of, seed, path=None):
    if path is not None:
        return grouped_split_ids(str(path), ids, ATTR, TRAIN_RATIO, VAL_RATIO, seed)
    with tempfile.TemporaryDirectory() as tmp:
        labels = pathlib.Path(tmp) / "labels.h5"
        write_labels(labels, gid_of, labels_only=True)
        return grouped_split_ids(str(labels), ids, ATTR, TRAIN_RATIO, VAL_RATIO, seed)


def sample_split_train(ids, seed):
    """The smp arms' train ids: mesh_dataset._resolve_split_ids without split_group_attr."""
    shuffled = list(ids)
    np.random.default_rng(seed).shuffle(shuffled)
    return set(shuffled[:int(len(shuffled) * TRAIN_RATIO)])


def subset_matches(src, dst, sample_ids):
    """A cut newer than its source that holds exactly these samples is already right."""
    try:
        if dst.stat().st_mtime < src.stat().st_mtime:
            return False
        with h5py.File(dst, "r") as f:
            return sorted(int(k) for k in f["data"].keys()) == sorted(sample_ids)
    except (OSError, KeyError, ValueError):
        return False


def write_subset(src, dst, sample_ids):
    """Copy the chosen samples (and the file's metadata) into a new HDF5, atomically.

    An existing cut that already holds them stays as it is: another runner
    may be about to read it.
    """
    if subset_matches(src, dst, sample_ids):
        return
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


HEADER = """%   ============================================================
%
%   SAOI B300 round 2, held-out part -- arm {arm}, board half {half}, {tag}:
%   geometry {gid}, {size} realizations, a TEST geometry of the grouped split.
%   {seen}
%
%   Generated by {script} from
%   {src}; only the part, the output paths and the
%   draw count differ. ../../split_{half}.json names the part's samples.
%
%   infer_dataset is one realization of the part (its geometry and condition
%   rows); eval_dataset is every realization of the part, that one included.
%
%   ============================================================
"""


def derive_config(method, arm, half, es, draws):
    src = CFG_DIRS[method] / f"config_infer_{arm}_{half}_s26fe_main.txt"
    lines = src.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.split()[:1] == ["model"])
    tag = es["tag"]
    new = {
        "log_file_dir": f"../../output/b300_r2/saoi/{arm}/{half}/infer_{tag}.log",
        "inference_output_dir": f"../../output/b300_r2/saoi/{arm}/{half}/infer/{tag}",
        "infer_dataset": f"{CUT_REL}/{es['infer']}  # one realization of the held-out part",
        "eval_dataset": f"{CUT_REL}/{es['compare']}  # every realization of the part: the histogram ground truth",
        "inference_samples": f"{draws}  # draws per scene",
    }
    seen_keys, body = set(), []
    for line in lines[start:]:
        m = re.match(r"(\S+)(\s+)", line)
        key = m.group(1).lower() if m and not line.startswith("%") else None
        if key in new:
            line = f"{m.group(1)}{m.group(2)}{new[key]}"
            seen_keys.add(key)
        elif "against 125 ground-truth realizations" in line:
            line = line.replace("against 125 ground-truth realizations",
                                f"against the part's {es['size']} realizations")
        body.append(line)
    if seen_keys != set(new):
        sys.exit(f"SAOI_HELDOUT FAIL {src.name}: no line for {sorted(set(new) - seen_keys)}")
    text = "\n".join(body) + "\n"
    for left in ("s26fe", "S26FE"):
        if left in text:
            sys.exit(f"SAOI_HELDOUT FAIL {src.name}: '{left}' survived: "
                     + next(x for x in text.splitlines() if left in x))
    seen = (f"{arm} trained on {es['seen_frac_smp']:.0%} of these realizations (random sample split): SEEN."
            if arm in SMP_ARMS else f"{arm} never trained on this geometry: UNSEEN.")
    return HEADER.format(arm=arm, half=half, tag=tag, gid=es["geometry_id"], size=es["size"], seen=seen,
                         script=pathlib.Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                         src=src.relative_to(REPO_ROOT).as_posix()) + text


@contextlib.contextmanager
def heldout_lock():
    """One writer at a time: two runners may start this script together."""
    with open(OUT_DIR / ".lock", "a+") as fh:
        try:
            import fcntl
        except ImportError:      # Windows
            import msvcrt
            fh.seek(0)
            while True:
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(1)
            try:
                yield
            finally:
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
            return
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def write_text_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--halves", nargs="+", default=["bot", "top"])
    ap.add_argument("--min-size", type=int, default=10,
                    help="skip a held-out part with fewer realizations: its histogram says nothing")
    ap.add_argument("--max-sets", type=int, default=6, help="held-out parts cut per half, largest first")
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--decimals", type=int, default=5,
                    help="rounding of coordinates/conditions in the grouping key")
    ap.add_argument("--plan-only", action="store_true", help="group and split only: no copy, no cuts, no configs")
    ap.add_argument("--method", choices=["mgnv", "both"], default="both",
                    help="whose inference configs to write; 'both' (kept for old command lines) "
                         "means every method in ARMS, now MGN-V only")
    args = ap.parse_args()
    args.methods = list(ARMS) if args.method == "both" else [args.method]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with heldout_lock():
        return run(args)


def run(args):
    rc, configs, done, brief = 0, {}, [], []
    for half in args.halves:
        src = DATA_DIR / f"saoi_train_{half}.h5"
        dst = DATA_DIR / f"saoi_train_{half}_bygeom.h5"
        if not src.is_file():
            print(f"SAOI_HELDOUT FAIL missing {src}")
            rc = 1
            continue
        seeds = set()
        for method in ARMS:
            for arm in ARMS[method]:
                cfg = CFG_DIRS[method] / f"config_train_{arm}_{half}.txt"
                seeds.add(int(config_value(cfg, "split_seed")))
                if arm in SMP_ARMS:
                    continue
                if config_value(cfg, "split_group_attr").lower() != ATTR:
                    sys.exit(f"SAOI_HELDOUT FAIL {cfg.name}: split_group_attr is not {ATTR}")
                if config_value(cfg, "dataset_dir") != f"../../dataset/SAOI/{dst.name}":
                    sys.exit(f"SAOI_HELDOUT FAIL {cfg.name}: dataset_dir is not ../../dataset/SAOI/{dst.name}")
        if len(seeds) != 1:
            sys.exit(f"SAOI_HELDOUT FAIL {half}: the arms disagree on split_seed {sorted(seeds)}")
        seed = seeds.pop()

        ids, geoms = group(src, args.decimals)
        gid_of = {sid: g["geometry_id"] for g in geoms for sid in g["sample_ids"]}
        n_parts = sum(len(g["parts"]) for g in geoms)
        print(f"{src}")
        print(f"  {len(ids)} samples = {len(geoms)} geometries = {n_parts} parts "
              f"(a part = a geometry under one set of condition rows)")
        if len(geoms) < 3:
            print(f"SAOI_HELDOUT FAIL {half}: {len(geoms)} geometries; a grouped split needs 3")
            rc = 1
            continue
        if len(geoms) > 0.5 * len(ids):
            print(f"SAOI_HELDOUT FAIL {half}: {len(geoms)} geometries for {len(ids)} samples -- the "
                  f"coordinates differ between realizations of one part. Rerun with --decimals 3.")
            rc = 1
            continue

        if args.plan_only:
            parts = grouped_split(ids, gid_of, seed)
        else:
            if dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime and labels_match(dst, gid_of):
                print(f"  {dst.name}: labels already match, reused")
            else:
                tmp = dst.with_name(dst.name + ".tmp")
                print(f"  copying to {dst.name} ({src.stat().st_size / 2**30:.2f} GiB) and labelling ...")
                shutil.copyfile(src, tmp)
                write_labels(tmp, gid_of)
                tmp.replace(dst)
            parts = grouped_split(ids, gid_of, seed, dst)   # from the file the trainers open
        role_of = {}
        for role, sids in zip(("train", "val", "test"), parts):
            for sid in sids:
                role_of[gid_of[sid]] = role
        for g in geoms:
            g["role"] = role_of[g["geometry_id"]]
        roles = {}
        for role in ("train", "val", "test"):
            gs = [g for g in geoms if g["role"] == role]
            roles[role] = {"geometries": len(gs), "samples": sum(g["size"] for g in gs),
                           "parts": sum(len(g["parts"]) for g in gs)}
            print(f"  {role:<5} {len(gs):>4} geometries  {roles[role]['parts']:>4} parts  "
                  f"{roles[role]['samples']:>6} samples")
        if len(geoms) < 20:
            print(f"  WARNING: {len(geoms)} geometries -- validation holds {roles['val']['geometries']}, "
                  f"test {roles['test']['geometries']}; the grp arms select on very few parts.")

        smp_train = sample_split_train(ids, seed)
        # Where each part sits among the training-role parts: the share of their
        # mean PVs below it. 0 or 1 = outside their range.
        train_pv = np.sort([p["pv_mean"] for g in geoms if g["role"] == "train" for p in g["parts"]])
        cands = sorted(((g, p) for g in geoms if g["role"] == "test" for p in g["parts"]),
                       key=lambda gp: (-gp[1]["size"], gp[1]["sample_ids"][0]))
        min_size = args.min_size
        if not any(p["size"] >= min_size for _, p in cands):
            min_size = 5
            print(f"  no test part has >= {args.min_size} realizations; lowering the floor to {min_size}")
        eval_sets = []
        for g, p in cands:
            if p["size"] < min_size or len(eval_sets) >= args.max_sets:
                continue
            k = len(eval_sets) + 1
            eval_sets.append({
                "tag": f"test{k}", "geometry_id": g["geometry_id"], "size": p["size"],
                "pv_mean": p["pv_mean"], "pv_sd": p["pv_sd"],
                "pv_rank_in_train": float(np.searchsorted(train_pv, p["pv_mean"]) / max(train_pv.size, 1)),
                "seen_frac_smp": sum(s in smp_train for s in p["sample_ids"]) / p["size"],
                "num_nodes": g["num_nodes"], "sample_ids": p["sample_ids"],
                "infer": f"geosplit_test{k}_infer_{half}.h5", "compare": f"geosplit_test{k}_compare_{half}.h5"})
        for es in eval_sets:
            print(f"  {es['tag']:<6} geometry {es['geometry_id']:<4} {es['size']:>4} realizations  "
                  f"PV {es['pv_mean']:.4g} +- {es['pv_sd']:.4g}  rank in train {es['pv_rank_in_train']:.2f}  "
                  f"smp-seen {es['seen_frac_smp']:.0%}")
        if not eval_sets:
            print(f"  WARNING: no test part has >= {min_size} realizations; this half gets no H score")

        plan = {"source": src.name, "copy": dst.name, "attribute": ATTR, "split_seed": seed,
                "ratios": [TRAIN_RATIO, VAL_RATIO, round(1 - TRAIN_RATIO - VAL_RATIO, 6)],
                "decimals": args.decimals, "min_size": min_size, "samples": len(ids),
                "geometries_total": len(geoms), "parts_total": n_parts, "roles": roles,
                "eval_sets": eval_sets, "geometries": geoms}
        out = OUT_DIR / f"split_{half}.json"
        write_text_atomic(out, json.dumps(plan, indent=1))
        print(f"  plan -> {out}")
        brief.append(f"{half} G{len(geoms)} v{roles['val']['geometries']} "
                     f"t{roles['test']['geometries']} H{len(eval_sets)}")
        if args.plan_only:
            print()
            continue

        CUT_DIR.mkdir(parents=True, exist_ok=True)
        for es in eval_sets:
            write_subset(src, CUT_DIR / es["infer"], es["sample_ids"][:1])
            write_subset(src, CUT_DIR / es["compare"], es["sample_ids"])
            for method in args.methods:
                for arm in ARMS[method]:
                    configs[method, f"config_infer_{arm}_{half}_{es['tag']}.txt"] = \
                        derive_config(method, arm, half, es, args.draws)
        print(f"  {len(eval_sets)} held-out part(s) cut into {CUT_DIR}")
        print()
        done.append(half)

    if done:
        for method in args.methods:
            out = CONFIG_OUT / method
            out.mkdir(parents=True, exist_ok=True)
            # A config left from an earlier split would be checked and run as if current.
            for old in out.glob("config_infer_*.txt"):
                if any(f"_{h}_test" in old.name for h in done):
                    old.unlink()
            mine = {name: body for (m, name), body in configs.items() if m == method}
            for name, body in mine.items():
                (out / name).write_text(body, encoding="utf-8", newline="\n")
            print(f"{len(mine)} inference config(s) -> {out}")
    if brief:
        print("SAOI_HELDOUT " + " | ".join(brief))
    return rc


if __name__ == "__main__":
    sys.exit(main())
