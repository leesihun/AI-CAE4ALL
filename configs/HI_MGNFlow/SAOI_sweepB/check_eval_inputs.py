#!/usr/bin/env python3
"""Validate the inference + histogram chain BEFORE it runs unattended.

Three failures have already cost us a run, and NONE of them raises:

  1. `eval_dataset` absent from the config. rollout.py prints
     "Skipped: no 'eval_dataset' set in config" and exits 0. The sweep reports
     a clean run with an empty warpage table -- the one metric the comparison
     rests on.

  2. `eval_dataset` pointing at the answer-stripped file instead of its `_orig`
     twin. The ground-truth column is all zeros, the histogram is drawn against
     it, and the plot looks perfectly plausible. Worse than (1): a skip is
     visible, a wrong axis is not.

  3. A path key missing from PATH_KEYS. The native parser lowercases every
     string value that is not a declared path key, so `dataset/SAOI/File.h5`
     silently becomes `dataset/saoi/file.h5`. Both directories exist here and
     hold different vintages, so the wrong one opens without error and yields
     wrong numbers. `eval_dataset` was missing exactly that way.

All three are silent, so nothing downstream can notice. Hence this check, run
before any GPU time is committed rather than after.

On failure it dumps everything about every dataset involved -- resolved path,
size, case-variant twins on disk, the HDF5 tree, and a row-by-row comparison of
the eval file against the training file, which is known to be well formed.
There is no second command to run: a failure here is meant to be diagnosable
from its own output.

Exit 0 = the histogram will have real ground truth. Exit 1 = it will not.
"""
import argparse
import datetime
import pathlib
import sys

# rollout.py::Z_DISP_CHANNEL -- the row the spread metric reads. Keep in sync;
# a mismatch here would validate a row the plot never looks at.
Z_DISP_CHANNEL = 5

# general_modules/mesh_dataset.py:169 enumerates samples as
#   sorted(int(k) for k in f['data'].keys())  ->  f['data/<id>/nodal_data']
# with nodal_data 3-D [feature, timestep, node]. Everything below assumes that
# same contract, so a file that does not match it is reported rather than
# silently mis-read.
DATA_GROUP = "data"
NODAL = "nodal_data"

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]          # configs/<Method>/<Sweep>/ -> repo root
# Native paths in a config are relative to the method repo: that is the cwd the
# launcher gives the native process.
METHOD_REPO = REPO_ROOT / "methods" / "HI_MGNFlow"

DATASET_KEYS = ("dataset_dir", "infer_dataset", "eval_dataset")


# ---------------------------------------------------------------------------
# config reading
# ---------------------------------------------------------------------------

def parse_config(path):
    """Flat `key<TAB>value` with `%` and `#` comments. Case preserved.

    Deliberately NOT the native parser: this must see what was written, not
    what the native parser would lowercase it into. Comparing the two is how
    failure (3) becomes visible.
    """
    cfg = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            cfg[parts[0]] = parts[1].split("#")[0].strip()
    return cfg


def native_value(key, raw):
    """What the native parser would hand the model for this key.

    Mirrors general_modules/load_config.py: values of keys outside PATH_KEYS
    are lowercased. Reported when it differs from what the config says.
    """
    try:
        sys.path.insert(0, str(METHOD_REPO))
        from general_modules.load_config import PATH_KEYS
    except Exception:
        return None
    finally:
        if sys.path and sys.path[0] == str(METHOD_REPO):
            sys.path.pop(0)
    return raw if key in PATH_KEYS else raw.lower()


# ---------------------------------------------------------------------------
# filesystem facts
# ---------------------------------------------------------------------------

def case_variants(path, root):
    """Components of `path` under `root` that have case-differing twins on disk.

    Linux is case-sensitive, so two spellings can both exist and hold different
    data. A config names one of them and nothing downstream notices the other.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return []
    findings, here = [], root
    for part in rel.parts:
        if not here.is_dir():
            break
        try:
            entries = list(here.iterdir())
        except OSError:
            break
        twins = [e.name for e in entries
                 if e.name.lower() == part.lower() and e.name != part]
        if twins:
            findings.append((part, twins))
        here = here / part
    return findings


def h5_report(path, max_samples):
    """Structured facts about one HDF5, or an error string. Never raises.

    Returns a dict so the checker and the dump render from the same reading --
    two code paths reading the file separately is how they drift apart.
    """
    rep = {"path": path, "exists": path.exists(), "error": None,
           "top": None, "n_samples": None, "ids": None, "shape": None,
           "dtype": None, "members": None, "rows": None, "ndim": None}
    if not path.exists():
        return rep
    try:
        st = path.stat()
        rep["size_mb"] = st.st_size / (1024 * 1024)
        rep["mtime"] = datetime.datetime.fromtimestamp(st.st_mtime)
    except OSError:
        pass
    try:
        import h5py
        import numpy as np
    except ImportError:
        rep["error"] = "h5py not importable in this interpreter"
        return rep
    try:
        with h5py.File(path, "r") as f:
            rep["top"] = list(f.keys())[:12]
            if DATA_GROUP not in f:
                rep["error"] = f"no /{DATA_GROUP} group -- not the mesh contract"
                return rep
            keys = list(f[DATA_GROUP].keys())
            rep["n_samples"] = len(keys)
            rep["ids"] = keys[:6]
            rows = []
            for sid in keys[:max_samples]:
                grp = f[f"{DATA_GROUP}/{sid}"]
                if not isinstance(grp, h5py.Group) or NODAL not in grp:
                    rep["error"] = (
                        f"/{DATA_GROUP}/{sid} has no {NODAL} "
                        f"(members={list(grp.keys())[:8]})"
                        if isinstance(grp, h5py.Group) else
                        f"/{DATA_GROUP}/{sid} is not a group")
                    return rep
                nd = grp[NODAL]
                if rep["shape"] is None:
                    rep["shape"], rep["dtype"] = nd.shape, str(nd.dtype)
                    rep["members"] = list(grp.keys())
                    rep["ndim"] = nd.ndim
                if nd.ndim != 3:
                    rep["error"] = (f"{NODAL} is {nd.ndim}-D {nd.shape}; the "
                                    f"contract is 3-D [feature, timestep, node]")
                    return rep
                block = np.asarray(nd[:, -1, :], dtype=np.float64)
                rows.append((sid, [(float(r.min()), float(r.max()),
                                    float(r.max() - r.min()),
                                    bool((r == 0).all()))
                                   for r in block]))
            rep["rows"] = rows
    except Exception as exc:                       # corrupt file, lock, ...
        rep["error"] = f"{type(exc).__name__}: {exc}"
    return rep


def degenerate_count(rep):
    """(n_checked, n_flat) on the z-displacement row, or None if unreadable."""
    if rep["rows"] is None:
        return None
    n = len(rep["rows"])
    flat = sum(1 for _, cols in rep["rows"]
               if Z_DISP_CHANNEL < len(cols) and cols[Z_DISP_CHANNEL][2] == 0.0)
    return n, flat


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def render_h5(rep, indent="    "):
    """Everything known about one file, for the failure dump."""
    L = [f"{indent}{rep['path']}"]
    for part, twins in case_variants(rep["path"], REPO_ROOT):
        L.append(f"{indent}  CASE: this path says '{part}', disk also has "
                 f"{twins} -- on Linux those are DIFFERENT directories")
    if not rep["exists"]:
        L.append(f"{indent}  MISSING")
        parent = rep["path"].parent
        if parent.is_dir():
            sibs = sorted(p.name for p in parent.iterdir())
            L.append(f"{indent}  {parent} holds {len(sibs)} entries:")
            for s in sibs[:25]:
                L.append(f"{indent}    {s}")
            if len(sibs) > 25:
                L.append(f"{indent}    ... {len(sibs) - 25} more")
        else:
            L.append(f"{indent}  parent directory does not exist either: {parent}")
        return L
    if "size_mb" in rep:
        L.append(f"{indent}  {rep['size_mb']:.1f} MB, modified {rep['mtime']}")
    if rep["error"]:
        L.append(f"{indent}  ERROR: {rep['error']}")
        if rep["top"] is not None:
            L.append(f"{indent}  top level: {rep['top']}")
        return L
    more = " ..." if rep["n_samples"] > len(rep["ids"]) else ""
    L.append(f"{indent}  /data holds {rep['n_samples']} sample(s); "
             f"ids {rep['ids']}{more}")
    L.append(f"{indent}  nodal_data {rep['shape']} [feature, timestep, node] "
             f"{rep['dtype']}; members {rep['members']}")
    for sid, cols in (rep["rows"] or [])[:2]:
        L.append(f"{indent}  sample {sid}, last timestep:")
        L.append(f"{indent}    row  min           max           spread        allzero")
        for r, (lo, hi, sp, zero) in enumerate(cols):
            mark = "  <== z-displacement" if r == Z_DISP_CHANNEL else ""
            L.append(f"{indent}    {r:>3}  {lo:<12.5g}  {hi:<12.5g}  "
                     f"{sp:<12.5g}  {str(zero):<5}{mark}")
    return L


def render_diagnosis(cfg_path, reports):
    """The full dump for one failing config: what was written, what was opened."""
    L = ["", "=" * 78,
         f"DIAGNOSIS  {cfg_path.name}", "=" * 78,
         f"  config: {cfg_path}"]
    cfg = parse_config(cfg_path)
    L.append("")
    L.append("  as written in the config / as the native parser would pass it:")
    for key in DATASET_KEYS:
        raw = cfg.get(key)
        if raw is None:
            L.append(f"    {key:<16} (ABSENT)")
            continue
        seen = native_value(key, raw)
        L.append(f"    {key:<16} {raw}")
        if seen is not None and seen != raw:
            L.append(f"    {'':<16} -> parser gives {seen}")
            L.append(f"    {'':<16}    LOWERCASED: '{key}' is not in PATH_KEYS, "
                     f"so the model opens a DIFFERENT path than this names.")
    for key in DATASET_KEYS:
        if key in reports:
            L.append("")
            L.append(f"  {key}:")
            L += render_h5(reports[key], indent="    ")

    # The training file is known-good, so a differing shape or row layout in the
    # eval file localises the problem immediately.
    tr, ev = reports.get("dataset_dir"), reports.get("eval_dataset")
    if tr and ev and tr["shape"] and ev["shape"]:
        L.append("")
        L.append("  training file vs eval file:")
        L.append(f"    features   {tr['shape'][0]:<8} vs {ev['shape'][0]}")
        L.append(f"    timesteps  {tr['shape'][1]:<8} vs {ev['shape'][1]}")
        L.append(f"    nodes      {tr['shape'][2]:<8} vs {ev['shape'][2]}")
        L.append(f"    samples    {tr['n_samples']:<8} vs {ev['n_samples']}")
        if tr["shape"][0] != ev["shape"][0]:
            L.append(f"    -> different feature count: row {Z_DISP_CHANNEL} is "
                     f"not the same quantity in both files")
        if tr["rows"] and ev["rows"]:
            t_live = [r for r, c in enumerate(tr["rows"][0][1]) if c[2] != 0.0]
            e_live = [r for r, c in enumerate(ev["rows"][0][1]) if c[2] != 0.0]
            L.append(f"    rows carrying variation:  train {t_live}   eval {e_live}")
            if Z_DISP_CHANNEL not in e_live and e_live:
                L.append(f"    -> row {Z_DISP_CHANNEL} is flat in the eval file "
                         f"but rows {e_live} are not; either this is the wrong "
                         f"file or Z_DISP_CHANNEL is wrong for it")
    return L


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(HERE))
    ap.add_argument("--max-samples", type=int, default=8,
                    help="samples read per file for the degeneracy check")
    ap.add_argument("--inspect", action="store_true",
                    help="dump the datasets' structure even when nothing fails")
    args = ap.parse_args()

    cfg_dir = pathlib.Path(args.config_dir)
    configs = sorted(cfg_dir.glob("config_infer_*.txt"))
    if not configs:
        print(f"FAIL: no config_infer_*.txt in {cfg_dir}", file=sys.stderr)
        print("      Generate them first:  python gen_sweep_configs.py",
              file=sys.stderr)
        return 1

    print(f"Checking the eval/histogram inputs of {len(configs)} inference configs")
    print(f"  method repo = {METHOD_REPO}")

    cache = {}                       # resolved path -> h5_report, read once

    def report_for(path):
        if path not in cache:
            cache[path] = h5_report(path, args.max_samples)
        return cache[path]

    def reports_for(cfg):
        out = {}
        for key in DATASET_KEYS:
            raw = cfg.get(key)
            if raw is not None:
                out[key] = report_for((METHOD_REPO / raw).resolve())
        return out

    bad, failing = 0, []
    for cfg_path in configs:
        cfg = parse_config(cfg_path)
        problems = []
        reports = reports_for(cfg)

        for key in DATASET_KEYS:
            raw = cfg.get(key)
            if raw is None:
                if key == "eval_dataset":
                    problems.append("no 'eval_dataset' key -- the histogram "
                                    "will be SKIPPED and the warpage table empty")
                elif key == "infer_dataset":
                    problems.append("no 'infer_dataset' key")
                continue
            seen = native_value(key, raw)
            if seen is not None and seen != raw:
                problems.append(
                    f"'{key}' is not in PATH_KEYS: the native parser lowercases "
                    f"it to {seen}, so the model opens a different file than "
                    f"this config names")

        p_infer = reports.get("infer_dataset")
        p_eval = reports.get("eval_dataset")
        if p_infer and p_eval:
            if p_infer["path"] == p_eval["path"]:
                problems.append("eval_dataset == infer_dataset -- ground truth "
                                "would be the answer-stripped file")
            for key, rep in (("infer_dataset", p_infer), ("eval_dataset", p_eval)):
                if not rep["exists"]:
                    problems.append(f"{key} not found: {rep['path']}")
                elif rep["error"]:
                    problems.append(f"{key} unreadable: {rep['error']}")
            for part, twins in case_variants(p_eval["path"], REPO_ROOT):
                problems.append(f"path case ambiguous: '{part}' vs {twins} on disk")

            dc = degenerate_count(p_eval)
            if dc:
                n, flat = dc
                if flat == n:
                    problems.append(
                        f"eval_dataset row {Z_DISP_CHANNEL} is flat in all {n} "
                        f"sampled entries -- GT spreads would be all zero")
                elif flat:
                    problems.append(
                        f"eval_dataset row {Z_DISP_CHANNEL} is flat in "
                        f"{flat}/{n} sampled entries")

        if problems:
            bad += 1
            failing.append((cfg_path, reports))
            print(f"  FAIL {cfg_path.name}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"  ok   {cfg_path.name}")

    if bad:
        # One dump per distinct set of datasets: every arm shares the same three
        # files, so dumping all 24 would bury the finding it is meant to expose.
        shown = set()
        for cfg_path, reports in failing:
            sig = tuple(sorted(str(r["path"]) for r in reports.values()))
            if sig in shown:
                continue
            shown.add(sig)
            for line in render_diagnosis(cfg_path, reports):
                print(line)

        print("", file=sys.stderr)
        print(f"{bad}/{len(configs)} inference configs would not produce a valid "
              f"histogram.", file=sys.stderr)
        print("Fix before spending GPU time. Most likely, in order:",
              file=sys.stderr)
        print("  1. stale configs   python gen_sweep_configs.py", file=sys.stderr)
        print("  2. unsynced code   the PATH_KEYS fix lives in "
              "methods/*/general_modules/load_config.py", file=sys.stderr)
        print("  3. wrong file      compare the dump above against the training "
              "file's structure", file=sys.stderr)
        return 1

    if args.inspect:
        for line in render_diagnosis(configs[0],
                                     reports_for(parse_config(configs[0]))):
            print(line)

    print(f"\nAll {len(configs)} inference configs have real ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
