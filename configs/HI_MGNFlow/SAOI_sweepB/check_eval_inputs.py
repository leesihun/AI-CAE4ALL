#!/usr/bin/env python3
"""Validate the inference + histogram chain BEFORE it runs unattended.

Two failures have already cost us a run, and NEITHER raises an exception:

  1. `eval_dataset` missing from the config. rollout.py prints
     "Skipped: no 'eval_dataset' set in config" and exits 0. The sweep then
     reports a clean run with an empty warpage table -- the one metric the
     whole comparison rests on.

  2. `eval_dataset` pointing at the answer-stripped inference file instead of
     its `_orig.h5` twin. The ground-truth column is then all zeros, the
     histogram is drawn against it, and the plot looks perfectly plausible.
     This is worse than (1): a skip is visible, a wrong axis is not.

Both are silent, so nothing downstream can notice. Hence this check, run
before any GPU time is committed rather than after.

Per `config_infer_*.txt` in this directory:
  - `eval_dataset` key present
  - `infer_dataset` and `eval_dataset` both resolve to files that exist
  - the two are not the same file
  - `eval_dataset` carries a non-degenerate z-displacement row: rollout.py
    reads `nodal_data[Z_DISP_CHANNEL, -1, :]` per sample and takes max-min, so
    a file whose row 5 is flat yields an all-zero "ground truth" distribution

Exit 0 = the histogram will have real ground truth. Exit 1 = it will not.
"""
import argparse
import pathlib
import sys

# rollout.py::Z_DISP_CHANNEL -- the row the spread metric is read from. Keep in
# sync; a mismatch here would validate a row the plot never looks at.
Z_DISP_CHANNEL = 5

HERE = pathlib.Path(__file__).resolve().parent
# configs/<Method>/<Sweep>/ -> repo root is three levels up.
REPO_ROOT = HERE.parents[2]
# Native paths in a config are relative to the method repo, because that is the
# cwd the launcher gives the native process.
METHOD_REPO = REPO_ROOT / "methods" / "HI_MGNFlow"


def parse_config(path):
    """Flat `key<TAB>value` with `#` comments -- enough for path keys."""
    cfg = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("%") or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        cfg[parts[0]] = parts[1].split("#")[0].strip()
    return cfg


def case_variants(path, root):
    """Components of `path` (below `root`) that have case-differing twins on disk.

    Returns a list of (configured_component, [other_spellings_present]).
    Empty list = the path is unambiguous. On a case-sensitive filesystem two
    spellings can both exist and hold different data, which is invisible to a
    config that names only one of them.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return []
    findings = []
    here = root
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


def render_case_note(path, indent="       "):
    """One line per ambiguous component, or nothing when the path is clean."""
    out = []
    for part, twins in case_variants(path, REPO_ROOT):
        out.append(f"{indent}CASE: config says '{part}', disk also has "
                   f"{twins} -- on Linux these are DIFFERENT directories")
    return out


def spread_stats(h5_path, max_samples):
    """(n_checked, n_degenerate, example) or None if h5py is unavailable."""
    try:
        import h5py
        import numpy as np
    except ImportError:
        return None
    n, degenerate, example = 0, 0, None
    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            raise RuntimeError(f"no /data group in {h5_path}")
        for sample_id in list(f["data"].keys())[:max_samples]:
            row = f[f"data/{sample_id}/nodal_data"][Z_DISP_CHANNEL, -1, :]
            spread = float(np.max(row) - np.min(row))
            n += 1
            if spread == 0.0:
                degenerate += 1
                if example is None:
                    example = sample_id
    return n, degenerate, example


def inspect(h5_path, max_samples):
    """Print what the file actually holds: sample count, shape, and which rows
    carry variation at the last timestep.

    The checker asserts row Z_DISP_CHANNEL is non-constant. If that assertion
    fires, the cause is one of: the file is the answer-stripped twin, the row
    layout differs here, or the part really is flat. Only the data separates
    those, hence this.
    """
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            print(f"    no /data group; top level = {list(f.keys())[:10]}")
            return
        keys = list(f["data"].keys())
        print(f"    samples: {len(keys)}   first: {keys[:4]}")
        for sample_id in keys[:max_samples]:
            grp = f[f"data/{sample_id}"]
            members = list(grp.keys())
            nd = grp["nodal_data"]
            print(f"    [{sample_id}] nodal_data {nd.shape} "
                  f"(rows, timesteps, nodes)  members={members}")
            block = nd[:, -1, :]          # every row, last timestep
            print("      row  min           max           spread        allzero")
            for r in range(block.shape[0]):
                row = np.asarray(block[r], dtype=np.float64)
                spread = float(row.max() - row.min())
                flag = "  <-- Z_DISP_CHANNEL" if r == Z_DISP_CHANNEL else ""
                print(f"      {r:>3}  {row.min():<12.5g}  {row.max():<12.5g}  "
                      f"{spread:<12.5g}  {bool(np.all(row == 0.0))!s:<5}{flag}")
            if nd.shape[1] > 1:
                first = np.asarray(nd[Z_DISP_CHANNEL, 0, :], dtype=np.float64)
                last = np.asarray(nd[Z_DISP_CHANNEL, -1, :], dtype=np.float64)
                print(f"      row {Z_DISP_CHANNEL} timestep 0 spread = "
                      f"{first.max() - first.min():.5g}, "
                      f"timestep -1 spread = {last.max() - last.min():.5g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-dir", default=str(HERE))
    ap.add_argument("--max-samples", type=int, default=8,
                    help="samples read per eval file for the degeneracy check")
    ap.add_argument("--inspect", action="store_true",
                    help="dump each distinct dataset's structure instead of "
                         "checking -- use when a check fires and you need to "
                         "know whether the file or the assumption is wrong")
    args = ap.parse_args()

    cfg_dir = pathlib.Path(args.config_dir)
    configs = sorted(cfg_dir.glob("config_infer_*.txt"))
    if not configs:
        print(f"FAIL: no config_infer_*.txt in {cfg_dir}", file=sys.stderr)
        print("      Generate them first:  python gen_sweep_configs.py", file=sys.stderr)
        return 1

    if args.inspect:
        seen = []
        for cfg_path in configs:
            cfg = parse_config(cfg_path)
            # dataset_dir first: the training file definitely carries
            # answers, so it is the reference for what a healthy row
            # layout looks like here.
            for key in ("dataset_dir", "infer_dataset", "eval_dataset"):
                val = cfg.get(key)
                if not val:
                    continue
                resolved = (METHOD_REPO / val).resolve()
                if resolved in [r for _, r in seen]:
                    continue
                seen.append((key, resolved))
        for key, resolved in seen:
            print(f"\n{key}: {resolved}")
            for line in render_case_note(resolved, indent="    "):
                print(line)
            if not resolved.exists():
                print("    MISSING")
                sibs = sorted(q.name for q in resolved.parent.glob("*")) \
                    if resolved.parent.is_dir() else []
                if sibs:
                    print(f"    {resolved.parent} holds: {sibs[:20]}")
                continue
            try:
                inspect(resolved, args.max_samples)
            except Exception as exc:
                print(f"    unreadable: {exc}")
        return 0

    print(f"Checking the eval/histogram inputs of {len(configs)} inference configs")
    print(f"  method repo = {METHOD_REPO}")

    bad = 0
    # One eval file is shared by every arm; check each distinct file once.
    checked_files = {}

    for cfg_path in configs:
        cfg = parse_config(cfg_path)
        name = cfg_path.name
        problems = []

        infer_ds = cfg.get("infer_dataset")
        eval_ds = cfg.get("eval_dataset")

        if not eval_ds:
            problems.append("no 'eval_dataset' key -- the histogram will be SKIPPED")
        if not infer_ds:
            problems.append("no 'infer_dataset' key")

        if infer_ds and eval_ds:
            p_infer = (METHOD_REPO / infer_ds).resolve()
            p_eval = (METHOD_REPO / eval_ds).resolve()
            if p_infer == p_eval:
                problems.append(
                    "eval_dataset == infer_dataset -- ground truth would be the "
                    "answer-stripped file (all-zero spreads)")
            if not p_infer.exists():
                problems.append(f"infer_dataset not found: {p_infer}")
            if not p_eval.exists():
                problems.append(f"eval_dataset not found: {p_eval}")
            elif p_eval not in checked_files:
                try:
                    stats = spread_stats(p_eval, args.max_samples)
                except Exception as exc:
                    stats = exc
                checked_files[p_eval] = stats

            # Both spellings can exist on Linux and hold different data; the
            # config names one and nothing downstream notices the other. Report
            # it whether or not the configured spelling resolved.
            for part, twins in case_variants(p_eval, REPO_ROOT):
                problems.append(
                    f"path case ambiguous: config says '{part}', disk also has "
                    f"{twins} -- on Linux these are DIFFERENT directories")

            stats = checked_files.get(p_eval)
            if isinstance(stats, Exception):
                problems.append(f"eval_dataset unreadable: {stats}")
            elif isinstance(stats, tuple):
                n, degenerate, example = stats
                if degenerate == n:
                    problems.append(
                        f"eval_dataset row {Z_DISP_CHANNEL} is flat in all {n} "
                        f"sampled entries -- GT spreads would be all zero "
                        f"(is this the '_orig' twin?)")
                elif degenerate:
                    problems.append(
                        f"eval_dataset row {Z_DISP_CHANNEL} is flat in "
                        f"{degenerate}/{n} sampled entries (e.g. {example})")

        if problems:
            bad += 1
            print(f"  FAIL {name}")
            for p in problems:
                print(f"       - {p}")
        else:
            print(f"  ok   {name}")

    if any(s is None for s in checked_files.values()):
        print("\nNOTICE: h5py not importable in this interpreter, so the "
              "all-zero ground-truth check was skipped.")
        print("        File existence and the eval==infer trap were still checked.")
        print("        Re-run under the method venv to get the full check.")

    if bad:
        print(f"\n{bad}/{len(configs)} inference configs would not produce a valid "
              f"histogram.", file=sys.stderr)
        print("Fix them before spending GPU time:", file=sys.stderr)
        print("  python gen_sweep_configs.py     # regenerates from the "
              "production configs", file=sys.stderr)
        return 1

    print(f"\nAll {len(configs)} inference configs have real ground truth.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
