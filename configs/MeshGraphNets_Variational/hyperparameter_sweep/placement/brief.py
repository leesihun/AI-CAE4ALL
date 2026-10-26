#!/usr/bin/env python3
"""A results block short enough to retype by hand, for one round of the B300 placement sweep.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/placement/brief.py r6
    python .../placement/brief.py r6 --long     # every number, not for typing

Standard library only; run it with any Python 3.8+, while the jobs run or
after. It prints a column legend and a numbered job index (read them, do not
type them), then the block to type back. Jobs are numbers (#, the jobs.txt
order), so a line is only numbers:

    ==== TYPE THIS BACK ====
    R6 135 41h D20/34 F1          round, node, run hours, done/jobs, failed
    floor 27 19 36                the 256 FE results split 128/128: median, 5%, 95%
    3 89 70 45 41 92 -            # w2 w2h b s sdg h   (one line per finished job)
    12@1.5 160 150 45 47 87       #@lam w2 w2h b s sdg (inflation jobs)
    run 14:40% 15:12%             training: # and epochs done
    inf 5 6                       inference running
    wait 16-33                    not started yet
    stop 7                        was running when the run died (run the round again)
    9 F:train CUDA out of memory  a failure: stage and the first error line
    ==== END ====
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
NODE = "135"
EPOCH_RE = re.compile(r"^Elapsed: ([\d.]+)s Epoch (\d+)")
LEGEND = """\
COLUMNS (do not type)   every number x100, rounded ("27" = 0.27)
  #     job number (JOBS below, the jobs.txt order)
  w2    the ranking: per geometry, W2 between the 1024 draws and the 256 FE results
        over the FE sd, then the RMS over the 8 held-out geometries. 0 = identical;
        ~15 a perfect sampler (estimate); 27 the floor; 100 = the exact FE mean
        with no spread, or the right shape shifted by 1 sd
  w2h   128 draws against a random half (128) of the FE results, median of 200
        splits: compare it with the floor line (a perfect sampler sits inside it)
  b     mean offset    s  width error |sd ratio - 1|    w2^2 = b^2 + s^2 + shape^2
  sdg   geometric mean of the sd ratios (100 = right width, below = too narrow)
  h     training hours ("-" = inference only, on an earlier checkpoint)
  #@lam one latent_inflation factor of an inflation job: 256 draws, so compare w2h
  floor 128 of each geometry's FE results against the other 128: median, 5%, 95%"""


def cfg(path):
    """Flat `key value` config -> {lowercased key: first token}, comments stripped."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("%", 1)[0].split("#", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                out.setdefault(parts[0].lower(), parts[1].split(",")[0].split()[0])
    return out


def role(path):
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("% Role:"):
                    return line[len("% Role:"):].strip()
    except OSError:
        pass
    return ""


def method_path(value):
    """A config path is relative to methods/<Name>/ (the native cwd), so '../../x' is REPO/x."""
    value = value.replace("\\", "/")
    return os.path.join(REPO, value[6:] if value.startswith("../../") else value)


def x100(v):
    return "-" if v is None else str(int(round(v * 100)))


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_tail(path, size=65536):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - size))
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def last_epoch(log):
    """(epochs done, seconds) from the last `Elapsed: Xs Epoch k` line (k is 0-based)."""
    for line in reversed(read_tail(log)):
        m = EPOCH_RE.match(line)
        if m:
            return int(m.group(2)) + 1, float(m.group(1))
    return None, None


def pid_alive(pid_file):
    pid = read_text(pid_file)
    if not pid:
        return False
    if os.name != "posix":
        return True   # os.kill(pid, 0) on Windows terminates the process; assume alive
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


class Job:
    """One train/infer config pair plus everything its run left on disk."""

    def __init__(self, num, jid, train_cfg, infer_cfg, status_dir):
        ci = cfg(os.path.join(REPO, infer_cfg))
        self.infer_only = train_cfg == "-"
        c = ci if self.infer_only else cfg(os.path.join(REPO, train_cfg))
        self.num, self.jid = num, jid
        self.role = role(os.path.join(REPO, infer_cfg))
        self.model = method_path(c["modelpath"])
        self.log = method_path(c["log_file_dir"])
        self.infer_dir = method_path(ci["inference_output_dir"])
        self.epochs = int(c.get("training_epochs", 0))
        self.status = {}
        for key in ("state", "err", "train_start", "train_end", "inferfail"):
            self.status[key] = read_text(os.path.join(status_dir, f"{jid}.{key}"))

    def hours(self):
        if self.infer_only:
            return None
        s, e = self.status.get("train_start"), self.status.get("train_end")
        try:
            if s:
                return ((float(e) if e else time.time()) - float(s)) / 3600
        except ValueError:
            pass
        secs = last_epoch(self.log)[1]
        return None if secs is None else secs / 3600

    def state(self):
        st = self.status.get("state")
        if st:
            return st
        if os.path.isfile(os.path.join(self.infer_dir, "spread_metrics.json")):
            return "done"
        return "queued"

    def failure(self):
        st = self.state()
        if st.startswith("fail"):
            stage = st.split()[1] if len(st.split()) > 1 else "?"
            return stage, (self.status.get("err") or "").replace("\n", " ")
        if self.status.get("inferfail"):
            return "infer", (self.status.get("err") or "").replace("\n", " ")
        return None

    def percent(self):
        done = last_epoch(self.log)[0]
        if done is None or not self.epochs:
            return None
        return int(100 * done / self.epochs)


def ranges(nums):
    """[3, 4, 5, 9] -> '3-5 9'."""
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("round", help="the round, e.g. r6")
    ap.add_argument("--long", action="store_true", help="also print shape and crps, and every metric per job")
    args = ap.parse_args()

    table = os.path.join(HERE, args.round, "jobs.txt")
    if not os.path.isfile(table):
        raise SystemExit(f"{args.round} has no jobs.txt ({table}); generate it with make_sweep.py")
    out_root = os.path.join(REPO, "output", f"b300_{args.round}")
    status_dir = os.path.join(out_root, "run_logs", NODE, "status")
    tag = args.round.upper()
    rows = []
    with open(table, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5 and not parts[0].startswith("#") and parts[0] == NODE:
                rows.append(parts[:5])
    jobs = [Job(k, j, t, i, status_dir) for k, (_n, j, _m, t, i) in enumerate(rows, 1)]

    print(LEGEND)
    print()
    print("JOBS (do not type)")
    width = max(len(j.jid) for j in jobs)
    for j in jobs:
        print(f"  {j.num:>2} {j.jid:<{width}}  {j.role}")
    print()

    abort = read_text(os.path.join(status_dir, "_abort"))
    if abort:
        print("==== TYPE THIS BACK ====")
        print(f"{tag} {NODE} ABORT {abort.splitlines()[0]}")
        print("==== END ====")
        return 0

    alive = any(pid_alive(os.path.join(out_root, f)) for f in os.listdir(out_root)
                if f.startswith(f"run_{NODE}") and f.endswith(".pid")) if os.path.isdir(out_root) else False
    lines, floor, k_ref = [], None, None
    run, inf, wait, stop, fails = [], [], [], [], []
    n_done = 0
    for j in jobs:
        f = j.failure()
        st = j.state()
        if f:
            fails.append(f"{j.num} F:{f[0]} {f[1][:40]}".rstrip())
            if st != "done":
                continue
        if st in ("train", "infer"):
            if not alive:
                stop.append(j.num)
            elif st == "train":
                p = j.percent()
                run.append(f"{j.num}:{'-' if p is None else p}%")
            else:
                inf.append(j.num)
            continue
        if st != "done":
            wait.append(j.num)
            continue
        n_done += f is None
        full = load_json(os.path.join(j.infer_dir, "spread_w2.json")) or {}
        m = load_json(os.path.join(j.infer_dir, "spread_metrics.json")) or {}
        w2, noise = full.get("summary") or {}, full.get("noise") or {}
        if noise.get("floor") and floor is None:
            floor, k_ref = noise["floor"].get("w2"), noise.get("k")
        half = ((noise.get("model") or {}).get("w2") or {}).get("p50")
        h = j.hours()
        cols = [x100(w2.get("w2")), x100(half), x100(w2.get("bias")), x100(w2.get("sd_gap")),
                x100(w2.get("sd_ratio_geomean")), "-" if h is None else f"{h:.0f}"]
        if args.long:
            cols += [x100(w2.get("shape")), x100(m.get("crps_norm"))]
        lines.append(f"{j.num} " + " ".join(cols) + ("" if w2 else " noplot"))
        for lam, b in sorted((full.get("by_inflation") or {}).items(), key=lambda kv: float(kv[0])):
            s = b.get("summary") or b
            bh = (((b.get("noise") or {}).get("model") or {}).get("w2") or {}).get("p50")
            lines.append(f"{j.num}@{lam} " + " ".join(
                [x100(s.get("w2")), x100(bh), x100(s.get("bias")), x100(s.get("sd_gap")),
                 x100(s.get("sd_ratio_geomean"))]))
        if args.long:
            print(f"  {j.num:>2} {j.jid}: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(w2.items())
                                                     if isinstance(v, (int, float)))
                  + " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()) if isinstance(v, (int, float))))

    s, e = read_text(os.path.join(status_dir, "_start")), read_text(os.path.join(status_dir, "_end"))
    try:
        hours = f"{((float(e) if e else time.time()) - float(s)) / 3600:.0f}" if s else "-"
    except ValueError:
        hours = "-"
    print("==== TYPE THIS BACK ====")
    print(f"{tag} {NODE} {hours}h D{n_done}/{len(jobs)} F{len(fails)}")
    if floor:
        print(f"floor {x100(floor.get('p50'))} {x100(floor.get('p5'))} {x100(floor.get('p95'))}"
              + ("" if k_ref in (None, 128) else f" k{k_ref}"))
    for line in lines:
        print(line)
    if run:
        print("run " + " ".join(run))
    if inf:
        print("inf " + ranges(inf))
    if wait:
        print("wait " + ranges(wait))
    if stop:
        print("stop " + ranges(stop))
    for line in fails:
        print(line)
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
