#!/usr/bin/env python3
"""A results block short enough to retype by hand, for the B300 round-4 sweep.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r4/brief.py         # every node that ran here
    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r4/brief.py --long  # every number, not for typing

Standard library only; run it with any Python 3.8+, while the jobs run or
after. It reads jobs.txt for the jobs of each node, each training config for
where the job writes, then the spread_metrics.json that inference leaves in
the job's infer/ directory (the 8 held-out ex3 geometries x 64 draws).
Every number is x100 and rounded, so "23" is 0.23.

Header   R4 <node> <h>h D<done>/<jobs> F<failed>
Line     job  w1 sd bias crps skill h
    w1    = w1_norm       (distribution distance / truth sd; 0 is perfect)
    sd    = sd_ratio      (generated width / true width; 100 is right)
    bias  = dmean_norm    (signed mean offset / truth sd; negative = generated mean below truth)
    crps  = crps_norm
    skill = spread_skill
    h     = training hours
A failure appends F:<stage> and the first error line. A job still training
prints "run NN%" (epochs done over the configured total; SAR and LDGN print
their stage and epoch, "vae e120" / "sar e40" / "ldgn e300", since their epoch
caps are only ceilings and the learning-rate stop ends them earlier). "wait" = no epoch logged yet, "inf" =
inference running. A node whose checks aborted prints only "ABORT <reason>".
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
NODES = ("135",)
OUT_ROOT = os.path.join(REPO, "output", "b300_r4")
EPOCH_RE = re.compile(r"^Elapsed: ([\d.]+)s Epoch (\d+)")
# Two-stage baselines: method -> second stage. Each stage logs beside
# log_file_dir as <root>_vae<ext>, then <root>_<stage><ext>.
STAGED = {"sar": "sar", "ldgn": "ldgn"}


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


def method_path(value):
    """A config path is relative to methods/<Name>/ (the native cwd), so '../../x' is REPO/x."""
    value = value.replace("\\", "/")
    return os.path.join(REPO, value[6:] if value.startswith("../../") else value)


def x100(v):
    return "-" if v is None else str(int(round(v * 100)))


def metrics(path):
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        return {k: float(v) for k, v in m.items() if isinstance(v, (int, float))}
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


def read_jobs():
    """jobs.txt -> [(node, job, method, train_cfg, infer_cfg)]."""
    rows = []
    with open(os.path.join(HERE, "jobs.txt"), encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) == 5 and not parts[0].startswith("#"):
                rows.append(tuple(parts))
    return rows


class Job:
    """One train/infer config pair plus everything its run left on disk."""

    def __init__(self, jid, method, train_cfg, infer_cfg, status_dir):
        c = cfg(os.path.join(REPO, train_cfg))
        ci = cfg(os.path.join(REPO, infer_cfg))
        self.jid, self.method = jid, method
        self.model = method_path(c["modelpath"])
        self.log = method_path(c["log_file_dir"])
        self.infer_dir = method_path(ci["inference_output_dir"])
        if method in STAGED:
            root, ext = os.path.splitext(self.log)
            self.stages = [(st, f"{root}_{st}{ext or '.log'}") for st in ("vae", STAGED[method])]
        self.epochs = int(c.get("training_epochs", 0))
        self.status = {}
        for key in ("state", "err", "train_start", "train_end", "inferfail"):
            self.status[key] = read_text(os.path.join(status_dir, f"{jid}.{key}"))

    def progress(self):
        """(flag text, hours) for a job still training: ' run NN%' / ' vae e120' / ' wait'."""
        if self.method in STAGED:
            for stage, log in reversed(self.stages):
                done, secs = last_epoch(log)
                if done is not None:
                    return f" {stage} e{done}", secs / 3600
            return " wait", None
        done, secs = last_epoch(self.log)
        if done is None:
            return " wait", None   # queued, or still building the hierarchy cache
        frac = done / self.epochs if self.epochs else 0.0
        return f" run {int(100 * frac)}%", secs / 3600

    def hours(self):
        s, e = self.status.get("train_start"), self.status.get("train_end")
        try:
            if s:
                return ((float(e) if e else time.time()) - float(s)) / 3600
        except ValueError:
            pass
        return self.progress()[1]

    def state(self):
        st = self.status.get("state")
        if st:
            return st
        if os.path.isfile(self.model):
            return "done"   # no state file: a model means training finished
        return "pending"

    def failure(self):
        st = self.state()
        if st.startswith("fail"):
            stage = st.split()[1] if len(st.split()) > 1 else "?"
            return stage, (self.status.get("err") or "").replace("\n", " ")
        inf = self.status.get("inferfail")
        if inf:
            return "infer", (self.status.get("err") or "").replace("\n", " ")
        return None

    def flags(self):
        f = self.failure()
        if f:
            return f" F:{f[0]} {f[1][:40]}".rstrip()
        st = self.state()
        if st in ("train", "queued", "pending"):
            return self.progress()[0] if st == "train" else " wait"
        if st == "infer":
            return " inf"
        return ""


def run_hours(status_dir):
    s = read_text(os.path.join(status_dir, "_start"))
    e = read_text(os.path.join(status_dir, "_end"))
    try:
        return ((float(e) if e else time.time()) - float(s)) / 3600 if s else None
    except ValueError:
        return None


def node_block(node, rows, long):
    status_dir = os.path.join(OUT_ROOT, "run_logs", node, "status")
    abort = read_text(os.path.join(status_dir, "_abort"))
    if abort:
        # Nothing ran this time; the job lines would be an older run's.
        return [f"R4 {node} ABORT {abort.splitlines()[0]}"]
    lines, n_done, n_fail = [], 0, 0
    jobs = [Job(j, m, t, i, status_dir) for n, j, m, t, i in rows if n == node]
    for job in jobs:
        n_done += job.state() == "done" and job.failure() is None
        n_fail += job.failure() is not None
        m = metrics(os.path.join(job.infer_dir, "spread_metrics.json")) or {}
        if long and m:
            print(f"  {job.jid:<16} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items())))
        h = job.hours()
        cols = [x100(m.get("w1_norm")), x100(m.get("sd_ratio")), x100(m.get("dmean_norm")),
                x100(m.get("crps_norm")), x100(m.get("spread_skill")),
                f"{h:.0f}" if h is not None else "-"]
        lines.append(f"{job.jid} " + " ".join(cols) + job.flags())
    h = run_hours(status_dir)
    head = f"R4 {node} {'-' if h is None else f'{h:.0f}'}h D{n_done}/{len(jobs)} F{n_fail}"
    return [head] + lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("node", nargs="?", choices=NODES,
                    help="one node; none: every node with a status directory here (else all three)")
    ap.add_argument("--long", action="store_true", help="print every metric per job first")
    args = ap.parse_args()

    rows = read_jobs()
    if args.node:
        nodes = [args.node]
    else:
        nodes = [n for n in NODES if os.path.isdir(os.path.join(OUT_ROOT, "run_logs", n, "status"))]
        nodes = nodes or list(NODES)
    block = []
    for node in nodes:
        block += node_block(node, rows, args.long)
    print("==== TYPE THIS BACK ====")
    print("\n".join(block))
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
