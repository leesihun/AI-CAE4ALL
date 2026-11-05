#!/usr/bin/env python3
"""Measured speed and memory of rounds 6 and 7 on node 135, to set round 8's --pack.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/speed.py

Standard library only, any Python 3.8+, on the node (it reads output/b300_r6 and
output/b300_r7). One line per group of training runs with the same batch size,
checkpointing and number of jobs on the card, median over the runs in it:

    ==== TYPE THIS BACK ====
    R8SPEED 135                   node
    b16 c0 n1 s540 p171 r190 k1   batch 16, no checkpointing, alone: 540 s/epoch, peak 171 GB, reserved 190, 1 run
    b16 c1 n2 s700 p48 r60 k3     batch 16, checkpointing, 2 jobs on the card, 3 runs
    gpu 0 98 181                  nvidia-smi now: card, % utilization, GB used
    ==== END ====

s = median seconds per epoch over a run's last 20 epochs (positive `Elapsed`
steps only, so a resume does not count). n = jobs training on that card at the
middle of that window (from the status files' train_start/train_end and the GPU
in run_135*.out). p/r = the largest `VRAM peak` / `reserved` the run logged.
Only runs at latent_dim 128, prior_hidden_dim 128 and prior_mp_layers 3 are
counted (the cost of the others differs for reasons other than packing).
"""
import os
import re
import statistics
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
SWEEP = os.path.dirname(HERE)
NODE = "135"
ROUNDS = ("r6", "r7")
WINDOW = 20
EPOCH_RE = re.compile(r"^Elapsed: ([\d.]+)s Epoch (\d+).*?VRAM peak=([\d.]+)GB reserved=([\d.]+)GB")
START_RE = re.compile(r"\[gpu (\d+)\] (\S+)\s+train start")


def cfg(path):
    out = {}
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split("%", 1)[0].strip().split(None, 1)
                if len(parts) == 2:
                    out.setdefault(parts[0].lower(), parts[1].split(",")[0].split()[0].lower())
    except OSError:
        pass
    return out


def read_float(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return float(fh.read().strip())
    except (OSError, ValueError):
        return None


def log_stats(path):
    """(median s/epoch over the last WINDOW steps, max peak, max reserved) from a train.log."""
    steps, peak, res, prev = [], 0.0, 0.0, None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = EPOCH_RE.match(line)
                if not m:
                    continue
                t = float(m.group(1))
                if prev is not None and t > prev:
                    steps.append(t - prev)
                prev = t
                peak, res = max(peak, float(m.group(3))), max(res, float(m.group(4)))
    except OSError:
        return None
    if not steps:
        return None
    return statistics.median(steps[-WINDOW:]), peak, res


def runs(rnd):
    """Every training run of a round: dicts with its GPU, interval and config."""
    sdir = os.path.join(SWEEP, rnd)
    out_root = os.path.join(REPO, "output", f"b300_{rnd}")
    status = os.path.join(out_root, "run_logs", NODE, "status")
    gpu = {}
    if os.path.isdir(out_root):
        for name in sorted(os.listdir(out_root)):
            if name.startswith(f"run_{NODE}") and name.endswith(".out"):
                with open(os.path.join(out_root, name), encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        m = START_RE.search(line)
                        if m:
                            gpu[m.group(2)] = m.group(1)   # the last start wins
    found = []
    try:
        lines = open(os.path.join(sdir, "jobs.txt"), encoding="utf-8").read().splitlines()
    except OSError:
        return found
    for line in lines:
        parts = line.split()
        if len(parts) < 5 or parts[0] != NODE or parts[3] == "-":
            continue
        jid, c = parts[1], cfg(os.path.join(REPO, parts[3]))
        if jid not in gpu or not c:
            continue
        log = c.get("log_file_dir", "")
        log = os.path.join(REPO, log[6:] if log.startswith("../../") else log)
        start, end = read_float(os.path.join(status, f"{jid}.train_start")), \
            read_float(os.path.join(status, f"{jid}.train_end"))
        found.append(dict(jid=jid, gpu=gpu[jid], start=start, end=end, log=log, c=c))
    return found


def main():
    groups = {}
    for rnd in ROUNDS:
        rs = runs(rnd)
        for r in rs:
            c = r["c"]
            if (c.get("latent_dim"), c.get("prior_hidden_dim"), c.get("prior_mp_layers")) != ("128", "128", "3"):
                continue
            st = log_stats(r["log"])
            if st is None or r["start"] is None:
                continue
            s, peak, res = st
            try:
                last = os.path.getmtime(r["log"])
            except OSError:
                continue
            mid = last - s * min(WINDOW, 10)
            n = sum(1 for o in rs if o["gpu"] == r["gpu"] and o["start"] is not None
                    and o["start"] <= mid <= (o["end"] if o["end"] and o["end"] >= o["start"] else time.time()))
            key = (int(c.get("batch_size", 0)), int(c.get("use_checkpointing") == "true"), max(n, 1))
            groups.setdefault(key, []).append((s, peak, res))
    print("==== TYPE THIS BACK ====")
    print(f"R8SPEED {NODE}")
    for (bs, ck, n), v in sorted(groups.items()):
        s = statistics.median(x[0] for x in v)
        p, r = max(x[1] for x in v), max(x[2] for x in v)
        print(f"b{bs} c{ck} n{n} s{s:.0f} p{p:.0f} r{r:.0f} k{len(v)}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        for line in smi.splitlines():
            i, u, m = [x.strip() for x in line.split(",")]
            print(f"gpu {i} {u} {float(m) / 1024:.0f}")
    except (OSError, subprocess.SubprocessError, ValueError):
        print("gpu -")
    print("==== END ====")


if __name__ == "__main__":
    main()
