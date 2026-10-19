#!/usr/bin/env python3
"""A results block short enough to retype by hand, for the B300 sweeps.

The HI-MGNflow twin of this file (and round 1, which ran HI-MGNflow only)
went with that route on 2026-10-19: it was merged into meshgraphnets-v as
latent_type local, and its old repo and configs are in junk/legacy_HI_MGNFlow/.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/brief.py         # round 2, MGN-V, both studies
    python configs/MeshGraphNets_Variational/hyperparameter_sweep/brief.py SAOI    # one study (or probabilistic)
    python configs/MeshGraphNets_Variational/hyperparameter_sweep/brief.py --long  # every number, not for typing

Standard library only; run it with any Python 3.8+, while the sweep runs or
after it. It reads the checked-in configs for where each arm writes, then the
spread_metrics.json files rollout.py leaves under each inference_output_dir.
Every number is x100 and rounded, so "23" is 0.23.

SAOI line   arm  Ow Os Ob  Hw Hs Hb  h
    O = the three official eval parts (s26fe_main, s26fe_sec, sm_l345u_main)
        on both board halves -- unseen parts for every arm.
    H = the held-out parts cut by saoi_heldout.py: SEEN parts for the
        *_smp arms, UNSEEN geometries for the others.
    w = mean w1_norm          (distribution distance / truth sd; 0 is perfect)
    s = geometric-mean sd_ratio (generated width / true width; 100 is right)
    b = mean |dmean_norm|     (mean offset / truth sd; 0 is perfect)
    h = training hours (the slower half)
ex3 line    arm  w1 sd bias crps skill h
    bias is signed (negative = generated mean below truth); skill = spread_skill.

A failure appends F:<stage>.<half> and the first error line; an arm still
training prints "run NN%" (epochs done over the configured total). A study
whose checks aborted prints only its header, ending in "ABORT <reason>".
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
METHOD_DIR = os.path.basename(os.path.dirname(HERE))    # MeshGraphNets_Variational
M = {"MeshGraphNets_Variational": "mgnv"}.get(METHOD_DIR)
OFFICIAL = ("s26fe_main", "s26fe_sec", "sm_l345u_main")
HALVES = ("bot", "top")
SWEEP = f"configs/{METHOD_DIR}/hyperparameter_sweep"

CAMPAIGNS = {
    "saoi": {"dirs": [f"{SWEEP}/SAOI"],
             "order": ["v_smp", "v_grp", "v_grp_rep", "v_nopart"],
             "status": f"output/b300_r2/saoi/run_logs/{M}/status", "split": "output/b300_r2/saoi/heldout"},
    "ex3": {"dirs": [f"{SWEEP}/probabilistic"],
            "order": ["v_base", "v_rep", "v_mp8", "v_mp12",
                      "v_mp8_pmin01", "v_mp8_aux100", "v_mp8_mmd10", "v_mp8_w256"],
            "status": f"output/b300_r2/ex3/run_logs/{M}/status"},
}

EPOCH_RE = re.compile(r"^Elapsed: ([\d.]+)s Epoch (\d+)")


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


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


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


class Job:
    """One training config plus everything its run left on disk."""

    def __init__(self, conf_path, arm, half, status_dir):
        c = cfg(conf_path)
        self.arm, self.half = arm, half
        self.model = method_path(c["modelpath"])
        self.root = os.path.dirname(self.model)
        self.log = method_path(c["log_file_dir"])
        self.epochs = int(c.get("training_epochs", 0))
        jid = arm if half is None else f"{arm}.{half}"
        self.status = {}
        if status_dir:
            for key in ("state", "err", "train_start", "train_end", "inferfail"):
                self.status[key] = read_text(os.path.join(status_dir, f"{jid}.{key}"))

    def progress(self):
        """(fraction of epochs done, hours from the epoch log's last Elapsed)."""
        frac, hours = None, None
        for line in reversed(read_tail(self.log)):
            m = EPOCH_RE.match(line)
            if m:
                done = int(m.group(2)) + 1
                frac = done / self.epochs if self.epochs else None
                hours = float(m.group(1)) / 3600
                break
        return frac, hours

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
        return "train" if os.path.isfile(self.log) else "pending"

    def failure(self):
        st = self.state()
        if st.startswith("fail"):
            stage = st.split()[1] if len(st.split()) > 1 else "?"
            return stage, (self.status.get("err") or "").replace("\n", " ")
        inf = self.status.get("inferfail")
        if inf:
            first = inf.splitlines()[0].split()
            return ("infer:" + first[0] if first else "infer"), (self.status.get("err") or "")
        return None

    def infer_sets(self):
        """{tag: metrics} for every inference this job wrote, SAOI layout."""
        out = {}
        for p in glob.glob(os.path.join(self.root, "infer", "*", "spread_metrics.json")):
            m = metrics(p)
            if m is not None:
                out[os.path.basename(os.path.dirname(p))] = m
        return out


def discover(camp, status_dir, halves):
    jobs = {}
    for d in CAMPAIGNS[camp]["dirs"]:
        for p in sorted(glob.glob(os.path.join(REPO, d, "config_train_*.txt"))):
            stem = os.path.basename(p)[len("config_train_"):-len(".txt")]
            half = None
            if halves:
                stem, _, half = stem.rpartition("_")
                if half not in HALVES:
                    continue
            jobs.setdefault(stem, []).append(Job(p, stem, half, status_dir))
    order = CAMPAIGNS[camp]["order"]
    arms = [a for a in order if a in jobs] + sorted(a for a in jobs if a not in order)
    return [(a, jobs[a]) for a in arms]


def run_hours(status_dir):
    s = read_text(os.path.join(status_dir, "_start")) if status_dir else None
    e = read_text(os.path.join(status_dir, "_end")) if status_dir else None
    try:
        return ((float(e) if e else time.time()) - float(s)) / 3600 if s else None
    except ValueError:
        return None


def tail_flags(jobs, expected_infers=None):
    """' run NN%' / ' F:stage.half err' suffix, or '' for a clean finished arm."""
    out = ""
    fails = [(j, j.failure()) for j in jobs]
    fails = [(j, f) for j, f in fails if f]
    if fails:
        j, (stage, err) = fails[0]
        where = stage if j.half is None else f"{stage}.{j.half}"
        out += f" F:{where} {err[:40]}".rstrip()
        if len(fails) > 1:
            out += f" (+{len(fails) - 1})"
    fr = [j.progress()[0] for j in jobs if j.state() in ("train", "pending", "queued")]
    if fr and all(f is None for f in fr):
        out += " wait"      # no epoch logged yet: queued, or still building the hierarchy cache
    elif fr:
        out += f" run {int(100 * sum(f or 0.0 for f in fr) / len(fr))}%"
    elif any(j.state().startswith("infer") for j in jobs):
        out += " inf"
    return out


def score_saoi_sets(sets):
    w = mean([m.get("w1_norm") for m in sets])
    s = gmean([m.get("sd_ratio") for m in sets])
    b = mean([abs(m["dmean_norm"]) if "dmean_norm" in m else None for m in sets])
    return [x100(w), x100(s), x100(b)]


def brief_saoi(camp, status_dir, long):
    lines, n_fail, n_done, n_jobs = [], 0, 0, 0
    for arm, jobs in discover(camp, status_dir, halves=True):
        O, H, T = [], [], []
        for j in jobs:
            n_jobs += 1
            n_done += j.state() == "done"
            n_fail += j.failure() is not None
            for tag, m in sorted(j.infer_sets().items()):
                (O if tag in OFFICIAL else T if tag.startswith("train_part") else H).append(m)
                if long:
                    print(f"  {arm:<12} {j.half} {tag:<14} " + " ".join(
                        f"{k}={m[k]:.3f}" for k in ("w1_norm", "sd_ratio", "dmean_norm", "crps_norm")
                        if k in m))
        hrs = [j.hours() for j in jobs]
        hrs = [h for h in hrs if h is not None]
        cols = score_saoi_sets(O) + score_saoi_sets(H)
        cols.append(f"{max(hrs):.0f}" if hrs else "-")
        lines.append(f"{arm} " + " ".join(cols) + tail_flags(jobs))
    return lines, n_jobs, n_done, n_fail


def brief_ex3(camp, status_dir, long):
    lines, n_fail, n_done, n_jobs = [], 0, 0, 0
    for arm, jobs in discover(camp, status_dir, halves=False):
        j = jobs[0]
        n_jobs += 1
        n_done += j.state() == "done"
        n_fail += j.failure() is not None
        m = metrics(os.path.join(j.root, "infer", "spread_metrics.json")) or {}
        if long and m:
            print(f"  {arm:<14} " + " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items())))
        h = j.hours()
        cols = [x100(m.get("w1_norm")), x100(m.get("sd_ratio")), x100(m.get("dmean_norm")),
                x100(m.get("crps_norm")), x100(m.get("spread_skill")), f"{h:.0f}" if h is not None else "-"]
        lines.append(f"{arm} " + " ".join(cols) + tail_flags(jobs))
    return lines, n_jobs, n_done, n_fail


def split_header():
    parts = []
    for half in HALVES:
        plan = os.path.join(REPO, CAMPAIGNS["saoi"]["split"], f"split_{half}.json")
        try:
            with open(plan, encoding="utf-8") as fh:
                p = json.load(fh)
            parts.append(f"{half} G{p['geometries_total']} v{p['roles']['val']['geometries']} "
                         f"t{p['roles']['test']['geometries']} H{len(p['eval_sets'])}")
        except (OSError, ValueError, KeyError):
            parts.append(f"{half} -")
    return " | ".join(parts)


def r2_block(which, long):
    """Header plus one line per arm for one round-2 study of this method."""
    status_dir = os.path.join(REPO, CAMPAIGNS[which]["status"])
    abort = read_text(os.path.join(status_dir, "_abort"))
    if abort:
        # Nothing ran this time; the counts and arm lines would be an older run's.
        return [f"R2 {which} {M} ABORT {abort.splitlines()[0]}"]
    if which == "saoi":
        lines, n, d, f = brief_saoi("saoi", status_dir, long)
        tail = f" | {split_header()}"
    else:
        lines, n, d, f = brief_ex3("ex3", status_dir, long)
        tail = ""
    h = run_hours(status_dir)
    head = f"R2 {which} {M} {'-' if h is None else f'{h:.0f}'}h D{d}/{n} F{f}"
    return [head + tail] + lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", type=str.lower, choices=["saoi", "probabilistic", "ex3"],
                    help="SAOI | probabilistic (= ex3) for one study of round 2; "
                         "none: both round-2 studies")
    ap.add_argument("--long", action="store_true", help="print every metric per eval set first")
    args = ap.parse_args()
    which = "ex3" if args.which == "probabilistic" else args.which

    if M is None:
        sys.exit(f"brief.py must sit in configs/MeshGraphNets_Variational/hyperparameter_sweep/, "
                 f"not {HERE}")
    block = []
    for study in ([which] if which else ["saoi", "ex3"]):
        block += r2_block(study, args.long)
    print("==== TYPE THIS BACK ====")
    print("\n".join(block))
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
