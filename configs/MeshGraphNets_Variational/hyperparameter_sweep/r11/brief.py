#!/usr/bin/env python3
"""A results block short enough to retype by hand, for B300 round 11 (latent placement; a z_g the decoder reads).

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/brief.py
    python .../r11/brief.py --long     # every number, not for typing

Standard library only; run it with any Python 3.8+, while the jobs run or
after. It prints a column legend and a numbered job index (read them, do not
type them), then the block to type back. Jobs are numbers (#, the jobs.txt
order); R1-R5 are runs of rounds 8-10 shown beside this round's (the base's
seeds 42/43/44 and r9's gfix_tf_k8, read where those rounds left them). The block
ends with one line per arm: its marks, w2 per seed, the mean and, once
diagnose_135.sh has run, where the error comes from:

    ==== TYPE THIS BACK ====
    R11 135 41h D10/16 F1         round, node, run hours, done/jobs, failed
    floor 27 19 36                the 256 FE results split 128/128: median, 5%, 95%
    run 14:40%:410 15:12%:395     training: #, epochs done, seconds of the last epoch
    inf 5 6                       inference running
    wait 12-14                    not started yet
    stop 7                        was running when the run died (run the round again)
    9 F:train CUDA out of memory  a failure: stage and the first error line
    d 9:tr                        a diagnosis part that failed (job:part)
    a0 66/95/88 m83 ho 27/31/87 tr 56/56/91 g0 l100        the base
    a4 += 70/75/80 m75 ho 27/33/76 tr 56/56/70 g0 l100     an arm: marks, w2 per seed, mean, diagnosis
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
ROUND = os.path.basename(HERE)   # r11
# Runs of earlier rounds shown beside this round's: (number, round, job, arm, seed, diag dir under output/).
# r8's run was diagnosed by r9's watcher; r9's and r10's own jobs under output/b300_rN/<job>/diag.
REFS = [
    ("R1", "r8", "hybrid_e360", "base", 42, "b300_r9/ref_r8_hybrid_e360/diag"),
    ("R2", "r9", "hybrid_base_s43", "base", 43, "b300_r9/hybrid_base_s43/diag"),
    ("R3", "r10", "hybrid_base_s44", "base", 44, "b300_r10/hybrid_base_s44/diag"),
    ("R4", "r9", "hybrid_gfix_tf_k8_s42", "tfk8", 42, "b300_r9/hybrid_gfix_tf_k8_s42/diag"),
    ("R5", "r9", "hybrid_gfix_tf_k8_s43", "tfk8", 43, "b300_r9/hybrid_gfix_tf_k8_s43/diag"),
]
NODE = "135"
EPOCH_RE = re.compile(r"^Elapsed: ([\d.]+)s Epoch (\d+)")
LEGEND = """\
COLUMNS (do not type)   every number x100, rounded ("27" = 0.27)
  aN    one line per arm (ARMS below), the last lines of the block:
        marks two characters, one per comparison (blank before the arm has two seeds):
              1st against the base (a0)      2nd against the arm it extends (ARMS: "vs")
              + every seed beats every seed of the other, - every seed loses, = neither
        w2    per seed, in seed order 42/43(/44). a0 is r8's hybrid_e360, r9's hybrid_base_s43
              and r10's hybrid_base_s44; the tfk8 line is r9's gfix_tf_k8 (42/43): rows R1-R5.
              Seed spread of one setting: the base 66/95/75 (r8/r9/r10), r6 88/88/107.
              global and wu_a1 have one seed, so no marks
        m     the mean over the seeds
        ho    the held-out geometries, mean over the seeds: floor / posterior sample / prior
              (W2 per geometry, median; from posterior_vs_prior_groups.py, diagnose_135.sh)
        tr    the same on 16 training geometries x 64 draws
        g l   how much the paired correlation (decoded draw vs its FE result, within a geometry)
              falls when z_g, or z_l, is swapped with another draw's: ~0 = the decoder ignores
              that part; large = that part carries the draw. On the hybrid arms: does g leave 0?
              ("-" on the global and local arms: one latent, nothing to swap)
        "-"   not diagnosed yet (or the part failed: a "d" line)
  run   #:% of the epochs done:seconds of the last epoch (r8 alone on a card: 277; r9 two per card)
  #     job number (JOBS below, the jobs.txt order)
  w2    per geometry, W2 between the 1024 draws and the 256 FE results
        over the FE sd, then the RMS over the 8 held-out geometries. 0 = identical;
        ~15 a perfect sampler (estimate); 27 the floor; 100 = the exact FE mean
        with no spread, or the right shape shifted by 1 sd
  w2h   (--long) 128 draws against a random half (128) of the FE results, median of 200
        splits: compare it with the floor line (a perfect sampler sits inside it)
  b     (--long) mean offset    s  width error |sd ratio - 1|    w2^2 = b^2 + s^2 + shape^2
  sdg   (--long) geometric mean of the sd ratios (100 = right width, below = too narrow)
  h     (--long) training hours ("-" = inference only, on an earlier checkpoint)
  floor 128 of each geometry's FE results against the other 128: median, 5%, 95%
"""

# One aN line read out, printed under the legend. Korean where the terminal can print it.
EXAMPLE = "a0 66/95/75 m78 ho 27/29/78 tr 56/57/82 g0 l100"
READING_KO = """\
ONE aN LINE (do not type)   %s
  66/95/75   seed 42/43/44 각각의 held-out W2
  m          그 평균
  ho a/b/c   held-out geometry 8개에서 floor / posterior / prior
  tr         학습 geometry 16개에서 같은 세 값. draw가 64개뿐이라 floor가 56으로 높습니다.
  g0 l100    z_g는 죽어 있고 z_l이 draw를 전부 실어 나른다는 뜻입니다. r8-r10은 모든 arm이 이랬습니다.
             이번 라운드는 hybrid arm에서 g가 0에서 올라가는지, 그리고 global / local / hybrid 중
             어느 쪽의 prior(ho, tr의 셋째 값)가 가장 낮은지를 봅니다.
""" % EXAMPLE
READING_EN = """\
ONE aN LINE (do not type)   %s
  66/95/75   the held-out W2 of seeds 42/43/44
  m          their mean
  ho a/b/c   on the 8 held-out geometries: floor / posterior / prior
  tr         the same three on 16 training geometries. Only 64 draws each, so the floor is high (56).
  g0 l100    z_g is dead and z_l carries the whole draw. Every arm of r8-r10 read the same;
             this round asks whether g leaves 0 on the hybrid arms, and which of global / local /
             hybrid has the lowest prior (the third number of ho and tr).
""" % EXAMPLE


def reading():
    """The Korean block, or the English one where stdout cannot encode Korean."""
    try:
        READING_KO.encode(getattr(sys.stdout, "encoding", None) or "ascii")
    except (UnicodeError, LookupError):
        return READING_EN
    return READING_KO


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
    """"-" = not there yet; "nan" = the number exists and is NaN or infinite (type it as it is)."""
    if v is None:
        return "-"
    if v != v or v in (float("inf"), float("-inf")):
        return "nan"
    return str(int(round(v * 100)))


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


def epoch_seconds(log):
    """Seconds of the last epoch: the difference of the last two `Elapsed:` lines."""
    found = []
    for line in reversed(read_tail(log)):
        m = EPOCH_RE.match(line)
        if m:
            found.append(float(m.group(1)))
            if len(found) == 2:
                return found[0] - found[1]
    return None


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
        ci = cfg(infer_cfg if os.path.isabs(infer_cfg) else os.path.join(REPO, infer_cfg))
        self.infer_only = train_cfg == "-"
        c = ci if self.infer_only else cfg(train_cfg if os.path.isabs(train_cfg) else os.path.join(REPO, train_cfg))
        self.num, self.jid = num, jid
        self.role = role(infer_cfg if os.path.isabs(infer_cfg) else os.path.join(REPO, infer_cfg))
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


# The arms, a0 = the base. arm -> the arm it extends (the 2nd mark).
ARMS = {
    "base": None,
    "global": "base",       # latent_type global (one seed)
    "local": "base",        # latent_type local, the base's GNN prior
    "local_tfk8": "tfk8",   # local with tfk8's prior
    "wu": "base",           # z_l withheld for 60 epochs, no anchor
    "wu_a1": "wu",          # + z_g anchor 1: the anchor's effect (one seed)
    "ld50": "base",         # local_dropout 0.5
    "cap05": "base",        # z_g sigma capped at 0.05
    "tfk8": "base",         # r9's gfix_tf_k8 (R4, R5)
    "tfk8_cn": "tfk8",      # + prior_cond_noise 0.1
    "tfk8_wu": "tfk8",      # + z_l withheld for 60 epochs
}
# hybrid_<arm>_s<seed>; the global and local arms are global_s42, local_s42, local_tfk8_s42.
JID_RE = re.compile(r"^(?:hybrid_)?(.+)_s(\d+)$")


def diag_scores(path):
    """(floor, posterior sample, prior W2 medians, {g_shuffle drop, l_shuffle drop}) of one diagnosis."""
    d = load_json(path)
    if not d:
        return None
    sc = d.get("scores") or {}
    w = [((sc.get(e) or {}).get("w2") or {}).get("p50") for e in ("floor", "posterior_sample", "prior")]
    pair = d.get("paired_within_geometry") or {}
    r_mu = (pair.get("posterior_mean") or {}).get("corr")
    drop = {p: (r_mu - pair[f"{p}_shuffle"]["corr"]) if r_mu is not None and f"{p}_shuffle" in pair
            else None for p in ("g", "l")}
    return w, drop


def mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return sum(vals) / len(vals) if vals else None


def arm_lines(arms):
    """arms: {arm: [(seed, w2 or None, ho, tr)]} -> the aN lines."""
    def w2s(arm):
        return [w for _s, w, _h, _t in sorted(arms.get(arm, [])) if w is not None]

    def mark(w, other):
        if len(w) < 2 or len(other) < 2 or any(v != v for v in w + other):   # a NaN seed: no mark
            return " "
        return "+" if max(w) < min(other) else "-" if min(w) > max(other) else "="

    out = []
    for k, (arm, anchor) in enumerate(ARMS.items()):
        rows = sorted(arms.get(arm, []))
        if not rows:
            continue
        w = w2s(arm)
        marks = ""
        if arm != "base":
            marks = mark(w, w2s("base")) + (mark(w, w2s(anchor)) if anchor != "base" else "")
            marks = marks.rstrip() + " " if marks.strip() else ""
        m = float("nan") if any(v != v for v in w) else mean(w)   # a NaN seed: no mean either
        line = f"a{k} {marks}" + "/".join(x100(v) for _s, v, _h, _t in rows) + f" m{x100(m)}"
        for tag, idx in (("ho", 2), ("tr", 3)):
            ds = [r[idx] for r in rows if r[idx]]
            line += f" {tag} " + "/".join(x100(mean([d[0][i] for d in ds])) for i in range(3))
            if tag == "ho":
                gl = " ".join(f"{p}{x100(mean([d[1][p] for d in ds]))}" for p in ("g", "l"))
        out.append(f"{line} {gl}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", action="store_true", help="also print shape and crps, and every metric per job")
    args = ap.parse_args()

    table = os.path.join(HERE, "jobs.txt")
    if not os.path.isfile(table):
        raise SystemExit(f"no jobs.txt ({table}); generate it with make_sweep.py")
    out_root = os.path.join(REPO, "output", f"b300_{ROUND}")
    status_dir = os.path.join(out_root, "run_logs", NODE, "status")
    tag = ROUND.upper()
    rows = []
    with open(table, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5 and not parts[0].startswith("#") and parts[0] == NODE:
                rows.append(parts[:5])
    jobs = [Job(k, j, t, i, status_dir) for k, (_n, j, _m, t, i) in enumerate(rows, 1)]
    # R1-R5: runs of rounds 8-10, read where those rounds left them.
    refs = []
    for num, rnd, name, arm, seed, ddir in REFS:
        rdir = os.path.join(HERE, "..", rnd)
        r = Job(num, f"{rnd}:{name}", os.path.join(rdir, f"config_train_{name}.txt"),
                os.path.join(rdir, f"config_infer_{name}.txt"),
                os.path.join(REPO, "output", f"b300_{rnd}", "run_logs", NODE, "status"))
        r.arm, r.seed, r.ddir = arm, seed, os.path.join(REPO, "output", *ddir.split("/"))
        r.role = f"{arm}, seed {seed}: round {rnd[1:]}'s {name} (not run here)"
        refs.append(r)

    print(LEGEND)
    print(reading())
    print("JOBS (do not type)")
    width = max(len(j.jid) for j in refs + jobs)
    for j in refs + jobs:
        print(f"  {j.num:>2} {j.jid:<{width}}  {j.role}")
    print()
    print("ARMS (do not type)")
    for k, (arm, anchor) in enumerate(ARMS.items()):
        print(f"  a{k} {arm}" + (f"   (vs a{list(ARMS).index(anchor)} {anchor})" if anchor else ""))
    print()

    abort = read_text(os.path.join(status_dir, "_abort"))
    if abort:
        print("==== TYPE THIS BACK ====")
        print(f"{tag} {NODE} ABORT {abort.splitlines()[0]}")
        print("==== END ====")
        return 0

    alive = any(pid_alive(os.path.join(out_root, f)) for f in os.listdir(out_root)
                if f.startswith(f"run_{NODE}") and f.endswith(".pid")) if os.path.isdir(out_root) else False
    ranked, infl, floor, k_ref = [], [], None, None
    arms = {}
    run, inf, wait, stop, fails = [], [], [], [], []
    n_done = 0
    for j in refs + jobs:
        is_ref = j in refs
        if is_ref and j.state() != "done":
            continue
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
                p, sec = j.percent(), epoch_seconds(j.log)
                run.append(f"{j.num}:{'-' if p is None else p}%:{'-' if sec is None else f'{sec:.0f}'}")
            else:
                inf.append(j.num)
            continue
        if st != "done":
            wait.append(j.num)
            continue
        n_done += f is None and not is_ref
        full = load_json(os.path.join(j.infer_dir, "spread_w2.json")) or {}
        mj = JID_RE.match(j.jid)
        arm, seed = (j.arm, j.seed) if is_ref else (mj.group(1), int(mj.group(2))) if mj else (None, None)
        if arm:
            ddir = j.ddir if is_ref else os.path.join(out_root, j.jid, "diag")
            arms.setdefault(arm, []).append(
                (seed, (full.get("summary") or {}).get("w2"),
                 *(diag_scores(os.path.join(ddir, f"posterior_vs_prior_groups_{t}.json")) for t in ("ho", "tr"))))
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
        shown = cols if args.long else cols[:1]
        ranked.append((w2.get("w2", float("inf")), str(j.num), f"{j.num} " + " ".join(shown) + ("" if w2 else " noplot")))
        for lam, b in sorted((full.get("by_inflation") or {}).items(), key=lambda kv: float(kv[0])):
            s = b.get("summary") or b
            bh = (((b.get("noise") or {}).get("model") or {}).get("w2") or {}).get("p50")
            icols = [x100(s.get("w2")), x100(bh), x100(s.get("bias")), x100(s.get("sd_gap")),
                     x100(s.get("sd_ratio_geomean"))]
            infl.append(f"{j.num}@{lam} " + " ".join(icols if args.long else icols[1:2]))
        if args.long:
            print(f"  {j.num:>2}: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(w2.items())
                                                     if isinstance(v, (int, float)))
                  + " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()) if isinstance(v, (int, float))))

    s, e = read_text(os.path.join(status_dir, "_start")), read_text(os.path.join(status_dir, "_end"))
    if alive:   # the first run's _end while a second run (LANE) still goes
        e = None
    try:
        hours = f"{((float(e) if e else time.time()) - float(s)) / 3600:.0f}" if s else "-"
    except ValueError:
        hours = "-"
    print("==== TYPE THIS BACK ====")
    print(f"{tag} {NODE} {hours}h D{n_done}/{len(jobs)} F{len(fails)}")
    if floor:
        print(f"floor {x100(floor.get('p50'))} {x100(floor.get('p5'))} {x100(floor.get('p95'))}"
              + ("" if k_ref in (None, 128) else f" k{k_ref}"))
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
    for f in sorted(os.listdir(status_dir)) if os.path.isdir(status_dir) else ():
        if f.startswith("_abort_"):   # a second run (LANE) that stopped in its checks
            msg = (read_text(os.path.join(status_dir, f)) or "").splitlines()
            print(f"ABORT lane {f[len('_abort_'):]} {msg[0] if msg else ''}".rstrip())
    dfail = read_text(os.path.join(out_root, "_diag", "diag_fail.txt")) or ""
    nums = {j.jid: j.num for j in jobs}
    dl = [f"{nums.get(p[1], p[1])}:{p[2]}" for p in (l.split() for l in dfail.splitlines()) if len(p) > 2]
    if dl:
        print("d " + " ".join(dl))
    for line in infl:
        print(line)
    if args.long:
        for _w, _n, line in sorted(ranked):
            print(line)
    for line in arm_lines(arms):
        print(line)
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
