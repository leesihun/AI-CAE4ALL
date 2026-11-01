#!/usr/bin/env python3
"""Generate B300 round 7: HI-MGN-hybrid only, one run per setting, 240 epochs as the base.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r7/make_sweep.py          # writes the configs here
    python .../r7/make_sweep.py --plan   # queue, slots and estimated hours; writes nothing

Standard library only, any Python 3.8+. The base is round 6's hybrid_e240
(w2 69, the best r6 hybrid run), which is not trained again: every job is it
with the changes of its ARM, seed 42, one run each. Edit this file, never a
generated config.

Long jobs get a card to themselves: run_135.sh lists their cards once and the
other cards twice, and these jobs come first in jobs.txt. run_node.sh deals
job s to slot s in its first round (on a restart too, over the unfinished
jobs), so they always land on the single-listed cards. ALONE_GPUS and
SHARED_GPUS must match run_135.sh's GPUS (checked).
"""
import argparse
import dataclasses
import heapq
import re
import sys
from decimal import Decimal
from pathlib import Path

ROUND = "r7"
NODE = "135"
PLACEMENT = "hybrid"

# ================================================================== the sweep
# Round 6's hybrid_ref (round 4's hybrid_mmd with mp_per_level 4, 6, 8, 6, 4 and the
# coarsening 4096, 256) at 240 epochs = round 6's hybrid_e240. beta_aux 10,
# voronoi_clusters and mp_per_level are fixed.
REF = dict(lr="1e-4", ep=240, bs=4, ar=1000, w=128, reg="mmd", mmd=1.0, kl=None, ld=32, ch=4,
           ph=128, pm=3, fb=4, pms=0.05)
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)
SEED = 42   # round 6's hybrid_e240 seed: every arm differs from the base by its change only

# arm -> (changes from REF, alone on a card, what it is). Run order = this order;
# the alone arms must come first.
# r6 hybrid: e60 91, e120 88/88/107 (3 seeds), e240 69 (the base), bs8 at e120 79.
ARMS = {
    # Epochs x batch size; learning rate not rescaled. Optimizer steps relative to
    # the base (e240 bs4): e480 2x, e480_bs8 1x, e480_bs16 0.5x, e360 1.5x, e240_bs8 0.5x.
    "e480": (dict(ep=480), True, "epochs 480 (2x the base's steps)"),
    "e480_bs8": (dict(ep=480, bs=8), True, "epochs 480, batch size 8 (the base's steps)"),
    "e480_bs16": (dict(ep=480, bs=16), True, "epochs 480, batch size 16 (half the base's steps)"),
    "e360": (dict(ep=360), False, "epochs 360 (1.5x the base's steps)"),
    "e240_bs8": (dict(bs=8), False, "batch size 8 (half the base's steps)"),
    # MMD weight (base 1)
    "e240_mmd0.1": (dict(mmd=0.1), False, "lambda_mmd 0.1"),
    "e240_mmd10": (dict(mmd=10.0), False, "lambda_mmd 10"),
    "e240_mmd100": (dict(mmd=100.0), False, "lambda_mmd 100"),
    "e240_mmd1000": (dict(mmd=1000.0), False, "lambda_mmd 1000"),
    "e240_mmd10000": (dict(mmd=10000.0), False, "lambda_mmd 10000"),
    # Per-sample KL instead of MMD. r3's kl_weight 1e-3 did nothing (alpha_recon 1000 dwarfs it).
    "e240_kl0.1": (dict(reg="kl", kl=0.1), False, "latent_reg kl, kl_weight 0.1"),
    "e240_kl1": (dict(reg="kl", kl=1.0), False, "latent_reg kl, kl_weight 1"),
    "e240_kl10": (dict(reg="kl", kl=10.0), False, "latent_reg kl, kl_weight 10"),
    "e240_kl100": (dict(reg="kl", kl=100.0), False, "latent_reg kl, kl_weight 100"),
}
ALONE_GPUS = ("0", "2", "3")        # one job each: the alone arms first, then whatever comes next
SHARED_GPUS = ("4", "5", "6", "7")  # two jobs each (MPS)

DISPLAY = "HI-MGN-hybrid (latent_type hybrid)"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REL = f"configs/MeshGraphNets_Variational/hyperparameter_sweep/{ROUND}"
SWEEP_REL = "configs/MeshGraphNets_Variational/hyperparameter_sweep"

# Estimated card-hours, ONLY for the queue plan (guesses except the anchor):
# 120 epochs at mp 4,6,8,6,4 took 34 h with one job per card and checkpointing on
# (round 5's hybrid_mp28). Round 6 ran at about 0.6-0.7x of these estimates.
H120 = 34.0
SHARE = 1.4            # two jobs on one card (MPS): guessed slowdown of each
BS_FACTOR = {2: 1.3, 4: 1.0, 8: 0.85, 16: 0.75}   # per epoch, guessed
INFER_H = 2.0

# Gradient checkpointing costs 1.52x per step (aarl 3090, bs 1, 2026-10-27), so it is off
# wherever the job fits without it. Peak memory without it, measured there at bs 1: 21 GB per
# 100k nodes at the reference, +1 GB fixed, x17.1/21 for the bf16 LayerNorm AMP now always
# uses. Worst batch in nodes = max of 200k random ex3 batches (mean sample 48k, largest 87k).
CKPT_SPEEDUP = 1.52
WORST_NODES = {2: 175e3, 4: 329e3, 8: 563e3, 16: 1045e3}
MEM_BUDGET_GB = {False: 120, True: 250}   # shared: ~144 GB per job; alone: the whole 288 GB B300
BF16_LN_MEM = 17.1 / 21


@dataclasses.dataclass(frozen=True)
class Spec:
    lr: str = REF["lr"]
    ep: int = REF["ep"]
    bs: int = REF["bs"]
    ar: int = REF["ar"]
    w: int = REF["w"]
    reg: str = REF["reg"]
    mmd: float = REF["mmd"]
    kl: object = REF["kl"]
    ld: int = REF["ld"]
    ch: int = REF["ch"]
    ph: int = REF["ph"]
    pm: int = REF["pm"]
    fb: int = REF["fb"]
    pms: float = REF["pms"]
    seed: int = SEED
    alone: bool = False


def peak_gb_without_checkpointing(spec):
    per_100k = (5.0 + 16.0 * spec.w / 128 + 4.7 * (spec.ph / 128 - 1)
                + 4.7 * (spec.pm / 3 - 1))
    return per_100k * BF16_LN_MEM * WORST_NODES[spec.bs] / 1e5 + 1.0


def use_checkpointing(spec):
    return peak_gb_without_checkpointing(spec) > MEM_BUDGET_GB[spec.alone]


# =========================================================== config bodies
BASE_TEXT = """\
model                            meshgraphnets-v
mode                             train
gpu_ids                          0
parallel_mode                    ddp
split_seed                       42
split_group_attr                 shape_idx
dataset_dir                      ../../dataset/probabilistic/ex3_shell_buckling.h5
infer_dataset                    ../../dataset/probabilistic/ex3_shell_buckling_infer.h5
modelpath                        -
log_file_dir                     -
inference_output_dir             -
infer_timesteps                  1
input_var                        3
output_var                       3
cond_var                         2
feature_loss_weights             1, 1, 1
use_node_types                   True
positional_features              4
time_integration                 ar_ot
training_epochs                  120
batch_size                       4
learningr                        0.0001
weight_decay                     0.0001
warmup_epochs                    5
num_workers                      4
prefetch_factor                  2
grad_accum_steps                 1
augment_geometry                 True
std_noise                        0.0
use_amp                          True
use_checkpointing                True
use_compile                      True
use_ema                          True
ema_decay                        0.99
use_parallel_stats               False
val_interval                     10
test_interval                    50
test_batch_idx                   0, 1
plot_feature_idx                 2
display_trainset                 True
display_testset                  True
write_preprocessing              False
use_world_edges                  False
use_multiscale                   True
edge_var                         8
latent_dim                       128
message_passing_num              28
use_deformed_edge                True
displacement_indices             0, 1, 2
coarsening_type                  voronoi_seedmean
multiscale_levels                2
voronoi_clusters                 4096, 256
mp_per_level                     4, 6, 8, 6, 4
hierarchy_variants               2
hierarchy_seed                   42
hierarchy_cache_keep             False
hierarchy_cache_dir              -
hierarchy_cache_build_workers    48
training_seed                    42
best_by                          crps
inference_samples                16
inference_batch_size             16
use_vae                          True
vae_latent_dim                   32
vae_encoder_mp_layers            3
vae_graph_aware                  True
latent_type                      hybrid
latent_ch                        4
vae_local_mp_layers              2
latent_reg                       mmd
recon_loss                       mse
alpha_recon                      1000
lambda_mmd                       1
mmd_bandwidth                    median
posterior_min_std                0.05
beta_aux                         10
z_conditioning                   adaln
use_conditional_prior            True
prior_type                       gnn_e2e
prior_family                     fm
prior_mp_layers                  3
prior_hidden_dim                 128
prior_nll_weight                 1
prior_grad_to_encoder            0.0
prior_fm_steps                   30
prior_fm_solver                  heun
prior_freeze_epoch               80
"""
EVAL = [
    ("eval_dataset", "../../dataset/probabilistic/ex3_shell_buckling_infer.h5"),
    ("spread_channel", "0, 1, 2"),
    ("spread_label", "u"),
    ("spread_stat", "mag_max"),
    ("make_histogram", "True"),
    ("show_histogram", "False"),
    ("histogram_bins", "30"),
    ("save_rollouts", "False"),
]
SAMPLES, INFER_BATCH = 4, 4   # draws per held-out scene; 256 scenes per geometry -> 1024 draws
JOBS_MARK = "## Jobs (generated by make_sweep.py; everything below is rewritten)"
PATH_KEYS = {"modelpath", "log_file_dir", "inference_output_dir", "hierarchy_cache_dir", "resume_training"}


class Body:
    """An ordered flat `key value` config body."""

    def __init__(self, pairs):
        self.kv = [[k, v] for k, v in pairs]

    def keys(self):
        return [k for k, _ in self.kv]

    def set(self, key, value, after=None):
        for pair in self.kv:
            if pair[0] == key:
                pair[1] = str(value)
                return self
        if after is None:
            self.kv.append([key, str(value)])
        else:
            self.kv.insert(self.keys().index(after) + 1, [key, str(value)])
        return self

    def drop(self, *keys):
        self.kv = [p for p in self.kv if p[0] not in keys]
        return self

    def text(self):
        return "".join(f"{k:<33}{v}\n" for k, v in self.kv)

    def as_dict(self):
        return {k: " ".join(v.split()) for k, v in self.kv}


BASE = [tuple(line.split(None, 1)) for line in BASE_TEXT.splitlines()]


def run_dir(job):
    return f"../../output/b300_{ROUND}/{job}"


def lr_value(lr):
    m, x = lr.split("e-")
    return format(Decimal(m) * Decimal(10) ** -int(x), "f")


def train_body(spec, job):
    b, run = Body(BASE), run_dir(job)
    b.set("modelpath", f"{run}/model.pth").set("log_file_dir", f"{run}/train.log")
    b.set("inference_output_dir", f"{run}/infer")
    b.set("hierarchy_cache_dir", f"../../output/b300_{ROUND}/mscache/mgnv_v{VORONOI[0]}_{VORONOI[1]}")
    b.set("training_epochs", spec.ep).set("prior_freeze_epoch", spec.ep * 2 // 3)
    b.set("batch_size", spec.bs).set("learningr", lr_value(spec.lr))
    b.set("latent_dim", spec.w).set("alpha_recon", spec.ar)
    b.set("posterior_min_std", f"{spec.pms:g}").set("training_seed", spec.seed)
    b.set("vae_latent_dim", spec.ld).set("latent_ch", spec.ch)
    if spec.reg == "kl":
        b.set("latent_reg", "kl").set("kl_weight", f"{spec.kl:g}", after="latent_reg")
        b.drop("lambda_mmd", "mmd_bandwidth")   # the launcher warns: ignored under kl
    else:
        b.set("lambda_mmd", f"{spec.mmd:g}")
    b.set("prior_mp_layers", spec.pm).set("prior_hidden_dim", spec.ph)
    if spec.fb != REF["fb"]:
        b.set("prior_fm_blocks", spec.fb, after="prior_hidden_dim")
    b.set("use_checkpointing", use_checkpointing(spec))
    b.set("resume_training", "True")
    return b


def infer_body(spec, job):
    b = train_body(spec, job)
    b.set("mode", "inference").set("log_file_dir", f"{run_dir(job)}/infer.log").set("batch_size", 8)
    b.drop("hierarchy_cache_dir", "hierarchy_cache_build_workers", "resume_training")
    b.set("inference_samples", SAMPLES).set("inference_batch_size", INFER_BATCH)
    for k, v in EVAL:
        b.set(k, v)
    return b


# ======================================================================= jobs
@dataclasses.dataclass
class Job:
    name: str
    spec: Spec
    change: str


def est_hours(spec, alone):
    """Card-hours of one job on a slot of the given kind (alone or two per card)."""
    ckpt = 1.0 if use_checkpointing(spec) else 1.0 / CKPT_SPEEDUP
    return (H120 * spec.ep / 120 * (spec.w / 128) ** 1.5 * BS_FACTOR[spec.bs]
            * (1.0 if alone else SHARE) * ckpt + INFER_H)


def build_jobs():
    jobs, seen_shared = [], False
    for arm, (changes, alone, what) in ARMS.items():
        if alone and seen_shared:
            raise SystemExit(f"{arm}: the alone arms must come first in ARMS")
        seen_shared |= not alone
        spec = dataclasses.replace(Spec(), alone=alone, **changes)
        jobs.append(Job(f"{PLACEMENT}_{arm}", spec, what + ("; alone on a card" if alone else "")))
    n_alone = sum(j.spec.alone for j in jobs)
    if n_alone > len(ALONE_GPUS):
        raise SystemExit(f"{n_alone} alone arms but {len(ALONE_GPUS)} ALONE_GPUS")
    return jobs


def slot_gpus():
    """The GPUS order run_135.sh must use: alone cards, then the shared cards twice."""
    return list(ALONE_GPUS) + list(SHARED_GPUS) * 2


def check_run_135():
    text = (HERE / "run_135.sh").read_text(encoding="utf-8")
    m = re.search(r'GPUS="\$\{GPUS:-([0-9 ]+)\}"', text)
    want = " ".join(slot_gpus())
    if not m or " ".join(m.group(1).split()) != want:
        raise SystemExit(f"run_135.sh GPUS default must be \"{want}\" (ALONE_GPUS, then SHARED_GPUS twice)")


# ==================================================================== headers
DATA = """\
% Dataset: shell buckling, 150 geometries x 64 imperfection draws (ex3_shell_buckling.h5); held-out file
% ex3_shell_buckling_infer.h5 = 8 unseen geometries x 256 draws. split_group_attr keeps a geometry's draws together.
% Fields: input [0:3] x,y,z | output [3:6] ux,uy,uz | conditioner [6:8] thickness, delta | [8] node type.
% Static (T=1): the model predicts the field in one step; the spread comes from the unseen imperfection.
"""


def header(job, kind):
    lines = [f"% ex3 B300 hybrid sweep ({ROUND}), job {job.name}: {DISPLAY}",
             f"% Role: {job.change}",
             "% Base: round 6's hybrid_e240 (hybrid_ref at 240 epochs; w2 69), not trained again.",
             f"% Generated by {REL}/make_sweep.py; edit that file, not this one.",
             f"% Run paths: output/b300_{ROUND}/<job>/; one shared hierarchy cache; resume_training True."
             if kind == "train" else
             "% 4 draws per held-out scene = 1024 per geometry against its 256 FE results (spread_metrics.json)."]
    if kind == "train" and job.spec.alone:
        lines.append("% Runs alone on a card (run_135.sh); use_checkpointing is set for that memory budget.")
    return "\n".join(lines) + "\n%\n" + DATA


def write(path, text):
    path.write_bytes(text.encode("utf-8"))   # LF, no BOM


def self_check():
    """The base equals round 6's hybrid_e240 except build-only keys and paths."""
    def read(p):
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("%", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = " ".join(parts[1].split())
        return out

    # hierarchy_cache_build_workers, use_checkpointing, use_compile: speed only.
    # amp_layernorm: retired 2026-10-30; r6 set it True, which is what AMP now always does.
    build_only = {"hierarchy_cache_build_workers", "use_checkpointing", "use_compile", "amp_layernorm"}
    ok = True
    for kind in ("train", "infer"):
        ref_path = REPO / SWEEP_REL / "r6" / f"config_{kind}_hybrid_e240.txt"
        if not ref_path.is_file():
            print(f"  self-check: r6/{ref_path.name} not found, skipped")
            continue
        body = train_body(Spec(), "base") if kind == "train" else infer_body(Spec(), "base")
        mine, ref = body.as_dict(), read(ref_path)
        diff = sorted(k for k in set(ref) | set(mine)
                      if k not in PATH_KEYS | build_only and ref.get(k) != mine.get(k))
        status = "ok" if not diff else "DIFFERS: " + ", ".join(f"{k} {ref.get(k)} -> {mine.get(k)}" for k in diff)
        ok &= not diff
        print(f"  self-check base ({kind}) vs r6/{ref_path.name}: {status}")
    return ok


def plan(jobs):
    """run_node.sh's dealing: job s on slot s, then each freed slot takes the next job."""
    gpus = slot_gpus()
    alone = [g in ALONE_GPUS for g in gpus]
    heap = [(0.0, s) for s in range(len(gpus))]
    out = []
    for k, j in enumerate(jobs):
        t, s = (0.0, k) if k < len(gpus) else heapq.heappop(heap)
        h = est_hours(j.spec, alone[s])
        out.append((j, gpus[s], alone[s], t, t + h))
        if k < len(gpus):
            heap.remove((0.0, k))
            heapq.heapify(heap)
        heapq.heappush(heap, (t + h, s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the queue and estimated hours; write nothing")
    ap.add_argument("--force", action="store_true", help=f"regenerate even though output/b300_{ROUND} exists")
    args = ap.parse_args()

    jobs = build_jobs()
    check_run_135()
    rows = plan(jobs)
    makespan = max(e for *_, e in rows)
    table = ["| # | job | change from the base (r6 hybrid_e240) | GPU | checkpointing | est. h | est. start-end (days) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, (j, gpu, alone, t, e) in enumerate(rows, 1):
        table.append(f"| {k} | {j.name} | {j.change} | {gpu}{' (alone)' if alone else ''} | "
                     f"{'on' if use_checkpointing(j.spec) else 'off'} | {e - t:.0f} | {t / 24:.1f}-{e / 24:.1f} |")
    summary = (f"{len(jobs)} trainings, seed {SEED}, one run each, on {len(slot_gpus())} slots "
               f"(GPUs {', '.join(ALONE_GPUS)} one job each, {', '.join(SHARED_GPUS)} two each); "
               f"estimated makespan {makespan / 24:.1f} days (34 h per 120 epochs alone with checkpointing, "
               f"x{SHARE} two per card, 1/{CKPT_SPEEDUP} without checkpointing, per-epoch batch factors "
               f"{BS_FACTOR}; round 6 ran at about 0.6-0.7x of such estimates).")
    if args.plan:
        print(summary)
        print("\n".join(table))
        return 0

    out = HERE
    if (REPO / "output" / f"b300_{ROUND}").exists() and not args.force:
        raise SystemExit(f"output/b300_{ROUND} exists: the sweep may be running; --force to regenerate anyway")
    if not self_check():
        raise SystemExit("self-check failed: the base drifted from round 6's hybrid_e240 (above)")
    for old in list(out.glob("config_*.txt")) + [out / "arms.json"]:
        if old.is_file():
            old.unlink()
    lines = []
    for j in jobs:
        write(out / f"config_train_{j.name}.txt", header(j, "train") + train_body(j.spec, j.name).text())
        write(out / f"config_infer_{j.name}.txt", header(j, "infer") + infer_body(j.spec, j.name).text())
        lines.append(f"{NODE} {j.name} mgnv {REL}/config_train_{j.name}.txt {REL}/config_infer_{j.name}.txt")
    write(out / "jobs.txt",
          f"# B300 hybrid sweep {ROUND}: node job method train_config|- infer_config. Generated by make_sweep.py.\n"
          f"# Run order = line order; the first {len(ALONE_GPUS)} run alone on GPUs {', '.join(ALONE_GPUS)}"
          " (run_135.sh); the slots take the next line as they free up.\n"
          + "\n".join(lines) + "\n")
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if JOBS_MARK in text:
        text = text[:text.index(JOBS_MARK)]
    write(readme, text + JOBS_MARK + "\n\n" + summary + "\n\n" + "\n".join(table) + "\n")
    print(summary)
    print(f"wrote {out} ({len(jobs)} jobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
