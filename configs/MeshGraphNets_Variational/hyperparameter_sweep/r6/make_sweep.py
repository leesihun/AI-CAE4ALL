#!/usr/bin/env python3
"""Generate the B300 placement sweep: one-at-a-time around one reference, one queue.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r6/make_sweep.py          # writes the configs here
    python .../r6/make_sweep.py --plan [--slots 14]   # queue and estimated hours; writes nothing

Standard library only, any Python 3.8+. Every training job is the reference
of its placement (REF below) with exactly one knob moved (KNOBS below); the
queue runs in the order of QUEUE on node 135's 7 cards, 2 jobs per card.
Edit this file, never a generated config.

Job names: <placement>_ref, <placement>_<token><value> (global_lr3e-4,
local_ch8, hybrid_bs16, ...), <placement>_ref_s43 (seed), <placement>_ref_infl /
<placement>_ref_t<T> (sampling only, on the reference checkpoint), ref_det.
"""
import argparse
import dataclasses
import heapq
import sys
import textwrap
from decimal import Decimal
from pathlib import Path

ROUND = "r6"
NODE = "135"
SLOTS = 14   # 7 cards x 2 jobs (run_135.sh lists the 7 cards twice)

# ================================================================== the sweep
# The reference of every placement: round 4's <placement>_mmd with mp_per_level
# 4, 6, 8, 6, 4 (the deterministic HI-MGN's) and the HI-MGN-V shell-buckling
# coarsening 4096, 256. beta_aux 10, voronoi_clusters and mp_per_level are fixed.
REF = dict(lr="1e-4", ep=120, bs=4, ar=1000, w=128, mmd=1.0, ld=32, ch=4, ph=128, pm=3, fb=4, pms=0.05)
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)

# token -> (Spec field, config key, values moved to, placements it applies to, what it is)
KNOBS = {
    "lr": ("lr", "learningr", ("3e-5", "3e-4", "1e-3"), "all", "learning rate"),
    "e": ("ep", "training_epochs", (60, 240), "all", "epochs (prior_freeze_epoch follows at 2/3)"),
    "bs": ("bs", "batch_size", (2, 8, 16), "all", "batch size (learning rate not rescaled)"),
    "ar": ("ar", "alpha_recon", (100, 10000), "all", "reconstruction weight"),
    "w": ("w", "latent_dim", (64, 256), "all", "backbone width (latent_dim)"),
    "mmd": ("mmd", "lambda_mmd", (0.1, 10, 100), "all", "MMD weight"),
    "ld": ("ld", "vae_latent_dim", {"global": (8, 16, 64, 128), "hybrid": (16, 64)}, ("global", "hybrid"),
           "global latent vector size"),
    "ch": ("ch", "latent_ch", {"local": (1, 2, 8, 16), "hybrid": (2, 8)}, ("local", "hybrid"),
           "latent channels per coarse node"),
    "ph": ("ph", "prior_hidden_dim", (64, 256), "all", "prior width"),
    "pm": ("pm", "prior_mp_layers", (2, 6), "all", "prior message-passing layers"),
    "fb": ("fb", "prior_fm_blocks", (2, 8), ("local", "hybrid"),
           "coarse FM prior blocks"),   # inert for global: its FM prior uses the MLP velocity
    "pms": ("pms", "posterior_min_std", (0.01, 0.1), "all", "posterior std floor"),
}
SEEDS = (43, 44)                       # the reference again: the seed noise every difference is read against
INFLATION_LIST = "1, 1.25, 1.5, 2"     # sampling only, on the reference checkpoint
TEMPERATURES = (0.5, 2)                # prior_temperature: FM start noise std x sqrt(T)
# Run order. Long jobs first so they do not trail; sampling-only jobs once the references are likely done
# (each waits for its reference checkpoint if not).
QUEUE = ("e240", "w256", "ref", "seed", "lr", "e60", "sampling", "bs", "ar", "w64", "mmd", "ld", "ch",
         "ph", "pm", "fb", "pms", "det")

PLACEMENTS = ("global", "local", "hybrid")
DISPLAY = {"global": "HI-MGN-V (latent_type global)", "local": "HI-MGNflow (latent_type local)",
           "hybrid": "HI-MGN-hybrid (latent_type hybrid)"}

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REL = "configs/MeshGraphNets_Variational/hyperparameter_sweep/r6"
SWEEP_REL = "configs/MeshGraphNets_Variational/hyperparameter_sweep"

# Estimated card-hours, ONLY for the queue plan (guesses except the anchor):
# 120 epochs at mp 4,6,8,6,4 took 34 h with one job per card (round 5's hybrid_mp28).
H120 = 34.0
SHARE = 1.4            # two jobs on one card (MPS): guessed slowdown of each
BS_FACTOR = {2: 1.3, 4: 1.0, 8: 0.85, 16: 0.75}   # guessed
INFER_H, INFL_H = 2.0, 4.0

# Gradient checkpointing costs 1.52x per step (aarl 3090, bs 1, 2026-10-27), so it is off
# wherever the job fits without it. Peak memory without it, measured there at bs 1: 21 GB per
# 100k nodes at the reference (13 at latent_dim 64, +4.7 at prior_hidden_dim 256; the same
# for every placement), +1 GB fixed. The prior_mp_layers term is a guess (same as width).
# Worst batch in nodes = max of 200k random ex3 batches (mean sample 48k, largest 87k).
CKPT_SPEEDUP = 1.52
WORST_NODES = {2: 175e3, 4: 329e3, 8: 563e3, 16: 1045e3}
MEM_BUDGET_GB = 120    # of ~144 GB per job (288 GB B300, two per card)
# amp_layernorm (LayerNorm on the bf16 activations): 17.1 instead of 21 GB per 100k nodes
# at the reference, same harness. Applied to every term (measured at the reference only).
AMP_LN_MEM = 17.1 / 21


def peak_gb_without_checkpointing(spec):
    per_100k = (5.0 + 16.0 * spec.w / 128 + 4.7 * (spec.ph / 128 - 1)
                + 4.7 * (spec.pm / 3 - 1))
    return per_100k * AMP_LN_MEM * WORST_NODES[spec.bs] / 1e5 + 1.0


def use_checkpointing(spec):
    return peak_gb_without_checkpointing(spec) > MEM_BUDGET_GB


@dataclasses.dataclass(frozen=True)
class Spec:
    placement: str
    lr: str = REF["lr"]
    ep: int = REF["ep"]
    bs: int = REF["bs"]
    ar: int = REF["ar"]
    w: int = REF["w"]
    mmd: float = REF["mmd"]
    ld: int = REF["ld"]
    ch: int = REF["ch"]
    ph: int = REF["ph"]
    pm: int = REF["pm"]
    fb: int = REF["fb"]
    pms: float = REF["pms"]
    seed: int = 42


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
amp_layernorm                    True
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
latent_type                      global
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
    b.set("latent_dim", spec.w).set("alpha_recon", spec.ar).set("lambda_mmd", f"{spec.mmd:g}")
    b.set("posterior_min_std", f"{spec.pms:g}").set("training_seed", spec.seed)
    b.set("vae_latent_dim", spec.ld).set("latent_type", spec.placement)
    if spec.placement != "global":
        b.set("latent_ch", spec.ch, after="latent_type").set("vae_local_mp_layers", 2, after="latent_ch")
    b.set("prior_mp_layers", spec.pm).set("prior_hidden_dim", spec.ph)
    if spec.fb != REF["fb"]:
        b.set("prior_fm_blocks", spec.fb, after="prior_hidden_dim")
    b.set("use_checkpointing", use_checkpointing(spec))
    b.set("resume_training", "True")
    return b


def infer_body(spec, job, modelpath=None, infl=False, temp=None):
    b = train_body(spec, job)
    b.set("mode", "inference").set("log_file_dir", f"{run_dir(job)}/infer.log").set("batch_size", 8)
    b.drop("hierarchy_cache_dir", "hierarchy_cache_build_workers", "resume_training")
    if modelpath:
        b.set("modelpath", modelpath)
    b.set("inference_samples", SAMPLES).set("inference_batch_size", 1 if infl else INFER_BATCH)
    for k, v in EVAL:
        b.set(k, v)
    if infl:
        b.set("latent_inflation", INFLATION_LIST)
    if temp is not None:
        b.set("prior_temperature", f"{temp:g}")
    return b


def det_train_body():
    keys = [k for k, _ in BASE]
    b, run = Body(BASE[: keys.index("training_seed") + 1]), run_dir("ref_det")
    b.set("modelpath", f"{run}/model.pth").set("log_file_dir", f"{run}/train.log")
    b.set("inference_output_dir", f"{run}/infer")
    b.set("hierarchy_cache_dir", f"../../output/b300_{ROUND}/mscache/mgnv_v{VORONOI[0]}_{VORONOI[1]}")
    for k, v in (("best_by", "recon"), ("use_vae", "False"), ("recon_loss", "mse"),
                 ("use_conditional_prior", "False"), ("use_checkpointing", "False"),
                 ("resume_training", "True")):
        b.set(k, v)
    return b


def det_infer_body():
    b = det_train_body()
    b.set("mode", "inference").set("log_file_dir", f"{run_dir('ref_det')}/infer.log").set("batch_size", 8)
    b.drop("hierarchy_cache_dir", "hierarchy_cache_build_workers", "resume_training")
    for k, v in EVAL:
        b.set(k, v)
    return b


# ======================================================================= jobs
@dataclasses.dataclass
class Job:
    name: str
    group: str            # QUEUE entry
    placement: str        # "" for ref_det
    spec: object          # Spec, None for ref_det
    change: str           # "learning rate 3e-4 (reference 1e-4)"
    infer_only: bool = False
    infl: bool = False
    temp: object = None
    hours: float = 0.0


def fmt(v):
    return v if isinstance(v, str) else f"{v:g}"


def est_hours(spec):
    if spec is None:
        return H120 * 0.6 * SHARE / CKPT_SPEEDUP + INFER_H   # no encoder, no prior (guess)
    ckpt = 1.0 if use_checkpointing(spec) else 1.0 / CKPT_SPEEDUP
    return H120 * spec.ep / 120 * (spec.w / 128) ** 1.5 * BS_FACTOR[spec.bs] * SHARE * ckpt + INFER_H


def build_jobs():
    groups = {g: [] for g in QUEUE}
    for p in PLACEMENTS:
        ref = Spec(p)
        groups["ref"].append(Job(f"{p}_ref", "ref", p, ref, "none (the reference)"))
        for s in SEEDS:
            groups["seed"].append(Job(f"{p}_ref_s{s}", "seed", p, dataclasses.replace(ref, seed=s),
                                      f"training_seed {s} (reference 42)"))
        for tok, (field, key, values, applies, what) in KNOBS.items():
            if applies != "all" and p not in applies:
                continue
            vals = values[p] if isinstance(values, dict) else values
            for v in vals:
                spec = dataclasses.replace(ref, **{field: v})
                name = f"{p}_{tok}{fmt(v)}"
                group = tok
                if tok in ("e", "w"):
                    group = f"{tok}{v}"
                if group not in groups:
                    raise SystemExit(f"{name}: group {group} is not in QUEUE")
                groups[group].append(Job(name, group, p, spec, f"{what}: {key} {fmt(v)} (reference {fmt(REF[field])})"))
        groups["sampling"].append(Job(f"{p}_ref_infl", "sampling", p, ref,
                                      f"sampling only: latent_inflation {INFLATION_LIST} on {p}_ref",
                                      infer_only=True, infl=True))
        for t in TEMPERATURES:
            groups["sampling"].append(Job(f"{p}_ref_t{fmt(t)}", "sampling", p, ref,
                                          f"sampling only: prior_temperature {fmt(t)} on {p}_ref",
                                          infer_only=True, temp=t))
    groups["det"].append(Job("ref_det", "det", "", None,
                             "deterministic HI-MGN (use_vae False) on the same backbone: the line a sampler must beat"))
    jobs = [j for g in QUEUE for j in groups[g]]
    for j in jobs:
        j.hours = (INFL_H if j.infl else INFER_H) if j.infer_only else est_hours(j.spec)
    names = [j.name for j in jobs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"duplicate job names: {sorted(dup)}")
    return jobs


# ==================================================================== headers
DATA = """\
% Dataset: shell buckling, 150 geometries x 64 imperfection draws (ex3_shell_buckling.h5); held-out file
% ex3_shell_buckling_infer.h5 = 8 unseen geometries x 256 draws. split_group_attr keeps a geometry's draws together.
% Fields: input [0:3] x,y,z | output [3:6] ux,uy,uz | conditioner [6:8] thickness, delta | [8] node type.
% Static (T=1): the model predicts the field in one step; the spread comes from the unseen imperfection.
"""


def header(job, kind):
    lines = [f"% ex3 B300 placement sweep ({ROUND}), job {job.name}: "
             + (DISPLAY[job.placement] if job.placement else "deterministic HI-MGN"),
             f"% Role: {job.change}",
             "% Reference: round 4's <placement>_mmd with mp_per_level 4, 6, 8, 6, 4 (message_passing_num 28);"
             " one knob moved.",
             f"% Generated by {REL}/make_sweep.py; edit that file, not this one.",
             "% Run paths: output/b300_r6/<job>/; one shared hierarchy cache; resume_training True."
             if kind == "train" else
             "% 4 draws per held-out scene = 1024 per geometry against its 256 FE results (spread_metrics.json)."]
    if job.infer_only:
        lines.append(f"% Trains nothing: samples {job.placement}_ref's checkpoint (waits for it in the queue).")
    return "\n".join(lines) + "\n%\n" + DATA


def write(path, text):
    path.write_bytes(text.encode("utf-8"))   # LF, no BOM


def self_check(jobs):
    """Each reference equals round 4's <placement>_mmd except mp_per_level; hybrid_ref equals round 5's hybrid_mp28."""
    def read(p):
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("%", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = " ".join(parts[1].split())
        return out

    ok = True
    # hierarchy_cache_build_workers only sets how many CPU processes build the
    # (identical) hierarchy cache; r6 raised it from 8 to 48 for the 128-core node.
    # use_checkpointing only trades memory for recompute; the gradients are the same.
    # use_compile (fused GnBlocks, eager fallback) and amp_layernorm are r6 speedups, set
    # on every job alike; amp_layernorm rounds LayerNorm in bf16, so r6 is not bit-equal to r4.
    build_only = {"hierarchy_cache_build_workers", "use_checkpointing", "use_compile", "amp_layernorm"}
    skip = PATH_KEYS | build_only | {"message_passing_num", "mp_per_level"}
    for p in PLACEMENTS:
        mine = train_body(Spec(p), f"{p}_ref").as_dict()
        for ref_path, ignore in ((REPO / SWEEP_REL / "r4" / f"config_train_{p}_mmd.txt", skip),
                                 (REPO / SWEEP_REL / "r5" / "config_train_hybrid_mp28.txt", PATH_KEYS | build_only)):
            if "r5" in str(ref_path) and p != "hybrid":
                continue
            if not ref_path.is_file():
                print(f"  self-check: {ref_path.name} not found, skipped")
                continue
            ref = read(ref_path)
            diff = sorted(k for k in set(ref) | set(mine)
                          if k not in ignore and ref.get(k) != mine.get(k))
            status = "ok" if not diff else "DIFFERS: " + ", ".join(f"{k} {ref.get(k)} -> {mine.get(k)}" for k in diff)
            ok &= not diff
            print(f"  self-check {p}_ref vs {ref_path.parent.name}/{ref_path.name}: {status}")
    return ok


def plan(jobs, slots):
    heap = [(0.0, s) for s in range(slots)]
    ends = []
    produced = {}
    for j in jobs:
        t, s = heapq.heappop(heap)
        if j.infer_only:
            t = max(t, produced.get(f"{j.placement}_ref", 0.0))
        end = t + j.hours
        if not j.infer_only:
            produced[j.name] = end
        ends.append((j, t, end))
        heapq.heappush(heap, (end, s))
    return ends


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the queue and estimated hours; write nothing")
    ap.add_argument("--slots", type=int, default=SLOTS)
    ap.add_argument("--force", action="store_true", help="regenerate even though output/b300_r6 exists")
    args = ap.parse_args()

    jobs = build_jobs()
    ends = plan(jobs, args.slots)
    makespan = max(e for _, _, e in ends)
    n_train = sum(not j.infer_only for j in jobs)
    table = ["| # | job | change from the reference | est. h | est. start-end (days) |",
             "| --- | --- | --- | --- | --- |"]
    for k, (j, t, e) in enumerate(ends, 1):
        table.append(f"| {k} | {j.name} | {j.change} | {j.hours:.0f} | {t / 24:.1f}-{e / 24:.1f} |")
    summary = (f"{len(jobs)} jobs ({n_train} trainings, {len(jobs) - n_train} sampling only) on {args.slots} slots; "
               f"estimated makespan {makespan / 24:.0f} days (guess: 34 h per 120 epochs on one card, "
               f"x{SHARE} for two per card, width^1.5, batch factors {BS_FACTOR}).")
    if args.plan:
        print(summary)
        print("\n".join(table))
        return 0

    out = HERE
    if (REPO / "output" / f"b300_{ROUND}").exists() and not args.force:
        raise SystemExit(f"output/b300_{ROUND} exists: the sweep may be running; --force to regenerate anyway")
    if not self_check(jobs):
        raise SystemExit("self-check failed: the references drifted from round 4 (above)")
    out.mkdir(exist_ok=True)
    for old in out.glob("config_*.txt"):
        old.unlink()
    rows = []
    for j in jobs:
        icfg = f"{REL}/config_infer_{j.name}.txt"
        if j.name == "ref_det":
            write(out / f"config_train_{j.name}.txt", header(j, "train") + det_train_body().text())
            write(out / f"config_infer_{j.name}.txt", header(j, "infer") + det_infer_body().text())
            tcfg = f"{REL}/config_train_{j.name}.txt"
        elif j.infer_only:
            body = infer_body(j.spec, j.name, modelpath=f"{run_dir(j.placement + '_ref')}/model.pth",
                              infl=j.infl, temp=j.temp)
            write(out / f"config_infer_{j.name}.txt", header(j, "infer") + body.text())
            tcfg = "-"
        else:
            write(out / f"config_train_{j.name}.txt", header(j, "train") + train_body(j.spec, j.name).text())
            write(out / f"config_infer_{j.name}.txt", header(j, "infer") + infer_body(j.spec, j.name).text())
            tcfg = f"{REL}/config_train_{j.name}.txt"
        rows.append(f"{NODE} {j.name} mgnv {tcfg} {icfg}")
    write(out / "jobs.txt",
          f"# B300 placement sweep {ROUND}: node job method train_config|- infer_config. Generated by make_sweep.py.\n"
          "# Run order = line order; 14 slots (7 cards x 2) take the next line as they free up.\n"
          + "\n".join(rows) + "\n")
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
