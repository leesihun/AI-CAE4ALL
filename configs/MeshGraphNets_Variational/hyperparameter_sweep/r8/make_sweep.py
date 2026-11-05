#!/usr/bin/env python3
"""Generate B300 round 8: HI-MGN-hybrid at batch 16, one run per setting.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/make_sweep.py            # writes the configs here
    python .../r8/make_sweep.py --plan          # queue, slots and estimated hours; writes nothing
    python .../r8/make_sweep.py --pack 3        # three jobs per card (checkpointing where needed)
    python .../r8/make_sweep.py --extra encdec  # append EXTRA_ARMS' set (or arms: --extra ld64,ch8)

Standard library only, any Python 3.8+. Round 7 showed that the batch, not the
step count, sets the cost: 480 epochs at batch 16 alone on a card (w2 68) beat
the base (r6 hybrid_e240, batch 4: 69) in less time than any batch-4 run, and
batch 8 at 240 epochs scored 74. So the base here is 240 epochs at batch 16,
trained in this round (job 2), and every other job changes one thing from it.
Seed 42, one run each. Edit this file, never a generated config.

--pack sets the jobs per card, and with it run_135.sh's GPUS and
THREADS_PER_JOB (rewritten here) and each job's use_checkpointing (on only where
the estimated peak of PACK jobs would not fit the card). use_checkpointing and
the slot layout do not enter the resume fingerprint, so the round can be
stopped, regenerated with another --pack and restarted: every job resumes.

--extra appends trainings after the others while the round runs, and they stay
in jobs.txt on every later regeneration. A config whose text does not change is
not rewritten (run_node.sh reruns a job whose config is newer than its result),
and without --force nothing an existing job reads may change. The running queue
read jobs.txt when it started; a second run of the round takes the extras as
its cards free up, and brief.py reports all of them in one block:
    LANE=b WAIT_FREE=1 JOBS="hybrid_ld64 hybrid_ch8" bash .../r8/run_135.sh
"""
import argparse
import dataclasses
import heapq
import re
import sys
from decimal import Decimal
from pathlib import Path

ROUND = "r8"
NODE = "135"
PLACEMENT = "hybrid"

# ================================================================== the sweep
# Round 6's hybrid_ref (mp_per_level 4, 6, 8, 6, 4, coarsening 4096, 256) at
# 240 epochs and batch 16. beta_aux 10, voronoi_clusters and mp_per_level are fixed.
REF = dict(lr="1e-4", ep=240, bs=16, ar=1000, w=128, reg="mmd", mmd=1.0, kl=None, ld=32, ch=4,
           ph=128, pm=3, fb=4, pms=0.05, fms=30)
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)
SEED = 42

# arm -> (changes from REF, what it is). Run order = this order: with one job per
# card the first 7 are the first wave. The longest job (e720) goes first so it
# does not set the makespan; then the questions that decide the next round (the
# base, the learning rate, the middle of the MMD and KL ranges).
# Known: r6 batch 4: e120 88/88/107 (3 seeds), e240 69; r7: e240 batch 8 74, e480 batch 16 68
# (= this base at 480 epochs). r6 at batch 4 / 120 epochs, all within its seed noise:
# prior_fm_blocks 8 82, prior_hidden_dim 256 90; prior_temperature 0.5 / 2 and
# latent_inflation > 1 worse.
ARMS = {
    "e720": (dict(ep=720), "720 epochs (with 240 and r7's 480: does it keep improving?)"),
    "base": (dict(), "the base: 240 epochs, batch 16, lr 1e-4, lambda_mmd 1"),
    "lr2e-4": (dict(lr="2e-4"), "learning rate 2e-4 (sqrt scaling from batch 4)"),
    "mmd10": (dict(mmd=10.0), "lambda_mmd 10"),
    "mmd100": (dict(mmd=100.0), "lambda_mmd 100"),
    "kl1": (dict(reg="kl", kl=1.0), "latent_reg kl, kl_weight 1"),
    "kl10": (dict(reg="kl", kl=10.0), "latent_reg kl, kl_weight 10"),
    "e360": (dict(ep=360), "360 epochs"),
    "fmbig": (dict(fb=8, ph=256), "larger flow-matching prior: prior_fm_blocks 8, prior_hidden_dim 256"),
    "bs32_lr2e-4": (dict(bs=32, lr="2e-4"), "batch 32 at learning rate 2e-4 (checkpointing always: ~330 GB without)"),
    "mmd0.1": (dict(mmd=0.1), "lambda_mmd 0.1"),
    "mmd1000": (dict(mmd=1000.0), "lambda_mmd 1000"),
    "mmd10000": (dict(mmd=10000.0), "lambda_mmd 10000"),
    "kl0.1": (dict(reg="kl", kl=0.1), "latent_reg kl, kl_weight 0.1"),
    "kl100": (dict(reg="kl", kl=100.0), "latent_reg kl, kl_weight 100"),
}
# Sampling only: arm -> (changes from REF, what it is). Each samples the base's
# checkpoint again (jobs.txt train column "-"; run_node.sh waits for the base) with
# only an inference key changed. prior_fm_steps: the ODE steps of the prior's
# sampler (heun). If 10, 30 and 100 score alike, the solver is not the error.
SAMPLE_ARMS = {
    "steps10": (dict(fms=10), "sampling only: prior_fm_steps 10 on the base's checkpoint"),
    "steps100": (dict(fms=100), "sampling only: prior_fm_steps 100 on the base's checkpoint"),
}
# Trainings appended while the round runs (--extra SET or --extra arm,arm), chosen by
# the base's posterior vs. prior diagnosis (diagnose.sh): posterior sample already far
# above the floor -> the encoder/decoder loses the width (encdec); posterior sample near
# the floor and prior far above it -> the prior does (prior). r6 tried vae_latent_dim
# 16/64 and latent_ch 2/8 at batch 4 / 120 epochs: all within its seed noise.
EXTRA_ARMS = {
    "encdec": {
        "ld64": (dict(ld=64), "vae_latent_dim 64"),
        "ld16": (dict(ld=16), "vae_latent_dim 16"),
        "ch8": (dict(ch=8), "latent_ch 8 (channels of the local latent)"),
        "ld64_ch8": (dict(ld=64, ch=8), "vae_latent_dim 64 and latent_ch 8"),
        "pms0.02": (dict(pms=0.02), "posterior_min_std 0.02"),
        "ar3000": (dict(ar=3000), "alpha_recon 3000"),
    },
    "prior": {
        "pm6": (dict(pm=6), "prior_mp_layers 6"),
        "ph256": (dict(ph=256), "prior_hidden_dim 256 (fmbig without the extra blocks)"),
        "fb8": (dict(fb=8), "prior_fm_blocks 8 (fmbig without the wider prior)"),
        "fmbig_pm6": (dict(fb=8, ph=256, pm=6), "prior_fm_blocks 8, prior_hidden_dim 256, prior_mp_layers 6"),
        "pms0.1": (dict(pms=0.1), "posterior_min_std 0.1 (a smoother target for the prior)"),
    },
}
EXTRA_ALL = {arm: v for arms in EXTRA_ARMS.values() for arm, v in arms.items()}
CARDS = ("0", "2", "3", "4", "5", "6", "7")   # GPU1 left out, as in rounds 6 and 7
PACK = 1                                      # jobs per card; --pack overrides
CPUS = 120                                    # of 128: torch threads + 4 DataLoader workers per job

DISPLAY = "HI-MGN-hybrid (latent_type hybrid)"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REL = f"configs/MeshGraphNets_Variational/hyperparameter_sweep/{ROUND}"
SWEEP_REL = "configs/MeshGraphNets_Variational/hyperparameter_sweep"

# Peak memory without gradient checkpointing (aarl 3090, 2026-10-27, bf16 LayerNorm):
# 17.1 GB per 100k nodes + 1 GB. Worst batch in nodes = max of 200k random ex3 batches
# (recomputed 2026-11-05: bs16 1.04M, bs32 1.90M). With checkpointing the peak is a
# GUESS (CKPT_MEM of it, from a 3090 bs4 run) until r6/r7's logged peaks replace it.
WORST_NODES = {4: 329e3, 8: 563e3, 16: 1045e3, 32: 1900e3}
GB_PER_100K = 17.1
CKPT_MEM = 0.3
CARD_GB = 260          # of 288 on a B300, the rest for the CUDA contexts and fragmentation

# Estimated hours, ONLY for the queue plan. EP_H[bs] = hours per epoch, alone on a
# card without checkpointing: batch 16 measured in this round (277 s, 135, 7 jobs, one
# per card, 2026-11-05); batch 32 guessed. Checkpointing x1.52 (aarl 3090). PACK jobs on
# one card: each runs SHARE[PACK] times slower (guess; r7 measures 1 and 2 per card).
EP_H = {16: 0.077, 32: 0.11}
CKPT_SLOWDOWN = 1.52
SHARE = {1: 1.0, 2: 1.5, 3: 2.1, 4: 2.7}
INFER_H = 2.0


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
    fms: int = REF["fms"]
    seed: int = SEED


def peak_gb(spec, ckpt):
    per_100k = GB_PER_100K * (spec.w / 128 + 4.7 / 21 * (spec.ph / 128 - 1) + 4.7 / 21 * (spec.pm / 3 - 1))
    gb = per_100k * WORST_NODES[spec.bs] / 1e5 + 1.0
    return gb * CKPT_MEM + 1.0 if ckpt else gb


def use_checkpointing(spec, pack):
    return peak_gb(spec, False) > CARD_GB / pack


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


def train_body(spec, job, pack):
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
    b.set("prior_fm_steps", spec.fms)
    b.set("use_checkpointing", use_checkpointing(spec, pack))
    b.set("resume_training", "True")
    return b


def infer_body(spec, job, pack, source=None):
    """The inference config of `job`; `source` = the job whose checkpoint it samples (default: its own)."""
    b = train_body(spec, source or job, pack)
    b.set("mode", "inference").set("log_file_dir", f"{run_dir(job)}/infer.log").set("batch_size", 8)
    b.set("inference_output_dir", f"{run_dir(job)}/infer")
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
    source: object = None   # sampling only: the job whose checkpoint it samples
    extra: bool = False     # appended with --extra: a second run (LANE=b WAIT_FREE=1) takes it


def est_hours(job, pack):
    if job.source:
        return INFER_H
    ckpt = CKPT_SLOWDOWN if use_checkpointing(job.spec, pack) else 1.0
    return EP_H[job.spec.bs] * job.spec.ep * ckpt * SHARE[pack] + INFER_H


def extra_arms(requested):
    """Extra arms already in jobs.txt (kept, in their order), then the requested ones
    (set names or arm names) not there yet."""
    table = HERE / "jobs.txt"
    have = []
    if table.is_file():
        for line in table.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and not parts[0].startswith("#"):
                arm = parts[1][len(PLACEMENT) + 1:]
                if arm in EXTRA_ALL and arm not in have:
                    have.append(arm)
    for item in requested:
        for name in filter(None, item.split(",")):
            arms = list(EXTRA_ARMS[name]) if name in EXTRA_ARMS else [name]
            for arm in arms:
                if arm not in EXTRA_ALL:
                    raise SystemExit(f"--extra {name}: not a set ({', '.join(EXTRA_ARMS)}) "
                                     f"or an arm ({', '.join(EXTRA_ALL)})")
                if arm not in have:
                    have.append(arm)
    return have


def build_jobs(extras=()):
    """The trainings in ARMS order, with the sampling-only jobs right after the first
    wave (one per card): they need only the base, which that wave trains. Then the
    extra arms."""
    base = f"{PLACEMENT}_base"
    train = [Job(f"{PLACEMENT}_{arm}", dataclasses.replace(Spec(), **changes), what)
             for arm, (changes, what) in ARMS.items()]
    sample = [Job(f"{base}_{arm}", dataclasses.replace(Spec(), **changes), what, source=base)
              for arm, (changes, what) in SAMPLE_ARMS.items()]
    extra = [Job(f"{PLACEMENT}_{arm}", dataclasses.replace(Spec(), **EXTRA_ALL[arm][0]), EXTRA_ALL[arm][1],
                 extra=True) for arm in extras]
    wave = len(CARDS)
    return train[:wave] + sample + train[wave:] + extra


def slot_gpus(pack):
    """run_135.sh's GPUS: every card once, then again, PACK times (job k -> slot k)."""
    return list(CARDS) * pack


def threads_per_job(pack):
    return max(2, CPUS // (len(CARDS) * pack) - 4)


def write_run_135(pack):
    text = f"""\
#!/usr/bin/env bash
# B300 round 8 (large-batch HI-MGN-hybrid sweep), node 135: see run_node.sh (order of work, overrides)
# and README.md (the jobs as one queue). WRITTEN BY make_sweep.py --pack {pack}: {pack} job(s) per card on
# GPUs {', '.join(CARDS)} (GPU1 left out), {len(CARDS) * pack} slots; each job gets {threads_per_job(pack)} torch threads + 4
# DataLoader workers. To change the packing, regenerate with another --pack (running jobs resume).
THREADS_PER_JOB="${{THREADS_PER_JOB:-{threads_per_job(pack)}}}" \\
GPUS="${{GPUS:-{' '.join(slot_gpus(pack))}}}" \\
    exec bash "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/run_node.sh" 135 "$@"
"""
    write(HERE / "run_135.sh", text)


# ==================================================================== headers
DATA = """\
% Dataset: shell buckling, 150 geometries x 64 imperfection draws (ex3_shell_buckling.h5); held-out file
% ex3_shell_buckling_infer.h5 = 8 unseen geometries x 256 draws. split_group_attr keeps a geometry's draws together.
% Fields: input [0:3] x,y,z | output [3:6] ux,uy,uz | conditioner [6:8] thickness, delta | [8] node type.
% Static (T=1): the model predicts the field in one step; the spread comes from the unseen imperfection.
"""


def header(job, kind, pack):
    lines = [f"% ex3 B300 large-batch hybrid sweep ({ROUND}), job {job.name}: {DISPLAY}",
             f"% Role: {job.change}",
             "% Base: hybrid_base of this round (r6 hybrid_e240 at batch 16).",
             f"% Generated by {REL}/make_sweep.py --pack {pack}; edit that file, not this one.",
             f"% Run paths: output/b300_{ROUND}/<job>/; one shared hierarchy cache; resume_training True."
             if kind == "train" else
             "% 4 draws per held-out scene = 1024 per geometry against its 256 FE results (spread_metrics.json)."]
    if job.source:
        lines.insert(4, f"% Trains nothing: samples {job.source}'s checkpoint (waits for it in the queue).")
    if job.extra:
        lines.insert(4, "% Added while the round ran (--extra); a second run takes it: LANE=b WAIT_FREE=1 JOBS=...")
    return "\n".join(lines) + "\n%\n" + DATA


def write(path, text):
    """LF, no BOM; an unchanged file keeps its mtime (run_node.sh compares them)."""
    data = text.encode("utf-8")
    if not (path.is_file() and path.read_bytes() == data):
        path.write_bytes(data)


def self_check():
    """The base equals round 6's hybrid_e240 except the batch, build-only keys and paths."""
    def read(p):
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("%", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = " ".join(parts[1].split())
        return out

    # amp_layernorm: retired 2026-10-30; r6 set it True, which is what AMP now always does.
    build_only = {"hierarchy_cache_build_workers", "use_checkpointing", "use_compile", "amp_layernorm"}
    intended = {"train": {"batch_size"}, "infer": {"batch_size"}}
    ok = True
    for kind in ("train", "infer"):
        ref_path = REPO / SWEEP_REL / "r6" / f"config_{kind}_hybrid_e240.txt"
        if not ref_path.is_file():
            print(f"  self-check: r6/{ref_path.name} not found, skipped")
            continue
        body = train_body(Spec(), "base", 1) if kind == "train" else infer_body(Spec(), "base", 1)
        mine, ref = body.as_dict(), read(ref_path)
        if kind == "infer":   # inference batch_size is the infer config's own (8), not the training batch
            intended[kind] = set()
        diff = sorted(k for k in set(ref) | set(mine)
                      if k not in PATH_KEYS | build_only | intended[kind] and ref.get(k) != mine.get(k))
        status = "ok" if not diff else "DIFFERS: " + ", ".join(f"{k} {ref.get(k)} -> {mine.get(k)}" for k in diff)
        ok &= not diff
        print(f"  self-check base ({kind}) vs r6/{ref_path.name}: {status}")
    return ok


def plan(jobs, pack):
    """run_node.sh's dealing: job s on slot s, then each freed slot takes the next job.

    A sampling-only job holds its slot until its source has trained. The extra jobs
    (a second run, WAIT_FREE, one slot per card) start on a card once the first run
    has started its last job and that card's slots have all finished."""
    gpus = slot_gpus(pack)
    heap = []
    out = []
    end = {}
    first = [j for j in jobs if not j.extra]
    for k, j in enumerate(first):
        t, s = (0.0, k) if k < len(gpus) else heapq.heappop(heap)
        if j.source:
            t = max(t, end[j.source] - INFER_H)
        h = est_hours(j, pack)
        out.append((j, gpus[s], t, t + h))
        end[j.name] = t + h
        heapq.heappush(heap, (t + h, s))
    last_start = max((t for _j, _g, t, _e in out), default=0.0)
    free = {c: max([last_start] + [e for _j, g, _t, e in out if g == c]) for c in CARDS}
    cards = [(t, c) for c, t in free.items()]
    heapq.heapify(cards)
    for j in jobs:
        if j.extra:
            t, c = heapq.heappop(cards)
            h = est_hours(j, 1)
            out.append((j, c, t, t + h))
            heapq.heappush(cards, (t + h, c))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the queue and estimated hours; write nothing")
    ap.add_argument("--pack", type=int, default=PACK, choices=sorted(SHARE), help="jobs per card")
    ap.add_argument("--force", action="store_true",
                    help=f"regenerate even though output/b300_{ROUND} exists and a job's config changes "
                         "(e.g. to change --pack)")
    ap.add_argument("--extra", action="append", default=[], metavar="SET|ARM,ARM",
                    help=f"append extra trainings: a set ({', '.join(EXTRA_ARMS)}) or arms "
                         f"({', '.join(EXTRA_ALL)}); arms added before stay")
    args = ap.parse_args()
    pack = args.pack

    jobs = build_jobs(extra_arms(args.extra))
    peaks = sorted((peak_gb(j.spec, use_checkpointing(j.spec, pack)) for j in jobs if not j.source),
                   reverse=True)
    if sum(peaks[:pack]) > CARD_GB:   # the worst card: the PACK largest jobs together
        raise SystemExit(f"the {pack} largest jobs need ~{sum(peaks[:pack]):.0f} GB together, over "
                         f"{CARD_GB}; {pack} per card does not fit (CKPT_MEM {CKPT_MEM} is a guess)")
    rows = plan(jobs, pack)
    makespan = max(e for *_, e in rows)
    table = ["| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, (j, gpu, t, e) in enumerate(rows, 1):
        ck = use_checkpointing(j.spec, pack)
        peak = "-" if j.source else f"{peak_gb(j.spec, ck):.0f}"
        table.append(f"| {k} | {j.name} | {j.change}{' (extra, second run)' if j.extra else ''} | {gpu} | "
                     f"{'-' if j.source else 'on' if ck else 'off'} | "
                     f"{peak} | {e - t:.0f} | {t / 24:.1f}-{e / 24:.1f} |")
    n_train = sum(1 for j in jobs if not j.source)
    extras = [j.name for j in jobs if j.extra]
    summary = (f"{n_train} trainings + {len(jobs) - n_train} sampling-only jobs, seed {SEED}, one run each; "
               f"--pack {pack}: {pack} per card on GPUs "
               f"{', '.join(CARDS)} = {len(slot_gpus(pack))} slots, {threads_per_job(pack)} torch threads per job. "
               f"Estimated makespan {makespan / 24:.1f} days (GUESSED speeds: {EP_H[16]} h per batch-16 epoch alone, "
               f"x{CKPT_SLOWDOWN} with checkpointing, x{SHARE[pack]} per job at {pack} per card; peaks with "
               f"checkpointing guessed at {CKPT_MEM} of the full peak).")
    if extras:
        summary += (f" Extra trainings ({len(extras)}): a second run takes them as the cards free up, "
                    f"LANE=b WAIT_FREE=1 JOBS=\"{' '.join(extras)}\" bash {REL}/run_135.sh; "
                    "their start-end assumes it was started with the first.")
    if args.plan:
        print(summary)
        print("\n".join(table))
        return 0

    out = HERE
    if not self_check():
        raise SystemExit("self-check failed: the base drifted from round 6's hybrid_e240 (above)")
    configs, lines = {}, []
    for j in jobs:
        if not j.source:
            configs[out / f"config_train_{j.name}.txt"] = \
                header(j, "train", pack) + train_body(j.spec, j.name, pack).text()
        configs[out / f"config_infer_{j.name}.txt"] = \
            header(j, "infer", pack) + infer_body(j.spec, j.name, pack, j.source).text()
        tcfg = "-" if j.source else f"{REL}/config_train_{j.name}.txt"
        lines.append(f"{NODE} {j.name} mgnv {tcfg} {REL}/config_infer_{j.name}.txt")
    stale = [p for p in out.glob("config_*.txt") if p not in configs]
    changed = [p for p, text in configs.items() if p.is_file() and p.read_bytes() != text.encode("utf-8")]
    if (stale or changed) and (REPO / "output" / f"b300_{ROUND}").exists() and not args.force:
        raise SystemExit(f"output/b300_{ROUND} exists (the sweep may be running) and this would "
                         + ", ".join(f"{'remove' if p in stale else 'change'} {p.name}" for p in stale + changed)
                         + "; --force to regenerate anyway")
    for p in stale:
        p.unlink()
    for p, text in configs.items():
        write(p, text)
    write(out / "jobs.txt",
          f"# B300 large-batch hybrid sweep {ROUND}: node job method train_config|- infer_config. "
          f"Generated by make_sweep.py --pack {pack}.\n"
          "# Run order = line order; job k starts on slot k, then each freed slot takes the next line.\n"
          + (f"# The last {len(extras)} (--extra) were added while the round ran; a second run takes them: "
             f"LANE=b WAIT_FREE=1 JOBS=\"...\".\n" if extras else "")
          + "\n".join(lines) + "\n")
    write_run_135(pack)
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if JOBS_MARK in text:
        text = text[:text.index(JOBS_MARK)]
    write(readme, text + JOBS_MARK + "\n\n" + summary + "\n\n" + "\n".join(table) + "\n")
    print(summary)
    print(f"wrote {out} ({len(jobs)} jobs, run_135.sh for --pack {pack}; "
          f"{len(changed)} config(s) changed, {len(stale)} removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
