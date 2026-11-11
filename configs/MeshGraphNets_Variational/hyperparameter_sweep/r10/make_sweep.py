#!/usr/bin/env python3
"""Generate B300 round 10: the HI-MGN-hybrid coarse prior against over-fitting, three seeds per arm.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r10/make_sweep.py           # writes the configs here
    python .../r10/make_sweep.py --plan         # jobs, cards and estimated hours; writes nothing
    python .../r10/make_sweep.py --pack 1       # one job per card (two waves; checkpointing off)

Standard library only, any Python 3.8+. Round 9 said where the prior's error is:

  - the posterior sample sits at the floor (ho 27/29-35), so every arm's gap is the prior's;
  - the GNN prior under-fits (training geometries: prior 81-101 against a floor of 56) but
    carries over to held-out geometries about as well as it fits;
  - the transformer prior fits the training geometries (65-72) and over-fits: held-out
    105-108. Only gfix_tf_k8 (8 FM draws per posterior sample) kept the gap small
    (tr 71, ho 76) and was the best arm (w2 78/66);
  - z_g stays dead (g ~ 0) even with gfix; prior_param x and latent_ch 2 hurt.

So round 10 keeps the transformer and k8 and adds regularization against over-fitting the
120 training geometries:

  dropout   0.1 in every transformer block (attention weights and both residual branches).
            Fixed in model/conditional_prior.py (TF_DROPOUT), not a key, so EVERY
            transformer arm here has it; r9's gfix_tf_k8 (seeds 42, 43) is the same setting
            without it, read from output/b300_r9.
  cn        prior_cond_noise 0.1: training-only Gaussian noise, 0.1 x each row's RMS, on
            the trunk context the prior reads (pooled node features and the pooled vector)
  d2        prior_fm_blocks 2: half the transformer depth
  ema       ema_decay 0.999 (from 0.99): ~2 epochs of averaging instead of ~0.2. It is the
            whole model's EMA, so the joint phase's simulator is averaged longer too; the
            ho posterior column shows whether that moved it.
  gnn_k8    the base's GNN prior with prior_fm_draws 8: does k8 help the under-fitting GNN?

Three seeds per arm (training_seed 42, 43, 44). The base already has two: r8's hybrid_e360
(seed 42) and r9's hybrid_base_s43, so only seed 44 of it trains here. Edit this file,
never a generated config.

--pack sets the jobs per card, and with it run_135.sh's GPUS and
THREADS_PER_JOB (rewritten here) and each job's use_checkpointing (on only where
the estimated peak of PACK jobs would not fit the card). The default 2 runs all
16 jobs at once on the 8 cards. use_checkpointing and the slot layout do not
enter the resume fingerprint, so the round can be stopped, regenerated with
another --pack (--force) and restarted: every job resumes.
"""
import argparse
import dataclasses
import sys
from decimal import Decimal
from pathlib import Path

ROUND = "r10"
NODE = "135"
PLACEMENT = "hybrid"

# ================================================================== the sweep
# r8's hybrid_e360: r6 hybrid_ref (mp_per_level 4, 6, 8, 6, 4, coarsening 4096, 256)
# at batch 16 and 360 epochs. beta_aux 10, voronoi_clusters and mp_per_level are fixed.
REF = dict(lr="1e-4", ep=360, bs=16, ar=1000, w=128, mmd=1.0, ld=32, ch=4, ph=128, pm=3, pms=0.05, fms=30,
           inj="all", mmdg=None, arch="gnn", param="velocity", k=1, fmb=4, ema=0.99, cn=0.0)
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)
REF_JOB = ("r8", "hybrid_e360")   # the base at seed 42, trained in round 8
SEEDS = (42, 43, 44)
GFIX = dict(mmdg=0.0, inj="coarsest")
TFD = dict(GFIX, arch="transformer", k=8)   # r9's gfix_tf_k8; the dropout comes with the code

# arm -> (changes from REF, what it is, seeds). Order = run order: job k on slot k,
# the slots being cards 0-7 once, then again (--pack 2), so job k and job k+8 share a card.
# Five arms x three seeds put an arm's seeds on three different cards; the base (seed 44
# only) comes last.
ARMS = {
    "gnn_k8": (dict(k=8), "the base's GNN prior with 8 FM draws per posterior sample (prior_fm_draws 8)", SEEDS),
    "tfd": (TFD, "r9's gfix_tf_k8 (gfix + transformer prior + prior_fm_draws 8), now with the fixed "
                 "transformer dropout 0.1", SEEDS),
    "tfd_cn": (dict(TFD, cn=0.1), "tfd + noise on the prior's condition (prior_cond_noise 0.1)", SEEDS),
    "tfd_d2": (dict(TFD, fmb=2), "tfd with 2 transformer blocks instead of 4 (prior_fm_blocks 2)", SEEDS),
    "tfd_ema": (dict(TFD, ema=0.999), "tfd with a 10x longer EMA (ema_decay 0.999)", SEEDS),
    "base": (dict(), "the base: r8 hybrid_e360 (seed 42 = r8's run, seed 43 = r9's hybrid_base_s43)", (44,)),
}
# Runs of earlier rounds that brief.py shows beside this round's (it reads them where they are).
# Their configs equal the setting they stand for except the seed and the paths (self_check).
REFERENCE_RUNS = {
    "base": [("r8", "hybrid_e360", 42), ("r9", "hybrid_base_s43", 43)],
    "tfk8_r9": [("r9", "hybrid_gfix_tf_k8_s42", 42), ("r9", "hybrid_gfix_tf_k8_s43", 43)],
}
CARDS = ("0", "1", "2", "3", "4", "5", "6", "7")   # all 8 cards from round 9 on
PACK = 2                                           # jobs per card; --pack overrides
CPUS = 120                                         # of 128: torch threads + 4 DataLoader workers per job

DISPLAY = "HI-MGN-hybrid (latent_type hybrid)"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REL = f"configs/MeshGraphNets_Variational/hyperparameter_sweep/{ROUND}"
SWEEP_REL = "configs/MeshGraphNets_Variational/hyperparameter_sweep"

# Peak memory without gradient checkpointing (aarl 3090, 2026-10-27, bf16 LayerNorm):
# 17.1 GB per 100k nodes + 1 GB; worst batch-16 ex3 batch 1.04M nodes. With
# checkpointing the peak is a GUESS (CKPT_MEM of it). The coarse prior (256 nodes per
# graph, also with 8 draws or as a transformer) is left out: it is under 1% of the nodes.
WORST_NODES = {16: 1045e3}
GB_PER_100K = 17.1
CKPT_MEM = 0.3
CARD_GB = 260          # of 288 on a B300, the rest for the CUDA contexts and fragmentation

# Estimated hours, ONLY for the plan. EP_H = hours per batch-16 epoch alone on a card
# without checkpointing (r8 measured 277 s on 135). Checkpointing x1.52 (aarl 3090);
# PACK jobs on one card each run SHARE[PACK] times slower (guess).
EP_H = {16: 0.077}
CKPT_SLOWDOWN = 1.52
SHARE = {1: 1.0, 2: 1.5, 3: 2.1}
INFER_H = 2.0


@dataclasses.dataclass(frozen=True)
class Spec:
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
    pms: float = REF["pms"]
    fms: int = REF["fms"]
    inj: str = REF["inj"]
    mmdg: object = REF["mmdg"]
    arch: str = REF["arch"]
    param: str = REF["param"]
    k: int = REF["k"]
    fmb: int = REF["fmb"]
    ema: float = REF["ema"]
    cn: float = REF["cn"]
    seed: int = SEEDS[0]


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
training_epochs                  360
batch_size                       16
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
prior_freeze_epoch               240
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
    if spec.inj != REF["inj"]:
        b.set("local_inject", spec.inj, after="vae_local_mp_layers")
    b.set("lambda_mmd", f"{spec.mmd:g}")
    if spec.mmdg is not None:
        b.set("lambda_mmd_g", f"{spec.mmdg:g}", after="lambda_mmd")
    b.set("prior_mp_layers", spec.pm).set("prior_hidden_dim", spec.ph)
    if spec.arch != REF["arch"]:
        b.set("prior_arch", spec.arch, after="prior_hidden_dim")
    if spec.param != REF["param"]:
        b.set("prior_param", spec.param, after="prior_arch" if "prior_arch" in b.keys() else "prior_hidden_dim")
    if spec.fmb != REF["fmb"]:
        b.set("prior_fm_blocks", spec.fmb,
              after="prior_param" if "prior_param" in b.keys()
              else "prior_arch" if "prior_arch" in b.keys() else "prior_hidden_dim")
    if spec.k != REF["k"]:
        b.set("prior_fm_draws", spec.k, after="prior_nll_weight")
    if spec.cn != REF["cn"]:
        b.set("prior_cond_noise", f"{spec.cn:g}",
              after="prior_fm_draws" if "prior_fm_draws" in b.keys() else "prior_nll_weight")
    if spec.ema != REF["ema"]:
        b.set("ema_decay", f"{spec.ema:g}")
    b.set("prior_fm_steps", spec.fms)
    b.set("use_checkpointing", use_checkpointing(spec, pack))
    b.set("resume_training", "True")
    return b


def infer_body(spec, job, pack):
    b = train_body(spec, job, pack)
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
    arm: str
    spec: Spec
    change: str


def build_jobs():
    """Seed 42 of every arm first (the first pass over the cards), then seed 43, then
    the third seeds: job k and job k + len(CARDS) share a card, and an arm's two seeds
    land on different cards."""
    jobs = []
    for seed in sorted({s for *_x, seeds in ARMS.values() for s in seeds}):
        for arm, (changes, what, seeds) in ARMS.items():
            if seed in seeds:
                spec = dataclasses.replace(Spec(), **changes, seed=seed)
                jobs.append(Job(f"{PLACEMENT}_{arm}_s{seed}", arm, spec, f"{what}; seed {seed}"))
    return jobs


def est_hours(job, pack):
    ckpt = CKPT_SLOWDOWN if use_checkpointing(job.spec, pack) else 1.0
    return EP_H[job.spec.bs] * job.spec.ep * ckpt * SHARE[pack] + INFER_H


def slot_gpus(pack):
    """run_135.sh's GPUS: every card once, then again, PACK times (job k -> slot k)."""
    return list(CARDS) * pack


def threads_per_job(pack):
    return max(2, CPUS // (len(CARDS) * pack) - 4)


def write_run_135(pack):
    text = f"""\
#!/usr/bin/env bash
# B300 round 10 (HI-MGN-hybrid coarse prior against over-fitting, 3 seeds), node 135: see run_node.sh (order of work, overrides)
# and README.md (the jobs as one queue). WRITTEN BY make_sweep.py --pack {pack}: {pack} job(s) per card on
# GPUs {', '.join(CARDS)}, {len(CARDS) * pack} slots; each job gets {threads_per_job(pack)} torch threads + 4
# DataLoader workers. To change the packing, regenerate with another --pack (running jobs resume).
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
THREADS_PER_JOB="${{THREADS_PER_JOB:-{threads_per_job(pack)}}}" \\
GPUS="${{GPUS:-{' '.join(slot_gpus(pack))}}}" \\
    bash "$HERE/run_node.sh" 135 "$@" || exit $?
# The jobs have started: diagnose_135.sh waits in the background and diagnoses each job as it
# finishes (brief.py prints the result); a second start finds it running and does nothing.
if [ "${{TRAIN:-1}}${{INFER:-1}}" != "00" ]; then
    bash "$HERE/diagnose_135.sh"
fi
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
    lines = [f"% ex3 B300 hybrid coarse-prior sweep ({ROUND}), job {job.name}: {DISPLAY}",
             *(["% The transformer prior trains with dropout 0.1 (TF_DROPOUT in model/conditional_prior.py)."]
               if job.spec.arch == "transformer" else []),
             f"% Role: {job.change}",
             f"% Base: {REF_JOB[0]} {REF_JOB[1]} (360 epochs, batch 16); seed 42 is that run, 43 r9's hybrid_base_s43.",
             f"% Generated by {REL}/make_sweep.py --pack {pack}; edit that file, not this one.",
             f"% Run paths: output/b300_{ROUND}/<job>/; one shared hierarchy cache; resume_training True."
             if kind == "train" else
             "% 4 draws per held-out scene = 1024 per geometry against its 256 FE results (spread_metrics.json)."]
    return "\n".join(lines) + "\n%\n" + DATA


def write(path, text):
    """LF, no BOM; an unchanged file keeps its mtime (run_node.sh compares them)."""
    data = text.encode("utf-8")
    if not (path.is_file() and path.read_bytes() == data):
        path.write_bytes(data)


def self_check():
    """Every reference run of an earlier round equals the setting it stands for here,
    except build-only keys and paths."""
    def read(p):
        out = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.split("%", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = " ".join(parts[1].split())
        return out

    build_only = {"hierarchy_cache_build_workers", "use_checkpointing", "use_compile"}
    stands_for = {"base": dict(), "tfk8_r9": TFD}
    ok = True
    for arm, runs in REFERENCE_RUNS.items():
        for rnd, name, seed in runs:
            spec = dataclasses.replace(Spec(), **stands_for[arm], seed=seed)
            for kind in ("train", "infer"):
                ref_path = REPO / SWEEP_REL / rnd / f"config_{kind}_{name}.txt"
                if not ref_path.is_file():
                    print(f"  self-check: {rnd}/{ref_path.name} not found, skipped")
                    continue
                body = train_body(spec, "x", 1) if kind == "train" else infer_body(spec, "x", 1)
                mine, ref = body.as_dict(), read(ref_path)
                diff = sorted(k for k in set(ref) | set(mine)
                              if k not in PATH_KEYS | build_only and ref.get(k) != mine.get(k))
                status = "ok" if not diff else "DIFFERS: " + ", ".join(
                    f"{k} {ref.get(k)} -> {mine.get(k)}" for k in diff)
                ok &= not diff
                print(f"  self-check {arm} seed {seed} ({kind}) vs {rnd}/{ref_path.name}: {status}")
    return ok


def plan(jobs, pack):
    """Job k on slot k; a freed slot takes the next job."""
    gpus = slot_gpus(pack)
    free = [0.0] * len(gpus)
    out = []
    for k, j in enumerate(jobs):
        s = k if k < len(gpus) else min(range(len(gpus)), key=free.__getitem__)
        t = free[s]
        free[s] = t + est_hours(j, pack)
        out.append((j, gpus[s], t, free[s]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the jobs and estimated hours; write nothing")
    ap.add_argument("--pack", type=int, default=PACK, choices=sorted(SHARE), help="jobs per card")
    ap.add_argument("--force", action="store_true",
                    help=f"regenerate even though output/b300_{ROUND} exists and a job's config changes "
                         "(e.g. to change --pack)")
    args = ap.parse_args()
    pack = args.pack

    jobs = build_jobs()
    peaks = sorted((peak_gb(j.spec, use_checkpointing(j.spec, pack)) for j in jobs), reverse=True)
    if sum(peaks[:pack]) > CARD_GB:   # the worst card: the PACK largest jobs together
        raise SystemExit(f"the {pack} largest jobs need ~{sum(peaks[:pack]):.0f} GB together, over "
                         f"{CARD_GB}; {pack} per card does not fit (CKPT_MEM {CKPT_MEM} is a guess)")
    rows = plan(jobs, pack)
    makespan = max(e for *_, e in rows)
    table = ["| # | job | change from the base | GPU | checkpointing | est. peak GB | est. h | est. start-end (days) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, (j, gpu, t, e) in enumerate(rows, 1):
        ck = use_checkpointing(j.spec, pack)
        table.append(f"| {k} | {j.name} | {j.change} | {gpu} | {'on' if ck else 'off'} | "
                     f"{peak_gb(j.spec, ck):.0f} | {e - t:.0f} | {t / 24:.1f}-{e / 24:.1f} |")
    summary = (f"{len(jobs)} trainings ({len(ARMS)} arms; seeds {', '.join(map(str, SEEDS))}; the base only 44, "
               f"its 42 = {REF_JOB[0]} {REF_JOB[1]}, its 43 = r9 hybrid_base_s43); --pack {pack}: {pack} per card on GPUs "
               f"{', '.join(CARDS)} = {len(slot_gpus(pack))} slots, {threads_per_job(pack)} torch threads per job. "
               f"Estimated makespan {makespan / 24:.1f} days (GUESSED speeds: {EP_H[16]} h per batch-16 epoch alone, "
               f"x{CKPT_SLOWDOWN} with checkpointing, x{SHARE[pack]} per job at {pack} per card; peaks with "
               f"checkpointing guessed at {CKPT_MEM} of the full peak).")
    if args.plan:
        print(summary)
        print("\n".join(table))
        return 0

    out = HERE
    if not self_check():
        raise SystemExit("self-check failed: a reference run differs from the setting it stands for (above)")
    configs, lines = {}, []
    for j in jobs:
        configs[out / f"config_train_{j.name}.txt"] = header(j, "train", pack) + train_body(j.spec, j.name, pack).text()
        configs[out / f"config_infer_{j.name}.txt"] = header(j, "infer", pack) + infer_body(j.spec, j.name, pack).text()
        lines.append(f"{NODE} {j.name} mgnv {REL}/config_train_{j.name}.txt {REL}/config_infer_{j.name}.txt")
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
          f"# B300 hybrid coarse-prior sweep {ROUND}: node job method train_config infer_config. "
          f"Generated by make_sweep.py --pack {pack}.\n"
          "# Run order = line order; job k starts on slot k, then each freed slot takes the next line.\n"
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
