#!/usr/bin/env python3
"""Generate aarl round 1: HI-MGN-hybrid on ex3, ablations first, then tuning.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/aarl1/make_sweep.py           # writes the configs here
    python .../aarl1/make_sweep.py --plan        # queue, slots and estimated hours; writes nothing
    python .../aarl1/make_sweep.py --pack 2      # two jobs per card (MPS), if the peak allows it

Standard library only, any Python 3.8+. Edit this file, never a generated config.

Why this round looks the way it does (README.md has the full reasoning, LOG.md the
journal): B300 rounds 6-8 already moved almost every hyperparameter of the hybrid one
at a time (lr, epochs, batch, alpha_recon, width, MMD/KL weight, latent sizes, prior
size, posterior floor, sampling temperature/inflation). They never REMOVED a part of
the method. So phase 1 is the ablation table, at round 6's hybrid_ref recipe (batch
4, 120 epochs), so every arm reads against r6's three reference seeds (88, 88, 107)
and r6's arm table; phase 2 is the tuning that phase 1 and the diagnosis point to.

aarl is a shared box: at most two cards (0 and 1) at a time. A 3090 has 24 GB, so
every training runs with gradient checkpointing (batch 4 peaks at ~57 GB without).
"""
import argparse
import dataclasses
import heapq
import sys
from decimal import Decimal
from pathlib import Path

ROUND = "aarl1"
NODE = "aarl"
PLACEMENT = "hybrid"

# ================================================================== the sweep
# r6 hybrid_ref: mp_per_level 4, 6, 8, 6, 4, coarsening 4096, 256, batch 4, 120 epochs.
REF = dict(lr="1e-4", ep=120, bs=4, ar=1000, w=128, reg="mmd", mmd=1.0, kl=None, ld=32, ch=4,
           ph=128, pm=3, fb=4, pms=0.05, fms=30,
           aux=10, aug=True, hv=2, ema=True, zc="adaln", ga=True, frz="auto", cond=2)
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)
SEED = 42

# Phase 1, the ablations: arm -> (changes from REF, what it removes). Queue order = this
# order. The reference first (the diagnosis and every comparison need it), then the
# parts most likely to matter. Known (r6, batch 4, 120 epochs, seed 42 unless said):
# ref 88/88/107 (seeds 42/43/44), mmd0.1 76, mmd100 71, ar100 86, ar10000 76, pm2 76,
# pm6 77, ph256 90, fb8 82, e60 91, e240 69; ref_det (use_vae False) ran once.
ARMS = {
    "ref": (dict(), "the reference: r6 hybrid_ref's recipe, retrained on aarl (3090, checkpointing)"),
    "mmd0": (dict(mmd=0.0), "no latent regularizer: lambda_mmd 0 (only the posterior std floor is left)"),
    "aux0": (dict(aux=0), "no peak-to-valley loss on the decoded field: beta_aux 0"),
    "nofreeze": (dict(frz=0), "no prior-only tail: prior_freeze_epoch 0 (encoder, decoder and prior "
                              "train together to the end)"),
    "noaug": (dict(aug=False), "no geometry augmentation: augment_geometry False (no Z rotation / reflection)"),
    "concat": (dict(zc="concat"), "z enters by concatenation instead of AdaLN-Zero: z_conditioning concat"),
    "flatenc": (dict(ga=False), "the posterior encoder does not see the geometry: vae_graph_aware False"),
    "hv1": (dict(hv=1), "one coarsening per sample instead of two: hierarchy_variants 1"),
    "noema": (dict(ema=False), "no weight EMA: use_ema False"),
    "nocond": (dict(cond=0), "the conditions (thickness, imperfection delta) are not inputs: cond_var 0"),
}
# Phase 2, the tuning: filled from phase 1, the reference's posterior vs. prior diagnosis
# and r8's batch-16 results (LOG.md says which result chose which arm). Empty until then.
TUNE = {}
CARDS = ("0", "1")      # the two cards this round may hold at once (aarl is shared)
PACK = 1                # jobs per card; --pack overrides
CPUS = 40               # of 80, shared with other users: torch threads + 4 DataLoader workers per job

DISPLAY = "HI-MGN-hybrid (latent_type hybrid)"
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REL = f"configs/MeshGraphNets_Variational/hyperparameter_sweep/{ROUND}"
SWEEP_REL = "configs/MeshGraphNets_Variational/hyperparameter_sweep"

# Reserved memory WITH checkpointing, GB, measured on aarl (3090, 2026-11-07, two-epoch probes,
# expandable_segments on): batch 4 peak 9.37 / reserved 10.0, batch 8 16.7 / 17.8. CARD_GB of 24
# leaves the CUDA contexts.
PEAK_GB = {4: 10.0, 8: 17.8}
CARD_GB = 23.0
# Hours, alone on a card with checkpointing (same probes, other users' jobs on cards 2-3): a joint
# epoch (train + test pass + pictures) and a prior-only epoch after prior_freeze_epoch; a CRPS
# validation every val_interval (10) epochs; START = hierarchy cache build + normalization stats.
# PACK jobs on one card under MPS: each runs SHARE[PACK] times slower (aarl 2026-10-27: 3 jobs
# 1.54x the throughput of one -> 1.95 each; 2 per card measured on this round 2026-11-07: 1.2 vs 0.76 s/it).
EP_H = {4: (0.41, 0.17), 8: (0.38, 0.16)}
VAL_H, START_H = 0.075, 0.75
SHARE = {1: 1.0, 2: 1.58, 3: 1.95}
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
    aux: float = REF["aux"]
    aug: bool = REF["aug"]
    hv: int = REF["hv"]
    ema: bool = REF["ema"]
    zc: str = REF["zc"]
    ga: bool = REF["ga"]
    frz: object = REF["frz"]
    cond: int = REF["cond"]
    seed: int = SEED


# =========================================================== config bodies
# Round 6's hybrid_ref, key by key (self_check compares it with r6's config).
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
hierarchy_cache_build_workers    32
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
    return f"../../output/{ROUND}/{job}"


def lr_value(lr):
    m, x = lr.split("e-")
    return format(Decimal(m) * Decimal(10) ** -int(x), "f")


def train_body(spec, job):
    b, run = Body(BASE), run_dir(job)
    b.set("modelpath", f"{run}/model.pth").set("log_file_dir", f"{run}/train.log")
    b.set("inference_output_dir", f"{run}/infer")
    b.set("hierarchy_cache_dir", f"../../output/{ROUND}/mscache/mgnv_v{VORONOI[0]}_{VORONOI[1]}")
    b.set("training_epochs", spec.ep)
    b.set("prior_freeze_epoch", spec.ep * 2 // 3 if spec.frz == "auto" else spec.frz)
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
    b.set("beta_aux", f"{spec.aux:g}").set("augment_geometry", spec.aug)
    b.set("hierarchy_variants", spec.hv).set("use_ema", spec.ema)
    b.set("z_conditioning", spec.zc).set("vae_graph_aware", spec.ga)
    b.set("cond_var", spec.cond)
    b.set("use_checkpointing", True)
    b.set("resume_training", "True")
    return b


def infer_body(spec, job, source=None):
    """The inference config of `job`; `source` = the job whose checkpoint it samples (default: its own)."""
    b = train_body(spec, source or job)
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
    phase: int
    source: object = None   # sampling only: the job whose checkpoint it samples


def est_hours(job, pack):
    if job.source:
        return INFER_H
    s = job.spec
    joint = s.ep if s.frz == 0 else (s.ep * 2 // 3 if s.frz == "auto" else s.frz)
    h_joint, h_prior = EP_H[s.bs]
    train = joint * h_joint + (s.ep - joint) * h_prior + s.ep // 10 * VAL_H
    return (train + START_H) * SHARE[pack] + INFER_H


def build_jobs():
    jobs = []
    for phase, arms in ((1, ARMS), (2, TUNE)):
        for arm, (changes, what) in arms.items():
            jobs.append(Job(f"{PLACEMENT}_{arm}", dataclasses.replace(Spec(), **changes), what, phase))
    return jobs


def slot_gpus(pack):
    """run_aarl.sh's GPUS: every card once, then again, PACK times (job k -> slot k)."""
    return list(CARDS) * pack


def threads_per_job(pack):
    return max(2, CPUS // (len(CARDS) * pack) - 4)


def write_run_aarl(pack):
    text = f"""\
#!/usr/bin/env bash
# aarl round 1 (HI-MGN-hybrid ex3 ablations and tuning): see run_node.sh (order of work, overrides),
# README.md (the design) and LOG.md (the journal). WRITTEN BY make_sweep.py --pack {pack}: {pack} job(s) per
# card on GPUs {', '.join(CARDS)} only (aarl is shared: two cards at most), {len(CARDS) * pack} slots; each job gets
# {threads_per_job(pack)} torch threads + 4 DataLoader workers. Another --pack: regenerate (running jobs resume).
THREADS_PER_JOB="${{THREADS_PER_JOB:-{threads_per_job(pack)}}}" \\
GPUS="${{GPUS:-{' '.join(slot_gpus(pack))}}}" \\
    exec bash "$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/run_node.sh" aarl "$@"
"""
    write(HERE / "run_aarl.sh", text)


# ==================================================================== headers
DATA = """\
% Dataset: shell buckling, 150 geometries x 64 imperfection draws (ex3_shell_buckling.h5); held-out file
% ex3_shell_buckling_infer.h5 = 8 unseen geometries x 256 draws. split_group_attr keeps a geometry's draws together.
% Fields: input [0:3] x,y,z | output [3:6] ux,uy,uz | conditioner [6:8] thickness, delta | [8] node type.
% Static (T=1): the model predicts the field in one step; the spread comes from the unseen imperfection.
"""


def header(job, kind):
    what = "ablation" if job.phase == 1 else "tuning"
    lines = [f"% ex3 aarl hybrid round ({ROUND}), phase {job.phase} ({what}), job {job.name}: {DISPLAY}",
             f"% Role: {job.change}",
             "% Reference: hybrid_ref of this round (= r6 hybrid_ref: batch 4, 120 epochs).",
             f"% Generated by {REL}/make_sweep.py; edit that file, not this one.",
             f"% Run paths: output/{ROUND}/<job>/; one shared hierarchy cache; resume_training True."
             if kind == "train" else
             "% 4 draws per held-out scene = 1024 per geometry against its 256 FE results (spread_metrics.json)."]
    if job.source:
        lines.insert(4, f"% Trains nothing: samples {job.source}'s checkpoint (waits for it in the queue).")
    return "\n".join(lines) + "\n%\n" + DATA


def write(path, text):
    """LF, no BOM; an unchanged file keeps its mtime (run_node.sh compares them)."""
    data = text.encode("utf-8")
    if not (path.is_file() and path.read_bytes() == data):
        path.write_bytes(data)


def self_check():
    """The reference equals round 6's hybrid_ref except build-only keys and paths."""
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
    ok = True
    for kind in ("train", "infer"):
        ref_path = REPO / SWEEP_REL / "r6" / f"config_{kind}_hybrid_ref.txt"
        if not ref_path.is_file():
            print(f"  self-check: r6/{ref_path.name} not found, skipped")
            continue
        body = train_body(Spec(), "ref") if kind == "train" else infer_body(Spec(), "ref")
        mine, ref = body.as_dict(), read(ref_path)
        diff = sorted(k for k in set(ref) | set(mine)
                      if k not in PATH_KEYS | build_only and ref.get(k) != mine.get(k))
        status = "ok" if not diff else "DIFFERS: " + ", ".join(f"{k} {ref.get(k)} -> {mine.get(k)}" for k in diff)
        ok &= not diff
        print(f"  self-check reference ({kind}) vs r6/{ref_path.name}: {status}")
    return ok


def plan(jobs, pack):
    """run_node.sh's dealing: job s on slot s, then each freed slot takes the next job."""
    gpus = slot_gpus(pack)
    heap, out, end = [], [], {}
    for k, j in enumerate(jobs):
        t, s = (0.0, k) if k < len(gpus) else heapq.heappop(heap)
        if j.source:
            t = max(t, end[j.source] - INFER_H)
        h = est_hours(j, pack)
        out.append((j, gpus[s], t, t + h))
        end[j.name] = t + h
        heapq.heappush(heap, (t + h, s))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the queue and estimated hours; write nothing")
    ap.add_argument("--pack", type=int, default=PACK, choices=sorted(SHARE), help="jobs per card")
    ap.add_argument("--force", action="store_true",
                    help=f"regenerate even though output/{ROUND} exists and a job's config changes")
    args = ap.parse_args()
    pack = args.pack

    jobs = build_jobs()
    peak = max(PEAK_GB[j.spec.bs] for j in jobs if not j.source)
    if peak * pack > CARD_GB:
        raise SystemExit(f"{pack} jobs per card need ~{peak * pack:.0f} GB together, over {CARD_GB}")
    rows = plan(jobs, pack)
    makespan = max(e for *_, e in rows)
    table = ["| # | phase | job | change from the reference | GPU | est. h | est. start-end (days) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, (j, gpu, t, e) in enumerate(rows, 1):
        table.append(f"| {k} | {j.phase} | {j.name} | {j.change} | {gpu} | {e - t:.0f} | "
                     f"{t / 24:.1f}-{e / 24:.1f} |")
    n_train = sum(1 for j in jobs if not j.source)
    summary = (f"{n_train} trainings + {len(jobs) - n_train} sampling-only jobs, seed {SEED}, one run each; "
               f"--pack {pack}: {pack} per card on GPUs {', '.join(CARDS)} = {len(slot_gpus(pack))} slots, "
               f"{threads_per_job(pack)} torch threads per job, gradient checkpointing on. Estimated makespan "
               f"{makespan / 24:.1f} days (batch 4 alone on a 3090, measured: {EP_H[4][0]} h per joint "
               f"epoch, {EP_H[4][1]} h per prior-only epoch; x{SHARE[pack]} per job at {pack} per card).")
    if args.plan:
        print(summary)
        print("\n".join(table))
        return 0

    out = HERE
    if not self_check():
        raise SystemExit("self-check failed: the reference drifted from round 6's hybrid_ref (above)")
    configs, lines = {}, []
    for j in jobs:
        if not j.source:
            configs[out / f"config_train_{j.name}.txt"] = header(j, "train") + train_body(j.spec, j.name).text()
        configs[out / f"config_infer_{j.name}.txt"] = \
            header(j, "infer") + infer_body(j.spec, j.name, j.source).text()
        tcfg = "-" if j.source else f"{REL}/config_train_{j.name}.txt"
        lines.append(f"{NODE} {j.name} mgnv {tcfg} {REL}/config_infer_{j.name}.txt")
    stale = [p for p in out.glob("config_*.txt") if p not in configs]
    changed = [p for p, text in configs.items() if p.is_file() and p.read_bytes() != text.encode("utf-8")]
    if (stale or changed) and (REPO / "output" / ROUND).exists() and not args.force:
        raise SystemExit(f"output/{ROUND} exists (the round may be running) and this would "
                         + ", ".join(f"{'remove' if p in stale else 'change'} {p.name}" for p in stale + changed)
                         + "; --force to regenerate anyway")
    for p in stale:
        p.unlink()
    for p, text in configs.items():
        write(p, text)
    write(out / "jobs.txt",
          f"# aarl hybrid round {ROUND}: node job method train_config|- infer_config. "
          f"Generated by make_sweep.py --pack {pack}.\n"
          "# Run order = line order; job k starts on slot k, then each freed slot takes the next line.\n"
          + "\n".join(lines) + "\n")
    write_run_aarl(pack)
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if JOBS_MARK in text:
        text = text[:text.index(JOBS_MARK)]
    write(readme, text + JOBS_MARK + "\n\n" + summary + "\n\n" + "\n".join(table) + "\n")
    print(summary)
    print(f"wrote {out} ({len(jobs)} jobs, run_aarl.sh for --pack {pack}; "
          f"{len(changed)} config(s) changed, {len(stale)} removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
