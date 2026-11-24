#!/usr/bin/env python3
"""Generate B300 round 14: the prior (GNN or transformer) x the latent regularizer (MMD or the
per-sample KL at 1, 10, 100). Every run is on the joint setting of round 12, one job per card.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/make_sweep.py           # writes the configs here (--pack 1)
    python .../r14/make_sweep.py --plan         # jobs, cards and estimated hours; writes nothing

Standard library only, any Python 3.8+. Every run is r12's `joint`: prior_freeze_epoch 0
(one objective for all 360 epochs) and the prior's loss reaching the encoder
(prior_grad_to_encoder 1). Sixteen runs, eight settings x seeds 42/43:

                      MMD            KL 1              KL 10              KL 100
    GNN prior         joint          joint_kl1         joint_kl10         joint_kl100
    transformer       joint_tfk8     joint_tfk8_kl1    joint_tfk8_kl10    joint_tfk8_kl100

Why this table. Of round 13's twelve settings only joint_kl10 changed the latent (a decoded
N(0,I) draw 551 -> 211) and had both seeds low (69/72); joint_kl1 went the other way (106/84).
The only prior that fitted the training geometries better without a warm-up was r9's tf_k8,
never run on joint. So the KL is read as a line of weights a factor 10 apart, per prior.

    joint            r12's joint, nothing else changed (the control; it died in round 13).
    joint_kl1        latent_reg kl, kl_weight 1: every sample's posterior is pulled to N(0,I)
                     (the plain VAE term). The MMD only asks the rows of a batch, pooled, to
                     look N(0,I); this asks it of every sample, so of every geometry too.
                     (round 13: 106/84.)
    joint_kl10       kl_weight 10. (round 13: 69/72.)
    joint_kl100      kl_weight 100: where the KL's price (a posterior off the floor, a decoder
                     that stops reading z) overtakes its gain. (r8, two parts: 99.)
    joint_tfk8       r9's gfix_tf_k8 on joint: prior_arch transformer, prior_fm_draws 8,
                     local_inject coarsest, lambda_mmd_g 0. Not the transformer alone.
    joint_tfk8_kl1   the same prior with kl_weight 1.
    joint_tfk8_kl10  ... kl_weight 10: the two changes that moved anything, together.
    joint_tfk8_kl100 ... kl_weight 100.

joint_kl1 and joint_kl10 repeat round 13's runs: the second wave has eight cards and six
other jobs, so they cost no time, and they say whether 69/72 comes back.

The base (two parts, not run here) is r8 hybrid_e360, r9 hybrid_base_s43 and r10
hybrid_base_s44. Edit this file, never a generated config.

--pack sets the jobs per card, and with it run_135.sh's GPUS and THREADS_PER_JOB
(rewritten here) and each job's use_checkpointing (on only where the estimated peak of
PACK jobs would not fit the card). The default here is 1: 8 jobs on the 8 cards at once,
checkpointing off, the other eight as cards free. run_135.sh sets MPS=0 (one job per card
has nothing to share). use_checkpointing and the slot layout do not enter the resume
fingerprint, so the round can be stopped, regenerated with another --pack (--force) and
restarted: every job resumes.
"""
import argparse
import dataclasses
import sys
from decimal import Decimal
from pathlib import Path

ROUND = "r14"
NODE = "135"

# ================================================================== the sweep
# r8's hybrid_e360: r6 hybrid_ref (mp_per_level 4, 6, 8, 6, 4, coarsening 4096, 256)
# at batch 16 and 360 epochs. beta_aux 10, voronoi_clusters and mp_per_level are fixed.
REF = dict(lr="1e-4", ep=360, bs=16, ar=1000, w=128, mmd=1.0, ld=32, ch=4, ph=128, pm=3, pms=0.05, fms=30,
           inj="all", mmdg=None, arch="gnn", param="velocity", k=1, fmb=4, cn=0.0, drop=0.0,
           lt="hybrid", cap=0.0, wu=0, anc=0.0, ldrop=0.0, frz="auto", genc=0.0, reg="mmd", klw="1e-3",
           tp=1.0, cw=0.0)
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)
REF_JOB = ("r8", "hybrid_e360")   # the base at seed 42, trained in round 8
SEEDS = (42, 43)

# setting -> (changes from REF, what it is, seeds). The run order is ORDER below.
# frz "auto" = prior_freeze_epoch at 2/3 of training_epochs (240); genc = prior_grad_to_encoder;
# reg/klw = latent_reg and kl_weight; inj = local_inject; mmdg = lambda_mmd_g; arch = prior_arch;
# k = prior_fm_draws.
JOINT = dict(frz=0, genc=1.0)
# r9's gfix_tf_k8 (r12's joint_tfk8): the transformer prior trained on 8 flow draws per sample,
# z_l read at the coarsest level only, no MMD on z_g.
TFK8 = dict(inj="coarsest", mmdg=0, arch="transformer", k=8)
KL_WEIGHTS = ("1", "10", "100")
SETTINGS = {   # the GNN prior's line, then the transformer prior's (brief.py's a1-a4, a5-a8)
    "joint": (JOINT, "r12's joint: no prior-only part (prior_freeze_epoch 0) and the prior's loss "
                     "reaches the encoder (prior_grad_to_encoder 1)", SEEDS),
}
for _w in KL_WEIGHTS:
    SETTINGS[f"joint_kl{_w}"] = (dict(JOINT, reg="kl", klw=_w),
                                 f"joint, per-sample KL to N(0,I) instead of the MMD (latent_reg kl, "
                                 f"kl_weight {_w})", SEEDS)
SETTINGS["joint_tfk8"] = (dict(JOINT, **TFK8), "joint with r9's tf_k8 prior: transformer, 8 flow draws per "
                                                "sample, z_l injected at the coarsest level only", SEEDS)
for _w in KL_WEIGHTS:
    SETTINGS[f"joint_tfk8_kl{_w}"] = (dict(JOINT, **TFK8, reg="kl", klw=_w),
                                      f"joint, r9's tf_k8 prior, per-sample KL (kl_weight {_w})", SEEDS)
# Run order: job k on slot k (the cards of SLOT_CARDS), then each freed card takes the next job.
# First eight = what is not known yet and decides the round: the control (it died in round 13),
# the transformer prior with the MMD and with KL 10, and KL 100 on the GNN prior; with round
# 13's joint_kl1 / joint_kl10 the GNN line is complete after the first wave. A setting's two
# seeds are never on one card (round 13 lost two whole settings to two cards): in the first
# eight they are two lines apart, in the last eight four lines apart, and each of the last
# eight takes a different freed card.
ORDER = [("joint", 42), ("joint_tfk8_kl10", 42), ("joint", 43), ("joint_tfk8_kl10", 43),
         ("joint_tfk8", 42), ("joint_kl100", 42), ("joint_tfk8", 43), ("joint_kl100", 43),
         ("joint_tfk8_kl100", 42), ("joint_tfk8_kl1", 42), ("joint_kl10", 42), ("joint_kl1", 42),
         ("joint_tfk8_kl100", 43), ("joint_tfk8_kl1", 43), ("joint_kl10", 43), ("joint_kl1", 43)]
# Runs of earlier rounds that brief.py shows beside this round's (it reads them where they are).
# Their configs equal the setting they stand for except the seed and the paths (self_check).
REFERENCE_RUNS = {
    "base": [("r8", "hybrid_e360", 42), ("r9", "hybrid_base_s43", 43), ("r10", "hybrid_base_s44", 44)],
}
CARDS = ("0", "1", "2", "3", "4", "5", "6", "7")   # all 8 cards from round 9 on
# The card of slot k. Cards 0 and 3 are last: both seeds of two settings died on them in round
# 13, so the first four jobs (the control and the main setting) are on 4-7.
SLOT_CARDS = ("4", "5", "6", "7", "1", "2", "0", "3")
PACK = 1                                           # jobs per card; --pack overrides
CPUS = 120                                         # of 128: torch threads + 4 DataLoader workers per job

DISPLAY = {"hybrid": "HI-MGN-hybrid (latent_type hybrid)", "local": "HI-MGNflow (latent_type local)",
           "global": "HI-MGN-V (latent_type global)"}
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
    cn: float = REF["cn"]
    drop: float = REF["drop"]
    lt: str = REF["lt"]
    cap: float = REF["cap"]
    wu: int = REF["wu"]
    anc: float = REF["anc"]
    ldrop: float = REF["ldrop"]
    frz: object = REF["frz"]
    genc: float = REF["genc"]
    reg: str = REF["reg"]
    klw: str = REF["klw"]
    tp: float = REF["tp"]
    cw: float = REF["cw"]
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
    b.set("training_epochs", spec.ep)
    b.set("prior_freeze_epoch", spec.ep * 2 // 3 if spec.frz == "auto" else spec.frz)
    if spec.genc != REF["genc"]:
        b.set("prior_grad_to_encoder", f"{spec.genc:.1f}")
    b.set("batch_size", spec.bs).set("learningr", lr_value(spec.lr))
    b.set("latent_dim", spec.w).set("alpha_recon", spec.ar)
    b.set("posterior_min_std", f"{spec.pms:g}").set("training_seed", spec.seed)
    if spec.cap != REF["cap"]:
        b.set("posterior_max_std_g", f"{spec.cap:g}", after="posterior_min_std")
    b.set("vae_latent_dim", spec.ld).set("latent_ch", spec.ch).set("latent_type", spec.lt)
    if spec.inj != REF["inj"]:
        b.set("local_inject", spec.inj, after="vae_local_mp_layers")
    last = "local_inject" if "local_inject" in b.keys() else "vae_local_mp_layers"
    for key, value, default in (("local_dropout", spec.ldrop, REF["ldrop"]),
                                ("local_warmup_epochs", spec.wu, REF["wu"]),
                                ("zg_anchor_weight", spec.anc, REF["anc"])):
        if value != default:
            b.set(key, f"{value:g}", after=last)
            last = key
    b.set("lambda_mmd", f"{spec.mmd:g}")
    if spec.cw != REF["cw"]:
        b.set("mmd_cond_width", f"{spec.cw:g}", after="mmd_bandwidth")
    if spec.mmdg is not None:
        b.set("lambda_mmd_g", f"{spec.mmdg:g}", after="lambda_mmd")
    if spec.reg == "kl":   # the per-sample KL replaces the MMD: its keys would be ignored
        b.set("latent_reg", "kl")
        b.set("kl_weight", lr_value(spec.klw) if "e-" in spec.klw else spec.klw, after="latent_reg")
        b.drop("lambda_mmd", "lambda_mmd_g", "mmd_bandwidth")
    b.set("prior_mp_layers", spec.pm).set("prior_hidden_dim", spec.ph)
    if spec.arch != REF["arch"]:
        b.set("prior_arch", spec.arch, after="prior_hidden_dim")
    if spec.param != REF["param"]:
        b.set("prior_param", spec.param, after="prior_arch" if "prior_arch" in b.keys() else "prior_hidden_dim")
    if spec.fmb != REF["fmb"]:
        b.set("prior_fm_blocks", spec.fmb,
              after="prior_param" if "prior_param" in b.keys()
              else "prior_arch" if "prior_arch" in b.keys() else "prior_hidden_dim")
    if spec.drop != REF["drop"]:
        b.set("prior_tf_dropout", f"{spec.drop:g}",
              after="prior_fm_blocks" if "prior_fm_blocks" in b.keys() else "prior_arch")
    if spec.k != REF["k"]:
        b.set("prior_fm_draws", spec.k, after="prior_nll_weight")
    if spec.cn != REF["cn"]:
        b.set("prior_cond_noise", f"{spec.cn:g}",
              after="prior_fm_draws" if "prior_fm_draws" in b.keys() else "prior_nll_weight")
    b.set("prior_fm_steps", spec.fms)
    if spec.tp != REF["tp"]:
        b.set("prior_fm_time_power", f"{spec.tp:g}", after="prior_fm_solver")
    if spec.lt == "global":   # no coarse-level latent: these two keys would be ignored
        b.drop("latent_ch", "vae_local_mp_layers")
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
    setting: str
    spec: Spec
    change: str


def job_name(setting, spec):
    """hybrid_<setting>_s<seed>."""
    return f"{'hybrid_' + setting if spec.lt == 'hybrid' else setting}_s{spec.seed}"


def build_jobs():
    """The jobs in the order of ORDER (see there)."""
    if sorted(ORDER) != sorted((s, seed) for s, (_c, _w, seeds) in SETTINGS.items() for seed in seeds):
        raise SystemExit("ORDER must list every seed of every setting exactly once")
    if sorted(SLOT_CARDS) != sorted(CARDS):
        raise SystemExit("SLOT_CARDS must be CARDS in another order")
    jobs = []
    for setting, seed in ORDER:
        changes, what, _seeds = SETTINGS[setting]
        spec = dataclasses.replace(Spec(), **changes, seed=seed)
        jobs.append(Job(job_name(setting, spec), setting, spec, f"{what}; seed {seed}"))
    return jobs


def est_hours(job, pack):
    ckpt = CKPT_SLOWDOWN if use_checkpointing(job.spec, pack) else 1.0
    return EP_H[job.spec.bs] * job.spec.ep * ckpt * SHARE[pack] + INFER_H


def slot_gpus(pack):
    """run_135.sh's GPUS: every card once (in the order of SLOT_CARDS), then again, PACK
    times (job k -> slot k)."""
    return list(SLOT_CARDS) * pack


def threads_per_job(pack):
    return max(2, CPUS // (len(CARDS) * pack) - 4)


def write_run_135(pack):
    text = f"""\
#!/usr/bin/env bash
# B300 round 14 (the prior, GNN or transformer, x the latent regularizer, MMD or KL 1, 10, 100; each on
# r12's joint), node 135: see run_node.sh (order of work, overrides) and README.md (the jobs as one queue).
# WRITTEN BY make_sweep.py --pack {pack}: {pack} job(s) per card on GPUs {', '.join(CARDS)}, {len(CARDS) * pack} slots
# (slot order {' '.join(slot_gpus(pack))}); each job gets {threads_per_job(pack)} torch threads + 4 DataLoader workers.
# MPS is OFF here, always (MPS=0 below is not an override): round 13 lost both seeds of two
# settings with two jobs per card under MPS. To change the packing, regenerate with another
# --pack (running jobs resume).
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MPS=0 \\
THREADS_PER_JOB="${{THREADS_PER_JOB:-{threads_per_job(pack)}}}" \\
GPUS="${{GPUS:-{' '.join(slot_gpus(pack))}}}" \\
    bash "$HERE/run_node.sh" 135 "$@" || exit $?
# The jobs have started: diagnose_135.sh waits in the background, diagnoses each job as it
# finishes (brief.py prints the result) and draws the per-geometry histograms at the end;
# a second start finds it running and does nothing.
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
    lines = [f"% ex3 B300 prior x latent regularizer ({ROUND}), job {job.name}: {DISPLAY[job.spec.lt]}",
             f"% Role: {job.change}",
             f"% Base: {REF_JOB[0]} {REF_JOB[1]} (360 epochs, batch 16); its seeds 42/43/44 are r8/r9/r10 runs.",
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
    stands_for = {"base": dict()}
    ok = True
    for setting, runs in REFERENCE_RUNS.items():
        for rnd, name, seed in runs:
            spec = dataclasses.replace(Spec(), **stands_for[setting], seed=seed)
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
                print(f"  self-check {setting} seed {seed} ({kind}) vs {rnd}/{ref_path.name}: {status}")
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
    summary = (f"{len(jobs)} trainings ({len(SETTINGS)} settings: "
               + "; ".join(f"{setting} seeds {'/'.join(map(str, seeds))}" for setting, (_c, _w, seeds) in SETTINGS.items())
               + "; the base is not run: its 42/43/44 = r8 hybrid_e360, r9 hybrid_base_s43, r10 hybrid_base_s44); "
               f"--pack {pack}: {pack} per card on GPUs "
               f"{', '.join(CARDS)} = {len(slot_gpus(pack))} slots (slot order {' '.join(slot_gpus(pack))}), "
               f"{threads_per_job(pack)} torch threads per job, MPS off. "
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
          f"# B300 prior x latent regularizer {ROUND}: node job method train_config infer_config. "
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
