#!/usr/bin/env python3
"""Generate B300 round 15: round 14's best setting (joint_tfk8_dec, 56 against the base's 78)
again, on the augmented ex3 dataset (210 geometries, was 150), with a third seed, a longer run
and the per-sample KL beside it; the base again on the new data; and stochastic_mode noise
with the energy score as a two-job check. 14 jobs, all at once, about 60 hours (the user,
2026-11-27).

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r15/make_sweep.py           # writes the configs here
    python .../r15/make_sweep.py --plan         # jobs, cards and estimated hours; writes nothing

Standard library only, any Python 3.8+.

    base                 seeds 42/43/44   two parts: the prior alone from epoch 160 (2/3), its loss
                                          does not reach the encoder. The control. Earlier rounds'
                                          base runs trained on 150 geometries and do not count here.
    joint_tfk8           seeds 42/43      round 14's a6 (74): no prior-only part, the prior's loss
                                          reaches the encoder; transformer prior, 8 flow draws per
                                          sample, z_l read at the coarsest level, no MMD on z_g.
    joint_tfk8_dec       seeds 42/43/44   round 14's a8 (55/57): joint_tfk8 with prior_trunk decoder
                                          (the prior reads the simulator's descending arm at z = 0).
                                          The round's main setting: does 56 come back on the new data
                                          and with a third seed.
    joint_tfk8_dec_kl10  seeds 42/43      the same with the per-sample KL (kl_weight 10) instead of
                                          the MMD. Round 14 never ran the three together: KL 10 helped
                                          the GNN prior (74) and hurt tfk8 without the trunk (84).
    joint_tfk8_dec_e450  seeds 42/43      the same as joint_tfk8_dec for 450 epochs (1.9x the updates),
                                          alone on a card without gradient checkpointing.
    noise_energy         seed 42          stochastic_mode noise (no encoder, no prior) with noise_loss
                                          energy: the fair energy score per graph instead of the CRPS
                                          per node. Round 14's noise (CRPS) was 125.
    noise_global_energy  seed 42          the same with latent_type global (one vector per sample);
                                          round 14's noise_global (CRPS) was 245.

The data (2026-11-26). ex3_shell_buckling.h5 now holds 210 geometries x 64 draws: the 150 of the
earlier rounds and 60 around the buckle / no-buckle transition (shape_idx 300-359). Each of the
60 shares a split_group with one training geometry, so `split_group_attr split_group` keeps the
earlier rounds' 120/15/15 partition and puts every new geometry in training: 180 geometries,
11,520 samples, 720 steps per epoch at batch 16 (was 480). ex3_shell_buckling_infer.h5 holds 16
held-out geometries x 256 draws: the 8 of the earlier rounds (shape_idx 150-157) and 8 more
(200-207). Scores are grouped per geometry by `eval_group_attr shape_idx`. brief.py gives the
held-out score on 150-157 (the per-seed numbers and m; only these compare with earlier rounds)
and on 200-207 (x) separately.

Epochs. 240 epochs of 720 steps are the 172,800 updates of round 14's 360 epochs of 480. The two
noise jobs train 200 epochs (144,000 updates): a noise step costs 1.14x a base step, and at 240
epochs their card would end 5-10 hours after the others; round 14's noise jobs were stopped at
about 150,000 updates.

Cards. ORDER names each job's card. Cards 0 and 3 run one 450-epoch job each without gradient
checkpointing; the other six run two jobs each with it (the two would not fit otherwise), paired
so that every card's work is about equal. No card holds two seeds of one setting. run_135.sh's
GPUS lists the cards in job order (job k starts on slot k) and sets MPS=0.

Edit this file, never a generated config. use_checkpointing and the card layout do not enter the
resume fingerprint, so the round can be stopped, regenerated (--force) and restarted: every job
resumes.
"""
import argparse
import dataclasses
import sys
from decimal import Decimal
from pathlib import Path

ROUND = "r15"
NODE = "135"

# ================================================================== the sweep
# The base: r8's hybrid_e360 (mp_per_level 4, 6, 8, 6, 4, coarsening 4096, 256, batch 16) at
# 240 epochs of the augmented dataset. beta_aux 10, voronoi_clusters and mp_per_level are fixed.
REF = dict(lr="1e-4", ep=240, bs=16, ar=1000, w=128, mmd=1.0, ld=32, ch=4, ph=128, pm=3, pms=0.05, fms=30,
           inj="all", mmdg=None, arch="gnn", param="velocity", k=1, fmb=4, cn=0.0, drop=0.0,
           lt="hybrid", cap=0.0, wu=0, anc=0.0, ldrop=0.0, frz="auto", genc=0.0, reg="mmd", klw="1e-3",
           tp=1.0, cw=0.0, trunk="fine", sm="vae", nd=2, nl="crps")
MP = (4, 6, 8, 6, 4)
VORONOI = (4096, 256)

# setting -> (changes from REF, what it is, seeds). The run order and the cards are ORDER below.
# frz "auto" = prior_freeze_epoch at 2/3 of training_epochs; genc = prior_grad_to_encoder;
# reg/klw = latent_reg and kl_weight; inj = local_inject; mmdg = lambda_mmd_g; arch = prior_arch;
# k = prior_fm_draws; trunk = prior_trunk; sm/nd/nl = stochastic_mode, noise_draws, noise_loss.
JOINT = dict(frz=0, genc=1.0)
# r9's gfix_tf_k8: the transformer prior trained on 8 flow draws per sample, z_l read at the
# coarsest level only, no MMD on z_g.
TFK8 = dict(inj="coarsest", mmdg=0, arch="transformer", k=8)
A8 = dict(JOINT, **TFK8, trunk="decoder")   # round 14's a8, joint_tfk8_dec
LONG_EPOCHS = 450
NOISE_DRAWS, NOISE_EPOCHS = 2, 200
NOISE = dict(sm="noise", nd=NOISE_DRAWS, nl="energy", ep=NOISE_EPOCHS)
A8_TEXT = ("no prior-only part and the prior's loss reaches the encoder (prior_freeze_epoch 0, "
           "prior_grad_to_encoder 1); transformer prior, 8 flow draws per sample, z_l read at the coarsest "
           "level, no MMD on z_g; the prior reads the simulator's descending arm at z = 0 (prior_trunk decoder)")
SETTINGS = {   # brief.py's a0-a6, in this order
    "base": (dict(), f"the base: the prior alone from epoch {REF['ep'] * 2 // 3}, its loss does not reach "
                     "the encoder", (42, 43, 44)),
    "joint_tfk8": (dict(JOINT, **TFK8), "round 14's a6: no prior-only part and the prior's loss reaches the "
                                        "encoder; transformer prior, 8 flow draws per sample, z_l read at the "
                                        "coarsest level, no MMD on z_g", (42, 43)),
    "joint_tfk8_dec": (A8, f"round 14's a8: {A8_TEXT}", (42, 43, 44)),
    "joint_tfk8_dec_kl10": (dict(A8, reg="kl", klw="10"),
                            "round 14's a8 with the per-sample KL to N(0,I) instead of the MMD (latent_reg kl, "
                            "kl_weight 10)", (42, 43)),
    f"joint_tfk8_dec_e{LONG_EPOCHS}": (dict(A8, ep=LONG_EPOCHS),
                                       f"round 14's a8 for {LONG_EPOCHS} epochs (the others: {REF['ep']})", (42, 43)),
    "noise_energy": (NOISE, f"stochastic_mode noise: no encoder, no prior; z_g and z_l ~ N(0,I), {NOISE_DRAWS} "
                            f"draws per batch, loss = their fair energy score per graph (noise_loss energy); "
                            f"{NOISE_EPOCHS} epochs", (42,)),
    "noise_global_energy": (dict(NOISE, lt="global"),
                            f"stochastic_mode noise with latent_type global (one N(0,I) vector per sample, no "
                            f"z_l), {NOISE_DRAWS} draws per batch, loss = their fair energy score per graph; "
                            f"{NOISE_EPOCHS} epochs", (42,)),
}
LONG = f"joint_tfk8_dec_e{LONG_EPOCHS}"
# (setting, seed, card), in run order: job k starts on slot k, whose card is the one named here.
# Cards 0 and 3 hold one 450-epoch job each (alone, so no gradient checkpointing); the other six
# hold two jobs. No card holds two seeds of one setting (build_jobs refuses otherwise). Every
# card's guessed work is about equal: a x1.0 job (base, joint_tfk8) with a x0.88 one
# (prior_trunk decoder), and the two noise jobs (x1.14 for 200 epochs = x0.95) together.
#   card 4: base 42 + joint_tfk8_dec 43          card 5: base 43 + joint_tfk8_dec 44
#   card 6: base 44 + joint_tfk8_dec_kl10 42     card 7: joint_tfk8_dec 42 + joint_tfk8 42
#   card 2: joint_tfk8 43 + joint_tfk8_dec_kl10 43
#   card 1: noise_energy 42 + noise_global_energy 42
ORDER = [("base", 42, "4"), ("base", 43, "5"), ("base", 44, "6"), ("joint_tfk8_dec", 42, "7"),
         ("joint_tfk8", 43, "2"), ("noise_energy", 42, "1"), (LONG, 42, "0"), (LONG, 43, "3"),
         ("joint_tfk8_dec", 43, "4"), ("joint_tfk8_dec", 44, "5"), ("joint_tfk8_dec_kl10", 42, "6"),
         ("joint_tfk8", 42, "7"), ("joint_tfk8_dec_kl10", 43, "2"), ("noise_global_energy", 42, "1")]
CARDS = ("0", "1", "2", "3", "4", "5", "6", "7")   # all 8 cards from round 9 on
CPUS = 120                                         # of 128: torch threads + 4 DataLoader workers per job

DISPLAY = {"hybrid": "HI-MGN-hybrid (latent_type hybrid)", "local": "HI-MGNflow (latent_type local)",
           "global": "HI-MGN-V (latent_type global)"}
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
REL = f"configs/MeshGraphNets_Variational/hyperparameter_sweep/{ROUND}"

# Peak memory without gradient checkpointing (aarl 3090, 2026-10-27, bf16 LayerNorm):
# 17.1 GB per 100k nodes + 1 GB. Worst batch of 16 over 240 epochs of the augmented dataset,
# drawn at random from the per-geometry node counts (2026-11-27): 1.14M nodes (the 150-geometry
# file: 1.05M). With checkpointing the peak is a GUESS (CKPT_MEM of it). The coarse prior (256
# nodes per graph, also with 8 draws or as a transformer) is left out: under 1% of the nodes.
WORST_NODES = {16: 1150e3}
GB_PER_100K = 17.1
CKPT_MEM = 0.3
CARD_GB = 260          # of 288 on a B300, the rest for the CUDA contexts and fragmentation

# Estimated hours, ONLY for the plan. EP_H = hours per batch-16 epoch alone on a card without
# checkpointing: r8 measured 277 s for 480 steps on 135; 720 steps and 1.2% more nodes per
# sample give 0.117. Checkpointing x1.52 (aarl 3090). With 2 jobs on one card each runs
# SHARE[2] times slower (guess); plan() lets a job speed up when its card's other job ends.
# Round 14 (3 per card, checkpointing) ran at about R14_RATIO of the hours this model gave.
EP_H = {16: 0.117}
CKPT_SLOWDOWN = 1.52
SHARE = {1: 1.0, 2: 1.5}
INFER_H = 4.0          # 4096 scenes x 4 draws, alone on a card (round 14: 2.0 for 2048)
R14_RATIO = 0.8


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
    trunk: str = REF["trunk"]
    sm: str = REF["sm"]
    nd: int = REF["nd"]
    nl: str = REF["nl"]
    seed: int = 42


# stochastic_mode noise, a GUESS from the fine-level block count (what sets both time and
# activation memory): the base runs 14 fine blocks per step (simulator 4 + 4 of mp_per_level,
# posterior encoder 3, prior trunk 3); noise runs the simulator's 8, NOISE_DRAWS times.
# prior_trunk decoder, the same kind of GUESS: the prior's own prior_mp_layers fine blocks (with
# their backward) are gone, and the simulator's 4 descending fine blocks run once more forward
# only, under no_grad (a forward ~1/3 of forward + backward). tf_k8 and KL are counted as 1.0.
FINE_BLOCKS_BASE, FINE_BLOCKS_SIM, SIM_DOWN_FINE_BLOCKS, FORWARD_ONLY = 14, 8, 4, 1 / 3


def cost_scale(spec):
    if spec.sm != "vae":
        return spec.nd * FINE_BLOCKS_SIM / FINE_BLOCKS_BASE                # noise: 16/14 = 1.14
    if spec.trunk == "decoder":
        return (FINE_BLOCKS_BASE - spec.pm + SIM_DOWN_FINE_BLOCKS * FORWARD_ONLY) / FINE_BLOCKS_BASE   # 0.88
    return 1.0


def peak_gb(spec, ckpt):
    per_100k = GB_PER_100K * (spec.w / 128 + 4.7 / 21 * (spec.ph / 128 - 1) + 4.7 / 21 * (spec.pm / 3 - 1))
    gb = per_100k * WORST_NODES[spec.bs] / 1e5 * cost_scale(spec) + 1.0
    return gb * CKPT_MEM + 1.0 if ckpt else gb


def use_checkpointing(spec, on_card):
    """On when `on_card` jobs like this one would not fit the card without it."""
    return peak_gb(spec, False) > CARD_GB / on_card


# =========================================================== config bodies
BASE_TEXT = """\
model                            meshgraphnets-v
mode                             train
gpu_ids                          0
parallel_mode                    ddp
split_seed                       42
split_group_attr                 split_group
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
training_epochs                  240
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
prior_freeze_epoch               160
"""
# The held-out file has no split_group (and needs none: inference never splits). Its scores are
# grouped per geometry, and so are the diagnosis tools' on the training file, where a
# split_group can hold two geometries.
EVAL_GROUP = "shape_idx"
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
# What stochastic_mode noise does not build: the posterior encoder, the latent regularizer, the
# auxiliary loss and the prior (ai_cae4all/specs/meshgraphnets_variational.py NOISE_INACTIVE_KEYS),
# and prior_freeze_epoch, which the spec refuses above 0.
NOISE_DROPPED = (
    "vae_encoder_mp_layers", "vae_graph_aware", "vae_local_mp_layers", "latent_reg", "kl_weight",
    "lambda_mmd", "lambda_mmd_g", "mmd_bandwidth", "mmd_cond_width", "alpha_recon", "beta_aux",
    "posterior_min_std", "posterior_max_std_g", "local_dropout", "use_conditional_prior", "prior_type",
    "prior_family", "prior_mp_layers", "prior_trunk", "prior_hidden_dim", "prior_arch", "prior_param",
    "prior_fm_blocks", "prior_tf_dropout", "prior_nll_weight", "prior_fm_draws", "prior_cond_noise",
    "prior_grad_to_encoder", "prior_fm_steps", "prior_fm_solver", "prior_fm_time_power",
    "prior_freeze_epoch",
)
SAMPLES, INFER_BATCH = 4, 4  # draws per held-out scene; 256 scenes per geometry -> 1024 draws
JOBS_MARK = "## Jobs (generated by make_sweep.py; everything below is rewritten)"


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


BASE = [tuple(line.split(None, 1)) for line in BASE_TEXT.splitlines()]


def run_dir(job):
    return f"../../output/b300_{ROUND}/{job}"


def lr_value(lr):
    m, x = lr.split("e-")
    return format(Decimal(m) * Decimal(10) ** -int(x), "f")


def train_body(spec, job, ckpt):
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
    if spec.trunk != REF["trunk"]:
        b.set("prior_trunk", spec.trunk, after="prior_mp_layers")
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
    if spec.sm != REF["sm"]:  # no encoder, no prior: their keys would be ignored (preflight warns)
        b.set("stochastic_mode", spec.sm, after="use_vae").set("noise_draws", spec.nd, after="stochastic_mode")
        if spec.nl != REF["nl"]:
            b.set("noise_loss", spec.nl, after="noise_draws")
        b.drop(*NOISE_DROPPED)
    b.set("use_checkpointing", ckpt)
    b.set("resume_training", "True")
    return b


def infer_body(spec, job, ckpt):
    b = train_body(spec, job, ckpt)
    b.set("mode", "inference").set("log_file_dir", f"{run_dir(job)}/infer.log").set("batch_size", 8)
    b.drop("hierarchy_cache_dir", "hierarchy_cache_build_workers", "resume_training")
    b.set("eval_group_attr", EVAL_GROUP, after="split_group_attr")
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
    card: str
    ckpt: bool = False


def job_name(setting, spec):
    """hybrid_<setting>_s<seed>; no prefix for latent_type global."""
    return f"{'hybrid_' + setting if spec.lt == 'hybrid' else setting}_s{spec.seed}"


def build_jobs():
    """The jobs in the order of ORDER, each on the card named there."""
    listed = sorted((s, seed) for s, seed, _card in ORDER)
    if listed != sorted((s, seed) for s, (_c, _w, seeds) in SETTINGS.items() for seed in seeds):
        raise SystemExit("ORDER must list every seed of every setting exactly once")
    jobs = []
    for setting, seed, card in ORDER:
        if card not in CARDS:
            raise SystemExit(f"ORDER: {setting} seed {seed} names card {card}, not one of {', '.join(CARDS)}")
        changes, what, _seeds = SETTINGS[setting]
        spec = dataclasses.replace(Spec(), **changes, seed=seed)
        jobs.append(Job(job_name(setting, spec), setting, spec, f"{what}; seed {seed}", card))
    for card in CARDS:
        here = [j for j in jobs if j.card == card]
        twice = sorted({j.setting for j in here if sum(o.setting == j.setting for o in here) > 1})
        if twice:
            raise SystemExit(f"card {card} holds two seeds of {', '.join(twice)}: change ORDER")
        if len(here) not in SHARE:
            raise SystemExit(f"card {card} holds {len(here)} jobs; ORDER must give every card 1 or 2")
        for j in here:
            j.ckpt = use_checkpointing(j.spec, len(here))
        peak = sum(peak_gb(j.spec, j.ckpt) for j in here)
        if peak > CARD_GB:
            raise SystemExit(f"card {card}: its jobs need ~{peak:.0f} GB together, over {CARD_GB} "
                             f"(CKPT_MEM {CKPT_MEM} is a guess)")
    return jobs


def train_hours(job):
    """Hours of the job's training alone on a card."""
    return EP_H[job.spec.bs] * job.spec.ep * cost_scale(job.spec) * (CKPT_SLOWDOWN if job.ckpt else 1.0)


def gpus(jobs):
    """run_135.sh's GPUS: the card of job k at position k (job k starts on slot k)."""
    return [j.card for j in jobs]


def threads_per_job(jobs):
    return max(2, CPUS // len(jobs) - 4)


def write_run_135(jobs):
    per_card = "; ".join(f"{c}: {' + '.join(j.name for j in jobs if j.card == c)}" for c in CARDS)
    text = f"""\
#!/usr/bin/env bash
# B300 round 15 (round 14's best setting, joint_tfk8_dec, on the augmented ex3 dataset: three seeds, a
# 450-epoch run, the per-sample KL beside it; the base again; stochastic_mode noise with the energy score),
# node 135: see run_node.sh (order of work, overrides) and README.md (the jobs as one queue).
# WRITTEN BY make_sweep.py: {len(jobs)} jobs at once on GPUs {', '.join(CARDS)}, the card of job k at position k of GPUS.
# Per card: {per_card}.
# Each job gets {threads_per_job(jobs)} torch threads + 4 DataLoader workers.
# MPS is OFF here, always (MPS=0 below is not an override): round 13 lost both seeds of two
# settings with two jobs per card under MPS. To change the layout, edit ORDER in make_sweep.py
# and regenerate (running jobs resume).
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MPS=0 \\
THREADS_PER_JOB="${{THREADS_PER_JOB:-{threads_per_job(jobs)}}}" \\
GPUS="${{GPUS:-{' '.join(gpus(jobs))}}}" \\
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
% Dataset: shell buckling, 210 geometries x 64 imperfection draws (ex3_shell_buckling.h5: the 150 of the
% earlier rounds and 60 around the buckle / no-buckle transition, shape_idx 300-359). split_group_attr
% split_group keeps the earlier 120/15/15 partition; every new geometry trains (180 geometries, 11,520 samples).
% Held-out file ex3_shell_buckling_infer.h5 = 16 unseen geometries x 256 draws (shape_idx 150-157, the 8 of
% the earlier rounds, and 200-207); scored per geometry (eval_group_attr shape_idx).
% Fields: input [0:3] x,y,z | output [3:6] ux,uy,uz | conditioner [6:8] thickness, delta | [8] node type.
% Static (T=1): the model predicts the field in one step; the spread comes from the unseen imperfection.
"""


def header(job, kind):
    lines = [f"% ex3 B300 round 15 ({ROUND}), job {job.name}: {DISPLAY[job.spec.lt]}",
             f"% Role: {job.change}",
             f"% Base: this round's hybrid_base_s42/43/44 ({REF['ep']} epochs of 720 steps, batch 16).",
             f"% Generated by {REL}/make_sweep.py; edit that file, not this one.",
             f"% Run paths: output/b300_{ROUND}/<job>/; one shared hierarchy cache; resume_training True; GPU {job.card}."
             if kind == "train" else
             "% 4 draws per held-out scene = 1024 per geometry against its 256 FE results (spread_metrics.json)."]
    return "\n".join(lines) + "\n%\n" + DATA


def write(path, text):
    """LF, no BOM; an unchanged file keeps its mtime (run_node.sh compares them)."""
    data = text.encode("utf-8")
    if not (path.is_file() and path.read_bytes() == data):
        path.write_bytes(data)


def plan(jobs):
    """Every job starts at 0 on its card and does its training, then its inference. With n jobs
    running on one card each runs SHARE[n] times slower than alone. Returns {job name: (hours
    until its training ends, hours until its inference ends)}."""
    out = {}
    for card in CARDS:
        left = {j.name: [train_hours(j), INFER_H] for j in jobs if j.card == card}
        t = 0.0
        while left:
            slow = SHARE[len(left)]
            dt = min(next(w for w in work if w > 1e-9) for work in left.values()) * slow
            t += dt
            for name in sorted(left):
                work = left[name]
                stage = 0 if work[0] > 1e-9 else 1
                work[stage] -= dt / slow
                if stage == 0 and work[0] <= 1e-9:
                    out[name] = [t, None]
                if work[0] <= 1e-9 and work[1] <= 1e-9:
                    out[name][1] = t
                    del left[name]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the jobs and estimated hours; write nothing")
    ap.add_argument("--force", action="store_true",
                    help=f"regenerate even though output/b300_{ROUND} exists and a job's config changes")
    args = ap.parse_args()

    jobs = build_jobs()
    ends = plan(jobs)
    makespan = max(e for _t, e in ends.values())
    table = ["| # | job | what it is | GPU | epochs | checkpointing | est. peak GB | est. training h | est. end h |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, j in enumerate(jobs, 1):
        t, e = ends[j.name]
        table.append(f"| {k} | {j.name} | {j.change} | {j.card} | {j.spec.ep} | {'on' if j.ckpt else 'off'} | "
                     f"{peak_gb(j.spec, j.ckpt):.0f} | {t:.0f} | {e:.0f} |")
    summary = (f"{len(jobs)} trainings ({len(SETTINGS)} settings: "
               + "; ".join(f"{setting} seeds {'/'.join(map(str, seeds))}" for setting, (_c, _w, seeds) in SETTINGS.items())
               + f"), all at once on GPUs {', '.join(CARDS)} (GPUS {' '.join(gpus(jobs))}), "
               f"{threads_per_job(jobs)} torch threads per job, MPS off. "
               f"Estimated end of the last inference: {makespan:.0f} h as modelled, about "
               f"{makespan * R14_RATIO:.0f} h if the round runs at round 14's {R14_RATIO} of the model "
               f"(GUESSED speeds: {EP_H[16]} h per batch-16 epoch alone, x{CKPT_SLOWDOWN} with checkpointing, "
               f"each job x{SHARE[2]} while 2 share a card, {INFER_H:.0f} h of inference per job; cost per job "
               f"from cost_scale: noise x1.14, prior_trunk decoder x0.88; peaks with checkpointing guessed at "
               f"{CKPT_MEM} of the full peak). The diagnosis (diagnose_135.sh) runs after that.")
    if args.plan:
        print(summary)
        print("\n".join(table))
        return 0

    out = HERE
    configs, lines = {}, []
    for j in jobs:
        configs[out / f"config_train_{j.name}.txt"] = header(j, "train") + train_body(j.spec, j.name, j.ckpt).text()
        configs[out / f"config_infer_{j.name}.txt"] = header(j, "infer") + infer_body(j.spec, j.name, j.ckpt).text()
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
          f"# B300 round 15 ({ROUND}): node job method train_config infer_config. Generated by make_sweep.py.\n"
          "# Run order = line order; job k starts on slot k (run_135.sh's GPUS names its card).\n"
          + "\n".join(lines) + "\n")
    write_run_135(jobs)
    readme = out / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if JOBS_MARK in text:
        text = text[:text.index(JOBS_MARK)]
    write(readme, text + JOBS_MARK + "\n\n" + summary + "\n\n" + "\n".join(table) + "\n")
    print(summary)
    print(f"wrote {out} ({len(jobs)} jobs, run_135.sh; {len(changed)} config(s) changed, {len(stale)} removed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
