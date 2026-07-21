"""Generate the Transolver ex2 capacity sweep: one-factor-at-a-time (OFAT) star, 8 cells.

=== THE DESIGN ===

An OFAT "star": pick a center point, then move ONE axis at a time while holding
the other two at the center. Every cell differs from the anchor in exactly one
hyperparameter, so any change in validation loss is attributable to that one
axis with no confounding.

    ANCHOR (center):  num_layers 10, latent_dim 256, slice_num 128

    num_layers axis :  6  <- [10] ->  20     (latent_dim 256, slice_num 128 held)
    latent_dim axis : 128 <- [256] -> 512    (num_layers 10, slice_num 128 held)
    slice_num axis  :  64 <- [128] -> 256    (num_layers 10, latent_dim 256 held)

The three axes share the anchor, so this is 3 + 3 + 3 - 2 = 7 unique cells,
plus cell 8 = a replicate of the existing ex2 config (L8 C256 M128) so the
sweep ties back to prior ex2 runs.

Cell 8 differs from the anchor only in num_layers, so it lands ON the layers
axis for free -- giving that axis 4 points {6, 8, 10, 20} instead of 3. That is
a bonus, not an accident to correct: depth is the axis the Transolver-3 paper
identifies as the one that actually scales, so extra resolution there is where
it is most useful.

8 cells = exactly one wave (4 two-GPU DDP pairs x 2 servers), so the entire
sweep runs concurrently in a single shot.

=== WHAT THIS DESIGN CAN AND CANNOT TELL YOU ===

CAN: the isolated main effect of each axis -- "does adding depth help on ex2?",
"does widening help?", "does raising slice_num help?" -- each answered cleanly
against a common reference point.

CANNOT: interactions. If depth only pays off at latent_dim 512, or if a high
slice_num only hurts on deep models, an OFAT star cannot see it -- every cell
sits on a line through the anchor, so no cell has two axes moved at once. If an
axis shows a strong effect here, the natural follow-up is a small 2x2 on that
axis crossed with the next most promising one.

=== WHY THESE RANGES ===

The existing ex2 config already sits at or above every published Transolver
config on slice_num -- the Transolver / Transolver++ / Transolver-3 papers all
use 32-64. Transolver-3's industrial-scale recipe is to add DEPTH at a fixed
latent_dim 256 (they run 16-24 layers). This star brackets both: layers reach
20 (Transolver-3 territory), slice_num reaches 256 (4x the published max), and
latent_dim reaches 512 (2x anything published for this task family).

=== HARDWARE ===

2x B300 servers, 8 GPUs each. Each cell is ONE 2-GPU DDP job -> 4 cells per
server, 8 concurrent, one wave. Cells 1-4 run on server A, cells 5-8 on server B.
Wave duration is set by the single most expensive cell (L10 C512 M128), since
every cell has its own dedicated GPU pair.

=== MEMORY: WHY use_checkpointing IS True ON EVERY CELL ===

ex2 graphs are ~200k nodes (dataset/ex2.h5: nodal_data (8, 50, 199993)), and
batch_size is PER-RANK under DDP (DistributedSampler, distributed_training.py),
so each GPU carries 4 x 200k nodes. The naive kernel materializes a
[heads, N, slice_num] softmax weight tensor per layer, promoted to fp32 by
_stable() in physics_attention.py:

    8 heads * 200k nodes * 256 slices * 4 B = 1.64 GB   per graph, per layer

The worst cell here (L10 C256 M256) would hold ~66 GB for those tensors alone
before counting the pre-softmax logits, which roughly double-to-triple it ->
130-200 GB against a B300's ~288 GB. That would probably fit, but not with
enough margin to risk losing the single wave to an OOM, so checkpointing stays
on. It is also what the current ex2 anchor config already uses, so this keeps
parity with prior runs rather than introducing a difference.

Note this is a weaker constraint than the earlier full-factorial design faced:
its L20/C512/M256 corner needed 260-390 GB and checkpointing was strictly
mandatory. Here it is precautionary.

=== KNOWN THROUGHPUT CAVEAT (uniform, does not confound) ===

distributed_training.py hardcodes pin_memory = torch.cuda.is_available(), which
is not config-reachable. On ~200k-node graphs pinning serializes the H2D copy of
large batches. It applies identically to all 8 cells, so it costs wall-clock
across the board but cannot bias the comparison between cells.

=== TIME INTEGRATION IS PINNED TO ar_ot, NOT LEFT IMPLICIT ===

Every cell sets `time_integration ar_ot` explicitly even though that is the
native default (CONFIGURATION_REFERENCE.md section 3.5), and it matches what all
five ex2 baseline configs currently use. Leaving it implicit would be fragile:
flipping ex2 to ar_rt drops optimizer steps per epoch by roughly T (49 here),
turns reported validation into ROLLOUT loss (not on the same scale as ar_ot
validation loss), and makes bptt retain one state tensor per unrolled step,
invalidating the memory budget above. If ex2 moves to ar_rt, re-derive this
sweep rather than flipping the key.

=== PINNED ACROSS ALL 8 CELLS ===

mlp_ratio 4, num_heads 8, 500 epochs, batch_size 4, lr 1e-4, weight_decay 1e-4,
dropout 0, std_noise 0.01, augment_geometry True, EMA, grad clip 3.0,
use_checkpointing True, time_integration ar_ot.

Regularization is deliberately NOT scaled with capacity (project decision): ex2
has only 40 training trajectories after the 80/10/10 split, so the larger cells
are ALLOWED to look worse from overfitting. That is the signal this sweep
measures, not noise to regularize away -- summarize_sweep.py reports the
min-vs-final gap that makes it visible.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- the OFAT star ---------------------------------------------------------
ANCHOR_LAYERS, ANCHOR_DIM, ANCHOR_SLICES = 10, 256, 128

LAYERS_AXIS = [6, 20]      # excursions from ANCHOR_LAYERS
DIM_AXIS = [128, 512]      # excursions from ANCHOR_DIM
SLICES_AXIS = [64, 256]    # excursions from ANCHOR_SLICES

# the existing ex2 config's shape -- ties the star back to prior runs, and
# happens to add a 4th point to the layers axis
BRIDGE = dict(layers=8, dim=ANCHOR_DIM, slices=ANCHOR_SLICES)

# ---- pinned across every cell ---------------------------------------------
MLP_RATIO = 4
NUM_HEADS = 8
EPOCHS = 500
BATCH_SIZE = 4        # PER-RANK under DDP
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
CHECKPOINTING = "True"
TIME_INTEGRATION = "ar_ot"

# ---- scheduling ------------------------------------------------------------
PAIRS_PER_SERVER = 4
SERVERS = ["A", "B"]


def cost(c):
    """N-scaled per-step work proxy: the two C->C input projections plus the
    mlp_ratio-4 MLP (~10 C^2) and the slice/deslice einsums (2 C M)."""
    return c["layers"] * (10 * c["dim"] ** 2 + 2 * c["dim"] * c["slices"])


def make_tag(c):
    return f"L{c['layers']}_C{c['dim']}_M{c['slices']}_mlp{MLP_RATIO}"


def build_cells():
    """Anchor first, then one axis at a time, then the ex2 bridge."""
    A = dict(layers=ANCHOR_LAYERS, dim=ANCHOR_DIM, slices=ANCHOR_SLICES)
    cells = [dict(n=1, role="ANCHOR (star center)", **A)]
    n = 2
    for v in LAYERS_AXIS:
        cells.append(dict(n=n, role=f"num_layers axis: {ANCHOR_LAYERS} -> {v}",
                          **{**A, "layers": v}))
        n += 1
    for v in DIM_AXIS:
        cells.append(dict(n=n, role=f"latent_dim axis: {ANCHOR_DIM} -> {v}",
                          **{**A, "dim": v}))
        n += 1
    for v in SLICES_AXIS:
        cells.append(dict(n=n, role=f"slice_num axis: {ANCHOR_SLICES} -> {v}",
                          **{**A, "slices": v}))
        n += 1
    cells.append(dict(n=n, role="ex2 bridge (existing config; also a 4th point "
                                "on the num_layers axis)", **BRIDGE))
    return cells


def assign_schedule(cells):
    """Single wave: cells 1-4 -> server A, cells 5-8 -> server B, each on its
    own 2-GPU pair."""
    for i, c in enumerate(cells):
        c["wave"] = 1
        c["server"] = SERVERS[i // PAIRS_PER_SERVER]
        pair_idx = i % PAIRS_PER_SERVER
        c["gpu"] = f"{2 * pair_idx},{2 * pair_idx + 1}"
    return cells


TRAIN_TMPL = """model          transolver  # backend selector -- {{transolver}}
mode           train       # run mode -- {{train|inference}}
gpu_ids        {gpu}         # cell {n} ({tag}) -- server {server}, 2-GPU DDP pair
parallel_mode  ddp         # parallelism -- {{ddp|node_shard}}
'
%   ex2 capacity sweep (OFAT star) cell {n}/{total}: {tag}
%   ROLE: {role}
%   Star center = num_layers {anchor_layers}, latent_dim {anchor_dim}, slice_num {anchor_slices}.
%   Each cell differs from that center in exactly ONE hyperparameter, so any
%   difference in validation loss is attributable to that one axis. See
%   gen_sweep_configs.py for the design, its interaction blind spot, and the
%   memory math behind use_checkpointing.
log_file_dir          ../../output/transolver/ex2_sweep/train{n}_{tag}.log     # log file path, relative to the backend output dir
modelpath             ../output/transolver/ex2_sweep/transolver_train{n}_{tag}.pth  # checkpoint path -- save target in train, load source in inference
dataset_dir           ../dataset/ex2.h5                                    # training mesh-state HDF5 path
infer_dataset         ../dataset/ex2.h5                                    # inference input HDF5 path
inference_output_dir  ../output/transolver/rollout/ex2_sweep/train{n}_{tag}  # rollout output directory
infer_timesteps       49                                                   # 50 stored timesteps -> 49 autoregressive rollout steps
split_seed            42                                                   # RNG seed for the deterministic 80/10/10 train/val/test split; default 42
'
%   Common parameters
input_var             4                   # physical-state input channel count -- positive int; temporal rollout needs input_var == output_var
output_var             4                  # predicted field/delta channel count -- positive int
positional_features    4                  # extra numeric positional feature channels -- int >= 0; default 0
use_node_types          True              # append one-hot node type (feature row 7 / Part No.) -- {{True|False}}; needs a >7-row file
feature_loss_weights    1.0, 1.0, 1.0, 1.0  # per-output-channel loss weights -- comma list, length must equal output_var, entries >= 0
'
%   Network parameters -- THIS CELL: num_layers={layers}, latent_dim={dim}, slice_num={slices}
coordinate_normalization  centered_isotropic  # coordinate scheme -- must be 'centered_isotropic'; enforced and recorded in the checkpoint data config
latent_dim                {dim}               # hidden width -- positive int, must be divisible by num_heads
num_layers                {layers}                  # Transolver block count -- positive int
num_heads                 {heads}                   # attention heads -- positive int, must divide latent_dim
slice_num                 {slices}                  # Physics-Attention learned slice count -- positive int
attention_kernel          naive               # attention kernel -- {{naive|slice_space}}; chunking and node_shard require slice_space; default naive
chunk_size                0                   # attention tiling size -- int >= 0, 0 = untiled; positive requires attention_kernel slice_space; default 0
infer_mode                direct              # inference mode -- {{direct|decoupled}}; default direct
infer_chunk_size          0                   # decoupled-inference chunk size -- int >= 0; default 0
mlp_ratio                 {mlp}                   # block MLP expansion ratio -- PINNED at the anchor value, not an axis in this sweep
dropout                   0.0                 # block dropout rate -- float in [0,1); default 0.0
temperature_init          0.5                 # initial slice-assignment temperature -- float; default 0.5
temperature_min           0.1                 # min slice-assignment temperature -- float; default 0.1
temperature_max           5.0                 # max slice-assignment temperature -- float; require 0 < min <= init <= max; default 5.0
small_output_init         False               # shrink final head at init -- {{True|False}}; default true when timesteps unknown/temporal, false for static T=1
'
%   Optimization -- pinned identically across all {total} cells (project decision: do not compensate capacity with regularization)
training_epochs   {epochs}     # epoch count -- matches the epoch budget used by every other ex2 baseline
batch_size        {batch_size}       # PER-RANK DataLoader batch size under DDP, before grad_accum_steps -- positive int
learningr         {lr}  # AdamW learning rate -- positive float; prefer decimal notation
weight_decay      {wd}  # AdamW weight decay -- float >= 0; default 0.0001
warmup_epochs     3       # linear LR warmup epochs before the cosine schedule -- int >= 0; default 3
num_workers       2       # DataLoader worker processes -- int >= 0
prefetch_factor   4       # batches prefetched per worker -- used only when num_workers > 0; default 4
grad_accum_steps  1       # gradient accumulation steps per optimizer step -- int >= 1; default 1
max_grad_norm     3.0     # gradient-clipping max norm -- float; default 3.0
'
%   Training regularization and runtime
std_noise          0.01   # lower than ex1 -- input-noise injection for autoregressive rollout stability (multi-step)
noise_gamma        1      # target-correction multiplier matching injected input noise -- default 1
augment_geometry   True   # kept ON for ex2 parity across all baselines (isotropy caveat overridden); matches MeshGraphNets ex2
use_amp            True   # bfloat16 autocast on supported CUDA -- {{True|False}}; native bf16 tensor cores on B300, no Turing MAGMA-fallback penalty
use_checkpointing  {ckpt}   # precautionary at this star's sizes (worst cell ~130-200 GB without it on 200k-node graphs); also matches the existing ex2 config
use_ema            True   # maintain EMA weights (inference prefers EMA when stored) -- {{True|False}}; default False
ema_decay          0.99   # EMA decay rate -- float in (0,1); default 0.999
use_compile        False  # torch.compile(dynamic=True) on the model -- {{True|False}}; default False
'
%   Time integration -- PINNED EXPLICITLY (CONFIGURATION_REFERENCE.md section 3.5).
%   ar_ot is the native default and what all five ex2 baselines use, but it is
%   pinned rather than left implicit because switching ex2 to ar_rt would
%   invalidate this sweep: optimizer steps per epoch drop by ~T (49), validation
%   becomes ROLLOUT loss (not comparable to ar_ot numbers), and bptt holds one
%   state tensor per unrolled step, breaking the memory budget above.
time_integration  {time_integration}   # ar_ot = one-step/teacher-forced; ar_rt = autoregressive rollout training
'
%   Evaluation
val_interval   5   # validation cadence in epochs -- positive int; default 1
test_interval  25  # periodic test-visualization cadence in epochs -- positive int; default 10
'
%   Dataset behavior
use_world_edges      False  # must stay False -- Transolver does not consume graph edges -- {{False}}
use_multiscale       False  # must stay False -- MGN multiscale hierarchy is unsupported here -- {{False}}
write_preprocessing  False  # must stay False -- writing normalization back into the source HDF5 is rejected -- {{False}}
"""


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main():
    # clear stale configs from any previous (differently sized) design
    for fname in os.listdir(HERE):
        if fname.startswith("config_train") and fname.endswith(".txt"):
            os.remove(os.path.join(HERE, fname))

    cells = assign_schedule(build_cells())
    total = len(cells)

    for c in cells:
        write(os.path.join(HERE, f"config_train{c['n']}.txt"),
              TRAIN_TMPL.format(
                  n=c["n"], total=total, tag=make_tag(c), role=c["role"],
                  gpu=c["gpu"], server=c["server"], layers=c["layers"],
                  dim=c["dim"], slices=c["slices"], heads=NUM_HEADS,
                  mlp=MLP_RATIO, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  lr=LEARNING_RATE, wd=WEIGHT_DECAY, ckpt=CHECKPOINTING,
                  time_integration=TIME_INTEGRATION,
                  anchor_layers=ANCHOR_LAYERS, anchor_dim=ANCHOR_DIM,
                  anchor_slices=ANCHOR_SLICES))

    lines = ["# server gpus cell tag  (generated by gen_sweep_configs.py; single wave)"]
    for c in cells:
        lines.append(f"{c['server']} {c['gpu']} {c['n']} {make_tag(c)}")
    write(os.path.join(HERE, "schedule.txt"), "\n".join(lines) + "\n")

    print(f"Wrote {total} configs + schedule.txt   (OFAT star, single wave)\n")
    hdr = (f"{'cell':>4} {'srv':>3} {'gpus':>5}  {'config':<22} "
           f"{'rel_cost':>8}  role")
    print(hdr)
    print("-" * (len(hdr) + 18))
    base = min(cost(c) for c in cells)
    for c in cells:
        print(f"{c['n']:>4} {c['server']:>3} {c['gpu']:>5}  {make_tag(c):<22} "
              f"{cost(c) / base:>7.1f}x  {c['role']}")
    slowest = max(cells, key=cost)
    print(f"\nAll {total} cells run CONCURRENTLY in one wave "
          f"({PAIRS_PER_SERVER} 2-GPU pairs x {len(SERVERS)} servers).")
    print(f"Wave duration is set by the slowest cell: {make_tag(slowest)} "
          f"({cost(slowest) / base:.1f}x the cheapest).")


if __name__ == "__main__":
    main()
