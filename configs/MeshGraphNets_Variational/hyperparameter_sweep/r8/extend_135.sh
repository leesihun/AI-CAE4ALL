#!/usr/bin/env bash
# B300 round 8, node 135: everything after the round's start in one command, nothing typed in between:
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/extend_135.sh
#
# Start it once, any time while the round (run_135.sh) runs; it goes to the
# background on its own, so the terminal can close. It then
#   1. waits until the round's base (hybrid_base) is trained and inferred;
#   2. runs the posterior vs. prior diagnosis on it (diagnose.sh, forward passes
#      only, on the round's card with the most free memory; a few minutes);
#   3. picks the extra trainings from that diagnosis (the same thresholds as
#      posterior_vs_prior_groups.py's verdict, gap = floor -> prior):
#        encode/decode costs > 2/3 of the gap, or the prior beats the posterior
#                                -> ENCDEC (latent size, channels, posterior floor, recon weight)
#        encode/decode costs < 1/3 -> PRIOR (prior depth, width, FM blocks)
#        in between, or no diagnosis (the base failed) -> BOTH (three of each)
#      and appends them to jobs.txt (make_sweep.py --extra; written on this node only);
#   4. runs them as a second run of the round (LANE=b WAIT_FREE=1): each card
#      takes one once the round has no job left to start and the card is idle,
#      so the node stays busy until about Sunday night;
#   5. ends with the brief.py block: every job of the round and the extras, with
#      the diagnosis as its "diag" lines (brief.py prints the same at any time).
#
#   progress:  tail -f output/b300_r8/extend_135.out   (the extras' jobs: run_135_b.out)
#   result:    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/brief.py
#   stop:      touch output/b300_r8/stop_135   (stops the round's queue too), or kill_all.sh
#
# Run it again after a stop or a crash: the diagnosis and the choice are kept
# (output/b300_r8/_diag/base_choice.txt; delete it to diagnose again), finished
# extras are skipped and unfinished ones resume.
#
# The script calls itself: EXTEND_PHASE=run is the background driver.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT" || exit 1
SWEEP_DIR="${SCRIPT_DIR#"$REPO_ROOT"/}"
ROUND="$(basename "$SCRIPT_DIR")"   # r8
NODE=135
PYTHON="${PYTHON:-python}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # diagnose.sh's GPU = nvidia-smi's index

OUT_ROOT="output/b300_$ROUND"
STATUS_DIR="$OUT_ROOT/run_logs/$NODE/status"
DIAG_DIR="$OUT_ROOT/_diag"
LOG="$OUT_ROOT/extend_$NODE.out"
PID_FILE="$OUT_ROOT/extend_$NODE.pid"
CHOICE="$DIAG_DIR/base_choice.txt"   # "<set> <arm,arm,...>"
BRIEF_LINES="$DIAG_DIR/base_brief.txt"   # the diag lines brief.py prints
BASE=hybrid_base
BASE_INFER="$SWEEP_DIR/config_infer_$BASE.txt"
PLACEMENT=hybrid

# The arms (make_sweep.py EXTRA_ARMS), six per set: six fit until Sunday night.
ENCDEC="ld64,ld16,ch8,ld64_ch8,pms0.02,ar3000"
PRIOR="pm6,ph256,fb8,fmbig_pm6,pms0.1,ld64"
BOTH="ld64,ch8,ar3000,pm6,ph256,fmbig_pm6"

pid_alive() { local p; p="$(tr -dc '0-9' < "$1" 2>/dev/null)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

# ================================================================ the terminal part
if [ "${EXTEND_PHASE:-}" != "run" ]; then
    if [ -f "$PID_FILE" ] && pid_alive "$PID_FILE"; then
        echo "extend_$NODE is already running (pid $(cat "$PID_FILE")): tail -f $LOG"
        echo "If it is not, delete $PID_FILE and start again."
        exit 0
    fi
    [ -f "$BASE_INFER" ] || { echo "no $BASE_INFER: git pull first"; exit 2; }
    "$PYTHON" -c "import cae_suite" 2>/dev/null \
        || { echo "launcher python '$PYTHON' cannot import cae_suite. Set PYTHON=/path/to/python."; exit 2; }
    for arms in "$ENCDEC" "$PRIOR" "$BOTH"; do   # every arm known to make_sweep.py (writes nothing)
        "$PYTHON" -B "$SWEEP_DIR/make_sweep.py" --plan --extra "$arms" > /dev/null \
            || { echo "make_sweep.py does not take --extra $arms"; exit 2; }
    done
    [ -f "$OUT_ROOT/run_$NODE.pid" ] && pid_alive "$OUT_ROOT/run_$NODE.pid" \
        || echo "note: the round (run_$NODE.sh) is not running here; this waits for its $BASE all the same"
    mkdir -p "$DIAG_DIR"
    SETSID=""
    command -v setsid >/dev/null 2>&1 && SETSID="setsid"
    EXTEND_PHASE=run nohup $SETSID bash "$SCRIPT" >> "$LOG" 2>&1 < /dev/null &
    echo "started in the background (pid $!); the terminal can close."
    echo "  progress : tail -f $LOG"
    echo "  result   : python $SWEEP_DIR/brief.py   (also the end of $LOG)"
    exit 0
fi

# ================================================================ the background driver
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT
log() { echo "[$(date '+%m-%d %H:%M')] $*"; }
stopped() { [ -e "$OUT_ROOT/stop_$NODE" ]; }
finish() {   # the block, whatever happened
    echo
    "$PYTHON" -B "$SWEEP_DIR/brief.py"
    exit "${1:-0}"
}
echo "=================================================================="
log "extend_$NODE started (host $(hostname))"

# ---------------------------------------------------------------- 1. the base
last=""
while :; do
    stopped && { log "$OUT_ROOT/stop_$NODE exists: nothing started"; finish 0; }
    st="$(cat "$STATUS_DIR/$BASE.state" 2>/dev/null)"
    case "$st" in done|fail*) break ;; esac
    msg="$BASE is '${st:-not started}'"
    pid_alive "$OUT_ROOT/run_$NODE.pid" || msg="$msg; the round is not running (bash $SWEEP_DIR/run_$NODE.sh)"
    [ "$msg" != "$last" ] && log "waits: $msg"
    last="$msg"
    sleep 300
done
log "$BASE: $st"

# ---------------------------------------------------------------- 2-3. diagnosis and choice
mkdir -p "$DIAG_DIR"
if [ -s "$CHOICE" ]; then
    log "choice kept from an earlier start ($CHOICE; delete it to diagnose again)"
else
    io="$(sed -n 's/^inference_output_dir[[:space:]]\{1,\}\([^[:space:]%#]*\).*/\1/p' "$BASE_INFER" | head -1)"
    JSON="$(dirname "${io#../../}")/diag/posterior_vs_prior_groups_${ROUND}_$BASE.json"
    STAMP="$DIAG_DIR/.stamp_$$"
    : > "$STAMP"
    GPUS="$(sed -n 's/^GPUS="\${GPUS:-\([^}]*\)}".*/\1/p' "$SWEEP_DIR/run_$NODE.sh")"
    GPU=""; best=-1
    while IFS=', ' read -r i f; do
        case " $GPUS " in *" $i "*) [ "${f:-0}" -gt "$best" ] 2>/dev/null && { best="$f"; GPU="$i"; } ;; esac
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null)
    GPU="${GPU:-${GPUS%% *}}"
    log "posterior vs. prior diagnosis of $BASE on GPU $GPU (${best} MiB free)"
    GPU="$GPU" PYTHON="$PYTHON" bash "$SWEEP_DIR/diagnose.sh" "$BASE_INFER"
    [ -f "$JSON" ] && [ "$JSON" -nt "$STAMP" ] || { log "no diagnosis ($JSON): the BOTH set"; JSON=""; }
    rm -f "$STAMP"
    "$PYTHON" -B - "$JSON" "$ENCDEC" "$PRIOR" "$BOTH" "$CHOICE" "$BRIEF_LINES" <<'EOF' \
        || { log "choosing the extras failed"; finish 1; }
import json, sys
path, sets, choice, brief = sys.argv[1], dict(zip(("encdec", "prior", "both"), sys.argv[2:5])), sys.argv[5], sys.argv[6]
x = lambda v: f"{100 * v:.0f}"
try:
    d = json.load(open(path, encoding="utf-8")) if path else None
except (OSError, ValueError) as e:
    d, path = None, f"{path}: {e}"
if d is None:
    pick, lines = "both", [f"diag none -> both"]
else:
    sc = d["scores"]
    f, mu, pz, pr = (sc[e]["w2"]["p50"] for e in ("floor", "posterior_mean", "posterior_sample", "prior"))
    total = pr - f   # posterior_vs_prior_groups.verdict, same order and thresholds
    if total <= 0:
        pick = "both"
    elif pr < pz - 0.02:
        pick = "encdec"
    else:
        share = (pz - f) / total
        pick = "prior" if share < 1 / 3 else "encdec" if share > 2 / 3 else "both"
    sd = "/".join(f"{sc[e]['sd_ratio_geomean']['p50']:.2f}" for e in ("posterior_mean", "posterior_sample", "prior"))
    med = d["per_group_w2_median"]
    lines = [f"diag w2 floor/mu/post/prior {x(f)}/{x(mu)}/{x(pz)}/{x(pr)} sd mu/post/prior {sd} -> {pick}",
             "diag geom " + " ".join(f"{g}:{x(m['floor'])}/{x(m['posterior_mean'])}/{x(m['posterior_sample'])}/"
                                     f"{x(m['prior'])}" for g, m in med.items())]
open(choice, "w", encoding="utf-8").write(f"{pick} {sets[pick]}\n")
open(brief, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
EOF
fi
read -r SET ARMS < "$CHOICE"
log "extras: $SET ($ARMS)"

# ---------------------------------------------------------------- 3b. jobs.txt
"$PYTHON" -B "$SWEEP_DIR/make_sweep.py" --extra "$ARMS" || { log "make_sweep.py --extra $ARMS failed"; finish 1; }
JOBS_B=""
for a in ${ARMS//,/ }; do
    j="${PLACEMENT}_$a"
    [ "$(cat "$STATUS_DIR/$j.state" 2>/dev/null)" = "done" ] && { log "$j already done"; continue; }
    JOBS_B="$JOBS_B $j"
done
JOBS_B="${JOBS_B# }"
[ -n "$JOBS_B" ] || { log "every extra is done"; finish 0; }
stopped && { log "$OUT_ROOT/stop_$NODE exists: the extras do not start"; finish 0; }

# ---------------------------------------------------------------- 4. the second run
log "second run: LANE=b WAIT_FREE=1 JOBS=\"$JOBS_B\" (its jobs: $OUT_ROOT/run_${NODE}_b.out)"
LANE=b WAIT_FREE=1 DETACH=0 JOBS="$JOBS_B" PYTHON="$PYTHON" bash "$SWEEP_DIR/run_$NODE.sh"
rc=$?
log "second run ended (rc=$rc)"

# ---------------------------------------------------------------- 5. the block
finish "$rc"
