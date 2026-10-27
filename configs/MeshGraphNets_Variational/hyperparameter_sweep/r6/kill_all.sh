#!/usr/bin/env bash
# Stop every sweep job of this repository on this node: the queue runners of
# every round (output/b300_*/run_*.pid) with everything they started, plus any
# python of yours still running from inside the repository (a trainer, an
# inference, a cache builder or a DataLoader worker whose runner already died).
#
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r6/kill_all.sh -n   # list only
#   bash configs/MeshGraphNets_Variational/hyperparameter_sweep/r6/kill_all.sh      # kill
#
# Processes are signalled by PID only, never by name pattern. Order: SIGSTOP all
# (no runner can start a further job meanwhile), SIGTERM all, SIGCONT, wait up to
# GRACE seconds (default 20), then SIGKILL what is left; then it scans again, in
# case a runner started a job between the scan and the SIGSTOP. Afterwards it
# removes the pid files and the hierarchy-cache .lock / .tmp.<pid> files a
# killed builder leaves (a .lock only counts as stale after 6 h, so every later
# job would wait on it).
#
# A killed training continues from its last <checkpoint>.resume when it is run
# again with resume_training true (run_node.sh appends it); others start over.
set -u

DRY=0
[ "${1:-}" = "-n" ] && DRY=1
GRACE="${GRACE:-20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../../.." && pwd -P)"
ME="$(id -u)"

# This shell and its ancestors (the ssh session, tmux, ...) are never touched.
declare -A SAFE=()
p=$$
while [ -n "$p" ] && [ "$p" != "0" ]; do
    SAFE[$p]=1
    p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
done

declare -A KIDS=() OWNER=() COMM=() KILL=() WHY=()
add_tree() {   # $1 pid, $2 reason
    local pid="$1" c
    [ -n "${SAFE[$pid]:-}" ] && return
    [ "${OWNER[$pid]:-}" = "$ME" ] || return
    [ -n "${KILL[$pid]:-}" ] && return
    KILL[$pid]=1; WHY[$pid]="$2"
    for c in ${KIDS[$pid]:-}; do add_tree "$c" "$2"; done
}

scan() {   # fills KILL/WHY from one ps snapshot
    KIDS=(); OWNER=(); COMM=(); KILL=(); WHY=()
    local pid ppid uid comm f cwd
    while read -r pid ppid uid comm; do
        KIDS[$ppid]="${KIDS[$ppid]:-} $pid"
        OWNER[$pid]=$uid
        COMM[$pid]=$comm
    done < <(ps -eo pid=,ppid=,uid=,comm=)
    # 1. queue runners of every round, with their whole process tree
    for f in "$REPO"/output/b300_*/run_*.pid; do
        [ -f "$f" ] || continue
        pid="$(tr -dc '0-9' < "$f")"
        [ -n "$pid" ] && [ -n "${OWNER[$pid]:-}" ] && add_tree "$pid" "runner ${f#"$REPO"/}"
    done
    # 2. any python of yours whose working directory is inside the repository
    #    (the native trainers run from methods/<Name>/, the launchers from the root)
    for pid in "${!COMM[@]}"; do
        [ "${OWNER[$pid]}" = "$ME" ] || continue
        case "${COMM[$pid]}" in python*) ;; *) continue ;; esac
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null)" || continue
        case "$cwd/" in "$REPO"/*) add_tree "$pid" "python in ${cwd#"$REPO"/}" ;; esac
    done
}

alive() { local p; for p in "$@"; do kill -0 "$p" 2>/dev/null && echo "$p"; done; }

list() {
    local pid
    for pid in $(printf '%s\n' "${!KILL[@]}" | sort -n); do
        printf '  %7s  %-16s %s\n' "$pid" "${COMM[$pid]}" "${WHY[$pid]}"
    done
}

leftovers() {   # cache build lock/tmp files: sweep caches and the default (next to the dataset)
    find "$REPO"/output/b300_*/mscache "$REPO"/dataset -type f \
         \( -name '*.mscache.*.h5.lock' -o -name '*.mscache.*.h5.tmp.*' \) 2>/dev/null
}

scan
if [ "${#KILL[@]}" -eq 0 ]; then
    echo "no sweep processes of uid $ME under $REPO"
else
    echo "${#KILL[@]} processes:"
    list
fi
LOCKS="$(leftovers)"
[ -n "$LOCKS" ] && echo "cache build leftovers:" && echo "$LOCKS" | sed "s#^$REPO/#  #"

if [ "$DRY" = "1" ]; then
    echo "(-n: nothing was signalled or removed)"
    exit 0
fi

total=0
for pass in 1 2 3; do
    [ "$pass" -gt 1 ] && { scan; [ "${#KILL[@]}" -eq 0 ] && break; echo "pass $pass: ${#KILL[@]} more:"; list; }
    [ "${#KILL[@]}" -eq 0 ] && break
    PIDS=("${!KILL[@]}")
    total=$((total + ${#PIDS[@]}))
    kill -STOP "${PIDS[@]}" 2>/dev/null
    kill -TERM "${PIDS[@]}" 2>/dev/null
    kill -CONT "${PIDS[@]}" 2>/dev/null
    for ((i = 0; i < GRACE; i++)); do
        [ -z "$(alive "${PIDS[@]}")" ] && break
        sleep 1
    done
    left="$(alive "${PIDS[@]}")"
    if [ -n "$left" ]; then
        echo "SIGKILL after ${GRACE}s:" $left
        kill -KILL $left 2>/dev/null
        sleep 2
        left="$(alive $left)"
        [ -n "$left" ] && echo "STILL ALIVE (uninterruptible?):" $left
    fi
done

# Pid files whose process is gone, then the cache leftovers (only now: a live
# builder could still have been writing them).
for f in "$REPO"/output/b300_*/run_*.pid; do
    [ -f "$f" ] || continue
    pid="$(tr -dc '0-9' < "$f")"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f -- "${f:?}"; fi
done
n_locks=0
while IFS= read -r f; do
    [ -n "$f" ] || continue
    rm -f -- "${f:?}" && n_locks=$((n_locks + 1))
done < <(leftovers)

echo "done: $total processes signalled, $n_locks cache build leftovers removed"
command -v nvidia-smi >/dev/null 2>&1 \
    && nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
