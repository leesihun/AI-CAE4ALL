#!/usr/bin/env python3
"""Dynamic shared-queue campaign runner for configs/campaigns/benchmarks_all/roster.tsv.

Replaces the old static per-GPU "lane" round-robin in train_all.sh/infer_all.sh
(jobs pre-assigned to a GPU by `i % num_gpus`, run strictly sequentially
within that lane) with a single pending queue serving every selected GPU.
Two problems that design had:

1. A hung job blocked its whole lane forever. The bash `lane_worker` awaited
   each job's process synchronously with no timeout, and the launcher
   (cae_suite/launcher.py) has no timeout on Popen.wait() either -- if a job
   deadlocks (e.g. one rank of a distributed job stuck in an NCCL/gloo
   collective after a sibling rank died), every later job queued on that same
   GPU never even started. This scheduler never blocks on wait(): it polls,
   and kills a job's whole process tree if its log file goes stale for
   STALL_TIMEOUT_MIN, freeing that GPU's slot for the next queued job.
2. One job per GPU regardless of size wasted capacity on small jobs (the
   roster's `light` column -- low-VRAM Neural_Operator arms on ex8/ex9,
   observed real peak VRAM ~0.1-0.2GB -- was parsed but never used). This
   scheduler admits jobs onto a GPU based on its *live* free VRAM
   (nvidia-smi-polled, not a static "N light jobs" guess), so multiple light
   jobs pack onto one GPU automatically, and a static round-robin's
   uneven-job-duration idling goes away as a side effect of sharing one queue.

Normally reached through train_all.sh/infer_all.sh, which now just `exec`
this with --mode train/infer and inherit every env var. Can also be invoked
directly (e.g. on Windows without bash):

    python configs/campaigns/benchmarks_all/campaign_runner.py --mode train
    CHECK_ONLY=1 python configs/campaigns/benchmarks_all/campaign_runner.py --mode train --gpus 0

Every existing env var still works (PYTHON, ROSTER, LABELS, GPUS, PREFLIGHT,
PREFLIGHT_FLAGS, CHECK_ONLY, LOG_ROOT, RUN_ID); each also has an equivalent
--flag that takes precedence. New knobs (env var + --flag), defaults tuned
for a multi-hour campaign:

    STALL_TIMEOUT_MIN       (default 60)   no log growth for this long -> kill
    VRAM_TARGET_UTIL_PERCENT (default 50)  admit a new job onto a GPU only
                                            while its current usage is below
                                            this % of total VRAM (a % target
                                            travels across heterogeneous GPU
                                            sizes better than a fixed MB figure)
    MAX_CONCURRENCY_PER_GPU (default 3)    hard concurrency cap per GPU
    ADMIT_WARMUP_SEC        (default 90)   cooldown after an admission before
                                            that GPU is reconsidered (lets a
                                            job's real peak VRAM materialize
                                            before the next free-VRAM read)
    POLL_INTERVAL_SEC       (default 20)   scheduler tick cadence
    BLOCKED_ABORT_MIN       (default 30)   if jobs are queued but NONE can be
                                            admitted (e.g. every GPU is above
                                            the VRAM target because of memory
                                            the campaign doesn't own) and
                                            nothing is running, stop with a
                                            diagnostic after this long rather
                                            than spinning forever; 0 = wait
                                            indefinitely

STALL_TIMEOUT_MIN trades off against how chatty the slowest selected method's
own per-epoch logging is -- a method that prints only once per (very long)
epoch on a big dataset needs a correspondingly generous timeout, or a healthy
job will look stalled. Jobs are launched with PYTHONUNBUFFERED=1 specifically
so this log-mtime signal is trustworthy (buffered stdout has previously
looked identical to a genuine hang).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from campaign.gpu_probe import NvidiaSmiProbe  # noqa: E402
from campaign.manifest import ManifestWriter  # noqa: E402
from campaign.process import Job, SubprocessLauncher  # noqa: E402
from campaign.roster import RosterError, load_roster, select_labels  # noqa: E402
from campaign.runtime_config import RuntimeConfigError, write_runtime_config  # noqa: E402
from campaign.scheduler import Scheduler  # noqa: E402


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True, choices=["train", "infer"])
    p.add_argument("--python", default=_env("PYTHON", "python"))
    p.add_argument("--roster", default=_env("ROSTER", str(SCRIPT_DIR / "roster.tsv")))
    p.add_argument("--labels", default=os.environ.get("LABELS") or None)
    p.add_argument("--gpus", default=_env("GPUS", "0 1 2 3 4 5 6 7"))
    p.add_argument("--preflight", default=_env("PREFLIGHT", "1"))
    p.add_argument("--preflight-flags", default=_env("PREFLIGHT_FLAGS", "--strict"))
    p.add_argument("--check-only", default=_env("CHECK_ONLY", "0"))
    p.add_argument("--log-root", default=os.environ.get("LOG_ROOT"))
    p.add_argument("--run-id", default=_env("RUN_ID", f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"))
    p.add_argument("--stall-timeout-min", type=float, default=float(_env("STALL_TIMEOUT_MIN", "60")))
    p.add_argument("--vram-target-util-percent", type=float, default=float(_env("VRAM_TARGET_UTIL_PERCENT", "50")))
    p.add_argument("--max-concurrency-per-gpu", type=int, default=int(_env("MAX_CONCURRENCY_PER_GPU", "3")))
    p.add_argument("--admit-warmup-sec", type=float, default=float(_env("ADMIT_WARMUP_SEC", "90")))
    p.add_argument("--poll-interval-sec", type=float, default=float(_env("POLL_INTERVAL_SEC", "20")))
    p.add_argument("--blocked-abort-min", type=float, default=float(_env("BLOCKED_ABORT_MIN", "30")))
    return p.parse_args(argv)


def _bool01(value: str, name: str) -> bool:
    if value not in ("0", "1"):
        print(f"ERROR: {name} must be 0 or 1 (got {value!r})", file=sys.stderr)
        sys.exit(2)
    return value == "1"


def _run_preflight(python_bin: str, config_path: Path, flags: list[str], log_path: Path, cwd: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as f:
        completed = subprocess.run(
            [python_bin, "AI_CAE4ALL_main.py", "--config", str(config_path), "--check", *flags],
            cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=False,
        )
    return completed.returncode == 0


def main(argv: list[str]) -> int:
    # Line-buffer even when stdout is redirected to a file (the default is
    # fully block-buffered in that case) -- otherwise progress lines sit
    # unflushed for a long time and an operator tailing the campaign log
    # sees nothing while jobs are, in fact, already running.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    args = parse_args(argv)
    if args.log_root is None:
        args.log_root = f"output/benchmarks_all/{args.mode}_runs"
    preflight = _bool01(args.preflight, "PREFLIGHT")
    check_only = _bool01(args.check_only, "CHECK_ONLY")

    roster_path = Path(args.roster)
    if not roster_path.is_file():
        print(f"ERROR: roster not found: {roster_path}", file=sys.stderr)
        return 2

    gpu_list = args.gpus.split()
    if not gpu_list:
        print("ERROR: GPUS is empty", file=sys.stderr)
        return 2

    entries = load_roster(roster_path)
    try:
        selected = select_labels(entries, args.labels, roster_path)
    except RosterError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if not selected:
        print("ERROR: no roster entries selected", file=sys.stderr)
        return 2
    for entry in selected:
        if not (REPO_ROOT / entry.train_config).is_file():
            print(f"ERROR: config not found for '{entry.label}': {REPO_ROOT / entry.train_config}", file=sys.stderr)
            return 2

    run_root = REPO_ROOT / args.log_root / args.run_id
    runtime_config_root = run_root / "runtime_configs"
    runtime_config_root.mkdir(parents=True, exist_ok=True)

    jobs = [
        Job(
            label=e.label, ex_slot=e.ex_slot, light=e.light, mode=args.mode,
            canonical_config=REPO_ROOT / e.train_config,
            log_path=run_root / f"{args.mode}_{e.label}.log",
        )
        for e in selected
    ]

    print(f"benchmarks_all {args.mode} campaign ({len(jobs)} arms)")
    print(f"  PYTHON       = {args.python}")
    print(f"  ROSTER       = {roster_path}")
    print(f"  GPUS         = {args.gpus}  ({len(gpu_list)} GPU(s), dynamic shared-queue admission)")
    print(f"  PREFLIGHT    = {int(preflight)} ({args.preflight_flags})")
    print(f"  CHECK_ONLY   = {int(check_only)}")
    print(f"  STALL_TIMEOUT_MIN       = {args.stall_timeout_min}")
    print(f"  VRAM_TARGET_UTIL_PERCENT = {args.vram_target_util_percent}")
    print(f"  MAX_CONCURRENCY_PER_GPU = {args.max_concurrency_per_gpu}")
    print(f"  ADMIT_WARMUP_SEC        = {args.admit_warmup_sec}")
    print(f"  RUN_ROOT     = {run_root}")

    if preflight:
        print(f"\nPreflighting all {len(jobs)} runtime configs before launch...")
        preflight_flags = args.preflight_flags.split()
        for i, job in enumerate(jobs):
            gpu = gpu_list[i % len(gpu_list)]
            job.preflight_runtime_config = runtime_config_root / f"{args.mode}_{job.label}.preflight.txt"
            try:
                write_runtime_config(args.mode, job.canonical_config, gpu, job.preflight_runtime_config)
            except RuntimeConfigError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            pf_log = run_root / f"{args.mode}_{job.label}.log.preflight"
            if not _run_preflight(args.python, job.preflight_runtime_config, preflight_flags, pf_log, REPO_ROOT):
                print(f"[{job.label}] PREFLIGHT FAILED -- see {pf_log}", file=sys.stderr)
                return 1
            print(f"[{job.label}] PREFLIGHT PASS")

    if check_only:
        kind = "training" if args.mode == "train" else "inference"
        print(f"\nCHECK_ONLY complete; no {kind} was launched.")
        return 0

    manifest = ManifestWriter(run_root)
    scheduler = Scheduler(
        jobs=jobs, gpu_list=gpu_list, python_bin=args.python, cwd=REPO_ROOT,
        runtime_config_root=runtime_config_root,
        probe=NvidiaSmiProbe(), launcher=SubprocessLauncher(), manifest=manifest,
        vram_target_util_percent=args.vram_target_util_percent,
        max_concurrency_per_gpu=args.max_concurrency_per_gpu,
        admit_warmup_sec=args.admit_warmup_sec,
        stall_timeout_sec=args.stall_timeout_min * 60,
        poll_interval_sec=args.poll_interval_sec,
        blocked_abort_sec=args.blocked_abort_min * 60,
    )

    print()
    started = time.time()
    rc = scheduler.run()
    elapsed = time.time() - started

    print()
    outcomes: dict[str, int] = {}
    for job in jobs:
        outcomes[job.outcome] = outcomes.get(job.outcome, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in sorted(outcomes.items()))
    print(f"benchmarks_all {args.mode} campaign finished in {elapsed:.0f}s (rc={rc}): {summary}")
    print(f"Manifest:        {manifest.manifest_path}")
    print(f"Events:          {manifest.events_path}")
    print(f"Transcripts:     {run_root}/{args.mode}_<label>.log")
    print(f"Runtime configs: {runtime_config_root}/")
    if rc == 0:
        if args.mode == "train":
            print("Next: bash configs/campaigns/benchmarks_all/infer_all.sh")
        else:
            print("Next: python configs/campaigns/benchmarks_all/score_rollouts.py")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
