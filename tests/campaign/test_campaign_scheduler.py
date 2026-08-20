"""Unit tests for the benchmarks_all campaign scheduler core -- no real
subprocesses, GPUs, or sleeping: FakeGpuProbe/FakeLauncher stand in for
hardware, and an injected clock/log_mtime_fn stand in for wall-clock time so
STALL_TIMEOUT_MIN-scale waits don't require real sleeping.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from campaign.gpu_probe import FakeGpuProbe
from campaign.manifest import ManifestWriter
from campaign.process import Job
from campaign.roster import RosterError, load_roster, select_labels
from campaign.runtime_config import RuntimeConfigError, write_runtime_config
from campaign.scheduler import Scheduler


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.killed: list[str] = []
        self.exit_codes: dict[str, int] = {}

    def start(self, job, python_bin, cwd) -> None:
        job.popen = object()
        job.pgid = id(job)
        job.outcome = "RUNNING"
        self.started.append(job.label)

    def poll(self, job):
        return self.exit_codes.get(job.label)

    def kill_process_tree(self, job) -> None:
        self.killed.append(job.label)

    def pids_owned_by(self, job, candidate_pids):
        return set()


def _stub_config(path: Path) -> Path:
    path.write_text("mode train\ngpu_ids 0\nkey value\n", encoding="utf-8")
    return path


def _make_job(tmp_path: Path, label: str, light: bool = False) -> Job:
    cfg = _stub_config(tmp_path / f"{label}_cfg.txt")
    return Job(label=label, ex_slot="ex4", light=light, mode="train",
               canonical_config=cfg, log_path=tmp_path / f"{label}.log")


# (free_mb, total_mb) fixtures: well under vs. well over a 50%-util target.
PLENTY_FREE = (90_000, 100_000)   # 10% used
MOSTLY_USED = (5_000, 100_000)    # 95% used


def _scheduler(tmp_path, jobs, gpu_list, **overrides):
    probe = overrides.pop("probe", None) or FakeGpuProbe()
    launcher = overrides.pop("launcher", None) or FakeLauncher()
    clock = overrides.pop("clock", lambda: 0.0)
    wall_clock = overrides.pop("wall_clock", lambda: 0.0)
    log_mtime_fn = overrides.pop("log_mtime_fn", lambda p: 0.0)
    kwargs = dict(
        vram_target_util_percent=50, max_concurrency_per_gpu=1,
        admit_warmup_sec=0, stall_timeout_sec=1800, poll_interval_sec=0,
        blocked_abort_sec=1800,
    )
    kwargs.update(overrides)
    manifest = ManifestWriter(tmp_path / "manifest_root")
    sched = Scheduler(
        jobs=jobs, gpu_list=gpu_list, python_bin="python", cwd=tmp_path,
        runtime_config_root=tmp_path / "runtime_configs",
        probe=probe, launcher=launcher, manifest=manifest,
        clock=clock, wall_clock=wall_clock, log_mtime_fn=log_mtime_fn,
        on_event=lambda msg: None, sleep=lambda s: None,
        **kwargs,
    )
    return sched, launcher, probe


# -- admission --

def test_max_concurrency_per_gpu_caps_admission(tmp_path):
    jobs = [_make_job(tmp_path, f"j{i}") for i in range(3)]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}, {"0": PLENTY_FREE}]
    sched, launcher, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, max_concurrency_per_gpu=1)

    sched.admit_tick()
    assert launcher.started == ["j0"]

    sched.admit_tick()  # GPU already at cap -- no 2nd admission
    assert launcher.started == ["j0"]


def test_vram_target_util_blocks_admission_when_over_target(tmp_path):
    jobs = [_make_job(tmp_path, "j0")]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": MOSTLY_USED}]  # 95% used, target is 50%
    sched, launcher, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, vram_target_util_percent=50)

    sched.admit_tick()
    assert launcher.started == []
    assert len(sched.pending) == 1


def test_admit_warmup_cooldown_blocks_reconsideration_until_elapsed(tmp_path):
    jobs = [_make_job(tmp_path, "j0"), _make_job(tmp_path, "j1")]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}, {"0": PLENTY_FREE}, {"0": PLENTY_FREE}]
    now = {"t": 0.0}
    sched, launcher, _ = _scheduler(
        tmp_path, jobs, ["0"], probe=probe, max_concurrency_per_gpu=5,
        admit_warmup_sec=90, clock=lambda: now["t"],
    )

    sched.admit_tick()
    assert launcher.started == ["j0"]

    now["t"] = 10.0  # within the 90s cooldown
    sched.admit_tick()
    assert launcher.started == ["j0"]

    now["t"] = 100.0  # cooldown elapsed
    sched.admit_tick()
    assert launcher.started == ["j0", "j1"]


def test_probe_unavailable_falls_back_to_one_job_per_gpu(tmp_path):
    jobs = [_make_job(tmp_path, "j0"), _make_job(tmp_path, "j1")]
    probe = FakeGpuProbe()  # no scripted responses -> always returns {gpu: None}
    sched, launcher, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, max_concurrency_per_gpu=5)

    sched.admit_tick()
    assert launcher.started == ["j0"]

    sched.admit_tick()  # probe still down, GPU already has a running job -> no stacking
    assert launcher.started == ["j0"]


def test_heavy_jobs_are_queued_before_light_jobs(tmp_path):
    jobs = [
        _make_job(tmp_path, "light_a", light=True),
        _make_job(tmp_path, "heavy_a", light=False),
        _make_job(tmp_path, "light_b", light=True),
        _make_job(tmp_path, "heavy_b", light=False),
    ]
    sched, _, _ = _scheduler(tmp_path, jobs, ["0"])
    assert [j.label for j in sched.pending] == ["heavy_a", "heavy_b", "light_a", "light_b"]


# -- blocked-campaign guard --

def test_blocked_campaign_aborts_instead_of_spinning_forever(tmp_path):
    """Every GPU permanently above the VRAM target (memory the campaign
    doesn't own) must not turn into a silent infinite wait."""
    jobs = [_make_job(tmp_path, "j0")]
    probe = FakeGpuProbe()
    now = {"t": 0.0}
    sched, launcher, _ = _scheduler(
        tmp_path, jobs, ["0"], probe=probe, clock=lambda: now["t"],
        vram_target_util_percent=50, blocked_abort_sec=1800,
    )

    probe.vram_responses = [{"0": MOSTLY_USED}]
    sched.tick()
    assert launcher.started == []
    assert sched.aborted_reason is None  # not yet -- still within the grace window

    now["t"] = 2000.0
    probe.vram_responses = [{"0": MOSTLY_USED}]
    sched.tick()
    assert sched.aborted_reason is not None
    assert "95% used" in sched.aborted_reason  # reports what it actually measured
    assert sched.run() == 1


def test_blocked_guard_resets_once_a_job_is_running(tmp_path):
    jobs = [_make_job(tmp_path, "j0"), _make_job(tmp_path, "j1")]
    probe = FakeGpuProbe()
    now = {"t": 0.0}
    sched, launcher, _ = _scheduler(
        tmp_path, jobs, ["0"], probe=probe, clock=lambda: now["t"], max_concurrency_per_gpu=1,
    )

    probe.vram_responses = [{"0": MOSTLY_USED}]
    sched.tick()
    assert sched._blocked_since is not None

    # A slot frees up and a job gets admitted -> no longer blocked.
    now["t"] = 100.0
    probe.vram_responses = [{"0": PLENTY_FREE}, {"0": PLENTY_FREE}]
    sched.tick()
    assert launcher.started == ["j0"]
    assert sched._blocked_since is None

    now["t"] = 5000.0  # long past the abort window, but a job is running
    probe.vram_responses = [{"0": MOSTLY_USED}]
    sched.tick()
    assert sched.aborted_reason is None


def test_keyboard_interrupt_kills_running_jobs_instead_of_orphaning_them(tmp_path):
    """Jobs run in their own process group, so Ctrl-C does not reach them --
    the scheduler must tear them down explicitly or they keep holding VRAM."""
    jobs = [_make_job(tmp_path, "j0")]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}]

    class InterruptingLauncher(FakeLauncher):
        def poll(self, job):
            raise KeyboardInterrupt

    launcher = InterruptingLauncher()
    sched, _, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, launcher=launcher)
    sched.admit_tick()
    assert launcher.started == ["j0"]

    assert sched.run() == 130
    assert launcher.killed == ["j0"]
    assert jobs[0].outcome == "INTERRUPTED"


# -- stall detection --

def test_stall_tick_kills_and_frees_slot_on_log_staleness(tmp_path):
    jobs = [_make_job(tmp_path, "j0")]
    jobs[0].log_path.write_text("", encoding="utf-8")
    now = {"t": 0.0}
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}]
    sched, launcher, _ = _scheduler(
        tmp_path, jobs, ["0"], probe=probe, wall_clock=lambda: now["t"],
        log_mtime_fn=lambda p: 0.0, stall_timeout_sec=1800,
    )
    sched.admit_tick()
    assert launcher.started == ["j0"]

    now["t"] = 1000.0  # under the 1800s threshold
    sched.stall_tick()
    assert launcher.killed == []
    assert jobs[0] in sched.gpus["0"].running

    now["t"] = 2000.0  # over threshold
    sched.stall_tick()
    assert launcher.killed == ["j0"]
    assert jobs[0].outcome == "STALL"
    assert sched.gpus["0"].running == []


# -- reap / outcomes --

def test_reap_tick_maps_exit_codes_to_outcomes(tmp_path):
    jobs = [_make_job(tmp_path, "ok"), _make_job(tmp_path, "bad")]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}, {"0": PLENTY_FREE}]
    sched, launcher, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, max_concurrency_per_gpu=5)
    sched.admit_tick()
    sched.admit_tick()
    assert launcher.started == ["ok", "bad"]

    launcher.exit_codes["ok"] = 0
    launcher.exit_codes["bad"] = 1
    sched.reap_tick()

    assert jobs[0].outcome == "SUCCESS"
    assert jobs[1].outcome == "FAILED"
    assert sched.gpus["0"].running == []


# -- orphan sweep --

def test_orphan_sweep_warns_when_owned_pid_survives(tmp_path):
    jobs = [_make_job(tmp_path, "j0")]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}]
    probe.compute_app_responses["0"] = [(4242, 100)]

    class OwningLauncher(FakeLauncher):
        def pids_owned_by(self, job, candidate_pids):
            return set(candidate_pids)  # everything reported on this GPU is "ours"

    launcher = OwningLauncher()
    sched, _, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, launcher=launcher, max_concurrency_per_gpu=5)
    sched.admit_tick()
    launcher.exit_codes["j0"] = 0
    sched.reap_tick()

    assert jobs[0].orphan_check == "WARN"
    assert "j0" in launcher.killed  # extra sweep-kill attempted


def test_orphan_sweep_clean_when_no_leftover_pids(tmp_path):
    jobs = [_make_job(tmp_path, "j0")]
    probe = FakeGpuProbe()
    probe.vram_responses = [{"0": PLENTY_FREE}]
    sched, launcher, _ = _scheduler(tmp_path, jobs, ["0"], probe=probe, max_concurrency_per_gpu=5)
    sched.admit_tick()
    launcher.exit_codes["j0"] = 0
    sched.reap_tick()

    assert jobs[0].orphan_check == "CLEAN"
    assert launcher.killed == []


# -- roster --

def test_select_labels_unknown_label_raises(tmp_path):
    roster = tmp_path / "roster.tsv"
    roster.write_text("label\ttrain_config\tex_slot\tlight\n"
                       "a\tconfigs/a.txt\tex4\t0\n", encoding="utf-8")
    entries = load_roster(roster)
    with pytest.raises(RosterError):
        select_labels(entries, "nonexistent_label", roster)


def test_select_labels_no_filter_returns_all_in_order(tmp_path):
    roster = tmp_path / "roster.tsv"
    roster.write_text("label\ttrain_config\tex_slot\tlight\n"
                       "a\tconfigs/a.txt\tex4\t0\n"
                       "b\tconfigs/b.txt\tex4\t1\n", encoding="utf-8")
    entries = load_roster(roster)
    selected = select_labels(entries, None, roster)
    assert [e.label for e in selected] == ["a", "b"]
    assert selected[1].light is True


# -- runtime config generation (parity with the old sed-based scripts) --

def test_write_runtime_config_train_mode_patches_gpu_ids_only(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("mode train\ngpu_ids 3\nother_key value  % a comment\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    write_runtime_config("train", src, "5", out)
    text = out.read_text(encoding="utf-8")
    assert "gpu_ids 5" in text
    assert "mode train" in text  # untouched in train mode
    assert "other_key value  % a comment" in text


def test_write_runtime_config_infer_mode_patches_mode_and_gpu(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("mode train\ngpu_ids 3\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    write_runtime_config("infer", src, "2", out)
    text = out.read_text(encoding="utf-8")
    assert "mode inference" in text
    assert "gpu_ids 2" in text


def test_write_runtime_config_infer_mode_requires_train_line(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("mode inference\ngpu_ids 3\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with pytest.raises(RuntimeConfigError):
        write_runtime_config("infer", src, "2", out)


def test_write_runtime_config_train_mode_requires_gpu_ids_line(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("mode train\n", encoding="utf-8")
    out = tmp_path / "out.txt"
    with pytest.raises(RuntimeConfigError):
        write_runtime_config("train", src, "2", out)
