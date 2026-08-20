"""Campaign records: an append-only JSONL event log plus a manifest TSV that
is fully rewritten after every job finalizes (cheap at this roster's scale,
~25 jobs, and unlike the old campaign.tsv -- built once up front, never
updated -- always reflects the true dynamic GPU assignment and outcome).

score_rollouts.py reads roster.tsv directly and never touches this manifest,
so its shape is free to evolve independently.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .process import Job

_COLUMNS = [
    "label", "ex_slot", "light", "mode", "canonical_config",
    "admission_runtime_config", "log", "gpu", "admitted_at", "finished_at",
    "duration_s", "outcome", "exit_code", "orphan_check",
]


class ManifestWriter:
    def __init__(self, run_root: Path) -> None:
        run_root.mkdir(parents=True, exist_ok=True)
        self.run_root = run_root
        self.events_path = run_root / "campaign_events.jsonl"
        self.manifest_path = run_root / "campaign_manifest.tsv"
        self._jobs: list[Job] = []

    def register(self, jobs: list[Job]) -> None:
        self._jobs = jobs
        self._write_manifest()

    def event(self, kind: str, job: Job, **extra: object) -> None:
        record: dict[str, object] = {"ts": time.time(), "event": kind, "label": job.label, "gpu": job.gpu}
        record.update(extra)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def job_finalized(self, job: Job) -> None:
        self._write_manifest()

    def _write_manifest(self) -> None:
        lines = ["\t".join(_COLUMNS)]
        for job in self._jobs:
            duration = (
                f"{job.finished_at - job.admitted_at:.1f}"
                if job.finished_at is not None and job.admitted_at is not None
                else ""
            )
            row = [
                job.label, job.ex_slot, "1" if job.light else "0", job.mode,
                str(job.canonical_config), str(job.admission_runtime_config or ""),
                str(job.log_path), job.gpu or "",
                f"{job.admitted_at:.3f}" if job.admitted_at is not None else "",
                f"{job.finished_at:.3f}" if job.finished_at is not None else "",
                duration, job.outcome,
                "" if job.exit_code is None else str(job.exit_code),
                job.orphan_check or "",
            ]
            lines.append("\t".join(row))
        self.manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
