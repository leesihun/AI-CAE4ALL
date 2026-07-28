"""Extract plot-ready training metrics from Studio-owned job logs.

The parser deliberately reads only persisted runtime evidence. It accepts the
compact epoch lines emitted by the model repositories without assuming one
fixed metric vocabulary, so newly added losses are surfaced automatically.
"""

from __future__ import annotations

import math
import re
from collections import OrderedDict
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EVENT_RE = re.compile(
    r"\b(?P<kind>epoch|iteration|iter|step)\s*(?:[:#=])?\s*[\[(]?\s*(?P<value>\d+)"
    r"(?:\s*/\s*\d+)?\s*[\])]?",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"^\s*\[studio\]\s+Step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*:", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
LABEL_TRIM_RE = re.compile(r"^[\s|,;:\-=\[\](){}]+|[\s|,;:\-=\[\](){}]+$")
KEY_RE = re.compile(r"[^a-z0-9]+")


def _metric_key(label: str) -> str:
    return KEY_RE.sub("_", label.lower()).strip("_")


def _metric_pairs(body: str) -> list[tuple[str, float]]:
    """Split a compact metric sequence without a hard-coded metric list."""

    pairs: list[tuple[str, float]] = []
    cursor = 0
    for match in NUMBER_RE.finditer(body):
        raw_label = LABEL_TRIM_RE.sub("", body[cursor : match.start()]).strip()
        cursor = match.end()
        if not raw_label or not re.search(r"[A-Za-z]", raw_label):
            continue
        raw_label = re.sub(r"\s+", " ", raw_label)
        words = raw_label.split()
        label = " ".join(words[-4:]).strip()
        if not label or len(label) > 64:
            continue
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if math.isfinite(value):
            pairs.append((label, value))
    return pairs


def parse_training_log(text: str) -> dict[str, Any]:
    """Return every metric found on epoch/iteration/step log lines."""

    series: OrderedDict[str, dict[str, Any]] = OrderedDict()
    active_step = 1
    total_steps = 1
    event_kinds: set[str] = set()

    for line_number, raw_line in enumerate(ANSI_RE.sub("", text or "").splitlines(), start=1):
        step_match = STEP_RE.search(raw_line)
        if step_match:
            active_step = int(step_match.group("step"))
            total_steps = max(total_steps, int(step_match.group("total")))
            # This orchestration line may contain digits in a block name such
            # as HDF5; it identifies a stage but is not a metric observation.
            continue
        event_match = EVENT_RE.search(raw_line)
        if not event_match:
            continue
        event_kind = event_match.group("kind").lower()
        if event_kind == "iter":
            event_kind = "iteration"
        event_kinds.add(event_kind)
        event_value = int(event_match.group("value"))
        for label, value in _metric_pairs(raw_line[event_match.end() :]):
            base_key = _metric_key(label)
            if not base_key:
                continue
            scoped_key = f"step_{active_step}__{base_key}" if total_steps > 1 else base_key
            display_label = f"Step {active_step} · {label}" if total_steps > 1 else label
            item = series.setdefault(
                scoped_key,
                {
                    "key": scoped_key,
                    "label": display_label,
                    "event": event_kind,
                    "points": [],
                },
            )
            item["points"].append(
                {"x": event_value, "y": value, "line": line_number, "event": event_kind}
            )

    metrics: list[dict[str, Any]] = []
    for item in series.values():
        values = [point["y"] for point in item["points"]]
        metrics.append(
            {
                **item,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "last": values[-1],
            }
        )
    if not event_kinds:
        x_label = "epoch"
    elif len(event_kinds) == 1:
        x_label = next(iter(event_kinds))
    else:
        x_label = "epoch / step"
    return {
        "x_label": x_label,
        "metrics": metrics,
        "metric_count": len(metrics),
        "point_count": sum(metric["count"] for metric in metrics),
    }


def _job_models(job: dict[str, Any]) -> list[str]:
    models = {
        str(step.get("route", {}).get("model", "")).strip()
        for step in job.get("steps", [])
        if isinstance(step, dict) and isinstance(step.get("route"), dict)
    }
    return sorted(model for model in models if model)


def _job_lineage(job: dict[str, Any]) -> list[dict[str, str]]:
    lineage: list[dict[str, str]] = []
    for step in job.get("steps", []):
        if not isinstance(step, dict):
            continue
        route = step.get("route") if isinstance(step.get("route"), dict) else {}
        lineage.append(
            {
                "node_id": str(step.get("node_id", "")),
                "node_type": str(step.get("node_type", "")),
                "model_id": str(route.get("model", "")),
                "mode": str(route.get("mode", "")),
                "label": str(step.get("label", "")),
            }
        )
    return lineage


def training_metrics_catalog(
    state: Any,
    job_id: str = "",
    node_id: str = "",
    model_id: str = "",
) -> dict[str, Any]:
    """Parse one or every Studio job, returning only jobs with metric series."""

    summaries = state.list_jobs()
    if job_id:
        summaries = [summary for summary in summaries if summary.get("id") == job_id]
        if not summaries:
            raise KeyError(job_id)

    items: list[dict[str, Any]] = []
    for summary in summaries:
        job = state.get_job(str(summary["id"]))
        lineage = _job_lineage(job)
        if node_id and not any(step["node_id"] == node_id for step in lineage):
            continue
        if model_id and not any(step["model_id"] == model_id for step in lineage):
            continue
        full_log = state.read_job_log(str(job["id"])) if hasattr(state, "read_job_log") else str(job.get("log", ""))
        parsed = parse_training_log(full_log)
        if not parsed["metrics"]:
            continue
        items.append(
            {
                "job_id": job["id"],
                "label": job.get("label", job["id"]),
                "status": job.get("status", "unknown"),
                "created_at": job.get("created_at"),
                "finished_at": job.get("finished_at"),
                "current_step": job.get("current_step", 0),
                "total_steps": job.get("total_steps", 0),
                "models": _job_models(job),
                "node_ids": [step["node_id"] for step in lineage if step["node_id"]],
                "lineage": lineage,
                "target_node_id": job.get("target_node_id", ""),
                "log_path": job.get("log_path", ""),
                **parsed,
            }
        )
    return {"items": items, "count": len(items), "source": "Studio job logs"}
