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
# A leading `[tag]` names the stage that printed the line (`[AE] Epoch 3/100`,
# `[model_split] epoch 3/100`). The tag must start with a letter, so a leading
# timestamp or counter is never mistaken for one.
EVENT_RE = re.compile(
    r"^\s*(?:\[(?P<tag>[A-Za-z][A-Za-z0-9_\- ]{0,31})\]\s*)?"
    r"[\[(]?(?P<kind>epoch|iteration|iter)\s*(?:[:#=])?\s*[\[(]?\s*(?P<value>\d+)"
    r"(?:\s*/\s*\d+)?\s*[\])]?",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"^\s*\[studio\]\s+Step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*:", re.IGNORECASE)
# The two-stage trainers (SDFFlow VAE->FM, SimulGen-VAE VAE->LC) run both stages
# in ONE process and restart the epoch counter; this header names the next one.
STAGE_HEADER_RE = re.compile(
    r"^\s*\[(?:pipeline|stage|phase)\s+\d+\s*/\s*\d+\]\s*(?:training|train|reusing)?\s*"
    r"(?P<name>[A-Za-z][A-Za-z0-9_\-]{0,23})",
    re.IGNORECASE,
)
# `f"{nan:.2e}"` prints `nan`: a diverged loss is still a value token, and it
# has to end the label before it the way a number does, or `TrainOpt: nan
# Valid: 1.2e-02` files the Valid value under a label named `TrainOpt: nan Valid`.
NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r"|(?<![A-Za-z0-9_])[-+]?(?i:nan|inf(?:inity)?)(?![A-Za-z0-9_])"
)
# Under DDP every rank draws a tqdm bar into the one pipe the Studio reads, so a
# rank>0 frame can land in front of rank 0's epoch print:
# `  0%|          | 0/40 [00:00<?, ?it/s]Epoch 1/3 TrainOpt: ...`.
PROGRESS_PREFIX_RE = re.compile(r"^.*?\d+%\|[^|]*\|[^\[\]]*\[[^\]]*\]")
LABEL_TRIM_RE = re.compile(r"^[\s|,;:\-=\[\](){}]+|[\s|,;:\-=\[\](){}]+$")
KEY_RE = re.compile(r"[^a-z0-9]+")
UNIT_PREFIX_RE = re.compile(
    r"^(?:%|ns|us|µs|ms|s|min|h|bytes?|kb|mb|gb|tb|kg|g|mm|cm|m|pa|kpa|mpa|gpa|j|w)\b\s*",
    re.IGNORECASE,
)
# `| Train recon=.. kl=.. | Valid recon=.. kl=..` names the split once per
# group; the bare `kl` labels inherit it so the two values stay two series.
GROUP_WORDS = frozenset({"train", "training", "valid", "validation", "val", "test", "eval", "evaluation"})
# Tags that mark how a line was produced rather than which stage printed it:
# `[model_split] epoch` is the whole run, so it never names a stage.
NON_STAGE_TAGS = frozenset({"model_split"})


def _metric_key(label: str) -> str:
    return KEY_RE.sub("_", label.lower()).strip("_")


def _metric_pairs(body: str) -> list[tuple[str, float]]:
    """Split a compact metric sequence without a hard-coded metric list."""

    pairs: list[tuple[str, float]] = []
    cursor = 0
    group = ""
    for match in NUMBER_RE.finditer(body):
        chunk = body[cursor : match.start()]
        cursor = match.end()
        # A label never spans a `|`/`;` separator, and a separator ends the
        # split group the previous labels belonged to.
        separator = max(chunk.rfind("|"), chunk.rfind(";"))
        if separator >= 0:
            chunk = chunk[separator + 1 :]
            group = ""
        raw_label = LABEL_TRIM_RE.sub("", chunk).strip()
        # A unit suffix belongs to the previous value, not the next label:
        # `VRAM peak=0.11GB reserved=0.12GB` must never create `GB reserved`.
        raw_label = UNIT_PREFIX_RE.sub("", raw_label)
        if not raw_label or not re.search(r"[A-Za-z]", raw_label):
            continue
        raw_label = re.sub(r"\s+", " ", raw_label)
        words = raw_label.split()[-4:]
        first = words[0].lower()
        if first in GROUP_WORDS:
            # `Train recon=` opens a group; a bare `Train:` is a value itself.
            group = words[0] if len(words) > 1 else ""
        elif len(words) > 1:
            group = ""
        elif group:
            words = [group, *words]
        label = " ".join(words).strip()
        if not label or len(label) > 64:
            continue
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        # Non-finite values are kept: the caller records them as divergence
        # rather than as chart points.
        pairs.append((label, value))
    return pairs


def _event_match(line: str) -> tuple[re.Match[str] | None, str]:
    """Match an epoch line, first as printed, then past glued progress frames."""

    match = EVENT_RE.search(line)
    if match:
        return match, line
    stripped = line
    for _ in range(4):
        prefix = PROGRESS_PREFIX_RE.match(stripped)
        if not prefix:
            break
        stripped = stripped[prefix.end() :]
        match = EVENT_RE.search(stripped)
        if match:
            return match, stripped
    return None, line


def _nonfinite_summary(trace: list[tuple[int, bool, float]]) -> dict[str, Any]:
    """Describe where a series was nan/inf, in the order the log printed it."""

    ranges: list[list[int]] = []
    open_range: list[int] | None = None
    for x, finite, _value in trace:
        if finite:
            open_range = None
            continue
        if open_range is None:
            open_range = [x, x]
            ranges.append(open_range)
        else:
            open_range[1] = x
    count = sum(1 for _x, finite, _value in trace if not finite)
    diverged = bool(trace) and not trace[-1][1]
    return {
        "nonfinite_count": count,
        "nonfinite_ranges": ranges,
        # A series whose last value is nan/inf ended diverged: its `last` is the
        # last finite value, which must not be read as where the run ended.
        "diverged": diverged,
        "diverged_at": ranges[-1][0] if diverged else None,
        "last_nonfinite": repr(trace[-1][2]) if diverged else None,
    }


def parse_training_log(
    text: str,
    *,
    training_steps: set[int] | None = None,
    declared_total_steps: int | None = None,
    stage_hints: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Return metrics on anchored epoch/iteration lines in training stages.

    ``stage_hints`` names the stage a single-stage step trains (``train_vae``
    -> ``vae``). It only enters ``compare_key``, so the VAE `LR` of a
    ``train_vae`` run lines up with the VAE `LR` of a ``mode train`` pipeline,
    whose header names the stage in the log itself.
    """

    active_step = 1
    total_steps = max(1, int(declared_total_steps or 1))
    event_kinds: set[str] = set()
    # One Studio step can hold several training stages: tagged lines belong to
    # their tag's stage, untagged ones to the current segment, which a stage
    # header or an epoch counter restarting at 0/1 closes. Without this, the
    # VAE and FM `LR` of an SDFFlow `mode train` run merge into one series
    # whose x runs backwards.
    stages: dict[int, OrderedDict[str, dict[str, Any]]] = {}
    segment: dict[int, int] = {}
    pending_header: dict[int, str] = {}
    last_x: dict[tuple[int, str, str], int] = {}
    observations: list[tuple[int, str, str, str, dict[str, Any]]] = []

    for line_number, raw_line in enumerate(ANSI_RE.sub("", text or "").splitlines(), start=1):
        step_match = STEP_RE.search(raw_line)
        if step_match:
            active_step = int(step_match.group("step"))
            total_steps = max(total_steps, int(step_match.group("total")))
            # This orchestration line may contain digits in a block name such
            # as HDF5; it identifies a stage but is not a metric observation.
            continue
        if training_steps is not None and active_step not in training_steps:
            continue
        header_match = STAGE_HEADER_RE.search(raw_line)
        if header_match:
            pending_header[active_step] = header_match.group("name")
            continue
        event_match, line = _event_match(raw_line)
        if not event_match:
            continue
        event_kind = event_match.group("kind").lower()
        if event_kind == "iter":
            event_kind = "iteration"
        event_kinds.add(event_kind)
        event_value = int(event_match.group("value"))

        step_stages = stages.setdefault(active_step, OrderedDict())
        tag = (event_match.group("tag") or "").strip()
        if tag:
            stage_id = f"tag:{tag.lower()}"
            step_stages.setdefault(stage_id, {"name": tag, "header": False})
        else:
            stage_id = f"segment:{segment.get(active_step, 0)}"
            header_name = pending_header.pop(active_step, "")
            previous = last_x.get((active_step, stage_id, event_kind))
            restarted = previous is not None and event_value < previous and event_value <= 1
            if stage_id in step_stages and (header_name or restarted):
                segment[active_step] = segment.get(active_step, 0) + 1
                stage_id = f"segment:{segment[active_step]}"
            step_stages.setdefault(stage_id, {"name": header_name, "header": bool(header_name)})
        last_x[(active_step, stage_id, event_kind)] = event_value

        for label, value in _metric_pairs(line[event_match.end() :]):
            base_key = _metric_key(label)
            if not base_key:
                continue
            observations.append(
                (
                    active_step,
                    stage_id,
                    base_key,
                    label,
                    {"x": event_value, "y": value, "line": line_number, "event": event_kind},
                )
            )

    # Stage names enter keys and labels only where they disambiguate (several
    # stages in one step), were announced by a header (`[Pipeline 2/2]
    # Training FM`), or were printed as a stage tag. A stage tag scopes from its
    # first line: HI-MGN's `mode train` prints `[AE]` for hours before the
    # first `[Prior]`, and its keys must not turn into `ae__*` when that
    # arrives. The tag stays silent only where it adds nothing: a non-stage tag
    # such as `[model_split]`, or `[AE]` in a `train_ae` step, whose hint
    # already names the stage.
    measured: dict[int, list[str]] = {}
    for step, stage_id, _base_key, _label, _point in observations:
        ids = measured.setdefault(step, [])
        if stage_id not in ids:
            ids.append(stage_id)
    # The step prefix follows the job's declared training steps, not the steps
    # seen so far, so a key never changes while the run is still writing it; a
    # trailing inference or export step is not a reason to scope the keys.
    if training_steps is not None:
        scope_steps = len(training_steps) > 1
    else:
        scope_steps = total_steps > 1
    series: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for step, stage_id, base_key, label, point in observations:
        key_parts: list[str] = []
        label_parts: list[str] = []
        if scope_steps:
            key_parts.append(f"step_{step}")
            label_parts.append(f"Step {step}")
        stage = stages[step][stage_id]
        stage_part = ""
        display = label
        tag_key = _metric_key(stage["name"]) if stage_id.startswith("tag:") else ""
        stage_tag = bool(tag_key) and tag_key not in NON_STAGE_TAGS and tag_key != _metric_key(
            (stage_hints or {}).get(step, "")
        )
        if len(measured[step]) > 1 or stage["header"] or stage_tag:
            stage_name = stage["name"] or f"Stage {measured[step].index(stage_id) + 1}"
            stage_part = _metric_key(stage_name)
            key_parts.append(stage_part)
            label_parts.append(stage_name)
            # `LC train` under the `LC` stage reads `LC · train`, not `LC · LC train`.
            words = label.split()
            if len(words) > 1 and words[0].lower() == stage_name.lower():
                display = " ".join(words[1:])
        scoped_key = "__".join([*key_parts, base_key])
        item = series.get(scoped_key)
        if item is None:
            # What Compare matches runs on: the same trainer's metric has the
            # same compare key whether it ran alone (`train_vae`), inside
            # `mode train`'s VAE->FM pipeline, or ahead of an inference step.
            compare_stage = stage_part or _metric_key((stage_hints or {}).get(step, ""))
            compare_parts = [compare_stage] if compare_stage else []
            item = series[scoped_key] = {
                "key": scoped_key,
                "label": " · ".join([*label_parts, display]),
                "compare_key": "__".join([*compare_parts, base_key]),
                "compare_label": " · ".join([*(label_parts[1:] if scope_steps else label_parts), display]),
                "event": point["event"],
                "points": [],
                "_trace": [],
            }
        finite = math.isfinite(point["y"])
        item["_trace"].append((point["x"], finite, point["y"]))
        if finite:
            item["points"].append(point)

    # Two training steps printing the same metric keep their step in the
    # compare key; there is no single series to line up with another run.
    compare_counts: dict[str, int] = {}
    for item in series.values():
        compare_counts[item["compare_key"]] = compare_counts.get(item["compare_key"], 0) + 1
    for item in series.values():
        if compare_counts[item["compare_key"]] > 1:
            item["compare_key"] = item["key"]
            item["compare_label"] = item["label"]

    metrics: list[dict[str, Any]] = []
    for item in series.values():
        trace = item.pop("_trace")
        values = [point["y"] for point in item["points"]]
        metrics.append(
            {
                **item,
                "count": len(values),
                # A series that was nan/inf on every epoch has no finite value;
                # it is still listed, since that is exactly what diverged.
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "last": values[-1] if values else None,
                **_nonfinite_summary(trace),
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


def _is_training_mode(mode: str) -> bool:
    return str(mode).strip().lower().startswith("train")


def _training_step_numbers(job: dict[str, Any]) -> set[int]:
    numbers: set[int] = set()
    for index, step in enumerate(job.get("steps", []), start=1):
        if not isinstance(step, dict):
            continue
        route = step.get("route") if isinstance(step.get("route"), dict) else {}
        if _is_training_mode(str(route.get("mode", ""))):
            numbers.add(index)
    return numbers


def _stage_hints(job: dict[str, Any]) -> dict[int, str]:
    """`train_vae` / `train_fm` / `train_lc` / `train_ae` / `train_prior` name their one stage."""

    hints: dict[int, str] = {}
    for index, step in enumerate(job.get("steps", []), start=1):
        if not isinstance(step, dict):
            continue
        route = step.get("route") if isinstance(step.get("route"), dict) else {}
        mode = str(route.get("mode", "")).strip().lower()
        if mode.startswith("train_") and len(mode) > len("train_"):
            hints[index] = mode[len("train_") :]
    return hints


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
        training_lineage = [step for step in lineage if _is_training_mode(step["mode"])]
        training_steps = _training_step_numbers(job)
        if not training_steps:
            continue
        if node_id and not any(step["node_id"] == node_id for step in training_lineage):
            continue
        if model_id and not any(step["model_id"] == model_id for step in training_lineage):
            continue
        full_log = state.read_job_log(str(job["id"])) if hasattr(state, "read_job_log") else str(job.get("log", ""))
        parsed = parse_training_log(
            full_log,
            training_steps=training_steps,
            declared_total_steps=len(job.get("steps", [])),
            stage_hints=_stage_hints(job),
        )
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
                "models": sorted({step["model_id"] for step in training_lineage if step["model_id"]}),
                "node_ids": [step["node_id"] for step in training_lineage if step["node_id"]],
                "lineage": lineage,
                "training_lineage": training_lineage,
                "target_node_id": job.get("target_node_id", ""),
                # Node ids repeat across every canvas made from one template;
                # this is what tells the GUI whether the run is this canvas's.
                "canvas_id": job.get("canvas_id", ""),
                "log_path": job.get("log_path", ""),
                **parsed,
            }
        )
    return {"items": items, "count": len(items), "source": "Studio job logs"}
