"""Offline, reproducible setpoint proposals for the accompanying NARX protocol.

Uses only the Python standard library. Does not communicate with equipment.
Temperatures remain unspecified unless base_profile_c and delta_c are supplied.
Every run starts and ends at the base profile. Times are nominal, not stability
guarantees. See REFLOW_NARX_EXPERIMENT_DESIGN.md before using the proposals.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


ZONES = 11
TICKS_PER_REFERENCE = 8
LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
FAMILIES = ("async", "group", "order")
DEFAULTS = {
    "base_profile_c": None,
    "delta_c": None,
    "minimum_setpoint_c": None,
    "maximum_setpoint_c": None,
    "reference_minutes": None,
    "seed": 20260907,
    "train_repeats_per_family": 2,
}


def vector(value, name, *, scalar=False):
    if value is None:
        return None
    if scalar and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = [value] * ZONES
    if not isinstance(value, list) or len(value) != ZONES:
        raise ValueError(f"{name} must have exactly {ZONES} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in value):
        raise ValueError(f"{name} must contain finite numbers")
    return [float(x) for x in value]


def validate_config(raw):
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    for name in ("seed", "train_repeats_per_family"):
        if isinstance(cfg[name], bool) or not isinstance(cfg[name], int):
            raise ValueError(f"{name} must be an integer")
    if cfg["train_repeats_per_family"] < 1:
        raise ValueError("train_repeats_per_family must be at least 1")
    ref = cfg["reference_minutes"]
    if ref is not None and (isinstance(ref, bool) or not isinstance(ref, (int, float))
                            or not math.isfinite(ref) or ref <= 0):
        raise ValueError("reference_minutes must be positive and finite")
    for name in ("base_profile_c", "delta_c", "minimum_setpoint_c", "maximum_setpoint_c"):
        cfg[name] = vector(cfg[name], name, scalar=name != "base_profile_c")
    base, delta = cfg["base_profile_c"], cfg["delta_c"]
    if (base is None) != (delta is None):
        raise ValueError("provide both base_profile_c and delta_c, or neither")
    low, high = cfg["minimum_setpoint_c"], cfg["maximum_setpoint_c"]
    if (low is None) != (high is None):
        raise ValueError("provide both setpoint bounds, or neither")
    if low is not None and base is None:
        raise ValueError("setpoint bounds require base_profile_c and delta_c")
    if delta is not None and any(x <= 0 for x in delta):
        raise ValueError("all delta_c entries must be positive")
    if low is not None:
        for i in range(ZONES):
            if low[i] > high[i] or base[i] - delta[i] < low[i] or base[i] + delta[i] > high[i]:
                raise ValueError(f"zone {i + 1}: base +/- delta exceeds supplied bounds")
    return cfg


def seed_for(seed, name):
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def make_run(name, split, family, seed, variant=0, zone=None):
    rng = random.Random(seed)
    run = {"run_id": name, "split": split, "family": family, "seed": seed,
           "sequence_group": name, "repeat_of": None, "segments": []}
    zero = [0.0] * ZONES

    def emit(q, ticks, label):
        run["segments"].append({"ticks": ticks, "q": list(q), "label": label})

    emit(zero, 16, "warmup_base_minimum")
    if family == "screen":
        first_sign = rng.choice((-1.0, 1.0))
        for sign in (first_sign, -first_sign):
            q = list(zero)
            q[zone] = sign
            emit(q, 8, f"zone_{zone + 1}_excursion")
            emit(zero, 8, "return_base")
    elif family == "async":
        q, remaining = list(zero), 64
        while remaining:
            count = rng.choices((1, 2, 3), weights=(6, 3, 1))[0]
            for i in rng.sample(range(ZONES), count):
                eligible = [v for v in LEVELS if v != q[i] and abs(v - q[i]) <= 1.0]
                q[i] = rng.choice(eligible)
            ticks = min(remaining, rng.choices((1, 2, 4, 8), weights=(2, 3, 3, 2))[0])
            emit(q, ticks, "random_subset_update")
            remaining -= ticks
    elif family == "group":
        sizes = [1, 3, 6, 11]
        rng.shuffle(sizes)
        for size in sizes:
            mode = ("positive" if variant % 2 == 0 else "negative") if size == 11 else rng.choice(
                ("positive", "negative", "mixed"))
            selected = rng.sample(range(ZONES), size)
            signs = ([1.0] * size if mode == "positive" else [-1.0] * size)
            if mode == "mixed":
                signs = [rng.choice((-1.0, 1.0)) for _ in selected]
                if size > 1:
                    signs[0], signs[1] = -1.0, 1.0
            q = list(zero)
            for i, sign in zip(selected, signs):
                q[i] = sign * rng.choice((0.5, 1.0))
            emit(q, 12, f"group_{size}_{mode}")
            emit(zero, 4, "return_base")
    elif family == "order":
        selected = rng.sample(range(ZONES), rng.choice((3, 6, 11)))
        cut = rng.randrange(1, len(selected))
        groups = [selected[:cut], selected[cut:]]
        q_final = list(zero)
        sign = 1.0 if variant % 2 == 0 else -1.0
        for i in selected:
            q_final[i] = sign * rng.choice((0.5, 1.0))
        gap = rng.choice((2, 4, 8))
        first = rng.randrange(2)
        for index in (first, 1 - first):
            q_first = [q_final[i] if i in groups[index] else 0.0 for i in range(ZONES)]
            emit(q_first, gap, f"group_{index + 1}_first")
            emit(q_final, 16 - gap, "both_groups")
            emit(zero, 16, "return_base")
    else:
        raise ValueError(f"unknown family: {family}")
    emit(zero, 8, "tail_base_minimum")
    return run


def build_campaign(cfg):
    runs = []
    screening_order = list(range(ZONES))
    random.Random(seed_for(cfg["seed"], "screen_order")).shuffle(screening_order)
    for zone in screening_order:
        name = f"train_screen_z{zone + 1:02d}"
        runs.append(make_run(name, "train", "screen", seed_for(cfg["seed"], name), zone=zone))
    for split, count in (("train", cfg["train_repeats_per_family"]), ("validation", 1), ("test", 1)):
        batch = []
        for variant in range(count):
            for family in FAMILIES:
                name = f"{split}_{family}_{variant + 1:02d}"
                batch.append(make_run(name, split, family, seed_for(cfg["seed"], name), variant))
        random.Random(seed_for(cfg["seed"], f"{split}_order")).shuffle(batch)
        runs.extend(batch)
    original = next(r for r in runs if r["run_id"] == "test_async_01")
    repeat = copy.deepcopy(original)
    repeat["run_id"] = "test_async_01_repeat"
    repeat["repeat_of"] = original["run_id"]
    runs.append(repeat)
    return runs


def audit(runs, cfg):
    ticks = 0
    changes = [0] * ZONES
    positive_ticks = [0] * ZONES
    negative_ticks = [0] * ZONES
    pair_ticks = [[0] * ZONES for _ in range(ZONES)]
    level_ticks = [Counter() for _ in range(ZONES)]
    signatures = {}
    split_ticks = Counter()
    split_runs = Counter()
    max_step = 0.0
    for run in runs:
        previous = [0.0] * ZONES
        expected = 56 if run["family"] == "screen" else 88
        actual = sum(s["ticks"] for s in run["segments"])
        if actual != expected:
            raise ValueError(f"unexpected duration in {run['run_id']}")
        if any(run["segments"][0]["q"]) or any(run["segments"][-1]["q"]):
            raise ValueError("every run must start and end at the base profile")
        signature = json.dumps([(s["ticks"], s["q"]) for s in run["segments"]])
        if signature in signatures and not run["repeat_of"]:
            raise ValueError(f"duplicate complete command sequence: {run['run_id']}")
        signatures[signature] = run["run_id"]
        split_runs[run["split"]] += 1
        for segment in run["segments"]:
            q, duration = segment["q"], segment["ticks"]
            if duration <= 0 or len(q) != ZONES or any(v not in LEVELS for v in q):
                raise ValueError("invalid segment")
            for i in range(ZONES):
                step = abs(q[i] - previous[i])
                if step > 1.0:
                    raise ValueError("normalized per-command change exceeds 1")
                max_step = max(max_step, step)
                changes[i] += int(step > 0)
                positive_ticks[i] += duration * (q[i] > 0)
                negative_ticks[i] += duration * (q[i] < 0)
                level_ticks[i][str(q[i])] += duration
                for j in range(ZONES):
                    pair_ticks[i][j] += duration * (q[i] != 0 and q[j] != 0)
            ticks += duration
            split_ticks[run["split"]] += duration
            previous = q
    for zone in range(ZONES):
        if not positive_ticks[zone] or not negative_ticks[zone]:
            raise ValueError(f"zone {zone + 1} lacks positive or negative excitation")
    ref_units = ticks / TICKS_PER_REFERENCE
    ref_minutes = cfg["reference_minutes"]
    return {
        "run_count": len(runs),
        "distinct_command_sequence_count": len(runs) - 1,
        "split_run_counts": dict(split_runs),
        "total_reference_units": ref_units,
        "split_reference_units": {k: v / 8 for k, v in split_ticks.items()},
        "nominal_hours": None if ref_minutes is None else ref_units * ref_minutes / 60,
        "hours_if_reference_minutes": {str(v): ref_units * v / 60 for v in (2, 5, 10, 20)},
        "zone_change_counts": changes,
        "zone_positive_reference_units": [v / 8 for v in positive_ticks],
        "zone_negative_reference_units": [v / 8 for v in negative_ticks],
        "zone_level_reference_units": [{k: v / 8 for k, v in c.items()} for c in level_ticks],
        "co_excited_pair_reference_units": [[v / 8 for v in row] for row in pair_ticks],
        "maximum_normalized_step": max_step,
        "setpoint_bounds_checked": cfg["minimum_setpoint_c"] is not None,
        "status": "normalized_proposal" if cfg["base_profile_c"] is None else "temperature_proposal",
        "excluded_time": ["pilot", "initial_heat_up", "extra_stabilization", "breaks", "production_recipe_validation"],
        "limitations": [
            "Coverage counters describe planned setpoints, not temperatures or heater power.",
            "Independent command sequences are distinct waveforms, not statistically independent thermal runs.",
            "Nominal baseline holds do not prove equal hidden states or thermal equilibrium.",
            "No equipment connection, NARX training, rank guarantee, or physical validation is performed.",
        ],
    }


def write_outputs(out, runs, cfg, report):
    # Exclusive output directory: never overwrite a previous campaign or a user file.
    out.mkdir(parents=True, exist_ok=False)
    payload = {"protocol_version": 1, "config": cfg, "report": report, "runs": runs}
    (out / "campaign.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["run_id", "split", "family", "sequence_group", "repeat_of", "segment", "label",
              "start_reference", "duration_reference", "start_seconds", "duration_seconds"]
    fields += [f"q_{i + 1:02d}" for i in range(ZONES)] + [f"setpoint_c_{i + 1:02d}" for i in range(ZONES)]
    with (out / "schedule.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            start = 0
            for index, segment in enumerate(run["segments"]):
                row = {k: run[k] for k in ("run_id", "split", "family", "sequence_group", "repeat_of")}
                row.update(segment=index, label=segment["label"], start_reference=start / 8,
                           duration_reference=segment["ticks"] / 8)
                ref = cfg["reference_minutes"]
                row["start_seconds"] = "" if ref is None else start / 8 * ref * 60
                row["duration_seconds"] = "" if ref is None else segment["ticks"] / 8 * ref * 60
                for i, q in enumerate(segment["q"]):
                    row[f"q_{i + 1:02d}"] = q
                    row[f"setpoint_c_{i + 1:02d}"] = "" if cfg["base_profile_c"] is None else (
                        cfg["base_profile_c"][i] + cfg["delta_c"][i] * q)
                writer.writerow(row)
                start += segment["ticks"]
    with (out / "run_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["run_id", "split", "family", "duration_reference", "duration_minutes", "repeat_of"])
        for run in runs:
            duration = sum(s["ticks"] for s in run["segments"]) / 8
            minutes = "" if cfg["reference_minutes"] is None else duration * cfg["reference_minutes"]
            writer.writerow([run["run_id"], run["split"], run["family"], duration, minutes, run["repeat_of"] or ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference-minutes", type=float, help="measured T_ref, or a clearly labeled planning scenario")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-repeats", type=int, help="runs per random family in the training split")
    parser.add_argument("--out", type=Path, help="new output directory; omit to print the audit only")
    args = parser.parse_args()
    raw = {} if args.config is None else json.loads(args.config.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        parser.error("config must be a JSON object")
    for key, value in (("reference_minutes", args.reference_minutes), ("seed", args.seed),
                       ("train_repeats_per_family", args.train_repeats)):
        if value is not None:
            raw[key] = value
    try:
        cfg = validate_config(raw)
        runs = build_campaign(cfg)
        report = audit(runs, cfg)
        if args.out is not None:
            write_outputs(args.out, runs, cfg, report)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
