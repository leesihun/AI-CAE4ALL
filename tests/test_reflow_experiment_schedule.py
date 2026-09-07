"""Check the offline reflow protocol's budgets, reproducibility and bounds."""

import csv
import importlib.util
import json
import random
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "docs/research/reflow/generate_schedule.py"
SPEC = importlib.util.spec_from_file_location("reflow_schedule", SCRIPT)
scheduler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scheduler)


@pytest.mark.parametrize("repeats,count,units", [(1, 21, 187), (2, 24, 220), (4, 30, 286)])
def test_documented_budgets(repeats, count, units):
    cfg = scheduler.validate_config({"train_repeats_per_family": repeats, "reference_minutes": 5})
    runs = scheduler.build_campaign(cfg)
    report = scheduler.audit(runs, cfg)
    assert report["run_count"] == count
    assert report["distinct_command_sequence_count"] == count - 1
    assert report["total_reference_units"] == units
    assert report["nominal_hours"] == pytest.approx(units / 12)
    assert report["split_run_counts"] == {"train": 11 + 3 * repeats, "validation": 3, "test": 4}


@pytest.mark.parametrize("seed", [0, 1, 42, 20260907, 123456789])
def test_random_proposals_obey_protocol(seed):
    cfg = scheduler.validate_config({"seed": seed})
    runs = scheduler.build_campaign(cfg)
    scheduler.audit(runs, cfg)
    for run in runs:
        prior = [0] * 11
        for segment in run["segments"]:
            assert segment["ticks"] >= 1
            assert all(-1 <= q <= 1 for q in segment["q"])
            assert all(abs(q - p) <= 1 for q, p in zip(segment["q"], prior))
            prior = segment["q"]
        assert prior == [0] * 11
        if run["family"] == "group":
            groups = [s for s in run["segments"] if s["label"].startswith("group_")]
            assert sorted(sum(q != 0 for q in s["q"]) for s in groups) == [1, 3, 6, 11]
        if run["family"] == "order":
            finals = [s["q"] for s in run["segments"] if s["label"] == "both_groups"]
            assert len(finals) == 2 and finals[0] == finals[1]


def test_new_training_runs_preserve_existing_waveforms_and_rng_state():
    before = random.getstate()
    original = {r["run_id"]: r for r in scheduler.build_campaign(scheduler.validate_config({}))}
    expanded = {r["run_id"]: r for r in scheduler.build_campaign(
        scheduler.validate_config({"train_repeats_per_family": 4}))}
    assert all(expanded[name] == run for name, run in original.items())
    assert random.getstate() == before
    assert scheduler.build_campaign(scheduler.validate_config({})) == list(original.values())


def test_repeated_waveform_stays_in_test_group():
    runs = {r["run_id"]: r for r in scheduler.build_campaign(scheduler.validate_config({}))}
    original, repeat = runs["test_async_01"], runs["test_async_01_repeat"]
    assert original["segments"] == repeat["segments"]
    assert original["sequence_group"] == repeat["sequence_group"]
    assert original["split"] == repeat["split"] == "test"
    assert repeat["repeat_of"] == original["run_id"]


@pytest.mark.parametrize("raw", [
    {"delta_c": 1},
    {"base_profile_c": [100] * 10, "delta_c": 1},
    {"base_profile_c": [100] * 11, "delta_c": 0},
    {"reference_minutes": float("nan")},
    {"reference_minutes": 0},
    {"seed": True},
    {"train_repeats_per_family": 0},
    {"minimum_setpoint_c": 0},
    {"unknown": 1},
    {"base_profile_c": [100] * 11, "delta_c": 10,
     "minimum_setpoint_c": 95, "maximum_setpoint_c": 200},
])
def test_invalid_or_incomplete_configs_are_rejected(raw):
    with pytest.raises(ValueError):
        scheduler.validate_config(raw)


def test_normalized_export_does_not_invent_temperatures_or_times(tmp_path):
    cfg = scheduler.validate_config({})
    runs = scheduler.build_campaign(cfg)
    report = scheduler.audit(runs, cfg)
    out = tmp_path / "normalized"
    scheduler.write_outputs(out, runs, cfg, report)
    with (out / "schedule.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    assert all(row["start_seconds"] == row["duration_seconds"] == "" for row in rows)
    assert all(row[f"setpoint_c_{i:02d}"] == "" for row in rows for i in range(1, 12))
    assert sum(float(row["duration_reference"]) for row in rows) == 220
    saved = json.loads((out / "campaign.json").read_text(encoding="utf-8"))
    assert saved["config"]["base_profile_c"] is None
    original_csv = (out / "schedule.csv").read_bytes()
    with pytest.raises(FileExistsError):
        scheduler.write_outputs(out, runs, cfg, report)
    assert (out / "schedule.csv").read_bytes() == original_csv


def test_physical_export_matches_user_values_and_time_units(tmp_path):
    # Synthetic values solely for the export contract; not an oven recommendation.
    cfg = scheduler.validate_config({"base_profile_c": [100 + i for i in range(11)],
                                    "delta_c": [i + 1 for i in range(11)],
                                    "minimum_setpoint_c": 80, "maximum_setpoint_c": 130,
                                    "reference_minutes": 2})
    runs = scheduler.build_campaign(cfg)
    report = scheduler.audit(runs, cfg)
    assert report["setpoint_bounds_checked"]
    out = tmp_path / "physical"
    scheduler.write_outputs(out, runs, cfg, report)
    with (out / "schedule.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(float(row["duration_seconds"]) for row in rows) == 220 * 120
    for row in rows:
        for i in range(11):
            expected = cfg["base_profile_c"][i] + cfg["delta_c"][i] * float(row[f"q_{i + 1:02d}"])
            assert float(row[f"setpoint_c_{i + 1:02d}"]) == expected
