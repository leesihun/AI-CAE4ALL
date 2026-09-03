"""Roster loading -- the same tab-separated (label, train_config, ex_slot,
light) format configs/campaigns/benchmarks_all/score_rollouts.py reads independently,
so this module must never change that format, only how it's consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class RosterError(Exception):
    pass


@dataclass(frozen=True)
class RosterEntry:
    label: str
    train_config: Path
    ex_slot: str
    light: bool


def load_roster(path: Path) -> list[RosterEntry]:
    entries: list[RosterEntry] = []
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            label, cfg, ex_slot, light = line.split("\t")
            entries.append(RosterEntry(
                label=label,
                train_config=Path(cfg),
                ex_slot=ex_slot,
                light=light.strip() == "1",
            ))
    return entries


def select_labels(
    entries: list[RosterEntry], labels: str | None, roster_path: Path
) -> list[RosterEntry]:
    """Mirrors the bash LABELS filter loop, including its hard-error-on-unknown-label behavior."""
    if not labels:
        return list(entries)
    by_label = {e.label: e for e in entries}
    selected: list[RosterEntry] = []
    for want in labels.split():
        entry = by_label.get(want)
        if entry is None:
            raise RosterError(f"unknown label '{want}' (see {roster_path})")
        selected.append(entry)
    return selected
