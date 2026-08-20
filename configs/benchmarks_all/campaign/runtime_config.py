"""Per-job runtime config generation.

Replicates train_all.sh/infer_all.sh's sed-based `gpu_ids` / `mode` patching
byte-for-byte (same regexes, same error conditions), rather than routing
through cae_suite.config_parser -- that parser has no serializer and
lowercases non-PATH_KEYS values, which would silently reformat configs the
original shell `sed` never touched.
"""

from __future__ import annotations

import re
from pathlib import Path

_GPU_IDS_RE = re.compile(r"^([ \t]*gpu_ids[ \t]+)\S+", re.MULTILINE)
_MODE_TRAIN_RE = re.compile(r"^([ \t]*mode[ \t]+)train([ \t]*)$", re.MULTILINE)


class RuntimeConfigError(Exception):
    pass


def write_runtime_config(mode: str, source_cfg: Path, gpu: str, out_path: Path) -> None:
    """mode='train': patch gpu_ids only (error if the config has no gpu_ids line,
    matching train_all.sh). mode='infer': patch mode train->inference (error if
    absent, matching infer_all.sh) and best-effort patch gpu_ids (silently a
    no-op if absent, matching infer_all.sh's unconditional sed)."""
    text = source_cfg.read_text(encoding="utf-8")

    if mode == "infer":
        if not _MODE_TRAIN_RE.search(text):
            raise RuntimeConfigError(
                f"no 'mode ... train' line found in {source_cfg} (already non-canonical?)")
        text = _MODE_TRAIN_RE.sub(lambda m: f"{m.group(1)}inference{m.group(2)}", text)
    else:
        if not _GPU_IDS_RE.search(text):
            raise RuntimeConfigError(f"no gpu_ids entry in {source_cfg}")

    text = _GPU_IDS_RE.sub(lambda m: f"{m.group(1)}{gpu}", text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
