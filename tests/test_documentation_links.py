"""Every relative link in every checked-in document must resolve on disk.

Scoped to the whole tree rather than three root files: the documentation lives
under docs/ and links across into methods/, configs/ and cae_suite/, so a moved
directory breaks links far from where the move happened.
"""
import re
from pathlib import Path
from urllib.parse import unquote

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "output", ".venv"}
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.I)


def _documents():
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
            continue
        yield path.relative_to(ROOT).as_posix()


def _prose(text):
    """Drop fenced code blocks; `a[i]` inside one is not a markdown link."""
    kept, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            kept.append(line)
    return "\n".join(kept)


def _unresolved(document, pattern):
    missing = []
    text = _prose(document.read_text(encoding="utf-8"))
    for raw_target in pattern.findall(text):
        target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
        if not target or target.startswith("#") or SCHEME_RE.match(target):
            continue
        path_text = unquote(target.split("#", 1)[0])
        if path_text and not (document.parent / path_text).exists():
            missing.append(path_text)
    return sorted(set(missing))


@pytest.mark.parametrize("relative_path", list(_documents()))
def test_document_relative_links_resolve(relative_path):
    document = ROOT / relative_path
    assert not _unresolved(document, LINK_RE)


@pytest.mark.parametrize("relative_path", list(_documents()))
def test_document_images_resolve(relative_path):
    document = ROOT / relative_path
    assert not _unresolved(document, IMAGE_RE)
