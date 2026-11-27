"""The docs/ twins of the generated paper surfaces must not drift.

`docs/` holds hand-written survey chapters, and it also holds copies of
three generated paper surfaces plus `data/RESULTS.md`. A copy that
nobody refreshes is a promise nobody keeps: the next regeneration
rewrites the original, and the copy keeps the old number — which is
exactly how the held-out RAGTruth tally stayed wrong in one file and
right in another.

These tests assert the *rule* — a mirror is rewritten from its source,
and a drifted mirror is detected — rather than freezing the current
content of each mirror.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.fill_comparison_table import (
    DOCS_MIRRORS,
    refresh_docs_mirrors,
)

ROOT = Path(__file__).resolve().parents[1]


def _normalised(path: Path) -> str:
    """Read a file with line endings normalised.

    Git checks out CRLF on Windows and LF elsewhere, so a byte comparison
    between a mirror and its source reports a difference that is not a
    difference in content.
    """
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_every_mirror_names_an_existing_source():
    for source, mirror in DOCS_MIRRORS.items():
        assert (ROOT / source).exists(), f"{mirror} has no source {source}"
        assert mirror.startswith("docs/")


def test_the_mirrors_on_disk_match_their_sources():
    """The live docs/ copies must agree with the paper they mirror."""
    for source, mirror in DOCS_MIRRORS.items():
        src = ROOT / source
        dst = ROOT / mirror
        assert dst.exists(), f"{mirror} is missing"
        assert _normalised(dst) == _normalised(src), (
            f"{mirror} has drifted from {source}. Run "
            f"`python scripts/fill_comparison_table.py` to refresh it."
        )


def test_a_drifted_mirror_is_detected_and_rewritten(tmp_path: Path):
    mirror = tmp_path / "comparison.md"
    mirror.write_text("A number that no longer exists", encoding="utf-8")

    changes = refresh_docs_mirrors(tmp_path, dry_run=True)

    assert any(c[0] == "docs/comparison.md" for c in changes)
    # A dry run reports the change without touching the file.
    assert mirror.read_text(encoding="utf-8") == "A number that no longer exists"

    changes = refresh_docs_mirrors(tmp_path, dry_run=False)

    assert any(c[0] == "docs/comparison.md" for c in changes)
    assert _normalised(mirror) == _normalised(ROOT / "Paper/comparison.md")


def test_refreshing_twice_reports_no_further_change(tmp_path: Path):
    refresh_docs_mirrors(tmp_path, dry_run=False)
    assert refresh_docs_mirrors(tmp_path, dry_run=True) == []


def test_a_missing_mirror_is_created_rather_than_reported_as_drift(
    tmp_path: Path,
):
    changes = refresh_docs_mirrors(tmp_path, dry_run=False)

    for _source, mirror in DOCS_MIRRORS.items():
        dst = tmp_path / Path(mirror).name
        assert dst.exists(), f"{mirror} was not created"
        assert any(c[0] == mirror for c in changes)
