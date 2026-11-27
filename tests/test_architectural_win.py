"""The architectural-win figure must match the survey it summarises.

The figure claims that no published pipeline implements all six of the
defining capabilities while CRISP does. That claim is only worth as much
as the survey behind it, so the figure reads the survey's own grid
rather than restating it: if a row is added, removed or re-marked in the
survey, the figure changes with it, and a test fails if the two drift
apart.

These tests assert the rule, not the current numbers: that the figure is
generated from the survey, that no published row reaches six of six, and
that the figure is written without any annotation that a caption would
duplicate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_architectural_win import _survey_grid  # noqa: E402

SURVEY = ROOT / "docs" / "related_pipeline_workflows.md"
FIGURE = ROOT / "data" / "graphs" / "architectural_win.png"


def test_the_survey_grid_is_readable():
    rows = _survey_grid()
    assert len(rows) >= 15, "the survey's landscape grid was not found"
    for name, caps in rows:
        assert len(caps) == 6, f"{name} has {len(caps)} capability columns"


def test_no_published_pipeline_implements_all_six():
    """This is the finding the figure asserts, so the survey must support it."""
    rows = _survey_grid()
    for name, caps in rows:
        assert sum(caps) < 6, (
            f"{name} now implements all six capabilities, so the figure's "
            f"headline is no longer true and the survey's own claim must "
            f"be revisited"
        )


def test_the_best_published_pipeline_is_noted_correctly():
    rows = _survey_grid()
    best = max(sum(c) for _, c in rows)
    assert best == 5, (
        f"the best published pipeline now reaches {best} of 6; the "
        f"comparison in the book must be re-checked"
    )


def test_the_figure_exists():
    assert FIGURE.exists()
    assert FIGURE.stat().st_size > 10_000


def test_the_figure_carries_no_annotation():
    """Journal style: the caption states the finding, the plot shows data.

    The generator writes the paper names, the capability tags and the
    per-row counts, and nothing else. A title or a summary sentence
    baked into the image would duplicate the caption and make the figure
    unusable in a two-column layout.
    """
    source = (ROOT / "scripts" / "plot_architectural_win.py").read_text(
        encoding="utf-8"
    )
    # No axis title, no suptitle, no in-plot annotation beyond the row
    # labels, the column tags and the per-row counts.
    assert "set_title" not in source
    assert "suptitle" not in source
    assert "ax.annotate" not in source
