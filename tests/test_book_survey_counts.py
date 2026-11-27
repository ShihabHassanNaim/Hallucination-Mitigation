"""The counts of the architectural survey must match the survey.

The central claim of the thesis is architectural: of the surveyed
pipelines, none implements all six capabilities, the two closest
implement five, and CRISP implements all six. The number of pipelines
and the coverage of each capability are read from the survey by
``thesis.survey``, and the prose takes them from there rather than
writing them out.

The check here is the one that matters if the survey is ever extended:
the counts the book states have to be the counts the survey produces,
and the "none reaches six, two reach five" claim has to hold against
the survey's own rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thesis import content as C  # noqa: E402
from thesis import survey  # noqa: E402


def test_the_pipeline_count_comes_from_the_survey():
    assert C.N_PIPELINES == len(survey.capability_grid())
    assert C.N_PIPELINE_WORDS == C.word(C.N_PIPELINES)


def test_the_coverage_counts_come_from_the_survey():
    counts = survey.coverage_counts()
    assert list(C.COVERAGE.values()) == counts
    assert len(C.COVERAGE) == len(survey.CAPABILITIES)


def test_no_surveyed_pipeline_reaches_all_six_capabilities():
    """The claim the whole architectural argument rests on."""
    reached_six = []
    reached_five = []
    for name, row in survey.capability_grid():
        # The row is (label, cells); a cell is filled when the
        # capability is implemented.
        cells = row[1] if isinstance(row, tuple) else row
        implemented = sum(
            1 for cell in cells if cell not in (False, None, "", 0)
        )
        if implemented >= 6:
            reached_six.append(name)
        elif implemented == 5:
            reached_five.append(name)
    assert not reached_six, (
        f"the survey now has pipelines that implement all six: {reached_six}"
    )
    assert len(reached_five) == 2, (
        f"the thesis says two pipelines reach five of six, but the survey "
        f"has {len(reached_five)}: {reached_five}"
    )


def test_the_thinnest_capabilities_are_the_ones_the_prose_names():
    """C5 and C6 are named as the two thinnest, so they have to be.

    The chapter originally named multi-signal fusion and post-hoc
    repair as the two thinnest. The survey says otherwise: post-hoc
    repair has four implementations, selective prediction five, and
    multi-signal fusion eight. A reader who counts the bars in Figure
    7.2 finds the prose wrong.
    """
    counts = survey.coverage_counts()
    labels = [c[0] for c in survey.CAPABILITIES]
    order = sorted(range(len(counts)), key=lambda i: counts[i])
    assert labels[order[0]] == "C5", (
        f"the thinnest capability is {labels[order[0]]}, not C5"
    )
    assert labels[order[1]] == "C6", (
        f"the second thinnest capability is {labels[order[1]]}, not C6"
    )
    assert labels[order[2]] == "C4", (
        f"the third thinnest capability is {labels[order[2]]}, not C4"
    )


def test_only_crag_implements_the_three_thinnest_capabilities():
    """The chapter says only one surveyed pipeline has all three."""
    grid = survey.capability_grid()
    all_three = [
        name for name, cells in grid
        if cells[3] and cells[4] and cells[5]
    ]
    assert all_three == ["CRAG"], (
        f"the survey now has {len(all_three)} pipelines with C4, C5 and "
        f"C6: {all_three}"
    )


def test_the_book_states_the_computed_counts():
    """The rendered book has to carry the numbers the survey gives."""
    from docx import Document

    book = ROOT / "Book" / "CRISP_test.docx"
    if not book.exists():
        pytest.skip(f"the book has not been built at {book}")
    doc = Document(str(book))
    text = "\n".join(p.text for p in doc.paragraphs)

    assert f"only {C.word(C.COVERAGE['C5'])} of the " in text, (
        "the coverage of C5 in the book does not match the survey"
    )
    assert f"next, with {C.word(C.COVERAGE['C6'])}," in text, (
        "the coverage of C6 in the book does not match the survey"
    )
    assert f"third thinnest, with {C.word(C.COVERAGE['C4'])}" in text, (
        "the coverage of C4 in the book does not match the survey"
    )
    assert f"of the {C.N_PIPELINE_WORDS} published pipelines" in text, (
        "the pipeline count in the book does not match the survey"
    )


def test_the_surveys_own_prose_names_the_thinnest_in_the_right_order():
    """The survey's prose must rank the capabilities as its grid does.

    The survey file is the authority for the book's tables, and it is
    also read as a standalone document by a reader. Its own §2 grid
    makes the ranking computable, so any statement about which
    categories are thinnest has to follow that grid. It once did not:
    the headline called C4 and C5 the two thinnest, while the grid
    beneath it showed C5=4, C6=5 and C4=8.
    """
    doc = survey.SURVEY.read_text(encoding="utf-8")
    # The markdown wraps, so the assertions are made against a version
    # of the text with the line breaks and the quote markers removed.
    flat = " ".join(doc.replace(">", " ").split())

    # The headline names the thinnest categories as C5 and C6, and calls
    # C4 the third thinnest.
    assert "two thinnest categories are **C5 (post-hoc span rewriting)**" in flat
    assert "**C6 (selective prediction)**" in flat
    assert "**C4 (learned multi-signal fusion)** is the third thinnest" in flat

    # The section headings must agree with that order.
    assert "## 3.4 The mitigation comparison (C5) — the thinnest category" in doc
    assert ("## 3.3 The fusion comparison (C4) — the third thinnest category"
            in doc)


def test_the_survey_states_that_only_crag_has_the_three_thinnest():
    """The survey makes the CRAG claim, so it has to be stated once."""
    doc = survey.SURVEY.read_text(encoding="utf-8")
    flat = " ".join(doc.replace(">", " ").split())
    assert ("Only one surveyed pipeline implements all three of C4, C5 and C6"
            in flat), (
        "the survey no longer states that CRAG alone has C4, C5 and C6"
    )

