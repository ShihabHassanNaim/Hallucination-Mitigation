"""The book's absolute claims must hold against the book's own tables.

A sentence of the form "none of the papers that evaluate on HaluEval
reports an ROC-AUC" is a claim about every row of the table that
follows it. If one row contradicts it, the chapter tells the reader
two different stories and a reviewer who counts the rows finds it.

Two defects of this kind were in the book. The HaluEval section
claimed that no published paper reports an ROC-AUC on HaluEval, while
the table beside it carried a Decoding Memories row with an AUROC and
the chapter's own prose named a second paper reporting an AUC. And the
architectural chapter claimed that a per-claim provenance record has
"no analogue in any of the surveyed pipelines", which the survey does
not record for any pipeline and therefore cannot support.

The checks here read the book's prose and its tables and refuse the
contradiction.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "Book" / "CRISP_test.docx"
SRC = ROOT / "scripts" / "thesis"
PKG_ROOT = ROOT / "scripts"

import sys  # noqa: E402

if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


def _sources() -> list[Path]:
    return sorted(SRC.glob("*.py"))


def _prose() -> str:
    """The prose of every chapter module, joined.

    The chapter prose is written as a series of adjacent Python string
    literals, one clause per line, so a sentence is split across several
    lines in the source. The text is joined before it is searched, or a
    claim written across a line break would not be found.
    """
    chunks = []
    for p in _sources():
        text = p.read_text(encoding="utf-8")
        # Join the adjacent string literals of one argument list.
        text = re.sub(r'"\s*\n\s*"', "", text)
        chunks.append(text)
    return "\n".join(chunks)


def test_no_source_claims_no_published_halueval_reports_an_auc():
    """The HaluEval section must not deny the AUROC rows it prints."""
    text = _prose()
    for pattern in (
        r"none of the papers that evaluate on HaluEval reports an ROC-AUC",
        r"no published HaluEval result reports an ROC-AUC",
    ):
        assert not re.search(pattern, text), (
            f"the book still claims '{pattern}', but its own table "
            f"carries AUROC rows"
        )


def test_the_halueval_table_lists_every_paper_that_reports_an_auc():
    """The AUROC rows the book names must be the ones it prints."""
    from thesis.chapters_7_9 import _halueval_rows  # type: ignore

    rows = _halueval_rows()
    named = [r for r in rows if "AUROC" in r[2] or "AUC" in r[2]]
    systems = [r[0] for r in named]
    assert "Decoding Memories" in systems, (
        "the HaluEval table no longer carries the Decoding Memories AUROC"
    )
    assert "The Semantic Illusion" in systems, (
        "the HaluEval table no longer carries the Semantic Illusion AUC"
    )
    # Only the three papers that report an AUC on HaluEval may claim one.
    # Every other published row must report a different metric.
    for row in rows:
        if row[0] in ("This project", "Decoding Memories",
                      "The Semantic Illusion"):
            continue
        if row[0].startswith("Factcheck-Bench") or row[0].startswith("GPTLODS"):
            continue
        assert "AUC" not in row[2], (
            f"{row[0]} is listed as reporting {row[2]}, which would "
            f"contradict the section's claim that only two published "
            f"results report an AUROC on HaluEval"
        )


def test_the_provenance_claim_is_scoped_to_the_survey():
    """The provenance record may not claim to have no analogue.

    The survey records no provenance reading for the published
    pipelines, so it cannot support a claim that none of them has the
    property. The book may state the property as CRISP's own, and may
    say that the survey records it for no published pipeline, but it may
    not claim that the capability is absent from systems the survey
    never examined on that dimension.
    """
    text = _prose()
    assert "no analogue in any of the" not in text, (
        "the book still claims the provenance record has no analogue in "
        "the surveyed pipelines, which the survey does not record"
    )
    assert "property that none of the" not in text, (
        "the book still claims that no surveyed pipeline provides the "
        "provenance record"
    )
    assert "property none of the" not in text, (
        "the abstract still claims that no surveyed pipeline provides the "
        "provenance record"
    )
    assert "provenance record that none" not in text, (
        "the contributions still claim that no surveyed pipeline provides "
        "the provenance record"
    )


def test_the_reverification_claim_is_the_one_the_survey_supports():
    """The uniqueness argument must rest on a dimension the survey has.

    The head-to-head table records "Re-verifies own edit" for CRISP and
    for none of the four published pipelines beside it, so that is the
    dimension on which the chapter may argue uniqueness.
    """
    sys_path = str(SRC)
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from thesis.comparison import head_to_head_table  # noqa: E402

    headers, body, _note = head_to_head_table()
    assert headers[0] == "Dimension"
    row = next(r for r in body if r[0] == "Re-verifies own edit")
    assert row[1] == "Yes", "CRISP must re-verify its own edit"
    for cell in row[2:]:
        assert cell == "No", (
            f"a published pipeline is listed as re-verifying its own edit: "
            f"{headers[row.index(cell)]}"
        )
    # The text may lean on this dimension.
    assert "re-verifies the repair it has made" in _prose(), (
        "the chapter no longer argues the case that the survey supports"
    )


def test_the_book_still_states_the_architectural_headline():
    """The fixes above must not have cost the book its main claim."""
    text = _prose()
    assert "none implements all" in text, (
        "the architectural headline claim has been lost"
    )
    assert "re-verifies its own edit" in text or (
        "re-verifies its own repair" in text
    ), "the re-verification argument has been lost"
