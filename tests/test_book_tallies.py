"""The tallies written in the prose must agree with the tables.

A sentence such as "wins against three of the five LLM-Check baselines
and ties the remaining two" is a claim about a table that sits three
paragraphs away. If the table changes and the sentence does not, the
chapter tells the reader two different stories, and a reviewer who
counts the rows finds it.

The check reads the verdicts out of every table that has a Verdict
column, groups the rows by the published system they name, and then
checks the prose that states a tally against the group it names.

Two earlier defects are what this guards against. The conclusion said
this project was "ahead of three of the four baselines reported in the
RAGTruth paper itself" when the table shows one win, two ties and one
loss. And a paragraph in Chapter 7 said "the three against LLM-Check"
were losses, when no LLM-Check row is a loss at all.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "Book" / "CRISP_test.docx"

VERDICT = re.compile(r"\b(WIN|LOSS|TIE)\b")

# The names by which a group of rows is described in the prose.
GROUP_NAMES = {
    "LLM": "LLM-Check",
    "RAGTruth paper": "the RAGTruth paper",
    "Keerthi et al.": "Keerthi et al.",
    "Mezghan et al.": "Mezghan et al.",
    "Gao et al.": "the desiderata work",
}

NUMBER = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


@pytest.fixture(scope="module")
def book():
    if not BOOK.exists():
        pytest.skip(f"the book has not been built at {BOOK}")
    from docx import Document

    return Document(str(BOOK))


@pytest.fixture(scope="module")
def groups(book):
    """The verdicts of every table, grouped by published system.

    The same system appears in more than one table — the summary table
    of Chapter 7 and the master table of every comparison — so the
    rows are filed by the table they are in as well, and the group is
    the pair.
    """
    out: dict[tuple[str, str], list[str]] = {}
    for index, table in enumerate(book.tables):
        header = [c.text.strip() for c in table.rows[0].cells]
        if "Verdict" not in header:
            continue
        for row in table.rows[1:]:
            cells = [c.text.strip() for c in row.cells]
            if not cells:
                continue
            found = VERDICT.search(" ".join(cells))
            if not found:
                continue
            group = re.split(r"[—\-–:]", cells[0])[0].strip()
            # "LLM-aaJ" is a desiderata detector, not an LLM-Check
            # probe, so the split on a dash would file it wrongly.
            if group == "LLM" and "LLM-Check" not in cells[0]:
                group = "LLM-aaJ"
            out.setdefault((group, str(index)), []).append(
                found.group(1)
            )
    return out


def _pick(groups, name, wins, ties, losses):
    """The group with the stated shape, or None."""
    for (group, _index), vs in groups.items():
        if group != name:
            continue
        if (vs.count("WIN"), vs.count("TIE"), vs.count("LOSS")) == (
            wins, ties, losses
        ):
            return vs
    return None


def test_the_llm_check_group_has_no_losses(groups):
    """The black-box probes are the ones this project clears."""
    vs = _pick(groups, "LLM", 3, 2, 0)
    assert vs is not None, (
        "no table of the LLM-Check probes shows three wins, two ties and "
        "no losses"
    )
    assert "LOSS" not in vs


def test_the_ragtruth_paper_group_is_one_win_two_ties_one_loss(groups):
    vs = _pick(groups, "RAGTruth paper", 1, 2, 1)
    assert vs is not None, (
        "no table of the RAGTruth paper's baselines shows one win, two "
        "ties and one loss"
    )
    assert len(vs) == 4


def test_the_prose_does_not_overstate_the_ragtruth_paper_result(book):
    """The conclusion must not claim more wins than the table has."""
    text = "\n".join(p.text for p in book.paragraphs)
    bad = [
        m.group(0)
        for m in re.finditer(
            r"[^.]*ahead of three of the four[^.]*\.", text, re.S
        )
    ]
    assert not bad, f"the prose overstates the result: {bad}"


def test_the_prose_does_not_call_the_llm_check_rows_losses(book):
    text = "\n".join(p.text for p in book.paragraphs)
    bad = [
        m.group(0)
        for m in re.finditer(
            r"[^.]*the three against LLM-Check[^.]*\.", text, re.S
        )
    ]
    assert not bad, f"the prose calls LLM-Check rows losses: {bad}"


def test_no_prose_tally_names_a_group_the_tables_do_not_have(book, groups):
    """A tally that names a comparison has to have rows to count."""
    text = "\n".join(p.text for p in book.paragraphs)
    # The largest number of rows any one table gives to a group.
    biggest: dict[str, int] = {}
    for (group, _index), vs in groups.items():
        biggest[group] = max(biggest.get(group, 0), len(vs))

    for key, phrase in GROUP_NAMES.items():
        if key not in biggest:
            continue
        for m in re.finditer(
            r"([^.]*?" + re.escape(phrase) + r"[^.]*\.)", text, re.S
        ):
            s = " ".join(m.group(1).split())
            for word, value in NUMBER.items():
                if not re.search(r"\b" + word + r"\b", s):
                    continue
                if value > biggest[key] + 1:
                    pytest.fail(
                        f"the prose states {word} for {phrase!r}, but the "
                        f"table has only {biggest[key]} rows"
                    )
