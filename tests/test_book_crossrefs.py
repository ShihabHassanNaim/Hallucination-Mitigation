"""Every cross-reference in the book must point at something that exists.

A cross-reference is a claim that something exists: "Table 7.7", "the
Section 5.7 figures". If the thing it names is absent, the reader is
sent to a place that is not there.

The check runs against the built .docx, because that is the artefact
the reader receives. The build is the expensive part, so the book is
built once for the module and every test reads it.

The test asserts the rule, not the current state: it finds dangling
references rather than checking a fixed list.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "Book" / "CRISP_test.docx"

TABLE_REF = re.compile(r"\bTable[s]? (\d+\.\d+)(?: to (\d+\.\d+))?")
FIGURE_REF = re.compile(r"\bFigure[s]? (\d+\.\d+)(?: to (\d+\.\d+))?")
CHAPTER_REF = re.compile(r"\bChapter (\d+)\b")
SECTION_REF = re.compile(r"\b[Ss]ection (\d+(?:\.\d+)*)")


def _visible(p) -> str:
    """The paragraph text as a reader sees it.

    Caption paragraphs carry a hidden SEQ field whose cached result is
    the field's own number. Word omits it, python-docx does not, so it
    has to be excluded or every caption reads "Table 3.11".
    """
    return "".join(r.text for r in p.runs if not r.font.hidden)


@pytest.fixture(scope="module")
def book():
    """The built book, built once for the module."""
    if not BOOK.exists():
        subprocess.run(
            [sys.executable, "scripts/build_thesis_book.py",
             "--out", str(BOOK)],
            cwd=ROOT, check=True, capture_output=True,
        )
    doc = Document(BOOK)

    tables: set[str] = set()
    figures: set[str] = set()
    headings: set[str] = set()
    heading_titles: dict[str, str] = {}
    chapters: set[str] = set()
    prose: list[str] = []
    table_captions: dict[str, str] = {}
    figure_captions: dict[str, str] = {}

    for p in doc.paragraphs:
        t = _visible(p).strip()
        if not t:
            continue
        style = p.style.name if p.style else ""
        if style == "Caption":
            if t.startswith("Table "):
                number = t.split(":", 1)[0].replace("Table ", "").strip()
                tables.add(number)
                table_captions[number] = t
            elif t.startswith("Figure "):
                number = t.split(":", 1)[0].replace("Figure ", "").strip()
                figures.add(number)
                figure_captions[number] = t
        elif style.startswith("Heading"):
            m = re.match(r"(\d+(?:\.\d+)*)\s+(.*)", t)
            if m:
                headings.add(m.group(1))
                heading_titles[m.group(1)] = m.group(2)
        else:
            prose.append(t)
        if t.startswith("CHAPTER "):
            chapters.add(t.split()[1])

    return {
        "tables": tables, "figures": figures, "headings": headings,
        "heading_titles": heading_titles, "chapters": chapters,
        "prose": prose,
        "table_captions": table_captions,
        "figure_captions": figure_captions,
    }


def _dangling(book) -> list[str]:
    problems: list[str] = []
    for text in book["prose"]:
        for m in TABLE_REF.finditer(text):
            for g in m.groups():
                if g and g not in book["tables"]:
                    problems.append(f"Table {g}")
        for m in FIGURE_REF.finditer(text):
            for g in m.groups():
                if g and g not in book["figures"]:
                    problems.append(f"Figure {g}")
        for m in CHAPTER_REF.finditer(text):
            if m.group(1) not in book["chapters"]:
                problems.append(f"Chapter {m.group(1)}")
        for m in SECTION_REF.finditer(text):
            ref = m.group(1)
            # A reference may name a subsection that exists, or a parent
            # whose children exist. Both are acceptable.
            if ref in book["headings"]:
                continue
            if any(h.startswith(ref + ".") for h in book["headings"]):
                continue
            problems.append(f"Section {ref}")
    return problems


def test_the_book_builds_and_has_content(book):
    assert len(book["chapters"]) == 10, book["chapters"]
    assert len(book["tables"]) >= 20
    assert len(book["figures"]) >= 15
    assert len(book["prose"]) > 300


def test_no_cross_reference_points_at_something_absent(book):
    problems = _dangling(book)
    assert not problems, (
        f"the book refers to {sorted(set(problems))}, which do not exist"
    )


def test_a_dangling_reference_is_detected(book):
    """The check must bite: a reference to a table that does not exist
    has to be reported, or the test above passes vacuously."""
    from copy import deepcopy

    broken = deepcopy(book)
    broken["tables"] = set(broken["tables"]) - {"7.7"}
    broken["prose"] = ["The comparison is shown in Table 7.7."]
    assert _dangling(broken) == ["Table 7.7"]


def test_the_caption_numbers_are_readable_without_the_hidden_field(book):
    """A caption reads 'Table 3.1', not 'Table 3.11'.

    The caption carries a hidden SEQ field for the List of Tables; Word
    omits it but python-docx includes it. This guards against the field
    and the literal number drifting apart.
    """
    doc = Document(BOOK)
    for p in doc.paragraphs:
        if not (p.style and p.style.name == "Caption"):
            continue
        t = _visible(p).strip()
        if not t.startswith("Table "):
            continue
        number = t.split(":", 1)[0].replace("Table ", "").strip()
        parts = number.split(".")
        assert len(parts) == 2, f"malformed caption number: {number!r}"
        assert parts[0].isdigit() and parts[1].isdigit(), number


# A cross-reference can name a section that exists and still be wrong: the
# number resolves, but the section it points at does not cover the thing
# the sentence is talking about. That failure is invisible to the
# dangling-reference check above, which only asks whether the target
# exists.
#
# Each entry is a regular expression that matches one reference together
# with the words that identify what it is being used for, and the heading
# that actually covers it. The heading is matched by a keyword that must
# appear in the title, so the test breaks when a section is renamed.
CONCEPT_TARGETS = [
    (r"interval rule of Section (\d+(?:\.\d+)*)", "rule", "5.7"),
    (r"verdict rule of Section (\d+(?:\.\d+)*)", "rule", "5.7"),
    (r"rule of Section (\d+(?:\.\d+)*)", "rule", "5.7"),
    (r"ablation of Section (\d+(?:\.\d+)*)", "ablation", "6.4"),
]

# The other form the book uses: "Section 5.7 on the comparison rule,
# Section 6.6 on the null mitigation result". Here the words that
# identify the concept follow the number rather than preceding it.
CONCEPT_AFTER = [
    (r"Section (\d+(?:\.\d+)*) on the comparison rule", "rule", "5.7"),
    (r"Section (\d+(?:\.\d+)*) on the null mitigation result",
     "mitigation", "6.6"),
    (r"Section (\d+(?:\.\d+)*) on why numbers", "protocol", "7.2.4"),
]


def _heading_titles(book) -> dict[str, str]:
    """The heading number mapped to its title."""
    return book["heading_titles"]


def test_a_reference_names_the_section_that_covers_the_concept(book):
    """Each concept reference resolves to a section about that concept.

    This is the check the dangling-reference test cannot make: Section
    5.6 exists, but it is about metric formulas, so a sentence that
    points at it for the comparison rule is wrong even though nothing
    dangles.
    """
    titles = _heading_titles(book)
    problems: list[str] = []
    for text in book["prose"]:
        for pattern, keyword, expected in CONCEPT_TARGETS + CONCEPT_AFTER:
            for m in re.finditer(pattern, text):
                ref = m.group(1)
                title = titles.get(ref, "")
                if ref != expected or keyword not in title.lower():
                    problems.append(
                        f"{m.group(0)!r} resolves to Section {ref} "
                        f"({title!r}), expected Section {expected}"
                    )
    assert not problems, sorted(set(problems))

# A table reference can name a table that exists and still be wrong: the
# number resolves, but the table it points at does not cover the thing the
# sentence is talking about. Inserting or removing a table renumbers every
# later table, and a reference that is not updated still resolves -- to the
# wrong table. That failure is invisible to the dangling-reference check.
#
# Each entry is a regular expression matching one reference together with
# the words that identify what it is used for, and a keyword that must
# appear in the caption of the table it names.
TABLE_CONCEPT_TARGETS = [
    (r"capabilities of Table (\d+\.\d+)", "capabilit", "7.1"),
    (r"shown in Table (\d+\.\d+)", "halueval", "7.9"),
    (r"summarised in Table (\d+\.\d+)", "blocks", "7.11"),
    (r"wins and ties in Table (\d+\.\d+)", "llm-check", "7.3"),
    (r"losses in Tables (\d+\.\d+) to (\d+\.\d+)", "attention", "7.4"),
]


def _table_captions(book) -> dict[str, str]:
    """The table number mapped to its caption.

    Tables and figures share a numbering space, so the two are kept
    apart: "Table 7.4" and "Figure 7.4" are different things.
    """
    return book["table_captions"]


def test_a_reference_names_the_table_that_covers_the_concept(book):
    """Each concept reference resolves to a table about that concept.

    This is the check the dangling-reference test cannot make: Table 7.7
    exists, but after a table is inserted it is the wrong table, and the
    sentence that points at it is wrong even though nothing dangles.
    """
    caps = _table_captions(book)
    problems: list[str] = []
    for text in book["prose"]:
        for pattern, keyword, expected in TABLE_CONCEPT_TARGETS:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                ref = m.group(1)
                caption = caps.get(ref, "")
                if ref != expected or keyword not in caption.lower():
                    problems.append(
                        f"{m.group(0)!r} resolves to Table {ref} "
                        f"({caption!r}), expected Table {expected}"
                    )
    assert not problems, sorted(set(problems))


def test_the_table_reference_check_bites(book):
    """The check must fail on a wrong-but-existing reference."""
    caps = _table_captions(book)
    # Table 7.4 is the Keerthi table; point the HaluEval sentence at it.
    wrong = "The comparison is shown in Table 7.4."
    assert "halueval" not in caps.get("7.4", "").lower()
    problems = []
    for pattern, keyword, expected in TABLE_CONCEPT_TARGETS:
        for m in re.finditer(pattern, wrong, re.IGNORECASE):
            ref = m.group(1)
            if ref != expected or keyword not in caps.get(ref, "").lower():
                problems.append(ref)
    assert problems == ["7.4"]
