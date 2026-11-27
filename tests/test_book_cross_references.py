"""The book's internal references must resolve.

A thesis that says "as discussed in Section 5.7" must have a Section
5.7, and a figure that is referred to must be a figure that exists.
These are the errors a reader finds first and a reviewer marks down
without comment.

Three things are checked against the built document:

*   Every numbered cross-reference in the prose, "Section 4.4.2", has
    a heading with that number.

*   Every chapter referred to, "Chapter 6", has a chapter heading.

*   The figure and table numbers run without a gap within each
    chapter, so that Figure 6.7 is followed by Figure 6.8 and not by
    Figure 6.9.

The caption numbers are read from the runs that are not hidden. A
caption carries a hidden SEQ field, and its cached result is part of
the paragraph's text, so the raw text reads "Figure 3.11: ..." where
the reader sees "Figure 3.1: ...".
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "Book" / "CRISP_test.docx"

HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\b")
CHAPTER_HEADING = re.compile(r"^\s*CHAPTER\s+(\d+)\b", re.I)
XREF = re.compile(r"\bSections?\s+([0-9][0-9.]*)")
CHAPTER = re.compile(r"\bChapter\s+(\d+)\b")
CAPTION = re.compile(r"^(Figure|Table)\s+(\d+)\.(\d+)")


def visible_text(p) -> str:
    """The text of a paragraph, without the hidden runs."""
    return "".join(
        run.text for run in p.runs if not run.font.hidden
    )


@pytest.fixture(scope="module")
def book():
    if not BOOK.exists():
        pytest.skip(f"the book has not been built at {BOOK}")
    from docx import Document

    return Document(str(BOOK))


@pytest.fixture(scope="module")
def headings(book):
    """The numbers of the headings, and the chapter numbers."""
    numbers: set[str] = set()
    chapters: set[str] = set()
    for p in book.paragraphs:
        if not p.style.name.lower().startswith("heading"):
            continue
        text = p.text.strip()
        cm = CHAPTER_HEADING.match(text)
        if cm:
            chapters.add(cm.group(1))
            continue
        m = HEADING.match(text)
        if m:
            numbers.add(m.group(1))
            if "." not in m.group(1):
                chapters.add(m.group(1))
    return numbers, chapters


def test_every_section_cross_reference_resolves(book, headings):
    numbers, _chapters = headings
    body = "\n".join(p.text for p in book.paragraphs)
    broken = []
    for m in XREF.finditer(body):
        target = m.group(1).rstrip(".")
        if target not in numbers:
            broken.append(target)
    assert not broken, f"sections referenced but absent: {sorted(set(broken))}"


def test_every_chapter_cross_reference_resolves(book, headings):
    _numbers, chapters = headings
    body = "\n".join(p.text for p in book.paragraphs)
    broken = sorted({
        m.group(1) for m in CHAPTER.finditer(body)
        if m.group(1) not in chapters
    })
    assert not broken, f"chapters referenced but absent: {broken}"


def test_the_book_has_every_chapter(book, headings):
    _numbers, chapters = headings
    assert len(chapters) >= 9, f"only {len(chapters)} chapters found"


@pytest.mark.parametrize("label", ["Figure", "Table"])
def test_the_caption_numbers_run_without_a_gap(book, label):
    """Figure 6.7 is followed by Figure 6.8, and not by Figure 6.9."""
    by_chapter = collections.defaultdict(list)
    for p in book.paragraphs:
        if p.style.name.lower() != "caption":
            continue
        m = CAPTION.match(visible_text(p).strip())
        if not m or m.group(1) != label:
            continue
        by_chapter[int(m.group(2))].append(int(m.group(3)))

    assert by_chapter, f"no {label.lower()} captions found"
    for chapter, numbers in sorted(by_chapter.items()):
        assert numbers == list(range(1, len(numbers) + 1)), (
            f"{label} {chapter}.x is numbered {numbers}, expected "
            f"{list(range(1, len(numbers) + 1))}"
        )
