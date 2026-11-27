"""The section numbering must be continuous.

A chapter that skips a number, or a subsection numbered under the wrong
section, is the kind of defect a reader notices in the table of contents
before reading a word. The numbers are written by hand in the generator,
so nothing catches a mistake in them.

The check is on the numbering itself: within each chapter the top-level
sections run 1, 2, 3 with no gaps, and each subsection's number begins
with the section it sits under.

The build is the expensive part, so the book is built once for the module.
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

NUMBER = re.compile(r"^(?P<chapter>\d+)\.(?P<rest>[\d.]*)\s")


@pytest.fixture(scope="module")
def book():
    """The built book, built once for the module."""
    if not BOOK.exists():
        subprocess.run(
            [sys.executable, "scripts/build_thesis_book.py",
             "--out", str(BOOK)],
            cwd=ROOT, check=True, capture_output=True,
        )
    return Document(BOOK)


def _headings(book) -> list[tuple[int, str, str]]:
    """Every heading as (level, number, title).

    The level is 1 for a chapter, 2 for a section, 3 for a subsection
    and 4 for a sub-subsection. The number is the string the heading
    carries before its title.
    """
    out = []
    for p in book.paragraphs:
        style = p.style.name
        if not style.startswith("Heading "):
            continue
        level = int(style.split()[-1])
        text = " ".join(p.text.split())
        m = NUMBER.match(text)
        if not m:
            # A chapter heading is "CHAPTER 7 Comparison ..." and carries
            # no dotted number.
            out.append((level, "", text))
            continue
        number = f"{m.group('chapter')}.{m.group('rest').rstrip('.')}".rstrip(".")
        out.append((level, number, text))
    return out


def test_the_top_level_sections_of_each_chapter_are_continuous(book):
    """Section 4.6 must not follow 4.4 with nothing between."""
    headings = _headings(book)
    by_chapter: dict[str, list[int]] = {}
    for level, number, _ in headings:
        if level != 2 or not number:
            continue
        chapter, _, rest = number.partition(".")
        by_chapter.setdefault(chapter, []).append(int(rest))

    problems = []
    for chapter, sections in sorted(by_chapter.items()):
        expected = list(range(1, len(sections) + 1))
        if sections != expected:
            problems.append(
                f"chapter {chapter} has sections {sections}, expected {expected}"
            )
    assert not problems, "; ".join(problems)


def test_a_subsection_sits_under_its_own_section(book):
    """Subsection 4.5.1 must be inside section 4.5, not inside 4.4."""
    headings = _headings(book)
    problems = []
    seen: set[str] = set()
    for level, number, text in headings:
        if not number:
            continue
        if level == 2:
            seen.add(number)
        elif level in (3, 4):
            parts = number.split(".")
            # The parent of a sub-subsection is the subsection above it.
            parent = ".".join(parts[:-1])
            if parent not in seen:
                problems.append(
                    f"'{text[:60]}' is numbered {number} but its parent "
                    f"section {parent} does not exist before it"
                )
            seen.add(number)
    assert not problems, "; ".join(problems)


def test_the_subsections_of_a_section_are_continuous(book):
    """Section 3.2 has 3.2.1 to 3.2.13 and nothing missing between."""
    headings = _headings(book)
    by_parent: dict[str, list[int]] = {}
    for level, number, _ in headings:
        if level != 3 or not number:
            continue
        parts = number.split(".")
        if len(parts) != 3:
            continue
        by_parent.setdefault(".".join(parts[:2]), []).append(int(parts[2]))

    problems = []
    for parent, subs in sorted(by_parent.items()):
        expected = list(range(1, len(subs) + 1))
        if subs != expected:
            problems.append(
                f"section {parent} has subsections {subs}, expected {expected}"
            )
    assert not problems, "; ".join(problems)


def test_the_numbers_of_the_sections_are_not_out_of_order(book):
    """A section that goes backwards is a sign of a mistake."""
    headings = _headings(book)
    seen: dict[str, int] = {}
    problems = []
    for level, number, text in headings:
        if level != 2 or not number:
            continue
        chapter, _, rest = number.partition(".")
        value = int(rest)
        if chapter in seen and value <= seen[chapter]:
            problems.append(f"'{text[:60]}' comes after a higher number")
        seen[chapter] = value
    assert not problems, "; ".join(problems)
