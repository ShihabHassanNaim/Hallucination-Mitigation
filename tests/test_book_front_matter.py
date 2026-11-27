"""The front matter must be present and built, not left as a placeholder.

The table of contents, the list of figures and the list of tables are
Word fields. A field that has never been resolved shows its placeholder
text, and the reader receives a book whose front matter says "Right-click
and choose Update Field to build the table of contents." That is not a
usable deliverable, and it is invisible to the cross-reference checks,
because the placeholder names nothing.

So the checks here are that each of the three lists exists, that it has
entries, that every entry carries a page number, and that the entry
matches a heading or a caption that is actually in the document.

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

PLACEHOLDER = "Update Field"
# A TOC entry ends in a tab and a page number.
ENTRY = re.compile(r"^(?P<text>.+?)\t(?P<page>\d+)$")


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


def _sections(doc):
    """The three front-matter lists, with their entries.

    The entries of each list are the paragraphs that follow its heading
    and carry one of Word's list styles.
    """
    paragraphs = doc.paragraphs
    wanted = {
        "TABLE OF CONTENTS": "toc",
        "LIST OF FIGURES": "list of figures",
        "LIST OF TABLES": "list of tables",
    }
    out = {}
    for i, p in enumerate(paragraphs):
        title = p.text.strip()
        if title not in wanted:
            continue
        entries = []
        for q in paragraphs[i + 1:]:
            style = q.style.name.lower()
            if style.startswith("toc ") or style.startswith("table of figures"):
                entries.append(q.text)
            elif q.text.strip():
                # The next heading ends the list.
                break
        out[wanted[title]] = entries
    return out


def test_the_three_lists_are_present(book):
    lists = _sections(book)
    for name in ("toc", "list of figures", "list of tables"):
        assert name in lists, f"the book has no {name}"
        assert lists[name], f"the {name} is empty"


def test_no_list_is_left_as_placeholder_text(book):
    for p in book.paragraphs:
        assert PLACEHOLDER not in p.text, (
            "the front matter still carries the placeholder text: "
            f"{p.text.strip()[:80]!r}"
        )


def test_every_entry_carries_a_page_number(book):
    lists = _sections(book)
    for name, entries in lists.items():
        for entry in entries:
            assert ENTRY.match(entry.strip()), (
                f"a {name} entry has no page number: {entry[:90]!r}"
            )


def test_the_page_numbers_are_plausible(book):
    """A page number is a page in the book, so it is neither zero nor
    beyond the last page."""
    lists = _sections(book)
    pages = []
    for entries in lists.values():
        for entry in entries:
            m = ENTRY.match(entry.strip())
            if m:
                pages.append(int(m.group("page")))
    assert pages, "no page numbers were found"
    assert min(pages) > 0, "a page number is zero"
    assert max(pages) < 400, "a page number is implausibly large"


def _caption_texts(book, label: str) -> set[str]:
    """The captions in the document that carry the given label.

    A caption is a paragraph whose text begins with the label and a
    number. A sentence in the prose that happens to begin the same way is
    not a caption, so the caption style is required as well.
    """
    out = set()
    for p in book.paragraphs:
        text = "".join(r.text for r in p.runs if not r.font.hidden).strip()
        if not re.match(rf"^{label} \d+\.\d+", text):
            continue
        if p.style.name.lower() != "caption":
            continue
        out.add(text.split("\t")[0].strip())
    return out


def test_the_contents_lists_every_chapter(book):
    """Each chapter heading in the document is in the contents."""
    lists = _sections(book)
    chapters = {
        " ".join(p.text.split())
        for p in book.paragraphs
        if p.style.name == "Heading 1"
    }
    chapters = {c for c in chapters if c.startswith("CHAPTER")}
    assert chapters, "the document has no chapter headings"
    listed = {" ".join(e.split("\t")[0].split()) for e in lists["toc"]}
    missing = {c for c in chapters if c not in listed}
    assert not missing, f"chapters missing from the contents: {sorted(missing)}"


def test_the_figure_list_covers_every_figure(book):
    """Every figure caption in the document is in the list of figures."""
    lists = _sections(book)
    captions = _caption_texts(book, "Figure")
    listed = {" ".join(e.split("\t")[0].split()) for e in lists["list of figures"]}
    missing = {c for c in captions if " ".join(c.split()) not in listed}
    assert not missing, (
        f"figures missing from the list of figures: {sorted(missing)[:5]}"
    )


def test_the_table_list_covers_every_table(book):
    """Every table caption in the document is in the list of tables."""
    lists = _sections(book)
    captions = _caption_texts(book, "Table")
    listed = {" ".join(e.split("\t")[0].split()) for e in lists["list of tables"]}
    missing = {c for c in captions if " ".join(c.split()) not in listed}
    assert not missing, (
        f"tables missing from the list of tables: {sorted(missing)[:5]}"
    )


def test_the_document_asks_word_to_refresh_the_fields(book):
    """The fields stay live, so an edit in Word refreshes the contents."""
    import zipfile
    with zipfile.ZipFile(BOOK) as z:
        settings = z.read("word/settings.xml").decode("utf-8")
    assert "updateFields" in settings, (
        "the document does not ask Word to refresh its fields on open"
    )
