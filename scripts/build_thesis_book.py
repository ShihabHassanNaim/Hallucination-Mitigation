"""Build the CRISP thesis book.

Usage:
    python scripts/build_thesis_book.py [--out PATH] [--no-finalise]

Produces a .docx in the DIU BSc project book format: Times New Roman
throughout, Letter page, the heading hierarchy and spacing rules from the
template's inline notes, front matter in the template's order, and all
numbers taken from the versioned evaluation artefacts.

The table of contents and the two lists of captions are Word fields, and
only Word can build them, so the finished document is opened through Word
and its fields are updated before the file is closed. That step can be
turned off with ``--no-finalise`` on a machine without Word.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thesis.renderer import Thesis  # noqa: E402
from thesis.frontmatter import build_front_matter  # noqa: E402
from thesis.chapters_1_3 import chapter_1, chapter_2, chapter_3  # noqa: E402
from thesis.chapters_4_6 import chapter_4, chapter_5, chapter_6  # noqa: E402
from thesis.chapters_7_9 import (  # noqa: E402
    chapter_7, chapter_8, chapter_9, references,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        # The default is the file the verification script and the book
        # tests read. Writing anywhere else by default means a plain
        # `python scripts/build_thesis_book.py` produces a document that
        # no test looks at, and the stale copy goes unnoticed.
        default=str(ROOT / "Book" / "CRISP_test.docx"),
        help="output .docx path",
    )
    ap.add_argument(
        "--no-finalise",
        action="store_true",
        help="do not open the document in Word to build the table of contents",
    )
    args = ap.parse_args()

    out = Path(args.out)
    t = Thesis(out)

    build_front_matter(t)
    chapter_1(t)
    chapter_2(t)
    chapter_3(t)
    chapter_4(t)
    chapter_5(t)
    chapter_6(t)
    chapter_7(t)
    chapter_8(t)
    chapter_9(t)
    references(t)

    t.save()
    print(f"wrote {out}")
    print(f"  chapters : {t.chapter}")
    print(f"  figures  : {t.fig_no} in the last chapter")

    if args.no_finalise:
        print("  contents : not built (--no-finalise)")
        return 0

    # Word is the only thing that can resolve the TOC, list-of-figures and
    # list-of-tables fields, so the document is finalised through it. A
    # machine without Word still gets a valid document, but the front
    # matter would show the placeholder text.
    try:
        from thesis.word_finalise import finalise
    except ImportError:
        print("  contents : not built (pywin32 is not installed)")
        return 0
    try:
        pages = finalise(out)
    except Exception as exc:
        print(f"  contents : NOT BUILT ({exc})")
        return 1
    print(f"  contents : built, {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
