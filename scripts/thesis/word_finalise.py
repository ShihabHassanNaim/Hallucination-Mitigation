"""Open the thesis in Word, update all fields (TOC, LoF, LoT, page refs),
repaginate, and report the page count.

Word is the only thing that can build a real TOC, so the deliverable is
finalised through it rather than by python-docx.

Word is a desktop application driven over COM, and it rejects a call that
arrives while it is still busy with the previous one. Every call that can
meet that state is therefore made through :func:`_retry`, which waits and
tries again rather than failing the build.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pythoncom
import win32com.client as win32

WD_FIELD_PAGE = 33   # wdFieldPage
WD_NUM_PAGES = 4    # wdNumberOfPages
WD_STAT_PAGES = 2   # wdStatisticPages
WD_DO_NOT_SAVE = 0
WD_SAVE = -1

# Word returns this when it is busy and the call has to be tried again.
_BUSY = (-2147418111, -2147417846, -2147417851)


def _retry(fn, what: str, attempts: int = 40, wait: float = 0.75):
    """Run a COM call, retrying while Word reports that it is busy."""
    last = None
    for _ in range(attempts):
        try:
            return fn()
        except Exception as exc:  # pywintypes.com_error
            code = getattr(exc, "args", [None])[0]
            if code not in _BUSY:
                raise
            last = exc
            time.sleep(wait)
    raise RuntimeError(f"Word stayed busy during {what}") from last


def _update_fields(doc) -> None:
    """Update every field in every story of the document."""
    for story in doc.StoryRanges:
        rng = story
        while rng is not None:
            _retry(rng.Fields.Update, "field update")
            rng = _retry(lambda r=rng: r.NextStoryRange, "story traversal")


def _update_tables(doc) -> None:
    """Update the table of contents and the two lists of captions."""
    for toc in doc.TablesOfContents:
        _retry(toc.Update, "table of contents update")
    for tof in doc.TablesOfFigures:
        _retry(tof.Update, "list of captions update")


def _toc_entries(doc) -> list[dict]:
    """The entries of the table of contents, as Word built them.

    Each entry is a heading and the page it landed on. The entries are
    read back out of the finished document so that the front matter can
    be checked against what the document actually contains.
    """
    out = []
    for toc in doc.TablesOfContents:
        for para in toc.Range.Paragraphs:
            text = para.Range.Text.replace("\r", "").replace("\x07", "").strip()
            if not text:
                continue
            page = para.Range.Information(3)  # wdActiveEndPageNumber
            out.append({"text": text, "page": int(page)})
    return out


def finalise(path: Path, update_toc: bool = True) -> int:
    path = path.resolve()
    pythoncom.CoInitialize()
    word = win32.gencache.EnsureDispatch("Word.Application")
    _retry(lambda: setattr(word, "Visible", False), "showing Word")
    _retry(lambda: setattr(word, "DisplayAlerts", 0), "setting alerts")
    doc = _retry(lambda: word.Documents.Open(str(path)), "opening the document")
    try:
        if update_toc:
            _update_fields(doc)
            _update_tables(doc)
            _retry(doc.Repaginate, "repagination")
            # A second pass: the TOC changes the pagination, so the page
            # numbers in the TOC must be recomputed against the new layout.
            _update_fields(doc)
            _update_tables(doc)
            _retry(doc.Repaginate, "repagination")

        pages = _retry(lambda: doc.ComputeStatistics(WD_STAT_PAGES), "page count")
        words = _retry(lambda: doc.ComputeStatistics(0), "word count")
        _retry(lambda: doc.Save(), "saving")
    finally:
        try:
            _retry(lambda: doc.Close(SaveChanges=WD_SAVE), "closing")
        except Exception:
            pass
        try:
            _retry(lambda: word.Quit(), "quitting")
        except Exception:
            pass
        pythoncom.CoUninitialize()
    # Word clears the flag when it updates the fields, so it is put back
    # afterwards. This runs once Word has closed the file.
    _restore_update_fields(path)
    return pages


def inspect(path: Path) -> list[dict]:
    """Read the built table of contents back out of a finalised document."""
    path = path.resolve()
    pythoncom.CoInitialize()
    word = win32.gencache.EnsureDispatch("Word.Application")
    _retry(lambda: setattr(word, "Visible", False), "showing Word")
    _retry(lambda: setattr(word, "DisplayAlerts", 0), "setting alerts")
    doc = _retry(lambda: word.Documents.Open(str(path)), "opening the document")
    try:
        entries = _toc_entries(doc)
        pages = _retry(lambda: doc.ComputeStatistics(WD_STAT_PAGES), "page count")
        _retry(lambda: doc.Close(SaveChanges=WD_DO_NOT_SAVE), "closing")
        return entries
    finally:
        try:
            _retry(lambda: word.Quit(), "quitting")
        except Exception:
            pass
        pythoncom.CoUninitialize()


def _restore_update_fields(path: Path) -> bool:
    """Put the ``updateFields`` flag back after Word has saved the file.

    Word clears the flag once it has updated the fields, so a document
    that is edited afterwards keeps a table of contents whose page numbers
    no longer match. Restoring the flag makes Word rebuild the contents
    the next time the document is opened.
    """
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document(str(path))
    settings = doc.settings.element
    node = settings.find(qn("w:updateFields"))
    if node is None:
        node = OxmlElement("w:updateFields")
        # The element has to sit at its place in the schema's sequence,
        # which is between w:defaultTabStop and w:characterSpacingControl.
        # An element appended at the end is outside the sequence and is
        # dropped when the file is written.
        anchor = settings.find(qn("w:defaultTabStop"))
        if anchor is not None:
            anchor.addnext(node)
        else:
            settings.insert(0, node)
    node.set(qn("w:val"), "true")
    doc.save(str(path))
    return True



if __name__ == "__main__":
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "CRISP_Thesis_Book.docx")
    n = finalise(p)
    print(f"pages: {n}")
