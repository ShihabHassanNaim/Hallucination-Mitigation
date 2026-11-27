"""Renderer that writes the CRISP thesis book to a .docx.

The formatting spec is taken from the DIU BSc project book template
(``BSc Project book format CSE DIU Latest.docx``) whose inline notes specify:

    page          Letter 8.5 x 11 in, margins L 1.25 in, R/T/B 1.0 in
    title         24 pt, Times New Roman
    chapter       20 pt, Times New Roman
    section 1.1   14 pt, camel case, line spacing 1.5
    subsection    13 pt, line spacing 1.5
    sub-sub      12 pt, line spacing 1.5
    body          12 pt, line spacing 1.15, justified
    figure cap   12 pt, centred, "Figure N.M"
    table cap    12 pt, left aligned, line spacing 1.5
    table body   12 pt
    equation     12 pt, set in a borderless table
    references   11 pt, line spacing 1.5

Everything is Times New Roman throughout.
"""

from __future__ import annotations

import copy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

FONT = "Times New Roman"

# --- sizes ------------------------------------------------------------------
SZ_TITLE = 24
SZ_CHAPTER = 20
SZ_SECTION = 14
SZ_SUBSECTION = 13
SZ_SUBSUBSECTION = 12
SZ_BODY = 12
SZ_TABLE = 12
SZ_REF = 11

LS_SECTION = 1.5
LS_BODY = 1.15
LS_REF = 1.5

BODY_W = Inches(8.5 - 1.25 - 1.0)  # 6.25 in text width


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------

def _set_font(run, size: float, bold: bool = False, italic: bool = False):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)


def _spacing(p, line: float, before: float = 0, after: float = 0):
    pf = p.paragraph_format
    pf.line_spacing = line
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)


def _no_borders(table):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "none")
        el.set(qn("w:sz"), "0")
        el.set(qn("w:space"), "0")
        borders.append(el)
    tbl_pr.append(borders)


def _grid_borders(table, sz: int = 4):
    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)
    tbl_pr.append(borders)


def _repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def _split_bold(text: str) -> list[tuple[str, bool]]:
    """Split a string on '**' markers into (text, is_bold) pairs.

    The chapter modules write the comparison verdicts as '**WIN**' so that
    they stand out in the source; the markers are turned into real bold runs
    here rather than being printed literally.
    """
    parts = text.split("**")
    out: list[tuple[str, bool]] = []
    for i, part in enumerate(parts):
        if part:
            out.append((part, i % 2 == 1))
    return out or [(text, False)]


def _page_field(paragraph):
    run = paragraph.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(fld)
    run._r.append(instr)
    run._r.append(end)
    _set_font(run, SZ_BODY)


def _field_run(paragraph, instr: str, cached: str, size: float = SZ_BODY,
               bold: bool = False, hidden: bool = False):
    """Insert a single Word field with a cached result.

    The field is written as begin / instruction / separate / cached result /
    end, which is the form Word itself produces, so that the document reads
    correctly in a reader that does not evaluate fields and updates correctly
    in one that does.
    """
    runs = []
    for kind, text in (("begin", None), ("instr", instr),
                       ("separate", None), ("text", cached),
                       ("end", None)):
        run = paragraph.add_run()
        _set_font(run, size, bold)
        run.font.hidden = hidden
        runs.append(run)
        if kind == "instr":
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
            run._r.append(el)
        elif kind == "text":
            run.text = text  # the cached field result
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
            run._r.append(el)
    return runs


def _seq_field(paragraph, label: str, number: int, bold: bool = False):
    """Insert a SEQ field. Returns the runs it created, so that the caller
    can hide them."""
    return _field_run(paragraph, f" SEQ {label} \\* ARABIC ", str(number),
                      bold=bold, hidden=True)


def _numbered_caption(paragraph, label: str, chapter: int, index: int,
                      bold: bool = False):
    """Render 'Figure N.M: caption'.

    The visible number is written literally, so the caption reads correctly
    even in a reader that does not evaluate fields.  A hidden SEQ field named
    after the label ("Figure" or "Table") is included so that Word's
    ``TOC \\c`` switch can collect the caption into the List of Figures or
    the List of Tables; the paragraph also carries the built-in Caption
    style, which that switch requires.

    Word's TOC omits hidden text when it copies a caption paragraph, so the
    collected entry reads "Figure 3.1: ..." and not the concatenation of the
    literal number with the field's cached result.
    """
    run = paragraph.add_run(f"{label} {chapter}.{index}")
    _set_font(run, SZ_BODY, bold)
    # Hidden SEQ counter, used only for the List of Figures / List of Tables.
    seq = _seq_field(paragraph, label, index, bold)
    for r in seq:
        r.font.hidden = True


def _field(paragraph, code: str, placeholder: str = ""):
    """Insert a Word field such as TOC or SEQ."""
    r1 = paragraph.add_run()
    _set_font(r1, SZ_BODY)
    f1 = OxmlElement("w:fldChar")
    f1.set(qn("w:fldCharType"), "begin")
    r1._r.append(f1)

    r2 = paragraph.add_run()
    _set_font(r2, SZ_BODY)
    it = OxmlElement("w:instrText")
    it.set(qn("xml:space"), "preserve")
    it.text = code
    r2._r.append(it)

    r3 = paragraph.add_run()
    _set_font(r3, SZ_BODY)
    f3 = OxmlElement("w:fldChar")
    f3.set(qn("w:fldCharType"), "separate")
    r3._r.append(f3)

    r4 = paragraph.add_run(placeholder)
    _set_font(r4, SZ_BODY)

    r5 = paragraph.add_run()
    _set_font(r5, SZ_BODY)
    f5 = OxmlElement("w:fldChar")
    f5.set(qn("w:fldCharType"), "end")
    r5._r.append(f5)


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------

class Thesis:
    def __init__(self, out_path: Path):
        self.doc = Document()
        self.out = Path(out_path)
        self.chapter = 0
        self.fig_no = 0
        self.tab_no = 0
        self.eq_no = 0
        self._setup_styles()
        self._setup_page()
        self._setup_footer()
        self._setup_update_fields()

    @staticmethod
    def _strip_style_numbering(style):
        """Remove the automatic outline numbering a heading style may carry."""
        ppr = style.element.find(qn("w:pPr"))
        if ppr is not None:
            numpr = ppr.find(qn("w:numPr"))
            if numpr is not None:
                ppr.remove(numpr)

    # -- setup ---------------------------------------------------------------

    def _setup_styles(self):
        st = self.doc.styles["Normal"]
        st.font.name = FONT
        st.font.size = Pt(SZ_BODY)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        st.element.rPr.rFonts.set(qn("w:cs"), FONT)
        pf = st.paragraph_format
        pf.line_spacing = LS_BODY
        pf.space_after = Pt(0)

        # The built-in heading styles are used so that Word's TOC field can
        # collect the headings; their appearance is then overridden to match
        # the template's specification (size, weight, spacing, Times New Roman).
        for name, size, before, after in (
            ("Heading 1", SZ_CHAPTER, 0, 18),
            ("Heading 2", SZ_SECTION, 12, 8),
            ("Heading 3", SZ_SUBSECTION, 10, 6),
            ("Heading 4", SZ_SUBSUBSECTION, 8, 4),
        ):
            try:
                hs = self.doc.styles[name]
            except KeyError:
                continue
            hs.font.name = FONT
            hs.font.size = Pt(size)
            hs.font.bold = True
            hs.font.italic = False
            hs.font.color.rgb = RGBColor(0, 0, 0)
            rpr = hs.element.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = OxmlElement("w:rFonts")
                rpr.append(rfonts)
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                rfonts.set(qn(attr), FONT)
            hpf = hs.paragraph_format
            hpf.line_spacing = LS_SECTION
            hpf.space_before = Pt(before)
            hpf.space_after = Pt(after)
            hpf.keep_with_next = True
            # Word's Heading styles carry a "keep with next" and a page-break
            # behaviour; the chapter heading manages its own page break.
            self._strip_style_numbering(hs)

        # Word's TOC \c switch collects a caption only if its paragraph carries
        # the built-in "Caption" style.  The style is set up to match the
        # template's specification directly, so applying it changes nothing
        # that is visible.
        try:
            cap = self.doc.styles["Caption"]
        except KeyError:
            cap = None
        if cap is not None:
            cap.font.name = FONT
            cap.font.size = Pt(SZ_BODY)
            cap.font.bold = False
            cap.font.italic = False
            cap.font.color.rgb = RGBColor(0, 0, 0)
            crpr = cap.element.get_or_add_rPr()
            crfonts = crpr.find(qn("w:rFonts"))
            if crfonts is None:
                crfonts = OxmlElement("w:rFonts")
                crpr.append(crfonts)
            for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                crfonts.set(qn(attr), FONT)
            cpf = cap.paragraph_format
            cpf.line_spacing = LS_BODY
            cpf.space_before = Pt(0)
            cpf.space_after = Pt(4)

    def _setup_page(self):
        s = self.doc.sections[0]
        s.page_width = Inches(8.5)
        s.page_height = Inches(11)
        s.left_margin = Inches(1.25)
        s.right_margin = Inches(1.0)
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)

    def _setup_footer(self):
        footer = self.doc.sections[0].footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _page_field(p)

    def _setup_update_fields(self):
        """Ask Word to refresh every field when the document is opened.

        The table of contents and the two lists of captions are fields.
        Setting ``w:updateFields`` makes Word rebuild them on open, so
        the page numbers stay right even after the document is edited.

        The element has to sit at its place in the schema's sequence, which
        is between ``w:defaultTabStop`` and ``w:characterSpacingControl``.
        An element appended at the end is outside the sequence and is
        dropped when the file is written.
        """
        settings = self.doc.settings.element
        node = settings.find(qn("w:updateFields"))
        if node is None:
            node = OxmlElement("w:updateFields")
            anchor = settings.find(qn("w:defaultTabStop"))
            if anchor is not None:
                anchor.addnext(node)
            else:
                settings.insert(0, node)
        node.set(qn("w:val"), "true")

    # -- paragraphs ----------------------------------------------------------

    def para(self, text="", size=SZ_BODY, bold=False, italic=False,
             align="justify", line=LS_BODY, before=0, after=6, indent=None,
             first_line=None):
        p = self.doc.add_paragraph()
        p.alignment = {
            "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
        }[align]
        _spacing(p, line, before, after)
        if indent is not None:
            p.paragraph_format.left_indent = Inches(indent)
        if first_line is not None:
            p.paragraph_format.first_line_indent = Inches(first_line)
        if text:
            run = p.add_run(text)
            _set_font(run, size, bold, italic)
        return p

    def rich(self, parts, size=SZ_BODY, align="justify", line=LS_BODY,
             before=0, after=6, indent=None):
        """parts: list of (text, style) where style in {'', 'b', 'i', 'bi'}."""
        p = self.para(align=align, line=line, before=before, after=after,
                      indent=indent)
        for text, st in parts:
            run = p.add_run(text)
            _set_font(run, size, "b" in st, "i" in st)
        return p

    # -- headings ------------------------------------------------------------

    # -- headings ------------------------------------------------------------

    def chapter_heading(self, text, page_break=True):
        if page_break:
            self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        self.chapter += 1
        self.fig_no = 0
        self.tab_no = 0
        # One Heading 1 paragraph carries both the chapter number and its
        # title, so that the table of contents shows a single entry per
        # chapter rather than two.
        p = self.doc.add_paragraph(style="Heading 1")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _spacing(p, 1.5, before=0, after=24)
        run = p.add_run(f"CHAPTER {self.chapter}")
        _set_font(run, SZ_CHAPTER, bold=True)
        run = p.add_run(f"\n{text}")
        _set_font(run, SZ_CHAPTER, bold=True)
        return p

    def h2(self, number, text):
        p = self.doc.add_paragraph(style="Heading 2")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _spacing(p, LS_SECTION, before=12, after=8)
        run = p.add_run(f"{number} {text}")
        _set_font(run, SZ_SECTION, bold=True)
        return p

    def h3(self, number, text):
        p = self.doc.add_paragraph(style="Heading 3")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _spacing(p, LS_SECTION, before=10, after=6)
        run = p.add_run(f"{number} {text}")
        _set_font(run, SZ_SUBSECTION, bold=True)
        return p

    def h4(self, number, text):
        p = self.doc.add_paragraph(style="Heading 4")
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        _spacing(p, LS_SECTION, before=8, after=4)
        run = p.add_run(f"{number} {text}")
        _set_font(run, SZ_SUBSUBSECTION, bold=True)
        return p

    # -- figures -------------------------------------------------------------

    def next_figure(self) -> str:
        """The number the next figure will carry.

        A sentence that refers to the figure that follows it needs the
        number before the figure is inserted. The number is the chapter
        number and the running count within the chapter, so this is the
        same string the caption will carry.
        """
        return f"{self.chapter}.{self.fig_no + 1}"

    def figure(self, image: Path, caption: str, width_in: float = 5.6,
               source: str | None = None):
        self.fig_no += 1
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _spacing(p, 1.0, before=8, after=2)
        p.add_run().add_picture(str(image), width=Inches(width_in))
        cap = self.doc.add_paragraph(style="Caption")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _spacing(cap, 1.15, before=0, after=4)
        # SEQ fields so that the List of Figures can collect these captions
        # and so that the numbers stay correct if a figure is moved.
        _numbered_caption(cap, "Figure", self.chapter, self.fig_no)
        run = cap.add_run(f": {caption}")
        _set_font(run, SZ_BODY)
        if source:
            s = self.doc.add_paragraph()
            s.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _spacing(s, 1.0, before=0, after=12)
            r = s.add_run(source)
            _set_font(r, 9, italic=True)
        else:
            _spacing(cap, 1.15, before=0, after=12)
        return self.fig_no

    # -- tables --------------------------------------------------------------

    def table(self, caption: str | None, headers: list[str], rows: list[list[str]],
              widths: list[float] | None = None, note: str | None = None,
              font_size: int = SZ_TABLE, align_first_left: bool = True):
        # A caption of None produces an unnumbered layout table. The signature
        # blocks in the front matter use this: they are furniture, not
        # numbered exhibits, and must not appear in the List of Tables.
        if caption is not None:
            self.tab_no += 1
            cap = self.doc.add_paragraph(style="Caption")
            cap.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _spacing(cap, LS_SECTION, before=12, after=4)
            _numbered_caption(cap, "Table", self.chapter, self.tab_no, bold=True)
            run = cap.add_run(f": {caption}")
            _set_font(run, SZ_BODY, bold=True)

        t = self.doc.add_table(rows=1, cols=len(headers))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        _grid_borders(t)

        hdr = t.rows[0]
        _repeat_header(hdr)
        for i, h in enumerate(headers):
            cell = hdr.cells[i]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _spacing(p, 1.0, 0, 0)
            r = p.add_run(h)
            _set_font(r, font_size, bold=True)

        for row in rows:
            cells = t.add_row().cells
            for i, val in enumerate(row):
                cell = cells[i]
                cell.text = ""
                p = cell.paragraphs[0]
                p.alignment = (WD_ALIGN_PARAGRAPH.LEFT if (i == 0 and align_first_left)
                               else WD_ALIGN_PARAGRAPH.CENTER)
                _spacing(p, 1.0, 2, 2)
                for text, bold in _split_bold(str(val)):
                    r = p.add_run(text)
                    _set_font(r, font_size, bold)

        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)

        if note:
            n = self.doc.add_paragraph()
            n.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _spacing(n, 1.0, before=3, after=12)
            r = n.add_run(note)
            _set_font(r, 10, italic=True)
        else:
            self.doc.add_paragraph().add_run().font.size = Pt(4)
        return self.tab_no

    # -- equations -----------------------------------------------------------

    def equation(self, latex: str, number_only: bool = False):
        """Set an equation in a borderless table, per the template's rule."""
        self.eq_no += 1
        t = self.doc.add_table(rows=1, cols=2)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        _no_borders(t)
        left, right = t.rows[0].cells
        left.width = Inches(5.0)
        right.width = Inches(1.25)
        for cell in (left, right):
            cell.text = ""
        p = left.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _spacing(p, 1.15, 6, 6)
        r = p.add_run(latex)
        _set_font(r, SZ_BODY, italic=not number_only)
        p2 = right.paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        _spacing(p2, 1.15, 6, 6)
        r2 = p2.add_run(f"({self.eq_no})")
        _set_font(r2, SZ_BODY)
        return self.eq_no

    # -- lists ---------------------------------------------------------------

    def bullets(self, items: list[str], size=SZ_BODY, indent=0.3):
        for it in items:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _spacing(p, LS_BODY, 0, 4)
            p.paragraph_format.left_indent = Inches(indent + 0.25)
            p.paragraph_format.first_line_indent = Inches(-0.25)
            r = p.add_run(f"\u2022  {it}")
            _set_font(r, size)

    def numbered(self, items: list[str], size=SZ_BODY, indent=0.3):
        for i, it in enumerate(items, 1):
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            _spacing(p, LS_BODY, 0, 4)
            p.paragraph_format.left_indent = Inches(indent + 0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            r = p.add_run(f"{i}.  {it}")
            _set_font(r, size)

    # -- misc ----------------------------------------------------------------

    def page_break(self):
        self.doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

    def blank_pages(self, n: int):
        for _ in range(n):
            self.page_break()

    def toc(self):
        p = self.doc.add_paragraph()
        # Collect by outline level rather than by built-in heading style:
        # the thesis uses direct formatting, so the levels are set explicitly
        # on each heading paragraph (see Thesis._outline).
        _field(p, 'TOC \\o "1-3" \\h \\z \\u',
               "Right-click and choose Update Field to build the table of contents.")

    def lof(self):
        p = self.doc.add_paragraph()
        # Collect by the "Figure" SEQ counter. The caption paragraphs carry a
        # SEQ field named "Chapter" and one named "FigureN"; Word's \\c switch
        # takes a single counter name, so the figure caption is given a
        # "Figure" SEQ of its own for the purpose of collection and the
        # visible number is written after it.
        _field(p, 'TOC \\h \\z \\c "Figure"',
               "Right-click and choose Update Field to build the list of figures.")

    def lot(self):
        p = self.doc.add_paragraph()
        _field(p, 'TOC \\h \\z \\c "Table"',
               "Right-click and choose Update Field to build the list of tables.")

    def save(self):
        self.doc.save(str(self.out))
        return self.out
