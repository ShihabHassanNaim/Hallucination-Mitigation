"""The book reports two benchmarks, and nothing may quietly reintroduce a third.

The book's evaluation is reported on HaluEval and RAGTruth. TruthfulQA
was removed from the book because the closed-book setting it requires
has no corpus to retrieve from, which makes several of the reported
rates degenerate. The artefacts for it still exist and are still
legitimate research records, so this test does not forbid the file; it
forbids the *book* from reporting on it.

The test asserts the rule, not the current state: it checks that the
book's own source files carry no TruthfulQA content and that the
figures the book embeds draw only the two reported datasets.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "scripts" / "thesis"

# The files that produce the book's text.
BOOK_SOURCES = [
    THESIS / "frontmatter.py",
    THESIS / "chapters_1_3.py",
    THESIS / "chapters_4_6.py",
    THESIS / "chapters_7_9.py",
]

# The plot scripts that produce the book's figures.
FIGURE_SOURCES = [
    ROOT / "scripts" / "plot_results.py",
    ROOT / "scripts" / "plot_headline.py",
    ROOT / "scripts" / "plot_architectural_win.py",
]

# A reference to TruthfulQA in the bibliography is fine: those are real
# published papers cited in the related-work chapter. What must not
# appear is TruthfulQA as a benchmark the book evaluates on.
BENCHMARK_USE = re.compile(
    r"(tqa_|TruthfulQA \(closed-book\)|closed-book TruthfulQA"
    r"|on TruthfulQA|and TruthfulQA|TruthfulQA and)",
    re.IGNORECASE,
)


def test_no_book_chapter_reports_on_truthfulqa():
    """No chapter may quote a TruthfulQA number or name it as a benchmark."""
    for path in BOOK_SOURCES:
        source = path.read_text(encoding="utf-8")
        hits = BENCHMARK_USE.findall(source)
        assert not hits, (
            f"{path.name} still reports on TruthfulQA: {hits[:4]}. The book "
            f"is reported on HaluEval and RAGTruth only."
        )


def test_no_book_chapter_quotes_a_truthfulqa_measurement():
    """The `tqa_` keys are the closed-book artefact's numbers."""
    for path in BOOK_SOURCES:
        source = path.read_text(encoding="utf-8")
        assert '"tqa_' not in source and "'tqa_" not in source, (
            f"{path.name} still reads a number from the TruthfulQA artefact."
        )


def test_the_figures_draw_only_the_two_reported_datasets():
    """The book figures must not draw a dataset the book does not report on."""
    for path in FIGURE_SOURCES:
        source = path.read_text(encoding="utf-8")
        assert "metrics_truthfulqa" not in source, (
            f"{path.name} still loads a TruthfulQA metrics file. The book's "
            f"figures draw HaluEval and RAGTruth only."
        )


def test_the_results_plot_declares_the_book_subset():
    """The restriction has to be explicit, not implicit in a filter.

    If the subset is deleted but the filters that use it stay, the file
    raises at import. If the filters are deleted but the subset stays,
    the figures silently go back to three datasets. So the subset and
    its use are checked together.
    """
    source = (ROOT / "scripts" / "plot_results.py").read_text(encoding="utf-8")
    assert "BOOK_DATASETS" in source, (
        "plot_results.py no longer declares which datasets the book uses"
    )
    assert source.count("BOOK_DATASETS") >= 2, (
        "BOOK_DATASETS is declared but never used, so the figures are no "
        "longer restricted"
    )


def test_the_audit_dataset_figure_lists_two_datasets():
    """The audit dataset figure shows the sets the book evaluated on."""
    source = (ROOT / "scripts" / "plot_audit_figures.py").read_text(encoding="utf-8")
    block = source.split("def fig_datasets()", 1)[1]
    assert "truthfulqa" not in block.lower(), (
        "the audit dataset figure still lists TruthfulQA as a third dataset"
    )
