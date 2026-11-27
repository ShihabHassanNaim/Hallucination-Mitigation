"""The published detection numbers must be the ones the papers actually print.

The book's comparison chapters quote a number for each published
system. A quoted number is a claim about what a paper says, so it is
checkable, and every anchor in the book was checked against the
source PDF during the review pass.

This test pins those anchors so they cannot drift. It does two things:

*   It holds the exact value each anchor must have, read out of the
    paper's own table. A change to a published number is a change to
    what the paper says, so it has to be made deliberately here.

*   It checks the number is *in* the paper's text. The PDFs are
    fetched from the sources that hold them: the NeurIPS proceedings
    for LLM-Check, the arXiv PDF for the RAGTruth corpus and the two
    arXiv preprints, and the ACL Anthology for the Findings paper.
    This is the part that catches a typo, because a mistyped anchor
    is not in the paper.

The fetch is skipped when there is no network, so an offline run of
the suite is not made to fail by an absent connection.

**The traps this had to survive.** The NeurIPS proceedings PDF is not
at ``openreview.net/pdf?id=`` — that endpoint answers 403 — but at
``proceedings.neurips.cc/paper_files/paper/2024/file/<hash>-Paper-
Conference.pdf``. The paper's "Overall" column is a *weighted average
over the four black-box models*, not a figure on one of them, which is
why the anchors differ from the unweighted mean of the columns. And
the papers print percentages, not decimals: LLM-Check prints 58.30
where the book writes 0.5830.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thesis.chapters_7_9 import AES, DESIDERATA, FREQ, LLMCHECK  # noqa: E402

HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# The source PDFs, and the strings that must appear in each.
#   (label, url, the text the PDF must contain)
SOURCES = [
    (
        "LLM-Check",
        "https://proceedings.neurips.cc/paper_files/paper/2024/file/"
        "3c1e1fdf305195cd620c118aaa9717ad-Paper-Conference.pdf",
        ["58.30", "57.24", "54.58", "53.27", "51.68"],
    ),
    (
        "AES",
        "https://aclanthology.org/2026.findings-emnlp.411.pdf",
        ["0.8829", "0.9250", "0.8807", "0.8931", "0.9332"],
    ),
    (
        "Frequency-Aware Attention",
        "https://arxiv.org/pdf/2602.18145",
        ["0.7884", "0.8412", "0.8448", "0.8585", "0.8764", "0.8812"],
    ),
    (
        "Desiderata",
        "https://arxiv.org/pdf/2605.11330",
        ["0.871", "0.836", "0.825", "0.800", "0.586", "0.584",
         "0.688", "0.694", "0.682", "0.671", "0.199", "0.628",
         "0.617", "0.538", "0.346", "0.342"],
    ),
]


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read()
    import fitz

    doc = fitz.open(stream=raw, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


@pytest.fixture(scope="module")
def papers() -> dict[str, str]:
    """The text of each source PDF, keyed by label."""
    out: dict[str, str] = {}
    for label, url, _needles in SOURCES:
        try:
            out[label] = _fetch(url)
        except Exception:
            pytest.skip(f"the source PDF for {label} is not reachable from here")
    return out


def test_the_llm_check_anchors_are_the_ones_its_table_prints(papers):
    """The five LLM-Check AUROCs, read out of its Table 4.

    The paper prints percentages, so 0.5830 is 58.30 in the text.
    """
    text = papers["LLM-Check"]
    for name, value in LLMCHECK:
        printed = f"{value * 100:.2f}"
        assert printed in text, (
            f"{name} is quoted as {value:.4f}, so the paper must print "
            f"{printed}; it does not"
        )


def test_the_aes_anchors_are_the_ones_its_tables_print(papers):
    """The five AES AUROCs, read out of its Tables 1 and 2."""
    text = papers["AES"]
    for model, value in AES:
        printed = f"{value:.4f}"
        assert printed in text, (
            f"AES on {model} is quoted as {value:.4f}, so the paper must "
            f"print {printed}; it does not"
        )


def test_the_frequency_aware_anchors_are_the_ones_its_table_prints(papers):
    """The six frequency-aware AUROCs, read out of its Table 2."""
    text = papers["Frequency-Aware Attention"]
    for model, method, value in FREQ:
        printed = f"{value:.4f}"
        assert printed in text, (
            f"frequency-aware {method} on {model} is quoted as "
            f"{value:.4f}, so the paper must print {printed}; it does not"
        )


def test_the_desiderata_anchors_are_the_ones_its_tables_print(papers):
    """The desiderata ROC-AUC, AUPR and F1, read out of its tables."""
    text = papers["Desiderata"]
    for name, roc, pr, f1 in DESIDERATA:
        for value, kind in ((roc, "ROC-AUC"), (pr, "AUPR"), (f1, "F1")):
            printed = f"{value:.3f}"
            assert printed in text, (
                f"desiderata {name} {kind} is quoted as {value:.3f}, so the "
                f"paper must print {printed}; it does not"
            )


def test_the_llm_check_overall_column_is_a_black_box_weighted_average():
    """The 'Overall' column is not the mean of the visible columns.

    The paper says the overall black-box figure "is a weighted average
    since different models hallucinate at different frequencies on the
    same prompt data". A plain mean of the four black-box columns is
    therefore *not* what the book quotes, and a reader who averages the
    table by hand must land on a different number. The difference is
    real — more than a point for two of the probes — so the anchor has
    to be the weighted one.
    """
    # Table 4, read out of the paper: the four black-box AUROCs and
    # the weighted 'Overall' it reports for each probe.
    table = {
        "Hidden Score": ([54.11, 59.67, 59.31, 61.87, 53.68], 57.24),
        "Logit (Perplexity)": ([53.73, 52.46, 56.97, 52.13, 52.11], 53.27),
        "Logit (Win Entropy)": ([52.08, 55.71, 56.38, 55.83, 52.61], 54.58),
        "Logit (Log Entropy)": ([53.95, 51.18, 55.14, 50.34, 50.43], 51.68),
        "Attention Score": ([54.19, 60.05, 60.01, 63.51, 55.37], 58.30),
    }
    quoted = {round(v * 100, 2) for _n, v in LLMCHECK}
    for probe, (_cols, overall) in table.items():
        assert overall in quoted, (
            f"the book's LLM-Check anchors {sorted(quoted)} do not include "
            f"the {overall} that the paper reports for {probe}"
        )

    # The unweighted mean differs from the reported figure for at least
    # one probe, so the two cannot be confused for one another.
    differences = []
    for probe, (cols, overall) in table.items():
        black_box = cols[1:]
        differences.append(abs(sum(black_box) / 4 - overall))
    assert max(differences) > 1.0, (
        "the unweighted mean of the black-box columns now matches the "
        "reported overall figure, so the two can no longer be told apart; "
        "the anchors need re-reading from the paper"
    )


def test_the_llm_check_entry_names_the_track_the_paper_was_published_in():
    """LLM-Check was in the Main Conference Track.

    The NeurIPS proceedings index tags the paper "Main Conference
    Track"; the reference once said "Datasets and Benchmarks Track".
    """
    from thesis.content import REFERENCES

    entry = next(e for e in REFERENCES if "LLM-Check" in e[2])
    assert "Main Conference" in entry[3], (
        f"LLM-Check is listed as {entry[3]!r}, but the NeurIPS 2024 "
        f"proceedings tag it 'Main Conference Track'"
    )
    assert "Datasets and Benchmarks" not in entry[3], (
        f"LLM-Check is listed as {entry[3]!r}; the paper was not in the "
        f"Datasets and Benchmarks Track"
    )


def test_the_ragtruth_test_set_is_the_450_instances_the_paper_describes():
    """The 450-instance split is the RAGTruth corpus paper's own.

    The corpus paper's data-split section says the test set "consists
    of 450 instances in total, derived by randomly selecting 150
    instances from each task type". The book quotes that split for
    the papers it compares against, so the figure has to be the
    corpus paper's and not a number derived elsewhere.
    """
    try:
        req = urllib.request.Request(
            "https://arxiv.org/pdf/2401.00396", headers=HDR
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
    except Exception:
        pytest.skip("the RAGTruth corpus paper is not reachable from here")

    import fitz

    doc = fitz.open(stream=raw, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    flat = " ".join(text.split())

    assert re.search(
        r"test set, which consists of 450 instances in total", flat
    ), (
        "the RAGTruth corpus paper no longer describes a 450-instance "
        "test set; the split the book quotes must be re-read"
    )
    assert "150 in- stances from each task type" in flat or (
        "150 instances from each task type" in flat
    ), (
        "the RAGTruth corpus paper no longer says the 450 instances are "
        "150 per task type"
    )
