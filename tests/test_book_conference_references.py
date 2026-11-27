"""The published-conference references must name the paper they cite.

The arXiv-backed references are checked by
``tests/test_book_arxiv_identifiers.py``, which queries the arXiv API.
The references published at a conference have no arXiv identifier to
query, so they are checked here against the ACL Anthology's own record
of each paper: the Anthology is a static, greppable index, and the
entries it holds are the ones a reader reaches by following a link.

The check has three parts, one per kind of error the list carried:

*   **The author.** Several entries named a researcher who did not
    write the paper — LLM-Check was attributed to a name that appears
    nowhere in its author list, and INSIDE to another.

*   **The title.** Six entries carried a title that no paper has,
    usually a plausible-sounding expansion of the real one.

*   **The page range.** The ranges were wrong on six entries, and a
    range that does not exist is as checkable an error as a wrong
    author.

The check is skipped when there is no network, so an offline run of
the suite is not made to fail by an absent connection.
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thesis.content import REFERENCES  # noqa: E402

ANTHOLOGY_BIB = "https://aclanthology.org/anthology.bib.gz"

# The reference list is a mix of arXiv preprints, conference papers,
# model checkpoints and books. The entries checked here are the ones
# published at an ACL conference, which the Anthology indexes. Each is
# (index in the list, the Anthology key it must match).
ANTHOLOGY_ENTRIES = {
    4: "manakul-etal-2023-selfcheckgpt",
    9: "su-etal-2026-attention",
    11: "li-etal-2023-halueval",
    30: "hu-etal-2024-embedding",
    31: "cheng-etal-2024-small",
    32: "beigi-etal-2024-internalinspector",
    33: "zhao-etal-2024-medico",
    35: "zhang-etal-2024-enhancing-hallucination",
    36: "wang-etal-2024-factcheck",
}


@pytest.fixture(scope="module")
def anthology() -> dict[str, dict]:
    """The Anthology's entries, as {key: {title, authors, pages}}.

    The file is a compressed copy of the Anthology's bibliography and
    is read once for the whole module, because it is ninety-eight
    megabytes.
    """
    try:
        with urllib.request.urlopen(ANTHOLOGY_BIB, timeout=120) as resp:
            raw = resp.read()
    except Exception:
        pytest.skip("the ACL Anthology index is not reachable from here")

    import gzip
    import io

    try:
        text = gzip.decompress(raw).decode("utf-8", "replace")
    except Exception:
        # A server that answered with the plain file.
        text = raw.decode("utf-8", "replace")

    marks = list(re.finditer(r"^@(\w+)\s*\{([^,]+),", text, re.M))
    out: dict[str, dict] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        block = text[m.start():end]
        title = re.search(r'\n\s*title = "(.*?)",\n', block, re.S)
        authors = re.findall(r'\n\s*author = "(.*?)",\n', block, re.S)
        pages = re.search(r'\n\s*pages = "(.*?)",\n', block)
        out[m.group(2)] = {
            "title": " ".join(title.group(1).split()) if title else "",
            "authors": " ".join(authors),
            "pages": pages.group(1) if pages else "",
        }
    return out


def _claimed_surname(author: str) -> str:
    """The surname the reference claims, with 'et al.' removed."""
    first = author.replace(" and ", ", ").split(",")[0].strip()
    first = re.sub(r"\s+et\s+al\.?$", "", first)
    return first.strip()


def _anthology_surnames(authors: str) -> set[str]:
    """The family names of an Anthology author list.

    The Anthology writes "Family, Given and Family, Given", so the
    family name is the part before the comma.
    """
    names = set()
    for chunk in authors.split(" and "):
        chunk = chunk.strip()
        if not chunk:
            continue
        family = chunk.split(",")[0].strip()
        if family:
            names.add(family.lower())
    return names


def _unescape(text: str) -> str:
    """The Anthology's brace-escaped capitals, unescaped.

    It writes "{H}alu{E}val" for "HaluEval" and "{LLM}s" for "LLMs",
    so a plain comparison would find nothing. It also writes a
    superscript in LaTeX, as in "InternalInspector $I^2$", which the
    book writes as "I²".
    """
    text = re.sub(r"\{(\w+)\}", r"\1", text)
    text = text.replace("$", "")
    # A LaTeX superscript, as in "I^2", becomes the character the book
    # uses, which is the Latin-1 one rather than the Unicode one.
    text = re.sub(r"\^(\d)", lambda m: chr(0xB2 + int(m.group(1)) - 2), text)
    return text


@pytest.mark.parametrize("number,key", sorted(ANTHOLOGY_ENTRIES.items()))
def test_the_entry_names_the_paper_the_anthology_holds(
    number: int, key: str, anthology: dict
):
    """The title, the author and the pages must match the record."""
    entry = REFERENCES[number - 1]
    author, _year, title, venue = entry
    found = anthology.get(key)
    assert found is not None, (
        f"reference {number} should point at the Anthology entry {key}, "
        f"which the Anthology does not hold"
    )

    # -- the title. The Anthology's own title, unescaped.
    real = _unescape(found["title"])
    real = real.replace("{", "").replace("}", "")
    assert title.lower().startswith(real.lower()[:40]), (
        f"reference {number} names the title {title!r}, but the Anthology "
        f"records it as {real!r}"
    )

    # -- the author.
    claimed = _claimed_surname(author)
    if " " in author:
        assert claimed.lower() in _anthology_surnames(found["authors"]), (
            f"reference {number} names {claimed}, who is not among the "
            f"authors of {key} ({found['authors'][:120]})"
        )

    # -- the pages, when the reference gives them.
    pages = re.search(r"pp\.\s*(\d+)\s*[-\u2013]\s*(\d+)", venue)
    if pages and found["pages"]:
        real_pages = found["pages"].replace("--", "-")
        assert pages.group(1) in real_pages, (
            f"reference {number} gives the page range {pages.group(0)}, "
            f"but {key} runs from {real_pages}"
        )


def test_the_llm_check_entry_names_its_authors():
    """LLM-Check was attributed to a researcher who did not write it.

    The paper is by Sriramanan, Bharti, Sadasivan, Saha, Kattakinda
    and Feizi. The entry once named Arzt, who is not among them.
    """
    entry = next(
        e for e in REFERENCES if "LLM-Check" in e[2]
    )
    assert "Sriramanan" in entry[0], (
        f"LLM-Check is attributed to {entry[0]!r}, but the paper is by "
        f"Sriramanan, Bharti, Sadasivan, Saha, Kattakinda and Feizi"
    )
    for name in ("Arzt",):
        for i, e in enumerate(REFERENCES, 1):
            assert name not in e[0], (
                f"reference {i} names {name}, who did not write the paper "
                f"it describes"
            )


def test_the_inside_entry_names_its_authors():
    """INSIDE is by Chen and colleagues, not by Grigoriadou.

    The paper is "INSIDE: LLMs' Internal States Retain the Power of
    Hallucination Detection", by Chen, Liu, Chen, Gu, Wu, Tao, Fu and
    Ye.
    """
    entry = next(
        e for e in REFERENCES if e[2].startswith("INSIDE:")
    )
    assert entry[0].startswith("Chen"), (
        f"INSIDE is attributed to {entry[0]!r}, but the paper is by "
        f"Chen, Liu, Chen, Gu, Wu, Tao, Fu and Ye"
    )
    for i, e in enumerate(REFERENCES, 1):
        assert "Grigoriadou" not in e[0], (
            f"reference {i} names Grigoriadou, who did not write the "
            f"paper it describes"
        )


def test_no_entry_claims_a_title_that_differs_from_the_paper():
    """The six titles that no paper has must not come back.

    Each of these was a plausible-sounding expansion of the real
    title. The check is on the distinctive words of the claimed title
    against the real one, so a shortened title still passes.
    """
    forbidden = [
        "Medico-centric Multimodal Knowledge-grounded Diagnosis",
        "A Comprehensive Re-evaluation of Hallucination Detection",
        "Benchmarking Fact-Checking for LLMs",
        "GPTLODS: A Framework for Using LLMs for Open Data Services",
        "Detecting and Evaluating Hallucinations in LLMs",
        "Markov Chain-based Multi-agent Debate for LLM Hallucination Detection",
        "Evaluating the Emergence of Small Language Model-based",
        "Detecting and Explaining Hallucinations in Large Language Models",
        "A Systematic Review of Hallucination Detection in Large Language Models",
        "LLMs' Internal States Reveal the Hallucination in TruthfulQA",
        "Detection of Hallucinations in Large Language Models for",
    ]
    for i, entry in enumerate(REFERENCES, 1):
        for bad in forbidden:
            assert bad not in entry[2], (
                f"reference {i} carries the title {entry[2]!r}, which no "
                f"paper has"
            )


def test_the_page_ranges_are_the_ones_the_papers_run_to():
    """A page range that does not exist is as wrong as a wrong author.

    The ranges were wrong on six entries. The check holds the exact
    ranges the Anthology records, so a range that is off by a page is
    caught as well as one that points at a different paper.
    """
    expected = {
        "SelfCheckGPT": ("9004", "9017"),
        "HaluEval: A Large-Scale": ("6449", "6464"),
        "Embedding and Gradient": ("1950", "1959"),
        "Small Agent Can Also Rock": ("14600", "14615"),
        "InternalInspector": ("12847", "12865"),
        "Medico: Towards": ("34", "45"),
        "Enhancing Hallucination Detection": ("13321", "13332"),
        "Factcheck-Bench": ("14199", "14230"),
        "When Attention Escapes": ("8678", "8696"),
    }
    for i, entry in enumerate(REFERENCES, 1):
        author, _year, title, venue = entry
        for key, (lo, hi) in expected.items():
            if key not in title:
                continue
            m = re.search(r"pp\.\s*(\d+)\s*[-\u2013]\s*(\d+)", venue)
            assert m, (
                f"reference {i} ({author}) should give the page range "
                f"{lo}-{hi}"
            )
            assert m.group(1) == lo and m.group(2) == hi, (
                f"reference {i} ({author}) gives the page range "
                f"{m.group(0)}, but the paper runs from {lo} to {hi}"
            )
