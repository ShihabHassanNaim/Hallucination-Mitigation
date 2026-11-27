"""Every arXiv identifier must point at the paper the book cites.

A reference is a claim that a particular document exists at a
particular place. The book carried eleven references whose identifier
or author did not match the paper it named: the FAISS entry pointed
at a statistical-physics preprint, the Self-critiquing Plans entry
pointed at a brain-stimulation paper, and eight entries named authors
who did not write the paper they described.

These are the errors a reviewer finds by following one link, and they
are the ones that cost a paper its credibility fastest. The check
queries the arXiv API for every identifier in the reference list and
compares the paper it finds with the entry that named it.

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

API = "http://export.arxiv.org/api/query?id_list={}"


def _fetch(ident: str) -> dict | None:
    """The arXiv entry for an identifier, or None if it cannot be read."""
    try:
        with urllib.request.urlopen(API.format(ident), timeout=25) as resp:
            data = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    entries = re.findall(r"<entry>(.*?)</entry>", data, re.S)
    if not entries:
        return None
    e = entries[0]
    title = re.search(r"<title>(.*?)</title>", e, re.S)
    authors = re.findall(r"<name>(.*?)</name>", e)
    return {
        "title": " ".join(title.group(1).split()) if title else "",
        "authors": authors,
    }


def _identifiers() -> list[tuple[int, str, tuple]]:
    out = []
    for i, entry in enumerate(REFERENCES, 1):
        for m in re.finditer(r"arXiv:(\d{4}\.\d{4,5})", entry[3]):
            out.append((i, m.group(1), entry))
    return out


def _surname(author: str) -> str:
    """The family name of an arXiv author name.

    arXiv writes names as "First Last", so the family name is the
    last word.
    """
    parts = author.strip().split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return parts[-1]


def _claimed_surname(author: str) -> str:
    """The surname the reference claims, with 'et al.' removed."""
    first = author.replace(" and ", ", ").split(",")[0].strip()
    # "Valmeekam et al." names Valmeekam; "Johnson, Douze and Jégou"
    # names Johnson.
    first = re.sub(r"\s+et\s+al\.?$", "", first)
    return first.strip()


def test_the_arxiv_entries_resolve():
    """Every identifier in the reference list must exist."""
    ids = _identifiers()
    assert len(ids) >= 10, "the reference list has almost no arXiv entries"
    unavailable = []
    for i, ident, entry in ids:
        found = _fetch(ident)
        if found is None:
            unavailable.append((i, ident))
            continue
        assert found["title"], f"reference {i} names an unreadable entry"
    if unavailable and len(unavailable) == len(ids):
        pytest.skip("the arXiv API is not reachable from here")
    assert not unavailable, (
        f"these references name an identifier that does not resolve: "
        f"{unavailable}"
    )


def test_the_author_of_each_entry_is_an_author_of_the_paper():
    """The surname in the reference must be on the paper."""
    ids = _identifiers()
    checked = 0
    for i, ident, entry in ids:
        found = _fetch(ident)
        if found is None:
            continue
        author, _year, _title, _venue = entry
        # A corporate author is a legal entity, not a person.
        if " " not in author:
            continue
        claimed = _claimed_surname(author)
        surnames = {_surname(a) for a in found["authors"]}
        # The accent of a name may differ between the two records.
        norm = {s.lower().replace("é", "e").replace("è", "e")
                for s in surnames}
        checked += 1
        assert claimed.lower().replace("é", "e") in norm, (
            f"reference {i} names {claimed}, who is not an author of "
            f"arXiv:{ident} (the paper is by "
            f"{', '.join(found['authors'][:4])})"
        )
    if checked == 0:
        pytest.skip("the arXiv API is not reachable from here")


def test_the_title_of_each_entry_matches_the_paper():
    """The title in the reference must be the paper's title.

    A reference may shorten a title, so the comparison is on the
    distinctive words of the title rather than on the whole string.
    The check is on the words that would be wrong if the entry named
    a different paper.
    """
    ids = _identifiers()
    checked = 0
    for i, ident, entry in ids:
        found = _fetch(ident)
        if found is None:
            continue
        _author, _year, title, _venue = entry
        # The distinctive words of the claimed title, minus the ones
        # that appear in almost any paper title.
        stop = {"the", "a", "an", "of", "for", "and", "in", "on", "to",
                "with", "via", "towards", "toward", "is", "are", "as"}
        words = [
            w.lower() for w in re.findall(r"[A-Za-z0-9]+", title)
            if len(w) > 3 and w.lower() not in stop
        ]
        if not words:
            continue
        paper = found["title"].lower()
        missing = [w for w in words if w not in paper]
        checked += 1
        # One missing word is a paraphrase; two means a different paper.
        assert len(missing) <= 1, (
            f"reference {i} names the title {title!r}, but arXiv:{ident} "
            f"is {found['title']!r} (words not in the paper: {missing})"
        )
    if checked == 0:
        pytest.skip("the arXiv API is not reachable from here")


def test_the_faiss_reference_points_at_the_faiss_paper():
    """The most-used library in the pipeline must be cited correctly.

    This entry pointed at arXiv:1906.00016, which is a paper on
    thermalised fluids, while the entry described FAISS.
    """
    entry = next(
        e for e in REFERENCES if "Billion-scale similarity search" in e[2]
    )
    assert "1702.08734" in entry[3], (
        f"the FAISS reference points at {entry[3]}, not at the paper that "
        f"describes billion-scale similarity search with GPUs"
    )
    assert "Johnson" in entry[0], (
        f"the FAISS reference is attributed to {entry[0]}"
    )


def test_the_self_critiquing_plans_reference_points_at_the_right_paper():
    """The entry must not point at a paper on brain stimulation."""
    entry = next(
        e for e in REFERENCES
        if "Self-critiquing Their Own Plans" in e[2]
    )
    assert "2310.08118" in entry[3], (
        f"the reference points at {entry[3]}, which is not the "
        f"self-critiquing plans paper"
    )
