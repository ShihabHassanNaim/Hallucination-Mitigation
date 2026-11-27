"""The citations must name the entry they point at.

A citation is a claim: "[16]" after the word "GLiNER" says that entry
16 is the GLiNER paper. The reference list is authoritative, and the
numbering of the prose has to agree with it. The numbering drifted
twice in this project — once when an entry was removed from the middle
of the list, and once because a block of tool citations had been
written against a different ordering of the list — so the agreement is
checked here rather than left to the eye.

The check has three parts:

*   Every entry in the list is cited at least once. An entry that
    nothing points at is dead weight.

*   Every citation resolves to an entry that exists.

*   The entry a citation points at must be one that its subject names.
    The subject is the name that precedes the bracket, and the entry
    may be named by its first author, by a word of its title, or by a
    model or dataset name it carries.

The third part is the one that bites. It has to tolerate the false
positives that a naive reading produces: Python list indexing in the
source, and model names written in several pieces.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from thesis.content import REFERENCES  # noqa: E402

FILES = [
    "chapters_1_3", "chapters_4_6", "chapters_7_9", "frontmatter",
]

# A bracketed group of numbers is a citation. A group is scanned only
# when every number in it is a real citation, so that a Python index
# such as ci[0] is not read as one.
CIT = re.compile(r"\[([0-9]+(?:\s*,\s*[0-9]+)*)\]")

# The string literals of the source, which is where the prose lives.
STRING = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)

# A word that may stand for a name.
WORD = re.compile(r"[A-Za-z][A-Za-z0-9.\-]*")

# Words that are never the subject of a citation. A sentence such as
# "on the two benchmarks [10]" is not a mis-citation.
STOP = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "both", "each", "two", "three", "four", "one", "and", "or", "of",
    "in", "on", "at", "to", "for", "from", "by", "with", "as", "is",
    "are", "was", "were", "be", "been", "has", "have", "had", "not",
    "no", "but", "so", "than", "then", "when", "where", "which", "who",
    "section", "sections", "chapter", "chapters", "table", "tables",
    "figure", "figures", "appendix", "equation", "equations", "see",
    "above", "below", "also", "only", "even", "such", "other", "others",
    "here", "there", "shown", "given", "used", "using", "make", "makes",
    "same", "all", "any", "their", "his", "her", "less", "more", "most",
    "least", "very", "quite", "rather", "still", "already", "always",
    "never", "often", "sometimes", "usually", "first", "second",
    "third", "last", "next", "own", "per", "via",
}

# Words that head a section, a table or a sentence rather than name
# the thing that is being cited.
HEADINGS = {
    "against", "the", "against the", "findings", "using", "from",
    "comparison", "comparing", "compared", "after", "before", "under",
    "over", "with", "without", "about", "into", "through", "between",
    "across", "within", "while", "when", "and", "or", "but", "so",
}


def prose(name: str) -> str:
    """The text of the string literals of one source file."""
    src = (ROOT / "scripts" / "thesis" / f"{name}.py").read_text(
        encoding="utf-8"
    )
    parts = []
    for part in STRING.findall(src):
        try:
            parts.append(part.encode().decode("unicode_escape"))
        except UnicodeDecodeError:
            parts.append(part)
    return "\n".join(parts)


def entry_names(entry) -> set[str]:
    """The names by which an entry may be referred to in the text."""
    author, _year, title, venue = entry
    names: set[str] = set()

    # The first author's surname.
    surname = author.split()[0].rstrip(",")
    if len(surname) > 2:
        names.add(surname.lower())

    # The words of the title, and the parts of a hyphenated one.
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", title):
        if len(word) > 3:
            names.add(word.lower())
            if "-" in word:
                names.add(word.split("-")[0].lower())

    # A model or dataset name carried by the venue, and the name of
    # the venue itself, so that a subject that names the paper by the
    # place it appeared, such as "ACL", still resolves.
    for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", venue):
        if len(word) > 2:
            names.add(word.lower())
    # The venue itself, so that a subject that names the paper by the
    # place it appeared, such as "Findings ACL", still resolves.
    names.add(" ".join(re.sub(r"[^a-z0-9 ]", " ", venue.lower()).split()))

    # The whole title, so that a subject that names the paper rather
    # than a word of it, such as "The Semantic Illusion", matches.
    names.add(" ".join(re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()))
    return names


def subject_of(text: str, at: int) -> list[str]:
    """The run of name-like words that ends at the citation.

    The run stops at a word that is neither capitalised nor carries a
    digit, so "the two benchmarks [10]" has no subject. A word such as
    "Against" or "Findings" is a section heading rather than a name,
    so those are dropped from the front of the run.
    """
    before = text[max(0, at - 70):at]
    run: list[str] = []
    for word in reversed(WORD.findall(before)):
        if word[0].isupper() or any(c.isdigit() for c in word):
            run.append(word)
        elif word.lower() in STOP:
            continue
        else:
            break
    run = list(reversed(run))
    # A leading word that heads a section or a table is not the name
    # of the thing being cited.
    while run and run[0].lower() in HEADINGS:
        run = run[1:]
    return run


def citations(name: str):
    """Every citation in one file, as (number, subject words)."""
    text = prose(name)
    out = []
    for m in CIT.finditer(text):
        numbers = [int(x) for x in re.findall(r"\d+", m.group(1))]
        # A group that contains a zero is an index, not a citation.
        if not numbers or 0 in numbers:
            continue
        # An index into a named variable, such as RAG_CI[0], is code.
        # It sits inside the braces of a formatted value, so the
        # nearest brace before it is an opening one.
        brace = text.rfind("{", 0, m.start())
        close = text.rfind("}", 0, m.start())
        if brace > close:
            continue
        out.append((numbers, subject_of(text, m.start())))
    return out


ALL = [c for name in FILES for c in citations(name)]


def test_the_reference_list_is_not_empty():
    assert len(REFERENCES) >= 40


def test_every_entry_is_cited_at_least_once():
    cited = {n for numbers, _ in ALL for n in numbers}
    missing = sorted(set(range(1, len(REFERENCES) + 1)) - cited)
    assert not missing, f"entries never cited: {missing}"


def test_no_citation_points_past_the_end_of_the_list():
    over = sorted({n for numbers, _ in ALL for n in numbers
                   if n > len(REFERENCES)})
    assert not over, f"citations with no entry: {over}"


@pytest.mark.parametrize(
    "numbers,subject",
    ALL,
    ids=lambda v: str(v)[:60],
)
def test_a_citation_names_the_entry_it_points_at(numbers, subject):
    """The subject before the bracket must belong to the entry."""
    if not subject:
        return
    for n in numbers:
        names = entry_names(REFERENCES[n - 1])
        # The subject as it is written, with the parts joined by a
        # space and the punctuation removed.
        words = [w.lower() for w in subject]
        spaced = " ".join(words)
        joined = "".join(words)
        flat = re.sub(r"[^a-z0-9]", "", " ".join(words))

        if any(
            candidate in names
            for candidate in (spaced, joined, flat)
        ):
            continue
        # A name that the subject repeats, such as a section heading
        # followed by the name itself.
        if len(words) > 1 and len(set(words)) < len(words):
            if any(w in names for w in words):
                continue
        # The subject may name the entry by the opening words of its
        # title, as in "The Semantic Illusion".
        if any(n.startswith(spaced + " ") for n in names):
            continue
        # The subject may carry a suffix the entry does not, such as
        # "gliner-bi-base-v1.0" for "GLiNER" or "mDeBERTa-v3" for
        # "mDeBERTa-v3-base".
        if any(n.startswith(joined) for n in names):
            continue
        if any(n.startswith(flat) for n in names):
            continue
        # The subject may be a checkpoint identifier built on the
        # entry's name, such as "gliner-bi-base-v1.0" for "GLiNER".
        if any(
            w.lower().startswith(n) and len(n) > 3
            for w in words
            for n in names
        ):
            continue
        # The subject may be only the tail of the entry's name, such as
        # "Illusion" for "The Semantic Illusion", or "mDeBERTa" for
        # "mDeBERTa-v3-base". It matches when it is a word of the
        # title rather than a stray word.
        if _is_a_word_of_the_title(subject[-1], REFERENCES[n - 1]):
            continue
        # The subject may carry a qualifier the entry does not, such
        # as "InternalInspector I2" or "Findings ACL", where the paper
        # is named by its first word and the rest is the venue.
        if _leads_the_title(subject[0], REFERENCES[n - 1]):
            continue
        pytest.fail(
            f"[{n}] after {subject[-1]!r} points at "
            f"{REFERENCES[n - 1][0]} {REFERENCES[n - 1][2][:44]!r}"
        )


def _is_a_word_of_the_title(word: str, entry) -> bool:
    """Whether a word appears in the title of an entry."""
    _author, _year, title, _venue = entry
    return word.lower() in {
        w.lower() for w in re.findall(r"[A-Za-z0-9]+", title)
    }


def _leads_the_title(word: str, entry) -> bool:
    """Whether a word is the first word of an entry's title."""
    _author, _year, title, _venue = entry
    words = re.findall(r"[A-Za-z0-9]+", title)
    return bool(words) and words[0].lower() == word.lower()


def test_no_phantom_paper_is_named():
    """No paper has the title "Conformal-RAG".

    The system that provides a conformal coverage guarantee is Sinha's
    Conformal RAG Guardrails, published under the title "The Semantic
    Illusion". Naming it "Conformal-RAG" cites a paper that does not
    exist.
    """
    for name in FILES:
        assert "Conformal-RAG" not in prose(name), (
            f"{name} names a paper that does not exist"
        )


def test_no_two_entries_share_an_arxiv_identifier():
    """Two references cannot point at the same preprint.

    The list once carried two entries with the identifier
    arXiv:2312.10997 under two different author lists, so one of the
    two named an author who did not write the paper. A reviewer who
    checks the identifier finds it.
    """
    seen: dict[str, int] = {}
    for i, entry in enumerate(REFERENCES, 1):
        for m in re.finditer(r"arXiv:(\d{4}\.\d{4,5})", entry[3]):
            ident = m.group(1)
            assert ident not in seen, (
                f"references {seen[ident]} and {i} share the identifier "
                f"arXiv:{ident}, so at most one of them is right"
            )
            seen[ident] = i


def test_the_author_and_the_year_agree_with_the_venue():
    """A reference's year must not contradict its venue.

    One entry read "Keerthi et al. (2025) ... Findings of EMNLP 2026",
    which is a self-contradiction a reviewer reads at a glance.
    """
    for i, entry in enumerate(REFERENCES, 1):
        author, year, _title, venue = entry
        # An arXiv identifier carries digits that look like a year, so
        # it is removed before the venue is read.
        venue = re.sub(r"arXiv:\d{4}\.\d{4,5}", "", venue)
        # A venue that names a year of its own.
        m = re.search(r"\b(19|20)\d{2}\b", venue)
        if not m:
            continue
        venue_year = m.group(0)
        assert year == venue_year, (
            f"reference {i} ({author}) gives the year as {year} but its "
            f"venue says {venue_year}: {venue}"
        )


def test_the_attention_method_is_attributed_to_its_real_authors():
    """"When Attention Escapes" is Su et al., not anyone else.

    The book attributed the paper to a name that is not among its
    authors. A fabricated author is the most serious citation error a
    review can find, and the project's own verified record of the paper
    gives the authors as Su, Wang, Ba, Chu, Zhong, Yan and Ren.
    """
    titles = [e[2] for e in REFERENCES]
    match = [t for t in titles if t.startswith("When Attention Escapes")]
    assert match, "the reference to 'When Attention Escapes' is missing"
    author = next(
        e[0] for e in REFERENCES if e[2].startswith("When Attention Escapes")
    )
    assert author == "Su et al.", (
        f"'When Attention Escapes' is attributed to {author}, but the "
        f"paper is by Su, Wang, Ba, Chu, Zhong, Yan and Ren"
    )
    for name in FILES:
        assert "Keerthi" not in prose(name), (
            f"{name} still attributes 'When Attention Escapes' to a "
            f"author who did not write it"
        )
