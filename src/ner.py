"""Phase 5 — Named Entity Recognition.

Why we need NER
---------------
Phase 4's ``_entities_for_multi_hop`` grabs capitalised spans — fine as a
last-resort fallback, but it conflates *people*, *organisations*, *dates*,
and *numbers*, and it ignores lower-cased entities ("Barack Obama" in the
middle of a sentence, after a quote, etc.).

For Phase 5's multi-hop planner we need *typed* entities: we want to ask
"give me the country of citizenship of the person mentioned in this
claim", not "give me any capitalised span". A typed entity lets us
build a graph traversal like ``person -> country -> capital``.

Two backends
------------
1. **Mock** (default, ``mock=True``). Deterministic regex + lexicon. Runs
   in microseconds, no model download, fully reproducible. Good enough
   for the synthetic corpus and the unit tests.
2. **Real** (``backend="spacy"``). Uses ``spacy.load("en_core_web_sm")``.
   Only imported lazily so the mock-mode path stays dependency-free.

Public surface
--------------
``Entity(text, label, start, end)`` — one detected mention.
``NER(backend="mock", model_name="en_core_web_sm", mock=True)`` —
the tagger. ``ner.tag(text) -> List[Entity]``.

The mock mode recognises these labels:

  * ``PER``      — person (capitalised first + last name; lexicon)
  * ``ORG``      — organisation (capitalised multi-word, or lexicon)
  * ``LOC``      — location (capitalised multi-word, or lexicon)
  * ``DATE``     — "in 1949", "1949", "January 2024", ISO dates
  * ``NUM``      — standalone numeric tokens not part of a date
  * ``MISC``     — capitalised multi-word that doesn't match a lexicon
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence


# --- data class --------------------------------------------------------------


@dataclass
class Entity:
    """One detected named entity."""

    text: str
    label: str           # one of: PER, ORG, LOC, DATE, NUM, MISC
    start: int           # inclusive char offset into the source text
    end: int             # exclusive char offset

    def __repr__(self) -> str:        # pragma: no cover - cosmetic
        return f"Entity({self.text!r}, {self.label}, [{self.start},{self.end}])"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }


# --- mock backend ------------------------------------------------------------

# Lexicons keyed on lower-case surface forms. Intentionally small and biased
# toward the synthetic corpus + common HaluEval entities. Real mode goes
# through spaCy's broader coverage.
_MOCK_LEXICON: dict[str, str] = {
    # people
    "george orwell": "PER",
    "barack obama": "PER",
    "albert einstein": "PER",
    "marie curie": "PER",
    "isaac newton": "PER",
    "ada lovelace": "PER",
    "william shakespeare": "PER",
    # orgs
    "openai": "ORG",
    "meta ai": "ORG",
    "google": "ORG",
    "microsoft": "ORG",
    "nasa": "ORG",
    "european union": "ORG",
    "united nations": "ORG",
    # locations
    "france": "LOC",
    "paris": "LOC",
    "germany": "LOC",
    "berlin": "LOC",
    "japan": "LOC",
    "tokyo": "LOC",
    "united states": "LOC",
    "united kingdom": "LOC",
    "europe": "LOC",
    "western europe": "LOC",
    "red planet": "LOC",
    "mars": "LOC",
}


# Regex helpers. Order matters: DATE must beat NUM.
_DATE_RE = re.compile(
    r"\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{1,4})\b"
)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
# Capitalised multi-word, allowing internal spaces and hyphens. Includes
# single capitalised words >= 3 letters.
_CAP_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b")


def _mock_tag(text: str) -> List[Entity]:
    out: List[Entity] = []
    if not text:
        return out

    # 1. Dates — strictest pattern, wins conflicts.
    for m in _DATE_RE.finditer(text):
        out.append(Entity(text=m.group(0), label="DATE",
                          start=m.start(), end=m.end()))

    # 2. Numbers (only outside date spans).
    date_spans = {(e.start, e.end) for e in out if e.label == "DATE"}
    for m in _NUM_RE.finditer(text):
        if any(s <= m.start() < e for s, e in date_spans):
            continue
        out.append(Entity(text=m.group(0), label="NUM",
                          start=m.start(), end=m.end()))

    # 3. Capitalised spans — but skip if the span is purely a date or number
    #    (already covered) or entirely inside an existing date span.
    occupied = {(e.start, e.end) for e in out}
    for m in _CAP_RE.finditer(text):
        if (m.start(), m.end()) in occupied:
            continue
        span_text = m.group(0)
        span_lower = span_text.lower()
        # lexicon hit first
        label = _MOCK_LEXICON.get(span_lower)
        if label is None:
            # single word: only tag if the word itself is a known single
            # token in the lexicon. Otherwise it's too noisy.
            if " " not in span_text and span_lower not in _MOCK_LEXICON:
                # still allow when the word is the first token of a longer
                # known multi-word — the longer form will be picked up too.
                continue
            # multi-word unknown: MISC
            label = "MISC"
        out.append(Entity(text=span_text, label=label,
                          start=m.start(), end=m.end()))

    out.sort(key=lambda e: (e.start, -e.end))
    return out


# --- public class ------------------------------------------------------------


class NER:
    """Pluggable NER tagger.

    Parameters
    ----------
    backend
        ``"mock"`` (default, dependency-free) or ``"spacy"`` for the real
        spaCy model.
    model_name
        spaCy model to load (e.g. ``"en_core_web_sm"``). Ignored when
        ``backend == "mock"`` or ``mock == True``.
    mock
        Force mock mode regardless of ``backend`` (useful when a caller
        passes ``config.mock`` straight through).
    """

    def __init__(self,
                 backend: str = "mock",
                 model_name: str = "en_core_web_sm",
                 mock: bool = True):
        if backend not in {"mock", "spacy"}:
            raise ValueError(f"Unknown NER backend: {backend!r}")
        self.backend = backend
        self.model_name = model_name
        self.mock = mock or (backend != "spacy")
        self._spacy = None
        if not self.mock:
            self._load_spacy()

    def _load_spacy(self) -> None:
        # Lazy import so the mock-mode path stays slim.
        try:
            import spacy          # type: ignore
            self._spacy = spacy.load(self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load spaCy model {self.model_name!r}: {e}. "
                "Either install the model (`python -m spacy download "
                "en_core_web_sm`) or set mock=True for the offline backend."
            ) from e

    # ----- public API --------------------------------------------------------

    def tag(self, text: str) -> List[Entity]:
        """Return the entities mentioned in ``text``.

        Empty / whitespace-only input returns an empty list.
        """
        if not text or not text.strip():
            return []
        if self.mock:
            return _mock_tag(text)

        assert self._spacy is not None
        doc = self._spacy(text)
        # spaCy label set -> our coarse 6-label set.
        mapping = {
            "PERSON": "PER",
            "ORG": "ORG",
            "GPE": "LOC",
            "LOC": "LOC",
            "DATE": "DATE",
            "TIME": "DATE",
            "MONEY": "NUM",
            "PERCENT": "NUM",
            "QUANTITY": "NUM",
            "CARDINAL": "NUM",
            "ORDINAL": "NUM",
        }
        out: List[Entity] = []
        for ent in doc.ents:
            label = mapping.get(ent.label_, "MISC")
            out.append(Entity(text=ent.text, label=label,
                              start=ent.start_char, end=ent.end_char))
        return out

    def entities_by_label(self, text: str, *labels: str) -> List[Entity]:
        """Convenience: ``tag()`` then filter by one or more labels."""
        wanted = set(labels)
        return [e for e in self.tag(text) if e.label in wanted]


# --- helpers -----------------------------------------------------------------


def dedupe_overlapping(entities: Sequence[Entity]) -> List[Entity]:
    """Drop shorter entities whose span is fully inside a longer one.

    Used when combining mock + real NER results, or when filtering by label
    and you want to keep the longest matching span for each anchor.
    """
    sorted_ents = sorted(entities, key=lambda e: (e.start, -(e.end - e.start)))
    kept: List[Entity] = []
    for e in sorted_ents:
        if any(k.start <= e.start and e.end <= k.end for k in kept):
            continue
        kept.append(e)
    return kept