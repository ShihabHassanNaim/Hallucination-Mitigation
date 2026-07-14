"""Phase 5 — Knowledge-Graph entity linker.

Why we need a KG
----------------
Multi-hop reasoning needs more than entity detection — it needs to know
*what kind of entity* something is, and what *attributes* follow from it.
For example:

    "George Orwell wrote 1984."

…has ``PER -> wrote -> WORK``. To verify the claim we'd want to:

  1. NER: find ``George Orwell`` (PER), ``1984`` (MISC).
  2. KG: resolve ``George Orwell -> QXXXX (canonical author)`` and
     ``1984 -> QXXXX (canonical work)``.
  3. KG: traverse ``author-of(QXXXX_per) -> work(QXXXX_work)``.
  4. RAG: re-query for each tail entity and union the evidence.

Phase 5 ships a *mini* knowledge graph — a deterministic JSON-shaped
in-memory store covering the entities Phase 4's synthetic corpus + the
most common HaluEval relations. The interface is the same one a real
Wikidata / ConceptNet / YAGO adapter would expose, so dropping in a real
KG later is a one-class change.

Schema
------
A ``KGEntity`` is::

    KGEntity(
        id           : canonical Wikidata-style id ("QXXXX")
        name         : canonical surface form ("George Orwell")
        type         : "PER" | "ORG" | "LOC" | "WORK" | "DATE" | "OTHER"
        surface_forms: list of aliases incl. the canonical name
        attributes   : dict of (relation_name -> List[str]) e.g.
                       {"country_of_citizenship": ["United Kingdom"],
                        "occupation": ["novelist", "journalist"]}
    )

Lookup is two-stage:

  1. Try exact (case-insensitive) match against ``surface_forms``.
  2. Try normalised match: lowercase, strip punctuation, drop
     determiners/articles. This catches ``"the United Kingdom"`` vs
     ``"United Kingdom"`` and similar noise.

The mini-KG
-----------
Hard-coded for the synthetic corpus + the HaluEval QA slice we ship.
It is intentionally *small and correct* rather than broad: the goal of
Phase 5 is to prove the *planner loop* works, not to be a production KG.
A real backend can be plugged in via ``KGLinker(custom_entities=...)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set


@dataclass
class KGEntity:
    """One node in the mini-KG."""

    id: str
    name: str
    type: str
    surface_forms: List[str] = field(default_factory=list)
    attributes: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "surface_forms": list(self.surface_forms),
            "attributes": {k: list(v) for k, v in self.attributes.items()},
        }


# --- normalisation ----------------------------------------------------------

# Stopwords to drop when fuzzy-matching surface forms. Mirrors the BM25
# stopword list so the two modules agree on what counts as content.
_KG_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
}


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, drop stopwords and extra whitespace."""
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            cur.append(ch)
        else:
            if cur:
                word = "".join(cur).strip()
                if word and word not in _KG_STOPWORDS:
                    out.append(word)
                cur = []
    if cur:
        word = "".join(cur).strip()
        if word and word not in _KG_STOPWORDS:
            out.append(word)
    return " ".join(out)


# --- mini-KG ----------------------------------------------------------------

# The seed KG. Each entity has an id, name, type, surface_forms, and a flat
# attributes dict. Attributes are lists because a relation can point to
# multiple values (e.g. Einstein's occupations).
_MINI_KG: List[KGEntity] = [
    KGEntity(
        id="Q1", name="George Orwell",
        type="PER",
        surface_forms=["George Orwell", "Orwell", "Eric Blair"],
        attributes={
            "country_of_citizenship": ["United Kingdom"],
            "occupation": ["novelist", "journalist", "essayist"],
            "notable_works": ["1984", "Animal Farm"],
            "born": ["1903"],
        },
    ),
    KGEntity(
        id="Q2", name="1984",
        type="WORK",
        surface_forms=["1984", "Nineteen Eighty-Four"],
        attributes={
            "author": ["George Orwell"],
            "genre": ["dystopian", "political fiction"],
            "publication_year": ["1949"],
            "original_language": ["English"],
        },
    ),
    KGEntity(
        id="Q3", name="Animal Farm",
        type="WORK",
        surface_forms=["Animal Farm"],
        attributes={
            "author": ["George Orwell"],
            "genre": ["political satire", "allegory"],
            "publication_year": ["1945"],
        },
    ),
    KGEntity(
        id="Q4", name="France",
        type="LOC",
        surface_forms=["France"],
        attributes={
            "capital": ["Paris"],
            "continent": ["Europe"],
            "official_language": ["French"],
        },
    ),
    KGEntity(
        id="Q5", name="Paris",
        type="LOC",
        surface_forms=["Paris"],
        attributes={
            "country": ["France"],
            "continent": ["Europe"],
        },
    ),
    KGEntity(
        id="Q6", name="Germany",
        type="LOC",
        surface_forms=["Germany"],
        attributes={
            "capital": ["Berlin"],
            "continent": ["Europe"],
            "official_language": ["German"],
        },
    ),
    KGEntity(
        id="Q7", name="Berlin",
        type="LOC",
        surface_forms=["Berlin"],
        attributes={
            "country": ["Germany"],
            "continent": ["Europe"],
        },
    ),
    KGEntity(
        id="Q8", name="United Kingdom",
        type="LOC",
        surface_forms=["United Kingdom", "UK", "Britain", "Great Britain"],
        attributes={
            "capital": ["London"],
            "continent": ["Europe"],
            "official_language": ["English"],
        },
    ),
    KGEntity(
        id="Q9", name="Mars",
        type="LOC",
        surface_forms=["Mars", "Red Planet"],
        attributes={
            "type": ["planet"],
            "in_solar_system": ["Sun"],
        },
    ),
    KGEntity(
        id="Q10", name="PyTorch",
        type="WORK",
        surface_forms=["PyTorch"],
        attributes={
            "developed_by": ["Meta AI"],
            "type": ["machine learning library"],
            "license": ["BSD"],
            "initial_release": ["2016"],
        },
    ),
    KGEntity(
        id="Q11", name="Meta AI",
        type="ORG",
        surface_forms=["Meta AI", "Meta"],
        attributes={
            "type": ["research lab"],
            "parent": ["Meta Platforms"],
            "headquarters": ["United States"],
        },
    ),
    KGEntity(
        id="Q12", name="OpenAI",
        type="ORG",
        surface_forms=["OpenAI"],
        attributes={
            "type": ["research lab"],
            "headquarters": ["United States"],
            "founded": ["2015"],
        },
    ),
    KGEntity(
        id="Q13", name="European Union",
        type="ORG",
        surface_forms=["European Union", "EU"],
        attributes={
            "headquarters": ["Brussels"],
            "type": ["political and economic union"],
            "member_states": ["France", "Germany"],
        },
    ),
]


# --- linker ------------------------------------------------------------------


class KGLinker:
    """Resolve a free-text entity mention to a ``KGEntity``.

    Parameters
    ----------
    custom_entities
        Optional additional ``KGEntity`` list to merge on top of the
        built-in mini-KG. Useful for plugging a domain-specific KG
        without re-implementing the lookup.
    """

    def __init__(self, custom_entities: Optional[Sequence[KGEntity]] = None):
        self._entities: List[KGEntity] = list(_MINI_KG)
        if custom_entities:
            # de-dupe by id; later wins
            existing = {e.id: i for i, e in enumerate(self._entities)}
            for e in custom_entities:
                if e.id in existing:
                    self._entities[existing[e.id]] = e
                else:
                    existing[e.id] = len(self._entities)
                    self._entities.append(e)
        # pre-compute normalised surface-form index for fuzzy lookup.
        self._index: Dict[str, str] = {}
        for ent in self._entities:
            for sf in ent.surface_forms + [ent.name]:
                self._index.setdefault(_normalise(sf), ent.id)

    # ----- public API --------------------------------------------------------

    @property
    def n_entities(self) -> int:
        return len(self._entities)

    def all_entities(self) -> List[KGEntity]:
        return list(self._entities)

    def lookup(self, text: str) -> Optional[KGEntity]:
        """Resolve ``text`` to a KG node, or ``None`` if unknown."""
        if not text or not text.strip():
            return None
        key = _normalise(text)
        eid = self._index.get(key)
        if eid is not None:
            return self._by_id(eid)
        # substring fallback: try every entity's normalised name and see
        # if the query is contained in it (or vice versa). This catches
        # "the Mars planet" vs "Mars" etc.
        for ent in self._entities:
            nname = _normalise(ent.name)
            if not nname:
                continue
            if nname in key or key in nname:
                return ent
            for sf in ent.surface_forms:
                nsf = _normalise(sf)
                if nsf and (nsf in key or key in nsf):
                    return ent
        return None

    def lookup_batch(self, texts: Iterable[str]) -> List[Optional[KGEntity]]:
        return [self.lookup(t) for t in texts]

    def get(self, entity_id: str) -> Optional[KGEntity]:
        return self._by_id(entity_id)

    def attribute(self, entity_id: str, relation: str) -> List[str]:
        ent = self._by_id(entity_id)
        if ent is None:
            return []
        return list(ent.attributes.get(relation, []))

    def expand(self, entity_id: str, relation: str) -> List[KGEntity]:
        """Traverse ``entity_id -[relation]-> tail_entity``.

        Returns one KGEntity per tail value whose name resolves in the KG.
        Used by the multi-hop planner to chain 2-hop queries.
        """
        tails = self.attribute(entity_id, relation)
        out: List[KGEntity] = []
        for t in tails:
            ent = self.lookup(t)
            if ent is not None:
                out.append(ent)
        return out

    # ----- internals ---------------------------------------------------------

    def _by_id(self, entity_id: str) -> Optional[KGEntity]:
        for ent in self._entities:
            if ent.id == entity_id:
                return ent
        return None


# --- convenience functions ---------------------------------------------------


def canonicalise(linker: KGLinker, mentions: Iterable[str]) -> List[KGEntity]:
    """Return the deduplicated list of KG nodes covering ``mentions``.

    Order: first-seen wins. Unresolved mentions are skipped silently.
    """
    seen: Set[str] = set()
    out: List[KGEntity] = []
    for m in mentions:
        ent = linker.lookup(m)
        if ent is None or ent.id in seen:
            continue
        seen.add(ent.id)
        out.append(ent)
    return out


def top_relations_for_type(entity_type: str) -> List[str]:
    """Return a small set of plausible relations to try per entity type.

    Used by ``MultiHopPlanner`` to decide which edges to expand first.
    Not exhaustive — the planner falls back to all attributes if none of
    the priority relations yield evidence.
    """
    table: Dict[str, List[str]] = {
        "PER":   ["country_of_citizenship", "occupation", "notable_works"],
        "ORG":   ["headquarters", "type", "parent"],
        "LOC":   ["capital", "continent", "country"],
        "WORK":  ["author", "publication_year", "genre"],
    }
    return table.get(entity_type, [])