"""Phase 5 — Multi-hop evidence aggregation.

What ``MultiHopPlanner`` does
-----------------------------
Given an *aggregated* claim (Phase 3 tag ``Provenance.AGGREGATED``),
single-hop retrieval is not enough — the claim composes information
from multiple passages. Phase 5 walks a 2-hop graph to gather supporting
evidence:

    1. **NER** — extract typed entities (PER / ORG / LOC / WORK / DATE)
       from the claim text.
    2. **KG link** — resolve each mention to a canonical ``KGEntity``.
       Unresolved mentions are skipped (we can't reason over them).
    3. **Plan sub-queries** — for each resolved entity, pick a small set
       of "priority relations" given its type (e.g. for ``PER`` we ask
       about citizenship / occupation / notable works) and form one
       sub-query per (entity, relation).
    4. **Execute** — issue each sub-query against the adaptive retriever
       (or any ``.retrieve(query, top_k)`` compatible object). Merge the
       hits with the original first-pass hits.
    5. **Trace** — record everything into a ``MultiHopTrace`` dataclass
       so Phase 6/7 can replay the policy and so offline analysis can
       audit the planner.

The trace is the contract surface. It has three roles:

  * ``hops``           — ordered list of sub-queries actually issued
  * ``sub_queries``    — flat list of query strings (mirrors ``hops``
                         but easier to JSON-dump)
  * ``entities``       — KG entities resolved from the claim
  * ``evidence``       — the merged, deduped hit list returned to the
                         pipeline (Phase 6 will use these as the
                         evidence for the NLI detector)
  * ``notes``          — free-text decisions for human inspection

This module is intentionally deterministic and has no LLM dependency;
it stays laptop-runnable in MOCK mode (the default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set, Tuple

from .kg_linker import KGEntity, KGLinker, canonicalise, top_relations_for_type
from .ner import Entity, NER
from .retriever import Hit


# --- trace dataclass ---------------------------------------------------------


@dataclass
class MultiHopTrace:
    """Audit trail of one multi-hop expansion."""

    claim_text: str = ""
    entities: List[KGEntity] = field(default_factory=list)
    unresolved_mentions: List[str] = field(default_factory=list)
    hops: List[dict] = field(default_factory=list)
    sub_queries: List[str] = field(default_factory=list)
    evidence: List[Hit] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "entities": [e.to_dict() for e in self.entities],
            "unresolved_mentions": list(self.unresolved_mentions),
            "hops": list(self.hops),
            "sub_queries": list(self.sub_queries),
            "evidence": [
                {"text": h.text, "score": float(h.score), "index": int(h.index)}
                for h in self.evidence
            ],
            "notes": list(self.notes),
        }


# --- planner -----------------------------------------------------------------


class MultiHopPlanner:
    """Plan and execute multi-hop evidence retrieval for one aggregated claim.

    Parameters
    ----------
    ner
        ``NER`` tagger. Default constructs a mock-mode tagger.
    linker
        ``KGLinker`` against the mini-KG. Default uses the built-in.
    max_entities
        Hard cap on entities considered per claim. Default 3.
    max_relations_per_entity
        How many of the priority relations to try per entity. Default 2.
    top_k_per_subquery
        ``top_k`` passed to the retriever for each sub-query. Default 3.
    """

    def __init__(self,
                 ner: Optional[NER] = None,
                 linker: Optional[KGLinker] = None,
                 max_entities: int = 3,
                 max_relations_per_entity: int = 2,
                 top_k_per_subquery: int = 3):
        self.ner = ner or NER(mock=True)
        self.linker = linker or KGLinker()
        if max_entities < 1:
            raise ValueError("max_entities must be >= 1")
        if max_relations_per_entity < 0:
            raise ValueError("max_relations_per_entity must be >= 0")
        if top_k_per_subquery < 1:
            raise ValueError("top_k_per_subquery must be >= 1")
        self.max_entities = int(max_entities)
        self.max_relations_per_entity = int(max_relations_per_entity)
        self.top_k_per_subquery = int(top_k_per_subquery)

    # ----- public API --------------------------------------------------------

    def plan(self, claim_text: str) -> MultiHopTrace:
        """Analyse a claim and produce a trace with NO retrieval.

        Useful when you want to inspect the planner's plan before
        committing retrieval budget.
        """
        trace = MultiHopTrace(claim_text=claim_text)
        ents, unresolved = self._resolve_entities(claim_text)
        trace.entities = ents
        trace.unresolved_mentions = unresolved
        trace.hops, trace.sub_queries = self._build_hops(ents)
        if not ents:
            trace.notes.append("no resolvable entities; skipping retrieval")
        elif not trace.sub_queries:
            trace.notes.append("entities resolved but no priority relations apply")
        return trace

    def execute(self,
                claim_text: str,
                retriever,
                seed_evidence: Optional[Sequence[Hit]] = None) -> MultiHopTrace:
        """Plan + run the sub-queries, merge the evidence.

        Parameters
        ----------
        claim_text
            The aggregated claim.
        retriever
            Anything exposing ``retrieve(query, top_k) -> List[Hit]``.
            Usually the adaptive retriever.
        seed_evidence
            Optional existing hits (e.g. from the first-pass single-hop
            retrieval) to merge with the sub-query results so we don't
            drop known-good evidence.
        """
        trace = self.plan(claim_text)

        merged_lists: List[Sequence[Hit]] = []
        if seed_evidence:
            merged_lists.append(list(seed_evidence))

        for hop in trace.hops:
            sub_q = hop["sub_query"]
            try:
                hits = retriever.retrieve(sub_q, top_k=self.top_k_per_subquery)
            except Exception as e:                       # pragma: no cover
                trace.notes.append(f"sub-query failed: {sub_q!r} ({e})")
                continue
            hop["evidence"] = [
                {"text": h.text, "score": float(h.score), "index": int(h.index)}
                for h in hits
            ]
            merged_lists.append(list(hits))
            trace.notes.append(
                f"hop: {hop['relation']}({hop['entity']}) "
                f"-> {len(hits)} hits"
            )

        merged = _merge_hits_lists(*merged_lists)
        # keep at most 3 * top_k hits so the trace doesn't blow up.
        cap = max(3 * self.top_k_per_subquery, len(seed_evidence or []))
        trace.evidence = merged[:cap]
        return trace

    # ----- internals ---------------------------------------------------------

    def _resolve_entities(self,
                          text: str) -> Tuple[List[KGEntity], List[str]]:
        """NER -> KG link, capped to max_entities."""
        raw_ents: List[Entity] = self.ner.tag(text)
        # Drop NUM/DATE: the mini-KG doesn't have numeric entities, and
        # we want to spend the budget on PER / ORG / LOC / WORK.
        mentionable = [e for e in raw_ents
                       if e.label in {"PER", "ORG", "LOC", "WORK", "MISC"}]
        mentions = [e.text for e in mentionable]

        resolved = canonicalise(self.linker, mentions)
        unresolved = [m for m in mentions
                      if self.linker.lookup(m) is None]
        # Cap, prefer earlier mentions.
        return resolved[:self.max_entities], unresolved

    def _build_hops(self,
                    ents: List[KGEntity]) -> Tuple[List[dict], List[str]]:
        """For each entity, build sub-queries for its priority relations."""
        hops: List[dict] = []
        sub_queries: List[str] = []
        for ent in ents:
            relations = top_relations_for_type(ent.type)[:self.max_relations_per_entity]
            for rel in relations:
                tails = self.linker.attribute(ent.id, rel)
                if not tails:
                    continue
                # Form the sub-query. Keep it simple: "<entity> <relation>"
                # — this lets BM25 (Phase 4) and dense retrieval both
                # latch onto the right evidence.
                sub_q = f"{ent.name} {rel.replace('_', ' ')}"
                # Optionally include the tail value to bias the query
                # toward the expected answer (e.g. "George Orwell country
                # of citizenship United Kingdom").
                if len(tails) == 1:
                    sub_q = f"{sub_q} {tails[0]}"
                hops.append({
                    "entity_id": ent.id,
                    "entity": ent.name,
                    "relation": rel,
                    "tails": list(tails),
                    "sub_query": sub_q,
                })
                sub_queries.append(sub_q)
        return hops, sub_queries


# --- helpers -----------------------------------------------------------------


def _merge_hits_lists(*lists: Sequence[Hit]) -> List[Hit]:
    """Union hit lists, keeping the highest score per document index,
    sorted by score descending.
    """
    by_idx: dict = {}
    order: List[int] = []
    for lst in lists:
        for h in lst:
            if h.index not in by_idx:
                by_idx[h.index] = h
                order.append(h.index)
            elif h.score > by_idx[h.index].score:
                by_idx[h.index] = Hit(text=h.text, score=float(h.score),
                                       index=int(h.index))
    merged = [by_idx[i] for i in order]
    merged.sort(key=lambda h: h.score, reverse=True)
    return merged