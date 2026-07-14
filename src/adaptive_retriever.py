"""Phase 4 — Adaptive evidence retrieval controller.

What ``AdaptiveRetriever`` does
-------------------------------
A vanilla one-shot retriever gives the generator the same fixed top-k
no matter the question. But three failure modes show up repeatedly:

  1. **Sparse evidence** — top-1 score is low; we should re-query with
     ``k -> 2k`` to widen the candidate pool before giving up.
  2. **Missing paraphrase** — dense retriever fails on lexically rare
     phrasings; query rewriting (heuristic variants) gives the BM25
     side another chance to surface the right hit.
  3. **Multi-hop claims** — a generated answer may contain claims whose
     provenance is ``AGGREGATED`` (Phase 3). Single-hop evidence isn't
     enough; we need to retrieve evidence for each entity named in the
     claim and union the hit lists.

This module wraps any ``Retriever``-compatible base (``Retriever`` or
``HybridRetriever``) and exposes a small policy object with three
strategies:

  * ``none``            — pass-through, no adaptation.
  * ``expand``          — widen ``k`` if top-1 score is below
                          ``expansion_score_threshold`` OR if marginal
                          gain between ranks is below
                          ``expansion_gap_threshold``.
  * ``multi_hop``       — after the initial pass + claim provenance is
                          known (post-generation), re-query for entities
                          mentioned in ``AGGREGATED`` claims.
  * ``auto``            — apply ``expand`` based on scores; apply
                          ``multi_hop`` based on provenance tags.

The controller records everything it did in a ``RetrievalTrace`` so
Phase 7's Adaptive Iteration Controller can decide whether to
regenerate, and so offline analysis can replay the policy.

This module has no heavy dependencies (no LLM, no spacy) on purpose;
it stays deterministic and laptop-runnable in MOCK mode.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

from .bm25 import BM25Index, tokenize
from .hybrid_retriever import HybridDiagnostics
from .retriever import Hit, Retriever


class Strategy(str, enum.Enum):
    """Which adaptive strategy was chosen for a query."""

    NONE = "none"
    EXPAND = "expand"
    REWRITE = "rewrite"
    MULTI_HOP = "multi_hop"
    EXPAND_AND_MULTI_HOP = "expand_and_multi_hop"
    REWRITE_AND_MULTI_HOP = "rewrite_and_multi_hop"

    def __str__(self) -> str:                       # pragma: no cover
        return self.value


@dataclass
class AdaptiveConfig:
    """Phase-4 policy knobs. All thresholds are in score units that match
    whatever the wrapped ``Retriever`` returns (dense: cosine in [-1, 1];
    BM25: positive; hybrid: RRF score)."""

    # Initial retrieval size. Doubled on expansion if eligible.
    initial_k: int = 5
    max_k: int = 20

    # Expansion triggers (relative to the wrapped retriever's score range).
    expansion_score_threshold: float = 0.20    # top-1 below -> expand
    expansion_gap_threshold: float = 0.05      # top1-top2 below -> expand

    # Query rewriting: how many heuristic rewrites to try before giving up.
    rewrite_variants: int = 2

    # Multi-hop: cap on how many entities we re-query for.
    multi_hop_max_entities: int = 3
    multi_hop_min_entity_len: int = 3

    def __post_init__(self) -> None:
        if self.initial_k < 1:
            raise ValueError("initial_k must be >= 1")
        if self.max_k < self.initial_k:
            raise ValueError("max_k must be >= initial_k")
        # No upper bound on the thresholds — a threshold >> 1.0 means
        # "always trigger", a threshold << 0 means "never trigger".
        # Both are legitimate policy choices.


@dataclass
class RetrievalTrace:
    """Audit trail of one adaptive retrieval call.

    Fields
    ------
    strategy        : which Strategy the controller picked.
    initial_k       : the k the controller started with.
    final_k         : the k actually used (may be larger after expand).
    rewrites        : list of query variants issued (excluding the original).
    multi_hop_queries: list of sub-queries issued for multi-hop.
    n_unique_hits   : number of distinct documents in the final hit list.
    top_score       : the top fused / dense score after adaptation.
    gap             : top1 - top2 score gap (a small gap -> expansion).
    notes           : free-text list of decisions for human inspection.
    extra           : dict for adapter-specific diagnostics (RRF, BM25, ...).
    """

    strategy: Strategy = Strategy.NONE
    initial_k: int = 5
    final_k: int = 5
    rewrites: List[str] = field(default_factory=list)
    multi_hop_queries: List[str] = field(default_factory=list)
    n_unique_hits: int = 0
    top_score: float = 0.0
    gap: float = 0.0
    notes: List[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "strategy": str(self.strategy),
            "initial_k": self.initial_k,
            "final_k": self.final_k,
            "rewrites": list(self.rewrites),
            "multi_hop_queries": list(self.multi_hop_queries),
            "n_unique_hits": self.n_unique_hits,
            "top_score": self.top_score,
            "gap": self.gap,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }


def _passes_expansion(hits: Sequence[Hit],
                      cfg: AdaptiveConfig) -> bool:
    """Decide whether the initial retrieval was weak enough to warrant
    expanding k.

    Two signals, OR-ed:
      * top-1 score below ``expansion_score_threshold``
      * top-1 -> top-2 gap below ``expansion_gap_threshold``
        (the corpus is fuzzy; we need more options)
    """
    if not hits:
        return True
    top1 = hits[0].score
    if top1 < cfg.expansion_score_threshold:
        return True
    if len(hits) >= 2:
        gap = top1 - hits[1].score
        if gap < cfg.expansion_gap_threshold:
            return True
    return False


def _gap(hits: Sequence[Hit]) -> float:
    if len(hits) < 2:
        return 0.0
    return float(hits[0].score - hits[1].score)


def _heuristic_rewrites(query: str, n: int) -> List[str]:
    """Cheap deterministic query rewrites.

    We do NOT call any LLM here; this stays laptop-runnable in MOCK
    mode. The variants are:

      * lowercase + strip punctuation
      * drop question words ("what", "who", "when"...) — these often
        don't help BM25 because they appear in nearly every retrieval
        query, and dropping them hurts nothing for fact questions.
      * swap common synonyms ("capital of X" <-> "X's capital")

    Synonyms below are intentionally conservative — they only target
    the question patterns we know Phase 2 tends to hallucinate on.
    """
    out: List[str] = []
    q = query.strip()
    if not q:
        return out

    cleaned = q.lower().strip("?.!,")

    drop_qwords = ("what", "who", "when", "where", "which", "how")
    tokens = cleaned.split()
    kept = [t for t in tokens if t not in drop_qwords]
    # Only emit the cleaned variant if it actually differs from the
    # question-word-stripped variant. Otherwise we'd emit both
    # "who wrote 1984" and "wrote 1984" — a wasted slot and a confusing
    # one (the first rewrite still has "who" in it).
    if not kept or kept == tokens:
        out.append(cleaned)
    else:
        out.append(" ".join(kept))

    # synonym swaps, applied to the no-question-word variant
    base = kept if kept else tokens
    swapped = " ".join(base)
    for src, dst in [
        ("capital of", "seat of government of"),
        ("author of", "writer of"),
        ("inventor of", "creator of"),
        ("born in", "birthplace in"),
    ]:
        if src in swapped:
            out.append(swapped.replace(src, dst, 1))
            break

    # de-dupe, keep order, cap to n
    seen: Set[str] = set()
    deduped: List[str] = []
    for v in out:
        if v and v not in seen and v != q:
            seen.add(v)
            deduped.append(v)
    return deduped[:n]


def _entities_for_multi_hop(text: str,
                            min_len: int = 3,
                            max_entities: int = 3) -> List[str]:
    """Pull a few simple "named entities" out of a claim for re-querying.

    A real Phase-5 system would use a NER model or a KG linker. For
    Phase 4 we stay lexical: take the longest contiguous capitalised
    spans (>= ``min_len`` chars) up to ``max_entities``. This is good
    enough to give the multi-hop hook real signal on the synthetic
    corpus and trivially generalises to HaluEval.
    """
    import re
    span_re = re.compile(r"[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*")
    spans = span_re.findall(text)
    out: List[str] = []
    seen: Set[str] = set()
    # prefer longest spans first
    for s in sorted(set(spans), key=len, reverse=True):
        if len(s.replace(" ", "")) < min_len:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_entities:
            break
    return out


def _merge_hits(*lists: Sequence[Hit]) -> List[Hit]:
    """Union hit lists, preserving first-seen order, max-scoring duplicate wins."""
    by_idx: dict = {}
    order: List[int] = []
    for lst in lists:
        for h in lst:
            if h.index not in by_idx:
                by_idx[h.index] = h
                order.append(h.index)
            else:
                # keep the higher score across sources
                if h.score > by_idx[h.index].score:
                    by_idx[h.index] = Hit(
                        text=h.text,
                        score=float(h.score),
                        index=h.index,
                    )
    return [by_idx[i] for i in order]


class AdaptiveRetriever:
    """Wraps any ``Retriever``-compatible base and applies adaptive policies.

    Parameters
    ----------
    base
        A ``Retriever``, ``HybridRetriever``, or any object exposing
        ``retrieve(query, top_k) -> List[Hit]`` and ``.n_docs``.
    config
        ``AdaptiveConfig`` with policy knobs.
    """

    def __init__(self, base,
                 config: Optional[AdaptiveConfig] = None):
        self.base = base
        self.config = config or AdaptiveConfig()
        self.last_trace: Optional[RetrievalTrace] = None

    @property
    def n_docs(self) -> int:
        # Dense ``Retriever`` exposes ``_docs`` (private) but not a public
        # ``n_docs`` property. Fall back to the length of whichever
        # corpus-shaped attribute is available on the base.
        if hasattr(self.base, "n_docs"):
            return int(self.base.n_docs)
        docs = getattr(self.base, "_docs", None)
        return len(docs) if docs is not None else 0

    # ----- public API ---------------------------------------------------------

    def build(self, documents: Sequence[str]) -> "AdaptiveRetriever":
        if not documents:
            raise ValueError("Cannot build adaptive retriever over empty corpus.")
        # The base retriever owns its own build() if it has one.
        if hasattr(self.base, "build"):
            self.base.build(documents)
        return self

    def retrieve(self, query: str,
                 top_k: Optional[int] = None,
                 aggregated_claim_text: Optional[str] = None) -> List[Hit]:
        """Adaptive retrieval entry point.

        Parameters
        ----------
        query
            The original user question.
        top_k
            Optional override for the initial k. Falls back to
            ``AdaptiveConfig.initial_k``.
        aggregated_claim_text
            If a downstream claim has ``Provenance.AGGREGATED``,
            pass its text here so the controller can re-query for
            the named entities. Leaving it ``None`` means: no
            multi-hop aggregation (single-hop pipeline).

        Returns
        -------
        List[Hit]
            Top hits after the adaptive policy has been applied.
            ``self.last_trace`` records the strategy and decisions.
        """
        cfg = self.config
        initial_k = min(top_k or cfg.initial_k, self.n_docs or top_k or cfg.initial_k)
        trace = RetrievalTrace(
            strategy=Strategy.NONE,
            initial_k=initial_k,
            final_k=initial_k,
        )

        # ----- 1. initial retrieval -----------------------------------------
        hits = self.base.retrieve(query, top_k=initial_k)
        top_score = hits[0].score if hits else 0.0
        trace.top_score = float(top_score)
        trace.gap = _gap(hits)
        if hasattr(self.base, "last_diagnostics") and \
                isinstance(self.base.last_diagnostics, HybridDiagnostics):
            trace.extra["hybrid"] = self.base.last_diagnostics.to_dict()

        notes: List[str] = []
        merged_lists: List[List[Hit]] = [hits]
        k_for_final = initial_k

        # ----- 2. expansion --------------------------------------------------
        expanded = _passes_expansion(hits, cfg)
        if expanded and initial_k < cfg.max_k:
            k_for_final = min(cfg.max_k, max(initial_k * 2, initial_k + 1))
            extra = self.base.retrieve(query, top_k=k_for_final)
            merged_lists.append(extra)
            notes.append(
                f"expand:k {initial_k}->{k_for_final} "
                f"(top_score={top_score:.3f}, gap={trace.gap:.3f})"
            )
            # recompute top of the union
            union = _merge_hits(*merged_lists)
            if union:
                top_score = float(union[0].score)
                trace.top_score = top_score
            trace.strategy = Strategy.EXPAND

        # ----- 3. query rewriting -------------------------------------------
        rewrites = _heuristic_rewrites(query, cfg.rewrite_variants)
        if rewrites:
            for variant in rewrites:
                rhits = self.base.retrieve(variant, top_k=k_for_final)
                merged_lists.append(rhits)
                trace.rewrites.append(variant)
            notes.append(
                f"rewrite: tried {len(rewrites)} variants"
            )
            if trace.strategy == Strategy.NONE:
                trace.strategy = Strategy.REWRITE

        # ----- 4. multi-hop --------------------------------------------------
        multi_hop_done = False
        if aggregated_claim_text:
            ents = _entities_for_multi_hop(
                aggregated_claim_text,
                min_len=cfg.multi_hop_min_entity_len,
                max_entities=cfg.multi_hop_max_entities,
            )
            for ent in ents:
                sub_q = ent  # current heuristic: re-query by entity alone.
                mh_hits = self.base.retrieve(sub_q, top_k=k_for_final)
                merged_lists.append(mh_hits)
                trace.multi_hop_queries.append(sub_q)
            if ents:
                multi_hop_done = True
                notes.append(
                    f"multi_hop: re-queried {len(ents)} entities "
                    f"from claim {aggregated_claim_text[:60]!r}"
                )

        # ----- 5. merge + final sort ----------------------------------------
        union = _merge_hits(*merged_lists)
        # If we widened k, keep wider; else cap to initial k.
        if union:
            union.sort(key=lambda h: h.score, reverse=True)
            final_k = max(k_for_final, initial_k)
            union = union[:final_k]
            trace.top_score = float(union[0].score) if union else 0.0
            trace.gap = _gap(union)
            trace.n_unique_hits = len({h.index for h in union})
            trace.final_k = final_k
        if multi_hop_done:
            if trace.strategy in (Strategy.NONE,):
                trace.strategy = Strategy.MULTI_HOP
            elif trace.strategy == Strategy.EXPAND:
                trace.strategy = Strategy.EXPAND_AND_MULTI_HOP
            elif trace.strategy == Strategy.REWRITE:
                trace.strategy = Strategy.REWRITE_AND_MULTI_HOP
        trace.notes = notes
        self.last_trace = trace
        return union

    # ----- persistence ------------------------------------------------------

    def save(self, out_dir: str) -> None:
        if hasattr(self.base, "save"):
            self.base.save(out_dir)

    @classmethod
    def load(cls, in_dir: str, base_factory) -> "AdaptiveRetriever":
        # Lazy factory so callers can swap in HybridRetriever vs dense
        # without this module pulling in heavy deps.
        base = base_factory(in_dir)
        return cls(base=base)