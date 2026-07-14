"""Phase 4 — Hybrid dense + BM25 retriever with reciprocal-rank fusion.

Design
------
The dense retriever excels at semantic paraphrase / topic-level matches
(e.g. "capital" <-> "seat of government"); the BM25 retriever excels at
exact-lexical matches for named entities and numbers. Combining them
gives complementary recall, especially in the long tail where a single
method misses.

We use **Reciprocal Rank Fusion** (Cormack et al., 2009) rather than
linear score blending, because RRF:

  * doesn't require score calibration between the two retrievers (BM25
    and cosine-IP live on very different scales),
  * is robust to outliers (a single dominant score can't drown out a
    hit that's consistently ranked top-3 by both),
  * has a single knob (the RRF constant ``k``) that's well-behaved in
    practice.

The standard RRF formula is::

    RRF(d) = sum_r w_r / (k + rank_r(d))

with rank_r(d) = 1-based rank of doc ``d`` under retriever ``r`` (or
omitted if ``d`` is not in its top-k), w_r = retriever weight, and k =
smoothing constant (60 in the original paper; we expose it via config).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .bm25 import BM25Hit, BM25Index
from .retriever import Hit, Retriever


@dataclass
class _RankEntry:
    """Internal record for one document's contributions to the fused score."""
    text: str
    index: int
    rrf: float = 0.0
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None


@dataclass
class HybridConfig:
    """Tunable knobs for HybridRetriever."""

    # RRF smoothing constant (Cormack et al. used k=60).
    rrf_k: float = 60.0
    # Retriever weights — sum does not need to be 1 because RRF normalises
    # by rank rather than raw score.
    weight_dense: float = 1.0
    weight_bm25: float = 1.0
    # If True, BM25 must be built from the same document list as the
    # dense retriever; we enforce this in HybridRetriever.build().
    require_same_corpus: bool = True


@dataclass
class HybridDiagnostics:
    """Snapshot of the last fusion call. Useful for the adaptive
    controller (Phase 4) and for the RAGResult.retrieval_trace field."""

    dense_top_score: Optional[float] = None
    bm25_top_score: Optional[float] = None
    fused_top_score: float = 0.0
    fused_top_rank: int = 0
    dense_only: List[str] = field(default_factory=list)
    bm25_only: List[str] = field(default_factory=list)
    overlap: List[str] = field(default_factory=list)
    n_unique: int = 0

    def to_dict(self) -> dict:
        return {
            "dense_top_score": self.dense_top_score,
            "bm25_top_score": self.bm25_top_score,
            "fused_top_score": self.fused_top_score,
            "fused_top_rank": self.fused_top_rank,
            "n_unique": self.n_unique,
            "n_dense_only": len(self.dense_only),
            "n_bm25_only": len(self.bm25_only),
            "n_overlap": len(self.overlap),
        }


class HybridRetriever:
    """Combines a dense ``Retriever`` with a BM25 ``BM25Index`` via RRF.

    Public surface mirrors ``Retriever.retrieve(query) -> List[Hit]`` so
    the rest of the pipeline doesn't care whether it's getting dense,
    BM25, or fused results.
    """

    def __init__(self,
                 dense: Retriever,
                 bm25: Optional[BM25Index] = None,
                 config: Optional[HybridConfig] = None):
        self.dense = dense
        self.bm25 = bm25 or BM25Index()
        self.config = config or HybridConfig()
        self._docs: List[str] = []
        self.last_diagnostics: Optional[HybridDiagnostics] = None

    # ----- index construction -------------------------------------------------

    def build(self, documents: Sequence[str]) -> "HybridRetriever":
        """Build both sub-indices over the same corpus."""
        if not documents:
            raise ValueError("Cannot build a hybrid index over an empty corpus.")
        self._docs = list(documents)
        # Dense index builds and embeds. BM25 builds from raw text.
        self.dense.build(self._docs)
        self.bm25.build(self._docs)
        return self

    @property
    def n_docs(self) -> int:
        return len(self._docs)

    # ----- query --------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> List[Hit]:
        """Run both retrievers, fuse with RRF, return top-k ``Hit``s.

        ``Hit.score`` carries the RRF fused score so downstream code
        (Phase 2 NLI, Phase 4 adaptive controller) can use a single,
        well-defined "how relevant is this evidence" number per document.
        """
        if not self._docs:
            raise RuntimeError("Hybrid index not built. Call build(documents) first.")
        k = min(top_k or self.dense.top_k, len(self._docs))
        if k <= 0:
            return []

        # Pull a wider candidate set from each side so fusion has room.
        # We over-fetch by 2x but cap at corpus size to avoid wasted work.
        cand_k = min(2 * k, len(self._docs))
        dense_hits = self.dense.retrieve(query, top_k=cand_k)
        bm25_hits = self.bm25.retrieve(query, top_k=cand_k)

        fused = self._rrf_fuse(dense_hits, bm25_hits)
        fused.sort(key=lambda e: e.rrf, reverse=True)
        top = fused[:k]

        self.last_diagnostics = self._build_diagnostics(dense_hits, bm25_hits, fused, k)

        return [
            Hit(text=e.text, score=float(e.rrf), index=int(e.index))
            for e in top
        ]

    # ----- persistence (delegates to dense; BM25 lives alongside) -------------

    def save(self, out_dir: str) -> None:
        self.dense.save(out_dir)
        self.bm25.save(out_dir)

    @classmethod
    def load(cls, in_dir: str, dense: Retriever) -> "HybridRetriever":
        dense = Retriever.load(in_dir, dense.embedder)
        bm25 = BM25Index.load(in_dir)
        h = cls(dense=dense, bm25=bm25)
        h._docs = list(dense._docs)
        return h

    # ----- internals ----------------------------------------------------------

    def _rrf_fuse(self,
                  dense_hits: Sequence[Hit],
                  bm25_hits: Sequence[BM25Hit]) -> List[_RankEntry]:
        """Compute RRF scores and aggregate per-document state."""
        entries: Dict[int, _RankEntry] = {}
        # dense contribution
        for rank, h in enumerate(dense_hits, start=1):
            entry = entries.setdefault(
                h.index,
                _RankEntry(text=h.text, index=h.index),
            )
            entry.dense_score = float(h.score)
            entry.dense_rank = rank
            entry.rrf += (self.config.weight_dense
                          / (self.config.rrf_k + rank))
        # bm25 contribution
        for rank, h in enumerate(bm25_hits, start=1):
            entry = entries.setdefault(
                h.index,
                _RankEntry(text=h.text, index=h.index),
            )
            entry.bm25_score = float(h.score)
            entry.bm25_rank = rank
            entry.rrf += (self.config.weight_bm25
                          / (self.config.rrf_k + rank))
        return list(entries.values())

    def _build_diagnostics(self,
                           dense_hits: Sequence[Hit],
                           bm25_hits: Sequence[BM25Hit],
                           fused: Sequence[_RankEntry],
                           top_k: int) -> HybridDiagnostics:
        d_top = dense_hits[0].score if dense_hits else None
        b_top = bm25_hits[0].score if bm25_hits else None
        fused_sorted = sorted(fused, key=lambda e: e.rrf, reverse=True)
        top_score = fused_sorted[0].rrf if fused_sorted else 0.0
        d_idx = {h.index for h in dense_hits}
        b_idx = {h.index for h in bm25_hits}
        d_texts = [h.text for h in dense_hits]
        b_texts = [h.text for h in bm25_hits]
        return HybridDiagnostics(
            dense_top_score=d_top,
            bm25_top_score=b_top,
            fused_top_score=float(top_score),
            fused_top_rank=1 if top_score > 0 else 0,
            dense_only=[t for t in d_texts if t not in b_texts],
            bm25_only=[t for t in b_texts if t not in d_texts],
            overlap=[t for t in d_texts if t in b_texts],
            n_unique=len(fused),
        )