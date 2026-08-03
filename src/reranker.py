"""Step 7 — Cross-encoder evidence reranking.

This module is an *optional* enhancement on top of Phase 4 hybrid retrieval.
It re-scores each (query, evidence) pair with a cross-encoder model and
re-orders the candidate list. The recommended model is **BAAI/bge-reranker-v2-m3**
(also fine for multilingual) but a deterministic token-overlap fallback is
shipped for mock mode.

When to use
-----------
The README recommends treating reranking as an enhancement (not a mandatory
step). Enable with ``CRISP_RERANKER=1`` (or ``reranker.enabled: true`` in
``configs/default.yaml``) and report its effect in an ablation study.

Plain-English summary
---------------------
After the hybrid retriever returns its top-K documents, this module asks
"which of these actually answers the question?" and reorders them so the
best evidence sits on top. The downstream NLI verifier then sees stronger
context, which usually lowers the hallucination rate.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .retriever import Hit

logger = logging.getLogger(__name__)


# --- public dataclasses -----------------------------------------------------

@dataclass
class RerankTrace:
    """Trace payload for one reranking call (mirrors `RetrievalTrace`)."""
    model_name: str
    input_top_k: int
    output_top_k: int
    scores: List[float] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)
    mock: bool = False

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "input_top_k": self.input_top_k,
            "output_top_k": self.output_top_k,
            "scores": [round(s, 4) for s in self.scores],
            "indices": list(self.indices),
            "mock": self.mock,
        }


# --- mock + real rerankers --------------------------------------------------

def _token_overlap_score(query: str, evidence: str) -> float:
    """Deterministic mock score: weighted overlap of query tokens in evidence.

    Uses the *full* query vocabulary (no stopword removal) and weights rare
    tokens higher so that the mock mirrors what a cross-encoder tends to do:
    reward documents that contain the meaningful query terms.
    """
    q_tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t]
    e_tokens = re.findall(r"[a-z0-9]+", evidence.lower())
    if not q_tokens or not e_tokens:
        return 0.0
    e_counts: Dict[str, int] = {}
    for t in e_tokens:
        e_counts[t] = e_counts.get(t, 0) + 1
    score = 0.0
    for qt in q_tokens:
        if qt in e_counts:
            # Reward matches; longer evidence is mildly penalised.
            score += 1.0 / math.sqrt(1.0 + e_counts[qt])
    return score / math.sqrt(max(1, len(e_tokens)))


class CrossEncoderReranker:
    """Wraps a cross-encoder model behind a small, mock-friendly API.

    Parameters
    ----------
    model_name : str
        HF model id of the cross-encoder. Default ``BAAI/bge-reranker-v2-m3``.
    top_k : int
        How many hits to keep after reranking.
    mock : bool
        If True, use the deterministic token-overlap heuristic. Also chosen
        automatically when ``CRISP_MOCK=1`` or the HF model fails to load.
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: Optional[str] = None,
                 top_k: int = 5,
                 mock: Optional[bool] = None):
        env_mock = os.environ.get("CRISP_MOCK", "").lower() in {"1", "true", "yes"}
        if mock is None:
            mock = env_mock
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_k = top_k
        self.mock = mock
        self._model = None
        self.last_trace: Optional[RerankTrace] = None
        if not self.mock:
            self._try_load_real_model()

    # ----- model loading ----------------------------------------------------

    def _try_load_real_model(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model_name)
            logger.info("Loaded cross-encoder reranker: %s", self.model_name)
        except Exception as e:  # pragma: no cover - optional dep
            logger.warning(
                "Could not load reranker '%s' (%s); falling back to mock scorer.",
                self.model_name, e,
            )
            self.mock = True

    # ----- public API -------------------------------------------------------

    def rerank(self, query: str, hits: Sequence[Hit],
               top_k: Optional[int] = None) -> List[Hit]:
        """Re-score and re-order ``hits`` for ``query``. Returns ``top_k`` items."""
        if not hits:
            self.last_trace = RerankTrace(self.model_name, 0, 0, mock=self.mock)
            return []
        k = top_k or self.top_k
        scored = self._score(query, hits)
        scored.sort(key=lambda h: h.score, reverse=True)
        out = list(scored[:k])
        self.last_trace = RerankTrace(
            model_name=self.model_name,
            input_top_k=len(hits),
            output_top_k=len(out),
            scores=[h.score for h in out],
            indices=[h.index for h in out],
            mock=self.mock,
        )
        return out

    # ----- internals --------------------------------------------------------

    def _score(self, query: str, hits: Sequence[Hit]) -> List[Hit]:
        if self.mock or self._model is None:
            return [
                Hit(text=h.text, score=_token_overlap_score(query, h.text), index=h.index)
                for h in hits
            ]
        try:
            pairs = [(query, h.text) for h in hits]
            raw = self._model.predict(pairs, show_progress_bar=False)
            return [
                Hit(text=h.text, score=float(s), index=h.index)
                for h, s in zip(hits, raw)
            ]
        except Exception as e:  # pragma: no cover - safety net
            logger.warning("Rerank inference failed (%s); using mock scores.", e)
            return [
                Hit(text=h.text, score=_token_overlap_score(query, h.text), index=h.index)
                for h in hits
            ]


# --- convenience wrappers ---------------------------------------------------

class RerankingRetriever:
    """Adapter: any base retriever + a CrossEncoderReranker.

    Use this when you want to slot reranking into `Pipeline` without writing
    glue code. ``Pipeline`` will check for a ``rerank`` method on the
    retriever and call it after the base ``retrieve``.
    """

    def __init__(self, base, reranker: CrossEncoderReranker):
        self.base = base
        self.reranker = reranker
        self.last_base_hits: List[Hit] = []
        self.last_rerank_trace: Optional[RerankTrace] = None

    def build(self, documents: List[str]) -> None:
        self.base.build(documents)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Hit]:
        base_hits = self.base.retrieve(query)
        self.last_base_hits = list(base_hits)
        reranked = self.reranker.rerank(query, base_hits, top_k=top_k)
        self.last_rerank_trace = self.reranker.last_trace
        return reranked

    def save(self, out_dir: str) -> None:
        self.base.save(out_dir)

    @classmethod
    def load(cls, in_dir: str, base_factory, reranker: CrossEncoderReranker):
        base = base_factory.load(in_dir, base_factory)
        return cls(base=base, reranker=reranker)
