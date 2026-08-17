"""Hybrid retrieval: dense FAISS + sparse BM25, fused by reciprocal rank."""
from __future__ import annotations

import numpy as np

from ..types import Evidence
from .bm25_index import BM25Index
from .faiss_index import FAISSIndex


class HybridRetriever:
    def __init__(self, dim: int, rrf_k: float = 60.0, k1: float = 1.5, b: float = 0.75):
        self.faiss = FAISSIndex(dim=dim)
        self.bm25 = BM25Index(k1=k1, b=b)
        self.rrf_k = rrf_k

    def add(self, doc_ids: list[str], docs: list[str], dense_vecs: np.ndarray) -> None:
        self.faiss.add(doc_ids, docs, dense_vecs)
        self.bm25.add(doc_ids, docs)

    @property
    def n_docs(self) -> int:
        return len(self.faiss.doc_ids)

    def search(self, query: str, query_vec: np.ndarray, top_k: int = 5) -> list[Evidence]:
        dense_hits = self.faiss.search(query_vec, top_k=top_k * 3)
        sparse_hits = self.bm25.search(query, top_k=top_k * 3)
        fused: dict[str, float] = {}
        text_lookup: dict[str, str] = {}

        for rank, ev in enumerate(dense_hits):
            fused[ev.doc_id] = fused.get(ev.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            text_lookup[ev.doc_id] = ev.text
        for rank, ev in enumerate(sparse_hits):
            fused[ev.doc_id] = fused.get(ev.doc_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            text_lookup[ev.doc_id] = ev.text

        if not fused:
            return []
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        out: list[Evidence] = []
        for doc_id, rrf_score in ordered[:top_k]:
            out.append(Evidence(doc_id=doc_id, text=text_lookup[doc_id], score=float(rrf_score)))
        return out