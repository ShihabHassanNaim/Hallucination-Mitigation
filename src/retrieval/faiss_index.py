"""FAISS dense index wrapper. Falls back to a numpy inner-product search
if faiss isn't installed.
"""
from __future__ import annotations

import numpy as np

from ..types import Evidence


class FAISSIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.docs: list[str] = []
        self.doc_ids: list[str] = []
        self.vecs: np.ndarray | None = None
        self._index = None
        self._use_faiss = False

    def add(self, doc_ids: list[str], docs: list[str], vecs: np.ndarray) -> None:
        self.doc_ids = list(doc_ids)
        self.docs = list(docs)
        self.vecs = np.asarray(vecs, dtype=np.float32)
        # normalise for cosine
        norms = np.linalg.norm(self.vecs, axis=1, keepdims=True) + 1e-12
        self.vecs = self.vecs / norms
        try:
            import faiss  # type: ignore
            self._index = faiss.IndexFlatIP(self.vecs.shape[1])
            self._index.add(self.vecs)
            self._use_faiss = True
        except Exception:
            self._index = None
            self._use_faiss = False

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[Evidence]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        qn = np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
        q = q / qn
        if self._index is not None and self._use_faiss:
            scores, idx = self._index.search(q, min(top_k, len(self.doc_ids)))
            scores = scores[0]
            idx = idx[0]
        else:
            sims = (self.vecs @ q[0]).astype(np.float32)
            order = np.argsort(-sims)[:top_k]
            idx = order
            scores = sims[order]
        out: list[Evidence] = []
        for i, s in zip(idx, scores):
            if i < 0 or i >= len(self.doc_ids):
                continue
            out.append(Evidence(doc_id=self.doc_ids[int(i)], text=self.docs[int(i)], score=float(s)))
        return out