"""FAISS-backed dense retriever.

The retriever owns:
  * an Embedder (real or mock)
  * a FAISS index (IndexFlatIP for correctness, swap for IVFPQ at scale)
  * a parallel list of document strings

Phase 4 (adaptive retrieval) will replace this with a hybrid BM25 + dense
retriever behind the same `retrieve(query) -> List[Hit]` interface.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:  # pragma: no cover - exercised on minimal installs
    faiss = None  # type: ignore
    _HAS_FAISS = False


@dataclass
class Hit:
    text: str
    score: float
    index: int


class Retriever:
    """Build an index over a list of documents, then query top-k."""

    def __init__(self, embedder, top_k: int = 5):
        self.embedder = embedder
        self.top_k = top_k
        self._docs: List[str] = []
        self._index = None

    # ----- index construction -------------------------------------------------

    def build(self, documents: Sequence[str]) -> "Retriever":
        if not documents:
            raise ValueError("Cannot build an index over an empty corpus.")
        self._docs = list(documents)
        embeddings = self.embedder.encode(self._docs, normalize=True)
        dim = embeddings.shape[1]

        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(dim)  # inner-product == cosine when normalized
            self._index.add(np.ascontiguousarray(embeddings))
        else:
            # NumPy fallback for environments without faiss. Slower but correct.
            self._index = _NumpyIPIndex(embeddings)
        return self

    # ----- query ---------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> List[Hit]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build(documents) first.")
        k = min(top_k or self.top_k, len(self._docs))
        if k <= 0:
            return []

        qvec = self.embedder.encode([query], normalize=True)
        scores, ids = self._index.search(qvec, k)
        scores = scores[0]
        ids = ids[0]

        hits: List[Hit] = []
        for score, idx in zip(scores, ids):
            if idx < 0:
                continue
            hits.append(Hit(text=self._docs[idx], score=float(score), index=int(idx)))
        return hits

    # ----- persistence ---------------------------------------------------------

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Documents are plain text — saved as JSONL for portability.
        with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
        # Embeddings: keep alongside so re-indexing is fast (optional).
        # Index itself: pickle via faiss when available; fall back to numpy.
        meta = {"top_k": self.top_k, "embedder": self.embedder.model_name, "mock": self.embedder.mock}
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        if _HAS_FAISS and hasattr(self._index, "serialize"):
            # Some faiss builds expose serialize/deserialize; default does not.
            pass

        # Universal fallback: pickle the index object. Works for faiss IndexFlat.
        with (out / "index.pkl").open("wb") as f:
            pickle.dump(self._index, f)

    @classmethod
    def load(cls, in_dir: str | Path, embedder) -> "Retriever":
        in_path = Path(in_dir)
        with (in_path / "docs.jsonl").open("r", encoding="utf-8") as f:
            docs = [json.loads(line)["text"] for line in f if line.strip()]
        meta = json.loads((in_path / "meta.json").read_text(encoding="utf-8"))
        r = cls(embedder=embedder, top_k=meta.get("top_k", 5))
        r._docs = docs
        with (in_path / "index.pkl").open("rb") as f:
            r._index = pickle.load(f)
        return r


class _NumpyIPIndex:
    """Minimal in-memory inner-product index used when faiss isn't installed."""

    def __init__(self, embeddings: np.ndarray):
        self._matrix = embeddings  # [n, d], rows L2-normalized.

    def search(self, queries: np.ndarray, k: int):
        # Cosine sim since rows are normalized.
        sims = queries @ self._matrix.T  # [q, n]
        # argpartition would be faster; keep argsort for simplicity / Phase 1.
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype(np.float32), idx.astype(np.int64)