"""BM25 sparse index. Pure-python implementation (no external deps)."""
from __future__ import annotations

import math
import re
from collections import Counter

from ..types import Evidence


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[list[str]] = []
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.N: int = 0
        self.doc_lens: list[int] = []

    def add(self, doc_ids: list[str], docs: list[str]) -> None:
        self.doc_ids = list(doc_ids)
        self.doc_texts = list(docs)
        self.docs = [_tokenize(d) for d in docs]
        self.doc_lens = [len(d) for d in self.docs]
        self.N = len(self.docs)
        self.avgdl = (sum(self.doc_lens) / self.N) if self.N else 0.0
        self.df = Counter()
        for d in self.docs:
            for term in set(d):
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(((self.N - n + 0.5) / (n + 0.5)) + 1.0)

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        qterms = _tokenize(query)
        if not qterms or not self.docs:
            return []
        scores = [0.0] * self.N
        for q in qterms:
            idf = self._idf(q)
            for i, d in enumerate(self.docs):
                tf = d.count(q)
                if tf == 0:
                    continue
                dl = self.doc_lens[i]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)
        order = sorted(range(self.N), key=lambda i: -scores[i])[:top_k]
        return [
            Evidence(doc_id=self.doc_ids[i], text=self.doc_texts[i], score=float(scores[i]))
            for i in order
            if scores[i] > 0
        ]