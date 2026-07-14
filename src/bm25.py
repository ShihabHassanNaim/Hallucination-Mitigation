"""Phase 4 — BM25 lexical retriever (pure Python, no external deps).

Why a separate module
---------------------
Phase 4 builds a *hybrid* dense + lexical retriever. We keep BM25 here
rather than inside ``retriever.py`` so the dense pipeline stays untouched
and the BM25 path is independently testable.

Why no external dep
-------------------
We deliberately avoid ``rank_bm25`` / ``pyserini`` so the full test
suite still runs in MOCK mode on a plain laptop without an internet
connection. The implementation is the textbook Robertson/Sparck Jones
BM25Okapi:

    score(D, Q) = sum_{t in Q} IDF(t) *
                  (f(t, D) * (k1 + 1)) /
                  (f(t, D) + k1 * (1 - b + b * |D| / avgdl))

with the IDF floor used by Lucene/Elasticsearch:

    IDF(t) = log(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

where N = number of documents, n(t) = number containing term t, f(t, D)
= term frequency in D, |D| = document length in tokens, avgdl = average
document length, and k1 / b are the standard tuning constants.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Light English stopword list. Intentionally small; aggressive removal
# hurts recall on short queries and BM25 is cheap enough to tolerate a
# few extra terms.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "was", "were", "will", "with",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords.

    Tokens shorter than 2 chars are also dropped; they tend to be noise
    ("x", "y", "to") that overwhelms BM25 scores.
    """
    text = (text or "").lower()
    out: List[str] = []
    for tok in _TOKEN_RE.findall(text):
        if len(tok) < 2:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


@dataclass
class BM25Hit:
    """BM25 retrieval result. Mirrors ``retriever.Hit`` but with raw BM25
    scores (no cosine assumption)."""

    text: str
    score: float
    index: int


class BM25Index:
    """A minimal BM25Okapi index over an in-memory document list.

    Parameters
    ----------
    k1, b
        Standard BM25 hyperparameters. Defaults (1.5, 0.75) match
        Elasticsearch/Lucene and are safe across most corpora.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 < 0:
            raise ValueError("k1 must be >= 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        # index state
        self._docs: List[str] = []
        self._doc_tokens: List[List[str]] = []
        self._doc_lens: List[int] = []
        self._avgdl: float = 0.0
        self._df: Dict[str, int] = {}
        self._tf: List[Dict[str, int]] = []  # per-doc term -> count
        self._doc_norms: List[float] = []    # precomputed L2-ish length factor

    # ----- index construction -------------------------------------------------

    def build(self, documents: Sequence[str]) -> "BM25Index":
        if not documents:
            raise ValueError("Cannot build BM25 over an empty corpus.")
        self._docs = list(documents)
        self._doc_tokens = [tokenize(d) for d in self._docs]
        self._doc_lens = [len(toks) for toks in self._doc_tokens]
        self._avgdl = (sum(self._doc_lens) / len(self._doc_lens)
                       if self._doc_lens else 0.0)
        # document frequencies and per-doc TF
        self._df = {}
        self._tf = []
        for toks in self._doc_tokens:
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        # precompute length-normalisation factors for speed
        denom = (1.0 - self.b) + self.b * self._avgdl
        if denom <= 0:
            # empty / degenerate corpus: disable length normalisation
            self._doc_norms = [1.0 for _ in self._doc_lens]
        else:
            self._doc_norms = [
                (1.0 - self.b) + self.b * (dl / self._avgdl)
                if self._avgdl > 0 else 1.0
                for dl in self._doc_lens
            ]
        return self

    # ----- query --------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> List[BM25Hit]:
        if not self._docs:
            raise RuntimeError("BM25 index not built. Call build(documents) first.")
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        k = min(top_k or len(self._docs), len(self._docs))
        if k <= 0:
            return []

        n = len(self._docs)
        scores = [0.0] * n
        for qt in q_tokens:
            df = self._df.get(qt, 0)
            if df == 0:
                continue
            # Lucene-style IDF with a +1 inside the log so unseen-after-
            # smoothing terms never get negative weights.
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, tf_map in enumerate(self._tf):
                f = tf_map.get(qt)
                if not f:
                    continue
                norm = self._doc_norms[i] or 1.0
                numerator = f * (self.k1 + 1.0)
                denominator = f + self.k1 * norm
                scores[i] += idf * (numerator / denominator)

        # argpartition would be faster for large n; argsort is fine for
        # Phase-4 scale and stays dependency-free.
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
        return [
            BM25Hit(text=self._docs[i], score=float(scores[i]), index=int(i))
            for i in ranked if scores[i] > 0
        ]

    # ----- introspection / persistence ---------------------------------------

    @property
    def n_docs(self) -> int:
        return len(self._docs)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def vocab_size(self) -> int:
        return len(self._df)

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "bm25_docs.jsonl").open("w", encoding="utf-8") as f:
            for d in self._docs:
                f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
        payload = {
            "k1": self.k1,
            "b": self.b,
            "avgdl": self._avgdl,
            "df": self._df,
            "tf": self._tf,
            "doc_lens": self._doc_lens,
        }
        (out / "bm25_index.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    @classmethod
    def load(cls, in_dir: str | Path) -> "BM25Index":
        in_path = Path(in_dir)
        idx = cls()
        with (in_path / "bm25_docs.jsonl").open("r", encoding="utf-8") as f:
            idx._docs = [json.loads(line)["text"] for line in f if line.strip()]
        payload = json.loads((in_path / "bm25_index.json").read_text(encoding="utf-8"))
        idx.k1 = float(payload["k1"])
        idx.b = float(payload["b"])
        idx._avgdl = float(payload["avgdl"])
        idx._df = {k: int(v) for k, v in payload["df"].items()}
        idx._tf = [{k: int(v) for k, v in d.items()} for d in payload["tf"]]
        idx._doc_lens = [int(x) for x in payload["doc_lens"]]
        idx._doc_tokens = [tokenize(d) for d in idx._docs]
        denom = (1.0 - idx.b) + idx.b * idx._avgdl
        if denom <= 0:
            idx._doc_norms = [1.0 for _ in idx._doc_lens]
        else:
            idx._doc_norms = [
                (1.0 - idx.b) + idx.b * (dl / idx._avgdl)
                if idx._avgdl > 0 else 1.0
                for dl in idx._doc_lens
            ]
        return idx