"""Lightweight in-memory model doubles for unit tests.

The production code is **always** real-model — there is no mock mode.
The unit-test suite, however, needs to exercise pipeline wiring
without paying the cost of downloading 6 HuggingFace checkpoints.
This module provides minimal stand-ins for every model slot on
``Pipeline``. Each fake has the same public method surface the real
model exposes, returns deterministic outputs, and is *not* a mock of
production behaviour — it's a focused stub that exists only so tests
can run.

Use ``make_fake_pipeline(cfg)`` to get a fully-injected ``Pipeline``
instance with every slot pre-filled.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.config import Config
from src.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Hash-bucketed deterministic embedder."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for d in range(self.dim):
                out[i, d] = (hash((t, d)) % 1000) / 500.0 - 1.0
        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
            out = out / norms
        return out


class FakeGenerator:
    """Echoes the question as the answer."""

    def generate(self, system: str, user: str, max_new_tokens: int | None = None) -> str:
        # Use the context if present, otherwise just answer the question.
        if "Context:" in user:
            after = user.split("Context:", 1)[1]
            answer = after.split("\n", 1)[-1].strip()
            if answer:
                return answer.split(".")[0] + "."
        return "fake answer."


class FakeReranker:
    """Constant 0.5 score for every doc."""

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        return [0.5] * len(docs)


class FakeNLI:
    """Always says 'supported'."""

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        return "supported", 0.9


class FakeGliNER:
    """Returns no entities."""

    def extract(self, text: str, labels: list[str] | None = None) -> list[dict]:
        return []


class FakeSpacyNER:
    """Returns no entities."""

    def extract(self, text: str) -> list[dict]:
        return []


class FakeXGB:
    """Returns 0.5 probability."""

    def predict_proba(self, features: list[float]) -> float:
        return 0.5


class FakeEEDC:
    """Returns a constant calibrated score."""

    def score(self, nli_entropy: float, retrieval_top1: float, self_consistency: float) -> float:
        return 0.7


class FakeCalibrator:
    """Identity temperature scaler."""

    def transform(self, x: float) -> float:
        return x


class FakeRetriever:
    """Returns empty results."""

    n_docs = 0

    def add(self, doc_ids: list[str], docs: list[str], vecs: np.ndarray) -> None:
        self.n_docs = len(docs)

    def search(self, query: str, query_vec: np.ndarray, top_k: int = 5) -> list[Any]:
        return []


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def make_fake_pipeline(cfg: Config | None = None) -> Pipeline:
    """Build a Pipeline with every model slot pre-filled with fakes.

    The real ``Pipeline.__post_init__`` is bypassed — instead we set
    every slot directly. No HuggingFace download happens.
    """
    if cfg is None:
        from src.config import load_config

        cfg = load_config()

    pipe = Pipeline.__new__(Pipeline)
    pipe.cfg = cfg
    pipe.config = cfg
    pipe.embedder = FakeEmbedder()
    pipe.retriever = FakeRetriever()
    pipe.nli = FakeNLI()
    pipe.generator = FakeGenerator()
    pipe.generator_qwen = None
    pipe.reranker = FakeReranker()
    pipe.gliner_ner = FakeGliNER()
    pipe.spacy_ner = FakeSpacyNER()
    pipe.early_classifier = FakeXGB()
    pipe.eedc = FakeEEDC()
    pipe.calibrator = FakeCalibrator()
    return pipe


__all__ = [
    "FakeEmbedder",
    "FakeGenerator",
    "FakeReranker",
    "FakeNLI",
    "FakeGliNER",
    "FakeSpacyNER",
    "FakeXGB",
    "FakeEEDC",
    "FakeCalibrator",
    "FakeRetriever",
    "make_fake_pipeline",
]