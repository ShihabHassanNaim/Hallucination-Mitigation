"""End-to-end smoke tests for the Phase 1 RAG pipeline.

These tests run with CRISP_MOCK=1, so no models are downloaded and no GPU
is required. They cover:
  - config loading & env overrides
  - embedder (mock mode)
  - retriever (build, save, load, retrieve)
  - generator (mock mode)
  - pipeline (single + batch, retrieval produces hits, answer is non-empty)
  - evaluation (per-item + aggregate)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Force MOCK mode for the entire test session — do this BEFORE importing
# the package, so modules see the env var at construction time.
os.environ["CRISP_MOCK"] = "1"
os.environ.setdefault("CRISP_INDEX_PATH", "data/test_index")

import pytest

from src.config import load_config
from src.data_loader import corpus_from_dataset, load_dataset
from src.embeddings import Embedder
from src.evaluation import aggregate, score_item, _is_refusal, _norm, _tokens
from src.generator import Generator
from src.pipeline import Pipeline
from src.retriever import Retriever


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_loads_defaults():
    cfg = load_config()
    assert cfg.mock is True
    assert cfg.retrieval.top_k >= 1
    assert cfg.generator.model_name
    assert "{question}" in cfg.prompt.user_template


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("CRISP_TOPK", "8")
    monkeypatch.setenv("CRISP_TEMP", "0.5")
    cfg = load_config()
    assert cfg.retrieval.top_k == 8
    assert cfg.generator.temperature == 0.5


# ---------------------------------------------------------------------------
# embeddings
# ---------------------------------------------------------------------------

def test_embedder_mock_deterministic():
    emb = Embedder(model_name="mock", dim=128, mock=True)
    v1 = emb.encode(["hello world"], normalize=True)[0]
    v2 = emb.encode(["hello world"], normalize=True)[0]
    # Constructor `dim` is honoured.
    assert v1.shape == (128,)
    # Determinism — same input -> same vector.
    assert (v1 == v2).all()
    # Normalised -> unit norm.
    import numpy as np
    assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-4

    # Sanity: a different dim also round-trips through the encode() contract.
    emb_small = Embedder(model_name="mock", dim=64, mock=True)
    vs = emb_small.encode(["alpha", "beta"], normalize=True)
    assert vs.shape == (2, 64)


def test_embedder_mock_different_texts_differ():
    emb = Embedder(model_name="mock", dim=128, mock=True)
    v1 = emb.encode(["france paris capital"])[0]
    v2 = emb.encode(["python programming language"])[0]
    import numpy as np
    sim = float(np.dot(v1, v2))
    assert sim < 0.5  # nothing semantically close in the mock embedder.


# ---------------------------------------------------------------------------
# retriever
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_corpus():
    return [
        "Paris is the capital of France.",
        "The novel '1984' was written by George Orwell.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Mars is known as the Red Planet due to iron oxide on its surface.",
        "PyTorch is a deep learning library primarily used with Python.",
    ]


def test_retriever_build_and_query(tiny_corpus, tmp_path):
    emb = Embedder(model_name="mock", dim=128, mock=True)
    r = Retriever(embedder=emb, top_k=3).build(tiny_corpus)
    hits = r.retrieve("What is the capital of France?")
    assert len(hits) == 3
    # Each hit has the expected fields.
    for h in hits:
        assert h.text in tiny_corpus
        assert -1.0 <= h.score <= 1.0
        assert h.index >= 0


def test_retriever_save_and_load(tiny_corpus, tmp_path):
    emb = Embedder(model_name="mock", dim=128, mock=True)
    r = Retriever(embedder=emb, top_k=2).build(tiny_corpus)
    out = tmp_path / "idx"
    r.save(out)

    assert (out / "docs.jsonl").exists()
    assert (out / "meta.json").exists()
    assert (out / "index.pkl").exists()

    r2 = Retriever.load(out, emb)
    hits1 = r.retrieve("PyTorch deep learning Python")
    hits2 = r2.retrieve("PyTorch deep learning Python")
    assert [h.text for h in hits1] == [h.text for h in hits2]


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

def test_generator_mock_returns_string():
    g = Generator(model_name="mock", mock=True)
    out = g.generate(
        system_prompt="Be concise.",
        user_prompt="Context:\nParis is the capital of France.\n\nQuestion: What is the capital of France?\n\nAnswer:",
    )
    assert isinstance(out, str)
    assert out.strip() != ""


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end(tiny_corpus, tmp_path):
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")

    assert result.answer.strip() != ""
    assert len(result.retrieved_docs) > 0
    assert result.iterations == 1
    # Phase 2: confidence is the mean EEDC across detected claims. With
    # mock embedder noise + mock generator nonsense, it won't be 1.0, but it
    # must still be in the [0, 1] interval and claim_verdicts must be populated.
    assert 0.0 <= result.confidence <= 1.0
    assert result.hallucination_rate >= 0.0
    assert "retrieve" in result.timings_ms
    assert "generate" in result.timings_ms
    assert "detect" in result.timings_ms
    assert result.timings_ms["total"] >= 0
    assert "{question}" not in result.prompt  # template was filled in.


def test_pipeline_save_and_load_index(tiny_corpus, tmp_path):
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    out = tmp_path / "idx"
    pipeline.save_index(str(out))

    cfg2 = load_config()
    pipeline2 = Pipeline(cfg2).load_index(str(out))
    r = pipeline2.run("Who wrote '1984'?")
    assert r.answer.strip() != ""
    assert len(r.retrieved_docs) > 0


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def test_synthetic_dataset_loads():
    items = load_dataset("synthetic")
    assert len(items) >= 5
    for it in items:
        assert "question" in it
        assert "knowledge" in it


def test_corpus_from_dataset_dedupes():
    items = [
        {"id": "1", "question": "q1", "knowledge": "Doc A"},
        {"id": "2", "question": "q2", "knowledge": "Doc A"},  # duplicate
        {"id": "3", "question": "q3", "knowledge": "Doc B"},
    ]
    docs = corpus_from_dataset(items)
    assert docs == ["Doc A", "Doc B"]


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_evaluation_helpers():
    assert _norm("Hello, World!") == "hello world"
    assert _tokens("Hello, World!") == ["hello", "world"]
    assert _is_refusal("I don't know.") is True
    assert _is_refusal("Paris.") is False


def test_evaluation_perfect_match():
    item = {
        "id": "1",
        "answer": "Paris",
        "reference_answer": "paris.",
        "retrieved_docs": [{"text": "Paris is the capital of France.", "score": 0.9}],
    }
    s = score_item(item)
    assert s.exact_match == 1.0
    assert s.token_f1 == 1.0
    assert s.refused == 0.0
    # 'paris' IS in the retrieved text -> ungrounded_rate = 0.
    assert s.ungrounded_token_rate == 0.0


def test_evaluation_ungrounded_rate():
    item = {
        "id": "1",
        "answer": "Tokyo is the capital.",
        "reference_answer": "Paris",
        "retrieved_docs": [{"text": "Paris is the capital of France.", "score": 0.9}],
    }
    s = score_item(item)
    # 'tokyo' is not in the retrieved doc -> 1 ungrounded of 4 tokens = 0.25
    assert 0.2 < s.ungrounded_token_rate < 0.3


def test_evaluation_aggregate():
    items = [
        {"id": "1", "answer": "Paris", "reference_answer": "Paris",
         "retrieved_docs": [{"text": "Paris", "score": 0.9}]},
        {"id": "2", "answer": "I don't know.", "reference_answer": "Orwell",
         "retrieved_docs": [{"text": "Orwell", "score": 0.5}]},
    ]
    scores = [score_item(it) for it in items]
    agg = aggregate(scores)
    assert agg["n"] == 2
    assert 0.0 <= agg["exact_match"] <= 1.0
    assert agg["refusal_rate"] == 0.5


def test_evaluation_json_roundtrip(tmp_path):
    out = tmp_path / "metrics.json"
    items = [
        {"id": "1", "answer": "Paris", "reference_answer": "Paris",
         "retrieved_docs": [{"text": "Paris", "score": 0.9}]},
    ]
    scores = [score_item(it) for it in items]
    agg = aggregate(scores)
    from src.evaluation import write_metrics
    write_metrics(agg, out)
    loaded = json.loads(out.read_text())
    assert loaded["n"] == 1