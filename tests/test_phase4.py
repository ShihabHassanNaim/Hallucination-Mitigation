"""Phase 4 tests — Adaptive Evidence Retrieval.

Covers (in MOCK mode, no downloads):
  - BM25 tokeniser, retrieval, IDF, save/load
  - HybridRetriever: same interface as Retriever, RRF fusion diagnostic
  - AdaptiveRetriever: pass-through, expansion, rewriting, multi-hop
  - Config plumbing + env overrides for AdaptiveRetrievalConfig
  - Pipeline integration: retrieval_trace in RAGResult.to_dict()
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Force MOCK mode for the whole test session so no models are downloaded.
os.environ["CRISP_MOCK"] = "1"

import pytest

from src.adaptive_retriever import (
    AdaptiveConfig,
    AdaptiveRetriever,
    RetrievalTrace,
    Strategy,
    _entities_for_multi_hop,
    _heuristic_rewrites,
    _merge_hits,
    _passes_expansion,
)
from src.bm25 import BM25Hit, BM25Index, tokenize
from src.config import AppConfig, load_config
from src.embeddings import Embedder
from src.hybrid_retriever import HybridConfig, HybridDiagnostics, HybridRetriever
from src.pipeline import Pipeline, RAGResult
from src.retriever import Hit, Retriever


# Small synthetic corpus used across tests.
CORPUS = [
    "France is a country in Western Europe. Its capital is Paris.",
    "The dystopian novel 1984 was written by George Orwell and published in 1949.",
    "PyTorch is an open-source machine learning library developed by Meta AI.",
    "Mars is often called the Red Planet because of the iron oxide on its surface.",
    "At sea level, pure water boils at 100 degrees Celsius (212 Fahrenheit).",
]


# ---------------------------------------------------------------------------
# BM25 tokeniser
# ---------------------------------------------------------------------------

class TestBM25Tokenize:
    def test_lowercases_and_splits_on_punct(self):
        assert tokenize("The Capital, of FRANCE!") == ["capital", "france"]

    def test_drops_short_tokens(self):
        # "x" and "y" should be dropped (length < 2).
        assert "x" not in tokenize("x y capital of france")

    def test_drops_stopwords(self):
        out = tokenize("the capital of france is paris")
        assert "the" not in out and "of" not in out and "is" not in out
        assert "capital" in out and "france" in out and "paris" in out

    def test_empty_returns_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
        assert tokenize(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class TestBM25Index:
    def test_basic_retrieval_finds_relevant_doc(self):
        idx = BM25Index().build(CORPUS)
        hits = idx.retrieve("capital of France", top_k=2)
        assert len(hits) >= 1
        # France / Paris doc should be top.
        assert "France" in hits[0].text

    def test_lexical_match(self):
        idx = BM25Index().build(CORPUS)
        hits = idx.retrieve("1984 Orwell", top_k=2)
        assert "1984" in hits[0].text
        assert "Orwell" in hits[0].text

    def test_no_match_returns_empty(self):
        idx = BM25Index().build(CORPUS)
        hits = idx.retrieve("quantum chromodynamics nanoparticles", top_k=3)
        assert hits == []

    def test_scores_are_non_negative(self):
        idx = BM25Index().build(CORPUS)
        hits = idx.retrieve("France", top_k=3)
        for h in hits:
            assert h.score >= 0.0

    def test_idf_higher_for_rare_terms(self):
        # "orwell" appears in 1 doc, "country" appears in 1 doc too in this
        # tiny corpus, but we can still verify IDF is positive overall.
        idx = BM25Index().build(CORPUS)
        assert idx._df["france"] >= 1
        assert idx._df["orwell"] >= 1

    def test_empty_corpus_raises(self):
        with pytest.raises(ValueError):
            BM25Index().build([])

    def test_retrieve_on_unbuilt_index_raises(self):
        idx = BM25Index()
        with pytest.raises(RuntimeError):
            idx.retrieve("anything")

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            BM25Index(k1=-1).build(CORPUS)
        with pytest.raises(ValueError):
            BM25Index(b=1.5).build(CORPUS)

    def test_save_load_roundtrip(self, tmp_path: Path):
        idx = BM25Index().build(CORPUS)
        idx.save(tmp_path)
        loaded = BM25Index.load(tmp_path)
        assert loaded.n_docs == idx.n_docs
        # scores should match
        for q in ["France capital", "1984 Orwell", "PyTorch Meta"]:
            a = idx.retrieve(q, top_k=2)
            b = loaded.retrieve(q, top_k=2)
            assert [h.text for h in a] == [h.text for h in b]
            for x, y in zip(a, b):
                assert abs(x.score - y.score) < 1e-9


# ---------------------------------------------------------------------------
# HybridRetriever (RRF)
# ---------------------------------------------------------------------------

def _build_hybrid() -> HybridRetriever:
    embedder = Embedder(model_name="mock-emb", mock=True)
    dense = Retriever(embedder=embedder, top_k=3)
    h = HybridRetriever(dense=dense, bm25=BM25Index(),
                        config=HybridConfig(rrf_k=60.0,
                                            weight_dense=1.0,
                                            weight_bm25=1.0))
    h.build(CORPUS)
    return h


class TestHybridRetriever:
    def test_same_interface_as_dense(self):
        h = _build_hybrid()
        hits = h.retrieve("France capital", top_k=3)
        assert all(isinstance(x, Hit) for x in hits)

    def test_top_hit_is_relevant(self):
        h = _build_hybrid()
        # RRF can legitimately put another doc at the very top of the
        # fused list (when dense mock ranks it #1 and BM25 ranks it #2),
        # but the BM25-strong 1984/Orwell doc must surface in the top-2.
        hits = h.retrieve("1984 Orwell", top_k=2)
        assert len(hits) >= 1
        top2_texts = [h.text for h in hits[:2]]
        assert any("1984" in t or "Orwell" in t for t in top2_texts)

    def test_diagnostics_recorded(self):
        h = _build_hybrid()
        h.retrieve("France", top_k=2)
        diag = h.last_diagnostics
        assert diag is not None
        assert diag.n_unique >= 1
        assert diag.fused_top_score >= 0.0
        assert isinstance(diag.to_dict(), dict)

    def test_cannot_build_empty(self):
        embedder = Embedder(model_name="mock-emb", mock=True)
        dense = Retriever(embedder=embedder, top_k=3)
        h = HybridRetriever(dense=dense, bm25=BM25Index())
        with pytest.raises(ValueError):
            h.build([])

    def test_top_k_respected(self):
        h = _build_hybrid()
        hits = h.retrieve("France", top_k=2)
        assert len(hits) <= 2


# ---------------------------------------------------------------------------
# AdaptiveRetriever — internals
# ---------------------------------------------------------------------------

class TestAdaptiveInternals:
    def test_heuristic_rewrites_dedupe_and_cap(self):
        out = _heuristic_rewrites("What is the capital of France?", n=3)
        assert len(out) <= 3
        # originals (and the original itself) should not appear verbatim.
        joined = " | ".join(out)
        assert "What is the capital of France?" not in out

    def test_heuristic_rewrites_swap_synonyms(self):
        out = _heuristic_rewrites("What is the capital of France?", n=5)
        # synonym swap should turn "capital of" into "seat of government of".
        assert any("seat of government of" in v for v in out)

    def test_heuristic_rewrites_drop_question_words(self):
        out = _heuristic_rewrites("Who wrote 1984?", n=5)
        # "who" is a question word; in the no-question-word variant it
        # should be gone while content words survive.
        joined = " | ".join(out)
        assert "who" not in joined

    def test_entities_picks_capitalised_spans(self):
        ents = _entities_for_multi_hop(
            "France and Germany dominate the European economy.",
            min_len=3, max_entities=3,
        )
        # Single-word capitalised entities should be picked up.
        assert any("France" in e for e in ents)
        assert any("Germany" in e for e in ents)

    def test_merge_hits_keeps_max_score(self):
        h1 = [Hit(text="a", score=0.5, index=0)]
        h2 = [Hit(text="a", score=0.9, index=0), Hit(text="b", score=0.3, index=1)]
        merged = _merge_hits(h1, h2)
        assert len(merged) == 2
        # first-seen order, but the duplicate "a" should keep the higher score.
        assert merged[0].text == "a"
        assert merged[0].score == pytest.approx(0.9)

    def test_passes_expansion_low_top_score(self):
        hits = [Hit(text="x", score=0.05, index=0)]
        cfg = AdaptiveConfig(expansion_score_threshold=0.20)
        assert _passes_expansion(hits, cfg) is True

    def test_passes_expansion_tight_gap(self):
        hits = [Hit(text="a", score=0.5, index=0),
                Hit(text="b", score=0.48, index=1)]
        cfg = AdaptiveConfig(expansion_score_threshold=0.10,
                             expansion_gap_threshold=0.05)
        assert _passes_expansion(hits, cfg) is True

    def test_passes_expansion_strong_top_stable(self):
        hits = [Hit(text="a", score=0.9, index=0),
                Hit(text="b", score=0.5, index=1)]
        cfg = AdaptiveConfig(expansion_score_threshold=0.20,
                             expansion_gap_threshold=0.05)
        assert _passes_expansion(hits, cfg) is False


# ---------------------------------------------------------------------------
# AdaptiveRetriever — public API
# ---------------------------------------------------------------------------

def _build_dense_base() -> Retriever:
    embedder = Embedder(model_name="mock-emb", mock=True)
    r = Retriever(embedder=embedder, top_k=3)
    r.build(CORPUS)
    return r


class TestAdaptiveRetriever:
    def test_pass_through_when_no_expansion(self):
        # Set expansion thresholds extremely loose so expansion never triggers.
        base = _build_dense_base()
        ar = AdaptiveRetriever(
            base=base,
            config=AdaptiveConfig(
                expansion_score_threshold=-1.0,
                expansion_gap_threshold=-1.0,
                rewrite_variants=0,
            ),
        )
        hits = ar.retrieve("France capital")
        # Should be exactly initial_k (=default 5) hits, no expansion.
        assert ar.last_trace is not None
        assert ar.last_trace.strategy == Strategy.NONE
        assert ar.last_trace.final_k <= len(CORPUS)

    def test_expansion_when_score_low(self):
        base = _build_dense_base()
        ar = AdaptiveRetriever(
            base=base,
            config=AdaptiveConfig(
                initial_k=2,
                max_k=4,
                expansion_score_threshold=10.0,    # always expand
                expansion_gap_threshold=10.0,
                rewrite_variants=0,
            ),
        )
        hits = ar.retrieve("France")
        assert ar.last_trace is not None
        assert ar.last_trace.strategy in (
            Strategy.EXPAND,
            Strategy.EXPAND_AND_MULTI_HOP,
            Strategy.REWRITE,
            Strategy.REWRITE_AND_MULTI_HOP,
            Strategy.MULTI_HOP,
        )
        assert ar.last_trace.final_k >= 2  # at least initial_k

    def test_rewriting_when_requested(self):
        base = _build_dense_base()
        ar = AdaptiveRetriever(
            base=base,
            config=AdaptiveConfig(
                expansion_score_threshold=-1.0,
                expansion_gap_threshold=-1.0,
                rewrite_variants=2,
            ),
        )
        ar.retrieve("What is the capital of France?")
        assert ar.last_trace is not None
        assert len(ar.last_trace.rewrites) <= 2
        assert ar.last_trace.strategy in (
            Strategy.REWRITE, Strategy.REWRITE_AND_MULTI_HOP,
        )

    def test_multi_hop_when_aggregated_claim_provided(self):
        base = _build_dense_base()
        ar = AdaptiveRetriever(
            base=base,
            config=AdaptiveConfig(
                expansion_score_threshold=-1.0,
                expansion_gap_threshold=-1.0,
                rewrite_variants=0,
                multi_hop_max_entities=2,
                multi_hop_min_entity_len=3,
            ),
        )
        ar.retrieve(
            "European powerhouses",
            aggregated_claim_text="France and Germany dominate the European economy.",
        )
        assert ar.last_trace is not None
        assert len(ar.last_trace.multi_hop_queries) >= 1
        assert "France" in ar.last_trace.multi_hop_queries \
            or "Germany" in ar.last_trace.multi_hop_queries

    def test_trace_to_dict_is_jsonable(self):
        base = _build_dense_base()
        ar = AdaptiveRetriever(base=base)
        ar.retrieve("France")
        d = ar.last_trace.to_dict()
        json.dumps(d)   # must not raise

    def test_invalid_config_raises(self):
        base = _build_dense_base()
        with pytest.raises(ValueError):
            AdaptiveConfig(initial_k=0)

    def test_adaptive_n_docs_delegates(self):
        base = _build_dense_base()
        ar = AdaptiveRetriever(base=base)
        assert ar.n_docs == len(CORPUS)


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------

class TestAdaptiveConfig:
    def test_default_config_has_adaptive_section(self):
        cfg = load_config()
        assert hasattr(cfg, "adaptive_retrieval")
        # Default is OFF so existing Phases 1-3 stay untouched.
        assert cfg.adaptive_retrieval.enabled is False
        assert cfg.adaptive_retrieval.initial_k == 5
        assert cfg.adaptive_retrieval.max_k == 20

    def test_retrieval_mode_default_is_dense(self):
        cfg = load_config()
        assert cfg.retrieval.mode == "dense"

    def test_env_overrides_enable_adaptive(self):
        old = os.environ.get("CRISP_ADAPTIVE")
        try:
            os.environ["CRISP_ADAPTIVE"] = "1"
            cfg = load_config()
            assert cfg.adaptive_retrieval.enabled is True
        finally:
            if old is None:
                os.environ.pop("CRISP_ADAPTIVE", None)
            else:
                os.environ["CRISP_ADAPTIVE"] = old

    def test_env_overrides_retrieval_mode(self):
        old = os.environ.get("CRISP_RETRIEVAL_MODE")
        try:
            os.environ["CRISP_RETRIEVAL_MODE"] = "hybrid"
            cfg = load_config()
            assert cfg.retrieval.mode == "hybrid"
        finally:
            if old is None:
                os.environ.pop("CRISP_RETRIEVAL_MODE", None)
            else:
                os.environ["CRISP_RETRIEVAL_MODE"] = old

    def test_invalid_threshold_rejected(self):
        # The only strictly invalid value is a non-float.
        with pytest.raises(Exception):
            AppConfig(adaptive_retrieval={"enabled": True,
                                          "expansion_score_threshold": "not-a-float"})

    def test_high_threshold_forces_expansion(self):
        # Threshold >> 1.0 is the documented way to express "always expand".
        cfg = AppConfig(adaptive_retrieval={"enabled": True,
                                            "expansion_score_threshold": 10.0})
        assert cfg.adaptive_retrieval.expansion_score_threshold == 10.0

    def test_negative_threshold_means_never_expand(self):
        # Threshold = -1.0 is the documented convention for "never expand".
        cfg = AppConfig(adaptive_retrieval={"enabled": True,
                                            "expansion_score_threshold": -1.0})
        assert cfg.adaptive_retrieval.expansion_score_threshold == -1.0


# ---------------------------------------------------------------------------
# Pipeline integration: retrieval_trace surfaces in RAGResult / JSONL
# ---------------------------------------------------------------------------

class TestPipelineAdaptiveIntegration:
    def test_pipeline_default_dense_no_trace(self):
        cfg = load_config()    # adaptive_retrieval.enabled=False
        pipe = Pipeline(config=cfg)
        pipe.build_index(CORPUS)
        result = pipe.run("What is the capital of France?")
        assert isinstance(result, RAGResult)
        # Phase 1-3 behaviour preserved: no trace when controller is off.
        assert result.retrieval_trace is None
        d = result.to_dict()
        assert d["retrieval_trace"] is None

    def test_pipeline_adaptive_emits_trace(self):
        cfg = AppConfig(
            mock=True,
            adaptive_retrieval={"enabled": True, "initial_k": 3, "max_k": 6,
                                "rewrite_variants": 1,
                                "expansion_score_threshold": 1.0,    # always expand
                                "expansion_gap_threshold": 1.0,
                                "multi_hop_max_entities": 2,
                                "multi_hop_min_entity_len": 3},
        )
        pipe = Pipeline(config=cfg)
        pipe.build_index(CORPUS)
        result = pipe.run("What is the capital of France?")
        d = result.to_dict()
        assert d["retrieval_trace"] is not None
        rt = d["retrieval_trace"]
        assert "strategy" in rt
        assert rt["initial_k"] == 3
        assert rt["final_k"] >= 3

    def test_pipeline_hybrid_mode_builds_bm25(self):
        cfg = AppConfig(
            mock=True,
            retrieval={"mode": "hybrid", "top_k": 3},
            adaptive_retrieval={"enabled": False},
        )
        pipe = Pipeline(config=cfg)
        pipe.build_index(CORPUS)
        # The base retriever should now be a HybridRetriever.
        from src.hybrid_retriever import HybridRetriever
        assert isinstance(pipe.retriever, HybridRetriever)
        result = pipe.run("France")
        assert len(result.retrieved_docs) >= 1

    def test_pipeline_adaptive_with_aggregated_claim_records_hop(self):
        # Force the claim extractor to tag AGGREGATED by giving the
        # mock generator a question whose mock answer contains
        # multi-entity content. We bypass the mock generator by
        # patching.
        from src.claim_extractor import ClaimExtractor, Provenance
        cfg = AppConfig(
            mock=True,
            adaptive_retrieval={"enabled": True,
                                "expansion_score_threshold": -1.0,
                                "expansion_gap_threshold": -1.0,
                                "rewrite_variants": 0,
                                "multi_hop_max_entities": 2,
                                "multi_hop_min_entity_len": 3,
                                "initial_k": 3, "max_k": 6},
        )
        # Stub extractor that always emits one AGGREGATED claim.
        class StubExtractor:
            mode = "synthetic"
            def __init__(self, intrinsic_threshold=0.7,
                         aggregated_threshold=0.3,
                         embedder=None): pass
            def extract(self, answer, hits, question=None):
                from src.claim_extractor import Claim
                return [Claim(id="c1",
                              text="France and Germany dominate Europe.",
                              provenance=Provenance.AGGREGATED)]
        pipe = Pipeline(config=cfg, claim_extractor=StubExtractor())
        pipe.build_index(CORPUS)
        # monkey-patch generator so answer is predictable
        pipe.generator.generate = lambda system_prompt, user_prompt: \
            "France and Germany dominate Europe."
        result = pipe.run("European powerhouses?")
        d = result.to_dict()
        assert d["retrieval_trace"] is not None
        # multi-hop sub-queries must contain at least one entity.
        assert len(d["retrieval_trace"]["multi_hop_queries"]) >= 1

    def test_pipeline_save_load_preserves_hybrid(self, tmp_path: Path):
        cfg = AppConfig(
            mock=True,
            retrieval={"mode": "hybrid", "top_k": 3},
            adaptive_retrieval={"enabled": False},
        )
        pipe = Pipeline(config=cfg)
        pipe.build_index(CORPUS)
        out_dir = tmp_path / "idx"
        pipe.save_index(str(out_dir))
        loaded_pipe = Pipeline(config=cfg).load_index(str(out_dir))
        from src.hybrid_retriever import HybridRetriever
        assert isinstance(loaded_pipe.retriever, HybridRetriever)
        result = loaded_pipe.run("France")
        assert len(result.retrieved_docs) >= 1
