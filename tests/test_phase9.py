"""Phase 9 — Step 7 (reranker) + Step 13 (evaluation) tests."""
import os
os.environ.setdefault("CRISP_MOCK", "1")

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.evaluation_metrics import (
    EvalBundle,
    bert_score_f1,
    bleu_score,
    ccr,
    confusion_matrix_3way,
    confusion_matrix_binary,
    evaluate_predictions,
    exact_match,
    hrr,
    load_predictions,
    mrr,
    ndcg_at_k,
    per_class_prf,
    precision_at_k,
    precision_recall_f1,
    recall_at_k,
    render_markdown_summary,
    rouge_l,
    vsr,
    write_metrics,
)
from src.pipeline import Pipeline
from src.reranker import CrossEncoderReranker, RerankingRetriever
from src.retriever import Hit


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestCrossEncoderReranker:
    def test_mock_reorders_by_token_overlap(self):
        r = CrossEncoderReranker(top_k=2, mock=True)
        hits = [
            Hit(text="George Orwell wrote the dystopian novel 1984.", score=0.5, index=0),
            Hit(text="The Eiffel Tower is in Paris.", score=0.5, index=1),
            Hit(text="George Orwell is a British author.", score=0.5, index=2),
        ]
        out = r.rerank("Who wrote 1984?", hits, top_k=2)
        # The strongest evidence ("wrote 1984") is promoted to rank 1, and the
        # irrelevant Eiffel Tower hit is correctly demoted out of top-2.
        assert len(out) == 2
        assert "1984" in out[0].text
        assert "Eiffel" not in out[0].text
        assert r.last_trace is not None
        assert r.last_trace.mock is True
        assert r.last_trace.input_top_k == 3
        assert r.last_trace.output_top_k == 2

    def test_empty_input(self):
        r = CrossEncoderReranker(top_k=5, mock=True)
        assert r.rerank("anything", []) == []
        assert r.last_trace.input_top_k == 0

    def test_top_k_caps_results(self):
        r = CrossEncoderReranker(top_k=3, mock=True)
        hits = [Hit(text=f"Orwell novel {i}", score=0.1, index=i) for i in range(5)]
        out = r.rerank("Orwell", hits, top_k=3)
        assert len(out) == 3


class TestRerankingRetriever:
    def test_wraps_base_retriever_and_reranks(self):
        # Use a tiny Pipeline + dense retriever to avoid mocking FAISS paths.
        cfg = load_config()
        pipe = Pipeline(cfg)
        pipe.build_index([
            "George Orwell wrote the dystopian novel 1984.",
            "The Eiffel Tower is in Paris.",
            "Orwell was a British author.",
        ])
        wrapped = RerankingRetriever(pipe.retriever, CrossEncoderReranker(top_k=2, mock=True))
        out = wrapped.retrieve("Who wrote 1984?")
        assert len(out) == 2
        assert all("Orwell" in h.text for h in out)
        assert wrapped.last_rerank_trace is not None


# ---------------------------------------------------------------------------
# Evaluation metrics — unit tests
# ---------------------------------------------------------------------------

class TestDetectionMetrics:
    def test_confusion_matrix_perfect(self):
        cm = confusion_matrix_binary([1, 0, 1, 0], [1, 0, 1, 0])
        assert cm == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}

    def test_precision_recall_f1(self):
        cm = {"tp": 2, "fp": 1, "fn": 0, "tn": 1}
        prf = precision_recall_f1(cm)
        assert prf["precision"] == pytest.approx(2 / 3, rel=1e-3)
        assert prf["recall"] == pytest.approx(1.0, rel=1e-3)
        assert prf["f1"] == pytest.approx(0.8, rel=1e-3)


class TestRetrievalMetrics:
    def test_recall_precision_mrr_ndcg(self):
        retrieved = [[0, 1, 2], [3, 4, 5]]
        gold = [[1], [4]]
        relevance = [[0, 1, 0], [0, 1, 0]]
        assert recall_at_k(retrieved, gold, k=3) == 1.0
        assert precision_at_k(retrieved, gold, k=3) == pytest.approx(2 / 6, rel=1e-3)
        # Gold doc is at index 1 of [0,1,2] (rank 2) and index 1 of [3,4,5] (rank 2)
        # so MRR = mean(1/2, 1/2) = 0.5
        assert mrr(retrieved, gold) == pytest.approx(0.5, rel=1e-3)
        assert ndcg_at_k(retrieved, relevance, k=3) > 0.0


class TestRegenerationMetrics:
    def test_hrr_full_reduction(self):
        assert hrr(100, 0) == 100.0
    def test_hrr_zero(self):
        assert hrr(50, 50) == 0.0
    def test_hrr_no_hallucinations(self):
        assert hrr(0, 0) == 0.0

    def test_ccr_partial(self):
        assert ccr(50, 42) == 84.0

    def test_vsr(self):
        assert vsr(20, 15) == 75.0


class TestGenerationMetrics:
    def test_exact_match(self):
        assert exact_match("Paris", "paris.") == 1.0
        assert exact_match("Paris", "London") == 0.0

    def test_bleu_perfect(self):
        assert bleu_score("the cat sat", "the cat sat") > 0.9

    def test_rouge_l_partial(self):
        score = rouge_l("the cat sat on the mat", "the cat is on the mat")
        assert 0.0 < score < 1.0

    def test_bert_score_proxy_runs(self):
        score = bert_score_f1(["the cat sat"], ["the cat sat"])
        assert score >= 0.0  # proxy may yield 1.0 in mock


class TestVerificationMatrix:
    def test_3way_confusion_and_prf(self):
        y_true = ["Supported", "Contradicted", "Unknown", "Supported"]
        y_pred = ["Supported", "Unknown", "Unknown", "Supported"]
        cm = confusion_matrix_3way(y_true, y_pred)
        assert cm["Supported"]["Supported"] == 2
        assert cm["Contradicted"]["Unknown"] == 1
        prf = per_class_prf(cm)
        assert prf["Supported"]["precision"] == 1.0
        assert "macro" in prf


# ---------------------------------------------------------------------------
# End-to-end evaluate_predictions + I/O
# ---------------------------------------------------------------------------

def _make_prediction(claim_text="Orwell wrote 1984",
                     reference="Orwell wrote 1984",
                     hallucinated=False,
                     nli_label="SUP",
                     retrieved=None,
                     gold_docs=None,
                     iteration_history=None):
    return {
        "id": "q1",
        "question": "Who wrote 1984?",
        "query": "Who wrote 1984?",
        "answer": claim_text,
        "final_answer": claim_text,
        "reference_answer": reference,
        "retrieved_docs": retrieved or [
            {"text": "Orwell wrote 1984.", "score": 0.9, "index": 0},
        ],
        "claim_verdicts": [
            {"claim_id": "c1", "claim_text": claim_text,
             "eedc_score": 0.1 if hallucinated else 0.9,
             "hallucinated": hallucinated,
             "nli_label": nli_label,
             "provenance": "extrinsic"}
        ],
        "iteration_history": iteration_history or [],
        "timings_ms": {"total": 12.0},
        "confidence": 0.9,
        "gold_doc_indices": gold_docs or [],
    }


class TestEvaluatePredictions:
    def test_full_bundle_keys(self):
        preds = [_make_prediction(), _make_prediction(hallucinated=True, nli_label="CON",
                                                     reference="Orwell wrote 1984")]
        bundle = evaluate_predictions(preds, k_for_retrieval=1)
        assert isinstance(bundle, EvalBundle)
        d = bundle.to_dict()
        for key in ("detection", "verification", "retrieval",
                    "regeneration", "generation", "overall"):
            assert key in d

    def test_overall_end_to_end_accuracy(self):
        preds = [_make_prediction(reference="Orwell wrote 1984"),
                 _make_prediction(reference="London is the capital.")]
        bundle = evaluate_predictions(preds, k_for_retrieval=1)
        # First one matches, second does not.
        assert bundle.overall["end_to_end_accuracy"] == 0.5

    def test_metrics_have_required_fields(self):
        preds = [_make_prediction()]
        bundle = evaluate_predictions(preds, k_for_retrieval=1)
        d = bundle.to_dict()
        assert "f1" in d["detection"]
        assert "roc_auc" in d["detection"]
        assert "mrr" in d["retrieval"]
        for k in ("hrr", "ccr", "vsr"):
            assert k in d["regeneration"]
        for k in ("exact_match", "bleu", "rouge_l", "bertscore_f1"):
            assert k in d["generation"]
        assert "avg_latency_ms" in d["overall"]


class TestIOHelpers:
    def test_load_write_roundtrip(self, tmp_path: Path):
        p = tmp_path / "preds.jsonl"
        out = tmp_path / "metrics.json"
        p.write_text(json.dumps(_make_prediction()) + "\n")
        loaded = load_predictions(p)
        assert len(loaded) == 1
        bundle = evaluate_predictions(loaded, k_for_retrieval=1)
        write_metrics(bundle.to_dict(), out)
        assert out.exists()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert "detection" in payload

    def test_markdown_renderer(self):
        preds = [_make_prediction()]
        bundle = evaluate_predictions(preds, k_for_retrieval=1)
        md = render_markdown_summary(bundle.to_dict())
        assert "# CRISP Evaluation Report" in md
        assert "detection" in md


# ---------------------------------------------------------------------------
# Pipeline integration — reranker is wired but inactive by default
# ---------------------------------------------------------------------------

class TestPipelineRerankerToggle:
    def test_reranker_off_by_default(self):
        cfg = load_config()
        assert cfg.reranker.enabled is False
        pipe = Pipeline(cfg)
        assert pipe.reranker is None
        pipe.build_index(["Orwell wrote 1984.", "Paris is the capital of France."])
        res = pipe.run("Who wrote 1984?")
        assert res.rerank_trace is None

    def test_reranker_on_via_injection(self):
        cfg = load_config()
        pipe = Pipeline(cfg, reranker=CrossEncoderReranker(top_k=2, mock=True))
        pipe.build_index([
            "Orwell wrote 1984.",
            "Paris is the capital of France.",
            "Orwell was British.",
        ])
        res = pipe.run("Who wrote 1984?")
        assert res.rerank_trace is not None
        assert res.rerank_trace["mock"] is True
        assert res.rerank_trace["output_top_k"] == 2