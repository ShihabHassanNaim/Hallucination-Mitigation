"""Flask front-end smoke test.

The production code is always real-model — there is no mock-mode flag.
Tests use the ``fake_pipeline`` fixture to inject a Pipeline with
lightweight in-memory doubles for every model slot, so the test suite
stays fast and offline. See ``tests/_fakes.py``.
"""
from __future__ import annotations

import importlib
import json

import pytest

from tests._fakes import make_fake_pipeline


def _client(monkeypatch, fake_pipeline):
    """Build a Flask app whose Pipeline is pre-injected with fakes."""
    frontend = importlib.import_module("frontend.app")
    monkeypatch.setattr(frontend, "Pipeline", lambda cfg=None: fake_pipeline)
    app = frontend.create_app()
    app.config["TESTING"] = True
    return app.test_client(), app


def test_flask_index_page(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/")
    assert r.status_code == 200
    assert b"Hallucination Pipeline" in r.data
    assert b"Build index" in r.data
    assert b"Run pipeline" in r.data


def test_flask_health(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.get_json()
    assert body["status"] == "ok"
    assert "index_loaded" in body


def test_flask_index_and_query(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.post("/api/index", json={"documents": ["Paris is the capital of France.", "France is in Europe."]})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["status"] == "indexed"
    assert body["n_documents"] >= 1

    r = client.post("/api/query", json={"query": "What is the capital of France?"})
    assert r.status_code == 200, r.get_json()
    res = r.get_json()
    assert res["query"] == "What is the capital of France?"
    assert res["decision"] in {"PASS", "REGEN", "ABSTAIN"}
    assert "claims" in res
    assert "evidence" in res
    assert len(res["step_traces"]) == 13


def test_flask_rejects_empty_query(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.post("/api/query", json={"query": "  "})
    assert r.status_code == 422
    assert "error" in r.get_json()


def test_flask_rejects_empty_docs(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.post("/api/index", json={"documents": []})
    assert r.status_code == 422


def test_flask_metrics(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.get_json()
    assert "rows" in body


def test_flask_structure_page_and_api(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    # Page
    r = client.get("/structure")
    assert r.status_code == 200
    assert b"Project Structure" in r.data
    assert b"File tree" in r.data
    # API
    r = client.get("/api/structure?depth=3")
    assert r.status_code == 200
    body = r.get_json()
    assert body["root"]
    assert body["total_files"] >= 1
    assert body["tree"]["type"] == "dir"
    # Top-level children should be the major folders.
    names = {c["name"] for c in body["tree"]["children"]}
    assert "src" in names
    assert "tests" in names
    assert "frontend" in names
    # Skipped dirs.
    assert ".venv" not in names
    assert "__pycache__" not in names
    # Depth param respected: at max_depth=3, no subdirectory appears at
    # depth 4 or deeper (files may be at depth 4 since the cutoff only
    # stops dir recursion).
    def _walk(n, d):
        if n["type"] == "dir":
            assert d <= 3, f"dir nested too deeply at depth {d}: {n['name']}"
            for c in n.get("children", []):
                _walk(c, d + 1)
        else:
            assert d <= 4, f"file nested too deeply at depth {d}: {n['name']}"

    for c in body["tree"]["children"]:
        _walk(c, 1)


def test_flask_structure_dotfiles_off(fake_pipeline, monkeypatch):
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/structure?depth=2")
    body = r.get_json()
    names = {c["name"] for c in body["tree"]["children"]}
    assert ".git" not in names
    r = client.get("/api/structure?depth=2&include_hidden=1")
    body = r.get_json()
    names = {c["name"] for c in body["tree"]["children"]}
    assert ".git" not in names  # .git still always skipped


# ---------------------------------------------------------------------------
# /api/evaluation
# ---------------------------------------------------------------------------


def _write_metrics(tmp_path, payload):
    """Helper: write a metrics.json under tmp_path/data/metrics.json."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_api_evaluation_returns_report_shape(tmp_path, monkeypatch, fake_pipeline):
    """GET /api/evaluation returns 200 with aggregate + per_query + n_queries,
    and aggregates numeric columns on the fly."""
    _write_metrics(
        tmp_path,
        {
            "rows": [
                {"confidence": 0.8, "decision": "PASS", "end_to_end_accuracy": 1.0,
                 "retrieval_recall_at_5": 1.0, "retrieval_precision_at_5": 0.6,
                 "retrieval_mrr": 1.0, "retrieval_ndcg": 1.0,
                 "claim_precision": 1.0, "claim_recall": 1.0, "claim_f1": 1.0,
                 "verification_success_rate": 1.0},
                {"confidence": 0.6, "decision": "ABSTAIN", "end_to_end_accuracy": 0.0,
                 "retrieval_recall_at_5": 0.5, "retrieval_precision_at_5": 0.4,
                 "retrieval_mrr": 0.5, "retrieval_ndcg": 0.5,
                 "claim_precision": 0.0, "claim_recall": 0.0, "claim_f1": 0.0,
                 "verification_success_rate": 1.0},
            ],
            "aggregate": {},
        },
    )
    monkeypatch.chdir(tmp_path)
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    body = r.get_json()
    assert "aggregate" in body
    assert "per_query" in body
    assert body["n_queries"] == 2
    assert body["aggregate"]["mean_confidence"] == pytest.approx(0.7)
    assert body["aggregate"]["stdev_confidence"] is not None  # n>=2
    assert body["aggregate"]["mean_end_to_end_accuracy"] == pytest.approx(0.5)
    assert body["source"] == "unknown"  # no precomputed aggregate


def test_api_evaluation_missing_file(tmp_path, monkeypatch, fake_pipeline):
    """Missing metrics.json → 200 with empty rows and source='missing'."""
    monkeypatch.chdir(tmp_path)
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    body = r.get_json()
    assert body["n_queries"] == 0
    assert body["per_query"] == []
    assert body["source"] == "missing"
    assert body["last_modified"] is None


def test_api_evaluation_preserves_precomputed(tmp_path, monkeypatch, fake_pipeline):
    """Pre-existing aggregate block from evaluate_full.py is preserved and
    detected in `source`."""
    _write_metrics(
        tmp_path,
        {
            "rows": [],
            "aggregate": {
                "detection_roc_auc": 0.91,
                "hallucination_reduction_rate": 0.5,
            },
        },
    )
    monkeypatch.chdir(tmp_path)
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/evaluation")
    assert r.status_code == 200
    body = r.get_json()
    assert body["aggregate"]["detection_roc_auc"] == 0.91
    assert body["aggregate"]["hallucination_reduction_rate"] == 0.5
    assert body["source"] == "evaluate_full"


# ---------------------------------------------------------------------------
# /api/accuracy
# ---------------------------------------------------------------------------


def test_api_accuracy_returns_headline_and_breakdown(tmp_path, monkeypatch, fake_pipeline):
    """GET /api/accuracy returns headline numbers, decision breakdown, and
    a per-query accuracy projection from data/metrics.json."""
    _write_metrics(
        tmp_path,
        {
            "rows": [
                {"query": "Capital of France?", "decision": "PASS",
                 "end_to_end_accuracy": 1.0, "claim_f1": 0.9, "confidence": 0.8},
                {"query": "Capital of Mars?", "decision": "ABSTAIN",
                 "end_to_end_accuracy": 0.0, "claim_f1": 0.0, "confidence": 0.4},
                {"query": "Author of 1984?", "decision": "PASS",
                 "end_to_end_accuracy": 1.0, "claim_f1": 0.8, "confidence": 0.75},
            ],
            "aggregate": {
                "detection_accuracy": 0.86,
                "end_to_end_accuracy": 0.66,
            },
        },
    )
    monkeypatch.chdir(tmp_path)
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/accuracy")
    assert r.status_code == 200
    body = r.get_json()
    # Headline surfaces top-level + mean_* keys.
    assert body["headline"]["detection_accuracy"] == pytest.approx(0.86)
    assert body["headline"]["end_to_end_accuracy"] == pytest.approx(0.66)
    assert body["headline"]["mean_claim_f1"] == pytest.approx((0.9 + 0.0 + 0.8) / 3)
    # Breakdown counts PASS/REGEN/ABSTAIN + correct/incorrect.
    bd = body["breakdown"]
    assert bd["decisions"]["PASS"] == 2
    assert bd["decisions"]["ABSTAIN"] == 1
    assert bd["decisions"]["REGEN"] == 0
    assert bd["total"] == 3
    assert bd["correct"] == 2
    assert bd["incorrect"] == 1
    # Per-query accuracy projection.
    pq = body["per_query"]
    assert len(pq) == 3
    assert pq[0]["correct"] is True
    assert pq[1]["correct"] is False
    assert pq[0]["query"] == "Capital of France?"
    # Meta block.
    meta = body["meta"]
    assert meta["n_queries"] == 3
    assert meta["source"] == "evaluate_full"


def test_api_accuracy_missing_file(tmp_path, monkeypatch, fake_pipeline):
    """No metrics.json → 200 with empty fields so the UI can render an
    empty state cleanly."""
    monkeypatch.chdir(tmp_path)
    client, _ = _client(monkeypatch, fake_pipeline)
    r = client.get("/api/accuracy")
    assert r.status_code == 200
    body = r.get_json()
    assert body["headline"] == {}
    assert body["breakdown"]["total"] == 0
    assert body["per_query"] == []
    assert body["meta"]["n_queries"] == 0
    assert body["meta"]["source"] == "missing"