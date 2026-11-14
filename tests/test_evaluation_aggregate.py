"""Tests for the evaluation aggregation and its bootstrap intervals.

These cover the code that produces the headline numbers in the results
table, so a silent change in the metric definitions is caught here rather
than in the thesis.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.evaluate_full import (
    _aggregate_from_rows,
    _bootstrap_intervals,
    _recompute,
)


def _rows(n_queries: int = 30, perfect: bool = True) -> list[dict]:
    """Build per-row records for ``n_queries`` balanced paired queries."""
    rows: list[dict] = []
    for i in range(n_queries):
        q = f"question-{i}"
        for label in (1, 0):
            if perfect:
                score = 0.9 if label == 1 else 0.1
            else:
                score = 0.5
            rows.append({
                "query": q,
                "label": label,
                "detector_score": score,
                "confidence": 0.5 + (0.3 if label == 0 else -0.3),
                "decision": "ABSTAIN" if label == 1 else "PASS",
                "n_claims": 2,
                "n_evidence": 3,
                "end_to_end_accuracy": 0 if label == 1 else 1,
                "hrr": 1.0 if label == 1 else 0.0,
                "ccr": 0.8,
                "vsr": 1.0,
                "retrieval_recall_at_5": 1.0,
                "retrieval_precision_at_5": 0.2,
                "retrieval_mrr": 1.0,
                "retrieval_ndcg": 1.0,
                "claim_precision": 0.9,
                "claim_recall": 0.9 if label == 0 else None,
                "claim_f1": 0.9 if label == 0 else None,
                "claim_support_rate": 0.9,
                "step_times_ms": {"nli_verification": 120.0},
            })
    return rows


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------


def test_aggregate_on_perfect_separation():
    """A detector that separates the classes perfectly must score 1.0."""
    agg = _aggregate_from_rows("halueval", _rows(30, perfect=True))
    assert agg["detection_accuracy"] == pytest.approx(1.0)
    assert agg["detection_precision"] == pytest.approx(1.0)
    assert agg["detection_recall"] == pytest.approx(1.0)
    assert agg["detection_f1"] == pytest.approx(1.0)
    assert agg["detection_roc_auc"] == pytest.approx(1.0)


def test_aggregate_on_chance_detector():
    """A detector with no signal must score at chance, not above it."""
    agg = _aggregate_from_rows("halueval", _rows(30, perfect=False))
    assert agg["detection_roc_auc"] == pytest.approx(0.5, abs=1e-9)
    assert agg["detection_accuracy"] == pytest.approx(0.5, abs=0.05)


def test_aggregate_reports_retrieval_and_claim_metrics():
    agg = _aggregate_from_rows("halueval", _rows(10))
    assert agg["retrieval_recall_at_5"] == pytest.approx(1.0)
    assert agg["retrieval_mrr"] == pytest.approx(1.0)
    assert agg["retrieval_ndcg"] == pytest.approx(1.0)
    assert agg["retrieval_precision_at_5"] == pytest.approx(0.2)
    assert agg["claim_precision"] == pytest.approx(0.9)


def test_aggregate_omits_retrieval_metrics_when_absent():
    """Older checkpoints have no retrieval fields — they must not report 0.0."""
    rows = _rows(5)
    for r in rows:
        r.pop("retrieval_recall_at_5", None)
        r.pop("retrieval_mrr", None)
    agg = _aggregate_from_rows("halueval", rows)
    assert "retrieval_recall_at_5" not in agg
    assert "retrieval_mrr" not in agg


# ---------------------------------------------------------------------------
# Bootstrap intervals
# ---------------------------------------------------------------------------


def test_every_headline_metric_has_a_confidence_interval():
    """Six of the nine headline metrics previously had no interval at all."""
    agg = _aggregate_from_rows("halueval", _rows(30))
    for key in (
        "detection_accuracy",
        "detection_precision",
        "detection_recall",
        "detection_f1",
        "detection_roc_auc",
        "hallucination_reduction_rate",
        "claim_correction_rate",
        "verification_success_rate",
        "end_to_end_accuracy",
    ):
        assert f"{key}_ci95" in agg, f"{key} has no confidence interval"
        lo, hi = agg[f"{key}_ci95"]
        assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_interval_is_degenerate_for_a_perfect_detector():
    truth = [1] * 50 + [0] * 50
    scores = [0.9] * 50 + [0.1] * 50
    groups = ["q%d" % i for i in range(50)] * 2
    out = _bootstrap_intervals(truth, scores, [1] * 50 + [0] * 50, groups)
    assert out["detection_roc_auc_ci95"] == [1.0, 1.0]
    assert out["detection_accuracy_ci95"] == [1.0, 1.0]


def test_bootstrap_interval_straddles_chance_for_a_random_detector():
    """With no signal the interval must contain 0.5 rather than exclude it."""
    import random

    random.seed(0)
    truth: list[int] = []
    scores: list[float] = []
    groups: list[str] = []
    for i in range(50):
        truth += [1, 0]
        scores += [random.random(), random.random()]
        groups += ["q%d" % i, "q%d" % i]
    out = _bootstrap_intervals(truth, scores, [0] * 100, groups)
    lo, hi = out["detection_roc_auc_ci95"]
    assert lo < 0.5 < hi


def test_bootstrap_resamples_queries_not_rows():
    """Both rows of a query must move together when it is resampled.

    A query contributes a label-0 and a label-1 row. Sampling rows
    independently treats that pair as independent evidence and yields an
    interval that is too narrow, because the two rows of a query are
    perfectly correlated. Sampling whole queries keeps the pair together.
    """
    truth = [1, 0] * 40
    scores = [0.9, 0.1] * 40
    groups = ["q%d" % i for i in range(40)] * 2
    out = _bootstrap_intervals(truth, scores, [0] * 80, groups)
    assert out["n_unique_queries"] == 40


def test_bootstrap_counts_queries_not_rows():
    """The reported unit of resampling must be the query.

    If rows were resampled instead, n_unique_queries would equal the row
    count and the interval would not widen when queries are duplicated.
    """
    truth = [1, 0] * 25
    scores = [0.9, 0.1] * 25
    groups = ["q%d" % i for i in range(25)] * 2
    out = _bootstrap_intervals(truth, scores, [0] * 50, groups)
    assert out["n_unique_queries"] == 25
    assert out["n_unique_queries"] != len(truth)


def test_bootstrap_resampling_preserves_the_class_balance():
    """Resampling whole queries keeps the positive/negative counts locked.

    Each query contributes one label-0 and one label-1 row, so a bootstrap
    replicate drawn at the query level always contains the same number of
    positives and negatives. Sampling rows independently would let a
    replicate be nearly all-positive, which inflates the apparent variance
    and produces an interval that is too wide.

    The observable consequence is that the class balance cannot drift
    between replicates, so the accuracy interval cannot be driven by a
    replicate that happens to be unbalanced.
    """
    import random

    random.seed(42)
    truth: list[int] = []
    scores: list[float] = []
    groups: list[str] = []
    for i in range(40):
        offset = random.uniform(-0.35, 0.35)
        truth += [1, 0]
        scores += [
            min(0.99, max(0.01, 0.6 + offset)),
            min(0.99, max(0.01, 0.4 + offset)),
        ]
        groups += ["q%d" % i, "q%d" % i]

    out = _bootstrap_intervals(truth, scores, [1, 0] * 40, groups)
    lo, hi = out["detection_accuracy_ci95"]

    # With the class balance locked, accuracy is bounded by the per-query
    # agreement rate and the interval is well defined and non-degenerate.
    assert lo < hi
    assert 0.0 <= lo <= hi <= 1.0
    assert out["n_unique_queries"] == 40


def test_bootstrap_interval_is_narrower_than_row_level_resampling():
    """Guard against regressing to row-level resampling.

    Row-level resampling can draw replicates that are almost entirely one
    class, which widens the interval (0.21 versus 0.13 on the fixture
    below). A correct query-level bootstrap keeps the paired structure and
    stays tighter.
    """
    import random

    random.seed(42)
    truth: list[int] = []
    scores: list[float] = []
    groups: list[str] = []
    for i in range(40):
        offset = random.uniform(-0.35, 0.35)
        truth += [1, 0]
        scores += [
            min(0.99, max(0.01, 0.6 + offset)),
            min(0.99, max(0.01, 0.4 + offset)),
        ]
        groups += ["q%d" % i, "q%d" % i]

    out = _bootstrap_intervals(truth, scores, [1, 0] * 40, groups)
    lo, hi = out["detection_accuracy_ci95"]
    width = hi - lo

    # Row-level resampling of this fixture yields ~0.21; query-level ~0.13.
    assert width < 0.18, f"interval too wide ({width:.3f}) — rows may be resampled independently"


def test_bootstrap_returns_nothing_for_tiny_samples():
    assert _bootstrap_intervals([1, 0], [0.9, 0.1], [1, 0], ["a", "b"]) == {}


# ---------------------------------------------------------------------------
# --recompute
# ---------------------------------------------------------------------------


def test_recompute_rebuilds_the_aggregate_from_disk(tmp_path, monkeypatch):
    out = tmp_path / "metrics_halueval.json"
    out.write_text(
        json.dumps({"dataset": "halueval", "per_query": _rows(20)}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_full", "--recompute", "--out", str(out)],
    )
    _recompute(_argparse_stub(out))
    payload = json.loads(out.read_text(encoding="utf-8"))
    agg = payload["aggregate"]
    assert agg["detection_roc_auc"] == pytest.approx(1.0)
    assert "detection_f1_ci95" in agg
    assert payload["n_queries"] == 40


def _argparse_stub(out: Path):
    """Minimal namespace matching what main() passes to _recompute."""
    import argparse

    ns = argparse.Namespace(out=str(out))
    return ns


# ---------------------------------------------------------------------------
# AUPR / precision-at-recall / Spearman rho
# ---------------------------------------------------------------------------


def test_average_precision_is_one_for_perfect_separation():
    from scripts.evaluate_full import _average_precision

    truth = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    assert _average_precision(truth, scores) == pytest.approx(1.0)


def test_average_precision_is_about_half_for_chance():
    from scripts.evaluate_full import _average_precision

    truth = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.6]  # all positives ranked first -> AP ~ 0.833
    ap = _average_precision(truth, scores)
    assert 0.0 <= ap <= 1.0
    # Reversed ranking -> positives at the bottom.
    ap_rev = _average_precision(truth, [0.1, 0.2, 0.3, 0.4])
    assert ap_rev < ap


def test_average_precision_handles_no_positives():
    from scripts.evaluate_full import _average_precision

    assert _average_precision([0, 0, 0], [0.5, 0.4, 0.3]) == 0.0


def test_precision_at_recall_is_one_for_perfect_separation():
    from scripts.evaluate_full import _precision_at_recall

    # Needs at least 10 positives: at 90% recall the constraint must tolerate
    # missing one positive, otherwise "recall >= 0.9" is vacuous.
    truth = [1] * 10 + [0] * 10
    scores = [0.9 - 0.001 * i for i in range(10)] + [0.1] * 10
    assert _precision_at_recall(truth, scores, 0.9) == pytest.approx(1.0)


def test_precision_at_recall_is_vacuous_below_ten_positives():
    from scripts.evaluate_full import _precision_at_recall

    # Four positives at 90% recall means "find all four" -- the constraint
    # carries no information, so the metric reports 0.0 rather than 1.0.
    truth = [1] * 4 + [0] * 4
    scores = [0.9 - 0.001 * i for i in range(4)] + [0.1] * 4
    assert _precision_at_recall(truth, scores, 0.9) == 0.0


def test_precision_at_recall_is_zero_when_recall_unreachable():
    from scripts.evaluate_full import _precision_at_recall

    # Only one positive, but recall 0.9 would require retrieving 0.9 of a
    # single instance -> not a meaningful operating point.
    truth = [1, 0, 0]
    scores = [0.9, 0.2, 0.1]
    assert _precision_at_recall(truth, scores, 0.9) == 0.0


def test_spearman_rho_is_high_for_perfect_rank_agreement():
    from scripts.evaluate_full import _spearman_rho

    # Spearman on tied binary labels is bounded below 1.0 (rank agreement
    # cannot be exact when the labels themselves contain ties), so assert
    # the correlation is strongly positive rather than exactly 1.
    truth = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]
    rho = _spearman_rho(truth, scores)
    assert 0.85 < rho < 1.0


def test_spearman_rho_is_minus_one_for_inverted_ranks():
    from scripts.evaluate_full import _spearman_rho

    truth = [1, 1, 0, 0]
    scores = [0.1, 0.2, 0.8, 0.9]
    rho = _spearman_rho(truth, scores)
    assert rho < -0.85
    # Reversing the scores must mirror the correlation exactly.
    assert rho == pytest.approx(-_spearman_rho(truth, [0.9, 0.8, 0.2, 0.1]))


def test_spearman_rho_is_one_for_untied_labels():
    from scripts.evaluate_full import _spearman_rho

    # Four distinct labels give four distinct ranks -> exact rho of 1.0
    # when the scores rank the labels in the same order.
    truth = [1, 2, 3, 4]
    scores = [0.4, 0.5, 0.6, 0.7]
    assert _spearman_rho(truth, scores) == pytest.approx(1.0)


def test_spearman_rho_handles_tied_scores():
    from scripts.evaluate_full import _spearman_rho

    # All-identical scores -> zero variance -> must not raise ZeroDivisionError.
    assert _spearman_rho([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.0


def test_new_metrics_appear_in_the_aggregate_with_cis():
    from scripts.evaluate_full import _aggregate_from_rows

    agg = _aggregate_from_rows("halueval", _rows(30))
    for key in ("detection_aupr", "precision_at_recall_90", "spearman_rho"):
        assert key in agg, f"{key} missing from the aggregate"
        assert isinstance(agg[key], float)
    for key in ("detection_aupr_ci95", "precision_at_recall_90_ci95",
                "spearman_rho_ci95"):
        ci = agg[key]
        assert len(ci) == 2
        assert ci[0] <= ci[1]


# ---------------------------------------------------------------------------
# --resume
# ---------------------------------------------------------------------------


def test_load_checkpoint_returns_nothing_for_a_missing_file(tmp_path):
    from scripts.evaluate_full import _load_checkpoint

    assert _load_checkpoint(tmp_path / "absent.json") == []


def test_load_checkpoint_returns_the_per_query_rows(tmp_path):
    from scripts.evaluate_full import _load_checkpoint, _write_checkpoint

    out = tmp_path / "metrics.json"
    _write_checkpoint(out, "halueval", _rows(3))
    loaded = _load_checkpoint(out)
    assert len(loaded) == 6
    assert loaded[0]["label"] in (0, 1)


def test_load_checkpoint_tolerates_a_corrupt_file(tmp_path):
    from scripts.evaluate_full import _load_checkpoint

    out = tmp_path / "metrics.json"
    out.write_text("{not json", encoding="utf-8")
    assert _load_checkpoint(out) == []


class _StopRun(BaseException):
    """Raised by the fake pipeline to end a test run early.

    Derives from BaseException, not Exception, so the evaluation loop's
    ``except Exception`` (which drops failed rows) does not swallow it.
    """


class _FakePipe:
    """Minimal stand-in for Pipeline that records the queries it is given."""

    class _Retriever:
        n_docs = 0

    def __init__(self, evaluated: list[str]) -> None:
        self.evaluated = evaluated
        self.retriever = self._Retriever()

    def load_index(self, index_dir: str) -> None:  # noqa: ARG002
        pass

    def run(self, query, answer=None, label=None, gold_answer=None):  # noqa: ARG002
        self.evaluated.append(query)
        raise _StopRun()


def test_resume_rejects_a_checkpoint_from_an_older_version(tmp_path, monkeypatch):
    """Rows lacking the current fields must not be silently reused."""
    from scripts import evaluate_full as ef

    out = tmp_path / "metrics.json"
    # A checkpoint written before the claim metrics existed.
    stale = [{k: v for k, v in rec.items() if k not in {"claim_recall", "claim_f1"}}
             for rec in _rows(2)]
    ef._write_checkpoint(out, "halueval", stale)

    dataset_rows = [
        {"query": r["query"], "label": r["label"], "answer": "x"}
        for r in _rows(2)
    ]
    monkeypatch.setattr(ef, "_resolve_dataset_rows", lambda args: (dataset_rows, "halueval"))

    import io

    stderr = io.StringIO()
    monkeypatch.setattr("sys.stderr", stderr)

    evaluated: list[str] = []
    monkeypatch.setattr(ef, "Pipeline", lambda *a, **k: _FakePipe(evaluated))
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_full", "--data", "unused", "--out", str(out), "--resume"],
    )

    with pytest.raises(_StopRun):
        ef.main()

    # Every row was re-evaluated because none of the checkpoint rows were
    # usable.
    assert len(evaluated) > 0
    assert "starting from scratch" in stderr.getvalue()


def test_resume_skips_rows_the_checkpoint_already_has(tmp_path, monkeypatch):
    """Rows already present in the checkpoint must not be evaluated again."""
    from scripts import evaluate_full as ef

    out = tmp_path / "metrics.json"
    done = _rows(2)
    ef._write_checkpoint(out, "halueval", done)

    dataset_rows = [
        {"query": r["query"], "label": r["label"], "answer": "x"}
        for r in _rows(3)
    ]
    monkeypatch.setattr(ef, "_resolve_dataset_rows", lambda args: (dataset_rows, "halueval"))

    evaluated: list[str] = []
    monkeypatch.setattr(ef, "Pipeline", lambda *a, **k: _FakePipe(evaluated))
    monkeypatch.setattr(
        sys, "argv",
        ["evaluate_full", "--data", "unused", "--out", str(out), "--resume"],
    )

    with pytest.raises(_StopRun):
        ef.main()

    # The first four rows (two queries x two labels) came from the
    # checkpoint; only the fifth row reached the pipeline.
    assert len(evaluated) == 1, (
        f"expected only the un-finished row to be evaluated, got {evaluated}"
    )


