"""Phase 8 — End-to-End Verification & Reporting tests.

Covers:
* ``ReliabilityLabel`` thresholds (well-supported vs unreliable).
* ``ReportBuilder.build`` consumes a RAGResult dict and yields a
  serialisable ``ReliabilityReport``.
* Renderers (``to_json``, ``to_markdown``, ``to_html``) produce valid,
  non-empty output.
* ``summarise_batch`` aggregates reliability labels correctly.
* ``write_report`` dispatches on file extension.
"""
from __future__ import annotations

import json
import os

os.environ["CRISP_MOCK"] = "1"
os.environ.setdefault("CRISP_INDEX_PATH", "data/test_index_phase8")

from pathlib import Path

import pytest

from src.reporting import (
    BatchSummary,
    ReliabilityLabel,
    ReliabilityReport,
    ReportBuilder,
    summarise_batch,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers — minimal RAGResult dict for testing
# ---------------------------------------------------------------------------


def _result(
    *,
    query: str = "What is the capital of France?",
    answer: str = "Paris.",
    confidence: float = 0.9,
    hallucination_rate: float = 0.0,
    num_claims: int = 1,
    num_flagged: int = 0,
    iterations: int = 1,
):
    """Build a minimal RAGResult for testing."""
    from src.claim_extractor import Claim, Provenance
    from src.detector import NLIPrediction
    from src.pipeline import ClaimVerdict, RAGResult

    verdicts = []
    for i in range(num_claims):
        claim = Claim(id=f"c{i}", text=f"Claim {i}",
                      provenance=Provenance.EXTRINSIC)
        nli = NLIPrediction(claim="x", evidence="y",
                            label="entail", probs=[0.9, 0.05, 0.05])
        verdicts.append(ClaimVerdict(
            claim=claim,
            evidence_text="Paris is the capital of France.",
            evidence_score=0.8,
            nli=nli,
            eedc_score=0.85,
            hallucinated=(i < num_flagged),
        ))
    return RAGResult(
        query=query,
        answer=answer,
        confidence=confidence,
        iterations=iterations,
        hallucination_rate=hallucination_rate,
        claim_verdicts=verdicts,
        timings_ms={},
    )


# ---------------------------------------------------------------------------
# ReliabilityLabel boundaries
# ---------------------------------------------------------------------------


@pytest.fixture
def builder() -> ReportBuilder:
    return ReportBuilder()


class TestReliabilityLabel:
    def test_well_supported_when_clean(self, builder):
        report = builder.build(_result(
            hallucination_rate=0.0, confidence=0.95, num_claims=3,
        ))
        assert report.label == ReliabilityLabel.RELIABLE

    def test_unreliable_when_most_flagged(self, builder):
        report = builder.build(_result(
            hallucination_rate=0.8, confidence=0.2, num_claims=5, num_flagged=4,
        ))
        assert report.label == ReliabilityLabel.UNRELIABLE

    def test_label_serialises_to_str(self, builder):
        report = builder.build(_result())
        assert isinstance(report.label.value, str)


# ---------------------------------------------------------------------------
# ReportBuilder.build
# ---------------------------------------------------------------------------


class TestReportBuilderBuild:
    def test_returns_reliability_report(self, builder):
        report = builder.build(_result())
        assert isinstance(report, ReliabilityReport)
        assert report.query == "What is the capital of France?"
        assert report.answer == "Paris."
        assert report.num_claims == 1
        assert report.num_flagged == 0
        assert report.mean_eedc == pytest.approx(0.85, abs=1e-6)
        assert 0.0 <= report.hallucination_rate <= 1.0
        assert isinstance(report.iterations, int)
        assert report.label == ReliabilityLabel.RELIABLE

    def test_handles_empty_verdicts(self, builder):
        report = builder.build(_result(num_claims=0))
        assert report.num_claims == 0
        assert report.mean_eedc == 1.0  # default when no claims


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class TestRenderers:
    def test_to_json_roundtrip(self, builder):
        report = builder.build(_result())
        payload = json.loads(report.to_json())
        assert payload["query"] == "What is the capital of France?"
        assert "label" in payload
        assert "mean_eedc" in payload

    def test_to_markdown_non_empty(self, builder):
        report = builder.build(_result())
        md = report.to_markdown()
        assert "Reliability Report" in md
        assert "What is the capital of France?" in md

    def test_to_html_contains_verdict_badge(self, builder):
        report = builder.build(_result())
        html = report.to_html()
        assert "<html" in html.lower()
        assert report.label.value in html


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------


class TestBatchSummary:
    def test_summarise_batch_counts_labels(self, builder):
        results = [
            _result(hallucination_rate=0.0, confidence=0.95),
            _result(hallucination_rate=0.6, confidence=0.4, num_flagged=3),
        ]
        reports = [builder.build(r) for r in results]
        summary = summarise_batch(reports)
        assert isinstance(summary, BatchSummary)
        assert summary.n == 2
        assert sum(summary.label_counts.values()) == 2
        # Mean hallucination rate is averaged across the batch.
        assert 0.0 <= summary.mean_hallucination_rate <= 1.0


# ---------------------------------------------------------------------------
# File dispatcher
# ---------------------------------------------------------------------------


class TestWriteReport:
    def test_write_json(self, builder, tmp_path: Path):
        report = builder.build(_result())
        out = tmp_path / "report.json"
        write_report(report, str(out))
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "label" in data

    def test_write_markdown(self, builder, tmp_path: Path):
        report = builder.build(_result())
        out = tmp_path / "report.md"
        write_report(report, str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "Reliability Report" in text

    def test_write_html(self, builder, tmp_path: Path):
        report = builder.build(_result())
        out = tmp_path / "report.html"
        write_report(report, str(out))
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "<html" in text.lower()

    def test_unknown_extension_falls_back_to_json(self, builder, tmp_path: Path):
        report = builder.build(_result())
        out = tmp_path / "report.xyz"
        # write_report dispatches unknown extensions to JSON by default
        # (defensive behaviour so the CLI doesn't crash on a typo).
        write_report(report, str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# CLI script smoke test
# ---------------------------------------------------------------------------


class TestReportCli:
    def test_cli_runs_end_to_end(self, tmp_path: Path):
        """Reproduce the scripts/report.py flow against a tiny
        preds.jsonl file via ``subprocess.run``."""
        import subprocess
        import sys as _sys

        # Build a preds.jsonl file in mock mode.
        from src.pipeline import Pipeline
        pipeline = Pipeline()
        pipeline.build_index(["France's capital is Paris."])
        preds_path = tmp_path / "preds.jsonl"
        preds_path.write_text(
            json.dumps(pipeline.run("Capital of France?").to_dict()) + "\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "reports"
        repo_root = Path(__file__).resolve().parent.parent
        result = subprocess.run(
            [_sys.executable, str(repo_root / "scripts" / "report.py"),
             "--in", str(preds_path),
             "--out-dir", str(out_dir),
             "--format", "json"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert (out_dir / "summary.json").exists()
