"""Shared types used across the 13 steps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Evidence:
    """A piece of evidence (retrieved doc / passage) with provenance."""

    doc_id: str
    text: str
    score: float = 0.0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "text": self.text, "score": self.score, "source": self.source}


@dataclass
class Claim:
    """An atomic factual claim extracted from the answer."""

    text: str
    provenance: str = "extrinsic"  # intrinsic | extrinsic | aggregated
    intrinsic_overlap: float = 0.0
    verdict: str = "unknown"       # supported | contradicted | unknown
    confidence: float = 0.0
    evidence: list[Evidence] = field(default_factory=list)
    citation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "provenance": self.provenance,
            "intrinsic_overlap": self.intrinsic_overlap,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "citation": self.citation,
        }


@dataclass
class ClaimVerdict:
    """NLI verdict for a single claim against a single evidence."""

    claim: str
    evidence_doc_id: str
    label: str        # supported | contradicted | unknown
    score: float      # model's entailment probability


@dataclass
class StepTrace:
    """Captures the input/output of one step for explainability."""

    step: int
    name: str
    started_at: float
    duration_ms: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 3),
            "payload": self.payload,
        }


@dataclass
class PipelineResult:
    """Final result returned to the API."""

    query: str
    answer: str
    claims: list[Claim]
    evidence: list[Evidence]
    confidence: float
    decision: str = "PASS"   # PASS | REGEN | ABSTAIN
    iterations: int = 1
    step_traces: list[StepTrace] = field(default_factory=list)
    report_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence": round(self.confidence, 4),
            "decision": self.decision,
            "iterations": self.iterations,
            "step_traces": [t.to_dict() for t in self.step_traces],
            "report_markdown": self.report_markdown,
        }
