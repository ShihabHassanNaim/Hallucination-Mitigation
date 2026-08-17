"""Step base class and registry.

Every pipeline step inherits Step and overrides .run(ctx). The orchestrator
in src/pipeline.py drives all 13 steps in order.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..types import StepTrace


@dataclass
class StepContext:
    """Mutable bag passed between steps within one pipeline.run() call."""

    query: str
    cfg: Any
    answer: str = ""
    claims: list[Any] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)
    feature_vector: dict[str, float] = field(default_factory=dict)
    verdicts: list[Any] = field(default_factory=list)
    eedc_score: float = 0.0
    confidence: float = 0.0
    decision: str = "PASS"
    iterations: int = 1
    decision_log: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Step(ABC):
    """Abstract base class for a single pipeline step."""

    number: int = 0
    name: str = "step"

    def __init__(self, ctx: StepContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def run(self) -> None:
        """Mutate self.ctx in place."""

    def trace(self, payload: dict[str, Any], duration_ms: float) -> StepTrace:
        return StepTrace(
            step=self.number,
            name=self.name,
            started_at=time.time(),
            duration_ms=duration_ms,
            payload=payload,
        )


def step_registry() -> list[type[Step]]:
    """Return the 13 step classes in order. Imported lazily to avoid cycles."""
    from . import (
        step01_query_understanding,
        step02_grounded_generation,
        step03_early_detection,
        step04_confidence_assessment,
        step05_claim_extraction,
        step06_hybrid_retrieval,
        step07_reranking,
        step08_nli_verification,
        step09_confidence_fusion,
        step10_regeneration,
        step11_final_gate,
        step12_explainable_response,
        step13_evaluation,
    )

    return [
        step01_query_understanding.Step01QueryUnderstanding,
        step02_grounded_generation.Step02GroundedGeneration,
        step03_early_detection.Step03EarlyDetection,
        step04_confidence_assessment.Step04ConfidenceAssessment,
        step05_claim_extraction.Step05ClaimExtraction,
        step06_hybrid_retrieval.Step06HybridRetrieval,
        step07_reranking.Step07Reranking,
        step08_nli_verification.Step08NLIVerification,
        step09_confidence_fusion.Step09ConfidenceFusion,
        step10_regeneration.Step10Regeneration,
        step11_final_gate.Step11FinalGate,
        step12_explainable_response.Step12ExplainableResponse,
        step13_evaluation.Step13Evaluation,
    ]
