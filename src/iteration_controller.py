"""Phase 7 — Adaptive Iteration Controller (AIC).

What ``AdaptiveIterationController`` does
-----------------------------------------
Phases 1–6 run once: query → retrieve → generate → verify → score. If a
claim looks hallucinated, we stop and report. Phase 7 closes the loop by
asking one more question:

    *Given what we know after this answer, should we keep iterating?*

The AIC inspects the verdict list from Phase 2 and decides:

  * ``ACCEPT``  – confidence is high and no claims are flagged. Stop.
  * ``EDIT``    – one or more claims are flagged and we can patch the
                  spans with :class:`EvidenceGuidedEditor`. No new
                  retrieval; bounded by ``max_edits_per_iteration``.
  * ``REGEN``   – flagged fraction is high enough that patching isn't
                  enough; regenerate the whole answer using the most
                  trusted evidence as additional context.
  * ``STOP``    – we've hit ``max_iterations`` or hallucination rate is
                  stuck; surface a warning instead of looping forever.

Each decision is recorded in an :class:`IterationRecord` so the trace is
reproducible offline.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

# NOTE: we deliberately avoid importing ``pipeline.ClaimVerdict`` here to
# break a circular import. The AIC only needs the public attributes
# ``.claim``, ``.hallucinated``, and ``.evidence_score``, so we treat the
# verdict objects as duck-typed.

logger = logging.getLogger(__name__)


class Action(str, enum.Enum):
    ACCEPT = "accept"
    EDIT = "edit"
    REGEN = "regen"
    STOP = "stop"

    def __str__(self) -> str:                    # pragma: no cover
        return self.value


@dataclass
class IterationConfig:
    """Knobs for the AIC policy."""
    max_iterations: int = 3
    # Stop if hallucination_rate falls below this threshold.
    accept_rate_threshold: float = 0.05
    # Patching range: max_edits <= low -> ACCEPT after edit; > high -> REGEN.
    max_edits_per_iteration: int = 2
    # Below this hallucinated fraction, accept (don't keep iterating).
    accept_rate: float = 0.10
    # Above this, regen whole answer (don't bother editing).
    regen_rate: float = 0.40
    # If hallucination rate is no longer dropping between iterations, STOP.
    min_improvement: float = 0.02


@dataclass
class IterationRecord:
    """One iteration of the AIC loop."""
    iteration: int
    action: str
    hallucination_rate: float
    num_flagged: int
    confidence: float
    edited_answer: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "action": self.action,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "num_flagged": self.num_flagged,
            "confidence": round(self.confidence, 4),
            "edited_answer": self.edited_answer,
            "notes": list(self.notes),
        }


class AdaptiveIterationController:
    """Decide ACCEPT / EDIT / REGEN / STOP after each pipeline pass."""

    def __init__(self, config: Optional[IterationConfig] = None):
        self.config = config or IterationConfig()

    # ----- public API --------------------------------------------------------

    def decide(self,
               claim_verdicts: Sequence[Any],
               confidence: float,
               iteration: int) -> tuple:
        """Return ``(action: Action, flagged_claims)``.

        ``flagged_claims`` is the list of verdict-like objects whose
        EEDC score is below the hallucination threshold; downstream code
        feeds them to the :class:`EvidenceGuidedEditor` when ``action is
        EDIT`` or uses them to flag the answer when ``action is REGEN``.

        Each verdict is duck-typed: we only read ``.hallucinated`` and
        ``.evidence_score``, so ``ClaimVerdict`` (from ``src.pipeline``)
        and simple test doubles both work.
        """
        flagged = [v for v in claim_verdicts if v.hallucinated]
        n = len(claim_verdicts)
        rate = (len(flagged) / n) if n else 0.0

        # Hard caps first.
        if iteration >= self.config.max_iterations:
            return Action.STOP, flagged
        if rate <= self.config.accept_rate_threshold:
            return Action.ACCEPT, flagged

        # Number-of-edits check.
        if len(flagged) > self.config.max_edits_per_iteration:
            return Action.REGEN, flagged
        if len(flagged) == 0:
            return Action.ACCEPT, flagged

        # Rate-based decision.
        if rate >= self.config.regen_rate:
            return Action.REGEN, flagged
        if rate <= self.config.accept_rate:
            return Action.ACCEPT, flagged

        # Middle ground: edit if any, regen otherwise.
        return Action.EDIT, flagged

    def should_stop(self,
                    history: Sequence[IterationRecord],
                    current_rate: float) -> bool:
        """Return True if iterating further is unlikely to help.

        Used to break out of the loop when the rate plateaus between
        iterations.
        """
        if len(history) < 2:
            return False
        prev = history[-1].hallucination_rate
        improvement = prev - current_rate
        if improvement < self.config.min_improvement and current_rate > self.config.accept_rate:
            return True
        return False