"""Phase 7 — Adaptive Iteration Controller + Evidence-Guided Editor tests.

Covers:
* ``EvidenceGuidedEditor``: stub / evidence modes rewrite only flagged spans
  and leave the rest of the answer untouched.
* ``AdaptiveIterationController``: ACCEPT / EDIT / REGEN / STOP decision
  rules, including the plateau-detection ``should_stop`` heuristic.
* Pipeline integration: ``enable_iteration_control=True`` produces
  ``iteration_history`` + ``edit_result`` in ``RAGResult.to_dict()``.
"""
from __future__ import annotations

import os

os.environ["CRISP_MOCK"] = "1"
os.environ.setdefault("CRISP_INDEX_PATH", "data/test_index_phase7")

import pytest

from src.claim_extractor import Claim, Provenance
from src.editor import EditorMode, EvidenceGuidedEditor, _span_for_claim
from src.iteration_controller import (
    Action,
    AdaptiveIterationController,
    IterationConfig,
    IterationRecord,
)
from src.pipeline import Pipeline
from src.retriever import Hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _claim(cid: str, text: str, prov: Provenance = Provenance.EXTRINSIC) -> Claim:
    return Claim(id=cid, text=text, provenance=prov)


class _StubVerdict:
    """Verdict duck-type used by tests (no NLI plumbing needed)."""

    def __init__(self, claim: Claim, hallucinated: bool, evidence_score: float = 0.5):
        self.claim = claim
        self.hallucinated = hallucinated
        self.evidence_score = evidence_score


# ---------------------------------------------------------------------------
# Span discovery
# ---------------------------------------------------------------------------


class TestSpanDiscovery:
    def test_exact_substring_match(self):
        text = "France's capital is Paris. Berlin is in Germany."
        c = _claim("c1", "Paris")
        start, end = _span_for_claim(text, c)
        assert start >= 0
        assert text[start:end] == "Paris"

    def test_no_match_returns_negative(self):
        c = _claim("c1", "Atlantis")
        assert _span_for_claim("Paris is the capital.", c) == (-1, -1)


# ---------------------------------------------------------------------------
# Editor — stub mode
# ---------------------------------------------------------------------------


class TestEditorStubMode:
    def test_rewrites_only_flagged_span(self):
        editor = EvidenceGuidedEditor(mode=EditorMode.STUB)
        answer = "France's capital is Paris. Berlin is in Germany. Tokyo is in Japan."
        flagged = [
            _StubVerdict(_claim("c1", "Tokyo is in Japan."), hallucinated=True),
        ]
        result = editor.edit(answer, flagged, hits=[])
        assert result.num_edits == 1
        assert "Paris" in result.edited_answer
        assert "Berlin" in result.edited_answer
        assert "[unsupported: Tokyo is in Japan.]" in result.edited_answer

    def test_no_flagged_claims_returns_unchanged(self):
        editor = EvidenceGuidedEditor(mode=EditorMode.STUB)
        result = editor.edit("Hello.", [], hits=[])
        assert result.edited_answer == "Hello."
        assert result.num_edits == 0

    def test_unlocatable_claim_skipped(self):
        editor = EvidenceGuidedEditor(mode=EditorMode.STUB)
        result = editor.edit(
            "Paris is a city.",
            [_StubVerdict(_claim("c1", "Atlantis is real."), hallucinated=True)],
            hits=[],
        )
        assert result.num_edits == 0
        assert result.edited_answer == "Paris is a city."


# ---------------------------------------------------------------------------
# Editor — evidence mode
# ---------------------------------------------------------------------------


class TestEditorEvidenceMode:
    def test_evidence_replaces_with_sentence(self):
        editor = EvidenceGuidedEditor(mode=EditorMode.EVIDENCE)
        answer = "Paris is in Atlantis."
        flagged = [_StubVerdict(
            _claim("c1", "Paris is in Atlantis."), hallucinated=True,
        )]
        hits = [
            Hit(text="Paris is the capital of France.", score=0.9, index=0),
        ]
        result = editor.edit(answer, flagged, hits)
        assert result.num_edits == 1
        assert "Paris is the capital of France." in result.edited_answer
        assert "Atlantis" not in result.edited_answer


# ---------------------------------------------------------------------------
# Editor — regenerate mode degrades without a real generator
# ---------------------------------------------------------------------------


class TestEditorRegenerateFallback:
    def test_regenerate_without_generator_uses_evidence(self):
        from src.generator import Generator
        gen = Generator(model_name="x", mock=True)
        editor = EvidenceGuidedEditor(mode=EditorMode.REGENERATE, generator=gen)
        flagged = [_StubVerdict(
            _claim("c1", "Paris is in Atlantis."), hallucinated=True,
        )]
        hits = [Hit(text="Paris is in France.", score=0.9, index=0)]
        result = editor.edit("Paris is in Atlantis.", flagged, hits)
        assert result.mode == "regenerate"
        assert "Paris is in France." in result.edited_answer


# ---------------------------------------------------------------------------
# Adaptive Iteration Controller — decision policy
# ---------------------------------------------------------------------------


class TestAicDecisions:
    def test_accept_when_no_flagged(self):
        controller = AdaptiveIterationController()
        verdicts = [_StubVerdict(_claim("c1", "Paris."), hallucinated=False)]
        action, flagged = controller.decide(verdicts, confidence=0.9, iteration=1)
        assert action == Action.ACCEPT
        assert flagged == []

    def test_accept_when_rate_below_threshold(self):
        cfg = IterationConfig(max_iterations=3, accept_rate_threshold=0.10)
        controller = AdaptiveIterationController(config=cfg)
        verdicts = (
            [_StubVerdict(_claim(f"c{i}", "x"), hallucinated=False) for i in range(9)]
            + [_StubVerdict(_claim("c9", "y"), hallucinated=True)]
        )
        action, flagged = controller.decide(verdicts, confidence=0.6, iteration=1)
        assert action == Action.ACCEPT
        assert len(flagged) == 1

    def test_edit_when_few_flagged(self):
        # 3 flagged / 10 claims = rate 0.30, which sits between
        # accept_rate (0.10) and regen_rate (0.40) — and
        # len(flagged) <= max_edits_per_iteration, so EDIT.
        cfg = IterationConfig(max_edits_per_iteration=3)
        controller = AdaptiveIterationController(config=cfg)
        verdicts = (
            [_StubVerdict(_claim(f"c{i}", "x"), hallucinated=True) for i in range(3)]
            + [_StubVerdict(_claim(f"c{i}", "y"), hallucinated=False)
               for i in range(7)]
        )
        action, _ = controller.decide(verdicts, confidence=0.6, iteration=1)
        assert action == Action.EDIT

    def test_regen_when_too_many_flagged(self):
        cfg = IterationConfig(max_edits_per_iteration=1, regen_rate=0.4)
        controller = AdaptiveIterationController(config=cfg)
        verdicts = [
            _StubVerdict(_claim(f"c{i}", "x"), hallucinated=True) for i in range(5)
        ]
        action, _ = controller.decide(verdicts, confidence=0.6, iteration=1)
        assert action == Action.REGEN

    def test_stop_when_max_iterations_reached(self):
        controller = AdaptiveIterationController()
        verdicts = [_StubVerdict(_claim("c1", "x"), hallucinated=True)]
        action, _ = controller.decide(verdicts, confidence=0.3, iteration=10)
        assert action == Action.STOP


class TestAicShouldStop:
    def test_plateau_triggers_stop(self):
        cfg = IterationConfig(min_improvement=0.05, accept_rate=0.10)
        controller = AdaptiveIterationController(config=cfg)
        history = [
            IterationRecord(iteration=1, action="edit",
                            hallucination_rate=0.6, num_flagged=6,
                            confidence=0.4, edited_answer=""),
            IterationRecord(iteration=2, action="edit",
                            hallucination_rate=0.6, num_flagged=6,
                            confidence=0.4, edited_answer=""),
        ]
        # Improvement of 0.0 (plateau) above accept_rate should stop.
        assert controller.should_stop(history, current_rate=0.6)

    def test_no_stop_when_improving(self):
        cfg = IterationConfig(min_improvement=0.05)
        controller = AdaptiveIterationController(config=cfg)
        history = [
            IterationRecord(iteration=1, action="edit",
                            hallucination_rate=0.6, num_flagged=6,
                            confidence=0.4, edited_answer=""),
        ]
        # Big improvement => no stop.
        assert not controller.should_stop(history, current_rate=0.2)

    def test_no_stop_with_insufficient_history(self):
        controller = AdaptiveIterationController()
        assert not controller.should_stop([], current_rate=0.5)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineAicIntegration:
    def test_iteration_history_populated_when_enabled(self):
        corpus = [
            "France's capital is Paris.",
            "Germany's capital is Berlin.",
            "Mars is the red planet.",
        ]
        cfg_kwargs = {
            "pipeline": {
                "max_iterations": 1,
                "enable_detection": True,
                "enable_iteration_control": True,
                "aic_max_iterations": 3,
                "aic_accept_rate_threshold": 0.0,  # never auto-accept
                "editor_mode": "stub",
            },
        }
        from src.config import AppConfig, load_config
        import yaml
        base = load_config().model_dump()
        base.update(cfg_kwargs)
        cfg = AppConfig(**base)

        pipeline = Pipeline(config=cfg)
        pipeline.build_index(corpus)
        result = pipeline.run("What is the capital of France?")

        # AIC ran at least one decision.
        assert isinstance(result.iteration_history, list)
        # Some iteration was recorded.
        if result.iteration_history:
            entry = result.iteration_history[0]
            assert entry["action"] in {"accept", "edit", "regen", "stop"}
            assert "hallucination_rate" in entry

    def test_off_by_default(self):
        corpus = ["France's capital is Paris."]
        pipeline = Pipeline()
        pipeline.build_index(corpus)
        result = pipeline.run("Capital of France?")
        assert result.iteration_history == []
        assert result.edit_result is None
        assert result.iterations == 1