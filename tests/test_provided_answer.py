"""Tests for verifying a supplied answer instead of generating one.

The evaluation measures detection on the dataset's own answers, so the
pipeline must be able to take an answer and judge it without generating a
different one. These tests cover that path.
"""
from __future__ import annotations

import pytest

from src.pipeline import Pipeline
from src.steps.step02_grounded_generation import Step02GroundedGeneration
from src.steps.step10_regeneration import Step10Regeneration


def test_run_accepts_a_provided_answer(fake_pipeline):
    """The answer under judgement must be the one supplied."""
    answer = "Suffolk County"
    res = fake_pipeline.run("What county is Lindenhurst in?", answer=answer)
    assert res.answer == answer


def test_provided_answer_bypasses_the_generator(fake_pipeline):
    """The generator must not be called when an answer is supplied."""
    calls: list[str] = []

    class _ExplodingGenerator:
        def generate(self, system: str, user: str) -> str:
            calls.append(user)
            raise AssertionError("generator must not run for a provided answer")

    # The session-scoped fixture is shared, so restore the generator
    # afterwards rather than leaving a stub behind for other tests.
    original = fake_pipeline.generator
    fake_pipeline.generator = _ExplodingGenerator()
    try:
        res = fake_pipeline.run("What county is Lindenhurst in?", answer="Suffolk County")
    finally:
        fake_pipeline.generator = original
    assert res.answer == "Suffolk County"
    assert not calls


def test_provided_answer_is_not_rewritten_by_regeneration():
    """A supplied answer is judged, not improved.

    Step 10 rewrites answers whose claims are unsupported. If it did that
    to a provided answer, the gate and the detector would score a
    different answer than the benchmark supplied.
    """
    from src.config import load_config
    from src.steps import StepContext
    from src.types import Claim, Evidence

    ctx = StepContext(query="q", cfg=load_config())
    ctx.metadata["provided_answer"] = "A factually wrong statement"
    ctx.answer = "A factually wrong statement"
    ctx.claims = [
        Claim(
            text="A factually wrong statement",
            verdict="contradicted",
            confidence=0.9,
        )
    ]
    ctx.evidence = [Evidence(doc_id="d1", text="the real answer", score=1.0)]

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == "A factually wrong statement"
    # The reduction action is still recorded.
    assert ctx.decision_log
    assert ctx.decision_log[-1]["decision"] == "KEEP"


def test_generation_step_marks_provided_answers(fake_pipeline):
    """The step trace must say the answer was provided, not generated."""
    from src.config import load_config
    from src.steps import StepContext

    ctx = StepContext(query="q", cfg=load_config())
    ctx.metadata["provided_answer"] = "given"
    ctx.metadata["pipeline"] = fake_pipeline
    Step02GroundedGeneration(ctx).run()
    assert ctx.answer == "given"
    assert ctx.metadata["step02_grounded_generation"]["provided"] is True
    assert ctx.metadata["step02_grounded_generation"]["backend"].endswith("(provided)")


def test_run_without_a_provided_answer_still_generates(fake_pipeline):
    """The default path must be unchanged."""
    res = fake_pipeline.run("What county is Lindenhurst in?")
    assert res.answer != ""


def test_provided_answer_differs_per_call(fake_pipeline):
    """Two calls with different answers must not share the answer.

    The evaluation relies on this: the faithful and the hallucinated row
    of a question are judged separately, and identical answers would make
    the detector score identical by construction.
    """
    a = fake_pipeline.run("q", answer="The correct answer")
    b = fake_pipeline.run("q", answer="A different, wrong answer")
    assert a.answer != b.answer
