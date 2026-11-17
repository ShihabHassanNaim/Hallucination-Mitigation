"""Regression tests for the closed-book (no-leak) evaluation path.

TruthfulQA's retrieval corpus was built from the dataset's own ``knowledge``
field, which is the set of *correct* answers. Retrieval therefore finds the
reference answer for every row, so any verification against the retrieved
document measures "is this answer the reference answer" -- the label.

These tests lock in the two protections that make the closed-book evaluation
honest:

1. ``retrieval_source: "none"`` makes step 6 retrieve nothing, so the NLI
   in step 8 has no premise that could contain the answer key.
2. ``grounding_source: "none"`` makes step 9 leave the grounding term out of
   the score, and the evidence-dependent components (``retrieval_top1``,
   ``nli``, ``coverage``) are left out too, because with no source they are
   constants rather than measurements.

Without these, the TruthfulQA detection number was 0.9589 with a near-binary
score distribution -- a detector reading the answer key.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def _context(fake_pipeline, answer: str = "The capital of France is Paris."):
    from src.steps import StepContext

    ctx = StepContext(query="What is the capital of France?", cfg=fake_pipeline.cfg)
    ctx.answer = answer
    ctx.metadata["pipeline"] = fake_pipeline
    return ctx


def test_retrieval_none_returns_no_evidence(fake_pipeline):
    """Step 6 must retrieve nothing when the dataset is closed-book."""
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval

    ctx = _context(fake_pipeline)
    ctx.metadata["retrieval_source"] = "none"
    Step06HybridRetrieval(ctx).run()

    assert ctx.evidence == []
    payload = ctx.metadata["step06_hybrid_retrieval"]
    assert payload["evidence_count"] == 0
    assert payload["retrieval_source"] == "none"


def test_retrieval_default_is_unchanged(fake_pipeline):
    """The default path must not be the closed-book path.

    The fake retriever holds no documents, so the default path reports the
    "no index loaded" warning rather than retrieving. What matters here is
    that the closed-book branch was *not* taken: its marker is the
    ``retrieval_source`` key, which only the closed-book branch writes.
    """
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval

    ctx = _context(fake_pipeline)
    Step06HybridRetrieval(ctx).run()

    payload = ctx.metadata["step06_hybrid_retrieval"]
    assert payload.get("retrieval_source") != "none"


def test_grounding_none_leaves_the_term_out_of_the_score(fake_pipeline):
    """With grounding switched off, the term must not contribute a weight."""
    from src.steps.step09_confidence_fusion import Step09ConfidenceFusion

    ctx = _context(fake_pipeline)
    ctx.metadata["grounding_source"] = "none"
    Step09ConfidenceFusion(ctx).run()

    payload = ctx.metadata["step09_confidence_fusion"]
    assert payload["grounding_used"] is False
    assert "grounding" not in payload["weights"]


def test_no_source_drops_the_evidence_dependent_components(fake_pipeline):
    """With no retrieved source, the components that need one are dropped.

    Without evidence the NLI has no premise, so ``nli`` and ``coverage``
    collapse to 0.0 and ``retrieval_top1`` has nothing to score. Those are
    constants, not measurements, and keeping them in would shift every
    closed-book row by the same amount with no discrimination.
    """
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval
    from src.steps.step09_confidence_fusion import Step09ConfidenceFusion

    ctx = _context(fake_pipeline)
    ctx.metadata["retrieval_source"] = "none"
    Step06HybridRetrieval(ctx).run()
    Step09ConfidenceFusion(ctx).run()

    payload = ctx.metadata["step09_confidence_fusion"]
    used = payload["weights"]
    for component in ("retrieval_top1", "nli", "coverage", "grounding"):
        assert component not in used, (
            f"{component} has no source to measure against on a closed-book "
            f"run and must be left out of the score"
        )


def test_closed_book_score_depends_on_the_answer_not_on_a_source(fake_pipeline):
    """A closed-book score must still vary with the answer.

    A near-binary score -- one value per label and nothing in between -- is
    the signature of a detector reading an answer key. On the leaked
    TruthfulQA run the score took essentially two values, one per label,
    because the retrieved document *was* the reference answer.

    The fake embedder is hash-based, so the query-answer cosine barely
    moves between answers. The component that does move is the step-3
    detector's probability, so the test drives that directly: with the
    detector probability changing, the closed-book score must change too.
    That proves the score is computed from the answer's own features rather
    than from a source it never saw.
    """
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval
    from src.steps.step09_confidence_fusion import Step09ConfidenceFusion

    scores = []
    for halluc_prob in (0.05, 0.30, 0.60, 0.90):
        ctx = _context(fake_pipeline)
        ctx.metadata["retrieval_source"] = "none"
        ctx.metadata["grounding_source"] = "none"
        Step06HybridRetrieval(ctx).run()
        # The step-3 detector publishes its probability here; step 9 turns
        # it into a trust component.
        ctx.metadata["hallucination_probability"] = halluc_prob
        Step09ConfidenceFusion(ctx).run()
        scores.append(ctx.confidence)

    assert len(set(round(s, 6) for s in scores)) == len(scores), (
        f"the closed-book score must vary with the answer's own detector "
        f"probability, got {scores}"
    )

    # A higher hallucination probability must mean a *lower* trust score.
    # ``detector_trust`` is 1 - hallucination_probability, so it is a trust
    # signal: the more likely the detector thinks the answer is
    # hallucinated, the less the pipeline should trust it. The measured
    # direction is unambiguous -- the detector's raw probability has AUC
    # 0.5389 read as a risk signal and detector_trust 0.4611, the exact
    # mirror (scripts/check_detector_direction.py).
    for lower_prob, higher_prob in zip(scores, scores[1:]):
        assert higher_prob < lower_prob, (
            f"trust must fall as the detector's hallucination probability "
            f"rises, got {scores}"
        )


def test_step09_documented_signs_are_the_truth_signs():
    """The trust/risk split must match what the module docstring claims."""
    from src.steps.step09_confidence_fusion import TRUST_COMPONENTS

    assert TRUST_COMPONENTS == {"nli", "coverage", "grounding", "detector_trust"}


# --- the artefact side of the same protection ------------------------------
#
# The tests above lock the *pipeline* into the closed-book path. These lock
# the *files*: the two retrieval-on TruthfulQA runs that predate the fix
# still carried the leaked numbers 0.9589 and 0.602764, and both are the
# shape of the label rather than a measurement of it (retrieval recovered
# the reference answer for 972-1000 rows). Leaving them in the repository
# as ordinary-looking metrics files is a trap: nothing reads them, but a
# future run that wrote to the old path, or a reader quoting the highest
# number in the tree, would put a leaked value into the paper.


def _load(path):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


LEAKED_TRUTHFULQA_FILES = (
    "data/metrics_truthfulqa.json",
    "data/metrics_truthfulqa_fitted.json",
)


@pytest.mark.parametrize("path", LEAKED_TRUTHFULQA_FILES)
def test_the_leaked_truthfulqa_files_are_quarantined(path):
    """The two leaked files must stay emptied and labelled, not restored."""
    blob = _load(path)

    assert blob.get("_QUARANTINED") is True, (
        f"{path} holds a leaked TruthfulQA measurement; it must stay "
        f"quarantined so the number cannot be quoted"
    )
    assert "detection_roc_auc" not in blob.get("aggregate", {}), (
        f"{path} must not carry an aggregate ROC-AUC"
    )
    assert not blob.get("per_query"), (
        f"{path} must not carry per-row scores"
    )


@pytest.mark.parametrize("path", LEAKED_TRUTHFULQA_FILES)
def test_the_quarantine_marker_names_the_honest_measurement(path):
    """A reader who opens the file must be pointed at the real number."""
    blob = _load(path)

    honest = blob.get("_honest_measurement", {})
    assert honest.get("path") == "data/metrics_truthfulqa_closedbook.json"
    # The honest file must actually exist and carry that value.
    real = _load(honest["path"])
    assert real["aggregate"]["detection_roc_auc"] == honest["detection_roc_auc"]


def test_the_honest_truthfulqa_number_is_the_one_the_paper_reads():
    """The generator must read the closed-book file, never a leaked one."""
    from scripts.fill_comparison_table import CROSS_DATASETS

    sources = {p for p, _ in CROSS_DATASETS}
    assert Path("data/metrics_truthfulqa_closedbook.json") in sources
    for leaked in LEAKED_TRUTHFULQA_FILES:
        assert Path(leaked) not in sources, (
            f"{leaked} is quarantined and must never be a paper source"
        )


def test_no_truthfulqa_run_has_retrieval_reading_the_answer_key():
    """The honest TruthfulQA file must show retrieval was switched off.

    ``retrieval_recall_at_5 == 1.0`` on nearly every row is the leak
    signature: the corpus was built from the reference answers, so
    retrieval hand-held the gold answer to the verifier.
    """
    real = _load("data/metrics_truthfulqa_closedbook.json")

    assert real.get("retrieval_source") == "none"
    assert real.get("grounding_source") == "none"
    for row in real.get("per_query", []):
        assert row.get("retrieval_recall_at_5") in (0.0, None), (
            "the closed-book run must not retrieve anything; a non-zero "
            "retrieval score means the answer key leaked back in"
        )
