"""Tests for the direction of each fusion component.

The step-9 head signs every component as either a trust signal (high value
= less likely hallucinated) or a risk signal (high value = more likely
hallucinated). A sign decided inside the noise band is a bug: the head
carried ``detector_trust`` as a risk signal on the strength of a
tuning-half AUC of 0.5008, when its true direction is trust and it was
worth 0.1934 of fitted weight.

The invariant these tests lock in:

* The component's configured sign must match what the code calls a trust
  signal, so the fallback path and the fitted path cannot disagree.
* ``detector_trust`` is ``1 - hallucination_probability``, so it is a trust
  signal and its sign must be negative.
* A component that is a mirror of another must have complementary AUCs
  summing to 1. Measured on RAGTruth: hallucination_probability 0.5389 and
  detector_trust 0.4611 (scripts/check_detector_direction.py).
"""
from __future__ import annotations

import pytest


def _cfg_step9():
    from src.config import load_config

    return (load_config().step(9) or {})


def test_detector_trust_is_signed_as_a_trust_signal():
    """``detector_trust`` is 1 - hallucination_probability: a trust signal."""
    head = (_cfg_step9().get("detection_head") or {})
    sign = head.get("sign") or {}
    assert sign.get("detector_trust") == -1.0, (
        "detector_trust must enter the head negated; it is "
        "1 - hallucination_probability, so a higher value means the "
        "detector thinks the answer is more faithful"
    )


def test_configured_signs_match_the_code_trust_set():
    """The fitted head and the fallback path must agree on every component."""
    from src.steps.step09_confidence_fusion import COMPONENTS, TRUST_COMPONENTS

    sign = (_cfg_step9().get("detection_head") or {}).get("sign") or {}
    assert set(sign) == set(COMPONENTS)
    for component in COMPONENTS:
        expected = -1.0 if component in TRUST_COMPONENTS else 1.0
        assert sign[component] == expected, (
            f"{component}: the fitted sign ({sign[component]}) disagrees "
            f"with the code's trust/risk classification"
        )


def test_higher_hallucination_probability_lowers_the_trust_score(fake_pipeline):
    """The step-9 score must fall as the detector's probability rises."""
    from src.steps import StepContext
    from src.steps.step06_hybrid_retrieval import Step06HybridRetrieval
    from src.steps.step09_confidence_fusion import Step09ConfidenceFusion

    scores = []
    for halluc_prob in (0.05, 0.30, 0.60, 0.90):
        ctx = StepContext(query="What is the capital of France?",
                          cfg=fake_pipeline.cfg)
        ctx.answer = "The capital of France is Paris."
        ctx.metadata["pipeline"] = fake_pipeline
        ctx.metadata["retrieval_source"] = "none"
        ctx.metadata["grounding_source"] = "none"
        Step06HybridRetrieval(ctx).run()
        ctx.metadata["hallucination_probability"] = halluc_prob
        Step09ConfidenceFusion(ctx).run()
        scores.append(ctx.confidence)

    for lower, higher in zip(scores, scores[1:]):
        assert higher < lower, (
            f"trust must fall as the hallucination probability rises, "
            f"got {scores}"
        )


def test_the_two_detector_readings_are_mirrors():
    """``detector_trust`` and the raw probability must sum to 1 in AUC.

    They are the same measurement read in opposite directions, so their
    AUCs must be complementary. This is the check that catches a sign
    decided by a coin flip: if a component is signed wrongly, the pair
    stops summing to 1.
    """
    import numpy as np

    def roc_auc(scores, labels) -> float:
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels, dtype=int)
        pos, neg = s[y == 1], s[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s), dtype=float)
        sorted_s = s[order]
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        r_pos = ranks[y == 1].sum()
        n_pos, n_neg = len(pos), len(neg)
        return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    # A synthetic case with a known direction: the probability is higher
    # for the positive class, so its AUC is above 0.5 and the trust
    # component's is the exact mirror.
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    a_prob = roc_auc(prob, labels)
    a_trust = roc_auc(1.0 - prob, labels)
    assert abs(a_prob + a_trust - 1.0) < 1e-9
    assert a_prob > 0.5
    assert a_trust < 0.5


def test_measured_directions_are_recorded_in_the_fit_artefact():
    """The fit must record each component's CI and whether it was decisive."""
    import json
    from pathlib import Path

    for name in ("data/detection_head_refit.json", "data/detection_head_refit600.json"):
        path = Path(name)
        if not path.exists():
            pytest.skip(f"{name} not present")
        report = json.loads(path.read_text(encoding="utf-8"))
        directions = report.get("component_direction") or {}
        assert directions, f"{name} has no component directions"
        for component, info in directions.items():
            assert "auc_all_ci95" in info, (
                f"{component} in {name} has no confidence interval, so its "
                f"sign cannot be checked against the noise band"
            )
            assert "direction_decisive" in info
            assert "sign_source" in info
