"""Phase 2 smoke tests — detector, EEDC, pipeline integration, calibration.

Run with CRISP_MOCK=1 so no real models are loaded.
"""
from __future__ import annotations

import json
import os

# Force MOCK mode for the test session.
os.environ["CRISP_MOCK"] = "1"

from pathlib import Path

import pytest

from src.claim_extractor import attach_evidence, extract_claims
from src.config import load_config
from src.detector import LABELS, NLIDetector, NLIPrediction
from src.eedc import (
    EEDCScorer,
    EEDCSignals,
    EEDCWeights,
    normalised_entropy,
    signals_from_prediction,
)


# ---------------------------------------------------------------------------
# detector mock
# ---------------------------------------------------------------------------

def test_detector_mock_returns_valid_label():
    d = NLIDetector(mock=True)
    pred = d.verify(
        claim="Paris is the capital of France.",
        evidence="France's capital city is Paris.",
    )
    assert pred.label in LABELS
    assert len(pred.probs) == 3
    assert abs(sum(pred.probs) - 1.0) < 1e-6
    for p in pred.probs:
        assert 0.0 <= p <= 1.0


def test_detector_mock_recognises_support():
    d = NLIDetector(mock=True)
    pred = d.verify(
        claim="Water boils at 100 degrees Celsius.",
        evidence="At standard atmospheric pressure, water boils at 100 degrees Celsius.",
    )
    # Strong lexical overlap + no negation -> should predict SUP.
    assert pred.label == "SUP"
    assert pred.sup_prob > 0.5


def test_detector_mock_recognises_contradiction():
    d = NLIDetector(mock=True)
    pred = d.verify(
        claim="Water boils at 50 degrees Celsius.",
        evidence="Water does not boil at 50 degrees Celsius.",
    )
    # High lexical overlap (50, celsius, water, boils) + negation cue -> CON.
    assert pred.label == "CON"
    assert pred.con_prob > 0.5


def test_detector_mock_nei_when_no_overlap():
    d = NLIDetector(mock=True)
    pred = d.verify(
        claim="The Mars rover landed on Europa.",
        evidence="Bananas are a tropical fruit grown in over 135 countries.",
    )
    assert pred.label == "NEI"


def test_detector_batch_matches_loop():
    d = NLIDetector(mock=True)
    pairs = [
        ("Paris is the capital of France.", "France's capital is Paris."),
        ("The sky is green.", "The sky is blue."),
    ]
    batch = d.verify_batch(*zip(*pairs))
    loop = [d.verify(c, e) for c, e in pairs]
    assert [p.label for p in batch] == [p.label for p in loop]


def test_nli_prediction_entropy_is_zero_when_certain():
    pred = NLIPrediction(claim="x", evidence="y", label="SUP", probs=[1.0, 0.0, 0.0])
    assert pred.entropy == 0.0


def test_nli_prediction_entropy_is_max_when_uniform():
    pred = NLIPrediction(claim="x", evidence="y", label="NEI",
                         probs=[1/3, 1/3, 1/3])
    import math
    assert abs(pred.entropy - math.log(3)) < 1e-6


# ---------------------------------------------------------------------------
# claim extractor
# ---------------------------------------------------------------------------

def test_extract_claims_splits_sentences():
    claims = extract_claims("Paris is the capital of France. Berlin is in Germany. The sky is blue.")
    assert len(claims) == 3
    assert [c.id for c in claims] == ["c1", "c2", "c3"]
    assert claims[0].text.startswith("Paris")


def test_extract_claims_drops_trivial_fragments():
    claims = extract_claims("Yes. No. The capital of France is Paris.")
    # "Yes." and "No." are dropped.
    assert len(claims) == 1
    assert "Paris" in claims[0].text


def test_extract_claims_handles_empty():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


def test_attach_evidence_pairs_claims_with_top_hit():
    claims = extract_claims("Paris is the capital of France. The sky is blue.")
    hits = [
        type("Hit", (), {"text": "Paris is the capital of France.", "score": 0.9, "index": 0})(),
    ]
    pairs = attach_evidence(claims, hits)
    assert len(pairs) == 2
    for c, h in pairs:
        assert h.text == "Paris is the capital of France."


# ---------------------------------------------------------------------------
# EEDC math
# ---------------------------------------------------------------------------

def test_entropy_normalisation_bounds():
    assert normalised_entropy([1.0, 0.0, 0.0]) == 0.0
    assert abs(normalised_entropy([1/3, 1/3, 1/3]) - 1.0) < 1e-6


def test_signals_vector_is_correct():
    s = EEDCSignals(nli_entropy_norm=0.5, retrieval_top1=0.8, self_consistency=0.9)
    # as_vector = [H, 1-r, 1-c]
    assert s.as_vector() == pytest.approx([0.5, 0.2, 0.1])


def test_eedc_score_in_unit_interval():
    scorer = EEDCScorer()
    for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for r in [0.0, 0.5, 1.0]:
            for c in [0.0, 1.0]:
                s = EEDCSignals(nli_entropy_norm=h, retrieval_top1=r, self_consistency=c)
                phi = scorer.score(s)
                assert 0.0 <= phi <= 1.0


def test_eedc_score_higher_entropy_means_lower_confidence():
    scorer = EEDCScorer()
    confident = EEDCSignals(nli_entropy_norm=0.0, retrieval_top1=1.0, self_consistency=1.0)
    uncertain = EEDCSignals(nli_entropy_norm=1.0, retrieval_top1=0.0, self_consistency=0.0)
    assert scorer.score(confident) > scorer.score(uncertain)


def test_eedc_monotonic_in_entropy():
    scorer = EEDCScorer()
    prev = 1.0
    for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = EEDCSignals(nli_entropy_norm=h, retrieval_top1=0.5, self_consistency=0.5)
        phi = scorer.score(s)
        assert phi <= prev + 1e-9
        prev = phi


def test_eedc_fit_separates_supported_from_unsupported():
    """Calibration should produce weights that rank supported > unsupported."""
    scorer = EEDCScorer()
    signals_supported = [
        EEDCSignals(nli_entropy_norm=0.0, retrieval_top1=1.0, self_consistency=1.0)
        for _ in range(20)
    ]
    signals_unsupported = [
        EEDCSignals(nli_entropy_norm=1.0, retrieval_top1=0.0, self_consistency=0.0)
        for _ in range(20)
    ]
    signals = signals_supported + signals_unsupported
    labels = [1] * 20 + [0] * 20
    scorer.fit(signals, labels)

    # After fitting, supported examples should score clearly higher.
    phi_sup = scorer.score(signals_supported[0])
    phi_unsup = scorer.score(signals_unsupported[0])
    assert phi_sup > phi_unsup
    # And separation should be substantial.
    assert phi_sup - phi_unsup > 0.5


def test_eedc_score_from_prediction():
    pred = NLIPrediction(claim="x", evidence="y", label="SUP", probs=[0.7, 0.1, 0.2])
    scorer = EEDCScorer()
    phi = scorer.score_prediction(pred, retrieval_top1=0.6, self_consistency=0.9)
    # Should match the manual three-step computation.
    expected = scorer.score_from_components(pred.probs, retrieval_top1=0.6, self_consistency=0.9)
    assert phi == pytest.approx(expected)


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_corpus():
    return [
        "Paris is the capital of France.",
        "The novel '1984' was written by George Orwell.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Mars is known as the Red Planet due to iron oxide on its surface.",
        "PyTorch is a deep learning library primarily used with Python.",
    ]


def test_pipeline_populates_claim_verdicts(tiny_corpus):
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus) if False else None
    from src.pipeline import Pipeline
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    assert result.claim_verdicts  # at least one claim
    for v in result.claim_verdicts:
        assert v.claim.text
        assert v.nli.label in LABELS
        assert 0.0 <= v.eedc_score <= 1.0
    # The synthetic answer should be detected with some confidence.
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.hallucination_rate <= 1.0
    # Phase 2 latency should be reported.
    assert "detect" in result.timings_ms


def test_pipeline_disable_detection(tiny_corpus, monkeypatch):
    monkeypatch.setenv("CRISP_DISABLE_DETECT", "1")
    cfg = load_config()
    assert cfg.pipeline.enable_detection is False
    from src.pipeline import Pipeline
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    assert result.claim_verdicts == []
    assert result.hallucination_rate == 0.0
    assert result.confidence == 1.0  # placeholder when detection is off


def test_pipeline_to_dict_includes_phase2(tiny_corpus):
    cfg = load_config()
    from src.pipeline import Pipeline
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("Who wrote '1984'?")
    d = result.to_dict()
    assert "claim_verdicts" in d
    assert "hallucination_rate" in d
    if d["claim_verdicts"]:
        v = d["claim_verdicts"][0]
        assert "nli_label" in v
        assert "eedc_score" in v
        assert "hallucinated" in v


# ---------------------------------------------------------------------------
# calibration script
# ---------------------------------------------------------------------------

def test_calibration_script_runs(tmp_path, monkeypatch):
    # Build a tiny synthetic calibration set: 30 "supported" + 30 "unsupported".
    records = []
    for _ in range(30):
        records.append({"nli_probs": [0.85, 0.05, 0.10],
                        "retrieval_top1": 0.9,
                        "self_consistency": 0.95, "label": 1})
    for _ in range(30):
        records.append({"nli_probs": [0.10, 0.20, 0.70],
                        "retrieval_top1": 0.1,
                        "self_consistency": 0.2, "label": 0})
    cal_path = tmp_path / "cal.jsonl"
    with cal_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    out_path = tmp_path / "weights.json"
    # Invoke the script as a module so we hit its argparse path.
    import runpy
    import sys
    sys.argv = ["calibrate_eedc.py", "--data", str(cal_path), "--out", str(out_path)]
    runpy.run_path(str(Path("scripts") / "calibrate_eedc.py"), run_name="__main__")

    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert set(payload.keys()) == {"alpha", "beta", "gamma", "delta"}


def test_pipeline_loads_calibrated_weights(tmp_path, monkeypatch):
    # Write a tiny weights file and ensure the pipeline picks it up.
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(
        {"alpha": 0.1, "beta": 0.2, "gamma": 0.3, "delta": -0.1}
    ), encoding="utf-8")
    monkeypatch.setenv("CRISP_EEDC_WEIGHTS", str(weights_path))
    cfg = load_config()
    assert Path(cfg.eedc.weights_path) == weights_path

    from src.pipeline import Pipeline
    pipeline = Pipeline(cfg)
    assert pipeline.eedc_scorer.weights.alpha == pytest.approx(0.1)
    assert pipeline.eedc_scorer.weights.beta == pytest.approx(0.2)
    assert pipeline.eedc_scorer.weights.gamma == pytest.approx(0.3)
    assert pipeline.eedc_scorer.weights.delta == pytest.approx(-0.1)