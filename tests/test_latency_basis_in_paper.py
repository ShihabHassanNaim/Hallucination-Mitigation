"""The paper must describe the latency band as the detection path.

The evaluation supplies the benchmark's answer, so step 2 skips
generation (``src/steps/step02_grounded_generation.py``). The reported
0.8-13.8 s band therefore covers retrieval, reranking and claim-level
verification -- not a generator call.

Three places in the prose asserted the opposite ("because it runs a
generator call", "One generator call"), which contradicted the paper's own
"detection path only" note and made the latency comparison read as
faster than it is. These tests keep the two in step.
"""
from pathlib import Path

PROSE_FILES = [
    Path("Paper/comparison.md"),
    Path("Paper/comparison_summary.tex"),
    Path("README.md"),
]

# Phrases that would claim the measured band includes generation.
GENERATOR_CALL_CLAIMS = [
    "because it runs a generator call",
    "One generator call",
]


def _existing():
    return [p for p in PROSE_FILES if p.exists()]


def test_no_prose_claims_the_band_includes_a_generator_call():
    offenders = []
    for path in _existing():
        text = path.read_text(encoding="utf-8")
        for claim in GENERATOR_CALL_CLAIMS:
            if claim in text:
                offenders.append(f"{path}: {claim!r}")
    assert not offenders, (
        "the latency prose claims the measured band includes a generator "
        "call, but the evaluation supplies the answer and step 2 skips "
        f"generation: {offenders}"
    )


def test_the_latency_prose_names_the_detection_path():
    """At least one place must state the band is the detection path."""
    found = False
    for path in _existing():
        text = path.read_text(encoding="utf-8")
        low = text.lower()
        if "detection path" in low or "detection-path" in low:
            found = True
            break
    assert found, (
        "no prose file states that the measured latency band is the "
        "detection path; without that caveat the band reads as a full "
        "generating pipeline"
    )


def test_the_latency_verdict_does_not_claim_latency_parity():
    """The pipeline is slower than LLM-Check; it must not claim parity."""
    for path in _existing():
        if path.name != "comparison.md":
            continue
        text = path.read_text(encoding="utf-8")
        assert "is competitive with the fastest single-pass detectors on latency" not in text, (
            "comparison.md still claims latency parity with the fastest "
            "single-pass detectors, which the measurement contradicts"
        )
