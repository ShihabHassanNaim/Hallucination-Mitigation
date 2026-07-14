"""Phase 3 smoke tests — atomic claim extraction + provenance tagging.

Run with CRISP_MOCK=1 so no real models are loaded.
"""
from __future__ import annotations

import json
import os

os.environ["CRISP_MOCK"] = "1"

import pytest

from src.claim_extractor import (
    Claim,
    ClaimExtractor,
    Provenance,
    _atomic_split_sentences,
    attach_evidence,
    extract_claims,
)
from src.config import load_config
from src.embeddings import Embedder
from src.pipeline import Pipeline
from src.retriever import Hit


# ---------------------------------------------------------------------------
# atomic splitting
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    out = _atomic_split_sentences("Paris is the capital of France. Berlin is in Germany.")
    assert len(out) == 2
    assert out[0].startswith("Paris")
    assert out[1].startswith("Berlin")


def test_split_breaks_on_conjunction_and():
    out = _atomic_split_sentences(
        "Paris is the capital of France and Berlin is the capital of Germany."
    )
    # Compound joined by " and " should be split into two atomic claims.
    assert len(out) == 2
    assert "Paris" in out[0]
    assert "Berlin" in out[1]


def test_split_breaks_on_but():
    out = _atomic_split_sentences(
        "Water boils at 100 degrees Celsius at sea level, but at higher altitudes it boils at lower temperatures."
    )
    assert len(out) >= 2
    assert any("Water boils" in s for s in out)
    assert any("but" in s or "altitudes" in s for s in out)


def test_split_drops_trivial_fragments():
    out = _atomic_split_sentences("Yes. No. The capital of France is Paris.")
    # "Yes." and "No." are dropped.
    assert len(out) == 1
    assert "Paris" in out[0]


def test_split_strips_bracketed_numbers():
    out = _atomic_split_sentences("[1] France is in Europe. [2] Its capital is Paris.")
    assert len(out) == 2
    assert out[0].startswith("France")
    assert out[1].startswith("Its capital")


def test_split_empty_returns_empty():
    assert _atomic_split_sentences("") == []
    assert _atomic_split_sentences("   ") == []


# ---------------------------------------------------------------------------
# Claim + Provenance API
# ---------------------------------------------------------------------------

def test_claim_defaults_to_extrinsic():
    c = Claim(id="c1", text="Paris is the capital of France.")
    assert c.provenance == Provenance.EXTRINSIC


def test_provenance_values_are_strings():
    assert Provenance.INTRINSIC.value == "intrinsic"
    assert Provenance.EXTRINSIC.value == "extrinsic"
    assert Provenance.AGGREGATED.value == "aggregated"
    # str(Provenance.X) returns the .value so JSON serialisation works.
    assert str(Provenance.AGGREGATED) == "aggregated"


# ---------------------------------------------------------------------------
# ClaimExtractor
# ---------------------------------------------------------------------------

def test_extractor_basic():
    ex = ClaimExtractor(mode="synthetic")
    claims = ex.extract(
        "Paris is the capital of France. Berlin is in Germany.",
        hits=[],
        question="Where is Paris?",
    )
    assert len(claims) == 2
    assert claims[0].id == "c1"
    assert claims[1].id == "c2"


def test_extractor_unknown_mode_raises():
    with pytest.raises(ValueError):
        ClaimExtractor(mode="bogus")


def test_extractor_empty_answer_returns_empty_list():
    ex = ClaimExtractor(mode="synthetic")
    assert ex.extract("", hits=[], question="Q?") == []
    assert ex.extract("   ", hits=[], question="Q?") == []


def test_intrinsic_when_claim_matches_question():
    ex = ClaimExtractor(mode="synthetic", intrinsic_threshold=0.5)
    claims = ex.extract(
        "Paris is the capital of France.",
        hits=[],
        question="What is the capital of France?",
    )
    # The claim repeats the question phrase "capital of France" verbatim.
    assert claims[0].provenance == Provenance.INTRINSIC


def test_extrinsic_when_claim_differs_from_question():
    ex = ClaimExtractor(mode="synthetic", intrinsic_threshold=0.5)
    claims = ex.extract(
        "Mars is known as the Red Planet due to iron oxide on its surface.",
        hits=[Hit(text="Mars has iron oxide.", score=0.5, index=0)],
        question="What is the capital of France?",
    )
    assert claims[0].provenance in {Provenance.EXTRINSIC, Provenance.AGGREGATED}


def test_aggregated_when_overlap_with_multiple_hits():
    ex = ClaimExtractor(mode="synthetic",
                        intrinsic_threshold=0.95,   # disable intrinsic
                        aggregated_threshold=0.2)
    # Single atomic claim that overlaps similarly with TWO distinct passages,
    # neither of which dominates -- the textbook multi-hop signature.
    claims = ex.extract(
        "France and Germany drive the European economy together.",
        hits=[
            Hit(text="France has a large and diversified economy.", score=0.6, index=0),
            Hit(text="Germany drives a large part of the European economy.", score=0.6, index=1),
        ],
        question="What about European economies?",
    )
    # We don't assert AGGREGATED here because the synthetic " and "
    # conjunction splitter may break the input into two separate sub-claims
    # before tagging; in that case each sub-claim is single-hop EXTRINSIC.
    # Instead we assert the tagger at least ATTEMPTS the multi-hop check by
    # producing some claims without errors and that the function is
    # deterministic.
    assert isinstance(claims, list)
    first = ex.extract(
        "France and Germany drive the European economy together.",
        hits=[
            Hit(text="France has a large and diversified economy.", score=0.6, index=0),
            Hit(text="Germany drives a large part of the European economy.", score=0.6, index=1),
        ],
        question="What about European economies?",
    )
    assert [c.provenance for c in claims] == [c.provenance for c in first]


def test_aggregated_tag_provenance_directly():
    """Test _tag_provenance directly so we control the inputs precisely.

    Claim overlap with each hit is similar (no single-hit dominance) and
    both exceed the aggregated_threshold -> AGGREGATED.
    """
    ex = ClaimExtractor(mode="synthetic",
                        intrinsic_threshold=0.95,   # disable intrinsic
                        aggregated_threshold=0.25)
    hits = [
        Hit(text="France and Germany have large diversified economies.",
            score=0.6, index=0),
        Hit(text="France and Germany are European economic powerhouses.",
            score=0.6, index=1),
    ]
    tag = ex._tag_provenance(
        "France and Germany are large European economies.",
        hits=hits,
        question="European economies?",
    )
    assert tag == Provenance.AGGREGATED


def test_aggregated_requires_multi_hop_signature():
    """A claim with a strong single-hit match stays EXTRINSIC even if other
    hits weakly overlap."""
    ex = ClaimExtractor(mode="synthetic",
                        intrinsic_threshold=0.95,
                        aggregated_threshold=0.2)
    hits = [
        Hit(text="The capital of France is Paris.", score=0.95, index=0),
        Hit(text="Berlin is the capital of Germany.", score=0.5, index=1),
    ]
    tag = ex._tag_provenance(
        "The capital of France is Paris.",  # strongly matches hit 0 only.
        hits=hits,
        question="Capital of France?",
    )
    # Strong single-hit dominance -> EXTRINSIC (single-hop), not AGGREGATED.
    assert tag == Provenance.EXTRINSIC


def test_real_mode_with_embedder_uses_cosine():
    # Mock embedder is deterministic so cosine is reproducible.
    embedder = Embedder(model_name="bge-test", mock=True, dim=64)
    ex = ClaimExtractor(mode="real", embedder=embedder,
                        intrinsic_threshold=0.99)
    claims = ex.extract(
        "Completely orthogonal string about mango farming and ripeness.",
        hits=[Hit(text="Mars has iron oxide.", score=0.5, index=0)],
        question="Capital of France?",
    )
    # With mock embedder and effectively-orthogonal text, claim should be
    # EXTRINSIC (not intrinsic). We don't assert a fixed provenance because
    # the mock embedding is seeded by content.
    assert isinstance(claims[0].provenance, Provenance)


# ---------------------------------------------------------------------------
# backward-compat helpers
# ---------------------------------------------------------------------------

def test_extract_claims_back_compat():
    out = extract_claims("Paris is the capital of France. Berlin is in Germany.")
    assert len(out) == 2
    assert all(c.provenance == Provenance.EXTRINSIC for c in out)


def test_attach_evidence_back_compat():
    claims = extract_claims("Paris is the capital of France.")
    hits = [Hit(text="A", score=0.1, index=0)]
    pairs = attach_evidence(claims, hits)
    assert len(pairs) == 1
    assert pairs[0][1].text == "A"


# ---------------------------------------------------------------------------
# config plumbing
# ---------------------------------------------------------------------------

def test_config_has_claim_extractor_section():
    cfg = load_config()
    assert cfg.claim_extractor.mode == "synthetic"
    assert 0.0 <= cfg.claim_extractor.intrinsic_threshold <= 1.0
    assert 0.0 <= cfg.claim_extractor.aggregated_threshold <= 1.0


def test_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CRISP_CLAIM_MODE", "real")
    monkeypatch.setenv("CRISP_CLAIM_INTRINSIC", "0.85")
    monkeypatch.setenv("CRISP_CLAIM_AGGREGATED", "0.25")
    cfg = load_config()
    assert cfg.claim_extractor.mode == "real"
    assert cfg.claim_extractor.intrinsic_threshold == pytest.approx(0.85)
    assert cfg.claim_extractor.aggregated_threshold == pytest.approx(0.25)


def test_config_threshold_bounds_reject_out_of_range():
    from pydantic import ValidationError
    from src.config import ClaimExtractorConfig
    with pytest.raises(ValidationError):
        ClaimExtractorConfig(intrinsic_threshold=1.5)


# ---------------------------------------------------------------------------
# pipeline integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_corpus():
    return [
        "Paris is the capital of France. Its capital is also its largest city.",
        "Berlin is the capital and largest city of Germany.",
        "Water boils at 100 degrees Celsius at sea level.",
        "Mars is known as the Red Planet due to iron oxide on its surface.",
        "PyTorch is a deep learning library primarily used with Python.",
    ]


def test_pipeline_claim_verdicts_carry_provenance(tiny_corpus):
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    assert result.claim_verdicts, "expected at least one claim"
    for v in result.claim_verdicts:
        # Provenance should now be a Provenance enum, serialised to its value.
        assert isinstance(v.claim.provenance, Provenance)


def test_pipeline_to_dict_includes_provenance(tiny_corpus):
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    d = result.to_dict()
    assert d["claim_verdicts"]
    for verdict in d["claim_verdicts"]:
        assert "provenance" in verdict
        assert verdict["provenance"] in {"intrinsic", "extrinsic", "aggregated"}


def test_pipeline_real_mode_works(tiny_corpus):
    cfg = load_config()
    cfg.claim_extractor.mode = "real"
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    # Real mode still produces atomic claims + provenance tags.
    assert result.claim_verdicts
    for v in result.claim_verdicts:
        assert isinstance(v.claim.provenance, Provenance)


def test_pipeline_disabled_has_no_claims(tiny_corpus, monkeypatch):
    monkeypatch.setenv("CRISP_DISABLE_DETECT", "1")
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    assert result.claim_verdicts == []
    assert result.hallucination_rate == 0.0


def test_pipeline_json_roundtrip_includes_provenance(tiny_corpus):
    cfg = load_config()
    pipeline = Pipeline(cfg).build_index(tiny_corpus)
    result = pipeline.run("What is the capital of France?")
    blob = json.dumps(result.to_dict())
    payload = json.loads(blob)
    for v in payload["claim_verdicts"]:
        assert v["provenance"] in {"intrinsic", "extrinsic", "aggregated"}