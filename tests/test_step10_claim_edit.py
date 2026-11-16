"""Tests for the per-claim rewrite in Step 10.

``claim_edit`` mode exists because ``evidence`` mode rebuilds the answer
from the supported claims alone. That discarded the original phrasing of
every claim the verifier did not explicitly mark as supported, and when no
claim was supported it substituted a raw retrieved passage for the answer.
On the length-matched HaluEval split that turned an already-correct answer
into a worse one -- a BERTScore of 1.000 fell to 0.811 -- so the measured
mitigation delta was negative.

These tests pin the behaviour that fixes it: the answer is edited in place,
only the flagged claim changes, and a correct answer is never degraded.
"""
from __future__ import annotations

import pytest

from src.steps import StepContext
from src.steps.step10_regeneration import Step10Regeneration, _find_span
from src.types import Claim, Evidence


def _ctx(cfg, answer: str, claims: list[Claim], evidence=None):
    ctx = StepContext(query="q", cfg=cfg)
    ctx.answer = answer
    ctx.claims = claims
    ctx.evidence = evidence or []
    # A generator has to be present, otherwise the step takes the
    # "gen is None" branch and never reaches the rewrite logic.
    class _Gen:
        def generate(self, system, user):
            return "GENERATED"
    ctx.metadata["pipeline"] = type("P", (), {"generator": _Gen()})()
    return ctx


def _claim(text: str, verdict: str, confidence: float, ev=None) -> Claim:
    c = Claim(text=text, verdict=verdict, confidence=confidence)
    c.evidence = ev or []
    return c


def _cfg(tmp_path, mode="claim_edit", min_conf=0.5):
    from src.config import Config, load_config

    base = load_config()
    raw = dict(base.raw)
    raw["step10_regeneration"] = {
        "mode": mode, "max_attempts": 2,
        "min_contradiction_confidence": min_conf,
    }
    return Config(raw=raw)


def test_a_flagged_claim_is_replaced_and_the_rest_is_untouched(tmp_path):
    """Only the flagged span changes; the other span is byte-identical."""
    answer = ("Suffolk County is the location of Lindenhurst. "
              "The End of the Innocence was a single by Donald Hugh Henley.")
    bad = "The End of the Innocence was a single by Donald Hugh Henley."
    good = "Suffolk County is the location of Lindenhurst."
    # The replacement has to be about the same subject, otherwise the
    # relevance guard rejects it -- which is the point of that guard.
    ev = Evidence(doc_id="doc-1",
                  text="Donald Hugh Henley released The End of the Innocence.",
                  score=0.9)

    ctx = _ctx(_cfg(tmp_path), answer,
               [_claim(good, "supported", 0.9),
                _claim(bad, "contradicted", 0.1, [ev])])

    Step10Regeneration(ctx).run()

    assert ctx.final_answer != answer, "nothing was rewritten"
    assert "Donald Hugh Henley released" in ctx.final_answer, (
        f"the evidence did not replace the claim: {ctx.final_answer!r}"
    )
    assert "The End of the Innocence was a single by Donald Hugh Henley." \
        not in ctx.final_answer, "the contradicted claim survived"
    # The supported claim keeps its exact wording.
    assert good in ctx.final_answer, (
        f"the untouched claim was altered: {ctx.final_answer!r}"
    )


def test_a_replacement_about_a_different_subject_is_rejected(tmp_path):
    """A replacement that does not mention the claim's subject is refused.

    The verifier can mark a correct claim as contradicted when the
    retrieved evidence is about a different subject. Swapping in a
    replacement about that other subject would replace a correct answer
    with a wrong one.
    """
    answer = "Memphis International Airport"
    ev = Evidence(doc_id="doc-1", text="The single was released by Geffen.", score=0.9)
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "contradicted", 0.1, [ev])])

    Step10Regeneration(ctx).run()

    rec = ctx.metadata["step10_regeneration"]
    assert rec["decision"] == "ABSTAIN"
    assert rec["guard"] == "replacement_not_relevant"
    assert ctx.final_answer == answer


def test_an_answer_with_no_usable_replacement_is_left_alone(tmp_path):
    """A flagged claim with no evidence must not cause the answer to be
    replaced by a retrieved passage -- that is what produced the
    1.000 -> 0.811 regression."""
    answer = "Suffolk County is the location of Lindenhurst."
    ctx = _ctx(_cfg(tmp_path), answer,
               [_claim(answer, "contradicted", 0.1, [])],
               evidence=[Evidence(doc_id="doc-9", text="Some other passage entirely.", score=0.5)])

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == answer, (
        f"the answer was replaced with a retrieved passage: {ctx.final_answer!r}"
    )
    rec = ctx.metadata["step10_regeneration"]
    assert rec["decision"] == "ABSTAIN"
    assert rec["guard"] == "no_usable_replacement"


def test_a_low_confidence_contradiction_does_not_rewrite(tmp_path):
    """The guard keeps a weakly-flagged answer intact.

    ``Claim.confidence`` is ``1 - contradiction_probability`` for a
    contradicted claim, so a contradiction strength below the threshold
    must not trigger a rewrite.
    """
    answer = "Suffolk County is the location of Lindenhurst."
    ev = Evidence(doc_id="doc-1", text="Some replacement text.", score=0.9)
    # confidence 0.9 -> contradiction strength 0.1, below the 0.5 threshold.
    ctx = _ctx(_cfg(tmp_path, min_conf=0.5), answer,
               [_claim(answer, "contradicted", 0.9, [ev])])

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == answer, (
        f"a weakly-flagged answer was rewritten: {ctx.final_answer!r}"
    )
    rec = ctx.metadata["step10_regeneration"]
    assert rec["decision"] == "ABSTAIN"
    assert rec["guard"] == "no_confident_contradiction"


def test_an_unknown_verdict_is_not_treated_as_a_contradiction(tmp_path):
    """An "unknown" verdict means the evidence was missing, not that the
    claim is false, so the answer is left alone."""
    answer = "Suffolk County is the location of Lindenhurst."
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "unknown", 0.3, [])])

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == answer
    assert ctx.metadata["step10_regeneration"]["decision"] == "KEEP"


def test_a_supported_answer_is_never_touched(tmp_path):
    """The common case: nothing is flagged, so nothing changes."""
    answer = "Suffolk County is the location of Lindenhurst."
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "supported", 0.95)])

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == answer
    assert ctx.metadata["step10_regeneration"]["decision"] == "KEEP"


def test_the_provided_answer_passes_through_untouched(tmp_path):
    """A benchmark-supplied answer is judged, not rewritten."""
    answer = "Suffolk County is the location of Lindenhurst."
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "contradicted", 0.1, [])])
    ctx.metadata["provided_answer"] = answer

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == answer
    assert ctx.metadata["step10_regeneration"]["mode"] == "provided_passthrough"


def test_find_span_locates_a_claim_with_different_whitespace():
    answer = "Suffolk  County is the location of Lindenhurst."
    span = _find_span(answer, "Suffolk County is the location of Lindenhurst.")
    assert span is not None
    assert answer[span[0]:span[1]].lower().startswith("suffolk")


def test_find_span_returns_none_for_text_that_is_not_present():
    assert _find_span("The cat sat on the mat.", "A dog stood on a rug.") is None


def test_the_replacement_is_a_sentence_not_a_whole_passage(tmp_path):
    """A retrieved passage must not be substituted wholesale.

    The evidence for a claim is a retrieved chunk, which is usually a
    paragraph. Substituting it wholesale for a short claim turned a
    four-word answer into a two-hundred-word passage about an unrelated
    subject. Only the best-matching sentence of the evidence is used.
    """
    answer = "Memphis International Airport"
    ev = Evidence(
        doc_id="doc-1038",
        text=("Kumasi International Airport (IATA: KMS, ICAO: DGSI) is an "
              "international airport in Ghana serving Kumasi, the capital "
              "of the Ashanti Region. Ghana has two international airports."),
        score=0.0,
    )
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "contradicted", 0.1, [ev])])

    Step10Regeneration(ctx).run()

    rec = ctx.metadata["step10_regeneration"]
    assert rec["decision"] == "ABSTAIN", (
        f"the answer was rewritten with an irrelevant passage: "
        f"{ctx.final_answer!r}"
    )
    assert rec["guard"] == "replacement_not_proportionate"


def test_a_proportionate_replacement_is_applied(tmp_path):
    """The guards must not block every rewrite."""
    answer = "Mary Ramsey is younger than Lee Ranaldo."
    ev = Evidence(doc_id="doc-5", text="Mary Ramsey was born in 1984.", score=0.9)
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "contradicted", 0.1, [ev])])

    Step10Regeneration(ctx).run()

    rec = ctx.metadata["step10_regeneration"]
    assert rec["decision"] == "EDITED"
    assert "1984" in ctx.final_answer


def test_a_correct_answer_is_never_degraded_by_a_false_contradiction(tmp_path):
    """The regression this whole change exists for.

    Before the fix, a correct answer scoring 1.000 against the gold
    answer fell to 0.811 because the rewrite replaced the whole answer
    with a retrieved passage. Now the answer is left alone when the
    evidence cannot sensibly replace the claim.
    """
    answer = "Memphis International Airport"
    ev = Evidence(doc_id="doc-1", text="A completely unrelated passage.", score=0.1)
    ctx = _ctx(_cfg(tmp_path), answer, [_claim(answer, "contradicted", 0.1, [ev])])

    Step10Regeneration(ctx).run()

    assert ctx.final_answer == answer, (
        f"a correct answer was degraded: {ctx.final_answer!r}"
    )
