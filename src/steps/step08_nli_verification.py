"""Step 8 — NLI Fact Verification (DeBERTa-v3-MNLI).

For every claim, score against each retrieved evidence and pick the best.

Claim-conditioned premise selection
------------------------------------
The NLI model truncates its premise to 512 tokens. When the retrieved
document is longer than that, only its opening reaches the model, so a
claim that the document supports *later* is judged "unknown" -- the
evidence was there, the model simply never saw it. On RAGTruth, whose
contexts run to 9900 characters, this made a third of all claims
undecidable and drove the detection score to chance.

So the premise is not the whole document. It is the part of the document
that is about the claim: the sentences with the highest token overlap
with the claim, padded with their immediate neighbours so the NLI still
sees a coherent passage. This is the same claim-conditioned selection the
regeneration step uses to pick a replacement, applied here so the
verifier and the rewriter agree on what "the evidence for this claim" is.
"""
from __future__ import annotations

import re

from . import Step, StepContext
from ..types import ClaimVerdict
from ..utils.chunker import split_sentences


def _claim_premise(evidence_text: str, claim_text: str,
                   max_chars: int = 1800) -> str:
    """The part of ``evidence_text`` that is about ``claim_text``.

    Selects the sentences with the highest token overlap with the claim and
    joins them in their original order. If the document is short enough to
    fit the NLI's window anyway, it is returned unchanged.
    """
    if not evidence_text:
        return ""
    if len(evidence_text) <= max_chars:
        return evidence_text

    sentences = split_sentences(evidence_text)
    if not sentences:
        # No sentence boundary: fall back to a window centred on the
        # claim's first matching token.
        return _window_around_match(evidence_text, claim_text, max_chars)

    claim_tokens = {t for t in re.findall(r"\w+", claim_text.lower())}
    if not claim_tokens:
        return evidence_text[:max_chars]

    scored = []
    for i, s in enumerate(sentences):
        tokens = set(re.findall(r"\w+", s.lower()))
        overlap = len(claim_tokens & tokens)
        scored.append((overlap, -i, i, s))
    scored.sort(reverse=True)

    # Take the best-matching sentences, then extend each with its
    # neighbours so the NLI sees a contiguous passage.
    chosen: set[int] = set()
    budget = max_chars
    for overlap, _, i, s in scored:
        if overlap == 0 and chosen:
            break
        chunk = sentences[i]
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(sentences) and j not in chosen:
                chosen.add(j)
                budget -= len(sentences[j])
        if budget <= 0:
            break

    if not chosen:
        return _window_around_match(evidence_text, claim_text, max_chars)
    picked = [sentences[i] for i in sorted(chosen)]
    return " ".join(picked)[:max_chars]


def _window_around_match(text: str, query: str, max_chars: int) -> str:
    """A window of ``text`` centred on where ``query`` first matches."""
    if not text:
        return ""
    tokens = re.findall(r"\w+", query.lower())
    if not tokens:
        return text[:max_chars]
    needle = tokens[0]
    pos = text.lower().find(needle)
    if pos < 0:
        return text[:max_chars]
    start = max(0, pos - max_chars // 3)
    return text[start:start + max_chars]


class Step08NLIVerification(Step):
    number = 8
    name = "nli_verification"

    def run(self) -> None:
        pipeline = self.ctx.metadata.get("pipeline")
        nli = getattr(pipeline, "nli", None)
        claims = self.ctx.claims
        evidence = self.ctx.evidence

        # A premise longer than this is truncated by the NLI anyway, so it
        # is shortened to the part that is about the claim first.
        max_premise_chars = int(
            (self.ctx.cfg.step(8) or {}).get("max_premise_chars", 1800)
        )

        verdicts: list[ClaimVerdict] = []
        claim_best: dict[int, tuple[str, float]] = {}

        for ci, claim in enumerate(claims):
            best_label = "unknown"
            best_score = 0.0
            best_doc = ""
            if nli is not None and evidence:
                for ev in evidence:
                    premise = _claim_premise(
                        ev.text or "", claim.text, max_premise_chars)
                    try:
                        lbl, score = nli.predict(premise=premise, hypothesis=claim.text)
                    except Exception:
                        lbl, score = "unknown", 0.0
                    v = ClaimVerdict(claim=claim.text, evidence_doc_id=ev.doc_id, label=lbl, score=score)
                    verdicts.append(v)
                    if lbl == "supported" and score > best_score:
                        best_label, best_score, best_doc = lbl, score, ev.doc_id
                    elif lbl == "contradicted" and best_label != "supported" and score > best_score:
                        best_label, best_score, best_doc = lbl, score, ev.doc_id
            claim_best[ci] = (best_label, best_score)
            claim.verdict = best_label
            if best_label == "supported":
                claim.confidence = float(best_score)
            else:
                claim.confidence = 1.0 - float(best_score) if best_label == "contradicted" else 0.3

        # Apply evidence to claims.
        for ci, claim in enumerate(claims):
            label, _ = claim_best.get(ci, ("unknown", 0.0))
            if label in {"supported", "contradicted"} and evidence:
                # Pick the best-matching evidence doc.
                best_ev = None
                best_score = -1.0
                for ev in evidence:
                    for v in verdicts:
                        if v.claim == claim.text and v.evidence_doc_id == ev.doc_id and v.label == label and v.score > best_score:
                            best_score = v.score
                            best_ev = ev
                if best_ev is not None:
                    claim.evidence = [best_ev]
                    claim.citation = best_ev.doc_id

        self.ctx.verdicts = verdicts
        # Update claims list to reflect verdicts.
        self.ctx.claims = claims
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "claim_count": len(claims),
            "verdict_count": len(verdicts),
            "verdict_distribution": {
                "supported": sum(1 for c in claims if c.verdict == "supported"),
                "contradicted": sum(1 for c in claims if c.verdict == "contradicted"),
                "unknown": sum(1 for c in claims if c.verdict == "unknown"),
            },
        }