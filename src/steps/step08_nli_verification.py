"""Step 8 — NLI Fact Verification (DeBERTa-v3-MNLI).

For every claim, score against each retrieved evidence and pick the best.
"""
from __future__ import annotations

from . import Step, StepContext
from ..types import ClaimVerdict


class Step08NLIVerification(Step):
    number = 8
    name = "nli_verification"

    def run(self) -> None:
        pipeline = self.ctx.metadata.get("pipeline")
        nli = getattr(pipeline, "nli", None)
        claims = self.ctx.claims
        evidence = self.ctx.evidence

        verdicts: list[ClaimVerdict] = []
        claim_best: dict[int, tuple[str, float]] = {}

        for ci, claim in enumerate(claims):
            best_label = "unknown"
            best_score = 0.0
            best_doc = ""
            if nli is not None and evidence:
                for ev in evidence:
                    try:
                        lbl, score = nli.predict(premise=ev.text, hypothesis=claim.text)
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