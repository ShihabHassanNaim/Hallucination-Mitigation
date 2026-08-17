"""Step 5 — Atomic Claim Extraction.

Splits the answer into atomic factual claims. Tags provenance via
intrinsic_overlap with the query.
"""
from __future__ import annotations

from . import Step, StepContext
from ..types import Claim
from ..utils.claim_extractor import extract_claims, intrinsic_overlap


class Step05ClaimExtraction(Step):
    number = 5
    name = "claim_extraction"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(5) or {}
        mode = blk.get("mode", "rule")
        intr_th = float(blk.get("intrinsic_threshold", 0.7))
        agg_th = float(blk.get("aggregated_threshold", 0.3))
        answer = self.ctx.answer or ""

        pipeline = self.ctx.metadata.get("pipeline")
        llm_generate = getattr(pipeline.generator, "generate", None) if getattr(pipeline, "generator", None) else None

        claims_text = extract_claims(answer, mode=mode, llm_generate=llm_generate)
        query = self.ctx.metadata.get("rewritten_query", self.ctx.query)
        claims: list[Claim] = []
        for txt in claims_text:
            ov = intrinsic_overlap(txt, query)
            if ov >= intr_th:
                prov = "intrinsic"
            elif ov <= agg_th:
                prov = "extrinsic"
            else:
                prov = "aggregated"
            claims.append(Claim(text=txt, provenance=prov, intrinsic_overlap=ov))

        self.ctx.claims = claims
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "mode": mode,
            "claim_count": len(claims),
            "claims": [c.text for c in claims],
            "provenances": [c.provenance for c in claims],
        }