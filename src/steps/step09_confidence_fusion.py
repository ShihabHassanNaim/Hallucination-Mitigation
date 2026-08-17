"""Step 9 — Confidence Fusion.

Combines similarity, retrieval, NLI, coverage, and EEDC into a final
trust score per claim and overall.
"""
from __future__ import annotations

from . import Step, StepContext


class Step09ConfidenceFusion(Step):
    number = 9
    name = "confidence_fusion"

    def run(self) -> None:
        cfg = self.ctx.cfg
        weights = (cfg.step(9) or {}).get("fusion", cfg.raw.get("step09_fusion_weights", {}))
        w_sim = float(weights.get("similarity", 0.30))
        w_ret = float(weights.get("retrieval", 0.20))
        w_nli = float(weights.get("nli", 0.25))
        w_cov = float(weights.get("coverage", 0.15))
        w_eedc = float(weights.get("eedc", 0.10))
        # Normalise.
        s = max(1e-9, w_sim + w_ret + w_nli + w_cov + w_eedc)
        w_sim, w_ret, w_nli, w_cov, w_eedc = (w_sim / s, w_ret / s, w_nli / s, w_cov / s, w_eedc / s)

        claims = self.ctx.claims
        evidence = self.ctx.evidence
        feat = self.ctx.feature_vector or {}
        cosine = feat.get("cosine", 0.0)
        if claims:
            nli_scores = [c.confidence if c.verdict == "supported" else 0.0 for c in claims]
            avg_nli = sum(nli_scores) / len(nli_scores)
        else:
            avg_nli = 0.0
        ret_top1 = evidence[0].score if evidence else 0.0
        coverage = (sum(1 for c in claims if c.verdict == "supported") / len(claims)) if claims else 0.0
        eedc = self.ctx.eedc_score or 0.0

        overall = w_sim * cosine + w_ret * ret_top1 + w_nli * avg_nli + w_cov * coverage + w_eedc * eedc

        # Per-claim scores:
        for c in claims:
            cs = w_nli * (c.confidence if c.verdict == "supported" else 0.0)
            cs += w_ret * (c.evidence[0].score if c.evidence else 0.0)
            cs += w_cov * (1.0 if c.verdict == "supported" else 0.0)
            cs += w_sim * cosine
            cs += w_eedc * eedc
            c.confidence = float(cs)

        self.ctx.confidence = float(overall)
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "weights": {
                "similarity": w_sim, "retrieval": w_ret, "nli": w_nli, "coverage": w_cov, "eedc": w_eedc,
            },
            "components": {
                "similarity": cosine, "retrieval_top1": ret_top1,
                "nli": avg_nli, "coverage": coverage, "eedc": eedc,
            },
            "overall_confidence": overall,
            "per_claim_confidence": [c.confidence for c in claims],
        }