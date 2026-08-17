"""Step 7 — Cross-Encoder Reranking (BGE-Reranker-v2)."""
from __future__ import annotations

from . import Step, StepContext


class Step07Reranking(Step):
    number = 7
    name = "reranking"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(7) or {}
        if not blk.get("enabled", True):
            self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {"enabled": False}
            return
        top_k = int(blk.get("top_k", 5))

        pipeline = self.ctx.metadata.get("pipeline")
        reranker = getattr(pipeline, "reranker", None)
        evidence = self.ctx.evidence
        query = self.ctx.metadata.get("rewritten_query", self.ctx.query)

        if reranker is None or not evidence:
            self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {"top_k": top_k, "evidence_count": len(evidence)}
            return

        try:
            scores = reranker.rerank(query, [e.text for e in evidence])
        except Exception:
            scores = [0.0] * len(evidence)
        for e, s in zip(evidence, scores):
            e.score = float(s)
        evidence.sort(key=lambda e: -e.score)
        self.ctx.evidence = evidence[:top_k]
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "top_k": top_k,
            "evidence_count": len(self.ctx.evidence),
            "scores": [e.score for e in self.ctx.evidence],
        }