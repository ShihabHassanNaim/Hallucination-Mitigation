"""Step 6 — Hybrid Evidence Retrieval.

Dense (BGE-M3 + FAISS) fused with sparse (BM25) via RRF.
"""
from __future__ import annotations

import numpy as np

from . import Step, StepContext


class Step06HybridRetrieval(Step):
    number = 6
    name = "hybrid_retrieval"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(6) or {}
        top_k = int(blk.get("top_k", 5))

        pipeline = self.ctx.metadata.get("pipeline")
        retriever = getattr(pipeline, "retriever", None)
        embedder = getattr(pipeline, "embedder", None)
        query = self.ctx.metadata.get("rewritten_query", self.ctx.query)

        evidence = []
        if retriever is None or embedder is None or getattr(retriever, "n_docs", 0) == 0:
            self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
                "top_k": top_k,
                "evidence_count": 0,
                "warning": "no index loaded",
            }
            return

        try:
            qv = embedder.encode([query])[0]
        except Exception:
            qv = np.zeros(embedder.dim, dtype=np.float32)
        evidence = retriever.search(query, qv, top_k=top_k)

        self.ctx.evidence = evidence
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "top_k": top_k,
            "evidence_count": len(evidence),
            "evidence_ids": [e.doc_id for e in evidence],
        }