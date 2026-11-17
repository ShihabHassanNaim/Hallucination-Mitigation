"""Step 6 — Hybrid Evidence Retrieval.

Dense (BGE-M3 + FAISS) fused with sparse (BM25) via RRF.

Closed-book datasets
--------------------
A closed-book dataset has no legitimate retrieval context. TruthfulQA is
the case here: its corpus was built from the dataset's own ``knowledge``
field, which is the set of *correct* answers. Retrieval therefore finds
the reference answer for every row, and any verification against the
retrieved document -- the NLI in step 8, the grounding term in step 9 --
would measure "is this answer the reference answer", which is the label.

For such a dataset the retrieval step is switched off and the pipeline
runs with no evidence at all, so the verification chain sees no source and
cannot read the answer key. The switch is ``retrieval_source: none`` in
the step-6 config, overridable per run through the context metadata
(``ctx.metadata["retrieval_source"]``), which is what the evaluation
harness sets per dataset.
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

        # "retrieved" (default) or "none". A closed-book dataset sets this
        # to "none" so the verification chain has no source to read the
        # answer key from.
        retrieval_src = blk.get("retrieval_source", "retrieved")
        override = self.ctx.metadata.get("retrieval_source")
        if override is not None:
            retrieval_src = override
        if retrieval_src == "none":
            self.ctx.evidence = []
            self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
                "top_k": top_k,
                "evidence_count": 0,
                "retrieval_source": "none",
                "warning": (
                    "closed-book dataset: retrieval disabled so the "
                    "verification chain cannot read the answer key"
                ),
            }
            return

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