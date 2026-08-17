"""Step 1 — Query Understanding.

Intent classification + NER (GLiNER + spaCy) + LLM-based query rewrite.
"""
from __future__ import annotations

import re
from . import Step, StepContext
from ..types import Claim
from ..utils.claim_extractor import intrinsic_overlap


class Step01QueryUnderstanding(Step):
    number = 1
    name = "query_understanding"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(1) or {}
        labels = blk.get("intent_labels", ["factual", "opinion", "multi-hop", "math", "other"])

        pipeline = self.ctx.metadata.get("pipeline")
        spacy_ner = getattr(pipeline, "spacy_ner", None) if pipeline else None
        gliner = getattr(pipeline, "gliner_ner", None) if pipeline else None
        llm = getattr(pipeline, "generator", None) if pipeline else None

        q = self.ctx.query

        # ---- intent ----
        intent = self._classify_intent(q, labels)
        # ---- NER ----
        entities: list[dict] = []
        if spacy_ner is not None:
            entities.extend(spacy_ner.extract(q))
        if gliner is not None:
            entities.extend(gliner.extract(q, labels=labels))
        # ---- rewrite ----
        variants = [q]
        rewrite_method = "identity"
        if llm is not None:
            try:
                rewritten = llm.generate(
                    system="You rewrite search queries to be more precise.",
                    user=f"Original: {q}\n\nReturn ONE rewrite, no commentary.",
                    max_new_tokens=64,
                )
                if rewritten and rewritten.strip() and rewritten.strip() != q:
                    variants.append(rewritten.strip())
                    rewrite_method = "llm"
            except Exception:
                pass
        # Heuristic fallback: prepend "What is" or remove trailing punctuation.
        if rewrite_method == "identity":
            r = re.sub(r"[?\.\!]+$", "", q).strip()
            if r and r != q:
                variants.append(r)
                rewrite_method = "heuristic"

        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "intent": intent,
            "entities": entities,
            "query_variants": variants,
            "rewrite_method": rewrite_method,
        }
        # Persist rewritten query for retrieval.
        self.ctx.metadata["rewritten_query"] = variants[0] if variants else q

    def _classify_intent(self, q: str, labels: list[str]) -> str:
        ql = q.lower()
        if any(t in ql for t in ["+", "-", "*", "/", "equals", "compute"]):
            return "math"
        if " and " in ql or " of " in ql and ql.count("?") > 1:
            return "multi-hop"
        if ql.startswith(("do you think", "is it good", "should")):
            return "opinion"
        if "?" in ql or any(ql.startswith(w) for w in ["who", "what", "when", "where", "why", "how"]):
            return "factual"
        return labels[-1] if labels else "other"