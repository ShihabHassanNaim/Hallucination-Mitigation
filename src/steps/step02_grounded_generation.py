"""Step 2 — Grounded Generation.

Calls Llama 3.1 (or Qwen 3 alt) with the prompt template and the
retrieved evidence (if any). Initial answer is stored in ctx.answer.
"""
from __future__ import annotations

from . import Step, StepContext


class Step02GroundedGeneration(Step):
    number = 2
    name = "grounded_generation"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(2) or {}
        backend = blk.get("backend", "llama")
        prompt = cfg.prompt

        pipeline = self.ctx.metadata.get("pipeline")
        if backend == "qwen" and getattr(pipeline, "generator_qwen", None) is not None:
            gen = pipeline.generator_qwen
        else:
            gen = getattr(pipeline, "generator", None)

        evidence = self.ctx.evidence
        ctx_text = "\n\n".join(e.text for e in evidence) if evidence else ""
        query = self.ctx.metadata.get("rewritten_query", self.ctx.query)
        user = prompt.get("user_template", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:").format(
            context=ctx_text, question=query
        )

        if gen is None:
            answer = "I don't have enough context to answer."
        else:
            answer = gen.generate(system=prompt.get("system", ""), user=user)

        self.ctx.answer = answer
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "backend": backend,
            "answer": answer,
            "evidence_count": len(evidence),
        }