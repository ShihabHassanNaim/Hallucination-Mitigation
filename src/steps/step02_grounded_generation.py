"""Step 2 — Grounded Generation.

Calls the generator with the prompt template and the retrieved evidence.
Initial answer is stored in ctx.answer.

Grounding note: retrieval and reranking run *before* this step (see
``src/steps/__init__.py`` — the registry orders step 6/7 ahead of step 2),
so ``ctx.evidence`` is populated here and the prompt template receives real
retrieved context. The step numbers are retained for traceability to the
architecture diagram, but the *execution* order is what matters.
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

        # A provided answer bypasses generation entirely. The pipeline is
        # then verifying a supplied answer rather than producing one, which
        # is what a hallucination-detection benchmark measures.
        provided = self.ctx.metadata.get("provided_answer")
        if provided is not None:
            answer = str(provided)
            backend = f"{backend} (provided)"
        elif gen is None:
            answer = "I don't have enough context to answer."
        else:
            answer = gen.generate(system=prompt.get("system", ""), user=user)

        self.ctx.answer = answer
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "backend": backend,
            "answer": answer,
            "evidence_count": len(evidence),
            "grounded": bool(evidence),
            "provided": provided is not None,
        }
