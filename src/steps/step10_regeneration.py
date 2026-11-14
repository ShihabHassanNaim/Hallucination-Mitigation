"""Step 10 — Evidence-Guided Regeneration.

Three modes:
  - stub:       no-op (caller passes intentionally unsupported answers through)
  - evidence:   replace unsupported spans with their evidence-grounded text
  - regenerate: full Llama 3.1 regeneration conditioned on verified evidence only
"""
from __future__ import annotations

from . import Step, StepContext
from ..types import Evidence


class Step10Regeneration(Step):
    number = 10
    name = "regeneration"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(10) or {}
        mode = blk.get("mode", "evidence")
        max_attempts = int(blk.get("max_attempts", 2))
        confidence = self.ctx.confidence
        claims = self.ctx.claims
        evidence = self.ctx.evidence

        bad = [c for c in claims if c.verdict in {"contradicted", "unknown"}]
        unsupported_ratio = len(bad) / max(1, len(claims))
        decision = "KEEP"
        output_text = self.ctx.answer

        pipeline = self.ctx.metadata.get("pipeline")
        gen = getattr(pipeline, "generator", None)

        # A provided answer is being *judged*, not improved. Rewriting it
        # here would mean the gate and the detector score a different answer
        # than the one the benchmark supplied. The reduction action is still
        # recorded so the HRR/CCR figures stay meaningful.
        provided = self.ctx.metadata.get("provided_answer")
        if provided is not None:
            self.ctx.final_answer = self.ctx.answer
            self.ctx.iterations = min(max_attempts, max(1, len(bad)))
            self.ctx.decision_log.append({
                "step": 10,
                "mode": "provided_passthrough",
                "unsupported_ratio": unsupported_ratio,
                "decision": "KEEP",
            })
            self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
                "step": 10,
                "mode": "provided_passthrough",
                "unsupported_ratio": unsupported_ratio,
                "decision": "KEEP",
            }
            return

        if mode == "stub" or not bad or gen is None:
            output_text = self.ctx.answer
            decision = "KEEP"
        elif mode == "evidence":
            supported = [c for c in claims if c.verdict == "supported"]
            pieces = []
            for c in supported:
                pieces.append(c.text)
            if pieces:
                output_text = " ".join(pieces)
                decision = "EDITED"
            else:
                # Use first evidence doc as the answer.
                if evidence:
                    output_text = evidence[0].text
                    decision = "EVIDENCE_REPLACE"
                else:
                    output_text = "I don't know."
                    decision = "ABSTAIN"
        elif mode == "regenerate":
            ctx_text = "\n\n".join(e.text for e in evidence)
            supported = [c for c in claims if c.verdict == "supported"]
            keep = "; ".join(c.text for c in supported)
            user = (
                "Context:\n" + ctx_text + "\n\n"
                "Original answer: " + (self.ctx.answer or "") + "\n\n"
                "Verified claims to keep: " + keep + "\n\n"
                "Rewrite the answer using ONLY the verified claims above and the context. "
                "Remove anything not supported. Output the new answer only."
            )
            try:
                output_text = gen.generate(system="You revise answers to be evidence-grounded.", user=user)
                decision = "REGENERATED"
            except Exception:
                output_text = self.ctx.answer
                decision = "KEEP"

        self.ctx.final_answer = output_text
        self.ctx.iterations = min(max_attempts, max(1, len(bad)))
        decision_record = {
            "step": 10,
            "mode": mode,
            "unsupported_ratio": unsupported_ratio,
            "decision": decision,
        }
        self.ctx.decision_log.append(decision_record)
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = decision_record