"""Step 12 — Explainable Final Response.

Renders the verified answer, per-claim verdicts, retrieved evidence,
confidence score, and decision history as Markdown + JSON.
"""
from __future__ import annotations

import json

from . import Step, StepContext


class Step12ExplainableResponse(Step):
    number = 12
    name = "explainable_response"

    def run(self) -> None:
        result = {
            "query": self.ctx.query,
            "answer": self.ctx.final_answer or self.ctx.answer,
            "confidence": round(self.ctx.confidence, 4),
            "decision": self.ctx.decision,
            "claims": [
                {
                    "text": c.text,
                    "verdict": c.verdict,
                    "confidence": round(c.confidence, 3),
                    "provenance": c.provenance,
                    "citation": c.citation,
                    "evidence": [e.to_dict() for e in c.evidence],
                }
                for c in self.ctx.claims
            ],
            "evidence": [e.to_dict() for e in self.ctx.evidence],
            "decision_history": self.ctx.decision_log,
        }
        json_blob = json.dumps(result, indent=2)

        # Markdown rendering.
        lines = [f"# Hallucination Pipeline Report", ""]
        lines.append(f"**Query:** {self.ctx.query}")
        lines.append(f"**Answer:** {result['answer']}")
        lines.append(f"**Confidence:** {result['confidence']:.2f}  ")
        lines.append(f"**Decision:** `{self.ctx.decision}`")
        lines.append("")
        lines.append("## Claims")
        for i, c in enumerate(self.ctx.claims, 1):
            lines.append(
                f"{i}. *({c.verdict}, conf={c.confidence:.2f})* {c.text}  \n"
                f"   - provenance: `{c.provenance}`  \n"
                f"   - citation: `{c.citation or '-'}`"
            )
        lines.append("")
        lines.append("## Evidence")
        for e in self.ctx.evidence:
            lines.append(f"- **{e.doc_id}** (score={e.score:.3f}): {e.text}")
        lines.append("")
        lines.append("## Decision history")
        for ev in self.ctx.decision_log:
            lines.append(f"- step {ev.get('step')}: {ev}")

        md = "\n".join(lines)
        self.ctx.metadata["report_markdown"] = md
        self.ctx.metadata["report_json"] = json_blob
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "markdown_bytes": len(md),
            "json_bytes": len(json_blob),
        }