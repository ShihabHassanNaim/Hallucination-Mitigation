"""Step 11 — Final Verification Gate.

Re-runs Step 3 + Step 8 on the regenerated answer, then picks one of
{PASS, REGENERATE (≤2 attempts), ABSTAIN}.
"""
from __future__ import annotations

from . import Step, StepContext


class Step11FinalGate(Step):
    number = 11
    name = "final_gate"

    def run(self) -> None:
        cfg = self.ctx.cfg
        blk = cfg.step(11) or {}
        pass_th = float(blk.get("pass_threshold", 0.70))
        abs_th = float(blk.get("abstain_threshold", 0.30))
        max_attempts = int(blk.get("max_regen_attempts", 2))

        score = self.ctx.confidence
        unsupported = sum(1 for c in self.ctx.claims if c.verdict in {"contradicted", "unknown"})
        total = max(1, len(self.ctx.claims))
        unsupp_ratio = unsupported / total

        attempts = getattr(self.ctx, "_regen_attempts", 0)
        if score >= pass_th and unsupp_ratio == 0.0:
            decision = "PASS"
        elif score < abs_th or unsupp_ratio >= 0.5:
            decision = "ABSTAIN"
        else:
            # Within the gray zone.
            if attempts < max_attempts:
                self.ctx._regen_attempts = attempts + 1
                decision = "REGEN"
            else:
                decision = "PASS" if score >= 0.5 else "ABSTAIN"

        self.ctx.decision = decision
        record = {
            "step": 11,
            "confidence": score,
            "unsupported_ratio": unsupp_ratio,
            "decision": decision,
            "pass_threshold": pass_th,
            "abstain_threshold": abs_th,
        }
        self.ctx.decision_log.append(record)
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = record