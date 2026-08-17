"""Step 4 — Confidence Assessment (EEDC + temperature scaling)."""
from __future__ import annotations

import math

import numpy as np

from . import Step, StepContext


class Step04ConfidenceAssessment(Step):
    number = 4
    name = "confidence_assessment"

    def run(self) -> None:
        pipeline = self.ctx.metadata.get("pipeline")
        eedc = getattr(pipeline, "eedc", None)
        calibrator = getattr(pipeline, "calibrator", None)

        # Build the 3 EEDC inputs from earlier steps + ctx.
        feat = self.ctx.feature_vector or {}
        cosine = feat.get("cosine", 0.0)
        # NLI entropy proxy: use variance of verdicts if available, else from features.
        verdicts = self.ctx.verdicts or []
        if verdicts:
            labels = [v.label for v in verdicts if hasattr(v, "label")]
            # entropy over {supported, contradicted, unknown}
            counts = {"supported": 0, "contradicted": 0, "unknown": 0}
            for lbl in labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            total = sum(counts.values()) or 1
            H = 0.0
            for c in counts.values():
                p = c / total
                if p > 0:
                    H -= p * math.log(p)
            H = H / math.log(3)  # normalise to [0,1]
        else:
            H = 1.0 - cosine  # pre-NLI proxy
        self_consistency = cosine  # use cosine as a proxy until multi-sample

        raw = 0.5
        if eedc is not None:
            try:
                raw = eedc.score(nli_entropy=H, retrieval_top1=cosine, self_consistency=self_consistency)
            except Exception:
                raw = 0.5

        calibrated = raw
        if calibrator is not None and hasattr(calibrator, "transform"):
            try:
                calibrated = float(np.asarray(calibrator.transform(np.array([raw])))[0])
            except Exception:
                calibrated = raw

        self.ctx.eedc_score = float(calibrated)
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "eedc_raw": raw,
            "eedc_calibrated": self.ctx.eedc_score,
            "nli_entropy": H,
            "self_consistency": self_consistency,
        }