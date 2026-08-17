"""Step 3 — Early Hallucination Detection.

Computes a feature vector [cosine, claim_count, intrinsic_overlap,
answer_length] and feeds it through the XGBoost (or sklearn) classifier.
"""
from __future__ import annotations

import numpy as np

from . import Step, StepContext


class Step03EarlyDetection(Step):
    number = 3
    name = "early_detection"

    def run(self) -> None:
        pipeline = self.ctx.metadata.get("pipeline")
        embedder = getattr(pipeline, "embedder", None)
        clf = getattr(pipeline, "early_classifier", None)

        answer = self.ctx.answer or ""
        query = self.ctx.metadata.get("rewritten_query", self.ctx.query)

        if embedder is not None and hasattr(embedder, "encode"):
            try:
                qv = embedder.encode([query])[0]
                av = embedder.encode([answer])[0]
                cosine = float(np.dot(qv, av) / (np.linalg.norm(qv) * np.linalg.norm(av) + 1e-12))
            except Exception:
                cosine = 0.0
        else:
            cosine = 0.0

        claim_count = max(1, len(self.ctx.claims))
        qtoks = set(query.lower().split())
        atoks = answer.lower().split()
        overlap = 0.0
        if qtoks and atoks:
            overlap = sum(1 for t in atoks if t in qtoks) / len(atoks)
        ans_len = len(atoks)

        features = [cosine, float(claim_count), overlap, float(ans_len)]
        self.ctx.feature_vector = {
            "cosine": cosine,
            "claim_count": float(claim_count),
            "intrinsic_overlap": overlap,
            "answer_length": ans_len,
        }

        prob = 0.5
        if clf is not None:
            try:
                prob = float(clf.predict_proba(features))
            except Exception:
                prob = 0.5
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "features": self.ctx.feature_vector,
            "hallucination_probability": prob,
        }