"""Step 3 — Early Hallucination Detection.

Computes a feature vector [cosine, claim_count, intrinsic_overlap,
answer_length] and feeds it through the XGBoost (or sklearn) classifier.
"""
from __future__ import annotations

import numpy as np

from . import Step, StepContext
from ..utils.claim_extractor import intrinsic_overlap


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
        # Use the canonical helper so the feature the detector is *fitted* on
        # is the same feature it is *served*. The training data is mined by
        # scripts/generate_detector_data_v2.py, which calls this function;
        # computing the overlap inline here used a different formula
        # (fraction of answer tokens present in the query), so the model was
        # being fed a feature it had never seen.
        overlap = intrinsic_overlap(answer, query)
        ans_len = len(answer.split())

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
        # Publish the probability so downstream fusion (step 9) can consume it.
        self.ctx.metadata["hallucination_probability"] = prob
        self.ctx.metadata[f"step{self.number:02d}_{self.name}"] = {
            "features": self.ctx.feature_vector,
            "hallucination_probability": prob,
        }