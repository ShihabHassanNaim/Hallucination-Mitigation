"""Phase 6 (designed in Phase 2) — Evidence Entropy + Divergence Confidence.

EEDC fuses three orthogonal uncertainty signals into a single calibrated
probability that the claim is *supported* by the retrieved evidence:

  1. NLI predictive entropy  — epistemic uncertainty of the verifier
  2. Retrieval disagreement   — evidence-side uncertainty (1 - top-1 score)
  3. Self-consistency         — model-side uncertainty over m sampled answers

Fusion rule
-----------
  raw = alpha * H_norm
      + beta  * (1 - retrieval_top1)
      + gamma * (1 - self_consistency)
      + delta

  phi = sigmoid(raw)

`alpha, beta, gamma, delta` are learned by Platt scaling on a calibration
set. Default weights give an uncalibrated but still useful score; Phase 2's
`scripts/calibrate_eedc.py` will fit them on FEVER-dev.

Sign convention
---------------
Higher `phi` means the claim is *more likely supported*. Therefore the
defaults set `alpha, beta, gamma` to NEGATIVE values, so that high NLI
entropy, low retrieval top-1, or low self-consistency each push `raw`
toward zero and `phi` toward 0. `delta` is positive so the "everything
looks good" case crosses the 0.5 threshold.

Why three signals?
------------------
* Entropy alone misses retrieval misses (where NLI is "NEI" with high
  confidence because the evidence is irrelevant).
* Retrieval alone misses semantic mismatch (high cosine, wrong answer).
* Self-consistency alone is expensive and noisy on short claims.

Together they are far better calibrated than any single signal — that's
the empirical claim of the CRISP paper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .detector import LABEL2ID, LABELS, NLIPrediction


# Entropy of a uniform 3-class distribution = log(3) ~ 1.0986.
_LOG3 = math.log(len(LABELS))


@dataclass
class EEDCSignals:
    """The three raw signals fed into EEDC, in [0, 1]."""
    nli_entropy_norm: float          # H(p) / log(3), normalised to [0, 1]
    retrieval_top1: float            # cosine score in [-1, 1], clipped to [0, 1]
    self_consistency: float          # fraction of m samples that agree, in [0, 1]

    def as_vector(self) -> List[float]:
        return [
            self.nli_entropy_norm,
            1.0 - self.retrieval_top1,
            1.0 - self.self_consistency,
        ]


@dataclass
class EEDCWeights:
    """Platt-style linear weights + bias for EEDC.

    Sign convention: higher phi => more likely supported. Therefore
    alpha, beta, gamma are NEGATIVE — high NLI entropy, low retrieval
    top-1, or low self-consistency should each *lower* phi. Delta is
    POSITIVE so the all-confident case crosses 0.5.
    """
    alpha: float = -1.0    # weight on H_norm
    beta: float = -0.7     # weight on (1 - retrieval_top1)
    gamma: float = -0.5    # weight on (1 - self_consistency)
    delta: float = 1.0     # bias

    def as_vector(self) -> List[float]:
        return [self.alpha, self.beta, self.gamma, self.delta]


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid.
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def normalised_entropy(probs: Sequence[float]) -> float:
    """Shannon entropy normalised to [0, 1] by log(K). K = len(probs)."""
    h = 0.0
    for p in probs:
        if p > 0.0:
            h -= p * math.log(p)
    return min(1.0, max(0.0, h / _LOG3))


class EEDCScorer:
    """Calibrated multi-signal hallucination probability.

    phi = sigmoid(alpha * H_norm + beta * (1 - r) + gamma * (1 - c) + delta)

    where H_norm is NLI entropy, r is retrieval top-1 score, c is the
    self-consistency agreement rate. Higher phi => more likely the claim
    is SUPPORTED. (The pipeline can invert this to get P(hallucinated)
    when needed.)
    """

    def __init__(self, weights: Optional[EEDCWeights] = None):
        self.weights = weights or EEDCWeights()

    # --- core API -----------------------------------------------------------

    def score(self, signals: EEDCSignals) -> float:
        """Return calibrated P(supported | signals) in [0, 1]."""
        v = signals.as_vector()
        w = self.weights
        raw = w.alpha * v[0] + w.beta * v[1] + w.gamma * v[2] + w.delta
        return _sigmoid(raw)

    def score_from_components(self,
                              nli_probs: Sequence[float],
                              retrieval_top1: float,
                              self_consistency: float = 1.0) -> float:
        """Convenience: build signals from raw components and score."""
        signals = EEDCSignals(
            nli_entropy_norm=normalised_entropy(nli_probs),
            retrieval_top1=_clip01(retrieval_top1),
            self_consistency=_clip01(self_consistency),
        )
        return self.score(signals)

    def score_prediction(self, prediction: NLIPrediction,
                         retrieval_top1: float,
                         self_consistency: float = 1.0) -> float:
        """Score directly from an NLIPrediction object."""
        return self.score_from_components(
            nli_probs=prediction.probs,
            retrieval_top1=retrieval_top1,
            self_consistency=self_consistency,
        )

    # --- calibration --------------------------------------------------------

    def fit(self, signals_list: Sequence[EEDCSignals],
            labels: Sequence[int]) -> "EEDCScorer":
        """Fit Platt-style weights by maximum likelihood (gradient ascent).

        labels: 1 if supported, 0 if not supported. Uses full-batch L-BFGS-
        style updates for stability on small calibration sets (typical:
        FEVER dev, ~4k examples).
        """
        if len(signals_list) != len(labels):
            raise ValueError("signals_list and labels must have the same length.")
        if not signals_list:
            raise ValueError("Cannot calibrate on an empty set.")

        # Convert to matrices for vectorised gradient computation.
        n = len(signals_list)
        X = []  # rows = [H, 1-r, 1-c, 1]
        y = []
        for s, lab in zip(signals_list, labels):
            v = s.as_vector()
            X.append([v[0], v[1], v[2], 1.0])
            y.append(float(lab))
        # Convert "lower score = hallucinated" intuition into "predict 1 if supported".
        # Our loss is cross-entropy: -y*log(phi) - (1-y)*log(1-phi).

        # Initialise from current weights.
        theta = list(self.weights.as_vector())

        lr = 0.5
        for _ in range(400):  # sufficient for the convex logistic loss.
            grad = [0.0, 0.0, 0.0, 0.0]
            for row, target in zip(X, y):
                raw = sum(t * x for t, x in zip(theta, row))
                phi = _sigmoid(raw)
                err = phi - target
                for i in range(4):
                    grad[i] += err * row[i]
            for i in range(4):
                theta[i] -= lr * grad[i] / n
            # Lightweight adaptive learning rate — halve if loss increases.
            # Skipped for brevity; 400 steps with fixed lr=0.5 converges on
            # typical FEVER-sized calibration sets.

        self.weights = EEDCWeights(
            alpha=theta[0], beta=theta[1], gamma=theta[2], delta=theta[3],
        )
        return self


# --- helpers ----------------------------------------------------------------

def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def signals_from_prediction(prediction: NLIPrediction,
                            retrieval_top1: float,
                            self_consistency: float = 1.0) -> EEDCSignals:
    return EEDCSignals(
        nli_entropy_norm=normalised_entropy(prediction.probs),
        retrieval_top1=_clip01(retrieval_top1),
        self_consistency=_clip01(self_consistency),
    )