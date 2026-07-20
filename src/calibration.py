"""Phase 6 — Confidence calibration on top of EEDC.

EEDC (``src.eedc``) produces a raw Platt-style sigmoid of (NLI entropy,
1 - retrieval top-1, 1 - self-consistency). On its own this is *monotonic*
but not calibrated: a phi of 0.7 means "more likely supported than 0.4" but
not necessarily "70 % supported". Phase 6 wraps the raw EEDC score with one
of two standard post-hoc calibrators:

  * ``TemperatureScaler`` — single-parameter T>0; monotonically rescales
    the logits so the sigmoid becomes well-calibrated without changing the
    ranking. Numpy-free and ~10 lines so it works on a laptop.
  * ``IsotonicCalibrator`` — non-parametric piecewise-constant isotonic
    regression. More flexible than temperature scaling (handles distortion)
    but needs more data (>>1k examples) and can overfit on small sets.

Both calibrators expose the same API:

    .fit(scores, labels)   —- scores in [0,1], labels {0,1}
    .transform(scores)     —- returns calibrated p in [0,1]
    .to_dict() / from_dict() —- for persistence to JSON

The :class:`CalibratedEEDC` glue composes an :class:`EEDCScorer` with one of
the calibrators so downstream code (``Pipeline._detect_and_score``) can swap
``eedc_scorer.score(signals)`` -> ``calibrated.score(signals)`` without any
call-site changes.

Calibration data format (matches ``scripts/calibrate_eedc.py``)
--------------------------------------------------------------
JSONL records:
    {"nli_probs": [sup, con, nei],
     "retrieval_top1": float in [-1, 1],
     "self_consistency": float in [0, 1],
     "label": 1 if supported, 0 if not}

The script ``scripts/calibrate_eedc.py`` is extended to optionally fit a
temperature or isotonic post-calibrator and write both the linear EEDC
weights AND the calibrator parameters into ``data/eedc_weights.json``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

from .eedc import EEDCScorer, EEDCSignals, EEDCWeights


# ---------------------------------------------------------------------------
# Calibrator protocol
# ---------------------------------------------------------------------------


class _BaseCalibrator:
    """Minimal interface every calibrator implements."""

    name = "base"

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "_BaseCalibrator":
        raise NotImplementedError

    def transform(self, scores: Sequence[float]) -> List[float]:
        raise NotImplementedError

    # ----- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.name}

    @classmethod
    def from_dict(cls, payload: dict) -> "_BaseCalibrator":
        kind = payload.get("kind") or payload.get("name")
        if kind == "temperature":
            return TemperatureScaler.from_dict(payload)
        if kind == "isotonic":
            return IsotonicCalibrator.from_dict(payload)
        if kind in (None, "none", "identity"):
            return IdentityCalibrator()
        raise ValueError(f"Unknown calibrator kind: {kind!r}")


class IdentityCalibrator(_BaseCalibrator):
    """No-op calibrator. Useful as a default and for tests."""

    name = "identity"

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IdentityCalibrator":
        return self

    def transform(self, scores: Sequence[float]) -> List[float]:
        return [float(s) for s in scores]

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.name}


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))


class TemperatureScaler(_BaseCalibrator):
    """Single-parameter post-hoc calibrator.

    Given a temperature T > 0, transform::

        p_cal = sigmoid(logit(p_raw) / T)

    When T == 1 the calibrator is the identity. T > 1 softens the
    sigmoid (helpful when raw scores are over-confident), T < 1 sharpens
    it (raw scores under-confident).
    """

    name = "temperature"

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "TemperatureScaler":
        """Fit T by maximising binary log-likelihood on ``scores``/``labels``.

        We do a 1-D Brent-style line search over a coarse grid plus local
        refinement. Returns the calibrator so this method is chainable.
        """
        if len(scores) != len(labels):
            raise ValueError("scores and labels must have the same length.")
        if not scores:
            raise ValueError("Cannot calibrate on an empty set.")

        z = [_logit(float(s)) for s in scores]
        y = [float(lab) for lab in labels]

        def neg_log_likelihood(t: float) -> float:
            nll = 0.0
            for zi, yi in zip(z, y):
                p = _sigmoid(zi / t)
                p = min(1.0 - 1e-12, max(1e-12, p))
                nll -= yi * math.log(p) + (1.0 - yi) * math.log(1.0 - p)
            return nll

        # Coarse grid over [0.05, 5.0].
        grid = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0]
        best_t = min(grid, key=neg_log_likelihood)

        # Local ternary refinement (~12 rounds — sufficient for unimodal).
        lo, hi = 0.05, 5.0
        for _ in range(40):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            if neg_log_likelihood(m1) < neg_log_likelihood(m2):
                hi = m2
            else:
                lo = m1
            if abs(hi - lo) < 1e-4:
                break
        refined_t = 0.5 * (lo + hi)
        self.temperature = refined_t if neg_log_likelihood(refined_t) < neg_log_likelihood(best_t) else best_t
        return self

    def transform(self, scores: Sequence[float]) -> List[float]:
        t = self.temperature
        if t == 1.0:
            return [float(s) for s in scores]
        return [_sigmoid(_logit(float(s)) / t) for s in scores]

    def to_dict(self) -> dict:
        return {"name": self.name, "kind": self.name, "temperature": self.temperature}

    @classmethod
    def from_dict(cls, payload: dict) -> "TemperatureScaler":
        return cls(temperature=float(payload.get("temperature", 1.0)))


# ---------------------------------------------------------------------------
# Isotonic regression (pool-adjacent-violators)
# ---------------------------------------------------------------------------


def _fit_isotonic(xs: Sequence[float], ys: Sequence[float]) -> List[Tuple[float, float]]:
    """Pool-adjacent-violators algorithm (PAV) for isotonic regression.

    Returns a sorted list of (xs_i, ys_i) break points where ``ys_i`` is the
    fitted piecewise-constant value for inputs up to ``xs_i``.
    """
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if not xs:
        return []

    pairs = sorted(zip(map(float, xs), map(float, ys)), key=lambda t: t[0])
    # Each block stores (mean_x_of_block, mean_y_of_block, weight).
    blocks: List[Tuple[float, float, int]] = []
    for x, y in pairs:
        blocks.append([x, y, 1])
        while len(blocks) >= 2:
            (mx1, my1, w1), (mx2, my2, w2) = blocks[-2], blocks[-1]
            if my1 > my2:  # non-monotone -> pool the last two blocks.
                w = w1 + w2
                mx = (mx1 * w1 + mx2 * w2) / w
                my = (my1 * w1 + my2 * w2) / w
                blocks[-2] = [mx, my, w]
                blocks.pop()
            else:
                break
    return [(b[0], b[1]) for b in blocks]


class IsotonicCalibrator(_BaseCalibrator):
    """Non-parametric isotonic regression calibrator.

    Fitted PAV blocks map raw scores -> calibrated labels. For an input
    ``p`` we binary-search the breakpoints by mean-x and return the
    mean-y of the enclosing block (clamped to [0, 1]).
    """

    name = "isotonic"

    def __init__(self):
        self._breakpoints: List[Tuple[float, float]] = []

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        labels_f = [float(lab) for lab in labels]
        self._breakpoints = _fit_isotonic(list(scores), labels_f)
        return self

    def transform(self, scores: Sequence[float]) -> List[float]:
        if not self._breakpoints:
            return [float(s) for s in scores]
        out: List[float] = []
        for raw in scores:
            p = float(raw)
            # Binary search the largest mean_x <= p.
            lo, hi = 0, len(self._breakpoints) - 1
            best = self._breakpoints[0][1]  # default to first block
            while lo <= hi:
                mid = (lo + hi) // 2
                mx, my = self._breakpoints[mid]
                if mx <= p:
                    best = my
                    lo = mid + 1
                else:
                    hi = mid - 1
            out.append(min(1.0, max(0.0, best)))
        return out

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.name,
            "breakpoints": [[mx, my] for mx, my in self._breakpoints],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "IsotonicCalibrator":
        cal = cls()
        cal._breakpoints = [
            (float(mx), float(my))
            for mx, my in payload.get("breakpoints", [])
        ]
        return cal


# ---------------------------------------------------------------------------
# Glue: EEDCScorer + calibrator
# ---------------------------------------------------------------------------


@dataclass
class CalibrationMetrics:
    """Hold calibration-quality metrics for reporting."""

    method: str                    # "identity", "temperature", "isotonic"
    n: int
    ece: float                     # Expected Calibration Error (lower is better)
    brier: float                   # Brier score (lower is better)
    log_loss: float                # Binary log-loss (lower is better)
    accuracy: float                # Accuracy at threshold 0.5

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "n": self.n,
            "ece": round(self.ece, 6),
            "brier": round(self.brier, 6),
            "log_loss": round(self.log_loss, 6),
            "accuracy": round(self.accuracy, 6),
        }


def expected_calibration_error(scores: Sequence[float],
                               labels: Sequence[int],
                               n_bins: int = 10) -> float:
    """Standard ECE: weighted mean of |bin_acc - bin_conf| across bins."""
    if not scores:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for s, lab in zip(scores, labels):
        idx = min(n_bins - 1, max(0, int(float(s) * n_bins)))
        bins[idx].append((float(s), int(lab)))
    n = len(scores)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        conf = sum(s for s, _ in bucket) / len(bucket)
        acc = sum(lab for _, lab in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf)
    return ece


def brier_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.0
    return sum((float(s) - int(lab)) ** 2 for s, lab in zip(scores, labels)) / len(scores)


def log_loss(scores: Sequence[float], labels: Sequence[int]) -> float:
    if not scores:
        return 0.0
    eps = 1e-12
    total = 0.0
    for s, lab in zip(scores, labels):
        p = min(1.0 - eps, max(eps, float(s)))
        y = int(lab)
        total -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
    return total / len(scores)


def accuracy_at(scores: Sequence[float], labels: Sequence[int],
                threshold: float = 0.5) -> float:
    if not scores:
        return 0.0
    correct = sum(1 for s, lab in zip(scores, labels) if (float(s) >= threshold) == bool(int(lab)))
    return correct / len(scores)


def evaluate_calibrator(calibrator: _BaseCalibrator,
                        raw_scores: Sequence[float],
                        labels: Sequence[int]) -> CalibrationMetrics:
    """Score a calibrator with ECE + Brier + log-loss + accuracy."""
    calibrated = calibrator.transform(raw_scores)
    return CalibrationMetrics(
        method=calibrator.name,
        n=len(raw_scores),
        ece=expected_calibration_error(calibrated, labels),
        brier=brier_score(calibrated, labels),
        log_loss=log_loss(calibrated, labels),
        accuracy=accuracy_at(calibrated, labels),
    )


class CalibratedEEDC:
    """Compose :class:`EEDCScorer` with an optional post-hoc calibrator.

    Use this instead of :class:`EEDCScorer` whenever you want the
    pipeline-level ``phi`` to mean *calibrated probability of support*
    rather than monotonic Platt log-odds.

    Parameters
    ----------
    scorer : EEDCScorer
        Underlying Platt-style scorer.
    calibrator : _BaseCalibrator | None
        Optional post-hoc calibrator; ``IdentityCalibrator()`` if omitted.
    """

    def __init__(self, scorer: EEDCScorer,
                 calibrator: Optional[_BaseCalibrator] = None):
        self.scorer = scorer
        self.calibrator: _BaseCalibrator = calibrator or IdentityCalibrator()

    # ----- API mirroring EEDCScorer so Pipeline can swap it in transparently

    def score(self, signals: EEDCSignals) -> float:
        raw = self.scorer.score(signals)
        return float(self.calibrator.transform([raw])[0])

    def score_from_components(self,
                              nli_probs: Sequence[float],
                              retrieval_top1: float,
                              self_consistency: float = 1.0) -> float:
        raw = self.scorer.score_from_components(
            nli_probs=nli_probs,
            retrieval_top1=retrieval_top1,
            self_consistency=self_consistency,
        )
        return float(self.calibrator.transform([raw])[0])

    def score_prediction(self, prediction, retrieval_top1: float,
                         self_consistency: float = 1.0) -> float:
        raw = self.scorer.score_prediction(
            prediction=prediction,
            retrieval_top1=retrieval_top1,
            self_consistency=self_consistency,
        )
        return float(self.calibrator.transform([raw])[0])

    # ----- calibration utilities --------------------------------------------

    def raw_score(self, signals: EEDCSignals) -> float:
        """Return the *uncalibrated* phi (for diagnostics + reports)."""
        return float(self.scorer.score(signals))

    def fit_calibrator(self,
                       signals_list: Sequence[EEDCSignals],
                       labels: Sequence[int],
                       method: str = "temperature") -> CalibrationMetrics:
        """Fit a new calibrator and replace the current one.

        ``signals_list`` are the same :class:`EEDCSignals` you'd use to score
        new data, so the fit sees the *raw* phi distribution that this
        scorer emits at inference time.
        """
        raw = [self.scorer.score(s) for s in signals_list]
        if method == "none":
            cal = IdentityCalibrator()
        elif method == "temperature":
            cal = TemperatureScaler().fit(raw, labels)
        elif method == "isotonic":
            cal = IsotonicCalibrator().fit(raw, labels)
        else:
            raise ValueError(f"Unknown calibration method: {method!r}")
        self.calibrator = cal
        return evaluate_calibrator(cal, raw, labels)

    # ----- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "weights": self.scorer.weights.as_vector(),
            "calibrator": self.calibrator.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CalibratedEEDC":
        weights = EEDCWeights(*payload["weights"])
        scorer = EEDCScorer(weights=weights)
        cal = _BaseCalibrator.from_dict(payload.get("calibrator", {}))
        return cls(scorer=scorer, calibrator=cal)

    def save(self, path) -> None:
        import json
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "CalibratedEEDC":
        import json
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
