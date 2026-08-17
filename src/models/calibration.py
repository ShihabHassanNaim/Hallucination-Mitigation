"""Temperature scaling + isotonic calibration for EEDC scores."""
from __future__ import annotations

import numpy as np


class TemperatureScaler:
    """Single-parameter temperature scaling.

    Fits T on (scores, labels) via NLL minimisation (closed-form on a
    1D grid is good enough; we use simple grid search).
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        # Convert scores to logits first via logit, then search T.
        eps = 1e-6
        s = np.clip(scores, eps, 1.0 - eps)
        logits = np.log(s / (1.0 - s))
        best_t, best_nll = 1.0, float("inf")
        for t in np.linspace(0.25, 4.0, 32):
            scaled = logits / t
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, eps, 1.0 - eps)
            nll = -(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)).mean()
            if nll < best_nll:
                best_nll = nll
                best_t = t
        self.temperature = float(best_t)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        eps = 1e-6
        s = np.clip(scores, eps, 1.0 - eps)
        logits = np.log(s / (1.0 - s)) / self.temperature
        return 1.0 / (1.0 + np.exp(-logits))


class IsotonicCalibrator:
    def __init__(self):
        self._xs: np.ndarray = np.array([0.0])
        self._ys: np.ndarray = np.array([0.0])

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels, dtype=np.float64).reshape(-1)
        order = np.argsort(scores)
        self._xs = scores[order]
        self._ys = labels[order]
        # Pool-Adjacent-Violators smoothing.
        for i in range(1, len(self._ys)):
            if self._ys[i] < self._ys[i - 1]:
                # Average neighbours to enforce monotonicity.
                j = i
                while j > 0 and self._ys[j] < self._ys[j - 1]:
                    self._ys[j - 1] = (self._ys[j - 1] + self._ys[j]) / 2.0
                    j -= 1
                self._ys[i] = max(self._ys[i], self._ys[i - 1])
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.interp(scores, self._xs, self._ys)


def from_config(cfg) -> object:
    """Pick a calibrator by cfg.step(4).method."""
    blk = cfg.step(4) or {}
    method = blk.get("method", "temperature")
    if method == "isotonic":
        return IsotonicCalibrator()
    return TemperatureScaler()