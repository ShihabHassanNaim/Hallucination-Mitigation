"""EEDC — Expected Entropy of Detection Confidence (Step 4/9).

Platt-style linear scoring over: NLI entropy, retrieval top-1,
self-consistency, bias. Returns a calibrated probability in [0, 1].
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from ..utils import log_fallback


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class EEDC:
    def __init__(
        self,
        alpha: float = -1.0,
        beta: float = -0.7,
        gamma: float = -0.5,
        delta: float = 1.0,
        weights_path: str = "data/eedc_weights.json",
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.weights_path = weights_path
        self._loaded = False

    def _maybe_load(self) -> None:
        if self._loaded:
            return
        p = Path(self.weights_path)
        if p.exists():
            try:
                w = json.loads(p.read_text(encoding="utf-8"))
                self.alpha = float(w.get("alpha", self.alpha))
                self.beta = float(w.get("beta", self.beta))
                self.gamma = float(w.get("gamma", self.gamma))
                self.delta = float(w.get("delta", self.delta))
            except Exception as exc:  # noqa: BLE001
                log_fallback("eedc_weights", repr(exc))
        self._loaded = True

    def score(
        self,
        nli_entropy: float,
        retrieval_top1: float,
        self_consistency: float,
    ) -> float:
        self._maybe_load()
        phi = (
            self.alpha * float(nli_entropy)
            + self.beta * (1.0 - float(retrieval_top1))
            + self.gamma * (1.0 - float(self_consistency))
            + self.delta
        )
        return _sigmoid(phi)


def from_config(cfg) -> EEDC:
    blk = cfg.step(4) or {}
    weights_path = blk.get("weights_path", "data/eedc_weights.json")
    return EEDC(
        alpha=-1.0,
        beta=-0.7,
        gamma=-0.5,
        delta=1.0,
        weights_path=weights_path,
    )