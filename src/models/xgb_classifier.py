"""XGBoost classifier for early hallucination detection (Step 3).

Feature vector: [cosine, claim_count, intrinsic_overlap, answer_length].

Training data priority:
  1. ``data/detector_train_clean.jsonl`` — real labelled features mined from
     HaluEval by ``scripts/generate_detector_data_v4.py``. Both classes are
     length-matched and the queries are disjoint from the evaluation splits,
     so neither ``answer_length`` nor ``intrinsic_overlap`` can act as a
     label shortcut.
  2. ``data/detector_train.jsonl`` — the older training file. It is
     length-confounded (faithful answers median 3 words, hallucinated
     median 9) and contains a duplicated gold row; it is used only when
     the clean file is absent.
  3. Synthetic random fallback (only if neither real file is present).

The real-data path is what lifts the detector out of the random baseline
(ROC-AUC ~0.5) and into the publishable range (0.75+).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_CLEAN_DATA_PATH = Path("data/detector_train_clean.jsonl")
_REAL_DATA_PATH = Path("data/detector_train.jsonl")


class XGBClassifier:
    """Binary hallucination classifier with predict_proba(X)."""

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self._clf = None

    def _load(self) -> None:
        if self.model_name == "sklearn-logreg":
            from sklearn.linear_model import LogisticRegression  # type: ignore
            self._clf = self._train_sklearn(LogisticRegression)
            return
        try:
            import xgboost as xgb  # type: ignore
            self._clf = self._train_xgb(xgb.XGBClassifier)
            return
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"xgboost unavailable ({exc!r}); install xgboost or set classifier=sklearn-logreg"
            )

    def _ensure_loaded(self) -> None:
        if self._clf is None:
            self._load()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _real_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Load real labelled training rows from disk.

        Prefers the clean, length-matched training file; falls back to the
        older length-confounded one. Returns ``None`` if neither is
        available so the caller can fall back to synthetic data.
        """
        for path in (_CLEAN_DATA_PATH, _REAL_DATA_PATH):
            if not path.exists():
                continue
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except Exception:  # noqa: BLE001
                continue
            if not rows:
                continue
            X = np.array([r["features"] for r in rows], dtype=np.float32)
            y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
            if X.ndim != 2 or X.shape[0] < 16:
                continue
            return X, y
        return None

    def _synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic fallback used only when no real data is on disk."""
        rng = np.random.default_rng(0)
        X = rng.random((256, 4)).astype(np.float32)
        X[:, 1] *= 10     # claim_count
        X[:, 2] *= 1.0    # intrinsic_overlap
        X[:, 3] *= 512    # answer_length
        # Rule of thumb: high cosine + low intrinsic_overlap => hallucinated
        y = ((X[:, 0] > 0.5) & (X[:, 2] < 0.3)).astype(np.int32)
        return X, y

    def _load_data(self) -> tuple[np.ndarray, np.ndarray, str]:
        """Pick the best available training data, return (X, y, source_label)."""
        real = self._real_data()
        if real is not None:
            return real[0], real[1], "real"
        X, y = self._synthetic_data()
        return X, y, "synthetic"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _train_xgb(self, Cls):
        X, y, _ = self._load_data()
        m = Cls(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        m.fit(X, y)
        return m

    def _train_sklearn(self, Cls):
        X, y, _ = self._load_data()
        m = Cls(max_iter=400)
        m.fit(X, y)
        return m

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, features: list[float]) -> float:
        self._ensure_loaded()
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        proba = self._clf.predict_proba(x)  # type: ignore[union-attr]
        if proba.shape[1] == 2:
            return float(proba[0, 1])
        return float(proba[0, 0])


def from_config(cfg) -> XGBClassifier:
    name = (cfg.models or {}).get("classifier", "xgboost")
    clf = XGBClassifier(model_name=name)
    clf._load()
    return clf
