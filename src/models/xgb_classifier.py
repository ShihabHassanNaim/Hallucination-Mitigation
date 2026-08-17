"""XGBoost classifier for early hallucination detection (Step 3).

Feature vector: [cosine, claim_count, intrinsic_overlap, answer_length].
Trains a small xgboost model on a synthetic labelled set on first use.
"""
from __future__ import annotations

import numpy as np


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

    def _synthetic_data(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        X = rng.random((256, 4)).astype(np.float32)
        X[:, 1] *= 10     # claim_count
        X[:, 2] *= 1.0    # intrinsic_overlap
        X[:, 3] *= 512    # answer_length
        # Rule of thumb: high cosine + low intrinsic_overlap => hallucinated
        y = ((X[:, 0] > 0.5) & (X[:, 2] < 0.3)).astype(np.int32)
        return X, y

    def _train_xgb(self, Cls):
        X, y = self._synthetic_data()
        m = Cls(n_estimators=10, max_depth=3, use_label_encoder=False, eval_metric="logloss")
        m.fit(X, y)
        return m

    def _train_sklearn(self, Cls):
        X, y = self._synthetic_data()
        m = Cls(max_iter=200)
        m.fit(X, y)
        return m

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