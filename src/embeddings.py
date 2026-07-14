"""Embedding wrapper.

Real mode uses sentence-transformers with BGE. Mock mode returns deterministic
random unit vectors so the pipeline structure can be exercised without
downloading multi-gigabyte models.
"""
from __future__ import annotations

import hashlib
import math
from typing import List

import numpy as np


class Embedder:
    """Encode texts -> dense vectors (numpy float32, shape [n, d])."""

    def __init__(self, model_name: str, dim: int = 1024, mock: bool = False):
        self.model_name = model_name
        self.dim = dim
        self.mock = mock
        self._model = None  # lazy

    def _load(self) -> None:
        if self.mock or self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # heavy import

        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Encode a list of strings into a [n, dim] float32 array."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self.mock:
            vecs = np.stack([self._mock_embed(t, self.dim) for t in texts]).astype(np.float32)
        else:
            self._load()
            vecs = self._model.encode(  # type: ignore[union-attr]
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False,
            ).astype(np.float32)

        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            vecs = vecs / norms
        return vecs

    @staticmethod
    def _mock_embed(text: str, dim: int) -> np.ndarray:
        """Deterministic unit vector derived from a SHA-256 of the text.

        Used only in tests / offline development. Not a real semantic embed.
        The `dim` argument controls the output dimensionality so the mock
        matches whatever the caller configured.
        """
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        v /= max(math.sqrt(float(np.dot(v, v))), 1e-12)
        return v