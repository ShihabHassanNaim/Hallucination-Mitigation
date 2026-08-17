"""BGE-M3 dense embedder.

Loads the BAAI/bge-m3 sentence-transformers checkpoint. Raises on failure.
"""
from __future__ import annotations

import numpy as np


class BGE_M3Embedder:
    """Dense embedder. ``encode(texts) -> ndarray[float32]`` shape (n, dim)."""

    def __init__(self, dim: int = 1024, model_name: str = "BAAI/bge-m3"):
        self.dim = dim
        self.model_name = model_name
        self._model = None

    def _load_hf(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"sentence-transformers not available: {exc}")
        m = SentenceTransformer(self.model_name, trust_remote_code=True)
        # bge-m3 outputs 1024-d vectors; if smaller model, override dim.
        test_vec = m.encode(["ping"], normalize_embeddings=True)
        self.dim = int(test_vec.shape[-1])
        self._model = m

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_hf()

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        self._ensure_loaded()
        vecs = self._model.encode(list(texts), normalize_embeddings=normalize)  # type: ignore[union-attr]
        return np.asarray(vecs, dtype=np.float32)


def from_config(cfg) -> BGE_M3Embedder:
    name = (cfg.models or {}).get("embedder", "BAAI/bge-m3")
    embedder = BGE_M3Embedder(model_name=name)
    embedder._load_hf()
    return embedder