"""BGE-Reranker-v2 cross-encoder (Step 7)."""
from __future__ import annotations

import os


def _pick_device() -> str:
    """Pick 'cuda' when a GPU is visible, else 'cpu'.

    Honours the ``HALLUC_RERANKER_DEVICE`` env var for manual override.
    """
    override = os.environ.get("HALLUC_RERANKER_DEVICE")
    if override:
        return override
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class BGEReranker:
    """Cross-encoder reranker. .rerank(query, docs) -> [(doc, score), ...]."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device or _pick_device()
        self._model = None

    def _load_hf(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"sentence-transformers missing: {exc}")
        self._model = CrossEncoder(self.model_name, device=self.device)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_hf()

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        self._ensure_loaded()
        if not docs:
            return []
        scores = self._model.predict([(query, d) for d in docs])  # type: ignore[union-attr]
        return [float(s) for s in scores]


def from_config(cfg) -> BGEReranker:
    name = (cfg.models or {}).get("reranker", "BAAI/bge-reranker-v2-m3")
    m = BGEReranker(model_name=name)
    m._load_hf()
    return m