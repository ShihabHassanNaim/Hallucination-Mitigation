"""BGE-Reranker-v2 cross-encoder (Step 7)."""
from __future__ import annotations


class BGEReranker:
    """Cross-encoder reranker. .rerank(query, docs) -> [(doc, score), ...]."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_hf(self) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"sentence-transformers missing: {exc}")
        self._model = CrossEncoder(self.model_name)

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