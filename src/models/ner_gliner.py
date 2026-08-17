"""GLiNER zero-shot NER wrapper.

Used by Step 1 (Query Understanding) for entity extraction beyond
spaCy's fixed labels.
"""
from __future__ import annotations


class GLiNERWrapper:
    def __init__(self, model_name: str = "knowledgator/gliner-bi-base-v1.0"):
        self.model_name = model_name
        self._model = None

    def _load_hf(self) -> None:
        try:
            from gliner import GLiNER  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"gliner not installed: {exc}")
        self._model = GLiNER.from_pretrained(self.model_name)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_hf()

    def extract(self, text: str, labels: list[str] | None = None) -> list[dict]:
        self._ensure_loaded()
        labels = labels or ["person", "organization", "location", "date", "product"]
        ents = self._model.predict_entities(text, labels)  # type: ignore[union-attr]
        return [
            {"text": e["text"], "label": e["label"], "score": float(e.get("score", 0.0))}
            for e in ents
        ]


def from_config(cfg) -> GLiNERWrapper:
    name = (cfg.models or {}).get("gliner", "knowledgator/gliner-bi-base-v1.0")
    m = GLiNERWrapper(model_name=name)
    m._load_hf()
    return m