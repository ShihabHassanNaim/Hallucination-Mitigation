"""spaCy NER wrapper (en_core_web_sm by default)."""
from __future__ import annotations


class SpacyNER:
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self._nlp = None

    def _load(self) -> None:
        try:
            import spacy  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"spacy not installed: {exc}")
        self._nlp = spacy.load(self.model_name)

    def _ensure_loaded(self) -> None:
        if self._nlp is None:
            self._load()

    def extract(self, text: str) -> list[dict]:
        self._ensure_loaded()
        doc = self._nlp(text)  # type: ignore[union-attr]
        return [
            {"text": ent.text, "label": ent.label_, "score": 1.0}
            for ent in doc.ents
        ]


def from_config(cfg) -> SpacyNER:
    name = (cfg.models or {}).get("spacy", "en_core_web_sm")
    m = SpacyNER(model_name=name)
    m._load()
    return m