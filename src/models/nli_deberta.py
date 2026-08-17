"""DeBERTa-v3-MNLI fact verifier (Step 8).

Returns labels: supported | contradicted | unknown.
"""
from __future__ import annotations


class DebertaNLIModel:
    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        self.model_name = model_name
        self._tok = None
        self._model = None

    def _load_hf(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"transformers missing: {exc}")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def _ensure_loaded(self) -> None:
        if self._model is None or self._tok is None:
            self._load_hf()

    def predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        self._ensure_loaded()
        import torch  # type: ignore

        inputs = self._tok(  # type: ignore[union-attr]
            premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits  # type: ignore[union-attr]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(probs.argmax())
        label = self._model.config.id2label[idx].lower()  # type: ignore[union-attr]
        if "entail" in label:
            label = "supported"
        elif "contradict" in label:
            label = "contradicted"
        else:
            label = "unknown"
        return label, float(probs[idx])


def from_config(cfg) -> DebertaNLIModel:
    name = (cfg.models or {}).get("nli", "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    m = DebertaNLIModel(model_name=name)
    m._load_hf()
    return m