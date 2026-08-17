"""HuggingFace model wrappers.

Each module exposes a ``from_config(cfg)`` factory that loads the
configured real HuggingFace checkpoint. Loading failures propagate as
``RuntimeError`` so the caller can surface a clear error.

The LLM generator slot is also wired to OpenRouter by default
(``openai/gpt-oss-20b:free``) — see ``generator_openrouter.py``. Set
``HALLUC_DISABLE_OPENROUTER=1`` to opt out.
"""
from .embedder_bge_m3 import BGE_M3Embedder, from_config as embedder_from_config
from .generator_llama import LlamaGenerator, from_config as llama_from_config
from .generator_qwen import QwenGenerator, from_config as qwen_from_config
from .generator_openrouter import OpenRouterGenerator, from_config as openrouter_from_config
from .xgb_classifier import XGBClassifier, from_config as xgb_from_config
from .ner_gliner import GLiNERWrapper, from_config as gliner_from_config
from .ner_spacy import SpacyNER, from_config as spacy_from_config
from .reranker_bge import BGEReranker, from_config as reranker_from_config
from .nli_deberta import DebertaNLIModel, from_config as nli_from_config
from .eedc import EEDC, from_config as eedc_from_config
from .calibration import TemperatureScaler, IsotonicCalibrator, from_config as calibration_from_config

__all__ = [
    "BGE_M3Embedder",
    "LlamaGenerator",
    "QwenGenerator",
    "OpenRouterGenerator",
    "XGBClassifier",
    "GLiNERWrapper",
    "SpacyNER",
    "BGEReranker",
    "DebertaNLIModel",
    "EEDC",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "embedder_from_config",
    "llama_from_config",
    "qwen_from_config",
    "openrouter_from_config",
    "xgb_from_config",
    "gliner_from_config",
    "spacy_from_config",
    "reranker_from_config",
    "nli_from_config",
    "eedc_from_config",
    "calibration_from_config",
]